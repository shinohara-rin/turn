"""Remote-only frozen-encoder turn heads. No dataset/network downloads here.

Inputs follow manifest.json [{id, split, npz, events}]. All times must denote
causal decision availability, not the timestamp of a frame whose future was used.
Only train/dev records are opened. Gate evaluation requires a separate explicit
invocation with --allow-gate and a frozen checkpoint/operating point.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def causal_history(times, vad):
    """[T,2,8]: current probability, quiet age, talk age, 5 trailing means."""
    times = np.asarray(times, dtype=np.float64)
    vad = np.asarray(vad, dtype=np.float32)
    if vad.shape != (len(times), 2) or not len(times):
        raise ValueError('Expected nonempty times [T], vad [T,2]')
    if not np.all(np.isfinite(vad)) or np.any((vad < 0) | (vad > 1)):
        raise ValueError('VAD probabilities must be finite in [0,1]')
    if np.any(np.diff(times) <= 0) or not np.all(np.isfinite(times)):
        raise ValueError('Times must be finite and strictly increasing')
    if len(times) > 1 and not np.allclose(np.diff(times), .16, atol=.002):
        raise ValueError('Cache must use 160 ms grid')
    talk = vad >= .5
    ages = np.zeros((len(times), 2, 2), np.float32)
    starts = np.full(2, times[0])
    for i in range(len(times)):
        if i:
            starts[talk[i] != talk[i-1]] = times[i]
        age = np.minimum(times[i] - starts, 10.)
        ages[i, :, 0] = age * (~talk[i])
        ages[i, :, 1] = age * talk[i]
    history = [vad[..., None], ages / 10.]
    cumulative = np.concatenate([np.zeros((1, 2)), np.cumsum(vad, axis=0)])
    for seconds in (.32, .64, 1.28, 2.56, 5.12):
        left = np.searchsorted(times, times-seconds, side='right')
        right = np.arange(len(times))+1
        history.append(((cumulative[right]-cumulative[left]) / (right-left)[:, None])[..., None])
    return np.concatenate(history, axis=-1).astype(np.float32)


def speaker_features(features, history, extras=None):
    """Share one head, with own-channel then other-channel features."""
    values = [np.asarray(features, np.float32), history]
    if extras is not None:
        values.append(np.asarray(extras, np.float32))
    if any(v.ndim != 3 or v.shape[:2] != history.shape[:2] for v in values):
        raise ValueError('Features/extras must have shape [T,2,D]')
    x = np.concatenate(values, axis=-1)
    if not np.all(np.isfinite(x)):
        raise ValueError('Nonfinite features')
    return np.concatenate([x, x[:, ::-1]], axis=-1)


def eot_targets(times, vad, events, background_weight=.02):
    """Positives start at anchor (never anticipate annotation); stop at resume.

    Explicit pause windows each receive total weight 1, as do each positive
    window. Background is capped to one small total weight per quiet episode,
    preventing long listening stretches from swamping annotated pauses.
    """
    times = np.asarray(times)
    quiet = vad < .5
    y = np.zeros_like(vad, dtype=np.float32)
    w = np.zeros_like(y)
    for s in range(2):
        starts = np.flatnonzero(quiet[:, s] & np.r_[True, ~quiet[:-1, s]])
        ends = np.flatnonzero(quiet[:, s] & np.r_[~quiet[1:, s], True]) + 1
        for start, end in zip(starts, ends):
            # Ignore initial listening before this channel has ever spoken.
            if start and np.any(~quiet[:start, s]):
                w[start:end, s] = background_weight / (end-start)
        for span in events.get('eot_negative_spans', []):
            if span['speaker'] == s+1:
                mask = (times >= span['start']) & (times <= span['end']) & quiet[:, s]
                w[mask, s] = 1. / max(1, mask.sum())
        for event in events.get('eot_positive_events', []):
            if event['speaker'] != s+1:
                continue
            anchor = event['time_s']
            # Detect a transition, not residual VAD high immediately after EOT.
            resume = np.flatnonzero((times > anchor) & ~quiet[:, s] & np.r_[False, quiet[:-1, s]])
            end = min(anchor+2.5, times[resume[0]] if len(resume) else np.inf)
            mask = (times >= anchor) & (times < end) & quiet[:, s]
            y[mask, s] = 1.
            w[mask, s] = 1. / max(1, mask.sum())
        for span in events.get('eot_excluded', []):
            if span['speaker'] == s+1:
                w[(times >= span['start']) & (times <= span['end']), s] = 0
    return y, w


def vap_targets(times, activity):
    """Training supervision only: own/other occupancy in future horizon bins."""
    activity = np.asarray(activity, np.float32)
    cumulative = np.concatenate([np.zeros((1, 2)), np.cumsum(activity, axis=0)])
    bins = []
    valid = []
    for lo, hi in ((0, .32), (.32, .64), (.64, 1.28), (1.28, 2.56)):
        left = np.searchsorted(times, times+lo, side='right')
        right = np.searchsorted(times, times+hi, side='right')
        occupancy = (cumulative[right]-cumulative[left])/np.maximum(right-left, 1)[:, None]
        bins.append(np.stack([occupancy, occupancy[:, ::-1]], axis=1))
        valid.append((times+hi <= times[-1]) & (right > left))
    # Each speaker has four own/other bin targets (8 total).
    target = np.concatenate(bins, axis=-1).astype(np.float32)
    mask = np.repeat(np.stack(valid, axis=-1), 2, axis=-1)[:, None, :]
    return target, np.broadcast_to(mask, target.shape).astype(np.float32)


def load_records(manifest, split):
    from leakage_guard import validate_manifest
    validate_manifest(manifest)
    root = Path(manifest).resolve().parent
    records = json.loads(Path(manifest).read_text())
    result = []
    for record in records:
        if record['split'] != split:
            continue
        with np.load(root / record['npz'], allow_pickle=False) as cache:
            t, v = cache['times'], cache['vad']
            h = causal_history(t, v)
            x = speaker_features(cache['features'], h, cache['extras'] if 'extras' in cache else None)
            events = json.loads((root / record['events']).read_text())
            y, w = eot_targets(t, v, events)
            a, am = vap_targets(t, cache['annotation_activity']) if split == 'train' and 'annotation_activity' in cache else (None, None)
            result.append(dict(id=record['id'], times=t, vad=v, x=x, y=y, w=w, aux=a, aux_mask=am, events=events))
    if not result:
        raise ValueError(f'No {split} records')
    return result


def commit_events(times, vad, probabilities, threshold, recommit_s=None):
    """Fire only after observed speech and during own-channel quiet."""
    output = {1: [], 2: []}
    for s in range(2):
        armed = False
        last_fire = -np.inf
        for t, v, p in zip(times, vad[:, s], probabilities[:, s]):
            if v >= .5:
                armed = True
                last_fire = -np.inf
            elif armed and p >= threshold:
                if last_fire == -np.inf or (recommit_s is not None and t-last_fire >= recommit_s):
                    output[s+1].append(float(t))
                    last_fire = t
    return output


def official_score(records, probabilities, threshold, recommit_s, turnbench_path):
    if str(turnbench_path) not in sys.path:
        sys.path.insert(0, str(turnbench_path))
    from turnbench.score import TaskScore, score_task, merge
    from turnbench.gold import AnchorEvent, Interval
    total = TaskScore()
    for record, probs in zip(records, probabilities):
        gold = record['events']
        prediction = commit_events(record['times'], record['vad'], probs, threshold, recommit_s)
        score = score_task([AnchorEvent(**e) for e in gold['eot_positive_events']],
                           [Interval(**e) for e in gold['eot_negative_spans']], prediction,
                           [Interval(**e) for e in gold.get('eot_excluded', [])])
        merge(total, score)
    latency = total.latency()
    def finite(x):
        return float(x) if np.isfinite(x) else None
    return dict(tp=total.tp, fn=total.fn, fp=total.fp, tn=total.tn,
                recall=finite(total.recall), fp_rate=finite(total.fp_rate),
                latency_ms={k: finite(getattr(latency, k)) for k in ('p10','p50','p90')})


def selection_key(row):
    score = row['score']
    fp, recall = score['fp_rate'], score['recall']
    qualified = fp is not None and fp <= .1
    return (qualified, recall if qualified and recall is not None else -1,
            -fp if fp is not None else -np.inf,
            -(score['latency_ms']['p50'] or 0))


def make_model(dim, hidden, auxiliary):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 9 if auxiliary else 1))


def predict(model, records, mean, std, device):
    import torch
    model.eval()
    result = []
    with torch.no_grad():
        for record in records:
            flat = record['x'].reshape(-1, len(mean))
            parts = []
            for start in range(0, len(flat), 8192):
                x = torch.as_tensor((flat[start:start+8192]-mean)/std, device=device)
                parts.append(model(x)[:, 0].sigmoid().cpu().numpy())
            result.append(np.concatenate(parts).reshape(-1, 2))
    return result


def train(args):
    import torch
    import torch.nn.functional as F
    if sys.platform == 'darwin':
        raise RuntimeError('Training is remote-only; run this CLI on Colab')
    from checkpointing import EpochRecovery
    config = vars(args).copy()
    config['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    recovery = EpochRecovery(args.out, config, getattr(args, 'resume', None))
    config = recovery.config
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    training = load_records(args.manifest, 'train')
    dev = load_records(args.manifest, 'dev')
    dim = training[0]['x'].shape[-1]
    x = np.concatenate([r['x'].reshape(-1, dim) for r in training])
    y = np.concatenate([r['y'].reshape(-1) for r in training])
    w = np.concatenate([r['w'].reshape(-1) for r in training])
    mean, std = x.mean(0), np.maximum(x.std(0), .01)
    auxiliary = args.variant == 'vap'
    if auxiliary:
        if any(r['aux'] is None for r in training):
            raise ValueError('VAP variant requires annotation_activity for every train record')
        aux = np.concatenate([r['aux'].reshape(-1, 8) for r in training])
        aux_mask = np.concatenate([r['aux_mask'].reshape(-1, 8) for r in training])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(dim, args.hidden, auxiliary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    if recovery.checkpoint is not None:
        mean, std = recovery.checkpoint['mean'], recovery.checkpoint['std']
    recovery.restore(model, optimizer)
    # Uniform frame sampling with importance weights preserves event weighting.
    w = w/max(float(w.mean()), 1e-8)
    for epoch in range(recovery.start_epoch, args.epochs+1):
        if recovery.pending_evaluation(epoch):
            checkpoint = recovery.checkpoint
            training_loss = checkpoint['training_state']['training_loss']
        else:
            model.train()
            losses = []
            for index in np.array_split(np.random.permutation(len(x)), max(1, int(np.ceil(len(x)/args.batch_size)))):
                xb = torch.as_tensor((x[index]-mean)/std, device=device)
                output = model(xb)
                loss = (F.binary_cross_entropy_with_logits(output[:,0], torch.as_tensor(y[index], device=device), reduction='none') * torch.as_tensor(w[index], device=device)).mean()
                if auxiliary:
                    mask = torch.as_tensor(aux_mask[index], device=device)
                    aux_loss = F.binary_cross_entropy_with_logits(output[:,1:], torch.as_tensor(aux[index], device=device), reduction='none')
                    loss = loss + args.aux_weight * (aux_loss*mask).sum()/mask.sum().clamp_min(1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            training_loss = float(np.mean(losses))
            checkpoint = dict(model=model.state_dict(), mean=mean, std=std, dim=dim, hidden=args.hidden, auxiliary=auxiliary, epoch=epoch, config=config)
            checkpoint = recovery.save_trained(checkpoint, optimizer, training_loss)
        probs = predict(model, dev, mean, std, device)
        rows = []
        for recommit in (None, 1.5):
            for threshold in np.linspace(0, 1, args.threshold_steps):
                score = official_score(dev, probs, float(threshold), recommit, args.turnbench_path)
                rows.append(dict(epoch=epoch, threshold=float(threshold), recommit_s=recommit, score=score))
        candidate = recovery.complete(checkpoint, optimizer, training_loss, rows, selection_key)
        print(json.dumps(candidate), flush=True)


def evaluate(args):
    import torch
    if args.split == 'gate' and not args.allow_gate:
        raise ValueError('Gate access requires explicit --allow-gate authorization')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    records = load_records(args.manifest, args.split)
    model = make_model(checkpoint['dim'], checkpoint['hidden'], checkpoint['auxiliary'])
    model.load_state_dict(checkpoint['model'])
    probs = predict(model, records, checkpoint['mean'], checkpoint['std'], 'cpu')
    op = checkpoint['operating_point']
    score = official_score(records, probs, op['threshold'], op['recommit_s'], args.turnbench_path)
    result = dict(split=args.split, checkpoint=args.checkpoint, operating_point=op, score=score)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--resume', nargs='?', const='auto', help='Resume latest epoch in --out, or a specified resumable checkpoint')
    fit.add_argument('--variant', choices=['mlp', 'vap'], default='mlp')
    fit.add_argument('--epochs', type=int, default=12)
    fit.add_argument('--hidden', type=int, default=128)
    fit.add_argument('--seed', type=int, default=42)
    fit.add_argument('--lr', type=float, default=.001)
    fit.add_argument('--batch-size', type=int, default=2048)
    fit.add_argument('--aux-weight', type=float, default=.2)
    fit.add_argument('--threshold-steps', type=int, default=51)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--split', choices=['dev', 'gate'], default='dev')
    evaluation.add_argument('--allow-gate', action='store_true')
    for command in (fit, evaluation):
        command.add_argument('--manifest', required=True)
        command.add_argument('--out', required=True)
        command.add_argument('--turnbench-path', default='/content/turnbench')
    args = parser.parse_args()
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
