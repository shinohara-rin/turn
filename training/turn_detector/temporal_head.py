"""Causal shared-speaker GRU over frozen paired features; remote training only.

Speaker channels are independent batch members, each seeing own/other paired
features. Hidden state resets per conversation and is detached, never reset,
between chronological TBPTT chunks. No padding or bidirectional recurrence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from heads import load_records, official_score, selection_key


def make_temporal_model(dim, hidden=128, projection=128, dropout=.1):
    """Lazy torch import lets contract tests run without installing torch."""
    import torch.nn as nn

    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(dim, projection), nn.GELU(), nn.Dropout(dropout))
            self.recurrent = nn.GRU(projection, hidden, num_layers=1, bidirectional=False)
            self.output = nn.Linear(hidden, 1)

        def forward(self, x, state=None):
            # x: [time, speaker-as-batch, paired features]
            encoded, state = self.recurrent(self.project(x), state)
            return self.output(encoded).squeeze(-1), state

    return TemporalHead()


def chunk_ranges(length, chunk_frames):
    if length <= 0 or chunk_frames <= 0:
        raise ValueError('Sequence and chunk lengths must be positive')
    return [(start, min(length, start+chunk_frames)) for start in range(0, length, chunk_frames)]


def supervised_weights(record, warmup_frames):
    """Mask fixed conversation-start warmup only, not each TBPTT boundary."""
    if warmup_frames < 0:
        raise ValueError('Warmup must be nonnegative')
    weight = record['w'].copy()
    weight[:warmup_frames] = 0
    return weight


def train_scaler(records):
    """Float64 streaming moments from training conversations, never dev/gate."""
    count = 0
    total = square = None
    for record in records:
        x = record['x'].reshape(-1, record['x'].shape[-1]).astype(np.float64)
        count += len(x)
        if total is None:
            total = x.sum(0)
            square = np.square(x).sum(0)
        else:
            total += x.sum(0)
            square += np.square(x).sum(0)
    if not count:
        raise ValueError('Empty training data')
    mean = total/count
    variance = np.maximum(square/count - mean**2, 0)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), .01).astype(np.float32)


def temporal_predict(model, records, mean, std, device='cpu', chunk_frames=256):
    import torch
    model.eval()
    result = []
    with torch.no_grad():
        for record in records:
            state = None  # A new conversation has no inherited hidden state.
            probabilities = []
            for start, end in chunk_ranges(len(record['x']), chunk_frames):
                x = torch.as_tensor((record['x'][start:end]-mean)/std, device=device)
                logits, state = model(x, state)
                probabilities.append(logits.sigmoid().cpu().numpy())
            result.append(np.concatenate(probabilities, axis=0))
    return result


def train(args):
    if sys.platform == 'darwin':
        raise RuntimeError('Training is remote-only; run on Colab')
    import torch
    import torch.nn.functional as F
    # Must exist in the integrated parent checkout; never silently skip guard.
    from leakage_guard import validate_manifest
    audit = validate_manifest(args.manifest)
    if args.epochs <= 0 or args.threshold_steps < 2 or args.chunk_frames <= 0 or args.warmup_frames < 0:
        raise ValueError('Invalid training configuration')
    from checkpointing import EpochRecovery
    config = vars(args).copy()
    config.update(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  leakage_audit=audit, torch_version=torch.__version__,
                  model='projection-GRU-EOT', bidirectional=False,
                  supervision='EOT only', sequence_batch='two independent speaker states')
    recovery = EpochRecovery(args.out, config, getattr(args, 'resume', None))
    config = recovery.config
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training = load_records(args.manifest, 'train')
    dev = load_records(args.manifest, 'dev')
    mean, std = train_scaler(training)
    weights = [supervised_weights(record, args.warmup_frames) for record in training]
    weight_mean = sum(float(w.sum()) for w in weights)/sum(w.size for w in weights)
    if weight_mean <= 0:
        raise ValueError('No weighted EOT supervision after warmup')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_temporal_model(len(mean), args.hidden, args.projection, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    if recovery.checkpoint is not None:
        mean, std = recovery.checkpoint['mean'], recovery.checkpoint['std']
    recovery.restore(model, optimizer, rng)
    for epoch in range(recovery.start_epoch, args.epochs+1):
        if recovery.pending_evaluation(epoch):
            checkpoint = recovery.checkpoint
            training_loss = checkpoint['training_state']['training_loss']
        else:
            model.train()
            loss_sum = frames = 0
            for index in rng.permutation(len(training)):
                record = training[index]
                state = None
                # Conversations can be shuffled; frames within one never are.
                for start, end in chunk_ranges(len(record['x']), args.chunk_frames):
                    x = torch.as_tensor((record['x'][start:end]-mean)/std, device=device)
                    target = torch.as_tensor(record['y'][start:end], device=device)
                    weight = torch.as_tensor(weights[index][start:end]/weight_mean, device=device)
                    logits, state = model(x, state)
                    # Truncation changes the backward graph, not deployment memory.
                    state = state.detach()
                    # Constant denominator avoids overweighting tiny final chunks.
                    loss = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weight).sum()/(args.chunk_frames*2)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    loss_sum += float(loss.detach())*args.chunk_frames*2
                    frames += (end-start)*2
            training_loss = loss_sum/max(frames, 1)
            checkpoint = dict(model=model.state_dict(), mean=mean, std=std, dim=len(mean),
                              hidden=args.hidden, projection=args.projection, dropout=args.dropout,
                              epoch=epoch, config=config)
            checkpoint = recovery.save_trained(checkpoint, optimizer, training_loss, rng)
        probabilities = temporal_predict(model, dev, mean, std, device, args.chunk_frames)
        rows = []
        for recommit in (None, 1.5):
            for threshold in np.linspace(0, 1, args.threshold_steps):
                score = official_score(dev, probabilities, float(threshold), recommit, args.turnbench_path)
                rows.append(dict(epoch=epoch, threshold=float(threshold), recommit_s=recommit, score=score))
        candidate = recovery.complete(checkpoint, optimizer, training_loss, rows, selection_key, rng)
        print(json.dumps(candidate), flush=True)


def evaluate(args):
    if args.split == 'gate' and not args.allow_gate:
        raise ValueError('Gate access requires explicit --allow-gate authorization')
    from leakage_guard import validate_manifest
    validate_manifest(args.manifest)
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    records = load_records(args.manifest, args.split)
    model = make_temporal_model(checkpoint['dim'], checkpoint['hidden'], checkpoint['projection'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['model'])
    probabilities = temporal_predict(model, records, checkpoint['mean'], checkpoint['std'],
                                     chunk_frames=checkpoint['config']['chunk_frames'])
    operating_point = checkpoint['operating_point']
    score = official_score(records, probabilities, operating_point['threshold'],
                           operating_point['recommit_s'], args.turnbench_path)
    result = dict(split=args.split, checkpoint=args.checkpoint, operating_point=operating_point, score=score)
    with Path(args.out).open('x') as file:
        file.write(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--resume', nargs='?', const='auto', help='Resume latest epoch in --out, or a specified resumable checkpoint')
    fit.add_argument('--epochs', type=int, default=12)
    fit.add_argument('--seed', type=int, default=42)
    fit.add_argument('--hidden', type=int, default=128)
    fit.add_argument('--projection', type=int, default=128)
    fit.add_argument('--dropout', type=float, default=.1)
    fit.add_argument('--chunk-frames', type=int, default=256)
    fit.add_argument('--warmup-frames', type=int, default=8)
    fit.add_argument('--lr', type=float, default=.001)
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
