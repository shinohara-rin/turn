"""Predeclared equal-probability causal-head ensembles; dev calibration only.

Candidates: MLP seeds42/17/2026, or MLP42 + GRU42. No learned weights or extra
threshold/policy search. Inference consumes in-memory x/times/vad and no labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np

CANDIDATES = {'mlp3': [('mlp', 42), ('mlp', 17), ('mlp', 2026)],
              'mlp_gru': [('mlp', 42), ('gru', 42)]}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_members(members, candidate, manifest_hash):
    expected = CANDIDATES[candidate]
    actual = []
    for member in members:
        kind, checkpoint = member['kind'], member['checkpoint']
        if kind not in ('mlp', 'gru'):
            raise ValueError('Explicit member kind must be mlp or gru')
        config = checkpoint['config']
        if config.get('manifest_sha256') != manifest_hash:
            raise ValueError('Mixed or changed source manifests are forbidden')
        if kind == 'mlp' and (checkpoint.get('auxiliary') is not False or config.get('variant') != 'mlp'):
            raise ValueError('MLP member is not the predeclared plain MLP')
        if kind == 'gru' and (config.get('model') != 'projection-GRU-EOT' or config.get('bidirectional') is not False):
            raise ValueError('GRU member must be the causal temporal model')
        actual.append((kind, config.get('seed')))
    if sorted(actual) != sorted(expected):
        raise ValueError('Members do not match the predeclared candidate seeds/kinds')
    if len({m['checkpoint']['dim'] for m in members}) != 1:
        raise ValueError('Member feature dimensions differ')


def aligned_mean(predictions):
    """Average [member][conversation] records, rejecting reordered/misaligned data."""
    if not predictions or not predictions[0]:
        raise ValueError('Nonempty member predictions required')
    output = []
    count = len(predictions[0])
    if any(len(p) != count for p in predictions):
        raise ValueError('Conversation counts differ')
    for i in range(count):
        reference = predictions[0][i]
        arrays = []
        for member in predictions:
            record = member[i]
            if record['id'] != reference['id'] or not np.array_equal(record['times'], reference['times']):
                raise ValueError('Conversation identity/time alignment mismatch')
            p = np.asarray(record['probabilities'])
            if p.shape != (len(reference['times']), 2) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Invalid probability shape/range')
            arrays.append(p)
        output.append(np.mean(np.stack(arrays), axis=0, dtype=np.float64).astype(np.float32))
    return output


class EnsemblePredictor:
    """Self-contained bundle inference. Does not open paths, events, or annotations."""
    def __init__(self, bundle, device='cpu'):
        if bundle.get('format') != 'equal-probability-turn-heads-v1':
            raise ValueError('Unsupported ensemble format')
        self.members = bundle['members']; self.device = device
        validate_members(self.members, bundle['candidate'], bundle['manifest_sha256'])
        equal = [1/len(self.members)]*len(self.members)
        if bundle.get('weights') != equal:
            raise ValueError('Only equal weights are supported')
        self.models = []
        from heads import make_model
        from temporal_head import make_temporal_model
        for member in self.members:
            c = member['checkpoint']
            model = (make_model(c['dim'], c['hidden'], False) if member['kind'] == 'mlp' else
                     make_temporal_model(c['dim'], c['hidden'], c['projection'], c['dropout']))
            model.load_state_dict(c['model'], strict=True)
            self.models.append(model.to(device).eval())

    def predict(self, records):
        from heads import predict
        from temporal_head import temporal_predict
        clean = []
        seen = set()
        for record in records:
            # Deliberately select only inference fields, even if caller has labels.
            r = {k: record[k] for k in ('id', 'x', 'times', 'vad')}
            if r['id'] in seen:
                raise ValueError('Duplicate conversation identity')
            seen.add(r['id'])
            x, times, vad = np.asarray(r['x']), np.asarray(r['times']), np.asarray(r['vad'])
            if x.ndim != 3 or x.shape[:2] != (len(times), 2) or vad.shape != (len(times), 2) or not len(times):
                raise ValueError('Record shapes are not aligned')
            if x.shape[-1] != self.members[0]['checkpoint']['dim']:
                raise ValueError('Feature dimension mismatch')
            if not all(np.isfinite(a).all() for a in (x, times, vad)) or np.any(np.diff(times) <= 0):
                raise ValueError('Nonfinite or nonchronological input')
            clean.append(r)
        predictions = []
        for member, model in zip(self.members, self.models):
            c = member['checkpoint']
            values = (predict(model, clean, c['mean'], c['std'], self.device) if member['kind'] == 'mlp' else
                      temporal_predict(model, clean, c['mean'], c['std'], self.device, c['config']['chunk_frames']))
            predictions.append([{'id': r['id'], 'times': r['times'], 'probabilities': p}
                                for r, p in zip(clean, values)])
        return aligned_mean(predictions)


def calibrate(args):
    if not Path('/content').is_dir():
        raise RuntimeError('Calibration is remote-only')
    import torch
    from leakage_guard import validate_manifest
    from heads import load_records, official_score, selection_key
    audit = validate_manifest(args.manifest)  # Before opening any dev arrays/labels.
    source_hash = file_hash(args.manifest)
    members = []
    for specification in args.member:
        kind, separator, path = specification.partition(':')
        if not separator or kind not in ('mlp', 'gru'):
            raise ValueError('Use --member mlp:/path.pt or gru:/path.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        keys = ['model', 'mean', 'std', 'dim', 'hidden', 'epoch', 'config']
        keys += ['auxiliary'] if kind == 'mlp' else ['projection', 'dropout']
        members.append({'kind': kind, 'checkpoint_sha256': file_hash(path),
                        'checkpoint': {key: state[key] for key in keys}})
    validate_members(members, args.candidate, source_hash)
    bundle = {'format': 'equal-probability-turn-heads-v1', 'candidate': args.candidate,
              'members': members, 'weights': [1/len(members)]*len(members),
              'manifest_sha256': source_hash, 'leakage_audit': audit}
    predictor = EnsemblePredictor(bundle, args.device)
    dev = load_records(args.manifest, 'dev')
    probabilities = predictor.predict(dev)
    rows = []
    for recommit in (None, 1.5):
        for threshold in np.linspace(0, 1, 51):
            score = official_score(dev, probabilities, float(threshold), recommit, args.turnbench_path)
            rows.append({'candidate': args.candidate, 'threshold': float(threshold),
                         'recommit_s': recommit, 'score': score})
    best = max(rows, key=selection_key)
    bundle['operating_point'] = best
    config = {'candidate': args.candidate, 'members': [{'kind': m['kind'], 'seed': m['checkpoint']['config']['seed'],
               'epoch': m['checkpoint']['epoch'], 'sha256': m['checkpoint_sha256']} for m in members],
              'weights': bundle['weights'], 'manifest_sha256': source_hash, 'leakage_audit': audit,
              'script_sha256': file_hash(__file__), 'calibration_split': 'dev',
              'threshold_steps': 51, 'recommit_s': [None, 1.5], 'learned_weights': False}
    bundle['config'] = config
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    for name, value in [('config', config), ('best', best), ('attempts', rows)]:
        (out/f'{name}.json').write_text(json.dumps(value, indent=2))
    temporary = out/'ensemble.pt.tmp'; torch.save(bundle, temporary); os.replace(temporary, out/'ensemble.pt')
    print(json.dumps(best), flush=True)


def evaluate(args):
    if args.split == 'gate' and not args.allow_gate:
        raise ValueError('Gate access requires explicit --allow-gate')
    import torch
    from leakage_guard import validate_manifest
    from heads import load_records, official_score
    audit = validate_manifest(args.manifest)
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if audit['split_sha256'] != bundle['leakage_audit']['split_sha256']:
        raise ValueError('Evaluation speaker split changed')
    if args.split == 'dev' and file_hash(args.manifest) != bundle['manifest_sha256']:
        raise ValueError('Development manifest changed')
    records = load_records(args.manifest, args.split)
    probabilities = EnsemblePredictor(bundle, args.device).predict(records)
    op = bundle['operating_point']
    result = {'split': args.split, 'operating_point': op,
              'score': official_score(records, probabilities, op['threshold'], op['recommit_s'], args.turnbench_path)}
    with Path(args.out).open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__); sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('calibrate'); fit.add_argument('--candidate', choices=CANDIDATES, required=True)
    fit.add_argument('--member', action='append', required=True)
    evaluation = sub.add_parser('evaluate'); evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--split', choices=['dev', 'gate'], default='dev')
    evaluation.add_argument('--allow-gate', action='store_true')
    for command in (fit, evaluation):
        command.add_argument('--manifest', required=True); command.add_argument('--out', required=True)
        command.add_argument('--device', default='cpu'); command.add_argument('--turnbench-path', default='/content/turnbench')
    args = parser.parse_args(); (calibrate if args.command == 'calibrate' else evaluate)(args)


if __name__ == '__main__':
    main()
