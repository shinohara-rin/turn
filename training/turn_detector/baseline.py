"""Remote-cache silence baseline; fixed 72-point dev sweep, no gate tuning."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from heads import causal_history, official_score, selection_key
from leakage_guard import validate_manifest

# Fixed before running any benchmark; same 160 ms decision grid as learned heads.
QUIET_SECONDS = (.16, .32, .48, .64, .96, 1.28, 1.60, 2., 2.4)
OTHER_SPEECH_SECONDS = (None, .16, .32, .64)
RECOMMIT_SECONDS = (None, 1.5)


def fixed_grid():
    return [dict(quiet_s=quiet, other_speech_s=other, recommit_s=recommit)
            for quiet in QUIET_SECONDS for other in OTHER_SPEECH_SECONDS
            for recommit in RECOMMIT_SECONDS]


def read_records(manifest, split):
    """Only selected split files are opened; embeddings/activity never read."""
    validate_manifest(manifest)
    root = Path(manifest).resolve().parent
    records = []
    for item in json.loads(Path(manifest).read_text()):
        if item['split'] != split:
            continue
        with np.load(root/item['npz'], allow_pickle=False) as cache:
            times, vad = cache['times'], cache['vad']
            history = causal_history(times, vad)
        events = json.loads((root/item['events']).read_text())
        records.append(dict(id=item['id'], times=times, vad=vad, events=events,
                            history=history))
    if not records:
        raise ValueError(f'No {split} records')
    return records


def baseline_probabilities(record, quiet_s, other_speech_s=None):
    """Binary decision eligibility from causal own quiet / other speech age.

    Age starts at the first frame classified in the new state. This deliberately
    makes no assumption about when between grid points the transition occurred.
    commit_events still enforces observed own speech and the episode policy.
    """
    vad = record['vad']
    history = record['history']
    # Float32 history encoding can differ at boundaries by a few nanoseconds.
    eligible = (vad < .5) & (history[:,:,1]*10 + 1e-6 >= quiet_s)
    if other_speech_s is not None:
        eligible &= (vad[:,::-1] >= .5) & (history[:,::-1,2]*10 + 1e-6 >= other_speech_s)
    return eligible.astype(np.float32)


def score_policy(records, policy, turnbench_path):
    probabilities = [baseline_probabilities(record, policy['quiet_s'], policy['other_speech_s'])
                     for record in records]
    return official_score(records, probabilities, .5, policy['recommit_s'], turnbench_path)


def sweep(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = read_records(args.manifest, 'dev')
    config = dict(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  selection_split='dev', fp_ceiling=.1, grid=fixed_grid(),
                  conversation_ids=[record['id'] for record in records])
    # Pin local scorer implementation without fetching anything.
    for name in ('score.py', 'gold.py'):
        path = Path(args.turnbench_path)/'turnbench'/name
        config[f'{name}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    with (out/'attempts.jsonl').open('x') as file:
        for policy in fixed_grid():
            row = dict(policy=policy, score=score_policy(records, policy, args.turnbench_path))
            rows.append(row)
            file.write(json.dumps(row)+'\n')
            file.flush()
            print(json.dumps(row), flush=True)
    best = max(rows, key=selection_key)
    best['qualified'] = best['score']['fp_rate'] is not None and best['score']['fp_rate'] <= .1
    best['selection_split'] = 'dev'
    (out/'best.json').write_text(json.dumps(best, indent=2))
    families = {}
    for name, use_other in (('silence_only', False), ('silence_and_other_speech', True)):
        candidates = [row for row in rows if (row['policy']['other_speech_s'] is not None) == use_other]
        families[name] = max(candidates, key=selection_key)
    (out/'family_best.json').write_text(json.dumps(families, indent=2))
    print(json.dumps(dict(best=best, families=families)), flush=True)


def evaluate(args):
    if args.split == 'gate' and not args.allow_gate:
        raise ValueError('Gate evaluation requires explicit --allow-gate authorization')
    frozen = json.loads(Path(args.policy).read_text())
    if frozen.get('selection_split') != 'dev':
        raise ValueError('Expected policy previously selected on dev')
    records = read_records(args.manifest, args.split)
    result = dict(split=args.split, policy=frozen['policy'],
                  policy_sha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
                  score=score_policy(records, frozen['policy'], args.turnbench_path))
    with Path(args.out).open('x') as file:
        file.write(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('sweep')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--policy', required=True)
    evaluation.add_argument('--split', choices=['dev', 'gate'], default='dev')
    evaluation.add_argument('--allow-gate', action='store_true')
    for command in (fit, evaluation):
        command.add_argument('--manifest', required=True)
        command.add_argument('--out', required=True)
        command.add_argument('--turnbench-path', default='/content/turnbench')
    args = parser.parse_args()
    (sweep if args.command == 'sweep' else evaluate)(args)


if __name__ == '__main__':
    main()
