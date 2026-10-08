"""Public TurnBench DEV scoring/calibration of one immutable inference artifact.

No model inference/fitting, test/golden source access, or adaptive search. The
predeclared sweep uses OUR serving policy, scored by the official event scorer.
"""
from pathlib import Path
import argparse
import json
import inspect
import hashlib
import os
import subprocess
import sys
import numpy as np
from benchmark_infer import protocol_phase
from benchmark_infer import sha, digest, require_equal, REVISIONS, operating_point
from checkpointing import atomic_text

DEV_SOURCE = 'mundo-ai/turn-benchmark-dev'
DEV_REVISION = REVISIONS['dev']
UPSTREAM_COMMIT = '38a6f874322430cb3ca71d8a52aa1e636e88bad8'


def save_json(path, value):
    atomic_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), path)


def validate_dev_identity(identity):
    """Before opening npz/labels: reject test, unknown, mixed, and legacy inputs."""
    freeze = identity['freeze']; protocol = freeze['protocol']; dataset = identity['dataset']
    if freeze.get('format') != 'turnbench-inference-freeze-v2':
        raise ValueError('A complete v2 inference identity is required')
    for part in (protocol, dataset):
        if part.get('source') != DEV_SOURCE or part.get('revision') != DEV_REVISION:
            raise ValueError('Only the exact public dev source/revision is allowed; no test/golden inputs')
    if protocol.get('split') != 'dev' or protocol.get('dev_cache') is not True:
        raise ValueError('Only explicitly enabled dev probability caches may be scored')
    if protocol.get('gate') not in ('mean', 'last') or protocol.get('history_vad') != 'mean':
        raise ValueError('Unknown serving/input VAD policy')
    if protocol.get('grid_samples') != 2560 or protocol.get('sample_rate') != 16000:
        raise ValueError('Unknown causal decision grid')
    phase = protocol_phase(protocol)
    if phase == 128 and not freeze.get('phase_qualification_sha256'):
        raise ValueError('Phase128 runtime qualification identity required')
    if phase == 128 and not freeze.get('sources', {}).get('experiment/phase128_full_arrays.py'):
        raise ValueError('Full phase128 input source identity required')
    if not freeze.get('checkpoint') or not dataset.get('shards'):
        raise ValueError('Immutable checkpoint and dataset shard identities required')
    ids = identity['ids']
    if not ids or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i.isdecimal() for i in ids):
        raise ValueError('Invalid conversation coverage')
    operating_point(protocol['operating_point'])
    if freeze.get('execution_mode') is not None:
        from dev_feature_contract import validate_derived
        validate_derived(identity)
    return freeze


def verify_scorer_sources(upstream, freeze):
    path = Path(upstream)
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    require_equal(revision, UPSTREAM_COMMIT, 'Pinned upstream source revision')
    subprocess.run(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD', '--'], check=True, capture_output=True)
    # Includes the upstream gold/scorer code and our actual serving policy.
    actual = {'turnbench/'+str(p.relative_to(path/'turnbench')): sha(p)
              for p in sorted((path/'turnbench').rglob('*.py'))}
    expected = {k: v for k, v in freeze['sources'].items() if k.startswith('turnbench/')}
    require_equal(expected, actual, 'Gold/scorer sources used by inference')
    require_equal(freeze['sources']['experiment/heads.py'], sha(Path(__file__).with_name('heads.py')), 'Serving policy source')
    return {'upstream_commit': revision, 'upstream_sources': actual,
            'serving_policy_sha256': sha(Path(__file__).with_name('heads.py')),
            'scoring_script_sha256': sha(__file__)}


def validate_arrays(arrays, gate, duration, decision_phase_ms=160):
    if decision_phase_ms not in (128, 160):
        raise ValueError('Unknown decision phase')
    if set(arrays) != {'times', 'vad', 'gate_vad', 'probabilities'}:
        raise ValueError('Unexpected dev probability cache fields')
    times = arrays['times']; vad = arrays['vad']; serving = arrays['gate_vad']; probabilities = arrays['probabilities']
    n = len(times)
    if times.shape != (n,) or not n or any(v.shape != (n, 2) for v in (vad, serving, probabilities)):
        raise ValueError('Unaligned dev cache shapes')
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite dev cache')
    if any(np.any((v < 0) | (v > 1)) for v in (vad, serving, probabilities)):
        raise ValueError('Invalid probability range')
    expected = (decision_phase_ms*16 + np.arange(n, dtype=np.float64)*2560)/16000
    if decision_phase_ms == 128 and times[-1]+.160 <= duration + 1e-8:
        raise ValueError('Incomplete phase128 tail grid')
    if not np.array_equal(times, expected) or times[-1] > duration + 1e-8:
        raise ValueError('Noncausal/incomplete decision availability grid')
    if gate == 'mean' and not np.array_equal(vad, serving):
        raise ValueError('Serving mean gate differs from frozen history VAD')


def exact_coverage(expected, actual):
    if len(set(actual)) != len(actual) or set(expected) != set(actual):
        raise ValueError('Inference IDs do not exactly cover pinned public dev')


def threshold_grid(mode='uniform51', caches=None):
    if mode == 'uniform51':
        return np.linspace(0, 1, 51).tolist(), {'mode': mode, 'spacing': 'uniform', 'range': [0, 1]}
    if mode != 'official-quantiles' or not caches:
        raise ValueError('Official quantile mode requires validated dev caches')
    # Use the actual pinned upstream schema/function, but NEVER its commit rule.
    from turnbench.sweep import ProbsFile, candidate_thetas
    probabilities = ProbsFile.model_validate({'schema_version': 1, 'task': 'eot', 'frame_rate_hz': 6.25,
        'probs': [{'conversation_id': cid,
                   'speaker_1': {'prob': arrays['probabilities'][:, 0].tolist()},
                   'speaker_2': {'prob': arrays['probabilities'][:, 1].tolist()}}
                  for cid, arrays in caches.items()]})
    values = candidate_thetas(probabilities, n=512)
    if values != sorted(set(values)) or not all(np.isfinite(t) and 0 <= t <= 1 for t in values):
        raise ValueError('Invalid official candidate grid')
    return values, {'mode': mode, 'quantile_count': 512, 'positive_raw_scores_only': True,
                    'uniform_union': '.01,.02,...,.99', 'spacing': 'sorted unique quantile union',
                    'candidate_function': 'turnbench.sweep.candidate_thetas',
                    'candidate_function_sha256': hashlib.sha256(inspect.getsource(candidate_thetas).encode()).hexdigest()}


def policies(thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 51)
    return [{'threshold': float(t), 'recommit_s': r} for r in (None, 1.5) for t in thresholds]


def summarize(score):
    latency = score.latency()
    def finite(value): return float(value) if np.isfinite(value) else None
    return {'tp': score.tp, 'fn': score.fn, 'fp': score.fp, 'tn': score.tn,
            'recall': finite(score.recall), 'fp_rate': finite(score.fp_rate),
            'latency_ms': {k: finite(getattr(latency, k)) for k in ('p10', 'p50', 'p90')}}


def choose_attempt(attempts):
    """Fixed recall/FPR/latency ranking; no fallback above the FPR ceiling."""
    from heads import selection_key
    eligible = [a for a in attempts if a['score']['fp_rate'] is not None and a['score']['fp_rate'] <= .1]
    return max(eligible, key=selection_key) if eligible else None


def main():
    if not Path('/content').is_dir():
        raise RuntimeError('Remote Colab only; no local benchmark loading')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inference', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--turnbench-path', default='/content/turnbench')
    ap.add_argument('--threshold-grid', choices=['uniform51', 'official-quantiles'], default='uniform51')
    args = ap.parse_args()
    source = Path(args.inference); identity_path = source/'inference.json'
    identity_hash = sha(identity_path); identity = json.loads(identity_path.read_text())
    require_equal(identity_hash, sha(identity_path), 'Inference identity during load')
    freeze = validate_dev_identity(identity)
    provenance = verify_scorer_sources(args.turnbench_path, freeze)
    ids = identity['ids']; protocol = freeze['protocol']; run_hash = digest(identity)
    # Hash check every cache BEFORE label resolution. A renamed test cache cannot
    # replace a dev cache without violating its inference transaction identity.
    caches = {}; transactions = {}; input_hashes = {'inference.json': identity_hash}
    if {p.stem for p in (source/'conversations').glob('*.json')} != set(ids):
        raise ValueError('Inference transaction coverage mismatch')
    for cid in ids:
        path = source/'conversations'/f'{cid}.json'; value = json.loads(path.read_text())
        require_equal(value['identity'], run_hash, 'Conversation inference identity')
        require_equal(value['payload_sha256'], digest(value['payload']), 'Conversation payload')
        require_equal(value['payload']['prediction']['conversation_id'], cid, 'Conversation ID')
        cache = source/f'{cid}.npz'
        require_equal(value['payload']['dev_cache_sha256'], sha(cache), 'Dev cache hash')
        with np.load(cache, allow_pickle=False) as arrays:
            cached = {k: arrays[k] for k in arrays.files}
        require_equal(value['payload']['dev_cache_sha256'], sha(cache), 'Dev cache during load')
        validate_arrays(cached, protocol['gate'], value['payload']['duration_s'], protocol_phase(protocol))
        caches[cid] = cached; transactions[cid] = value['payload']
        input_hashes[str(path.relative_to(source))] = sha(path)
        input_hashes[str(cache.relative_to(source))] = sha(cache)
    prediction_path = source/'predictions-dev.json'
    input_hashes['predictions-dev.json'] = sha(prediction_path)
    sys.path.insert(0, args.turnbench_path)
    from turnbench.data import resolve_dataset, conversation, conversation_ids
    from turnbench.gold import events_for_conversation
    from turnbench.score import TaskScore, score_task, merge
    from turnbench.submission import Submission, ConversationPrediction, SpeakerEvents, validate_coverage, validate_event_times
    from heads import commit_events
    fixed_submission = Submission.model_validate(json.loads(prediction_path.read_text()))
    validate_coverage(fixed_submission, ids)
    fixed_by_id = {p.conversation_id: p for p in fixed_submission.predictions}
    for cid in ids:
        require_equal(fixed_by_id[cid].model_dump(), transactions[cid]['prediction'], 'Final/transaction prediction')
    thresholds, grid_provenance = threshold_grid(args.threshold_grid, caches)
    grid_provenance.update(candidates=thresholds, unique_threshold_count=len(thresholds),
                           operating_point_attempts=2*len(thresholds), upstream_commit=UPSTREAM_COMMIT,
                           sweep_source_sha256=provenance['upstream_sources']['turnbench/sweep.py'])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    save_json(out/'provenance.json', {'purpose': 'public TurnBench dev development/calibration only',
        'source': DEV_SOURCE, 'revision': DEV_REVISION, 'test_accessed': False,
        'candidate_checkpoint_sha256': freeze['checkpoint'], 'input_hashes': input_hashes,
        'inference_identity': identity, 'scorer': provenance,
        'search': {**grid_provenance, 'recommit_s': [None, 1.5],
                   'fpr_ceiling': .1, 'policy': 'heads.commit_events', 'ranking': 'heads.selection_key'}})
    # No caller-controlled source or revision, and no test ID lookup.
    if Path('/content/.hf_token').is_file():
        os.environ['HF_TOKEN'] = Path('/content/.hf_token').read_text().strip()
    dataset = resolve_dataset(DEV_SOURCE, revision=DEV_REVISION, skip_audio=True)
    official_ids = conversation_ids(dataset)
    exact_coverage(official_ids, ids)
    validate_coverage(fixed_submission, official_ids)
    gold = {}; durations = {}
    for cid in ids:
        conv = conversation(dataset, cid)
        gold[cid] = events_for_conversation(conv); durations[cid] = conv.duration_s
        if abs(transactions[cid]['duration_s'] - conv.duration_s) > 1/16000:
            raise ValueError('Public gold/inference duration mismatch')
        validate_arrays(caches[cid], protocol['gate'], conv.duration_s, protocol_phase(protocol))
    def evaluate(op, label):
        total = TaskScore(); predictions = []; per_conversation = []
        for cid in ids:
            a = caches[cid]
            events = commit_events(a['times'], a['gate_vad'], a['probabilities'], op['threshold'], op['recommit_s'])
            row = ConversationPrediction(conversation_id=cid, speaker_1=SpeakerEvents(eot=events[1], interruption=[]),
                                         speaker_2=SpeakerEvents(eot=events[2], interruption=[]))
            validate_event_times(row, durations[cid]); predictions.append(row)
            if label == 'fixed':
                require_equal(row.model_dump(), fixed_by_id[cid].model_dump(), 'Frozen policy reproduction')
            g = gold[cid]
            score = score_task(g.eot_positive_events, g.eot_negative_spans, events, g.eot_excluded)
            merge(total, score)
            per_conversation.append({'id': cid, 'score': summarize(score), 'latencies_ms': score.latencies_ms})
        submission = Submission(schema_version=1, predictions=predictions)
        validate_coverage(submission, official_ids)
        atomic_text(submission.model_dump_json(indent=2), out/f'predictions-{label}.json')
        result = {**op, 'score': summarize(total), 'predictions': f'predictions-{label}.json',
                  'per_conversation': per_conversation}
        save_json(out/f'score-{label}.json', result)
        return result
    fixed = evaluate(operating_point(protocol['operating_point']), 'fixed')
    attempts = []
    for i, op in enumerate(policies(thresholds)):
        attempts.append(evaluate(op, f'sweep-{i:03d}'))
    best = choose_attempt(attempts)
    save_json(out/'report.json', {'source': DEV_SOURCE, 'revision': DEV_REVISION, 'task': 'eot',
        'test_accessed': False, 'candidate_checkpoint_sha256': freeze['checkpoint'],
        'fixed_operating_point': fixed, 'threshold_grid': grid_provenance, 'attempt_count': len(attempts),
        'attempts': attempts, 'best_dev_operating_point': best,
        'qualification': 'No threshold meets the fixed <=0.10 FPR ceiling' if best is None else 'dev calibration only',
        'limitation': 'Public dev used for development/calibration; not a held-out test claim. Test and otoSpeech gate remain sealed.'})
    if best is not None:
        save_json(out/'operating-point.json', {'threshold': best['threshold'], 'recommit_s': best['recommit_s']})
        save_json(out/'operating-point-provenance.json', {'candidate_checkpoint_sha256': freeze['checkpoint'],
            'inference_identity_sha256': run_hash, 'threshold_grid': grid_provenance, 'report_sha256': sha(out/'report.json'),
            'operating_point_sha256': sha(out/'operating-point.json'), 'source': DEV_SOURCE, 'revision': DEV_REVISION,
            'requires_parent_candidate_selection_before_final_test_freeze': True})
    print(json.dumps({'event': 'PUBLIC_DEV_SCORING_COMPLETE', 'fixed': fixed['score'],
                      'best_dev': best['score'] if best else None, 'threshold_grid': args.threshold_grid,
                      'policies': len(attempts), 'test_accessed': False}), flush=True)


if __name__ == '__main__': main()
