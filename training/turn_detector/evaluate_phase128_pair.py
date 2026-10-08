"""Matched-event evaluation of two fixed guarded otoSpeech pilot heads.

No threshold search, fitting, audio access, public benchmark or sealed gate.
--out is one JSON report file. All results are exploratory development evidence.
"""
from pathlib import Path
import argparse
import bisect
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import numpy as np

HORIZON_POLICY = 'original gold complete-window boundary censor with overlap propagation; baseline N-1 vs phase N rows'
UPSTREAM_COMMIT = '38a6f874322430cb3ca71d8a52aa1e636e88bad8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def windows(events, tau_pre, tau_max):
    anchors = {}
    for event in events:
        anchors.setdefault(event['speaker'], []).append(event['time_s'])
    for values in anchors.values(): values.sort()
    result = []
    # Stable ordering and original ordinal distinguish even duplicate anchors.
    for ordinal, event in sorted(enumerate(events), key=lambda item: item[1]['time_s']):
        times = anchors[event['speaker']]; i = bisect.bisect_right(times, event['time_s'])
        right = min(event['time_s']+tau_max, times[i] if i < len(times) else np.inf)
        result.append({'ordinal': ordinal, 'speaker': event['speaker'], 'time_s': event['time_s'],
                       'start': event['time_s']-tau_pre, 'end': right})
    return result


def matched_score(gold, predictions, api):
    """Use actual official ledgers; assert trace matches official score_task."""
    excluded = [api.Interval(**v) for v in gold.get('eot_excluded', [])]
    ledgers = {s: api.PredictionLedger(api.scoreable_times(predictions[s], [v for v in excluded if v.speaker == s]))
               for s in (1, 2)}
    matches = []
    for event in windows(gold['eot_positive_events'], api.TAU_PRE_S, api.TAU_MAX_S):
        ledger = ledgers[event['speaker']]
        ledger.positive_windows.append((event['start'], event['end']))
        committed = ledger.claim_first_in(event['start'], event['end'])
        matches.append({**event, 'committed_s': committed,
                        'latency_ms': None if committed is None else (committed-event['time_s'])*1000})
    score = api.score_task([api.AnchorEvent(**v) for v in gold['eot_positive_events']],
                          [api.Interval(**v) for v in gold['eot_negative_spans']], predictions, excluded)
    latencies = [m['latency_ms'] for m in matches if m['committed_s'] is not None]
    require(score.tp == len(latencies) and score.fn == len(matches)-len(latencies), 'Official matching count drift')
    require(score.latencies_ms == latencies, 'Official matched-anchor latency drift')
    return score, matches


def summarize(score):
    def finite(value): return float(value) if np.isfinite(value) else None
    latency = score.latency()
    return {'tp': score.tp, 'fn': score.fn, 'fp': score.fp, 'tn': score.tn,
            'recall': finite(score.recall), 'fp_rate': finite(score.fp_rate),
            'latency_ms': {name: finite(getattr(latency, name)) for name in ('p10', 'p50', 'p90')}}


def paired_matches(baseline, phase):
    require(len(baseline) == len(phase), 'Anchor coverage differs')
    counts = {'common_detected': 0, 'gained': 0, 'lost': 0, 'missed_both': 0}
    paired = []
    for a, b in zip(baseline, phase):
        keys = ('ordinal', 'speaker', 'time_s', 'start', 'end')
        require(all(a[k] == b[k] for k in keys), 'Paired anchor/window identity differs')
        found_a = a['committed_s'] is not None; found_b = b['committed_s'] is not None
        status = ('common_detected' if found_a and found_b else 'lost' if found_a else 'gained' if found_b else 'missed_both')
        counts[status] += 1
        paired.append({**{k: a[k] for k in keys}, 'status': status,
                       'baseline_commit_s': a['committed_s'], 'phase_commit_s': b['committed_s'],
                       'baseline_latency_ms': a['latency_ms'], 'phase_latency_ms': b['latency_ms'],
                       'gain_ms': a['latency_ms']-b['latency_ms'] if status == 'common_detected' else None})
    gains = [p['gain_ms'] for p in paired if p['gain_ms'] is not None]
    return {**counts, 'median_gain_ms': float(np.median(gains)) if gains else None,
            'mean_gain_ms': float(np.mean(gains)) if gains else None, 'anchors': paired}


def advancement(baseline, phase, median_gain):
    valid_recall = baseline['recall'] is not None and phase['recall'] is not None
    tests = {'phase_fpr_at_most_0_10': phase['fp_rate'] is not None and phase['fp_rate'] <= .10,
             'recall_loss_at_most_0_005': valid_recall and phase['recall'] >= baseline['recall']-.005,
             'common_detected_median_gain_at_least_16ms': median_gain is not None and median_gain >= 16.}
    return {'passed': all(tests.values()), 'criteria': tests,
            'recall_delta_phase_minus_baseline': phase['recall']-baseline['recall'] if valid_recall else None,
            'interpretation': 'Predeclared development advancement screen only; no held-out quality claim or automatic deployment.'}


def verify_upstream(root):
    require(subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == UPSTREAM_COMMIT,
            'Pinned official scorer revision changed')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'], check=True, capture_output=True)
    return {name: sha(root/'turnbench'/name) for name in ('gold.py', 'score.py')}


def verified_cache(manifest, expected_arm, official_sources):
    from leakage_guard import validate_manifest
    audit = validate_manifest(manifest)
    require(audit['counts'] == {'train': 16, 'dev': 6, 'gate': 0}, 'Require original16/6 pilot; gate/test forbidden')
    root = manifest.resolve().parent
    rows = json.loads(manifest.read_text()); config = json.loads((root/'phase128-config.json').read_text())
    complete = json.loads((root/'complete.json').read_text()); config_hash = digest(config)
    require(complete.get('complete') is True and complete['manifest_sha256'] == sha(manifest) and
            complete['config_sha256'] == config_hash and complete['audit'] == audit, 'Incomplete/changed paired cache')
    require(config['variant'] == 'phase128-common-rows-paired-pilot-v1' and config['horizon_policy'] == HORIZON_POLICY,
            'Require complete-window censor protocol; older right-censored cache rejected')
    require(config['test_or_gate_data_loaded'] is False and config['official_sources'] == official_sources,
            'Cache source/scorer provenance differs')
    require(config['official_tolerances'] == {'tau_pre_s': .25, 'tau_max_s': 3.}, 'Official tolerance drift')
    require(config['dependencies'] == {name: importlib.metadata.version(name) for name in config['dependencies']},
            'Preparation runtime dependency versions changed')
    for name, expected in config['sources'].items():
        require(sha(Path(__file__).with_name(name)) == expected, 'Paired preparation source changed: '+name)
    proofs = {}
    for row in rows:
        require(row['split'] in ('train', 'dev') and row.get('encoder_checkpoint_sha256') is None, 'Non-pilot/frozen record')
        proof = json.loads((root/(row['id']+'.phase-provenance.json')).read_text())
        require(proof['config_sha256'] == config_hash and proof['arm'] == expected_arm and proof['output_record'] == row,
                'Record preparation provenance differs')
        require(proof['source_hashes'] == config['source_inventory'][row['id']], 'Source inventory mismatch')
        # Hash every artifact without parsing train arrays/events.
        for key in ('npz', 'events'):
            require(sha(root/row[key]) == row[key+'_sha256'], 'Paired cache bytes changed')
        proofs[row['id']] = proof
    return rows, proofs, config, audit


def load_checkpoint(path, manifest):
    import torch
    identity = sha(path); checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require(sha(path) == identity, 'Checkpoint changed while loading')
    config = checkpoint['config']
    require(config['manifest_sha256'] == sha(manifest), 'Checkpoint belongs to a different paired manifest')
    require(config.get('variant') == 'mlp' and config.get('seed') == 42 and config.get('epochs') == 12 and
            checkpoint.get('auxiliary') is False, 'Require prespecified plain MLP seed42/12epoch head')
    op = checkpoint['operating_point']
    require(np.isfinite(op['threshold']) and 0 <= op['threshold'] <= 1 and op['recommit_s'] in (None, 1.5), 'Invalid fixed operating point')
    return checkpoint, identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-manifest', 'phase-manifest', 'baseline-checkpoint', 'phase-checkpoint', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--turnbench-path', type=Path, default=Path('/content/turnbench'))
    args = ap.parse_args()
    if not Path('/content').is_dir(): raise RuntimeError('Remote CPU Colab only')
    if args.out.exists(): raise ValueError('Refusing to overwrite paired evidence')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    official_sources = verify_upstream(args.turnbench_path)
    sys.path.insert(0, str(args.turnbench_path))
    from turnbench import score as official
    from heads import load_records, make_model, predict, commit_events
    arms = {}
    for name, manifest, checkpoint_path, expected_arm in (
        ('baseline', args.baseline_manifest, args.baseline_checkpoint, 'baseline160'),
        ('phase', args.phase_manifest, args.phase_checkpoint, 'phase128')):
        rows, proofs, config, audit = verified_cache(manifest, expected_arm, official_sources)
        checkpoint, checkpoint_hash = load_checkpoint(checkpoint_path, manifest)
        records = load_records(manifest, 'dev')
        model = make_model(checkpoint['dim'], checkpoint['hidden'], False)
        model.load_state_dict(checkpoint['model'], strict=True); model.eval()
        probabilities = predict(model, records, checkpoint['mean'], checkpoint['std'], 'cpu')
        arms[name] = {'records': records, 'probabilities': probabilities, 'proofs': proofs,
                      'rows': rows, 'config': config, 'audit': audit, 'checkpoint': checkpoint, 'manifest': manifest,
                      'proof_sha256': {cid: sha(manifest.parent/(cid+'.phase-provenance.json')) for cid in proofs},
                      'checkpoint_sha256': checkpoint_hash, 'total': official.TaskScore()}
    a = arms['baseline']; b = arms['phase']
    require(a['config'] == b['config'], 'Arms were prepared under different configurations')
    require([(r['id'], r['split']) for r in a['rows']] == [(r['id'], r['split']) for r in b['rows']], 'Pair manifest coverage/order differs')
    ids = [r['id'] for r in a['records']]
    require(ids == [r['id'] for r in b['records']], 'Pair dev order differs')
    conversations = []; all_pairs = []
    for index, cid in enumerate(ids):
        ra = a['records'][index]; rb = b['records'][index]
        pa = a['proofs'][cid]; pb = b['proofs'][cid]
        for key in ('common_horizon_s', 'source_duration_s', 'mapping', 'censor_counts', 'raw_files', 'source_resample_delay_s'):
            require(pa[key] == pb[key], 'Paired preparation metadata differs: '+key)
        h = pa['common_horizon_s']
        require(0 < h <= pa['source_duration_s'] and len(rb['times']) == len(ra['times'])+1, 'Unexpected shared horizon/row counts')
        for record, proof in ((ra, pa), (rb, pb)):
            require(len(record['times']) == proof['output_rows'] and np.all(record['times'] <= h), 'Beyond-horizon decisions')
        require(np.array_equal(ra['times'], np.arange(1, len(ra['times'])+1)*2560/16000), 'Baseline grid drift')
        require(np.array_equal(rb['times'], (np.arange(1, len(rb['times'])+1)*2560-512)/16000), 'Phase128 grid drift')
        require(rb['times'][-1] == h and ra['events'] == rb['events'], 'Shared gold/horizon differs')
        gold = ra['events']
        require(all(0 <= w['start'] <= w['end'] <= h for w in windows(gold['eot_positive_events'], official.TAU_PRE_S, official.TAU_MAX_S)),
                'Incomplete positive matching window survived censor')
        require(all(0 <= v['start']-official.TAU_PRE_S <= v['end'] <= h for v in gold['eot_negative_spans']), 'Incomplete negative scoring window')
        predictions = {}; matched = {}; scores = {}
        for name, arm, record in (('baseline', a, ra), ('phase', b, rb)):
            op = arm['checkpoint']['operating_point']
            prediction = commit_events(record['times'], record['vad'], arm['probabilities'][index], op['threshold'], op['recommit_s'])
            score, matches = matched_score(gold, prediction, official)
            official.merge(arm['total'], score)
            predictions[name] = prediction; matched[name] = matches; scores[name] = summarize(score)
        paired = paired_matches(matched['baseline'], matched['phase'])
        all_pairs.extend([{**p, 'conversation_id': cid} for p in paired['anchors']])
        conversations.append({'id': cid, 'common_horizon_s': h, 'censor_counts': pa['censor_counts'],
                              'predictions': predictions, 'scores': scores, 'paired': paired})
    aggregate = {name: summarize(arm['total']) for name, arm in arms.items()}
    for name, arm in arms.items():
        expected = arm['checkpoint']['operating_point']['score']
        require(all(aggregate[name][k] == expected[k] for k in ('tp', 'fn', 'fp', 'tn')), 'Frozen operating-point score reproduction failed: '+name)
    gains = [p['gain_ms'] for p in all_pairs if p['gain_ms'] is not None]
    counts = {status: sum(p['status'] == status for p in all_pairs) for status in ('common_detected', 'gained', 'lost', 'missed_both')}
    median_gain = float(np.median(gains)) if gains else None
    from gate_evaluate import actor_components
    metadata = json.loads((args.baseline_manifest.resolve().parent.parent/'metadata.json').read_text())
    by_id = {r['_dir']: r for r in metadata}
    groups = actor_components(ids, {cid: [by_id[cid][f'speaker_{s}_actor_id'] for s in (1, 2)] for cid in ids})
    report = {'experiment': 'n13 phase128 matched-event pilot', 'development_only': True,
        'test_or_gate_or_public_benchmark_data_loaded': False, 'search_or_fitting_performed': False,
        'upstream_commit': UPSTREAM_COMMIT, 'official_sources': official_sources,
        'evaluator_sha256': sha(__file__), 'preparation_config_sha256': digest(a['config']),
        'artifacts': {name: {'checkpoint_sha256': arm['checkpoint_sha256'], 'audit': arm['audit'],
                            'fixed_operating_point': arm['checkpoint']['operating_point'],
                            'record_provenance_sha256': arm['proof_sha256'],
                            'complete_sha256': sha(arm['manifest'].parent/'complete.json')} for name, arm in arms.items()},
        'aggregate': aggregate, 'paired': {**counts, 'median_gain_ms': median_gain,
            'mean_gain_ms': float(np.mean(gains)) if gains else None, 'positive_gain_means': 'phase earlier on same detected anchor'},
        'advancement': advancement(aggregate['baseline'], aggregate['phase'], median_gain),
        'uncertainty': {'confidence_intervals': None, 'actor_connected_components': len(groups),
                        'component_sizes': [len(g) for g in groups],
                        'reason': 'Point estimates only: repeated actors and six development conversations do not establish independent event samples. No IID conversation/event bootstrap.'},
        'conversations': conversations,
        'limitations': ['Matched latency conditions on anchors detected by both heads; gained/lost anchors are reported separately.',
                       'Conservative shared complete-window censor may propagate through overlapping tail windows; counts are reported.',
                       'Each operating point was already selected on this development data; advancement is exploratory.']}
    from checkpointing import atomic_text
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), args.out)
    print(json.dumps({'event': 'PHASE128_PAIRED_DEV_COMPLETE', 'aggregate': aggregate,
                      'paired': report['paired'], 'advancement': report['advancement']}), flush=True)


if __name__ == '__main__': main()
