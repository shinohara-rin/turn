"""One-shot, frozen otoSpeech speaker gate; NOT official TurnBench evaluation.

prepare freezes metadata/weights/dev rationale without opening gate audio/labels.
run requires that freeze and a persistent ledger; no tuning or fitting exists.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import re
import sys
import types
import numpy as np
from benchmark_infer import (sha, digest, require_equal, source_lock, dependency_lock,
    model_kind, predictor, operating_point, verify_continued, MODEL_REPO, MODEL_REVISION, MODEL_FILE, validate_manifest_phase)
from checkpointing import atomic_text, _atomic_write
from leakage_guard import TRAIN_SOURCE, TRAIN_REVISION


def write_json(path, value):
    atomic_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), path)


def gate_membership(plan, metadata):
    if plan.get('revision') != TRAIN_REVISION:
        raise ValueError('Wrong frozen otoSpeech revision')
    rows = {r['_dir']: r for r in metadata}
    if len(rows) != len(metadata):
        raise ValueError('Duplicate metadata conversation')
    ids = plan['splits']['gate']
    if len(ids) != 20 or len(set(ids)) != len(ids):
        raise ValueError('Exactly the original 20 gate conversations required')
    seen = set(); actors = {part: set() for part in ('train', 'dev', 'gate')}
    for part in actors:
        for cid in plan['splits'][part]:
            if cid in seen or cid not in rows or not re.fullmatch(r'[A-Za-z0-9_-]+', cid):
                raise ValueError('Duplicate, missing, or unsafe split conversation')
            seen.add(cid)
            for speaker in (1, 2):
                actor = rows[cid][f'speaker_{speaker}_actor_id']
                if plan['assignments'][actor] != part:
                    raise ValueError('Actor does not belong to frozen partition')
                actors[part].add(actor)
    if any(actors[a] & actors[b] for a, b in [('train', 'dev'), ('train', 'gate'), ('dev', 'gate')]):
        raise ValueError('Actor leakage across partitions')
    return {'ids': ids, 'actors': {cid: [rows[cid][f'speaker_{s}_actor_id'] for s in (1, 2)] for cid in ids}}


def fixed_policies(policies, rationale):
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('Written selection rationale using dev evidence required')
    if len(policies) < 2 or sum(p.get('role') == 'selected' for p in policies) != 1:
        raise ValueError('Exactly one selected candidate and at least one fixed baseline required')
    names = [p['name'] for p in policies]
    if len(set(names)) != len(names) or not all(re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in names):
        raise ValueError('Unique safe policy names required')
    for policy in policies:
        if policy['role'] not in ('selected', 'baseline'):
            raise ValueError('Only selected and baseline roles permitted')
        if set(policy['operating_point']) != {'threshold', 'recommit_s'}:
            raise ValueError('Operating point must be fixed, without search options')
        operating_point(policy['operating_point'])


def runtime_identity():
    # Match qualification's import precedence before freezing runtime identity.
    if sys.path[0] != '/content/turnbench': sys.path.insert(0, '/content/turnbench')
    # Match initialized NeMo/Lhotse import precedence in every runtime snapshot.
    from nemo.collections.asr.models import ASRModel
    from evaluate_phase128_pair import verify_upstream
    official_sources = verify_upstream(Path('/content/turnbench'))
    import torch
    import silero_vad
    torch.set_num_threads(2)
    return {'sources': source_lock('/content/turnbench'), 'dependencies': dependency_lock(), 'official_sources': official_sources,
            'silero': sha(Path(silero_vad.__file__).parent / 'data/silero_vad.jit'),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device': torch.cuda.get_device_name(0), 'threads': 2,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32}


def policy_artifacts(policy, plan):
    import torch
    state_hash = sha(policy['checkpoint'])
    state = torch.load(policy['checkpoint'], map_location='cpu', weights_only=False)
    require_equal(state_hash, sha(policy['checkpoint']), 'Checkpoint while loading')
    args = types.SimpleNamespace(**{k: policy.get(k) for k in ('encoder_checkpoint', 'encoder_identity', 'encoder_causality')})
    continued, encoder_identity = verify_continued(args, state)
    manifest_hash = sha(policy['feature_manifest'])
    members = [m['checkpoint'] for m in state['members']] if model_kind(state) == 'ensemble' else [state]
    for member in members:
        require_equal(member['config']['manifest_sha256'], manifest_hash, 'Head feature manifest')
    rows = json.loads(Path(policy['feature_manifest']).read_text())
    expected_encoder = encoder_identity['checkpoint'] if encoder_identity else None
    if not rows or any(r['split'] not in ('train', 'dev') or r.get('source') != TRAIN_SOURCE or
                       r.get('revision') != TRAIN_REVISION or r.get('encoder_checkpoint_sha256') != expected_encoder or
                       r['id'] not in plan['splits'].get(r['split'], []) for r in rows):
        raise ValueError('Head lineage must contain only approved train/dev records and matching encoder')
    grid = policy_grid(policy, rows, members)
    return {'checkpoint': state_hash, 'feature_manifest': manifest_hash, 'kind': model_kind(state),
            'continued': encoder_identity, 'grid': grid}, state, ({'encoder': continued['encoder']} if continued is not None else None)


def policy_grid(policy, records, members):
    phase = policy.get('decision_phase_ms', 160)
    if type(phase) is not int or phase not in (160, 128):
        raise ValueError('Only explicit160 or full-source128 decision grids are supported')
    validate_manifest_phase(records, phase)
    for member in members:
        config = member['config']
        if config.get('decision_phase_ms', phase) != phase or config.get('grid_samples', 2560) != 2560:
            raise ValueError('Checkpoint grid conflicts with policy/manifest')
        contract = 'phase128-full-v1' if phase == 128 else None
        if (config.get('grid_start_samples', phase*16) != phase*16 or config.get('sample_rate', 16000) != 16000 or
                config.get('decision_phase_contract', contract) != contract or config.get('phase_grid_contract', contract) != contract):
            raise ValueError('Checkpoint decision start/sample rate/contract differs')
    if (phase == 128) != bool(policy.get('phase_qualification')):
        raise ValueError('Full phase128 requires runtime qualification; default160 must omit it')
    return {'decision_phase_ms': phase, 'grid_samples': 2560, 'grid_start_samples': phase*16,
            'decision_phase_contract': 'phase128-full-v1' if phase == 128 else None}


def gate_protocol(artifacts):
    return {'batch_size': 1, 'grid_samples': 2560, 'sample_rate': 16000,
            'policy_grids': {name: value['grid'] for name, value in artifacts.items()},
            'history_gate': 'mean', 'feature_dtype': 'float16',
            'bootstrap_seed': 20261007, 'bootstrap_replicates': 2000,
            'bootstrap_unit': 'actor-connected-component', 'minimum_ci_components': 5,
            'labels': 'prepare_data single-annotator published floor; identical full gold across grids'}


def phase_qualification_identity(runtime, base_hash, artifact):
    from phase128_qualification import qualification_identity
    numerical = {k: runtime[k] for k in ('device', 'cuda', 'cudnn', 'tf32_matmul', 'tf32_cudnn')}
    return qualification_identity({'base_model': {'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'sha256': base_hash},
        'continued_encoder': artifact['continued'], 'silero_weights': runtime['silero'],
        'sources': runtime['sources'], 'dependencies': runtime['dependencies'],
        'protocol': {**numerical, 'batch_size': 1, 'torch_threads': runtime['threads']}})


def qualify_policies(policies, artifacts, runtime, base_hash):
    """Validate existing synthetic evidence before any gate ledger/content access."""
    for policy in policies:
        artifact = artifacts[policy['name']]
        if artifact['grid']['decision_phase_ms'] == 128:
            from phase128_qualification import validate_qualification
            if not runtime['sources'].get('experiment/phase128_full_arrays.py'):
                raise ValueError('Full phase128 helper source missing from frozen runtime')
            path = Path(policy['phase_qualification']); before = sha(path)
            report = json.loads(path.read_text())
            validate_qualification(report, phase_qualification_identity(runtime, base_hash, artifact))
            require_equal(before, sha(path), 'Phase qualification while loading')
            artifact['phase_qualification_sha256'] = before
        else:
            artifact['phase_qualification_sha256'] = None


def decision_grid(samples, phase):
    if phase not in (160, 128): raise ValueError('Unknown decision phase')
    return np.arange(phase*16, samples+1, 2560, dtype=np.int64)


def inputs_for_grid(wave, raw, phase):
    """Same causal VAD/energy construction as qualified benchmark inference."""
    ends = decision_grid(len(wave), phase)
    if not len(ends): raise ValueError('Empty gate decision grid')
    if phase == 128:
        from phase128_full_arrays import inputs_at_decisions
        vad, energy = inputs_at_decisions(wave, raw, ends)
        last = raw[ends//512-1]
    else:
        if len(wave) % 2560 or raw.shape != (len(wave)//512, 2):
            raise ValueError('Baseline inputs must use complete original160ms blocks')
        n = len(ends); blocks = raw.reshape(n, 5, 2)
        vad = blocks.mean(1); last = blocks[:, -1]
        energy = np.log(np.sqrt((wave.reshape(n, 2560, 2)**2).mean(axis=1))+1e-7)[..., None]
    return {'times': ends/16000, 'vad': vad, 'vad_last': last, 'energy': energy}


def remote_metadata(config, membership):
    """Metadata only: no gate file download or content access during freeze."""
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(TRAIN_SOURCE, revision=TRAIN_REVISION, files_metadata=True)
    require_equal(info.sha, TRAIN_REVISION, 'Source revision')
    entries = {e.rfilename: e for e in info.siblings}
    files = {}
    for cid in membership['ids']:
        for s in (1, 2):
            for suffix in ('audio.wav', 'annotation_a.srt'):
                name = f'{cid}/speaker_{s}_{suffix}'; entry = entries[name]
                lfs = entry.lfs
                value = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
                if value:
                    files[name] = {'algorithm': 'sha256', 'hash': value}
                elif entry.blob_id:
                    files[name] = {'algorithm': 'git-blob-sha1', 'hash': entry.blob_id}
                else:
                    raise ValueError('Missing pinned file identity')
    info = api.model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    require_equal(info.sha, MODEL_REVISION, 'Base revision')
    entry = next(e for e in info.siblings if e.rfilename == MODEL_FILE)
    lfs = entry.lfs
    value = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    if not value:
        raise ValueError('Base model hash unavailable')
    require_equal(sha(config['base_model']), value, 'Pinned base encoder')
    return files


def prepare(config_path, freeze_path):
    config = json.loads(Path(config_path).read_text())
    fixed_policies(config['policies'], config['selection_rationale_dev_only'])
    if config['gate'] not in ('mean', 'last'):
        raise ValueError('One common fixed serving gate required')
    if not config.get('dev_evidence'):
        raise ValueError('Dev selection evidence paths required')
    membership = gate_membership(json.loads(Path(config['split']).read_text()), json.loads(Path(config['metadata']).read_text()))
    plan = json.loads(Path(config['split']).read_text())
    artifacts = {p['name']: policy_artifacts(p, plan)[0] for p in config['policies']}
    runtime = runtime_identity(); base_hash = sha(config['base_model'])
    qualify_policies(config['policies'], artifacts, runtime, base_hash)
    freeze = {'format': 'otospeech-gate-once-v2', 'config': config, 'membership': membership,
              'source': TRAIN_SOURCE, 'revision': TRAIN_REVISION, 'official_turnbench': False,
              'split_sha256': sha(config['split']), 'metadata_sha256': sha(config['metadata']),
              'dev_evidence': {str(p): sha(p) for p in config['dev_evidence']},
              'artifacts': artifacts, 'base_model_sha256': sha(config['base_model']),
              'files': remote_metadata(config, membership), 'runtime': runtime,
              'protocol': gate_protocol(artifacts)}
    if Path(freeze_path).exists():
        raise ValueError('Freeze already exists; refusing overwrite')
    write_json(freeze_path, freeze)
    print('GATE_FREEZE_WRITTEN_NO_GATE_CONTENT_OPENED', flush=True)


def verify_gate_file(path, expected):
    if expected['algorithm'] == 'sha256':
        actual = sha(path)
    elif expected['algorithm'] == 'git-blob-sha1':
        h = hashlib.sha1(f'blob {Path(path).stat().st_size}\0'.encode())
        with Path(path).open('rb') as file:
            for block in iter(lambda: file.read(1024*1024), b''):
                h.update(block)
        actual = h.hexdigest()
    else:
        raise ValueError('Unapproved file hash algorithm')
    require_equal(expected['hash'], actual, 'Gate file '+str(path))


def ledger_open(path, freeze_hash, out, resume):
    """Caller holds an exclusive advisory lock for the complete evaluation."""
    identity = {'freeze_sha256': freeze_hash, 'output': str(Path(out).resolve()),
                'purpose': 'one fixed otoSpeech gate evaluation; not official TurnBench',
                'official_turnbench_accessed': False}
    path = Path(path)
    if path.exists():
        row = json.loads(path.read_text())
        require_equal(row['identity'], identity, 'Gate access ledger')
        if not resume:
            raise ValueError('Gate already opened; only exact interrupted run may resume')
    else:
        if resume:
            raise ValueError('No gate access ledger to resume')
        row = {'identity': identity, 'accessed': [], 'completed': [], 'status': 'started'}
        write_json(path, row)
    return row


def cached_npz(path, identity):
    marker = Path(str(path)+'.json')
    if not marker.exists():
        return None
    evidence = json.loads(marker.read_text())
    require_equal(evidence['identity'], identity, 'Gate cache identity')
    require_equal(evidence['sha256'], sha(path), 'Gate cache bytes')
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k] for k in value.files}


def save_npz(path, identity, **arrays):
    _atomic_write(path, lambda file: np.savez_compressed(file, **arrays))
    write_json(str(path)+'.json', {'identity': identity, 'sha256': sha(path)})


def score_prediction(events, prediction):
    from turnbench.score import score_task
    from turnbench.gold import AnchorEvent, Interval
    return asdict(score_task([AnchorEvent(**v) for v in events['eot_positive_events']],
        [Interval(**v) for v in events['eot_negative_spans']], prediction,
        [Interval(**v) for v in events.get('eot_excluded', [])]))


def summarize(scores):
    counts = {k: sum(s[k] for s in scores) for k in ('tp', 'fn', 'fp', 'tn')}
    latencies = [v for s in scores for v in s['latencies_ms']]
    counts.update(recall=counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None,
                  fp_rate=counts['fp']/(counts['fp']+counts['tn']) if counts['fp']+counts['tn'] else None,
                  latency_ms={f'p{p}': float(np.percentile(latencies, p)) if latencies else None for p in (10, 50, 90)})
    return counts


def actor_components(ids, actors):
    """Union conversations sharing either actor, including transitive chains."""
    if len(set(ids)) != len(ids) or set(ids) != set(actors):
        raise ValueError('Cluster membership must cover each conversation exactly once')
    parent = list(range(len(ids)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, cid in enumerate(ids):
        if len(actors[cid]) != 2:
            raise ValueError('Exactly two actor identities required')
        for actor in actors[cid]:
            if actor in seen:
                parent[find(i)] = find(seen[actor])
            else:
                seen[actor] = i
    groups = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def paired_report(rows, policies, ids, actors, seed=20261007, replicates=2000):
    if len(rows) != len(ids):
        raise ValueError('Score and actor membership alignment mismatch')
    names = [p['name'] for p in policies]
    selected = next(p['name'] for p in policies if p['role'] == 'selected')
    output = {'aggregate': {n: summarize([r[n] for r in rows]) for n in names}, 'paired_differences': {}}
    groups = actor_components(ids, actors)
    enough = len(groups) >= 5
    rng = np.random.default_rng(seed)
    # Identical sampled actor-connected components for every paired policy.
    draws = ([i for cluster in draw for i in groups[cluster]]
             for draw in rng.integers(0, len(groups), size=(replicates if enough else 0, len(groups))))
    draws = list(draws)
    def metrics(s): return {'recall': s['recall'], 'fp_rate': s['fp_rate'], 'latency_p50_ms': s['latency_ms']['p50']}
    output['aggregate_95_intervals'] = {}
    for name in names:
        samples = {key: [] for key in metrics(output['aggregate'][name])}
        for draw in draws:
            values = metrics(summarize([rows[i][name] for i in draw]))
            for key, value in values.items():
                if value is not None:
                    samples[key].append(value)
        output['aggregate_95_intervals'][name] = {key: {
            'percentile_95_interval': np.percentile(v, [2.5, 97.5]).tolist() if v else None,
            'valid_replicates': len(v)} for key, v in samples.items()}
    for baseline in (n for n in names if n != selected):
        values = {key: [] for key in metrics(output['aggregate'][selected])}
        for draw in draws:
            a = metrics(summarize([rows[i][selected] for i in draw]))
            b = metrics(summarize([rows[i][baseline] for i in draw]))
            for key in values:
                if a[key] is not None and b[key] is not None:
                    values[key].append(a[key]-b[key])
        a = metrics(output['aggregate'][selected]); b = metrics(output['aggregate'][baseline])
        output['paired_differences'][baseline] = {key: {
            'selected_minus_baseline': a[key]-b[key] if a[key] is not None and b[key] is not None else None,
            'percentile_95_interval': np.percentile(v, [2.5, 97.5]).tolist() if v else None,
            'valid_replicates': len(v)} for key, v in values.items()}
    output['uncertainty'] = {'unit': 'paired actor-connected-component cluster bootstrap', 'seed': seed,
        'requested_replicates': replicates, 'independent_components': len(groups),
        'component_sizes': [len(g) for g in groups], 'components': [[ids[i] for i in g] for g in groups],
        'confidence_intervals_suppressed': not enough,
        'reason': None if enough else 'Insufficient independent groups: fewer than five actor-connected components.',
        'limitation': 'Cluster bootstrap assumes actor-connected groups are independent. Latency conditions on each model\'s detected positives. No model selection or gate retuning permitted.'}
    return output


def run(freeze_path, resume):
    import torch
    import fcntl
    freeze = json.loads(Path(freeze_path).read_text()); config = freeze['config']
    require_equal(freeze['format'], 'otospeech-gate-once-v2', 'Freeze format')
    require_equal(freeze['source'], TRAIN_SOURCE, 'Source'); require_equal(freeze['revision'], TRAIN_REVISION, 'Revision')
    fixed_policies(config['policies'], config['selection_rationale_dev_only'])
    require_equal(freeze['split_sha256'], sha(config['split']), 'Split')
    require_equal(freeze['metadata_sha256'], sha(config['metadata']), 'Metadata')
    membership = gate_membership(json.loads(Path(config['split']).read_text()), json.loads(Path(config['metadata']).read_text()))
    require_equal(freeze['membership'], membership, 'Gate membership')
    require_equal(freeze['runtime'], runtime_identity(), 'Frozen runtime/code')
    require_equal(freeze['base_model_sha256'], sha(config['base_model']), 'Base model')
    require_equal(freeze['dev_evidence'], {str(p): sha(p) for p in config['dev_evidence']}, 'Dev selection evidence')
    states = {}; continuations = {}; verified_artifacts = {}
    for policy in config['policies']:
        artifact, state, continuation = policy_artifacts(policy, json.loads(Path(config['split']).read_text()))
        verified_artifacts[policy['name']] = artifact
        states[policy['name']] = state
        key = artifact['continued']['checkpoint'] if artifact['continued'] else 'base'
        continuations[key] = continuation
    qualify_policies(config['policies'], verified_artifacts, freeze['runtime'], freeze['base_model_sha256'])
    require_equal(freeze['artifacts'], verified_artifacts, 'Frozen candidate artifacts/grids/qualifications')
    require_equal(freeze['protocol'], gate_protocol(verified_artifacts), 'Frozen gate protocol')
    if config['gate'] not in ('mean', 'last'):
        raise ValueError('Invalid common gate')
    out = Path(config['out']); ledger = Path(config['access_ledger'])
    if not str(out.resolve()).startswith('/content/drive/') or not str(ledger.resolve()).startswith('/content/drive/'):
        raise ValueError('Gate output and access ledger must persist on mounted Drive')
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('Fresh gate run requires an empty output directory')
    out.mkdir(parents=True, exist_ok=True); ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(ledger)+'.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze_hash = sha(freeze_path)
    ledger_row = ledger_open(ledger, freeze_hash, out, resume)
    if (out/'freeze.json').exists():
        require_equal(json.loads((out/'freeze.json').read_text()), freeze, 'Output freeze')
    else:
        write_json(out/'freeze.json', freeze)
    from encoder import causal_resample, ParakeetStreamingEncoder
    from vad_batch import causal_vad_batch
    from heads import causal_history, speaker_features, commit_events
    from turnbench.submission import ConversationPrediction, SpeakerEvents, validate_event_times
    import soundfile as sf
    import prepare_data
    # The shared label builder accepts the verified remote directory directly.
    dataset = Path(config['dataset_dir']).resolve()
    if not str(dataset).startswith('/content/drive/'):
        raise ValueError('Gate source must be staged on mounted Drive')
    encoders = {}; predictors = {}; results = []; phase_detector = None
    for cid in membership['ids']:
        folder = out/cid; folder.mkdir(exist_ok=True)
        identity = {'freeze_sha256': freeze_hash, 'id': cid, 'actors': membership['actors'][cid]}
        final = folder/'predictions.json'
        if final.exists():
            value = json.loads(final.read_text())
            require_equal(value['identity'], identity, 'Conversation identity')
            require_equal(value['payload_sha256'], digest(value['payload']), 'Conversation predictions')
            require_equal(value['transaction_sha256'], digest({k: v for k, v in value.items() if k != 'transaction_sha256'}), 'Conversation transaction')
            for rel, expected in value['cache_hashes'].items():
                require_equal(sha(folder/rel), expected, 'Completed cache')
            results.append(value['payload']['scores'])
            if cid not in ledger_row['completed']:
                ledger_row['completed'].append(cid); write_json(ledger, ledger_row)
            continue
        if cid not in ledger_row['accessed']:
            ledger_row['accessed'].append(cid); write_json(ledger, ledger_row)
        # All four source files verified after ledger entry, before parsing any.
        for s in (1, 2):
            for suffix in ('audio.wav', 'annotation_a.srt'):
                rel = f'{cid}/speaker_{s}_{suffix}'
                verify_gate_file(dataset/rel, freeze['files'][rel])
        channels = [sf.read(dataset/cid/f'speaker_{s}_audio.wav', dtype='float32') for s in (1, 2)]
        if channels[0][1] != channels[1][1] or channels[0][0].shape != channels[1][0].shape or channels[0][0].ndim != 1:
            raise ValueError('Stereo source alignment mismatch')
        duration = len(channels[0][0])/channels[0][1]
        wave, _ = causal_resample(np.stack([v[0] for v in channels], axis=1), channels[0][1])
        source_end = len(channels[0][0])*16000//channels[0][1]
        if len(wave) < source_end: raise ValueError('Resampler returned incomplete source coverage')
        wave = wave[:source_end]
        shared_grids = {}; waves = {}; events = None
        phases = sorted({a['grid']['decision_phase_ms'] for a in freeze['artifacts'].values()})
        for phase in phases:
            waves[phase] = wave if phase == 128 else wave[:len(wave)//2560*2560]
            phase_wave = waves[phase]
            shared_path = folder/f'shared-phase{phase}.npz'
            shared_identity = {**identity, 'decision_phase_ms': phase}
            shared = cached_npz(shared_path, shared_identity)
            if shared is None:
                if phase == 128:
                    from phase128_full_arrays import native_silero_all
                    if phase_detector is None:
                        from silero_vad import load_silero_vad
                        phase_detector = load_silero_vad().eval()
                    raw = native_silero_all(phase_wave, detector=phase_detector)
                else:
                    raw = causal_vad_batch([phase_wave], reduction='raw')[0]
                shared = inputs_for_grid(phase_wave, raw, phase)
                gold, _ = prepare_data.gold_and_activity(cid, shared['times'], directory=dataset/cid)
                shared['events_json'] = np.asarray(json.dumps(gold))
                save_npz(shared_path, shared_identity, **shared)
            if not np.array_equal(shared['times'], decision_grid(source_end, phase)/16000):
                raise ValueError('Shared cache decision grid differs from full source duration')
            gold = json.loads(str(shared['events_json']))
            if events is not None: require_equal(gold, events, 'Full gold/exclusions across decision grids')
            events = gold; shared_grids[phase] = shared
        predictions = {}; scores = {}
        for policy in config['policies']:
            name = policy['name']; artifact = freeze['artifacts'][name]
            key = artifact['continued']['checkpoint'] if artifact['continued'] else 'base'
            phase = artifact['grid']['decision_phase_ms']; shared = shared_grids[phase]; times = shared['times']
            phase_wave = waves[phase]
            feature_path = folder/f'features-{key}-phase{phase}.npz'
            feature_identity = {**identity, 'encoder': key, 'grid': artifact['grid']}
            feature = cached_npz(feature_path, feature_identity)
            if feature is None:
                if key not in encoders:
                    from nemo.collections.asr.models import ASRModel
                    model = ASRModel.restore_from(config['base_model'], map_location='cpu')
                    if continuations[key] is not None:
                        model.encoder.load_state_dict(continuations[key]['encoder'], strict=True)
                    encoders[key] = ParakeetStreamingEncoder(model=model, device='cuda')
                options = {'decision_samples': [decision_grid(len(phase_wave), phase)]} if phase == 128 else {}
                sequence = encoders[key].extract_batch([phase_wave], **options)[0]
                if not np.array_equal(sequence.available_at_s, times):
                    raise ValueError('Encoder changed causal availability grid')
                feature = {'features': sequence.features.astype(np.float16), 'times': sequence.available_at_s}
                save_npz(feature_path, feature_identity, **feature)
            if not np.array_equal(feature['times'], times): raise ValueError('Encoder cache grid differs from policy inputs')
            if name not in predictors:
                predictors[name] = predictor(states[name], 'cuda')
            record = {'id': cid, 'times': times, 'vad': shared['vad'],
                      'x': speaker_features(feature['features'], causal_history(times, shared['vad']), shared['energy'])}
            probability = predictors[name]([record])[0]
            gate = shared['vad'] if config['gate'] == 'mean' else shared['vad_last']
            op = operating_point(policy['operating_point'])
            prediction = commit_events(times, gate, probability, op['threshold'], op['recommit_s'])
            row = ConversationPrediction(conversation_id=cid, speaker_1=SpeakerEvents(eot=prediction[1], interruption=[]),
                                         speaker_2=SpeakerEvents(eot=prediction[2], interruption=[]))
            validate_event_times(row, duration)
            predictions[name] = row.model_dump(); scores[name] = score_prediction(events, prediction)
        payload = {'predictions': predictions, 'scores': scores, 'duration_s': duration}
        caches = {p.name: sha(p) for p in folder.glob('*.npz*') if p.is_file()}
        transaction = {'identity': identity, 'payload': payload, 'payload_sha256': digest(payload), 'cache_hashes': caches}
        transaction['transaction_sha256'] = digest(transaction)
        write_json(final, transaction)
        results.append(scores); ledger_row['completed'].append(cid); write_json(ledger, ledger_row)
        print(json.dumps({'gate_completed': len(results), 'total': len(membership['ids'])}), flush=True)
    report = paired_report(results, config['policies'], membership['ids'], membership['actors'])
    report.update(decision_grids=freeze['protocol']['policy_grids'], freeze_sha256=freeze_hash, official_turnbench=False, source=TRAIN_SOURCE, revision=TRAIN_REVISION,
                  selection_rationale_dev_only=config['selection_rationale_dev_only'], ids=membership['ids'],
                  per_conversation=[{'id': cid, 'scores': row} for cid, row in zip(membership['ids'], results)])
    write_json(out/'report.json', report)
    ledger_row.update(status='complete', report_sha256=sha(out/'report.json')); write_json(ledger, ledger_row)
    print('FROZEN_OTOSPEECH_GATE_COMPLETE_NOT_OFFICIAL_TURNBENCH', flush=True)


def main():
    if not Path('/content/turnbench/turnbench').is_dir():
        raise RuntimeError('Prepared Colab only')
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', required=True); p.add_argument('--freeze', required=True)
    p = sub.add_parser('run'); p.add_argument('--freeze', required=True); p.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.command == 'prepare': prepare(args.config, args.freeze)
    else: run(args.freeze, args.resume)


if __name__ == '__main__': main()
