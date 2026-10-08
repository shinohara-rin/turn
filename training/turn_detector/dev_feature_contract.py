"""Pure validation contracts for public-dev encoder feature reuse."""
import numpy as np
from benchmark_infer import digest,require_equal,sha

MODE='derived_dev_features_v1'
ATOL=1e-6
RTOL=1e-5


def validate_features(arrays,probability_arrays,duration,decision_phase_ms=160):
    from score_benchmark_dev import validate_arrays
    if set(arrays)!={'times','features','vad','extras'}:raise ValueError('Unexpected encoder feature fields')
    n=len(arrays['times'])
    if arrays['features'].shape!=(n,2,512) or arrays['features'].dtype!=np.float16:
        raise ValueError('Expected float16 Parakeet encoder features [T,2,512]')
    if arrays['extras'].shape!=(n,2,1):raise ValueError('Expected energy extras [T,2,1]')
    if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite encoder feature cache')
    if not np.array_equal(arrays['times'],probability_arrays['times']) or not np.array_equal(arrays['vad'],probability_arrays['vad']):
        raise ValueError('Feature/probability grid or history mismatch')
    # Check generic grid/ranges; caller additionally checks actual serving policy.
    validate_arrays(probability_arrays,'last',duration,decision_phase_ms)


def validate_parent(identity):
    from score_benchmark_dev import validate_dev_identity
    if identity['freeze'].get('execution_mode') is not None:raise ValueError('Reuse requires original end-to-end inference')
    freeze=validate_dev_identity(identity)
    if freeze['protocol'].get('dev_feature_cache') is not True:raise ValueError('Parent lacks explicit dev feature export')
    return freeze


def validate_derived(identity):
    freeze=identity['freeze']
    if freeze.get('execution_mode')!=MODE:raise ValueError('Unknown derived execution mode')
    derivation=freeze['derivation'];parent=derivation['parent_identity'];pf=validate_parent(parent)
    require_equal(derivation['parent_identity_digest'],digest(parent),'Derived parent identity')
    require_equal(identity['dataset'],parent['dataset'],'Derived dataset')
    require_equal(identity['ids'],parent['ids'],'Derived coverage')
    require_equal(freeze['base_model'],pf['base_model'],'Derived base encoder')
    require_equal(freeze['continued_encoder'],pf['continued_encoder'],'Derived continued encoder')
    require_equal(freeze['dependencies'],pf['dependencies'],'Derived dependencies')
    validate_reuse_sources(pf['sources'],freeze['sources'])
    for field in ('source','revision','split','history_vad','grid_samples','sample_rate','gate'):
        require_equal(freeze['protocol'][field],pf['protocol'][field],'Derived '+field)
    for field in ('grid_start_samples','decision_phase_contract'):
        require_equal(freeze['protocol'].get(field),pf['protocol'].get(field),'Derived '+field)
    require_equal(freeze.get('phase_qualification_sha256'),pf.get('phase_qualification_sha256'),'Derived phase qualification')
    inventory=derivation['feature_inventory']
    if set(inventory)!=set(parent['ids']) or any(not all(row.get(k) for k in
            ('features_sha256','probabilities_sha256','transaction_sha256')) for row in inventory.values()):
        raise ValueError('Incomplete parent feature inventory')
    replay=derivation['baseline_replay']
    if replay.get('passed') is not True or replay.get('events_exact') is not True or replay.get('checkpoint')!=pf['checkpoint']:
        raise ValueError('Missing successful baseline head replay')
    if replay.get('atol')!=ATOL or replay.get('rtol')!=RTOL:raise ValueError('Replay tolerance changed')
    if set(replay.get('conversations',{}))!=set(parent['ids']):raise ValueError('Incomplete replay coverage')
    if derivation['head_audit'].get('manifest_sha256')!=freeze['feature_manifest'] or derivation['head_audit'].get('passed') is not True:
        raise ValueError('Missing guarded head training manifest')


def replay_check(actual,expected,actual_prediction,expected_prediction):
    np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=RTOL)
    require_equal(actual_prediction,expected_prediction,'Baseline replay events')
    return float(np.max(np.abs(actual-expected)))


def validate_reuse_sources(saved,current):
    names = ['heads.py','temporal_head.py','ensemble_heads.py','encoder.py','vad_batch.py','benchmark_infer.py']
    if 'experiment/phase128_full_arrays.py' in saved:
        names.append('phase128_full_arrays.py')
    for name in names:
        require_equal(saved['experiment/'+name],current['experiment/'+name],'Feature/inference source '+name)


def checked_npz(path,expected):
    require_equal(sha(path),expected,'Parent array hash')
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    require_equal(sha(path),expected,'Parent array during load')
    return arrays
