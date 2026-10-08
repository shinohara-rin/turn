"""Small pure-Python provenance contracts for frozen/continued feature caches."""
from pathlib import Path
import hashlib


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_continuation(state, expected_audit, source_hash, model_repo, model_revision):
    if state.get('format_version') != 1:
        raise ValueError('Unsupported continuation checkpoint format')
    if state.get('model_repo') != model_repo or state.get('model_revision') != model_revision:
        raise ValueError('Continuation base model revision mismatch')
    if state.get('source_sha256') != source_hash:
        raise ValueError('Continuation training source changed')
    if state.get('audit') != expected_audit:
        raise ValueError('Continuation training audit or frozen speaker split changed')
    if expected_audit.get('loaded_partitions') != ['train'] or expected_audit.get('test_data_loaded') is not False:
        raise ValueError('Continuation training provenance is not train-only')
    if not isinstance(state.get('update'), int) or state['update'] < 1:
        raise ValueError('Continuation checkpoint has no completed update')
    if not state.get('encoder'):
        raise ValueError('Continuation checkpoint has no encoder state')


def validate_cache_identity(provenance, identity):
    """Legacy frozen caches remain valid; continued lineage must match exactly."""
    if provenance.get('encoder_checkpoint_sha256') != identity.get('encoder_checkpoint_sha256'):
        raise ValueError('Cache mixes frozen/different continued encoder weights')
    if identity.get('encoder_checkpoint_sha256') is not None:
        for key, value in identity.items():
            if provenance.get(key) != value:
                raise ValueError('Continued cache provenance mismatch: '+key)
