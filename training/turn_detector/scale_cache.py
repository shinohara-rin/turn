"""Metadata-only full train/dev selection and remote cache/dataset reuse helpers.

Never selects gate/test or excluded cross-partition conversations. No downloads
occur here. The CLI derives an explicit frozen selection; extraction is separate.
"""
from pathlib import Path
import argparse
import json
from cache_identity import sha256_file, validate_cache_identity
from leakage_guard import TRAIN_SOURCE, TRAIN_REVISION

VAD_PROTOCOL = 'silero-mean-five-causal-32ms-v1'
DATA_FILES = tuple(f'speaker_{s}_{suffix}' for s in (1, 2) for suffix in ('audio.wav', 'annotation_a.srt'))


def validate_selection(records, split_path):
    split_path = Path(split_path)
    plan = json.loads(split_path.read_text())
    metadata = json.loads((split_path.parent/'metadata.json').read_text())
    if plan.get('revision') != TRAIN_REVISION:
        raise ValueError('Unapproved frozen source revision')
    by_id = {str(row['_dir']): row for row in metadata}
    if len(by_id) != len(metadata):
        raise ValueError('Duplicate source metadata identities')
    seen = set(); result = []
    for record in records:
        cid, part = str(record['id']), record['split']
        if not cid.isdigit() or cid in seen:
            raise ValueError('Duplicate or invalid conversation identity')
        if part not in ('train', 'dev'):
            raise ValueError('Only train/dev extraction is allowed; gate/test prohibited')
        if record.get('source', TRAIN_SOURCE) != TRAIN_SOURCE or record.get('revision', TRAIN_REVISION) != TRAIN_REVISION:
            raise ValueError('Unapproved selection source or revision')
        if cid not in plan['splits'][part] or cid not in by_id:
            raise ValueError('Conversation not in frozen split/source metadata')
        for speaker in (1, 2):
            actor = by_id[cid][f'speaker_{speaker}_actor_id']
            if plan['assignments'][actor] != part:
                raise ValueError('Conversation crosses frozen actor partitions')
        seen.add(cid)
        result.append({'id': cid, 'split': part, 'source': TRAIN_SOURCE, 'revision': TRAIN_REVISION})
    if not result:
        raise ValueError('Empty selection')
    return result


def full_selection(split_path):
    plan = json.loads(Path(split_path).read_text())
    records = [{'id': cid, 'split': part} for part in ('train', 'dev') for cid in plan['splits'][part]]
    return validate_selection(records, split_path)


def verified_drive_directory(dataset_root, cid):
    """Use only complete pinned HF local-dir files, checked via HF commit metadata.

    No raw dataset is copied. Missing/unverified files cause the caller's bounded
    per-conversation HF fallback. HF local-dir metadata first line is commit SHA.
    """
    root = Path(dataset_root)
    directory = root/cid
    if not all((directory/name).is_file() for name in DATA_FILES):
        return None
    for name in DATA_FILES:
        metadata = root/'.cache'/'huggingface'/'download'/cid/(name+'.metadata')
        if not metadata.is_file():
            return None
        lines = metadata.read_text().splitlines()
        if len(lines) < 3 or lines[0] != TRAIN_REVISION:
            return None
        try:
            if (directory/name).stat().st_mtime-1 > float(lines[2]):
                return None
        except ValueError:
            return None
    return directory


def cleanup_created_fallback(directory):
    """Called only for a directory absent before this run's remote HF fallback."""
    directory = Path(directory)
    for name in DATA_FILES:
        (directory/name).unlink(missing_ok=True)
    try:
        directory.rmdir()
    except OSError:
        pass  # Never remove unexpected files from another process.


def reuse_verified_cache(manifest_path, out, selection, identity, encoder_source_hash, split_path, policy_records=None):
    """Reuse immutable pilot arrays by reference; copy only small provenance JSON.

    Checks selected ids, split, encoder identity/code, gate protocol and both file
    hashes before emitting records. Pilot files are never changed or relinked.
    """
    from leakage_guard import validate_manifest
    manifest_path = Path(manifest_path); out = Path(out)
    if manifest_path.parent.resolve() == out.resolve():
        raise ValueError('Reuse source must differ from output directory')
    validate_manifest(manifest_path, split_path=split_path)
    approved = {(r['id'], r['split']) for r in selection}
    reused = []
    for record in json.loads(manifest_path.read_text()):
        if (record['id'], record['split']) not in approved:
            continue
        provenance_path = manifest_path.parent/f"{record['id']}.provenance.json"
        provenance = json.loads(provenance_path.read_text())
        prior_identity = provenance.get('cache_assembly_identity', provenance)
        if identity.get('encoder_checkpoint_sha256'):
            # Expanding the policy manifest and IO entrypoint must not invalidate
            # identical encoder weights/features. Verify every substantive field;
            # preserve old extractor provenance and record the new assembly below.
            stable_identity = {k:v for k,v in identity.items()
                               if k not in ('cache_entrypoint_sha256','policy_cache_manifest_sha256')}
            validate_cache_identity(prior_identity, stable_identity)
            prior_policy = prior_identity.get('policy_cache_manifest_sha256')
            if prior_policy != identity.get('policy_cache_manifest_sha256'):
                policy = (policy_records or {}).get(record['id'])
                if policy is None or policy.get('reused_from_manifest_sha256') != prior_policy:
                    raise ValueError('Expanded policy cache lacks verified ancestry to original pilot policy')
                for key in ('npz','events'):
                    if sha256_file(policy[key]) != policy[key+'_sha256']:
                        raise ValueError('Expanded policy ancestor checksum mismatch')
        else:
            validate_cache_identity(prior_identity, identity)
        if record.get('encoder_checkpoint_sha256') != identity.get('encoder_checkpoint_sha256'):
            raise ValueError('Reuse manifest encoder identity mismatch')
        if provenance.get('encoder_source_sha256') != encoder_source_hash:
            raise ValueError('Reuse encoder source changed')
        # Original audited pilot omitted a protocol field; its fixed policy was
        # exactly five-frame mean. Other explicit policies are never accepted.
        if provenance.get('vad_protocol', VAD_PROTOCOL) != VAD_PROTOCOL:
            raise ValueError('Reuse VAD policy changed')
        copied = dict(record)
        for key in ('npz', 'events'):
            source = (manifest_path.parent/record[key]).resolve()
            if sha256_file(source) != record[key+'_sha256']:
                raise ValueError('Reuse file checksum mismatch')
            copied[key] = str(source)
        destination = out/provenance_path.name
        assembled = {**provenance, 'cache_assembly_identity': identity,
                     'reused_from_provenance_sha256': sha256_file(provenance_path),
                     'reused_from_manifest_sha256': sha256_file(manifest_path)}
        if destination.exists():
            if json.loads(destination.read_text()) != assembled:
                raise ValueError('Existing reused provenance changed')
        else:
            destination.write_text(json.dumps(assembled,indent=2))
        copied['reused_from_manifest_sha256'] = sha256_file(manifest_path)
        reused.append(copied)
    return reused


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = full_selection(args.split)
    with args.out.open('x') as handle:
        json.dump(records, handle, indent=2)
    print(json.dumps({'selection': str(args.out), 'sha256': sha256_file(args.out),
                      'split_sha256': sha256_file(args.split),
                      'counts': {p: sum(r['split'] == p for r in records) for p in ('train', 'dev')},
                      'gate_or_test_selected': False}), flush=True)


if __name__ == '__main__':
    main()
