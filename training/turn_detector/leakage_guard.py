"""Fail closed before any fitted component can open data arrays or labels."""
from pathlib import Path
import hashlib,json
TRAIN_SOURCE='otoearth/otoSpeech-full-duplex-turn-104h'
TRAIN_REVISION='46f520297f434edf804389f82f9075a59d2f8268'

def validate_manifest(manifest, split_path=None):
    manifest=Path(manifest)
    records=json.loads(manifest.read_text())
    split_path=Path(split_path) if split_path else manifest.parent.parent/'split.json'
    plan=json.loads(split_path.read_text())
    if plan['revision'] != TRAIN_REVISION:
        raise ValueError('Unapproved training revision')
    metadata=json.loads((split_path.parent/'metadata.json').read_text())
    by_id={r['_dir']:r for r in metadata}
    seen=set();actors={p:set() for p in ('train','dev','gate')};counts={p:0 for p in actors}
    for r in records:
        if r.get('source') != TRAIN_SOURCE:
            raise ValueError('Benchmark or unidentified source prohibited from training manifest')
        if r.get('revision') != TRAIN_REVISION:
            raise ValueError('Source revision must be pinned')
        split=r['split'];cid=r['id']
        if split not in actors or cid not in plan['splits'][split]:
            raise ValueError('Conversation not in frozen speaker split')
        if cid in seen:
            raise ValueError('Duplicate conversation')
        seen.add(cid)
        row=by_id[cid]
        for s in (1,2):
            actor=row[f'speaker_{s}_actor_id']
            if plan['assignments'][actor] != split:
                raise ValueError('Actor assigned to another partition')
            actors[split].add(actor)
        counts[split]+=1
    for a,b in [('train','dev'),('train','gate'),('dev','gate')]:
        if actors[a] & actors[b]:
            raise ValueError(f'Speaker leakage: {a}/{b}')
    return {'passed':True,'source':TRAIN_SOURCE,'revision':TRAIN_REVISION,'counts':counts,
            'speaker_counts':{k:len(v) for k,v in actors.items()},
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'test_data_loaded':False}
