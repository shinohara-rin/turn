"""Stage the datasets into the `turnbench-datasets` volume (a fresh Modal workspace has none).

Downloads from Hugging Face with the `huggingface` Modal secret (HF_TOKEN; both datasets are
gated): otoSpeech metadata for every conversation (the actor split needs all of them), audio
and annotations only for the conversations prep will use, and TurnBench dev. TurnBench test
is never downloaded.

    modal run stage.py --train 131 --dev 16 --extra 24
    modal run prep.py::main --train 131 --dev 16 --extra 24
"""
import json

import modal

from common import OTO, TB_DEV, VOLUMES, datasets

app = modal.App('ssl-turn-stage')
image = (modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub==0.35.3', 'hf_transfer', 'numpy')
         .env({'HF_HUB_ENABLE_HF_TRANSFER': '1'}).add_local_python_source('common'))
OTO_REPO = 'otoearth/otoSpeech-full-duplex-turn-104h'
TB_REPO = 'mundo-ai/turn-benchmark-dev'
SECRETS = [modal.Secret.from_name('huggingface')]


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, cpu=2, memory=4096, timeout=3600)
def fetch(repo, local_dir, patterns):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type='dataset', local_dir=local_dir, allow_patterns=patterns, max_workers=16)
    datasets.commit()
    return len(patterns)


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=2048, timeout=600)
def plan(train, dev, extra):
    """prep.make_split's split and prep.extra_ids' donors, from the staged metadata."""
    import os
    import numpy as np
    datasets.reload()
    rows = []
    for d in sorted(os.listdir(OTO)):
        p = f'{OTO}/{d}/metadata.json'
        if os.path.exists(p):
            rows.append(dict(json.load(open(p)), _dir=d))
    actors = sorted({r[f'speaker_{s}_actor_id'] for r in rows for s in (1, 2)})
    rng = np.random.default_rng(20261007)
    rng.shuffle(actors)
    n = len(actors)
    assign = {a: ('train' if i < int(n * .6) else 'dev' if i < int(n * .8) else 'gate') for i, a in enumerate(actors)}
    import hashlib
    splits = {k: [] for k in ('train', 'dev', 'gate', 'excluded_cross_partition')}
    for r in rows:
        a, b = (assign[r[f'speaker_{s}_actor_id']] for s in (1, 2))
        splits[a if a == b else 'excluded_cross_partition'].append(r['_dir'])
    for k in splits:
        splits[k].sort(key=lambda x: hashlib.sha256(('pilot-v1:' + x).encode()).hexdigest())
    by = {r['_dir']: r for r in rows}
    no_gate = [c for c in splits['excluded_cross_partition']
               if 'gate' not in {assign[by[c][f'speaker_{s}_actor_id']] for s in (1, 2)}]
    return dict(counts={k: len(v) for k, v in splits.items()},
                cids=splits['train'][:train] + splits['dev'][:dev] + no_gate[:extra])


@app.local_entrypoint()
def main(train: int = 131, dev: int = 16, extra: int = 24, groups: int = 8):
    oto_dir = OTO
    tb_dir = TB_DEV.rsplit('/data', 1)[0]
    meta = fetch.spawn(OTO_REPO, oto_dir, ['*/metadata.json'])
    tb = fetch.spawn(TB_REPO, tb_dir, ['data/*', '*.md', '*.json'])
    meta.get()
    p = plan.remote(train, dev, extra)
    print('split', p['counts'], '->', len(p['cids']), 'conversations to fetch')
    pats = [[f'{c}/speaker_*' for c in p['cids'][i::groups]] for i in range(groups)]
    print(sum(fetch.starmap([(OTO_REPO, oto_dir, g) for g in pats])), 'conversations staged')
    tb.get()
    print('TurnBench dev staged')
