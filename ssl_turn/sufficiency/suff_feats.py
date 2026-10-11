"""Pull cached streaming FastConformer frames at each sufficiency candidate (Modal CPU, no GPU).

For candidate (conversation, speaker, t): the speaker's channel frame that ends at or before t,
the other channel's frame, and the mean of the speaker's last 1 s, from
{WORK}/feats_asr/{tbdev,oto}/{cid}.npy (float16 [T, 2, 1024], 80 ms frames, causal encoder).

    cd ssl_turn/pipeline && modal run ../sufficiency/suff_feats.py --labels suff_dev.json --out suff_feats.npz
"""
import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from common import VOLUMES, WORK  # noqa: E402

app = modal.App('ssl-turn-suff-feats')
image = modal.Image.debian_slim(python_version='3.12').pip_install('numpy').add_local_python_source('common')


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=16384, timeout=1800)
def pull(items):
    import numpy as np
    out, cache = [], {}
    for cid, s, t in items:
        corpus, c = cid.split('/', 1)
        path = f'{WORK}/feats_asr/{"tbdev" if corpus == "tb" else "oto"}/{c}.npy'
        if path not in cache:
            cache.clear()
            cache[path] = np.load(path, mmap_mode='r')
        f = cache[path]
        i = min(len(f), int(t / 0.08)) - 1
        own, oth = f[i, s - 1].astype(np.float32), f[i, 2 - s].astype(np.float32)
        last = f[max(0, i - 12):i + 1, s - 1].astype(np.float32).mean(0)
        out.append(np.concatenate([own, oth, last]).astype(np.float16))
    return np.stack(out)


@app.local_entrypoint()
def main(labels: str, out: str = 'suff_feats.npz'):
    import numpy as np
    d = json.load(open(labels))
    keys = sorted(d)
    items = [(cid, x[0], x[1]) for cid in keys for x in d[cid]]
    chunks = [[it for it in items if it[0] == cid] for cid in keys]
    X = np.concatenate(list(pull.map(chunks)))
    np.savez_compressed(out, X=X, cid=np.array([i[0] for i in items]), s=np.array([i[1] for i in items]),
                        t=np.array([i[2] for i in items]))
    print('wrote', out, X.shape)
