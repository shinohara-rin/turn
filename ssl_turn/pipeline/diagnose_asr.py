"""Would an ASR encoder (word content) help the floor head? Probe test against Cat and MTD.

Same decision events, readouts and equal-data split as diagnose.py (r012 cache: 19 oto
train / 4 selection conversations among the 23 with MTD features, TurnBench dev for report).
For each event time t (readout frame i, features ending at (i+1) * 80 ms), each channel's
last WINDOW_S of 16 kHz audio ending at t is encoded and the final 80 ms frame is kept, plus
the mean of the last 13 frames (1.04 s), for a middle layer and the encoder output. Nothing
after t is seen, so the features are causal at the readout.

Encoders (NeMo FastConformer, 80 ms frames):
  tdt:    nvidia/parakeet-tdt-0.6b-v2, full attention inside the trailing window.
  stream: nvidia/stt_en_fastconformer_hybrid_large_streaming_multi at attention context
          [70, 0] (no lookahead): the cache-aware streaming encoder, causal by construction.

Probes (linear / MLP, picked on the 4 selection conversations) on: Cat head input, MTD, each
ASR encoder, and Cat + ASR concatenated (the fusion question).

    modal run diagnose_asr.py
"""
import json

import modal

from common import VOLUMES, work

WORK = '/work'

app = modal.App('ssl-turn-diag-asr')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1', 'scipy')
         .add_local_dir(str(__import__('pathlib').Path(__file__).resolve().parent.parent), '/root/ssl_turn',
                        ignore=['**/__pycache__'])
         .add_local_python_source('common'))
ENCODERS = {'tdt': 'nvidia/parakeet-tdt-0.6b-v2',
            'stream': 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'}
WINDOW_S = 16.0
FRAME = 1280  # 80 ms at 16 kHz
MEAN_FRAMES = 13


def to16k(x24):
    """24 -> 16 kHz, one-sided FIR as in encode_mtd.to16k (no lookahead)."""
    import numpy as np
    from scipy.signal import firwin, upfirdn
    up, down = 2, 3
    half = 10 * max(up, down)
    taps = firwin(2 * half + 1, 1 / max(up, down), window=('kaiser', 5.0)) * up
    count = (len(x24) * up + down - 1) // down
    return upfirdn(taps, x24.astype(np.float32), up=up, down=down, axis=0)[:count].astype(np.float32)


def auc(y, s):
    import numpy as np
    from scipy.stats import rankdata
    y, s = np.asarray(y), np.asarray(s, np.float64)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def fpr_at(y, s, tpr=0.9):
    import numpy as np
    y, s = np.asarray(y), np.asarray(s)
    th = np.quantile(s[y == 1], 1 - tpr)
    return float((s[y == 0] >= th).mean())


def train_probe(Xtr, ytr, Xsel, ysel, kind, wd, dev='cuda', epochs=60, seed=0):
    """As diagnose.train_probe: class-balanced BCE, early stopping on selection AUC."""
    import torch
    torch.manual_seed(seed)
    D = Xtr.shape[1]
    net = (torch.nn.Linear(D, 1) if kind == 'linear' else
           torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(D, 512), torch.nn.GELU(),
                               torch.nn.Dropout(0.3), torch.nn.Linear(512, 1))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    pos = ytr.float().mean()
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).clamp(max=50))
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for a in range(0, len(Xtr), 1024):
            idx = perm[a:a + 1024]
            loss = lossf(net(Xtr[idx]).squeeze(-1), ytr[idx].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                a_ = auc(ysel.cpu().numpy(), net(Xsel).squeeze(-1).float().cpu().numpy())
            if a_ > best:
                best, best_state = a_, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def encode_events(name, keys, batch=48):
    """keys: sorted unique (split_dir, cid, i). Returns {key: [2, 2, 2, D]} fp16 as
    (channel, now/mean, mid/final layer)."""
    import time
    import numpy as np
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(ENCODERS[name], map_location='cuda').eval()
    if name == 'stream':
        model.encoder.set_default_att_context_size([70, 0])
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    layers = model.encoder.layers
    mid = {}
    layers[len(layers) // 2].register_forward_hook(lambda m, i, o: mid.__setitem__('x', o))
    W = int(WINDOW_S * 16000)
    out, t0, audio = {}, time.time(), {}
    jobs = [(k, c) for k in keys for c in (0, 1)]
    for b in range(0, len(jobs), batch):
        chunk = jobs[b:b + batch]
        wins, lens = [], []
        for (d, cid, i), c in chunk:
            if (d, cid) not in audio:
                audio.clear() if len(audio) > 4 else None
                a24 = np.load(f'{WORK}/audio/{d}/{cid}.npy').astype(np.float32)
                audio[(d, cid)] = [to16k(a24[:, ch]) for ch in (0, 1)]
            x = audio[(d, cid)][c]
            e = min((i + 1) * FRAME, len(x))
            w = x[max(0, e - W):e]
            wins.append(w)
            lens.append(len(w))
        L = max(lens)
        sig = torch.zeros(len(wins), L)
        for k, w in enumerate(wins):
            sig[k, :len(w)] = torch.from_numpy(w)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            feats, flen = model.preprocessor(input_signal=sig.cuda(), length=torch.tensor(lens).cuda())
            enc, elen = model.encoder(audio_signal=feats, length=flen)
        fin = enc.transpose(1, 2).float()          # [B, T, D]
        md = mid['x'].float()                      # [B, T, D]
        for k, ((key, c), n) in enumerate(zip(chunk, elen.tolist())):
            lo = max(0, n - MEAN_FRAMES)
            v = torch.stack([torch.stack([md[k, n - 1], fin[k, n - 1]]),
                             torch.stack([md[k, lo:n].mean(0), fin[k, lo:n].mean(0)])])  # [now/mean, layer, D]
            out.setdefault(key, [None, None])[c] = v.half().cpu().numpy()
        if b % (batch * 50) == 0:
            print(f'{name}: {b}/{len(jobs)} windows, {time.time() - t0:.0f}s', flush=True)
    return {k: np.stack(v) for k, v in out.items()}


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=8, memory=49152, timeout=7200)
def run(run='r012_fine', encoders=('stream', 'tdt')):
    import os
    import time
    import numpy as np
    import torch
    t0 = time.time()
    z = np.load(f'{WORK}/runs/diag/cache_{run}.npz', allow_pickle=True)
    events, F, M = list(z['events']), z['F'], z['M']
    has_mtd = np.array([x['mtd'] for x in events])
    sel = np.where(has_mtd)[0]
    mpos = {i: k for k, i in enumerate(sel)}
    ev = [events[i] for i in sel]
    keys = sorted({('tbdev' if x['split'] == 'tbdev' else 'oto', x['cid'], int(x['i'])) for x in ev})
    print(f'{len(ev)} events, {len(keys)} readouts', flush=True)
    os.makedirs(f'{WORK}/runs/diag', exist_ok=True)
    asr = {}
    for name in encoders:
        path = f'{WORK}/runs/diag/asr_{name}.npz'
        if os.path.exists(path):
            zz = np.load(path, allow_pickle=True)
            asr[name] = dict(zip(map(tuple, zz['keys'].tolist()), zz['X']))
        else:
            enc = encode_events(name, keys)
            np.savez(path, keys=np.array([list(k) for k in enc], dtype=object), X=np.stack(list(enc.values())))
            work.commit()
            asr[name] = enc
            torch.cuda.empty_cache()

    def asr_feat(name, layer):
        """[n, 4 * D]: own now, other now, own mean, other mean (as diagnose.gather)."""
        rows = []
        for x in ev:
            v = asr[name][('tbdev' if x['split'] == 'tbdev' else 'oto', x['cid'], int(x['i']))]
            c = x['c']
            ls = [0, 1] if layer == 'both' else [layer]
            rows.append(np.concatenate([v[c, 0, ls].ravel(), v[1 - c, 0, ls].ravel(),
                                        v[c, 1, ls].ravel(), v[1 - c, 1, ls].ravel()]))
        return np.stack(rows).astype(np.float32)

    tap_cols = np.concatenate([np.arange(k * 1280, (k + 1) * 1280) for k in (1, 2, 3)] +
                              [np.arange(4 * 1280, 4 * 1280 + 768)])  # head input: taps 15/23/31 + final
    cat = F[sel][..., tap_cols].reshape(len(sel), -1).astype(np.float32)
    mtd = M[[mpos[i] for i in sel]].reshape(len(sel), -1).astype(np.float32)
    feats = {'cat': cat, 'mtd': mtd}
    for name in asr:
        feats[name] = asr_feat(name, 'both')
        feats[f'{name}_final'] = asr_feat(name, 1)
        feats[f'cat+{name}'] = np.concatenate([cat, feats[name]], 1)
    feats['cat+mtd'] = np.concatenate([cat, mtd], 1)

    mtd_ids = sorted(set(json.load(open(f'{WORK}/mtd_train_ids.json'))))
    sel_ids = set(mtd_ids[::6])
    sp = np.array([x['split'] for x in ev])
    cid = np.array([x['cid'] for x in ev])
    task = np.array([x['task'] for x in ev])
    y = np.array([x['y'] for x in ev])
    cats = [x['cat'] for x in ev]
    m_train = (sp == 'train') & ~np.isin(cid, list(sel_ids))
    m_sel = (sp == 'train') & np.isin(cid, list(sel_ids))
    m_tb = sp == 'tbdev'
    res = {}
    for t in ('eot24', 'eot48', 'int40'):
        mt = task == t
        for fname, X in feats.items():
            Xt = torch.from_numpy(X).cuda()
            tr, se = np.where(mt & m_train)[0], np.where(mt & m_sel)[0]
            mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-3
            Xt = (Xt - mu) / sd
            yt = torch.from_numpy(y).cuda()
            for kind in ('linear', 'mlp'):
                best = (-1, None, None)
                for wd in ((1e-2, 1e-1, 1.0) if kind == 'linear' else (1e-2, 1e-1)):
                    net, a = train_probe(Xt[tr], yt[tr], Xt[se], yt[se], kind, wd)
                    if a > best[0]:
                        best = (a, wd, net)
                with torch.no_grad():
                    s = torch.cat([best[2](Xt[i:i + 4096]).squeeze(-1) for i in range(0, len(Xt), 4096)])
                s = s.float().cpu().numpy()
                r = dict(wd=best[1])
                for nm, m in (('train', mt & m_train), ('sel', mt & m_sel), ('tbdev', mt & m_tb)):
                    r[nm] = dict(auc=auc(y[m], s[m]), fpr90=fpr_at(y[m], s[m]))
                if t.startswith('eot'):  # long holds vs all ends, TB dev
                    m = mt & m_tb & ((y == 1) | np.array(['long_hold' in c for c in cats]))
                    r['tbdev_long_hold'] = dict(auc=auc(y[m], s[m]), fpr90=fpr_at(y[m], s[m]))
                res[f'{t}/{fname}/{kind}'] = r
                print(t, fname, kind, json.dumps({k: v for k, v in r.items() if k != 'wd'}), flush=True)
            del Xt
            torch.cuda.empty_cache()
    json.dump(res, open(f'{WORK}/runs/diag/diag_asr.json', 'w'), indent=1)
    work.commit()
    res['wall_s'] = time.time() - t0
    return res


@app.local_entrypoint()
def main(encoders: str = 'stream,tdt'):
    res = run.remote(encoders=tuple(encoders.split(',')))
    print(f'wall {res.pop("wall_s"):.0f}s')
    for t in ('eot24', 'eot48', 'int40'):
        print(f'\n== {t}  (AUC train / sel / TB dev' + (' / TB long holds)' if t != 'int40' else ')'))
        for k, v in res.items():
            if k.startswith(t + '/'):
                cells = [v['train']['auc'], v['sel']['auc'], v['tbdev']['auc']]
                if 'tbdev_long_hold' in v:
                    cells.append(v['tbdev_long_hold']['auc'])
                print(f'{k[len(t) + 1:]:28s} ' + ' '.join(f'{c:.3f}' for c in cells))
