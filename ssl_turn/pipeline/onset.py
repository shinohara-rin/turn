"""Onset classifier for INT: what kind of vocalisation has just started?

After turn-1-mini's own-channel head. For each VAD vocalisation (policy.speech_segments on
vad.py output) and each decision time t = onset + k (k = 0.16 ... 1.2 s), an MLP reads causal
inputs and predicts the vocalisation's class:
  none (no annotator label), turn, floor_taking, attempt (non-floor-taking interruption),
  backchannel, noncontent  (labels.FINE_GROUPS; majority fine label in [onset, onset + 1 s]).
Inputs at frame j (the 12.5 Hz frame known at t = (j + 1) * 80 ms): streaming FastConformer
features (feats_asr) of the own channel at j, their mean from the onset frame to j, the other
channel at j, and voice-activity context (time since onset, own and other speech fractions,
silence before the onset, the other speaker's run length at the onset).

Trained on otoSpeech train (TurnBench is never used for training), early-stopped on oto dev.
Writes a TB dev INT track p(floor_taking) for policy_score.py:
  {WORK}/runs/{run}/probs.npz  key '{model}/tbdev/{cid}/int_onset'  [T, 2], 0 outside windows.

    modal run onset.py --run onset_r1
"""
import json

import modal

from common import VOLUMES, WORK, gpu_image, setup_path, work

app = modal.App('ssl-turn-onset')
GROUPS = ('none', 'turn', 'floor_taking', 'attempt', 'backchannel', 'noncontent')
FPS = 12.5
K_TRAIN = (0.16, 0.24, 0.32, 0.4, 0.56, 0.72, 0.96, 1.2)
K_MAX = 1.2
N_SCALARS = 8


def vocalisations(vad_u8, vad_th=0.7, min_gap=0.2):
    from policy import speech_segments
    p = vad_u8.astype('float32') / 255
    return [speech_segments(p[:, c], vad_th, min_gap) for c in (0, 1)], p


def scalars(p, segs, c, a, prev_end, t):
    """Causal voice-activity context at time t for a vocalisation of channel c starting at a."""
    import numpy as np
    hop = 0.032
    now = int(t / hop)                    # chunks fully known by t
    win = lambda x, s: float(x[max(int(s / hop), 0):now].mean()) if now > int(s / hop) else 0.0
    own, oth = p[:, c] > 0.5, p[:, 1 - c] > 0.5
    other_run = 0.0
    for s, e in segs[1 - c]:
        if s <= a <= e + 0.2:
            other_run = min(a - s, 5.0)
            break
        if s > a:
            break
    return np.array([t - a, win(own, a), win(oth, a), win(oth, t - 2.0), float(oth[max(now - 3, 0):now].any()),
                     min(a - prev_end, 5.0) if prev_end is not None else 5.0, other_run,
                     float(other_run > 0)], np.float32)


def label_of(fine, times, c, a, b):
    """Group index of the vocalisation from per-frame fine labels (labels.FINE indices)."""
    import numpy as np
    import labels as lb
    m = (times >= a) & (times < min(b, a + 1.0))
    x = fine[m, c]
    x = x[x > 0]
    if not len(x):
        return 0
    top = np.bincount(x).argmax()
    for g, name in enumerate(GROUPS[1:], 1):
        if top in lb.FINE_GROUPS[name]:
            return g
    return 0


def rows_for(feats, vad_u8, ks=K_TRAIN, every=False):
    """Decision points for one conversation. Returns X [N, 3 * D + N_SCALARS] fp16 and meta
    rows (channel, onset, end, frame j)."""
    import numpy as np
    segs, p = vocalisations(vad_u8)
    T, _, D = feats.shape
    X, meta = [], []
    for c in (0, 1):
        prev_end = None
        for a, b in segs[c]:
            j0 = int(a * FPS)
            if every:
                js = range(int(np.ceil((a + 0.16) * FPS)) - 1, int(np.floor(min(b + 1 / FPS, a + K_MAX) * FPS)))
            else:
                js = sorted({int(np.ceil((a + k) * FPS)) - 1 for k in ks if a + k <= b + 1 / FPS})
            csum = None
            for j in js:
                if j >= T or j < j0:
                    continue
                t = (j + 1) / FPS
                own = feats[j0:j + 1, c].astype(np.float32)
                x = np.concatenate([feats[j, c].astype(np.float32), own.mean(0), feats[j, 1 - c].astype(np.float32),
                                    scalars(p, segs, c, a, prev_end, t)])
                X.append(x.astype(np.float16))
                meta.append((c, a, b, j))
            prev_end = b
    D3 = 3 * D + N_SCALARS
    return (np.stack(X) if X else np.zeros((0, D3), np.float16)), meta


def build(split, cid, every=False):
    import numpy as np
    setup_path()
    feats = np.load(f'{WORK}/feats_asr/{split}/{cid}.npy', mmap_mode='r')
    vad = np.load(f'{WORK}/vad/{split}/{cid}.npy')
    X, meta = rows_for(feats, vad, every=every)
    y = np.zeros(len(meta), np.int64)
    if split == 'oto':
        z = np.load(f'{WORK}/labels/oto/{cid}.npz')
        lab = {}
        for i, (c, a, b, j) in enumerate(meta):
            if (c, a) not in lab:
                lab[c, a] = label_of(z['fine'], z['times'], c, a, b)
            y[i] = lab[c, a]
    return X, y, meta, feats.shape[0]


class MLP:
    @staticmethod
    def make(d_in, hidden=256, n_out=len(GROUPS), dropout=0.2):
        import torch.nn as nn
        return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))


def auc(pos, neg):
    import numpy as np
    from scipy.stats import rankdata
    if not len(pos) or not len(neg):
        return float('nan')
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@app.function(image=gpu_image, volumes=VOLUMES, gpu='L4', cpu=8, memory=65536, timeout=5400)
def fit(run_name='onset_r1', seeds=(1, 2), epochs=8, hidden=256, lr=1e-3):
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    setup_path()
    t0 = time.time()
    split = json.load(open(f'{WORK}/split.json'))['splits']
    tb = sorted(n[:-4] for n in os.listdir(f'{WORK}/vad/tbdev'))
    with ThreadPoolExecutor(8) as ex:
        tr = list(ex.map(lambda c: build('oto', c), split['train']))
        dv = list(ex.map(lambda c: build('oto', c), split['dev']))
        tbd = list(ex.map(lambda c: build('tbdev', c, True), tb))
    Xtr = torch.from_numpy(np.concatenate([r[0] for r in tr]))
    ytr = torch.from_numpy(np.concatenate([r[1] for r in tr]))
    Xdv = torch.from_numpy(np.concatenate([r[0] for r in dv]))
    ydv = np.concatenate([r[1] for r in dv])
    kdv = np.array([(j + 1) / FPS - a for r in dv for (c, a, b, j) in r[2]])
    counts = np.bincount(ytr.numpy(), minlength=len(GROUPS))
    print(f'rows train {len(ytr)} dev {len(ydv)}; train classes', dict(zip(GROUPS, counts.tolist())),
          f'{time.time() - t0:.0f}s', flush=True)
    dev = 'cuda'
    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(GROUPS), dtype=torch.float32, device=dev).clamp(max=50)

    def predict(net, X, bs=8192):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                out.append(torch.softmax(net(X[i:i + bs].to(dev).float()), -1).cpu())
        net.train()
        return torch.cat(out).numpy() if out else np.zeros((0, len(GROUPS)))

    def dev_auc(P):  # floor_taking vs attempt + backchannel, at k <= 0.4 s
        ft = P[:, GROUPS.index('floor_taking')]
        m = kdv <= 0.4 + 1e-6
        pos = ft[m & (ydv == GROUPS.index('floor_taking'))]
        neg = ft[m & np.isin(ydv, [GROUPS.index('attempt'), GROUPS.index('backchannel')])]
        return auc(pos, neg)

    report, tracks = {}, {}
    for seed in seeds:
        torch.manual_seed(seed)
        net = MLP.make(Xtr.shape[1], hidden).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
        best, best_state = -1, None
        n, bs = len(ytr), 1024
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = torch.nn.functional.cross_entropy(net(Xtr_d[idx].float()), ytr_d[idx], weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
            a = dev_auc(predict(net, Xdv))
            print(f'seed {seed} epoch {ep} loss {loss.item():.3f} oto dev AUC(ft vs bc+attempt, k<=0.4) {a:.3f}',
                  flush=True)
            if a > best:
                best, best_state = a, {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        report[f'mlp_s{seed}'] = dict(oto_dev_auc=best)
        for cid, (X, _, meta, T) in zip(tb, tbd):
            P = predict(net, torch.from_numpy(X))[:, GROUPS.index('floor_taking')]
            tr_ = np.zeros((T, 2), np.float16)
            for (c, a, b, j), v in zip(meta, P):
                tr_[j, c] = v
            tracks[f'mlp_s{seed}/tbdev/{cid}/int_onset'] = tr_
    os.makedirs(f'{WORK}/runs/{run_name}', exist_ok=True)
    np.savez(f'{WORK}/runs/{run_name}/probs.npz', **tracks)
    json.dump(report, open(f'{WORK}/runs/{run_name}/train.json', 'w'), indent=1)
    work.commit()
    return report


@app.local_entrypoint()
def main(run: str = 'onset_r1', epochs: int = 8):
    print(fit.remote(run, epochs=epochs))
