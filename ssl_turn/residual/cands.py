"""Candidate-level dataset for a residual EOT model: one row per (VAD offset, probe delay).

A candidate is a speech offset t of speaker s (policy.speech_segments, vad_th 0.5, gaps < 0.2 s
merged). It is probed at t + d for d in DELAYS (d < 0.2 only for eot-bench, where pauses are given) while s is still silent (s's next vocalisation
starts after t + d). Labels, from TurnBench-format gold (TB dev 3-annotator, otoSpeech 1-annotator):

  1   a gold EOT anchor of s lies in [t - 0.3, t + 0.8] (anchors sit at the VAD offset: median
      +0.04 s, 90% within +0.38 s on TB dev)
  0   t + d falls inside a scored EOT negative span of s ([start - 0.25, end]) and in no positive
      window, i.e. a fire there would be a false positive
  -1  neither (a fire there is free under the scorer); not used for training

Features, all known at t + d (FastConformer frame i is known at (i + 1) / 12.5 s):
  own frame at the probe, mean of own frames over the last 1 s before t, other's frame at the
  probe (1024 each), and scalars: d, own vocalisation length, other active at the probe, other's
  current vocalisation length, time since other last spoke, own previous pause length.

    python cands.py            # writes $DATA/cands/{oto,tbdev}.npz
"""
import numpy as np

import data
from policy import speech_segments

DELAYS = tuple(round(0.1 * i, 1) for i in range(1, 16)) + (2.0, 2.5)  # eot-bench grid is 0.1 s
FPS = 12.5
VAD_TH, MIN_GAP = 0.5, 0.2


def frame_at(tau, T):
    """Index of the last encoder frame known at time tau."""
    return int(np.clip(np.floor(tau * FPS) - 1, 0, T - 1))


def conv_rows(feats, vad, g):
    """Rows for one conversation: (X [n, 3072] f16, S [n, k] f32, meta [n, 5], y [n])."""
    T = feats.shape[0]
    segs = [speech_segments(vad[:, s], VAD_TH, MIN_GAP) for s in (0, 1)]
    ev = g['events']
    X, S, M, Y = [], [], [], []
    for s in (0, 1):
        own, oth = segs[s], segs[1 - s]
        anchors = np.array(sorted(e['time_s'] for e in ev['eot_positive_events'] if e['speaker'] == s + 1))
        negs = [(e['start'] - 0.25, e['end']) for e in ev['eot_negative_spans'] if e['speaker'] == s + 1]
        pos_w = [(a - 0.25, a + 3.0) for a in anchors]
        for k, (a0, t) in enumerate(own):
            nxt = own[k + 1][0] if k + 1 < len(own) else np.inf
            prev_gap = a0 - own[k - 1][1] if k else 10.0
            is_pos = bool(len(anchors)) and bool(((anchors >= t - 0.3) & (anchors <= t + 0.8)).any())
            i0, i1 = frame_at(t - 1.0, T), frame_at(t, T)
            tail = feats[i0:i1 + 1, s].astype(np.float32).mean(0)
            for d in DELAYS:
                tau = t + d
                if nxt <= tau or tau > g['duration_s']:
                    break
                if is_pos:
                    y = 1
                elif any(lo <= tau <= hi for lo, hi in negs) and not any(lo <= tau <= hi for lo, hi in pos_w):
                    y = 0
                else:
                    y = -1
                # other speaker's state at tau, causally (a vocalisation is ongoing until b + MIN_GAP)
                act, cur, since = 0.0, 0.0, 10.0
                for b0, b1 in oth:
                    if b0 > tau:
                        break
                    if tau <= b1 + MIN_GAP:
                        act, cur, since = 1.0, tau - b0, 0.0
                    else:
                        since = min(tau - b1, 10.0)
                i = frame_at(tau, T)
                X.append(np.concatenate([feats[i, s], tail.astype(np.float16), feats[i, 1 - s]]))
                S.append([d, min(t - a0, 10.0), act, min(cur, 5.0), since, min(prev_gap, 10.0)])
                M.append([s + 1, t, d, k, 0])
                Y.append(y)
    if not X:
        return None
    return (np.stack(X).astype(np.float16), np.array(S, np.float32), np.array(M, np.float32),
            np.array(Y, np.int8))


def build(split, cids, gold):
    parts = []
    for c in cids:
        f = np.load(data.DATA / f'feats/{split}/{c}.npy', mmap_mode='r')
        v = np.load(data.DATA / f'vad/{split}/{c}.npy').astype(np.float32) / 255
        r = conv_rows(f, v, gold[c])
        if r is not None:
            parts.append((c, r))
    cid = np.concatenate([np.full(len(r[3]), int(c)) for c, r in parts])
    X, S, M, Y = (np.concatenate([r[j] for _, r in parts]) for j in range(4))
    return dict(X=X, S=S, M=M, y=Y, cid=cid)


def eotb_rows(feats, lab):
    """Rows for one eot-bench voice-agent training session (latency/eotb_data.py: concatenated
    single user turns of other languages, silent agent). A candidate is each silent run of the
    user after speech; label 1 if the floor is OPEN there (the turn's final span), else 0 (a hold)."""
    import labels as lb
    T = min(len(feats), len(lab['act']))
    talk = lab['act'][:T, 0] != lb.A['SILENT']
    is_open = lab['floor'][:T, lb.F['OPEN']] > 0.5
    d = np.diff(np.r_[0, talk.astype(np.int8), 0])
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    X, S, M, Y = [], [], [], []
    prev_gap = 10.0
    for k, (a, b) in enumerate(zip(on, off)):
        nxt = on[k + 1] if k + 1 < len(on) else T
        t = b / FPS
        tail = feats[max(b - 13, 0):b, 0].astype(np.float32).mean(0).astype(np.float16)
        y = int(is_open[min(b, T - 1)])
        for dd in DELAYS:
            tau = t + dd
            if nxt / FPS <= tau or tau > T / FPS:
                break
            i = frame_at(tau, T)
            X.append(np.concatenate([feats[i, 0], tail, feats[i, 1]]))
            S.append([dd, min((b - a) / FPS, 10.0), 0.0, 0.0, 10.0, min(prev_gap, 10.0)])
            M.append([1, t, dd, k, 0])
            Y.append(y)
        prev_gap = (nxt - b) / FPS
    return (np.stack(X).astype(np.float16), np.array(S, np.float32), np.array(M, np.float32),
            np.array(Y, np.int8))


def build_eotb():
    parts = []
    for i, p in enumerate(sorted((data.DATA / 'labels/eotb').glob('*.npz'))):
        f = np.load(data.DATA / f'feats/eotb/{p.stem}.npy', mmap_mode='r')
        parts.append((i, eotb_rows(f, np.load(p))))
    cid = np.concatenate([np.full(len(r[3]), 100000 + i) for i, r in parts])
    X, S, M, Y = (np.concatenate([r[j] for _, r in parts]) for j in range(4))
    return dict(X=X, S=S, M=M, y=Y, cid=cid)


def main():
    import sys
    out = data.DATA / 'cands'
    if sys.argv[1:] == ['eotb']:
        d = build_eotb()
        np.savez(out / 'eotb_train.npz', **d)
        print('eotb', d['X'].shape, np.bincount(d['y']))
        return
    out = data.DATA / 'cands'
    out.mkdir(exist_ok=True)
    g = data.load_gold('tbdev')
    d = build('tbdev', sorted(g, key=int), g)
    np.savez(out / 'tbdev.npz', **d)
    print('tbdev', d['X'].shape, np.bincount(d['y'] + 1))
    g = data.load_gold('oto')
    for name in ('train', 'dev'):
        d = build('oto', data.oto_split(name), g)
        np.savez(out / f'oto_{name}.npz', **d)
        print('oto', name, d['X'].shape, np.bincount(d['y'] + 1))


if __name__ == '__main__':
    main()
