"""Score residual triggers through the TB dev EOT policy.

The rules policy (analyze.RULES_EOT, vad_th 0.5) runs unchanged; a candidate score read at the
probe delays of cands.py adds
  early  fire at t + d for the first probe d >= d_min whose score >= th
  veto   when the rules' first fire comes from the other-speaker trigger and the score at the
         last probe before it is < lo, wait for the deadline instead
Scores: 'r019' (eot_q read at the probes, bgaug_s1), 'res' (train_res.py), 'avg' (mean of
the two probabilities). Reports, per family, the best recall at FP <= budget on all 38
conversations and a split-half held-out estimate (pick on 19, score on 19, 200 splits).

    python policy_res.py --tag res1
"""
import argparse
import itertools
import random

import numpy as np
from turnbench.score import score_task

import data
from analyze import RULES_EOT, HYB_EOT, candidates, gold_sets
from policy import speech_segments, eot_events
from train_res import load, r019_at_probes


def per_conv_counts(gold, vad, cids, P, probes=None, th=None, d_min=0.24, lo=None, score_track=None):
    """[(tp, fn, fp, tn, latencies)] per conversation."""
    rows = []
    for c in cids:
        g = gold[c]
        segs = [speech_segments(vad[c][:, s], P.vad_th, P.min_gap) for s in (0, 1)]
        acc = [0, 0, 0, 0, []]
        for s in (0, 1):
            if score_track is not None:  # continuous model trigger inside policy.py (existing hybrid)
                fires = eot_events(segs[s], segs[1 - s], g['duration_s'], P, score_track[c]['eot_q'][:, s])
            else:
                fires = []
                for r in candidates(segs[s], segs[1 - s], g['duration_s'], P):
                    commit, confirm = r['commit'], r['confirm']
                    pr = probes.get((c, s + 1, round(r['t'], 3)), []) if probes else []
                    if lo is not None and commit is not None and r['kind'] == 'other':
                        seen = [p for d, p in pr if r['t'] + d <= commit]
                        if seen and seen[-1] < lo:
                            dl = r['trig']['deadline']
                            commit = round(min(dl, g['duration_s']), 3) if dl < r['cancel'] else None
                    if th is not None:
                        hit = [d for d, p in pr if d >= d_min and p >= th]
                        if hit:
                            commit = round(min(commit if commit is not None else np.inf, r['t'] + hit[0]), 3)
                    fires += [x for x in (commit, confirm) if x is not None]
                fires = sorted(set(fires))
            ts = score_task(*gold_sets(g, 'eot', s + 1)[:2], {s + 1: fires}, gold_sets(g, 'eot', s + 1)[2])
            acc = [acc[0] + ts.tp, acc[1] + ts.fn, acc[2] + ts.fp, acc[3] + ts.tn, acc[4] + ts.latencies_ms]
        rows.append(acc)
    return rows


def agg(rows, idx=None):
    rows = rows if idx is None else [rows[i] for i in idx]
    tp, fn, fp, tn = (sum(r[j] for r in rows) for j in range(4))
    lat = [x for r in rows for x in r[4]]
    return dict(recall=tp / (tp + fn), fp=fp / (fp + tn), p50=float(np.median(lat)) if lat else np.nan)


def pick(scores, budget, fam):
    ok = [k for k in fam if scores[k]['fp'] <= budget]
    return max(ok, key=lambda k: (scores[k]['recall'], -scores[k]['p50'])) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='res1')
    ap.add_argument('--splits', type=int, default=200)
    args = ap.parse_args()
    gold = data.load_gold('tbdev')
    cids = sorted(gold, key=int)
    vad = data.load_vad('tbdev', cids)
    tb = load('tbdev')
    res = np.load(data.DATA / f'cands/scores_{args.tag}.npz')['tbdev']
    r019 = r019_at_probes(tb, 'tbdev')
    tracks = data.load_tracks('bgaug_s1', 'tbdev')
    keys = [(str(c), int(m[0]), round(float(m[1]), 3)) for c, m in zip(tb['cid'], tb['M'])]
    srcs = {'r019': r019, 'res': res, 'avg': (r019 + res) / 2}
    probes = {}
    for name, sc in srcs.items():
        pb = {}
        for k, d, p in zip(keys, tb['M'][:, 2], sc):
            pb.setdefault(k, []).append((round(float(d), 2), float(p)))
        probes[name] = pb

    grid = [('rules', {}, None)]
    grid += [('r019 continuous (policy.py)', dict(P=RULES_EOT.but(eot_th=th, mw_model=mw)), 'track')
             for th, mw in itertools.product((0.5, 0.7, 0.8, 0.9, 0.95), (0.2, 0.32, 0.48))]
    for name in srcs:
        for th, dm in itertools.product((0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95), (0.24, 0.32, 0.48)):
            grid.append((f'{name} early', dict(probes=probes[name], th=th, d_min=dm), None))
        for th, dm, lo in itertools.product((0.7, 0.8, 0.9, 0.95), (0.24, 0.32, 0.48), (0.05, 0.1, 0.2, 0.3)):
            grid.append((f'{name} early+veto', dict(probes=probes[name], th=th, d_min=dm, lo=lo), None))
    R = []
    for fam, kw, mode in grid:
        P = kw.pop('P', RULES_EOT)
        R.append(per_conv_counts(gold, vad, cids, P, score_track=tracks if mode == 'track' else None, **kw))
    full = [agg(r) for r in R]
    n = len(cids)
    rng = random.Random(0)
    halves = []
    for _ in range(args.splits):
        a = rng.sample(range(n), n // 2)
        halves.append((a, [i for i in range(n) if i not in a]))
    fams = {}
    for k, (f, _, _) in enumerate(grid):
        fams.setdefault(f, []).append(k)
    for budget in (0.08, 0.10):
        print(f'== EOT TB dev, FP budget {budget}')
        for f, ks in fams.items():
            k = pick(full, budget, ks)
            if k is None:
                print(f'  {f:30s} none')
                continue
            held = []
            for a, b in halves:
                kk = pick([agg(R[j], a) if j in ks else None for j in range(len(R))], budget, ks)
                if kk is not None:
                    hb = agg(R[kk], b)
                    held.append((hb['recall'], hb['fp'], hb['p50']))
            h = np.array(held)
            kw = {x: y for x, y in grid[k][1].items() if x != 'probes'}
            print(f"  {f:30s} R {full[k]['recall']:.3f} FP {full[k]['fp']:.3f} p50 {full[k]['p50']:4.0f} | held-out "
                  f"R {h[:, 0].mean():.3f} FP {h[:, 1].mean():.3f} p50 {np.median(h[:, 2]):4.0f} ({len(h)}) {kw}")


if __name__ == '__main__':
    main()
