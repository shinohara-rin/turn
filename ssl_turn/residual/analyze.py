"""Where does VAD + rules go wrong on TB dev, and how much of the model's edge sits there?

Each EOT fire is traced back to its VAD candidate (a speech offset) and the trigger that
committed it (deadline, other speaker, model, confirmation); each gold positive and negative
span is attributed with the pinned TurnBench scorer's own claiming rule. Then oracle
residual policies bound what a model that only acts on VAD candidates could gain:

  veto     drop every candidate whose fires are all false positives
  early@d  move a candidate that the rules get right to t + d (t = VAD offset)

    python analyze.py [--model bgaug_s1]
"""
import argparse

import numpy as np
from turnbench.gold import AnchorEvent, Interval
from turnbench.score import TAU_MAX_S, TAU_PRE_S, score_task

import data
from policy import Policy, first_above, speech_segments, eot_events, int_events

RULES_EOT = Policy(vad_th=0.5, min_wait=0.6, other_dur=0.3, eot_th=None)        # rules, budget 0.10
HYB_EOT = {0.10: Policy(vad_th=0.5, min_wait=0.6, other_dur=0.3, eot_th=0.8),    # bgaug_s1, 0.10
           0.08: Policy()}                                                       # bgaug_s2, 0.08
RULES_INT = {0.05: Policy(int_th=None, int_dur=1.0), 0.10: Policy(vad_th=0.5, int_th=None, int_dur=0.8)}
HYB_INT = {0.05: Policy(int_th=0.15, int_dur=1.2), 0.10: Policy()}


def candidates(own, other, duration_s, P, score=None, fps=12.5):
    """policy.eot_events, one record per VAD offset: offset t, commit time, which trigger
    committed, confirmation time (or None); cancelled candidates have commit None."""
    out = []
    horizon = max(P.deadline, P.confirm or 0)
    for k, (_, t) in enumerate(own):
        cancel = np.inf
        for a, b in own[k + 1:]:
            if a - t > horizon:
                break
            if a + P.resume_min <= b + P.min_gap:
                cancel = a + P.resume_min
                break
        trig = {'deadline': t + P.deadline}
        earliest = t + P.min_wait
        for a, b in other:
            if b < t - P.min_gap:
                continue
            if a > t + P.deadline:
                break
            tt = max(a + P.other_dur, earliest)
            if tt <= b + P.min_gap:
                trig['other'] = tt
                break
        if P.eot_th is not None:
            m = first_above(score, t + P.read_from, min(t + P.deadline, cancel), P.eot_th, fps)
            if m is not None:
                trig['model'] = max(m, t + P.mw_model)
        kind = min(trig, key=trig.get)
        tc = min(trig[kind], duration_s)
        rec = dict(t=t, cancel=cancel, commit=None, kind=kind, confirm=None, trig=trig)
        if tc < cancel:
            rec['commit'] = round(tc, 3)
            if P.confirm and cancel > t + P.confirm and t + P.confirm <= duration_s:
                rec['confirm'] = round(t + P.confirm, 3)
        out.append(rec)
    return out


def attribute(pos, neg, excl, fires):
    """Mirror of turnbench.score.score_task for one speaker, keeping who claimed what.
    Returns ({pos index: fire time or None}, {neg index: [FP fire times]})."""
    fires = sorted(f for f in fires if not any(e.start <= f <= e.end for e in excl))
    claimed, windows, hit = set(), [], {}
    nxt = sorted(e.time_s for e in pos)
    for i, e in sorted(enumerate(pos), key=lambda x: x[1].time_s):
        j = np.searchsorted(nxt, e.time_s, side='right')
        w0, w1 = e.time_s - TAU_PRE_S, min(e.time_s + TAU_MAX_S, nxt[j] if j < len(nxt) else np.inf)
        windows.append((w0, w1))
        hit[i] = None
        for k, f in enumerate(fires):
            if w0 <= f <= w1 and k not in claimed:
                claimed.add(k)
                hit[i] = f
                break
    fp = {}
    for i, s in enumerate(neg):
        fp[i] = [f for k, f in enumerate(fires) if k not in claimed and s.start - TAU_PRE_S <= f <= s.end
                 and not any(a <= f <= b for a, b in windows)]
    return hit, fp


def gold_sets(g, task, spk):
    e = g['events']
    return ([AnchorEvent(**x) for x in e[f'{task}_positive_events'] if x['speaker'] == spk],
            [Interval(**x) for x in e[f'{task}_negative_spans'] if x['speaker'] == spk],
            [Interval(**x) for x in e[f'{task}_excluded'] if x['speaker'] == spk])


def run_eot(gold, vad, tracks, P, cids, mutate=None):
    """Score EOT with optional per-candidate mutation; returns totals and per-candidate rows."""
    tp = fn = fp = tn = 0
    lat, rows = [], []
    for c in cids:
        g = gold[c]
        segs = [speech_segments(vad[c][:, s], P.vad_th, P.min_gap) for s in (0, 1)]
        for s in (0, 1):
            score = tracks[c]['eot_q'][:, s] if tracks and P.eot_th is not None else None
            cands = candidates(segs[s], segs[1 - s], g['duration_s'], P, score)
            ref = eot_events(segs[s], segs[1 - s], g['duration_s'], P, score)
            fires = sorted(set(x for r in cands for x in (r['commit'], r['confirm']) if x is not None))
            assert fires == ref, (c, s)
            pos, neg, excl = gold_sets(g, 'eot', s + 1)
            if mutate is not None:
                cands = mutate(c, s, cands, pos, neg, excl)
                fires = sorted(set(x for r in cands for x in (r['commit'], r['confirm']) if x is not None))
            ts = score_task(pos, neg, {s + 1: fires}, excl)
            tp, fn, fp, tn = tp + ts.tp, fn + ts.fn, fp + ts.fp, tn + ts.tn
            lat += ts.latencies_ms
            hit, fps_ = attribute(pos, neg, excl, fires)
            assert sum(v is not None for v in hit.values()) == ts.tp
            assert sum(bool(v) for v in fps_.values()) == ts.fp
            byfire = {}
            for i, f in hit.items():
                if f is not None:
                    byfire[f] = ('tp', pos[i].time_s)
            for i, fs in fps_.items():
                for f in fs:
                    byfire.setdefault(f, ('fp', neg[i].start))
            for r in cands:
                r.update(cid=c, spk=s + 1, commit_res=byfire.get(r['commit'], ('free',))[0] if r['commit'] else None,
                         confirm_res=byfire.get(r['confirm'], ('free',))[0] if r['confirm'] else None)
                rows.append(r)
            for i, f in hit.items():
                if f is None:
                    rows.append(dict(cid=c, spk=s + 1, miss=pos[i].time_s))
    return dict(recall=tp / (tp + fn), fp=fp / (fp + tn), p50=float(np.median(lat)), tp=tp, fn=fn, fp_n=fp,
                neg=fp + tn), rows


def fmt(r):
    return f"R {r['recall']:.3f} FP {r['fp']:.3f} p50 {r['p50']:.0f} (TP {r['tp']} FN {r['fn']} FP {r['fp_n']}/{r['neg']})"


def oracle_veto(c, s, cands, pos, neg, excl):
    """Drop a candidate whose fires include a false positive and no true positive. Uses gold."""
    hit, fps_ = attribute(pos, neg, excl, sorted(set(x for r in cands for x in (r['commit'], r['confirm']) if x)))
    tps = {f for f in hit.values() if f is not None}
    bad = {f for fs in fps_.values() for f in fs}
    out = []
    for r in cands:
        fs = {r['commit'], r['confirm']} - {None}
        if fs & bad and not fs & tps:
            r = dict(r, commit=None, confirm=None)
        out.append(r)
    return out


def oracle_early(d):
    def f(c, s, cands, pos, neg, excl):
        hit, _ = attribute(pos, neg, excl, sorted(set(x for r in cands for x in (r['commit'], r['confirm']) if x)))
        tps = {f for f in hit.values() if f is not None}
        return [dict(r, commit=round(min(r['commit'], r['t'] + d), 3)) if r['commit'] in tps else r for r in cands]
    return f


def both(*fs):
    def f(*a):
        cands = a[2]
        for g in fs:
            cands = g(a[0], a[1], cands, *a[3:])
        return cands
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='bgaug_s1')
    args = ap.parse_args()
    gold = data.load_gold('tbdev')
    cids = sorted(gold, key=int)
    vad = data.load_vad('tbdev', cids)
    tracks = {m: data.load_tracks(m, 'tbdev') for m in ('bgaug_s1', 'bgaug_s2')}

    print('== EOT, TB dev')
    R, rows = run_eot(gold, vad, None, RULES_EOT, cids)
    print('rules (budget 0.10)          ', fmt(R))
    H, hrows = run_eot(gold, vad, tracks['bgaug_s1'], HYB_EOT[0.10], cids)
    print('rules + r019 (budget 0.10)   ', fmt(H))
    H8, _ = run_eot(gold, vad, tracks['bgaug_s2'], HYB_EOT[0.08], cids)
    print('rules + r019 (budget 0.08)   ', fmt(H8))

    # Where rule fires come from and how they land.
    cand = [r for r in rows if 'miss' not in r]
    print(f'\nVAD candidates {len(cand)}; cancelled by resumption {sum(r["commit"] is None for r in cand)}')
    for kind in ('other', 'deadline'):
        ks = [r for r in cand if r['commit'] is not None and r['kind'] == kind]
        res = {x: sum(r['commit_res'] == x for r in ks) for x in ('tp', 'fp', 'free')}
        print(f'  first fire by {kind:8s}: {len(ks):5d}  tp {res["tp"]} fp {res["fp"]} unlabeled/no-effect {res["free"]}')
    cf = [r for r in cand if r['confirm'] is not None]
    print(f'  confirmation fires      : {len(cf):5d}  tp {sum(r["confirm_res"] == "tp" for r in cf)} '
          f'fp {sum(r["confirm_res"] == "fp" for r in cf)}')
    print(f'  misses (no fire in window): {sum("miss" in r for r in rows)}')

    # Latency of rule TPs by trigger, vs where the hybrid committed the same candidate.
    hy = {(r['cid'], r['spk'], r['t']): r for r in hrows if 'miss' not in r}
    for kind in ('other', 'deadline'):
        ks = [r for r in cand if r['commit_res'] == 'tp' and r['kind'] == kind]
        d_off = np.array([r['commit'] - r['t'] for r in ks])
        moved = [hy[(r['cid'], r['spk'], r['t'])] for r in ks if (r['cid'], r['spk'], r['t']) in hy]
        gain = np.array([r['commit'] - m['commit'] for r, m in zip(ks, moved) if m['commit'] is not None])
        print(f'  rule TPs by {kind:8s}: n {len(ks)}, commit - offset p50 {np.median(d_off):.2f} s; '
              f'hybrid fires same candidate earlier in {np.mean(gain > 0.01):.0%}, median gain '
              f'{np.median(gain[gain > 0.01]) if (gain > 0.01).any() else 0:.2f} s')

    print('\nOracle residuals on the rules policy (gold-informed upper bounds):')
    for name, mut in [('veto false candidates', oracle_veto), ('early @0.16', oracle_early(0.16)),
                      ('early @0.32', oracle_early(0.32)), ('early @0.48', oracle_early(0.48)),
                      ('veto + early @0.32', both(oracle_veto, oracle_early(0.32)))]:
        r, _ = run_eot(gold, vad, None, RULES_EOT, cids, mut)
        print(f'  {name:24s}', fmt(r))

    print('\n== INT, TB dev')
    for b in (0.05, 0.10):
        for name, P, tr in [('rules', RULES_INT[b], None), ('rules + r019', HYB_INT[b], tracks['bgaug_s2' if b == 0.05 else 'bgaug_s1'])]:
            tp = fn = fp = tn = 0
            lat = []
            for c in cids:
                g = gold[c]
                for s in (0, 1):
                    seg = speech_segments(vad[c][:, s], P.vad_th, P.min_gap)
                    ev = int_events(seg, P, None if tr is None else tr[c]['int_ft'][:, s])
                    pos, neg, excl = gold_sets(g, 'int', s + 1)
                    ts = score_task(pos, neg, {s + 1: ev}, excl)
                    tp, fn, fp, tn = tp + ts.tp, fn + ts.fn, fp + ts.fp, tn + ts.tn
                    lat += ts.latencies_ms
            print(f'  budget {b} {name:14s} R {tp / (tp + fn):.3f} FP {fp / (fp + tn):.3f} p50 {np.median(lat):.0f} '
                  f'(TP {tp} FN {fn} FP {fp}/{fp + tn})')


if __name__ == '__main__':
    main()
