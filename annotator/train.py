"""Train (and cross-check) the hindsight labeler from per-conversation activity + gold.

Data layout (see README): <dir>/<cid>.npz with `silero_u8_32ms` [N, 2] (Silero speech
probability x 255 per channel, 32 ms) and <dir>/<cid>.gold.json with TurnBench-style events.

    python -m annotator.train --data tbdev=/path/tbdev oto=/path/oto --out model.joblib
    python -m annotator.train --data tbdev=... oto=... --eval   # leave-one-dataset-out + 2-fold
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import features as F


def load_dir(path):
    convs = {}
    for f in sorted(glob.glob(os.path.join(path, '*.npz'))):
        cid = os.path.basename(f)[:-4]
        p = np.load(f)['silero_u8_32ms'] / 255.0
        convs[cid] = dict(activity=F.smooth(F.resample_probs(p, 0.032)),
                          gold=json.load(open(os.path.join(path, f'{cid}.gold.json'))))
    return convs


def build(convs, task):
    rows, X, y = [], [], []
    for cid, c in convs.items():
        r, x = F.table(c['activity'], task)
        rows += [(cid, s, t) for s, t, _ in r]
        X.append(x)
        y += [F.label(c['gold'], task, s, t) for s, t, _ in r]
    return rows, np.concatenate(X), np.array(y)


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)


def fit(X, y):
    m = y >= 0
    return model().fit(X[m], y[m])


def evaluate(convs, task, rows, scores, labels, budget=0.10):
    """Official TurnBench scorer over a threshold sweep: best recall at FP <= budget."""
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import score_task
    scores, labels = np.asarray(scores), np.asarray(labels)
    best = (0.0, None, None)
    for th in np.unique(np.quantile(scores, np.linspace(0, 1, 201))):
        fires = {}
        for (cid, s, t), v in zip(rows, scores):
            if v >= th:
                fires.setdefault(cid, {1: [], 2: []})[s].append(t + F.FIRE_OFFSET_S[task])
        tot = dict(tp=0, fn=0, fp=0, tn=0)
        for cid, c in convs.items():
            g = c['gold']
            r = score_task([AnchorEvent(**e) for e in g[f'{task}_positive_events']],
                           [Interval(**e) for e in g[f'{task}_negative_spans']],
                           {s: sorted(fires.get(cid, {}).get(s, [])) for s in (1, 2)},
                           [Interval(**e) for e in g[f'{task}_excluded']])
            for k in tot:
                tot[k] += getattr(r, k)
        rec, fp = tot['tp'] / max(1, tot['tp'] + tot['fn']), tot['fp'] / max(1, tot['fp'] + tot['tn'])
        if fp <= budget and rec > best[0]:
            best = (rec, fp, float(th))
    # precision as a label source: share of fired candidates that land in a gold positive window
    m = (scores >= best[2]) & (labels >= 0) if best[2] is not None else np.zeros(len(scores), bool)
    prec = float((labels[m] == 1).mean()) if m.any() else None
    out = dict(recall=round(best[0], 3), fp=None if best[1] is None else round(best[1], 3),
               precision=None if prec is None else round(prec, 3), threshold=best[2])
    out.update(precision_recall(convs, task, rows, scores, labels))
    return out


def precision_recall(convs, task, rows, scores, labels, targets=(0.7, 0.8, 0.9)):
    """Annotator view: event recall (share of gold positives with a fired candidate in their
    window) at the lowest threshold whose fired-candidate precision reaches each target."""
    m = labels >= 0
    rows = [r for r, k in zip(rows, m) if k]; scores, labels = scores[m], labels[m]
    order = np.argsort(-scores); hits = np.cumsum(labels[order] == 1); prec = hits / np.arange(1, len(order) + 1)
    res = {}
    for target in targets:
        ok = np.flatnonzero(prec >= target)
        if not len(ok):
            res[f'recall@P{target}'] = 0.0; continue
        th = scores[order][ok[-1]]
        fired = {}
        for (cid, s, t), v in zip(rows, scores):
            if v >= th:
                fired.setdefault((cid, s), []).append(t + F.FIRE_OFFSET_S[task])
        n = found = 0
        for cid, c in convs.items():
            for e in c['gold'][f'{task}_positive_events']:
                n += 1
                found += any(e['time_s'] - F.TAU_PRE_S <= t <= e['time_s'] + F.TAU_MAX_S for t in fired.get((cid, e['speaker']), []))
        res[f'recall@P{target}'] = round(found / max(1, n), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='name=dir pairs')
    ap.add_argument('--out', help='joblib path for {task: model}')
    ap.add_argument('--eval', action='store_true')
    a = ap.parse_args()
    sets = {k: load_dir(v) for k, v in (d.split('=', 1) for d in a.data)}
    if a.eval:
        for task in F.TASKS:
            built = {k: build(v, task) for k, v in sets.items()}
            for test, (rows, X, y) in built.items():
                cids = sorted(sets[test]); half = set(cids[0::2]); sc = np.zeros(len(y))
                for k in (0, 1):
                    tr = np.array([(r[0] in half) == (k == 0) for r in rows])
                    sc[~tr] = fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
                print(task, test, '2-fold', evaluate(sets[test], task, rows, sc, y), flush=True)
                others = [b for k, b in built.items() if k != test]
                if others:
                    m = fit(np.concatenate([b[1] for b in others]), np.concatenate([b[2] for b in others]))
                    print(task, test, 'trained on the rest', evaluate(sets[test], task, rows, m.predict_proba(X)[:, 1], y), flush=True)
    if a.out:
        import joblib
        models, thresholds, dev = {}, {}, {}
        allc = {f'{k}/{c}': v for k, cs in sets.items() for c, v in cs.items()}
        for task in F.TASKS:
            rows, X, y = build(allc, task)
            models[task] = fit(X, y)
            # operating point: max recall at FP <= 0.10 on cross-fitted (2-fold) scores
            cids = sorted(allc); half = set(cids[0::2]); sc = np.zeros(len(y))
            for k in (0, 1):
                tr = np.array([(r[0] in half) == (k == 0) for r in rows])
                sc[~tr] = fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
            dev[task] = evaluate(allc, task, rows, sc, y)
            thresholds[task] = dev[task]['threshold']
            print(task, 'cross-fitted operating point', dev[task], flush=True)
        joblib.dump(dict(models=models, thresholds=thresholds, dev=dev, data=sorted(sets)), a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
