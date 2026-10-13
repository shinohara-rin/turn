"""Train (and cross-check) the hindsight labeler from per-conversation activity + gold.

Data layout (see README): <dir>/<cid>.npz with `silero_u8_32ms` [N, 2] (Silero speech
probability x 255 per channel, 32 ms) and <dir>/<cid>.gold.json with TurnBench-style events.
With --feats, <featdir>/<cid>.npy holds encoder frames from content.py ([T, 2, 1024] for a
stereo model, [T, 1, 1024] for a mono one); they are pooled around each candidate, reduced by
PCA (fit on all candidates, unsupervised) and appended to the timing features.

    python -m annotator.train --data tbdev=/path/tbdev oto=/path/oto --out model.joblib
    python -m annotator.train --data tbdev=... oto=... --eval   # 2-fold per set + cross-set
    python -m annotator.train --data ... --feats tbdev=/feats/tbdev oto=/feats/oto --out m.joblib
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import features as F


def load_dir(path, feats=None):
    convs = {}
    for f in sorted(glob.glob(os.path.join(path, '*.npz'))):
        cid = os.path.basename(f)[:-4]
        p = np.load(f)['silero_u8_32ms'] / 255.0
        convs[cid] = dict(activity=F.smooth(F.resample_probs(p, 0.032)),
                          gold=json.load(open(os.path.join(path, f'{cid}.gold.json'))))
        if feats:
            convs[cid]['feats'] = os.path.join(feats, f'{cid}.npy')
    return convs


def build(convs, task):
    rows, X, y = [], [], []
    for cid, c in convs.items():
        r, x = F.table(c['activity'], task)
        rows += [(cid, s, t) for s, t, _ in r]
        X.append(x)
        y += [F.label(c['gold'], task, s, t) for s, t, _ in r]
    return rows, np.concatenate(X), np.array(y)


def content_table(convs, rows):
    """Pooled encoder vectors for rows (cid, speaker, time_s), reading each conversation's frames once."""
    from . import content as C
    by, Z = {}, None
    for i, r in enumerate(rows):
        by.setdefault(r[0], []).append(i)
    for cid, idx in by.items():
        z = C.table(np.load(convs[cid]['feats'], mmap_mode='r'), [rows[i][1:] for i in idx])
        if Z is None:
            Z = np.zeros((len(rows), z.shape[1]), np.float32)
        Z[idx] = z
    return Z


def design(convs, task, n_pca=0):
    """rows, X, y for all candidates; with n_pca, X gets PCA-reduced content columns too."""
    rows, X, y = build(convs, task)
    pca = None
    if n_pca:
        from sklearn.decomposition import PCA
        Z = content_table(convs, rows)
        pca = PCA(n_pca, random_state=0, svd_solver='randomized').fit(Z)
        X = np.concatenate([X, pca.transform(Z)], 1)
    return rows, X, y, pca


def cross_fit(rows, X, y):
    """Scores from 2-fold cross-fitting by conversation."""
    cids = sorted({r[0] for r in rows}); half = set(cids[0::2]); sc = np.zeros(len(y))
    for k in (0, 1):
        tr = np.array([(r[0] in half) == (k == 0) for r in rows])
        sc[~tr] = fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    return sc


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
            res[f'recall@P{target}'] = 0.0; res[f'threshold@P{target}'] = None; continue
        th = scores[order][ok[-1]]
        res[f'threshold@P{target}'] = float(th)
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
    ap.add_argument('--feats', nargs='*', default=[], help='name=dir pairs of encoder frames (content.py)')
    ap.add_argument('--pca', type=int, default=32, help='content dimensions kept (with --feats)')
    ap.add_argument('--out', help='joblib path for the model bundle')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--calibrate', nargs='*', help='sets whose labels pick the stored thresholds (default: all)')
    a = ap.parse_args()
    feats = dict(d.split('=', 1) for d in a.feats)
    sets = {k: load_dir(v, feats.get(k)) for k, v in (d.split('=', 1) for d in a.data)}
    if feats and set(feats) != set(sets):
        ap.error('--feats needs a directory for every --data set')
    n_pca = a.pca if feats else 0
    cal = set(a.calibrate or sets)
    allc = {f'{k}/{c}': v for k, cs in sets.items() for c, v in cs.items()}
    models, thresholds, dev, pcas = {}, {}, {}, {}
    for task in F.TASKS:
        rows, X, y, pcas[task] = design(allc, task, n_pca)
        of = {k: np.array([r[0].startswith(k + '/') for r in rows]) for k in sets}
        if a.eval:
            for test, m in of.items():
                rr = [r for r, k in zip(rows, m) if k]
                conv = {c: v for c, v in allc.items() if c.startswith(test + '/')}
                print(task, test, '2-fold', evaluate(conv, task, rr, cross_fit(rr, X[m], y[m]), y[m]), flush=True)
                rest = ~m
                if rest.any():
                    sc = fit(X[rest], y[rest]).predict_proba(X[m])[:, 1]
                    print(task, test, 'trained on the rest', evaluate(conv, task, rr, sc, y[m]), flush=True)
        if a.out:
            models[task] = fit(X, y)
            # thresholds from cross-fitted (2-fold) scores on the --calibrate sets: max recall at
            # TurnBench FP <= 0.10, and the precision targets of precision_recall
            sc, m = cross_fit(rows, X, y), np.array([r[0].split('/')[0] in cal for r in rows])
            dev[task] = evaluate({c: v for c, v in allc.items() if c.split('/')[0] in cal}, task,
                                 [r for r, k in zip(rows, m) if k], sc[m], y[m])
            thresholds[task] = dev[task]['threshold']
            print(task, 'cross-fitted operating point', dev[task], flush=True)
    if a.out:
        import joblib
        content = None
        if n_pca:
            first = next(iter(allc.values()))['feats']
            content = dict(pca=pcas, channels=int(np.load(first, mmap_mode='r').shape[1]))
        joblib.dump(dict(models=models, thresholds=thresholds, dev=dev, data=sorted(sets), content=content), a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
