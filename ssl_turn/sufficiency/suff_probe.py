"""Can the streaming encoder predict reply sufficiency? Grouped-CV linear probes.

    python suff_probe.py suff_dev.json suff_feats.npz [comp_dev.json]

Target: judge score >= 8 at candidates before the turn end. 5-fold CV grouped by conversation,
standardize + PCA 64 + L2 logistic regression (as semantics/probe_fit.py). Baselines use only
causal facts: words so far, seconds since the turn started, whether the point is a pause.
"""
import json
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_auc(X, y, g, pca=True):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        steps = [StandardScaler()] + ([PCA(64, random_state=0)] if pca and X.shape[1] > 64 else []) + \
                [LogisticRegression(C=0.1, max_iter=2000)]
        m = make_pipeline(*steps).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main(lab, feats, comp=None):
    d = json.load(open(lab))
    F = np.load(feats)
    X = F['X'].astype(np.float32)
    rows = [(cid, *x) for cid in sorted(d) for x in d[cid]]
    assert len(rows) == len(X)
    start = {}
    for cid, s, t, kind, sc, n, N, end, *_ in rows:
        k = (cid, s, end)
        start[k] = min(start.get(k, t), t - n * 0.3)  # rough turn start: ~0.3 s per word before the first candidate
    keep = np.array([r[3] != 'end' and r[4] is not None for r in rows])
    y = np.array([(r[4] or 0) >= 8 for r in rows])
    g = np.array([r[0] for r in rows])
    n = np.array([r[5] for r in rows], float)
    el = np.array([r[2] - start[(r[0], r[1], r[7])] for r in rows])
    pause = np.array([r[3] == 'pause' for r in rows], float)
    base = np.stack([np.log1p(n), np.log1p(np.maximum(el, 0)), pause], 1)
    corp = np.array([r[0].split('/')[0] for r in rows])
    kind = np.array([r[3] for r in rows])
    sets = {'position (words so far, elapsed, pause)': base,
            'encoder frame own+other': X[:, :2048],
            'encoder own+other+last 1 s': X,
            'encoder + position': np.concatenate([X, base], 1)}
    k = keep
    print(f'{k.sum()} candidates, base rate {y[k].mean():.2f}')
    for name, Z in sets.items():
        p = cv_auc(Z[k], y[k], g[k], pca=Z.shape[1] > 64)
        parts = '  '.join(f'{c} {roc_auc_score(y[k][m], p[m]):.3f}' for c, m in
                          [(c, corp[k] == c) for c in ('tb', 'oto')] + [(c, kind[k] == c) for c in ('mid', 'pause')])
        print(f'  {name:42s} AUC {roc_auc_score(y[k], p):.3f}   ({parts})')


if __name__ == '__main__':
    main(*sys.argv[1:])
