"""Cross-validated pause-onset probes on probe_eotbench.py's dump: end of turn vs mid-turn pause,
at a fixed silence length, 5 folds grouped by turn. Standardized features + PCA + L2 logistic
regression (C from an inner CV), so 1024-d inputs do not overfit 400 turns.

    python probe_fit.py probe_en.npz [text_en.json]
"""
import json
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_auc(X, y, groups, pca=64, seed=0):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        steps = [StandardScaler()]
        if pca and X.shape[1] > pca:
            steps.append(PCA(pca, random_state=seed))
        steps.append(LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), max_iter=2000))
        m = make_pipeline(*steps).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p), p


def logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))[:, None]


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    text = {}
    if len(sys.argv) > 2:
        text = {(o['turn'], o['span']): o['text_p'] for o in json.load(open(sys.argv[2]))}
    for q in (0.1, 0.3, 0.6):
        m = np.isclose(z['q'], q)
        y, g = z['label'][m], z['turn'][m]
        feats = {'r019 eot_q (zero-shot)': None,
                 'r019 eot_q (recalibrated)': logit(z['eot_q'][m]),
                 'r019 head hidden': z['hid'][m].astype(np.float32),
                 'FastConformer frame': z['enc'][m].astype(np.float32),
                 'FastConformer last words': z['pre'][m].astype(np.float32),
                 'FC frame + last words': np.hstack([z['enc'][m], z['pre'][m]]).astype(np.float32)}
        if 'pred' in z.files:
            feats['RNNT prediction net'] = z['pred'][m].astype(np.float32)
            feats['RNNT joint hidden'] = z['joint'][m].astype(np.float32)
            feats['FC + RNNT pred + joint'] = np.hstack([z['enc'][m], z['pre'][m], z['pred'][m],
                                                         z['joint'][m]]).astype(np.float32)
        if text:
            tp = np.array([text[(t, s)] for t, s in zip(z['turn'][m], z['span'][m])])
            feats['text model, true words'] = logit(tp)
            feats['r019 eot_q + text'] = np.hstack([logit(z['eot_q'][m]), logit(tp)])
            feats['FC + RNNT + text'] = np.hstack([feats.get('FC + RNNT pred + joint', feats['FC frame + last words']),
                                                   logit(tp)])
        print(f'--- silence {q} s: {int((y == 0).sum())} pauses, {int(y.sum())} ends')
        for name, X in feats.items():
            auc = roc_auc_score(y, z['eot_q'][m]) if X is None else cv_auc(X, y, g)[0]
            print(f'{name:32s} AUC {auc:.3f}')
