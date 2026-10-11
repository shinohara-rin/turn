"""Train a small residual EOT classifier on VAD candidates only (CPU), and compare it with the
r019 head's eot_q read at the same probe times.

    python train_res.py [--hidden 256] [--feats all|frame|scalars] [--tag res1]

Writes $DATA/cands/scores_{tag}.npz: per-row scores for tbdev and oto_dev (all rows, labeled
or not), used by policy_res.py.
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import data
from cands import DELAYS, frame_at

torch.set_num_threads(4)


def load(name):
    return dict(np.load(data.DATA / f'cands/{name}.npz'))


COLS = {'all': slice(0, 3072), 'frame': slice(0, 1024), 'own': slice(0, 2048), 'scalars': slice(0, 0)}


class Inputs:
    """Standardized [encoder features | log scalars], built per batch (the fp16 matrix stays as is)."""

    def __init__(self, d, which, ref=None):
        self.X, self.cols = d['X'], COLS[which]
        S = d['S'].copy()
        S[:, 1:] = np.log1p(S[:, 1:])
        self.S = S
        if ref is None:
            i = np.random.default_rng(0).choice(len(S), min(len(S), 50000), replace=False)
            x = self.raw(np.sort(i))
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-4
        else:
            self.mu, self.sd = ref.mu, ref.sd

    def raw(self, idx):
        return np.concatenate([self.X[idx, self.cols].astype(np.float32), self.S[idx]], 1)

    def __call__(self, idx):
        return torch.tensor((self.raw(idx) - self.mu) / self.sd)

    def __len__(self):
        return len(self.S)


def r019_at_probes(d, split, model='bgaug_s1'):
    """eot_q of r019 at each row's probe time (same causal frame rule)."""
    tr = data.load_tracks(model, split)
    out = np.full(len(d['y']), np.nan, np.float32)
    for c in np.unique(d['cid']):
        q = tr[str(c)]['eot_q']
        for i in np.flatnonzero(d['cid'] == c):
            spk, t, dd = d['M'][i, :3]
            out[i] = q[frame_at(t + dd, len(q)), int(spk) - 1]
    return out


def auc_by_delay(d, score):
    lab = d['y'] >= 0
    res = {}
    for dd in DELAYS:
        m = lab & np.isclose(d['M'][:, 2], dd)
        if m.sum() and len(np.unique(d['y'][m])) == 2:
            res[dd] = roc_auc_score(d['y'][m], score[m])
    m = lab
    res['all'] = roc_auc_score(d['y'][m], score[m])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--feats', default='all')
    ap.add_argument('--tag', default='res1')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--extra', default='', help='extra training sets, e.g. eotb_train (cands.py eotb)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tr, dv, tb = load('oto_train'), load('oto_dev'), load('tbdev')
    for name in filter(None, args.extra.split(',')):
        ex = load(name)
        tr = {k: np.concatenate([tr[k], ex[k]]) for k in tr}
    Xtr = Inputs(tr, args.feats)
    Xdv, Xtb = Inputs(dv, args.feats, Xtr), Inputs(tb, args.feats, Xtr)
    lab = np.flatnonzero(tr['y'] >= 0)
    yt = torch.tensor(tr['y'][lab], dtype=torch.float32)
    pos_w = float((yt == 0).sum() / (yt == 1).sum())
    dim = Xtr(lab[:1]).shape[1]
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(dim, args.hidden), torch.nn.GELU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(args.hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    def pred(X):
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(net(X(np.arange(i, min(i + 8192, len(X))))).squeeze(1)).numpy()
                                   for i in range(0, len(X), 8192)])
    best, state = -1, None
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lossf(net(Xtr(np.sort(lab[b.numpy()]))).squeeze(1), yt[np.sort(b.numpy())]).backward()
            opt.step()
        net.eval()
        a = auc_by_delay(dv, pred(Xdv))['all']
        if a > best:
            best, state, best_ep = a, {k: v.clone() for k, v in net.state_dict().items()}, ep
    net.load_state_dict(state)
    net.eval()
    print(f'best oto-dev AUC {best:.3f} at epoch {best_ep}')
    s_dv, s_tb = pred(Xdv), pred(Xtb)
    np.savez(data.DATA / f'cands/scores_{args.tag}.npz', tbdev=s_tb, oto_dev=s_dv)
    torch.save(dict(state=net.state_dict(), mu=Xtr.mu, sd=Xtr.sd, feats=args.feats, hidden=args.hidden),
               data.DATA / f'cands/model_{args.tag}.pt')
    for name, d, s, split in [('oto dev', dv, s_dv, 'oto'), ('TB dev', tb, s_tb, 'tbdev')]:
        r = r019_at_probes(d, split)
        a, b = auc_by_delay(d, s), auc_by_delay(d, r)
        print(f'{name:8s} AUC  ' + '  '.join(f'{k}: res {a[k]:.3f} / r019 {b[k]:.3f}' for k in a))


if __name__ == '__main__':
    main()
