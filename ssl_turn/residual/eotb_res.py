"""Residual EOT classifier on LiveKit eot-bench (en): one score per 0.1 s grid point of each
silence span, built exactly like cands.py rows (the span is the VAD candidate; the agent channel
is silent, so 'other' is never active).

    python eotb_res.py --tags res2 [--ref RUN_DIR]

Writes $DATA/eotb/{tag}/predictions.parquet + manifest.json in the harness layout, ready for
`eot-harness compute-metrics`. The grid (ids, spans, timestamps, labels) is copied from an
existing harness run (default: the committed r019 bgaug_s1 eot_q run).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data
from cands import frame_at
from train_res import COLS

REF = Path('/mnt/project-files/eot-bench/runs/ssl_turn__r019_bgaug_s1_eot_q')


def rows_for(grid, feats):
    """Feature rows (X [n, 3072] f16, S [n, 6]) for one turn's grid rows."""
    T = feats.shape[0]
    X, S = [], []
    g = grid.assign(start=grid.timestamp - grid.silence_dur)
    bounds = g.groupby('span_index').agg(start=('start', 'first'), end=('timestamp', 'max')).sort_index()
    prev_end, prev_len = {}, {}
    last_end, last_len = 0.0, 10.0
    for si, b in bounds.iterrows():
        prev_end[si], prev_len[si] = last_end, last_len
        last_end, last_len = b.end, b.end - b.start
    for r in g.itertuples():
        t = round(r.start, 3)
        d = r.silence_dur
        i0, i1 = frame_at(t - 1.0, T), frame_at(t, T)
        tail = feats[i0:i1 + 1, 0].astype(np.float32).mean(0).astype(np.float16)
        i = frame_at(r.timestamp, T)
        X.append(np.concatenate([feats[i, 0], tail, feats[i, 1]]))
        S.append([min(d, 2.5), min(t - prev_end[r.span_index], 10.0), 0.0, 0.0, 10.0,
                  min(prev_len[r.span_index], 10.0)])
    return np.stack(X), np.array(S, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='res2')
    ap.add_argument('--ref', default=str(REF))
    args = ap.parse_args()
    ref = Path(args.ref)
    grid = pd.read_parquet(ref / 'predictions.parquet')
    feats = np.load(data.DATA / 'feats/eotb_en.npz')
    Xs, Ss = [], []
    for rid, g in grid.groupby('id', sort=False):
        X, S = rows_for(g, feats[rid])
        Xs.append(X)
        Ss.append(S)
    order = np.concatenate([g.index.values for _, g in grid.groupby('id', sort=False)])
    X, S = np.concatenate(Xs), np.concatenate(Ss)
    S[:, 1:] = np.log1p(S[:, 1:])
    manifest = json.loads((ref / 'manifest.json').read_text())
    for tag in args.tags.split(','):
        ck = torch.load(data.DATA / f'cands/model_{tag}.pt', weights_only=False)
        cols = COLS[ck['feats']]
        dim = len(ck['mu'])
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(dim, ck['hidden']), torch.nn.GELU(),
                                  torch.nn.Dropout(0.3), torch.nn.Linear(ck['hidden'], 1))
        net.load_state_dict(ck['state'])
        net.eval()
        x = (np.concatenate([X[:, cols].astype(np.float32), S], 1) - ck['mu']) / ck['sd']
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(x, dtype=torch.float32)).squeeze(1)).numpy()
        out = grid.copy()
        out.loc[order, 'p_eot'] = p
        run_id = f'ssl_turn__residual_{tag}'
        d = data.DATA / 'eotb' / run_id
        d.mkdir(parents=True, exist_ok=True)
        out.to_parquet(d / 'predictions.parquet', index=False)
        m = dict(manifest, model_run_id=run_id)
        m['model'] = dict(m['model'], adapter='turn/ssl_turn/residual/eotb_res.py', adapter_id=run_id,
                          display_name=f'residual {tag} (candidate-only head on FastConformer)')
        (d / 'manifest.json').write_text(json.dumps(m, indent=2))
        print('wrote', d)


if __name__ == '__main__':
    main()
