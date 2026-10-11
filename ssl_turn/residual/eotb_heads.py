"""Run full ssl_turn heads (r019, r025) on CPU over the locally encoded eot-bench features
(encode_eotb.py), and write harness predictions for each head and for head + residual blends.

    python eotb_heads.py --heads r019_bgaug_s1,r025_eotb5_s1 --blend res3

The head forward is eotbench/modal_eot.py's (one pass over the whole turn, eot_q =
p(released) x p(user SILENT)); r019_bgaug_s1 reproduces the committed Modal run (815 ms @5%).
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch

import data
import model as m

FRAME_S = 0.08


def build_head(cfg):
    assert cfg.get('feats') == 'asr' and not cfg.get('taps'), cfg
    return m.TurnModel(tap_layers=0, final_dim=1024, dim=cfg.get('dim', 256), heads=cfg.get('heads', 4),
                       layers=cfg.get('layers', 4), window_s=cfg.get('window_s', 20.0), dropout=cfg.get('dropout', 0.1))


def head_tracks(name, feats):
    ck = torch.load(data.DATA / f'{name}.pt', map_location='cpu', weights_only=False)
    net = build_head(ck['cfg']).eval()
    net.load_state_dict(ck['state'])
    out = {}
    for rid in feats.files:
        x = torch.from_numpy(feats[rid].astype(np.float32))[None]
        with torch.no_grad():
            o = net(None, x)
        floor = o['floor'][0].softmax(-1).numpy()
        silent = o['act'][0].softmax(-1)[:, 0, 0].numpy()
        out[rid] = np.clip((floor[:, 2] + floor[:, 1]) * silent, 0, 1)
    return out


def at_grid(grid, tracks):
    p = np.empty(len(grid), np.float32)
    for k, r in enumerate(grid.itertuples()):
        tr = tracks[r.id]
        t = int(math.floor(r.timestamp / FRAME_S + 1e-6)) - 1
        p[k] = float(tr[min(t, len(tr) - 1)]) if t >= 0 else 0.0
    return p


def write(grid, p, run_id, manifest, note):
    d = data.DATA / 'eotb' / run_id
    d.mkdir(parents=True, exist_ok=True)
    grid.assign(p_eot=p).to_parquet(d / 'predictions.parquet', index=False)
    mm = dict(manifest, model_run_id=run_id)
    mm['model'] = dict(mm['model'], adapter='turn/ssl_turn/residual/eotb_heads.py', adapter_id=run_id, display_name=note)
    (d / 'manifest.json').write_text(json.dumps(mm, indent=2))
    print('wrote', d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', default='r019_bgaug_s1,r025_eotb5_s1')
    ap.add_argument('--blend', default='res3')
    a = ap.parse_args()
    ref = data.DATA / 'eotb/ssl_turn__residual_res2'
    grid = pd.read_parquet(ref / 'predictions.parquet')
    manifest = json.loads((ref / 'manifest.json').read_text())
    feats = np.load(data.DATA / 'feats/eotb_en.npz')
    for h in a.heads.split(','):
        p = at_grid(grid, head_tracks(h, feats))
        write(grid, p, f'ssl_turn__cpu_{h}', manifest, f'{h} eot_q (CPU)')
        for tag in filter(None, a.blend.split(',')):
            r = pd.read_parquet(data.DATA / f'eotb/ssl_turn__residual_{tag}/predictions.parquet').p_eot.values
            write(grid, (p + r) / 2, f'ssl_turn__blend_{h}_{tag}', manifest, f'mean({h}, residual {tag})')


if __name__ == '__main__':
    main()
