"""Harness predictions from a tracks.npz ({model}/{variant}/{turn id} -> [T] score on the 80 ms
grid, e.g. distill/colab.py eot), on the eot-bench grid of an existing run.

    python eotb_tracks.py tracks.npz TAG [--variants eot_q,yld_q]
"""
import argparse
import json

import numpy as np
import pandas as pd

import data
from eotb_heads import at_grid, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks')
    ap.add_argument('tag')
    ap.add_argument('--variants', default='eot_q,yld_q')
    a = ap.parse_args()
    ref = data.DATA / 'eotb/ssl_turn__residual_res2'
    grid = pd.read_parquet(ref / 'predictions.parquet').drop(columns='p_eot')
    manifest = json.loads((ref / 'manifest.json').read_text())
    z = np.load(a.tracks)
    keys = {}
    for k in z.files:
        m, v, rid = k.split('/', 2)
        keys.setdefault((m, v), {})[rid] = z[k]
    for (m, v), tr in sorted(keys.items()):
        if v in a.variants.split(','):
            write(grid, at_grid(grid, tr), f'ssl_turn__{a.tag}_{m}_{v}', manifest, f'{a.tag} {m} {v}')


if __name__ == '__main__':
    main()
