"""Local (CPU) loaders for the residual study: cached Silero VAD, gold events and model tracks
pulled from the Modal volume `ssl-turn-work` (no GPU):

    modal volume get ssl-turn-work /vad/tbdev/ $DATA/vad/
    modal volume get ssl-turn-work /vad/oto/ $DATA/vad/
    modal volume get ssl-turn-work /gold/tbdev.json $DATA/gold/tbdev.json
    modal volume get ssl-turn-work /gold/oto/ $DATA/gold/
    modal volume get ssl-turn-work /runs/r019_asr_bgaug/probs.npz $DATA/r019_probs.npz
    modal volume get ssl-turn-work /feats_asr/tbdev/ $DATA/feats/     (residual model only)
    modal volume get ssl-turn-work /feats_asr/oto/ $DATA/feats/
    modal volume get ssl-turn-work /split.json $DATA/split.json

DATA defaults to ~/data (env RESIDUAL_DATA).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # policy.py, labels.py
DATA = Path(os.environ.get('RESIDUAL_DATA', Path.home() / 'data'))


def oto_split(name):
    """Conversation ids of one otoSpeech split ('train' 131, 'dev' 16, 'gate' 20)."""
    return sorted(json.load(open(DATA / 'split.json'))['splits'][name], key=int)


def load_gold(split):
    if split == 'tbdev':
        return json.load(open(DATA / 'gold/tbdev.json'))
    return {p.stem: json.load(open(p)) for p in (DATA / 'gold/oto').glob('*.json')}


def load_vad(split, cids):
    return {c: np.load(DATA / f'vad/{split}/{c}.npy').astype(np.float32) / 255 for c in cids}


def score_variants(post, silent=None, fine=None):
    """Copy of pipeline/score.py::score_variants for the tracks used here (eot_q, int_ft),
    without the Modal import."""
    import labels as lb
    now = post[:, 0]
    eot_now = np.stack([now[:, 2] + now[:, 1], now[:, 2] + now[:, 0]], 1)
    out = dict(eot=np.clip(eot_now, 0, 1))
    if silent is not None:
        out['eot_q'] = np.clip(eot_now * silent, 0, 1)
    if fine is not None:
        out['int_ft'] = np.clip(fine[..., lb.FINE_GROUPS['floor_taking']].sum(-1), 0, 1)
    return out


def load_tracks(model, split, path=None):
    """{cid: {'eot_q': [T, 2], 'int_ft': [T, 2]}} for one checkpoint of a probs.npz."""
    z = np.load(path or DATA / 'r019_probs.npz')
    out = {}
    for k in z.files:
        m, sp, cid, t = k.split('/')
        if m == model and sp == split and t == 'post':
            base = k[:-len('post')]
            out[cid] = score_variants(z[k].astype(np.float32), z[base + 'silent'].astype(np.float32),
                                      z[base + 'fine'].astype(np.float32))
    return out
