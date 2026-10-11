"""How r019 treats listener backchannels: TurnBench dev (new speakers) vs otoSpeech dev.

For each annotator-'a' backchannel by speaker c while the other speaker o holds a turn (o has a
Normal Turn / Strong Floor Hold segment covering or within 1 s of the backchannel):
  int_fire   max int_nobc[c] over [start, end + 0.4 s] > th_int (c 'takes the floor')
  eot_fire   max eot_q[o] over [start, end + 0.5 s] > th_eot, and o speaks again within 1.5 s of
             the backchannel end (o never yielded)
  bc_mass    mean fine-head backchannel + non-content mass for c over the segment
Thresholds default to the playground's (r019 TB dev operating points).

    python bc_check.py probs.npz ann.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 2)[0])
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/pipeline')
import labels as lb  # noqa: E402
from score import score_variants  # noqa: E402

FPS = 12.5
BC = ('Continuer Backchannel', 'Acknowledgement Backchannel', 'Reaction Backchannel')
HOLD = ('Normal Turn', 'Strong Floor Hold', 'Filler')


def frames(a, b, T):
    return slice(max(0, int(a * FPS)), min(T, int(np.ceil(b * FPS)) + 1))


def analyse(z, ann, model, split, th_eot=0.64, th_int=0.18):
    rows = []
    for cid, segs in ann[split].items():
        k = f'{model}/{split}/{cid}'
        if f'{k}/post' not in z.files:
            continue
        post, silent, fine = (z[f'{k}/{t}'].astype(np.float32) for t in ('post', 'silent', 'fine'))
        v = score_variants(post, silent, fine)
        bcm = fine[..., lb.FINE_GROUPS['backchannel'] + lb.FINE_GROUPS['noncontent']].sum(-1)
        T = len(post)
        a_segs = [s for s in segs if s[1] == 'a']
        for s, _, a, b, label, text in a_segs:
            if label not in BC:
                continue
            c, o = s - 1, 2 - s
            other = [x for x in a_segs if x[0] == o + 1 and x[4] in HOLD]
            if not any(x[2] - 1.0 < b and x[3] + 1.0 > a for x in other):
                continue
            resumes = any(b - 0.5 <= x[2] <= b + 1.5 or (x[2] < b and x[3] > b + 0.3) for x in other)
            w_int = frames(a, b + 0.4, T)
            w_eot = frames(a, b + 0.5, T)
            rows.append(dict(cid=cid, label=label, dur=b - a, text=text,
                             int_fire=bool(v['int_nobc'][w_int, c].max() > th_int),
                             int_max=float(v['int_nobc'][w_int, c].max()),
                             eot_fire=bool(resumes and v['eot_q'][w_eot, o].max() > th_eot),
                             resumes=resumes, bc_mass=float(bcm[frames(a, b, T), c].mean())))
    return rows


def summary(rows):
    r = rows
    f = lambda key, sub: np.mean([x[key] for x in sub]) if sub else float('nan')
    out = dict(n=len(r), int_fire=f('int_fire', r), eot_fire=f('eot_fire', [x for x in r if x['resumes']]),
               bc_mass=f('bc_mass', r))
    for lab in BC:
        sub = [x for x in r if x['label'] == lab]
        out[lab.split()[0]] = (len(sub), round(f('int_fire', sub), 3), round(f('bc_mass', sub), 2))
    for lo, hi in ((0, 0.4), (0.4, 0.8), (0.8, 99)):
        sub = [x for x in r if lo <= x['dur'] < hi]
        out[f'dur{lo}-{hi}'] = (len(sub), round(f('int_fire', sub), 3))
    return out


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    ann = json.load(open(sys.argv[2]))
    for model in ('bgaug_s1', 'bgaug_s2', 'fine1_bal1_s1'):
        for split in ('oto', 'tbdev'):
            print(model, split, summary(analyse(z, ann, model, split)))
