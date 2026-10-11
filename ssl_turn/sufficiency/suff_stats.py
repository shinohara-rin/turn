"""How early does a turn become reply-sufficient, and how does that relate to "speaker is done"?

    python suff_stats.py suff_dev.json [comp_dev.json]

suff_dev.json from suff_labels.py; comp_dev.json (optional) from semantics/complete_labels.py on
the same oto dev conversations, matched by (conversation, speaker, pause end time).
"""
import json
import sys
from collections import defaultdict

import numpy as np

TH = 8   # score >= TH counts as sufficient


def main(src, comp=None):
    d = json.load(open(src))
    for corpus in ('tb', 'oto'):
        rows = [(cid, *x) for cid, v in d.items() if cid.startswith(corpus) for x in v]
        if not rows:
            continue
        print(f'\n== {corpus}: {len(rows)} candidates')
        for kind in ('mid', 'pause'):
            sc = np.array([r[4] for r in rows if r[3] == kind and r[4] is not None])
            print(f'  {kind:5s} n={len(sc):5d} mean {sc.mean():.1f}  sufficient(>={TH}) {np.mean(sc >= TH):.2f}')
        turns = defaultdict(list)
        for cid, s, t, kind, sc, n, N, end, *_ in rows:
            turns[(cid, s, end)].append((t, kind, sc, n, N))
        leads, fracs, long_leads = [], [], []
        for key, c in turns.items():
            c.sort()
            if len(c) < 2:
                continue
            # earliest point after which every later candidate is sufficient (stable)
            first = c[-1]
            for x in reversed(c[:-1]):
                if x[2] is None or x[2] < TH:
                    break
                first = x
            lead = key[2] - first[0]
            leads.append(lead)
            fracs.append(first[3] / first[4])
            if first[4] >= 15:
                long_leads.append(lead)
        leads, long_leads = np.array(leads), np.array(long_leads)
        print(f'  turns with >=2 candidates: {len(leads)}; stable-sufficient lead before turn end: '
              f'median {np.median(leads):.2f} s, mean {leads.mean():.2f} s, >1 s in {np.mean(leads > 1):.0%}, '
              f'>3 s in {np.mean(leads > 3):.0%}; median fraction of words needed {np.median(fracs):.2f}')
        if len(long_leads):
            print(f'  turns of >=15 words ({len(long_leads)}): median lead {np.median(long_leads):.2f} s, '
                  f'>1 s in {np.mean(long_leads > 1):.0%}')
    if comp:
        c = json.load(open(comp))
        cmap = {(f'oto/{cid}', x[0], round(x[1], 2)): x for cid, v in c.items() for x in v}
        pairs = []
        for cid, v in d.items():
            for s, t, kind, sc, n, N, end, *_ in v:
                m = cmap.get((cid, s, round(t, 2)))
                if m and kind == 'pause' and sc is not None and m[2] is not None:
                    pairs.append((sc, m[2], m[3] not in (0, s)))   # suff, comp, other speaker takes next
        if pairs:
            p = np.array(pairs, dtype=float)
            print(f'\n== oto dev mid-turn pauses matched to completion labels: {len(p)}')
            print(f'  Spearman-ish corr (Pearson on scores) {np.corrcoef(p[:, 0], p[:, 1])[0, 1]:.2f}')
            for lo, hi in ((0, 4), (4, 8), (8, 11)):
                m = (p[:, 1] >= lo) & (p[:, 1] < hi)
                print(f'  completion {lo}-{hi - 1}: n={m.sum():4d} sufficient {np.mean(p[m, 0] >= TH):.2f}')
            suff, done = p[:, 0] >= TH, p[:, 1] >= TH
            print(f'  sufficient but not done (wake early, keep listening): {np.mean(suff & ~done):.2f}; '
                  f'done but not sufficient: {np.mean(done & ~suff):.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
