"""Speculative drafting: hit rate at commit, and can we tell live whether the current draft is final?

    python suff_spec.py suff_dev.json [--sim sim.json] [--rescore rescore.json]

Every candidate's draft in suff_dev.json was judged against the full turn, so keeping the draft
from candidate j until the commit is a hit when score_j >= 8. Redraft schedules compared:
  pause : redraft at segment ends inside the turn (VAD pauses)
  clause: also at clause boundaries inside segments
The turn-end pause itself always yields a fresh full draft (score 10); speculation only matters
when the LLM is slower than the EOT delay, so the hit rates below are for the last draft started
BEFORE the speaker's final pause.

Causal "is the current draft final?" signals at candidate j (target score_j >= 8):
  - position: words so far, candidate index
  - jaccard(draft_{j-1}, draft_j): word overlap of consecutive drafts
  - LLM similarity of consecutive drafts (--sim, from this script's --make-sim)
  - rescoring (--rescore, from suff_rescore.py): per-token log-prob change of draft_{j-1} when
    the new words are added to the prompt
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np

TH = 8

SIM = """Each item has two short replies a listener drafted at two moments while someone was talking.
Rate 0-10 how much they say the same thing (10 = same content and intent, wording aside; 0 = they
respond to different things). Reply with one line per item: "<number>: <score>".

"""


def load(src):
    d = json.load(open(src))
    turns = defaultdict(list)
    for cid, v in d.items():
        for s, t, kind, sc, n, N, end, tail, reply, *_ in v:
            turns[(cid, s, end)].append(dict(t=t, kind=kind, sc=sc, n=n, N=N, reply=reply or ''))
    for c in turns.values():
        c.sort(key=lambda x: x['n'])
    return turns


def words(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def jacc(a, b):
    A, B = words(a), words(b)
    return len(A & B) / max(1, len(A | B))


def pairs(turns):
    """Consecutive (prev, cur) candidates before the turn end, for the stability signals."""
    out = []
    for key, c in turns.items():
        pre = [x for x in c if x['kind'] != 'end' and x['sc'] is not None]
        for j in range(1, len(pre)):
            out.append((key, j, pre[j - 1], pre[j]))
    return out


def hit_rates(turns):
    from_end = []
    for sched in ('pause', 'clause'):
        hits, leads, n_draft, n_turns, none = [], [], [], 0, 0
        for key, c in turns.items():
            n_turns += 1
            pre = [x for x in c if x['kind'] != 'end' and x['sc'] is not None and (sched == 'clause' or x['kind'] == 'pause')]
            n_draft.append(len(pre) + 1)
            if not pre:
                none += 1
                continue
            hits.append(pre[-1]['sc'] >= TH)
            leads.append(key[2] - pre[-1]['t'])
        hits, leads = np.array(hits), np.array(leads)
        print(f'  {sched:6s}: {n_turns} turns, {none / n_turns:.0%} have no draft point before the final pause; '
              f'where they do, last pre-final draft is a hit in {hits.mean():.0%} '
              f'(lead before turn end median {np.median(leads):.2f} s); drafts per turn {np.mean(n_draft):.1f}')
        from_end.append(hits.mean())
    return from_end


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def make_sim(turns, dst):
    import suff_labels as L
    P = pairs(turns)
    items = [dict(cid=k[0], s=k[1], end=k[2], a=a['reply'], b=b['reply'], n=b['n']) for k, j, a, b in P]
    sc, cost = L.run_batches(SIM, items, lambda it: f'A: "{it["a"]}"  B: "{it["b"]}"', L.score_of)
    json.dump({f'{it["cid"]}|{it["s"]}|{it["end"]}|{it["n"]}': s for it, s in zip(items, sc)}, open(dst, 'w'))
    print(f'sim: {len(items)} pairs ${cost:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--make-sim')
    ap.add_argument('--sim')
    ap.add_argument('--rescore')
    a = ap.parse_args()
    turns = load(a.src)
    if a.make_sim:
        return make_sim(turns, a.make_sim)
    print('== hit rate of the last draft started before the final pause')
    hit_rates(turns)

    P = pairs(turns)
    sim = json.load(open(a.sim)) if a.sim else {}
    res = json.load(open(a.rescore)) if a.rescore else {}
    rows = []
    for (cid, s, end), j, prev, cur in P:
        k = f'{cid}|{s}|{end}|{cur["n"]}'
        rows.append(dict(y=cur['sc'] >= TH, n=cur['n'], j=j, jac=jacc(prev['reply'], cur['reply']),
                         sim=sim.get(k), res=res.get(k)))
    y = np.array([r['y'] for r in rows])
    print(f'\n== is the current draft final? {len(rows)} candidates with a previous draft, base rate {y.mean():.2f}')
    feats = {'words so far': [r['n'] for r in rows],
             'jaccard(prev, cur draft)': [r['jac'] for r in rows]}
    if sim:
        feats['LLM similarity(prev, cur draft)'] = [r['sim'] if r['sim'] is not None else 5 for r in rows]
    if res:
        feats['rescore: logp change of prev draft'] = [r['res'] if r['res'] is not None else np.nan for r in rows]
    for name, f in feats.items():
        f = np.array(f, float)
        m = ~np.isnan(f)
        print(f'  {name:36s} AUC {auc(y[m], f[m]):.3f}')
    if sim:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        X = np.stack([np.log1p(feats['words so far']), feats['jaccard(prev, cur draft)'],
                      feats['LLM similarity(prev, cur draft)']] + ([feats['rescore: logp change of prev draft']] if res else []), 1).astype(float)
        g = np.array([k[0] for k, *_ in P])
        m = ~np.isnan(X).any(1)   # rescoring covers a random subset
        X, yy, g = X[m], y[m], g[m]
        p = np.zeros(len(yy))
        for tr, te in GroupKFold(5).split(X, yy, g):
            p[te] = LogisticRegression().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        print(f'  {"combined (logistic, 5-fold)":36s} AUC {auc(yy, p):.3f}  (n={len(yy)})')
        s = np.array(feats['LLM similarity(prev, cur draft)'])
        for th in (8, 9, 10):
            m = s >= th
            print(f'  LLM similarity >= {th}: {m.mean():.0%} of candidates, draft final in {y[m].mean():.0%} '
                  f'(vs {y[~m].mean():.0%} otherwise)')


if __name__ == '__main__':
    main()
