"""Causal LLM gate: from (context, words so far) only, one call predicts whether answering now
would be right. Scored against the draft+judge labels (which saw the full turn).

    python suff_gate.py ann.json suff_dev.json [--n 3000] [--model glm-5-3-flash]
"""
import argparse
import json
import random

import numpy as np

import suff_labels as L

GATE = """Below are moments from casual two-person conversations. Each shows the other person's last
line (context) and what the speaker has said so far (lower case, no punctuation). The speaker may
keep talking. Rate 0-10 how likely it is that the listener already knows what to reply, i.e. that
whatever the speaker says next will NOT change the right response (10 = the reply is clear now;
0 = the point, question or request is still to come).
Reply with one line per item: "<number>: <score>".

"""


def main():
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument('ann')
    ap.add_argument('suff')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--model', default='glm-5-3-flash')
    a = ap.parse_args()
    L.MODEL = a.model
    ann, d = json.load(open(a.ann)), json.load(open(a.suff))
    data = {**{f'tb/{k}': v for k, v in ann['tbdev'].items()}, **{f'oto/{k}': v for k, v in ann['oto'].items()}}
    items = []
    for cid in sorted(data):
        conv = L.conversation(data[cid])
        lab = {(x[0], x[6], x[4]): x for x in d.get(cid, [])}
        for tr in conv:
            for n, t, kind in tr['cands']:
                x = lab.get((tr['s'], tr['end'], n))
                if kind != 'end' and x and x[3] is not None:
                    items.append(dict(cid=cid, s=tr['s'], end=tr['end'], ctx=L.context_at(conv, tr['s'], t),
                                      prefix=' '.join(tr['words'][:n]), y=x[3], kind=kind))
    random.Random(2).shuffle(items)
    items = items[:a.n]
    fmt = lambda it: f'context: "{it["ctx"][-200:] or "(none)"}"  speaker so far: "{" ".join(it["prefix"].split()[-80:])}"'  # noqa: E731
    sc, cost = L.run_batches(GATE, items, fmt, L.score_of)
    p = np.array([(it['y'] >= 8, s) for it, s in zip(items, sc) if s is not None], float)
    print(f'{a.model}: {len(p)} items ${cost:.3f}  gate AUC vs labels {roc_auc_score(p[:, 0], p[:, 1]):.3f}')


if __name__ == '__main__':
    main()
