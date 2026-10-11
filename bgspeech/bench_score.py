"""Score bgbench outputs (bench_modal.py fetch) and write a report card.

Run with the TurnBench checkout's venv (plus `modal`, which ssl_turn/pipeline/score.py
imports), from inside the checkout:
    .venv/bin/python /path/to/bgspeech/bench_score.py --plan DIR/plan.json \
        --vap DIR/vap.npz --ssl r016_asr=DIR/r016_asr.npz --members fine1_bal1_s1,fine1_bal1_s2 \
        --out DIR/report

Each model's thresholds are its TurnBench dev operating point on clean audio (highest
recall at FP <= 0.10 over all 38 conversations), held fixed under every background.
Per condition: EOT / INT recall and FP rate, overall and on the user channel (the one with
the background), p50 latency, and false INT per minute on the user channel while the user is
silent and the other speaker talks ("the TV cut the agent off"). TurnBench itself counts FPs
only inside annotated negative spans, so it misses most of those.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ssl_turn'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ssl_turn' / 'pipeline'))
import bench  # noqa: E402
from turnbench.data import DEV_DATASET, conversation, resolve_dataset  # noqa: E402
from turnbench.gold import events_for_conversation  # noqa: E402
from turnbench.score import TaskScore, merge, score_task  # noqa: E402

FP_BUDGET = 0.10


def commit(p, fps, theta, refractory_s, recommit_s):
    from score import commit as ssl_commit  # ssl_turn/pipeline/score.py: rising edge + optional re-commit
    return ssl_commit(p, fps, theta, refractory_s, recommit_s)


class Model:
    """tracks[cond][task][cid] -> [T, 2] scores; commit rule per task."""

    def __init__(self, name, fps, tracks, refractory, recommit):
        self.name, self.fps, self.tracks = name, fps, tracks
        self.refractory, self.recommit = refractory, recommit
        self._ev = {}

    def events(self, cond, task, cid, theta):
        key = (cond, task, cid, theta)
        if key not in self._ev:
            p = self.tracks[cond][task][cid]
            self._ev[key] = {s + 1: commit(p[:, s], self.fps, theta, self.refractory[task], self.recommit[task])
                             for s in (0, 1)}
        return self._ev[key]


def load_vap(path, ids):
    z = np.load(path)
    tracks = {}
    for cond in bench.CONDITIONS:
        if all(f'{c}/{cond}' in z.files for c in ids):
            p = {c: z[f'{c}/{cond}'].astype(np.float64) for c in ids}
            tracks[cond] = {'eot': {c: 1 - v for c, v in p.items()}, 'int': p}
    return Model('VAP (oto)', 50.0, tracks, {'eot': 2.0, 'int': 2.0}, {'eot': None, 'int': None})


def load_ssl(path, ids, run, member):
    from score import score_variants
    z = np.load(path)
    tracks = {}
    for cond in bench.CONDITIONS:
        pre = lambda c: f'{c}/{member}@{cond}/tbdev/{c}/'
        if all(pre(c) + 'post' in z.files for c in ids):
            tracks[cond] = {'eot': {}, 'int': {}}
            for c in ids:
                v = score_variants(*(z[pre(c) + k].astype(np.float32) for k in ('post', 'silent', 'fine')))
                tracks[cond]['eot'][c], tracks[cond]['int'][c] = v['eot_q'], v['int_nobc']
    # ssl_turn's commit policy (HANDOFF.md): 0.5 s refractory, EOT re-commit after 1 s.
    return Model(f'{run} {member}', 12.5, tracks, {'eot': 0.5, 'int': 0.5}, {'eot': 1.0, 'int': None})


def score(model, cond, task, theta, ids, gold, plan):
    out = {k: TaskScore() for k in ('all', 'user', 'other')}
    for cid in ids:
        ev = model.events(cond, task, cid, theta)
        g = gold[cid]
        pos, neg, exc = ((g.eot_positive_events, g.eot_negative_spans, g.eot_excluded) if task == 'eot'
                         else (g.int_positive_events, g.int_negative_spans, g.int_excluded))
        merge(out['all'], score_task(pos, neg, ev, exc))
        user = plan['items'][cid]['user']
        for key, spk in (('user', user), ('other', 3 - user)):
            merge(out[key], score_task([e for e in pos if e.speaker == spk], [s for s in neg if s.speaker == spk],
                                       ev, exc))
    return out


def op_point(model, task, ids, gold, plan):
    pooled = np.concatenate([model.tracks['clean'][task][c].ravel() for c in ids])
    grid = np.unique(np.concatenate([np.quantile(pooled, np.linspace(0, 1, 129)), np.arange(1, 100) * 0.01]))
    best = None
    for th in grid:
        s = score(model, 'clean', task, float(th), ids, gold, plan)['all']
        if s.fp_rate <= FP_BUDGET and (best is None or s.recall > best[1].recall):
            best = (float(th), s)
    return best[0]


def false_int_rate(model, cond, theta, ids, convs, plan):
    """INT fires per minute on each channel while its speaker is silent and the other talks."""
    fires, minutes = {'user': 0, 'other': 0}, {'user': 0.0, 'other': 0.0}
    for cid in ids:
        ev = model.events(cond, 'int', cid, theta)
        n = len(model.tracks[cond]['int'][cid])
        user = plan['items'][cid]['user']
        ann = convs[cid].annotations
        for key, spk in (('user', user), ('other', 3 - user)):
            listening = ~bench.activity(ann, spk, n, model.fps, 0.3) & bench.activity(ann, 3 - spk, n, model.fps)
            fires[key] += sum(1 for t in ev[spk] if listening[min(n - 1, int(round(t * model.fps)) - 1)])
            minutes[key] += listening.sum() / model.fps / 60
    return {k: fires[k] / minutes[k] for k in fires}


def summary(s: TaskScore):
    return dict(recall=s.recall, fp_rate=s.fp_rate, p50_ms=s.latency().p50, tp=s.tp, fn=s.fn, fp=s.fp, tn=s.tn)


def evaluate(model, ids, gold, convs, plan):
    theta = {t: op_point(model, t, ids, gold, plan) for t in ('eot', 'int')}
    res = {'theta_clean': theta}
    for cond in bench.CONDITIONS:
        if cond not in model.tracks:
            continue
        r = res[cond] = {t: {k: summary(v) for k, v in score(model, cond, t, theta[t], ids, gold, plan).items()}
                         for t in ('eot', 'int')}
        r['false_int_per_min'] = false_int_rate(model, cond, theta['int'], ids, convs, plan)
        e, i, f = r['eot'], r['int'], r['false_int_per_min']
        print(f"{model.name:>26} {cond:>8} | EOT {e['all']['recall']:.3f}/{e['all']['fp_rate']:.3f}"
              f" user {e['user']['recall']:.3f} p50 {e['user']['p50_ms']:4.0f}ms | INT {i['all']['recall']:.3f}"
              f"/{i['all']['fp_rate']:.3f} user-FP {i['user']['fp_rate']:.3f} | false INT/min user {f['user']:.2f}",
              flush=True)
    return res


def row(results, cond):
    """Mean over a model's seeds of the report-card numbers for one condition."""
    rs = [r[cond] for r in results if cond in r]
    m = lambda f: float(np.mean([f(r) for r in rs]))
    return dict(eot=m(lambda r: r['eot']['all']['recall']), eot_fp=m(lambda r: r['eot']['all']['fp_rate']),
                eot_user=m(lambda r: r['eot']['user']['recall']),
                int=m(lambda r: r['int']['all']['recall']), int_fp_user=m(lambda r: r['int']['user']['fp_rate']),
                false_int=m(lambda r: r['false_int_per_min']['user']))


def report(groups, n_convs, hours):
    """Markdown report card: headline per background group, then every condition."""
    lines = [f'# bgbench report ({bench.VERSION})', '',
             f'{n_convs} TurnBench dev conversations, {hours:.1f} h. Background in the user channel only; '
             'thresholds fixed at each model\'s clean operating point (FP <= 0.10). Seed means where a model '
             'has several seeds. "false INT/min": INT fires on the user channel per minute of the user '
             'listening (silent while the other speaker talks).', '',
             '## Headline (mean over each group\'s conditions)', '',
             '| model | clean EOT / INT | group | user EOT recall (drop) | user INT FP | false INT/min |',
             '|---|---|---|---|---|---|']
    for name, results in groups.items():
        clean = row(results, 'clean')
        for g, conds in bench.GROUPS.items():
            rs = [row(results, c) for c in conds if all(c in r for r in results)]
            if not rs:
                continue
            u = np.mean([r['eot_user'] for r in rs])
            lines.append(f"| {name} | {clean['eot']:.3f} / {clean['int']:.3f} | {g} | {u:.3f} "
                         f"({u - clean['eot_user']:+.3f}) | {np.mean([r['int_fp_user'] for r in rs]):.3f} | "
                         f"{np.mean([r['false_int'] for r in rs]):.1f} |")
    lines += ['', '## All conditions', '',
              '| model | condition | EOT recall / FP | user EOT recall | INT recall | user INT FP | false INT/min |',
              '|---|---|---|---|---|---|---|']
    for name, results in groups.items():
        for cond in bench.CONDITIONS:
            if not all(cond in r for r in results):
                continue
            r = row(results, cond)
            lines.append(f"| {name} | {cond} | {r['eot']:.3f} / {r['eot_fp']:.3f} | {r['eot_user']:.3f} | "
                         f"{r['int']:.3f} | {r['int_fp_user']:.3f} | {r['false_int']:.1f} |")
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--vap')
    ap.add_argument('--ssl', action='append', default=[],
                    help='LABEL=path.npz[:member,member] (repeatable; members default to --members)')
    ap.add_argument('--members', default='fine1_bal1_s1,fine1_bal1_s2')
    ap.add_argument('--out', required=True, help='output prefix: writes OUT.json and OUT.md')
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    assert plan['version'] == bench.VERSION, (plan['version'], bench.VERSION)
    ids = plan['ids']
    ds = resolve_dataset(DEV_DATASET, skip_audio=True)
    convs = {c: conversation(ds, c) for c in ids}
    gold = {c: events_for_conversation(convs[c]) for c in ids}
    hours = sum(c.duration_s for c in convs.values()) / 3600
    print(f'{len(ids)} conversations, {hours:.2f} h')
    groups, results = {}, {}
    if a.vap:
        m = load_vap(a.vap, ids)
        results[m.name] = evaluate(m, ids, gold, convs, plan)
        groups['VAP (oto)'] = [results[m.name]]
    for spec in a.ssl:
        run, path = spec.split('=', 1)
        path, _, members = path.partition(':')
        groups[run] = []
        for member in (members or a.members).split(','):
            m = load_ssl(path, ids, run, member)
            results[m.name] = evaluate(m, ids, gold, convs, plan)
            groups[run].append(results[m.name])
    Path(a.out + '.json').write_text(json.dumps(dict(version=bench.VERSION, results=results), indent=1))
    Path(a.out + '.md').write_text(report(groups, len(ids), hours))
    print(Path(a.out + '.md').read_text())


if __name__ == '__main__':
    main()
