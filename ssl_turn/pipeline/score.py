"""TurnBench-style sweep scoring with the pinned evaluator's own functions (CPU).

Probabilities: /work/runs/{run}/probs.npz with keys '{split}/{cid}/{task}' -> [T, 2]
on the 12.5 Hz grid, where frame i covers [i*80 ms, (i+1)*80 ms) and commits at its end,
exactly the turnbench.sweep grid. Rising-edge commit, 2 s refractory, and score_task
are imported from turnbench@38a6f87 unchanged.

Gold:
  tbdev: turnbench.gold.events_for_conversation on the 3-annotator dev set (cached once)
  oto:   single-annotator otoSpeech dev events from prep.py, used for model selection
"""
import json

import modal

from common import TB_DEV, VOLUMES, cpu_image, work

app = modal.App('ssl-turn-score')


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=16384, timeout=3600)
def cache_tbdev_gold():
    import os
    from dataclasses import asdict
    from turnbench.data import conversation, conversation_ids, resolve_dataset
    from turnbench.gold import events_for_conversation
    path = '/work/gold/tbdev.json'
    if os.path.exists(path):
        return len(json.load(open(path)))
    ds = resolve_dataset(TB_DEV)
    out = {}
    for cid in conversation_ids(ds):
        conv = conversation(ds, cid)
        out[cid] = dict(duration_s=conv.duration_s, events=asdict(events_for_conversation(conv)))
    os.makedirs('/work/gold', exist_ok=True)
    json.dump(out, open(path, 'w'))
    work.commit()
    return len(out)


def load_gold(split, cids):
    if split == 'tbdev':
        g = json.load(open('/work/gold/tbdev.json'))
        return {c: g[c] for c in cids}
    return {c: json.load(open(f'/work/gold/oto/{c}.json')) for c in cids}


def sweep_task(probs, gold, task, thetas=None, fps=12.5):
    import numpy as np
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import TaskScore, merge, score_task
    from turnbench.sweep import commit_events
    pooled = np.concatenate([p.ravel() for p in probs.values()])
    pooled = pooled[pooled > 0]
    if thetas is None:
        grid = np.round(np.arange(1, 100) * 0.01, 2)
        thetas = np.unique(np.concatenate([np.quantile(pooled, np.linspace(0, 1, 256)), grid]))
    key = 'eot' if task == 'eot' else 'int'
    golds = {}
    for cid in probs:
        e = gold[cid]['events']
        golds[cid] = ([AnchorEvent(**x) for x in e[f'{key}_positive_events']],
                      [Interval(**x) for x in e[f'{key}_negative_spans']],
                      [Interval(**x) for x in e[f'{key}_excluded']])
    rows = []
    for theta in thetas:
        total = TaskScore()
        for cid, p in probs.items():
            ev = {s + 1: commit_events(p[:, s], fps, float(theta)) for s in (0, 1)}
            merge(total, score_task(*golds[cid][:2], ev, golds[cid][2]))
        lat = total.latency()
        rows.append(dict(theta=float(theta), recall=total.recall, fp=total.fp_rate, p50=lat.p50, p10=lat.p10,
                         p90=lat.p90, tp=total.tp, fn=total.fn, fpn=total.fp))
    return rows


def operating_point(rows, budget=0.10):
    ok = [r for r in rows if r['fp'] == r['fp'] and r['fp'] <= budget]
    return max(ok, key=lambda r: (r['recall'], -r['p50'] if r['p50'] == r['p50'] else 0)) if ok else None


def at_theta(rows, theta):
    return min(rows, key=lambda r: abs(r['theta'] - theta))


@app.function(image=cpu_image, volumes=VOLUMES, cpu=8, memory=16384, timeout=3600)
def score_run(run, variants=None):
    """Sweep every (model, task variant) in a run on oto dev (selection) and tbdev (report)."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    work.reload()
    z = np.load(f'/work/runs/{run}/probs.npz')
    by = {}
    for k in z.files:
        model, split, cid, task = k.split('/')
        by.setdefault((model, task), {}).setdefault(split, {})[cid] = z[k]
    jobs = {}
    with ProcessPoolExecutor(8) as pool:
        for (model, task), splits in by.items():
            if variants and task not in variants:
                continue
            for split, probs in splits.items():
                jobs[(model, task, split)] = pool.submit(sweep_task, probs, load_gold(split, list(probs)),
                                                         'eot' if task.startswith('eot') else 'int')
        rows = {k: f.result() for k, f in jobs.items()}
    report = []
    for (model, task, split), r in sorted(rows.items()):
        if split != 'oto':
            continue
        sel = operating_point(r)
        tb = rows.get((model, task, 'tbdev'))
        entry = dict(model=model, task=task, oto=sel)
        if tb:
            entry['tbdev_at_oto_theta'] = at_theta(tb, sel['theta']) if sel else None
            entry['tbdev_swept'] = operating_point(tb)
        report.append(entry)
    json.dump(dict(report=report, rows={'|'.join(k): v for k, v in rows.items()}),
              open(f'/work/runs/{run}/score.json', 'w'))
    work.commit()
    return report


def fmt(r):
    if not r:
        return '            —'
    return f"R {r['recall']:.3f} FP {r['fp']:.3f} p50 {r['p50']:.0f}ms θ {r['theta']:.3f}"


@app.local_entrypoint()
def main(run: str):
    for e in score_run.remote(run):
        print(f"{e['model']:>14} {e['task']:<8} oto[{fmt(e['oto'])}]  tbdev@oto-θ[{fmt(e.get('tbdev_at_oto_theta'))}]"
              f"  tbdev-swept[{fmt(e.get('tbdev_swept'))}]")
