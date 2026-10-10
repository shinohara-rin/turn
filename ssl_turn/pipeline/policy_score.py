"""Score a run through the commit policy (../policy.py) on TB dev: VAD + rules, with the run's
score tracks as extra triggers. Needs {WORK}/vad/tbdev (vad.py) and {WORK}/gold/tbdev.json
(`modal run score.py::cache_tbdev_gold`).

    modal run policy_score.py::main --run r019_asr_bgaug --model bgaug_s1
    modal run policy_score.py::sweep --run r019_asr_bgaug --models bgaug_s1,bgaug_s2
    modal run policy_score.py::export --run r019_asr_bgaug --model bgaug_s1

`sweep` re-selects the policy settings on a grid and reports, per family (rules only,
rules + model, rules + model with a median-latency cap), the dev operating point and a
split-half estimate: settings picked on 19 random conversations, scored on the other 19.
"""
import json

import modal

from common import SRC, VOLUMES, WORK, cpu_image, work

app = modal.App('ssl-turn-policy')
BUDGETS = {'eot': (0.08, 0.10), 'int': (0.05, 0.10)}


def _setup():
    import sys
    from common import setup_path
    setup_path()
    sys.path.insert(0, '/root/ssl_turn/pipeline')  # score.py


def load_inputs(run, model, eot_var, int_var, int_run='', int_model=''):
    """INT tracks come from `int_run` / `int_model` when given (e.g. onset.py), else from run / model."""
    import numpy as np
    from score import load_gold, load_tracks
    work.reload()
    gold = json.load(open(f'{WORK}/gold/tbdev.json'))
    cids = sorted(gold, key=int)
    vad = {c: np.load(f'{WORK}/vad/tbdev/{c}.npy').astype(np.float32) / 255 for c in cids}
    eot = load_tracks(run, model, eot_var, 'tbdev') if model else {}
    int_run, int_model = int_run or run, int_model or model
    intr = load_tracks(int_run, int_model, int_var, 'tbdev') if int_model else {}
    return cids, load_gold('tbdev', cids), vad, eot, intr


def score_conv(task, events, gold):
    """One conversation's TaskScore for one task."""
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import score_task
    key = 'eot' if task == 'eot' else 'int'
    e = gold['events']
    return score_task([AnchorEvent(**x) for x in e[f'{key}_positive_events']],
                      [Interval(**x) for x in e[f'{key}_negative_spans']], events,
                      [Interval(**x) for x in e[f'{key}_excluded']])


def summarize(scores):
    import numpy as np
    tp, fn, fp, tn = (sum(getattr(s, k) for s in scores) for k in ('tp', 'fn', 'fp', 'tn'))
    lat = [x for s in scores for x in s.latencies_ms]
    return dict(recall=tp / (tp + fn), fp=fp / (fp + tn), p50=float(np.median(lat)) if lat else float('nan'),
                tp=tp, fn=fn, fp_n=fp)


def per_conv(task, P, cids, gold, vad, eot, intr):
    from policy import conversation_events
    out = []
    for c in cids:
        ev = conversation_events(vad[c], gold[c]['duration_s'], P, eot.get(c), intr.get(c))
        out.append(score_conv(task, ev[task], gold[c]))
    return out


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=1800)
def evaluate(run='', model='', eot_var='eot_q', int_var='int_ft', overrides=None, int_run='', int_model=''):
    """Both tasks at the default Policy (plus overrides)."""
    _setup()
    from policy import Policy
    cids, gold, vad, eot, intr = load_inputs(run, model, eot_var, int_var, int_run, int_model)
    P = Policy().but(**(overrides or {}))
    if not model:
        P = P.but(eot_th=None, int_th=None)
    return {t: summarize(per_conv(t, P, cids, gold, vad, eot, intr)) for t in ('eot', 'int')}


INT_THRESHOLDS = {'int_ft': (0.05, 0.1, 0.15, 0.2, 0.3, 0.5), 'int_nobc': (0.3, 0.4, 0.5, 0.7, 0.9),
                  'int_onset': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}


def _grid(task, models, eot_vars, int_vars):
    """[(model, score variant, Policy)]; model '' is rules only."""
    import itertools
    from policy import Policy
    base, out = Policy(), []
    if task == 'eot':
        for vt, od, rm, cf, dl in itertools.product((0.5, 0.7), (0.2, 0.3), (0.3, 0.5), (None, 2.5), (2.0, 2.5)):
            rules = base.but(vad_th=vt, other_dur=od, resume_min=rm, confirm=cf, deadline=dl, eot_th=None,
                             min_wait=0.6)
            out.append(('', '', rules))
            for m, v, th, mw, mwm in itertools.product(models, eot_vars, (0.5, 0.7, 0.8, 0.9, 0.95), (0.6, 0.8),
                                                       (0.2, 0.32, 0.48)):
                out.append((m, v, rules.but(eot_th=th, min_wait=mw, mw_model=mwm)))
    else:
        for vt, d in itertools.product((0.5, 0.7), [round(0.5 + 0.1 * i, 1) for i in range(8)]):
            out.append(('', '', base.but(vad_th=vt, int_dur=d, int_th=None, int_fallback=True)))
            for m, v in itertools.product(models, int_vars):
                for th, fr, fb in itertools.product(INT_THRESHOLDS.get(v, (0.3, 0.5, 0.7, 0.9)), (0.16, 0.24),
                                                    (True, False)):
                    out.append((m, v, base.but(vad_th=vt, int_dur=d, int_th=th, int_from=fr, int_fallback=fb)))
    return out


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=3600)
def _sweep_chunk(run, task, items, int_run=''):
    _setup()
    from policy import Policy
    cache, rows = {}, []
    for m, v, kw in items:
        if (m, v) not in cache:
            cache[m, v] = load_inputs(run, m, v, v, int_run if task == 'int' else '')
        rows.append([(s.tp, s.fn, s.fp, s.tn, s.latencies_ms)
                     for s in per_conv(task, Policy(**kw), *cache[m, v])])
    return rows


def _agg(rows, idx=None):
    import numpy as np
    rows = rows if idx is None else [rows[i] for i in idx]
    tp, fn, fp, tn = (sum(r[j] for r in rows) for j in range(4))
    lat = [x for r in rows for x in r[4]]
    return dict(recall=tp / (tp + fn), fp=fp / (fp + tn), p50=float(np.median(lat)) if lat else float('nan'))


def _pick(scores, budget, family, cap):
    ok = [k for k in family if scores[k]['fp'] <= budget and scores[k]['p50'] <= cap]
    return max(ok, key=lambda k: (scores[k]['recall'], -scores[k]['p50'])) if ok else None


@app.local_entrypoint()
def sweep(run: str, models: str, eot_vars: str = 'eot_q', int_vars: str = 'int_ft,int_nobc', splits: int = 200,
          caps: str = '500,400', tasks: str = 'eot,int', int_run: str = '', int_models: str = ''):
    import random
    from dataclasses import asdict
    import numpy as np
    import sys
    sys.path.insert(0, str(SRC))  # policy.py, for the grid
    from policy import Policy
    models = [m for m in models.split(',') if m]
    base, report = Policy(), {}
    for task in tasks.split(','):
        ms = [m for m in int_models.split(',') if m] if task == 'int' and int_models else models
        grid = [(m, v, asdict(P)) for m, v, P in _grid(task, ms, eot_vars.split(','), int_vars.split(','))]
        chunks = [grid[i:i + 64] for i in range(0, len(grid), 64)]
        R = [r for rows in _sweep_chunk.starmap([(run, task, c, int_run) for c in chunks]) for r in rows]
        n = len(R[0])
        rng = random.Random(0)
        halves = []
        for _ in range(splits):
            a = rng.sample(range(n), n // 2)
            halves.append((a, [i for i in range(n) if i not in a]))
        full = [_agg(r) for r in R]
        SA = [[_agg(r, a) for r in R] for a, _ in halves]
        SB = [[_agg(r, b) for r in R] for _, b in halves]
        rules = [k for k, g in enumerate(grid) if not g[0]]
        model = [k for k, g in enumerate(grid) if g[0]]
        fams = [('rules', rules, 1e9), ('rules+model', model, 1e9)]
        fams += [(f'rules+model p50<={c}', model, float(c)) for c in caps.split(',') if c]
        for budget in BUDGETS[task]:
            for name, fam, cap in fams:
                k = _pick(full, budget, fam, cap)
                if k is None:
                    print(task, budget, name, 'none')
                    continue
                held = []
                for i in range(splits):
                    kk = _pick(SA[i], budget, fam, cap)
                    if kk is not None:
                        held.append((SB[i][kk]['recall'], SB[i][kk]['fp']))
                held = np.array(held)
                row = dict(dev=full[k], model=grid[k][0], variant=grid[k][1], policy=grid[k][2],
                           heldout=dict(recall=float(held[:, 0].mean()), fp=float(held[:, 1].mean()),
                                        fp_p95=float(np.quantile(held[:, 1], 0.95)), n=len(held)))
                report[f'{task}/{budget}/{name}'] = row
                d, h = row['dev'], row['heldout']
                print(f"{task} {budget} {name}: R {d['recall']:.3f} FP {d['fp']:.3f} p50 {d['p50']:.0f} | "
                      f"held-out R {h['recall']:.3f} FP {h['fp']:.3f} (p95 {h['fp_p95']:.3f}) | {row['model']} "
                      f"{row['variant']}", {k: v for k, v in row['policy'].items() if v != getattr(base, k)})
    out = f'policy_sweep_{run}_{tasks.replace(",", "_")}.json'
    json.dump(report, open(out, 'w'), indent=1)
    print('wrote', out)


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=1800)
def export_predictions(run, model, eot_var='eot_q', int_var='int_ft', name='predictions-dev-policy', int_run='',
                       int_model=''):
    """Official predictions JSON for TB dev at the default Policy, validated and scored with the
    pinned evaluator."""
    import os
    from pathlib import Path
    _setup()
    from policy import Policy, conversation_events
    from turnbench.check import check_predictions
    from turnbench.durations import load_durations
    cids, gold, vad, eot, intr = load_inputs(run, model, eot_var, int_var, int_run, int_model)
    P, preds, scores = Policy(), [], {'eot': [], 'int': []}
    for c in cids:
        dur = gold[c]['duration_s']
        ev = conversation_events(vad[c], dur, P, eot.get(c), intr.get(c))
        for t in scores:
            scores[t].append(score_conv(t, ev[t], gold[c]))
        entry = dict(conversation_id=c)
        for s in (1, 2):
            entry[f'speaker_{s}'] = dict(eot=[x for x in ev['eot'][s] if x <= dur],
                                         interruption=[x for x in ev['int'][s] if x <= dur])
        preds.append(entry)
    out = f'{WORK}/runs/{run}/{name}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(schema_version=1, predictions=preds), open(out, 'w'))
    work.commit()
    check_predictions(Path(out), load_durations('dev'))  # raises on any schema/coverage/time violation
    return out, {t: summarize(s) for t, s in scores.items()}


@app.local_entrypoint()
def export(run: str, model: str, eot_var: str = 'eot_q', int_var: str = 'int_ft', int_run: str = '', int_model: str = ''):
    out, res = export_predictions.remote(run, model, eot_var, int_var, int_run=int_run, int_model=int_model)
    print(out)
    for t, r in res.items():
        print(t, f"R {r['recall']:.3f} FP {r['fp']:.3f} p50 {r['p50']:.0f}")


@app.local_entrypoint()
def main(run: str = '', model: str = '', eot_var: str = 'eot_q', int_var: str = 'int_ft', int_run: str = '',
         int_model: str = ''):
    for t, r in evaluate.remote(run, model, eot_var, int_var, int_run=int_run, int_model=int_model).items():
        print(t, f"R {r['recall']:.3f} FP {r['fp']:.3f} p50 {r['p50']:.0f}")
