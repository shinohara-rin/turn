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


def score_variants(post, silent=None):
    """post [T, 1 + H, 4]: floor now and at 0.4/0.8/1.6 s (HELD_0, HELD_1, OPEN, CONTESTED)
    -> per-speaker [T, 2] score tracks."""
    import numpy as np
    now, f04, f08, f16 = post[:, 0], post[:, 1], post[:, 2], post[:, 3]
    def released(p):  # speaker c no longer holds: OPEN or held by the other
        return np.stack([p[:, 2] + p[:, 1], p[:, 2] + p[:, 0]], 1)
    def holds(p):
        return np.stack([p[:, 0], p[:, 1]], 1)
    eot_now, eot_f04 = released(now), released(f04)
    clip = lambda v: np.clip(v, 0, 1).astype(np.float32)

    def past_max(x, frames):  # causal: max over the previous `frames` frames (excluding now)
        out = np.zeros_like(x)
        for k in range(1, frames + 1):
            out[k:] = np.maximum(out[k:], x[:-k])
        return out

    held_now = holds(now)
    other = np.stack([now[:, 1], now[:, 0]], 1)
    out = dict(eot=clip(eot_now), eot_f04=clip(eot_f04), eot_mix=clip((eot_now + eot_f04) / 2),
               int=clip(holds(f08)), int_f04=clip(holds(f04)), int_f16=clip(holds(f16)),
               int_mix=clip((holds(f04) + holds(f08) + holds(f16)) / 3))
    # Transition scores: released now x c held the floor recently (a hand-over, not a state).
    for frames in (5, 10, 20):
        held_recent = past_max(held_now, frames)
        out[f'eot_tr{frames}'] = clip(eot_now * held_recent)
        out[f'eot_f04_tr{frames}'] = clip(eot_f04 * held_recent)
    out['int_tr10'] = clip(holds(f08) * past_max(other, 10))
    if silent is not None:  # gate EOT by the speaker's own causal p(SILENT) from the act head
        out['eot_q'] = clip(eot_now * silent)
        out['eot_f04_q'] = clip(eot_f04 * silent)
        out['eot_mix_q'] = clip((eot_now + eot_f04) / 2 * silent)
        out['eot_q_sqrt'] = clip(eot_now * np.sqrt(silent))
        out['int_spk'] = clip(holds(f04) * (1 - silent))  # c must be vocalizing to take the floor
    return out


def load_tracks(run, model, task, split):
    """Per-conversation [T, 2] score track for one model and score variant."""
    import numpy as np
    work.reload()
    z = np.load(f'/work/runs/{run}/probs.npz')
    tracks = {}
    for k in z.files:
        m_, sp, cid, t = k.split('/')
        if m_ == model and sp == split:
            if t == 'post':
                sk = k[:-len('post')] + 'silent'
                tracks[cid] = score_variants(z[k].astype(np.float32),
                                             z[sk].astype(np.float32) if sk in z.files else None)[task]
            elif t == task:
                tracks[cid] = z[k]
    return tracks


def load_gold(split, cids):
    if split == 'tbdev':
        g = json.load(open('/work/gold/tbdev.json'))
        return {c: g[c] for c in cids}
    return {c: json.load(open(f'/work/gold/oto/{c}.json')) for c in cids}


def commit(p, fps, theta, refractory_s=2.0, recommit_s=None):
    """Pinned rising-edge commit; optionally re-commit once the score has stayed above
    theta for `recommit_s` since the last commit (one re-commit per continuous run)."""
    import numpy as np
    from turnbench.sweep import commit_events
    times = commit_events(p, fps, theta, refractory_s=refractory_s)
    if not recommit_s:
        return times
    above = np.asarray(p) > theta
    extra, k = [], int(round(recommit_s * fps))
    for t in times:
        i = int(round(t * fps)) - 1  # frame index of the commit
        j = i + k
        if j < len(above) and above[i:j + 1].all():
            extra.append((j + 1) / fps)
    return sorted(set(times) | set(extra))


def sweep_task(probs, gold, task, thetas=None, fps=12.5, refractory_s=2.0, recommit_s=None):
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
            ev = {s + 1: commit(p[:, s], fps, float(theta), refractory_s, recommit_s) for s in (0, 1)}
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
def score_run(run, variants=None, refractories=(2.0,), ensembles=None, recommits=(None,)):
    """Sweep every (model, task variant) in a run on oto dev (selection) and tbdev (report)."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    work.reload()
    z = np.load(f'/work/runs/{run}/probs.npz')
    by, posts = {}, {}
    for k in z.files:
        model, split, cid, task = k.split('/')
        if task == 'post':
            sk = k[:-len('post')] + 'silent'
            posts.setdefault(model, {})[(split, cid)] = (z[k].astype(np.float32),
                                                         z[sk].astype(np.float32) if sk in z.files else None)
        elif task != 'silent':
            by.setdefault((model, task), {}).setdefault(split, {})[cid] = z[k]
    for name, members in (ensembles or {}).items():  # average posteriors across models
        members = [m_ for m_ in members if m_ in posts]
        keys = set.intersection(*(set(posts[m_]) for m_ in members))
        posts[name] = {key: (np.mean([posts[m_][key][0] for m_ in members], 0),
                             np.mean([posts[m_][key][1] for m_ in members], 0)) for key in keys}
    for model, items in posts.items():
        for (split, cid), (post, silent) in items.items():
            for name, v in score_variants(post, silent).items():
                by.setdefault((model, name), {}).setdefault(split, {})[cid] = v
    jobs = {}
    with ProcessPoolExecutor(8) as pool:
        for (model, task), splits in by.items():
            if variants and task not in variants:
                continue
            for split, probs in splits.items():
                for r in refractories:
                    for rc in recommits:
                        name = (task if r == 2.0 else f'{task}@r{r}') + (f'+rc{rc}' if rc else '')
                        jobs[(model, name, split)] = pool.submit(sweep_task, probs, load_gold(split, list(probs)),
                                                                 'eot' if task.startswith('eot') else 'int', None,
                                                                 12.5, r, rc)
        rows = {k: f.result() for k, f in jobs.items()}
    report = []
    for (model, task) in sorted({(m_, t) for m_, t, _ in rows}):
        r = rows.get((model, task, 'oto'))
        sel = operating_point(r) if r else None
        tb = rows.get((model, task, 'tbdev'))
        entry = dict(model=model, task=task, oto=sel)
        if tb:
            entry['tbdev_at_oto_theta'] = at_theta(tb, sel['theta']) if sel else None
            entry['tbdev_swept'] = operating_point(tb)
        report.append(entry)
    json.dump(dict(report=report, rows={'|'.join(k): v for k, v in rows.items()}),
              open(f'/work/runs/{run}/score' + ('_' + '-'.join(variants) if variants else '') + '.json', 'w'))
    work.commit()
    return report


def fmt(r):
    if not r:
        return '            —'
    return f"R {r['recall']:.3f} FP {r['fp']:.3f} p50 {r['p50']:.0f}ms θ {r['theta']:.3f}"


@app.local_entrypoint()
def main(run: str, variants: str = '', refractories: str = '2.0', ensembles: str = '', recommits: str = ''):
    ens = json.loads(ensembles) if ensembles else None
    rcs = tuple(float(x) if x else None for x in recommits.split(',')) if recommits else (None,)
    for e in score_run.remote(run, variants.split(',') if variants else None,
                              tuple(float(x) for x in refractories.split(',')), ens, rcs):
        print(f"{e['model']:>14} {e['task']:<8} oto[{fmt(e['oto'])}]  tbdev@oto-θ[{fmt(e.get('tbdev_at_oto_theta'))}]"
              f"  tbdev-swept[{fmt(e.get('tbdev_swept'))}]")


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=16384, timeout=1800)
def analyze(run, model, task, theta, split='tbdev', refractory_s=2.0):
    """Error anatomy at one operating point: per conversation type, FP by pause length,
    latency distribution, and how many gold events never get a fire within 3 s."""
    import numpy as np
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import TaskScore, merge, score_task
    from turnbench.sweep import commit_events
    work.reload()
    probs = load_tracks(run, model, task, split)
    gold = load_gold(split, list(probs))
    types = json.load(open('/work/gold/tbdev_types.json')) if split == 'tbdev' else {}
    key = 'eot' if task.startswith('eot') else 'int'
    by_type, fp_len, neg_len, lat = {}, [], [], []
    for cid, p in probs.items():
        e = gold[cid]['events']
        pos = [AnchorEvent(**x) for x in e[f'{key}_positive_events']]
        neg = [Interval(**x) for x in e[f'{key}_negative_spans']]
        exc = [Interval(**x) for x in e[f'{key}_excluded']]
        ev = {s + 1: commit_events(p[:, s], 12.5, float(theta), refractory_s=refractory_s) for s in (0, 1)}
        sc = score_task(pos, neg, ev, exc)
        merge(by_type.setdefault(types.get(cid, 'all'), TaskScore()), sc)
        for span in neg:
            fired = any(span.start <= t <= span.end for t in ev[span.speaker])
            (fp_len if fired else neg_len).append(span.end - span.start)
        lat += [l for l in getattr(sc, 'latencies', [])]
    out = {t: dict(recall=s.recall, fp=s.fp_rate, tp=s.tp, fn=s.fn, n_fp=s.fp) for t, s in by_type.items()}
    q = lambda a: [float(np.round(v, 2)) for v in np.quantile(a, [.1, .5, .9])] if len(a) else None
    out['fp_pause_len_q10_50_90'] = q(fp_len)
    out['clean_pause_len_q10_50_90'] = q(neg_len)
    return out


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=16384, timeout=1800)
def miss_anatomy(run, model, task, theta, split='tbdev', refractory_s=2.0):
    """Why EOT positives are missed at threshold theta (rising-edge rule, 2 s refractory):
    'prefired' = score already above theta when the window opens (no fresh edge),
    'refractory' = an edge in the window was suppressed by an earlier commit,
    'never' = score stays below theta for the whole window, plus the peak score."""
    import numpy as np
    from turnbench.sweep import commit_events
    tracks = load_tracks(run, model, task, split)
    gold = load_gold(split, list(tracks))
    cats, peaks, quiet_len = {}, [], []
    fps = 12.5
    for cid, p in tracks.items():
        e = gold[cid]['events']
        fires = {s + 1: commit_events(p[:, s], fps, float(theta), refractory_s=refractory_s) for s in (0, 1)}
        pos = sorted(e['eot_positive_events'], key=lambda x: (x['speaker'], x['time_s']))
        for i, ev in enumerate(pos):
            s, t = ev['speaker'], ev['time_s']
            nxt = [x['time_s'] for x in pos if x['speaker'] == s and x['time_s'] > t]
            lo, hi = t - 0.25, min(t + 3.0, nxt[0] if nxt else t + 3.0)
            if any(lo <= f <= hi for f in fires[s]):
                cats['hit'] = cats.get('hit', 0) + 1
                continue
            a, b = int(max(0, np.floor(lo * fps))), int(min(len(p), np.ceil(hi * fps)))
            w = p[a:b, s - 1]
            if len(w) == 0:
                continue
            above = w > theta
            if above[0] and a > 0 and p[a - 1, s - 1] > theta:
                c = 'prefired'
            elif above.any():
                c = 'refractory'
            else:
                c = 'never'
                peaks.append(float(w.max()))
            cats[c] = cats.get(c, 0) + 1
    q = lambda v: [round(float(x), 3) for x in np.quantile(v, [.1, .25, .5, .75, .9])] if v else None
    return dict(categories=cats, never_peak_quantiles=q(peaks))


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=16384, timeout=1800)
def export_predictions(run, eot, intr, refractory_s=0.5, name='predictions-dev', recommit_eot=None,
                       recommit_int=None):
    """Write an official predictions JSON for TurnBench dev at a fixed operating point,
    validate it with turnbench.check and score it with turnbench.score (pinned evaluator).

    eot / intr: (model, score variant, theta). Committing uses the same rising-edge rule
    with `refractory_s`; times are frame ends on the 12.5 Hz grid, clipped to duration."""
    import os
    import numpy as np
    from turnbench.check import check_predictions
    from turnbench.data import resolve_dataset
    from turnbench.durations import load_durations
    from turnbench.score import score_submission
    from turnbench.submission import load_submission
    from turnbench.sweep import commit_events
    eot_tracks = load_tracks(run, eot[0], eot[1], 'tbdev')
    int_tracks = load_tracks(run, intr[0], intr[1], 'tbdev')
    gold = json.load(open('/work/gold/tbdev.json'))
    preds = []
    for cid in sorted(gold, key=lambda c: int(c)):
        dur = gold[cid]['duration_s']
        entry = dict(conversation_id=cid)
        for s in (0, 1):
            ev = {}
            for key, tracks, theta, rc in (('eot', eot_tracks, eot[2], recommit_eot),
                                           ('interruption', int_tracks, intr[2], recommit_int)):
                times = commit(tracks[cid][:, s], 12.5, float(theta), refractory_s, rc)
                ev[key] = [round(t, 3) for t in times if t <= dur]
            entry[f'speaker_{s + 1}'] = ev
        preds.append(entry)
    out = f'/work/runs/{run}/{name}.json'
    json.dump(dict(schema_version=1, predictions=preds), open(out, 'w'))
    work.commit()
    from pathlib import Path
    check_predictions(Path(out), load_durations('dev'))  # raises on any schema/coverage/time violation
    ds = resolve_dataset(TB_DEV, skip_audio=True)
    sc = score_submission(load_submission(Path(out)), ds)
    res = {}
    for task, t in (('eot', sc.task_eot), ('int', sc.task_int)):
        lat = t.latency()
        res[task] = dict(recall=t.recall, fp=t.fp_rate, p10=lat.p10, p50=lat.p50, p90=lat.p90, tp=t.tp, fn=t.fn,
                         fp_n=t.fp)
    json.dump(res, open(f'/work/runs/{run}/{name}.score.json', 'w'), indent=1)
    work.commit()
    return res
