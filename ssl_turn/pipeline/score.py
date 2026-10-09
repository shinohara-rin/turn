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


def score_variants(post, silent=None, fine=None):
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
    if fine is not None and silent is not None:  # fine annotator-label posteriors [T, 2, len(FINE)]
        from common import setup_path
        setup_path()
        import labels as lb
        grp = {k: fine[..., v].sum(-1) for k, v in lb.FINE_GROUPS.items()}
        not_bc = np.clip(1 - grp['backchannel'] - grp['noncontent'], 0, 1)
        out['int_nobc'] = clip(holds(f04) * (1 - silent) * not_bc)
        out['int_ft'] = clip(grp['floor_taking'])
        out['int_claim'] = clip(holds(f04) * (grp['floor_taking'] + grp['turn']))
        other_bc = (grp['backchannel'] + grp['noncontent'])[:, ::-1]
        out['eot_nobc'] = clip(eot_now * silent * np.clip(1 - other_bc, 0, 1))
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
                sk, fk = k[:-len('post')] + 'silent', k[:-len('post')] + 'fine'
                tracks[cid] = score_variants(z[k].astype(np.float32),
                                             z[sk].astype(np.float32) if sk in z.files else None,
                                             z[fk].astype(np.float32) if fk in z.files else None)[task]
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
            sk, fk = k[:-len('post')] + 'silent', k[:-len('post')] + 'fine'
            posts.setdefault(model, {})[(split, cid)] = (z[k].astype(np.float32),
                                                         z[sk].astype(np.float32) if sk in z.files else None,
                                                         z[fk].astype(np.float32) if fk in z.files else None)
        elif task not in ('silent', 'fine'):
            by.setdefault((model, task), {}).setdefault(split, {})[cid] = z[k]
    for name, members in (ensembles or {}).items():  # average posteriors across models
        members = [m_ for m_ in members if m_ in posts]
        keys = set.intersection(*(set(posts[m_]) for m_ in members))
        avg = lambda key, i: (None if posts[members[0]][key][i] is None
                              else np.mean([posts[m_][key][i] for m_ in members], 0))
        posts[name] = {key: (avg(key, 0), avg(key, 1), avg(key, 2)) for key in keys}
    for model, items in posts.items():
        for (split, cid), (post, silent, fine) in items.items():
            for name, v in score_variants(post, silent, fine).items():
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


FUNCTION_WORDS = {'and', 'but', 'so', 'or', 'because', 'um', 'uh', 'like', 'the', 'a', 'to', 'of', 'that', 'if',
                  'with', 'i', 'you', 'we', 'is', 'was', 'which', 'then'}


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=32768, timeout=3600)
def residuals(run, eot, intr, refractory_s=0.5, recommit_eot=1.0):
    """Content-level anatomy of errors at a fixed operating point on TurnBench dev, using the
    three annotators' segments and transcripts. For each candidate factor it reports counts
    and error rates, so factors are compared against base rates rather than read from anecdotes."""
    import numpy as np
    from collections import defaultdict
    from turnbench.data import conversation, resolve_dataset
    from turnbench.gold import CANONICAL
    ds = resolve_dataset(TB_DEV, skip_audio=True)
    gold = json.load(open('/work/gold/tbdev.json'))
    types = json.load(open('/work/gold/tbdev_types.json'))
    eot_tr = load_tracks(run, eot[0], eot[1], 'tbdev')
    int_tr = load_tracks(run, intr[0], intr[1], 'tbdev')
    fps = 12.5
    rows = dict(eot_pos=[], eot_neg=[], int_pos=[], int_neg=[])

    def last_words(text, n=2):
        w = [x.strip('.,?!"\'').lower() for x in text.split()]
        return w[-n:] if w else []

    def seg_at_end(segs, t, tol=0.5):
        best = min(segs, key=lambda x: abs(x[1] - t), default=None)
        return best if best and abs(best[1] - t) <= tol else None

    def seg_covering(segs, a, b):
        return [x for x in segs if x[0] < b and x[1] > a]

    for cid in sorted(gold, key=int):
        conv = conversation(ds, cid)
        ann = conv.annotations
        e = gold[cid]['events']
        for s in (1, 2):
            o = 3 - s
            own = {a: ann.get((s, a), []) for a in ('a', 'b', 'c')}
            oth = {a: ann.get((o, a), []) for a in ('a', 'b', 'c')}
            ef = commit(eot_tr[cid][:, s - 1], fps, float(eot[2]), refractory_s, recommit_eot)
            inf = commit(int_tr[cid][:, s - 1], fps, float(intr[2]), refractory_s, None)
            pos = sorted(x['time_s'] for x in e['eot_positive_events'] if x['speaker'] == s)
            for i, t in enumerate(pos):
                hi = min(t + 3.0, pos[i + 1] if i + 1 < len(pos) else t + 3.0)
                hit = [f for f in ef if t - 0.25 <= f <= hi]
                w = eot_tr[cid][max(0, int((t - 0.25) * fps)):int(hi * fps) + 1, s - 1]
                seg = seg_at_end(own['a'], t) or seg_at_end(own['b'], t) or seg_at_end(own['c'], t)
                agree = sum(seg_at_end(own[a], t, 0.25) is not None for a in own)
                nxt = [x for x in oth['a'] if x[0] > t - 3 and CANONICAL.get(x[2]) in ('Turn', 'Interruption')
                       and x[1] > t]
                gap = (min(x[0] for x in nxt) - t) if nxt else None
                earlier = [f for f in ef if t - 3 <= f < t - 0.25]
                rows['eot_pos'].append(dict(
                    cid=cid, type=types.get(cid), hit=bool(hit), lat=(hit[0] - t) if hit else None,
                    peak=float(w.max()) if len(w) else 0.0, fired_before=bool(earlier),
                    label=seg[2] if seg else None, dur=(seg[1] - seg[0]) if seg else None,
                    text=seg[3][-80:] if seg else '', last=last_words(seg[3]) if seg else [], agree=agree,
                    gap=gap))
            for span in (x for x in e['eot_negative_spans'] if x['speaker'] == s):
                a, b = span['start'], span['end']
                fired = [f for f in ef if a <= f <= b]
                prev = seg_at_end(own['a'], a, 0.3)
                bc = [x for x in seg_covering(oth['a'], a, b) if CANONICAL.get(x[2]) == 'Backchannel']
                rows['eot_neg'].append(dict(
                    cid=cid, type=types.get(cid), fired=bool(fired), dur=b - a,
                    fire_after=(fired[0] - a) if fired else None, prev_label=prev[2] if prev else None,
                    prev_text=prev[3][-80:] if prev else '', last=last_words(prev[3]) if prev else [],
                    other_backchannel=bool(bc)))
            ipos = sorted(x['time_s'] for x in e['int_positive_events'] if x['speaker'] == s)
            for i, t in enumerate(ipos):
                hi = min(t + 3.0, ipos[i + 1] if i + 1 < len(ipos) else t + 3.0)
                hit = [f for f in inf if t - 0.25 <= f <= hi]
                seg = min(own['a'], key=lambda x: abs(x[0] - t), default=None)
                rows['int_pos'].append(dict(cid=cid, hit=bool(hit), lat=(hit[0] - t) if hit else None,
                                            label=seg[2] if seg else None, text=seg[3][:80] if seg else ''))
            for span in (x for x in e['int_negative_spans'] if x['speaker'] == s):
                a, b = span['start'], span['end']
                fired = [f for f in inf if a <= f <= b]
                labs = [x[2] for an in own.values() for x in seg_covering(an, a, b)]
                lab = max(set(labs), key=labs.count) if labs else None
                txt = next((x[3] for x in seg_covering(own['a'], a, b)), '')
                rows['int_neg'].append(dict(cid=cid, type=types.get(cid), fired=bool(fired), dur=b - a, label=lab,
                                            text=txt[:60], other_talking=bool(seg_covering(oth['a'], a, b))))
    json.dump(rows, open(f'/work/runs/{run}/residuals.json', 'w'))
    work.commit()

    def table(items, key, flag, bins=None):
        g = defaultdict(lambda: [0, 0])
        for r in items:
            v = key(r)
            if bins is not None and v is not None:
                v = next((lab for lo, hi, lab in bins if lo <= v < hi), 'other')
            g[v][0] += 1
            g[v][1] += bool(flag(r))
        return {str(k): dict(n=n, rate=round(m / n, 3)) for k, (n, m) in sorted(g.items(), key=lambda x: -x[1][0])}

    ep, en, ip, inn = rows['eot_pos'], rows['eot_neg'], rows['int_pos'], rows['int_neg']
    miss = lambda r: not r['hit']
    out = dict(
        counts=dict(eot_pos=len(ep), eot_miss=sum(map(miss, ep)), eot_neg=len(en), eot_fp=sum(r['fired'] for r in en),
                    int_pos=len(ip), int_miss=sum(map(miss, ip)), int_neg=len(inn), int_fp=sum(r['fired'] for r in inn)),
        eot_miss_by_gap=table(ep, lambda r: r['gap'], miss, [(-99, -0.3, 'overlap>0.3s'), (-0.3, 0, 'overlap<0.3s'),
                                                             (0, 0.3, 'gap<0.3s'), (0.3, 1, 'gap0.3-1s'), (1, 99, 'gap>1s')]),
        eot_miss_by_agree=table(ep, lambda r: r['agree'], miss),
        eot_miss_by_label=table(ep, lambda r: r['label'], miss),
        eot_miss_by_dur=table(ep, lambda r: r['dur'], miss, [(0, 0.6, '<0.6s'), (0.6, 1.5, '0.6-1.5s'), (1.5, 4, '1.5-4s'),
                                                             (4, 999, '>4s')]),
        eot_miss_by_last_function_word=table(ep, lambda r: bool(r['last']) and r['last'][-1] in FUNCTION_WORDS, miss),
        eot_miss_by_fired_before=table(ep, lambda r: r['fired_before'], miss),
        eot_miss_by_type=table(ep, lambda r: r['type'], miss),
        eot_miss_peak_q=[round(float(x), 3) for x in np.quantile([r['peak'] for r in ep if miss(r)], [.1, .25, .5, .75, .9])],
        eot_fp_by_dur=table(en, lambda r: r['dur'], lambda r: r['fired'], [(0, 0.3, '<0.3s'), (0.3, 0.6, '0.3-0.6s'),
                                                                           (0.6, 1.2, '0.6-1.2s'), (1.2, 2.5, '1.2-2.5s'),
                                                                           (2.5, 999, '>2.5s')]),
        eot_fp_by_prev_label=table(en, lambda r: r['prev_label'], lambda r: r['fired']),
        eot_fp_by_last_function_word=table(en, lambda r: bool(r['last']) and r['last'][-1] in FUNCTION_WORDS,
                                           lambda r: r['fired']),
        eot_fp_by_other_backchannel=table(en, lambda r: r['other_backchannel'], lambda r: r['fired']),
        eot_fp_fire_after_q=[round(float(x), 2) for x in np.quantile([r['fire_after'] for r in en if r['fired']],
                                                                     [.1, .5, .9])],
        int_fp_by_label=table(inn, lambda r: r['label'], lambda r: r['fired']),
        int_fp_by_dur=table(inn, lambda r: r['dur'], lambda r: r['fired'], [(0, 0.4, '<0.4s'), (0.4, 0.8, '0.4-0.8s'),
                                                                          (0.8, 1.5, '0.8-1.5s'), (1.5, 999, '>1.5s')]),
        int_fp_by_other_talking=table(inn, lambda r: r['other_talking'], lambda r: r['fired']),
        int_misses=[dict(label=r['label'], text=r['text']) for r in ip if miss(r)],
        examples=dict(
            eot_miss=[dict(t=r['text'], label=r['label'], gap=r['gap'], agree=r['agree'], peak=round(r['peak'], 2))
                      for r in ep if miss(r)][:25],
            eot_fp=[dict(t=r['prev_text'], label=r['prev_label'], dur=round(r['dur'], 2)) for r in en if r['fired']][:25],
            int_fp=[dict(t=r['text'], label=r['label'], dur=round(r['dur'], 2)) for r in inn if r['fired']][:25]))
    json.dump(out, open(f'/work/runs/{run}/residuals_summary.json', 'w'), indent=1)
    work.commit()
    return out


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=32768, timeout=1800)
def dump_transcripts():
    """Text + timing only (no labels) for the LLM oracle: TurnBench dev annotator 'a' and
    otoSpeech dev SRTs. Returns {split: {cid: {speaker: [[start, end, text], ...]}}}."""
    import re
    from turnbench.data import conversation, conversation_ids, resolve_dataset
    out = {'tbdev': {}, 'oto': {}}
    ds = resolve_dataset(TB_DEV, skip_audio=True)
    for cid in conversation_ids(ds):
        ann = conversation(ds, cid).annotations
        out['tbdev'][cid] = {s: [[a, b, t] for a, b, _, t in ann.get((s, 'a'), [])] for s in (1, 2)}
    split = json.load(open('/work/split.json'))['splits']
    import srt
    for cid in split['dev']:
        segs = {}
        for s in (1, 2):
            rows = []
            for e in srt.parse(open(f'/datasets/otoearth/otoSpeech-full-duplex-turn-104h/{cid}/speaker_{s}_annotation_a.srt').read()):
                m = re.match(r'\[([^]]+)\]\s*(.*)', e.content, re.S)
                rows.append([e.start.total_seconds(), e.end.total_seconds(), (m[2] if m else e.content).strip()])
            segs[s] = rows
        out['oto'][cid] = segs
    return out


def _text_tracks(recs, conv, T, fps=12.5, latency=0.3):
    """Per-frame causal text features [T, 2] for eot and int (NaN = no text yet)."""
    import numpy as np
    eot = np.full((T, 2), np.nan, np.float32)
    intr = np.full((T, 2), np.nan, np.float32)
    end_t = (np.arange(T) + 1) / fps  # frame i commits at its end
    for r in sorted(recs, key=lambda r: r['t']):
        if r['p'] is None:
            continue
        s = r['speaker']
        segs = conv[str(s)]
        i = int(r['id'].split('/')[3])
        lo = np.searchsorted(end_t, r['t'])
        if r['kind'] == 'eot':
            nxt = segs[i + 1][0] if i + 1 < len(segs) else 1e9
            hi = np.searchsorted(end_t, nxt)
            eot[lo:hi, s - 1] = r['p']
        else:
            hi = np.searchsorted(end_t, segs[i][1] + latency)
            intr[lo:hi, s - 1] = r['p']  # later (longer-prefix) records overwrite earlier ones
    return eot, intr


def _fp_at(rows, target):
    ok = [r for r in rows if r['recall'] >= target and r['fp'] == r['fp']]
    if not ok:
        return None
    b = min(ok, key=lambda r: (r['fp'], r['p50']))
    return dict(fp=round(b['fp'], 4), p50=round(b['p50'], 0), theta=round(b['theta'], 4))


@app.function(image=cpu_image, volumes=VOLUMES, cpu=8, memory=32768, timeout=3600)
def oracle_fusion(run, model, weights=(0.0, 0.25, 0.5, 0.75, 1.0), control=None):
    """Fuse LLM text-oracle tracks into the audio scores and sweep (see llm_oracle.py).

    control='shuffle' permutes the LLM answers among queries of the same split and kind,
    keeping every query's timing (annotated segment boundaries) but destroying its content:
    whatever gain survives the shuffle is a timing leak, not language understanding."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    work.reload()
    transcripts = json.load(open('/work/llm/transcripts.json'))
    recs = {}
    for line in open('/work/llm/oracle.jsonl'):
        r = json.loads(line)
        recs.setdefault((r['split'], r['cid']), {})[r['id']] = r  # last answer per id wins
    if control == 'shuffle':
        rng = np.random.default_rng(0)
        for split in ('oto', 'tbdev'):
            for kind in ('eot', 'int'):
                group = [r for (sp, _), d in recs.items() if sp == split for r in d.values() if r['kind'] == kind]
                ps = [r['p'] for r in group]
                rng.shuffle(ps)
                for r, p in zip(group, ps):
                    r['p'] = p
    out, jobs = {}, {}
    with ProcessPoolExecutor(8) as pool:
        for split in ('oto', 'tbdev'):
            base_eot = load_tracks(run, model, 'eot_q', split)
            base_int = load_tracks(run, model, 'int_nobc', split)
            gold = load_gold(split, list(base_eot))
            text = {cid: _text_tracks(list(recs.get((split, cid), {}).values()), transcripts[split][cid],
                                      len(base_eot[cid])) for cid in base_eot}
            for w in weights:
                fe, fi = {}, {}
                for cid in base_eot:
                    te, ti = text[cid]
                    te = np.where(np.isnan(te), 0.5, te)
                    ti = np.where(np.isnan(ti), 0.5, ti)
                    fe[cid] = np.clip(base_eot[cid], 1e-6, 1) ** (1 - w) * te ** w
                    fi[cid] = np.clip(base_int[cid], 1e-6, 1) ** (1 - w) * ti ** w
                jobs[(split, w, 'eot')] = pool.submit(sweep_task, fe, gold, 'eot', None, 12.5, 0.5, 1.0)
                jobs[(split, w, 'int')] = pool.submit(sweep_task, fi, gold, 'int', None, 12.5, 0.5, None)
        for (split, w, task), f in jobs.items():
            rows = f.result()
            op = operating_point(rows)
            tg = (0.90, 0.92) if task == 'eot' else (0.95, 0.97)
            out[f'{split}|{task}|w{w}'] = dict(
                op=dict(recall=round(op['recall'], 4), fp=round(op['fp'], 4), p50=round(op['p50'], 0)) if op else None,
                **{f'fp@{t}': _fp_at(rows, t) for t in tg})
    coverage = {}
    for split in ('oto', 'tbdev'):
        rs = [r for (sp, _), d in recs.items() if sp == split for r in d.values()]
        coverage[split] = dict(n=len(rs), ok=sum(r['p'] is not None for r in rs))
    out['coverage'] = coverage
    json.dump(out, open(f'/work/runs/{run}/oracle_fusion_{model}{"_" + control if control else ""}.json', 'w'), indent=1)
    work.commit()
    return out



TASK_SCORE = {'eot': ('eot_q', 1.0), 'int': ('int_nobc', None)}  # score variant, recommit_s


@app.function(image=cpu_image, volumes=VOLUMES, cpu=8, memory=32768, timeout=3600)
def verifier_fires(run, model, budgets=(0.30, 0.20, 0.15, 0.10, 0.07, 0.05), split='tbdev'):
    """Audio-model commit events at a few thresholds; an LLM verifier then keeps or vetoes each
    fire. Query times are the model's own fires, so text availability carries no gold timing."""
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(2) as pool:
        tracks = {t: load_tracks(run, model, v, split) for t, (v, _) in TASK_SCORE.items()}
        gold = load_gold(split, list(tracks['eot']))
        rows = {t: pool.submit(sweep_task, tracks[t], gold, t, None, 12.5, 0.5, rc) for t, (_, rc) in TASK_SCORE.items()}
        for t, (_, rc) in TASK_SCORE.items():
            rs = rows[t].result()
            out[t] = {}
            for b in budgets:
                op = operating_point(rs, b)
                fires = {cid: {s + 1: commit(p[:, s], 12.5, op['theta'], 0.5, rc) for s in (0, 1)}
                         for cid, p in tracks[t].items()}
                out[t][str(b)] = dict(theta=op['theta'], recall=op['recall'], fp=op['fp'], p50=op['p50'], fires=fires)
    json.dump(out, open(f'/work/llm/fires_{model}.json', 'w'))
    work.commit()
    return {t: {b: (d['theta'], round(d['recall'], 4), round(d['fp'], 4),
                    sum(len(v) for f in d['fires'].values() for v in f.values())) for b, d in x.items()}
            for t, x in out.items()}


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=16384, timeout=3600)
def verifier_score(model, answers, cuts=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), control=None,
                   split='tbdev'):
    """Score verifier-gated fires. answers: {task: {cid: {speaker: {time: confidence}}}} (str keys).
    A fire is kept when the LLM's confidence that it is a true event is >= cut (no answer -> kept).
    control='shuffle' permutes confidences among fires of the same task (same fires, no content)."""
    import random
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import TaskScore, merge, score_task
    work.reload()
    allf = json.load(open(f'/work/llm/fires_{model}.json'))
    gold = None
    if control == 'shuffle':
        rnd = random.Random(0)
        for t, x in answers.items():
            keys = [(c, s, k) for c, d in x.items() for s, e in d.items() for k in e]
            vals = [x[c][s][k] for c, s, k in keys]
            rnd.shuffle(vals)
            for (c, s, k), v in zip(keys, vals):
                x[c][s][k] = v
    out = {}
    for t, by in allf.items():
        key = 'eot' if t == 'eot' else 'int'
        for b, d in by.items():
            if gold is None:
                gold = load_gold(split, list(d['fires']))
            for cut in cuts:
                total, kept, n = TaskScore(), 0, 0
                for cid, f in d['fires'].items():
                    e = gold[cid]['events']
                    ev = {}
                    for s, times in f.items():
                        a = answers.get(t, {}).get(cid, {}).get(s, {})
                        ev[int(s)] = [x for x in times if a.get(f'{x:.2f}', 1.0) >= cut]
                        kept += len(ev[int(s)])
                        n += len(times)
                    merge(total, score_task([AnchorEvent(**x) for x in e[f'{key}_positive_events']],
                                            [Interval(**x) for x in e[f'{key}_negative_spans']], ev,
                                            [Interval(**x) for x in e[f'{key}_excluded']]))
                lat = total.latency()
                out[f'{t}|{b}|{cut}'] = dict(theta=d['theta'], recall=round(total.recall, 4), fp=round(total.fp_rate, 4),
                                             p50=lat.p50, kept=kept, fires=n)
    return out


def _queries_above(p, theta, step):
    """Query frames: each rising edge of p > theta, then every `step` frames while it stays above."""
    out, last = [], None
    for i in range(len(p)):
        if p[i] <= theta:
            last = None
        elif last is None or i - last >= step:
            out.append(i)
            last = i
    return out


@app.function(image=cpu_image, volumes=VOLUMES, cpu=8, memory=32768, timeout=3600)
def verifier_candidates(run, model, budget=0.30, step=5, split='tbdev'):
    """LLM-verifier query times taken from the audio model's own timeline (no gold timing).

    theta_low = the audio score's operating point at a generous FP budget; queries go at each
    rising edge above theta_low and every `step` frames while the score stays above it."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = {'eot': 'eot_q', 'int': 'int_nobc'}
    out = {}
    with ProcessPoolExecutor(2) as pool:
        tracks = {t: load_tracks(run, model, v, split) for t, v in tasks.items()}
        gold = load_gold(split, list(tracks['eot']))
        rows = {t: pool.submit(sweep_task, tracks[t], gold, t, None, 12.5, 0.5, 1.0 if t == 'eot' else None)
                for t in tasks}
        for t in tasks:
            op = operating_point(rows[t].result(), budget)
            q = {cid: [[i, s + 1] for s in (0, 1) for i in _queries_above(p[:, s], op['theta'], step)]
                 for cid, p in tracks[t].items()}
            out[t] = dict(theta_low=op['theta'], recall=op['recall'], fp=op['fp'], queries=q,
                          n=sum(map(len, q.values())))
    json.dump(out, open(f'/work/llm/candidates_{model}.json', 'w'))
    work.commit()
    return {t: {k: v for k, v in d.items() if k != 'queries'} for t, d in out.items()}


@app.local_entrypoint()
def verify(answers: str, out: str, model: str = 'fine1_bal1_s1'):
    """Score llm_verifier.py output (local JSONL) with and without the shuffled-answer control."""
    recs = {}
    for line in open(answers):
        r = json.loads(line)
        if r.get('answer') is not None:
            recs[(r['kind'], r['text'])] = r
    res = {}
    for field, cuts in (('answer', (0.0, 0.5)), ('p', (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95))):
        ans = {}
        for r in recs.values():
            v = r[field] if r.get(field) is not None else r['answer']
            for cid, s, t in r['at']:
                ans.setdefault(r['kind'], {}).setdefault(cid, {}).setdefault(s, {})[t] = v
        calls = {c: verifier_score.spawn(model, ans, cuts, c) for c in (None, 'shuffle')}
        res[field] = {str(c): f.get() for c, f in calls.items()}
    json.dump(dict(answered=len(recs), **res), open(out, 'w'), indent=1)
    print('answered', len(recs))


def _events(conv, annotators, min_agreement):
    """TurnBench event sets built from a subset of the annotators (gold.py with ANNOTATORS and
    MIN_AGREEMENT patched; one annotator with agreement 1 = that annotator's own reading)."""
    from dataclasses import asdict
    import turnbench.gold as G
    saved = G.ANNOTATORS, G.MIN_AGREEMENT
    G.ANNOTATORS, G.MIN_AGREEMENT = tuple(annotators), min_agreement
    try:
        return asdict(G.events_for_conversation(conv))
    finally:
        G.ANNOTATORS, G.MIN_AGREEMENT = saved


@app.function(image=cpu_image, volumes=VOLUMES, cpu=8, memory=16384, timeout=3600)
def human_ceiling(model='fine1_bal1_s1', budget='0.1'):
    """How well does one annotator agree with the others, under TurnBench's own scoring?

    Each annotator's own labels become a 'system' (EOT fires at their turn ends, INT fires at
    their interruption onsets, zero latency, full hindsight). Scored against (a) gold built from
    the other two annotators (2-of-2 agreement) and (b) the official 3-annotator gold, which
    includes the annotator (optimistic). The audio model's fires at its operating point
    (score.verifier_fires) are scored the same way, and its errors are split by how many
    annotators individually mark an event there."""
    from turnbench.data import ANNOTATORS, conversation, conversation_ids, resolve_dataset
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import TaskScore, merge, score_task
    work.reload()
    ds = resolve_dataset(TB_DEV, skip_audio=True)
    official = json.load(open('/work/gold/tbdev.json'))
    fires = json.load(open(f'/work/llm/fires_{model}.json'))
    model_fires = {t: {cid: {int(s): v for s, v in f.items()} for cid, f in fires[t][budget]['fires'].items()}
                   for t in ('eot', 'int')}
    own, loo = {}, {}
    cids = conversation_ids(ds)
    for cid in cids:
        conv = conversation(ds, cid)
        if _events(conv, ANNOTATORS, 2) != official[cid]['events']:  # patched builder == official gold
            raise ValueError(f'gold rebuild mismatch for {cid}')
        for x in ANNOTATORS:
            own[(cid, x)] = _events(conv, [x], 1)
            loo[(cid, x)] = _events(conv, [a for a in ANNOTATORS if a != x], 2)

    def score(gold_of, fires_of, task):
        total = TaskScore()
        for cid in cids:
            e = gold_of(cid)
            merge(total, score_task([AnchorEvent(**x) for x in e[f'{task}_positive_events']],
                                    [Interval(**x) for x in e[f'{task}_negative_spans']], fires_of(cid),
                                    [Interval(**x) for x in e[f'{task}_excluded']]))
        lat = total.latency()
        return dict(recall=round(total.recall, 4), fp=round(total.fp_rate, 4), p50=round(lat.p50, 0) if lat.p50 == lat.p50 else None,
                    tp=total.tp, fn=total.fn, fpn=total.fp)

    def human_fires(cid, x, task):
        out = {1: [], 2: []}
        for ev in own[(cid, x)][f'{task}_positive_events']:
            out[ev['speaker']].append(ev['time_s'])
        return {s: sorted(v) for s, v in out.items()}

    res = {}
    for task in ('eot', 'int'):
        for x in ANNOTATORS:
            res[f'{task}|human {x} vs other two'] = score(lambda c: loo[(c, x)], lambda c: human_fires(c, x, task), task)
            res[f'{task}|human {x} vs official'] = score(lambda c: official[c]['events'], lambda c: human_fires(c, x, task), task)
            res[f'{task}|model vs other two (w/o {x})'] = score(lambda c: loo[(c, x)], lambda c: model_fires[task][c], task)
        res[f'{task}|model vs official'] = score(lambda c: official[c]['events'], lambda c: model_fires[task][c], task)

    # Error anatomy on the official gold: how many annotators individually mark an event here?
    def n_marking(cid, task, speaker, a, b):
        return sum(any(ev['speaker'] == speaker and a <= ev['time_s'] <= b for ev in own[(cid, x)][f'{task}_positive_events'])
                   for x in ANNOTATORS)
    anatomy = {}
    for task in ('eot', 'int'):
        pos, neg = {}, {}
        for cid in cids:
            e = official[cid]['events']
            mf = model_fires[task][cid]
            for ev in e[f'{task}_positive_events']:
                k = n_marking(cid, task, ev['speaker'], ev['time_s'] - 0.3, ev['time_s'] + 0.3)
                hit = any(ev['time_s'] - 0.25 <= t <= ev['time_s'] + 3.0 for t in mf[ev['speaker']])
                d = pos.setdefault(k, [0, 0]); d[0] += 1; d[1] += hit
            for sp in e[f'{task}_negative_spans']:
                k = n_marking(cid, task, sp['speaker'], sp['start'] - 0.25, sp['end'])
                fired = any(sp['start'] <= t <= sp['end'] for t in mf[sp['speaker']])
                d = neg.setdefault(k, [0, 0]); d[0] += 1; d[1] += fired
        anatomy[task] = dict(
            positives_by_annotators_marking={k: dict(n=v[0], model_recall=round(v[1] / v[0], 3)) for k, v in sorted(pos.items())},
            negatives_by_annotators_marking_event={k: dict(n=v[0], model_fp=round(v[1] / v[0], 3)) for k, v in sorted(neg.items())})
    out = dict(scores=res, anatomy=anatomy, model=model, budget=budget)
    json.dump(out, open('/work/human_ceiling.json', 'w'), indent=1)
    work.commit()
    return out
