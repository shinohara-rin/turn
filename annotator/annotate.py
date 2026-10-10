"""Label EOT and interruption events in two-speaker audio with hindsight.

    python -m annotator.annotate --model model.joblib --mode mono a.wav b.wav --out labels/
    python -m annotator.annotate --model model.joblib --mode stereo conv.wav --out labels/

Writes <out>/<stem>.json with every candidate and its score per task, plus `*_positive_events`
at the chosen thresholds (TurnBench event format; speaker 1/2 are diarizer slots in mono mode,
not identities). Thresholds default to the operating points stored by train.py (max recall at
TurnBench FP <= 0.10); --precision 0.7|0.8|0.9 picks the stored higher-precision ones instead.

Also writes <out>/<stem>.labels.txt, an Audacity label track of the positive events
(File > Import > Labels) for a human to confirm or reject. Lower the thresholds to trade more
proposals to reject for fewer missed events.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import activity as A
from . import content as C
from . import features as F

MERGE_S = 1.0


def annotate(a, models, thresholds, content=None, E=None):
    """a: activity [T, 2]; content: the bundle's content entry, with E the encoder frames
    (content.encode) when the model uses them."""
    out = {}
    for task in F.TASKS:
        rows, X = F.table(a, task)
        if content:
            X = np.concatenate([X, content['pca'][task].transform(C.table(E, rows))], 1) if len(rows) else \
                np.zeros((0, X.shape[1] + content['pca'][task].n_components_), np.float32)
        sc = models[task].predict_proba(X)[:, 1] if len(rows) else np.zeros(0)
        out[f'{task}_candidates'] = [dict(speaker=s, time_s=round(t + F.FIRE_OFFSET_S[task], 2), kind=k,
                                          score=round(float(v), 4)) for (s, t, k), v in zip(rows, sc)]
        events = []  # one event per run of positive candidates (overlap check points repeat every 0.4 s)
        for c in out[f'{task}_candidates']:
            if c['score'] < thresholds[task]:
                continue
            if events and events[-1]['speaker'] == c['speaker'] and c['time_s'] - events[-1]['_last'] <= MERGE_S:
                events[-1]['_last'] = c['time_s']; events[-1]['score'] = max(events[-1]['score'], c['score'])
                continue
            events.append(dict(speaker=c['speaker'], time_s=c['time_s'], _last=c['time_s'], score=c['score']))
        out[f'{task}_positive_events'] = [dict(speaker=e['speaker'], time_s=e['time_s'], score=e['score']) for e in events]
    return out


def audacity_labels(res):
    """Audacity label track lines: start, end, text (point labels, sorted by time)."""
    rows = sorted((e['time_s'], f"{task.upper()} spk{e['speaker']} {e['score']:.2f}")
                  for task in F.TASKS for e in res[f'{task}_positive_events'])
    return ''.join(f'{t:.2f}\t{t:.2f}\t{text}\n' for t, text in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', nargs='+')
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', choices=('mono', 'stereo'), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--eot-threshold', type=float, help='default: the model file\'s FP<=0.10 point')
    ap.add_argument('--int-threshold', type=float)
    ap.add_argument('--precision', type=float, choices=(0.7, 0.8, 0.9),
                    help='use the stored thresholds where proposals reach this precision instead')
    ap.add_argument('--device', default='cpu', help='encoder device for content models (cpu or cuda)')
    a = ap.parse_args()
    import joblib
    bundle = joblib.load(a.model); models = bundle['models']
    th = dict(bundle['thresholds'])
    if a.precision is not None:  # cross-fitted threshold where proposals reach this precision
        for task in F.TASKS:
            t = bundle['dev'][task].get(f'threshold@P{a.precision}')
            if t is None:
                ap.error(f'{a.model} has no {task} threshold for precision {a.precision}')
            th[task] = t
    if a.eot_threshold is not None:
        th['eot'] = a.eot_threshold
    if a.int_threshold is not None:
        th['int'] = a.int_threshold
    os.makedirs(a.out, exist_ok=True)
    content = bundle.get('content')
    if content and content['channels'] != (2 if a.mode == 'stereo' else 1):
        ap.error(f"{a.model} is a {'stereo' if content['channels'] == 2 else 'mono'} content model")
    encoder = C.load_model(a.device) if content else None
    diarizer = A.Sortformer() if a.mode == 'mono' else None
    for path in a.audio:
        act = A.mono_activity(path, diarizer) if a.mode == 'mono' else A.stereo_activity(path)
        E = C.encode(A.load_16k(path), a.mode, encoder) if content else None
        res = dict(audio=os.path.abspath(path), mode=a.mode, frame_s=F.FRAME_S, thresholds=th,
                   duration_s=round(len(act) * F.FRAME_S, 2), **annotate(act, models, th, content, E))
        stem = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(a.out, f'{stem}.activity.npy'), act)
        json.dump(res, open(os.path.join(a.out, f'{stem}.json'), 'w'))
        open(os.path.join(a.out, f'{stem}.labels.txt'), 'w').write(audacity_labels(res))
        print(stem, {t: len(res[f'{t}_positive_events']) for t in F.TASKS}, flush=True)


if __name__ == '__main__':
    main()
