"""Hindsight turn-taking labeler: activity -> candidate points -> timing features -> labels.

Everything here works on per-speaker voice activity [T, 2] (bool, 20 ms frames), so the same
code serves real stereo (Silero per channel) and mono audio (diarization). It is offline: the
features look up to 4 s past each candidate, which is the point of an annotator.
"""
from __future__ import annotations

import numpy as np

FRAME_S = 0.02
TASKS = ('eot', 'int')
FIRE_OFFSET_S = dict(int=0.1, eot=0.4)  # label time = candidate + offset (see README)
MIN_GAP_S = 0.2       # silence before an INT onset / after an EOT offset
OVERLAP_STEP_S = 0.4  # extra INT check points inside each overlap
OVERLAP_MAX_S = 3.0
DEDUPE_S = 0.3
TAU_PRE_S, TAU_MAX_S = 0.25, 3.0  # TurnBench matching window
N_FEATURES = 1 + 12 + 5


def resample_probs(p, src_frame_s, n_frames=None, threshold=0.5):
    """Speaker probabilities [N, K] on a src_frame_s grid -> bool [T, K] on the 20 ms grid."""
    p = np.asarray(p)
    if n_frames is None:
        n_frames = int(len(p) * src_frame_s / FRAME_S)
    idx = np.minimum((np.arange(n_frames) * FRAME_S / src_frame_s).astype(int), len(p) - 1)
    return p[idx] > threshold


def smooth(a, fill_gap_s=0.16, min_on_s=0.1):
    """Close gaps shorter than fill_gap_s, then drop blips shorter than min_on_s, per column."""
    a = np.array(a, bool, copy=True)
    for c in range(a.shape[1]):
        x = a[:, c]
        for value, shortest in ((False, round(fill_gap_s / FRAME_S)), (True, round(min_on_s / FRAME_S))):
            for s, e in runs(x == value):
                if s > 0 and e < len(x) and e - s < shortest:
                    x[s:e] = not value
    return a


def runs(x):
    """Start/end indices of True runs in a 1-D bool array."""
    d = np.diff(np.r_[0, np.asarray(x, np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fold_to_two(b):
    """[T, K] bool slots -> [T, 2]: keep the two busiest slots and fold each other slot into the
    one it co-occurs with least. Diarizers often split one person across two slots; in a
    known two-speaker recording a split slot rarely overlaps its own other half."""
    b = np.asarray(b, bool)
    order = np.argsort(-b.sum(0))
    a = b[:, order[:2]].copy()
    if a.shape[1] < 2:
        a = np.concatenate([a, np.zeros((len(a), 2 - a.shape[1]), bool)], 1)
    for k in order[2:]:
        if b[:, k].mean() < 0.01:
            continue
        co = [(b[:, k] & a[:, j]).sum() / max(1, b[:, k].sum()) for j in (0, 1)]
        a[:, int(np.argmin(co))] |= b[:, k]
    return a


def candidates(a, task):
    """Candidate (speaker 1|2, time_s, kind) points from activity alone. kind 0 = onset/offset,
    1 = check point inside an overlap (INT only)."""
    out = []
    near = round(0.1 / FRAME_S)
    for c in (0, 1):
        segs, other = runs(a[:, c]), a[:, 1 - c]
        for i, (s, e) in enumerate(segs):
            if task == 'int':
                gap = (s - segs[i - 1][1]) * FRAME_S if i else np.inf
                if gap >= MIN_GAP_S and other[max(0, s - near): s + near + 1].any():
                    out.append((c + 1, round(s * FRAME_S, 2), 0))
            else:
                gap = (segs[i + 1][0] - e) * FRAME_S if i + 1 < len(segs) else np.inf
                if gap >= MIN_GAP_S:
                    out.append((c + 1, round(e * FRAME_S, 2), 0))
        if task == 'int':
            for s, e in runs(a[:, c] & other):
                for k in range(1, int(OVERLAP_MAX_S / OVERLAP_STEP_S) + 1):
                    t = s + round(k * OVERLAP_STEP_S / FRAME_S)
                    if t < e:
                        out.append((c + 1, round(t * FRAME_S, 2), 1))
    out.sort(key=lambda r: (r[0], r[1]))
    kept = []
    for r in out:
        if kept and kept[-1][0] == r[0] and r[1] - kept[-1][1] < DEDUPE_S:
            continue
        kept.append(r)
    return kept


def _run_length(x, j, step):
    """Signed length (s) of the run containing x[j], walking in direction step (capped at 10 s)."""
    j = min(max(j, 0), len(x) - 1)
    v, k = x[j], j
    while 0 <= k < len(x) and x[k] == v and abs(k - j) < 500:
        k += step
    return abs(k - j) * FRAME_S * (1 if v else -1)


def features(a, speaker, t, kind):
    i = round(t / FRAME_S)
    me, other = a[:, speaker - 1], a[:, 2 - speaker]
    f = [kind]
    for lo, hi in ((-3, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4)):
        sl = slice(max(0, i + int(lo / FRAME_S)), max(0, i + int(hi / FRAME_S)))
        f += [me[sl].mean() if me[sl].size else 0.0, other[sl].mean() if other[sl].size else 0.0]
    f += [_run_length(me, i + 1, 1), _run_length(other, i, 1), _run_length(other, i, -1),
          _run_length(me, i - 1, -1), _run_length(me & other, i - 1, -1)]
    return f


def table(a, task):
    """Candidates and their feature matrix for one conversation."""
    rows = candidates(a, task)
    X = np.array([features(a, s, t, k) for s, t, k in rows], np.float32).reshape(len(rows), N_FEATURES)
    return rows, X


def label(gold, task, speaker, t):
    """1 if a fire at t + offset would hit a gold positive window, -1 inside an excluded span
    (no reliable gold; not trained on), else 0. Unlabeled regions count as negatives: TurnBench's
    scorer ignores fires there, but an annotator's events there would be false labels."""
    t = t + FIRE_OFFSET_S[task]
    for e in gold[f'{task}_positive_events']:
        if e['speaker'] == speaker and e['time_s'] - TAU_PRE_S <= t <= e['time_s'] + TAU_MAX_S:
            return 1
    for e in gold[f'{task}_excluded']:
        if e['speaker'] == speaker and e['start'] <= t <= e['end']:
            return -1
    return 0
