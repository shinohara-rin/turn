"""Event commit policy: per-channel voice activity plus optional model score tracks -> TurnBench events.

After turn-1-mini (P99Lab, 2026-10): on two-channel TurnBench EOT, voice activity and a fixed
policy carry the result; a model score only adds an earlier trigger. Every rule is causal: an
event is stamped no earlier than the last audio its condition depends on.

Voice activity: Silero probabilities per channel on a 32 ms grid (`HOP_S`), the value for
chunk i known at (i + 1) * HOP_S. Model tracks: [T, 2] scores on the 12.5 Hz grid, frame i
known at (i + 1) / 12.5 s (the ssl_turn convention in pipeline/score.py).

EOT, per speaker c and other speaker o. A candidate opens at each speech offset t of c
(gaps shorter than `min_gap` are not offsets). It commits at the earliest of:
  - `deadline` s of silence (t + deadline);
  - o has been vocalising for `other_dur` s (not before t + min_wait);
  - the model's EOT score >= `eot_th`, read from t + read_from (not before t + mw_model);
and is cancelled when c resumes and keeps talking for `resume_min` s. Without a cancellation,
a second "confirmation" event is committed at t + confirm (None: off). TurnBench places about
7% of EOT labels 1-3 s after the last detected speech; one late fire covers them, and a second
fire costs nothing where the first already matched. If the audio ends first, the event is at
the end.

INT, per vocalisation of c starting at a: at most one event, when the model's INT score
reaches `int_th` between a + int_from and a + int_dur (and before the vocalisation ends), else
(if `int_fallback`) at a + int_dur when the vocalisation lasts that long.

Defaults are the TB dev operating points selected with r019_asr_bgaug (see RESULTS.md,
"Commit policy"): EOT (track eot_q) at FP budget 0.08 with median latency <= 0.5 s, INT
(track int_ft, the fine head's floor-taking probability) at budget 0.10.
"""
from dataclasses import dataclass, replace

import numpy as np

HOP_S = 0.032
FPS = 12.5


@dataclass(frozen=True)
class Policy:
    vad_th: float = 0.7          # Silero speech threshold
    min_gap: float = 0.2         # silences shorter than this do not end a vocalisation
    # EOT
    min_wait: float = 0.8
    other_dur: float = 0.2
    deadline: float = 2.5
    resume_min: float = 0.5
    confirm: float | None = 2.5
    eot_th: float | None = 0.8   # None: rules only
    read_from: float = 0.16
    mw_model: float = 0.32
    # INT
    int_th: float | None = 0.1   # None: duration rule only (default track: int_ft, low-valued)
    int_from: float = 0.16
    int_dur: float = 0.8
    int_fallback: bool = True

    def but(self, **kw):
        return replace(self, **kw)


def speech_segments(p, vad_th, min_gap, hop=HOP_S):
    """[(start_s, end_s)] of speech in one channel's VAD track, short gaps merged.

    Merging delays when an offset is known by up to `min_gap`; every policy wait is longer."""
    on = np.asarray(p) > vad_th
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        a, b = a * hop, b * hop
        if out and a - out[-1][1] < min_gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [tuple(s) for s in out]


def first_above(score, t0, t1, th, fps=FPS):
    """Commit time of the first frame known in (t0, t1] with score >= th, or None."""
    if score is None:
        return None
    i0 = max(int(np.ceil(t0 * fps)) - 1, 0)
    i1 = min(int(np.floor(t1 * fps)) - 1, len(score) - 1)
    if i1 < i0:
        return None
    hit = np.flatnonzero(np.asarray(score[i0:i1 + 1]) >= th)
    return (i0 + hit[0] + 1) / fps if len(hit) else None


def eot_events(own, other, duration_s, P, score=None, fps=FPS):
    """EOT commit times for the speaker of `own` (speech segments); `score` is its EOT track."""
    events = []
    horizon = max(P.deadline, P.confirm or 0)
    for k, (_, t) in enumerate(own):
        cancel = np.inf
        for a, b in own[k + 1:]:
            if a - t > horizon:
                break
            if b - a >= P.resume_min:
                cancel = a + P.resume_min
                break
        triggers = [t + P.deadline]
        earliest = t + P.min_wait
        for a, b in other:
            if b < t - P.min_gap:
                continue
            if a > t + P.deadline:
                break
            tt = max(a + P.other_dur, earliest)
            if b - a >= P.other_dur and tt <= b + P.min_gap:
                triggers.append(tt)
                break
        if P.eot_th is not None:
            m = first_above(score, t + P.read_from, min(t + P.deadline, cancel), P.eot_th, fps)
            if m is not None:
                triggers.append(max(m, t + P.mw_model))
        tc = min(min(triggers), duration_s)
        if tc < cancel:
            events.append(tc)
            if P.confirm and cancel > t + P.confirm and t + P.confirm <= duration_s:
                events.append(t + P.confirm)
    return sorted(set(round(x, 3) for x in events))


def int_events(own, P, score=None, fps=FPS):
    """INT commit times for the speaker of `own`; `score` is its INT track."""
    events = []
    for a, b in own:
        t = None
        if P.int_th is not None:
            t = first_above(score, a + P.int_from, min(b + 1 / fps, a + P.int_dur), P.int_th, fps)
        if t is None and P.int_fallback and b - a >= P.int_dur:
            t = a + P.int_dur
        if t is not None:
            events.append(round(t, 3))
    return sorted(set(events))


def conversation_events(vad, duration_s, P, eot=None, intr=None, fps=FPS, hop=HOP_S):
    """vad [T, 2] speech probabilities; eot / intr optional [T', 2] score tracks.

    Returns {'eot': {1: [...], 2: [...]}, 'int': {...}}; speaker s is channel s - 1."""
    segs = [speech_segments(vad[:, c], P.vad_th, P.min_gap, hop) for c in (0, 1)]
    col = lambda x, c: None if x is None else np.asarray(x[:, c], np.float32)
    return {
        'eot': {c + 1: eot_events(segs[c], segs[1 - c], duration_s, P, col(eot, c), fps) for c in (0, 1)},
        'int': {c + 1: int_events(segs[c], P, col(intr, c), fps) for c in (0, 1)},
    }
