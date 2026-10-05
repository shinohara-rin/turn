"""Energy VAD per channel. TurnBench segments audio with VAD before annotators
label each segment, so we do the same: all three annotators label the same
segments, and disagreement is purely about the label."""

import numpy as np

FRAME_S = 0.01


def segments(audio: np.ndarray, sr: int, *, rel_db: float = 28.0, floor_db: float = -50.0,
             min_speech_s: float = 0.08, merge_gap_s: float = 0.2, pad_s: float = 0.02) -> list[tuple[float, float]]:
    hop = int(FRAME_S * sr)
    n = len(audio) // hop
    if n == 0:
        return []
    frames = audio[: n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    ref = np.percentile(db, 99)
    active = db > max(floor_db, ref - rel_db)
    spans: list[list[float]] = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        start, end = i * FRAME_S, j * FRAME_S
        if spans and start - spans[-1][1] < merge_gap_s:
            spans[-1][1] = end
        else:
            spans.append([start, end])
        i = j
    dur = len(audio) / sr
    return [(round(max(0.0, s - pad_s), 3), round(min(dur, e + pad_s), 3))
            for s, e in spans if e - s >= min_speech_s]
