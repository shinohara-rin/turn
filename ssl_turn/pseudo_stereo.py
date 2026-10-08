"""Pseudo-stereo from a single-channel two-speaker mix, and calibration against real stereo.

Two routes produce one channel per speaker from podcast audio:
  1. DialogueSidon (DuplexChat's separator): separates overlap, but *re-synthesizes*
     speech through a diffusion model and DAC vocoder, and removes noise. Breaths, room
     tone and bleed, which are turn-taking cues present in TurnBench audio, may be
     lost. The weights are CC-BY-NC-4.0.
  2. `gated_stereo` below: each channel keeps the *original* mixture, at full
     gain while its diarized speaker talks and attenuated to a bleed level otherwise.
     Acoustics stay real and the result resembles headset bleed, but overlaps are
     not separated: both channels then contain both talkers.

Neither is trustworthy until calibrated. otoSpeech has true per-speaker channels,
so mixing them to mono, running a route, and comparing with the originals gives
a ground-truth measurement, and the same comparison can be repeated downstream:
train on pseudo-stereo otoSpeech versus real otoSpeech.
"""
from __future__ import annotations

import numpy as np


def gated_stereo(mono, sample_rate, segments, bleed_db=-24.0, ramp_ms=20.0, rng=None, jitter_db=4.0):
    """mono [N]; segments: iterable of (start_s, end_s, speaker in {0,1}) -> [N, 2] float32.

    Gains ramp linearly over `ramp_ms` at segment edges. The bleed level is drawn per
    channel with +-jitter_db, so a fixed attenuation is not a giveaway that the
    model could key on.
    """
    x = np.asarray(mono, np.float32)
    if x.ndim != 1:
        raise ValueError('mono must be 1-D')
    rng = rng or np.random.default_rng(0)
    active = np.zeros((len(x), 2), np.float32)
    for start, end, speaker in segments:
        if speaker not in (0, 1):
            raise ValueError('speaker must be 0 or 1')
        lo, hi = max(0, int(round(start * sample_rate))), min(len(x), int(round(end * sample_rate)))
        active[lo:hi, speaker] = 1.0
    ramp = max(1, int(ramp_ms * sample_rate / 1000))
    if ramp > 1:
        kernel = np.ones(ramp, np.float32) / ramp
        active = np.stack([np.convolve(active[:, s], kernel, mode='same') for s in range(2)], 1)
    floor = 10 ** ((bleed_db + rng.uniform(-jitter_db, jitter_db, size=2)) / 20)
    gain = floor + (1 - floor) * np.clip(active, 0, 1)
    return (x[:, None] * gain).astype(np.float32)


def frame_activity(segments, frames, frame_s=0.08):
    """Diarization/annotation segments -> [frames, 2] activity on the model grid."""
    a = np.zeros((frames, 2), np.float32)
    centers = (np.arange(frames) + 0.5) * frame_s
    for start, end, speaker in segments:
        a[(centers >= start) & (centers < end), speaker] = 1
    return a


def resolve_permutation(reference, estimate):
    """Swap estimated channels if that agrees better with reference activity [T, 2]."""
    same = (reference == estimate).mean()
    swapped = (reference == estimate[:, ::-1]).mean()
    return (estimate[:, ::-1], True) if swapped > same else (estimate, False)


def activity_agreement(reference, estimate):
    """Frame-level agreement of binary activity [T, 2], after permutation resolution.

    Reports overall accuracy, per-class F1, and overlap recall: overlap is where
    interruptions live and where single-channel pipelines fail first.
    """
    ref = np.asarray(reference) > 0.5
    est, swapped = resolve_permutation(ref, np.asarray(estimate) > 0.5)
    tp = (ref & est).sum()
    precision = tp / max(1, est.sum())
    recall = tp / max(1, ref.sum())
    overlap = ref.all(1)
    return dict(
        accuracy=float((ref == est).mean()),
        f1=float(2 * precision * recall / max(1e-9, precision + recall)),
        overlap_frames=int(overlap.sum()),
        overlap_recall=float(est[overlap].all(1).mean()) if overlap.any() else float('nan'),
        swapped=bool(swapped),
    )
