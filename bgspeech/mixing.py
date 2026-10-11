"""Background-speech mixtures for TurnBench-format stereo conversations.

The deployment case: a voice agent hears the user through one microphone, and a
TV / YouTube video / another person talks somewhere in the room. That speech lands
only in the *user* channel. We simulate it by adding a different conversation
(both of its speakers, summed: a dialogue, like a podcast playing) to one channel,
passed through a far-field room response and a loudspeaker band-pass, at a given
SNR relative to the user's own active speech level.

Background donors come from conversations outside the evaluated subset, so no
background voice is ever a foreground speaker.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from scipy import signal

SPEAKERS = (1, 2)
ANNOTATORS = ("a", "b", "c")


# ---- data access (row by row: a whole TB dev shard decodes to tens of GB) ----

def parquet_files(data: str | Path) -> list[Path]:
    """`data`: a directory of parquet shards, or one shard."""
    p = Path(data)
    return [p] if p.is_file() else sorted(p.glob("*.parquet"))


def list_ids(data_dir) -> list[str]:
    import pyarrow.parquet as pq
    ids = []
    for f in parquet_files(data_dir):
        ids += pq.read_table(f, columns=["conversation_id"])["conversation_id"].to_pylist()
    return sorted(ids)


def iter_rows(data_dir, ids: set[str] | None = None, audio: bool = True):
    """Yield one dict per conversation: id, annotations[(spk, ann)], audio[spk] (wav, sr)."""
    import pyarrow.parquet as pq
    import soundfile as sf
    cols = ["conversation_id"] + [f"speaker_{s}_annotation_{a}" for s in SPEAKERS for a in ANNOTATORS]
    if audio:
        cols += [f"speaker_{s}_audio" for s in SPEAKERS]
    for f in parquet_files(data_dir):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1, columns=cols):
            row = batch.to_pylist()[0]
            cid = row["conversation_id"]
            if ids is not None and cid not in ids:
                continue
            out = {"id": cid, "annotations": {
                (s, a): [(e["start_s"], e["end_s"], e["label"]) for e in row[f"speaker_{s}_annotation_{a}"]]
                for s in SPEAKERS for a in ANNOTATORS}}
            if audio:
                out["audio"] = {s: sf.read(io.BytesIO(row[f"speaker_{s}_audio"]["bytes"]), dtype="float32",
                                           always_2d=False) for s in SPEAKERS}
            yield out


def activity_mask(annotations, speaker: int, n: int, rate: float) -> np.ndarray:
    """Union over annotators of every annotated segment of `speaker`, as a boolean
    mask with `n` frames at `rate` Hz."""
    m = np.zeros(n, bool)
    for a in ANNOTATORS:
        for start, end, _label in annotations[(speaker, a)]:
            m[max(0, int(start * rate)):min(n, int(np.ceil(end * rate)))] = True
    return m


# ---- acoustics ---------------------------------------------------------------

def far_field_rir(sr: int, rng: np.random.Generator, rt60=(0.3, 0.7), drr_db=(-3.0, 3.0)) -> np.ndarray:
    """Synthetic room impulse response: a direct path plus an exponentially decaying
    noise tail (RT60 and direct-to-reverberant ratio drawn from the given ranges)."""
    t60 = rng.uniform(*rt60)
    n = int(sr * t60 * 1.2)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / t60)  # -60 dB at t60
    tail[: int(0.0025 * sr)] = 0.0  # first reflections arrive after the direct path
    drr = 10 ** (rng.uniform(*drr_db) / 10)
    tail *= np.sqrt(1.0 / (drr * np.sum(tail ** 2)))
    h = tail
    h[0] = 1.0
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def loudspeaker(x: np.ndarray, sr: int, lo=150.0, hi=6000.0) -> np.ndarray:
    """Band-limit like a laptop / TV speaker."""
    sos = signal.butter(4, [lo, min(hi, 0.45 * sr)], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def active_rms(x: np.ndarray, sr: int, mask: np.ndarray | None = None, mask_rate: float = 100.0) -> float:
    """RMS over speech-active samples: inside `mask` if given, else over 20 ms frames
    within 30 dB of the loud (95th percentile) frames."""
    if mask is not None:
        idx = np.repeat(mask, int(round(sr / mask_rate)))[: len(x)]
        sel = x[: len(idx)][idx]
        return float(np.sqrt(np.mean(sel ** 2) + 1e-12)) if sel.size else 0.0
    hop = int(0.02 * sr)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    keep = e > np.percentile(e, 95) - 30
    return float(np.sqrt(np.mean(frames[keep] ** 2)))


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def donor_dialogue(row: dict, sr: int) -> np.ndarray:
    """Both speakers of a donor conversation summed into one track at `sr`."""
    (w1, s1), (w2, s2) = row["audio"][1], row["audio"][2]
    w1, w2 = resample_to(w1, s1, sr), resample_to(w2, s2, sr)
    m = min(len(w1), len(w2))
    return w1[:m] + w2[:m]


def concat_offset(donors: list[np.ndarray], n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Donor tracks concatenated until long enough, cut to n samples at a random offset."""
    parts, total = [], 0
    for d in donors:
        parts.append(d)
        total += len(d)
        if total >= n + sr * 60:
            break
    bg = np.concatenate(parts)
    while len(bg) < n:
        bg = np.concatenate([bg, bg])
    off = int(rng.integers(0, max(1, len(bg) - n)))
    return bg[off: off + n]


def playback(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Through a far-field room and a small loudspeaker."""
    return loudspeaker(signal.fftconvolve(x, far_field_rir(sr, rng))[:len(x)].astype(np.float32), sr)


def background_track(donors: list[np.ndarray], n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A far-field 'podcast playing' track of length n: donor dialogues (from
    `donor_dialogue`), concatenated until long enough, from a random offset."""
    return playback(concat_offset(donors, n, sr, rng), sr, rng)


def mix(user: np.ndarray, sr: int, bg: np.ndarray, snr_db: float, user_mask: np.ndarray,
        bg_level: np.ndarray | None = None) -> np.ndarray:
    """user + bg scaled so that user active-speech level / bg active level = snr_db.
    `bg_level` (default: bg itself) is the track whose level sets the gain, so a gated
    copy of a background is scaled exactly like the ungated one."""
    ref = bg if bg_level is None else bg_level
    gain = active_rms(user, sr, user_mask) / (active_rms(ref, sr) + 1e-9) * 10 ** (-snr_db / 20)
    y = user + gain * bg
    peak = np.max(np.abs(y))
    return (y / peak * 0.99).astype(np.float32) if peak > 0.99 else y.astype(np.float32)


def smooth_gate(mask: np.ndarray, mask_rate: float, n: int, sr: int, pad_s=0.2, ramp_s=0.05) -> np.ndarray:
    """Sample-rate gain: 1 inside `mask` (dilated by pad_s), 0 elsewhere, with linear ramps."""
    pad = int(pad_s * mask_rate)
    m = np.convolve(mask.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    g = np.repeat(m.astype(np.float32), int(round(sr / mask_rate)))[:n]
    g = np.pad(g, (0, max(0, n - len(g))))
    k = max(1, int(ramp_s * sr))
    return np.convolve(g, np.ones(k) / k, mode="same").astype(np.float32)
