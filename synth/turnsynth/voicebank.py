"""Voice-prompt bank for zero-shot TTS backends (IndexTTS2).

A bank is a directory of wav clips plus voices.json:

    {"globe_S_013519": {"file": "globe_S_013519.wav", "gender": "female", "accent": "...", "source": "..."}}

IndexTTS copies the recording quality of its prompt, so the bank decides
how clean the output is. `build_libritts` (the default) uses LibriTTS-R
parquet shards (mythicinfinity/libritts_r, CC BY 4.0, studio-quality restored
audiobook speech); gender is not in the shards, so it is taken from median
F0 and speakers in the ambiguous 150-180 Hz band are skipped. On 20 lines
of the casual example a LibriTTS-R prompt gave estimated PESQ 3.5 and SI-SDR
21 dB against 2.9 / 15 dB for GLOBE (torchaudio SQUIM). `build_globe` uses
MushanW/GLOBE_V2 (CC0, Common Voice: more accents, but noisy microphones).
Both join a few utterances per speaker into a 6-10 s clip.
"""

import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

GLOBE_URL = "https://huggingface.co/datasets/MushanW/GLOBE_V2/resolve/main/data/{split}-{i:05d}-of-{n:05d}.parquet"


def load_bank(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    bank = json.loads((path / "voices.json").read_text())
    for meta in bank.values():
        meta["path"] = str(path / meta["file"])
    return bank


def build_globe(shards: list[str | Path], out: str | Path, *, per_gender: int = 40, accents: list[str] | None = None,
                min_s: float = 6.0, max_s: float = 10.0, seed: int = 0) -> dict[str, dict]:
    import pyarrow.parquet as pq

    rows = defaultdict(list)
    for shard in shards:
        t = pq.read_table(shard)
        for r in t.to_pylist():
            if r["gender"] not in ("female", "male"):
                continue
            if accents and not any(a.lower() in r["accent"].lower() for a in accents):
                continue
            rows[r["speaker_id"]].append(r)
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank: dict[str, dict] = {}
    by_gender: dict[str, list[str]] = defaultdict(list)
    for spk, utts in rows.items():
        if sum(u["duration"] for u in utts) >= min_s:
            by_gender[utts[0]["gender"]].append(spk)
    for gender, speakers in by_gender.items():
        for spk in rng.permutation(sorted(speakers))[:per_gender]:
            clip, sr = _join(rows[spk], max_s)
            if len(clip) < min_s * sr:
                continue
            name = f"globe_{spk}"
            sf.write(out / f"{name}.wav", clip, sr)
            u = rows[spk][0]
            bank[name] = {"file": f"{name}.wav", "gender": gender, "accent": u["accent"], "source": "GLOBE_V2 (CC0)"}
    (out / "voices.json").write_text(json.dumps(bank, indent=1))
    return bank


def build_libritts(shards: list[str | Path], out: str | Path, *, per_gender: int = 40,
                   min_s: float = 6.0, max_s: float = 10.0, seed: int = 0) -> dict[str, dict]:
    import librosa
    import pyarrow.parquet as pq

    rows = defaultdict(list)
    for shard in shards:
        for r in pq.read_table(shard, columns=["audio", "speaker_id"]).to_pylist():
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            rows[str(r["speaker_id"])].append({"audio": r["audio"], "duration": len(audio) / sr})
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    for spk in rng.permutation(sorted(rows)):
        clip, sr = _join(rows[spk], max_s)
        if len(clip) < min_s * sr:
            continue
        y = librosa.resample(clip, orig_sr=sr, target_sr=16000)
        f0 = float(np.nanmedian(librosa.pyin(y, fmin=60, fmax=400, sr=16000)[0]))
        gender = "female" if f0 >= 180 else "male" if f0 <= 150 else None
        if gender is None or counts[gender] >= per_gender:
            continue
        counts[gender] += 1
        name = f"libritts_{spk}"
        sf.write(out / f"{name}.wav", clip, sr)
        bank[name] = {"file": f"{name}.wav", "gender": gender, "median_f0": round(f0), "source": "LibriTTS-R (CC BY 4.0)"}
    (out / "voices.json").write_text(json.dumps(bank, indent=1))
    return bank


def _join(utts: list[dict], max_s: float) -> tuple[np.ndarray, int]:
    pieces, sr, total = [], None, 0.0
    for u in sorted(utts, key=lambda u: -u["duration"]):
        audio, rate = sf.read(io.BytesIO(u["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr is None:
            sr = rate
        elif rate != sr:
            continue
        audio = _trim_silence(audio, sr)
        if total + len(audio) / sr > max_s:
            continue  # too long to add; a shorter one may still fit
        pieces += [audio, np.zeros(int(0.25 * sr), np.float32)]
        total += len(audio) / sr + 0.25
    clip = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)
    peak = np.abs(clip).max() if len(clip) else 1.0
    return (0.9 * clip / max(peak, 1e-6)).astype(np.float32), sr or 24000


def _trim_silence(audio: np.ndarray, sr: int, pad: float = 0.05) -> np.ndarray:
    from turnsynth.tts import energy_bounds

    lo, hi = energy_bounds(audio, sr)
    return audio[max(0, int((lo - pad) * sr)): int((hi + pad) * sr)]
