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
For Japanese and Chinese, `build_emilia` reads Emilia-YODAS tars (CC BY 4.0,
in-the-wild, filtered on DNSMOS) and `build_aishell3` AISHELL-3 (Apache 2.0,
studio Mandarin). Each bank entry carries a `language` (missing = English),
and TTS backends only draw voices of the script's language. All builders
join a few utterances per speaker into a 6-10 s clip.
"""

import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

GLOBE_URL = "https://huggingface.co/datasets/MushanW/GLOBE_V2/resolve/main/data/{split}-{i:05d}-of-{n:05d}.parquet"


def load_bank(path: str | Path) -> dict[str, dict]:
    """One bank directory, or several joined with commas (e.g. one per language)."""
    bank: dict[str, dict] = {}
    for p in str(path).split(","):
        p = Path(p)
        for name, meta in json.loads((p / "voices.json").read_text()).items():
            bank[name] = dict(meta, path=str(p / meta["file"]))
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
    import pyarrow.parquet as pq

    rows = defaultdict(list)
    for shard in shards:
        for r in pq.read_table(shard, columns=["audio", "speaker_id"]).to_pylist():
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            rows[str(r["speaker_id"])].append({"audio": r["audio"], "duration": len(audio) / sr})
    order = np.random.default_rng(seed).permutation(sorted(rows))
    return _write_bank(rows, order, out, prefix="libritts", source="LibriTTS-R (CC BY 4.0)", language="en",
                       per_gender=per_gender, min_s=min_s, max_s=max_s)


def build_aishell3(root: str | Path, out: str | Path, *, per_gender: int = 40, min_s: float = 6.0,
                   max_s: float = 10.0, seed: int = 0) -> dict[str, dict]:
    """Mandarin bank from AISHELL-3 (AISHELL/AISHELL-3 on the Hub, Apache 2.0; studio
    recordings, 44.1 kHz). `root` holds spk-info.txt and */wav/<speaker>/*.wav."""
    root = Path(root)
    genders = {}
    for line in (root / "spk-info.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and not line.startswith("#"):
            genders[parts[0]] = parts[2]
    rows = defaultdict(list)
    for f in sorted(root.glob("*/wav/*/*.wav")):
        rows[f.parent.name].append({"audio": {"bytes": f.read_bytes()}, "duration": sf.info(f).duration})
    order = np.random.default_rng(seed).permutation(sorted(rows))
    return _write_bank(rows, order, out, prefix="aishell3", source="AISHELL-3 (Apache 2.0)", language="zh",
                       per_gender=per_gender, min_s=min_s, max_s=max_s, genders=genders)


def build_emilia(tars: list[str | Path], out: str | Path, *, language: str, per_gender: int = 40,
                 min_dnsmos: float = 3.3, min_s: float = 6.0, max_s: float = 10.0) -> dict[str, dict]:
    """Bank from Emilia-YODAS webdataset tars (e.g. TTS-AGI/emilia-yodas JA/*.tar, CC BY 4.0;
    in-the-wild YouTube speech, so only clips with DNSMOS >= min_dnsmos are used, and the
    speakers whose clips score best come first)."""
    import tarfile

    rows = defaultdict(list)
    for path in tars:
        with tarfile.open(path) as tar:
            pending: dict[str, dict] = {}
            for m in tar:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit(".", 1)
                pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
                rec = pending[key]
                if "json" in rec and "mp3" in rec:
                    meta = json.loads(rec["json"])
                    if meta.get("dnsmos", 0.0) >= min_dnsmos and 2.0 <= meta.get("duration", 0.0) <= max_s:
                        rows[meta["speaker"]].append({"audio": {"bytes": rec["mp3"]}, "duration": meta["duration"],
                                                      "dnsmos": meta["dnsmos"]})
                    del pending[key]
    order = sorted(rows, key=lambda k: -np.mean([u["dnsmos"] for u in rows[k]]))
    return _write_bank(rows, order, out, prefix=f"emilia_{language}", source="Emilia-YODAS (CC BY 4.0)",
                       language=language, per_gender=per_gender, min_s=min_s, max_s=max_s)


def _write_bank(rows: dict[str, list[dict]], order, out: str | Path, *, prefix: str, source: str, language: str,
                per_gender: int, min_s: float, max_s: float, genders: dict[str, str] | None = None) -> dict[str, dict]:
    """Join each speaker's utterances into a clip, in `order`, until per_gender voices of each gender.
    Gender comes from `genders` or else from median F0 (the ambiguous 150-180 Hz band is skipped)."""
    import librosa

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    for spk in order:
        if all(counts[g] >= per_gender for g in ("female", "male")):
            break
        clip, sr = _join(rows[spk], max_s)
        if len(clip) < min_s * sr:
            continue
        y = librosa.resample(clip, orig_sr=sr, target_sr=16000)
        f0 = float(np.nanmedian(librosa.pyin(y, fmin=60, fmax=400, sr=16000)[0]))
        gender = (genders or {}).get(spk) or ("female" if f0 >= 180 else "male" if f0 <= 150 else None)
        if gender not in ("female", "male") or counts[gender] >= per_gender:
            continue
        counts[gender] += 1
        name = f"{prefix}_{spk}"
        sf.write(out / f"{name}.wav", clip, sr)
        bank[name] = {"file": f"{name}.wav", "gender": gender, "language": language, "median_f0": round(f0),
                      "source": source}
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
