"""TTS backends. Each returns mono float32 audio and per-word timings.

Word timings are what let the renderer anchor backchannels and interruptions
on a specific word of the host turn and cut interrupted turns on a word
boundary, so they are part of the interface rather than an extra.

Backends:
  kokoro  Kokoro-82M (Apache-2.0). Runs on CPU at a few x real time, 50+ voices,
          returns word timestamps. The default.
  dummy   Shaped noise bursts per word. No model; for tests.
IndexTTS2 (MultiTalk's choice, emotion control, GPU) slots in behind the same
interface; see README.
"""

import zlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Speech:
    audio: np.ndarray  # mono float32
    sample_rate: int
    words: list[Word]

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


class TTS(Protocol):
    sample_rate: int

    def voices(self, gender: str) -> list[str]: ...

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> Speech: ...


class DummyTTS:
    """Deterministic noise bursts, ~0.28 s per word with 40 ms gaps."""

    sample_rate = 24000

    def voices(self, gender: str) -> list[str]:
        return [f"dummy_{gender}_{i}" for i in range(4)]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> Speech:
        rng = np.random.default_rng(zlib.crc32(f"{voice}|{text}".encode()))
        sr = self.sample_rate
        pieces, words, t = [np.zeros(int(0.05 * sr), np.float32)], [], 0.05
        for w in text.split():
            dur = (0.12 + 0.04 * len(w.strip(",.?!"))) / speed
            n = int(dur * sr)
            env = np.hanning(n).astype(np.float32)
            pieces.append(0.3 * env * rng.standard_normal(n).astype(np.float32))
            words.append(Word(w, t, t + dur))
            t += dur
            gap = 0.04 + (0.15 if w.endswith((",", ".", "?", "!")) else 0.0)
            pieces.append(np.zeros(int(gap * sr), np.float32))
            t += gap
        pieces.append(np.zeros(int(0.05 * sr), np.float32))
        return Speech(np.concatenate(pieces), sr, words)


class KokoroTTS:
    """hexgrad/Kokoro-82M via the `kokoro` package (American English voices)."""

    sample_rate = 24000
    VOICES = {
        "female": ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
                   "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river"],
        "male": ["am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam",
                 "am_onyx", "am_adam", "am_santa"],
    }

    def __init__(self, device: str | None = None):
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)

    def voices(self, gender: str) -> list[str]:
        return list(self.VOICES.get(gender, self.VOICES["female"] + self.VOICES["male"]))

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> Speech:
        audio, words, offset = [], [], 0.0
        for result in self.pipeline(text, voice=voice, speed=speed):
            if result.audio is None:
                continue
            chunk = result.audio.detach().cpu().numpy().astype(np.float32)
            # Kokoro's timestamps run late at chunk edges (the last token absorbs
            # trailing silence); clamp them to where the audio has energy.
            lo, hi = energy_bounds(chunk, self.sample_rate)
            for tok in result.tokens or []:
                if tok.start_ts is None or tok.end_ts is None or not any(c.isalnum() for c in tok.text):
                    continue
                start, end = max(tok.start_ts, lo), min(tok.end_ts, hi)
                words.append(Word(tok.text, offset + start, offset + max(end, start + 0.02)))
            audio.append(chunk)
            offset += len(chunk) / self.sample_rate
        if not audio:
            return Speech(np.zeros(int(0.1 * self.sample_rate), np.float32), self.sample_rate, [])
        return Speech(np.concatenate(audio), self.sample_rate, _attach_punct(words, text))


def energy_bounds(audio: np.ndarray, sr: int, rel_db: float = 35.0) -> tuple[float, float]:
    """First and last 10 ms frame within rel_db of the loudest frame."""
    hop = int(0.01 * sr)
    n = len(audio) // hop
    if n == 0:
        return 0.0, len(audio) / sr
    db = 10 * np.log10(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    idx = np.flatnonzero(db > db.max() - rel_db)
    return idx[0] * 0.01, (idx[-1] + 1) * 0.01


def _attach_punct(words: list[Word], text: str) -> list[Word]:
    """Kokoro tokenizes "don't" -> "do","n't" and drops punctuation; map tokens back
    onto the script's whitespace words so word indices match Item.words."""
    script_words = text.split()
    if len(words) == len(script_words):
        return [Word(sw, w.start, w.end) for sw, w in zip(script_words, words)]
    out, j = [], 0
    for sw in script_words:
        target = "".join(c for c in sw.lower() if c.isalnum())
        if j >= len(words):
            break
        start, end, acc = words[j].start, words[j].end, ""
        while j < len(words) and len(acc) < len(target):
            acc += "".join(c for c in words[j].text.lower() if c.isalnum())
            end = words[j].end
            j += 1
        out.append(Word(sw, start, end))
    return out


def make_tts(name: str, **kwargs) -> TTS:
    if name == "dummy":
        return DummyTTS()
    if name == "kokoro":
        return KokoroTTS(**kwargs)
    raise ValueError(f"unknown TTS backend {name!r}")
