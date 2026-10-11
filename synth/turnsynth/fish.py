"""Fish Audio S2 backend (hosted API, no GPU).

Zero-shot from the same voice bank as IndexTTS: the bank clip and its
transcript go in as an inline reference (Fish needs the transcript; bank
clips without one are transcribed once with faster-whisper and cached in
the bank's transcripts.json). Like IndexTTS it reads whole turns in floor
passes, and the speaker's latest rendered line rides along as a second
reference, so pitch and pace carry from line to line.

Delivery: the strongest script
emotion axis (if >= EMOTION_MIN) becomes an S2 inline tag such as "[happy]"
in front of the pass. S2 does not speak the tag and the timestamp stream
leaves it out. The script's per-line speed is NOT passed on by default:
S2's `prosody.speed` time-stretches the output, and any value but 1 took
SQUIM PESQ from 4.3 to 2.8-3.1 on the same sentence (`speed=True` turns it
on anyway).

Word timings come from the API's own alignment
(/v1/tts/stream/with-timestamp), mapped onto the script words; if that
fails (text normalization changed the words), MMS_FA forced alignment as
for IndexTTS. `timings="mms"` forces MMS_FA, and every call keeps both in
`self.log` so the two can be compared.

Auth: the agent proxy injects the key for api.fish.audio; elsewhere set
FISH_API_KEY.
"""

import base64
import difflib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from turnsynth.lang import spoken
from turnsynth.tts import Context, Speech, Word, squeeze_silence

URL = "https://api.fish.audio/v1/tts/stream/with-timestamp"
EMOTION_MIN = 0.3
TAGS = {"happy": "happy", "angry": "angry", "sad": "sad", "afraid": "scared", "disgusted": "disgusted",
        "melancholic": "melancholic", "surprised": "surprised", "calm": "calm"}


class FishTTS:
    sample_rate = 24000
    whole_turn = True
    speaker_pass = True

    def __init__(self, bank: str, *, model: str = "s2.1-pro-free", pass_mode: str = "floor", pass_words: int = 60,
                 history: bool = True, history_max_s: float = 8.0, emotion_tags: bool = True,
                 timings: str = "fish", latency: str = "normal", temperature: float = 0.7, top_p: float = 0.7,
                 retry_below: float | None = 2.9, speed: bool = False, asr_device: str = "cpu"):
        from turnsynth.align import Aligner
        from turnsynth.voicebank import load_bank

        self.bank = load_bank(bank)
        self.bank_dirs = [Path(p) for p in str(bank).split(",")]
        self.model, self.latency, self.temperature, self.top_p = model, latency, temperature, top_p
        self.speaker_pass = pass_mode in ("speaker", "floor")
        self.whole_turn = True
        self.pass_split, self.pass_words = pass_mode, pass_words
        self.history, self.history_max_s, self.emotion_tags = history, history_max_s, emotion_tags
        self.timings = timings
        self.use_speed = speed
        self.retry_below = retry_below
        self.aligner = Aligner(device="cpu")
        self.asr_device = asr_device
        self._refs: dict[str, tuple[bytes, str]] = {}
        self.log: list[dict] = []  # per call: text, fish and MMS_FA word times, seconds of audio, wall time

    def voices(self, gender: str, language: str = "en") -> list[str]:
        names = [k for k, v in self.bank.items() if v.get("language", "en") == language]
        return [k for k in names if self.bank[k].get("gender") == gender] or names

    def synthesize(self, text: str, voice: str, speed: float = 1.0, context: Context | None = None,
                   language: str = "en") -> Speech:
        context = context or Context()
        words = text.split()
        refs = [self._reference(voice)]
        if self.history and context.history:
            last = context.history[-1]
            clip = squeeze_silence(last.audio, last.sample_rate)[-int(self.history_max_s * last.sample_rate):]
            if len(clip) > 0.8 * last.sample_rate and last.words:
                # Only whole lines: a reference transcript must match its audio.
                if len(clip) == len(squeeze_silence(last.audio, last.sample_rate)):
                    refs.append((_wav_bytes(clip, last.sample_rate), spoken([w.text for w in last.words], language)))
        tag = self._tag(context.emotion) if self.emotion_tags else ""
        body = {"text": tag + spoken(words, language), "format": "pcm", "sample_rate": self.sample_rate,
                "latency": self.latency, "temperature": self.temperature, "top_p": self.top_p,
                "prosody": {"speed": float(speed) if self.use_speed else 1.0},
                "references": [{"audio": a, "text": t} for a, t in refs]}
        t0 = time.time()
        audio, segs = self._call(body)
        wall = time.time() - t0
        fish = map_segments(words, segs, language)
        mms = self.aligner(audio, self.sample_rate, words, language) if len(audio) else []
        self.log.append({"text": body["text"], "voice": voice, "language": language, "seconds": len(audio) / self.sample_rate,
                         "wall": wall, "segments": segs,
                         "fish": None if fish is None else [(w.start, w.end) for w in fish],
                         "mms": [(w.start, w.end) for w in mms], "audio": audio})
        use = fish if (self.timings == "fish" and fish is not None) else mms
        return Speech(audio, self.sample_rate, use)

    def _tag(self, emotion: dict[str, float] | None) -> str:
        if not emotion:
            return ""
        k, v = max(emotion.items(), key=lambda kv: kv[1])
        return f"[{TAGS.get(k, k)}] " if v >= EMOTION_MIN else ""

    def _call(self, body: dict, tries: int = 5) -> tuple[np.ndarray, list[dict]]:
        import msgpack
        import requests

        headers = {"Content-Type": "application/msgpack", "model": self.model}
        if os.environ.get("FISH_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['FISH_API_KEY']}"
        for attempt in range(tries):
            try:
                r = requests.post(URL, data=msgpack.packb(body), headers=headers, stream=True, timeout=180)
                if r.status_code in (429, 500, 502, 503):
                    raise RuntimeError(f"fish {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    raise ValueError(f"fish {r.status_code}: {r.text[:500]}")
                r.encoding = "utf-8"
                pcm, aligned = bytearray(), {}
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    pcm += base64.b64decode(ev["audio_base64"])
                    if ev.get("alignment"):
                        aligned[ev["chunk_seq"]] = (ev["chunk_audio_offset_sec"], ev["alignment"]["segments"])
                break
            except (RuntimeError, requests.RequestException):
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        audio = np.frombuffer(bytes(pcm[: len(pcm) // 2 * 2]), dtype="<i2").astype(np.float32) / 32768.0
        segs = [{"text": s["text"], "start": off + s["start"], "end": off + s["end"]}
                for _, (off, ss) in sorted(aligned.items()) for s in ss]
        return audio, segs

    def _reference(self, voice: str) -> tuple[bytes, str]:
        if voice not in self._refs:
            meta = self.bank[voice]
            text = meta.get("text") or self._transcript(voice)
            self._refs[voice] = (Path(meta["path"]).read_bytes(), text)
        return self._refs[voice]

    def _transcript(self, voice: str) -> str:
        """Whisper transcript of a bank clip, cached next to the bank's voices.json."""
        path = Path(self.bank[voice]["path"]).parent / "transcripts.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        if voice not in cache:
            from faster_whisper import WhisperModel

            lang = self.bank[voice].get("language", "en")
            if not hasattr(self, "_whisper"):
                self._whisper = WhisperModel("small", device=self.asr_device)
            import librosa

            audio, _ = librosa.load(self.bank[voice]["path"], sr=16000, mono=True)
            parts, _ = self._whisper.transcribe(audio, language=lang, beam_size=5)
            cache[voice] = "".join(p.text for p in parts).strip()
            path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        return cache[voice]


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chars(text: str) -> str:
    return "".join(c for c in text.lower().replace("’", "'") if c.isalnum())


def map_segments(words: list[str], segs: list[dict], language: str = "en",
                 min_matched: float = 0.9) -> list[Word] | None:
    """Times for the script words from Fish's alignment segments.

    Segments are words for English and runs of characters for ja/zh, so both
    sides are flattened to alphanumeric characters, matched with difflib,
    and each script word spans from its first to its last matched character
    (a character's time is interpolated inside its segment). Words with no
    match are filled from their neighbours. None if under `min_matched` of
    the script characters matched (the API normalized the text too far).
    """
    from turnsynth.align import _fill

    if not segs:
        return None
    stream, times = [], []
    for s in segs:
        cs = _chars(s["text"])
        step = (s["end"] - s["start"]) / max(len(cs), 1)
        for i, c in enumerate(cs):
            stream.append(c)
            times.append((s["start"] + i * step, s["start"] + (i + 1) * step))
    script, owner = [], []
    for k, w in enumerate(words):
        for c in _chars(w):
            script.append(c)
            owner.append(k)
    if not script:
        return None
    sm = difflib.SequenceMatcher(None, script, stream, autojunk=False)
    hit: dict[int, list[int]] = {}
    matched = 0
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            hit.setdefault(owner[a + i], []).append(b + i)
        matched += n
    if matched < min_matched * len(script):
        return None
    out: list[Word | None] = []
    for k, w in enumerate(words):
        idx = hit.get(k)
        out.append(Word(w, times[idx[0]][0], times[idx[-1]][1]) if idx else None)
    duration = max(s["end"] for s in segs)
    return _fill(words, out, duration)
