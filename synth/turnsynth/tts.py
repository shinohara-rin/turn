"""TTS backends. Each returns mono float32 audio and per-word timings.

Word timings are what let the renderer anchor backchannels and interruptions
on a specific word of the host turn and cut interrupted turns on a word
boundary, so they are part of the interface rather than an extra.

Backends:
  kokoro  Kokoro-82M (Apache-2.0). Runs on CPU at a few x real time, 50+ voices,
          returns word timestamps. The default.
  indextts IndexTTS2 / IndexTTS-2.5 (MultiTalk's choice). Zero-shot from a
          voice-prompt bank, emotion vectors, GPU. Synthesizes whole turns
          with dialogue context (see IndexTTS and render.synthesize_item).
  dummy   Shaped noise bursts per word. No model; for tests.
"""

import itertools
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from turnsynth.script import EMOTIONS


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


@dataclass
class Context:
    """What the dialogue so far tells a contextual backend about the next utterance."""

    history: list[Speech] = field(default_factory=list)  # this speaker's earlier utterances, oldest first
    partner: Speech | None = None  # the other speaker's latest utterance
    emotion: dict[str, float] | None = None  # script-level emotion for this item (IndexTTS2's 8 axes)


class TTS(Protocol):
    sample_rate: int

    def voices(self, gender: str) -> list[str]: ...

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> Speech: ...


# A backend with `whole_turn = True` gets each item as one text, with pauses
# written as punctuation, plus a Context; render re-times the pauses itself.


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


class IndexTTS:
    """IndexTTS-2.5 (default) or IndexTTS2, zero-shot from a voice bank.

    Where MultiTalk calls IndexTTS2 once per short utterance with a fixed
    reference clip, this backend gives the model the dialogue context it can
    take:

    - whole turns: one call per item with the full text (pauses become
      punctuation, so the model knows the sentence goes on and keeps
      continuation intonation); max_text_tokens_per_segment is raised so
      IndexTTS does not split the turn into independently generated pieces.
    - rolling speaker prompt: the speaker prompt is the bank clip (timbre
      anchor) followed by the last few seconds of that speaker's rendered
      speech, so pitch, energy and pace carry from line to line, and the
      s2mel stage continues acoustically from the end of the prompt.
    - emotion: an item's `emotion` vector (written by the script LLM, which
      sees the whole dialogue) drives IndexTTS2's emotion control; without
      one, `entrain` > 0 uses the partner's last utterance as the emotion
      reference at that strength, otherwise the rolling prompt sets it.
    Word timings come from forced alignment (align.py). context=False turns
    all of this off (chunk-by-chunk calls from the bank clip), for comparison.
    """

    sample_rate = 22050
    whole_turn = True
    EMOTIONS = EMOTIONS  # IndexTTS2's emotion-vector order

    def __init__(self, model_dir: str, bank: str, *, version: str = "2.5", device: str | None = None,
                 half: bool = True, anchor_s: float = 8.0, history_s: float = 3.0, entrain: float = 0.0,
                 emo_alpha: float = 0.8, context: bool = True, pass_mode: str = "floor", pass_words: int = 60):
        import soundfile as sf

        from turnsynth.align import Aligner
        from turnsynth.voicebank import load_bank

        model_dir = str(model_dir)
        if version == "2.5":
            from indextts.infer_v2_5 import IndexTTS2
            self.model = IndexTTS2(cfg_path=f"{model_dir}/config.yaml", model_dir=model_dir, device=device,
                                   use_bf16=half)
            self.model.low_vram = False  # it would split turns at 40 characters
        else:
            from indextts.infer_v2 import IndexTTS2
            self.model = IndexTTS2(cfg_path=f"{model_dir}/config.yaml", model_dir=model_dir, device=device,
                                   use_fp16=half)
        self.version = version
        # context=False is the MultiTalk-style ablation: chunk by chunk, fixed bank prompt, no emotion.
        self.whole_turn = context
        # pass_mode="speaker": render reads each speaker's lines in long passes (render.synthesize_by_speaker).
        self.speaker_pass = context and pass_mode in ("speaker", "floor")
        self.pass_split = pass_mode
        self.pass_words = pass_words
        self.bank = load_bank(bank)
        self.anchor_s, self.history_s = anchor_s, history_s
        self.entrain, self.emo_alpha = entrain, emo_alpha
        self.aligner = Aligner(device="cpu" if device in (None, "cpu") else device)
        self._sf = sf
        self._tmp = Path(tempfile.mkdtemp(prefix="turnsynth-itts-"))
        self._n = itertools.count()
        self._anchors: dict[str, np.ndarray] = {}

    def voices(self, gender: str) -> list[str]:
        names = [k for k, v in self.bank.items() if v.get("gender") == gender]
        return names or list(self.bank)

    def synthesize(self, text: str, voice: str, speed: float = 1.0, context: Context | None = None) -> Speech:
        context = context if (context is not None and self.whole_turn) else Context()
        kwargs = dict(max_text_tokens_per_segment=600, interval_silence=0, verbose=False)
        if self.version == "2.5":
            kwargs.update(lang="en", duration_factor=1.0 / speed)
        if context.emotion:
            kwargs.update(emo_vector=[float(context.emotion.get(k, 0.0)) for k in self.EMOTIONS],
                          emo_alpha=self.emo_alpha)
        elif self.entrain > 0 and context.partner is not None and context.partner.duration > 1.0:
            kwargs.update(emo_audio_prompt=self._write(context.partner.audio), emo_alpha=self.entrain)
        try:
            audio, words = self._infer(text, voice, context, kwargs)
        finally:
            for f in self._tmp.glob("*.wav"):
                f.unlink()
        return Speech(audio, self.sample_rate, words)

    def _infer(self, text: str, voice: str, context: Context, kwargs: dict) -> tuple[np.ndarray, list[Word]]:
        prompt = self._write(self._prompt(voice, context.history))
        try:
            result = self.model.infer(spk_audio_prompt=prompt, text=text, output_path=None, **kwargs)
        finally:
            Path(prompt).unlink()
        if result is None:
            return np.zeros(int(0.1 * self.sample_rate), np.float32), []
        sr, wav = result
        audio = np.asarray(wav, dtype=np.float32).reshape(len(wav), -1).mean(axis=1) / 32768.0
        return audio, self.aligner(audio, sr, text.split())

    def _anchor(self, voice: str) -> np.ndarray:
        if voice not in self._anchors:
            import torch
            import torchaudio.functional as F

            audio, sr = self._sf.read(self.bank[voice]["path"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != self.sample_rate:
                audio = F.resample(torch.from_numpy(audio), sr, self.sample_rate).numpy()
            self._anchors[voice] = audio[: int(self.anchor_s * self.sample_rate)]
        return self._anchors[voice]

    def _prompt(self, voice: str, history: list[Speech]) -> np.ndarray:
        """Bank clip, then the last `history_s` seconds of this speaker's own speech, silences squeezed.

        A short tail carries the speaker's current pitch, energy and pace into
        the next line at little cost: on 20 lines a 3 s tail kept estimated
        PESQ at 3.5. Longer tails of generated speech fed the model's own
        slowdowns and artifacts back into every later line.
        """
        sr = self.sample_rate
        parts, need = [self._anchor(voice)], int(self.history_s * sr)
        recent: list[np.ndarray] = []
        for sp in reversed(history):
            if need <= 0:
                break
            clip = squeeze_silence(sp.audio, sr)[-need:]
            recent.insert(0, clip)
            need -= len(clip)
        if recent:
            parts += [np.zeros(int(0.15 * sr), np.float32), *recent]
        return np.concatenate(parts)

    def _write(self, audio: np.ndarray) -> str:
        path = self._tmp / f"{next(self._n)}.wav"
        self._sf.write(path, audio, self.sample_rate)
        return str(path)


def squeeze_silence(audio: np.ndarray, sr: int, max_gap: float = 0.15) -> np.ndarray:
    """Shorten every silence to max_gap, so a prompt carries voice and pace but not long pauses."""
    from turnsynth.vad import segments

    segs = segments(audio, sr, merge_gap_s=max_gap, pad_s=0.0)
    if not segs:
        return audio
    gap = np.zeros(int(max_gap * sr), np.float32)
    parts = []
    for s, e in segs:
        parts += [audio[int(s * sr): int(e * sr)], gap]
    return np.concatenate(parts[:-1])


def energy_bounds(audio: np.ndarray, sr: int, rel_db: float = 35.0) -> tuple[float, float]:
    """First and last 10 ms frame within rel_db of the loudest frame."""
    hop = int(0.01 * sr)
    n = len(audio) // hop
    if n == 0:
        return 0.0, len(audio) / sr
    db = 10 * np.log10(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    idx = np.flatnonzero(db > db.max() - rel_db)
    return idx[0] * hop / sr, (idx[-1] + 1) * hop / sr


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
    if name == "indextts":
        return IndexTTS(**kwargs)
    raise ValueError(f"unknown TTS backend {name!r}")
