"""Word timings for TTS backends that do not report them (IndexTTS2).

CTC forced alignment of the known script words with torchaudio's MMS_FA
model (wav2vec2, 1100+ languages, romanized character set). The renderer
needs word ends to anchor backchannels and interruptions and to re-time
within-turn pauses, so alignment is to the script text, not to an ASR
transcript. Japanese and Chinese words are aligned through their romanization
(lang.romanize). Words with no alignable characters (digits, symbols) take their
time from their neighbours.
"""

import numpy as np

from turnsynth.lang import romanize
from turnsynth.tts import Word


class Aligner:
    def __init__(self, device: str = "cpu"):
        import torchaudio

        bundle = torchaudio.pipelines.MMS_FA
        self.device = device
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model(with_star=False).to(device).eval()
        self.dictionary = bundle.get_dict(star=None)

    def tokens(self, word: str) -> list[int]:
        word = word.lower().replace("’", "'")
        return [self.dictionary[c] for c in word if c in self.dictionary and c != "-"]

    def __call__(self, audio: np.ndarray, sr: int, words: list[str], language: str = "en") -> list[Word]:
        import torch
        import torchaudio.functional as F

        wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        if sr != self.sample_rate:
            wave = F.resample(wave, sr, self.sample_rate)
        toks = [self.tokens(w) for w in romanize(words, language)]
        flat = [t for ts in toks for t in ts]
        if not flat:
            return _spread(words, len(audio) / sr)
        with torch.inference_mode():
            emission, _ = self.model(wave.to(self.device))
            labels, scores = F.forced_align(emission, torch.tensor([flat], dtype=torch.int32, device=self.device), blank=0)
        spans = F.merge_tokens(labels[0], scores[0].exp())
        sec = wave.size(1) / emission.size(1) / self.sample_rate
        out: list[Word | None] = []
        i = 0
        for w, ts in zip(words, toks):
            if not ts:
                out.append(None)
                continue
            first, last = spans[i], spans[i + len(ts) - 1]
            out.append(Word(w, first.start * sec, last.end * sec))
            i += len(ts)
        return _fill(words, out, len(audio) / sr)


def _spread(words: list[str], duration: float) -> list[Word]:
    step = duration / max(len(words), 1)
    return [Word(w, i * step, (i + 1) * step) for i, w in enumerate(words)]


def _fill(words: list[str], aligned: list[Word | None], duration: float) -> list[Word]:
    """Give unaligned words a slice of the gap between their aligned neighbours.

    Punctuation-only tokens (a standalone "、" or "？" in a ja/zh script) take no
    time: they sit at the end of the word before them, so a line that ends in
    one still ends where its last spoken word does.
    """
    spoken = [i for i, w in enumerate(words) if any(c.isalnum() for c in w)]
    if not spoken:
        return _spread(words, duration)
    filled = _fill_gaps([words[i] for i in spoken], [aligned[i] for i in spoken], duration)
    out: list[Word] = []
    by_index = dict(zip(spoken, filled))
    for i, w in enumerate(words):
        if i in by_index:
            out.append(by_index[i])
        else:
            t = out[-1].end if out else filled[0].start
            out.append(Word(w, t, t))
    return out


def _fill_gaps(words: list[str], aligned: list[Word | None], duration: float) -> list[Word]:
    out = list(aligned)
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        lo = out[i - 1].end if i > 0 else 0.0
        hi = out[j].start if j < len(out) else duration
        if hi - lo < 0.05 * (j - i):  # no gap to speak of: share the neighbours' time
            lo, hi = (out[i - 1].start if i > 0 else lo), (out[j].end if j < len(out) else hi)
        step = (hi - lo) / (j - i)
        for k in range(i, j):
            out[k] = Word(words[k], lo + (k - i) * step, lo + (k - i + 1) * step)
        i = j
    return out
