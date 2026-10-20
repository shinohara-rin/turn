"""Text-only timing: a TTS backend whose output is word timings from a duration model.

Tapes for a text LLM (tape.py) need when each word is said and how many words fit in a cell, not a
voice, so a script can be placed and checked (ground.py) in milliseconds instead of a TTS pass. A word
lasts its syllables over the articulation rate (conversational English runs about 5 syllables/s, and
TurnBench's conversation types 198-211 words/min with pauses), stretched before punctuation (phrase-final
lengthening); fillers take a fixed time, and punctuation adds a short pause. Pace and per-item speed
come in through `speed`, as for any backend. ja/zh count kana/characters instead of syllables.

The audio is a shaped noise burst per word, as in DummyTTS, so render's energy-based trimming and cuts
work unchanged. Every constant is a placeholder to calibrate: on forced-aligned conversational speech for
people, and on the agent's own TTS for the agent.
"""

import re
import zlib
from dataclasses import dataclass

import numpy as np

from turnsynth import lang as L
from turnsynth.tts import Speech, Word

FILLERS_EN = {"um", "uh", "er", "erm", "hmm", "mm", "mhm", "mmhm", "uhhuh", "ah", "oh"}
UNIT_RATE = {"ja": 7.5, "zh": 5.0}  # kana or characters per second


def syllables(word: str) -> int:
    """Vowel groups, less a silent final e. Rough, but within a syllable on most conversational words."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    n = len(re.findall(r"[aeiouy]+", w))
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


@dataclass
class DurationTTS:
    sample_rate: int = 8000
    rate: float = 5.0  # syllables per second at speed 1.0
    final_lengthening: float = 1.35  # a word before punctuation is this much longer
    filler_s: float = 0.35
    clause_pause: float = 0.12  # after , ; :
    sentence_pause: float = 0.25  # after . ? !

    def voices(self, gender: str, language: str = "en") -> list[str]:
        return [f"duration_{gender}_{i}" for i in range(4)]

    def word_seconds(self, word: str, speed: float = 1.0, language: str = "en") -> float:
        if language in L.CJK:
            d = max(1, len(L.units(word, language))) / UNIT_RATE[language]
        elif L.bare(word, language).replace(" ", "") in FILLERS_EN:
            d = self.filler_s
        else:
            d = syllables(word) / self.rate
        if word[-1] in L.SENTENCE_END + L.CLAUSE_END:
            d *= self.final_lengthening
        return d / speed

    def synthesize(self, text: str, voice: str, speed: float = 1.0, language: str = "en") -> Speech:
        rng = np.random.default_rng(zlib.crc32(f"{voice}|{text}".encode()))
        sr = self.sample_rate
        pieces, words, t = [np.zeros(int(0.05 * sr), np.float32)], [], 0.05
        for w in text.split():
            n = max(1, int(self.word_seconds(w, speed, language) * sr))
            pieces.append(0.3 * np.hanning(n).astype(np.float32) * rng.standard_normal(n).astype(np.float32))
            words.append(Word(w, t, t + n / sr))
            t += n / sr
            gap = self.sentence_pause if w[-1] in L.SENTENCE_END else self.clause_pause if w[-1] in L.CLAUSE_END else 0.0
            if gap:
                pieces.append(np.zeros(int(gap * sr), np.float32))
                t += int(gap * sr) / sr
        pieces.append(np.zeros(int(0.05 * sr), np.float32))
        return Speech(np.concatenate(pieces), sr, words)
