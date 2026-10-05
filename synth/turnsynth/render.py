"""Render a script onto a two-channel timeline.

MultiTalk places turns sequentially with a same-channel non-overlap invariant
and then compresses gaps into fixed overlap ranges. TurnBench needs the timing
itself to be realistic (it is what the benchmark scores), so instead every
placement is sampled from the calibrated timing model in config.py:

  floor turns       start at previous floor end + FTO (or + a pause when the
                    same speaker continues)
  interruptions     start just after the host's `after_word`; the host is cut
                    on a word boundary a sampled yield delay later
  backchannels and  start just after the host's `after_word`; dropped if the
  failed attempts   host was already cut or the listener's channel is busy

The output carries both channels and, per placed item, its realized word
timings and intended fine label. That is the generator's annotation; the
annotators in annotate.py map it onto VAD segments.
"""

from dataclasses import dataclass, field

import numpy as np

from turnsynth.config import TYPES, Timing
from turnsynth.script import Item, Script
from turnsynth.tts import TTS, Speech, Word

FADE_S = 0.03


@dataclass
class Placed:
    item: Item
    speech: Speech  # local timeline; audio already truncated if cut
    start: float
    cut: bool = False
    dropped: str | None = None  # reason if not rendered

    @property
    def end(self) -> float:
        return self.start + self.speech.duration

    @property
    def words(self) -> list[Word]:
        return [Word(w.text, self.start + w.start, self.start + w.end) for w in self.speech.words]

    @property
    def speech_end(self) -> float:
        ws = self.speech.words
        return self.start + (ws[-1].end if ws else self.speech.duration)

    def word_end(self, k: int) -> float:
        """Global end time of the host's k-th word (1-based)."""
        return self.start + self.speech.words[min(k, len(self.speech.words)) - 1].end


@dataclass
class Rendered:
    conversation_id: str
    script: Script
    audio: dict[int, np.ndarray]  # speaker index 1/2 -> mono float32
    sample_rate: int
    placed: list[Placed]
    voices: dict[str, str]
    speaker_index: dict[str, int] = field(default_factory=lambda: {"A": 1, "B": 2})

    @property
    def duration(self) -> float:
        return len(self.audio[1]) / self.sample_rate


def synthesize_item(tts: TTS, item: Item, voice: str, timing: Timing, rng: np.random.Generator) -> Speech:
    """Synthesize chunk by chunk, inserting the scripted within-turn pauses."""
    sr = tts.sample_rate
    audio, words, offset = [], [], 0.0
    for text, pause in item.chunks:
        sp = tts.synthesize(text, voice)
        audio.append(sp.audio)
        words += [Word(w.text, offset + w.start, offset + w.end) for w in sp.words]
        offset += sp.duration
        if pause is not None:
            gap = timing.pause(rng, pause)
            audio.append(np.zeros(int(gap * sr), np.float32))
            offset += gap
    # Trim leading/trailing TTS silence so placement times are speech times.
    speech = Speech(np.concatenate(audio), sr, words)
    return _trim(speech)


def _trim(speech: Speech, pad: float = 0.02) -> Speech:
    if not speech.words:
        return speech
    sr = speech.sample_rate
    t0 = max(0.0, speech.words[0].start - pad)
    t1 = min(speech.duration, speech.words[-1].end + pad)
    audio = speech.audio[int(t0 * sr): int(t1 * sr)]
    words = [Word(w.text, w.start - t0, w.end - t0) for w in speech.words]
    return Speech(audio, sr, words)


def _cut(speech: Speech, at: float) -> Speech:
    sr = speech.sample_rate
    n = int(at * sr)
    audio = speech.audio[:n].copy()
    fade = min(len(audio), int(FADE_S * sr))
    if fade:
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return Speech(audio, sr, [w for w in speech.words if w.end <= at + 1e-6])


def assign_voices(tts: TTS, script: Script, rng: np.random.Generator) -> dict[str, str]:
    """Draw voices per dialogue, without replacement, matched on gender (MultiTalk §2.1.3)."""
    used: set[str] = set()
    voices = {}
    for spk in ("A", "B"):
        gender = str(script.speakers[spk].get("gender", "female")).lower()
        pool = [v for v in tts.voices(gender) if v not in used] or [v for v in tts.voices("any") if v not in used]
        voices[spk] = str(rng.choice(pool))
        used.add(voices[spk])
    return voices


def render(script: Script, tts: TTS, *, conversation_id: str, seed: int = 0,
           timing: Timing | None = None, bleed_db: float | None = None) -> Rendered:
    timing = timing or Timing()
    rng = np.random.default_rng(seed)
    ctype = TYPES[script.conversation_type]
    voices = assign_voices(tts, script, rng)
    placed: dict[int, Placed] = {}
    busy: dict[str, list[tuple[float, float]]] = {"A": [], "B": []}

    def is_free(spk: str, start: float, end: float) -> bool:
        g = timing.min_same_channel_gap
        return all(end + g <= s or start >= e + g for s, e in busy[spk])

    # Pass 1: the floor sequence.
    prev: Placed | None = None
    for item in script.items:
        if not item.is_floor:
            continue
        speech = synthesize_item(tts, item, voices[item.speaker], timing, rng)
        channel_end = max((e for _, e in busy[item.speaker]), default=0.0)
        if prev is None:
            start = 0.5
        elif item.type == "interruption":
            host = prev
            start = host.word_end(item.after_word) + rng.uniform(0.0, timing.onset_jitter)
        elif prev.item.speaker == item.speaker:
            start = prev.speech_end + timing.pause(rng)
        else:
            fto = timing.fto(rng, ctype)
            if prev.item.type == "interruption" and prev.item.host is not None \
                    and script.item(prev.item.host).speaker == item.speaker:
                fto = max(fto, 0.0)  # someone who was just cut off lets the interrupter finish
            start = max(prev.speech_end + fto, prev.start + 0.3)
        start = max(start, channel_end + timing.min_same_channel_gap)
        p = Placed(item, speech, start)
        if item.type == "interruption" and prev is not None:
            _cut_host(prev, p.start, p.start + timing.yield_delay(rng), p.start + speech.duration, busy,
                      timing.yield_min, timing.min_talk_after_yield)
        placed[item.id] = p
        busy[item.speaker].append((p.start, p.end))
        prev = p

    # Pass 2: overlays (backchannels, non-floor-taking interruptions).
    for item in script.items:
        if item.is_floor:
            continue
        host = placed.get(item.host)
        speech = synthesize_item(tts, item, voices[item.speaker], timing, rng)
        p = Placed(item, speech, 0.0)
        placed[item.id] = p
        if host is None or host.dropped:
            p.dropped = "no_host"
            continue
        if item.after_word > len(host.speech.words) - 1:
            p.dropped = "host_cut"
            continue
        # Try the anchor word, then the next two word boundaries if the channel is busy.
        for k in range(item.after_word, min(item.after_word + 3, len(host.speech.words))):
            start = host.word_end(k) + rng.uniform(0.05, timing.onset_jitter + 0.05)
            # A backchannel may trail just past the host's last word; a failed
            # attempt must end while the host is still talking, or it would
            # have taken the floor after all.
            limit = host.speech_end + 0.3 if item.type == "backchannel" else host.speech_end - 0.2
            if start + speech.duration <= limit and is_free(item.speaker, start, start + speech.duration):
                p.start = start
                busy[item.speaker].append((p.start, p.end))
                break
        else:
            p.dropped = "no_room"

    # Mix.
    sr = tts.sample_rate
    live = [p for p in placed.values() if not p.dropped]
    total = max(p.end for p in live) + 0.5
    channels = {1: np.zeros(int(total * sr) + 1, np.float32), 2: np.zeros(int(total * sr) + 1, np.float32)}
    index = {"A": 1, "B": 2}
    for p in live:
        ch = channels[index[p.item.speaker]]
        i0 = int(round(p.start * sr))
        seg = p.speech.audio[: len(ch) - i0]
        ch[i0: i0 + len(seg)] += seg
    if bleed_db is not None:
        gain = 10 ** (bleed_db / 20)
        a, b = channels[1].copy(), channels[2].copy()
        channels[1] += gain * b
        channels[2] += gain * a
    peak = max(np.abs(channels[1]).max(), np.abs(channels[2]).max(), 1e-6)
    if peak > 0.99:
        for k in channels:
            channels[k] *= 0.99 / peak
    order = {it.id: i for i, it in enumerate(script.items)}
    return Rendered(conversation_id, script, channels, sr,
                    sorted(placed.values(), key=lambda p: order[p.item.id]), voices)


def _cut_host(host: Placed, onset: float, cut_time: float, interrupter_end: float,
              busy: dict[str, list[tuple[float, float]]], min_yield: float, min_after: float) -> None:
    """Cut the host on a word boundary after the interrupter's onset.

    The host keeps talking at least `min_yield` s (shorter reads as an early
    smooth transition, not a barge-in) and stops at least `min_after` s before
    the interrupter finishes (or the floor never visibly changed hands). Within
    that window it stops at the first word ending after `cut_time`, or earlier
    at one of its own pauses: people trail off rather than resume over someone.
    """
    words = host.speech.words
    lo, hi = onset + min_yield - host.start, interrupter_end - min_after - host.start
    target = min(max(cut_time - host.start, lo), max(lo, hi))
    at = None
    for w, nxt in zip(words, words[1:]):
        if w.end < onset - host.start + 0.1:
            continue
        if nxt.start - w.end > 0.3 or (w.end >= lo and w.end >= target):
            at = w.end + 0.02
            break
    if at is None:
        return  # host finishes on its own: the scripted interruption did not happen
    old_end = host.end
    host.speech = _cut(host.speech, at)
    host.cut = True
    spans = busy[host.item.speaker]
    for i, (s, e) in enumerate(spans):
        if abs(e - old_end) < 1e-9 and abs(s - host.start) < 1e-9:
            spans[i] = (s, host.end)
