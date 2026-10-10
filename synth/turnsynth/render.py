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

from turnsynth import lang as L
from turnsynth.config import TYPES, Timing
from turnsynth.script import Item, Script
from turnsynth.tts import TTS, Context, Speech, Word

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


def synthesize_item(tts: TTS, item: Item, voice: str, timing: Timing, rng: np.random.Generator,
                    context: Context | None = None, language: str = "en") -> Speech:
    """Synthesize one item with its scripted within-turn pauses.

    A whole-turn backend (IndexTTS) gets the full text in one call, with the
    pauses written as punctuation so prosody runs across them, and the pauses
    are then re-timed on the aligned word boundaries. Other backends are
    called chunk by chunk with silence inserted between chunks.
    """
    if getattr(tts, "whole_turn", False):
        text, cuts = spoken_text(item, language)
        sp = tts.synthesize(text, voice, context=context, language=language)
        words = item.words
        if len(sp.words) == len(words):
            sp = Speech(sp.audio, sp.sample_rate, [Word(w, x.start, x.end) for w, x in zip(words, sp.words)])
            sp = _retime(sp, [(k, timing.pause(rng, pause)) for k, pause in cuts if k < len(words)])
        return _trim(sp)
    sr = tts.sample_rate
    audio, words, offset = [], [], 0.0
    for text, pause in item.chunks:
        sp = tts.synthesize(text, voice, language=language)
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


def spoken_text(item: Item, language: str = "en") -> tuple[str, list[tuple[int, float]]]:
    """Text for a whole-turn backend, and the scripted pauses as (after word k, seconds).

    A pause becomes a comma unless the chunk already ends in punctuation: a
    hold pause keeps continuation intonation that way, where synthesizing the
    chunk alone ends it like a finished sentence.
    """
    parts, cuts, n = [], [], 0
    for text, pause in item.chunks:
        n += len(text.split())
        if pause is not None:
            if text[-1] not in L.SENTENCE_END + L.CLAUSE_END:
                text += L.COMMA[language]
            cuts.append((n, pause))
        parts.append(text)
    return " ".join(parts), cuts


def _retime(speech: Speech, cuts: list[tuple[int, float]]) -> Speech:
    """Set the silence after word k to `gap` seconds for each (k, gap).

    The silence runs from where word k has died away to where word k+1's
    sound begins, not between the aligned word times (see _trim).
    """
    if not cuts:
        return speech
    sr, ws = speech.sample_rate, speech.words
    db = _frame_db(speech.audio, sr)
    pieces, shifts, pos, shift = [], [], 0, 0.0
    for k, gap in cuts:
        a, b = ws[k - 1].end, ws[k].start
        if b < a:
            a = b = (a + b) / 2
        a = _decay_end(db, a, b)
        b = _onset_start(db, b, a)
        i0, i1 = int(a * sr), int(b * sr)
        pieces += [_fade(speech.audio[pos:i0].copy(), sr, out=True), np.zeros(int(gap * sr), np.float32)]
        shift += gap - (b - a)
        shifts.append((k, shift))
        pos = i1
    pieces.append(_fade(speech.audio[pos:].copy(), sr, out=False))
    words, j, cur = [], 0, 0.0
    for i, w in enumerate(ws):
        while j < len(shifts) and i >= shifts[j][0]:
            cur = shifts[j][1]
            j += 1
        words.append(Word(w.text, w.start + cur, w.end + cur))
    return Speech(np.concatenate(pieces), sr, words)


def _trim(speech: Speech, max_head: float = 0.2, max_tail: float = 0.5) -> Speech:
    """Cut a line from its first sound to where its last word has died away.

    Forced-alignment word ends land before the voice and breath decay, and
    cutting there sounds like airflow stopping dead. The cut goes where the
    level falls DECAY_DB under the line's peak (at most max_tail past the
    aligned end), with a fade. Word times keep marking the words, so
    placement and labels still use the aligned speech end.
    """
    if not speech.words:
        return speech
    sr = speech.sample_rate
    db = _frame_db(speech.audio, sr)
    first, last = speech.words[0].start, speech.words[-1].end
    # No quiet frame before the first word: the previous line runs into it, so start near the word.
    t0 = _onset_start(db, first, max(0.0, first - max_head), fallback=first - 0.03)
    t1 = _decay_end(db, last, min(speech.duration, last + max_tail))
    audio = speech.audio[int(t0 * sr): int(t1 * sr)].copy()
    audio = _fade(_fade(audio, sr, out=False, length=0.01), sr, out=True, length=0.05)
    words = [Word(w.text, w.start - t0, w.end - t0) for w in speech.words]
    return Speech(audio, sr, words)


DECAY_DB = 45.0  # a sound has ended once it is this far under the line's peak
HOP_S = 0.005


def _frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """Level of each HOP_S frame relative to the loudest frame, in dB."""
    hop = max(1, int(HOP_S * sr))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(1)
    db = 10 * np.log10(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return db - db.max()


QUIET_S = 0.08  # longer than a stop closure, so a final "t" or "k" release is kept


def _quiet(db: np.ndarray, i: int, j: int) -> np.ndarray:
    """Frames in [i, j) that start a run of QUIET_S under -DECAY_DB (runs may end at j)."""
    m = max(1, int(QUIET_S / HOP_S))
    loud = np.concatenate([db[i:j] >= -DECAY_DB, np.zeros(m - 1, bool)])
    if len(loud) < m:
        return np.zeros(0, int)
    run_loud = np.lib.stride_tricks.sliding_window_view(loud, m).any(axis=1)
    return np.flatnonzero(~run_loud)


def _decay_end(db: np.ndarray, t: float, limit: float) -> float:
    """Start of the first quiet run in [t, limit], or limit if the sound never dies away."""
    i, j = int(t / HOP_S), int(limit / HOP_S)
    quiet = _quiet(db, i, j)
    return (i + quiet[0]) * HOP_S if len(quiet) else max(t, limit)


def _onset_start(db: np.ndarray, t: float, limit: float, fallback: float | None = None) -> float:
    """End of the last quiet run in [limit, t]; if there is none, fallback (default limit)."""
    i, j = int(limit / HOP_S), int(t / HOP_S)
    quiet = _quiet(db[::-1], len(db) - j, len(db) - i)
    if len(quiet):
        return (j - quiet[0]) * HOP_S
    return min(t, limit if fallback is None else max(limit, fallback))


def _fade(audio: np.ndarray, sr: int, out: bool, length: float = FADE_S) -> np.ndarray:
    n = min(len(audio), int(length * sr))
    if n:
        ramp = np.sin(np.linspace(0.0, np.pi / 2, n, dtype=np.float32)) ** 2
        audio[-n:] *= ramp[::-1] if out else ramp
    return audio


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
        pool = [v for v in tts.voices(gender, script.language) if v not in used] \
            or [v for v in tts.voices("any", script.language) if v not in used]
        if not pool:
            raise ValueError(f"no {script.language} voices in this TTS backend or voice bank")
        voices[spk] = str(rng.choice(pool))
        used.add(voices[spk])
    return voices


def synthesize_by_item(tts: TTS, script: Script, voices: dict[str, str], timing: Timing,
                       rng: np.random.Generator) -> dict[int, Speech]:
    """One call per item in script order, so a contextual backend hears the dialogue as it unfolds."""
    speeches: dict[int, Speech] = {}
    history: dict[str, list[Speech]] = {"A": [], "B": []}
    for item in script.items:
        other = "B" if item.speaker == "A" else "A"
        ctx = Context(history=list(history[item.speaker]), partner=history[other][-1] if history[other] else None,
                      emotion=item.emotion)
        speeches[item.id] = synthesize_item(tts, item, voices[item.speaker], timing, rng, ctx, script.language)
        history[item.speaker].append(speeches[item.id])
    return speeches


def synthesize_by_speaker(tts: TTS, script: Script, voices: dict[str, str], timing: Timing,
                          rng: np.random.Generator, max_words: int | None = None) -> dict[int, Speech]:
    """Read each speaker's lines in long passes, then cut the pass back into items.

    Per-item calls give every line its own prosodic start and end, which
    sounds disconnected from the speaker's previous line. Here all of one
    speaker's items, in order, go into as few calls as the model's length
    allows (`max_words` per pass; the rolling prompt carries over between
    passes), so one line flows into the next as it does when a person talks.
    With the backend's pass_split="floor", a pass also ends whenever the
    other speaker takes the floor: a speaker's lines within one floor are
    read together, but a real turn end stays the end of a reading, so it
    keeps its final fall.
    Items are cut apart where each one's sound starts and dies away, and their
    within-item pauses re-timed as usual. An item whose aligned length is
    implausible (the model skipped or slurred it) is re-synthesized alone.
    """
    max_words = max_words or getattr(tts, "pass_words", 110)
    by_floor = getattr(tts, "pass_split", "speaker") == "floor"
    speeches: dict[int, Speech] = {}
    for spk in ("A", "B"):
        blocks: list[list[Item]] = [[]]
        handover = False  # the other speaker took the floor since this speaker's last item
        for it in script.items:
            if it.speaker != spk:
                handover |= it.is_floor
                continue
            if blocks[-1] and (sum(len(b.words) for b in blocks[-1]) + len(it.words) > max_words
                               or (by_floor and handover)):
                blocks.append([])
            blocks[-1].append(it)
            handover = False
        history: list[Speech] = []
        for block in blocks:
            texts, cuts = [], []
            for it in block:
                text, c = spoken_text(it, script.language)
                if text[-1] not in L.SENTENCE_END:
                    text += L.ELLIPSIS[script.language] if text[-1] not in L.CLAUSE_END else ""
                texts.append(text)
                cuts.append(c)
            sp = tts.synthesize(" ".join(texts), voices[spk],
                                context=Context(history=list(history)), language=script.language)
            history.append(sp)
            n_words = [len(it.words) for it in block]
            if len(sp.words) != sum(n_words):
                for it in block:
                    speeches[it.id] = synthesize_item(tts, it, voices[spk], timing, rng,
                                                      Context(history=list(history)), script.language)
                continue
            # Each item's piece runs from the previous item's last word to the
            # next item's first, so _trim can find where its sound starts and ends.
            starts, ends, k = [0.0], [], 0
            for n in n_words[:-1]:
                k += n
                ends.append(sp.words[k].start)
                starts.append(sp.words[k - 1].end)
            ends.append(sp.duration)
            k = 0
            for it, n, c, t0, t1 in zip(block, n_words, cuts, starts, ends):
                ws = sp.words[k: k + n]
                k += n
                span = ws[-1].end - ws[0].start
                if span < 0.07 * n or span > 1.2 * n + 0.5:
                    speeches[it.id] = synthesize_item(tts, it, voices[spk], timing, rng,
                                                      Context(history=list(history)), script.language)
                    continue
                sr = sp.sample_rate
                piece = Speech(sp.audio[int(t0 * sr): int(t1 * sr)].copy(), sr,
                               [Word(w, x.start - t0, x.end - t0) for w, x in zip(it.words, ws)])
                piece = _retime(piece, [(j, timing.pause(rng, pause)) for j, pause in c if j < n])
                speeches[it.id] = _trim(piece)
    return speeches


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

    if getattr(tts, "speaker_pass", False):
        speeches = synthesize_by_speaker(tts, script, voices, timing, rng)
    else:
        speeches = synthesize_by_item(tts, script, voices, timing, rng)

    # Pass 1: the floor sequence.
    prev: Placed | None = None
    for item in script.items:
        if not item.is_floor:
            continue
        speech = speeches[item.id]
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
        speech = speeches[item.id]
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
