"""Grounding: when each speaker can have heard, understood and known what.

The generator writes the whole dialogue at once and sees every line; the speakers in it must not.
Two constraints follow from that, both checked on the rendered timeline:

  causality  a line that depends on the other speaker's words (`responds_to`, and for the agent its
             `after` anchor too, since it can only time an onset off words it has received) can't
             start before its speaker could have perceived them and reacted. render(ground=...)
             moves such lines and logs each move.
  knowledge  a line may only use or reveal facts its speaker started with (`facts`) or heard and had
             time to take in (`reveals` of the other speaker's lines). A fact said in words that were
             cut off, not yet heard, or spoken while the listener was talking over them isn't known.
             leaks() lists every violation with a reason code.

How a speaker perceives depends on who listens. A person recognizes a word before it ends
(`recognize_at`, a stand-in for its uniqueness point) and starts reacting `react` later. The cascade
agent (`Script.agent`) only gets a word once its ASR commits it, `asr_delay` after the word ends; it
decides only at cell boundaries (`cell`, 8 ticks of 80 ms) and is heard `tts_lead` after it decides.
When the user barges in, the agent stops by its pipeline's rule (`stop`): a hard cut on the user's voice,
`vad_latency` after the onset (Frankenvoice), or at the first boundary after the ASR delivers the
interrupting words (DuplexCascade, arXiv 2603.09180). Every constant is a placeholder to calibrate.
"""

import math
from dataclasses import dataclass

from turnsynth import lang as L
from turnsynth.script import SPEAKERS, Script, resolve_after
from turnsynth.tts import Word


@dataclass(frozen=True)
class Grounding:
    cell: float = 0.64  # the agent decides at multiples of this (s)
    asr_delay: float = 0.45  # a user word's end -> committed by the agent's ASR
    tts_lead: float = 0.40  # the agent decides -> its audio starts
    stop: str = "hard_cut"  # how the agent stops for a barge-in: hard_cut | cell
    vad_latency: float = 0.15  # the user's onset -> hard cut
    stop_latency: float = 0.08  # the agent decides to stop -> its audio stops (one TTS frame)
    recognize_at: float = 0.6  # a person recognizes a word this far into it
    react: float = 0.2  # a person, from recognizing a word to starting to react to it
    grasp: float = 0.2  # from hearing a fact's last word to being able to use it
    mask_overlap: bool = True  # a person talking over words doesn't take them in

    def boundary(self, t: float) -> float:
        """The first cell boundary at or after t."""
        return math.ceil(t / self.cell - 1e-9) * self.cell

    def reaction(self, agent: bool, w: Word) -> float:
        """Earliest time a reaction to word w can be heard."""
        if agent:
            return self.boundary(w.end + self.asr_delay) + self.tts_lead
        return w.start + self.recognize_at * (w.end - w.start) + self.react

    def agent_stop(self, onset: float, first_word: Word) -> float:
        """When the agent stops for a barge-in starting at `onset` whose first word is `first_word`."""
        if self.stop == "hard_cut":
            return onset + self.vad_latency
        return self.boundary(first_word.end + self.asr_delay) + self.stop_latency


@dataclass
class Move:
    """A line the renderer had to start later than the script put it, because of what it depends on."""

    item: int
    scripted: float
    start: float
    needs: str  # the quoted words it had to hear first

    def __str__(self) -> str:
        return (f"#{self.item} causality: scripted at {self.scripted:.2f}s, but {self.needs!r} can't be acted on "
                f"before {self.start:.2f}s -> moved")


def onset_bound(ground: Grounding, script: Script, item, placed: dict, anchor=None) -> tuple[float, str | None]:
    """The earliest start for `item` and the quote that sets it (0, None if nothing does).

    `placed` maps item ids to Placed; `anchor` is (host Placed, word count) for the word the renderer
    times the item's onset off: for the agent that word is a dependency too.
    """
    agent = script.agent == item.speaker
    deps = []
    if item.responds_to and item.responds_item in placed:
        host = placed[item.responds_item]
        # For a person reacting to the host it overlays, a quote at or before the anchor adds nothing:
        # the renderer already starts a reaction time after the anchor.
        if agent or anchor is None or host is not anchor[0] or item.responds_word > anchor[1]:
            deps.append((host, item.responds_word, item.responds_to))
    if agent and anchor is not None:
        host, k = anchor
        deps.append((host, k, " ".join(host.item.words[max(0, k - 3):k])))
    best: tuple[float, str | None] = (0.0, None)
    for host, k, quote in deps:
        words = host.words
        if not words:
            continue
        # A word cut off before it was said can't be waited for; the knowledge check reports what depends on it.
        t = ground.reaction(agent, words[min(k, len(words)) - 1])
        if t > best[0]:
            best = (t, quote)
    return best


@dataclass
class Leak:
    """An item that uses or reveals a fact its speaker can't know at its onset."""

    item: int
    speaker: str
    fact: str
    code: str  # never_said | not_yet_said | not_yet_heard | cut_off | spoken_over
    source: int | None = None  # the other speaker's item that says it, if any

    WHY = {
        "never_said": "never said to them",
        "not_yet_said": "only said later",
        "not_yet_heard": "said, but not heard and taken in yet",
        "cut_off": "its words were cut off",
        "spoken_over": "its words were spoken while they were talking over them",
    }

    def __str__(self) -> str:
        src = f" (#{self.source})" if self.source is not None else ""
        return (f"#{self.item} belief leak: {self.speaker} uses {self.fact!r}, {self.WHY[self.code]}{src} -> "
                f"regenerate from {self.speaker}'s view")


def _carrier(placed, quote: str, language: str) -> tuple[list[Word], bool]:
    """The words of `placed` that carry a revealed fact (global times), and whether all were said."""
    words = placed.item.words
    end = resolve_after(words, quote, language)
    n = len([w for w in quote.split() if L.bare(w, language)])
    start = max(0, end - n)
    said = placed.words
    if end > len(said):
        return [], False
    return said[start:end], True


def _talking(rendered, speaker: str, t: float) -> bool:
    return any(p.item.speaker == speaker and not p.dropped and any(w.start <= t <= w.end for w in p.words)
               for p in rendered.placed)


def knows(ground: Grounding, rendered, speaker: str, fact: str, onset: float) -> tuple[bool, str, int | None]:
    """Whether `speaker` knows `fact` at `onset`; if not, the reason code and the item that says it."""
    script = rendered.script
    if fact in script.facts(speaker):
        return True, "", None
    agent = script.agent == speaker
    other = [p for p in rendered.placed if not p.dropped and p.item.speaker != speaker and fact in p.item.reveals]
    best: tuple[str, int | None] = ("never_said", None)
    rank = ["never_said", "not_yet_said", "cut_off", "spoken_over", "not_yet_heard"]
    for p in sorted(other, key=lambda p: p.start):
        words, said = _carrier(p, p.item.reveals[fact], script.language)
        if p.start >= onset:
            code = "not_yet_said"
        elif not said:
            code = "cut_off"
        elif not agent and ground.mask_overlap and any(_talking(rendered, speaker, (w.start + w.end) / 2) for w in words):
            code = "spoken_over"
        else:
            last = words[-1].end
            # The agent decides tts_lead before it is heard, on what its ASR has committed by then.
            ready = last + ground.asr_delay <= onset - ground.tts_lead + 1e-9 if agent else last + ground.grasp <= onset
            if ready:
                return True, "", p.item.id
            code = "not_yet_heard"
        if rank.index(code) > rank.index(best[0]):
            best = (code, p.item.id)
    return False, best[0], best[1]


def leaks(ground: Grounding, rendered) -> list[Leak]:
    """Every fact an item uses or reveals that its speaker can't know at its onset, in onset order."""
    script = rendered.script
    live = sorted((p for p in rendered.placed if not p.dropped), key=lambda p: p.start)
    # A fact nobody declared belongs to whoever reveals it first: they brought it into the conversation.
    declared = {f for s in SPEAKERS for f in script.facts(s)}
    first: dict[str, str] = {}
    for p in live:
        for f in p.item.reveals:
            if f not in declared:
                first.setdefault(f, p.item.speaker)
    out, known_own = [], {s: set() for s in SPEAKERS}
    for p in live:
        spk = p.item.speaker
        for fact in list(p.item.uses) + [f for f in p.item.reveals if f not in p.item.uses]:
            if fact in known_own[spk] or first.get(fact) == spk:
                continue
            ok, code, src = knows(ground, rendered, spk, fact, p.start)
            if ok:
                known_own[spk].add(fact)
            else:
                out.append(Leak(p.item.id, spk, fact, code, src))
    return out
