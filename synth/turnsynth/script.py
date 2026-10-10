"""Dialogue script schema, parsing and the four-layer quality filter.

A script is what the generator LLM emits in stage 2 (see prompts.py). It is a
list of items in rough temporal order:

    {"id": 3, "speaker": "A", "type": "turn", "style": "normal",
     "text": "So I went there <pause 0.7> and it was closed."}
    {"id": 4, "speaker": "B", "type": "backchannel", "kind": "continuer",
     "text": "mhm", "host": 3, "after": "I went there"}
    {"id": 5, "speaker": "B", "type": "interruption", "floor_taking": true,
     "stance": "competitive", "text": "Wait, closed?", "host": 3, "after": "it was closed."}

`turn` and floor-taking `interruption` items form the floor sequence.
Backchannels and non-floor-taking interruptions are overlays anchored on a
host turn of the other speaker: `after` quotes the host's words right before
the overlay starts (LLMs miscount words, but copy text reliably), and parse
resolves it to `after_word`, the number of host words spoken by then (which
may also be given directly). For a
floor-taking interruption the host text is what the speaker *would* have said;
the renderer cuts it shortly after the interrupter starts.

Japanese and Chinese scripts (`"language": "ja"|"zh"`) put a space between
words, since `after_word` counts words; see lang.py.

Delivery is optional per item: `"emotion"` ({axis: weight}, IndexTTS2's
eight axes) and `"speed"` (rate relative to the speaker's usual pace, which
`speakers[X]["pace"]` sets), so the TTS is told how a line is said instead of
guessing from its words.

`<pause X>` inside a turn marks a within-turn pause of X seconds: the EOT
hard negatives. The filter rejects scripts with structured reason codes, as in
MultiTalk §2.1.2, so prompt iterations can be tied to failure-rate deltas.
"""

import re
from dataclasses import dataclass, field

from turnsynth import labels, lang as L

PAUSE_RE = re.compile(r"<pause\s+([0-9.]+)\s*>")
SPEAKERS = ("A", "B")
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
ITEM_TYPES = ("turn", "backchannel", "interruption")
SPEED_RANGE = (0.7, 1.4)  # an item's "speed" and a speaker's "pace", as multiples of a typical rate


def pace(speaker: dict) -> float:
    """A speaker's usual speaking rate (`"pace"` in `speakers`, default 1.0)."""
    return float(speaker.get("pace", 1.0) or 1.0)


@dataclass
class Item:
    id: int
    speaker: str
    type: str
    text: str
    style: str = "normal"
    kind: str = "acknowledgement"
    floor_taking: bool = True
    stance: str = "competitive"
    host: int | None = None
    after_word: int | None = None
    after: str | None = None  # quoted host words before the onset; resolved to after_word
    emotion: dict[str, float] | None = None  # optional, for emotion-controllable TTS
    speed: float | None = None  # optional speaking rate relative to the speaker's own pace (1.0 = usual)

    @property
    def chunks(self) -> list[tuple[str, float | None]]:
        """Split text at <pause X> markers -> [(chunk_text, pause_after_s | None)]."""
        parts = PAUSE_RE.split(self.text)
        out: list[tuple[str, float | None]] = []
        for i in range(0, len(parts), 2):
            chunk = " ".join(parts[i].split())
            pause = float(parts[i + 1]) if i + 1 < len(parts) else None
            if chunk:
                out.append((chunk, pause))
            elif out and pause is not None:  # adjacent markers: merge into the previous pause
                prev_text, prev_pause = out[-1]
                out[-1] = (prev_text, (prev_pause or 0.0) + pause)
        return out

    @property
    def words(self) -> list[str]:
        return PAUSE_RE.sub(" ", self.text).split()

    @property
    def is_floor(self) -> bool:
        return self.type == "turn" or (self.type == "interruption" and self.floor_taking)

    @property
    def fine_label(self) -> str:
        if self.type == "turn":
            return labels.TURN_STYLE.get(self.style, labels.NORMAL_TURN)
        if self.type == "backchannel":
            return labels.BACKCHANNEL_KIND.get(self.kind, labels.BC_ACK)
        return labels.interruption_label(self.floor_taking, self.stance)


@dataclass
class Script:
    conversation_type: str
    scenario: str
    speakers: dict[str, dict]
    items: list[Item]
    meta: dict = field(default_factory=dict)
    language: str = "en"  # en | ja | zh; ja/zh text is space-segmented into words (see lang.py)

    def item(self, item_id: int) -> Item:
        return self._by_id[item_id]

    def __post_init__(self) -> None:
        self._by_id = {item.id: item for item in self.items}

    def to_json(self) -> dict:
        def item_json(it: Item) -> dict:
            d = {"id": it.id, "speaker": it.speaker, "type": it.type, "text": it.text}
            if it.type == "turn":
                d["style"] = it.style
            if it.type == "backchannel":
                d["kind"] = it.kind
            if it.type == "interruption":
                d["floor_taking"] = it.floor_taking
                d["stance"] = it.stance
            if it.type != "turn":
                d["host"] = it.host
                d["after_word"] = it.after_word
                if it.after:
                    d["after"] = it.after
            if it.emotion:
                d["emotion"] = it.emotion
            if it.speed is not None:
                d["speed"] = it.speed
            return d

        return {
            "language": self.language,
            "conversation_type": self.conversation_type,
            "scenario": self.scenario,
            "speakers": self.speakers,
            "turns": [item_json(it) for it in self.items],
            "meta": self.meta,
        }


class ScriptError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def parse(obj: dict) -> Script:
    """Layer 1: schema and value ranges. Raises ScriptError(code, detail)."""
    try:
        items = []
        for raw in obj["turns"]:
            item = Item(
                id=int(raw["id"]),
                speaker=str(raw["speaker"]),
                type=str(raw["type"]),
                text=str(raw["text"]).strip(),
                style=str(raw.get("style", "normal")),
                kind=str(raw.get("kind", "acknowledgement")),
                floor_taking=bool(raw.get("floor_taking", True)),
                stance=str(raw.get("stance", "competitive")),
                host=None if raw.get("host") is None else int(raw["host"]),
                after_word=None if raw.get("after_word") is None else int(raw["after_word"]),
                after=None if raw.get("after") is None else str(raw["after"]),
                emotion=None if not raw.get("emotion") else {str(k): float(v) for k, v in raw["emotion"].items()},
                speed=None if raw.get("speed") is None else float(raw["speed"]),
            )
            items.append(item)
        script = Script(
            conversation_type=str(obj["conversation_type"]),
            scenario=str(obj.get("scenario", "")),
            speakers=dict(obj["speakers"]),
            items=items,
            meta=dict(obj.get("meta", {})),
            language=L.check(str(obj.get("language", "en"))),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ScriptError("schema", repr(e)) from e

    if set(script.speakers) != set(SPEAKERS):
        raise ScriptError("schema", f"speakers must be exactly {SPEAKERS}")
    for spk, info in script.speakers.items():
        try:
            ok = SPEED_RANGE[0] <= pace(info) <= SPEED_RANGE[1]
        except (TypeError, ValueError, AttributeError):
            ok = False
        if not ok:
            raise ScriptError("schema", f"speaker {spk}: pace {info.get('pace')} outside {SPEED_RANGE}")
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ScriptError("schema", "duplicate ids")
    seen: dict[int, Item] = {}
    last_floor: Item | None = None
    for it in items:
        if it.speaker not in SPEAKERS:
            raise ScriptError("schema", f"item {it.id}: bad speaker {it.speaker}")
        if it.type not in ITEM_TYPES:
            raise ScriptError("schema", f"item {it.id}: bad type {it.type}")
        if not it.words:
            raise ScriptError("schema", f"item {it.id}: empty text")
        if it.type == "turn" and it.style not in labels.TURN_STYLE:
            raise ScriptError("schema", f"item {it.id}: bad style {it.style}")
        if it.type == "backchannel" and it.kind not in labels.BACKCHANNEL_KIND:
            raise ScriptError("schema", f"item {it.id}: bad kind {it.kind}")
        if it.type == "interruption" and it.stance not in ("competitive", "cooperative"):
            raise ScriptError("schema", f"item {it.id}: bad stance {it.stance}")
        if it.type != "turn":
            host = seen.get(it.host) if it.host is not None else None
            if host is None or not host.is_floor or host.speaker == it.speaker:
                raise ScriptError("anchor", f"item {it.id}: host must be an earlier floor item of the other speaker")
            if it.after:
                it.after_word = resolve_after(host.words, it.after, script.language)
                if it.after_word is None:
                    raise ScriptError("anchor", f"item {it.id}: 'after' {it.after!r} is not in host {host.id}")
            if it.after_word is None or not 1 <= it.after_word < len(host.words):
                raise ScriptError("anchor", f"item {it.id}: after_word out of range for host {host.id}")
            if it.type == "interruption" and it.floor_taking and host is not last_floor:
                raise ScriptError("anchor", f"item {it.id}: a floor-taking interruption must cut the current floor holder")
        if it.emotion and (set(it.emotion) - set(EMOTIONS) or not all(0.0 <= v <= 1.0 for v in it.emotion.values())):
            raise ScriptError("schema", f"item {it.id}: emotion keys must be from {EMOTIONS} with weights in [0, 1]")
        if it.speed is not None and not SPEED_RANGE[0] <= it.speed <= SPEED_RANGE[1]:
            raise ScriptError("schema", f"item {it.id}: speed {it.speed} outside {SPEED_RANGE}")
        for chunk, pause in it.chunks:
            if pause is not None and not 0.1 <= pause <= 5.0:
                raise ScriptError("schema", f"item {it.id}: pause {pause} out of range")
        seen[it.id] = it
        if it.is_floor:
            last_floor = it
    return script


def resolve_after(words: list[str], quote: str, language: str = "en") -> int | None:
    """Number of host words up to and including the first occurrence of `quote`."""
    key = lambda w: L.bare(w, language)
    target = [key(w) for w in PAUSE_RE.sub(" ", quote).split() if key(w)]
    hay = [key(w) for w in words]
    n = len(target)
    if not n:
        return None
    for i in range(len(hay) - n + 1):
        if hay[i: i + n] == target:
            j = i + n
            # Count punctuation-only tokens right after the quote as already spoken.
            while j < len(hay) and not hay[j]:
                j += 1
            return j
    return None


BANNED_PATTERNS = [
    (re.compile(r"[\[\(\*].*?[\]\)\*]"), "stage_direction"),
    (re.compile(r"\bas an ai\b", re.I), "role_break"),
]


def check(script: Script, *, min_items: int = 20) -> list[str]:
    """Layers 2-4. Returns a list of reason codes (empty = accept)."""
    reasons: list[str] = []
    floor = [it for it in script.items if it.is_floor]
    # Layer 2: spoken-length sanity.
    if len(script.items) < min_items:
        reasons.append("too_short")
    for it in script.items:
        n = len(it.words)
        if it.type == "backchannel" and n > 4:
            reasons.append("long_backchannel")
        if it.type == "turn" and n > 120:
            reasons.append("long_turn")
    run, longest = 1, 1
    for prev, cur in zip(floor, floor[1:]):
        run = run + 1 if cur.speaker == prev.speaker else 1
        longest = max(longest, run)
    if longest > 4:
        reasons.append("monologue_run")
    # Layer 3: participation.
    words = {s: sum(len(it.words) for it in script.items if it.speaker == s) for s in SPEAKERS}
    total = sum(words.values()) or 1
    if min(words.values()) / total < 0.15:
        reasons.append("participation")
    # Layer 4: content quality.
    for prev, cur in zip(script.items, script.items[1:]):
        if prev.text.strip().lower() == cur.text.strip().lower() and prev.type == cur.type == "turn":
            reasons.append("verbatim_repeat")
            break
    for it in script.items:
        clean = PAUSE_RE.sub(" ", it.text)
        for pattern, code in BANNED_PATTERNS:
            if pattern.search(clean):
                reasons.append(code)
    if not any(PAUSE_RE.search(it.text) for it in floor):
        reasons.append("no_pauses")
    # ja/zh must be segmented into words, or after_word and turn lengths mean nothing.
    if script.language in L.CJK:
        words = [w for it in script.items for w in it.words]
        if sum(len(L.units(w, script.language)) for w in words) / max(len(words), 1) > 3.5:
            reasons.append("unsegmented")
    # A barge-in needs room to happen: the host must have words left to be cut
    # off in, and the interrupter must keep talking once the host stops.
    for it in script.items:
        if it.type == "interruption" and it.floor_taking:
            host = script.item(it.host)
            if len(host.words) - it.after_word < 6 or len(it.words) < 5:
                reasons.append("weak_interruption")
                break
    return sorted(set(reasons))
