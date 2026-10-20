"""Tapes: a conversation as the cascade agent sees it, one line per decision cell.

The agent's LLM decides every `cell` seconds (ground.py). A cell holds what reached the agent in the
window that just ended (the user's words its ASR committed, whether its VAD heard the user), then the
agent's pipeline state and what it did: its action and the words it sent to TTS. The same cells come
from a rendered script (cells(), for training data) and from a model running live (pilot.py), so a
model tried zero-shot sees what it would later be trained on.

History is what was heard. When the user cuts the agent off, the words it sent but never said are
taken out of the cells that sent them and `[cut]` marks where its voice stopped, so a model never
conditions on speech the user didn't hear (DuplexCascade keeps them, its server.py:605).

Two renderings: line() is readable, for prompting an untrained model; micro_turn() is DuplexCascade's
format (arXiv 2603.09180) in Gemma 4's turn markers, for training, with `<|voice|>` added for a cell
the VAD heard but the ASR committed nothing in.
"""

import re
from dataclasses import dataclass, field

from turnsynth.ground import Grounding
from turnsynth.tts import Word

STATES = ("silent", "speaking", "cut")  # the agent's turn: none open | open and audible | the user cut it
ACTIONS = {
    "silent": ("listen", "respond", "backchannel"),
    "speaking": ("continue", "finish", "stop"),  # stop: only when the agent stops itself (Grounding.stop "cell")
    "cut": ("listen", "resume", "respond"),
}
STATE_TEXT = {"silent": "silent", "speaking": "speaking", "cut": "cut off"}


@dataclass
class Cell:
    t: float  # end of the window: when the agent decides
    user: list[str] = field(default_factory=list)  # words the ASR committed in the window
    voice: bool = False  # the VAD heard the user in the window
    state: str = "silent"
    action: str = "listen"
    say: list[str] = field(default_factory=list)  # sent to TTS and said
    unsaid: list[str] = field(default_factory=list)  # sent to TTS, then cut before they were said
    cut: bool = False  # the agent's voice was cut during this cell's words

    @property
    def sent(self) -> list[str]:
        return self.say + self.unsaid


def no_periods(word: str) -> str:
    """A user word without its periods ("if." -> "if", "U.S." -> "US"); a decimal point between digits stays."""
    return re.sub(r"\.(?!\d)|(?<!\d)\.", "", word)


def user_text(cell: Cell, periods: bool = True) -> str:
    user = cell.user if periods else [w for w in map(no_periods, cell.user) if w]
    words = f'"{" ".join(user)}" ' if user else ""
    return f"{words}({'talking' if cell.voice else 'silent'})"


def agent_text(cell: Cell) -> str:
    """The agent's action as heard, e.g. `continue "but they [cut]"`."""
    words = cell.say + (["[cut]"] if cell.cut else [])
    return cell.action + (f' "{" ".join(words)}"' if words else "")


def prefix(cell: Cell, periods: bool = True) -> str:
    """A line up to the agent's action: what the agent has in front of it when it decides."""
    return f"{cell.t:6.2f} | user: {user_text(cell, periods)} | agent ({STATE_TEXT[cell.state]}):"


def line(cell: Cell, periods: bool = True) -> str:
    return f"{prefix(cell, periods)} {agent_text(cell)}"


CONTROL = {
    "user_talking": "<|user is talking|>",
    "user_thinking": "<|user is thinking|>",
    "finish_talking": "<|user finish talking|>",
    "interruption": "<|user interruption|>",
    "backchannel": "<|user backchannel|>",
    "agent_backchannel": "<|assistant_backchannel|>",
    "no_voice": "<|no voice|>",
    "voice": "<|voice|>",
}


def control(cell: Cell) -> str:
    """DuplexCascade's control token for a cell: what the agent took the user to be doing."""
    if cell.action == "respond":
        return CONTROL["interruption"] if cell.state == "cut" else CONTROL["finish_talking"]
    if cell.action == "resume":
        return CONTROL["backchannel"]
    if cell.action == "backchannel":
        return CONTROL["agent_backchannel"]
    if cell.action == "stop":
        return CONTROL["interruption"]
    if cell.action == "listen":
        return CONTROL["user_talking"] if cell.voice or cell.user else CONTROL["user_thinking"]
    return ""  # continue / finish: the agent's own turn goes on


def micro_turn(cell: Cell) -> tuple[str, str]:
    """(user turn, model turn) in Gemma 4's markers. The model turn is the training target."""
    user = " ".join(cell.user) or CONTROL["voice" if cell.voice else "no_voice"]
    words = " ".join(cell.say + (["[cut]"] if cell.cut else []))
    model = " ".join(x for x in (control(cell), words) if x)
    return f"<|turn>user\n{user}<turn|>\n", f"<|turn>model\n{model}<turn|>\n"


def cells(rendered, ground: Grounding, until: float | None = None) -> list[Cell]:
    """The tape of a rendered script with an agent speaker (`Script.agent`), from the agent's side.

    The agent sends a word one `tts_lead` before it is heard, so the cell deciding at t sends the
    words heard in [t + tts_lead, t + tts_lead + cell). A turn the user cut short also sent, before
    the cut, words it never said (`Placed.uncut`); they are the `unsaid` of the cells that sent them.
    Every agent floor item opens with `respond`: a rendered script doesn't say which ones resume.
    """
    agent = rendered.script.agent
    if agent is None:
        raise ValueError("cells() needs a script with an agent speaker")
    live = [p for p in rendered.placed if not p.dropped]
    user = sorted((w for p in live if p.item.speaker != agent for w in p.words), key=lambda w: w.start)
    floor = [p for p in live if p.item.speaker == agent and p.item.is_floor]
    bcs = [p for p in live if p.item.speaker == agent and p.item.type == "backchannel"]
    until = until if until is not None else max([p.end for p in live] + [0.0]) + 2 * ground.cell
    out: list[Cell] = []
    state, current = "silent", None
    t = ground.cell
    while t <= until + 1e-9:
        lo, hi = t + ground.tts_lead, t + ground.tts_lead + ground.cell
        v0, v1 = t - ground.cell - ground.vad_latency, t - ground.vad_latency
        cell = Cell(t, [w.text for w in user if t - ground.cell < w.end + ground.asr_delay <= t],
                    any(w.start < v1 and w.end > v0 for w in user))
        if current is not None and current.cut and state == "speaking" and t >= current.end:
            state = "cut"  # the agent's pipeline stopped it for the user's barge-in
        cell.state = state
        sending = None
        for p in floor:
            said = [w for w in p.words if lo <= w.start < hi]
            # Until its voice stops; under "cell" the cell that decides to stop sends nothing.
            sends = t < p.end - (ground.stop_latency if ground.stop == "cell" else 0.0) - 1e-6
            unsaid = [w for w in _uncut(p)[len(p.words):] if lo <= w.start < hi] if sends else []
            if said or unsaid:
                sending = p
                cell.say, cell.unsaid = [w.text for w in said], [w.text for w in unsaid]
                break
        if sending is not None:
            cell.action = "continue" if sending is current else "respond"
            current, state = sending, "speaking"
            # [cut] goes after the last word heard (on the first cell that sent any, if none was).
            said = sending.words
            last = said[-1].start if said else None
            cell.cut = sending.cut and (lo <= last < hi if said else not out or out[-1].action == "listen" or
                                        out[-1].state != "speaking")
        elif state == "silent" and any(lo <= p.start < hi for p in bcs):
            bc = next(p for p in bcs if lo <= p.start < hi)
            cell.action, cell.say = "backchannel", [w.text for w in bc.words]
        elif state == "speaking" and current is not None and current.cut:
            # Still talking until the pipeline stops it; under Grounding.stop "cell" the model stops it.
            stops = ground.stop == "cell" and t + ground.stop_latency >= current.end - 1e-6
            cell.action, state = ("stop", "cut") if stops else ("continue", state)
        elif state == "speaking" and current is not None:
            done = current.words[-1].start < lo
            cell.action, state = ("finish", "silent") if done else ("continue", state)
        else:
            cell.action = "listen"
        out.append(cell)
        t = round(t + ground.cell, 6)
    return out


def _uncut(p) -> list:
    """Global word times of what a cut item was going to say (empty if it wasn't cut)."""
    return [Word(w.text, p.start + w.start, p.start + w.end) for w in (p.uncut or [])]
