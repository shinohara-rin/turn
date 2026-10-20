"""Closed-loop pilot: an untrained LLM as the cascade agent's turn-taking layer, on the cell tape.

Every `cell` the agent model gets the tape so far (tape.py) and writes one action under a grammar that
allows only what its pipeline state permits (tape.ACTIONS) and only as many words as its TTS queue has
room for (the pacing grant). Everything else is the agent's pipeline, as in ground.py: its ASR commits
a user word `asr_delay` after the word ends, its VAD reports voice, its words are timed by the duration
model and heard `tts_lead` after it writes them, and user voice cuts its audio (`stop` "hard_cut",
Frankenvoice) or the model has to stop itself (`stop` "cell", DuplexCascade). Words it never got to
say are taken off the tape.

The user is a person playing a scenario: an opening turn, then optional lines (a backchannel, a
barge-in) timed off the agent's audio as they hear it, a reaction time after the word that sets them.

The report is FDB-style turn-taking (Full-Duplex-Bench, arXiv 2503.04721), from the realized timeline:
whether the agent took the floor at the right moments, and how fast.

    python -m turnsynth.pilot examples/pilot/scenarios.json --agent llama --url http://127.0.0.1:8090 --out out/pilot
    python -m turnsynth.pilot examples/pilot/scenarios.json --agent rules --out out/pilot-rules
"""

import argparse
import json
import re
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from turnsynth import lang as L
from turnsynth import tape as T
from turnsynth.duration import DurationTTS
from turnsynth.ground import Grounding
from turnsynth.script import PAUSE_RE
from turnsynth.tts import Word

AGENT_BACKCHANNELS = ("mhm", "yeah", "right", "okay", "uh-huh", "I see")
USER_BACKCHANNELS = {"mhm", "mm", "mmhm", "yeah", "yep", "yes", "okay", "ok", "right", "uh-huh", "uhhuh", "sure", "i", "see", "got", "it"}
WORD_S = 0.3  # a word's average length with its pause, for the pacing grant
MAX_WORDS = 6


@dataclass
class Line:
    kind: str  # backchannel | interruption
    text: str
    after_s: float = 1.5  # said once the agent's current answer has been audible this long


@dataclass
class Scenario:
    id: str
    probe: str  # turn_end | pause | backchannel | interruption
    user: str  # the opening turn, with <pause X> markers
    lines: list[Line] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)  # what the rules agent says

    @staticmethod
    def parse(d: dict) -> "Scenario":
        return Scenario(d["id"], d["probe"], d["user"], [Line(**x) for x in d.get("lines", [])], d.get("replies", []))


@dataclass
class AgentWord:
    text: str
    start: float
    end: float
    cell: int  # index of the cell that sent it
    bc: bool = False
    said: bool = True


@dataclass
class Decision:
    action: str
    words: list[str]
    ms: float = 0.0  # model time
    forced: bool = False  # no call: the grammar left one choice


class Run:
    """One scenario in virtual time."""

    def __init__(self, sc: Scenario, agent, ground: Grounding, tts: DurationTTS, limit: float = 40.0):
        self.sc, self.agent, self.g, self.tts, self.limit = sc, agent, ground, tts, limit
        self.cells: list[T.Cell] = []
        self.decisions: list[Decision] = []
        self.words: list[AgentWord] = []
        self.user: list[Word] = []
        self.lines: list[dict] = []  # placed user lines: {kind, text, onset, end, words}
        self.state = "silent"
        self.queue_end = 0.0
        self.turn_start: float | None = None  # first audio of the agent's current answer
        self.cut_at: float | None = None
        opening = self._say(sc.user, 0.6)
        self.opening = {"kind": "opening", "text": sc.user, "onset": opening[0].start, "end": opening[-1].end,
                        "chunks": self._chunk_ends(sc.user, opening)}

    # --- the user -------------------------------------------------------------------------------
    def _say(self, text: str, start: float) -> list[Word]:
        parts, t, out = PAUSE_RE.split(text), start, []
        for i in range(0, len(parts), 2):
            chunk = " ".join(parts[i].split())
            if chunk:
                for w in chunk.split():
                    d = self.tts.word_seconds(w)
                    out.append(Word(w, t, t + d))
                    t += d + (self.tts.sentence_pause if w[-1] in L.SENTENCE_END else
                              self.tts.clause_pause if w[-1] in L.CLAUSE_END else 0.0)
            if i + 1 < len(parts):
                t = (out[-1].end if out else t) + float(parts[i + 1])
        self.user += out
        self.user.sort(key=lambda w: w.start)
        return out

    @staticmethod
    def _chunk_ends(text: str, words: list[Word]) -> list[tuple[float, float]]:
        """(end of a chunk's last word, start of the next chunk) for each scripted pause."""
        parts, n, out = PAUSE_RE.split(text), 0, []
        for i in range(0, len(parts) - 1, 2):
            n += len(parts[i].split())
            if 0 < n < len(words):
                out.append((words[n - 1].end, words[n].start))
        return out

    def _voice(self) -> list[tuple[float, float]]:
        """The user's voiced regions (words closer than 0.25 s merge)."""
        out = []
        for w in self.user:
            if out and w.start - out[-1][1] < 0.25:
                out[-1] = (out[-1][0], max(out[-1][1], w.end))
            else:
                out.append((w.start, w.end))
        return out

    def _place_lines(self) -> None:
        """Place the next user line once the agent's answer has been audible for `after_s`."""
        k = len(self.lines)
        if k >= len(self.sc.lines) or self.turn_start is None or (self.lines and self.turn_start < self.lines[-1]["end"]):
            return  # each line waits for the agent to (re)start talking after the previous one
        line = self.sc.lines[k]
        voice = [(s + self.g.vad_latency, e + 0.1) for s, e in self._voice()]
        heard = [w for w in self.words if not w.bc and w.said and w.start >= self.turn_start
                 and not any(w.start < e and w.end > s for s, e in voice)]
        due = self.turn_start + line.after_s
        onset = None
        if line.kind == "backchannel":
            # At the end of a phrase, or a little later if the agent doesn't end one.
            w = next((w for w in heard if w.end >= due and w.text[-1] in ",.;:?!"), None)
            w = w or next((w for w in heard if w.end >= due + 2.0), None)
            onset = w.end + 0.15 if w else None
        else:
            w = next((w for w in heard if w.end >= due), None)
            onset = self.g.reaction(False, w) if w else None
        if onset is None:
            return
        words = self._say(line.text, max(onset, self.user[-1].end + 0.05 if self.user else onset))
        self.lines.append({"kind": line.kind, "text": line.text, "onset": words[0].start, "end": words[-1].end,
                           "after_word": w.text})

    # --- the agent's pipeline -------------------------------------------------------------------
    def _schedule(self, words: list[str], at: float, cell: int, bc: bool = False) -> float:
        t = at
        for w in words:
            d = self.tts.word_seconds(w)
            self.words.append(AgentWord(w, t, t + d, cell, bc))
            t += d + (self.tts.sentence_pause if w[-1] in L.SENTENCE_END else
                      self.tts.clause_pause if w[-1] in L.CLAUSE_END else 0.0)
        return t

    def _cut(self, c: float) -> None:
        """Stop the agent's voice at c: words not yet mostly said are unsaid, and leave the tape."""
        last = None
        for w in self.words:
            if not w.bc and w.said and w.start + self.g.recognize_at * (w.end - w.start) > c:
                w.said = False
                cell = self.cells[w.cell]
                del cell.say[len(cell.say) - 1 - cell.say[::-1].index(w.text)]  # unsaid words end the cell's
                cell.unsaid.append(w.text)
            elif not w.bc and w.said and w.start < c:
                last = w
        sent = [w for w in self.words if not w.bc and w.start >= (self.turn_start or 0) - 1e-9]
        if sent:
            # [cut] goes on the cell of the last word heard, or of the first unsaid one if none was.
            k = last.cell if last is not None and last.start >= (self.turn_start or 0) - 1e-9 else sent[0].cell
            self.cells[k].cut = True
        self.queue_end = min(self.queue_end, c)
        self.state, self.cut_at = "cut", c

    def _hard_cut(self, t: float) -> None:
        """User voice the VAD has reported by t stops the agent's answer (not its backchannels)."""
        if self.g.stop != "hard_cut" or self.state == "cut" or self.turn_start is None:
            return
        answer = [w for w in self.words if not w.bc and w.said and w.start >= self.turn_start - 1e-9]
        for s, e in self._voice():
            v0, v1 = s + self.g.vad_latency, e + 0.1
            if v0 > t:
                break
            if v1 < self.turn_start:
                continue
            hit = [w for w in answer if w.end > v0 and w.start < v1]
            if hit:
                self._cut(max(v0, hit[0].start))
                return
            if self.state == "speaking" and v0 >= self.turn_start:
                self._cut(v0)  # the user took the floor while the open answer had nothing playing
                return

    def grant(self, t: float) -> int:
        start = max(t + self.g.tts_lead, self.queue_end)
        budget = t + self.g.tts_lead + 2 * self.g.cell - start
        return max(0, min(MAX_WORDS, int(budget / WORD_S + 0.5)))

    def step(self, t: float) -> None:
        g = self.g
        self._hard_cut(t)
        cell = T.Cell(t, [w.text for w in self.user if t - g.cell < w.end + g.asr_delay <= t + 1e-9])
        v0, v1 = t - g.cell - g.vad_latency, t - g.vad_latency
        cell.voice = any(s < v1 and e > v0 for s, e in self._voice())
        cell.state = self.state
        self.cells.append(cell)
        allowed = [a for a in T.ACTIONS[self.state] if a != "stop" or g.stop == "cell"]
        n = self.grant(t)
        if self.state == "speaking" and n == 0 and (g.stop == "hard_cut" or not cell.voice):
            d = Decision("continue", [], forced=True)
        else:
            t0 = time.time()
            d = self.agent.decide(self, allowed, max(n, 1))
            d.ms = (time.time() - t0) * 1000
        self.decisions.append(d)
        cell.action, cell.say = d.action, list(d.words)
        k = len(self.cells) - 1
        at = max(t + g.tts_lead, self.queue_end)
        if d.action == "backchannel":
            self._schedule(d.words, t + g.tts_lead, k, bc=True)
        elif d.action == "stop":
            self._cut(t + g.stop_latency)
        elif d.action in ("respond", "resume", "continue", "finish") and (d.words or d.action == "finish"):
            if d.action in ("respond", "resume"):
                at = t + g.tts_lead if self.state == "cut" else at
                self.turn_start, self.cut_at = at, None
            self.queue_end = self._schedule(d.words, at, k)
            self.state = "silent" if d.action == "finish" else "speaking"
        self._place_lines()

    def done(self, t: float) -> bool:
        g = self.g
        if t >= self.limit:
            return True
        if len(self.lines) < len(self.sc.lines):
            started = self.turn_start is not None
            return t > (self.opening["end"] + 8.0 if not started else self.turn_start + 12.0)
        end = max([self.opening["end"]] + [x["end"] for x in self.lines])
        answered = [w for w in self.words if not w.bc and w.said and w.start > end]
        talked = sum(w.end - w.start for w in answered)
        return (talked >= 1.5 and self.state != "cut") or t > end + 7.0 or (self.state == "silent" and answered and t > end + g.cell)

    def run(self) -> "Run":
        t = self.g.cell
        while not self.done(t):
            self.step(round(t, 6))
            t += self.g.cell
        return self

    def tape(self, until: int | None = None) -> list[str]:
        return [T.line(c) for c in self.cells[:until]]


# --- agents -----------------------------------------------------------------------------------

SYSTEM = """You are the turn-taking layer of a voice assistant on a live call. You hear the user through a speech recognizer and talk through a text-to-speech voice. Time runs in cells of 0.64 seconds, and you act once per cell.

Each line of the tape is one cell:
  <time> | user: "<words>" (talking or silent) | agent (<state>): <action>
- user: the words the speech recognizer confirmed in that cell (they arrive about half a second after the user says them), and whether the user was making sound.
- agent state: silent (you don't have the floor), speaking (you are partway through an answer), cut off (the user started talking over you and your voice was stopped).
- Your words are spoken aloud 0.4 s after you write them, a few words per cell, so carry your sentence on in the next cell.
- [cut] marks where your voice was stopped. Words you wrote after it were never heard, so they are not on the tape.

Actions:
  listen               stay quiet: the user is talking, pausing mid-thought, or you are waiting
  respond "<words>"    start your answer to what the user just said
  backchannel "mhm"    a short acknowledgement while the user keeps the floor
  continue "<words>"   the next words of your answer
  finish "<words>"     the last words of your answer, or just finish when you have said them
  resume "<words>"     go on with an answer the user cut off only to acknowledge you (mhm, yeah, okay)

Take turns like a person:
- Respond once the user has finished a thought: a question or a complete sentence, then silence. Don't start while they are talking, or while they pause in the middle of a sentence (after "and", "um", "the", a comma). Wait for them.
- Once they are done, respond promptly, usually in the first silent cell after their last words.
- If a brief acknowledgement cut you off, resume from where you were cut. If the user is taking the floor (a question, a correction, "wait"), listen until they finish, then respond to what they said.
- Talk like a person on the phone: one or two short sentences, no lists."""

EXAMPLE = Scenario("example", "example",
                   "hey, do you know if the pharmacy on Main Street is, um, <pause 2.4> open on Sundays?",
                   [Line("backchannel", "yeah", 1.0), Line("interruption", "oh wait, I meant the one on Oak Street.", 1.2)],
                   ["It is, from ten to four, but it closes early on holidays.",
                    "Ah, the Oak Street one is closed on Sundays."])


class Rules:
    """A punctuation-and-silence baseline (and the writer of the prompt's example tape).

    Takes the floor on the first silent cell after sentence-final punctuation, backchannels in the
    second silent cell of a mid-sentence pause, resumes after a cut if all the user said since was a
    backchannel, and says the scenario's `replies` in order.
    """

    def __init__(self):
        self.reply, self.pos, self.base, self.bc_at = -1, 0, 0, -1

    def plan(self, run: Run) -> list[str]:
        r = run.sc.replies or ["Sure, so the short answer is that it depends, but I can walk you through it."]
        return r[min(self.reply, len(r) - 1)].split()

    def decide(self, run: Run, allowed: list[str], n: int) -> Decision:
        cells, t = run.cells, run.cells[-1].t
        heard = [w for c in cells for w in c.user]
        quiet = 0
        for c in reversed(cells):
            if c.voice or c.user:
                break
            quiet += 1
        done = bool(heard) and heard[-1][-1] in ".?!" and not cells[-1].voice
        if run.state == "speaking":
            plan = self.plan(run)
            words = plan[self.pos:self.pos + n]
            self.pos += len(words)
            return Decision("finish" if self.pos >= len(plan) else "continue", words)
        if run.state == "cut":
            since = [w.text for w in run.user if w.start >= run.cut_at - run.g.vad_latency - 0.05
                     and w.end + run.g.asr_delay <= t + 1e-9]
            if since and all(L.bare(w, "en") in USER_BACKCHANNELS for w in since):
                if cells[-1].voice:
                    return Decision("listen", [])
                # Pick the reply up from the first word that wasn't heard.
                self.base = self.pos = self.base + sum(1 for w in run.words if not w.bc and w.said
                                                       and w.start >= run.turn_start - 1e-9)
                words = self.plan(run)[self.pos:self.pos + n]
                self.pos += len(words)
                return Decision("resume", words)
            return self._respond(run, n) if done and since else Decision("listen", [])
        if done:
            return self._respond(run, n)
        last_user = max((w.end for w in run.user if w.end + run.g.asr_delay <= t + 1e-9), default=None)
        holds = last_user is not None and not any(not w.bc and w.start > last_user for w in run.words)
        if holds and quiet == 2 and self.bc_at < len(heard):
            self.bc_at = len(heard)
            return Decision("backchannel", ["mhm"])
        return Decision("listen", [])

    def _respond(self, run: Run, n: int) -> Decision:
        self.reply, self.base = self.reply + 1, 0
        words = self.plan(run)[:n]
        self.pos = len(words)
        return Decision("respond", words)


def grammar(allowed: list[str], n: int) -> str:
    """GBNF for one cell's action: the state's actions, at most n words."""
    alts = []
    for a in allowed:
        if a == "listen":
            alts.append('" listen"')
        elif a == "backchannel":
            alts.append('" backchannel " bc')
        elif a == "finish":
            alts.append('" finish" (" " say)?')
        elif a == "stop":
            alts.append('" stop"')
        else:
            alts.append(f'" {a} " say')
    rep = f"{{0,{n - 1}}}" if n > 1 else "{0,0}"
    return "\n".join([
        "root ::= " + " | ".join(alts),
        f'say ::= "\\"" word (" " word){rep} "\\""',
        "word ::= [A-Za-z0-9$%&'] [A-Za-z0-9$%&'-]{0,18} [,.?!;:]?",
        "bc ::= " + " | ".join(f'"\\"{b}\\""' for b in AGENT_BACKCHANNELS),
    ])


def example_tape() -> str:
    run = Run(EXAMPLE, Rules(), Grounding(), DurationTTS()).run()
    return "\n".join(run.tape())


def parse_action(text: str) -> Decision:
    m = re.match(r'\s*(\w+)(?:\s+"([^"]*)")?', text)
    if not m:
        raise ValueError(f"unparsable action {text!r}")
    return Decision(m.group(1), (m.group(2) or "").split())


class LlamaCpp:
    """A llama.cpp server (/completion), the tape continued in the model's turn under a GBNF grammar.

    The prompt is Gemma 4's chat format: instructions as the system turn, the example tape as the user
    turn, and the live tape as the model's own turn with thinking off. Each cell's prompt extends the
    last one, so the server's prompt cache only processes the new line (a cut re-processes from the
    first line it edits).
    """

    def __init__(self, url: str, temperature: float = 0.0, example: str | None = None):
        self.url, self.temperature = url.rstrip("/"), temperature
        ex = example if example is not None else example_tape()
        self.head = (f"<|turn>system\n{SYSTEM}<turn|>\n<|turn>user\nAn example tape from another call:\n\n{ex}\n\n"
                     "Now the live call. Continue the tape with the agent's action on each new line.<turn|>\n"
                     "<|turn>model\n<|channel>thought\n<channel|>")
        self.calls = []

    def prompt(self, run: Run) -> str:
        return self.head + "".join(l + "\n" for l in run.tape(-1)) + T.prefix(run.cells[-1])

    def decide(self, run: Run, allowed: list[str], n: int) -> Decision:
        body = {"prompt": self.prompt(run), "grammar": grammar(allowed, n), "n_predict": 64,
                "temperature": self.temperature, "cache_prompt": True, "stop": ["\n"]}
        req = urllib.request.Request(self.url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        self.calls.append(r.get("timings", {}))
        d = parse_action(r["content"])
        if d.action not in allowed:
            raise ValueError(f"grammar let through {d.action!r}")
        return d


# --- scoring ----------------------------------------------------------------------------------

def _first(run: Run, after: float, actions: tuple[str, ...]):
    for c in run.cells:
        if c.t > after + 1e-9 and c.action in actions:
            return c
    return None


def _audio_after(run: Run, t: float) -> float | None:
    w = next((w for w in run.words if not w.bc and w.said and w.start >= t - 1e-9), None)
    return w.start if w else None


def score(run: Run) -> dict:
    """Turn-taking outcomes of one run, FDB-style."""
    op, sc = run.opening, run.sc
    out = {"id": sc.id, "probe": sc.probe}
    first = _first(run, 0.0, ("respond",))
    # Taking the floor before the user finished: in a pause that's FDB's pause-handling takeover.
    out["cut_in"] = bool(first and first.t < op["end"])
    reply = _first(run, op["end"], ("respond",))
    out["answered"] = reply is not None and reply.t <= op["end"] + 7.0
    a = _audio_after(run, reply.t + run.g.tts_lead - 1e-6) if reply else None
    out["latency"] = round(a - op["end"], 2) if a is not None and out["answered"] else None
    out["backchannels"] = sum(c.action == "backchannel" for c in run.cells)
    out["backchannels_in_pauses"] = sum(any(s - 0.1 <= c.t - run.g.tts_lead <= e for s, e in op["chunks"]) and c.action == "backchannel"
                                        for c in run.cells)
    for k, line in enumerate(run.lines):
        key = line["kind"]
        nxt = run.lines[k + 1]["onset"] if k + 1 < len(run.lines) else float("inf")
        act = next((c for c in run.cells if line["onset"] < c.t < nxt and c.action in ("resume", "respond", "stop")), None)
        res = {"onset": round(line["onset"], 2), "after_word": line["after_word"], "action": act.action if act else None}
        if key == "backchannel":
            res["ok"] = act is not None and act.action == "resume"
        else:
            res["ok"] = act is not None and act.action == "respond"
            res["early"] = bool(act and act.t < line["end"])
        a = _audio_after(run, act.t + run.g.tts_lead - 1e-6) if act else None
        res["latency"] = round(a - line["end"], 2) if a is not None else None
        out.setdefault("lines", []).append(res)
    if len(run.lines) < len(sc.lines):
        out["lines_missing"] = len(sc.lines) - len(run.lines)
    calls = [d.ms for d in run.decisions if not d.forced]
    out["calls"], out["model_ms_mean"] = len(calls), round(statistics.mean(calls)) if calls else 0
    return out


def summary(scores: list[dict]) -> dict:
    def rate(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 2) if xs else None

    by = lambda p: [s for s in scores if s["probe"] == p]  # noqa: E731
    pause, turn = by("pause"), [s for s in scores if s["probe"] != "pause"]
    bc = [l for s in by("backchannel") for l in s.get("lines", [])]
    intr = [l for s in by("interruption") for l in s.get("lines", [])]
    return {
        "pause: took the floor mid-utterance (lower is better)": rate([s["cut_in"] for s in pause]),
        "turn end: cut in before the user finished (lower)": rate([s["cut_in"] for s in turn]),
        "turn end: answered (higher)": rate([s["answered"] for s in scores]),
        "turn end: latency median s": med([s["latency"] for s in scores]),
        "user backchannel: resumed (higher)": rate([l["ok"] for l in bc]),
        "user backchannel: resume latency median s": med([l["latency"] for l in bc if l["ok"]]),
        "interruption: answered it (higher)": rate([l["ok"] for l in intr]),
        "interruption: latency median s": med([l["latency"] for l in intr if l["ok"]]),
        "agent backchannels per scenario": round(statistics.mean(s["backchannels"] for s in scores), 2) if scores else None,
        "lines never placed": sum(s.get("lines_missing", 0) for s in scores),
        "model calls": sum(s["calls"] for s in scores),
        "model ms mean": round(statistics.mean(s["model_ms_mean"] for s in scores if s["calls"])) if scores else 0,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios")
    ap.add_argument("--agent", choices=["rules", "llama"], default="rules")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--stop", choices=["hard_cut", "cell"], default="hard_cut")
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    scenarios = [Scenario.parse(d) for d in json.loads(Path(args.scenarios).read_text())]
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.id in keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ground = Grounding(stop=args.stop)
    scores = []
    for sc in scenarios:
        agent = Rules() if args.agent == "rules" else LlamaCpp(args.url)
        run = Run(sc, agent, ground, DurationTTS()).run()
        s = score(run)
        scores.append(s)
        (out / f"{sc.id}.tape.txt").write_text("\n".join(run.tape()) + "\n")
        print(json.dumps(s), flush=True)
    (out / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    summ = summary(scores)
    (out / "summary.json").write_text(json.dumps({"agent": args.agent, "stop": args.stop, "ground": asdict(ground), **summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
