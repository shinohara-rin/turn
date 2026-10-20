import json
from pathlib import Path

import pytest

from turnsynth import tape as T
from turnsynth.duration import DurationTTS, syllables
from turnsynth.ground import Grounding, leaks
from turnsynth.pilot import EXAMPLE, Rules, Run, Scenario, grammar, parse_action, score
from turnsynth.render import render
from turnsynth.script import ScriptError, parse

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def interstellar():
    return parse(json.loads((EXAMPLES / "ground" / "interstellar.json").read_text()))


def test_duration_model():
    tts = DurationTTS()
    assert syllables("cat") == 1 and syllables("tonight") == 2 and syllables("make") == 1
    assert tts.word_seconds("tonight,") > tts.word_seconds("tonight")  # phrase-final lengthening
    assert tts.word_seconds("tonight", speed=2.0) == pytest.approx(tts.word_seconds("tonight") / 2)
    sp = tts.synthesize("so um, I want to watch something.", "duration_female_0")
    assert [w.text for w in sp.words][:2] == ["so", "um,"]
    assert all(a.end <= b.start for a, b in zip(sp.words, sp.words[1:]))


def test_parse_rejects_bad_grounding():
    raw = json.loads((EXAMPLES / "ground" / "interstellar.json").read_text())
    raw["turns"][2]["reveals"] = {"plot_wormhole": "words that are not in the line"}
    with pytest.raises(ScriptError):
        parse(raw)
    raw = json.loads((EXAMPLES / "ground" / "interstellar.json").read_text())
    raw["agent"] = "C"
    with pytest.raises(ScriptError):
        parse(raw)


def test_causality_moves_and_leaks():
    script = interstellar()
    g = Grounding()
    r = render(script, DurationTTS(), conversation_id="t", ground=g)
    moved = {m.item for m in r.moves}
    assert {3, 5} <= moved  # the agent's replies wait for its ASR and the next cell
    placed = {p.item.id: p for p in r.placed}
    last_user = placed[1].words[-1]
    assert placed[3].start >= g.boundary(last_user.end + g.asr_delay) + g.tts_lead - 1e-6
    codes = {(lk.item, lk.fact): lk.code for lk in leaks(g, r)}
    assert codes[(3, "u_two_hours")] == "never_said"
    # The hard cut stops the agent before it says "black hole"; under DuplexCascade's cell rule the user
    # talks over it instead. Either way the user can't know it.
    assert codes[(7, "plot_blackhole")] == "cut_off"
    rc = render(script, DurationTTS(), conversation_id="t", ground=Grounding(stop="cell"))
    assert {(lk.item, lk.fact): lk.code for lk in leaks(Grounding(stop="cell"), rc)}[(7, "plot_blackhole")] == "spoken_over"


def test_agent_hard_cut_is_immediate():
    g = Grounding()
    r = render(interstellar(), DurationTTS(), conversation_id="t", ground=g)
    placed = {p.item.id: p for p in r.placed}
    assert placed[3].cut and placed[3].end == pytest.approx(placed[4].start + g.vad_latency, abs=1e-3)


@pytest.mark.parametrize("stop", ["hard_cut", "cell"])
def test_tape_shows_only_heard_words(stop):
    g = Grounding(stop=stop)
    r = render(interstellar(), DurationTTS(), conversation_id="t", ground=g)
    cells = T.cells(r, g)
    agent = [w for c in cells for w in c.say if c.action != "backchannel"]
    heard = [w.text for p in r.placed if p.item.speaker == "B" and p.item.is_floor for w in p.words]
    assert agent == heard
    assert sum(c.cut for c in cells) == sum(p.cut for p in r.placed)
    for c in cells:
        assert c.action in T.ACTIONS[c.state]
    if stop == "cell":
        assert [c.action for c in cells].count("stop") == 2


def test_micro_turn_controls():
    cell = T.Cell(1.28, ["Inception."], False, "silent", "respond", ["Both", "are"])
    user, model = T.micro_turn(cell)
    assert user == "<|turn>user\nInception.<turn|>\n"
    assert model == "<|turn>model\n<|user finish talking|> Both are<turn|>\n"
    assert T.micro_turn(T.Cell(1.92, [], True))[0] == "<|turn>user\n<|voice|><turn|>\n"


def test_pilot_rules_example():
    run = Run(EXAMPLE, Rules(), Grounding(), DurationTTS()).run()
    s = score(run)
    assert not s["cut_in"] and s["answered"]
    assert [x["action"] for x in s["lines"]] == ["resume", "respond"]
    assert all(x["ok"] for x in s["lines"])
    tape = "\n".join(run.tape())
    assert "[cut]" in tape and 'backchannel "mhm"' in tape
    # Words cut before they were heard are not on the tape.
    unsaid = [w.text for w in run.words if not w.said]
    assert unsaid and sorted(unsaid) == sorted(w for c in run.cells for w in c.unsaid)


def test_pilot_scenarios_with_rules():
    scenarios = [Scenario.parse(d) for d in json.loads((EXAMPLES / "pilot" / "scenarios.json").read_text())]
    assert len(scenarios) == 24
    scores = [score(Run(sc, Rules(), Grounding(), DurationTTS()).run()) for sc in scenarios]
    pause = [s for s in scores if s["probe"] == "pause"]
    assert not any(s["cut_in"] for s in pause)
    for s in scores:
        assert "lines_missing" not in s, s["id"]
    # The rules baseline takes the floor on punctuation, so it never answers an unpunctuated question.
    assert not next(s for s in scores if s["id"] == "turn-latte-nopunct")["answered"]


def test_grammar_and_parse():
    g = grammar(["listen", "respond", "backchannel"], 3)
    assert g.startswith('root ::= " listen" | " respond " say | " backchannel " bc')
    assert "{0,2}" in g
    d = parse_action(' respond "It is, from"')
    assert d.action == "respond" and d.words == ["It", "is,", "from"]
    assert parse_action(" finish").words == []


def test_tape_without_periods():
    assert [T.no_periods(w) for w in ["if.", "U.S.", "3.5", "...", "now?"]] == ["if", "US", "3.5", "", "now?"]
    cell = T.Cell(1.28, ["is", "it", "open."], False, "silent", "listen")
    assert T.line(cell, periods=False) == '  1.28 | user: "is it open" (silent) | agent (silent): listen'
    assert T.line(T.Cell(1.92, ["..."], True), periods=False).startswith("  1.92 | user: (talking)")
