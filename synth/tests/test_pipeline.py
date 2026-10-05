import json
from pathlib import Path

import numpy as np
import pytest

from turnsynth import labels
from turnsynth.export import to_row, write_shard
from turnsynth.pipeline import process
from turnsynth.render import render
from turnsynth.script import ScriptError, check, parse
from turnsynth.tts import DummyTTS

EXAMPLES = sorted((Path(__file__).parent.parent / "examples" / "scripts").glob("*.json"))


def load(path: Path):
    return parse(json.loads(path.read_text()))


def test_canonical_map_matches_turnbench():
    gold = pytest.importorskip("turnbench.gold")
    # "Regular Turn" is a defensive alias in turnbench that occurs in no split.
    assert labels.CANONICAL == {k: v for k, v in gold.CANONICAL.items() if k != "Regular Turn"}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_pass_filter(path):
    assert check(load(path)) == []


def test_anchor_errors_are_reason_coded():
    obj = json.loads(EXAMPLES[0].read_text())
    bc = next(t for t in obj["turns"] if t["type"] == "backchannel")
    bc["after_word"] = 999
    with pytest.raises(ScriptError) as e:
        parse(obj)
    assert e.value.code == "anchor"


def test_stage_directions_rejected():
    obj = json.loads(EXAMPLES[0].read_text())
    obj["turns"][0]["text"] = "(laughs) Oh, you're back!"
    assert "stage_direction" in check(parse(obj))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_render_invariants(path):
    script = load(path)
    r = render(script, DummyTTS(), conversation_id="1", seed=3)
    assert len(r.audio[1]) == len(r.audio[2])
    # No speaker ever overlaps themselves.
    for spk in ("A", "B"):
        spans = sorted((p.start, p.end) for p in r.placed if not p.dropped and p.item.speaker == spk)
        assert all(e <= s2 for (_, e), (s2, _) in zip(spans, spans[1:]))
    # A realized barge-in leaves the host talking >= yield_min, then stops it.
    for p in r.placed:
        if p.item.type == "interruption" and p.item.floor_taking:
            host = next(h for h in r.placed if h.item.id == p.item.host)
            if host.cut:
                assert host.speech_end - p.start >= 1.0 - 1e-6
                assert host.speech_end < p.end


def _oracle(dataset):
    from turnbench.data import conversation, conversation_ids
    from turnbench.gold import events_for_conversation
    from turnbench.submission import Submission

    preds = []
    for cid in conversation_ids(dataset):
        ev = events_for_conversation(conversation(dataset, cid))
        spk = {}
        for s in (1, 2):
            spk[f"speaker_{s}"] = {
                "eot": sorted({a.time_s for a in ev.eot_positive_events if a.speaker == s}),
                "interruption": sorted({a.time_s for a in ev.int_positive_events if a.speaker == s}),
            }
        preds.append({"conversation_id": cid, **spk})
    return Submission.model_validate({"schema_version": 1, "predictions": preds})


def test_turnbench_scorer_reads_export(tmp_path):
    pytest.importorskip("turnbench.score")
    from turnbench.data import resolve_dataset
    from turnbench.score import score_submission

    rows = []
    for n, path in enumerate(EXAMPLES):
        res = process(load(path), DummyTTS(), conversation_id=str(900000 + n), seed=n)
        assert not res.reasons
        meta = {"conversation_type": res.rendered.script.conversation_type, "source": "test", "generator": "hand",
                "tts": "dummy", "speaker_1_voice": "", "speaker_2_voice": "", "speaker_1_gender": "",
                "speaker_2_gender": "", "annotator_a": "", "annotator_b": "", "annotator_c": "",
                "agreement": res.agreement}
        rows.append(to_row(res.rendered, res.tracks, meta))
    write_shard(rows, tmp_path / "synth-0.parquet")

    dataset = resolve_dataset(str(tmp_path))
    totals = score_submission(_oracle(dataset), dataset)
    assert totals.task_eot.tp > 10 and totals.task_int.tp >= 1
    assert totals.task_eot.recall == 1.0 and totals.task_eot.fp_rate == 0.0
    assert totals.task_int.recall == 1.0


def test_energy_bounds():
    from turnsynth.tts import energy_bounds

    sr = 24000
    audio = np.zeros(sr, np.float32)
    audio[int(0.2 * sr): int(0.6 * sr)] = 0.5
    lo, hi = energy_bounds(audio, sr)
    assert abs(lo - 0.2) < 0.011 and abs(hi - 0.6) < 0.011


def test_generate_with_fake_llm():
    from turnsynth.generate import generate_script
    from turnsynth.llm import FunctionLLM

    example = json.loads(EXAMPLES[1].read_text())  # casual

    def fake(system, user):
        if "Design one" in user:
            return "```json\n" + json.dumps({"scenario": example["scenario"], "speakers": example["speakers"]}) + "\n```"
        return "Here you go.\n```json\n" + json.dumps({"turns": example["turns"]}) + "\n```"

    script, log = generate_script(FunctionLLM(fake), "Casual", seed=1)
    assert script is not None and log[-1]["reasons"] == []
    assert len(script.items) == len(example["turns"])


def test_llm_judge_uses_model_labels_and_falls_back():
    from turnsynth import annotate
    from turnsynth.judge import LLMJudge
    from turnsynth.llm import FunctionLLM

    r = render(load(EXAMPLES[0]), DummyTTS(), conversation_id="1")
    segs = annotate.segment(r)
    text = {s: [" ".join(w for w, _ in annotate.segment_words(r, seg)) for seg in segs[s]] for s in (1, 2)}
    # The fake judge labels only even ids, and one with an invalid label.
    fake = FunctionLLM(lambda system, user: "```json\n" + json.dumps(
        {str(i): labels.BC_CONTINUER if i else "Not A Label" for i in range(0, 400, 2)}) + "\n```")
    tracks = LLMJudge(fake)(segs, text)
    flat = sorted((row for s in (1, 2) for row in tracks[s]), key=lambda row: row[0])
    assert flat[2][2] == labels.BC_CONTINUER
    assert flat[0][2] != "Not A Label"
    assert len(tracks[1]) == len(segs[1]) and len(tracks[2]) == len(segs[2])


class WholeTurnTTS(DummyTTS):
    """DummyTTS behind the whole-turn interface; records the context it was given."""

    whole_turn = True

    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice, speed=1.0, context=None):
        self.calls.append((text, context))
        return super().synthesize(text, voice, speed)


def test_whole_turn_backend_gets_context_and_retimed_pauses():
    from turnsynth.config import Timing
    from turnsynth.render import spoken_text, synthesize_item
    from turnsynth.script import Item

    item = Item(1, "A", "turn", "I think we should <pause 0.9> probably leave early. <pause 1.2> Unless it rains")
    text, cuts = spoken_text(item)
    assert text == "I think we should, probably leave early. Unless it rains" and cuts == [(4, 0.9), (7, 1.2)]

    timing = Timing()  # a scripted pause length is used as written
    sp = synthesize_item(WholeTurnTTS(), item, "v", timing, np.random.default_rng(0))
    assert [w.text for w in sp.words] == item.words
    for k, gap in cuts:
        assert abs(sp.words[k].start - sp.words[k - 1].end - gap) < 0.01

    tts = WholeTurnTTS()
    script = load(EXAMPLES[0])
    render(script, tts, conversation_id="1")
    assert len(tts.calls) == len(script.items)  # one call per item, in script order
    _, ctx = tts.calls[-1]
    assert len(ctx.history) == sum(it.speaker == script.items[-1].speaker for it in script.items) - 1
    assert ctx.partner is not None


def test_emotion_field_validated():
    obj = json.loads(EXAMPLES[0].read_text())
    obj["turns"][0]["emotion"] = {"happy": 0.4}
    assert parse(obj).items[0].emotion == {"happy": 0.4}
    obj["turns"][0]["emotion"] = {"giddy": 0.4}
    with pytest.raises(ScriptError):
        parse(obj)


def test_vad_times_at_22050():
    from turnsynth import vad

    sr = 22050
    audio = np.zeros(200 * sr, np.float32)
    audio[int(150.0 * sr): int(151.0 * sr)] = 0.5
    (s, e), = vad.segments(audio, sr)
    assert abs(s - 149.98) < 0.015 and abs(e - 151.02) < 0.015
