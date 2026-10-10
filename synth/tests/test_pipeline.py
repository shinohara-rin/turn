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
    bc["after"] = "words the host never says"
    with pytest.raises(ScriptError) as e:
        parse(obj)
    assert e.value.code == "anchor"
    del bc["after"]
    bc["after_word"] = 999
    with pytest.raises(ScriptError) as e:
        parse(obj)
    assert e.value.code == "anchor"


def test_after_quote_resolves_to_word_count():
    from turnsynth.script import resolve_after

    words = "So we get there on Friday, right, and the campsite".split()
    assert resolve_after(words, "on friday, Right") == 7
    assert resolve_after(["我", "帮", "你", "看看？", "我"], "帮 你 看看", "zh") == 4
    assert resolve_after(words, "not there") is None


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

    example = json.loads(next(p for p in EXAMPLES if p.stem == "casual_weekend").read_text())

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

    def synthesize(self, text, voice, speed=1.0, context=None, language="en"):
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


def test_speaker_pass_cuts_items_back_apart():
    class PassTTS(WholeTurnTTS):
        speaker_pass = True
        pass_words = 60

    tts = PassTTS()
    script = load(EXAMPLES[0])
    r = render(script, tts, conversation_id="1")
    words = {s: sum(len(it.words) for it in script.items if it.speaker == s) for s in "AB"}
    assert len(tts.calls) < len(script.items)
    assert sum(len(t.split()) for t, _ in tts.calls) == words["A"] + words["B"]
    for p in r.placed:
        assert [w.text for w in p.speech.words][: len(p.speech.words)] == p.item.words[: len(p.speech.words)]
        if not p.cut:
            assert len(p.speech.words) == len(p.item.words)


def test_voice_clip_skips_utterances_too_long_to_fit():
    import io

    import soundfile as sf

    from turnsynth.voicebank import _join

    def utt(seconds):
        buf = io.BytesIO()
        sf.write(buf, 0.3 * np.sin(np.arange(int(seconds * 16000)) * 0.05).astype(np.float32), 16000, format="WAV")
        return {"audio": {"bytes": buf.getvalue()}, "duration": seconds}

    clip, sr = _join([utt(14.0), utt(5.0), utt(3.0)], max_s=10.0)
    assert 7.5 <= len(clip) / sr <= 10.0


def test_trim_keeps_the_decay_after_the_aligned_word_end():
    from turnsynth.render import _trim
    from turnsynth.tts import Speech, Word

    sr = 22050
    t = np.arange(int(1.5 * sr)) / sr
    env = np.where(t < 0.2, 0.0, np.where(t < 0.8, 1.0, np.exp(-(t - 0.8) / 0.05)))
    audio = (0.5 * env * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    # The aligner ends the word at 0.8 s, while the voice dies away over ~0.25 s.
    out = _trim(Speech(audio, sr, [Word("so.", 0.25, 0.8)]))
    assert out.duration > 0.6 + 0.15
    assert abs(out.audio[-int(0.005 * sr):]).max() < 1e-3
    assert out.words[0].end == pytest.approx(0.8 - (0.25 - out.words[0].start), abs=1e-6)


def test_cjk_scripts_are_segmented_and_spoken_without_spaces():
    from turnsynth import lang
    from turnsynth.render import spoken_text

    for name in ("casual_kyoto_ja.json", "casual_chengdu_zh.json"):
        script = parse(json.loads((Path(__file__).parent.parent / "examples" / "scripts" / name).read_text()))
        assert script.language in lang.CJK
        item = next(it for it in script.items if it.type == "turn" and len(it.chunks) > 1)
        text, cuts = spoken_text(item, script.language)
        assert len(text.split()) == len(item.words) and cuts[0][0] < len(item.words)
        said = lang.spoken(text.split(), script.language)
        assert " " not in said and lang.COMMA[script.language] in said
    unsegmented = json.loads((Path(__file__).parent.parent / "examples" / "scripts" / "casual_chengdu_zh.json").read_text())
    unsegmented["turns"] = [dict(t, text=t["text"].replace(" ", "")) for t in unsegmented["turns"] if t["type"] == "turn"]
    assert "unsegmented" in check(parse(unsegmented))


def test_cjk_error_rate_counts_characters_and_rule_judge_knows_aizuchi():
    from turnsynth import lang
    from turnsynth.judge import _lexical

    assert lang.units("今天 ， 下雨 。", "zh") == list("今天下雨")
    assert lang.spoken(["買っ", "た", "iPhone", "15", "は"], "ja") == "買ったiPhone 15は"
    assert _lexical("うんうん。", "ja")[2] is not None
    assert _lexical("对对对", "zh")[2] is not None
    assert _lexical("我觉得不是", "zh")[2] is None


def test_standalone_punctuation_takes_no_time_in_alignment():
    from turnsynth.align import _fill
    from turnsynth.tts import Word

    words = ["你", "回来", "了", "！"]
    out = _fill(words, [Word("你", 0.1, 0.3), Word("回来", 0.4, 0.8), Word("了", 0.9, 1.0), None], 2.0)
    assert out[-1].start == out[-1].end == 1.0


def test_speed_and_emotion_reach_the_tts_and_split_passes():
    class PassTTS(WholeTurnTTS):
        speaker_pass = True
        pass_words = 60

        def synthesize(self, text, voice, speed=1.0, context=None, language="en"):
            self.speeds.append(speed)
            return super().synthesize(text, voice, speed, context, language)

    obj = json.loads(next(p for p in EXAMPLES if p.stem == "casual_weekend").read_text())
    script = parse(obj)
    assert script.speakers["A"]["pace"] == 1.1 and script.items[0].speed == 1.1
    tts = PassTTS()
    tts.speeds = []
    render(script, tts, conversation_id="1")
    assert len(set(round(s, 2) for s in tts.speeds)) > 3  # pace varies from pass to pass
    assert all(ctx.emotion for _, ctx in tts.calls)
    plain = parse({**obj, "turns": [{k: v for k, v in t.items() if k not in ("speed", "emotion")} for t in obj["turns"]]})
    flat = PassTTS()
    flat.speeds = []
    render(plain, flat, conversation_id="1")
    assert len(flat.calls) < len(tts.calls)  # delivery changes end a pass

    obj["turns"][0]["speed"] = 2.0
    with pytest.raises(ScriptError):
        parse(obj)
