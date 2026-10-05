"""Script -> rendered audio -> three annotator tracks -> TurnBench row."""

from dataclasses import dataclass

from turnsynth import annotate
from turnsynth.judge import LLMJudge, Transcriber, rule_judge
from turnsynth.render import Rendered, render
from turnsynth.script import Script
from turnsynth.tts import TTS


@dataclass
class Result:
    rendered: Rendered
    tracks: dict[str, dict[int, annotate.Track]]
    agreement: float
    wer: float | None
    reasons: list[str]


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm(text: str) -> list[str]:
    return ["".join(c for c in w.lower() if c.isalnum()) for w in text.split() if any(c.isalnum() for c in w)]


def process(script: Script, tts: TTS, *, conversation_id: str, seed: int = 0,
            transcriber: Transcriber | None = None, judge: LLMJudge | None = None,
            bleed_db: float | None = None, min_agreement: float = 0.6, max_wer: float = 0.35) -> Result:
    rendered = render(script, tts, conversation_id=conversation_id, seed=seed, bleed_db=bleed_db)
    return annotate_rendered(rendered, transcriber=transcriber, judge=judge, min_agreement=min_agreement,
                             max_wer=max_wer)


def annotate_rendered(rendered: Rendered, *, transcriber: Transcriber | None = None, judge: LLMJudge | None = None,
                      min_agreement: float = 0.6, max_wer: float = 0.35) -> Result:
    segs = annotate.segment(rendered)
    script_text = {s: [" ".join(w for w, _ in annotate.segment_words(rendered, seg)) for seg in segs[s]] for s in (1, 2)}
    if transcriber is not None:
        asr_text = {s: transcriber.transcribe(rendered.audio[s], rendered.sample_rate, segs[s]) for s in (1, 2)}
        ref = [w for s in (1, 2) for t in script_text[s] for w in _norm(t)]
        hyp = [w for s in (1, 2) for t in asr_text[s] for w in _norm(t)]
        wer = _edit_distance(ref, hyp) / max(len(ref), 1)
    else:
        # No ASR: the judge reads the script words that fall in each segment,
        # i.e. a perfect-ASR transcript. Fine for smoke tests, not for data.
        asr_text, wer = script_text, None
    tracks = {
        "a": annotate.intent_track(rendered, segs),
        "b": (judge or rule_judge)(segs, asr_text),
        "c": annotate.geometry_track(segs, script_text),
    }
    agreement = annotate.agreement([tracks["a"], tracks["b"], tracks["c"]])
    reasons = []
    if agreement < min_agreement:
        reasons.append("low_agreement")
    if wer is not None and wer > max_wer:
        reasons.append("high_wer")
    if not segs[1] or not segs[2]:
        reasons.append("silent_channel")
    return Result(rendered, tracks, agreement, wer, reasons)
