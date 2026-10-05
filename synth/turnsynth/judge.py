"""Annotator b: a post-hoc, transcription-based judge.

The judge never sees the script. It gets each channel transcribed by ASR,
segment by segment, and labels every VAD segment from the interleaved, timed
transcript, the way a human annotator would from listening. This catches what
the generator's own labels can't: TTS that swallowed a word, a "backchannel"
long enough to read as a turn, an interruption that landed at a transition
point.
"""

import json
from dataclasses import dataclass

import numpy as np

from turnsynth import labels as L
from turnsynth.annotate import Segment, Track, overlap_class
from turnsynth.llm import LLM, extract_json

BC_CONTINUER = {"mhm", "mm", "m", "mmm", "mmhm", "uh-huh", "uhhuh", "hm", "hmm", "mm-hmm"}
BC_REACTION = {"wow", "oh", "really", "no", "whoa", "god", "nice", "ha", "haha", "oof", "yikes", "damn", "seriously", "what", "huh", "ah", "aw"}
BC_ACK = {"yeah", "yes", "yep", "right", "okay", "ok", "sure", "true", "exactly", "totally", "uh", "i", "see", "gotcha", "got", "it", "cool", "alright", "fair", "definitely", "absolutely", "makes", "sense"}
FILLERS = {"um", "uh", "er", "erm", "so", "like", "well", "hmm"}


class Transcriber:
    """faster-whisper on each segment clip, padded a little for context."""

    def __init__(self, model: str = "base.en", device: str = "auto", compute_type: str = "default"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sr: int, segs: list[Segment]) -> list[str]:
        out = []
        for seg in segs:
            i0, i1 = int(max(0.0, seg.start - 0.1) * sr), int((seg.end + 0.1) * sr)
            clip = audio[i0:i1]
            if sr != 16000:
                clip = np.interp(np.arange(0, len(clip), sr / 16000), np.arange(len(clip)), clip).astype(np.float32)
            parts, _ = self.model.transcribe(clip, language="en", beam_size=1, vad_filter=False,
                                             condition_on_previous_text=False)
            out.append(" ".join(p.text.strip() for p in parts).strip())
        return out


def _norm(text: str) -> list[str]:
    return [w.strip(".,?!;:\"'").lower() for w in text.split() if w.strip(".,?!;:\"'")]


def rule_judge(segs: dict[int, list[Segment]], text: dict[int, list[str]]) -> dict[int, Track]:
    """Offline fallback: lexical cues from the transcript decide backchannels,
    fillers and non-speech; overlap geometry decides the rest. It shares the
    geometry annotator's timing rules, so it is far less independent than the
    LLM judge; use it for smoke tests."""
    tracks: dict[int, Track] = {1: [], 2: []}
    for spk in (1, 2):
        for i, (seg, t) in enumerate(zip(segs[spk], text[spk])):
            words = _norm(t)
            geo = overlap_class(segs, spk, i)
            if not words:
                label = L.NONLINGUISTIC
            elif all(w in FILLERS for w in words):
                label = L.FILLER
            elif geo is not None and len(words) <= 3 and all(w in BC_CONTINUER | BC_REACTION | BC_ACK for w in words):
                if any(w in BC_CONTINUER for w in words):
                    label = L.BC_CONTINUER
                elif any(w in BC_REACTION for w in words):
                    label = L.BC_REACTION
                else:
                    label = L.BC_ACK
            elif geo in ("short", "failed"):
                label = L.NFT_COMPETITIVE
            elif geo == "took":
                label = L.FT_COMPETITIVE
            elif geo == "overlap":
                label = L.OVERLAP
            else:
                label = L.NORMAL_TURN
            tracks[spk].append((seg.start, seg.end, label, t))
    return tracks


JUDGE_SYSTEM = f"""You are an expert conversation-analysis annotator labelling a two-speaker recording for a turn-taking benchmark.

You receive a timed transcript: every line is one voice-activity segment on one speaker's channel, with its id, speaker, start and end time in seconds, and the ASR transcript of that segment. Segments are sorted by start time; segments of the two speakers can overlap in time. ASR can mishear short tokens.

Label every segment with exactly one of these labels:

{L.JUDGE_DEFINITIONS}

Judge from content and timing together: a segment starting while the other speaker is mid-sentence, away from a point where they could finish, is an interruption or a backchannel attempt; whether the floor changed hands depends on who keeps talking afterwards. A speaker resuming after their own short pause is still holding their turn.

Return only a JSON object mapping each segment id (as a string) to its label, inside a ```json fenced block."""


@dataclass
class LLMJudge:
    llm: LLM
    window: int = 120
    context: int = 20

    def __call__(self, segs: dict[int, list[Segment]], text: dict[int, list[str]]) -> dict[int, Track]:
        rows = sorted(
            [(seg.start, spk, i, seg, t) for spk in (1, 2) for i, (seg, t) in enumerate(zip(segs[spk], text[spk]))],
            key=lambda r: (r[0], r[1]),
        )
        labels: dict[int, str] = {}
        for w0 in range(0, len(rows), self.window):
            lo = max(0, w0 - self.context)
            chunk = rows[lo: w0 + self.window]
            lines = []
            for n, (_, spk, _, seg, t) in enumerate(chunk, start=lo):
                tag = " (context only, already labelled)" if n < w0 else ""
                lines.append(f"[{n}] S{spk} {seg.start:.2f}-{seg.end:.2f}: {t or '<no words>'}{tag}")
            reply = self.llm.complete(JUDGE_SYSTEM, "\n".join(lines))
            try:
                parsed = extract_json(reply)
            except (ValueError, json.JSONDecodeError):
                parsed = {}
            for key, value in parsed.items():
                if str(key).isdigit() and int(key) >= w0 and value in L.FINE_LABELS:
                    labels[int(key)] = value
        fallback = rule_judge(segs, text)
        tracks: dict[int, Track] = {1: [], 2: []}
        for n, (_, spk, i, seg, t) in enumerate(rows):
            label = labels.get(n, fallback[spk][i][2])
            tracks[spk].append((seg.start, seg.end, label, t))
        for spk in tracks:
            tracks[spk].sort(key=lambda r: r[0])
        return tracks
