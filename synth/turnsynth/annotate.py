"""Annotators that label VAD segments with TurnBench fine labels.

TurnBench triple-annotates every VAD segment and the scorer keeps a label only
where 2 of 3 annotators agree (turnbench/gold.py). We fill the three annotator
slots with three independent views of the same synthetic audio:

  a  intent    the generator LLM's label for the item the segment came from
  b  judge     a post-hoc judge that sees only ASR transcripts + timing
               (judge.py: LLM judge, or a lexical rule judge offline)
  c  geometry  a rule-based labeler that sees only the two VAD tracks

Where the renderer failed to realize the intent (a scripted interruption that
landed after the host had finished, a "backchannel" that turned into a turn),
the views disagree and the scorer's own consensus masks the segment, exactly
as it masks no-majority regions in the human data.
"""

from dataclasses import dataclass

from turnsynth import labels as L
from turnsynth.render import Rendered
from turnsynth.vad import segments as vad_segments

Track = list[tuple[float, float, str, str]]  # (start_s, end_s, fine_label, text)

# A floor claim starting this close to the end of the other's run is an early
# smooth transition, not a barge-in (TurnBench FTOs reach about -0.8 s).
OVERLAP_MAX_S = 0.9


@dataclass(frozen=True)
class Segment:
    speaker: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment(rendered: Rendered) -> dict[int, list[Segment]]:
    return {
        spk: [Segment(spk, s, e) for s, e in vad_segments(rendered.audio[spk], rendered.sample_rate)]
        for spk in (1, 2)
    }


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def segment_words(rendered: Rendered, seg: Segment) -> list[tuple[str, int]]:
    """Script words (with their item id) whose midpoint falls in the segment."""
    out = []
    for p in rendered.placed:
        if p.dropped or rendered.speaker_index[p.item.speaker] != seg.speaker:
            continue
        for w in p.words:
            if seg.start <= (w.start + w.end) / 2 < seg.end:
                out.append((w.text, p.item.id))
    return out


def intent_track(rendered: Rendered, segs: dict[int, list[Segment]]) -> dict[int, Track]:
    """Annotator a: the generator's labels, carried through rendering."""
    tracks: dict[int, Track] = {1: [], 2: []}
    first_seg_of_item: dict[int, Segment] = {}
    for spk in (1, 2):
        for seg in segs[spk]:
            best, best_ov = None, 0.0
            for p in rendered.placed:
                if p.dropped or rendered.speaker_index[p.item.speaker] != spk:
                    continue
                ov = _overlap(seg.start, seg.end, p.start, p.end)
                if ov > best_ov:
                    best, best_ov = p, ov
            text = " ".join(w for w, _ in segment_words(rendered, seg))
            if best is None:
                tracks[spk].append((seg.start, seg.end, L.NONSPEECH_NOISE, text))
                continue
            label = best.item.fine_label
            first = first_seg_of_item.setdefault(best.item.id, seg)
            # Only the onset segment of an interruption is the interruption; the
            # rest of the interrupter's turn is ordinary floor-holding speech.
            if best.item.type == "interruption" and best.item.floor_taking and first is not seg:
                label = L.NORMAL_TURN
            tracks[spk].append((seg.start, seg.end, label, text))
    return tracks


def runs(segs: list[Segment], max_gap: float = 0.5) -> list[tuple[float, float]]:
    """Merge a channel's segments separated by short gaps into speech runs."""
    out: list[list[float]] = []
    for s in segs:
        if out and s.start - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], s.end)
        else:
            out.append([s.start, s.end])
    return [(a, b) for a, b in out]


def other_run_at(other_runs: list[tuple[float, float]], t: float) -> tuple[float, float] | None:
    """The other speaker's run that is mid-way at time t (started before, lasts past t + 0.1)."""
    return next(((a, b) for a, b in other_runs if a < t and b > t + 0.1), None)


def _run_end(segs: list[Segment], start_index: int, max_gap: float = 0.5) -> float:
    end = segs[start_index].end
    for s in segs[start_index + 1:]:
        if s.start - end > max_gap:
            break
        end = max(end, s.end)
    return end


def overlap_class(segs: dict[int, list[Segment]], spk: int, i: int) -> str | None:
    """Classify segment i of `spk` by overlap geometry.

    None      the other speaker was not mid-speech at onset
    overlap   the other was about to finish anyway (early smooth transition)
    took      the other went quiet while this speaker kept going (barge-in)
    short     brief, and the other kept the floor (backchannel-like)
    failed    long, and the other kept the floor (failed attempt)
    """
    other = 3 - spk
    seg = segs[spk][i]
    o_run = other_run_at(runs(segs[other]), seg.start)
    if o_run is None:
        return None
    s_end = _run_end(segs[spk], i)
    # Follow the other's speech while it keeps resuming (gaps <= 0.7 s) before this
    # speaker's run ends: resuming over us after a short pause is not yielding.
    o_last = o_run[1]
    for o in segs[other]:
        if o.start <= o_last or o.start >= s_end:
            continue
        if o.start - o_last > 0.7:
            break
        o_last = max(o_last, o.end)
    if o_last - seg.start <= OVERLAP_MAX_S:
        return "overlap"
    if o_last < s_end - 0.3:
        return "took"
    return "short" if seg.duration < 1.0 else "failed"


GEOMETRY_LABEL = {None: L.NORMAL_TURN, "overlap": L.OVERLAP, "took": L.FT_COMPETITIVE,
                  "short": L.BC_ACK, "failed": L.NFT_COMPETITIVE}


def geometry_track(segs: dict[int, list[Segment]], text: dict[int, list[str]] | None = None) -> dict[int, Track]:
    """Annotator c: labels from the two VAD tracks alone (see overlap_class)."""
    return {
        spk: [(seg.start, seg.end, GEOMETRY_LABEL[overlap_class(segs, spk, i)], text[spk][i] if text else "")
              for i, seg in enumerate(segs[spk])]
        for spk in (1, 2)
    }


def agreement(tracks: list[dict[int, Track]]) -> float:
    """Share of segments whose canonical label has a 2-of-3 majority."""
    total = agree = 0
    for spk in (1, 2):
        for rows in zip(*(t[spk] for t in tracks)):
            canon = [L.CANONICAL[r[2]] for r in rows]
            total += 1
            agree += max(canon.count(c) for c in canon) >= 2
    return agree / max(total, 1)
