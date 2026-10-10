"""Conversation types and the timing model, calibrated on TurnBench.

Per-type rates come from TurnBench Table III (raw per-annotator dynamics) and
the corpus-wide timing facts in §IV-B: floor-transfer offset (FTO) median
-281 ms (-151 ms excluding interruptions), 64% of transfers start in overlap,
non-overlapping gap median 0.38 s, intra-speaker pause median 0.51 s, and an
interrupted speaker keeps talking a median 1.48 s after the interrupter's onset.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConversationType:
    name: str
    description: str
    words_per_min: float
    mean_turn_s: float
    backchannels_per_min: float
    interruptions_per_min: float
    # Mean of the smooth-transition FTO (s). More negative = more overlap.
    fto_mean: float


# Descriptions follow TurnBench §III-A.
TYPES: dict[str, ConversationType] = {
    t.name: t
    for t in [
        ConversationType("Argumentative", "structured disagreement: longer turns, fewer backchannels, competitive interruptions", 206, 10.0, 3.09, 2.48, -0.12),
        ConversationType("Casual", "unstructured social talk with topic drift, humor and many backchannels", 211, 8.1, 5.20, 2.05, -0.22),
        ConversationType("Collaborative", "shared reasoning with no fixed answer; frequent overlap and cooperative interruptions", 207, 7.9, 3.97, 2.64, -0.25),
        ConversationType("Instructional", "an expert guides a learner: asymmetric turns, confirmation backchannels", 202, 10.5, 4.30, 1.52, -0.05),
        ConversationType("Narrative", "one speaker tells a story to an active listener: backchannels, little floor competition", 198, 10.2, 4.76, 1.50, -0.08),
        ConversationType("Task-Oriented", "goal-directed exchange with clarifications", 205, 8.9, 5.00, 1.75, -0.10),
    ]
}


@dataclass(frozen=True)
class Timing:
    """Sampling distributions for rendering a script onto a timeline (seconds)."""

    fto_sd: float = 0.40
    fto_min: float = -0.8
    fto_max: float = 2.0
    pause_median: float = 0.51
    pause_sigma: float = 0.45
    pause_min: float = 0.25
    pause_max: float = 2.5
    yield_median: float = 1.4  # interrupted speaker's talk-on after interrupter onset
    yield_sigma: float = 0.45
    yield_min: float = 1.0  # above annotate.OVERLAP_MAX_S, so a barge-in never reads as a smooth overlap
    yield_max: float = 3.0
    min_talk_after_yield: float = 0.5  # interrupter keeps talking this long after the host stops
    onset_jitter: float = 0.15  # (unused since overlays got reaction delays; kept for old configs)
    # Listener reaction time after the anchor word ends: a backchannel answers
    # a phrase end it saw coming; an interruption reacts to what was just said.
    backchannel_delay: tuple[float, float] = (0.1, 0.35)
    interrupt_delay_median: float = 0.3
    interrupt_delay_sigma: float = 0.4
    interrupt_delay_min: float = 0.12
    interrupt_delay_max: float = 0.8
    min_same_channel_gap: float = 0.15

    def fto(self, rng: np.random.Generator, ctype: ConversationType) -> float:
        return float(np.clip(rng.normal(ctype.fto_mean, self.fto_sd), self.fto_min, self.fto_max))

    def pause(self, rng: np.random.Generator, requested: float | None = None) -> float:
        if requested is not None:
            return float(np.clip(requested, self.pause_min, self.pause_max))
        return float(np.clip(rng.lognormal(np.log(self.pause_median), self.pause_sigma), self.pause_min, self.pause_max))

    def reaction(self, rng: np.random.Generator, kind: str) -> float:
        if kind == "backchannel":
            return float(rng.uniform(*self.backchannel_delay))
        return float(np.clip(rng.lognormal(np.log(self.interrupt_delay_median), self.interrupt_delay_sigma),
                             self.interrupt_delay_min, self.interrupt_delay_max))

    def yield_delay(self, rng: np.random.Generator) -> float:
        return float(np.clip(rng.lognormal(np.log(self.yield_median), self.yield_sigma), self.yield_min, self.yield_max))
