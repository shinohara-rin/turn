"""Per-frame TurnBench label states, the training target in place of a bare p(EOT).

TurnBench gold is computed from annotated segments (turnbench/gold.py at 38a6f87).
This module projects the same gold onto the 12.5 Hz grid as one categorical state
per speaker per frame:

  speaking: TURN, INT_FLOOR (floor-taking interruption), INT_ATTEMPT (non-floor-taking),
            BACKCHANNEL, LAUGHTER, NONCONTENT (noise, bleed, non-linguistic)
  quiet:    HOLD (mid-turn pause; an EOT negative span), YIELD (from an EOT anchor until
            the speaker resumes or the 3 s deadline), LISTEN (everything else)

States are targets, never inputs. Labels that only resolve later, such as whether
an onset takes the floor, are learned as calibrated beliefs from causal inputs.
Scores for the official sweep are read straight off the state posterior:
EOT = p(YIELD), INT = p(INT_FLOOR).
"""
from __future__ import annotations

import numpy as np

STATES = ('LISTEN', 'HOLD', 'YIELD', 'TURN', 'INT_FLOOR', 'INT_ATTEMPT', 'BACKCHANNEL', 'LAUGHTER',
          'NONCONTENT')
S = {name: i for i, name in enumerate(STATES)}
QUIET = (S['LISTEN'], S['HOLD'], S['YIELD'])
SPEAKING = tuple(i for i in range(len(STATES)) if i not in QUIET)
# Canonical TurnBench labels (gold.LABEL_MAP keys) -> speaking state. When segments
# overlap on one channel, the earlier entry in this order wins.
CANONICAL_TO_STATE = {
    'Interruption': 'INT_FLOOR',
    'NonFloorTakingInterruption': 'INT_ATTEMPT',
    'Turn': 'TURN',
    'Backchannel': 'BACKCHANNEL',
    'Laughter': 'LAUGHTER',
    'NonContent': 'NONCONTENT',
}
TAU_MAX_S = 3.0  # gold.TAU_MAX_S: the scorer's latency deadline


def frame_states(times, segments, events, yield_s=TAU_MAX_S):
    """times [T] frame availability times (s); segments: iterable of
    (speaker in {1,2}, start, end, canonical_label); events: asdict(ConversationEvents).

    Returns (state [T, 2] int64, weight [T, 2] float32). A weight of 0 marks frames
    inside the gold's excluded (no-majority) spans, which the scorer also ignores.
    """
    t = np.asarray(times, np.float64)
    state = np.full((len(t), 2), S['LISTEN'], np.int64)
    weight = np.ones((len(t), 2), np.float32)
    speaking = np.zeros((len(t), 2), bool)
    priority = {name: i for i, name in enumerate(CANONICAL_TO_STATE)}
    rank = np.full((len(t), 2), len(priority))
    onsets = {1: [], 2: []}
    for speaker, start, end, label in segments:
        if label not in CANONICAL_TO_STATE:
            continue  # e.g. AwkwardSilence: quiet, not a vocalisation
        m = (t >= start) & (t < end)
        c = speaker - 1
        better = m & (priority[label] < rank[:, c])
        state[better, c] = S[CANONICAL_TO_STATE[label]]
        rank[better, c] = priority[label]
        speaking[m, c] = True
        if label in ('Turn', 'Interruption'):  # gold's "resumes": next floor-claiming start
            onsets[speaker].append(start)
    for span in events.get('eot_negative_spans', []):
        c = span['speaker'] - 1
        m = (t >= span['start']) & (t < span['end']) & ~speaking[:, c]
        state[m, c] = S['HOLD']
    for anchor in events.get('eot_positive_events', []):
        c, a = anchor['speaker'] - 1, anchor['time_s']
        resume = min([o for o in onsets[anchor['speaker']] if o > a], default=np.inf)
        m = (t >= a) & (t < min(a + yield_s, resume)) & ~speaking[:, c]
        state[m, c] = S['YIELD']
    for key, kinds in (('eot_excluded', QUIET), ('int_excluded', SPEAKING)):
        for span in events.get(key, []):
            for c in ((span['speaker'] - 1,) if 'speaker' in span else (0, 1)):
                m = (t >= span['start']) & (t < span['end']) & np.isin(state[:, c], kinds)
                weight[m, c] = 0
    return state, weight


def activity(state):
    """Binary per-speaker activity implied by states (VAP supervision)."""
    return np.isin(state, SPEAKING).astype(np.float32)


def turnbench_scores(state_logits):
    """[..., 2, len(STATES)] logits -> dict of [..., 2] scores for EOT and INT."""
    import torch
    p = torch.softmax(state_logits, -1)
    return dict(eot=p[..., S['YIELD']], int=p[..., S['INT_FLOOR']])


# Mixed-speaker (mono) audio has no channel per speaker. Speakers are instead
# assigned to slots in order of first speech (Sortformer's arrival-time ordering),
# which is causal and needs no permutation search. Each slot gets activity (an explicit
# diarization target, overlap allowed) and the same TurnBench label states as stereo.
# Stereo audio mixed down to mono therefore gives exact per-slot labels.


def arrival_order(activity):
    """[T, C] activity -> channel indices sorted by first active frame; never-active last."""
    a = np.asarray(activity) > 0.5
    first = np.where(a.any(0), a.argmax(0), len(a))
    return np.argsort(first, kind='stable')


def slot_targets(state, weight):
    """Stereo states [T, 2] -> arrival-ordered (slot_state [T, 2], slot_weight [T, 2],
    slot_activity [T, 2])."""
    order = arrival_order(activity(state))
    return state[:, order], weight[:, order], activity(state)[:, order]


def slot_activity_from_segments(segments, frames, slots=2, frame_s=0.08):
    """Diarization output [(start_s, end_s, speaker_label)] -> arrival-ordered [frames, slots].

    For podcast mono audio where only diarization (no label states) is available.
    Speakers beyond `slots` raise: DuplexChat clips are selected to contain two.
    """
    labels = sorted({spk for _, _, spk in segments}, key=lambda k: min(s for s, _, l in segments if l == k))
    if len(labels) > slots:
        raise ValueError(f'{len(labels)} speakers > {slots} slots')
    a = np.zeros((frames, slots), np.float32)
    centers = (np.arange(frames) + 0.5) * frame_s
    for start, end, spk in segments:
        a[(centers >= start) & (centers < end), labels.index(spk)] = 1
    return a
