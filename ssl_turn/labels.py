"""Floor ownership targets: a joint floor state plus per-speaker vocal acts.

Turn-taking is modeled as *who holds the floor*, not as per-speaker end-of-turn flags:

  floor (joint, one per frame): HELD_0, HELD_1, OPEN, CONTESTED
    HELD_c     speaker c holds the floor: talking, or pausing without yielding (a hold)
    OPEN       nobody holds it: the last holder yielded and nobody has claimed it yet
    CONTESTED  both speakers are bidding for the floor at once (an overlap involving an
               interruption or attempt). Overlap is a state of the floor, not two
               independent p(speaking) values
  Backchannels never contest: they are not claims.
  Hand-offs (the next speaker starts before the current one finishes, with no
  interruption label) are not contests either. The target is *soft* across the
  overlap and moves linearly from HELD_old to HELD_new, with p(HELD_0) + p(HELD_1) = 1.
  act (per speaker): SILENT, CLAIM, BACKCHANNEL, LAUGHTER, NONCONTENT
    CLAIM is floor-claiming speech: turns, and interruptions whether or not they succeed

How a contest resolves is learned from floor *projection*: the floor state at
t + 0.4/0.8/1.6 s, a floor-level analogue of VAP. Events are transitions:
  EOT for c:  HELD_c -> OPEN or HELD_other while c is quiet
  INT for c:  HELD_other -> CONTESTED -> HELD_c (floor-taking); resolving back to HELD_other
              is a failed attempt, and a backchannel never leaves HELD_other

Gold comes from the pinned TurnBench builder (turnbench/gold.py at 38a6f87): annotated
segments give claims and acts; EOT anchors decide whether a holder's silence is a hold
or a yield, exactly as the gold's floor construction does. Targets only, never inputs.
"""
from __future__ import annotations

import numpy as np

FLOOR = ('HELD_0', 'HELD_1', 'OPEN', 'CONTESTED')
F = {name: i for i, name in enumerate(FLOOR)}
ACTS = ('SILENT', 'CLAIM', 'BACKCHANNEL', 'LAUGHTER', 'NONCONTENT')
A = {name: i for i, name in enumerate(ACTS)}
# Canonical TurnBench labels (gold.LABEL_MAP keys) -> act; earlier wins on overlap.
CANONICAL_TO_ACT = {
    'Turn': 'CLAIM',
    'Interruption': 'CLAIM',
    'NonFloorTakingInterruption': 'CLAIM',
    'Backchannel': 'BACKCHANNEL',
    'Laughter': 'LAUGHTER',
    'NonContent': 'NONCONTENT',
}
HORIZONS_S = (0.4, 0.8, 1.6)
FRAME_S = 0.08
ANCHOR_TOL_S = 0.05


def _in(t, start, end):
    return (t >= start) & (t < end)


def floor_targets(times, segments, events):
    """times [T] frame times (s); segments: iterable of (speaker in {1,2}, start, end,
    canonical_label); events: asdict(ConversationEvents) from the pinned gold builder.

    Returns dict(floor [T, 4] soft target, floor_w [T], act [T, 2], act_w [T, 2]). Weights are 0 in the
    gold's no-majority spans: turn-view disputes for the floor, label-view disputes for
    acts. Agreed failed interruptions stay supervised: they define a lost contest.
    """
    t = np.asarray(times, np.float64)
    T = len(t)
    act = np.full((T, 2), A['SILENT'], np.int64)
    rank = np.full((T, 2), len(CANONICAL_TO_ACT))
    priority = {k: i for i, k in enumerate(CANONICAL_TO_ACT)}
    claim_end = np.full((T, 2), np.nan)  # end time of the claim segment covering each frame
    bidding = np.zeros((T, 2), bool)     # inside an interruption or attempt
    attempts = set()
    for speaker, start, end, label in segments:
        if label not in CANONICAL_TO_ACT:
            continue  # e.g. AwkwardSilence
        c, m = speaker - 1, _in(t, start, end)
        better = m & (priority[label] < rank[:, c])
        act[better, c] = A[CANONICAL_TO_ACT[label]]
        rank[better, c] = priority[label]
        if CANONICAL_TO_ACT[label] == 'CLAIM':
            claim_end[m, c] = np.fmax(claim_end[m, c], end)
        if label == 'NonFloorTakingInterruption':
            attempts.add((speaker, start, end))
        if label in ('Interruption', 'NonFloorTakingInterruption'):
            bidding[m, c] = True
    anchors = {1: [], 2: []}
    for e in events.get('eot_positive_events', []):
        anchors[e['speaker']].append(e['time_s'])

    claiming = act == A['CLAIM']
    hard = np.empty(T, np.int64)
    holder, last_end = None, {0: None, 1: None}
    for i in range(T):
        for c in (0, 1):
            if claiming[i, c]:
                last_end[c] = claim_end[i, c]
        n = claiming[i].sum()
        if n == 2:
            hard[i] = F['CONTESTED']  # refined below: only bids stay contested
        elif n == 1:
            holder = int(np.argmax(claiming[i]))
            hard[i] = holder
        elif holder is None:
            hard[i] = F['OPEN']
        else:
            # Silent floor: still held unless the holder's last claim ended at an EOT anchor.
            yielded = any(abs(a - last_end[holder]) <= ANCHOR_TOL_S for a in anchors[holder + 1])
            hard[i] = F['OPEN'] if yielded else holder
    floor = np.eye(len(FLOOR), dtype=np.float32)[hard]

    # Overlap runs without an interruption or attempt: a hand-off or plain co-talk.
    both = claiming.all(1)
    i = 0
    while i < T:
        if not both[i]:
            i += 1
            continue
        j = i
        while j < T and both[j]:
            j += 1
        if not bidding[i:j].any():
            before = hard[i - 1] if i and hard[i - 1] < 2 else None
            after = hard[j] if j < T and hard[j] < 2 else None
            if before is not None and after is not None and before != after:
                alpha = (np.arange(j - i) + 0.5) / (j - i)  # linear hand-off
                floor[i:j] = 0
                floor[i:j, before] = 1 - alpha
                floor[i:j, after] = alpha
            elif before is not None or after is not None:
                keep = before if before is not None else after  # co-talk that took nothing
                floor[i:j] = np.eye(len(FLOOR), dtype=np.float32)[keep]
        i = j

    floor_w = np.ones(T, np.float32)
    for span in events.get('eot_excluded', []):
        floor_w[_in(t, span['start'], span['end'])] = 0
    act_w = np.ones((T, 2), np.float32)
    for span in events.get('int_excluded', []):
        if (span['speaker'], span['start'], span['end']) in attempts:
            continue
        act_w[_in(t, span['start'], span['end']), span['speaker'] - 1] = 0
    return dict(floor=floor, floor_w=floor_w, act=act, act_w=act_w)


def floor_projection(floor, floor_w, horizons_s=HORIZONS_S, frame_s=FRAME_S):
    """Soft floor target at each horizon ahead: ([T, H, 4], [T, H] weight; 0 past the end)."""
    T = len(floor)
    steps = [int(round(h / frame_s)) for h in horizons_s]
    target = np.zeros((T, len(steps), floor.shape[-1]), np.float32)
    weight = np.zeros((T, len(steps)), np.float32)
    for j, k in enumerate(steps):
        target[:T - k, j] = floor[k:]
        weight[:T - k, j] = floor_w[k:]
    return target, weight


def activity(act):
    """Per-speaker vocal activity implied by acts (VAP supervision)."""
    return (np.asarray(act) != A['SILENT']).astype(np.float32)


def turnbench_scores(floor_logits, future_logits, int_horizon=1):
    """floor_logits [..., 4], future_logits [..., H, 4] -> per-speaker scores [..., 2].

    EOT for c: p(floor is OPEN or held by the other), read while c is quiet.
    INT for c: p(c holds the floor at the INT horizon), i.e. c wins the contest.
    """
    import torch
    now = floor_logits.softmax(-1)
    ahead = future_logits[..., int_horizon, :].softmax(-1)
    eot = torch.stack([now[..., F['OPEN']] + now[..., F['HELD_1']],
                       now[..., F['OPEN']] + now[..., F['HELD_0']]], -1)
    return dict(eot=eot, int=ahead[..., :2])


# Mixed-speaker (mono) audio has no channel per speaker. Speakers go into slots in
# order of first speech (Sortformer's arrival-time ordering), which is causal and needs
# no permutation search. HELD_c then refers to slot c. Stereo audio mixed down to mono
# gives exact slot targets.


def arrival_order(activity):
    """[T, C] activity -> channel indices sorted by first active frame; never-active last."""
    a = np.asarray(activity) > 0.5
    first = np.where(a.any(0), a.argmax(0), len(a))
    return np.argsort(first, kind='stable')


def to_slots(targets):
    """Channel-indexed targets (floor_targets output, plus optional 'future') -> slot-indexed,
    adding 'slot_activity' for the diarization head."""
    order = arrival_order(activity(targets['act']))
    columns = np.r_[order, 2, 3]  # slot k takes channel order[k]'s HELD column
    out = dict(targets, floor=targets['floor'][..., columns], act=targets['act'][:, order],
               act_w=targets['act_w'][:, order])
    if 'future' in targets:
        out['future'] = targets['future'][..., columns]
    out['slot_activity'] = activity(out['act'])
    return out


def slot_activity_from_segments(segments, frames, slots=2, frame_s=FRAME_S):
    """Diarization output [(start_s, end_s, speaker_label)] -> arrival-ordered [frames, slots].

    For podcast mono audio where only diarization (no floor labels) is available.
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
