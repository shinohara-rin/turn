"""TurnBench label vocabulary.

The fine labels are the annotator labels TurnBench's scorer understands
(turnbench/gold.py LABEL_MAP). We emit exactly these strings so the official
scorer builds gold from our synthetic data with no changes.
"""

NORMAL_TURN = "Normal Turn"
STRONG_FLOOR_HOLD = "Strong Floor Hold"
BOUNDED_RESPONSE = "Bounded Response"
FILLER = "Filler"
OVERLAP = "Overlap"
FT_COMPETITIVE = "Floor-taking Competitive Interruption"
FT_COOPERATIVE = "Floor-taking Cooperative Interruption"
NFT_COMPETITIVE = "Non-floor Taking Competitive Interruption"
NFT_COOPERATIVE = "Non-floor Taking Cooperative Interruption"
BC_ACK = "Acknowledgement Backchannel"
BC_CONTINUER = "Continuer Backchannel"
BC_REACTION = "Reaction Backchannel"
LAUGHTER = "Laughter"
AWKWARD_SILENCE = "Awkward Silence"
NONSPEECH_NOISE = "Non-Speech Noise"
CHANNEL_BLEED = "Channel Bleed"
NONLINGUISTIC = "Speech, Non-Linguistic"

FINE_LABELS = (
    NORMAL_TURN, STRONG_FLOOR_HOLD, BOUNDED_RESPONSE, FILLER, OVERLAP,
    FT_COMPETITIVE, FT_COOPERATIVE, NFT_COMPETITIVE, NFT_COOPERATIVE,
    BC_ACK, BC_CONTINUER, BC_REACTION,
    LAUGHTER, AWKWARD_SILENCE, NONSPEECH_NOISE, CHANNEL_BLEED, NONLINGUISTIC,
)

# Canonical collapse, mirroring turnbench/gold.py (kept in sync by a test).
CANONICAL = {
    NORMAL_TURN: "Turn", STRONG_FLOOR_HOLD: "Turn", BOUNDED_RESPONSE: "Turn",
    FILLER: "Turn", OVERLAP: "Turn",
    FT_COMPETITIVE: "Interruption", FT_COOPERATIVE: "Interruption",
    NFT_COMPETITIVE: "NonFloorTakingInterruption", NFT_COOPERATIVE: "NonFloorTakingInterruption",
    BC_ACK: "Backchannel", BC_CONTINUER: "Backchannel", BC_REACTION: "Backchannel",
    LAUGHTER: "Laughter", AWKWARD_SILENCE: "AwkwardSilence",
    NONSPEECH_NOISE: "NonContent", CHANNEL_BLEED: "NonContent", NONLINGUISTIC: "NonContent",
}

# Script-level vocab the generator LLM writes -> fine label.
TURN_STYLE = {
    "normal": NORMAL_TURN,
    "strong_floor_hold": STRONG_FLOOR_HOLD,
    "bounded_response": BOUNDED_RESPONSE,
    "filler": FILLER,
}
BACKCHANNEL_KIND = {
    "acknowledgement": BC_ACK,
    "continuer": BC_CONTINUER,
    "reaction": BC_REACTION,
}


def interruption_label(floor_taking: bool, stance: str) -> str:
    cooperative = stance == "cooperative"
    if floor_taking:
        return FT_COOPERATIVE if cooperative else FT_COMPETITIVE
    return NFT_COOPERATIVE if cooperative else NFT_COMPETITIVE


# Definitions given to the post-hoc judge (paraphrasing the TurnBench taxonomy).
JUDGE_DEFINITIONS = """\
Normal Turn: speech that takes or holds the conversational floor.
Strong Floor Hold: a turn segment where the speaker audibly fights to keep the floor (raises voice, talks over an attempt to take it).
Bounded Response: a short but complete answer that is a turn in its own right (e.g. "yes, Tuesday works").
Filler: a filled pause ("um", "uh", "so...") by the floor holder, holding the floor.
Overlap: floor-claiming speech that starts while the other speaker is finishing (an early smooth transition, not a barge-in).
Floor-taking Competitive Interruption: the listener starts speaking mid-turn (not at a transition point) and takes the floor; the other speaker stops. Competitive = against the speaker's wishes.
Floor-taking Cooperative Interruption: as above but supportive (finishing their sentence, eager agreement) and the floor still changes hands.
Non-floor Taking Competitive Interruption: the listener tries to take the floor mid-turn but the speaker keeps going and the attempt fails.
Non-floor Taking Cooperative Interruption: a supportive mid-turn contribution longer than a backchannel that does not take the floor.
Acknowledgement Backchannel: short listener token signalling understanding ("yeah", "right", "okay") without claiming the floor.
Continuer Backchannel: short listener token inviting the speaker to go on ("mhm", "uh-huh").
Reaction Backchannel: short listener emotional reaction ("wow", "oh no", "really?") without claiming the floor.
Laughter: laughter.
Awkward Silence: a marked silence that belongs to nobody's turn.
Non-Speech Noise: non-speech sound (cough, breath, click).
Channel Bleed: the other speaker's voice leaking into this channel.
Speech, Non-Linguistic: vocal sound with no words (sigh, hum)."""
