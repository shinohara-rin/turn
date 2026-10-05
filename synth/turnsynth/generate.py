"""Two-pass script synthesis (MultiTalk §2.1.2), retargeted at TurnBench.

Pass 1 (creative design) fixes the world: scenario, two speakers with stable
personas, and a trajectory that fits one of TurnBench's six conversation
types. Pass 2 realizes the dialogue under that design, with the turn-taking
dynamics written as explicit, labelled script items: within-turn pauses,
backchannels and interruptions anchored on a word of the host turn. Those
labels are annotator a. The script then goes through the filter in script.py.
"""

import json

import numpy as np

from turnsynth.config import TYPES, ConversationType
from turnsynth.llm import LLM, extract_json
from turnsynth.script import Script, ScriptError, check, parse

TOPIC_SEEDS = {
    "Argumentative": ["whether remote work is better than office work", "if tipping culture should be abolished",
                      "whether college is still worth it", "pineapple on pizza, seriously", "whether cities should ban cars downtown"],
    "Casual": ["a disastrous weekend trip", "a new neighbour", "favourite comfort food", "a show they're both watching",
               "what they did last weekend"],
    "Collaborative": ["planning a surprise party on a small budget", "naming a new cafe", "fixing a team's broken standup",
                      "designing the perfect board game night", "how to split chores fairly"],
    "Instructional": ["how to make sourdough starter", "setting up a home network", "learning to change a bike tyre",
                      "getting started with a budget spreadsheet", "how to prune a tomato plant"],
    "Narrative": ["the time they got lost abroad", "their worst job interview", "how they met their best friend",
                  "a strange thing that happened on the subway", "a childhood holiday"],
    "Task-Oriented": ["booking a group dinner for Friday", "rescheduling a dentist appointment", "choosing a used car",
                      "sorting out a wrong delivery", "planning a moving day"],
}

DESIGN_SYSTEM = """You design realistic two-person spoken conversations for a speech dataset.
Write ordinary people, not assistants. Never quote the seed verbatim."""

DESIGN_USER = """Design one {type} conversation ({type_desc}).
Starting topic: {topic}. Topics may drift naturally.

Return a ```json block with:
{{"scenario": "<4-6 sentences: setting, relationship, what is at stake, how the talk will develop>",
  "speakers": {{
    "A": {{"name": "...", "gender": "female|male", "summary": "...", "background": "...", "personality": "...", "speaking_style": "<pace, verbal habits, how often they backchannel or interrupt>"}},
    "B": {{ same fields }}
  }}}}
Every person named in the scenario must be A or B."""

DIALOGUE_SYSTEM = """You write spoken two-person dialogue scripts that will be rendered with text-to-speech and used to train and evaluate turn-taking models. The timing of who speaks when matters as much as the words.

Output format: a ```json block {"turns": [item, ...]} where each item is one of

  {"id": int, "speaker": "A"|"B", "type": "turn", "style": "normal"|"bounded_response"|"strong_floor_hold"|"filler", "text": str}
  {"id": int, "speaker": "A"|"B", "type": "backchannel", "kind": "acknowledgement"|"continuer"|"reaction", "text": str, "host": int, "after_word": int}
  {"id": int, "speaker": "A"|"B", "type": "interruption", "floor_taking": bool, "stance": "competitive"|"cooperative", "text": str, "host": int, "after_word": int}

Rules:
- ids increase by 1 from 1. Items are in temporal order of onset.
- turn: one stretch of floor-holding speech. Consecutive turns by the same speaker mean they paused and resumed with no handover.
- Inside a turn, write <pause X> (X in seconds, 0.3-2.0) where the speaker pauses but keeps the floor. Use them often, as people do: mid-sentence while thinking ("I think we should <pause 0.8> probably leave early"), after a filler ("so, um <pause 0.6> where was I"), and sometimes after a complete sentence before adding more ("That's the plan. <pause 0.9> Unless it rains."). These within-turn pauses are the hard cases, so roughly one turn in three should have at least one.
- backchannel: a 1-3 word listener response ("mhm", "yeah", "right", "oh wow") that does NOT take the floor. host is the id of the other speaker's turn it lands in; after_word is how many words of the host have been spoken when it starts (1 <= after_word < host word count). Place them at natural phrase boundaries inside long host turns.
- interruption with floor_taking=true: the listener barges in mid-turn and takes the floor. host must be the floor item right before it. Write the host's full intended sentence, with at least 8 more words after after_word; it will be cut off automatically a second or two after the interruption starts, so those words are only partly heard. The interruption itself is at least 5 words, so the interrupter is still talking after the host gives up. The next items should follow on from the interruption.
- interruption with floor_taking=false: the listener tries to come in (or makes a short supportive remark longer than a backchannel), but the host keeps talking; it is a listener overlay like a backchannel.
- style: bounded_response = a short complete answer; filler = a turn that is only a filled pause; strong_floor_hold = the speaker pushes on to keep the floor.
- Turn lengths: about 60% short (1-15 words), 30% medium (15-40), 10% long (40-90); no two long turns in a row.
- Punctuate for the voice: the TTS reads each item in one go, so a comma or "..." before a <pause X> where the thought continues keeps the voice up, and a full stop or "?" ends it. Questions get "?".
- Optional on any item: "emotion": {name: weight} with names from happy, angry, sad, afraid, disgusted, melancholic, surprised, calm and weights summing to at most 0.8, for how the line should sound given what was just said (a sharp "angry": 0.4 retort, a "surprised": 0.5 "oh wow"). Leave it out for neutral lines; use it where the delivery matters, and keep each speaker consistent from line to line.
- Text is spoken English exactly as it should be pronounced: disfluencies ("uh", "um", "I mean"), contractions and self-repairs are welcome. No stage directions, brackets, parentheses, asterisks, emoji or speaker names in the text.
- Never break character or mention being an AI."""

DIALOGUE_USER = """Conversation design:
{design}

Conversation type: {type} ({type_desc}).
Target about {n_items} items in total, covering roughly {minutes:.0f} minutes of talk. For this type, aim for about {n_bc} backchannels and {n_int} interruptions (about {n_ft} of them floor-taking), spread through the conversation.

Write the script."""


def target_counts(ctype: ConversationType, minutes: float) -> dict[str, int]:
    n_bc = round(ctype.backchannels_per_min * minutes)
    n_int = round(ctype.interruptions_per_min * minutes)
    n_turns = round(minutes * 60 / ctype.mean_turn_s * 1.6)  # segments, not whole turns
    return {"n_bc": n_bc, "n_int": n_int, "n_ft": round(n_int * 0.6), "n_items": n_turns + n_bc + n_int}


def generate_script(llm: LLM, conversation_type: str, *, topic: str | None = None, minutes: float = 4.0,
                    seed: int = 0, max_attempts: int = 3) -> tuple[Script | None, list[dict]]:
    """Returns (script or None, attempt log with reason codes)."""
    rng = np.random.default_rng(seed)
    ctype = TYPES[conversation_type]
    topic = topic or str(rng.choice(TOPIC_SEEDS[conversation_type]))
    log: list[dict] = []
    design = extract_json(llm.complete(DESIGN_SYSTEM, DESIGN_USER.format(type=ctype.name, type_desc=ctype.description, topic=topic)))
    counts = target_counts(ctype, minutes)
    for attempt in range(max_attempts):
        reply = llm.complete(DIALOGUE_SYSTEM, DIALOGUE_USER.format(
            design=json.dumps(design, indent=1), type=ctype.name, type_desc=ctype.description,
            minutes=minutes, **counts))
        try:
            obj = extract_json(reply)
            obj = {"conversation_type": ctype.name, "scenario": design.get("scenario", ""),
                   "speakers": design["speakers"], "turns": obj["turns"],
                   "meta": {"topic": topic, "seed": seed, "attempt": attempt, "targets": counts}}
            script = parse(obj)
        except (ScriptError, ValueError, KeyError) as e:
            log.append({"attempt": attempt, "reasons": [getattr(e, "code", "schema")], "detail": str(e)})
            continue
        reasons = check(script)
        log.append({"attempt": attempt, "reasons": reasons})
        if not reasons:
            return script, log
    return None, log
