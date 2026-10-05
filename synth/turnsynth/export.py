"""Write conversations in TurnBench's parquet layout.

One row per conversation with `conversation_id` (numeric string),
`speaker_{1,2}_audio` ({bytes, path}, FLAC), `speaker_{1,2}_annotation_{a,b,c}`
(list of {start_s, end_s, label, text}) and a `metadata` struct. A directory
of these shards is a valid `--dataset` for every turnbench command:

    python -m turnbench.score predictions.json --dataset out/parquet
"""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from turnsynth.annotate import Track
from turnsynth.render import Rendered

ANNOTATION_TYPE = pa.list_(pa.struct([
    ("start_s", pa.float64()), ("end_s", pa.float64()), ("label", pa.string()), ("text", pa.string()),
]))
AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
METADATA_TYPE = pa.struct([
    ("conversation_type", pa.string()), ("source", pa.string()), ("generator", pa.string()),
    ("tts", pa.string()), ("speaker_1_voice", pa.string()), ("speaker_2_voice", pa.string()),
    ("speaker_1_gender", pa.string()), ("speaker_2_gender", pa.string()),
    ("annotator_a", pa.string()), ("annotator_b", pa.string()), ("annotator_c", pa.string()),
    ("agreement", pa.float64()),
])
SCHEMA = pa.schema(
    [("conversation_id", pa.string())]
    + [(f"speaker_{s}_audio", AUDIO_TYPE) for s in (1, 2)]
    + [(f"speaker_{s}_annotation_{a}", ANNOTATION_TYPE) for s in (1, 2) for a in "abc"]
    + [("metadata", METADATA_TYPE)]
)


def flac_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def to_row(rendered: Rendered, tracks: dict[str, dict[int, Track]], metadata: dict) -> dict:
    row = {"conversation_id": rendered.conversation_id, "metadata": metadata}
    for s in (1, 2):
        row[f"speaker_{s}_audio"] = {"bytes": flac_bytes(rendered.audio[s], rendered.sample_rate),
                                     "path": f"{rendered.conversation_id}_speaker_{s}.flac"}
        for a in "abc":
            row[f"speaker_{s}_annotation_{a}"] = [
                {"start_s": float(st), "end_s": float(en), "label": lab, "text": txt}
                for st, en, lab, txt in tracks[a][s]
            ]
    return row


def write_shard(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def intent_record(rendered: Rendered) -> dict:
    """Full generator-side record: script plus where each item actually landed.
    Useful as dense training targets beyond the TurnBench label columns."""
    return {
        "conversation_id": rendered.conversation_id,
        "script": rendered.script.to_json(),
        "voices": rendered.voices,
        "placed": [
            {"id": p.item.id, "speaker": p.item.speaker, "label": p.item.fine_label,
             "start": round(p.start, 3), "end": round(p.end, 3), "cut": p.cut, "dropped": p.dropped,
             "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in p.words] if not p.dropped else []}
            for p in rendered.placed
        ],
    }


def write_intent(rendered: Rendered, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{rendered.conversation_id}.json").write_text(json.dumps(intent_record(rendered)))
