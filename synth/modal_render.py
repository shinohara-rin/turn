"""Render scripts into TurnBench-format parquet on Modal GPUs.

Kokoro and faster-whisper both run on CPU (see README), but a GPU makes a few
hundred hours practical. Script generation needs no GPU: run `turnsynth
scripts` locally, then fan the rendering out here.

Prereqs (once, on a machine where the Modal CLI can connect):
    pip install modal && modal token set ...
    modal secret create anthropic ANTHROPIC_API_KEY=...   # only for --judge llm
Run:
    modal run modal_render.py --scripts out/scripts --out out/synth --judge llm
Each container renders one chunk of scripts and returns a parquet shard, the
intent records and its report lines; they land under --out.
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libsndfile1", "espeak-ng")
    .add_local_dir(HERE, "/root/synth", copy=True, ignore=[".venv", "out", "**/__pycache__"])
    .run_commands(
        "pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124",
        "pip install '/root/synth[llm,tts,asr]'",
        "pip uninstall -y spacy-curated-transformers || true",
        # Warm the model caches into the image.
        "python -c \"from turnsynth.tts import KokoroTTS; KokoroTTS(device='cpu').synthesize('warm up', 'af_heart')\"",
        "python -c \"from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu')\"",
    )
)
app = modal.App("turnsynth-render", image=image)


@app.function(gpu="L4", timeout=4 * 3600, secrets=[modal.Secret.from_name("anthropic", required_keys=[])])
def render_chunk(scripts: list[tuple[str, str]], id_offset: int, seed: int, judge: str, asr: str,
                 bleed_db: float | None) -> dict:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    from turnsynth.export import SCHEMA, intent_record, to_row
    from turnsynth.judge import LLMJudge, Transcriber
    from turnsynth.llm import AnthropicLLM
    from turnsynth.pipeline import process
    from turnsynth.script import ScriptError, parse
    from turnsynth.tts import KokoroTTS

    tts = KokoroTTS(device="cuda")
    transcriber = Transcriber(asr, device="cuda", compute_type="float16") if asr else None
    llm_judge = LLMJudge(AnthropicLLM(effort="low")) if judge == "llm" else None
    rows, intents, report = [], [], []
    for n, (name, text) in enumerate(scripts):
        try:
            script = parse(json.loads(text))
        except ScriptError as e:
            report.append({"script": name, "accepted": False, "reasons": [e.code]})
            continue
        cid = str(id_offset + n)
        res = process(script, tts, conversation_id=cid, seed=seed + id_offset + n,
                      transcriber=transcriber, judge=llm_judge, bleed_db=bleed_db)
        report.append({"script": name, "conversation_id": cid, "accepted": not res.reasons,
                       "reasons": res.reasons, "agreement": round(res.agreement, 3),
                       "wer": None if res.wer is None else round(res.wer, 3),
                       "duration_s": round(res.rendered.duration, 1)})
        if res.reasons:
            continue
        sp = script.speakers
        rows.append(to_row(res.rendered, res.tracks, {
            "conversation_type": script.conversation_type, "source": "turnsynth", "generator": "llm",
            "tts": "kokoro", "speaker_1_voice": res.rendered.voices["A"], "speaker_2_voice": res.rendered.voices["B"],
            "speaker_1_gender": sp["A"].get("gender", ""), "speaker_2_gender": sp["B"].get("gender", ""),
            "annotator_a": "generator-intent", "annotator_b": f"judge:{judge}+asr:{asr}",
            "annotator_c": "geometry", "agreement": res.agreement,
        }))
        intents.append(intent_record(res.rendered))
    shard = b""
    if rows:
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), buf)
        shard = buf.getvalue()
    return {"shard": shard, "intents": intents, "report": report}


@app.local_entrypoint()
def main(scripts: str, out: str, chunk: int = 20, judge: str = "llm", asr: str = "small.en",
         bleed_db: float = 0.0, id_offset: int = 900000, seed: int = 0):
    paths = sorted(Path(scripts).glob("*.json"))
    paths = [p for p in paths if p.name != "generation_log.jsonl"]
    items = [(p.name, p.read_text()) for p in paths]
    chunks = [items[i: i + chunk] for i in range(0, len(items), chunk)]
    args = [(c, id_offset + i * chunk, seed, judge, asr, bleed_db or None) for i, c in enumerate(chunks)]
    out_dir = Path(out)
    (out_dir / "parquet").mkdir(parents=True, exist_ok=True)
    (out_dir / "intent").mkdir(parents=True, exist_ok=True)
    report = []
    for i, result in enumerate(render_chunk.starmap(args)):
        if result["shard"]:
            (out_dir / "parquet" / f"synth-{args[i][1]:06d}.parquet").write_bytes(result["shard"])
        for rec in result["intents"]:
            (out_dir / "intent" / f"{rec['conversation_id']}.json").write_text(json.dumps(rec))
        report += result["report"]
        print(f"chunk {i + 1}/{len(chunks)}: {sum(r['accepted'] for r in result['report'])}/{len(result['report'])} accepted")
    (out_dir / "render_report.jsonl").write_text("\n".join(json.dumps(r) for r in report) + "\n")
