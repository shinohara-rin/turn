"""Command line.

    turnsynth scripts --n 12 --minutes 4 --out out/scripts          # LLM: pass 1 + pass 2 + filter
    turnsynth render out/scripts --tts kokoro --asr base.en --judge llm --out out/synth
    turnsynth stats out/synth/parquet
"""

import argparse
import json
from pathlib import Path

from turnsynth.config import TYPES


def cmd_scripts(args) -> None:
    from turnsynth.generate import generate_script
    from turnsynth.llm import AnthropicLLM

    llm = AnthropicLLM(model=args.model, effort=args.effort)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    types = args.types.split(",") if args.types else list(TYPES)
    log = []
    for i in range(args.n):
        ctype = types[i % len(types)]
        seed = args.seed + i
        script, attempts = generate_script(llm, ctype, minutes=args.minutes, seed=seed)
        log.append({"index": i, "type": ctype, "seed": seed, "accepted": script is not None, "attempts": attempts})
        if script is not None:
            (out / f"{i:05d}.json").write_text(json.dumps(script.to_json(), indent=1))
        print(json.dumps(log[-1]))
    (out / "generation_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")


def cmd_render(args) -> None:
    from turnsynth.export import to_row, write_intent, write_shard
    from turnsynth.judge import LLMJudge, Transcriber
    from turnsynth.pipeline import process
    from turnsynth.script import ScriptError, parse
    from turnsynth.tts import make_tts

    paths = sorted(Path(args.scripts).glob("*.json")) if Path(args.scripts).is_dir() else [Path(args.scripts)]
    tts = make_tts(args.tts)
    transcriber = Transcriber(args.asr, device=args.device) if args.asr else None
    judge = None
    if args.judge == "llm":
        from turnsynth.llm import AnthropicLLM

        judge = LLMJudge(AnthropicLLM(model=args.judge_model, effort="low"))
    out = Path(args.out)
    rows, report = [], []
    for n, path in enumerate(paths):
        try:
            script = parse(json.loads(path.read_text()))
        except ScriptError as e:
            report.append({"script": path.name, "accepted": False, "reasons": [e.code]})
            continue
        cid = str(args.id_offset + n)
        res = process(script, tts, conversation_id=cid, seed=args.seed + n, transcriber=transcriber,
                      judge=judge, bleed_db=args.bleed_db, min_agreement=args.min_agreement)
        rec = {"script": path.name, "conversation_id": cid, "accepted": not res.reasons, "reasons": res.reasons,
               "agreement": round(res.agreement, 3), "wer": None if res.wer is None else round(res.wer, 3),
               "duration_s": round(res.rendered.duration, 1),
               "dropped": [p.item.id for p in res.rendered.placed if p.dropped]}
        report.append(rec)
        print(json.dumps(rec))
        if res.reasons and not args.keep_rejected:
            continue
        sp = script.speakers
        meta = {
            "conversation_type": script.conversation_type, "source": "turnsynth", "generator": script.meta.get("generator", "llm"),
            "tts": args.tts, "speaker_1_voice": res.rendered.voices["A"], "speaker_2_voice": res.rendered.voices["B"],
            "speaker_1_gender": sp["A"].get("gender", ""), "speaker_2_gender": sp["B"].get("gender", ""),
            "annotator_a": "generator-intent", "annotator_b": f"judge:{args.judge}" + (f"+asr:{args.asr}" if args.asr else "+script-text"),
            "annotator_c": "geometry", "agreement": res.agreement,
        }
        rows.append(to_row(res.rendered, res.tracks, meta))
        write_intent(res.rendered, out / "intent")
        if args.wav:
            import soundfile as sf

            import numpy as np

            (out / "wav").mkdir(parents=True, exist_ok=True)
            sf.write(out / "wav" / f"{cid}.wav", np.stack([res.rendered.audio[1], res.rendered.audio[2]], 1), res.rendered.sample_rate)
        if len(rows) >= args.shard_size:
            write_shard(rows, out / "parquet" / f"synth-{int(rows[0]['conversation_id']):06d}.parquet")
            rows = []
    if rows:
        write_shard(rows, out / "parquet" / f"synth-{int(rows[0]['conversation_id']):06d}.parquet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "render_report.jsonl").write_text("\n".join(json.dumps(r) for r in report) + "\n")


def cmd_stats(args) -> None:
    from turnsynth.stats import dataset_stats

    print(json.dumps(dataset_stats(args.dataset), indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="turnsynth")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("scripts", help="generate dialogue scripts with an LLM")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--types", default="", help="comma-separated TurnBench types (default: all six, round robin)")
    s.add_argument("--minutes", type=float, default=4.0)
    s.add_argument("--model", default="claude-opus-5-5")
    s.add_argument("--effort", default="medium")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_scripts)

    r = sub.add_parser("render", help="TTS + annotate + export TurnBench parquet")
    r.add_argument("scripts", help="a script JSON or a directory of them")
    r.add_argument("--tts", default="kokoro", choices=["kokoro", "dummy"])
    r.add_argument("--asr", default="", help="faster-whisper model for the judge's transcripts (e.g. base.en); empty = script text")
    r.add_argument("--device", default="auto")
    r.add_argument("--judge", default="rules", choices=["rules", "llm"])
    r.add_argument("--judge-model", default="claude-opus-5-5")
    r.add_argument("--bleed-db", type=float, default=None, help="cross-channel bleed level, e.g. -32")
    r.add_argument("--min-agreement", type=float, default=0.6)
    r.add_argument("--keep-rejected", action="store_true")
    r.add_argument("--id-offset", type=int, default=900000, help="numeric conversation ids start here")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shard-size", type=int, default=50)
    r.add_argument("--wav", action="store_true", help="also write stereo wavs for listening")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_render)

    t = sub.add_parser("stats", help="corpus stats via turnbench's gold code")
    t.add_argument("dataset")
    t.set_defaults(fn=cmd_stats)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
