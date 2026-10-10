"""Command line.

    turnsynth scripts --n 12 --minutes 4 --out out/scripts          # LLM: pass 1 + pass 2 + filter
    turnsynth render out/scripts --tts kokoro --asr base.en --judge llm --out out/synth
    turnsynth voices libritts/data/dev.clean/*.parquet --out voices     # prompt bank for --tts indextts
    turnsynth voices emilia/JA/JA-B000000.tar --source emilia --language ja --out voices-ja
    turnsynth scripts --language ja ...; turnsynth render ... --voice-bank voices-ja --asr small
    turnsynth render out/scripts --tts indextts --index-model-dir ckpt/IndexTTS-2.5 --voice-bank voices ...
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
        script, attempts = generate_script(llm, ctype, minutes=args.minutes, seed=seed, language=args.language)
        log.append({"index": i, "type": ctype, "language": args.language, "seed": seed, "accepted": script is not None, "attempts": attempts})
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
    tts_kwargs = {}
    if args.tts == "kokoro" and args.device != "auto":
        tts_kwargs = {"device": args.device}
    if args.tts == "indextts":
        tts_kwargs = {"model_dir": args.index_model_dir, "bank": args.voice_bank, "version": args.index_version,
                      "device": None if args.device == "auto" else args.device, "entrain": args.entrain,
                      "context": not args.no_dialogue_context, "pass_mode": args.index_pass,
                      "emo_alpha": args.emo_alpha, "text_emotion": args.text_emotion}
    tts = make_tts(args.tts, **tts_kwargs)
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
            "annotator_c": "geometry", "agreement": res.agreement, "language": script.language,
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


def cmd_voices(args) -> None:
    from turnsynth.voicebank import build_aishell3, build_emilia, build_globe, build_libritts

    if args.source == "libritts":
        bank = build_libritts(args.shards, args.out, per_gender=args.per_gender, seed=args.seed)
    elif args.source == "aishell3":
        bank = build_aishell3(args.shards[0], args.out, per_gender=args.per_gender, seed=args.seed)
    elif args.source == "emilia":
        bank = build_emilia(args.shards, args.out, language=args.language, per_gender=args.per_gender)
    else:
        bank = build_globe(args.shards, args.out, per_gender=args.per_gender,
                           accents=args.accents.split(",") if args.accents else None, seed=args.seed)
    genders = [v["gender"] for v in bank.values()]
    print(f"{len(bank)} voices ({genders.count('female')} female, {genders.count('male')} male) -> {args.out}")


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
    s.add_argument("--language", default="en", choices=["en", "ja", "zh"],
                   help="ja/zh need --tts indextts and a multilingual --asr (e.g. small) at render time")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_scripts)

    r = sub.add_parser("render", help="TTS + annotate + export TurnBench parquet")
    r.add_argument("scripts", help="a script JSON or a directory of them")
    r.add_argument("--tts", default="kokoro", choices=["kokoro", "indextts", "dummy"])
    r.add_argument("--index-model-dir", default="checkpoints/IndexTTS-2.5", help="IndexTTS weights (with config.yaml)")
    r.add_argument("--index-version", default="2.5", choices=["2.5", "2"])
    r.add_argument("--voice-bank", default="voices", help="directory with voices.json (turnsynth voices); several comma-separated, e.g. one per language")
    r.add_argument("--no-dialogue-context", action="store_true",
                   help="IndexTTS ablation: per-chunk calls from the fixed bank clip, as in MultiTalk")
    r.add_argument("--index-pass", default="floor", choices=["turn", "speaker", "floor"],
                   help="IndexTTS: one call per line; each speaker's lines read in long passes and cut apart; "
                        "or the same but a pass ends when the other speaker takes the floor")
    r.add_argument("--entrain", type=float, default=0.0,
                   help="IndexTTS: emotion strength borrowed from the partner's last line when the script gives none")
    r.add_argument("--emo-alpha", type=float, default=0.8,
                   help="IndexTTS: strength of script (or text-derived) emotion vectors; the rest comes from the speaker prompt")
    r.add_argument("--text-emotion", action="store_true",
                   help="IndexTTS: for lines without a script emotion, read one off the text with IndexTTS's QwenEmotion")
    r.add_argument("--asr", default="", help="faster-whisper model for the judge's transcripts (e.g. base.en; a multilingual one such as small for ja/zh); empty = script text")
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

    v = sub.add_parser("voices", help="build a voice-prompt bank (English, Japanese or Chinese voices)")
    v.add_argument("shards", nargs="+", help="parquet shards, Emilia tars, or the AISHELL-3 root directory")
    v.add_argument("--source", default="libritts", choices=["libritts", "globe", "aishell3", "emilia"],
                   help="libritts: mythicinfinity/libritts_r shards (clean English, default); globe: MushanW/GLOBE_V2 "
                        "shards; aishell3: AISHELL/AISHELL-3 root (zh); emilia: TTS-AGI/emilia-yodas tars (--language)")
    v.add_argument("--language", default="ja", choices=["ja", "zh", "en"], help="language of --source emilia tars")
    v.add_argument("--out", required=True)
    v.add_argument("--per-gender", type=int, default=40)
    v.add_argument("--accents", default="", help="comma-separated substrings of GLOBE accent names to keep")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(fn=cmd_voices)

    t = sub.add_parser("stats", help="corpus stats via turnbench's gold code")
    t.add_argument("dataset")
    t.set_defaults(fn=cmd_stats)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
