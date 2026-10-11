# Usage: python tools/fish_render.py <scripts dir> <voice bank> <out dir> <whisper model>  (env TAGS=0, SEED0=n)
# Render scripts with --tts fish and keep the per-call log (fish vs MMS word times) for the timing comparison.
import json, os, pickle, sys
from pathlib import Path
import numpy as np, soundfile as sf
from turnsynth.fish import FishTTS
from turnsynth.judge import Transcriber
from turnsynth.pipeline import process
from turnsynth.script import parse

scripts, bank, out, asr = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
timings = sys.argv[5] if len(sys.argv) > 5 else "fish"
out.mkdir(parents=True, exist_ok=True)
tr = Transcriber(asr, device="cpu")
for n, p in enumerate(sorted(Path(scripts).glob("*.json"))):
    tts = FishTTS(bank, timings=timings, emotion_tags=os.environ.get("TAGS", "1") == "1")
    script = parse(json.loads(p.read_text()))
    res = process(script, tts, conversation_id=str(900000 + n), seed=n + int(os.environ.get("SEED0", "0")), transcriber=tr)
    r = res.rendered
    rec = {"script": p.name, "agreement": round(res.agreement, 3), "wer": round(res.wer, 4), "reasons": res.reasons,
           "duration_s": round(r.duration, 1), "voices": r.voices, "calls": len(tts.log),
           "audio_s": round(sum(c["seconds"] for c in tts.log), 1), "wall_s": round(sum(c["wall"] for c in tts.log), 1),
           "dropped": [q.item.id for q in r.placed if q.dropped]}
    print(json.dumps(rec), flush=True)
    sf.write(out / f"{p.stem}.wav", np.stack([r.audio[1], r.audio[2]], 1), r.sample_rate)
    pickle.dump({"log": tts.log, "rec": rec, "words": {q.item.id: [(w.text, w.start, w.end) for w in q.words] for q in r.placed if not q.dropped}},
                open(out / f"{p.stem}.pkl", "wb"))
