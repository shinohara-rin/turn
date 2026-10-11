"""Cut eot-bench en rows into listening-test clips: trim leading silence, pad 2 s after the
1.5 s final span (3.5 s = harness max timeout), encode mp3, write 4 shuffled blocks of 100."""
import base64, io, json, random, subprocess, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf

src, out = sys.argv[1], sys.argv[2]
rows = pq.read_table(src).to_pylist()
clips = []
for r in rows:
    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    assert sr == 16000 and x.ndim == 1
    first = r["words"][0]["start"] if r["words"] else 0.0
    trim = max(0.0, min(first, r["silence_spans"][0]["start"]) - 0.3)
    trim = round(trim, 2)
    y = np.concatenate([x[int(trim * sr):], np.zeros(2 * sr, np.float32)])
    pk = np.abs(y).max()
    if pk > 0: y = y * min(1.0, 0.89 / pk)  # peak-normalise quiet rows (no clipping)
    wav = io.BytesIO(); sf.write(wav, y, sr, format="WAV", subtype="PCM_16")
    mp3 = subprocess.run(["ffmpeg", "-v", "error", "-f", "wav", "-i", "-", "-ac", "1", "-ar", "16000",
                          "-b:a", "32k", "-f", "mp3", "-"], input=wav.getvalue(), capture_output=True, check=True).stdout
    spans = [[round(s["start"] - trim, 3), round(s["end"] - trim, 3)] for s in r["silence_spans"]]
    msgs = r["messages"]
    agent = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    clips.append({"id": r["id"], "trim": trim, "dur": round(len(y) / sr, 3), "spans": spans,
                  "agent": agent, "mp3": base64.b64encode(mp3).decode()})
random.Random(20261011).shuffle(clips)
for b in range(4):
    blk = clips[b * 100:(b + 1) * 100]
    with open(f"{out}/block{b + 1}.json", "w") as f:
        json.dump({"block": b + 1, "clips": blk}, f, separators=(",", ":"))
meta = [{k: v for k, v in c.items() if k != "mp3"} | {"block": i // 100 + 1} for i, c in enumerate(clips)]
json.dump(meta, open(f"{out}/clips_meta.json", "w"), indent=0)
print(len(clips), sum(len(c["mp3"]) for c in clips) / 1e6, "MB b64")
