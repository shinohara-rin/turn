# Usage: python tools/score_audio.py <stereo audio> <script.json> <whisper model> <label>
# Score a stereo render against its script, the same way for any TTS: per-channel
# VAD -> faster-whisper -> error rate vs that speaker's script words (in order),
# and SQUIM (PESQ/STOI/SI-SDR) on 2 s windows of each channel's speech.
import json, sys
import numpy as np, soundfile as sf, librosa, torch, torchaudio
from turnsynth.judge import Transcriber
from turnsynth.lang import spoken, units
from turnsynth.script import parse
from turnsynth.vad import segments

def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

class Seg:
    def __init__(s, a, b): s.start, s.end = a, b

squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
def squim_windows(x, sr):
    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    out = []
    for a, b in segments(x, 16000, merge_gap_s=0.5):
        clip = x[int(a * 16000): int(b * 16000)]
        for i in range(0, max(1, len(clip) - 32000 + 1), 16000):
            w = clip[i: i + 32000]
            if len(w) < 16000: continue
            with torch.inference_mode():
                stoi, pesq, sisdr = [t.item() for t in squim(torch.from_numpy(w.astype(np.float32))[None])]
            out.append((pesq, stoi, sisdr))
    return np.array(out)

def main(audio_path, script_path, asr, label):
    script = parse(json.load(open(script_path)))
    lang = script.language
    x, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    tr = Transcriber(asr, device="cpu")
    ref_all, hyp_all, wins = [], [], []
    for ch, spk in ((0, "A"), (1, "B")):
        a = x[:, ch]
        segs = [Seg(*s) for s in segments(a, sr)]
        hyp = tr.transcribe(a, sr, segs, lang)
        ref = spoken([w for it in script.items if it.speaker == spk for w in it.words], lang)
        ref_all += units(ref, lang); hyp_all += units(" ".join(hyp), lang)
        wins.append(squim_windows(a, sr))
    w = np.concatenate(wins)
    rec = {"label": label, "lang": lang, "err": round(edit(ref_all, hyp_all) / len(ref_all), 4),
           "pesq_med": round(float(np.median(w[:, 0])), 2), "pesq_p10": round(float(np.percentile(w[:, 0], 10)), 2),
           "stoi_med": round(float(np.median(w[:, 1])), 3), "sisdr_med": round(float(np.median(w[:, 2])), 1),
           "windows": len(w), "under_2.9": round(float(np.mean(w[:, 0] < 2.9)), 3)}
    print(json.dumps(rec), flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:5])
