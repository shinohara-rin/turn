# Usage: python tools/fish_timing.py out/*.pkl  (logs written by tools/fish_render.py)
# Fish API word times vs MMS_FA on the same audio. At pauses (a quiet run of
# >= 150 ms, 35 dB under the call's peak, between two words) the silence's
# own edges are the reference: word end vs where the sound stops, next word
# start vs where it starts again.
import pickle, sys, json
import numpy as np
HOP = 0.005
def frame_db(a, sr):
    h = int(HOP * sr); n = len(a) // h
    db = 10 * np.log10(np.mean(a[: n * h].reshape(n, h) ** 2, 1) + 1e-12)
    return db - db.max()
def longest_quiet(loud, i, j):
    best, cur, s = (0, 0, 0), 0, None
    for k in range(max(i, 0), min(j, len(loud))):
        if not loud[k]:
            if s is None: s = k
            if k - s + 1 > best[0]: best = (k - s + 1, s, k + 1)
        else: s = None
    return best
res = {"start": [], "end": [], "fish_end": [], "mms_end": [], "fish_on": [], "mms_on": [], "n_calls": 0, "fish_none": 0}
for path in sys.argv[1:]:
    for c in pickle.load(open(path, "rb"))["log"]:
        res["n_calls"] += 1
        if c["fish"] is None: res["fish_none"] += 1; continue
        F, M = np.array(c["fish"]), np.array(c["mms"])
        words = c["text"].split()
        res["start"] += list(F[:, 0] - M[:, 0]); res["end"] += list(F[:, 1] - M[:, 1])
        loud = frame_db(c["audio"], 24000) > -35
        for k in range(len(F) - 1):
            if F[k + 1, 0] - F[k, 1] <= 0 and M[k + 1, 0] - M[k, 1] <= 0.05: continue
            i = int((min(F[k, 1], M[k, 1]) - 0.15) / HOP); j = int((max(F[k + 1, 0], M[k + 1, 0]) + 0.15) / HOP)
            n, a, b = longest_quiet(loud, i, j)
            if n * HOP < 0.15: continue
            end, on = a * HOP, b * HOP
            res["fish_end"].append(F[k, 1] - end); res["mms_end"].append(M[k, 1] - end)
            res["fish_on"].append(F[k + 1, 0] - on); res["mms_on"].append(M[k + 1, 0] - on)
def s(x):
    x = np.array(x); return {"n": len(x), "median": round(float(np.median(x)), 3), "mean_abs": round(float(np.mean(np.abs(x))), 3),
                             "p90_abs": round(float(np.percentile(np.abs(x), 90)), 3), "within_50ms": round(float(np.mean(np.abs(x) <= 0.05)), 3)}
print(json.dumps({"calls": res["n_calls"], "fish_unmapped": res["fish_none"],
                  **{k: s(res[k]) for k in ("start", "end", "fish_end", "mms_end", "fish_on", "mms_on")}}, indent=1))
