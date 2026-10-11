"""Score human listening-test presses with eot-bench's harness rules.

Input: trial records (from the artifact store or the page's saved file) + clips_meta.json.
The caller keeps talking after a press, so every hold gets its own decision. A press commits a
reply that, like any harness policy, lands at least DELAY (0.2 s) into a pause, in the first pause
still open then. Cut-off = reply inside a 0.2-5 s hold (or the hold outlasting the timeout); latency
on the true end from final-silence onset, capped at the timeout. Sweeping the timeout T gives a
human + silence-timeout curve.
"""
import argparse, glob, json, math
import numpy as np

MIN_HOLD, MAX_HOLD, DELAY = 0.2, 5.0, 0.2


def fires(presses, spans):
    """Each press -> (span index, fire time): first pause still open at max(press, start + DELAY)."""
    out, last = [], len(spans) - 1
    for p in presses or []:
        for j, (a, b) in enumerate(spans):
            f = max(p, a + DELAY)
            if f < (math.inf if j == last else b - 1e-6):
                out.append((j, f))
                break
    return out


def score(trials, meta, timeout=3.5):
    """Every 0.2-5 s hold is scored (the caller keeps talking after a press), as in the harness."""
    holds = cut = 0
    lat, raw = [], []
    for t in trials:
        spans = meta[t["id"]]["spans"]
        last = len(spans) - 1
        fs = fires(t.get("presses", [t["press"]] if t.get("press") is not None else []), spans)
        for j, (a, b) in enumerate(spans[:last]):
            if not (MIN_HOLD - 1e-6 <= b - a <= MAX_HOLD + 1e-6):
                continue
            holds += 1
            cut += any(k == j for k, _ in fs) or (b - a) > timeout + 1e-6
        e = spans[last][0]
        fe = [f for k, f in fs if k == last]
        lat.append(min(timeout, min(fe) - e) if fe else timeout)
        rp = [p - e for p in t.get("presses", []) if p >= e - 1.5]
        if rp:
            raw.append(min(rp))
    return dict(n=len(trials), holds=holds, cut=cut, cut_rate=cut / max(holds, 1), n_eot=len(lat),
                mean_lat=float(np.mean(lat)) if lat else float("nan"),
                med_lat=float(np.median(lat)) if lat else float("nan"),
                first_press_vs_end_p10_50_90=[float(x) for x in np.percentile(raw, [10, 50, 90])] if raw else None)


def load(paths):
    out = []
    for p in paths:
        j = json.load(open(p))
        if isinstance(j, dict) and "trials" in j:
            out += j["trials"]
        elif isinstance(j, dict) and "id" in j:
            out.append(j.get("data", j))
        elif isinstance(j, list):
            out += j
    return [t.get("data", t) for t in out]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="trial JSON files or a glob")
    ap.add_argument("--meta", default="humantest/clips_meta.json")
    ap.add_argument("--skip-first", type=int, default=5, help="warm-up trials dropped per block")
    a = ap.parse_args()
    meta = {c["id"]: c for c in json.load(open(a.meta))}
    trials = load(sorted(p for f in a.files for p in glob.glob(f)))
    trials = [t for t in trials if t.get("i", 99) >= a.skip_first]
    print(json.dumps(score(trials, meta), indent=1))
    print("timeout  cut%   mean_lat_ms")
    for T in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        s = score(trials, meta, T)
        print(f"{T:6.2f} {100 * s['cut_rate']:6.1f} {1000 * s['mean_lat']:8.0f}")
