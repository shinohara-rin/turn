"""Run the official TurnBench VAP baseline (oto checkpoint) on clean TB dev
conversations and on background-speech mixtures of them.

Run from inside a TurnBench checkout set up by vap/setup.sh:
    cd work/turnbench && .venv/bin/python /path/to/bgspeech/run_vap.py \
        --data <dir with TB dev parquet shards> --out <dir> [--n 12] [--shard 0/2]

Writes <out>/probs/<condition>/<cid>.npy (p_now, [T, 2] at 50 Hz, float16),
<out>/plan.json (subset, user channel, donors) and one listening sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, ".")
import mixing  # noqa: E402

CONDITIONS = {"clean": None, "snr10": 10.0, "snr5": 5.0, "snr0": 0.0,
              # oracle target-speaker gate: background only where the user is talking
              # (upper bound for any front end that silences non-user speech)
              "gate5": 5.0, "gate0": 0.0}


def make_plan(data_dir, n: int, seed: int) -> dict:
    ids = mixing.list_ids(data_dir)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(ids))
    subset, donors = sorted(order[:n]), order[n:]
    plan = {"seed": seed, "subset": subset, "donor_pool": donors, "items": {}}
    for i, cid in enumerate(subset):
        plan["items"][cid] = {"user": int(rng.integers(1, 3)),
                              "donors": [donors[(2 * i + k) % len(donors)] for k in range(3)]}
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = make_plan(a.data, a.n, a.seed)
        plan_path.write_text(json.dumps(plan, indent=1))
    k, m = map(int, a.shard.split("/"))
    mine = plan["subset"][k::m]
    conds = a.conditions.split(",")

    from baselines.vap.predict import _load_model, _step_extraction, SAMPLE_RATE
    model = _load_model("oto", "cpu")

    need_donors = {d for cid in mine for d in plan["items"][cid]["donors"]}
    donor_rows = {r["id"]: mixing.donor_dialogue(r, SAMPLE_RATE) for r in mixing.iter_rows(a.data, need_donors)}
    for row in mixing.iter_rows(a.data, set(mine)):
        cid, item = row["id"], plan["items"][row["id"]]
        user = item["user"]
        rng = np.random.default_rng(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
        chans = {s: mixing.resample_to(*row["audio"][s], SAMPLE_RATE) for s in (1, 2)}
        n = min(len(chans[1]), len(chans[2]))
        chans = {s: c[:n] for s, c in chans.items()}
        mask = mixing.activity_mask(row["annotations"], user, int(np.ceil(n / SAMPLE_RATE * 100)), 100.0)
        bg = mixing.background_track([donor_rows[d] for d in item["donors"]], n, SAMPLE_RATE, rng)
        for cond in conds:
            path = out / "probs" / cond / f"{cid}.npy"
            if path.exists():
                continue
            snr = CONDITIONS[cond]
            ch = dict(chans)
            if snr is not None:
                b = bg
                if cond.startswith("gate"):
                    b = bg * mixing.smooth_gate(mask, 100.0, n, SAMPLE_RATE)
                ch[user] = mixing.mix(chans[user], SAMPLE_RATE, b, snr, mask, bg_level=bg)
            wav = torch.from_numpy(np.stack([ch[1], ch[2]]))[None]
            t0 = time.time()
            with torch.inference_mode():
                p = _step_extraction(wav, model, "cpu")["p_now"][0].numpy()
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, p.astype(np.float16))
            print(f"{cid} {cond} {n / SAMPLE_RATE:.0f}s audio in {time.time() - t0:.0f}s", flush=True)
            if cid == plan["subset"][0] and cond == "snr5":
                import soundfile as sf
                s0, s1 = 60 * SAMPLE_RATE, 120 * SAMPLE_RATE
                sf.write(out / f"sample-{cid[:8]}-snr5-user{user}.wav",
                         np.stack([ch[1][s0:s1], ch[2][s0:s1]], 1), SAMPLE_RATE)


if __name__ == "__main__":
    main()
