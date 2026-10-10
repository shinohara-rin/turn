"""Score VAP probabilities on clean vs background-speech conditions.

Run from inside the TurnBench checkout:
    .venv/bin/python /path/to/bgspeech/score_bg.py --out <run_vap out dir>

For each condition it reports, on the evaluated subset:
- TurnBench EOT / INT at the official clean dev thresholds (a deployed model does not
  know the room is noisy), split by whether the event is on the contaminated
  ("user") channel or the clean one;
- the same after re-picking the threshold on that condition (oracle re-tune);
- false INT fires per minute on the user channel while the user is silent and the
  other speaker talks. TurnBench only counts FPs inside backchannel spans, so it
  cannot see the main deployment harm: the agent stops talking because the TV did.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turnbench.data import DEV_DATASET, conversation, resolve_dataset  # noqa: E402
from turnbench.gold import events_for_conversation  # noqa: E402
from turnbench.score import TaskScore, merge, score_task  # noqa: E402
from turnbench.sweep import (ConversationProbs, ProbsFile, SpeakerProbs, commit_events,  # noqa: E402
                             frame_count, operating_point, sweep)

FPS = 50.0
THETA = {"eot": 0.9161, "int": 0.8591}  # official VAP-oto dev operating point (clean)


def activity(conv, speaker, n, pad_s=0.3):
    m = np.zeros(n, bool)
    for a in "abc":
        for start, end, *_ in conv.annotations[(speaker, a)]:
            m[max(0, int((start - pad_s) * FPS)):min(n, int((end + pad_s) * FPS))] = True
    return m


def fmt(s: TaskScore):
    lat = s.latency().p50
    return f"R {s.recall:.3f} FP {s.fp_rate:.3f} p50 {lat:4.0f}ms (tp {s.tp} fn {s.fn} fp {s.fp} tn {s.tn})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    ds = resolve_dataset(DEV_DATASET, skip_audio=True)
    conds = [c for c in ("clean", "snr10", "snr5", "snr0", "gate5", "gate0") if (out / "probs" / c).is_dir()]
    ids = [cid for cid in plan["subset"] if all((out / "probs" / c / f"{cid}.npy").exists() for c in conds)]
    convs = {cid: conversation(ds, cid) for cid in ids}
    gold = {cid: events_for_conversation(convs[cid]) for cid in ids}
    print(f"{len(ids)} conversations, {sum(c.duration_s for c in convs.values()) / 3600:.2f} h; conditions {conds}")
    results = {}
    for cond in conds:
        res = {}
        p = {cid: np.load(out / "probs" / cond / f"{cid}.npy").astype(np.float64) for cid in ids}
        for task in ("eot", "int"):
            score = {k: TaskScore() for k in ("all", "user", "other")}
            entries = []
            for cid in ids:
                n = frame_count(convs[cid].duration_s, FPS)
                pp = p[cid][:n]
                if len(pp) < n:
                    pp = np.pad(pp, ((0, n - len(pp)), (0, 0)), mode="edge")
                sc = {s: (1 - pp[:, s - 1]) if task == "eot" else pp[:, s - 1] for s in (1, 2)}
                entries.append(ConversationProbs(conversation_id=cid, speaker_1=SpeakerProbs(prob=sc[1].tolist()),
                                                 speaker_2=SpeakerProbs(prob=sc[2].tolist())))
                ev = {s: commit_events(sc[s], FPS, THETA[task]) for s in (1, 2)}
                g = gold[cid]
                pos, neg, exc = ((g.eot_positive_events, g.eot_negative_spans, g.eot_excluded) if task == "eot"
                                 else (g.int_positive_events, g.int_negative_spans, g.int_excluded))
                merge(score["all"], score_task(pos, neg, ev, exc))
                user = plan["items"][cid]["user"]
                for key, spk in (("user", user), ("other", 3 - user)):
                    merge(score[key], score_task([e for e in pos if e.speaker == spk],
                                                 [s for s in neg if s.speaker == spk], ev, exc))
            rows = sweep(ProbsFile(schema_version=1, task=task, frame_rate_hz=FPS, probs=entries), ds)
            op = operating_point(rows)
            res[task] = {k: {"recall": v.recall, "fp_rate": v.fp_rate, "p50_ms": v.latency().p50,
                             "tp": v.tp, "fn": v.fn, "fp": v.fp, "tn": v.tn} for k, v in score.items()}
            res[task]["retuned"] = None if op is None else {"theta": op.theta, "recall": op.recall,
                                                            "fp_rate": op.fp_rate, "p50_ms": op.lat_p50}
            print(f"[{cond}] {task.upper()} all:   {fmt(score['all'])}")
            print(f"[{cond}] {task.upper()} user:  {fmt(score['user'])}")
            print(f"[{cond}] {task.upper()} other: {fmt(score['other'])}")
            if op is not None:
                print(f"[{cond}] {task.upper()} re-tuned θ {op.theta:.4f}: R {op.recall:.3f} FP {op.fp_rate:.3f}"
                      f" p50 {op.lat_p50:.0f}ms")
            else:
                print(f"[{cond}] {task.upper()} re-tuned: no θ reaches FP ≤ 0.10")
        # False INT fires on the user channel while the user is silent and the other speaker talks
        # ("agent gets cut off"), and the same for the clean channel as a control.
        fires = {"user": 0, "other": 0}
        minutes = {"user": 0.0, "other": 0.0}
        for cid in ids:
            n = frame_count(convs[cid].duration_s, FPS)
            pp = p[cid][:n]
            user = plan["items"][cid]["user"]
            for key, spk in (("user", user), ("other", 3 - user)):
                silent = ~activity(convs[cid], spk, len(pp))
                listening = silent & activity(convs[cid], 3 - spk, len(pp), pad_s=0.0)
                ev = commit_events(pp[:, spk - 1], FPS, THETA["int"])
                fires[key] += sum(1 for t in ev if listening[min(len(pp) - 1, int(t * FPS) - 1)])
                minutes[key] += listening.sum() / FPS / 60
        res["false_int_per_min"] = {k: fires[k] / minutes[k] for k in fires}
        print(f"[{cond}] false INT/min while silent & other talks: user {fires['user'] / minutes['user']:.2f}"
              f"  clean channel {fires['other'] / minutes['other']:.2f}")
        results[cond] = res
    (out / "scores.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
