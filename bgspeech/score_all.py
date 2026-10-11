"""Score VAP and ssl_turn on clean vs background-speech TurnBench dev mixtures.

Run with the TurnBench checkout's venv (plus `modal`, which ssl_turn/pipeline/score.py
imports), from inside the checkout:
    .venv/bin/python /path/to/bgspeech/score_all.py --vap <run_vap out> --ssl <probs.npz> --out scores.json

Thresholds are picked once per model on the *clean* audio of the same conversations
(highest recall at FP <= 0.10, TurnBench's dev rule) and then held fixed: a deployed
model does not know someone switched the TV on. Reported per condition:
- EOT / INT recall, FP rate and p50 latency, overall and split by channel ("user" =
  the channel that received the background, "other" = the clean one);
- false INT fires per minute on each channel while that speaker is silent and the
  other one talks ("the agent gets cut off by the TV"). TurnBench itself only counts
  FPs inside annotated backchannel / pause spans, so it misses most of these;
- the operating point re-picked on that condition (what re-tuning alone could recover).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl_turn"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl_turn" / "pipeline"))
from turnbench.data import DEV_DATASET, conversation, resolve_dataset  # noqa: E402
from turnbench.gold import events_for_conversation  # noqa: E402
from turnbench.score import TaskScore, merge, score_task  # noqa: E402

CONDS = ("clean", "snr10", "snr5", "snr0", "gate5", "gate0", "snrm5", "near5", "near0", "tv5", "tv0", "music0")
FP_BUDGET = 0.10


def commit(p, fps, theta, refractory_s, recommit_s):
    from score import commit as ssl_commit  # ssl_turn/pipeline/score.py: rising edge + optional re-commit
    return ssl_commit(p, fps, theta, refractory_s, recommit_s)


def activity(conv, speaker, n, fps, pad_s):
    m = np.zeros(n, bool)
    for a in "abc":
        for start, end, *_ in conv.annotations[(speaker, a)]:
            m[max(0, int((start - pad_s) * fps)):min(n, int(np.ceil((end + pad_s) * fps)))] = True
    return m


class Model:
    """Per-conversation [T, 2] score tracks for each condition and task, plus the commit rule."""

    def __init__(self, name, fps, tracks, refractory, recommit):
        self.name, self.fps, self.tracks = name, fps, tracks  # tracks[cond][task][cid]
        self.refractory, self.recommit = refractory, recommit  # per task

    def events(self, cond, task, cid, theta):
        p = self.tracks[cond][task][cid]
        return {s + 1: commit(p[:, s], self.fps, theta, self.refractory[task], self.recommit[task]) for s in (0, 1)}


def score(model, cond, task, theta, ids, gold, plan):
    out = {k: TaskScore() for k in ("all", "user", "other")}
    for cid in ids:
        ev = model.events(cond, task, cid, theta)
        g = gold[cid]
        pos, neg, exc = ((g.eot_positive_events, g.eot_negative_spans, g.eot_excluded) if task == "eot"
                         else (g.int_positive_events, g.int_negative_spans, g.int_excluded))
        merge(out["all"], score_task(pos, neg, ev, exc))
        user = plan["items"][cid]["user"]
        for key, spk in (("user", user), ("other", 3 - user)):
            merge(out[key], score_task([e for e in pos if e.speaker == spk], [s for s in neg if s.speaker == spk],
                                       ev, exc))
    return out


def thetas_for(model, cond, task, ids):
    pooled = np.concatenate([model.tracks[cond][task][c].ravel() for c in ids])
    return np.unique(np.concatenate([np.quantile(pooled, np.linspace(0, 1, 129)), np.arange(1, 100) * 0.01]))


def op_point(model, cond, task, ids, gold, plan):
    best = None
    for th in thetas_for(model, cond, task, ids):
        s = score(model, cond, task, float(th), ids, gold, plan)["all"]
        if s.fp_rate <= FP_BUDGET and (best is None or s.recall > best[1].recall):
            best = (float(th), s)
    return best


def summary(s: TaskScore):
    return dict(recall=s.recall, fp_rate=s.fp_rate, p50_ms=s.latency().p50, tp=s.tp, fn=s.fn, fp=s.fp, tn=s.tn)


def false_int_rate(model, cond, theta, ids, convs, plan):
    """INT fires per minute on a channel while its speaker is silent and the other talks."""
    fires, minutes = {"user": 0, "other": 0}, {"user": 0.0, "other": 0.0}
    for cid in ids:
        ev = model.events(cond, "int", cid, theta)
        n = len(model.tracks[cond]["int"][cid])
        user = plan["items"][cid]["user"]
        for key, spk in (("user", user), ("other", 3 - user)):
            listening = ~activity(convs[cid], spk, n, model.fps, 0.3) & activity(convs[cid], 3 - spk, n, model.fps, 0.0)
            fires[key] += sum(1 for t in ev[spk] if listening[min(n - 1, int(round(t * model.fps)) - 1)])
            minutes[key] += listening.sum() / model.fps / 60
    return {k: fires[k] / minutes[k] for k in fires}


def load_vap(out_dir, ids):
    tracks = {}
    for cond in CONDS:
        d = Path(out_dir) / "probs" / cond
        if d.is_dir() and all((d / f"{c}.npy").exists() for c in ids):
            p = {c: np.load(d / f"{c}.npy").astype(np.float64) for c in ids}
            tracks[cond] = {"eot": {c: 1 - v for c, v in p.items()}, "int": p}
    return Model("vap", 50.0, tracks, {"eot": 2.0, "int": 2.0}, {"eot": None, "int": None})


def load_ssl(npz, ids, member):
    from score import score_variants
    z = np.load(npz)
    tracks = {}
    for cond in CONDS:
        pre = f"{member}@{cond}/tbdev/"
        if all(pre + f"{c}/post" in z.files for c in ids):
            tracks[cond] = {"eot": {}, "int": {}}
            for c in ids:
                v = score_variants(z[pre + f"{c}/post"].astype(np.float32), z[pre + f"{c}/silent"].astype(np.float32),
                                   z[pre + f"{c}/fine"].astype(np.float32))
                tracks[cond]["eot"][c], tracks[cond]["int"][c] = v["eot_q"], v["int_nobc"]
    # The current best commit policy (ssl_turn/HANDOFF.md): 0.5 s refractory, EOT re-commit after 1 s.
    return Model(member, 12.5, tracks, {"eot": 0.5, "int": 0.5}, {"eot": 1.0, "int": None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vap")
    ap.add_argument("--ssl")
    ap.add_argument("--ssl-models", default="fine1_bal1_s1,fine1_bal1_s2")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    ids = list(plan["subset"])
    ds = resolve_dataset(DEV_DATASET, skip_audio=True)
    convs = {c: conversation(ds, c) for c in ids}
    gold = {c: events_for_conversation(convs[c]) for c in ids}
    models = []
    if a.vap:
        models.append(load_vap(a.vap, ids))
    if a.ssl:
        models += [load_ssl(a.ssl, ids, m) for m in a.ssl_models.split(",")]
    print(f"{len(ids)} conversations, {sum(c.duration_s for c in convs.values()) / 3600:.2f} h")
    results = {}
    for model in models:
        res = results[model.name] = {}
        theta = {}
        for task in ("eot", "int"):
            th, s = op_point(model, "clean", task, ids, gold, plan)
            theta[task] = th
        res["theta_clean"] = theta
        for cond in CONDS:
            if cond not in model.tracks:
                continue
            r = res[cond] = {}
            for task in ("eot", "int"):
                sc = score(model, cond, task, theta[task], ids, gold, plan)
                r[task] = {k: summary(v) for k, v in sc.items()}
                rt = op_point(model, cond, task, ids, gold, plan)
                r[task]["retuned"] = None if rt is None else dict(theta=rt[0], **summary(rt[1]))
            r["false_int_per_min"] = false_int_rate(model, cond, theta["int"], ids, convs, plan)
            e, i, f = r["eot"], r["int"], r["false_int_per_min"]
            print(f"{model.name:>14} {cond:>6} | EOT all {e['all']['recall']:.3f}/{e['all']['fp_rate']:.3f}"
                  f" user {e['user']['recall']:.3f} p50 {e['user']['p50_ms']:4.0f}ms other {e['other']['recall']:.3f}"
                  f" | INT all {i['all']['recall']:.3f}/{i['all']['fp_rate']:.3f} user-FP {i['user']['fp_rate']:.3f}"
                  f" | falseINT/min user {f['user']:.2f} other {f['other']:.2f}"
                  f" | retuned EOT {e['retuned']['recall'] if e['retuned'] else float('nan'):.3f}"
                  f" INT {i['retuned']['recall'] if i['retuned'] else float('nan'):.3f}", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
