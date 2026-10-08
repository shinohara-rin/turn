"""Summarize head runs (best.json + attempts.jsonl per run dir) into a markdown table.

  python summarize_runs.py RUNS_DIR > RESULTS.md      # RUNS_DIR/<run-name>/{best.json,attempts.jsonl}
Per run: the epoch-12 (final) operating point chosen by heads.selection_key (max recall under
FPR<=0.10) and the best point over all epochs (what best.json records). Runs are grouped by
(encoder kind, head type) and averaged over seeds. Dev-set numbers, selected on the same dev set:
optimistic by construction; compare arms, not absolutes.
"""
import json, re, sys
from pathlib import Path
from statistics import mean

def key(row):
    s = row['score']; fp, rc = s['fp_rate'], s['recall']
    ok = fp is not None and fp <= .1
    return (ok, rc if ok and rc is not None else -1, -fp if fp is not None else -1e9, -(s['latency_ms']['p50'] or 0))

def load(run):
    rows = [json.loads(l) for l in (run / 'attempts.jsonl').read_text().splitlines() if l.strip()]
    best = json.loads((run / 'best.json').read_text())
    final_epoch = max(r['epoch'] for r in rows)
    final = max((r for r in rows if r['epoch'] == final_epoch), key=key)
    return final_epoch, final, best

def fmt(r):
    s = r['score']
    return dict(recall=s['recall'], fpr=s['fp_rate'], p50=s['latency_ms']['p50'], ok=s['fp_rate'] <= .1)

def main(root):
    groups = {}
    for run in sorted(Path(root).iterdir()):
        m = re.fullmatch(r'full-(frozen|continued)-(mlp|gru)-seed(\d+)', run.name)
        if not m or not (run / 'best.json').exists():
            continue
        epochs, final, best = load(run)
        groups.setdefault((m[1], m[2]), []).append((int(m[3]), epochs, fmt(final), fmt(best), best['epoch']))
    print('| encoder | head | seed | epochs | final: recall / FPR / p50 ms | best-epoch (e): recall / FPR / p50 ms |')
    print('|---|---|---|---|---|---|')
    for (enc, head), runs in sorted(groups.items()):
        for seed, ep, f, b, be in sorted(runs):
            print(f"| {enc} | {head} | {seed} | {ep} | {f['recall']:.4f} / {f['fpr']:.4f} / {f['p50']:.0f} | {b['recall']:.4f} / {b['fpr']:.4f} / {b['p50']:.0f} (e{be}) |")
        if len(runs) > 1:
            fm = {k: mean(r[2][k] for r in runs) for k in ('recall', 'fpr', 'p50')}
            bm = {k: mean(r[3][k] for r in runs) for k in ('recall', 'fpr', 'p50')}
            print(f"| {enc} | {head} | **mean** | | {fm['recall']:.4f} / {fm['fpr']:.4f} / {fm['p50']:.0f} | {bm['recall']:.4f} / {bm['fpr']:.4f} / {bm['p50']:.0f} |")

if __name__ == '__main__':
    main(sys.argv[1])
