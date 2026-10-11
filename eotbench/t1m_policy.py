"""turn-1-mini-style commit rule on eot-bench predictions, swept like the eot-bench harness.

    python eotbench/t1m_policy.py RUN_DIR/predictions.parquet [...]

The round-2 hybrid policy (/mnt/project-files/turn-1-mini-check/hybrid.py) commits a pause at
min(VAD deadline, first time the model score is >= th at or after t + mw_model). The eot-bench
harness policy is the same three knobs (threshold, action_delay, timeout) but fires at
max(action_delay, first crossing): a score that crossed early and fell back still fires at the
delay. Here `t1m` reads the score only once the delay has passed (hybrid.first_above). `harness`
reimplements the harness rule and must reproduce `eot-harness compute-metrics`.

Not carried over (no effect or no input on eot-bench): the 2.5 s confirmation event and the
resumption cancel (the first fire in a pause decides a cutoff), the other-speaker trigger (no
agent audio), and the VAD itself (pauses are the dataset's silence spans).
Metric as metrics._policy_sweep_rows: holds of 0.2-5 s cut off if the fire is before the span ends;
latency on true ends is the fire time, or the timeout when the model never fires.
"""
import sys

import numpy as np
import pandas as pd

EPS = 1e-9
THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2)
DELAYS = np.round(np.arange(0.2, 1.01, 0.1), 2)       # harness DEFAULT_ACTION_DELAYS
TIMEOUTS = np.round(np.arange(1.0, 3.51, 0.5), 2)     # harness DEFAULT_TIMEOUTS


def spans_of(pred):
    out = []
    for (rid, k), g in pred.sort_values('silence_dur').groupby(['id', 'span_index'], sort=False):
        dur = g.silence_dur.max()
        label = g.label.iloc[0]
        if label == 'hold' and not (0.2 - EPS <= dur <= 5.0 + EPS):
            continue
        out.append((label, dur, g.silence_dur.to_numpy(), g.p_eot.to_numpy()))
    return out


def fire_times(spans, rule):
    """[len(THRESHOLDS), len(DELAYS), n_spans] model fire time (inf = never)."""
    F = np.full((len(THRESHOLDS), len(DELAYS), len(spans)), np.inf)
    for j, (_, _, ts, p) in enumerate(spans):
        above = p[None, :] > THRESHOLDS[:, None] - EPS if rule == 't1m' else p[None, :] > THRESHOLDS[:, None]
        for i, d in enumerate(DELAYS):
            if rule == 'harness':  # first crossing anywhere, acted on no earlier than the delay
                first = np.where(above.any(1), ts[above.argmax(1)], np.inf)
                F[:, i, j] = np.maximum(d, first)
            else:                  # first grid time at/after the delay whose score is >= th
                ok = above & (ts[None, :] >= d - EPS)
                F[:, i, j] = np.where(ok.any(1), ts[ok.argmax(1)], np.inf)
    return F


def sweep(spans, rule):
    F = fire_times(spans, rule)
    hold = np.array([s[0] == 'hold' for s in spans])
    dur = np.array([s[1] for s in spans])
    rows = []
    for a, th in enumerate(THRESHOLDS):
        for b, d in enumerate(DELAYS):
            f = F[a, b]
            model_cut = np.isfinite(f[hold]) & (dur[hold] > f[hold] + EPS)
            for to in TIMEOUTS:
                if to + EPS < d:
                    continue
                cut = (dur[hold] > to + EPS) | model_cut
                detect = np.isfinite(f[~hold]) & (f[~hold] <= to + EPS)
                lat = np.where(detect, f[~hold], to)
                rows.append(dict(threshold=th, action_delay=d, timeout=to, cutoff_rate=cut.mean(),
                                 mean_latency=lat.mean()))
    return pd.DataFrame(rows)


def operating_points(df):
    out = {}
    for b in (0.05, 0.10):
        f = df[df.cutoff_rate <= b + EPS]
        out[f'lat@{int(b * 100)}%'] = f.mean_latency.min() if len(f) else np.nan
    for b in (0.3, 0.6):
        f = df[df.mean_latency <= b + EPS]
        out[f'cut@{int(b * 1000)}ms'] = f.cutoff_rate.min() if len(f) else np.nan
    return out


def main():
    rows = []
    for path in sys.argv[1:]:
        spans = spans_of(pd.read_parquet(path))
        for rule in ('harness', 't1m'):
            rows.append(dict(run=path.split('/')[-2], rule=rule, **operating_points(sweep(spans, rule))))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
