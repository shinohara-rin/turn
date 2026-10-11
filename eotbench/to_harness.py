"""Turn modal_eot.py score tracks into eot-bench predictions (github.com/livekit/eot-bench).

    python eotbench/to_harness.py tracks.npz EN_DIR [--models a,b] [--variant eot_q] [--tag r019]

EN_DIR is an eot-bench span-set directory, e.g. the committed
output/livekit__eot-bench-data__validation__min_silence_100ms/en (its span_set.parquet defines the
spans). For each (model, variant) this writes EN_DIR/ssl_turn__{tag}_{model}_{variant}/predictions.parquet
and manifest.json; then run `eot-harness compute-metrics` and `eot-harness compare-models`.

Timing: track frame t is the causal score of audio before (t + 1) * 80 ms, so the score at a grid
timestamp ts (eot-bench's 0.1 s grid over each silence span, harness io._time_grid) is frame
floor(ts / 0.08) - 1, or 0 before the first frame exists.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_S = 0.08
STEP = 0.1        # eot-bench default inference_interval
EPS = 1e-9        # eot-bench io._EPS


def time_grid(start, end, step=STEP):
    """eot-bench io._time_grid."""
    n = max(0, int(math.floor((end - start) / step + EPS)))
    values = [round(start + i * step, 6) for i in range(n + 1)]
    if values and values[-1] < end - EPS:
        values.append(round(end, 6))
    elif not values:
        values = [round(start, 6), round(end, 6)]
    return values


def predictions(spans, tracks):
    rows = []
    for s in spans.itertuples():
        track = tracks[s.id].astype(np.float32)
        for ts in time_grid(s.start, s.end):
            t = int(math.floor(ts / FRAME_S + 1e-6)) - 1
            p = float(track[min(t, len(track) - 1)]) if t >= 0 else 0.0
            rows.append(dict(id=s.id, language=s.language, span_index=int(s.span_index), timestamp=ts,
                             silence_dur=round(ts - s.start, 6), p_eot=p, label=s.label))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks')
    ap.add_argument('span_dir')
    ap.add_argument('--models', default='')
    ap.add_argument('--variants', default='eot_q')
    ap.add_argument('--tag', default='r019')
    a = ap.parse_args()
    span_dir = Path(a.span_dir)
    spans = pd.read_parquet(span_dir / 'span_set.parquet')
    span_manifest = json.loads((span_dir / 'span_set_manifest.json').read_text())
    z = np.load(a.tracks)
    keys = {}
    for k in z.files:
        model, variant, rid = k.split('/', 2)
        keys.setdefault((model, variant), {})[rid] = k
    models = a.models.split(',') if a.models else sorted({m for m, _ in keys})
    for model in models:
        for variant in a.variants.split(','):
            tracks = {rid: z[k] for rid, k in keys[(model, variant)].items()}
            missing = set(spans.id) - set(tracks)
            assert not missing, f'{len(missing)} turns without a track'
            run_id = f'ssl_turn__{a.tag}_{model}_{variant}'
            out = span_dir / run_id
            out.mkdir(exist_ok=True)
            df = predictions(spans, tracks)
            df.to_parquet(out / 'predictions.parquet', index=False)
            manifest = dict(harness_version=span_manifest.get('harness_version'),
                            span_set_id=span_manifest.get('span_set_id'), model_run_id=run_id,
                            dataset=span_manifest['dataset'],
                            model=dict(adapter='turn/eotbench/modal_eot.py', adapter_id=run_id,
                                       inference_interval=STEP, display_name=f'ssl_turn {a.tag} {model} ({variant})',
                                       note='streaming FastConformer + floor head; agent channel silent'),
                            language=spans.language.iloc[0], inference_interval=STEP)
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(run_id, len(df), 'rows')


if __name__ == '__main__':
    main()
