"""Frame targets for the completion head from complete_labels.py output.

For each labelled pause (speaker c, end time b, LLM score 0-10), every frame from b until anyone
vocalises again (labels/oto activity), capped at MAX_S, gets comp[t, c] = score / 10 with weight 1.
The target is constant over the silence, so it cannot be learned from silence length. All other
frames have weight 0. Writes /work/labels_comp/oto/{cid}.npz (comp, comp_w: float32 [T, 2]).

    modal run comp_targets.py --labels comp_train.json,comp_dev.json
    modal run comp_targets.py --labels ... --max-s 0.32 --out-dir labels_comp_onset   # onset only (r022)
"""
import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from common import VOLUMES, WORK, cpu_image, work  # noqa: E402

app = modal.App('ssl-turn-comp-targets')
FRAME_S = 0.08
MAX_S = 3.0


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=4096, timeout=1800)
def build(labels, max_s=MAX_S, out_dir='labels_comp'):
    import os
    import numpy as np
    os.makedirs(f'{WORK}/{out_dir}/oto', exist_ok=True)
    stats = dict(convs=0, pauses=0, frames=0, mean=0.0)
    for cid, items in labels.items():
        path = f'{WORK}/labels/oto/{cid}.npz'
        if not os.path.exists(path):
            continue
        act = np.load(path)['activity']
        T = len(act)
        comp, w = np.zeros((T, 2), np.float32), np.zeros((T, 2), np.float32)
        quiet = act.sum(1) == 0
        for spk, end, score, *_ in items:
            if score is None:
                continue
            # first frame whose centre is at/after the word end (activity marks frames by centre, prep.py);
            # r021/r022 were trained on ceil(end / FRAME_S), which skipped that frame for ~half the pauses
            c, i = spk - 1, max(0, int(np.ceil(end / FRAME_S - 0.5 - 1e-6)))
            stop = min(T, i + int(round(max_s / FRAME_S)))
            n = 0
            while i < stop and quiet[i]:
                comp[i, c], w[i, c] = score / 10, 1.0
                i += 1
                n += 1
            stats['pauses'] += n > 0
            stats['frames'] += n
            stats['mean'] += n * score / 10
        np.savez_compressed(f'{WORK}/{out_dir}/oto/{cid}.npz', comp=comp, comp_w=w)
        stats['convs'] += 1
    stats['mean'] /= max(1, stats['frames'])
    work.commit()
    return stats


@app.local_entrypoint()
def main(labels: str, max_s: float = MAX_S, out_dir: str = 'labels_comp'):
    merged = {}
    for f in labels.split(','):
        merged.update(json.load(open(f)))
    print(build.remote(merged, max_s, out_dir))
