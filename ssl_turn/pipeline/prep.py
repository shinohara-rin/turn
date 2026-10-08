"""CPU prep on Modal: frozen actor split, otoSpeech labels, causal 24 kHz audio, TurnBench dev audio.

Outputs on the ssl-turn-work volume:
  /work/split.json                          attempt 1's actor split (seed 20261007), recomputed
  /work/audio/{oto,tbdev}/{cid}.npy         float16 [N, 2] at 24 kHz, causal one-sided FIR
  /work/labels/oto/{cid}.npz                floor/future/act targets + activity on the 80 ms grid
  /work/gold/oto/{cid}.json                 single-annotator TurnBench-style events (for oto-dev scoring)

    modal run prep.py::main --train 32
"""
import json

import modal

from common import OTO, TB_DEV, VOLUMES, cpu_image, setup_path, work

app = modal.App('ssl-turn-prep')
FRAME_S = 0.08
SR = 24000
ACTIVITY_EXCLUDED = ('Awkward Silence', 'Non-Speech Noise', 'Channel Bleed')


def causal_resample(x, sr_in, sr_out=SR):
    """One-sided FIR (same design as attempt 1's encoder.causal_resample): output n uses
    only input up to its own time; the filter's group delay is kept, never compensated."""
    import numpy as np
    from math import gcd
    from scipy.signal import firwin, upfirdn
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    half = 10 * max(up, down)
    taps = firwin(2 * half + 1, 1 / max(up, down), window=('kaiser', 5.0)) * up
    count = (len(x) * up + down - 1) // down
    return upfirdn(taps, x, up=up, down=down, axis=0)[:count].astype(np.float32)


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=4096, timeout=1800)
def make_split():
    import hashlib, os
    import numpy as np
    rows = []
    for d in sorted(os.listdir(OTO)):
        p = f'{OTO}/{d}/metadata.json'
        if os.path.exists(p):
            rows.append(dict(json.load(open(p)), _dir=d))
    actors = sorted({r[f'speaker_{s}_actor_id'] for r in rows for s in (1, 2)})
    rng = np.random.default_rng(20261007)
    rng.shuffle(actors)
    n = len(actors)
    assign = {a: ('train' if i < int(n * .6) else 'dev' if i < int(n * .8) else 'gate') for i, a in enumerate(actors)}
    splits = {k: [] for k in ('train', 'dev', 'gate', 'excluded_cross_partition')}
    for r in rows:
        a, b = (assign[r[f'speaker_{s}_actor_id']] for s in (1, 2))
        splits[a if a == b else 'excluded_cross_partition'].append(r['_dir'])
    for k in splits:
        splits[k].sort(key=lambda x: hashlib.sha256(('pilot-v1:' + x).encode()).hexdigest())
    meta = {r['_dir']: {k: v for k, v in r.items() if k != '_dir'} for r in rows}
    out = dict(seed=20261007, splits=splits, conversation_type={k: v['conversation_type'] for k, v in meta.items()})
    counts = {k: len(v) for k, v in splits.items()}
    if counts != dict(train=131, dev=16, gate=20, excluded_cross_partition=253):
        raise ValueError(f'split does not reproduce attempt 1: {counts}')
    os.makedirs('/work', exist_ok=True)
    json.dump(out, open('/work/split.json', 'w'), indent=1)
    work.commit()
    return counts


def parse_srt(path):
    import re
    import srt
    segs = []
    for e in srt.parse(open(path).read()):
        m = re.match(r'\[([^]]+)\]\s*(.*)', e.content, re.S)
        if not m:
            raise ValueError(f'unknown annotation format {path}')
        segs.append((e.start.total_seconds(), e.end.total_seconds(), m[1]))
    return segs


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=1800)
def oto_item(cid):
    """Labels + causal 24 kHz audio for one otoSpeech conversation (train/dev only)."""
    import os
    from dataclasses import asdict
    import numpy as np
    import soundfile as sf
    setup_path()
    from turnbench.gold import CANONICAL, TURN_CANONICAL, ConsensusEvent, ConsensusViews, build_conversation_events
    import labels as lb
    if os.path.exists(f'/work/labels/oto/{cid}.npz') and os.path.exists(f'/work/audio/oto/{cid}.npy'):
        return cid, 'cached'
    src = f'{OTO}/{cid}'
    chans, sr = [], None
    for s in (1, 2):
        x, sr = sf.read(f'{src}/speaker_{s}_audio.wav', dtype='float32')
        chans.append(x)
    n = min(map(len, chans))
    audio = causal_resample(np.stack([c[:n] for c in chans], 1), sr)
    duration = n / sr
    T = int(np.floor(duration / FRAME_S))
    times = (np.arange(T) + 1) * FRAME_S
    segments, turn, fine = [], [], []
    activity = np.zeros((T, 2), np.float32)
    for s in (1, 2):
        for start, end, label in parse_srt(f'{src}/speaker_{s}_annotation_a.srt'):
            if label in CANONICAL:
                fine.append(ConsensusEvent(s, start, end, CANONICAL[label]))
                segments.append((s, start, end, CANONICAL[label]))
            if label in TURN_CANONICAL:
                turn.append(ConsensusEvent(s, start, end, 'Turn'))
            if label not in ACTIVITY_EXCLUDED:
                activity[(times - FRAME_S / 2 >= start) & (times - FRAME_S / 2 < end), s - 1] = 1
    events = asdict(build_conversation_events(ConsensusViews(turn, [], fine, [])))
    y = lb.floor_targets(times, segments, events)
    future, future_w = lb.floor_projection(y['floor'], y['floor_w'])
    for d in ('audio/oto', 'labels/oto', 'gold/oto'):
        os.makedirs(f'/work/{d}', exist_ok=True)
    np.save(f'/work/audio/oto/{cid}.npy', audio.astype(np.float16))
    np.savez_compressed(f'/work/labels/oto/{cid}.npz', floor=y['floor'], floor_w=y['floor_w'], act=y['act'],
                        act_w=y['act_w'], future=future, future_w=future_w, activity=activity, times=times)
    json.dump(dict(duration_s=duration, events=events), open(f'/work/gold/oto/{cid}.json', 'w'))
    work.commit()
    return cid, round(duration, 1), sr, int(T)


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=16384, timeout=3600)
def tbdev_audio(cids):
    """Decode + causally resample TurnBench dev conversations (audio only; gold stays in the scorer)."""
    import os
    import numpy as np
    from turnbench.data import conversation, resolve_dataset
    ds = resolve_dataset(TB_DEV)
    os.makedirs('/work/audio/tbdev', exist_ok=True)
    out = []
    for cid in cids:
        conv = conversation(ds, cid)
        chans = [conv.audio(s) for s in (1, 2)]
        sr = chans[0][1]
        n = min(len(c[0]) for c in chans)
        audio = causal_resample(np.stack([c[0][:n] for c in chans], 1), sr)
        np.save(f'/work/audio/tbdev/{cid}.npy', audio.astype(np.float16))
        out.append((cid, sr, round(conv.duration_s, 2)))
    work.commit()
    return out


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=1800)
def tbdev_ids():
    from turnbench.data import conversation_ids, resolve_dataset
    from turnbench.durations import load_durations
    ds = resolve_dataset(TB_DEV, skip_audio=True)
    return conversation_ids(ds)


@app.local_entrypoint()
def main(train: int = 32, dev: int = 16, tbdev: bool = True):
    counts = make_split.remote()
    print('split', counts)
    import io
    buf = io.BytesIO()
    for chunk in work.read_file('split.json'):
        buf.write(chunk)
    split = json.loads(buf.getvalue())
    cids = split['splits']['train'][:train] + split['splits']['dev'][:dev]
    for r in oto_item.map(cids):
        print('oto', r)
    if not tbdev:
        return
    ids = tbdev_ids.remote()
    print('tbdev conversations', len(ids))
    groups = [ids[i::8] for i in range(8)]
    for r in tbdev_audio.map(groups):
        print('tbdev', r)
