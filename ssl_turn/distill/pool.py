"""Unlabeled distillation pool: People's Speech English (MLCommons/peoples_speech `clean`, CC-BY /
CC-BY-SA), as single-user sessions with a silent agent channel (the eot-bench layout).

    modal run ssl_turn/distill/pool.py --shards 20
    cd ssl_turn/distill && modal run distill.py::teacher --run r025_eotb5 --name eotb5_s2 --tag r025s2 --splits pool

The d001/d002 students matched the teacher on TurnBench but not on new audio (eot-bench, background
speech); they saw ~45 h of audio. This pool adds many more speakers, microphones and rooms. It has
no labels: the teacher's logits are the only targets.

People's Speech rows are ~15 s consecutive cuts of long archive.org recordings (ids
`{source}_{index:05d}.flac`). Per shard, rows are put back in source order and concatenated; a
0.8-2.5 s near-silent gap separates sources. The result is cut into 3000-frame (4 min) sessions:
  audio16/pool/{shard}_{k:02d}.npy     float16 [T * 1280] mono user audio
  feats_asr/pool/{shard}_{k:02d}.npy   float16 [T, 2, 1024] FastConformer of the user audio and of
                                       digital silence (the agent), for the teacher pass only
"""
import sys
from pathlib import Path

import modal

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / 'pipeline'))
from common import VOLUMES, WORK, work  # noqa: E402

DATASET = 'MLCommons/peoples_speech'
SHARDS = 804
SESSION_FRAMES = 3000
FRAME = 1280

app = modal.App('ssl-turn-distill-pool')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                      'scipy==1.15.3', 'soundfile==0.13.1', 'huggingface_hub')
         .add_local_python_source('common', 'encode_asr'))


def pick_shards(n, seed=20261011):
    """n shard indices spread over the train split (deterministic)."""
    import random
    return sorted(random.Random(seed).sample(range(SHARDS), n))


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=32768, timeout=3600)
def build(shard):
    import io
    import os
    import time
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    import encode_asr
    t0 = time.time()
    path = hf_hub_download(DATASET, f'clean/train-{shard:05d}-of-{SHARDS:05d}.parquet', repo_type='dataset')
    table = pq.read_table(path, columns=['id', 'audio'])
    ids = table.column('id').to_pylist()
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    rng = np.random.default_rng(shard)
    pieces, prev = [], None
    for i in order:
        src = ids[i].rsplit('_', 1)[0]
        wav, sr = sf.read(io.BytesIO(table.column('audio')[i].as_py()['bytes']), dtype='float32')
        assert sr == 16000, sr
        wav = wav.mean(1) if wav.ndim > 1 else wav
        if prev is not None and src != prev:
            pieces.append(rng.normal(0, 1e-4, int(rng.uniform(0.8, 2.5) * 16000)).astype(np.float32))
        pieces.append(wav)
        prev = src
    audio = np.concatenate(pieces)
    del table, pieces
    enc = encode_asr.load_model()
    silent = encode_asr.encode_channel(enc, np.zeros(SESSION_FRAMES * FRAME, np.float32), SESSION_FRAMES)
    for d in ('audio16', 'feats_asr'):
        os.makedirs(f'{WORK}/{d}/pool', exist_ok=True)
    n = 0
    for k, s in enumerate(range(0, len(audio) // FRAME - 250, SESSION_FRAMES)):  # drop a < 20 s tail
        w = audio[s * FRAME:(s + SESSION_FRAMES) * FRAME]
        T = len(w) // FRAME
        w = w[:T * FRAME]
        user = encode_asr.encode_channel(enc, w, T)
        name = f'{shard:03d}_{k:02d}'
        np.save(f'{WORK}/feats_asr/pool/{name}.npy', np.stack([user, silent[:T]], 1))
        np.save(f'{WORK}/audio16/pool/{name}.npy', w.astype(np.float16))
        n += T
    work.commit()
    return dict(shard=shard, sessions=k + 1, hours=n * 0.08 / 3600, wall_s=round(time.time() - t0))


@app.local_entrypoint()
def main(shards: int = 20):
    tot = 0.0
    for r in build.map(pick_shards(shards)):
        print(r, flush=True)
        tot += r['hours']
    print(f'{tot:.1f} h')


@app.local_entrypoint()
def one(shard: int = 14):
    print(build.remote(shard))
