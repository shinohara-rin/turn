"""Voice-agent user turns as training data: LiveKit eot-bench in languages we never evaluate.

    modal run ssl_turn/latency/eotb_data.py --langs de,fr,it,nl,pt,ar,hi,id,ko,tr

eot-bench rows are single user turns from voice-agent sessions (16 kHz mono, CC-BY-4.0) with
the dataset's silence spans: the last one is the true end of turn, earlier ones are holds. That
is the domain our zero-shot EOT is slow on (otoSpeech is two-party chat). English (the benchmark
we report) and es / ja / zh (the multilingual thread's benchmarks) are never used here.

Each turn is encoded like eotbench/modal_eot.py: user audio in channel 0 through the streaming
FastConformer, the agent channel 1 the encoding of digital silence. Turns are concatenated into
~4 min sessions so train.py can draw 30 s crops. Labels (labels.py format):
  act    user CLAIM from the first word to the start of the final span, except inside holds;
         the agent channel is unsupervised (act_w 0)
  floor  HELD_0 from the first word to the final span, OPEN in the leading silence and the final
         span; projections stop at each turn's end (the agent's reply is not in the data)
  fine   unsupervised (fine_w 0); vap_mask 0 (no agent behaviour to project)
Writes {WORK}/feats_asr/eotb/{session}.npy and {WORK}/labels/eotb/{session}.npz.
"""
import sys
from pathlib import Path

import modal

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / 'pipeline'))
from common import VOLUMES, WORK, work  # noqa: E402

DATASET = 'livekit/eot-bench-data'
REVISION = 'ca9d98a9686b920a2d8c9eb984224ba9be74e4dd'
FRAME_S = 0.08
SESSION_FRAMES = 3000

app = modal.App('ssl-turn-eotb-data')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                      'scipy==1.15.3', 'soundfile==0.13.1', 'huggingface_hub')
         .add_local_dir(str(SRC), '/root/ssl_turn', ignore=['**/__pycache__', 'pipeline/**'])
         .add_local_python_source('common', 'encode_asr'))


def turn_labels(T, words, spans):
    """Per-frame targets for one turn of T frames (frame t ends at (t + 1) * 80 ms)."""
    import numpy as np
    sys.path.insert(0, '/root/ssl_turn')
    import labels as lb
    t = (np.arange(T) + 1) * FRAME_S - FRAME_S / 2   # frame centres
    spans = sorted(spans, key=lambda s: s['start'])
    first = min(w['start'] for w in words) if words else spans[0]['start']
    final = spans[-1]['start']
    speaking = (t >= first) & (t < final)
    for s in spans[:-1]:
        speaking &= ~((t >= s['start']) & (t < s['end']))
    act = np.full((T, 2), lb.A['SILENT'], np.int64)
    act[speaking, 0] = lb.A['CLAIM']
    floor = np.zeros((T, 4), np.float32)
    held = (t >= first) & (t < final)
    floor[held, lb.F['HELD_0']] = 1
    floor[~held, lb.F['OPEN']] = 1
    floor_w = np.ones(T, np.float32)
    future, future_w = lb.floor_projection(floor, floor_w)
    act_w = np.zeros((T, 2), np.float32)
    act_w[:, 0] = 1
    return dict(floor=floor, floor_w=floor_w, act=act, act_w=act_w, future=future, future_w=future_w,
                activity=lb.activity(act), fine=np.where(speaking, lb.FI['Normal Turn'], 0)[:, None].repeat(2, 1),
                fine_w=np.zeros((T, 2), np.float32), vap_mask=np.zeros(T, bool))


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=32768, timeout=7200)
def build(lang):
    import io
    import json
    import os
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, '/root/ssl_turn')
    import encode_asr
    assert lang not in ('en', 'es', 'ja', 'zh'), lang
    path = hf_hub_download(DATASET, f'data/{lang}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=REVISION)
    table = pq.read_table(path, columns=['id', 'audio', 'silence_spans', 'words'])
    enc = encode_asr.load_model()
    rows = table.to_pylist()
    longest = max(len(sf.read(io.BytesIO(r['audio']['bytes']))[0]) for r in rows) // encode_asr.FRAME + 1
    silent = encode_asr.encode_channel(enc, np.zeros(longest * encode_asr.FRAME, np.float32), longest)
    os.makedirs(f'{WORK}/feats_asr/eotb', exist_ok=True)
    os.makedirs(f'{WORK}/labels/eotb', exist_ok=True)
    feats, labs, k, n_turns, total = [], [], 0, 0, 0

    def flush():
        nonlocal feats, labs, k
        if not feats:
            return
        name = f'{lang}_{k:02d}'
        np.save(f'{WORK}/feats_asr/eotb/{name}.npy', np.concatenate(feats))
        np.savez_compressed(f'{WORK}/labels/eotb/{name}.npz',
                            **{key: np.concatenate([l[key] for l in labs]) for key in labs[0]})
        feats, labs, k = [], [], k + 1

    for r in rows:
        wav, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        assert sr == 16000, sr
        wav = wav.mean(1) if wav.ndim > 1 else wav
        T = len(wav) // encode_asr.FRAME
        spans = [s for s in r['silence_spans'] if s['end'] - s['start'] > 0]
        if T < 10 or not spans:
            continue
        user = encode_asr.encode_channel(enc, wav[:T * encode_asr.FRAME], T)
        feats.append(np.stack([user, silent[:T]], 1))
        labs.append(turn_labels(T, r['words'] or [], spans))
        n_turns += 1
        total += T
        if sum(len(f) for f in feats) >= SESSION_FRAMES:
            flush()
    flush()
    work.commit()
    return dict(lang=lang, turns=n_turns, sessions=k, minutes=total * FRAME_S / 60)


@app.local_entrypoint()
def main(langs: str = 'de,fr,it,nl,pt,ar,hi,id,ko,tr'):
    for r in build.map(langs.split(',')):
        print(r)
