"""Run bgbench (bench.py) on Modal: prepare TB dev once, then run each model on every
condition. Mixtures are rendered on the fly in each runner (deterministic, cheap), so the
benchmark audio is defined by bench.py rather than stored.

Volume `turn-bgbench` (upload once from a machine that has the data):
    /tbdev/*.parquet      TurnBench dev shards (mundo-ai/turn-benchmark-dev)
    /musan/fma/*.wav      MUSAN music/fma
    /heads/<run>/*.pt     ssl_turn head checkpoints (train.py output)
Writes /bench/{plan.json, audio/<cid>.npy, ann/<cid>.json} and /runs/<model>/<cid>.npz.

    modal run bench_modal.py::prep
    modal run bench_modal.py::vap
    modal run bench_modal.py::asr --run r016_asr --members fine1_bal1_s1,fine1_bal1_s2
    modal run bench_modal.py::fetch --model vap --out DIR     # then bench_score.py
    modal run bench_modal.py::samples --out DIR              # listening clips
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
vol = modal.Volume.from_name('turn-bgbench', create_if_missing=True)
V = '/vol'
app = modal.App('turn-bgbench')


def _code(image):
    if not modal.is_local():
        return image
    return (image.add_local_dir(str(HERE), '/root/bgspeech', ignore=['**/__pycache__', '*.pyc'])
            .add_local_dir(str(HERE.parent / 'ssl_turn'), '/root/ssl_turn', ignore=['**/__pycache__']))


cpu_image = _code(modal.Image.debian_slim(python_version='3.11')
                  .apt_install('libsndfile1')
                  .pip_install('numpy==2.2.6', 'scipy==1.15.3', 'soundfile==0.13.1', 'pyarrow==19.0.1'))
TURNBENCH_SHA = '38a6f874322430cb3ca71d8a52aa1e636e88bad8'
VAP_SHA = 'f39a78b23a6dccdbedd106e00b48c410b8739f5d'
TB = '/root/turnbench'
vap_image = _code(
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'libsndfile1', 'ffmpeg')
    .run_commands(
        f'git clone https://github.com/SesameAILabs/turnbench {TB} && git -C {TB} checkout {TURNBENCH_SHA}',
        f'git clone https://github.com/ErikEkstedt/VoiceActivityProjection {TB}/baselines/vap/VoiceActivityProjection'
        f' && git -C {TB}/baselines/vap/VoiceActivityProjection checkout {VAP_SHA}',
        f'pip install -e {TB}',
        'pip install torch==2.7.0 torchaudio==2.7.0 einops tqdm',
        f'pip install -e {TB}/baselines/vap/VoiceActivityProjection --no-deps')
    .pip_install('scipy==1.15.3', 'soundfile==0.13.1', 'pyarrow==19.0.1'))
asr_image = _code(
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
    .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                 'scipy==1.15.3', 'soundfile==0.13.1')
    .env({'SSL_TURN_WORK': V}))


def _paths():
    for p in ('/root/bgspeech', '/root/ssl_turn', '/root/ssl_turn/pipeline'):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---- shared container helpers -------------------------------------------------------

def _load(cid):
    import numpy as np
    a = np.load(f'{V}/bench/audio/{cid}.npy').astype(np.float32)
    ann = {tuple((int(k.split(':')[0]), k.split(':')[1])): [tuple(e) for e in v]
           for k, v in json.load(open(f'{V}/bench/ann/{cid}.json')).items()}
    return a, ann


def _music():
    import numpy as np
    import soundfile as sf
    import bench
    import mixing
    d = f'{V}/musan/fma'
    out = []
    for f in bench.music_files([f for f in os.listdir(d) if f.endswith('.wav')]):
        x, sr = sf.read(f'{d}/{f}', dtype='float32', always_2d=True)
        out.append(mixing.resample_to(x.mean(1), sr, bench.SR))
    return out


def _conditions(cid, plan, music, conds):
    """Yield (cond, stereo [n, 2] float32 at bench.SR) for each condition."""
    import numpy as np
    import bench
    a, ann = _load(cid)
    item = plan['items'][cid]
    u = item['user'] - 1
    n = len(a)
    mask = bench.activity(ann, item['user'], int(np.ceil(n / bench.SR * bench.MASK_RATE)))
    styles = {bench.CONDITIONS[c][0] for c in conds} - {None}
    bgs = bench.backgrounds(cid, n, [_load(d)[0] for d in item['donors']], music, styles=styles) if styles else {}
    for cond in conds:
        x = a.copy()
        x[:, u] = bench.render(cid, a[:, u], mask, bgs, cond)
        yield cond, x


def _plan():
    return json.load(open(f'{V}/bench/plan.json'))


# ---- stage 0: decode TB dev once -------------------------------------------------------

@app.function(image=cpu_image, volumes={V: vol}, cpu=4, memory=32768, timeout=3600)
def prep_shard(path):
    import numpy as np
    _paths()
    import mixing
    from prep import causal_resample  # ssl_turn/pipeline/prep.py: the pipeline's 24 kHz audio
    import bench
    os.makedirs(f'{V}/bench/audio', exist_ok=True)
    os.makedirs(f'{V}/bench/ann', exist_ok=True)
    ids = []
    for row in mixing.iter_rows(path):
        (w1, s1), (w2, s2) = row['audio'][1], row['audio'][2]
        assert s1 == s2
        n = min(len(w1), len(w2))
        a = causal_resample(np.stack([w1[:n], w2[:n]], 1), s1, bench.SR)
        np.save(f'{V}/bench/audio/{row["id"]}.npy', a.astype(np.float16))
        json.dump({f'{s}:{k}': v for (s, k), v in row['annotations'].items()},
                  open(f'{V}/bench/ann/{row["id"]}.json', 'w'))
        ids.append(row['id'])
        print(row['id'], f'{n / s1:.0f}s', flush=True)
    vol.commit()
    return ids


@app.function(image=cpu_image, volumes={V: vol}, cpu=1, memory=2048, timeout=600)
def write_plan():
    _paths()
    import bench
    vol.reload()
    ids = sorted(f[:-4] for f in os.listdir(f'{V}/bench/audio') if f.endswith('.npy'))
    plan = bench.make_plan(ids)
    json.dump(plan, open(f'{V}/bench/plan.json', 'w'), indent=1)
    vol.commit()
    return len(ids)


@app.local_entrypoint()
def prep():
    shards = sorted(f'{V}/tbdev/{f}' for f in ['dev-0.parquet', 'dev-1.parquet', 'dev-2.parquet'])
    print(sum(len(x) for x in prep_shard.map(shards)), 'conversations decoded')
    print(write_plan.remote(), 'conversations in plan')


# ---- VAP (official oto checkpoint) -------------------------------------------------------

@app.function(image=vap_image, volumes={V: vol}, gpu='L4', cpu=4, memory=32768, timeout=3600)
def vap_group(cids):
    import numpy as np
    import torch
    _paths()
    sys.path.insert(0, TB)
    import bench
    import mixing
    from baselines.vap.predict import _load_model, _step_extraction, SAMPLE_RATE
    model = _load_model('oto', 'cuda')
    plan, music = _plan(), _music()
    os.makedirs(f'{V}/runs/vap', exist_ok=True)
    for cid in cids:
        if os.path.exists(f'{V}/runs/vap/{cid}.npz'):
            continue
        t0, out = time.time(), {}
        for cond, x in _conditions(cid, plan, music, list(bench.CONDITIONS)):
            w = np.stack([mixing.resample_to(x[:, c], bench.SR, SAMPLE_RATE) for c in (0, 1)])
            with torch.inference_mode():
                out[cond] = _step_extraction(torch.from_numpy(w)[None], model, 'cuda')['p_now'][0].cpu().numpy() \
                    .astype(np.float16)
        np.savez_compressed(f'{V}/runs/vap/{cid}.npz', **out)
        vol.commit()
        print(cid, f'{time.time() - t0:.0f}s', flush=True)
    return len(cids)


# ---- ssl_turn heads on the streaming FastConformer encoder --------------------------------

@app.function(image=asr_image, volumes={V: vol}, gpu='L4', cpu=8, memory=32768, timeout=7200)
def asr_group(cids, run, members):
    import numpy as np
    import torch
    _paths()
    import bench
    import encode_asr as ea
    import train as tr
    enc = ea.load_model()
    models, configs = {}, {}
    for m in members:
        ck = torch.load(f'{V}/heads/{run}/{m}.pt', map_location='cuda', weights_only=False)
        assert ck['cfg'].get('feats') == 'asr', ck['cfg']
        configs[m] = ck['cfg']
        models[m] = tr.build_model(ck['cfg']).cuda()
        models[m].load_state_dict(ck['state'])
    tr.LOADED = []  # no tap columns: features are the 1024-d ASR vectors
    plan, music = _plan(), _music()
    os.makedirs(f'{V}/runs/{run}', exist_ok=True)
    for cid in cids:
        if os.path.exists(f'{V}/runs/{run}/{cid}.npz'):
            continue
        t0, out = time.time(), {}
        u = plan['items'][cid]['user'] - 1
        clean_f = None
        for cond, x in _conditions(cid, plan, music, list(bench.CONDITIONS)):
            chans = [ea.to16k(x[:, c]) for c in (0, 1)]
            n = min(len(c) for c in chans)
            T = n // ea.FRAME
            if clean_f is None:  # 'clean' comes first: encode both channels once
                assert cond == 'clean'
                clean_f = np.stack([ea.encode_channel(enc, chans[c][:n], T) for c in (0, 1)], 1)
                f = clean_f
            else:
                f = clean_f[:T].copy()
                f[:, u] = ea.encode_channel(enc, chans[u][:n], T)
            X = torch.from_numpy(f).cuda()
            res = tr.infer_all(models, configs, (('tbdev', X, [0, len(f)], [cid]),), 'cuda')
            for k, v in res.items():
                m, rest = k.split('/', 1)
                out[f'{m}@{cond}/{rest}'] = v
        np.savez_compressed(f'{V}/runs/{run}/{cid}.npz', **out)
        vol.commit()
        print(cid, f'{time.time() - t0:.0f}s', flush=True)
    return len(cids)


def _groups(k, limit=0):
    ids = read_plan.remote()['ids'][:limit or None]
    return [g for g in (ids[i::k] for i in range(k)) if g]


@app.function(image=cpu_image, volumes={V: vol}, cpu=1, memory=1024, timeout=300)
def read_plan():
    return _plan()


@app.local_entrypoint()
def vap(groups: int = 8, limit: int = 0):
    print(sum(vap_group.map(_groups(groups, limit))), 'conversations')


@app.local_entrypoint()
def asr(run: str = 'r016_asr', members: str = 'fine1_bal1_s1,fine1_bal1_s2', groups: int = 8, limit: int = 0):
    gs = _groups(groups, limit)
    print(sum(asr_group.starmap([(g, run, members.split(',')) for g in gs])), 'conversations')


# ---- downloads ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={V: vol}, cpu=2, memory=8192, timeout=1800)
def pack(model):
    """All of a model's per-conversation outputs as one npz (keys '<cid>/<key>')."""
    import io
    import numpy as np
    vol.reload()
    d = f'{V}/runs/{model}'
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.npz'):
            with np.load(f'{d}/{f}') as z:
                for k in z.files:
                    out[f'{f[:-4]}/{k}'] = z[k]
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return buf.getvalue()


@app.local_entrypoint()
def fetch(model: str, out: str):
    os.makedirs(out, exist_ok=True)
    Path(out, f'{model}.npz').write_bytes(pack.remote(model))
    Path(out, 'plan.json').write_text(json.dumps(read_plan.remote(), indent=1))
    print('wrote', out)


@app.function(image=cpu_image, volumes={V: vol}, cpu=4, memory=16384, timeout=1800)
def sample_clips(cid, start_s=60.0, dur_s=30.0):
    import io
    import numpy as np
    import soundfile as sf
    _paths()
    import bench
    plan, music = _plan(), _music()
    out = {}
    for cond, x in _conditions(cid, plan, music, list(bench.CONDITIONS)):
        a, b = int(start_s * bench.SR), int((start_s + dur_s) * bench.SR)
        buf = io.BytesIO()
        sf.write(buf, x[a:b], bench.SR, format='FLAC')
        out[cond] = buf.getvalue()
    return plan['items'][cid]['user'], out


@app.local_entrypoint()
def samples(out: str, cid: str = ''):
    plan = read_plan.remote()
    cid = cid or plan['ids'][0]
    user, clips = sample_clips.remote(cid)
    os.makedirs(out, exist_ok=True)
    for cond, b in clips.items():
        Path(out, f'{cid[:8]}-user{user}-{cond}.flac').write_bytes(b)
    print('wrote', len(clips), 'clips to', out)
