"""Train the content models on Modal, where ssl_turn's cached encoder frames live.

    modal run annotator/modal_content.py --data /path/to/annotator-data     # both models
    modal run annotator/modal_content.py --data ... --only stereo --eval --calibrate tbdev

Volume `ssl-turn-work` (ssl_turn/pipeline): /work/audio/{split}/{cid}.npy (24 kHz stereo) and
/work/feats_asr/{split}/{cid}.npy (per-channel encoder frames, the stereo model's input). The mono
model needs the mix encoded the way annotate.py encodes mono audio; encode_mono writes that to
/work/feats_asr_mono once (L4 GPU). Training runs on CPU. Bundles are written next to --data as
labeler-content-{stereo,mono}.joblib.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
TURNBENCH_COMMIT = '38a6f874322430cb3ca71d8a52aa1e636e88bad8'
work = modal.Volume.from_name('ssl-turn-work')
app = modal.App('annotator-content')

gpu_image = (modal.Image.debian_slim(python_version='3.11')
             .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
             .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1', 'scipy==1.15.3',
                          'soundfile==0.13.1')
             .add_local_dir(HERE, '/root/annotator', ignore=['**/__pycache__']))
cpu_image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
             .pip_install(f'git+https://github.com/SesameAILabs/turnbench@{TURNBENCH_COMMIT}', 'scipy',
                          'scikit-learn', 'joblib')
             .add_local_dir(HERE, '/root/annotator', ignore=['**/__pycache__']))


@app.function(image=gpu_image, volumes={'/work': work}, gpu='L4', cpu=4, memory=16384, timeout=7200)
def encode_mono(items):
    import sys
    sys.path.insert(0, '/root')
    import numpy as np
    from scipy.signal import resample_poly
    from annotator import content as C
    model = C.load_model('cuda')
    for split, cid in items:
        mix = np.load(f'/work/audio/{split}/{cid}.npy').astype(np.float32).mean(1)
        x16 = resample_poly(mix, 2, 3).astype(np.float32)[:, None]  # 24 -> 16 kHz, as annotate's load_16k
        os.makedirs(f'/work/feats_asr_mono/{split}', exist_ok=True)
        np.save(f'/work/feats_asr_mono/{split}/{cid}.tmp.npy', C.encode(x16, 'mono', model))
        os.replace(f'/work/feats_asr_mono/{split}/{cid}.tmp.npy', f'/work/feats_asr_mono/{split}/{cid}.npy')
    work.commit()
    return len(items)


@app.function(image=cpu_image, volumes={'/work': work}, cpu=8, memory=32768, timeout=7200)
def train(data, feats_root, args):
    """data: {set: {filename: bytes}}; runs annotator.train with --feats from the volume."""
    import subprocess
    import tempfile
    work.reload()
    tmp = tempfile.mkdtemp()
    for name, files in data.items():
        os.makedirs(f'{tmp}/{name}')
        for f, b in files.items():
            open(f'{tmp}/{name}/{f}', 'wb').write(b)
    out = f'{tmp}/model.joblib'
    cmd = (['python', '-m', 'annotator.train', '--data'] + [f'{k}={tmp}/{k}' for k in data]
           + ['--feats'] + [f'{k}={feats_root}/{k}' for k in data] + args + ['--out', out])
    subprocess.run(cmd, cwd='/root', check=True)
    return open(out, 'rb').read()


@app.local_entrypoint()
def main(data: str, only: str = '', eval: bool = False, calibrate: str = ''):
    sets = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)))
    blobs = {s: {f: open(os.path.join(data, s, f), 'rb').read() for f in os.listdir(os.path.join(data, s))}
             for s in sets}
    modes = [only] if only else ['stereo', 'mono']
    if 'mono' in modes and not os.environ.get('SKIP_MONO_ENCODE'):
        items = [(s, f[:-4]) for s in sets for f in sorted(blobs[s]) if f.endswith('.npz')]
        print('encoding', sum(encode_mono.map([items[i::4] for i in range(4)])), 'mixes')
    extra = (['--eval'] if eval else []) + (['--calibrate'] + calibrate.split(',') if calibrate else [])
    for mode, root in zip(modes, ['/work/feats_asr' if m == 'stereo' else '/work/feats_asr_mono' for m in modes]):
        path = os.path.join(data, '..', 'annotator-model', f'labeler-content-{mode}.joblib')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(train.remote(blobs, root, extra))
        print('saved', os.path.abspath(path))
