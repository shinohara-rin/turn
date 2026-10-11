"""FastConformer features of a silent channel (digital zeros), as eot-bench's agent channel sees it.

    modal run ssl_turn/latency/silent_feats.py

Writes {WORK}/silent_asr.npy float16 [T, 1024] (30 s) for the `mute` training augmentation in
train.py, which replaces one channel of a crop with these frames.
"""
import sys
from pathlib import Path

import modal

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / 'pipeline'))
from common import VOLUMES, WORK, work  # noqa: E402

app = modal.App('ssl-turn-silent')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1', 'scipy==1.15.3',
                      'soundfile==0.13.1', 'huggingface_hub')
         .add_local_python_source('common', 'encode_asr'))


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=16384, timeout=1800)
def silent(seconds=30):
    import numpy as np
    import encode_asr
    enc = encode_asr.load_model()
    T = int(seconds / 0.08)
    x = encode_asr.encode_channel(enc, np.zeros(T * encode_asr.FRAME, np.float32), T)
    np.save(f'{WORK}/silent_asr.npy', x)
    work.commit()
    d = np.abs(np.diff(x.astype(np.float32), axis=0)).max(1)
    return dict(T=T, max_step_change=[float(v) for v in d[:5]] + [float(d[5:].max())],
                norm=float(np.linalg.norm(x[-1].astype(np.float32))))


@app.local_entrypoint()
def main():
    print(silent.remote())
