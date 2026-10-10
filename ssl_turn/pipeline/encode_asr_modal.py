"""Run encode_asr.py (streaming FastConformer features) on Modal L4s.

    modal run encode_asr_modal.py [--groups 8]

Encodes every prepped conversation (/work/audio/{oto,tbdev}) that has no
/work/feats_asr file yet. ray_run.py runs the same encode on a local GPU.
"""
import modal

from common import VOLUMES, WORK, work

app = modal.App('ssl-turn-encode-asr')
asr_image = (modal.Image.debian_slim(python_version='3.11')
             .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
             .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                          'scipy==1.15.3', 'soundfile==0.13.1')
             .add_local_python_source('common', 'encode_asr'))


@app.function(image=asr_image, volumes=VOLUMES, gpu='L4', cpu=8, memory=32768, timeout=7200)
def encode_group(items):
    import encode_asr
    out = encode_asr.encode(items)
    work.commit()
    return out


@app.function(image=asr_image, volumes=VOLUMES, cpu=1, memory=1024, timeout=300)
def todo():
    import os
    work.reload()
    items = []
    for split in ('oto', 'tbdev'):
        have = set(os.listdir(f'{WORK}/feats_asr/{split}')) if os.path.isdir(f'{WORK}/feats_asr/{split}') else set()
        for f in sorted(os.listdir(f'{WORK}/audio/{split}')):
            if f.endswith('.npy') and f not in have:
                items.append((split, f[:-4]))
    return items


@app.local_entrypoint()
def main(groups: int = 8):
    items = todo.remote()
    print(len(items), 'conversations to encode')
    for r in encode_group.map([g for g in (items[i::groups] for i in range(groups)) if g]):
        print(r)
