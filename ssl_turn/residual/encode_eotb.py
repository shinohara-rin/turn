"""FastConformer features for LiveKit eot-bench (en validation, rev ca9d98a9) on CPU, as
eotbench/modal_eot.py encodes them on Modal: user in channel 0, agent channel silent.

    ASR_DEVICE=cpu python encode_eotb.py      # -> $DATA/feats/eotb_en.npz {id: [T, 2, 1024] f16}
"""
import io
import os
import sys
import time

os.environ.setdefault('ASR_DEVICE', 'cpu')
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

import data
sys.path.insert(0, str(data.HERE.parent / 'pipeline'))
import encode_asr  # noqa: E402

DATASET, REVISION = 'livekit/eot-bench-data', 'ca9d98a9'


def main(limit=None):
    path = hf_hub_download(DATASET, 'data/en/validation-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
    table = pq.read_table(path, columns=['id', 'audio'])
    enc = encode_asr.load_model()
    out, t0, audio_s = {}, time.time(), 0.0
    for n, (rid, audio) in enumerate(zip(table.column('id').to_pylist(), table.column('audio').to_pylist())):
        if limit and n >= limit:
            break
        wav, sr = sf.read(io.BytesIO(audio['bytes']), dtype='float32')
        assert sr == 16000, sr
        wav = wav.mean(1) if wav.ndim > 1 else wav
        T = len(wav) // encode_asr.FRAME
        audio_s += len(wav) / sr
        user = encode_asr.encode_channel(enc, wav[:T * encode_asr.FRAME], T)
        agent = encode_asr.encode_channel(enc, np.zeros(T * encode_asr.FRAME, np.float32), T)
        out[rid] = np.stack([user, agent], 1)
        if n % 20 == 0:
            print(f'{n} turns, {audio_s:.0f} s audio, {time.time() - t0:.0f} s wall', flush=True)
    np.savez(data.DATA / 'feats/eotb_en.npz', **out)
    print(f'{len(out)} turns, {audio_s / 60:.1f} min, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
