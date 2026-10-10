"""Reference outputs for bench.html: stream a stereo clip through encoder_k1.onnx + head.onnx
(fp32, onnxruntime CPU; equal to the PyTorch steps to ~1e-6) and write

    sample.f32  16 kHz float32, planar (channel 0 then channel 1)
    ref.json    per-frame eot / int score tracks, frames 0 .. T

    python make_ref.py AUDIO --models ../../../web_models --seconds 60 --out .

AUDIO is a prep.py audio cache (24 kHz [N, 2] .npy) or a 16 kHz stereo .wav.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / 'pipeline')]
import stream_encoder as se  # noqa: E402


def load(path, seconds):
    if path.endswith('.npy'):
        from encode_asr import to16k
        a24 = np.load(path).astype(np.float32)[:24000 * seconds]
        return np.stack([to16k(a24[:, c]) for c in (0, 1)], 1)
    from scipy.io import wavfile
    sr, a = wavfile.read(path)
    assert sr == 16000 and a.ndim == 2 and a.shape[1] == 2, 'need a 16 kHz stereo wav'
    a = a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0)
    return a[:16000 * seconds]


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--models', default='web_models')
    ap.add_argument('--seconds', type=int, default=60)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    wav = load(a.audio, a.seconds).astype(np.float32)
    T = len(wav) // 1280
    wav = wav[:T * 1280]
    wav.T.copy().tofile(f'{a.out}/sample.f32')
    e = ort.InferenceSession(f'{a.models}/encoder_k1.onnx', providers=['CPUExecutionProvider'])
    h = ort.InferenceSession(f'{a.models}/head.onnx', providers=['CPUExecutionProvider'])
    names = [i.name for i in e.get_inputs()][2:]
    st = [np.zeros(i.shape, np.int64 if 'int64' in i.type else np.float32) for i in e.get_inputs()[2:]]
    kv = [np.zeros(i.shape, np.float32) for i in h.get_inputs()[2:]]
    eot, intr = [], []

    def head(feat, t):
        o = h.run(None, dict(feat=feat, n=np.array([min(t, 249)], np.int64), kcache=kv[0], vcache=kv[1]))
        kv[:] = o[5:7]
        eot.extend(o[3].tolist())
        intr.extend(o[4].tolist())

    head(np.zeros((2, 1, 1024), np.float32), 0)
    t0 = time.time()
    for t, w in se.windows(wav, T + 1):
        feat, *st = e.run(None, dict(audio=w, t=np.array([t], np.int64), **dict(zip(names, st))))
        head(feat, t)
    print(f'{len(eot)} frames; native onnxruntime CPU fp32: {(time.time() - t0) / T * 1000:.0f} ms per 80 ms frame')
    json.dump(dict(eot=eot, int=intr), open(f'{a.out}/ref.json', 'w'))


if __name__ == '__main__':
    main()
