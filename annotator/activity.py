"""Per-speaker activity [T, 2] at 20 ms from audio.

- Stereo (one speaker per channel): Silero VAD (ONNX) on each channel.
- Mono (two speakers mixed): NVIDIA Sortformer 4spk v1 (offline, CC-BY-NC-4.0) in overlapping
  windows, stitched by slot permutation, then folded to two speakers.

Silero needs `onnxruntime` and the `silero-vad` wheel's model file; Sortformer needs NeMo
(`nemo_toolkit[asr]`). Both run on CPU (~8 s and ~2 min per 15-min conversation).
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np

from . import features as F

SR = 16000


def load_16k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    return x


def silero_model_path():
    import silero_vad  # pip install silero-vad --no-deps is enough; only the ONNX file is used
    return glob.glob(os.path.join(os.path.dirname(silero_vad.__file__), 'data', 'silero_vad.onnx'))[0]


def silero_probs(x16, model_path=None):
    """[n, B] float32 at 16 kHz -> [chunks, B] speech probability at 32 ms (B channels in lockstep)."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_path or silero_model_path(), so)
    B, N = x16.shape[1], len(x16) // 512
    x = np.ascontiguousarray(x16[:N * 512].T.reshape(B, N, 512))
    state = np.zeros((2, B, 128), np.float32); ctx = np.zeros((B, 64), np.float32)
    sr = np.array(SR, np.int64); out = np.empty((N, B), np.float32)
    for i in range(N):
        p, state = sess.run(None, {'input': np.concatenate([ctx, x[:, i]], 1), 'state': state, 'sr': sr})
        out[i] = p[:, 0]; ctx = x[:, i, -64:]
    return out


def stereo_activity(path):
    x = load_16k(path)
    if x.shape[1] != 2:
        raise ValueError(f'{path}: expected 2 channels, got {x.shape[1]}')
    return F.smooth(F.resample_probs(silero_probs(x), 0.032))


class Sortformer:
    WIN_S, HOP_S, FRAME_S = 240.0, 180.0, 0.08

    def __init__(self, name='nvidia/diar_sortformer_4spk-v1'):
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
        self.torch = torch
        self.model = SortformerEncLabelModel.from_pretrained(name, map_location='cpu').eval()

    def probs(self, mono):
        """mono [n] at 16 kHz -> [T, 4] slot probabilities at 80 ms, stitched across windows."""
        T = int(len(mono) / SR / self.FRAME_S) + 1
        out = np.zeros((T, 4), np.float32); have = np.zeros(T, bool); start = 0.0
        while True:
            seg = mono[int(start * SR): int((start + self.WIN_S) * SR)]
            with self.torch.inference_mode():
                _, p = self.model.diarize(audio=[seg], batch_size=1, sample_rate=SR,
                                          include_tensor_outputs=True, verbose=False)
            p = p[0].squeeze(0).float().numpy()
            f0 = int(round(start / self.FRAME_S)); n = min(len(p), T - f0); p = p[:n]
            ov = have[f0:f0 + n]
            if ov.any():  # slot order of this window that best matches what is already stitched
                perm = min(itertools.permutations(range(p.shape[1])),
                           key=lambda pm: np.abs(out[f0:f0 + n][ov] - p[ov][:, pm]).sum())
                p = p[:, perm]
            out[f0:f0 + n] = np.where(ov[:, None], 0.5 * out[f0:f0 + n] + 0.5 * p, p)
            have[f0:f0 + n] = True
            if start + self.WIN_S >= len(mono) / SR:
                return out
            start += self.HOP_S


def mono_activity(path, diarizer=None):
    x = load_16k(path).mean(1)
    p = (diarizer or Sortformer()).probs(x)
    n = int(len(x) / SR / F.FRAME_S)
    return F.smooth(F.fold_to_two(F.resample_probs(p, Sortformer.FRAME_S, n)))
