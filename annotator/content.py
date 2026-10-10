"""Content features: what is being said around each candidate, from a speech encoder.

NVIDIA's cache-aware streaming FastConformer (`stt_en_fastconformer_hybrid_large_streaming_multi`,
attention context [70, 0]), set up exactly as ssl_turn/pipeline/encode_asr.py so models trained
on that pipeline's cached features (`feats_asr`) apply here: 10 s chunks with 20 s of left
context, a middle layer and the output per 80 ms frame (2 x 512 = 1024-d). Stereo audio is
encoded per channel ([T, 2, 1024]); mono audio as the mix ([T, 1, 1024]).

Per candidate the frames are averaged in a few windows around it, for the candidate's own channel
and the other channel (stereo) or the mix (mono). train.py reduces these with PCA and appends them
to the timing features. Needs NeMo (`nemo_toolkit[asr]`); runs on CPU (slower) or GPU.
"""
from __future__ import annotations

import numpy as np

MODEL = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
SR = 16000
FRAME = 1280         # 80 ms at 16 kHz
FRAME_S = FRAME / SR
CHUNK = 125          # output frames per chunk (10 s)
LEFT = 250           # left-context frames re-encoded per chunk (20 s)
DIM = 1024
WINDOWS = ((-1.0, 0.0), (0.0, 0.5), (0.5, 1.5))  # seconds around the candidate


def load_model(device='cpu'):
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location=device).eval()
    model.encoder.set_default_att_context_size([70, 0])
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    return model


def encode_channel(model, wav, T, batch=8):
    """wav: 16 kHz float32 [n]; returns [T, 1024] float16 (frame k + 1 <- encoder frame k), as in
    ssl_turn/pipeline/encode_asr.py."""
    import torch
    device = next(model.parameters()).device
    layers = model.encoder.layers
    mid = {}
    h = layers[len(layers) // 2].register_forward_hook(lambda m, i, o: mid.__setitem__('x', o))
    out = np.zeros((T, DIM), np.float16)
    starts = list(range(0, T, CHUNK))
    try:
        for b in range(0, len(starts), batch):
            sigs, lens, keep = [], [], []
            for s in starts[b:b + batch]:
                a, e = max(0, s - LEFT), min(T, s + CHUNK)
                sigs.append(wav[a * FRAME:e * FRAME]); lens.append((e - a) * FRAME); keep.append((a, s, e))
            sig = torch.zeros(len(sigs), max(lens))
            for k, w in enumerate(sigs):
                sig[k, :len(w)] = torch.from_numpy(np.ascontiguousarray(w))
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                f, fl = model.preprocessor(input_signal=sig.to(device), length=torch.tensor(lens).to(device))
                enc, el = model.encoder(audio_signal=f, length=fl)
            z = torch.cat([mid['x'].float(), enc.transpose(1, 2).float()], -1).half().cpu().numpy()
            for k, (a, s, e) in enumerate(keep):
                frames = np.arange(s, e); j = frames - a - 1
                ok = (j >= 0) & (j < int(el[k]))
                out[frames[ok]] = z[k, j[ok]]
    finally:
        h.remove()
    return out


def encode(x16, mode, model):
    """x16: [n, C] float32 at 16 kHz -> [T, 2, 1024] (stereo, per channel) or [T, 1, 1024] (mono mix)."""
    chans = [x16[:, 0], x16[:, 1]] if mode == 'stereo' else [x16.mean(1)]
    T = len(x16) // FRAME
    return np.stack([encode_channel(model, c[:T * FRAME].astype(np.float32), T) for c in chans], 1)


def pooled(E, speaker, t):
    """Mean encoder frame per window around t: (own, other) channel for stereo E, the mix for mono E."""
    chans = (speaker - 1, 2 - speaker) if E.shape[1] == 2 else (0,)
    out = []
    for lo, hi in WINDOWS:
        a = max(0, int((t + lo) / FRAME_S)); b = max(a + 1, int((t + hi) / FRAME_S))
        seg = E[a:b]
        for c in chans:
            out.append(seg[:, c].astype(np.float32).mean(0) if len(seg) else np.zeros(E.shape[2], np.float32))
    return np.concatenate(out)


def table(E, rows):
    """Pooled content vectors for candidate rows (speaker, time_s, ...) of one conversation."""
    n = len(WINDOWS) * E.shape[1] * E.shape[2]
    return np.array([pooled(E, r[0], r[1]) for r in rows], np.float32).reshape(len(rows), n)
