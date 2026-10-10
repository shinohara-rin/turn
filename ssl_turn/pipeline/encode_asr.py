"""Causal streaming-ASR features: NVIDIA's cache-aware FastConformer at attention context [70, 0].

`nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` (114M) is trained for cache-aware
streaming: causal convolutions and chunked-limited attention. With context [70, 0] (no
lookahead) its offline forward is a streaming encoder. Each channel is encoded in 10 s chunks
with 20 s of left audio context (the rest of the receptive field is cut), and per 80 ms frame
we keep a middle layer and the output (2 x 512 = 1024-d).

Causality: encoder frame k sees mel frames up to 8k + 7, i.e. audio up to (k + 1) * 80 ms plus
the 16 ms STFT half-window. It is stored at our frame k + 1, which may use audio before
(k + 2) * 80 ms, so every stored frame is strictly causal. Frame 0 is zeros.

Audio is the cached 24 kHz causal resample (prep.py), decimated causally to 16 kHz.
Output: {WORK}/feats_asr/{split}/{cid}.npy float16 [T, 2, 1024], T = samples // 1280.
diagnose_asr.py has the probe evidence for this encoder.
"""
import os
import time

from common import WORK

MODEL = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
FRAME = 1280            # 80 ms at 16 kHz
DEVICE = os.environ.get('ASR_DEVICE', 'cuda')
CHUNK = 125             # output frames per chunk (10 s)
LEFT = 250              # left-context frames re-encoded per chunk (20 s)


def to16k(x24):
    """24 -> 16 kHz, one-sided FIR as in encode_mtd.to16k (no lookahead)."""
    import numpy as np
    from scipy.signal import firwin, upfirdn
    up, down = 2, 3
    half = 10 * max(up, down)
    taps = firwin(2 * half + 1, 1 / max(up, down), window=('kaiser', 5.0)) * up
    count = (len(x24) * up + down - 1) // down
    return upfirdn(taps, x24.astype(np.float32), up=up, down=down, axis=0)[:count].astype(np.float32)


def load_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location=DEVICE).eval()
    model.encoder.set_default_att_context_size([70, 0])
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    return model


def encode_channel(model, wav, T, batch=32):
    """wav: 16 kHz float32 numpy; returns [T, 1024] float16 (frame k + 1 <- encoder frame k)."""
    import numpy as np
    import torch
    layers = model.encoder.layers
    mid = {}
    h = layers[len(layers) // 2].register_forward_hook(lambda m, i, o: mid.__setitem__('x', o))
    out = np.zeros((T, 1024), np.float16)
    starts = list(range(0, T, CHUNK))
    try:
        for b in range(0, len(starts), batch):
            sigs, lens, keep = [], [], []
            for s in starts[b:b + batch]:
                a = max(0, s - LEFT)
                e = min(T, s + CHUNK)
                sigs.append(wav[a * FRAME:e * FRAME])
                lens.append((e - a) * FRAME)
                keep.append((a, s, e))
            sig = torch.zeros(len(sigs), max(lens))
            for k, w in enumerate(sigs):
                sig[k, :len(w)] = torch.from_numpy(w)
            with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
                f, fl = model.preprocessor(input_signal=sig.to(DEVICE), length=torch.tensor(lens).to(DEVICE))
                enc, el = model.encoder(audio_signal=f, length=fl)
            z = torch.cat([mid['x'].float(), enc.transpose(1, 2).float()], -1).half().cpu().numpy()
            for k, (a, s, e) in enumerate(keep):
                n = int(el[k])
                # encoder frame j (window-relative) -> our frame a + j + 1; keep our frames s..e-1
                for_frames = np.arange(s, e)
                j = for_frames - a - 1
                ok = (j >= 0) & (j < n)
                out[for_frames[ok]] = z[k, j[ok]]
    finally:
        h.remove()
    return out


def encode(items):
    import numpy as np
    model = load_model()
    t0, done = time.time(), 0.0
    for split, cid in items:
        a24 = np.load(f'{WORK}/audio/{split}/{cid}.npy').astype(np.float32)
        chans = [to16k(a24[:, c]) for c in (0, 1)]
        n = min(len(c) for c in chans)
        T = n // FRAME
        out = np.stack([encode_channel(model, chans[c][:n], T) for c in (0, 1)], 1)
        os.makedirs(f'{WORK}/feats_asr/{split}', exist_ok=True)
        np.save(f'{WORK}/feats_asr/{split}/{cid}.tmp.npy', out)
        os.replace(f'{WORK}/feats_asr/{split}/{cid}.tmp.npy', f'{WORK}/feats_asr/{split}/{cid}.npy')
        done += 2 * n / 16000
        print(f'{split}/{cid}: {T} frames; {done / (time.time() - t0):.0f} channel-s/s', flush=True)
    return dict(items=len(items), channel_seconds=done, wall_s=time.time() - t0)
