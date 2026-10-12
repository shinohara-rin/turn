"""Student-side input transforms for distillation (the teacher's targets stay those of clean audio).

acoustic()     room reverb, microphone EQ and band limits applied to the student's input only, so
               the student must give the clean teacher's answer for audio recorded elsewhere
vad_frames()   Silero VAD (the deployed pipeline already runs it) summarised per 80 ms frame, as an
               extra student input (`vad_in`)
"""
import numpy as np

SR = 16000
FRAME = 1280          # 80 ms
CHUNK = 512           # Silero step, 32 ms
SILERO = 'silero-vad==6.2.0'


# ---- acoustic augmentation ------------------------------------------------------------------------

def acoustic(wav, p=0.5, rir_p=0.7, band_p=0.3, rir_s=0.5):
    """wav [B, S, C] float (GPU) -> same shape. Each non-silent channel is transformed with
    probability p: an optional synthetic room impulse response (direct path at t = 0, so the
    timing the teacher's targets refer to is unchanged), a random smooth EQ (spectral tilt plus two
    peaks/dips) and an optional low-pass at 3.4-7 kHz (telephone / narrowband codecs). Its RMS is
    kept, the per-channel gain augmentation handles level."""
    import torch
    B, S, C = wav.shape
    x = wav.permute(0, 2, 1).reshape(B * C, S)
    dev = x.device
    sel = (torch.rand(B * C, device=dev) < p) & (x.abs().amax(1) > 0)
    if not bool(sel.any()):
        return wav
    xs = x[sel].float()
    n = xs.shape[0]
    L = int(rir_s * SR)
    nfft = 1 << (S + L - 1).bit_length()
    f = torch.fft.rfftfreq(nfft, 1 / SR).to(dev)
    # reverb: exponentially decaying noise tail behind a unit direct path
    t = torch.arange(L, device=dev) / SR
    rt60 = torch.empty(n, 1, device=dev).uniform_(0.15, 0.8)
    tail = torch.randn(n, L, device=dev) * torch.exp(-6.9 * t / rt60)
    tail[:, :int(0.002 * SR)] = 0
    drr_db = torch.empty(n, 1, device=dev).uniform_(0, 15)
    tail = tail / tail.norm(dim=1, keepdim=True).clamp_min(1e-6) * 10 ** (-drr_db / 20)
    tail = tail * (torch.rand(n, 1, device=dev) < rir_p)
    rir = tail
    rir[:, 0] = 1
    H = torch.fft.rfft(rir, nfft)
    # EQ: tilt (dB per octave around 1 kHz) and two Gaussian bumps on a log-frequency axis
    lf = torch.log2((f + 50) / 1000)[None]
    g_db = torch.empty(n, 1, device=dev).uniform_(-3, 3) * lf
    for _ in range(2):
        fc = torch.empty(n, 1, device=dev).uniform_(np.log2(200 / 1000), np.log2(6000 / 1000))
        g = torch.empty(n, 1, device=dev).uniform_(-9, 9)
        g_db = g_db + g * torch.exp(-0.5 * ((lf - fc) / 0.5) ** 2)
    cut = torch.empty(n, 1, device=dev).uniform_(3400, 7000)
    lowpass = -60 * torch.sigmoid((f[None] - cut) / 150)
    g_db = g_db + lowpass * (torch.rand(n, 1, device=dev) < band_p)
    G = 10 ** (g_db / 20)
    y = torch.fft.irfft(torch.fft.rfft(xs, nfft) * H * G, nfft)[:, :S]
    y = y * (xs.pow(2).mean(1, keepdim=True).sqrt() / y.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8))
    x = x.clone()
    x[sel] = y.to(x.dtype)
    return x.view(B, C, S).permute(0, 2, 1)


# ---- Silero VAD per 80 ms frame -------------------------------------------------------------------

_SESSION = None


def silero_session():
    """onnxruntime session of the Silero model bundled with the silero-vad package (as pipeline/vad.py)."""
    global _SESSION
    if _SESSION is None:
        import os
        from importlib.util import find_spec
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        path = os.path.join(os.path.dirname(find_spec('silero_vad').origin), 'data', 'silero_vad.onnx')
        _SESSION = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    return _SESSION


def silero(x16):
    """x16 [S, B] float32 at 16 kHz -> [S // 512, B] speech probabilities (pipeline/vad.silero)."""
    sess = silero_session()
    B, N = x16.shape[1], len(x16) // CHUNK
    x = np.ascontiguousarray(x16[:N * CHUNK].T.reshape(B, N, CHUNK)).astype(np.float32)
    state = np.zeros((2, B, 128), np.float32)
    ctx = np.zeros((B, 64), np.float32)
    sr = np.array(SR, np.int64)
    out = np.empty((N, B), np.float32)
    for i in range(N):
        p, state = sess.run(None, {'input': np.concatenate([ctx, x[:, i]], 1), 'state': state, 'sr': sr})
        out[i] = p[:, 0]
        ctx = x[:, i, -64:]
    return out


def to_frames(p, T):
    """Chunk probabilities p [N, B] -> uint8 [T, B, 2]: the latest chunk that ends inside 80 ms frame t
    (audio before (t + 1) * 80 ms) and the max over the chunks ending inside the frame."""
    N, B = p.shape
    t = np.arange(T)
    last = (t + 1) * FRAME // CHUNK - 1
    first = t * FRAME // CHUNK
    out = np.zeros((T, B, 2), np.float32)
    ok = last >= 0
    lastc = np.clip(last, 0, N - 1)
    out[ok, :, 0] = p[lastc[ok]]
    mx = np.full((T, B), -1.0, np.float32)
    for k in range(3):  # at most 3 chunks end inside one 80 ms frame
        i = first + k
        m = (i <= last) & (i < N)
        mx[m] = np.maximum(mx[m], p[np.clip(i[m], 0, N - 1)])
    out[:, :, 1] = np.maximum(mx, 0)
    out[~ok] = 0
    return np.round(out * 255).astype(np.uint8)


def vad_frames(wav):
    """wav [S, C] float 16 kHz -> uint8 [S // 1280, C, 2]."""
    T = len(wav) // FRAME
    return to_frames(silero(np.asarray(wav, np.float32)), T)


def vad_frames_batch(wavs):
    """Several mono signals at once (one Silero batch, zero-padded to the longest) -> list of uint8
    [len // 1280, 2]. Zero padding only follows each signal, so results equal one-at-a-time runs."""
    S = max(len(w) for w in wavs)
    x = np.zeros((S, len(wavs)), np.float32)
    for j, w in enumerate(wavs):
        x[:len(w), j] = w
    p = silero(x)
    return [to_frames(p[:, j:j + 1], len(w) // FRAME)[:, 0] for j, w in enumerate(wavs)]
