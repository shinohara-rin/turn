"""Streaming step of the causal FastConformer feature extractor, for ONNX export.

encode_asr.py encodes offline (10 s chunks, 20 s left context). This module produces the same
features K 80 ms frames at a time (K = 1 by default) with NeMo's cache-aware streaming caches,
so a browser can run it live. Each call takes a raw 16 kHz audio window and returns our frames
t .. t+K-1 as 1024-d features (encoder frame t - 1 is our frame t, as encode_asr stores it).
Encoder frame k sees mel frames up to 8k (causal stride-2 convs), i.e. audio up to sample
1280 k + 256, so our frame t only needs audio up to 1280 t - 1024: the window ends there and
frame t is ready 0.8 frames early. Larger K amortises per-call overhead (WebGPU dispatch) at
the cost of up to (K - 1) x 80 ms extra latency on the earlier frames of a chunk.

Mel features are computed in-graph (pre-emphasis, 512-point DFT as a matmul, NeMo's filter
bank, log, stft's reflect padding at sample 0) so the browser only feeds raw samples.
"""
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv2D

FRAME = 1280                     # 80 ms at 16 kHz = 8 mel hops
HOP, NFFT = 160, 512
START = -4097                    # window = samples 1280 t + START ... 1280 t + START + win(K) - 1
CACHE = 70


def mels_in(K=1):
    return 17 + 8 * (K - 1)      # 9 cached + 8 new mel frames per encoder frame (streaming_cfg)


def win(K=1):
    return (mels_in(K) - 1) * HOP + NFFT + 1   # 3073 samples for K = 1 (+1 for pre-emphasis)


WIN = win(1)


def load_encoder(path=None):
    import nemo.collections.asr as nemo_asr
    name = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    m = (nemo_asr.models.ASRModel.restore_from(path, map_location='cpu') if path else
         nemo_asr.models.ASRModel.from_pretrained(name, map_location='cpu')).eval()
    m.encoder.set_default_att_context_size([70, 0])
    m.encoder.setup_streaming_params()
    m.preprocessor.featurizer.dither = 0.0
    m.preprocessor.featurizer.pad_to = 0
    return m


class StreamStep(nn.Module):
    """inputs: audio [B, win(K)] float32 (samples 1280 t - 4097 ... 1280 (t + K - 1) - 1025, zeros
    before 0), t [1] int64 (first frame of the chunk, >= 1), cache_len [B] int64, then per layer i
    cache_ch_i [B, 70, 512] and cache_time_i [B, 512, 8]. outputs: feat [B, K, 1024], cache_len_out and the
    updated per-layer caches. Caches are separate tensors, not one stacked [L, ...] tensor: a
    17-way Concat needs 18 storage buffers in one WebGPU shader, over the common limit of 8."""

    def __init__(self, model, K=1):
        super().__init__()
        fz = model.preprocessor.featurizer
        self.enc = model.encoder
        assert self.enc.streaming_cfg.drop_extra_pre_encoded == 2
        window = torch.zeros(NFFT)
        w = fz.window.float()
        off = (NFFT - len(w)) // 2
        window[off:off + len(w)] = w
        k = torch.arange(NFFT, dtype=torch.float64)[:, None]
        f = torch.arange(NFFT // 2 + 1, dtype=torch.float64)[None]
        ang = 2 * torch.pi * k * f / NFFT
        self.register_buffer('cos', (torch.cos(ang) * window[:, None].double()).float())
        self.register_buffer('sin', (torch.sin(ang) * window[:, None].double()).float())
        self.register_buffer('fb', fz.fb[0].float().T.contiguous())          # [257, 80]
        self.guard = float(fz.log_zero_guard_value)
        self.preemph = float(fz.preemph)
        self.mid = len(self.enc.layers) // 2
        self.K, self.M, self.W = K, mels_in(K), win(K)

    def mel(self, audio, t):
        x = audio[:, 1:] - self.preemph * audio[:, :-1]                      # [B, WIN-1]
        # torch.stft(center=True) reflect-pads the pre-emphasised signal at sample 0
        base = 1280 * t + START + 1                                          # global index of x[:, 0]
        g = base + torch.arange(self.W - 1)
        src = torch.where(g < 0, -g, g) - base
        x = x[:, src.clamp(0, self.W - 2)]
        idx = torch.arange(self.M)[:, None] * HOP + torch.arange(NFFT)[None]
        frames = x[:, idx]                                                   # [B, M, 512]
        re, im = frames @ self.cos, frames @ self.sin
        m = torch.log((re * re + im * im) @ self.fb + self.guard)            # [B, M, 80]
        j = 8 * t - 24 + torch.arange(self.M)                                # global mel index
        return m.masked_fill((j < 0)[None, :, None], 0.0)

    def pre_encode(self, m, t):
        """NeMo's causal dw-striding subsampling on the mel window, last K outputs. Offline,
        every CausalConv2D zero-pads before the start of the audio; frames of this window that lie
        before it are zeroed at each conv input so the first steps match offline exactly
        (otherwise ReLU(bias) leaks in and poisons the attention caches for ~80 s)."""
        x = m[:, None]                                                       # [B, 1, M, 80]
        first = 8 * t - 24                                                   # global index of x[:, :, 0]
        stage = 0
        for mod in self.enc.pre_encode.conv:
            if isinstance(mod, CausalConv2D) and stage:
                g = first + torch.arange(x.shape[2])
                x = x * (g >= 0).to(x.dtype)[None, None, :, None]
            x = mod(x)
            if isinstance(mod, CausalConv2D):
                stage += 1
                first = first // 2
        b, c, T, f = x.shape
        return self.enc.pre_encode.out(x.transpose(1, 2).reshape(b, T, -1))[:, 2:]

    def forward(self, audio, t, cache_len, *caches):
        enc = self.enc
        L = len(enc.layers)
        cache_ch, cache_time = caches[:L], caches[L:]
        m = self.mel(audio, t)
        B = m.shape[0]
        x = self.pre_encode(m, t)                                            # [B, K, 512]
        assert x.shape[1] == self.K
        x, pos = enc.pos_enc(x=x, cache_len=CACHE)
        # query j (chunk frame) sees keys j .. j + 70 of [70 cached, K new]; True = hidden
        key = torch.arange(CACHE + self.K)[None]
        q = torch.arange(self.K)[:, None]
        hide = (key < q) | (key > q + CACHE)                                 # [K, 70 + K]
        att_mask = hide[None] | (key < (CACHE - cache_len)[:, None])[:, None]   # [B, K, 70 + K]
        pad_mask = torch.zeros(B, self.K, dtype=torch.bool)
        new_ch, new_time = [], []
        mid = None
        for i, layer in enumerate(enc.layers):
            x, c, tm = layer(x=x, att_mask=att_mask, pos_emb=pos, pad_mask=pad_mask,
                             cache_last_channel=cache_ch[i], cache_last_time=cache_time[i])
            new_ch.append(c)
            new_time.append(tm)
            if i == self.mid:
                mid = x
        feat = torch.cat([mid, x], -1)                                       # [B, K, 1024]
        return (feat, torch.clamp(cache_len + self.K, max=CACHE), *new_ch, *new_time)

    def init_state(self, B=2):
        """[cache_len, cache_ch_0 .. cache_ch_16, cache_time_0 .. cache_time_16]"""
        L, D = len(self.enc.layers), self.enc.d_model
        return ([torch.zeros(B, dtype=torch.int64)] + [torch.zeros(B, CACHE, D) for _ in range(L)] +
                [torch.zeros(B, D, self.enc.conv_context_size[0]) for _ in range(L)])

    def state_names(self):
        L = len(self.enc.layers)
        return ['cache_len'] + [f'cache_ch_{i}' for i in range(L)] + [f'cache_time_{i}' for i in range(L)]


def windows(wav, T, K=1):
    """wav [N] or [N, C] float32 16 kHz -> generator of (t, window [C, win(K)]) for chunks
    starting at t = 1, 1 + K, ... while t + K - 1 < T."""
    import numpy as np
    w = wav if wav.ndim == 2 else wav[:, None]
    W = win(K)
    pad = np.zeros((W, w.shape[1]), np.float32)
    w = np.concatenate([pad, w.astype(np.float32), pad])
    for t in range(1, T - K + 1, K):
        s = 1280 * t + START + W
        yield t, w[s:s + W].T
