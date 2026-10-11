"""Small end-to-end student: 16 kHz stereo audio -> the teacher's floor outputs at 12.5 Hz.

The teacher is the deployed pipeline's model part: the frozen streaming FastConformer (114M) plus
an ssl_turn floor head (model.TurnModel, ~3M). The student replaces both with one causal network:

  MelFrontend   fixed-weight log-mel (25 ms Hann window, 10 ms hop, 80 bins) as a strided Conv1d,
                so ONNX export and streaming need no STFT op
  Encoder       per channel (shared weights): a causal conv over 10 ms frames, 8 frames stacked into
                one 80 ms frame, then `layers` blocks of [depthwise causal conv + windowed causal
                attention + feed-forward]
  TurnModel     the same floor head as the teacher (cross-channel attention, every output head),
                reading the student encoder instead of FastConformer
  feat          a linear map from the encoder to the teacher's 1024-d FastConformer frame
                (feature distillation; training only)

Frame alignment matches the teacher's cached features: frame t uses audio before (t + 1) * 80 ms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as m  # noqa: E402

SR = 16000
HOP = 160            # 10 ms
WIN = 400            # 25 ms
N_FFT = 512
SUB = 8              # 10 ms frames per 80 ms frame
FRAME = HOP * SUB    # 1280 samples


def mel_filterbank(n_mels=80, n_fft=N_FFT, sr=SR, fmin=0.0, fmax=8000.0):
    """[n_fft // 2 + 1, n_mels] triangular filters on the HTK mel scale, Slaney-normalized."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(x):
        return 700.0 * (10 ** (x / 2595.0) - 1.0)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((len(freqs), n_mels), np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[:, i] = np.maximum(0, np.minimum(up, down)) * 2.0 / (hi - lo)
    return fb


class MelFrontend(nn.Module):
    """wav [N, S] float -> log-mel [N, S // HOP, n_mels]. Frame k covers samples
    [(k + 1) * HOP - WIN, (k + 1) * HOP): causal, zero-padded on the left."""

    def __init__(self, n_mels=80):
        super().__init__()
        n = np.arange(WIN)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / WIN)
        k = np.arange(N_FFT // 2 + 1)[:, None]
        ang = 2 * np.pi * k * n[None] / N_FFT
        basis = np.concatenate([np.cos(ang) * window, -np.sin(ang) * window]).astype(np.float32)
        self.register_buffer('basis', torch.from_numpy(basis)[:, None])           # [2 * bins, 1, WIN]
        self.register_buffer('fb', torch.from_numpy(mel_filterbank(n_mels)))     # [bins, n_mels]

    def forward(self, wav):
        with torch.autocast(wav.device.type, enabled=False):  # fp32 even under autocast: wide dynamic range
            x = F.pad(wav.float()[:, None], (WIN - HOP, 0))
            spec = F.conv1d(x, self.basis, stride=HOP)                            # [N, 2 * bins, K]
            re, im = spec.chunk(2, 1)
            power = (re * re + im * im).transpose(1, 2)                           # [N, K, bins]
            return torch.log(power @ self.fb + 1e-6)


class CausalConv(nn.Module):
    """Depthwise causal conv over 80 ms frames with a pre-norm and a pointwise gate."""

    def __init__(self, dim, kernel=9, dropout=0.1):
        super().__init__()
        self.kernel = kernel
        self.norm = nn.LayerNorm(dim)
        self.pw_in = nn.Linear(dim, 2 * dim)
        self.dw = nn.Conv1d(dim, dim, kernel, groups=dim)
        self.pw_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # [N, T, D]
        h = F.glu(self.pw_in(self.norm(x)), -1).transpose(1, 2)
        h = self.dw(F.pad(h, (self.kernel - 1, 0)))
        return x + self.drop(self.pw_out(F.silu(h.transpose(1, 2))))


class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=4, window_s=5.6, kernel=9, n_mels=80, stem=128, dropout=0.1):
        super().__init__()
        self.mel = MelFrontend(n_mels)
        self.mel_norm = nn.LayerNorm(n_mels)
        self.stem = nn.Conv1d(n_mels, stem, 5)             # causal over 10 ms frames (left pad 4)
        self.proj = nn.Sequential(nn.Linear(SUB * stem, dim), nn.GELU(), nn.Linear(dim, dim))
        window = int(round(window_s / m.FRAME_S))
        self.convs = nn.ModuleList(CausalConv(dim, kernel, dropout) for _ in range(layers))
        self.blocks = nn.ModuleList(m.CausalBlock(dim, heads, window, dropout=dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.window = window

    def forward(self, wav):
        """wav [N, T * FRAME] -> [N, T, dim]."""
        N = wav.shape[0]
        x = self.mel_norm(self.mel(wav))                   # [N, 8T, mels]
        x = F.gelu(self.stem(F.pad(x.transpose(1, 2), (4, 0)))).transpose(1, 2)
        T = x.shape[1] // SUB
        x = self.proj(x[:, :T * SUB].reshape(N, T, -1))
        for conv, block in zip(self.convs, self.blocks):
            x = block(conv(x))
        return self.norm(x)


class Student(nn.Module):
    """forward(wav [B, S, C]) with C = 2 channels (one per speaker) -> TurnModel outputs + 'feat'."""

    def __init__(self, dim=256, layers=6, heads=4, window_s=5.6, kernel=9, head_dim=192, head_layers=3,
                 head_window_s=20.0, feat_dim=1024, dropout=0.1, yld=True, vad_in=False):
        super().__init__()
        self.encoder = Encoder(dim, layers, heads, window_s, kernel, dropout=dropout)
        self.head = m.TurnModel(tap_layers=0, final_dim=dim, dim=head_dim, heads=4, layers=head_layers,
                                window_s=head_window_s, dropout=dropout, yld=yld)
        self.feat = nn.Linear(dim, feat_dim) if feat_dim else None
        self.vad_proj = nn.Linear(2, dim) if vad_in else None  # Silero VAD per frame (augment.vad_frames)

    def encode(self, wav):
        B, S, C = wav.shape
        z = self.encoder(wav.permute(0, 2, 1).reshape(B * C, S))
        return z.view(B, C, z.shape[1], -1).permute(0, 2, 1, 3)  # [B, T, C, D]

    def forward(self, wav, with_feat=False, vad=None):
        """vad: uint8 [B, T, C, 2] (augment.vad_frames), required when built with vad_in."""
        z = self.encode(wav)
        if self.vad_proj is not None:
            z = z + self.vad_proj(vad[:, :z.shape[1]].to(z.dtype) / 255)
        out = self.head(None, z)
        if with_feat and self.feat is not None:
            out['feat'] = self.feat(z)
        return out


def count_params(net, skip=('feat.', 'head.slot_', 'head.source', 'head.mono')):
    """Parameters used at inference (stereo): drops the training-only feature map and the unused
    mono-slot heads."""
    return sum(p.numel() for n, p in net.named_parameters() if not n.startswith(skip))


PRESETS = {  # name: Student kwargs
    's8m': dict(dim=256, layers=6, head_dim=192, head_layers=3),
    's3m': dict(dim=192, layers=4, head_dim=128, head_layers=2),
    's1m': dict(dim=128, layers=4, head_dim=96, head_layers=2),
}


def context_frames(cfg):
    """Left context (80 ms frames) after which a Student output equals the full-sequence one."""
    if cfg.get('arch') == 'fc':
        import student_fc
        return student_fc.context_frames(cfg)
    enc = cfg.get('layers', 6) * (int(round(cfg.get('window_s', 5.6) / m.FRAME_S)) - 1 + cfg.get('kernel', 9) - 1) + 2
    return enc + m.context_frames(cfg.get('head_layers', 3), cfg.get('head_window_s', 20.0))


def build(cfg):
    if cfg.get('arch') == 'fc':  # bottom layers of the pretrained FastConformer (needs NeMo)
        import student_fc
        return student_fc.build(cfg)
    kw = {**PRESETS.get(cfg.get('preset', ''), {}), **{k: v for k, v in cfg.items() if k in (
        'dim', 'layers', 'heads', 'window_s', 'kernel', 'head_dim', 'head_layers', 'head_window_s', 'feat_dim',
        'dropout', 'yld', 'vad_in')}}
    return Student(**kw)


if __name__ == '__main__':
    for name, kw in PRESETS.items():
        net = Student(**kw)
        print(name, f'{count_params(net) / 1e6:.2f}M inference params',
              f'(encoder {sum(p.numel() for p in net.encoder.parameters()) / 1e6:.2f}M)')
