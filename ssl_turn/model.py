"""Stereo causal turn model over frozen (or adapted) Cat encoder features at 12.5 Hz.

Per frame t (available at (t+1)*80 ms) and speaker s it predicts:
  - vap:  256-way discrete Voice Activity Projection over the next 2 s (shared, joint)
  - eot:  this speaker's current pause is a turn end
  - int:  the other speaker's current onset takes the floor from s (scored for the onset speaker)
  - vad:  speaker s is talking now (auxiliary, also usable as the commit gate)
VAP is self-supervised from per-channel activity, so it is the objective that
scales to unlabeled (or pseudo-labeled) two-channel audio.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_S = 0.08
# VAP projection bins (frames). Ekstedt & Skantze use 0.2/0.4/0.6/0.8 s; the 80 ms
# grid gives 0.24/0.40/0.56/0.80 s, still spanning the same 2.0 s horizon.
VAP_BINS = (3, 5, 7, 10)
VAP_HORIZON = sum(VAP_BINS)
VAP_CLASSES = 2 ** (2 * len(VAP_BINS))


def vap_labels(activity, bins=VAP_BINS, threshold=0.5):
    """activity [T, 2] in [0,1] at 12.5 Hz -> (class [T] int64, valid [T] bool).

    Bit order: speaker 0 bins (near->far), then speaker 1 bins. A bin is active when
    the speaker talks for >= threshold of it. Supervision only, never an input.
    """
    a = np.asarray(activity, np.float32)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('activity must be [T, 2]')
    T = len(a)
    c = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(a, 0)])
    labels = np.zeros(T, np.int64)
    bit = 0
    for s in range(2):
        start = 1  # projection starts at the next frame
        for width in bins:
            lo = np.minimum(np.arange(T) + start, T)
            hi = np.minimum(lo + width, T)
            frac = (c[hi, s] - c[lo, s]) / width
            labels |= (frac >= threshold).astype(np.int64) << bit
            bit += 1
            start += width
    valid = np.arange(T) + 1 + sum(bins) <= T
    return labels, valid


def alibi_slopes(heads):
    return torch.tensor([2 ** (-8 * (i + 1) / heads) for i in range(heads)])


class CausalBlock(nn.Module):
    """Pre-norm attention block; queries attend causally within `window` frames."""

    def __init__(self, dim, heads, window, cross=False, dropout=0.1):
        super().__init__()
        self.heads, self.window, self.cross = heads, window, cross
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim) if cross else None
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)
        self.register_buffer('slopes', alibi_slopes(heads), persistent=False)

    def bias(self, T, device, dtype):
        i = torch.arange(T, device=device)
        delta = (i[:, None] - i[None, :]).to(dtype)
        allowed = (delta >= 0) & (delta < self.window)
        bias = -self.slopes.to(device, dtype)[:, None, None] * delta
        return bias.masked_fill(~allowed, float('-inf'))

    def forward(self, x, context=None):
        B, T, D = x.shape
        h = self.norm_q(x)
        src = self.norm_kv(context) if self.cross else h
        q = self.q(h).view(B, T, self.heads, -1).transpose(1, 2)
        k, v = self.kv(src).view(B, T, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias(T, x.device, q.dtype),
                                           dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.out(y.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(x))


class TurnModel(nn.Module):
    """inputs: taps [B, T, 2, L, 1280] and/or final [B, T, 2, 768] (channel = speaker)."""

    def __init__(self, tap_layers=4, tap_dim=1280, final_dim=768, dim=256, heads=4, layers=4,
                 window_s=20.0, dropout=0.1):
        super().__init__()
        window = int(round(window_s / FRAME_S))
        self.tap_weights = nn.Parameter(torch.zeros(tap_layers)) if tap_layers else None
        self.tap_proj = nn.Sequential(nn.LayerNorm(tap_dim), nn.Linear(tap_dim, dim)) if tap_layers else None
        self.final_proj = nn.Sequential(nn.LayerNorm(final_dim), nn.Linear(final_dim, dim)) if final_dim else None
        self.channel = nn.Parameter(torch.zeros(2, dim))
        self.self_blocks = nn.ModuleList(CausalBlock(dim, heads, window, dropout=dropout) for _ in range(layers))
        self.cross_blocks = nn.ModuleList(CausalBlock(dim, heads, window, cross=True, dropout=dropout)
                                          for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.vap = nn.Linear(2 * dim, VAP_CLASSES)
        self.speaker = nn.Linear(2 * dim, 3)  # eot, int, vad from [own, other]

    def embed(self, taps=None, final=None):
        x = 0
        if self.tap_proj is not None:
            w = torch.softmax(self.tap_weights, 0)
            x = x + self.tap_proj((taps * w[:, None]).sum(-2))
        if self.final_proj is not None:
            x = x + self.final_proj(final)
        return x + self.channel  # [B, T, 2, dim]

    def forward(self, taps=None, final=None):
        x = self.embed(taps, final)
        B, T, _, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(2 * B, T, D)  # speakers as batch, shared weights
        for sa, ca in zip(self.self_blocks, self.cross_blocks):
            x = sa(x)
            other = x.view(B, 2, T, D).flip(1).reshape(2 * B, T, D)
            x = ca(x, other)
        x = self.norm(x).view(B, 2, T, D).permute(0, 2, 1, 3)  # [B, T, 2, D]
        pair = torch.cat([x, x.flip(2)], -1)  # own, other
        per_speaker = self.speaker(pair)
        return dict(vap=self.vap(torch.cat([x[:, :, 0], x[:, :, 1]], -1)),
                    eot=per_speaker[..., 0], int=per_speaker[..., 1], vad=per_speaker[..., 2])


def vap_speaker_probabilities(vap_logits, near_bins=2):
    """p(speaker s active in the first `near_bins` VAP bins), marginalised from 256 classes."""
    probs = vap_logits.softmax(-1)
    classes = torch.arange(VAP_CLASSES, device=vap_logits.device)
    out = []
    for s in range(2):
        bits = sum(((classes >> (s * len(VAP_BINS) + b)) & 1) for b in range(near_bins))
        out.append((probs * (bits > 0).to(probs.dtype)).sum(-1))
    return torch.stack(out, -1)


def loss(outputs, batch, weights=(1.0, 1.0, 1.0, 0.5)):
    """batch: vap [B,T] long, vap_valid [B,T] bool, and for k in eot/int/vad: k [B,T,2], k_w [B,T,2]."""
    terms = {}
    v = batch['vap_valid']
    terms['vap'] = F.cross_entropy(outputs['vap'][v], batch['vap'][v]) if v.any() else outputs['vap'].sum() * 0
    for name in ('eot', 'int', 'vad'):
        if name not in batch:
            continue
        w = batch[name + '_w']
        bce = F.binary_cross_entropy_with_logits(outputs[name], batch[name], reduction='none')
        terms[name] = (bce * w).sum() / w.sum().clamp_min(1e-6)
    total = sum(weights[i] * terms[k] for i, k in enumerate(('vap', 'eot', 'int', 'vad')) if k in terms)
    return total, {k: float(t.detach()) for k, t in terms.items()}
