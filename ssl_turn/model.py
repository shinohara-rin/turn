"""Causal floor-ownership model over frozen (or adapted) Cat encoder features at 12.5 Hz.

Two input modes share one trunk (see labels.py for the target definitions):
  stereo [.., T, 2, ..] (one channel per speaker; the TurnBench condition):
    - floor:  joint floor state now (HELD_0, HELD_1, OPEN, CONTESTED)
    - future: floor state at each projection horizon (how contests and pauses resolve)
    - act:    per-speaker vocal act (SILENT, CLAIM, BACKCHANNEL, LAUGHTER, NONCONTENT)
    - vap:    256-way Voice Activity Projection; activity only, so it scales to unlabeled stereo
  mono [.., T, 1, ..] (mixed-speaker audio), speakers in arrival-order slots:
    - floor/future/act over slots, plus slot_activity: streaming diarization with overlap
Floor heads are speaker-equivariant: HELD_c comes from a shared [own, other] head and
OPEN/CONTESTED from a symmetric pooled head, so swapping speakers swaps HELD_0/HELD_1.
Frame t is available at (t+1)*80 ms.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labels import ACTS, FINE, HORIZONS_S

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


# Data sources for heterogeneous training. The model is told which pipeline produced
# its input channels; evaluation always uses REAL_STEREO, the TurnBench condition.
REAL_STEREO, GATED_PODCAST, SEPARATED_PODCAST = 0, 1, 2
SOURCES = 3


class TurnModel(nn.Module):
    """inputs: taps [B, T, C, L, 1280] and/or final [B, T, C, 768], with C=2 (channel =
    speaker) or C=1 (mixed mono), plus optional source ids [B] (default REAL_STEREO)."""

    def __init__(self, tap_layers=4, tap_dim=1280, final_dim=768, dim=256, heads=4, layers=4,
                 window_s=20.0, dropout=0.1, sources=SOURCES, slots=2):
        super().__init__()
        window = int(round(window_s / FRAME_S))
        self.tap_weights = nn.Parameter(torch.zeros(tap_layers)) if tap_layers else None
        self.tap_proj = nn.Sequential(nn.LayerNorm(tap_dim), nn.Linear(tap_dim, dim)) if tap_layers else None
        self.final_proj = nn.Sequential(nn.LayerNorm(final_dim), nn.Linear(final_dim, dim)) if final_dim else None
        self.channel = nn.Parameter(torch.zeros(2, dim))
        self.mono = nn.Parameter(torch.zeros(dim))
        self.source = nn.Embedding(sources, dim)
        nn.init.zeros_(self.source.weight)
        self.self_blocks = nn.ModuleList(CausalBlock(dim, heads, window, dropout=dropout) for _ in range(layers))
        self.cross_blocks = nn.ModuleList(CausalBlock(dim, heads, window, cross=True, dropout=dropout)
                                          for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        steps = 1 + len(HORIZONS_S)  # now + projection horizons
        self.steps = steps
        self.vap = nn.Linear(2 * dim, VAP_CLASSES)
        self.held = nn.Linear(2 * dim, steps)        # HELD_c logit per step, from [own, other]
        self.shared = nn.Linear(dim, 2 * steps)      # OPEN, CONTESTED per step, from own + other
        self.act = nn.Linear(2 * dim, len(ACTS))
        self.fine = nn.Linear(2 * dim, len(FINE))   # per-speaker fine annotator label (labels.FINE)
        self.slots = slots
        self.slot_floor = nn.Linear(dim, steps * (slots + 2))
        self.slot_act = nn.Linear(dim, slots * len(ACTS))
        self.slot_fine = nn.Linear(dim, slots * len(FINE))
        self.slot_activity = nn.Linear(dim, slots)

    def embed(self, taps=None, final=None, source=None):
        x = 0
        if self.tap_proj is not None:
            w = torch.softmax(self.tap_weights, 0)
            x = x + self.tap_proj((taps * w[:, None]).sum(-2))
        if self.final_proj is not None:
            x = x + self.final_proj(final)
        x = x + (self.channel if x.shape[2] == 2 else self.mono)  # [B, T, C, dim]
        if source is None:
            source = torch.full((x.shape[0],), REAL_STEREO, dtype=torch.long, device=x.device)
        return x + self.source(source)[:, None, None]

    def forward(self, taps=None, final=None, source=None):
        x = self.embed(taps, final, source)
        B, T, C, D = x.shape
        if C not in (1, 2):
            raise ValueError('expected 1 (mono) or 2 (stereo) channels')
        x = x.permute(0, 2, 1, 3).reshape(C * B, T, D)  # channels as batch, shared weights
        for sa, ca in zip(self.self_blocks, self.cross_blocks):
            x = sa(x)
            if C == 2:
                x = ca(x, x.view(B, 2, T, D).flip(1).reshape(2 * B, T, D))
        x = self.norm(x).view(B, C, T, D).permute(0, 2, 1, 3)  # [B, T, C, D]
        if C == 1:
            h = x[:, :, 0]
            floor = self.slot_floor(h).view(B, T, self.steps, self.slots + 2)
            return dict(floor=floor[:, :, 0], future=floor[:, :, 1:],
                        act=self.slot_act(h).view(B, T, self.slots, len(ACTS)),
                        fine=self.slot_fine(h).view(B, T, self.slots, len(FINE)),
                        slot_activity=self.slot_activity(h))
        pair = torch.cat([x, x.flip(2)], -1)                       # [B, T, 2, 2D]: own, other
        held = self.held(pair).permute(0, 1, 3, 2)                 # [B, T, steps, 2]
        shared = self.shared(x.sum(2)).view(B, T, self.steps, 2)   # [B, T, steps, 2]
        floor = torch.cat([held, shared], -1)                      # HELD_0, HELD_1, OPEN, CONTESTED
        return dict(floor=floor[:, :, 0], future=floor[:, :, 1:], act=self.act(pair), fine=self.fine(pair),
                    vap=self.vap(torch.cat([x[:, :, 0], x[:, :, 1]], -1)))


def vap_speaker_probabilities(vap_logits, near_bins=2):
    """p(speaker s active in the first `near_bins` VAP bins), marginalised from 256 classes."""
    probs = vap_logits.softmax(-1)
    classes = torch.arange(VAP_CLASSES, device=vap_logits.device)
    out = []
    for s in range(2):
        bits = sum(((classes >> (s * len(VAP_BINS) + b)) & 1) for b in range(near_bins))
        out.append((probs * (bits > 0).to(probs.dtype)).sum(-1))
    return torch.stack(out, -1)


def _weighted_ce(logits, target, weight):
    """Hard (long, [...]) or soft (float, [..., C]) targets; soft ones carry hand-offs."""
    target = target.flatten(0, -2) if target.is_floating_point() else target.flatten()
    ce = F.cross_entropy(logits.flatten(0, -2), target, reduction='none')
    w = weight.flatten()
    return (ce * w).sum() / w.sum().clamp_min(1e-6)


def loss(outputs, batch, weights=None):
    """Targets (labels.floor_targets / floor_projection / to_slots), each with a *_w weight:
      floor [B,T,4] soft, future [B,T,H,4] soft, act [B,T,2] long, slot_activity [B,T,2] float (mono),
      plus vap [B,T] long with vap_valid [B,T] bool (stereo).

    Partial labels: a source lacking a target omits it or zeroes its weight, e.g. podcast
    mono with only pyannote output has slot_activity alone. Per-source trust (human vs
    pseudo) is folded into the weights by the loader. Each term is normalized by its own
    weight mass, so the sampler's mixing ratio, not raw volume, sets each source's share.
    """
    weights = {**dict(vap=1.0, floor=1.0, future=1.0, act=0.5, fine=0.5, slot_activity=1.0), **(weights or {})}
    terms = {}
    if 'vap' in outputs and 'vap' in batch:
        terms['vap'] = _weighted_ce(outputs['vap'], batch['vap'], batch['vap_valid'].float())
    for name in ('floor', 'future', 'act', 'fine'):
        if name in batch and name in outputs and weights.get(name, 0) > 0:
            terms[name] = _weighted_ce(outputs[name], batch[name], batch[name + '_w'])
    if 'slot_activity' in outputs and 'slot_activity' in batch:
        w = batch['slot_activity_w']
        bce = F.binary_cross_entropy_with_logits(outputs['slot_activity'], batch['slot_activity'], reduction='none')
        terms['slot_activity'] = (bce * w).sum() / w.sum().clamp_min(1e-6)
    total = sum(weights[k] * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}
