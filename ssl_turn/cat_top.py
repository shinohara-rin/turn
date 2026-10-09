"""Fine-tunable upper layers of the Cat encoder, run from cached lower-layer taps.

Cat's 12.5 Hz top stage is 32 causal transformer layers (d1280, RoPE, 10 s window) and an
output projection to 768. Features cached by encode.py include tap 15 (the output of
top-stage layer 15), so layers 16-31 + the projection can be re-run, and adapted, from the
cache without ever touching audio or the lower 0.6B parameters. RoPE is relative, so a crop
taken mid-conversation sees the same positions as the full encode; only the 10 s window per
layer means a crop needs left context (warm-up frames) to match the full encode.

Adaptation is LoRA on every linear map of the tuned layers (attention in/out, both FFN
matrices); the pretrained weights stay frozen.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

WINDOW = 125  # frames: the top stage's 10 s attention window at 12.5 Hz


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x with W frozen and B initialised to zero (starts exact)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale


class CatTop(nn.Module):
    """Top-stage layers [first, 32) + output projection, from tap `first - 1`.

    forward(x [N, T, 1280]) -> {layer index: [N, T, 1280] for each requested tap, 'final': [N, T, 768]}
    """

    def __init__(self, cat_encoder, first=16, taps=(23, 31), rank=16, alpha=32.0, dropout=0.0,
                 train_norms=False, checkpoint=False, base_dtype=None):
        super().__init__()
        top = cat_encoder.encoder[-1]
        self.first = first
        self.layers = copy.deepcopy(nn.ModuleList(top.transformer.layers[first:]))  # leave the encoder intact
        self.output_proj = copy.deepcopy(top.output_proj)
        self.taps = tuple(taps)
        self.checkpoint = checkpoint
        for p in self.parameters():
            p.requires_grad_(False)
        if base_dtype is not None:  # frozen matmul weights stored in the autocast dtype: no per-step casts
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.to(base_dtype)
        if rank:
            for layer in self.layers:
                attn = layer.self_attn
                attn.in_projs[0] = LoRALinear(attn.in_projs[0], rank, alpha, dropout)
                attn.out_projs[0] = LoRALinear(attn.out_projs[0], rank, alpha, dropout)
                layer.linear1 = LoRALinear(layer.linear1, rank, alpha, dropout)
                layer.linear2 = LoRALinear(layer.linear2, rank, alpha, dropout)
        if train_norms:
            for layer in self.layers:
                for p in list(layer.norm1.parameters()) + list(layer.norm2.parameters()):
                    p.requires_grad_(True)

    @property
    def context(self):
        """Left context (frames) after which outputs equal the full-sequence encode."""
        return len(self.layers) * (self.layers[0].self_attn.context - 1)

    def forward(self, x):
        out = {}
        for i, layer in enumerate(self.layers, start=self.first):
            if self.checkpoint and self.training:  # recompute activations in backward: less VRAM, +~1/3 compute
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if i in self.taps:
                out[i] = x
        out['final'] = self.output_proj(x)
        return out


class Tuned(nn.Module):
    """CatTop in front of a floor head: tap (first - 1) in, head taps/final from the adapted layers."""

    def __init__(self, top, head, taps):
        super().__init__()
        self.top, self.head, self.taps = top, head, list(taps)

    def forward(self, x, warm=0):
        """x [B, T, C, 1280] cached tap (first - 1); the first `warm` frames are context only."""
        B, T, C, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * C, T, D)
        out = self.top(h)
        out[self.top.first - 1] = h
        taps = torch.stack([out[t].to(out['final'].dtype) for t in self.taps], -2) if self.taps else None
        if taps is not None:
            taps = taps.view(B, C, T, len(self.taps), D).permute(0, 2, 1, 3, 4)[:, warm:]
        final = out['final'].view(B, C, T, -1).permute(0, 2, 1, 3)[:, warm:]
        return self.head(taps, final)


def trainable_state(module):
    names = {n for n, p in module.named_parameters() if p.requires_grad}
    return {k: v.detach().clone() for k, v in module.state_dict().items() if k in names}
