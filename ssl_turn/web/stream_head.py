"""Streaming step of the ssl_turn head (TurnModel, stereo) with attention KV caches, for ONNX export.
K frames per call (K = 1 by default).

TurnModel runs causal ALiBi attention over a 20 s window (W = 250 frames): per layer a self
block per channel, then a cross block whose keys are the other channel's self-block output.
Offline inference (train.infer_all) re-runs the window per chunk; here each block keeps the
keys/values of the last W - 1 frames, so a step costs K queries per channel per block.

Outputs per step, as score.score_variants reads them: floor posteriors now and at 0.4/0.8/1.6 s
(HELD_0, HELD_1, OPEN, CONTESTED), per-speaker p(SILENT) and fine-label posteriors, and the two
score tracks the bgbench commit policy uses: eot_q and int_nobc (one value per speaker).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import labels as lb  # noqa: E402
import model as tm  # noqa: E402

NEG = -1e4  # masked logit; ALiBi biases stay above -70, and -1e4 is safe in fp16


def build(cfg):
    return tm.TurnModel(tap_layers=0, final_dim=1024, dim=cfg.get('dim', 256), heads=cfg.get('heads', 4),
                        layers=cfg.get('layers', 4), window_s=cfg.get('window_s', 20.0), dropout=0.0)


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    assert cfg.get('feats') == 'asr' and not cfg.get('taps') and not cfg.get('enroll'), cfg
    net = build(cfg)
    net.load_state_dict(ck['state'])
    return net.eval(), cfg


class HeadStep(nn.Module):
    """inputs: feat [2, K, 1024] (channel = speaker), n [1] int64 (valid cached past frames,
    min(t, W - 1) for a chunk starting at frame t), kcache/vcache [2 * layers, 2, heads, W - 1, dh]
    (self and cross blocks interleaved). outputs, per frame of the chunk: post [K, 4, 4],
    silent [K, 2], fine [K, 2, 18], eot [K, 2], int [K, 2]; then the updated caches."""

    def __init__(self, net, K=1):
        super().__init__()
        self.net, self.K = net, K
        self.blocks = [b for pair in zip(net.self_blocks, net.cross_blocks) for b in pair]
        self.W = self.blocks[0].window
        self.H = self.blocks[0].heads
        self.dim = net.norm.normalized_shape[0]
        bc = lb.FINE_GROUPS['backchannel'] + lb.FINE_GROUPS['noncontent']
        self.register_buffer('bc_mask', torch.zeros(len(lb.FINE)).index_fill_(0, torch.tensor(bc), 1.0))

    def attend(self, blk, x, src, kc, vc, n):
        C, Q, D = x.shape
        H, dh = self.H, D // self.H
        q = blk.q(blk.norm_q(x)).view(C, Q, H, dh).transpose(1, 2)        # [C, H, Q, dh]
        k, v = blk.kv(src).view(C, Q, 2, H, dh).permute(2, 0, 3, 1, 4)
        K, V = torch.cat([kc, k], 2), torch.cat([vc, v], 2)              # [C, H, W - 1 + Q, dh]
        key = torch.arange(self.W - 1 + Q)
        delta = (self.W - 1 + torch.arange(Q))[:, None] - key[None]      # [Q, keys] distance to query
        ok = (delta >= 0) & (delta < self.W) & (key[None] >= self.W - 1 - n)
        bias = -blk.slopes[:, None, None] * delta.to(x.dtype)[None]      # [H, Q, keys]
        bias = torch.where(ok[None], bias, torch.full_like(bias, NEG))
        att = ((q @ K.transpose(-1, -2)) * dh ** -0.5 + bias[None]).softmax(-1)
        y = (att @ V).transpose(1, 2).reshape(C, Q, D)
        x = x + blk.out(y)
        return x + blk.ff(x), K[:, :, Q:], V[:, :, Q:]

    def forward(self, feat, n, kcache, vcache):
        net = self.net
        x = net.final_proj(feat) + net.channel.mean(0) + net.source.weight[tm.REAL_STEREO]   # [2, K, D]
        ks, vs = [], []
        for i, blk in enumerate(self.blocks):
            src = blk.norm_kv(x.flip(0)) if blk.cross else blk.norm_q(x)
            x, k, v = self.attend(blk, x, src, kcache[i], vcache[i], n)
            ks.append(k)
            vs.append(v)
        x = net.norm(x).transpose(0, 1)                                  # [K, 2, D]
        pair = torch.cat([x, x.flip(1)], -1)
        held = net.held(pair).transpose(1, 2)                            # [K, steps, 2]
        shared = net.shared(x.sum(1)).view(-1, net.steps, 2)
        post = torch.cat([held, shared], -1).softmax(-1)                 # [K, steps, 4]
        silent = net.act(pair).softmax(-1)[..., 0]                       # [K, 2]
        fine = net.fine(pair).softmax(-1)                                # [K, 2, 18]
        now, f04 = post[:, 0], post[:, 1]
        released = torch.stack([now[:, 2] + now[:, 1], now[:, 2] + now[:, 0]], -1)
        eot = (released * silent).clamp(0, 1)
        not_bc = (1 - fine @ self.bc_mask).clamp(0, 1)
        intr = (f04[:, :2] * (1 - silent) * not_bc).clamp(0, 1)
        return post, silent, fine, eot, intr, torch.stack(ks), torch.stack(vs)

    def init_state(self):
        shape = (len(self.blocks), 2, self.H, self.W - 1, self.dim // self.H)
        return torch.zeros(shape), torch.zeros(shape)
