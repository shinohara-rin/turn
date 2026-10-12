"""Streaming step of the distilled student (student.py), K 80 ms frames per call, for ONNX export.

One graph does what the deployed pipeline needs two for (web/stream_encoder.StreamStep, a 114M
FastConformer, then web/stream_head.HeadStep): raw 16 kHz samples in, floor outputs out.

State per call (all fixed-size tensors):
  wav_tail [2, 240]                last samples of the previous call (the 25 ms mel window)
  mel_tail [2, 4, 80]              last normalized mel frames (the stem conv's left context)
  conv_i   [2, dim, kernel - 1]    per encoder layer, the depthwise conv's left context
  enc_k / enc_v [layers, 2, heads, We - 1, dh]  encoder attention KV caches (We = window frames)
  head_k / head_v                  the head's KV caches (stream_head.HeadStep layout)
  n_enc / n_head [1] int64         valid cached frames (min(t, W - 1) at chunk start t)
Outputs per frame: post [K, 4, 4], silent [K, 2], fine [K, 2, 18], yld [K, 2], eot [K, 2]
(eot_q), int [K, 2] (int_ft, the fine head's floor-taking probability), then the new state.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'web'))
import student as st  # noqa: E402
import stream_head as sh  # noqa: E402
import labels as lb  # noqa: E402
import model as tm  # noqa: E402

NEG = sh.NEG


class StudentStep(nn.Module):
    def __init__(self, net: st.Student, K=1):
        super().__init__()
        self.net, self.K = net, K
        enc = net.encoder
        self.enc = enc
        self.We = enc.window
        self.H = enc.blocks[0].heads
        self.D = enc.norm.normalized_shape[0]
        self.kernel = enc.convs[0].kernel
        self.head = sh.HeadStep(net.head, K)
        ft = lb.FINE_GROUPS['floor_taking']
        self.register_buffer('ft_mask', torch.zeros(len(lb.FINE)).index_fill_(0, torch.tensor(ft), 1.0))

    def attend(self, blk, x, kc, vc, n):
        """Self-attention block with a KV cache (stream_head.HeadStep.attend, encoder window)."""
        C, Q, D = x.shape
        H, dh, W = self.H, D // self.H, self.We
        h = blk.norm_q(x)
        q = blk.q(h).view(C, Q, H, dh).transpose(1, 2)
        k, v = blk.kv(h).view(C, Q, 2, H, dh).permute(2, 0, 3, 1, 4)
        K, V = torch.cat([kc, k], 2), torch.cat([vc, v], 2)
        key = torch.arange(W - 1 + Q)
        delta = (W - 1 + torch.arange(Q))[:, None] - key[None]
        ok = (delta >= 0) & (delta < W) & (key[None] >= W - 1 - n)
        bias = -blk.slopes[:, None, None] * delta.to(x.dtype)[None]
        bias = torch.where(ok[None], bias, torch.full_like(bias, NEG))
        att = ((q @ K.transpose(-1, -2)) * dh ** -0.5 + bias[None]).softmax(-1)
        y = (att @ V).transpose(1, 2).reshape(C, Q, D)
        x = x + blk.out(y)
        return x + blk.ff(x), K[:, :, Q:], V[:, :, Q:]

    def encode(self, audio, wav_tail, mel_tail, convs, enc_k, enc_v, n_enc):
        enc = self.enc
        x = torch.cat([wav_tail, audio], 1)                                  # [2, 240 + K * 1280]
        spec = F.conv1d(x[:, None], enc.mel.basis, stride=st.HOP)            # [2, 2 * bins, 8K]
        re, im = spec.chunk(2, 1)
        mel = enc.mel_norm(torch.log((re * re + im * im).transpose(1, 2) @ enc.mel.fb + 1e-6))
        m = torch.cat([mel_tail, mel], 1)                                   # [2, 4 + 8K, 80]
        h = F.gelu(enc.stem(m.transpose(1, 2))).transpose(1, 2)             # [2, 8K, stem]
        h = enc.proj(h.reshape(2, self.K, -1))                              # [2, K, D]
        new_convs, ks, vs = [], [], []
        for i, (conv, blk) in enumerate(zip(enc.convs, enc.blocks)):
            g = F.glu(conv.pw_in(conv.norm(h)), -1).transpose(1, 2)         # [2, D, K]
            gc = torch.cat([convs[i], g], 2)
            new_convs.append(gc[:, :, -(self.kernel - 1):])
            h = h + conv.pw_out(F.silu(conv.dw(gc).transpose(1, 2)))
            h, k, v = self.attend(blk, h, enc_k[i], enc_v[i], n_enc)
            ks.append(k)
            vs.append(v)
        z = enc.norm(h)
        return z, x[:, -(st.WIN - st.HOP):], m[:, -4:], new_convs, torch.stack(ks), torch.stack(vs)

    def forward(self, audio, n_enc, n_head, wav_tail, mel_tail, enc_k, enc_v, head_k, head_v, *convs):
        z, wav_tail, mel_tail, convs, enc_k, enc_v = self.encode(audio, wav_tail, mel_tail, list(convs), enc_k,
                                                                 enc_v, n_enc)
        net, hd = self.net.head, self.head
        x = net.final_proj(z) + net.channel.mean(0) + net.source.weight[tm.REAL_STEREO]
        ks, vs = [], []
        for i, blk in enumerate(hd.blocks):
            src = blk.norm_kv(x.flip(0)) if blk.cross else blk.norm_q(x)
            x, k, v = hd.attend(blk, x, src, head_k[i], head_v[i], n_head)
            ks.append(k)
            vs.append(v)
        x = net.norm(x).transpose(0, 1)                                      # [K, 2, D]
        pair = torch.cat([x, x.flip(1)], -1)
        held = net.held(pair).transpose(1, 2)
        shared = net.shared(x.sum(1)).view(-1, net.steps, 2)
        post = torch.cat([held, shared], -1).softmax(-1)
        silent = net.act(pair).softmax(-1)[..., 0]
        fine = net.fine(pair).softmax(-1)
        yld = net.yld(pair)[..., 0].sigmoid() if net.yld is not None else silent * 0
        now = post[:, 0]
        released = torch.stack([now[:, 2] + now[:, 1], now[:, 2] + now[:, 0]], -1)
        eot = (released * silent).clamp(0, 1)
        intr = (fine @ self.ft_mask).clamp(0, 1)
        return (post, silent, fine, yld, eot, intr, wav_tail, mel_tail, enc_k, enc_v, torch.stack(ks),
                torch.stack(vs), *convs)

    def init_state(self):
        L, dh = len(self.enc.blocks), self.D // self.H
        hk, hv = self.head.init_state()
        return dict(wav_tail=torch.zeros(2, st.WIN - st.HOP), mel_tail=torch.zeros(2, 4, self.enc.mel_norm.normalized_shape[0]),
                    enc_k=torch.zeros(L, 2, self.H, self.We - 1, dh), enc_v=torch.zeros(L, 2, self.H, self.We - 1, dh),
                    head_k=hk, head_v=hv,
                    convs=[torch.zeros(2, self.D, self.kernel - 1) for _ in range(L)])


def run_stream(step, wav):
    """wav [S, 2] float -> per-frame outputs over the whole signal, K frames per call (reference)."""
    K = step.K
    T = wav.shape[0] // st.FRAME // K * K
    s = step.init_state()
    outs = []
    with torch.no_grad():
        for t in range(0, T, K):
            audio = wav[t * st.FRAME:(t + K) * st.FRAME].T.contiguous()
            n_enc = torch.tensor([min(t, step.We - 1)])
            n_head = torch.tensor([min(t, step.head.W - 1)])
            o = step(audio, n_enc, n_head, s['wav_tail'], s['mel_tail'], s['enc_k'], s['enc_v'], s['head_k'],
                     s['head_v'], *s['convs'])
            outs.append(o[:6])
            s = dict(wav_tail=o[6], mel_tail=o[7], enc_k=o[8], enc_v=o[9], head_k=o[10], head_v=o[11],
                     convs=list(o[12:]))
    return [torch.cat([o[i] for o in outs]) for i in range(6)]


def state_names(L):
    return ['wav_tail', 'mel_tail', 'enc_k', 'enc_v', 'head_k', 'head_v'] + [f'conv_{i}' for i in range(L)]


def export(net, path, K=1):
    step = StudentStep(net.eval(), K).eval()
    s = step.init_state()
    L = len(s['convs'])
    args = (torch.zeros(2, K * st.FRAME), torch.tensor([3]), torch.tensor([3]), s['wav_tail'], s['mel_tail'],
            s['enc_k'], s['enc_v'], s['head_k'], s['head_v'], *s['convs'])
    names = state_names(L)
    torch.onnx.export(step, args, path, opset_version=17, do_constant_folding=True, dynamo=False,
                      input_names=['audio', 'n_enc', 'n_head'] + names,
                      output_names=['post', 'silent', 'fine', 'yld', 'eot', 'int'] + [n + '_out' for n in names])
    return step
