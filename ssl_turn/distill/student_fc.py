"""Student that keeps the teacher's own pretraining: the bottom `layers` blocks of the streaming
FastConformer (encode_asr.MODEL, attention context [70, 0], fine-tuned) under the same floor head.

The small students (student.py) match the teacher on TurnBench but not on new recordings
(eot-bench, background audio); this tests whether keeping some of the ASR-pretrained encoder
closes that gap at a fraction of the teacher's size (17 layers -> `layers`).

Frame alignment is encode_asr.py's: encoder frame k (audio up to (k + 1) * 80 ms plus the 16 ms
STFT half-window) is our frame k + 1, so every frame is causal; frame 0 is zeros.
Config keys: arch 'fc', layers (default 4), head_dim / head_layers / head_window_s, dropout, yld,
vad_in, feat_dim, enc_lr (training: the encoder's learning rate, default 1e-4).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'pipeline'))
import model as m  # noqa: E402

MODEL = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
FRAME = 1280


def load_encoder(layers):
    import nemo.collections.asr as nemo_asr
    asr = nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location='cpu').eval()
    enc = asr.encoder
    enc.set_default_att_context_size([70, 0])
    pre = asr.preprocessor
    pre.featurizer.dither = 0.0
    pre.featurizer.pad_to = 0
    norm = getattr(pre.featurizer, 'normalize', None)
    print(f'FastConformer: keeping {layers} of {len(enc.layers)} layers; mel normalize={norm!r}', flush=True)
    enc.layers = nn.ModuleList(list(enc.layers)[:layers])
    if hasattr(enc, 'layer_drop_probs'):
        enc.layer_drop_probs = list(enc.layer_drop_probs)[:layers]
    enc.n_layers = layers
    return pre, enc


class FCStudent(nn.Module):
    def __init__(self, layers=4, head_dim=192, head_layers=3, head_window_s=20.0, dropout=0.1, yld=True,
                 vad_in=False, feat_dim=1024):
        super().__init__()
        self.pre, self.enc = load_encoder(layers)
        d = self.enc.d_model
        self.head = m.TurnModel(tap_layers=0, final_dim=d, dim=head_dim, heads=4, layers=head_layers,
                                window_s=head_window_s, dropout=dropout, yld=yld)
        self.feat = nn.Linear(d, feat_dim) if feat_dim else None
        self.vad_proj = nn.Linear(2, d) if vad_in else None

    def encoder_parameters(self):
        return list(self.enc.parameters())

    def encode(self, wav):
        """wav [B, S, C] -> [B, T, C, d], T = S // 1280."""
        B, S, C = wav.shape
        T = S // FRAME
        x = wav.permute(0, 2, 1).reshape(B * C, S).float()
        n = torch.full((B * C,), S, device=x.device, dtype=torch.long)
        with torch.autocast(x.device.type, enabled=False):
            f, fl = self.pre(input_signal=x, length=n)
        e, _ = self.enc(audio_signal=f, length=fl)            # [BC, d, T']
        e = e.transpose(1, 2)
        z = e.new_zeros(B * C, T, e.shape[-1])
        k = min(T - 1, e.shape[1])
        z[:, 1:1 + k] = e[:, :k]                               # our frame k + 1 <- encoder frame k
        return z.view(B, C, T, -1).permute(0, 2, 1, 3)

    def forward(self, wav, with_feat=False, vad=None):
        z = self.encode(wav)
        if self.vad_proj is not None:
            z = z + self.vad_proj(vad[:, :z.shape[1]].to(z.dtype) / 255)
        out = self.head(None, z)
        if with_feat and self.feat is not None:
            out['feat'] = self.feat(z)
        return out


def build(cfg):
    return FCStudent(**{k: v for k, v in cfg.items() if k in (
        'layers', 'head_dim', 'head_layers', 'head_window_s', 'dropout', 'yld', 'vad_in', 'feat_dim')})


def context_frames(cfg):
    """Left context after which chunked inference equals the full-sequence one: per layer the
    attention window (70 frames) and the causal conv kernel (9), plus the subsampling convs."""
    return cfg.get("layers", 4) * (70 + 16) + 8 + m.context_frames(cfg.get('head_layers', 3),
                                                                 cfg.get('head_window_s', 20.0))
