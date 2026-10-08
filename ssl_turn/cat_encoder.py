"""Encoder half of OpenMOSS MOSS-Audio-Tokenizer ("Cat") as a causal feature extractor.

Cat is a 1.6B-parameter (encoder ~0.9B) pure causal-transformer codec trained from
scratch on ~3M hours of audio with reconstruction plus LLM semantic alignment.
Encoder: 24 kHz mono, patch 240 samples (100 Hz) -> 12L/d768 -> patch 2 (50 Hz) ->
12L/d768 -> patch 2 (25 Hz) -> 12L/d768 -> patch 2 (12.5 Hz) -> 32L/d1280 -> 768.
Every attention is causal with a 10 s window and patches never overlap, so frame k
depends only on samples [0, 1920*(k+1)) at 24 kHz: available at (k+1)*80 ms.

Only the pinned upstream module classes are used; the quantizer and decoder are
never built or downloaded. Primary sources (inspected 2026-10-08):
https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer (Apache-2.0)
https://arxiv.org/abs/2602.10934
"""
from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn as nn

CAT_REPO = 'OpenMOSS-Team/MOSS-Audio-Tokenizer'
CAT_REVISION = '3cd226ba2947efa357ef453bcad111b6eafba782'
CAT_CODE_SHA256 = {
    'configuration_moss_audio_tokenizer.py': '349b7ff7e1b3f160f9c80df9a0311672b326b8b73e90459122fb39e6878962bf',
    'modeling_moss_audio_tokenizer.py': '65cae7744845f1b8ac65957e918cea508efe331a38e87b882b7530b6c8d7caa5',
}
ENCODER_SHARD = 'model-00001-of-00002.safetensors'
SAMPLE_RATE = 24000
HOP = 1920  # samples per 12.5 Hz frame
FRAME_S = HOP / SAMPLE_RATE
TOP_LAYERS = 32
TOP_DIM = 1280
OUT_DIM = 768


def _sha256(path):
    from hf_slice import sha256
    return sha256(path)


def fetch_code(directory):
    """Download pinned upstream config/modeling code and verify hashes."""
    from huggingface_hub import hf_hub_download
    directory = Path(directory)
    for name in ('config.json', *CAT_CODE_SHA256):
        hf_hub_download(CAT_REPO, name, revision=CAT_REVISION, local_dir=directory)
    for name, digest in CAT_CODE_SHA256.items():
        if _sha256(directory / name) != digest:
            raise ValueError(f'{name} does not match pinned SHA256')
    return directory


def load_upstream(directory):
    """Import the pinned modeling file as a private package; returns (module, config)."""
    directory = Path(directory).resolve()
    for name, digest in CAT_CODE_SHA256.items():
        if _sha256(directory / name) != digest:
            raise ValueError(f'{name} does not match pinned SHA256')
    package = '_cat_upstream_' + CAT_REVISION[:12]
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, directory / '__init__.py',
                                                      submodule_search_locations=[str(directory)])
        if not (directory / '__init__.py').exists():
            (directory / '__init__.py').write_text('')
        sys.modules[package] = importlib.util.module_from_spec(spec)
    modeling = importlib.import_module(package + '.modeling_moss_audio_tokenizer')
    configuration = importlib.import_module(package + '.configuration_moss_audio_tokenizer')
    config = configuration.MossAudioTokenizerConfig(**json.loads((directory / 'config.json').read_text()))
    return modeling, config


def fetch_encoder_weights(directory, out_name='cat_encoder.safetensors'):
    """Range-read only encoder.* tensors from the pinned shard (~3.5 GB fp32, not 7 GB)."""
    from hf_slice import fetch_tensors
    out = Path(directory) / out_name
    return out, fetch_tensors(CAT_REPO, CAT_REVISION, ENCODER_SHARD, 'encoder.', out)


class CatEncoder(nn.Module):
    """Causal Cat encoder with optional taps on the 12.5 Hz top-stage layers.

    forward(wave [B, samples] at 24 kHz) -> dict(final=[B, T, 768], taps=[B, T, L, 1280])
    with T = samples // 1920. Use `stream()` for long conversations: it keeps the
    upstream ring KV caches, matching the full-sequence result for the same window.
    """

    def __init__(self, modeling, config, taps=(7, 15, 23, 31)):
        super().__init__()
        rate = float(config.sampling_rate)
        self.encoder = nn.ModuleList()
        for kwargs in config.encoder_kwargs:
            kwargs = dict(kwargs)
            if kwargs['module_type'] == 'PatchedPretransform':
                self.encoder.append(modeling.MossAudioTokenizerPatchedPretransform(**kwargs, is_downsample=True))
            else:
                self.encoder.append(modeling.MossAudioTokenizerProjectedTransformer(
                    **kwargs, context=int(rate * config.causal_transformer_context_duration)))
            rate /= self.encoder[-1].downsample_ratio
        self.hop = int(config.downsample_rate)
        self.streaming_module = modeling.StreamingModule
        self.ring_cache = modeling.RingKVCache
        self.attention = modeling.MossAudioTokenizerMultiheadAttention
        top = self.encoder[-1].transformer.layers
        self.taps = tuple(int(t) for t in taps)
        if any(t < 0 or t >= len(top) for t in self.taps):
            raise ValueError('tap outside top-stage layers')
        self._captured = {}
        for t in self.taps:
            top[t].register_forward_hook(lambda m, i, o, t=t: self._captured.__setitem__(t, o))

    def load_encoder_state(self, path):
        from safetensors.torch import load_file
        state = load_file(str(path))
        state = {k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(state, strict=True)
        return self

    def forward(self, wave):
        if wave.ndim != 2:
            raise ValueError('wave must be [B, samples]')
        usable = wave.shape[1] // self.hop * self.hop
        x = wave[:, None, :usable]
        lengths = torch.full((wave.shape[0],), usable, dtype=torch.long, device=wave.device)
        self._captured.clear()
        for module in self.encoder:
            x, lengths = module(x, lengths)
        taps = [self._captured[t] for t in self.taps]
        return dict(final=x.transpose(1, 2), taps=torch.stack(taps, dim=2) if taps else None)

    def _widen_caches(self, chunk_frames):
        """Upstream sizes each ring cache to exactly `context` keys, so a chunk of T
        tokens evicts keys its earliest queries still need: chunked encodes drift
        from the full-sequence result once audio exceeds the window. The attention
        mask already enforces `delta < context` by position, so extra capacity
        restores exact equivalence for any chunk size."""
        for module in self.encoder.modules():
            if isinstance(module, self.attention) and module._streaming_state is not None:
                old = module._streaming_state.kv_cache
                _, batch, heads, _, dim = old.cache.shape
                # Lowest stage runs at 8 tokens per 12.5 Hz frame.
                module._streaming_state.kv_cache = self.ring_cache(
                    batch, heads, dim, module.context + 8 * chunk_frames,
                    respect_exec_mask=old.respect_exec_mask, device=old.cache.device, dtype=old.cache.dtype)

    def stream(self, wave, chunk_frames=25):
        """Chunked streaming encode, identical to forward() on the whole sequence."""
        if chunk_frames < 1:
            raise ValueError('chunk_frames must be positive')
        step = chunk_frames * self.hop
        usable = wave.shape[1] // self.hop * self.hop
        finals, taps = [], []
        with ExitStack() as stack:
            for module in self.encoder:
                if isinstance(module, self.streaming_module):
                    stack.enter_context(module.streaming(batch_size=wave.shape[0]))
            self._widen_caches(chunk_frames)
            for start in range(0, usable, step):
                out = self(wave[:, start:min(start + step, usable)])
                finals.append(out['final'])
                if out['taps'] is not None:
                    taps.append(out['taps'])
        return dict(final=torch.cat(finals, 1), taps=torch.cat(taps, 1) if taps else None)


def available_at(frames, resample_delay_s=0.0):
    """Exclusive audio endpoint (s) at which each 12.5 Hz frame becomes computable."""
    import numpy as np
    return (np.arange(frames) + 1) * FRAME_S + resample_delay_s


def build(code_dir, weights=None, taps=(7, 15, 23, 31), device='cpu', dtype=torch.float32):
    modeling, config = load_upstream(code_dir)
    encoder = CatEncoder(modeling, config, taps=taps)
    if weights is not None:
        encoder.load_encoder_state(weights)
    return encoder.to(device=device, dtype=dtype).eval()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Fetch pinned Cat code and encoder-only weights.')
    ap.add_argument('directory')
    args = ap.parse_args()
    fetch_code(args.directory)
    print(json.dumps(fetch_encoder_weights(args.directory)[1], indent=2))
