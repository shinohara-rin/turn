"""MOSS-Transcribe-Diarize's audio encoder as an exactly causal feature extractor.

MOSS-Transcribe-Diarize 0.9B (OpenMOSS, Apache-2.0, arXiv 2601.01554) is a
Whisper-medium-shaped encoder (24L, d1024, 80-bin log-mel, 50 Hz), merged 4x to
12.5 Hz and fed to a Qwen3-0.6B decoder. It is trained end to end for
speaker-attributed, timestamped transcription in 50+ languages, and won the 2026
MLC-SLM challenge (two-speaker conversations in 14 languages).

The encoder is *not* causal: its processor cuts audio into independent 30 s chunks,
zero-pads each one at the end, normalizes each chunk's log-mel by that chunk's own
maximum, and attends bidirectionally within the chunk. The trailing-window mode
below is still exactly causal. At each step it re-encodes the last <= 30 s ending
*now*, zero-padded on the right, and takes the final 12.5 Hz token. This is how
the model saw the last chunk of every training file, so it is in distribution. It
is expensive: one full 30 s encoder pass per decision step. That is fine for frozen
probes and offline teacher use; deployment would need a causal student.

Primary sources (inspected 2026-10-08):
https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
https://arxiv.org/abs/2601.01554
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MTD_REPO = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'
MTD_REVISION = '704aa4a9c304e8520be88901e0d1960158ef5b15'
MTD_SHARD = 'model-00000-of-00001.safetensors'
PREFIX = 'model.whisper_encoder.'
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 480000      # processor n_samples (30 s)
TOKEN_SAMPLES = 1280        # hop 160 x encoder stride 2 x merge 4 = one 12.5 Hz token
MERGE = 4
DIM = 1024


def fetch(directory):
    """Pinned config plus encoder-only weights (range-read from the 0.9B checkpoint)."""
    from huggingface_hub import hf_hub_download
    from hf_slice import fetch_tensors
    directory = Path(directory)
    hf_hub_download(MTD_REPO, 'config.json', revision=MTD_REVISION, local_dir=directory)
    return fetch_tensors(MTD_REPO, MTD_REVISION, MTD_SHARD, PREFIX, directory / 'mtd_encoder.safetensors')


def whisper_config(directory=None, **overrides):
    from transformers import WhisperConfig
    audio = json.loads((Path(directory) / 'config.json').read_text())['audio_config'] if directory else {}
    audio.update(overrides)
    audio.pop('model_type', None)
    return WhisperConfig(**audio)


def feature_extractor(num_mel_bins=80):
    from transformers import WhisperFeatureExtractor
    return WhisperFeatureExtractor(feature_size=num_mel_bins, sampling_rate=SAMPLE_RATE, hop_length=160,
                                   chunk_length=30, n_fft=400)


class TrailingWindowEncoder(nn.Module):
    """forward(wave [N] at 16 kHz, steps) -> dict(final [K, 4096], taps [K, L, 1024], available_at [K]).

    Step k (1-based) ends at k*80 ms. Its window is wave[max(0, end - window_s) : end],
    padded to 30 s exactly as the upstream processor pads a final chunk.
    `final` is the merged last token (upstream time_merge layout), and `taps` are
    hidden states averaged over that token's four 50 Hz frames.
    """

    def __init__(self, encoder, fe, taps=(6, 12, 18, 24), window_s=30.0):
        super().__init__()
        self.encoder, self.fe, self.taps = encoder, fe, tuple(taps)
        self.window = int(round(window_s * SAMPLE_RATE / TOKEN_SAMPLES)) * TOKEN_SAMPLES
        if not TOKEN_SAMPLES <= self.window <= CHUNK_SAMPLES:
            raise ValueError('window must be within (80 ms, 30 s]')

    def load(self, path):
        from safetensors.torch import load_file
        state = {k[len(PREFIX):]: v for k, v in load_file(str(path)).items()}
        self.encoder.load_state_dict(state, strict=True)
        return self

    @torch.no_grad()
    def windows(self, chunks):
        """Encode up to-30 s windows (each a multiple of 80 ms); one token per window."""
        padded = [np.pad(c, (0, CHUNK_SAMPLES - len(c))).astype(np.float32) for c in chunks]
        mel = self.fe(padded, sampling_rate=SAMPLE_RATE, padding='max_length', return_tensors='pt')['input_features']
        p = next(self.encoder.parameters())
        out = self.encoder(mel.to(p.device, p.dtype), output_hidden_states=True)
        tokens = [len(c) // TOKEN_SAMPLES for c in chunks]  # == upstream _compute_audio_token_length
        rows = torch.arange(len(chunks))
        frames = torch.tensor([[MERGE * (n - 1) + j for j in range(MERGE)] for n in tokens])
        final = out.last_hidden_state[rows[:, None], frames].flatten(1)          # [B, 4096]
        taps = torch.stack([out.hidden_states[t][rows[:, None], frames].mean(1) for t in self.taps], 1)
        return final, taps

    def forward(self, wave, steps=None, batch=8):
        wave = np.asarray(wave, np.float32)
        steps = steps or len(wave) // TOKEN_SAMPLES
        finals, taps = [], []
        for lo in range(1, steps + 1, batch):
            ends = [k * TOKEN_SAMPLES for k in range(lo, min(lo + batch, steps + 1))]
            f, t = self.windows([wave[max(0, e - self.window):e] for e in ends])
            finals.append(f)
            taps.append(t)
        return dict(final=torch.cat(finals), taps=torch.cat(taps),
                    available_at=np.arange(1, steps + 1) * TOKEN_SAMPLES / SAMPLE_RATE)


def build(directory, weights=None, taps=(6, 12, 18, 24), window_s=30.0, device='cpu', dtype=torch.float32,
          **config_overrides):
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    config = whisper_config(directory, **config_overrides)
    model = TrailingWindowEncoder(WhisperEncoder(config), feature_extractor(config.num_mel_bins), taps, window_s)
    if weights is not None:
        model.load(weights)
    return model.to(device=device, dtype=dtype).eval()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Fetch pinned MOSS-Transcribe-Diarize encoder-only weights.')
    ap.add_argument('directory')
    print(json.dumps(fetch(ap.parse_args().directory), indent=2))
