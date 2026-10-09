"""MOSS-Transcribe-Diarize trailing-window encoder: exact causality and upstream parity.

Structural tests use a shrunken random Whisper encoder. Set MTD_DIR=<dir from
`python mtd_encoder.py <dir>`> for real weights. Set MTD_UPSTREAM=<dir holding the pinned
processing_moss_transcribe_diarize.py> to check features against the upstream processor.
"""
import importlib.util
import os
import unittest

import numpy as np
import torch

import mtd_encoder as me

MTD_DIR = os.environ.get('MTD_DIR')
UPSTREAM = os.environ.get('MTD_UPSTREAM')


def speechlike(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * me.SAMPLE_RATE)) / me.SAMPLE_RATE
    gate = (np.sin(2 * np.pi * 0.6 * t) > 0).astype(np.float32)
    return (0.2 * np.sin(2 * np.pi * 150 * t) * gate + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


class Tiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        torch.manual_seed(0)
        config = WhisperConfig(num_mel_bins=80, d_model=64, encoder_layers=2, encoder_attention_heads=4,
                               encoder_ffn_dim=128, max_source_positions=1500)
        cls.model = me.TrailingWindowEncoder(WhisperEncoder(config), me.feature_extractor(), taps=(1, 2),
                                             window_s=4.0).eval()

    def test_future_audio_never_changes_past_steps(self):
        wave = speechlike(3.0)
        mutated = wave.copy()
        cut = 20 * me.TOKEN_SAMPLES
        mutated[cut:] = np.random.default_rng(1).standard_normal(len(wave) - cut).astype(np.float32)
        a, b = self.model(wave), self.model(mutated)
        torch.testing.assert_close(a['final'][:20], b['final'][:20], rtol=0, atol=0)
        torch.testing.assert_close(a['taps'][:20], b['taps'][:20], rtol=0, atol=0)
        self.assertGreater((a['final'][20:] - b['final'][20:]).abs().max().item(), 1e-4)
        self.assertEqual(a['available_at'][19], 20 * 0.08)

    def test_shapes_and_window_limit(self):
        out = self.model(speechlike(6.0), steps=70)
        self.assertEqual(out['final'].shape, (70, me.MERGE * 64))
        self.assertEqual(out['taps'].shape, (70, 2, 64))
        with self.assertRaises(ValueError):
            me.TrailingWindowEncoder(self.model.encoder, self.model.fe, window_s=31)


@unittest.skipUnless(UPSTREAM, 'set MTD_UPSTREAM to the pinned upstream code directory')
class UpstreamParity(unittest.TestCase):
    def test_features_and_token_count_match_processor(self):
        spec = importlib.util.spec_from_file_location(
            'mtd_processing', os.path.join(UPSTREAM, 'processing_moss_transcribe_diarize.py'))
        up = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(up)
        fe = me.feature_extractor()
        clip = speechlike(2.4)
        feats, lengths, _ = up._audios_to_input_features(fe, [clip], audio_merge_size=me.MERGE)
        ours = fe([np.pad(clip, (0, me.CHUNK_SAMPLES - len(clip)))], sampling_rate=me.SAMPLE_RATE,
                  padding='max_length', return_tensors='pt')['input_features']
        torch.testing.assert_close(feats, ours, rtol=0, atol=0)
        self.assertEqual(int(lengths[0]), len(clip) // me.TOKEN_SAMPLES)


@unittest.skipUnless(MTD_DIR, 'set MTD_DIR with fetched weights')
class RealWeights(unittest.TestCase):
    def test_strict_load_and_exact_causality(self):
        model = me.build(MTD_DIR, os.path.join(MTD_DIR, 'mtd_encoder.safetensors'), window_s=8.0)
        self.assertEqual(sum(p.numel() for p in model.parameters()) // 10**6, 307)
        wave = speechlike(1.6, seed=3)
        mutated = wave.copy()
        mutated[12 * me.TOKEN_SAMPLES:] = 0.3
        a, b = model(wave), model(mutated)
        self.assertEqual(a['final'].shape, (20, me.MERGE * me.DIM))
        self.assertTrue(torch.isfinite(a['final']).all())
        torch.testing.assert_close(a['final'][:12], b['final'][:12], rtol=0, atol=0)
        self.assertGreater((a['final'][12:] - b['final'][12:]).abs().max().item(), 1e-2)


if __name__ == '__main__':
    unittest.main()
