"""Cat encoder contracts: upstream parity, exact causality, streaming == full.

Structural tests use a shrunken config with random weights (fast, no download).
Set CAT_DIR=<dir from `python cat_encoder.py <dir>`> to also run them on the
real pinned encoder weights.
"""
import copy
import os
import unittest

import torch

import cat_encoder as ce

CODE_DIR = os.environ.get('CAT_CODE_DIR') or os.environ.get('CAT_DIR')


def tiny_config(config):
    config = copy.deepcopy(config)
    for kwargs in config.encoder_kwargs + config.decoder_kwargs:
        if kwargs['module_type'] == 'Transformer':
            kwargs.update(num_layers=2, d_model=64, dim_feedforward=128, num_heads=4)
    config.encoder_kwargs[-1]['num_layers'] = 4
    config.quantizer_kwargs = dict(config.quantizer_kwargs, num_quantizers=2)
    # Short context so the window edge is exercised within a few seconds of audio.
    config.causal_transformer_context_duration = 1.6
    return config


@unittest.skipUnless(CODE_DIR, 'set CAT_CODE_DIR or CAT_DIR to the fetched pinned code')
class TinyStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.modeling, config = ce.load_upstream(CODE_DIR)
        cls.config = tiny_config(config)
        cls.encoder = ce.CatEncoder(cls.modeling, cls.config, taps=(1, 3)).eval()
        # Non-trivial layer scales so attention paths actually matter.
        with torch.no_grad():
            for name, p in cls.encoder.named_parameters():
                if 'layer_scale' in name:
                    p.fill_(1.0)

    def wave(self, seconds, seed=1):
        g = torch.Generator().manual_seed(seed)
        return 0.1 * torch.randn(2, int(seconds * ce.SAMPLE_RATE), generator=g)

    def test_matches_upstream_full_model(self):
        model = self.modeling.MossAudioTokenizerModel(self.config).eval()
        model.encoder.load_state_dict(self.encoder.encoder.state_dict())
        wave = self.wave(2.0)
        with torch.no_grad():
            ours = self.encoder(wave)['final']
            theirs = model._encode_frame(wave[:, None], None).encoder_hidden_states.transpose(1, 2)
        torch.testing.assert_close(ours, theirs, rtol=0, atol=0)

    def test_future_mutation_does_not_change_past_frames(self):
        wave = self.wave(3.2)
        cut_frame = 17
        mutated = wave.clone()
        mutated[:, cut_frame * ce.HOP:] = torch.randn_like(mutated[:, cut_frame * ce.HOP:])
        with torch.no_grad():
            a, b = self.encoder(wave), self.encoder(mutated)
        torch.testing.assert_close(a['final'][:, :cut_frame], b['final'][:, :cut_frame], rtol=0, atol=1e-6)
        torch.testing.assert_close(a['taps'][:, :cut_frame], b['taps'][:, :cut_frame], rtol=0, atol=1e-6)
        self.assertGreater((a['final'][:, cut_frame] - b['final'][:, cut_frame]).abs().max().item(), 1e-4)

    def test_streaming_matches_full_sequence(self):
        wave = self.wave(4.0)  # longer than the shrunken 1.6 s context
        with torch.no_grad():
            full = self.encoder(wave)
            for chunk in (1, 3, 20):
                streamed = self.encoder.stream(wave, chunk_frames=chunk)
                torch.testing.assert_close(streamed['final'], full['final'], rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(streamed['taps'], full['taps'], rtol=1e-4, atol=1e-5)

    def test_partial_frame_dropped(self):
        wave = self.wave(1.0)[:, : 5 * ce.HOP + 100]
        with torch.no_grad():
            self.assertEqual(self.encoder(wave)['final'].shape[1], 5)


WEIGHTS = os.path.join(os.environ.get('CAT_DIR', ''), 'cat_encoder.safetensors')


@unittest.skipUnless(os.environ.get('CAT_DIR') and os.path.exists(WEIGHTS), 'set CAT_DIR with fetched weights')
class RealWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.encoder = ce.build(os.environ['CAT_DIR'], WEIGHTS)

    def test_causal_and_streaming_on_real_weights(self):
        g = torch.Generator().manual_seed(2)
        # 12.8 s exceeds the 10 s attention window, exercising the cache edge.
        t = torch.arange(int(12.8 * ce.SAMPLE_RATE)) / ce.SAMPLE_RATE
        # Voiced-ish harmonic signal with amplitude bursts plus a little noise.
        wave = (0.2 * torch.sin(2 * torch.pi * 140 * t) * (torch.sin(2 * torch.pi * 0.7 * t) > 0)
                + 0.01 * torch.randn(len(t), generator=g)).repeat(2, 1)
        wave[1] = torch.roll(wave[1], 7000)
        cut = 140
        mutated = wave.clone()
        mutated[:, cut * ce.HOP:] = 0.3 * torch.randn(2, wave.shape[1] - cut * ce.HOP, generator=g)
        with torch.no_grad():
            full = self.encoder(wave)
            other = self.encoder(mutated)
            streamed = self.encoder.stream(wave, chunk_frames=2)
        self.assertEqual(full['final'].shape, (2, 160, ce.OUT_DIM))
        self.assertEqual(full['taps'].shape, (2, 160, 4, ce.TOP_DIM))
        self.assertTrue(torch.isfinite(full['final']).all())
        torch.testing.assert_close(other['final'][:, :cut], full['final'][:, :cut], rtol=0, atol=1e-5)
        self.assertGreater((other['final'][:, cut:] - full['final'][:, cut:]).abs().max().item(), 1e-2)
        torch.testing.assert_close(streamed['final'], full['final'], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(streamed['taps'], full['taps'], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
