import unittest

import numpy as np
import torch

import model as m


class VapLabels(unittest.TestCase):
    def test_bits_and_validity(self):
        T = 60
        a = np.zeros((T, 2), np.float32)
        a[11:14, 0] = 1   # speaker 0 talks in frames 11..13
        a[14:40, 1] = 1   # speaker 1 takes over at frame 14
        labels, valid = m.vap_labels(a)
        self.assertEqual(valid.sum(), T - m.VAP_HORIZON)
        # From frame 10, speaker 0 fills the first bin (frames 11-13) only.
        self.assertEqual(labels[10] & 0b1111, 0b0001)
        # Speaker 1 fills bins 2..4 (frames 14-35) from frame 10.
        self.assertEqual(labels[10] >> 4, 0b1110)
        # From frame 0 only the far bin (frames 16-25) is filled, by speaker 1.
        self.assertEqual(labels[0], 0b1000 << 4)
        self.assertTrue(np.all(labels < m.VAP_CLASSES))


class TurnModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = m.TurnModel(tap_layers=2, tap_dim=32, final_dim=16, dim=32, heads=4, layers=2,
                                 window_s=1.6, dropout=0.0).eval()
        self.taps = torch.randn(2, 50, 2, 2, 32)
        self.final = torch.randn(2, 50, 2, 16)

    def test_shapes(self):
        out = self.model(self.taps, self.final)
        self.assertEqual(out['vap'].shape, (2, 50, m.VAP_CLASSES))
        for k in ('eot', 'int', 'vad'):
            self.assertEqual(out[k].shape, (2, 50, 2))

    def test_causal_in_time_for_both_channels(self):
        cut = 30
        taps, final = self.taps.clone(), self.final.clone()
        taps[:, cut:] = torch.randn_like(taps[:, cut:])
        final[:, cut:, 1] = torch.randn_like(final[:, cut:, 1])
        a, b = self.model(self.taps, self.final), self.model(taps, final)
        for k in a:
            torch.testing.assert_close(a[k][:, :cut], b[k][:, :cut], rtol=0, atol=1e-6)
            self.assertGreater((a[k][:, cut:] - b[k][:, cut:]).abs().max().item(), 1e-4)

    def test_speaker_swap_equivariance(self):
        a = self.model(self.taps, self.final)
        # Swapping channels must swap per-speaker outputs (channel embedding aside).
        with torch.no_grad():
            self.model.channel.zero_()
        a = self.model(self.taps, self.final)
        b = self.model(self.taps.flip(2), self.final.flip(2))
        for k in ('eot', 'int', 'vad'):
            torch.testing.assert_close(a[k], b[k].flip(-1), rtol=1e-5, atol=1e-5)

    def test_source_conditioning(self):
        default = self.model(self.taps, self.final)
        real = self.model(self.taps, self.final, torch.tensor([m.REAL_STEREO] * 2))
        torch.testing.assert_close(default['vap'], real['vap'])
        with torch.no_grad():
            self.model.source.weight.normal_()
        podcast = self.model(self.taps, self.final, torch.tensor([m.GATED_PODCAST] * 2))
        real = self.model(self.taps, self.final, torch.tensor([m.REAL_STEREO] * 2))
        self.assertGreater((podcast['eot'] - real['eot']).abs().max().item(), 1e-4)

    def test_loss_and_vap_marginal(self):
        out = self.model(self.taps, self.final)
        labels, valid = m.vap_labels(np.random.default_rng(0).random((50, 2)) > .5)
        batch = dict(vap=torch.from_numpy(labels).repeat(2, 1), vap_valid=torch.from_numpy(valid).repeat(2, 1))
        for k in ('eot', 'int', 'vad'):
            batch[k] = torch.randint(0, 2, (2, 50, 2)).float()
            batch[k + '_w'] = torch.rand(2, 50, 2)
        total, parts = m.loss(out, batch)
        total.backward()
        self.assertTrue(np.isfinite(float(total)))
        self.assertEqual(set(parts), {'vap', 'eot', 'int', 'vad'})
        p = m.vap_speaker_probabilities(out['vap'])
        self.assertEqual(p.shape, (2, 50, 2))
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == '__main__':
    unittest.main()
