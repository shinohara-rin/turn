import unittest

import numpy as np
import torch

import labels as lb
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
        self.H = len(lb.HORIZONS_S)

    def test_shapes(self):
        out = self.model(self.taps, self.final)
        self.assertEqual(out['vap'].shape, (2, 50, m.VAP_CLASSES))
        self.assertEqual(out['floor'].shape, (2, 50, len(lb.FLOOR)))
        self.assertEqual(out['future'].shape, (2, 50, self.H, len(lb.FLOOR)))
        self.assertEqual(out['act'].shape, (2, 50, 2, len(lb.ACTS)))
        mono = self.model(self.taps[:, :, :1], self.final[:, :, :1])
        self.assertEqual(set(mono), {'floor', 'future', 'act', 'slot_activity'})
        self.assertEqual(mono['floor'].shape, (2, 50, len(lb.FLOOR)))
        self.assertEqual(mono['slot_activity'].shape, (2, 50, 2))
        scores = lb.turnbench_scores(out['floor'], out['future'])
        self.assertEqual(scores['eot'].shape, (2, 50, 2))
        self.assertEqual(scores['int'].shape, (2, 50, 2))

    def test_causal_in_time_stereo_and_mono(self):
        cut = 30
        taps, final = self.taps.clone(), self.final.clone()
        taps[:, cut:] = torch.randn_like(taps[:, cut:])
        final[:, cut:, 1] = torch.randn_like(final[:, cut:, 1])
        for channels in (slice(0, 2), slice(0, 1)):
            a = self.model(self.taps[:, :, channels], self.final[:, :, channels])
            b = self.model(taps[:, :, channels], final[:, :, channels])
            for k in a:
                torch.testing.assert_close(a[k][:, :cut], b[k][:, :cut], rtol=0, atol=1e-6)
                self.assertGreater((a[k][:, cut:] - b[k][:, cut:]).abs().max().item(), 1e-4)

    def test_speaker_swap_swaps_floor_holder(self):
        with torch.no_grad():
            self.model.channel.zero_()
        a = self.model(self.taps, self.final)
        b = self.model(self.taps.flip(2), self.final.flip(2))
        swap = [lb.F['HELD_1'], lb.F['HELD_0'], lb.F['OPEN'], lb.F['CONTESTED']]
        torch.testing.assert_close(a['floor'], b['floor'][..., swap], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a['future'], b['future'][..., swap], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a['act'], b['act'].flip(2), rtol=1e-5, atol=1e-5)

    def test_source_conditioning(self):
        default = self.model(self.taps, self.final)
        real = self.model(self.taps, self.final, torch.tensor([m.REAL_STEREO] * 2))
        torch.testing.assert_close(default['floor'], real['floor'])
        with torch.no_grad():
            self.model.source.weight.normal_()
        podcast = self.model(self.taps, self.final, torch.tensor([m.GATED_PODCAST] * 2))
        real = self.model(self.taps, self.final, torch.tensor([m.REAL_STEREO] * 2))
        self.assertGreater((podcast['floor'] - real['floor']).abs().max().item(), 1e-4)

    def test_partial_label_losses(self):
        out = self.model(self.taps, self.final)
        labels, valid = m.vap_labels(np.random.default_rng(0).random((50, 2)) > .5)
        vap_only = dict(vap=torch.from_numpy(labels).repeat(2, 1), vap_valid=torch.from_numpy(valid).repeat(2, 1))
        _, parts = m.loss(out, vap_only)
        self.assertEqual(set(parts), {'vap'})
        floor = dict(floor=torch.randint(0, 4, (2, 50)), floor_w=torch.ones(2, 50),
                     future=torch.randint(0, 4, (2, 50, self.H)), future_w=torch.ones(2, 50, self.H),
                     act=torch.randint(0, len(lb.ACTS), (2, 50, 2)), act_w=torch.ones(2, 50, 2))
        total, parts = m.loss(out, dict(vap_only, **floor))
        total.backward()
        self.assertEqual(set(parts), {'vap', 'floor', 'future', 'act'})
        self.assertTrue(np.isfinite(float(total)))
        mono = self.model(self.taps[:, :, :1], self.final[:, :, :1])
        diar = dict(slot_activity=torch.randint(0, 2, (2, 50, 2)).float(), slot_activity_w=torch.ones(2, 50, 2))
        _, parts = m.loss(mono, diar)
        self.assertEqual(set(parts), {'slot_activity'})
        _, parts = m.loss(mono, dict(diar, **floor))
        self.assertEqual(set(parts), {'slot_activity', 'floor', 'future', 'act'})


if __name__ == '__main__':
    unittest.main()
