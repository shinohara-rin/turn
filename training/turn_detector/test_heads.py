"""Synthetic contract checks only; no training or real dataset access."""
import unittest
import numpy as np
from heads import causal_history, speaker_features, eot_targets, vap_targets, commit_events


class Contracts(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(40, dtype=np.float64)*.16
        self.v = np.zeros((40, 2), np.float32)
        self.v[:4, 0] = 1
        self.v[12:15, 0] = 1

    def test_causal_and_speaker_order(self):
        h = causal_history(self.t, self.v)
        altered = self.v.copy()
        altered[20:] = 1
        np.testing.assert_array_equal(h[:20], causal_history(self.t, altered)[:20])
        f = np.zeros((40, 2, 3), np.float32)
        f[:, 1] = 3
        x = speaker_features(f, h)
        self.assertEqual(x.shape, (40, 2, 22))
        np.testing.assert_array_equal(x[:, 0, :11], x[:, 1, 11:])
        self.assertAlmostEqual(h[5, 0, 1], .016, places=5)

    def test_labels_no_early_positive_and_stop_at_resume(self):
        events = {'eot_positive_events':[{'speaker':1,'time_s':.8}],
                  'eot_negative_spans':[{'speaker':1,'start':2.4,'end':3.2}],
                  'eot_excluded':[{'speaker':1,'start':1.1,'end':1.3}]}
        y, w = eot_targets(self.t, self.v, events)
        self.assertEqual(y[self.t < .8].sum(), 0)
        self.assertEqual(y[self.t >= 1.92].sum(), 0)
        self.assertEqual(w[(self.t>=1.1)&(self.t<=1.3), 0].sum(), 0)
        self.assertEqual(w[:,1].sum(), 0)  # Never spoke: no huge listening negatives.
        self.assertEqual(y[5,0], 1)

    def test_future_targets_are_not_input_features(self):
        activity = np.zeros_like(self.v)
        activity[10:20,0] = 1
        targets, mask = vap_targets(self.t, activity)
        self.assertEqual(targets.shape, (40,2,8))
        np.testing.assert_array_equal(targets[:,0,::2],targets[:,1,1::2])
        self.assertEqual(mask[-1].sum(), 0)
        self.assertGreater(targets[8,0,0], 0)

    def test_commit_policy(self):
        p = np.ones_like(self.v)
        once = commit_events(self.t, self.v, p, .5)
        self.assertEqual(once[1], [float(self.t[4]),float(self.t[15])])
        self.assertEqual(once[2], [])
        repeats = commit_events(self.t, self.v, p, .5, 1.5)
        self.assertGreater(len(repeats[1]), len(once[1]))
        self.assertTrue(all(self.v[np.argmin(abs(self.t-t)),0] < .5 for t in repeats[1]))

    def test_grid_validation(self):
        with self.assertRaises(ValueError):
            causal_history(self.t*2, self.v)


if __name__ == '__main__':
    unittest.main()
