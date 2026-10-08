import unittest

import numpy as np

import podcast_subset as ps
import pseudo_stereo as pst


def rows(feeds=40, episodes=5, clips=6):
    out = []
    for f in range(feeds):
        for e in range(episodes):
            for c in range(clips):
                out.append(dict(rss_url=f'https://feed{f}/rss', audio_url=f'https://feed{f}/ep{e}.mp3',
                                dialogue_idx=c, duration_sec=60.0 + 7 * c, speakers=['A', 'B']))
    return out


class Subset(unittest.TestCase):
    def test_feed_disjoint_partitions(self):
        r = rows()
        train = ps.select(r, 1e9, 'train', val_fraction=0.2)
        val = ps.select(r, 1e9, 'val', val_fraction=0.2)
        self.assertTrue(train and val)
        self.assertFalse({x['rss_url'] for x in train} & {x['rss_url'] for x in val})

    def test_nested_as_hours_grow(self):
        r = rows()
        small, large = ps.select(r, 2, feed_cap_h=1e3), ps.select(r, 6, feed_cap_h=1e3)
        key = lambda x: (x['audio_url'], x['dialogue_idx'])
        self.assertTrue({key(x) for x in small} <= {key(x) for x in large})

    def test_caps(self):
        out = ps.select(rows(), 1e9, val_fraction=0.0, feed_cap_h=0.2, episode_cap_h=0.05)
        per_feed, per_episode = {}, {}
        for x in out:
            per_feed[x['rss_url']] = per_feed.get(x['rss_url'], 0) + x['duration_sec']
            per_episode[x['audio_url']] = per_episode.get(x['audio_url'], 0) + x['duration_sec']
        self.assertLessEqual(max(per_feed.values()), 0.2 * 3600)
        self.assertLessEqual(max(per_episode.values()), 0.05 * 3600)


class PseudoStereo(unittest.TestCase):
    def test_gating_keeps_real_audio_with_bleed(self):
        sr = 1000
        mono = np.ones(3 * sr, np.float32)
        y = pst.gated_stereo(mono, sr, [(0.0, 1.0, 0), (2.0, 3.0, 1)], bleed_db=-20, ramp_ms=1, jitter_db=0)
        self.assertAlmostEqual(float(y[500, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(y[500, 1]), 0.1, places=5)
        self.assertAlmostEqual(float(y[1500, 0]), 0.1, places=5)  # silence: both at bleed
        self.assertAlmostEqual(float(y[2500, 1]), 1.0, places=5)

    def test_agreement_resolves_permutation_and_overlap(self):
        a = pst.frame_activity([(0, 1.0, 0), (0.8, 2.0, 1)], 25)
        m = pst.activity_agreement(a, a[:, ::-1])
        self.assertTrue(m['swapped'])
        self.assertEqual(m['accuracy'], 1.0)
        self.assertGreater(m['overlap_frames'], 0)
        self.assertEqual(m['overlap_recall'], 1.0)
        # An estimate that misses the overlap (one speaker at a time).
        est = a.copy()
        est[a.all(1), 1] = 0
        self.assertEqual(pst.activity_agreement(a, est)['overlap_recall'], 0.0)


if __name__ == '__main__':
    unittest.main()
