"""Pure synthetic timing, causality and split access checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from heads import causal_history, commit_events, eot_targets
from baseline import baseline_probabilities, fixed_grid, read_records
from leakage_guard import TRAIN_SOURCE, TRAIN_REVISION


class BaselineContracts(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(30)*.16
        self.vad = np.zeros((30,2), np.float32)
        self.vad[:5,0] = 1
        self.vad[8:20,1] = 1
        self.record = dict(times=self.times, vad=self.vad,
                           history=causal_history(self.times,self.vad))

    def test_small_unique_prespecified_grid(self):
        grid = fixed_grid()
        self.assertEqual(len(grid), 72)
        self.assertEqual(len({tuple(p.values()) for p in grid}), 72)

    def test_own_silence_waits_from_first_quiet_frame(self):
        p = baseline_probabilities(self.record, .32)
        self.assertEqual(p[6,0], 0)
        self.assertEqual(p[7,0], 1)
        fires = commit_events(self.times,self.vad,p,.5)
        self.assertEqual(fires[1], [self.times[7]])
        # Other channel cannot emit before its first speech.
        self.assertTrue(all(t > self.times[19] for t in fires[2]))

    def test_other_speech_must_be_sustained_and_current(self):
        p = baseline_probabilities(self.record,.32,.32)
        self.assertEqual(p[9,0], 0)
        self.assertEqual(p[10,0], 1)
        self.assertEqual(p[20,0], 0)
        altered = self.vad.copy()
        altered[15:] = 1
        future = dict(vad=altered,history=causal_history(self.times,altered))
        np.testing.assert_array_equal(p[:15],baseline_probabilities(future,.32,.32)[:15])

    def test_dev_loader_does_not_open_gate_or_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'cache'
            root.mkdir()
            np.savez(root/'dev.npz',times=self.times,vad=self.vad)
            (root/'dev.json').write_text('{}')
            manifest = [dict(id='d',split='dev',npz='dev.npz',events='dev.json')]
            for split in ('train','gate'):
                manifest.append(dict(id=split,split=split,npz='missing.npz',events='missing.json'))
            plan = dict(revision=TRAIN_REVISION, splits={}, assignments={})
            metadata = []
            for record in manifest:
                record.update(source=TRAIN_SOURCE, revision=TRAIN_REVISION)
                split, cid = record['split'], record['id']
                plan['splits'][split] = [cid]
                actors = [f'{split}-a', f'{split}-b']
                plan['assignments'].update({actor: split for actor in actors})
                metadata.append(dict(_dir=cid, speaker_1_actor_id=actors[0], speaker_2_actor_id=actors[1]))
            (root.parent/'split.json').write_text(json.dumps(plan))
            (root.parent/'metadata.json').write_text(json.dumps(metadata))
            (root/'manifest.json').write_text(json.dumps(manifest))
            records = read_records(root/'manifest.json','dev')
            self.assertEqual([r['id'] for r in records],['d'])

    def test_background_episode_weight_does_not_grow_with_length(self):
        empty = dict(eot_positive_events=[],eot_negative_spans=[])
        _,w = eot_targets(self.times,self.vad,empty)
        np.testing.assert_allclose(w[:,0].sum(),.02,atol=1e-7)
        long_times = np.arange(300)*.16
        long_vad = np.zeros((300,2),np.float32)
        long_vad[:5,0] = 1
        _,long_w = eot_targets(long_times,long_vad,empty)
        np.testing.assert_allclose(long_w[:,0].sum(),.02,atol=1e-7)
        self.assertEqual(long_w[:,1].sum(),0)


if __name__ == '__main__':
    unittest.main()
