"""Floor targets must agree with the pinned TurnBench gold builder.

The integration tests need the pinned evaluator importable (turnbench@38a6f87, in its
own environment); the builder is the source of truth for EOT/INT gold.
"""
import unittest
from dataclasses import asdict

import numpy as np

import labels as lb

try:
    from turnbench.gold import ConsensusEvent, ConsensusViews, build_conversation_events
except ImportError:  # pragma: no cover
    build_conversation_events = None

# speaker, start, end, canonical label
SEGMENTS = [
    (1, 0.0, 2.0, 'Turn'),
    (2, 1.0, 1.3, 'Backchannel'),                   # floor stays with speaker 1
    (1, 2.6, 4.0, 'Turn'),                          # 2.0-2.6 is a hold; 4.0 is a yield
    (2, 3.0, 3.4, 'NonFloorTakingInterruption'),    # contest that speaker 1 wins
    (2, 4.5, 6.0, 'Turn'),
    (1, 5.5, 7.0, 'Interruption'),                  # contest that speaker 1 takes
    (2, 7.5, 7.8, 'NonContent'),
    (2, 8.5, 10.0, 'Turn'),
    (1, 9.6, 11.0, 'Turn'),                         # latched start: a hand-off, not a contest
]
TURN_VIEW = ('Turn', 'Interruption')


@unittest.skipUnless(build_conversation_events, 'pinned turnbench not importable')
class AgreesWithGold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        turn = [ConsensusEvent(s, a, b, 'Turn') for s, a, b, l in SEGMENTS if l in TURN_VIEW]
        fine = [ConsensusEvent(s, a, b, l) for s, a, b, l in SEGMENTS]
        cls.events = asdict(build_conversation_events(ConsensusViews(turn, [], fine, [])))
        cls.times = (np.arange(160) + 1) * 0.08
        cls.y = lb.floor_targets(cls.times, SEGMENTS, cls.events)
        cls.future, _ = lb.floor_projection(cls.y['floor'], cls.y['floor_w'])

    def floor_at(self, time_s):
        return lb.FLOOR[self.y['floor'][np.searchsorted(self.times, time_s)].argmax()]

    def test_gold_itself(self):
        eot = {(e['speaker'], e['time_s']) for e in self.events['eot_positive_events']}
        self.assertTrue({(1, 4.0), (2, 6.0), (1, 7.0), (2, 10.0)} <= eot)
        self.assertEqual([(e['speaker'], e['time_s']) for e in self.events['int_positive_events']], [(1, 5.5)])

    def test_eot_positives_release_the_floor_and_negatives_hold_it(self):
        for e in self.events['eot_positive_events']:
            if e['time_s'] + 0.1 < self.times[-1]:
                other = 'HELD_%d' % (2 - e['speaker'])
                self.assertIn(self.floor_at(e['time_s'] + 0.05), ('OPEN', other), e)
        for span in self.events['eot_negative_spans']:
            self.assertEqual(self.floor_at((span['start'] + span['end']) / 2), 'HELD_%d' % (span['speaker'] - 1))

    def test_interruptions_are_contests_with_outcomes(self):
        i = np.searchsorted(self.times, 5.6)
        self.assertEqual(self.floor_at(5.6), 'CONTESTED')
        self.assertEqual(lb.FLOOR[self.future[i, 1].argmax()], 'HELD_0')   # speaker 1 takes the floor
        j = np.searchsorted(self.times, 3.1)
        self.assertEqual(self.floor_at(3.1), 'CONTESTED')
        self.assertEqual(lb.FLOOR[self.future[j, 1].argmax()], 'HELD_0')   # failed attempt: speaker 1 keeps it
        self.assertEqual(self.floor_at(1.1), 'HELD_0')            # backchannel: no contest
        for span in self.events['int_negative_spans']:
            mid = np.searchsorted(self.times, (span['start'] + span['end']) / 2)
            self.assertIn(lb.ACTS[self.y['act'][mid, span['speaker'] - 1]], ('BACKCHANNEL', 'NONCONTENT'))

    def test_cotalk_before_a_bid_is_not_contested(self):
        segs = [(1, 0.0, 3.0, 'Turn'), (2, 1.0, 1.6, 'Turn'), (2, 1.6, 2.4, 'Interruption')]
        turn = [ConsensusEvent(s, a, b, 'Turn') for s, a, b, l in segs if l in TURN_VIEW]
        fine = [ConsensusEvent(s, a, b, l) for s, a, b, l in segs]
        ev = asdict(build_conversation_events(ConsensusViews(turn, [], fine, [])))
        t = (np.arange(40) + 1) * 0.08
        f = lb.floor_targets(t, segs, ev)['floor']
        at = lambda x: lb.FLOOR[f[np.searchsorted(t, x)].argmax()]
        self.assertEqual(at(1.3), 'HELD_0')      # co-talk before the bid keeps the holder
        self.assertEqual(at(2.0), 'CONTESTED')   # the bid itself is a contest

    def test_handoff_overlap_is_soft_not_contested(self):
        run = (self.times >= 9.6) & (self.times < 10.0)
        f = self.y['floor'][run]
        np.testing.assert_allclose(f[:, lb.F['HELD_0']] + f[:, lb.F['HELD_1']], 1.0)
        self.assertTrue((f[:, lb.F['CONTESTED']] == 0).all())
        held_new = f[:, lb.F['HELD_0']]
        self.assertTrue((np.diff(held_new) > 0).all())          # moves monotonically to speaker 1
        self.assertLess(held_new[0], 0.5)
        self.assertGreater(held_new[-1], 0.5)
        self.assertEqual(self.floor_at(10.3), 'HELD_0')
        np.testing.assert_allclose(self.y['floor'].sum(1), 1.0)

    def test_open_floor_and_weights(self):
        self.assertEqual(self.floor_at(4.2), 'OPEN')
        self.assertEqual(self.floor_at(7.6), 'OPEN')   # noise does not claim the floor
        self.assertEqual(self.floor_at(9.0), 'HELD_1')
        # The failed attempt is in int_excluded for scoring, but stays supervised here.
        self.assertTrue((self.y['act_w'] == 1).all())

    def test_scores_read_off_floor(self):
        import torch
        logits = torch.full((1, 4), -10.0)
        logits[0, lb.F['OPEN']] = 10
        future = torch.full((1, 3, 4), -10.0)
        future[0, :, lb.F['HELD_1']] = 10
        s = lb.turnbench_scores(logits, future)
        self.assertGreater(s['eot'][0, 0].item(), 0.99)
        self.assertGreater(s['int'][0, 1].item(), 0.99)
        self.assertLess(s['int'][0, 0].item(), 0.01)


class Fine(unittest.TestCase):
    def test_priority_alias_and_groups(self):
        t = (np.arange(30) + 1) * 0.08
        segs = [(1, 0.0, 1.0, 'Regular Turn'), (1, 0.5, 0.9, 'Floor-taking Cooperative Interruption'),
                (2, 0.2, 0.6, 'Reaction Backchannel'), (2, 1.2, 1.6, 'Awkward Silence'), (2, 0.0, 2.0, 'Made Up')]
        f = lb.fine_acts(t, segs)
        at = lambda s, c: lb.FINE[f[np.searchsorted(t, s), c]]
        self.assertEqual(at(0.3, 0), 'Normal Turn')
        self.assertEqual(at(0.7, 0), 'Floor-taking Cooperative Interruption')   # higher priority wins
        self.assertEqual(at(0.4, 1), 'Reaction Backchannel')
        self.assertEqual(at(1.4, 1), 'Awkward Silence')
        self.assertEqual(at(1.9, 1), 'SILENT')                                  # unknown label ignored
        self.assertEqual(sorted(sum(lb.FINE_GROUPS.values(), [])), list(range(1, len(lb.FINE) - 1)))


class Slots(unittest.TestCase):
    def test_arrival_order_relabels_floor_and_acts(self):
        floor = np.eye(4, dtype=np.float32)[[2, 1, 1, 3, 0]]
        floor[3] = [0.25, 0.75, 0, 0]  # a hand-off frame
        y = dict(floor=floor, floor_w=np.ones(5, np.float32),
                 act=np.array([[0, 0], [0, 1], [0, 1], [1, 1], [1, 0]]), act_w=np.ones((5, 2), np.float32))
        s = lb.to_slots(y)  # channel 1 speaks first -> slot 0
        self.assertEqual([lb.FLOOR[i] for i in s['floor'].argmax(1)], ['OPEN', 'HELD_0', 'HELD_0', 'HELD_0', 'HELD_1'])
        np.testing.assert_allclose(s['floor'][3], [0.75, 0.25, 0, 0])
        np.testing.assert_array_equal(s['slot_activity'][3], [1, 1])
        np.testing.assert_array_equal(s['act'][:, 0], y['act'][:, 1])

    def test_fine_labels_follow_slot_order(self):
        y = dict(floor=np.eye(4, dtype=np.float32)[[2, 1, 1]], floor_w=np.ones(3, np.float32),
                 act=np.array([[0, 0], [0, 1], [1, 1]]), act_w=np.ones((3, 2), np.float32),
                 fine=np.array([[0, 0], [0, 5], [7, 5]]), fine_w=np.array([[1, 1], [1, 1], [0, 1]], np.float32))
        s = lb.to_slots(y)  # channel 1 speaks first -> slot 0
        np.testing.assert_array_equal(s['fine'], y['fine'][:, ::-1])
        np.testing.assert_array_equal(s['fine_w'], y['fine_w'][:, ::-1])

    def test_diarization_segments_in_arrival_order(self):
        segs = [(1.0, 2.0, 'SPEAKER_04'), (0.2, 0.8, 'SPEAKER_02'), (1.5, 1.9, 'SPEAKER_02')]
        a = lb.slot_activity_from_segments(segs, 30)
        self.assertEqual(a[5].tolist(), [1, 0])
        self.assertEqual(a[20].tolist(), [1, 1])
        with self.assertRaises(ValueError):
            lb.slot_activity_from_segments(segs + [(3, 4, 'SPEAKER_09')], 60)


class Exclusions(unittest.TestCase):
    def test_disputes_zero_their_target(self):
        t = (np.arange(50) + 1) * 0.08
        events = dict(eot_excluded=[dict(speaker=1, start=2.0, end=3.0)],
                      int_excluded=[dict(speaker=2, start=0.0, end=1.0)])
        y = lb.floor_targets(t, [(1, 0.0, 1.0, 'Turn')], events)
        self.assertTrue((y['floor_w'][(t >= 2.0) & (t < 3.0)] == 0).all())
        self.assertTrue((y['act_w'][t < 1.0, 1] == 0).all())
        self.assertTrue((y['act_w'][:, 0] == 1).all())


if __name__ == '__main__':
    unittest.main()
