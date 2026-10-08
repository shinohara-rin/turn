"""Frame states must agree with the pinned TurnBench gold builder.

The integration test needs the pinned evaluator importable (`pip install -e
turnbench@38a6f87`); the builder is the source of truth for EOT/INT gold.
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
    (2, 1.0, 1.3, 'Backchannel'),
    (1, 2.6, 4.0, 'Turn'),          # pause 2.0-2.6 is a hold; 4.0 hands the floor over
    (2, 4.5, 6.0, 'Turn'),
    (1, 5.5, 7.0, 'Interruption'),  # floor-taking barge-in; speaker 2 yields at 6.0
    (2, 7.5, 7.8, 'NonContent'),
    (2, 8.5, 10.0, 'Turn'),
]


@unittest.skipUnless(build_conversation_events, 'pinned turnbench not importable')
class AgreesWithGold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        turn = [ConsensusEvent(s, a, b, 'Turn') for s, a, b, l in SEGMENTS if l in ('Turn', 'Interruption')]
        fine = [ConsensusEvent(s, a, b, l) for s, a, b, l in SEGMENTS]
        cls.events = asdict(build_conversation_events(ConsensusViews(turn, [], fine, [])))
        cls.times = (np.arange(150) + 1) * 0.08
        cls.state, cls.weight = lb.frame_states(cls.times, SEGMENTS, cls.events)

    def at(self, time_s, speaker):
        return lb.STATES[self.state[np.searchsorted(self.times, time_s), speaker - 1]]

    def test_gold_itself(self):
        eot = sorted((e['speaker'], e['time_s']) for e in self.events['eot_positive_events'])
        self.assertIn((1, 4.0), eot)
        self.assertIn((2, 6.0), eot)
        self.assertEqual([(e['speaker'], e['time_s']) for e in self.events['int_positive_events']], [(1, 5.5)])

    def test_states_follow_gold(self):
        for e in self.events['eot_positive_events']:
            if e['time_s'] + 0.1 < self.times[-1]:
                self.assertEqual(self.at(e['time_s'] + 0.05, e['speaker']), 'YIELD', e)
        for span in self.events['eot_negative_spans']:
            self.assertEqual(self.at((span['start'] + span['end']) / 2, span['speaker']), 'HOLD', span)
        for e in self.events['int_positive_events']:
            self.assertEqual(self.at(e['time_s'] + 0.05, e['speaker']), 'INT_FLOOR', e)
        for span in self.events['int_negative_spans']:
            self.assertIn(self.at((span['start'] + span['end']) / 2, span['speaker']), ('BACKCHANNEL', 'NONCONTENT'))

    def test_listen_and_yield_end_on_resume(self):
        self.assertEqual(self.at(3.0, 2), 'LISTEN')
        self.assertEqual(self.at(8.3, 2), 'YIELD')   # speaker 2 yielded at 6.0, within 3 s
        self.assertEqual(self.at(9.0, 2), 'TURN')
        np.testing.assert_array_equal(lb.activity(self.state)[:, 0] > 0,
                                      np.isin(self.state[:, 0], lb.SPEAKING))


class Mono(unittest.TestCase):
    def test_priority_merge(self):
        S = lb.S
        state = np.array([[S['TURN'], S['BACKCHANNEL']],
                          [S['TURN'], S['INT_FLOOR']],
                          [S['YIELD'], S['LISTEN']],
                          [S['HOLD'], S['BACKCHANNEL']],
                          [S['HOLD'], S['LISTEN']]])
        weight = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0]], np.float32)
        mono, w = lb.mono_states(state, weight)
        self.assertEqual([lb.STATES[i] for i in mono], ['TURN', 'INT_FLOOR', 'YIELD', 'BACKCHANNEL', 'HOLD'])
        np.testing.assert_array_equal(w, [1, 1, 1, 1, 0])


class Exclusions(unittest.TestCase):
    def test_excluded_spans_zero_weight_for_their_task(self):
        t = (np.arange(50) + 1) * 0.08
        events = dict(eot_excluded=[dict(speaker=1, start=2.0, end=3.0)],
                      int_excluded=[dict(speaker=1, start=0.0, end=1.0)])
        state, weight = lb.frame_states(t, [(1, 0.0, 1.0, 'NonFloorTakingInterruption')], events)
        self.assertTrue((weight[(t >= 2.0) & (t < 3.0), 0] == 0).all())
        self.assertTrue((weight[t < 1.0, 0] == 0).all())
        self.assertTrue((weight[:, 1] == 1).all())


if __name__ == '__main__':
    unittest.main()
