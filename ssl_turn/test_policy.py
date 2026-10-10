"""Commit policy (policy.py): rule semantics on hand-made voice activity, and causality."""
import unittest

import numpy as np

import policy as pol
from policy import Policy

RULES = Policy(eot_th=None, int_th=None)


def vad_track(segments, duration_s, hop=pol.HOP_S):
    """[T] speech probability with 1.0 inside the given (start_s, end_s) segments."""
    t = (np.arange(int(duration_s / hop)) + 0.5) * hop
    p = np.zeros(len(t), np.float32)
    for a, b in segments:
        p[(t >= a) & (t < b)] = 1.0
    return p


class TestSegments(unittest.TestCase):
    def test_short_gaps_merge(self):
        p = vad_track([(1.0, 2.0), (2.1, 3.0), (4.0, 5.0)], 6.0)
        segs = pol.speech_segments(p, 0.5, 0.2)
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0][0], 0.992, places=3)
        self.assertAlmostEqual(segs[0][1], 3.008, places=3)


class TestEOT(unittest.TestCase):
    def test_silence_deadline_and_confirmation(self):
        own, other = [(0.0, 2.0)], []
        ev = pol.eot_events(own, other, 10.0, RULES.but(deadline=2.0, confirm=2.5))
        self.assertEqual(ev, [4.0, 4.5])
        self.assertEqual(pol.eot_events(own, other, 10.0, RULES.but(deadline=2.5, confirm=2.5)), [4.5])
        self.assertEqual(pol.eot_events(own, other, 10.0, RULES.but(confirm=None)), [4.5])

    def test_other_speaker_trigger_waits_for_min_wait(self):
        own, other = [(0.0, 2.0)], [(2.1, 5.0)]
        ev = pol.eot_events(own, other, 10.0, RULES.but(min_wait=0.8, other_dur=0.2, confirm=None))
        self.assertEqual(ev, [2.8])                       # other started 2.1 + 0.2 < 2.0 + 0.8
        ev = pol.eot_events(own, [(2.7, 5.0)], 10.0, RULES.but(min_wait=0.6, other_dur=0.3, confirm=None))
        self.assertEqual(ev, [3.0])                       # 2.7 + 0.3

    def test_long_resumption_cancels_short_one_does_not(self):
        P = RULES.but(resume_min=0.5, deadline=2.5, confirm=None)
        self.assertEqual(pol.eot_events([(0.0, 2.0), (3.0, 6.0)], [], 10.0, P)[0], 6.0 + 2.5)  # cancelled at 3.5
        ev = pol.eot_events([(0.0, 2.0), (3.0, 3.3)], [], 10.0, P)  # a 0.3 s "yeah" does not cancel
        self.assertIn(4.5, ev)

    def test_confirmation_skipped_when_cancelled(self):
        P = RULES.but(min_wait=0.6, other_dur=0.2, resume_min=0.5, confirm=2.5)
        ev = pol.eot_events([(0.0, 2.0), (3.5, 6.0)], [(2.1, 3.0)], 10.0, P)
        self.assertIn(2.6, ev)
        self.assertNotIn(4.5, ev)                         # resumed at 3.5 for 0.5 s -> cancelled at 4.0

    def test_model_trigger(self):
        score = np.zeros(125, np.float32)                 # 10 s at 12.5 Hz
        score[int(2.4 * 12.5):] = 0.9                     # first frame >= 0.8 ends at 2.48
        P = Policy(eot_th=0.8, mw_model=0.32, read_from=0.16, confirm=None)
        self.assertEqual(pol.eot_events([(0.0, 2.0)], [], 10.0, P, score), [2.48])
        score[:] = 0.9                                    # high already: wait for mw_model
        self.assertEqual(pol.eot_events([(0.0, 2.0)], [], 10.0, P, score), [2.32])

    def test_end_of_stream(self):
        self.assertEqual(pol.eot_events([(0.0, 2.0)], [], 3.0, RULES), [3.0])


class TestINT(unittest.TestCase):
    def test_duration_fallback(self):
        P = RULES.but(int_dur=0.9)
        self.assertEqual(pol.int_events([(1.0, 1.5), (3.0, 5.0)], P), [3.9])

    def test_model_trigger_and_window(self):
        score = np.zeros(125, np.float32)
        score[int(3.3 * 12.5)] = 0.8                      # frame 41 ends at 3.36
        P = Policy(int_th=0.5, int_from=0.16, int_dur=0.9, int_fallback=False)
        self.assertEqual(pol.int_events([(3.0, 5.0)], P, score), [3.36])
        self.assertEqual(pol.int_events([(2.0, 5.0)], P, score), [])  # outside onset + 0.9 s


class TestCausality(unittest.TestCase):
    def test_truncation_keeps_earlier_events(self):
        rng = np.random.default_rng(0)
        T = int(300 / pol.HOP_S)
        vad = (rng.random((T, 2)) < 0.5).astype(np.float32)
        vad = np.repeat(vad[::25], 25, 0)[:T]             # 0.8 s blocks of speech / silence
        eot = rng.random((int(300 * 12.5), 2)).astype(np.float32)
        intr = rng.random((int(300 * 12.5), 2)).astype(np.float32)
        full = pol.conversation_events(vad, 300.0, Policy(), eot, intr)
        cut = 150.0
        part = pol.conversation_events(vad[:int(cut / pol.HOP_S)], cut, Policy(), eot[:int(cut * 12.5)],
                                       intr[:int(cut * 12.5)])
        for task in ('eot', 'int'):
            for s in (1, 2):
                before = [t for t in full[task][s] if t < cut - 1e-6]
                # the truncated run may add end-of-stream events at `cut`; nothing earlier may change
                self.assertEqual(before, [t for t in part[task][s] if t < cut - 1e-6])


if __name__ == '__main__':
    unittest.main()
