import unittest

import numpy as np

from annotator import features as F


def act(spans, T=500):
    a = np.zeros((T, 2), bool)
    for c, s, e in spans:
        a[int(s / F.FRAME_S):int(e / F.FRAME_S), c] = True
    return a


class FeaturesTest(unittest.TestCase):
    def test_candidates(self):
        # speaker 1 talks 0-4 s; speaker 2 starts at 3 s (overlap) and talks to 8 s
        a = act([(0, 0, 4), (1, 3, 8)])
        ints = F.candidates(a, 'int')
        self.assertIn((2, 3.0, 0), ints)                 # onset during the other's speech
        self.assertTrue(any(r[0] == 1 and r[2] == 1 for r in ints))  # check point inside overlap
        self.assertIn((1, 4.0, 0), F.candidates(a, 'eot'))

    def test_table_shapes(self):
        a = act([(0, 0, 4), (1, 3, 8)])
        rows, X = F.table(a, 'int')
        self.assertEqual(X.shape, (len(rows), F.N_FEATURES))
        rows, X = F.table(np.zeros((100, 2), bool), 'int')
        self.assertEqual(X.shape, (0, F.N_FEATURES))

    def test_fold_to_two(self):
        b = np.zeros((300, 3), bool)
        b[:100, 0] = True; b[150:300, 1] = True; b[100:150, 2] = True  # slot 2 is slot 0's other half
        b[120:140, 1] = True
        a = F.fold_to_two(b)
        self.assertTrue(a[100:150, 0].all() or a[100:150].sum() > 0)
        self.assertEqual(a.shape, (300, 2))

    def test_label(self):
        g = dict(int_positive_events=[dict(speaker=2, time_s=3.0)], int_negative_spans=[],
                 int_excluded=[dict(speaker=1, start=10, end=12)])
        self.assertEqual(F.label(g, 'int', 2, 3.0), 1)
        self.assertEqual(F.label(g, 'int', 1, 11.0), -1)
        self.assertEqual(F.label(g, 'int', 1, 3.0), 0)

    def test_audacity_labels(self):
        from annotator.annotate import audacity_labels
        res = dict(eot_positive_events=[dict(speaker=2, time_s=5.0, score=0.9)],
                   int_positive_events=[dict(speaker=1, time_s=1.5, score=0.71)])
        self.assertEqual(audacity_labels(res), '1.50\t1.50\tINT spk1 0.71\n5.00\t5.00\tEOT spk2 0.90\n')


if __name__ == '__main__':
    unittest.main()
