"""Synthetic scorer contracts, without real benchmark, gold or model access."""
import copy
import ast
import os
from pathlib import Path
import types
from unittest.mock import patch
import unittest
import numpy as np
import score_benchmark_dev as dev


def identity():
    return {'ids': ['1', '2'], 'dataset': {'source': dev.DEV_SOURCE, 'revision': dev.DEV_REVISION, 'shards': {'x': 'hash'}},
            'freeze': {'format': 'turnbench-inference-freeze-v2', 'checkpoint': 'fixed-weight-hash',
            'protocol': {'source': dev.DEV_SOURCE, 'revision': dev.DEV_REVISION, 'split': 'dev', 'dev_cache': True,
                         'gate': 'mean', 'history_vad': 'mean', 'grid_samples': 2560, 'sample_rate': 16000,
                         'operating_point': {'threshold': .5, 'recommit_s': None}}}}


class DevContracts(unittest.TestCase):
    def test_only_public_dev_source_before_loading(self):
        dev.validate_dev_identity(identity())
        for location in ('dataset', 'protocol'):
            for source in ('mundo-ai/turn-benchmark-test', 'mundo-ai/turn-benchmark-test-golden', '/content/test-renamed-dev'):
                value = identity(); part = value['dataset'] if location == 'dataset' else value['freeze']['protocol']
                part['source'] = source
                with self.subTest(source=source, location=location), self.assertRaises(ValueError): dev.validate_dev_identity(value)
        for key, value in [('split', 'test'), ('dev_cache', False), ('revision', 'unversioned')]:
            invalid = identity(); invalid['freeze']['protocol'][key] = value
            with self.assertRaises(ValueError): dev.validate_dev_identity(invalid)
        invalid = identity(); invalid['ids'] = ['1', '1']
        with self.assertRaises(ValueError): dev.validate_dev_identity(invalid)

    def test_exact_dev_coverage_rejects_foreign_and_duplicate_ids(self):
        dev.exact_coverage(['1', '2'], ['2', '1'])
        for ids in [['1', '3'], ['1'], ['1', '2', '2']]:
            with self.assertRaises(ValueError): dev.exact_coverage(['1', '2'], ids)

    def test_fixed_102_policy_grid(self):
        policies = dev.policies()
        self.assertEqual(len(policies), 102)
        self.assertEqual(policies[0], {'threshold': 0., 'recommit_s': None})
        self.assertEqual(policies[50], {'threshold': 1., 'recommit_s': None})
        self.assertEqual(policies[51], {'threshold': 0., 'recommit_s': 1.5})
        self.assertEqual(policies[-1], {'threshold': 1., 'recommit_s': 1.5})
        self.assertEqual(len({(p['threshold'], p['recommit_s']) for p in policies}), 102)

    def test_official_quantiles_match_pinned_algorithm(self):
        root = Path(os.environ.get('TURNBENCH_SOURCE', '/tmp/pardon-turnbench-source'))
        if not (root/'turnbench/sweep.py').is_file():
            root = Path('/content/turnbench')
        source = root/'turnbench/sweep.py'
        if not source.is_file():
            self.skipTest('Pinned upstream source checkout not available')
        # Execute only the source function AST, without importing CLI/data loaders.
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'candidate_thetas')
        namespace = {'ProbsFile': object, 'N_CANDIDATES': 512}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        official = namespace['candidate_thetas']
        class Schema:
            @classmethod
            def model_validate(cls, value):
                self.assertEqual(value['schema_version'], 1)
                self.assertEqual(value['task'], 'eot')
                self.assertEqual(value['frame_rate_hz'], 6.25)
                return types.SimpleNamespace(probs=[types.SimpleNamespace(
                    speaker_1=types.SimpleNamespace(prob=r['speaker_1']['prob']),
                    speaker_2=types.SimpleNamespace(prob=r['speaker_2']['prob'])) for r in value['probs']])
        stub = types.SimpleNamespace(ProbsFile=Schema, candidate_thetas=official)
        with patch.dict('sys.modules', {'turnbench.sweep': stub}):
            caches = {'1': {'probabilities': np.array([[0., .001], [.002, .98], [0., 1.]])},
                      '2': {'probabilities': np.array([[.3, .3], [.8, 0.]])}}
            values, proof = dev.threshold_grid('official-quantiles', caches)
            pooled = np.concatenate([v['probabilities'].ravel() for v in caches.values()])
            pooled = pooled[pooled > 0]
            expected = np.unique(np.concatenate([np.quantile(pooled, np.linspace(0, 1, 512)),
                                                 [round(.01*i, 2) for i in range(1, 100)]])).tolist()
            self.assertEqual(values, expected)
            self.assertNotIn(0., values)
            self.assertEqual(len(dev.policies(values)), 2*len(expected))
            self.assertEqual(proof['quantile_count'], 512)
            self.assertEqual(len(proof['candidate_function_sha256']), 64)
            zeros, _ = dev.threshold_grid('official-quantiles', {'1': {'probabilities': np.zeros((2, 2))}})
            self.assertEqual(zeros, [round(.01*i, 2) for i in range(1, 100)])
            self.assertEqual(len(dev.policies(zeros)), 198)
        defaults, proof = dev.threshold_grid()
        self.assertEqual(defaults, np.linspace(0, 1, 51).tolist())
        self.assertEqual(dev.policies(defaults), dev.policies())

    def test_probability_shape_range_availability_and_gate(self):
        arrays = {'times': np.array([.16, .32]), 'vad': np.zeros((2, 2)),
                  'gate_vad': np.zeros((2, 2)), 'probabilities': np.full((2, 2), .5)}
        dev.validate_arrays(arrays, 'mean', .4)
        for key, value in [('times', np.array([0., .16])), ('probabilities', np.ones((1, 2))),
                           ('probabilities', np.full((2, 2), np.nan)), ('probabilities', np.full((2, 2), 1.1)),
                           ('gate_vad', np.ones((2, 2)))]:
            bad = copy.deepcopy(arrays); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): dev.validate_arrays(bad, 'mean', .4)
        with self.assertRaises(ValueError): dev.validate_arrays(arrays, 'mean', .3)
        bad = copy.deepcopy(arrays); bad['annotation'] = np.zeros(1)
        with self.assertRaises(ValueError): dev.validate_arrays(bad, 'mean', .4)

    def test_selection_never_exceeds_ceiling(self):
        def row(recall, fp, latency=100):
            return {'score': {'recall': recall, 'fp_rate': fp, 'latency_ms': {'p50': latency}}}
        over = row(.99, .10001); qualified = row(.9, .1)
        self.assertIs(dev.choose_attempt([over, qualified]), qualified)
        self.assertIsNone(dev.choose_attempt([over]))
        lower_fp = row(.9, .05)
        self.assertIs(dev.choose_attempt([qualified, lower_fp]), lower_fp)
        faster = row(.9, .05, 50)
        self.assertIs(dev.choose_attempt([lower_fp, faster]), faster)


if __name__ == '__main__': unittest.main()
