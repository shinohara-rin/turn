"""Synthetic gate contracts: no real dataset, model weights or runtime access."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import gate_evaluate as gate


def plan_and_metadata():
    plan = {'revision': gate.TRAIN_REVISION, 'splits': {'train': ['train'], 'dev': ['dev'],
            'gate': [f'gate{i}' for i in range(20)]}, 'assignments': {}}
    metadata = []
    for part, ids in plan['splits'].items():
        for cid in ids:
            row = {'_dir': cid}
            for s in (1, 2):
                actor = f'{cid}-actor{s}'
                row[f'speaker_{s}_actor_id'] = actor; plan['assignments'][actor] = part
            metadata.append(row)
    return plan, metadata


class GateContracts(unittest.TestCase):
    def test_membership_rejects_leakage_and_subset(self):
        plan, metadata = plan_and_metadata()
        self.assertEqual(len(gate.gate_membership(plan, metadata)['ids']), 20)
        broken = copy.deepcopy(plan); broken['splits']['gate'].pop()
        with self.assertRaises(ValueError): gate.gate_membership(broken, metadata)
        broken = copy.deepcopy(metadata); broken[-1]['speaker_1_actor_id'] = metadata[0]['speaker_1_actor_id']
        with self.assertRaises(ValueError): gate.gate_membership(plan, broken)
        broken = copy.deepcopy(plan); broken['splits']['train'].append('gate0')
        with self.assertRaises(ValueError): gate.gate_membership(broken, metadata)
        broken = copy.deepcopy(plan); broken['revision'] = 'benchmark'
        with self.assertRaises(ValueError): gate.gate_membership(broken, metadata)

    def test_selection_is_fixed(self):
        policies = [{'name': 'candidate', 'role': 'selected', 'operating_point': {'threshold': .5, 'recommit_s': None}},
                    {'name': 'control', 'role': 'baseline', 'operating_point': {'threshold': .8, 'recommit_s': 1.5}}]
        gate.fixed_policies(policies, 'Selected by frozen dev score')
        with self.assertRaises(ValueError): gate.fixed_policies(policies, '')
        with self.assertRaises(ValueError): gate.fixed_policies(policies[:1], 'dev')
        broken = copy.deepcopy(policies); broken[1]['role'] = 'selected'
        with self.assertRaises(ValueError): gate.fixed_policies(broken, 'dev')
        broken = copy.deepcopy(policies); broken[0]['operating_point']['search'] = True
        with self.assertRaises(ValueError): gate.fixed_policies(broken, 'dev')

    def test_ledger_only_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ledger.json'; out = Path(tmp)/'out'
            gate.ledger_open(path, 'frozen-a', out, False)
            gate.ledger_open(path, 'frozen-a', out, True)
            with self.assertRaises(ValueError): gate.ledger_open(path, 'frozen-a', out, False)
            with self.assertRaises(ValueError): gate.ledger_open(path, 'frozen-b', out, True)
            with self.assertRaises(ValueError): gate.ledger_open(path, 'frozen-a', Path(tmp)/'new', True)

    def test_cache_provenance_and_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cache.npz'; identity = {'id': 'synthetic', 'freeze': 'fixed'}
            path.write_bytes(b'partial')
            self.assertIsNone(gate.cached_npz(path, identity))
            gate.save_npz(path, identity, times=np.array([.16, .32]))
            np.testing.assert_array_equal(gate.cached_npz(path, identity)['times'], [.16, .32])
            with self.assertRaises(ValueError): gate.cached_npz(path, {'id': 'other'})
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): gate.cached_npz(path, identity)

    def test_source_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic'; path.write_bytes(b'hello')
            expected = {'algorithm': 'git-blob-sha1', 'hash': hashlib.sha1(b'blob 5\0hello').hexdigest()}
            gate.verify_gate_file(path, expected)
            gate.verify_gate_file(path, {'algorithm': 'sha256', 'hash': gate.sha(path)})
            path.write_bytes(b'world')
            with self.assertRaises(ValueError): gate.verify_gate_file(path, expected)

    def test_paired_uncertainty_no_reselection(self):
        policies = [{'name': 'chosen', 'role': 'selected'}, {'name': 'baseline', 'role': 'baseline'}]
        score = {'tp': 2, 'fn': 1, 'fp': 1, 'tn': 3, 'latencies_ms': [200., 500.]}
        rows = [{'chosen': score, 'baseline': copy.deepcopy(score)} for _ in range(5)]
        ids = [str(i) for i in range(5)]; actors = {i: [i+'a', i+'b'] for i in ids}
        result = gate.paired_report(rows, policies, ids, actors, replicates=20)
        for metric in result['paired_differences']['baseline'].values():
            self.assertEqual(metric['selected_minus_baseline'], 0)
            self.assertEqual(metric['percentile_95_interval'], [0, 0])
            self.assertEqual(metric['valid_replicates'], 20)
        self.assertEqual(result['aggregate']['chosen']['tp'], 10)
        self.assertEqual(result, gate.paired_report(rows, policies, ids, actors, replicates=20))


    def test_actor_components_transitive_and_small_sample_suppression(self):
        ids = ['a', 'b', 'c', 'd']
        actors = {'a': ['1', '2'], 'b': ['2', '3'], 'c': ['3', '4'], 'd': ['5', '6']}
        self.assertEqual(gate.actor_components(ids, actors), [[0, 1, 2], [3]])
        policies = [{'name': 'chosen', 'role': 'selected'}, {'name': 'baseline', 'role': 'baseline'}]
        score = {'tp': 2, 'fn': 1, 'fp': 1, 'tn': 3, 'latencies_ms': [200., 500.]}
        rows = [{'chosen': score, 'baseline': score} for _ in ids]
        report = gate.paired_report(rows, policies, ids, actors, replicates=20)
        self.assertEqual(report['uncertainty']['component_sizes'], [3, 1])
        self.assertTrue(report['uncertainty']['confidence_intervals_suppressed'])
        for policy in report['aggregate_95_intervals'].values():
            for metric in policy.values():
                self.assertIsNone(metric['percentile_95_interval'])
                self.assertEqual(metric['valid_replicates'], 0)
        for metric in report['paired_differences']['baseline'].values():
            self.assertIsNone(metric['percentile_95_interval'])
            self.assertEqual(metric['valid_replicates'], 0)


if __name__ == '__main__': unittest.main()
