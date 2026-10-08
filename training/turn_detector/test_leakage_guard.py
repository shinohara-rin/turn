"""Synthetic source, split and identity isolation tests; no real data access."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from leakage_guard import TRAIN_REVISION, TRAIN_SOURCE, validate_manifest
from baseline import read_records
from heads import load_records


class LeakageContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'cache').mkdir()
        self.manifest = self.root/'cache'/'manifest.json'
        self.records = []
        self.metadata = []
        self.plan = dict(revision=TRAIN_REVISION, splits={}, assignments={})
        for split in ('train', 'dev', 'gate'):
            cid = f'{split}-conversation'
            a, b = f'{split}-actor-a', f'{split}-actor-b'
            self.records.append(dict(id=cid, split=split, source=TRAIN_SOURCE,
                                     revision=TRAIN_REVISION, npz='never-open.npz', events='never-open.json'))
            self.metadata.append(dict(_dir=cid, speaker_1_actor_id=a, speaker_2_actor_id=b))
            self.plan['splits'][split] = [cid]
            self.plan['assignments'].update({a:split, b:split})
        self.write_fixture()

    def write_fixture(self):
        self.manifest.write_text(json.dumps(self.records))
        (self.root/'metadata.json').write_text(json.dumps(self.metadata))
        (self.root/'split.json').write_text(json.dumps(self.plan))

    def reject(self, message):
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, message):
            validate_manifest(self.manifest)

    def test_valid_disjoint_fixture_never_opens_arrays(self):
        report = validate_manifest(self.manifest)
        self.assertTrue(report['passed'])
        self.assertFalse(report['test_data_loaded'])
        self.assertEqual(report['counts'], dict(train=1,dev=1,gate=1))
        self.assertEqual(report['speaker_counts'],dict(train=2,dev=2,gate=2))

    def test_benchmark_source_disguised_as_train_is_rejected(self):
        for source in ('sesame/turnbench', 'sesame/turnbench-test', '', None):
            with self.subTest(source=source):
                self.records[0]['source'] = source
                self.reject('source prohibited')

    def test_wrong_or_missing_record_revision_is_rejected(self):
        for revision in ('main', 'wrong-hash', None):
            with self.subTest(revision=revision):
                self.records[0]['revision'] = revision
                self.reject('revision must be pinned')

    def test_wrong_plan_revision_is_rejected(self):
        self.plan['revision'] = 'main'
        self.reject('Unapproved training revision')

    def test_repeated_actor_across_partitions_is_rejected(self):
        self.metadata[1]['speaker_1_actor_id'] = self.metadata[0]['speaker_1_actor_id']
        self.reject('Actor assigned to another partition')

    def test_duplicate_conversation_is_rejected(self):
        self.records.append(copy.deepcopy(self.records[0]))
        self.reject('Duplicate conversation')

    def test_wrong_membership_is_rejected(self):
        self.records[0]['split'] = 'dev'
        self.reject('Conversation not in frozen speaker split')

    def test_both_loaders_validate_all_sources_before_opening_arrays(self):
        # A contaminated train record blocks a dev request before even safe dev
        # arrays are opened; validation cannot be bypassed by choosing a split.
        self.records[0]['source'] = 'sesame/turnbench-test'
        self.write_fixture()
        for loader in (read_records, load_records):
            with self.subTest(loader=loader.__module__):
                with patch('numpy.load',side_effect=AssertionError('Array read before guard')) as opened:
                    with self.assertRaisesRegex(ValueError,'source prohibited'):
                        loader(self.manifest, 'dev')
                    opened.assert_not_called()

    def test_optimization_flag_cannot_disable_guard(self):
        self.records[0]['source'] = 'sesame/turnbench-test'
        self.write_fixture()
        result = subprocess.run([sys.executable,'-O','-c',
            'from leakage_guard import validate_manifest; import sys; validate_manifest(sys.argv[1])',
            str(self.manifest)],cwd=Path(__file__).resolve().parent,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('ValueError: Benchmark or unidentified source prohibited',result.stderr)


if __name__ == '__main__':
    unittest.main()
