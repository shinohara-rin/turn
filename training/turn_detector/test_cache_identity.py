"""Metadata-only fail-closed contracts; no datasets, models, or GPU required."""
import copy
import unittest
from cache_identity import validate_cache_identity, validate_continuation


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.audit = {'loaded_partitions': ['train'], 'test_data_loaded': False,
                      'split_sha256': 'frozen-split', 'training_ids': ['1', '2']}
        self.state = {'format_version': 1, 'model_repo': 'base', 'model_revision': 'rev',
                      'source_sha256': 'source', 'audit': self.audit, 'update': 25,
                      'encoder': {'weight': 'placeholder'}}

    def validate(self, state):
        validate_continuation(state, self.audit, 'source', 'base', 'rev')

    def test_matching_checkpoint(self):
        self.validate(self.state)

    def test_wrong_base_revision(self):
        state = {**self.state, 'model_revision': 'unapproved'}
        with self.assertRaisesRegex(ValueError, 'revision'):
            self.validate(state)

    def test_training_source_changed(self):
        with self.assertRaisesRegex(ValueError, 'source'):
            self.validate({**self.state, 'source_sha256': 'changed'})

    def test_frozen_split_changed(self):
        state = copy.deepcopy(self.state); state['audit']['split_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'audit'):
            self.validate(state)

    def test_nontraining_access(self):
        audit = {**self.audit, 'loaded_partitions': ['train', 'gate']}
        with self.assertRaisesRegex(ValueError, 'train-only'):
            validate_continuation({**self.state, 'audit': audit}, audit, 'source', 'base', 'rev')

    def test_no_completed_update(self):
        with self.assertRaisesRegex(ValueError, 'completed update'):
            self.validate({**self.state, 'update': 0})

    def test_legacy_frozen_cache(self):
        validate_cache_identity({'encoder_source_sha256': 'old'}, {})

    def test_never_mix_frozen_and_continued(self):
        for left, right in [({}, {'encoder_checkpoint_sha256': 'new'}),
                            ({'encoder_checkpoint_sha256': 'new'}, {}),
                            ({'encoder_checkpoint_sha256': 'old'}, {'encoder_checkpoint_sha256': 'new'})]:
            with self.assertRaisesRegex(ValueError, 'mixes'):
                validate_cache_identity(left, right)

    def test_continued_resume_requires_complete_identity(self):
        identity = {'encoder_checkpoint_sha256': 'new', 'continuation_split_sha256': 'split',
                    'cache_entrypoint_sha256': 'code'}
        validate_cache_identity(dict(identity), identity)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_cache_identity({'encoder_checkpoint_sha256': 'new'}, identity)


if __name__ == '__main__':
    unittest.main()
