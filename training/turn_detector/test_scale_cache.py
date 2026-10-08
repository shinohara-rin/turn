"""Metadata-only synthetic scaling safeguards; no datasets or models."""
from pathlib import Path
import json
import tempfile
import time
import unittest
from scale_cache import (full_selection,validate_selection,verified_drive_directory,
                         reuse_verified_cache,DATA_FILES,TRAIN_SOURCE,TRAIN_REVISION)
from cache_identity import sha256_file


class ScaleCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.plan = {'revision': TRAIN_REVISION,
                     'assignments': {'a': 'train', 'b': 'dev', 'c': 'gate'},
                     'splits': {'train': ['1'], 'dev': ['2'], 'gate': ['3'], 'excluded_cross_partition': ['4']}}
        self.split = self.root/'split.json'; self.split.write_text(json.dumps(self.plan))
        rows = [{'_dir': cid, 'speaker_1_actor_id': actor, 'speaker_2_actor_id': actor}
                for cid, actor in [('1','a'), ('2','b'), ('3','c')]]
        (self.root/'metadata.json').write_text(json.dumps(rows))

    def tearDown(self):
        self.temp.cleanup()

    def test_full_selection_never_gate(self):
        selected = full_selection(self.split)
        self.assertEqual([(r['id'],r['split']) for r in selected], [('1','train'),('2','dev')])

    def test_selection_rejects_gate_source_duplicate_and_cross_split(self):
        bad = [[{'id':'3','split':'gate'}], [{'id':'1','split':'train','source':'benchmark'}],
               [{'id':'1','split':'dev'}], [{'id':'1','split':'train'}]*2,
               [{'id':'../1','split':'train'}]]
        for records in bad:
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_selection(records,self.split)

    def test_drive_requires_every_file_and_pinned_metadata(self):
        drive = self.root/'drive'; directory = drive/'1'; directory.mkdir(parents=True)
        for name in DATA_FILES:
            (directory/name).write_bytes(b'synthetic')
        self.assertIsNone(verified_drive_directory(drive,'1'))
        for name in DATA_FILES:
            path = drive/'.cache/huggingface/download/1'/(name+'.metadata'); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(f'{TRAIN_REVISION}\nsynthetic-etag\n{time.time()}\n')
        self.assertEqual(verified_drive_directory(drive,'1'),directory)
        path.write_text('wrong-revision\nsynthetic-etag\n0\n')
        self.assertIsNone(verified_drive_directory(drive,'1'))

    def test_reuse_preserves_original_bytes_and_checks_hashes(self):
        pilot = self.root/'pilot'; pilot.mkdir(); output = self.root/'scaled'; output.mkdir()
        records = []
        for cid, split in [('1','train'),('2','dev')]:
            npz = pilot/f'{cid}.npz'; event = pilot/f'{cid}.events.json'
            npz.write_bytes(b'synthetic-not-an-array'); event.write_text('{}')
            (pilot/f'{cid}.provenance.json').write_text(json.dumps({'encoder_source_sha256':'encoder'}))
            records.append({'id':cid,'split':split,'source':TRAIN_SOURCE,'revision':TRAIN_REVISION,
                            'npz':str(npz),'events':str(event),'npz_sha256':sha256_file(npz),'events_sha256':sha256_file(event)})
        manifest = pilot/'manifest.json'; manifest.write_text(json.dumps(records))
        before = {p.name:p.read_bytes() for p in pilot.iterdir()}
        reused = reuse_verified_cache(manifest,output,full_selection(self.split),{},'encoder',self.split)
        self.assertEqual(len(reused),2)
        self.assertEqual(before,{p.name:p.read_bytes() for p in pilot.iterdir()})
        self.assertFalse((output/'1.npz').exists())  # immutable reference, no large copy
        (pilot/'1.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            reuse_verified_cache(manifest,output,full_selection(self.split),{},'encoder',self.split)

    def test_continued_reuse_requires_expanded_policy_ancestry(self):
        pilot=self.root/'continued';pilot.mkdir();output=self.root/'full';output.mkdir()
        npz=pilot/'1.npz';npz.write_bytes(b'synthetic-continued-features')
        events=pilot/'1.events.json';events.write_text('{}')
        old={'encoder_checkpoint_sha256':'weights','cache_entrypoint_sha256':'old-io',
             'policy_cache_manifest_sha256':'original-policy','continuation_source_sha256':'training-code'}
        new={**old,'cache_entrypoint_sha256':'new-io','policy_cache_manifest_sha256':'expanded-policy'}
        provenance=pilot/'1.provenance.json'
        provenance.write_text(json.dumps({**old,'encoder_source_sha256':'encoder'}))
        record={'id':'1','split':'train','source':TRAIN_SOURCE,'revision':TRAIN_REVISION,
                'encoder_checkpoint_sha256':'weights','npz':str(npz),'events':str(events),
                'npz_sha256':sha256_file(npz),'events_sha256':sha256_file(events)}
        manifest=pilot/'manifest.json';manifest.write_text(json.dumps([record]))
        with self.assertRaisesRegex(ValueError,'ancestry'):
            reuse_verified_cache(manifest,output,full_selection(self.split),new,'encoder',self.split)
        policy={**record,'reused_from_manifest_sha256':'original-policy'}
        reused=reuse_verified_cache(manifest,output,full_selection(self.split),new,'encoder',self.split,
                                    policy_records={'1':policy})
        self.assertEqual(len(reused),1)
        assembled=json.loads((output/'1.provenance.json').read_text())
        self.assertEqual(assembled['cache_assembly_identity'],new)
        self.assertEqual(assembled['cache_entrypoint_sha256'],'old-io')
        self.assertEqual(json.loads(provenance.read_text())['policy_cache_manifest_sha256'],'original-policy')


if __name__ == '__main__':
    unittest.main()
