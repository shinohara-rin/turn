"""Pure atomic checks locally; opt-in synthetic Torch checks on remote CPU.

RUN_CHECKPOINT_TORCH_TESTS=1 python -m unittest discover -s experiments/turn_detector -p test_checkpointing.py
No real dataset is opened. Torch integration uses tiny generated tensors only.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from checkpointing import _atomic_write, atomic_text, _validate_config


class AtomicFileContracts(unittest.TestCase):
    def test_failed_writer_keeps_prior_complete_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'epoch-001.pt'
            path.write_bytes(b'prior complete checkpoint')
            def fail(file):
                file.write(b'partial')
                self.assertEqual(path.read_bytes(), b'prior complete checkpoint')
                raise RuntimeError('simulated writer failure')
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                _atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), b'prior complete checkpoint')
            self.assertEqual(list(Path(d).iterdir()), [path])
            atomic_text('replacement complete', path)
            self.assertEqual(path.read_text(), 'replacement complete')

    def test_resume_rejects_protocol_and_manifest_changes(self):
        saved = dict(seed=42, lr=.001, epochs=12, manifest_sha256='stable', out='old')
        _validate_config(saved, dict(saved, out='new', resume='auto'))
        for key,value in [('seed',43),('lr',.01),('epochs',13),('manifest_sha256','changed')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                _validate_config(saved, dict(saved, **{key:value}))


REMOTE_TORCH = os.environ.get('RUN_CHECKPOINT_TORCH_TESTS') == '1' and importlib.util.find_spec('torch') is not None


@unittest.skipUnless(REMOTE_TORCH, 'Opt in to synthetic Torch checks on remote CPU with RUN_CHECKPOINT_TORCH_TESTS=1')
class EpochResumeContracts(unittest.TestCase):
    def setUp(self):
        import torch
        import checkpointing
        self.torch = torch
        self.recovery = checkpointing.EpochRecovery
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.threads)

    def assert_nested_equal(self,a,b):
        if isinstance(a,self.torch.Tensor):
            self.torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,np.ndarray):
            np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                self.assert_nested_equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):
                self.assert_nested_equal(left,right)
        else:
            self.assertEqual(a,b)

    def test_cpu_global_and_generator_rng_resume_equivalence(self):
        from checkpointing import capture_rng,restore_rng,atomic_torch_save
        torch = self.torch
        torch.manual_seed(4)
        np.random.seed(4)
        generator = np.random.default_rng(4)
        state = capture_rng(generator)
        expected = (torch.rand(4),np.random.random(4),generator.random(4))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'state.pt'
            atomic_torch_save(state,path)
            loaded = torch.load(path,weights_only=False,map_location='cpu')
        restore_rng(loaded,generator)
        actual = (torch.rand(4),np.random.random(4),generator.random(4))
        self.assert_nested_equal(expected,actual)
        self.assertIn('torch_cuda',loaded)

    def test_auto_resume_selects_newest_mirrored_epoch_and_phase(self):
        from checkpointing import EpochRecovery, atomic_torch_save
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = dict(seed=42, epochs=3)
            state = dict(version=1, phase='complete', attempts=[], best=None, best_inference=None)
            older = dict(epoch=1, config=config, training_state=state)
            newer = dict(epoch=2, config=config, training_state=dict(state, phase='trained'))
            atomic_torch_save(older, root/'epoch-001.pt')
            atomic_torch_save(newer, root/'latest.pt')
            recovery = EpochRecovery(root, config, 'auto')
            self.assertEqual(recovery.start_epoch, 2)
            self.assertTrue(recovery.pending_evaluation(2))
            # Same epoch: completed scoring outranks pending scoring.
            atomic_torch_save(newer, root/'epoch-002.pt')
            newer['training_state']['phase'] = 'complete'
            atomic_torch_save(newer, root/'latest.pt')
            recovery = EpochRecovery(root, config, 'auto')
            self.assertEqual(recovery.start_epoch, 3)
            self.assertFalse(recovery.pending_evaluation(2))

    def test_both_trainers_resume_pending_scoring_or_complete_epoch_exactly(self):
        import heads
        import temporal_head
        from checkpointing import EpochRecovery
        array = np.random.default_rng(11).normal(size=(12,2,4)).astype(np.float32)
        record = dict(id='synthetic', x=array, y=(array[:,:,0]>0).astype(np.float32),
                      w=np.ones((12,2),np.float32), aux=None, aux_mask=None)
        def synthetic_score(records,probabilities,threshold,recommit,path):
            # Deterministic substitute for scorer: validates loop state only.
            return dict(tp=1,fn=1,fp=0,tn=2,recall=float(np.mean(probabilities[0])),
                        fp_rate=0.,latency_ms=dict(p10=0.,p50=0.,p90=0.))
        for module in (heads,temporal_head):
            for interruption in ('trained','complete'):
                with self.subTest(module=module.__name__,interruption=interruption), tempfile.TemporaryDirectory() as d:
                    root=Path(d)
                    manifest=root/'synthetic-manifest.json'
                    manifest.write_text('[]')
                    common=dict(command='train',manifest=str(manifest),turnbench_path='unused',epochs=2,
                                seed=42,hidden=5,lr=.001,threshold_steps=2,resume=None)
                    if module is heads:
                        common.update(variant='mlp',batch_size=7,aux_weight=.2)
                    else:
                        common.update(projection=6,dropout=.1,chunk_frames=5,warmup_frames=2)
                    full=argparse.Namespace(**common,out=str(root/'full'))
                    resumed=argparse.Namespace(**common,out=str(root/'resumed'))
                    @contextlib.contextmanager
                    def synthetic_guard():
                        # Restoring all of sys.modules would unload Torch's lazy
                        # extension modules while their C++ registrations persist.
                        previous = sys.modules.get('leakage_guard')
                        sys.modules['leakage_guard'] = types.SimpleNamespace(validate_manifest=lambda path: dict(passed=True))
                        try:
                            yield
                        finally:
                            if previous is None:
                                sys.modules.pop('leakage_guard', None)
                            else:
                                sys.modules['leakage_guard'] = previous
                    with patch.object(module,'load_records',return_value=[record]), patch.object(module,'official_score',side_effect=synthetic_score), synthetic_guard(), patch('torch.cuda.is_available',return_value=False), contextlib.redirect_stdout(io.StringIO()):
                        module.train(full)
                        if interruption == 'trained':
                            original=EpochRecovery.save_trained
                            def interrupt(manager,*args,**kwargs):
                                result=original(manager,*args,**kwargs)
                                raise RuntimeError('synthetic interruption after training checkpoint')
                            target='save_trained'
                        else:
                            original=EpochRecovery._write_artifacts
                            def interrupt(manager,*args,**kwargs):
                                if manager.best is not None:
                                    raise RuntimeError('synthetic interruption after completed checkpoint')
                                return original(manager,*args,**kwargs)
                            target='_write_artifacts'
                        with patch.object(EpochRecovery,target,interrupt), self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                            module.train(resumed)
                        resumed.resume='auto'
                        module.train(resumed)
                        # Resuming an already finished run is a no-op, not duplicate rows.
                        module.train(resumed)
                    a=self.torch.load(root/'full'/'latest.pt',weights_only=False,map_location='cpu')
                    b=self.torch.load(root/'resumed'/'latest.pt',weights_only=False,map_location='cpu')
                    self.assertEqual(a['epoch'],2)
                    self.assertEqual(b['epoch'],2)
                    self.assert_nested_equal(a['model'],b['model'])
                    for key in ('optimizer','rng','attempts','best'):
                        self.assert_nested_equal(a['training_state'][key],b['training_state'][key])
                    self.assertEqual((root/'full'/'attempts.jsonl').read_text(),(root/'resumed'/'attempts.jsonl').read_text())
                    self.assertEqual((root/'full'/'best.json').read_text(),(root/'resumed'/'best.json').read_text())
                    attempts=[json.loads(line) for line in (root/'resumed'/'attempts.jsonl').read_text().splitlines()]
                    self.assertEqual(len(attempts),8)
                    self.assertEqual(len({(row['epoch'],row['threshold'],row['recommit_s']) for row in attempts}),8)
                    best=self.torch.load(root/'resumed'/'best.pt',weights_only=False,map_location='cpu')
                    self.assertIn('operating_point',best)
                    self.assert_nested_equal(best['model'],b['training_state']['best_inference']['model'])


if __name__ == '__main__':
    unittest.main()
