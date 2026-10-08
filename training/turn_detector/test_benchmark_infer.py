"""Synthetic protocol tests only; no benchmark, audio, weights or Torch needed."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import benchmark_infer as infer
from checkpointing import _atomic_write


class Contracts(unittest.TestCase):
    def test_dispatch(self):
        self.assertEqual(infer.model_kind({'auxiliary': False, 'model': {}}), 'mlp')
        self.assertEqual(infer.model_kind({'config': {'model': 'projection-GRU-EOT', 'bidirectional': False}}), 'gru')
        self.assertEqual(infer.model_kind({'format': 'equal-probability-turn-heads-v1'}), 'ensemble')
        for value in [{}, {'config': {'model': 'projection-GRU-EOT', 'bidirectional': True}}]:
            with self.assertRaises(ValueError): infer.model_kind(value)
        calls = []
        class Fake:
            def load_state_dict(self, state, strict): calls.append(('load', strict))
            def to(self, device): return self
            def eval(self): return self
        heads = types.SimpleNamespace(make_model=lambda *a: Fake(), predict=lambda *a: ['mlp'])
        temporal = types.SimpleNamespace(make_temporal_model=lambda *a: Fake(), temporal_predict=lambda *a: ['gru', a[-1]])
        ensemble = types.SimpleNamespace(EnsemblePredictor=lambda *a: types.SimpleNamespace(predict=lambda records: ['ensemble']))
        with patch.dict('sys.modules', heads=heads, temporal_head=temporal, ensemble_heads=ensemble):
            state = {'model': {}, 'dim': 2, 'hidden': 3, 'mean': 0, 'std': 1, 'auxiliary': False}
            self.assertEqual(infer.predictor(state, 'cpu')([]), ['mlp'])
            state.update(config={'model': 'projection-GRU-EOT', 'bidirectional': False, 'chunk_frames': 256}, projection=2, dropout=.1)
            self.assertEqual(infer.predictor(state, 'cpu')([]), ['gru', 256])
            self.assertEqual(infer.predictor({'format': 'equal-probability-turn-heads-v1'}, 'cpu')([]), ['ensemble'])
        self.assertEqual(calls, [('load', True), ('load', True)])

    def test_freeze_rejects_each_identity_change(self):
        reference = {'sources': {'heads.py': 'a'}, 'weights': 'b', 'gate': 'mean', 'op': .5,
                     'dependencies': {'torch': '1'}, 'batch': 4}
        for key in reference:
            changed = dict(reference); changed[key] = 'modified'
            with self.subTest(key=key), self.assertRaises(ValueError):
                infer.require_equal(reference, changed, 'freeze')
        infer.require_equal(reference, json.loads(json.dumps(reference)), 'freeze')

    def test_source_closure_detects_new_and_changed_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); exp = root / 'exp'; up = root / 'turnbench'
            exp.mkdir(); up.mkdir()
            (exp / 'heads.py').write_text('a'); (up / 'submission.py').write_text('b')
            with patch.object(infer, 'ROOT', exp):
                original = infer.source_lock(root)
                (exp / 'new_helper.py').write_text('c')
                self.assertNotEqual(original, infer.source_lock(root))
                (exp / 'new_helper.py').unlink(); (up / 'submission.py').write_text('changed')
                self.assertNotEqual(original, infer.source_lock(root))

    def test_dependency_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'module.py').write_text('a')
            distribution = types.SimpleNamespace(metadata={'Name': 'fake'}, version='1',
                files=[Path('module.py')], locate_file=lambda name: root / name)
            with patch.object(infer.importlib.metadata, 'distributions', return_value=[distribution]):
                original = infer.dependency_lock()
                (root / 'module.py').write_text('b')
                self.assertNotEqual(original, infer.dependency_lock())

    def test_duplicate_distributions_and_import_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp)/name for name in ('system', 'local')]
            for root in roots:
                root.mkdir(); (root/'module.py').write_text(root.name)
            def distribution(root, version):
                return types.SimpleNamespace(metadata={'Name': 'Python_Apt'}, version=version,
                    files=[Path('module.py')], locate_file=lambda name: root/name)
            variants = [distribution(roots[0], '1'), distribution(roots[1], '2')]
            with patch.object(infer.importlib.metadata, 'distributions', return_value=variants), \
                 patch.object(infer.sys, 'path', [str(roots[1]), str(roots[0])]):
                original = infer.dependency_lock()
                self.assertEqual(len(original['packages']['python-apt']), 2)
                self.assertEqual({v['version'] for v in original['packages']['python-apt']}, {'1', '2'})
                self.assertEqual({v['origin'] for v in original['packages']['python-apt']}, {str(r.resolve()) for r in roots})
                variants.reverse()
                self.assertEqual(original, infer.dependency_lock())  # Discovery order is not import order.
                infer.sys.path.reverse()
                self.assertNotEqual(original, infer.dependency_lock())
                infer.sys.path.reverse()
                (roots[0]/'module.py').write_text('changed system implementation')
                self.assertNotEqual(original, infer.dependency_lock())
                variants.append(variants[0])
                self.assertEqual(len(infer.dependency_lock()['packages']['python-apt']), 3)

    def test_resume_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'; identity = {'freeze': {'gate': 'mean'}, 'ids': ['1']}
            infer.open_run(out, identity, False)
            infer.open_run(out, identity, True)
            with self.assertRaises(ValueError): infer.open_run(out, {'freeze': {'gate': 'last'}}, True)
            with self.assertRaises(FileExistsError): infer.open_run(out, identity, False)

    def test_atomic_interruption_retains_complete_previous_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '1.json'; path.write_bytes(b'complete')
            def interrupted(file):
                file.write(b'partial'); raise RuntimeError('simulated crash')
            with self.assertRaises(RuntimeError): _atomic_write(path, interrupted)
            self.assertEqual(path.read_bytes(), b'complete')
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_resampled_roundup_never_extends_source_duration(self):
        import numpy as np
        class Table:
            def column(self, name):
                return [types.SimpleNamespace(as_py=lambda: {'bytes': b'synthetic'})]
        sound = types.SimpleNamespace(read=lambda *a, **k: (np.zeros(7679, np.float32), 48000))
        encoder = types.SimpleNamespace(causal_resample=lambda *a: (np.zeros((2560, 2), np.float32), 0))
        with patch.dict('sys.modules', soundfile=sound, encoder=encoder):
            with self.assertRaisesRegex(ValueError, 'shorter than decision frame'):
                infer.decode_audio(Table(), {'1': 0}, '1')
        sound.read = lambda *a, **k: (np.zeros(15359, np.float32), 48000)
        encoder.causal_resample = lambda *a: (np.zeros((5120, 2), np.float32), 0)
        with patch.dict('sys.modules', soundfile=sound, encoder=encoder):
            wave, duration, _ = infer.decode_audio(Table(), {'1': 0}, '1')
            self.assertEqual(len(wave), 2560)
            self.assertLessEqual(len(wave)/16000, duration)

    def test_operating_point_contract(self):
        self.assertEqual(infer.operating_point({'threshold': .5, 'recommit_s': None}), {'threshold': .5, 'recommit_s': None})
        for threshold, recommit in [(float('nan'), None), (1.1, None), (.5, 2)]:
            with self.assertRaises(ValueError): infer.operating_point({'threshold': threshold, 'recommit_s': recommit})


if __name__ == '__main__': unittest.main()
