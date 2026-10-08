"""Parent-run synthetic contracts; no datasets, pretrained models, or gate access.

python /content/turn-recreation/test_ensemble_heads.py
Uses tiny randomly initialized existing MLP/GRU architectures for inference tests.
"""
import copy
import unittest
import numpy as np
from ensemble_heads import aligned_mean, validate_members, EnsemblePredictor


def member(kind, seed, dim=3):
    from heads import make_model
    from temporal_head import make_temporal_model
    config = {'seed': seed, 'manifest_sha256': 'frozen-manifest', 'chunk_frames': 3}
    if kind == 'mlp':
        config['variant'] = 'mlp'; model = make_model(dim, 4, False)
        extra = {'auxiliary': False}
    else:
        config.update(model='projection-GRU-EOT', bidirectional=False)
        model = make_temporal_model(dim, hidden=4, projection=4, dropout=0.)
        extra = {'projection': 4, 'dropout': 0.}
    checkpoint = {'model': model.state_dict(), 'config': config, 'mean': np.zeros(dim, np.float32),
                  'std': np.ones(dim, np.float32), 'dim': dim, 'hidden': 4, 'epoch': 1, **extra}
    return {'kind': kind, 'checkpoint': checkpoint}


class EnsembleTests(unittest.TestCase):
    def test_equal_probability_mean(self):
        def p(value):
            return [{'id': 'a', 'times': np.array([.16, .32]),
                     'probabilities': np.full((2, 2), value, np.float32)}]
        np.testing.assert_allclose(aligned_mean([p(.1), p(.3), p(.8)])[0], .4, atol=1e-7)

    def test_alignment_and_identity_fail_closed(self):
        a = [{'id': 'a', 'times': np.array([.16]), 'probabilities': np.array([[.2, .8]])}]
        for change in [{'id': 'b'}, {'times': np.array([.32])}, {'probabilities': np.array([[np.nan, .2]])},
                       {'probabilities': np.array([[1.1, .2]])}]:
            b = [{**a[0], **change}]
            with self.assertRaises(ValueError):
                aligned_mean([a, b])
        with self.assertRaises(ValueError):
            aligned_mean([a, []])

    def test_manifest_and_seed_identity(self):
        members = [member('mlp', 42), member('gru', 42)]
        validate_members(members, 'mlp_gru', 'frozen-manifest')
        altered = copy.deepcopy(members); altered[1]['checkpoint']['config']['manifest_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'source manifests'):
            validate_members(altered, 'mlp_gru', 'frozen-manifest')
        altered = copy.deepcopy(members); altered[0]['checkpoint']['config']['seed'] = 17
        with self.assertRaisesRegex(ValueError, 'seeds/kinds'):
            validate_members(altered, 'mlp_gru', 'frozen-manifest')

    def test_bundle_mean_causality_and_no_label_access(self):
        from heads import make_model, predict
        from temporal_head import make_temporal_model, temporal_predict
        members = [member('mlp', 42), member('gru', 42)]
        bundle = {'format': 'equal-probability-turn-heads-v1', 'members': members,
                  'candidate': 'mlp_gru', 'weights': [.5, .5], 'manifest_sha256': 'frozen-manifest'}
        predictor = EnsemblePredictor(bundle)
        class InferenceOnly(dict):
            def __getitem__(self, key):
                if key not in ('id', 'x', 'times', 'vad'):
                    raise AssertionError('Attempted annotation access: '+key)
                return super().__getitem__(key)
        rng = np.random.default_rng(61)
        r = InferenceOnly(id='synthetic', times=np.arange(1, 13)*.16,
                          x=rng.normal(size=(12, 2, 3)).astype(np.float32), vad=np.zeros((12, 2), np.float32))
        actual = predictor.predict([r])[0]
        direct = []
        for m in members:
            c = m['checkpoint']
            if m['kind'] == 'mlp':
                model = make_model(3, 4, False); model.load_state_dict(c['model'])
                direct.append(predict(model, [r], c['mean'], c['std'], 'cpu')[0])
            else:
                model = make_temporal_model(3, 4, 4, 0.); model.load_state_dict(c['model'])
                direct.append(temporal_predict(model, [r], c['mean'], c['std'], chunk_frames=3)[0])
        np.testing.assert_allclose(actual, (direct[0].astype(np.float64)+direct[1])/2, atol=1e-7)
        changed = InferenceOnly(r); changed['x'] = r['x'].copy(); changed['x'][6:] *= 100
        np.testing.assert_allclose(actual[:6], predictor.predict([changed])[0][:6], atol=1e-7)
        shortened = InferenceOnly({k: (r[k][:6] if k != 'id' else r[k]) for k in ('id', 'x', 'times', 'vad')})
        np.testing.assert_allclose(actual[:6], predictor.predict([shortened])[0], atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'equal weights'):
            EnsemblePredictor({**bundle, 'weights': [.6, .4]})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            predictor.predict([r, r])


if __name__ == '__main__':
    unittest.main()
