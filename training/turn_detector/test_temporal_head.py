"""Synthetic only. Torch forward tests skip if torch is not already installed."""
import importlib.util
import unittest
import numpy as np
from temporal_head import chunk_ranges, supervised_weights, train_scaler, make_temporal_model, temporal_predict


class TemporalArrayContracts(unittest.TestCase):
    def test_chunk_ranges_cover_in_order_without_padding(self):
        self.assertEqual(chunk_ranges(11,4),[(0,4),(4,8),(8,11)])
        with self.assertRaises(ValueError):
            chunk_ranges(11,0)

    def test_warmup_only_masks_conversation_start(self):
        original = np.ones((11,2),np.float32)
        weights = supervised_weights(dict(w=original),3)
        self.assertEqual(weights[:3].sum(),0)
        np.testing.assert_array_equal(weights[3:],1)
        np.testing.assert_array_equal(original,1)

    def test_streaming_scaler_matches_training_arrays(self):
        x = np.arange(48,dtype=np.float32).reshape(6,2,4)
        records = [dict(x=x[:2]),dict(x=x[2:])]
        mean,std = train_scaler(records)
        np.testing.assert_allclose(mean,x.reshape(-1,4).mean(0))
        np.testing.assert_allclose(std,x.reshape(-1,4).std(0),rtol=1e-6)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch absent: run forward tests in existing remote torch runtime')
class TemporalTorchContracts(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(5)
        self.model = make_temporal_model(4,hidden=5,projection=6,dropout=.1).eval()
        self.x = torch.randn(21,2,4)

    def test_future_mutation_cannot_change_past(self):
        with self.torch.no_grad():
            first,_ = self.model(self.x)
            mutated = self.x.clone()
            mutated[12:] = 1000
            second,_ = self.model(mutated)
        self.torch.testing.assert_close(first[:12],second[:12],rtol=0,atol=0)

    def test_chunked_forward_matches_contiguous_and_resets_conversation(self):
        records = [dict(x=self.x.numpy()),dict(x=self.x.numpy())]
        predicted = temporal_predict(self.model,records,np.zeros(4,np.float32),np.ones(4,np.float32),chunk_frames=7)
        with self.torch.no_grad():
            full,_ = self.model(self.x)
        np.testing.assert_allclose(predicted[0],full.sigmoid().numpy(),rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(predicted[0],predicted[1])

    def test_speaker_hidden_states_are_independent(self):
        with self.torch.no_grad():
            first,_ = self.model(self.x)
            mutated = self.x.clone()
            mutated[:,1] = 1000
            second,_ = self.model(mutated)
        self.torch.testing.assert_close(first[:,0],second[:,0],rtol=0,atol=0)

    def test_detach_truncates_gradient_without_resetting_state(self):
        x = self.x.clone().requires_grad_()
        _,state = self.model(x[:7])
        logits,_ = self.model(x[7:],state.detach())
        logits.sum().backward()
        self.torch.testing.assert_close(x.grad[:7],self.torch.zeros_like(x.grad[:7]),rtol=0,atol=0)
        self.assertGreater(float(x.grad[7:].abs().sum()),0)


if __name__ == '__main__':
    unittest.main()
