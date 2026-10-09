"""CatTop: re-running the Cat top stage from a cached tap (shrunken config, random weights)."""
import unittest

import torch

import cat_encoder as ce
import cat_top
from test_cat_encoder import CODE_DIR, tiny_config


class CatTopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        modeling, config = ce.load_upstream(CODE_DIR)
        cls.encoder = ce.CatEncoder(modeling, tiny_config(config), taps=(1, 3)).eval()
        with torch.no_grad():
            cls.out = cls.encoder(torch.randn(1, 1920 * 60))

    def top(self, rank=4):
        return cat_top.CatTop(self.encoder, first=2, taps=(3,), rank=rank).eval()

    def test_reproduces_cached_upper_tap_and_final(self):
        with torch.no_grad():
            o = self.top()(self.out['taps'][:, :, 0])
        torch.testing.assert_close(o[3], self.out['taps'][:, :, 1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(o['final'], self.out['final'], rtol=1e-4, atol=1e-4)

    def test_encoder_untouched_and_only_lora_trainable(self):
        top = self.top()
        names = [n for n, p in top.named_parameters() if p.requires_grad]
        self.assertTrue(names and all(n.endswith(('.A', '.B')) for n in names))
        self.assertFalse(any(isinstance(m, cat_top.LoRALinear) for m in self.encoder.modules()))

    def test_chunk_with_context_is_exact_after_adaptation(self):
        top = self.top()
        with torch.no_grad():
            for m in top.modules():
                if isinstance(m, cat_top.LoRALinear):
                    m.B.normal_(0, 0.05)
            x = self.out['taps'][:, :, 0]
            full = top(x)['final']
            s = 50
            lo = max(0, s - top.context)
            part = top(x[:, lo:])['final'][:, s - lo:]
        torch.testing.assert_close(part, full[:, s:], rtol=1e-4, atol=1e-4)

    def test_causal(self):
        top = self.top()
        with torch.no_grad():
            x = self.out['taps'][:, :, 0].clone()
            a = top(x)['final']
            x[:, 40:] += 1.0
            b = top(x)['final']
        torch.testing.assert_close(a[:, :40], b[:, :40])


if __name__ == '__main__':
    unittest.main()
