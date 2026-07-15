"""CPU unit tests for xgrammar constrained decoding on the TPU backend.

Covers the packed-bitmask torch fallback and verifies that xgrammar dispatches
TPU tensors through it.
"""

import inspect
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


class TestXGrammarBitmaskTorchFallback(unittest.TestCase):
    def test_torch_fallback_masks_disallowed_tokens(self):
        from xgrammar import apply_token_bitmask_inplace

        # 2 requests, vocab=70 -> packed bitmask is [2, ceil(70/32)] = [2, 3].
        bs, vocab = 2, 70
        logits = torch.zeros((bs, vocab), dtype=torch.float32)

        # Allow {3, 17, 42} for req0; allow {0, 31, 63} for req1.
        allowed = [{3, 17, 42}, {0, 31, 63}]
        bitmask = torch.zeros((bs, (vocab + 31) // 32), dtype=torch.int32)
        for r, ids in enumerate(allowed):
            for tok in ids:
                bitmask[r, tok // 32] |= 1 << (tok % 32)

        apply_token_bitmask_inplace(logits, bitmask, backend="torch_native")

        for r, ids in enumerate(allowed):
            for tok in range(vocab):
                if tok in ids:
                    self.assertEqual(logits[r, tok].item(), 0.0)
                else:
                    self.assertTrue(
                        logits[r, tok].item() == float("-inf"),
                        f"req{r} token {tok} should be -inf, got {logits[r, tok].item()}",
                    )


class TestXGrammarTpuDispatch(unittest.TestCase):
    def _get_backend_module(self):
        try:
            from sglang.srt.constrained import xgrammar_backend
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"xgrammar backend unavailable: {exc}")
        return xgrammar_backend

    def test_grammar_apply_vocab_mask_routes_tpu_through_torch_fallback(self):
        mod = self._get_backend_module()
        src = inspect.getsource(mod.XGrammarGrammar.apply_vocab_mask)
        self.assertIn('"tpu"', src)
        self.assertIn("apply_token_bitmask_inplace", src)
        self.assertIn('backend="torch_compile"', src)

    def test_backend_apply_vocab_mask_routes_tpu_through_torch_fallback(self):
        mod = self._get_backend_module()
        src = inspect.getsource(mod.XGrammarGrammarBackend.apply_vocab_mask)
        self.assertIn('"tpu"', src)
        self.assertIn("apply_token_bitmask_inplace", src)
        self.assertIn('backend="torch_compile"', src)


if __name__ == "__main__":
    unittest.main()
