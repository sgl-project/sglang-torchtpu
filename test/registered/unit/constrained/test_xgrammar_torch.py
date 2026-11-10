"""CPU unit tests for xgrammar constrained decoding on the TPU backend.

Verifies that xgrammar dispatches TPU tensors through the torch fallback.
"""

import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


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
