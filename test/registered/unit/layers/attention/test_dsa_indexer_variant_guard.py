"""The rope-free / Hadamard-free (Kimi-K3.1) DSA indexer is rejected off-TPU."""

import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"
_FLAGS = ("_is_cuda", "_is_hip", "_is_npu", "_is_xpu")


class TestDSAIndexerVariantGuard(unittest.TestCase):
    def _with_platform(self, flag):
        patches = [patch(f"{_INDEXER}.{f}", f == flag) for f in _FLAGS]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_k31_variants_rejected_on_gpu_paths(self):
        for flag in _FLAGS:
            self._with_platform(flag)
            for skip_rope, apply_hadamard in ((True, True), (False, False)):
                with self.subTest(flag=flag, skip_rope=skip_rope):
                    with self.assertRaises(NotImplementedError):
                        dsa_indexer._check_indexer_variant_supported(
                            skip_rope, apply_hadamard
                        )

    def test_default_indexer_unaffected(self):
        for flag in _FLAGS:
            self._with_platform(flag)
            dsa_indexer._check_indexer_variant_supported(False, True)

    def test_k31_variant_allowed_elsewhere(self):
        # The attention backend that overrides forward_indexer owns this case.
        self._with_platform(None)
        dsa_indexer._check_indexer_variant_supported(True, False)


if __name__ == "__main__":
    unittest.main()
