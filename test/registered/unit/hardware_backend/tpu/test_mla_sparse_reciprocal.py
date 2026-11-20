"""The sparse MLA kernel's softmax normalization takes ``pltpu.reciprocal``
where jax has it (jax 0.11 on) and ``pl.reciprocal``, the same op, on older
jax."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HAS_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("jax", "torch_tpu")
)


@unittest.skipUnless(_HAS_DEPS, "needs jax and torch_tpu")
class TestSparseMlaReciprocal(CustomTestCase):
    def test_reciprocal_follows_jax(self):
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.attention.pallas.mla_sparse import kernel

        if hasattr(pltpu, "reciprocal"):
            self.assertIs(kernel._reciprocal, pltpu.reciprocal)
        else:
            self.assertIs(kernel._reciprocal, pl.reciprocal)


if __name__ == "__main__":
    unittest.main()
