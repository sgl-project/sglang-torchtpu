"""The sparse MLA kernel's softmax normalization takes ``pltpu.reciprocal``
where jax has it (jax 0.11 on) and ``pl.reciprocal``, the same op, on older
jax."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# Needs jax and torch_tpu, which the CPU runners do not install; the TPU unit
# stage's time budget is spent, so test_k3_programs_nightly.py runs it in the
# TPU nightly.
register_tpu_ci(
    est_time=10,
    suite="nightly-tpu-4-v7",
    nightly=True,
    disabled="runs in the TPU nightly through test_k3_programs_nightly.py",
)

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
