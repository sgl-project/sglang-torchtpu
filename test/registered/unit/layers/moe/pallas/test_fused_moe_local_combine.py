"""Selection of the destination-major SparseCore combine for local MoE output.

Pure dispatch logic, no TPU: ``_use_destination_major_combine`` reads static
shapes and a mocked ``get_tpu_info``. Cases pin the kernel's own limits and the
boundary with the dense SparseCore kernel (16 lanes divisible by top-k and
whole 16384-row waves). Registered on both lanes and skipped where JAX is
unavailable.
"""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - JAX is absent on some CPU CI agents
    _HAS_JAX = False

if _HAS_JAX:
    from jax.experimental.pallas import tpu as pltpu

    from sglang.srt.layers.moe.pallas.fused_moe_op import (
        _use_destination_major_combine,
    )


def _tpu_info(sc_lanes):
    if sc_lanes == "no-tpu-info":

        def raise_unsupported():
            raise ValueError("unsupported device kind")

        return raise_unsupported
    sparse_core = (
        None
        if sc_lanes is None
        else SimpleNamespace(num_lanes=sc_lanes, num_cores=2, num_subcores=16)
    )
    return lambda: SimpleNamespace(
        sparse_core=sparse_core, num_lanes=128, vmem_capacity_bytes=64 << 20
    )


# (dtype, hidden, topk, num_tokens, source_rows, sc_lanes, selected)
_CASES = {
    "tokens-2048": ("bf16", 4096, 10, 2048, None, 16, True),
    "tokens-2047": ("bf16", 4096, 10, 2047, None, 16, True),
    "tokens-64": ("bf16", 4096, 10, 64, None, 16, False),
    "tokens-256": ("bf16", 4096, 10, 256, None, 16, False),
    "tokens-511": ("bf16", 4096, 10, 511, None, 16, False),
    "tokens-512": ("bf16", 4096, 10, 512, None, 16, True),
    "tokens-544-half-padded": ("bf16", 4096, 10, 544, None, 16, False),
    "tokens-682-half-padded": ("bf16", 4096, 10, 682, None, 16, False),
    "tokens-683": ("bf16", 4096, 10, 683, None, 16, True),
    "tokens-1025": ("bf16", 4096, 10, 1025, None, 16, True),
    "topk-3-tokens-512-padded": ("bf16", 4096, 3, 512, 4096, 16, True),
    "topk-2-below-vmem-cutoff": ("bf16", 4096, 2, 1024, None, 16, False),
    "source-below-vmem-cutoff": ("bf16", 4096, 2, 1024, 2457, 16, False),
    "source-above-vmem-cutoff": ("bf16", 4096, 2, 1024, 2458, 16, True),
    "fp32": ("f32", 4096, 10, 2048, None, 16, False),
    "fp16": ("f16", 4096, 10, 2048, None, 16, False),
    "width-512-below-alignment": ("bf16", 512, 10, 1024, None, 16, False),
    "width-2048-tokens-1023": ("bf16", 2048, 10, 1023, None, 16, False),
    "width-2048-tokens-1024": ("bf16", 2048, 10, 1024, None, 16, True),
    "width-2048-tokens-1100-padded": ("bf16", 2048, 10, 1100, None, 16, False),
    "width-2880-unaligned": ("bf16", 2880, 4, 2048, None, 16, False),
    "width-4608": ("bf16", 4608, 10, 2048, None, 16, True),
    "width-5120-tokens-255": ("bf16", 5120, 10, 255, None, 16, False),
    "width-5120-tokens-256": ("bf16", 5120, 10, 256, None, 16, True),
    "width-5120-tokens-300-padded": ("bf16", 5120, 10, 300, None, 16, False),
    "width-7168": ("bf16", 7168, 10, 2048, None, 16, True),
    "topk-1": ("bf16", 4096, 1, 2048, None, 16, False),
    "topk-1-large": ("bf16", 4096, 1, 4096, None, 16, False),
    "topk-2": ("bf16", 4096, 2, 2048, None, 16, True),
    "topk-3": ("bf16", 4096, 3, 2048, None, 16, True),
    "topk-4": ("bf16", 4096, 4, 2048, None, 16, True),
    "topk-8-dense-kernel": ("bf16", 4096, 8, 2048, None, 16, False),
    "topk-8-unaligned-rows": ("bf16", 4096, 8, 2000, None, 16, True),
    "topk-4-dense-kernel": ("bf16", 4096, 4, 4096, None, 16, False),
    "topk-2-dense-kernel": ("bf16", 4096, 2, 8192, None, 16, False),
    # BF16 rows pack in pairs, so 16 routes leave the dense kernel no output block.
    "topk-16-dense-fallback": ("bf16", 4096, 16, 1024, None, 16, True),
    "no-sc": ("bf16", 4096, 10, 2048, None, None, False),
    "sc-lanes-8": ("bf16", 4096, 10, 2048, None, 8, False),
    "no-tpu-info": ("bf16", 4096, 10, 2048, None, "no-tpu-info", False),
    "source-limit": ("bf16", 4096, 10, 2048, 1 << 20, 16, True),
    "source-over-limit": ("bf16", 4096, 10, 2048, (1 << 20) + 1, 16, False),
}


@unittest.skipUnless(_HAS_JAX, "JAX is required for the selection helper")
class TestDestinationMajorCombineSelection(CustomTestCase):
    def test_selection(self):
        dtypes = {"bf16": jnp.bfloat16, "f32": jnp.float32, "f16": jnp.float16}
        for name, (
            dtype,
            hidden,
            topk,
            tokens,
            rows,
            sc_lanes,
            selected,
        ) in _CASES.items():
            rows = tokens * topk if rows is None else rows
            # Abstract shapes exercise the 20-bit boundary without allocating
            # a multi-gigabyte route tensor.
            gmm2_out = jax.ShapeDtypeStruct((rows, hidden), dtypes[dtype])
            with mock.patch.object(pltpu, "get_tpu_info", _tpu_info(sc_lanes)):
                actual = _use_destination_major_combine(
                    gmm2_out=gmm2_out,
                    unsort_idx=jax.ShapeDtypeStruct((tokens * topk,), jnp.int32),
                    num_tokens=tokens,
                    topk=topk,
                )
            self.assertEqual(actual, selected, name)


if __name__ == "__main__":
    unittest.main()
