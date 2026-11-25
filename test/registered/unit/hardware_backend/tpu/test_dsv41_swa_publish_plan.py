# SPDX-License-Identifier: Apache-2.0
"""The SWA publish plan (paged_swa.swa_publish_plan / write_swa_pages(plan=)) must
hand the SparseCore writer exactly the inputs the per-call native prepare does:
built from one layer's records, it serves every layer of the forward."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.kernels import paged_swa

PAGE = 64
LANES_CORES = 256


def _reference(records, values, base, capacity, page_size, lanes, cores):
    """The writer inputs as write_swa_pages built them before the plan split."""
    valid = (records >= max(base, page_size)) & (records < base + capacity)
    local = jnp.where(valid, records - base, capacity)
    order = jnp.argsort(local, stable=True)
    ids = local[order]
    next_ids = jnp.concatenate((ids[1:], jnp.array([capacity], jnp.int32)))
    # TC BF16 tiles pack adjacent token rows into one physical word row.
    pairs = ids // 2
    boundary = (ids < capacity) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
    )
    even = ids % 2 == 0
    has_odd = (~even) | (next_ids // 2 == pairs)
    bits = jax.lax.bitcast_convert_type(values[order], jnp.uint16).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
    low = jnp.where(even[:, None], bits, jnp.uint32(0))
    high = jnp.where(even[:, None], next_bits, bits)
    high = jnp.where(has_odd[:, None], high, jnp.uint32(0))
    words = jax.lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        has_odd, jnp.int32(-65536), jnp.int32(0)
    )
    padding = -records.size % (lanes * cores)
    destinations = jnp.pad(
        jnp.where(boundary, pairs, -1), (0, padding), constant_values=-1
    )
    # SC cannot reshape a lane vector for a column broadcast in the kernel.
    masks = jnp.broadcast_to(
        jnp.pad(masks, (0, padding))[:, None], (records.size + padding, 512)
    )
    words = jnp.pad(words, ((0, padding), (0, 0)))
    return destinations, masks, words


def _records(rng, n, capacity):
    # Unique valid records, null-page and out-of-range rows, and -1 padding.
    valid = rng.choice(np.arange(PAGE, capacity), size=n - 6, replace=False)
    extra = np.array([-1, -1, 3, 0, capacity + 5, capacity * 2])
    out = np.concatenate([valid, extra]).astype(np.int32)
    rng.shuffle(out)
    return jnp.asarray(out)


def _values(rng, n):
    bits = rng.integers(0, 2**16, size=(n, 512), dtype=np.uint16)
    return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)


class TestSwaPublishPlan(CustomTestCase):
    def test_plan_reused_across_layers_matches_the_native_prepare(self):
        rng = np.random.default_rng(0)
        for n, pages, base in ((48, 40, 0), (262, 300, 0), (1000, 200, 200 * PAGE)):
            capacity = pages * PAGE
            records = (
                _records(rng, n, base + capacity)
                if base == 0
                else (_records(rng, n, capacity) + base)
            )
            plan = paged_swa._plan_bank(
                records, base=base, capacity=capacity, page_size=PAGE
            )
            for _layer in range(3):
                values = _values(rng, n)
                want = _reference(
                    records, values, base, capacity, PAGE, LANES_CORES // 16, 16
                )
                got = paged_swa._planned_prepare(plan, values, lanes_cores=LANES_CORES)
                native = paged_swa._native_prepare(
                    records,
                    values,
                    base=base,
                    capacity=capacity,
                    page_size=PAGE,
                    lanes_cores=LANES_CORES,
                )
                for a, b, c in zip(want, got, native, strict=True):
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertTrue(bool(jnp.array_equal(a, b)))
                    self.assertTrue(bool(jnp.array_equal(a, c)))


if __name__ == "__main__":
    unittest.main()
