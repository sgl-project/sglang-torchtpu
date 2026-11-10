"""CPU checks that the lane-packed DSV4 RoPE table is bit-exact with the plain one.

torch_tpu stores the plain [1M, 64] float32 cos/sin table dim-0-minor, so every
Pallas RoPE call that took it whole relayouted all 256 MB first (~6 copies per
decode step). The served table now packs two positions per 128-lane row
and the kernels receive only the batch's rows. Here the new path must equal the
old one bit for bit:

* the lane table and its row lookup against the plain table (odd max_pos too);
* the Pallas rope (forward / inverse), rope_quant and qnorm_rope kernels, run in
  TPU interpret mode, at decode, verify (K + 1 = 6 rows per request), draft
  (K = 5 rows per request) and prefill token counts;
* the CSA2 inverse-RoPE table (dsv41 custom_ops).

The copy this removes is inserted by the TPU compiler's layout assignment, so it
is checked on TPU: an XProf of decode shows no %copy.*.cos_sin_cache_hbm_ref.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsv4 import dsv4_memory_pool
from sglang_tpu.dsv4.dsv4_memory_pool import (
    get_or_build_cos_sin_cache,
    get_or_build_rope_lane_table,
    lookup_cos_sin,
)

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

K = 5
DECODE, VERIFY, DRAFT, PREFILL = (8, 16), (6, 48), (5, 40), (1024,)


def make_freqs(max_pos: int) -> torch.Tensor:
    """DSV4-style complex64 freqs_cis [max_pos, 32] (non-trivial angles)."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return torch.polar(torch.ones_like(angles), angles)


def make_tables(max_pos: int):
    """(freqs, plain, lanes). The caches key on id(freqs): clear them so a
    freed earlier tensor's id cannot hand back another test's table."""
    dsv4_memory_pool._COS_SIN_CACHE.clear()
    dsv4_memory_pool._ROPE_LANE_TABLE.clear()
    freqs = make_freqs(max_pos)
    plain = get_or_build_cos_sin_cache(freqs)
    return freqs, plain, get_or_build_rope_lane_table(freqs)


def bits(x) -> np.ndarray:
    a = np.asarray(x)
    return a.view(np.dtype(f"u{a.dtype.itemsize}"))


class TestRopeLaneTable(CustomTestCase):
    def test_lane_table_is_the_plain_table_reshaped(self):
        for max_pos in (4096, 4097):
            freqs, plain, lanes = make_tables(max_pos)
            self.assertEqual(tuple(lanes.shape), ((max_pos + 1) // 2, 128))
            self.assertEqual(lanes.dtype, torch.float32)
            self.assertIsNone(lanes._base)  # own storage, not a view graph input
            flat = lanes.reshape(-1, 64)
            self.assertTrue(torch.equal(flat[:max_pos], plain))
            self.assertIs(get_or_build_rope_lane_table(freqs), lanes)
            positions = torch.tensor([0, 1, 2, 3, 4095, max_pos - 1, 777, 1024])
            self.assertTrue(
                torch.equal(lookup_cos_sin(lanes, positions), plain[positions])
            )
            self.assertTrue(
                torch.equal(lookup_cos_sin(plain, positions), plain[positions])
            )


class TestRopeLaneTableKernels(CustomTestCase):
    MAX_POS = 1 << 14

    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp

        _, plain, lanes = make_tables(cls.MAX_POS)
        cls.plain = jnp.asarray(plain.numpy())
        cls.lanes = jnp.asarray(lanes.numpy())

    def interpret(self):
        """Run every pallas_call in TPU interpret mode on the host."""
        import contextlib

        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu import pallas_export
        from sglang_tpu.dsv4 import rope

        pallas_call = pl.pallas_call

        def interpreted(*args, **kwargs):
            return pallas_call(*args, interpret=pltpu.InterpretParams(), **kwargs)

        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(rope.pl, "pallas_call", interpreted))
        stack.enter_context(
            patch.object(
                pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
            )
        )
        stack.enter_context(pallas_export.tpu_info_for_cpu_export())
        return stack

    def assert_same(self, fn, x, positions):
        """fn(x, positions, table) on the plain and the lane table, bitwise."""
        import jax
        import jax.numpy as jnp

        with self.interpret():
            # x is donated: hand each call its own copy.
            old = fn(jnp.copy(x), positions, self.plain)
            new = fn(jnp.copy(x), positions, self.lanes)
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_array_equal(bits(a), bits(b))
        return old

    def inputs(self, tokens, shape, seed):
        import jax.numpy as jnp

        rng = np.random.default_rng(seed)
        positions = rng.integers(0, self.MAX_POS, tokens).astype(np.int32)
        positions[:2] = (0, self.MAX_POS - 1)  # first/last row, even/odd lanes
        x = jnp.asarray(rng.standard_normal((tokens, *shape)), jnp.bfloat16)
        return x, jnp.asarray(positions)

    def test_rope_forward_inverse(self):
        import functools

        from sglang_tpu.dsv4.rope import rope

        for tokens in DECODE + VERIFY + DRAFT + PREFILL:
            for shape in ((512,), (4, 512)):
                x, positions = self.inputs(tokens, shape, tokens)
                for inverse in (False, True):
                    with self.subTest(tokens=tokens, shape=shape, inverse=inverse):
                        out = self.assert_same(
                            functools.partial(rope, inverse=inverse), x, positions
                        )
                        # The rotation happened (not a no-op on both paths).
                        self.assertFalse(np.array_equal(bits(out), bits(x)))

    def test_rope_quant_and_qnorm_rope(self):
        from sglang_tpu.dsv4.rope import qnorm_rope, rope_quant

        for tokens in DECODE + VERIFY + DRAFT + PREFILL:
            with self.subTest(tokens=tokens):
                x, positions = self.inputs(tokens, (4, 128), tokens)
                self.assert_same(rope_quant, x, positions)
                x, positions = self.inputs(tokens, (4, 512), tokens + 1)
                self.assert_same(qnorm_rope, x, positions)

    def test_csa2_inverse_rope_table(self):
        from sglang_tpu.dsv41.custom_ops import inverse_rope

        for tokens in DECODE + VERIFY + DRAFT + PREFILL:
            _, positions = self.inputs(tokens, (1,), tokens)
            np.testing.assert_array_equal(
                bits(inverse_rope(self.plain, positions)),
                bits(inverse_rope(self.lanes, positions)),
            )


if __name__ == "__main__":
    unittest.main()
