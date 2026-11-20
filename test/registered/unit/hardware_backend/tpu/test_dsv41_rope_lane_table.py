# SPDX-License-Identifier: Apache-2.0
"""The lane-packed DeepSeek-V4.1 RoPE table must give the plain table's results
bit for bit: the table itself, the CPU reference, the backend's table routing,
the Pallas rope / qnorm_rope / rope_quant kernels (TPU interpret mode) and the
CSA2 inverse-RoPE table, including odd ``max_pos`` and clipped positions."""

import contextlib
import importlib.util
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
from sglang_tpu.models.deepseek_v41.attention_backend import (
    DSV41TPUAttnBackend,
    rope_reference,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_HAS_JAX = importlib.util.find_spec("jax") is not None


def make_freqs(max_pos: int) -> torch.Tensor:
    """DeepSeek-V4-style complex64 freqs_cis ``[max_pos, 32]``."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return torch.polar(torch.ones_like(angles), angles)


def make_tables(max_pos: int):
    """(freqs, plain, lanes). The caches key on id(freqs); clearing them keeps a
    freed tensor's reused id from returning another test's table."""
    dsv4_memory_pool._COS_SIN_CACHE.clear()
    dsv4_memory_pool._ROPE_LANE_TABLE.clear()
    freqs = make_freqs(max_pos)
    return freqs, get_or_build_cos_sin_cache(freqs), get_or_build_rope_lane_table(freqs)


def edge_positions(max_pos: int, tokens: int, seed: int) -> np.ndarray:
    """Random in-range positions led by 0, 1 and the last two positions."""
    positions = np.random.default_rng(seed).integers(0, max_pos, tokens)
    positions[:4] = (0, 1, max_pos - 2, max_pos - 1)
    return positions.astype(np.int32)


def bits(x) -> np.ndarray:
    a = np.asarray(x)
    return a.view(np.dtype(f"u{a.dtype.itemsize}"))


class TestRopeLaneTable(CustomTestCase):
    def test_lane_table_rows_are_the_plain_rows(self):
        for max_pos in (4096, 4097):
            with self.subTest(max_pos=max_pos):
                freqs, plain, lanes = make_tables(max_pos)
                self.assertEqual(tuple(lanes.shape), ((max_pos + 1) // 2, 128))
                self.assertIsNone(lanes._base)
                flat = lanes.reshape(-1, 64)
                self.assertTrue(torch.equal(flat[:max_pos], plain))
                # The odd padding slot repeats the last position, for clipping.
                self.assertTrue(torch.equal(flat[-1], plain[-1]))
                self.assertIs(get_or_build_rope_lane_table(freqs), lanes)
                positions = torch.from_numpy(edge_positions(max_pos, 64, max_pos))
                self.assertTrue(
                    torch.equal(lookup_cos_sin(lanes, positions), plain[positions])
                )

    def test_rope_reference_on_lane_table(self):
        _, plain, lanes = make_tables(4097)
        positions = torch.from_numpy(edge_positions(4097, 32, 0))
        x = torch.randn(32, 4, 512, dtype=torch.bfloat16)
        for inverse in (False, True):
            old = rope_reference(x, positions, plain, inverse=inverse)
            new = rope_reference(x, positions, lanes, inverse=inverse)
            self.assertTrue(torch.equal(old, new))
            self.assertFalse(torch.equal(new, x))

    def test_plain_table_rejects_the_lane_width(self):
        """A plain 128-wide table would be read as lane-packed."""
        from sglang_tpu.dsv4.dsv4_memory_pool import get_or_build_cos_sin_cache

        freqs = torch.polar(torch.ones(8, 64), torch.zeros(8, 64))
        with self.assertRaises(ValueError):
            get_or_build_cos_sin_cache(freqs)

    def test_prewarm_gives_each_freqs_one_lane_table(self):
        """Layers sharing a freqs_cis share one table; another freqs_cis gets its own."""
        dsv4_memory_pool._COS_SIN_CACHE.clear()
        dsv4_memory_pool._ROPE_LANE_TABLE.clear()
        main, compressed = make_freqs(64), make_freqs(66)

        def layer(freqs, layer_id):
            module = torch.nn.Module()
            module.register_buffer("freqs_cis", freqs, persistent=False)
            module.attn_mqa = torch.nn.Module()
            module.attn_mqa.layer_id = layer_id
            return module

        layers = [layer(main, 0), layer(compressed, 1), layer(main, 2)]
        backend = SimpleNamespace(
            model_runner=SimpleNamespace(model=torch.nn.ModuleList(layers)),
            rope_owners={},
        )
        DSV41TPUAttnBackend._prewarm_model_caches(backend)
        tables = [m.freqs_cis for m in layers]
        self.assertEqual(tuple(tables[0].shape), (32, 128))
        self.assertEqual(tuple(tables[1].shape), (33, 128))
        self.assertIs(tables[0], tables[2])
        self.assertEqual(dsv4_memory_pool._COS_SIN_CACHE, {})
        self.assertEqual(backend.rope_owners, dict(enumerate(layers)))


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestRopeLaneTableKernels(CustomTestCase):
    # Odd: the last position sits in the even half of the padded last row.
    MAX_POS = 4097
    # Decode, DSpark verify, a partial tile and three 128-token tiles.
    TOKENS = (8, 48, 100, 384)

    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp

        _, plain, lanes = make_tables(cls.MAX_POS)
        cls.plain = jnp.asarray(plain.numpy())
        cls.lanes = jnp.asarray(lanes.numpy())

    def interpret(self):
        import jax
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        stack = contextlib.ExitStack()
        stack.enter_context(jax.default_device(jax.devices("cpu")[0]))
        stack.enter_context(use_abstract_mesh(mesh))
        stack.enter_context(pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()))
        return stack

    def assert_same(self, fn, x, positions):
        """fn on the plain and the lane table gives the same bits."""
        import jax
        import jax.numpy as jnp

        with self.interpret():
            # x is donated: each call gets its own copy.
            old = fn(jnp.copy(x), positions, self.plain)
            new = fn(jnp.copy(x), positions, self.lanes)
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_array_equal(bits(a), bits(b))
        return old

    def inputs(self, tokens, shape, seed):
        import jax.numpy as jnp

        x = np.random.default_rng(seed).standard_normal((tokens, *shape))
        positions = edge_positions(self.MAX_POS, tokens, seed)
        return jnp.asarray(x, jnp.bfloat16), jnp.asarray(positions)

    def test_rope(self):
        import functools

        from sglang_tpu.dsv4.rope import rope

        for tokens in self.TOKENS:
            for shape in ((512,), (4, 512)):
                x, positions = self.inputs(tokens, shape, tokens)
                for inverse in (False, True):
                    with self.subTest(tokens=tokens, shape=shape, inverse=inverse):
                        out = self.assert_same(
                            functools.partial(rope, inverse=inverse), x, positions
                        )
                        self.assertFalse(np.array_equal(bits(out), bits(x)))

    def test_rope_quant_and_qnorm_rope(self):
        from sglang_tpu.dsv4.rope import qnorm_rope, rope_quant

        for tokens in self.TOKENS:
            with self.subTest(tokens=tokens):
                x, positions = self.inputs(tokens, (4, 128), tokens)
                self.assert_same(rope_quant, x, positions)
                x, positions = self.inputs(tokens, (4, 512), tokens + 1)
                self.assert_same(qnorm_rope, x, positions)

    def test_rows_past_the_table_end_clamp_to_the_last_position(self):
        """An out-of-range jnp.take fills NaN; lane rows clamp instead."""
        import jax.numpy as jnp

        from sglang_tpu.dsv4.rope import cos_sin_rows

        last = self.MAX_POS - 1
        positions = jnp.asarray([-1, 0, last, self.MAX_POS, self.MAX_POS + 9])
        rows = np.asarray(cos_sin_rows(self.lanes, positions))
        self.assertTrue(np.isfinite(rows).all())
        np.testing.assert_array_equal(bits(rows[0]), bits(self.plain[0]))
        for row in rows[2:]:
            np.testing.assert_array_equal(bits(row), bits(self.plain[last]))

    def test_csa2_inverse_rope_clips_like_the_plain_table(self):
        """Positions past either end clip to the first / last position."""
        import jax.numpy as jnp

        from sglang_tpu.models.deepseek_v41.ops.csa2 import _inverse_rope

        positions = edge_positions(self.MAX_POS, 64, 0)
        positions[4:8] = (-1, self.MAX_POS, self.MAX_POS + 1, self.MAX_POS + 9)
        positions = jnp.asarray(positions)
        np.testing.assert_array_equal(
            bits(_inverse_rope(self.plain, positions)),
            bits(_inverse_rope(self.lanes, positions)),
        )


if __name__ == "__main__":
    unittest.main()
