"""SGLANG_TPU_MUSE_KERNEL_A1_PARTS / A1_AG_UNDER_D give the same A1 inputs and outputs.

Runs the Pallas kernels in TPU interpret mode on one JAX CPU device (a 1-rank "model"
mesh, so the all-gathers are identities and only the row bookkeeping is under test):

* Kernel D run as two row ranges (the AG_UNDER_D split, second call writing new_stream in
  place) gives the full call's new_stream, and its gathered first rows plus the remaining
  rows are the full call's next_attn_in, bit for bit. That is the next layer's A1 input.
* A1 (v5x4 two-half body) with uneven parts 2,3,2,1, and with part 0 handed in already
  gathered (the AG_UNDER_D op), gives the equal-parts outputs bit for bit.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

import contextlib
import functools
import unittest
from unittest import mock

import numpy as np

from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _interpret_on_cpu():
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    real = pl.pallas_call

    def interpret(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams()
        return real(*args, **kwargs)

    with mock.patch.object(pl, "pallas_call", interpret):
        with jax.default_device(jax.devices("cpu")[0]):
            yield


def _one_rank_shard_map(body, n_in: int, n_out: int):
    import jax
    from jax.sharding import PartitionSpec as P

    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("model",))
    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("model"),) * n_in,
            out_specs=(P("model"),) * n_out,
            check_vma=False,
        )
    )


def _assert_bits_equal(test, a, b):
    a, b = np.asarray(a), np.asarray(b)
    test.assertEqual(a.shape, b.shape)
    test.assertEqual(a.dtype, b.dtype)
    test.assertTrue(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


class TestMuseA1AgUnderD(CustomTestCase):
    def test_kernel_d_row_split_gives_full_next_attn_in(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_d import (
            pallas_combine_unpool_residual_nextnorm_4slab,
        )
        from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
            _d_with_next_in_gather,
        )

        t_local, topk, rows0, n_src = 192, 8, 64, 512
        keys = jax.random.split(jax.random.key(0), 8)
        args = (
            jax.random.normal(keys[0], (n_src, 32, 128), jnp.float32).astype(
                jnp.bfloat16
            ),
            jax.random.randint(keys[1], (t_local, topk), 0, n_src, jnp.int32),
            jax.random.uniform(keys[2], (t_local, topk), jnp.float32).astype(
                jnp.bfloat16
            ),
            jax.random.normal(keys[3], (t_local, 8192), jnp.float32),
            (jax.random.normal(keys[4], (4096, 8192), jnp.float32) * 0.1).astype(
                jnp.float8_e4m3fn
            ),
            jnp.full((8192,), 0.02, jnp.float32),
            jax.random.normal(keys[5], (8192,), jnp.float32) * 0.1,
            jnp.full((8192,), 0.9, jnp.float32),
            jnp.full((8192,), 1.1, jnp.float32),
            jax.random.normal(keys[6], (8192,), jnp.float32) * 0.1,
        )
        d_fn = functools.partial(
            pallas_combine_unpool_residual_nextnorm_4slab, tok_tile=64
        )
        # Per-rank blocks of the replicated inputs: add the leading mesh dim.
        stacked = tuple(a[None] for a in args)

        def full(*xs):
            stream, next_in = d_fn(*(x[0] for x in xs))
            return stream[None], next_in[None]

        def split(*xs):
            stream, rest, gathered = _d_with_next_in_gather(
                d_fn=d_fn,
                d_args=tuple(x[0] for x in xs),
                rows0=rows0,
                split=True,
                axis_name="model",
            )
            return stream[None], rest[None], gathered[None]

        with _interpret_on_cpu():
            ref_stream, ref_next = _one_rank_shard_map(full, 10, 2)(*stacked)
            stream, rest, gathered = _one_rank_shard_map(split, 10, 3)(*stacked)
        _assert_bits_equal(self, stream, ref_stream)
        self.assertEqual(gathered.shape, (1, 1, rows0, 8192))
        _assert_bits_equal(
            self, jnp.concatenate([gathered[0, 0], rest[0]]), ref_next[0]
        )

    def test_native_single_kv_head_matches_duplicated_heads(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1, reference

        # A TP16 shard stores one K and one V in the same 128 lanes. In the
        # two-head control K and V occupy separate tiles.
        t, hidden, q_size = 256, 256, 512
        keys = jax.random.split(jax.random.key(721), 4)
        x = jax.random.normal(keys[0], (t, hidden)).astype(jnp.bfloat16)
        w = jax.random.normal(keys[1], (hidden, 2 * q_size + 128)).astype(
            jnp.float8_e4m3fn
        )
        scales = jax.random.uniform(keys[2], (1, w.shape[1]), minval=0.1, maxval=0.5)
        theta = jax.random.normal(keys[3], (t, 32))
        _, cos, sin = reference.build_batch_cos_sin_128(
            jnp.arange(t), jnp.cos(theta), jnp.sin(theta)
        )

        def duplicate_kv(x):
            qg, k, v = jnp.split(x, (2 * q_size, 2 * q_size + 64), axis=-1)
            return jnp.concatenate((qg, k, k, v, v), axis=-1)

        def run(w, scales, heads):
            return kernel_a1.pallas_qkvg_qk_rope_kv(
                x, w, scales, cos, sin, 8, heads, k_scale=0.7, v_scale=1.3, bk=hidden
            )

        with _interpret_on_cpu():
            q, gate, kv = run(w, scales, 1)
            q2, gate2, kv2 = run(duplicate_kv(w), duplicate_kv(scales), 2)
        _assert_bits_equal(self, q, q2)
        _assert_bits_equal(self, gate, gate2)
        _assert_bits_equal(self, kv[:, :64], kv2[:, :64])
        _assert_bits_equal(self, kv[:, 64:], kv2[:, 128:192])

    def test_a1_uneven_parts_and_gathered_part0_match_equal_parts(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.muse_prefill_kernels.integration import (
            _a1_body,
            _a1_first_rows,
        )

        h_q, h_kv, k_dim, t_shard, parts = 4, 2, 256, 1024, (2, 3, 2, 1)
        n_dim = 2 * h_q * 64 + 2 * h_kv * 64
        keys = jax.random.split(jax.random.key(1), 4)
        x = jax.random.normal(keys[0], (t_shard, k_dim), jnp.float32).astype(
            jnp.bfloat16
        )
        w = (jax.random.normal(keys[1], (k_dim, n_dim), jnp.float32) * 0.1).astype(
            jnp.float8_e4m3fn
        )
        w_scale = jnp.full((1, n_dim), 0.05, jnp.float32)
        angle = jax.random.uniform(keys[2], (t_shard, 128), jnp.float32) * 3.0
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        rows0 = _a1_first_rows(t_shard, variant="v5x4", parts=parts)
        self.assertEqual(rows0, 256)

        def run(**kw):
            body = _a1_body(h_q, h_kv, "v5x4", **kw)
            return _one_rank_shard_map(body, 5, 3)(x, w, w_scale, cos, sin)

        with _interpret_on_cpu():
            ref = run()
            uneven = run(parts=parts)
            body = _a1_body(h_q, h_kv, "v5x4", parts=parts, ag_under_d=True)
            # [tp, rows0, H] per rank, as kernel D's tiled=False all-gather emits it.
            ag0 = x[None, :rows0]
            agd = _one_rank_shard_map(body, 6, 3)(ag0, x[rows0:], w, w_scale, cos, sin)
        for r, u, a in zip(ref, uneven, agd):
            _assert_bits_equal(self, u, r)
            _assert_bits_equal(self, a, r)


if __name__ == "__main__":
    unittest.main()
