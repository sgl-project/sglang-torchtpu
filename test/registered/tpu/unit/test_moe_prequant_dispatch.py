"""Pre-quantized MoE dispatch: bit-equality checks.

* ``sparse_core.row_gather(x, idx) == x[idx]`` (pure-DMA SparseCore row gather).
* ``gmm_v2(prequantize_lhs_fp8(x), lhs_scale=...)`` equals ``gmm_v2(x)`` with its
  in-kernel activation quantization, bit for bit (fused and unfused activation).
* ``_fused_moe_jax`` with pre-quantized dispatch equals the bf16-gather fallback,
  including dropped (-1) slots, zero router weights and padded token counts.

Run both scale layouts (the flag is read when gmm_v2 is imported):
    SGLANG_PLATFORM=tpu SGLANG_TPU_FP8_MOE_FP32_SCALES=1 python \\
      test/registered/tpu/unit/test_moe_prequant_dispatch.py
    SGLANG_PLATFORM=tpu SGLANG_TPU_FP8_MOE_FP32_SCALES=0 python ...
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

# Platform discovery defaults JAX_PLATFORMS to cpu; select TPU before importing jax.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU backend visible to jax.devices()"


def _bits_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: np.uint16, 4: np.uint32, 1: np.uint8}[a.dtype.itemsize]
    return bool((a.view(view) == b.view(view)).all())


def _activations(key, m, k):
    """bf16 rows with mixed magnitudes, all-zero rows (zero scale), and zero blocks."""
    k1, k2 = jax.random.split(key)
    mag = jnp.exp2(jax.random.randint(k1, (m, 1), -6, 6).astype(jnp.float32))
    x = jax.random.normal(k2, (m, k), jnp.float32) * mag
    x = x.at[::7].set(0.0)
    x = x.at[1::11, :512].set(0.0)
    return x.astype(jnp.bfloat16)


def _fp8_weights(key, g, k, n):
    k1, k2 = jax.random.split(key)
    w = (jax.random.normal(k1, (g, k, n), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    s = jax.random.uniform(k2, (g, 1, 1, n), minval=1e-3, maxval=1e-2)
    return w, s


def _group_sizes(sizes):
    return jnp.asarray(sizes, jnp.int32)


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestRowGather(unittest.TestCase):
    def test_matches_take(self):
        from sglang_tpu.moe.pallas.sparse_core.row_gather import row_gather

        key = jax.random.PRNGKey(0)
        for rows, width, n, dtype in [
            (300, 128, 16, jnp.uint32),
            (300, 128, 1000, jnp.uint32),  # not a multiple of 16: padded + sliced
            (513, 1536, 4096, jnp.uint32),
            (8192, 1536, 65536, jnp.uint32),
            (1024, 384, 2080, jnp.float32),
        ]:
            with self.subTest(rows=rows, width=width, n=n, dtype=dtype):
                k1, k2, key = jax.random.split(key, 3)
                x = jax.random.bits(k1, (rows, width), jnp.uint32)
                if dtype == jnp.float32:
                    x = jax.lax.bitcast_convert_type(x, jnp.float32)
                idx = jax.random.randint(k2, (n,), 0, rows, jnp.int32)
                got = jax.jit(row_gather)(x, idx)
                self.assertTrue(_bits_equal(got, x[idx]))

    def test_gather_end_matches_prefix(self):
        from sglang_tpu.moe.pallas.sparse_core.row_gather import row_gather

        key = jax.random.PRNGKey(9)
        k1, k2 = jax.random.split(key)
        x = jax.random.bits(k1, (1024, 256), jnp.uint32)
        idx = jax.random.randint(k2, (8192,), 0, 1024, jnp.int32)
        fn = jax.jit(lambda a, i, e: row_gather(a, i, gather_end=e))
        for end in (0, 17, 1024, 4097, 8192):
            with self.subTest(end=end):
                got = fn(x, idx, jnp.asarray([end], jnp.int32))
                self.assertTrue(_bits_equal(got[:end], x[idx[:end]]))


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestPrequantGmm(unittest.TestCase):
    def _check(self, k, n, sizes, fuse_act, seed):
        from sglang_tpu.moe.pallas import gmm_v2 as g

        key = jax.random.PRNGKey(seed)
        g_n = len(sizes)
        m = int(sum(sizes))
        assert m % 16 == 0, "gmm_v2 rows are a multiple of the bf16 sublane tiling"
        kx, kw = jax.random.split(key)
        x = _activations(kx, m, k)
        w, s = _fp8_weights(kw, g_n, k, n)
        gs = _group_sizes(sizes)
        tiles = g.TileSizes(tile_m=256, tile_k=k, tile_n=n // 2 if fuse_act else n)
        common = dict(
            group_sizes=gs,
            rhs_scale=s,
            zero_initialize=False,
            preferred_element_type=jnp.bfloat16,
            tile_info=tiles,
            fuse_act=fuse_act,
            maybe_quantize_lhs=True,
        )
        ref = jax.jit(lambda x_: g.gmm_v2(lhs=x_, rhs=w, **common))(x)
        q, qs = g.prequantize_lhs_fp8(x)
        got = jax.jit(lambda q_, s_: g.gmm_v2(lhs=q_, rhs=w, lhs_scale=s_, **common))(
            q, qs
        )
        # Rows past the groups are never written; compare the grouped rows.
        self.assertTrue(_bits_equal(got[:m], ref[:m]))

    def test_fused_silu(self):
        self._check(5120, 256, [0, 37, 256, 300, 1, 0, 512, 46], "silu", 1)

    def test_fused_clamped_silu(self):
        self._check(
            4096, 256, [16, 0, 200, 300, 1, 0, 495, 12], "silu_and_mul_clamp:7.0", 6
        )

    def test_unfused(self):
        self._check(1024, 512, [16, 0, 400, 3, 97, 0, 0, 508], None, 2)

    def test_prequant_supported(self):
        from sglang_tpu.moe.pallas import gmm_v2 as g

        w, s = _fp8_weights(jax.random.PRNGKey(3), 4, 5120, 256)
        self.assertTrue(g.prequant_supported(2048, w, s, 4, fuse_act="silu"))
        w_bf16 = w.astype(jnp.bfloat16)
        self.assertFalse(g.prequant_supported(2048, w_bf16, None, 4, fuse_act="silu"))
        w_odd, s_odd = _fp8_weights(jax.random.PRNGKey(4), 4, 640, 256)
        self.assertFalse(g.prequant_supported(2048, w_odd, s_odd, 4, fuse_act="silu"))


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoePrequantDispatch(unittest.TestCase):
    E, H, TWO_N, TOPK = 16, 1024, 256, 8

    def _run(self, prequant, x, w1, s1, w2, s2, wts, ids, *, global_e=0, offset=0):
        from sglang_tpu.envs import envs
        from sglang_tpu.moe.pallas import fused_moe_op

        body = functools.partial(
            fused_moe_op._fused_moe_jax,
            topk=self.TOPK,
            activation_name="silu",
            use_sc_gather=1,
            permute_dense_threshold=0,
            global_num_experts=global_e,
            expert_group_offset=offset,
        )

        def moe(x_, w1_, w2_, wt_, id_, s1_, s2_):
            return body(x_, w1_, w2_, wt_, id_, s1_, s2_, None, None)

        # Prequant eligibility is evaluated while tracing: give each mode its own
        # function and keep it alive so JAX cannot reuse a freed function's cache entry.
        moe.__name__ = f"moe_prequant_{prequant}_ge{global_e}"
        fn = jax.jit(moe)
        self._keep_alive.append(fn)
        gate_ctx = (
            contextlib.nullcontext()
            if prequant
            else mock.patch.object(
                fused_moe_op, "_prequant_dispatch_ok", return_value=False
            )
        )
        with envs.SGLANG_TPU_FUSED_MOE_EP_COUNTING_ROUTE.override(True), gate_ctx:
            hlo = fn.lower(x, w1, w2, wts, ids, s1, s2).as_text()
            out = fn(x, w1, w2, wts, ids, s1, s2)
        return out, hlo

    def setUp(self):
        self._keep_alive = []

    def test_on_equals_off(self):
        key = jax.random.PRNGKey(5)
        kw1, kw2, kr = jax.random.split(key, 3)
        w1, s1 = _fp8_weights(kw1, self.E, self.H, self.TWO_N)
        w2, s2 = _fp8_weights(kw2, self.E, self.TWO_N // 2, self.H)
        for m in (256, 1000, 2048):
            with self.subTest(m=m):
                kx, ki, kt, kr = jax.random.split(kr, 4)
                x = _activations(kx, m, self.H)
                g = jax.random.gumbel(ki, (m, self.E))
                ids = jax.lax.top_k(g, self.TOPK)[1].astype(jnp.int32)
                ids = ids.at[::13].set(-1)  # dropped (padding) tokens
                wts = jax.nn.softmax(jax.random.normal(kt, (m, self.TOPK)), -1)
                wts = wts.at[::5, 3].set(0.0).astype(jnp.bfloat16)
                on, hlo_on = self._run(True, x, w1, s1, w2, s2, wts, ids)
                off, hlo_off = self._run(False, x, w1, s1, w2, s2, wts, ids)
                self.assertIn("prequantize_lhs_fp8", hlo_on)
                self.assertIn("sc_row_gather", hlo_on)
                self.assertNotIn("prequantize_lhs_fp8", hlo_off)
                self.assertTrue(_bits_equal(on, off))

    def test_ep_counting_route_on_equals_off(self):
        # EP=4 (topk/ep = 2 >= 2) exercises ep_route + dynamic gather_end.
        key = jax.random.PRNGKey(13)
        kw1, kw2, kx, ki, kt = jax.random.split(key, 5)
        global_e, offset, m = self.E * 4, self.E, 512
        w1, s1 = _fp8_weights(kw1, self.E, self.H, self.TWO_N)
        w2, s2 = _fp8_weights(kw2, self.E, self.TWO_N // 2, self.H)
        x = _activations(kx, m, self.H)
        ids = jax.lax.top_k(jax.random.gumbel(ki, (m, global_e)), self.TOPK)[1].astype(
            jnp.int32
        )
        wts = jax.nn.softmax(jax.random.normal(kt, (m, self.TOPK)), -1).astype(
            jnp.bfloat16
        )
        on, hlo_on = self._run(
            True, x, w1, s1, w2, s2, wts, ids, global_e=global_e, offset=offset
        )
        off, hlo_off = self._run(
            False, x, w1, s1, w2, s2, wts, ids, global_e=global_e, offset=offset
        )
        self.assertIn("prequantize_lhs_fp8", hlo_on)
        self.assertIn("sc_row_gather", hlo_on)
        self.assertNotIn("prequantize_lhs_fp8", hlo_off)
        self.assertTrue(_bits_equal(on, off))

    def test_gate_skips_small_batches_and_low_fanout(self):
        key = jax.random.PRNGKey(19)
        kw1, kw2, kr = jax.random.split(key, 3)
        w1, s1 = _fp8_weights(kw1, self.E, self.H, self.TWO_N)
        w2, s2 = _fp8_weights(kw2, self.E, self.TWO_N // 2, self.H)
        for label, m, global_e, offset in (
            ("small_m", 64, 0, 0),
            ("ep8_low_fanout", 2048, self.E * 8, self.E),
        ):
            with self.subTest(case=label):
                kx, ki, kt, kr = jax.random.split(kr, 4)
                x = _activations(kx, m, self.H)
                route_e = global_e or self.E
                ids = jax.lax.top_k(jax.random.gumbel(ki, (m, route_e)), self.TOPK)[
                    1
                ].astype(jnp.int32)
                wts = jax.nn.softmax(jax.random.normal(kt, (m, self.TOPK)), -1).astype(
                    jnp.bfloat16
                )
                _, hlo = self._run(
                    True, x, w1, s1, w2, s2, wts, ids, global_e=global_e, offset=offset
                )
                self.assertNotIn("prequantize_lhs_fp8", hlo)


if __name__ == "__main__":
    unittest.main()
