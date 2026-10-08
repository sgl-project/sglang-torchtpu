"""Single-token (bs=1) fused-MoE route-meta path vs the sort/gather path."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

import torch  # noqa: F401

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a JAX TPU backend"

# Kimi-K3 per-chip shape, scaled down: top-16 over 28 local experts per rank.
HIDDEN, INTER, LOCAL, EP, TOPK = 256, 128, 28, 4, 16
GLOBAL = LOCAL * EP
DENSE_THRESHOLD = 512  # Kimi-K3 recipe value


def _inputs(m, seed):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((m, HIDDEN)) * 0.3, jnp.bfloat16)
    w1 = jnp.asarray(
        rng.standard_normal((GLOBAL, HIDDEN, 2 * INTER)) * 0.05, jnp.bfloat16
    )
    w2 = jnp.asarray(rng.standard_normal((GLOBAL, INTER, HIDDEN)) * 0.05, jnp.bfloat16)
    ids = np.stack([rng.choice(GLOBAL, TOPK, replace=False) for _ in range(m)])
    w = rng.random((m, TOPK)).astype(np.float32)
    w = jnp.asarray(w / w.sum(-1, keepdims=True), jnp.bfloat16)
    return x, w1, w2, w, jnp.asarray(ids, jnp.int32)


def _moe(
    x,
    w1,
    w2,
    w,
    ids,
    *,
    rank=None,
    threshold,
    no_combine=False,
    w1_scale=None,
    w2_scale=None,
    activation="silu",
    num_global=GLOBAL,
):
    from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax

    kwargs = {}
    if rank is not None:
        local = slice(rank * LOCAL, (rank + 1) * LOCAL)
        w1, w2 = w1[local], w2[local]
        if w1_scale is not None:
            w1_scale, w2_scale = w1_scale[local], w2_scale[local]
        kwargs = dict(global_num_experts=num_global, expert_group_offset=local.start)
    out = _fused_moe_jax(
        x,
        w1,
        w2,
        w,
        ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        topk=TOPK,
        activation_name=activation,
        no_combine=no_combine,
        use_sc_gather=0,
        permute_dense_threshold=threshold,
        **kwargs,
    )
    return np.asarray(jax.block_until_ready(out).astype(jnp.float32))


def _mxfp4_weights(num_experts, hidden, inter, seed):
    """Random HF-layout MXFP4 experts converted to the native packed layout."""
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        process_mxfp4_moe_weights_native,
    )

    g = torch.Generator().manual_seed(seed)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    # Scales up to 2^3 drive gate/up past SiTU's beta and linear_beta.
    e8 = lambda *s: torch.randint(124, 131, s, dtype=torch.uint8, generator=g)
    w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
        u8(num_experts, 2 * inter, hidden // 2),
        e8(num_experts, 2 * inter, hidden // 32),
        torch.zeros(num_experts, 2 * inter, dtype=torch.bfloat16),
        u8(num_experts, hidden, inter // 2),
        e8(num_experts, hidden, inter // 32),
        torch.zeros(num_experts, hidden, dtype=torch.bfloat16),
        w13_interleaved=False,
    )
    return [jnp.asarray(t.numpy()) for t in (w13, w2, w13_s, w2_s)]


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestSingleTokenRouteMetaMxfp4Situ(CustomTestCase):
    """Kimi-K3 expert config: packed MXFP4 W4A16, SiTU(4, 25), top-16."""

    HIDDEN, INTER, EP = 1024, 768, 2
    ACT = "situ:4.0:25.0"

    def _compare(self, *, rank, num_global, seed):
        from sglang.srt.layers.moe.pallas import route_meta

        w1, w2, w1_s, w2_s = _mxfp4_weights(num_global, self.HIDDEN, self.INTER, 0)
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.standard_normal((1, self.HIDDEN)), jnp.bfloat16)
        ids = rng.choice(num_global, TOPK, replace=False)[None]
        w = rng.random((1, TOPK)).astype(np.float32)
        w = jnp.asarray(w / w.sum(-1, keepdims=True), jnp.bfloat16)
        if rank is not None:
            local = (ids >= rank * LOCAL) & (ids < (rank + 1) * LOCAL)
            self.assertTrue(local.any() and not local.all())
        run = lambda threshold: _moe(
            x,
            w1,
            w2,
            w,
            jnp.asarray(ids, jnp.int32),
            rank=rank,
            threshold=threshold,
            w1_scale=w1_s,
            w2_scale=w2_s,
            activation=self.ACT,
            num_global=num_global,
        )
        with mock.patch.object(
            route_meta,
            "single_token_route_meta",
            wraps=route_meta.single_token_route_meta,
        ) as spy:
            out = run(DENSE_THRESHOLD)
        self.assertTrue(spy.called)
        ref = run(0)  # argsort + gather path
        self.assertTrue(np.all(np.isfinite(out)))
        scale = max(float(np.max(np.abs(ref))), 1e-6)
        self.assertLess(float(np.max(np.abs(out - ref))) / scale, 1e-2)

    def test_bs1_all_local(self):
        self._compare(rank=None, num_global=LOCAL, seed=1)

    def test_bs1_expert_parallel_mixed_local_ids(self):
        # Seed picks ids split across both ranks; check both windows.
        for rank in range(self.EP):
            with self.subTest(rank=rank):
                self._compare(rank=rank, num_global=LOCAL * self.EP, seed=2)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestSingleTokenRouteMeta(CustomTestCase):
    def test_kernel_matches_stable_sort(self):
        from sglang.srt.layers.moe.pallas.route_meta import single_token_route_meta

        rng = np.random.default_rng(0)
        n = TOPK
        for trial in range(10):
            if trial == 0:  # repeated ids: stability within a group
                ids = rng.integers(0, 4, TOPK)
            elif trial == 1:  # padded token: TopK marks every slot -1
                ids = np.full(TOPK, -1)
            elif trial == 2:  # some -1 and out-of-range slots
                ids = rng.choice(GLOBAL, TOPK, replace=False)
                ids[::3] = -1
                ids[1] = GLOBAL + 3
            else:
                ids = rng.choice(GLOBAL, TOPK, replace=False)
            w = rng.random(TOPK).astype(np.float32)
            for offset, num_experts in ((0, GLOBAL), (LOCAL, LOCAL), (0, LOCAL)):
                ws, gs = single_token_route_meta(
                    jnp.asarray(ids[None], jnp.int32),
                    jnp.asarray(w[None]),
                    n=n,
                    offset=offset,
                    num_experts=num_experts,
                )
                local = ids - offset
                keep = (local >= 0) & (local < num_experts)
                order = np.argsort(np.where(keep, local, num_experts), kind="stable")
                ref_ws = np.zeros(n, np.float32)
                ref_ws[: keep.sum()] = w[order[: keep.sum()]]
                ref_gs = np.bincount(local[keep], minlength=num_experts)
                np.testing.assert_array_equal(np.asarray(ws), ref_ws)
                np.testing.assert_array_equal(np.asarray(gs), ref_gs)

    def _check(self, m, *, rank, expect_route_meta):
        from sglang.srt.layers.moe.pallas import route_meta

        inputs = _inputs(m, seed=m - 1)
        with mock.patch.object(
            route_meta,
            "single_token_route_meta",
            wraps=route_meta.single_token_route_meta,
        ) as spy:
            out = _moe(*inputs, rank=rank, threshold=DENSE_THRESHOLD)
        self.assertEqual(spy.called, expect_route_meta)
        ref = _moe(*inputs, rank=rank, threshold=0)  # argsort + gather path
        self.assertTrue(np.all(np.isfinite(out)))
        # The reference combines in bf16, the new path in f32.
        scale = max(float(np.max(np.abs(ref))), 1e-6)
        self.assertLess(float(np.max(np.abs(out - ref))) / scale, 1e-2)
        return out, ref

    def test_bs1_all_local(self):
        self._check(1, rank=None, expect_route_meta=True)

    def test_bs1_expert_parallel(self):
        outs = [self._check(1, rank=r, expect_route_meta=True)[0] for r in range(EP)]
        # Per-rank partials (masked prefix, no group_offset) sum to the full MoE.
        full = _moe(*_inputs(1, seed=0), threshold=0)
        scale = float(np.max(np.abs(full)))
        self.assertLess(float(np.max(np.abs(sum(outs) - full))) / scale, 2e-2)

    def test_bs1_no_local_experts_is_zero(self):
        x, w1, w2, w, _ = _inputs(1, seed=7)
        ids = jnp.asarray(np.arange(LOCAL, LOCAL + TOPK)[None], jnp.int32)
        out = _moe(x, w1, w2, w, ids, rank=0, threshold=DENSE_THRESHOLD)
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_multi_token_skips_route_meta(self):
        self._check(8, rank=None, expect_route_meta=False)
        self._check(8, rank=1, expect_route_meta=False)

    def test_bs1_no_combine_keeps_existing_path(self):
        from sglang.srt.layers.moe.pallas import route_meta

        inputs = _inputs(1, seed=3)
        with mock.patch.object(
            route_meta,
            "single_token_route_meta",
            wraps=route_meta.single_token_route_meta,
        ) as spy:
            _moe(*inputs, threshold=DENSE_THRESHOLD, no_combine=True)
        self.assertFalse(spy.called)


def _poison_unwritten(gmm_v2):
    """Fill the rows a no-zero-init grouped matmul leaves unwritten with NaN."""

    def wrapped(*args, **kwargs):
        out = gmm_v2(*args, **kwargs)
        if kwargs["zero_initialize"] or kwargs["group_offset"] is not None:
            return out
        written = jnp.arange(out.shape[0])[:, None] < jnp.sum(kwargs["group_sizes"])
        return jnp.where(written, out, jnp.nan).astype(out.dtype)

    return wrapped


def _kind_weights(kind, num_experts, hidden, inter, seed):
    """Kernel weight kwargs for bf16 (+bias), FP8 channel/block and INT4 W4A16."""
    rng = np.random.default_rng(seed)
    kw = dict(activation="silu")
    if kind in ("bf16", "bf16_bias"):
        kw["w1"] = jnp.asarray(
            rng.standard_normal((num_experts, hidden, 2 * inter)) * 0.05, jnp.bfloat16
        )
        kw["w2"] = jnp.asarray(
            rng.standard_normal((num_experts, inter, hidden)) * 0.05, jnp.bfloat16
        )
        if kind == "bf16_bias":
            kw["activation"] = "swiglu_oai"
            kw["w1_bias"] = jnp.asarray(
                rng.standard_normal((num_experts, 1, 2 * inter)) * 0.1, jnp.float32
            )
            kw["w2_bias"] = jnp.asarray(
                rng.standard_normal((num_experts, 1, hidden)) * 0.1, jnp.float32
            )
    elif kind in ("fp8_ch", "fp8_blk"):
        for name, shape in (
            ("w1", (num_experts, hidden, 2 * inter)),
            ("w2", (num_experts, inter, hidden)),
        ):
            wf = rng.standard_normal(shape).astype(np.float32) * 0.05
            if kind == "fp8_ch":
                scale = np.abs(wf).max(axis=1, keepdims=True) / 448.0
                q = wf / scale
                scale = scale[:, None]  # [g, 1, 1, n]
            else:
                g, k, n = shape
                blocks = wf.reshape(g, k // 128, 128, n // 128, 128)
                scale = np.abs(blocks).max(axis=(2, 4)) / 448.0
                q = (blocks / scale[:, :, None, :, None]).reshape(shape)
                scale = scale[:, :, None, :]  # [g, k/128, 1, n/128]
            kw[name] = jnp.asarray(q, jnp.float8_e4m3fn)
            kw[name + "_scale"] = jnp.asarray(scale, jnp.float32)
    elif kind == "int4":
        for name, (k, n) in (("w1", (hidden, 2 * inter)), ("w2", (inter, hidden))):
            kw[name] = jnp.asarray(
                rng.integers(-(2**31), 2**31, (num_experts, k // 8, n)).astype(np.int32)
            )
            kw[name + "_scale"] = jnp.asarray(
                np.abs(rng.standard_normal((num_experts, k // 32, 1, n))) * 0.004
                + 0.001,
                jnp.float32,
            )
        kw["w_quant_bits"] = 4
    else:
        raise ValueError(kind)
    return kw


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestSingleTokenRouteMetaWeightKinds(CustomTestCase):
    """bs=1 route-meta path under jit, unwritten GMM rows poisoned with NaN."""

    E, EP, H, N = 16, 2, 512, 256

    def _run(self, kw, x, w, ids, *, rank, threshold, poison, expect_route_meta=None):
        import importlib

        from sglang.srt.layers.moe.pallas import fused_moe_op, route_meta

        gmm_mod = importlib.import_module("sglang.srt.layers.moe.pallas.gmm_v2")
        names = ("w1", "w2", "w1_scale", "w2_scale", "w1_bias", "w2_bias")
        weights = {name: kw.get(name) for name in names}
        static = dict(w_quant_bits=kw.get("w_quant_bits", 0))
        if rank is not None:
            local = slice(rank * self.E, (rank + 1) * self.E)
            weights = {k: None if v is None else v[local] for k, v in weights.items()}
            static.update(
                global_num_experts=self.E * self.EP, expert_group_offset=local.start
            )
        topk = int(ids.shape[1])
        fn = jax.jit(
            lambda x, w, ids, weights: fused_moe_op._fused_moe_jax(
                x,
                topk_weights=w,
                topk_ids=ids,
                topk=topk,
                activation_name=kw["activation"],
                permute_dense_threshold=threshold,
                **weights,
                **static,
            )
        )
        gmm = _poison_unwritten(gmm_mod.gmm_v2) if poison else gmm_mod.gmm_v2
        with (
            mock.patch.object(gmm_mod, "gmm_v2", gmm),
            mock.patch.object(
                route_meta,
                "single_token_route_meta",
                wraps=route_meta.single_token_route_meta,
            ) as spy,
        ):
            out = jax.block_until_ready(fn(x, w, ids, weights))
        self.assertEqual(
            spy.called,
            threshold > 0 if expect_route_meta is None else expect_route_meta,
        )
        return np.asarray(out.astype(jnp.float32))

    def test_top10_threshold_boundary(self):
        """One token pads to 8 rows at top-10 (80 routed rows, Qwen3.5's top-k),
        so the route-meta kernel needs a threshold above 80."""
        kw = _kind_weights("bf16", self.E * self.EP, self.H, self.N, seed=10)
        rng = np.random.default_rng(110)
        x = jnp.asarray(rng.standard_normal((1, self.H)) * 0.5, jnp.bfloat16)
        ids = jnp.asarray(
            rng.choice(self.E * self.EP, 10, replace=False)[None], jnp.int32
        )
        w = rng.random((1, 10)).astype(np.float32)
        w = jnp.asarray(w / w.sum(-1, keepdims=True), jnp.bfloat16)
        ref = self._run(kw, x, w, ids, rank=None, threshold=0, poison=False)
        scale = max(float(np.max(np.abs(ref))), 1e-6)
        for threshold, taken in ((80, False), (81, True)):
            with self.subTest(threshold=threshold):
                out = self._run(
                    kw,
                    x,
                    w,
                    ids,
                    rank=None,
                    threshold=threshold,
                    poison=False,
                    expect_route_meta=taken,
                )
                self.assertLess(float(np.max(np.abs(out - ref))) / scale, 1e-2)

    def test_weight_kinds_and_topk(self):
        cases = [("bf16", k) for k in (8, 10, 16)] + [
            (kind, k)
            for kind in ("bf16_bias", "fp8_ch", "fp8_blk", "int4")
            for k in (8, 10)
        ]
        for kind, topk in cases:
            kw = _kind_weights(kind, self.E * self.EP, self.H, self.N, seed=topk)
            rng = np.random.default_rng(100 + topk)
            x = jnp.asarray(rng.standard_normal((1, self.H)) * 0.5, jnp.bfloat16)
            ids = jnp.asarray(
                rng.choice(self.E * self.EP, topk, replace=False)[None], jnp.int32
            )
            w = rng.random((1, topk)).astype(np.float32)
            w = jnp.asarray(w / w.sum(-1, keepdims=True), jnp.bfloat16)
            partial = []
            for rank in (None, 0, 1):
                with self.subTest(kind=kind, topk=topk, rank=rank):
                    run = lambda **a: self._run(kw, x, w, ids, rank=rank, **a)
                    out = run(threshold=DENSE_THRESHOLD, poison=True)
                    clean = run(threshold=DENSE_THRESHOLD, poison=False)
                    ref = run(threshold=0, poison=False)
                    self.assertTrue(np.all(np.isfinite(out)))
                    np.testing.assert_array_equal(out, clean)
                    scale = max(float(np.max(np.abs(ref))), 1e-6)
                    self.assertLess(float(np.max(np.abs(out - ref))) / scale, 1e-2)
                    if rank is None:
                        full = out
                    else:
                        partial.append(out)
            with self.subTest(kind=kind, topk=topk, check="ep partials sum"):
                scale = max(float(np.max(np.abs(full))), 1e-6)
                err = float(np.max(np.abs(sum(partial) - full))) / scale
                self.assertLess(err, 2e-2)

    def test_padded_slots_are_dropped(self):
        # TopK marks a padded token's slots -1. Some -1 slots match dropping
        # them through a zero weight; an all -1 token outputs zeros, even NaN.
        kw = _kind_weights("bf16", self.E * self.EP, self.H, self.N, seed=0)
        rng = np.random.default_rng(5)
        x = jnp.asarray(rng.standard_normal((1, self.H)), jnp.bfloat16)
        ids = rng.choice(self.E * self.EP, TOPK, replace=False)[None]
        w = rng.random((1, TOPK)).astype(np.float32)
        w = w / w.sum(-1, keepdims=True)
        padded = ids.copy()
        padded[:, ::3] = -1
        zero_weight = w.copy()
        zero_weight[:, ::3] = 0
        bf16 = lambda a: jnp.asarray(a, jnp.bfloat16)
        i32 = lambda a: jnp.asarray(a, jnp.int32)
        nan_x = jnp.full_like(x, jnp.nan)
        for rank in (None, 0, 1):
            with self.subTest(rank=rank):
                run = lambda *a: self._run(
                    kw, *a, rank=rank, threshold=DENSE_THRESHOLD, poison=True
                )
                dropped = run(x, bf16(w), i32(padded))
                weighted = run(x, bf16(zero_weight), i32(ids))
                np.testing.assert_allclose(dropped, weighted, rtol=2**-7, atol=1e-6)
                pad_token = run(nan_x, bf16(w), i32(np.full_like(ids, -1)))
                np.testing.assert_array_equal(pad_token, np.zeros_like(pad_token))


if __name__ == "__main__":
    unittest.main()
