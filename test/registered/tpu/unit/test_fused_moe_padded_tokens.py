"""Pallas fused MoE drops slots with negative top-k ids (TopK's padded tokens)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=150, suite="stage-a-test-tpu-unit", chips=1)

import importlib
import itertools
import unittest
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

H, N = 512, 256
SITU = "situ:4.0:25.0"

# Routing shapes: (local experts, global experts, EP rank, top-k, dense threshold).
ROUTES = {
    "top2": (8, 16, 1, 2, 2048),
    "qwen3_coder": (20, 160, 3, 8, 2048),  # EP8, top-8, recipe threshold
    "qwen35": (64, 512, 2, 10, 1024),  # EP8, top-10, recipe threshold
    "k3": (28, 896, 5, 16, 512),  # EP32, top-16, recipe threshold
}


def _silu(gate, up):
    return gate / (1.0 + np.exp(-gate)) * up


def _situ(gate, up, beta=4.0, linear_beta=25.0):
    gate_act = beta * np.tanh(gate / beta) / (1.0 + np.exp(-gate))
    return gate_act * linear_beta * np.tanh(up / linear_beta)


def _moe_ref(hidden, w1, w2, ids, w, offset, act):
    """float32 MoE over this rank's experts; other ids contribute nothing."""
    out = np.zeros((hidden.shape[0], w2.shape[2]), np.float32)
    for t in range(hidden.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j]) - offset
            if 0 <= e < w1.shape[0]:
                gate_up = hidden[t] @ w1[e]
                out[t] += w[t, j] * (act(gate_up[:N], gate_up[N:]) @ w2[e])
    return out


def _bf16(a):
    return np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))


def _make_weights(kind, num_experts, rng):
    """Kernel args and the float32 reference weights (None: no reference)."""
    kw = dict(w1_scale=None, w2_scale=None, w_quant_bits=0, act="silu")
    ref = None
    if kind == "bf16":
        w1 = jnp.asarray(
            rng.standard_normal((num_experts, H, 2 * N)) / 16, jnp.bfloat16
        )
        w2 = jnp.asarray(rng.standard_normal((num_experts, N, H)) / 11, jnp.bfloat16)
        ref = (np.asarray(w1.astype(jnp.float32)), np.asarray(w2.astype(jnp.float32)))
    elif kind in ("fp8_ch", "fp8_blk"):
        w1f = (rng.standard_normal((num_experts, H, 2 * N)) / 16).astype(np.float32)
        w2f = (rng.standard_normal((num_experts, N, H)) / 11).astype(np.float32)

        def quant(wf):
            if kind == "fp8_ch":
                s = np.abs(wf).max(axis=1, keepdims=True) / 448.0  # [g, 1, n]
                q = jnp.asarray(wf / s, jnp.float8_e4m3fn)
                deq = np.asarray(q.astype(jnp.float32)) * s
                return q, jnp.asarray(s[:, None], jnp.float32), deq
            g, k, n = wf.shape
            blocks = wf.reshape(g, k // 128, 128, n // 128, 128)
            s = np.abs(blocks).max(axis=(2, 4)) / 448.0  # [g, k/128, n/128]
            q = jnp.asarray(
                (blocks / s[:, :, None, :, None]).reshape(g, k, n), jnp.float8_e4m3fn
            )
            deq = (
                np.asarray(q.astype(jnp.float32)).reshape(blocks.shape)
                * s[:, :, None, :, None]
            ).reshape(g, k, n)
            return q, jnp.asarray(s[:, :, None, :], jnp.float32), deq

        w1, kw["w1_scale"], ref1 = quant(w1f)
        w2, kw["w2_scale"], ref2 = quant(w2f)
        ref = (ref1, ref2)
    elif kind in ("int4", "int4_vmem_fp8"):
        words = lambda *s: jnp.asarray(
            rng.integers(-(2**31), 2**31, s, dtype=np.int64).astype(np.int32)
        )
        scale = lambda *s: jnp.asarray(
            np.abs(rng.standard_normal(s)) * 0.004 + 0.001, jnp.float32
        )
        w1, w2 = words(num_experts, H // 8, 2 * N), words(num_experts, N // 8, H)
        kw.update(
            w1_scale=scale(num_experts, H // 32, 1, 2 * N),
            w2_scale=scale(num_experts, N // 32, 1, H),
            w_quant_bits=4,
        )
        if kind == "int4_vmem_fp8":
            channel = lambda *s: jnp.asarray(
                np.abs(rng.standard_normal(s)) + 0.5, jnp.float32
            )
            kw["w1_channel_scale"] = channel(num_experts, 1, 2 * N)
            kw["w2_channel_scale"] = channel(num_experts, 1, H)
            kw["w1_scale"] = kw["w1_scale"] / kw["w1_channel_scale"][:, None]
            kw["w2_scale"] = kw["w2_scale"] / kw["w2_channel_scale"][:, None]
    elif kind == "mxfp4":
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_bf16,
            process_mxfp4_moe_weights_native,
        )

        g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
        e8 = lambda *s: torch.randint(122, 127, s, dtype=torch.uint8, generator=g)
        zeros = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
        hf = (
            u8(num_experts, 2 * N, H // 2),
            e8(num_experts, 2 * N, H // 32),
            zeros(num_experts, 2 * N),
            u8(num_experts, H, N // 2),
            e8(num_experts, H, N // 32),
            zeros(num_experts, H),
        )
        w13, w13_s, _, w2t, w2_s, _ = process_mxfp4_moe_weights_native(
            *hf, w13_interleaved=False
        )
        w13_ref, _, w2_ref, _ = process_mxfp4_moe_weights_bf16(
            *hf, w13_interleaved=False
        )
        w1, w2 = jnp.asarray(w13.numpy()), jnp.asarray(w2t.numpy())
        kw.update(
            w1_scale=jnp.asarray(w13_s.numpy()),
            w2_scale=jnp.asarray(w2_s.numpy()),
            act=SITU,
        )
        ref = (w13_ref.float().numpy(), w2_ref.float().numpy())
    else:
        raise ValueError(kind)
    return w1, w2, kw, ref


def _poisoned_gmm(gmm_v2):
    """gmm_v2 with every row it does not compute set to NaN when not zero-init."""

    def gmm(*args, **kwargs):
        out = gmm_v2(*args, **kwargs)
        if kwargs.get("zero_initialize", True):
            return out
        sizes = kwargs["group_sizes"]
        first = 0 if kwargs.get("group_offset") is None else kwargs["group_offset"][0]
        group = jnp.arange(sizes.shape[0])
        start = jnp.sum(jnp.where(group < first, sizes, 0))
        local = (group >= first) & (group < first + kwargs["rhs"].shape[0])
        end = start + jnp.sum(jnp.where(local, sizes, 0))
        rows = jnp.arange(out.shape[0])[:, None]
        return jnp.where((rows >= start) & (rows < end), out, jnp.nan).astype(out.dtype)

    return gmm


def _routing(m, route, ep, rng):
    """Top-k ids with about half on this rank's experts; normalized weights."""
    e_local, e_global, rank, topk, _ = route
    offset = rank * e_local if ep else 0
    ids = []
    for _ in range(m):
        n_local = topk if not ep else topk // 2
        local = rng.choice(e_local, size=n_local, replace=False) + offset
        if ep:
            rest = np.setdiff1d(np.arange(e_global), local)
            local = np.concatenate([local, rng.choice(rest, topk - n_local, False)])
        ids.append(rng.permutation(local))
    w = rng.random((m, topk)).astype(np.float32)
    return np.stack(ids).astype(np.int32), w / w.sum(-1, keepdims=True)


def _ulp_close(a, b):
    return bool(np.all(np.abs(a - b) <= 2**-7 * np.maximum(np.abs(a), np.abs(b))))


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestFusedMoEPaddedTokens(CustomTestCase):
    _weights = {}
    _fns = {}

    def _weights_for(self, kind, route_name):
        key = (kind, route_name)
        if key not in self._weights:
            rng = np.random.default_rng(len(self._weights))
            self._weights[key] = _make_weights(kind, ROUTES[route_name][0], rng)
        return self._weights[key]

    def _fn(self, kind, route_name, ep, local_ids, thr, sc, no_combine):
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        key = (kind, route_name, ep, local_ids, thr, sc, no_combine)
        if key not in self._fns:
            w1, w2, kw, _ = self._weights_for(kind, route_name)
            e_local, e_global, rank, topk, _ = ROUTES[route_name]
            route_kw = {}
            if ep and not local_ids:
                route_kw = dict(
                    global_num_experts=e_global, expert_group_offset=rank * e_local
                )
            weights = (
                w1,
                w2,
                kw["w1_scale"],
                kw["w2_scale"],
                kw.get("w1_channel_scale"),
                kw.get("w2_channel_scale"),
            )
            call = jax.jit(
                lambda h, tw, ti, w1, w2, s1, s2, c1, c2: _fused_moe_jax(
                    h,
                    w1,
                    w2,
                    tw,
                    ti,
                    s1,
                    s2,
                    None,
                    None,
                    topk,
                    kw["act"],
                    no_combine=no_combine,
                    use_sc_gather=sc,
                    w_quant_bits=kw["w_quant_bits"],
                    permute_dense_threshold=thr,
                    w1_channel_scale=c1,
                    w2_channel_scale=c2,
                    **route_kw,
                )
            )

            gmm_mod = importlib.import_module("sglang_tpu.moe.pallas.gmm_v2")
            poisoned = _poisoned_gmm(gmm_mod.gmm_v2)

            def run(hidden, ids, w):
                with mock.patch.object(gmm_mod, "gmm_v2", poisoned):
                    out = call(
                        jnp.asarray(hidden).astype(jnp.bfloat16),
                        jnp.asarray(w).astype(jnp.bfloat16),
                        jnp.asarray(ids, jnp.int32),
                        *weights,
                    )
                return np.asarray(jax.block_until_ready(out).astype(jnp.float32))

            self._fns[key] = run
        return self._fns[key]

    def _check(
        self,
        kind,
        route_name,
        m,
        *,
        ep,
        dense,
        sc=0,
        no_combine=False,
        ref_tol=None,
    ):
        e_local, e_global, rank, topk, recipe_thr = ROUTES[route_name]
        thr = recipe_thr if dense else 0
        offset = rank * e_local if ep else 0
        run = self._fn(kind, route_name, ep, False, thr, sc, no_combine)
        rng = np.random.default_rng(1000 * m + topk)
        ids, w = _routing(m, ROUTES[route_name], ep, rng)
        hidden = rng.standard_normal((m, H)).astype(np.float32)
        route_experts = e_global if ep else e_local
        for nv in sorted({0, 1, m // 2, m}):
            tag = f"{kind} {route_name} m={m} nv={nv} ep={ep} thr={thr} sc={sc} nc={no_combine}"
            pad = slice(nv, None)
            clean_h, clean_ids, clean_w = hidden.copy(), ids.copy(), w.copy()
            clean_h[pad], clean_ids[pad], clean_w[pad] = 0, -1, 0
            bad_h, bad_ids, bad_w = hidden.copy(), ids.copy(), w.copy()
            bad_h[pad] = np.nan
            bad_h[pad, ::7] = np.inf
            bad_ids[pad] = rng.choice(
                [-1, -7, -(2**20), route_experts, route_experts + 5],
                size=(m - nv, topk),
            )
            bad_w[pad] = np.nan
            # Pre-change contract: padded slots clamped to expert 0, weight 0.
            clamp_ids = clean_ids.copy()
            clamp_ids[pad] = 0

            clean = run(clean_h, clean_ids, clean_w)
            bad = run(bad_h, bad_ids, bad_w)
            clamp = run(clean_h, clamp_ids, clean_w)

            self.assertTrue(np.all(np.isfinite(bad[:nv])), tag)
            np.testing.assert_array_equal(bad[:nv], clean[:nv], err_msg=tag)
            np.testing.assert_array_equal(bad[nv:], 0, err_msg=tag)
            np.testing.assert_array_equal(clean[nv:], 0, err_msg=tag)
            self.assertTrue(_ulp_close(clean[:nv], clamp[:nv]), tag)

            ref_weights = self._weights_for(kind, route_name)[3]
            if no_combine or ref_tol is None or nv == 0 or ref_weights is None:
                continue
            act = _situ if kind == "mxfp4" else _silu
            ref = _moe_ref(
                _bf16(hidden[:nv]),
                *ref_weights,
                ids[:nv],
                _bf16(w[:nv]),
                offset,
                act,
            )
            err = float(np.max(np.abs(clean[:nv] - ref)) / (np.max(np.abs(ref)) + 1e-9))
            self.assertLess(err, ref_tol, tag)

    def test_bf16_all_topk_paths(self):
        for route_name, m, ep, dense in itertools.product(
            ROUTES, (1, 5, 20), (True, False), (False, True)
        ):
            with self.subTest(route=route_name, m=m, ep=ep, dense=dense):
                self._check("bf16", route_name, m, ep=ep, dense=dense, ref_tol=0.03)
        # Even m: odd counts need ragged_gather_v2's row padding (covered in
        # test_fused_moe_sc_gather_odd_tokens.py).
        for route_name, ep in itertools.product(ROUTES, (True, False)):
            with self.subTest(route=route_name, m=20, ep=ep, sc=1):
                self._check("bf16", route_name, 20, ep=ep, dense=False, sc=1)

    def test_non_finite_live_token_stays_in_its_own_row(self):
        """The one-hot permute is a matmul over every token, so one live token
        with a NaN/inf hidden state (0 * NaN) turned every token's output NaN."""
        m, poisoned = 5, 2
        others = [i for i in range(m) if i != poisoned]
        for route_name, ep in itertools.product(ROUTES, (True, False)):
            recipe_thr = ROUTES[route_name][4]
            run = self._fn("bf16", route_name, ep, False, recipe_thr, 0, False)
            rng = np.random.default_rng(7)
            ids, w = _routing(m, ROUTES[route_name], ep, rng)
            hidden = rng.standard_normal((m, H)).astype(np.float32)
            clean = run(hidden, ids, w)
            bad_h, bad_w = hidden.copy(), w.copy()
            bad_h[poisoned] = np.nan
            bad_h[poisoned, ::5] = np.inf
            # The router derives the weights from the same hidden state, so in
            # serving they are NaN too; poisoning them alone isolates the combine.
            bad_w[poisoned] = np.nan
            inputs = {
                "hidden": (bad_h, w),
                "hidden+weights": (bad_h, bad_w),
                "weights": (hidden, bad_w),
            }
            for poison, (h, tw) in inputs.items():
                with self.subTest(route=route_name, ep=ep, poison=poison):
                    bad = run(h, ids, tw)
                    self.assertTrue(np.all(np.isfinite(bad[others])))
                    np.testing.assert_array_equal(bad[others], clean[others])

    def test_permute_follows_the_gather_combine(self):
        # m < threshold <= m * topk: the combine gathers, so the permute must
        # too. A one-hot permute feeding the SparseCore combine gave wrong
        # tokens on Kimi-K3 whenever moe_front or latent_up ran static fp8.
        from sglang_tpu.moe.pallas import fused_moe_op

        e_local, e_global, rank, topk, thr = ROUTES["k3"]
        m = 40
        self.assertLess(m, thr)
        self.assertGreaterEqual(m * topk, thr)
        w1, w2, kw, _ = self._weights_for("bf16", "k3")
        rng = np.random.default_rng(m)
        hidden = jnp.asarray(rng.standard_normal((m, H)), jnp.bfloat16)
        for ep, sc in itertools.product((True, False), (0, 1)):
            with self.subTest(ep=ep, sc=sc):
                ids, w = _routing(m, ROUTES["k3"], ep, rng)
                route_kw = (
                    dict(
                        global_num_experts=e_global, expert_group_offset=rank * e_local
                    )
                    if ep
                    else {}
                )

                def run(threshold):
                    # A fresh jit per call, so the spy sees the trace.
                    fn = jax.jit(
                        lambda h: fused_moe_op._fused_moe_jax(
                            h,
                            w1,
                            w2,
                            jnp.asarray(w, jnp.bfloat16),
                            jnp.asarray(ids),
                            kw["w1_scale"],
                            kw["w2_scale"],
                            None,
                            None,
                            topk,
                            kw["act"],
                            use_sc_gather=sc,
                            w_quant_bits=kw["w_quant_bits"],
                            permute_dense_threshold=threshold,
                            **route_kw,
                        )
                    )
                    with mock.patch.object(
                        jax.nn, "one_hot", wraps=jax.nn.one_hot
                    ) as one_hot:
                        out = jax.block_until_ready(fn(hidden))
                    classes = [call.args[1] for call in one_hot.call_args_list]
                    return np.asarray(out.astype(jnp.float32)), classes

                mixed, classes = run(thr)
                # Traced (the group-size count is a one-hot over experts), with
                # no one-hot over the m tokens.
                self.assertTrue(classes)
                self.assertNotIn(m, classes)
                gather, _ = run(0)
                np.testing.assert_array_equal(mixed, gather)

    def test_quantized_weights(self):
        cases = [
            ("fp8_ch", "qwen3_coder", 0.1),
            ("fp8_ch", "qwen35", 0.1),
            ("fp8_blk", "qwen3_coder", 0.1),
            ("int4", "qwen3_coder", None),
            ("int4_vmem_fp8", "qwen3_coder", None),
            ("mxfp4", "k3", 0.03),
        ]
        for (kind, route_name, tol), m, ep, dense in itertools.product(
            cases, (1, 20), (True, False), (False, True)
        ):
            with self.subTest(kind=kind, route=route_name, m=m, ep=ep, dense=dense):
                self._check(kind, route_name, m, ep=ep, dense=dense, ref_tol=tol)

    def test_no_combine(self):
        for ep in (True, False):
            for sc in (0, 1):
                with self.subTest(ep=ep, sc=sc):
                    self._check(
                        "bf16",
                        "qwen3_coder",
                        6,
                        ep=ep,
                        dense=False,
                        sc=sc,
                        no_combine=True,
                    )

    def test_local_ids_with_dropped_slots_match_global_ids(self):
        # NVFP4 hands the kernel local ids with non-local slots as -1 and no
        # EP window: real tokens then carry dropped slots on the all-local path.
        e_local, e_global, rank, topk, recipe_thr = ROUTES["k3"]
        offset = rank * e_local
        rng = np.random.default_rng(7)
        m = 20
        ids, w = _routing(m, ROUTES["k3"], True, rng)
        hidden = rng.standard_normal((m, H)).astype(np.float32)
        local = ids - offset
        owned = (local >= 0) & (local < e_local)
        local_ids = np.where(owned, local, -1)
        for thr, sc in ((0, 0), (0, 1), (recipe_thr, 0)):
            with self.subTest(thr=thr, sc=sc):
                global_run = self._fn("bf16", "k3", True, False, thr, sc, False)
                local_run = self._fn("bf16", "k3", True, True, thr, sc, False)
                expected = global_run(hidden, ids, w)
                actual = local_run(hidden, local_ids, w * owned)
                self.assertTrue(np.all(np.isfinite(actual)))
                self.assertTrue(_ulp_close(actual, expected), (thr, sc))


if __name__ == "__main__":
    unittest.main()
