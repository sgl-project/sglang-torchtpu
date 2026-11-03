"""Pallas ``_fused_moe_jax`` (production kernel body) vs a pure-JAX
reference MoE.

This file isolates the Pallas kernel's correctness from the
PyTorch <-> JAX <-> torch_tpu shuttle by calling the **production**
``_fused_moe_jax`` directly with JAX inputs (no torch custom op in the
loop), comparing against a dense JAX reference built from
``jnp.einsum`` + one-hot expert masking.

Both sides run on TPU under JAX, and the kernel under test is the
exact function ``_build_fused_moe_op`` exports to ``jax_op``. Any
production drift (sort order, pad amount, activation gating, tile
selection) lands here.

Run as:
    SGLANG_PLATFORM=tpu python \\
      test/manual/tpu/test_pallas_fused_moe.py
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=25, suite="stage-a-test-tpu-unit", chips=1)

import unittest

try:
    import jax
    import jax.numpy as jnp

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


def _call_production_kernel(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    activation: str,
):
    """Invoke the production ``_fused_moe_jax`` for the unquant path.

    ``w*_scale`` / ``w*_bias`` are ``None`` -- the kernel signature
    accepts them positionally so a future quantised branch can plug in
    without churning this test.
    """
    # Lazy import so the module loads on hosts without JAX (the
    # @skipUnless decorator below will skip the whole class there).
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    topk = int(topk_ids.shape[-1])
    return _fused_moe_jax(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        None,  # w1_scale
        None,  # w2_scale
        None,  # w1_bias
        None,  # w2_bias
        topk,
        activation,
    )


def _jax_reference_moe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation: str,
):
    """Pure-JAX MoE reference -- dense einsum over a one-hot expert mask.

    Algebraically equivalent to ``_fused_moe_jax`` but with no Pallas
    grid, no grouped-matmul, no padding -- the diff vs the kernel can
    only be the kernel's own arithmetic.
    """
    m, k = hidden_states.shape
    g, _k, two_n = w1.shape
    assert _k == k
    n = two_n // 2

    hs = hidden_states.astype(jnp.float32)
    w1f = w1.astype(jnp.float32)
    w2f = w2.astype(jnp.float32)
    tw = topk_weights.astype(jnp.float32)

    # one-hot mask: [m, topk, g] picks the routed expert for each
    # (token, slot). Multiplying by it selects w1[e] / w2[e] per slot.
    onehot = jax.nn.one_hot(topk_ids, g, dtype=jnp.float32)  # [m, topk, g]

    # gmm1[m,j,2n] = sum_{e} onehot[m,j,e] * (hs[m,:] @ w1[e])
    gmm1 = jnp.einsum("mjg,mk,gkn->mjn", onehot, hs, w1f)

    gate, up = gmm1[..., :n], gmm1[..., n:]
    if activation == "silu":
        act = jax.nn.silu(gate) * up
    elif activation == "gelu":
        act = jax.nn.gelu(gate) * up
    else:
        raise AssertionError(f"unsupported activation in reference: {activation}")

    gmm2 = jnp.einsum("mjg,mjn,gnk->mjk", onehot, act, w2f)  # [m, topk, k]
    return jnp.einsum("mj,mjk->mk", tw, gmm2)


def _make_inputs(m, k, n, g, topk, seed, dtype=None):
    """JAX-on-TPU inputs with magnitudes chosen to stay inside bf16's
    accurate range (sub-saturated gates, sub-overflow accumulators).
    """
    if dtype is None:
        dtype = jnp.bfloat16
    key = jax.random.PRNGKey(seed)
    kh, kw1, kw2, ktw, kids = jax.random.split(key, 5)
    hidden = (jax.random.normal(kh, (m, k), dtype=jnp.float32) * 0.5).astype(dtype)
    w1 = (jax.random.normal(kw1, (g, k, 2 * n), dtype=jnp.float32) * 0.05).astype(dtype)
    w2 = (jax.random.normal(kw2, (g, n, k), dtype=jnp.float32) * 0.05).astype(dtype)
    topk_ids = jax.random.randint(kids, (m, topk), 0, g, dtype=jnp.int32)
    topk_weights = jax.nn.softmax(
        jax.random.normal(ktw, (m, topk), dtype=jnp.float32), axis=-1
    ).astype(dtype)
    return hidden, w1, w2, topk_weights, topk_ids


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoEJaxKernelVsReference(unittest.TestCase):
    """Compare the production ``_fused_moe_jax`` body to a dense JAX
    reference. Any drift in the production kernel's permute / GMM1 /
    activation / GMM2 / un-permute path lands here as a numerics diff,
    independent of the torch_tpu bridge.
    """

    def test_glm_clamped_swiglu(self):
        # Exact projections straddle both clamps. A negative gate must remain
        # negative (only its upper bound is clamped); up is clamped on both sides.
        hidden = jnp.ones((3, 128), dtype=jnp.bfloat16)
        gate = jnp.tile(jnp.array([20.0, -20.0]), 64)
        up = jnp.tile(jnp.array([-20.0, 20.0]), 64)
        w1 = jnp.broadcast_to(jnp.concatenate([gate, up]) / 128, (4, 128, 256)).astype(
            jnp.bfloat16
        )
        w2 = jnp.broadcast_to(jnp.eye(128), (4, 128, 128)).astype(jnp.bfloat16)
        ids = jnp.array([[0, 1], [2, 3], [1, 2]], dtype=jnp.int32)
        weights = jnp.full((3, 2), 0.5, dtype=jnp.bfloat16)
        out = _call_production_kernel(
            hidden, w1, w2, weights, ids, activation="silu_and_mul_clamp:10.0"
        )
        expected = jnp.broadcast_to(
            jnp.array([-100.0, -200.0 / (1.0 + jnp.exp(20.0))] * 64), (3, 128)
        )
        self.assertLess(
            float(jnp.max(jnp.abs(out.astype(jnp.float32) - expected))), 0.001
        )

    def _run_and_diff(self, *, m, k, n, g, topk, activation, seed, tol, force_ids=None):
        hidden, w1, w2, w_tk, ids = _make_inputs(m, k, n, g, topk, seed)
        if force_ids is not None:
            ids = force_ids

        out = jax.block_until_ready(
            _call_production_kernel(hidden, w1, w2, w_tk, ids, activation=activation)
        )
        ref = _jax_reference_moe(hidden, w1, w2, w_tk, ids, activation)

        out32 = jnp.asarray(out).astype(jnp.float32)
        ref32 = ref.astype(jnp.float32)

        self.assertEqual(out32.shape, ref32.shape)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(out32))),
            "Pallas kernel produced non-finite output",
        )
        max_diff = float(jnp.max(jnp.abs(out32 - ref32)))
        self.assertLess(
            max_diff,
            tol,
            f"Pallas vs dense JAX max diff = {max_diff} "
            f"(activation={activation}, m={m}, k={k}, n={n}, g={g}, topk={topk})",
        )
        return out32, ref32, max_diff

    # ---- shape-aligned (no M padding) cases ---------------------------
    def test_silu_aligned_m_matches_reference(self):
        """m*topk = 64 -- a multiple of the 32-row sublane tile, so the
        kernel's ``pad_m=0`` fast path runs. Pinning this avoids a
        regression in the no-pad branch.
        """
        self._run_and_diff(
            m=32,
            k=128,
            n=128,
            g=4,
            topk=2,
            activation="silu",
            seed=0,
            tol=5e-2,
        )

    def test_gelu_aligned_m_matches_reference(self):
        """Same shape, different gate. Catches drift between the silu /
        gelu branches in ``_activation``."""
        self._run_and_diff(
            m=32,
            k=128,
            n=128,
            g=4,
            topk=2,
            activation="gelu",
            seed=1,
            tol=5e-2,
        )

    # ---- M-padding cases ----------------------------------------------
    def test_silu_unaligned_m_pad_path_matches_reference(self):
        """m*topk = 24 -- exercises the pad-to-32 branch.

        Without the pad, ``gmm_v2.zero_out_start`` does
        ``zero_ref.reshape(-1, size_lhs_sublane, num_lanes)`` and raises
        a Pallas ``ValueError`` at JAX trace time. Either failure mode
        is a clean signal.
        """
        self._run_and_diff(
            m=3,
            k=128,
            n=128,
            g=4,
            topk=8,
            activation="silu",
            seed=2,
            tol=5e-2,
        )

    def test_silu_odd_topk_matches_reference(self):
        """topk=3 -- the m*topk % 32 != 0 path with a less symmetric
        ratio (m=8, topk=3 -> 24 rows). Picks up regressions in the
        pad/un-permute path that aligned shapes would miss."""
        self._run_and_diff(
            m=8,
            k=128,
            n=128,
            g=4,
            topk=3,
            activation="silu",
            seed=3,
            tol=5e-2,
        )

    # ---- expert-coverage cases ----------------------------------------
    def test_silu_full_expert_coverage_matches_reference(self):
        """Round-robin routing -- every expert is hit at least once,
        so ``group_sizes`` has no zero entries. Pair with the next
        test to cover both arms of the gmm_v2 empty-group branch.
        """
        m, topk, g = 64, 2, 4
        flat = jnp.arange(m * topk, dtype=jnp.int32) % g
        ids = flat.reshape(m, topk)
        self._run_and_diff(
            m=m,
            k=128,
            n=128,
            g=g,
            topk=topk,
            activation="silu",
            seed=4,
            tol=5e-2,
            force_ids=ids,
        )

    def test_silu_single_expert_dominates_matches_reference(self):
        """All routing collapses to expert 0 -- ``group_sizes`` is
        ``[m*topk, 0, 0, ...]``. The kernel must still process empty
        expert groups without leaking writes into their output rows.
        """
        m, topk, g = 16, 2, 4
        ids = jnp.zeros((m, topk), dtype=jnp.int32)
        self._run_and_diff(
            m=m,
            k=128,
            n=128,
            g=g,
            topk=topk,
            activation="silu",
            seed=5,
            tol=5e-2,
            force_ids=ids,
        )

    def test_non_lane_aligned_hidden_with_padded_weights_matches_reference(self):
        """Exercises pre-gather activation K-padding and post-sum K-slicing
        when model hidden_size (192) is narrower than the 128-lane-padded
        weight width (256) and m (19) is padded to a sublane multiple (20).
        """
        m, k, k_padded, n, g, topk = 19, 192, 256, 128, 4, 4
        hidden, w1, w2, w_tk, ids = _make_inputs(m, k, n, g, topk, seed=6)
        w1_pad = jnp.pad(w1, ((0, 0), (0, k_padded - k), (0, 0)))
        w2_pad = jnp.pad(w2, ((0, 0), (0, 0), (0, k_padded - k)))

        out = jax.block_until_ready(
            _call_production_kernel(
                hidden, w1_pad, w2_pad, w_tk, ids, activation="silu"
            )
        )
        ref = _jax_reference_moe(hidden, w1, w2, w_tk, ids, "silu")

        out32 = jnp.asarray(out).astype(jnp.float32)
        ref32 = ref.astype(jnp.float32)
        self.assertEqual(out32.shape, (m, k))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out32))))
        self.assertLess(float(jnp.max(jnp.abs(out32 - ref32))), 5e-2)


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoEJaxKernelGuards(unittest.TestCase):
    """Negative tests pinning the kernel's documented error contracts
    (raised via the module-level ``_fuse_act_name`` helper)."""

    def test_swiglu_oai_maps_to_kernel_name(self):
        """SGLang spells the GPT-OSS gating ``swiglu_oai``; gmm_v2's
        in-kernel ``fuse_act`` spells it ``swigluoai``. Both spellings must
        map through, since the activation is applied inside GMM1."""
        from sglang_tpu.moe.pallas.fused_moe_op import _fuse_act_name

        self.assertEqual(_fuse_act_name("swiglu_oai"), "swigluoai")
        self.assertEqual(_fuse_act_name("swigluoai"), "swigluoai")

    def test_unknown_activation_raises_not_implemented(self):
        """A typo or future activation name shouldn't fall through to a
        default branch -- it must surface as NotImplementedError naming
        the offending value, so a model that requests a new activation
        gets a clear failure mode.
        """
        from sglang_tpu.moe.pallas.fused_moe_op import _fuse_act_name

        with self.assertRaises(NotImplementedError) as cm:
            _fuse_act_name("relu2")
        self.assertIn("relu2", str(cm.exception))


def _pad_moe_weights(w1, w1_bias, w2, w2_bias, *, k_padded, n_padded):
    """Zero-pad unpadded MoE weights the way ``prepare_tpu_mxfp4_moe_weights``
    lays them out: hidden (w1's K, w2's N) to ``k_padded``, and each of gate
    and up (w1's N halves) and w2's K to ``n_padded``."""
    _, k, two_n = w1.shape
    n = two_n // 2
    w1_pad = ((0, 0), (0, k_padded - k), (0, n_padded - n))
    b1_pad = ((0, 0), (0, 0), (0, n_padded - n))
    w1 = jnp.concatenate(
        [jnp.pad(w1[..., :n], w1_pad), jnp.pad(w1[..., n:], w1_pad)], axis=-1
    )
    w1_bias = jnp.concatenate(
        [jnp.pad(w1_bias[..., :n], b1_pad), jnp.pad(w1_bias[..., n:], b1_pad)],
        axis=-1,
    )
    w2 = jnp.pad(w2, ((0, 0), (0, n_padded - n), (0, k_padded - k)))
    w2_bias = jnp.pad(w2_bias, ((0, 0), (0, 0), (0, k_padded - k)))
    return w1, w1_bias, w2, w2_bias


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoEPaddedIntermediate(unittest.TestCase):
    """GPT-OSS's MoE at TP1: hidden = intermediate = 2880. The loader pads the
    intermediate to ``gmm_v2.pad_size_n_to_fewest_tiles`` (3072 on tpu7x)
    rather than the 128-aligned 2944 that GMM1 splits into 128-wide tiles.
    With 128 MiB of VMEM 2944 is one tile and the two widths coincide."""

    HIDDEN = INTER = 2880
    K = 2944  # hidden padded to 128 lanes

    def _padded_inter(self):
        from sglang_tpu.moe.pallas.gmm_v2 import pad_size_n_to_fewest_tiles

        return pad_size_n_to_fewest_tiles(
            self.K, self.INTER, jnp.bfloat16, fuse_act="swigluoai"
        )

    def _gmm1_tile_n(self, size_m, size_n):
        """The N tile make_gmm_configs runs for GMM1 on this chip."""
        from sglang_tpu.moe.pallas import gmm_v2

        spec = jax.ShapeDtypeStruct
        return gmm_v2.make_gmm_configs(
            spec((size_m, self.K), jnp.bfloat16),
            spec((1, self.K, 2 * size_n), jnp.bfloat16),
            None,
            None,
            spec((1,), jnp.int32),
            spec((1,), jnp.int32),
            tile_info=gmm_v2.calculate_tiling,
            vmem_limit_bytes=gmm_v2.default_vmem_limit_bytes(),
            out_dtype=None,
            acc_dtype=None,
            maybe_quantize_lhs=True,
            zero_initialize=True,
            fuse_act="swigluoai",
        ).tiles.tile_n

    def test_padded_width_runs_fewer_gmm1_tiles(self):
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental.pallas import tpu as pltpu

        # The loader sizes the width against the per-chip metadata
        # tpu_info_for_cpu_export swaps in, so that has to match what the
        # kernel sees in every field the tiling reads.
        chip = tpu_info.chip_version_from_device_kind(jax.devices()[0].device_kind)
        export = tpu_info.get_tpu_info_for_chip(chip, 1)
        runtime = pltpu.get_tpu_info()
        for field in ("vmem_capacity_bytes", "num_lanes", "mxu_column_size"):
            with self.subTest(field=field):
                self.assertEqual(getattr(export, field), getattr(runtime, field))
        self.assertEqual(
            export.get_sublane_tiling(jnp.bfloat16),
            runtime.get_sublane_tiling(jnp.bfloat16),
        )

        padded = self._padded_inter()
        prefill_m = 4096
        tile_n = self._gmm1_tile_n(prefill_m, padded)
        self.assertLessEqual(
            padded // tile_n, self.K // self._gmm1_tile_n(prefill_m, self.K)
        )
        if runtime.vmem_capacity_bytes == 64 * 2**20:
            self.assertEqual((padded, tile_n), (3072, 1024))
        # Decode steps route fewer rows, which only widens the tile.
        for size_m in (16, 32, 128):
            with self.subTest(size_m=size_m):
                self.assertGreaterEqual(self._gmm1_tile_n(size_m, padded), tile_n)

    def test_padded_intermediate_matches_the_128_aligned_width(self):
        """Zero gate/up columns and bias give zero activations, which meet
        zero rows of w2, so only the accumulation order may differ."""
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        g, topk = 8, 4
        hidden, w1, w2, topk_weights, topk_ids = _make_inputs(
            512, self.HIDDEN, self.INTER, g, topk, seed=0
        )
        kb1, kb2 = jax.random.split(jax.random.PRNGKey(1))
        w1_bias = jax.random.normal(kb1, (g, 1, 2 * self.INTER), jnp.float32) * 0.05
        w2_bias = jax.random.normal(kb2, (g, 1, self.HIDDEN), jnp.float32) * 0.05
        widths = {
            "lane_aligned": self.K,
            "fewest_tiles": self._padded_inter(),
        }
        weights = {
            name: _pad_moe_weights(
                w1, w1_bias, w2, w2_bias, k_padded=self.K, n_padded=n_padded
            )
            for name, n_padded in widths.items()
        }
        # 8 tokens is a decode step (32 routed rows), 512 a prefill chunk.
        for m in (8, 512):
            outs = {}
            for name, (pw1, pb1, pw2, pb2) in weights.items():
                outs[name] = jax.block_until_ready(
                    _fused_moe_jax(
                        hidden[:m],
                        pw1,
                        pw2,
                        topk_weights[:m],
                        topk_ids[:m],
                        None,
                        None,
                        pb1,
                        pb2,
                        topk,
                        "swigluoai",
                    )
                ).astype(jnp.float32)
            base, got = outs["lane_aligned"], outs["fewest_tiles"]
            with self.subTest(m=m, widths=widths):
                self.assertTrue(bool(jnp.all(jnp.isfinite(got))))
                max_diff = float(jnp.max(jnp.abs(got - base)))
                # A couple of bf16 ulps at the output's largest magnitude.
                tol = 2**-6 * max(1.0, float(jnp.max(jnp.abs(base))))
                self.assertLess(max_diff, tol)


if __name__ == "__main__":
    unittest.main()
