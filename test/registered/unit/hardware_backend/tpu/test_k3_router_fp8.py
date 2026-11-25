"""``KernelConfig.decode_router_fp8``: the decode MoE router as FP8 e4m3 with
one FP32 scale per expert column (rank_loader.router_fp8_families), the
scales multiplying the FP32 products before the sigmoid.

At load, each layer's router becomes rank_loader.quantize_columns_fp8 of its
BF16 weight and the colocated prefill keeps its own BF16 router. In Pallas
interpret mode on a rank pair of host devices, at 8 and 16 rows (null rows at
16) with stale NaN bytes under every router column, ``_route_hidden_states``
with the FP8 router must give the same routes bit for bit with both router
halves and with the pair's score exchange, and with the argmax and the
presorted top-k. With power-of-two scales its scaled raw scores match the FP32
product of the dequantized weights, and its experts and route counts equal
(probabilities and totals within 4 FP32 ulps) those of the BF16 router holding
the dequantized weights (exact in BF16; the CPU backend sums BF16 x BF16 and
BF16 x FP8 products in other orders). With amax / 448 scales the scaled raw
scores match the FP32 product and every rank owns the experts of its top-16."""

import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.ci.gate_matrix import full_matrix
from sglang.test.test_utils import CustomTestCase

# Whole kernels in Pallas interpret mode on host devices, minutes a case: the
# TPU nightly runs it (the CPU runners do not install jax).
register_tpu_ci(
    est_time=240,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "kimi-k3", "k3-programs"],
)


class TestRouterFp8Families(CustomTestCase):
    """The FP8 router families at load and their config."""

    def test_families_replace_the_router_per_layer(self):
        import ml_dtypes

        from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

        rng = np.random.default_rng(0)
        router = (
            rng.standard_normal((3, 256, 896)) * rng.uniform(1e-3, 0.05, 896)
        ).astype(ml_dtypes.bfloat16)
        router[1, :, 7] = 0
        families = {"router": router.copy(), "router_bias": np.zeros((3, 1, 896))}
        converted = rank_loader.router_fp8_families(families)
        self.assertEqual(sorted(families), ["router_bias"])  # BF16 router popped
        self.assertEqual(sorted(converted), ["router", "router_scale"])
        values, scale = converted["router"], converted["router_scale"]
        self.assertEqual(values.dtype, ml_dtypes.float8_e4m3fn)
        self.assertEqual((values.shape, scale.shape), ((3, 256, 896), (3, 1, 896)))
        self.assertEqual(scale.dtype, np.float32)
        for layer in range(3):
            want_values, want_scale = rank_loader.quantize_columns_fp8(router[layer])
            np.testing.assert_array_equal(
                values[layer].view(np.uint8), want_values.view(np.uint8)
            )
            np.testing.assert_array_equal(scale[layer], want_scale)
        np.testing.assert_array_equal(scale[1, 0, 7], 1)  # all-zero expert column
        # Each expert's largest weight lands on 448.
        live = np.abs(router.astype(np.float32)).max(axis=1) > 0
        peaks = np.abs(values.astype(np.float32)).max(axis=1)
        np.testing.assert_array_equal(peaks[live], 448)

    def test_config_and_colocated_prefill(self):
        from types import SimpleNamespace
        from unittest import mock

        from sglang_tpu.k3_programs import sglang_colocated as colocated
        from sglang_tpu.k3_programs.kimi_k3.config import KernelConfig

        self.assertTrue(KernelConfig(decode_router_fp8=True).decode_router_fp8)
        with self.assertRaises(ValueError):
            KernelConfig(decode_router_fp8=True, attention_dp=2)
        for fp8 in (False, True):
            config = SimpleNamespace(attention_dp=1, decode_router_fp8=fp8)
            with mock.patch(
                "sglang_tpu.k3_programs.kimi_k3.config.active", return_value=config
            ):
                moe = colocated.shared_dense()["moe"]
            self.assertEqual("router" in moe, not fp8)
            self.assertIn("router_bias", moe)


SCRIPT = """
    import dataclasses, os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import ml_dtypes, numpy as np, jax, jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk
    from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

    H, E, K = 7168, 896, 16
    FIELDS = [f.name for f in dataclasses.fields(dmk.MixtureOfExpertsWeights)]
    mesh = Mesh(np.asarray(jax.devices()), ("tp",))

    def run(slots, presorted, hidden, router, bias, scale, live):
        B = hidden.shape[0]
        dtype = router.dtype
        fp8 = scale is not None

        def kernel(live_ref, hidden_ref, bias_ref, *refs):
            if fp8:
                scale_ref, router_ref, *refs = refs
            else:
                scale_ref, (router_ref, *refs) = None, refs
            (raw_out, scores_out, experts_out, probabilities_out, counts_out,
             totals_out, router_vmem, router_semaphores, scores_vmem, sends,
             receives) = refs
            weights = dmk.MixtureOfExpertsWeights(**dict(
                dict.fromkeys(FIELDS), routing_projection=router_ref,
                routing_projection_vmem=router_vmem,
                routing_copy_semaphores=router_semaphores, routing_bias=bias_ref,
                routing_scale=scale_ref))
            rank = jax.lax.axis_index("tp")
            wait_half = weights.wait_for_routing_projection_half if slots == 2 else None

            def load_router():
                # Stale arena bytes (NaN) under every router column, then the
                # arena request's copies; one slot is waited for up front.
                router_vmem[...] = jnp.full(router_vmem.shape, jnp.nan, router_vmem.dtype)
                scores_vmem[...] = jnp.full(scores_vmem.shape, jnp.nan, jnp.float32)
                dmk._router_copy(router_ref, router_vmem, router_semaphores, False,
                                 initialize_padding=False)
                if slots == 1:
                    weights.wait_for_routing_projection_copy()

            load_router()
            dmk._project_routing_scores(rank, hidden_ref[...], router_vmem, scores_vmem,
                                        sends.at[0], receives.at[0],
                                        wait_for_projection_half=wait_half)
            raw_out[...] = scores_vmem[...]
            load_router()
            live_rows = (jnp.stack([live_ref[b] for b in range(B)]) != 0) if live else None
            experts, probabilities, counts, totals = dmk._route_hidden_states(
                hidden_ref[...], router_vmem, bias_ref, scores_vmem, sends.at[0],
                receives.at[0], selected_expert_count=K, local_expert_count=112,
                expert_tensor_parallel=1, on_local_selection=lambda *unused: None,
                after_selection=lambda unused: None, live_rows=live_rows,
                wait_for_projection_half=wait_half, presorted=presorted,
                routing_scale_ref=scale_ref)
            scores_out[...] = scores_vmem[...]
            experts_out[...] = experts
            probabilities_out[...] = probabilities
            counts_out[...] = jnp.broadcast_to(counts[:, None], counts_out.shape)
            totals_out[...] = jnp.broadcast_to(totals[:, None], totals_out.shape)

        vmem = pltpu.VMEM
        shapes = ((B, 1024, jnp.float32), (B, 1024, jnp.float32), (B, K, jnp.int32),
                  (B, K, jnp.float32), (B, 128, jnp.int32), (B, 128, jnp.float32))
        inputs = (hidden, bias, *((scale,) if fp8 else ()), router)

        def body(live_rows, *operands):
            outputs = pl.pallas_call(
                kernel,
                out_shape=tuple(jax.ShapeDtypeStruct(s[:2], s[2]) for s in shapes),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    in_specs=[pl.BlockSpec(memory_space=vmem)] * (len(operands) - 1)
                    + [pl.BlockSpec(memory_space=pl.ANY)],
                    out_specs=[pl.BlockSpec(memory_space=vmem)] * len(shapes),
                    scratch_shapes=[vmem((slots, H, 512), dtype),
                                    pltpu.SemaphoreType.DMA((2,)),
                                    vmem((B, 1024), jnp.float32),
                                    pltpu.SemaphoreType.DMA((1,)),
                                    pltpu.SemaphoreType.DMA((1,))]),
                compiler_params=pltpu.CompilerParams(has_side_effects=True),
                interpret=pltpu.InterpretParams(),
            )(live_rows, *operands)
            return tuple(x[None] for x in outputs)

        f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(),) * (1 + len(inputs)),
                                  out_specs=(P("tp"),) * len(shapes), check_vma=False))
        live_rows = (np.arange(B) % 3 != 1) if live else np.ones(B, bool)
        return [np.asarray(x) for x in f(live_rows.astype(np.int32), *inputs)]

    names = ("raw scores", "final scores", "experts", "probabilities", "counts", "totals")

    def owned_routes(outputs, want, B, live):
        # Rank l holds experts 112 l .. 112 l + 111 (expert_tensor_parallel 1).
        for lane in range(2):
            local, counts = outputs[2][lane], outputs[4][lane, :, 0]
            for row in range(B):
                if live and row % 3 == 1:
                    assert counts[row] == 0, (B, row)
                    continue
                owned = sorted(int(e) + 112 * lane for e in local[row, :counts[row]])
                assert owned == [int(e) for e in want[row] if e // 112 == lane], (B, row)

    def top16(exact, bias):
        corrected = 1 / (1 + np.exp(-exact.astype(np.float64))) + bias[0]
        return np.sort(np.argsort(-corrected, axis=1, kind="stable")[:, :K], axis=1)

    rng = np.random.default_rng(0)
    for B, live in ((8, False), (16, True)):
        hidden = rng.standard_normal((B, H)).astype(ml_dtypes.bfloat16)
        weight = (rng.standard_normal((H, E)) * H**-0.5
                  * rng.uniform(0.25, 4, E)).astype(ml_dtypes.bfloat16)
        bias = (0.1 * rng.standard_normal((1, E))).astype(np.float32)
        # Power-of-two scales (at least amax / 448): the dequantized weights are
        # exact in BF16, so the BF16 router can hold them.
        values, scale = rank_loader.quantize_columns_fp8(weight)
        pow2 = np.exp2(np.ceil(np.log2(scale))).astype(np.float32)
        values2 = (weight.astype(np.float32) / pow2).astype(ml_dtypes.float8_e4m3fn)
        dequantized = (values2.astype(np.float32) * pow2).astype(ml_dtypes.bfloat16)
        assert np.array_equal(dequantized.astype(np.float32),
                              values2.astype(np.float32) * pow2)
        exchanged = run(1, False, hidden, values2, bias, pow2, live)
        projected = run(2, False, hidden, values2, bias, pow2, live)
        presorted = run(2, True, hidden, values2, bias, pow2, live)
        # Both halves on each rank == the pair's score exchange, bit for bit.
        for name, a, b in zip(names, exchanged, projected):
            assert a.tobytes() == b.tobytes(), (B, "both halves", name)
        # The presorted top-k == the argmax loop (the score rows differ: only
        # the argmax loop masks its winners).
        for name, a, b in list(zip(names, projected, presorted))[2:]:
            assert a.tobytes() == b.tobytes(), (B, "presorted", name)
        raw = projected[0]
        assert np.array_equal(raw[0], raw[1]), B  # both lanes hold the same rows
        # Stale NaN bytes under the padding columns never reach the scores.
        assert not np.isnan(raw).any() and not raw[..., E:].any(), B
        # The raw rows are the FP8 products before their expert scales.
        exact = hidden.astype(np.float32) @ dequantized.astype(np.float32)
        scaled = raw[0, :, :E] * pow2[0]
        assert np.abs(scaled - exact).max() <= 1e-5 * np.abs(exact).max(), B
        # Against the BF16 router holding the dequantized weights (the CPU
        # backend sums BF16 x BF16 and BF16 x FP8 products in other orders, so
        # the FP32 sums agree to rounding): the same experts and route counts,
        # probabilities and totals within a few FP32 ulps.
        reference = run(2, False, hidden, dequantized, bias, None, live)
        for index in (2, 4):
            assert np.array_equal(projected[index], reference[index]), (B, names[index])
        for index in (3, 5):
            np.testing.assert_allclose(projected[index], reference[index],
                                       rtol=4 * 2.0**-23, atol=0)
        want = top16(exact, bias)
        owned_routes(projected, want, B, live)
        assert projected[4][..., 0].sum(axis=1).all(), B  # both lanes own routes
        # amax / 448 scales against the FP32 product of the dequantized weights.
        ours = run(2, True, hidden, values, bias, scale, live)
        exact = hidden.astype(np.float32) @ (values.astype(np.float32) * scale)
        scaled = ours[0][0, :, :E] * scale[0]
        assert np.abs(scaled - exact).max() <= 1e-5 * np.abs(exact).max(), B
        owned_routes(ours, top16(exact, bias), B, live)
    print("OK")
"""


class TestRouterFp8Routing(CustomTestCase):
    """The FP8 router in ``_route_hidden_states`` (see the module docstring)."""

    def test_fp8_routes_match_the_dequantized_router(self):
        try:
            from jax.experimental.pallas import tpu as pltpu
        except ImportError:
            pltpu = None
        if not hasattr(pltpu, "InterpretParams"):
            # The TPU nightly runs this file: there it fails instead of skipping.
            self.assertFalse(full_matrix(), "no Pallas TPU interpret mode")
            self.skipTest("no Pallas TPU interpret mode")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1500,
        )
        self.assertIn(
            "OK", result.stdout, result.stdout[-2000:] + result.stderr[-3000:]
        )


if __name__ == "__main__":
    unittest.main()
