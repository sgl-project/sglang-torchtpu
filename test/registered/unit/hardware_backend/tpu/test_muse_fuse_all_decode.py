# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for Muse Spark 1119 Fuse-All decode integration.

Covers default-off env guardrails, TorchTPU op naming/fake registration,
hybrid EP4xTP2 / EP8xTP2 coordinate & weight-sharding equivalence, NVFP4
gmm_v2->fused-RS and replica-bank layout equivalence, MuseSparkMoE
prepare_fused_decode / forward guardrails & fallback, router tie-breaking,
replica planner invariants, and TPU-target Mosaic export.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import fused_rs_sequence_parallel as seq
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.models.muse_spark import (
    MuseSparkForConditionalGeneration,
    MuseSparkMoE,
    MuseSparkRMSNorm,
    _convert_gmm_v2_nvfp4_to_fused_rs_layout,
    _format_nvfp4_replica_bank_from_raw,
    _is_decode_ep_tp2_layout,
    _resolve_ep_tp2_coords,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _make_stub_moe(
    *,
    world: int,
    ep_size: int,
    tp_experts: int,
    tp_rank: int,
    block: int,
    handles_moe_reduction: bool = True,
    norm_dtype: torch.dtype = torch.bfloat16,
    device: str = "meta",
) -> tuple[MuseSparkMoE, SimpleNamespace]:
    """Construct a minimal MuseSparkMoE + parallel namespace for guardrail tests."""
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    nn.Module.__init__(moe)
    moe.tp_size = world
    moe.moe_ep_size = ep_size
    moe.moe_tp_size = tp_experts
    moe._fused_decode_op = None
    moe._fused_decode_batches = (16, 32)
    moe._fused_decode_num_replicas = 0
    moe._fused_decode_mode = ""
    moe._decode_fused_router = False
    moe.quantized_experts = True
    moe.num_experts = 256
    moe.top_k = 8
    moe.expert_hidden_size = 4096
    moe.route_scale = 1.0
    moe.route_eps = 1e-15
    moe.router_multiplier = 0.17677669529663687
    moe.oneshot_gather = False
    moe._manual_post_gather = True
    moe.gate = Mock(
        weight=torch.empty((256, 8192), dtype=torch.float32, device=device),
        side_effect=lambda r: (r[:, :256], None),
    )
    moe.e_score_correction_bias = torch.empty(256, dtype=torch.float32, device=device)
    moe.pre_expert_proj = lambda x: (x[:, :4096], None)
    moe.pre_expert_norm = nn.Identity()
    moe.post_expert_proj = lambda x: (x[:, : 8192 // world], None)
    moe._route = Mock(return_value=SimpleNamespace(topk_weights=None))
    moe._gather_columns = Mock(
        side_effect=lambda x, transpose_2d=True: x.new_empty(
            (x.shape[0], 8192)
            if transpose_2d
            else (world * x.shape[0], 8192 // world)
        )
    )

    e_owner = 256 // ep_size
    i_part = 4096 // tp_experts
    n1 = 2 * i_part
    ex = Mock(handles_moe_reduction=handles_moe_reduction, side_effect=lambda x, r: x)
    ex._buffers = {}
    if tp_experts == 1:
        ex._tpu_fused_rs_group_offset = torch.tensor(
            [tp_rank * e_owner], dtype=torch.int32
        )
    ex._tpu_fused_rs_has_post_expert_norm = True
    ex._tpu_fused_rs_post_expert_norm_weight = torch.empty(
        4096, dtype=norm_dtype, device=device
    )
    fp4 = torch.float4_e2m1fn_x2
    for name, shape, dt in (
        ("w13_weight", (e_owner, 4096, n1 // 2), fp4),
        ("w2_weight", (e_owner, i_part, 2048), fp4),
        ("w13_weight_scale", (e_owner, 4096 // block, n1), torch.bfloat16),
        ("w2_weight_scale", (e_owner, i_part // block, 4096), torch.bfloat16),
    ):
        setattr(ex, name, torch.empty(shape, dtype=dt, device=device))
    moe.experts = ex

    ep_rank, tp_half = (
        (tp_rank, 0) if tp_experts == 1 else _resolve_ep_tp2_coords(tp_rank, ep_size)
    )
    parallel = SimpleNamespace(
        tp_size=world,
        tp_rank=tp_rank,
        moe_ep_size=ep_size,
        moe_ep_rank=ep_rank,
        moe_tp_size=tp_experts,
        moe_tp_rank=tp_half,
        moe_dp_size=1,
        pp_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        attn_dcp_size=1,
    )
    return moe, parallel


class TestMuseFuseAllEnvDefaults(CustomTestCase):
    def test_new_env_flags_default_off(self):
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_EP16.get())
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NATIVE_EP8.get())
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL.get())
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2.get())
        self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS.get(), 0)
        self.assertEqual(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_STATS_PATH.get(), ""
        )
        self.assertEqual(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_COST_MODE.get(), "tiles"
        )
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE.get())
        self.assertFalse(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.get()
        )
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE.get())
        self.assertEqual(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MODE.get(), "full"
        )
        self.assertTrue(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_KMAJOR.get())
        self.assertEqual(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MAX_BS.get(), 128
        )
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_STATIC_HOLDER_MASK.get())
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_BRIDGE64.get())
        self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME.get(), 0)
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL.get())
        self.assertEqual(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SELECTION_MODE.get(),
            "global_top",
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            default_sparse_decode_config,
        )

        with tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME.override(2):
            self.assertEqual(default_sparse_decode_config(64, 8, 256).spec_prime_parts, 2)
            self.assertEqual(default_sparse_decode_config(128, 8, 256).spec_prime_parts, 2)
            self.assertEqual(default_sparse_decode_config(256, 8, 256).spec_prime_parts, 0)

    def test_hybrid_ep_tp2_rejected_by_default_and_gated_when_opted_in(self):
        for ep_size, tp_size in ((4, 8), (8, 16)):
            parallel = SimpleNamespace(
                moe_ep_size=ep_size,
                moe_tp_size=2,
                tp_size=tp_size,
                moe_dp_size=1,
            )
            self.assertFalse(_is_decode_ep_tp2_layout(parallel))
            with patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel):
                with self.assertRaisesRegex(ValueError, "EP or expert TP, not both"):
                    MuseSparkMoE(
                        SimpleNamespace(post_norm_eps=1e-8),
                        layer_idx=1,
                        quant_config=None,
                        prefix="model.layers.1.mlp",
                    )
            with tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2.override(True):
                self.assertTrue(_is_decode_ep_tp2_layout(parallel))
                # Invalid widths stay rejected even when DECODE_EP_TP2=1.
                for bad in (
                    SimpleNamespace(moe_ep_size=2, moe_tp_size=4, tp_size=8, moe_dp_size=1),
                    SimpleNamespace(moe_ep_size=4, moe_tp_size=2, tp_size=16, moe_dp_size=1),
                    SimpleNamespace(moe_ep_size=4, moe_tp_size=2, tp_size=8, moe_dp_size=2),
                    SimpleNamespace(moe_ep_size=2, moe_tp_size=2, tp_size=4, moe_dp_size=1),
                ):
                    self.assertFalse(_is_decode_ep_tp2_layout(bad))


class TestMuseFuseAllOpNamesAndFake(CustomTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch_tpu") and importlib.util.find_spec("jax"),
        "requires TorchTPU and JAX",
    )
    def test_op_naming_validation(self):
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            muse_full_moe_decode_op_name,
            muse_fuse_all_decode_op_name,
        )

        self.assertEqual(
            muse_full_moe_decode_op_name(full_mode="full", ep=8, block=256, num_replicas=0),
            "sglang_tpu_pallas::muse_decode_full_full_ep8_k256_r0",
        )
        self.assertEqual(
            muse_full_moe_decode_op_name(
                full_mode="stage_a",
                ep=16,
                block=16,
                num_replicas=2,
                replica_cost_mode="tiles",
            ),
            "sglang_tpu_pallas::muse_decode_full_stage_a_ep16_k16_r2_tiles",
        )
        self.assertEqual(
            muse_full_moe_decode_op_name(
                full_mode="stage_b",
                ep=8,
                block=256,
                num_replicas=4,
                replica_cost_mode="weighted",
            ),
            "sglang_tpu_pallas::muse_decode_full_stage_b_ep8_k256_r4_weighted",
        )
        for bad_full in (
            dict(full_mode="bogus", ep=8, block=256),
            dict(full_mode="full", ep=4, block=256),
            dict(full_mode="full", ep=8, block=64),
            dict(full_mode="full", ep=8, block=256, num_replicas=3),
            dict(full_mode="full", ep=8, block=256, replica_cost_mode="invalid"),
        ):
            with self.subTest(bad_full=bad_full), self.assertRaises(ValueError):
                muse_full_moe_decode_op_name(**bad_full)

        self.assertEqual(
            muse_fuse_all_decode_op_name(mode="sparse", ep=8, tp=1, block=256, num_replicas=0),
            "sglang_tpu_pallas::muse_decode_sparse_ep8_k256_r0",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(
                mode="sparse", ep=16, tp=1, block=16, num_replicas=2, replica_cost_mode="tiles"
            ),
            "sglang_tpu_pallas::muse_decode_sparse_ep16_k16_r2_tiles",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(
                mode="ep_tp2", ep=4, tp=2, block=256, num_replicas=1, replica_cost_mode="active"
            ),
            "sglang_tpu_pallas::muse_decode_ep4_tp2_k256_r1_active",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(
                mode="ep_tp2", ep=8, tp=2, block=16, num_replicas=0, chipmate_xor=2
            ),
            "sglang_tpu_pallas::muse_decode_ep8_tp2_k16_r0_x2",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(
                mode="chipmate", ep=8, tp=1, block=256, num_replicas=0, chipmate_xor=1
            ),
            "sglang_tpu_pallas::muse_decode_chipmate_ep8_k256_r0_x1",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(
                mode="fp8_sparse", ep=8, tp=1, block=256, num_replicas=0
            ),
            "sglang_tpu_pallas::muse_decode_fp8_sparse_ep8_r0",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(
                mode="fp8_sparse",
                ep=16,
                tp=1,
                block=256,
                num_replicas=2,
                replica_cost_mode="tiles",
            ),
            "sglang_tpu_pallas::muse_decode_fp8_sparse_ep16_r2_tiles",
        )

        for bad_kwargs in (
            dict(mode="invalid", ep=8, tp=1, block=256),
            dict(mode="sparse", ep=4, tp=1, block=256),
            dict(mode="sparse", ep=8, tp=2, block=256),
            dict(mode="ep_tp2", ep=16, tp=2, block=256),
            dict(mode="ep_tp2", ep=4, tp=1, block=256),
            dict(mode="sparse", ep=8, tp=1, block=128),
            dict(mode="sparse", ep=8, tp=1, block=256, num_replicas=3),
            dict(mode="sparse", ep=8, tp=1, block=256, replica_cost_mode="bogus"),
            dict(mode="ep_tp2", ep=4, tp=2, block=256, chipmate_xor=3),
        ):
            with self.subTest(bad_kwargs=bad_kwargs), self.assertRaises(ValueError):
                muse_fuse_all_decode_op_name(**bad_kwargs)

        import jax.numpy as jnp
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            _validate_fuse_all_shapes,
        )

        x_bf16 = SimpleNamespace(shape=(64, 4096), dtype=jnp.bfloat16)
        w1_fp8 = SimpleNamespace(dtype=jnp.dtype("float8_e4m3fn"))
        s1_2d = SimpleNamespace(shape=(32, 8192))
        for fp8_mode in ("fp8_sparse", "sparse_fp8"):
            _validate_fuse_all_shapes(
                "sglang_tpu_pallas::muse_decode_fp8_sparse_ep8_r0",
                fp8_mode,
                256,
                x_bf16,
                w1_fp8,
                s1_2d,
            )
            with self.assertRaises(ValueError):
                _validate_fuse_all_shapes(
                    "sglang_tpu_pallas::muse_decode_fp8_sparse_ep8_r0",
                    fp8_mode,
                    256,
                    x_bf16,
                    x_bf16,
                    s1_2d,
                )


class TestMuseEpTp2ShardingAndLayoutEquivalence(CustomTestCase):
    def test_resolve_ep_tp2_coords_bijection_and_chipmate_pairing(self):
        for ep_size in (4, 8):
            world = ep_size * 2
            for xor in (1, 2, 4):
                seen = set()
                for rank in range(world):
                    chip, half = _resolve_ep_tp2_coords(rank, ep_size, chipmate_xor=xor)
                    mate_chip, mate_half = _resolve_ep_tp2_coords(
                        rank ^ xor, ep_size, chipmate_xor=xor
                    )
                    self.assertEqual(chip, mate_chip)
                    self.assertEqual(half + mate_half, 1)
                    seen.add((chip, half))
                self.assertEqual(len(seen), world)

    def test_ep_tp2_load_expert_stack_matches_unsharded_swiglu_down(self):
        torch.manual_seed(7)
        num_experts, hidden, inter = 8, 16, 12
        w13_full = torch.randn(num_experts, 2 * inter, hidden, dtype=torch.float32)
        w2_full = torch.randn(num_experts, hidden, inter, dtype=torch.float32)
        x = torch.randn(5, hidden, dtype=torch.float32)

        for xor in (1, 2, 4):
            with tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2.override(True):
                # EP4xTP2 across 8 ranks -> 2 experts per chip, inter/2 per chipmate.
                loaded_w13 = {}
                loaded_w2 = {}
                for rank in range(8):
                    parallel = SimpleNamespace(
                        tp_rank=rank,
                        tp_size=8,
                        moe_ep_size=4,
                        moe_ep_rank=rank // 2,
                        moe_tp_size=2,
                        moe_tp_rank=rank % 2,
                        moe_dp_size=1,
                        pp_size=1,
                    )
                    p13 = nn.Parameter(
                        torch.empty(2, inter, hidden, dtype=torch.float32),
                        requires_grad=False,
                    )
                    p2 = nn.Parameter(
                        torch.empty(2, hidden, inter // 2, dtype=torch.float32),
                        requires_grad=False,
                    )
                    with (
                        patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
                        patch(
                            "sglang_tpu.models.muse_spark._default_tpu7x_chipmate_xor",
                            return_value=xor,
                        ),
                    ):
                        MuseSparkForConditionalGeneration._load_expert_stack(
                            SimpleNamespace(),
                            {},
                            p13,
                            w13_full,
                            "model.layers.1.mlp.experts.w13_weight",
                        )
                        MuseSparkForConditionalGeneration._load_expert_stack(
                            SimpleNamespace(),
                            {},
                            p2,
                            w2_full,
                            "model.layers.1.mlp.experts.w2_weight",
                        )
                    loaded_w13[rank] = p13.data.clone()
                    loaded_w2[rank] = p2.data.clone()

            # For every chipmate pair (r, r ^ xor), sum of partial down-projections
            # must equal the unsharded expert output for every expert on that chip.
            for rank in range(8):
                mate = rank ^ xor
                if rank > mate:
                    continue
                chip, _ = _resolve_ep_tp2_coords(rank, 4, chipmate_xor=xor)
                for le in range(2):
                    ge = chip * 2 + le
                    g_ref = x @ w13_full[ge, :inter].t()
                    u_ref = x @ w13_full[ge, inter:].t()
                    z_ref = (torch.nn.functional.silu(g_ref) * u_ref) @ w2_full[ge].t()

                    z_sum = torch.zeros_like(z_ref)
                    for r in (rank, mate):
                        g_p = x @ loaded_w13[r][le, : inter // 2].t()
                        u_p = x @ loaded_w13[r][le, inter // 2 :].t()
                        z_sum = z_sum + (
                            torch.nn.functional.silu(g_p) * u_p
                        ) @ loaded_w2[r][le].t()
                    torch.testing.assert_close(z_sum, z_ref, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_gmm_v2_to_fused_rs_and_replica_formatting_bit_exact(self):
        from sglang_tpu.fp4_moe import prepare_tpu_nvfp4_moe_weights
        from sglang_tpu.moe.nvfp4_requant import compact_tpu_nvfp4_block_scales

        # Verify chunked conversion (> 4 experts) avoids torch.cat on float4_e2m1fn_x2.
        torch.manual_seed(11)
        e, h, i, g = 6, 256, 256, 16
        w13_raw = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)
        w13_s = torch.rand((e, 2 * i, h // g), dtype=torch.float32) * 0.5 + 0.1
        w13_s2 = torch.rand((e, 2), dtype=torch.float32) * 0.5 + 0.5
        w2_raw = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
        w2_s = torch.rand((e, h, i // g), dtype=torch.float32) * 0.5 + 0.1
        w2_s2 = torch.rand((e,), dtype=torch.float32) * 0.5 + 0.5

        def _make_layer():
            layer = nn.Module()
            layer.moe_runner_config = SimpleNamespace(is_gated=True)
            layer.w13_weight = nn.Parameter(w13_raw.clone(), requires_grad=False)
            layer.w13_weight_scale = nn.Parameter(w13_s.clone(), requires_grad=False)
            layer.w13_weight_scale_2 = nn.Parameter(w13_s2.clone(), requires_grad=False)
            layer.w2_weight = nn.Parameter(w2_raw.clone(), requires_grad=False)
            layer.w2_weight_scale = nn.Parameter(w2_s.clone(), requires_grad=False)
            layer.w2_weight_scale_2 = nn.Parameter(w2_s2.clone(), requires_grad=False)
            return layer

        ref_layer = _make_layer()
        gmm_layer = _make_layer()
        with (
            patch("sglang_tpu.fp4_moe.prepare_tpu_fused_rs_post_expert_norm"),
            patch("sglang_tpu.moe.pallas.fused_moe_rs_op.prepare_fused_moe_rs"),
        ):
            prepare_tpu_nvfp4_moe_weights(
                ref_layer,
                group_size=g,
                align_to=256,
                scale_dtype=torch.bfloat16,
                use_fused_rs=True,
            )
            prepare_tpu_nvfp4_moe_weights(
                gmm_layer,
                group_size=g,
                align_to=256,
                scale_dtype=torch.float32,
                use_fused_rs=False,
            )

        _convert_gmm_v2_nvfp4_to_fused_rs_layout(gmm_layer)
        self.assertTrue(gmm_layer.__dict__.get("_tpu_ep_tp2_packed_fp4", False))
        self.assertTrue(
            torch.equal(
                gmm_layer.w13_weight.view(torch.uint8),
                ref_layer.w13_weight.view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                gmm_layer.w2_weight.view(torch.uint8),
                ref_layer.w2_weight.view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(gmm_layer.w13_weight_scale, ref_layer.w13_weight_scale)
        )
        self.assertTrue(
            torch.equal(gmm_layer.w2_weight_scale, ref_layer.w2_weight_scale)
        )

        # Replica bank formatting matches both K16 and compacted K256 bit-for-bit.
        raw_dict = {
            "w13_weight": w13_raw[:2],
            "w13_weight_scale": w13_s[:2],
            "w13_weight_scale_2": w13_s2[:2],
            "w2_weight": w2_raw[:2],
            "w2_weight_scale": w2_s[:2],
            "w2_weight_scale_2": w2_s2[:2],
        }
        rw1, rs1, rw2, rs2 = _format_nvfp4_replica_bank_from_raw(
            raw_dict, device=torch.device("cpu"), compact_k256=False
        )
        self.assertTrue(
            torch.equal(rw1.view(torch.uint8), ref_layer.w13_weight[:2].view(torch.uint8))
        )
        self.assertTrue(torch.equal(rs1, ref_layer.w13_weight_scale[:2]))
        self.assertTrue(
            torch.equal(rw2.view(torch.uint8), ref_layer.w2_weight[:2].view(torch.uint8))
        )
        self.assertTrue(torch.equal(rs2, ref_layer.w2_weight_scale[:2]))

        with patch(
            "sglang_tpu.moe.nvfp4_requant._get_compiled_compact",
            side_effect=lambda: __import__(
                "sglang_tpu.moe.nvfp4_requant", fromlist=["compact_nvfp4_blocks"]
            ).compact_nvfp4_blocks,
        ):
            compact_tpu_nvfp4_block_scales(ref_layer, 256)
        rw1_c, rs1_c, rw2_c, rs2_c = _format_nvfp4_replica_bank_from_raw(
            raw_dict, device=torch.device("cpu"), compact_k256=True
        )
        self.assertTrue(
            torch.equal(
                rw1_c.view(torch.uint8), ref_layer.w13_weight[:2].view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(rs1_c, ref_layer.w13_weight_scale[:2]))
        self.assertTrue(
            torch.equal(
                rw2_c.view(torch.uint8), ref_layer.w2_weight[:2].view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(rs2_c, ref_layer.w2_weight_scale[:2]))


class TestMuseFuseAllModelGuardrailsAndDispatch(CustomTestCase):
    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_prepare_and_forward_across_modes_blocks_and_replicas(self):
        configs = [
            ("sparse", 8, 8, 1, 5, 256, 0),
            ("sparse", 8, 8, 1, 3, 16, 1),
            ("sparse", 16, 16, 1, 9, 256, 2),
            ("ep_tp2", 8, 4, 2, 5, 256, 0),
            ("ep_tp2", 8, 4, 2, 6, 16, 1),
            ("ep_tp2", 16, 8, 2, 11, 256, 2),
            ("chipmate", 8, 8, 1, 2, 256, 0),
        ]
        bridge = "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op"
        for mode, world, ep_size, tp_experts, tp_rank, block, num_replicas in configs:
            with self.subTest(
                mode=mode, world=world, ep=ep_size, tp=tp_experts, block=block, r=num_replicas
            ):
                moe, parallel = _make_stub_moe(
                    world=world,
                    ep_size=ep_size,
                    tp_experts=tp_experts,
                    tp_rank=tp_rank,
                    block=block,
                    handles_moe_reduction=(tp_experts == 1),
                )
                ex = moe.experts
                if num_replicas > 0:
                    moe.attach_preloaded_replicas(
                        rep_w1=torch.empty(
                            (num_replicas, *ex.w13_weight.shape[1:]),
                            dtype=ex.w13_weight.dtype,
                            device="meta",
                        ),
                        rep_s1=torch.empty(
                            (num_replicas, *ex.w13_weight_scale.shape[1:]),
                            dtype=ex.w13_weight_scale.dtype,
                            device="meta",
                        ),
                        rep_w2=torch.empty(
                            (num_replicas, *ex.w2_weight.shape[1:]),
                            dtype=ex.w2_weight.dtype,
                            device="meta",
                        ),
                        rep_s2=torch.empty(
                            (num_replicas, *ex.w2_weight_scale.shape[1:]),
                            dtype=ex.w2_weight_scale.dtype,
                            device="meta",
                        ),
                        replica_meta=torch.zeros(
                            (4, ep_size * num_replicas), dtype=torch.int32, device="meta"
                        ),
                    )
                graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
                seen_arity = []

                def fused(*args):
                    seen_arity.append(len(args))
                    return args[3]

                build = Mock(return_value=fused)
                fallback_build = Mock(side_effect=lambda **kw: Mock(side_effect=lambda *a: a[1]))
                fake_mod = SimpleNamespace(
                    SUPPORTED_FUSE_ALL_BATCHES=(16, 32, 64, 128, 256),
                    get_muse_fuse_all_decode_op=build,
                    get_muse_ep_tp2_fallback_op=fallback_build,
                )
                with (
                    patch.dict("os.environ", WORLD_SIZE=str(world)),
                    patch(
                        "sglang.srt.runtime_context.get_exec",
                        return_value=SimpleNamespace(graph=graph),
                    ),
                    patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch("torch.distributed.get_rank", return_value=tp_rank),
                    patch.dict(sys.modules, {bridge: fake_mod}),
                ):
                    # Negative guardrails must prevent op binding.
                    for obj, key, val in (
                        (graph, "tpu_compile_tracer", "direct"),
                        (graph, "use_spmd_compile", True),
                        (parallel, "pp_size", 2),
                        (parallel, "attn_dp_size", 2),
                        (moe, "_eplb_num_replicas", 1),
                        (moe, "route_scale", 2.0),
                        (ex, "_tpu_nvfp4_qblock", 256),
                    ):
                        with patch.object(obj, key, val, create=True):
                            moe.prepare_fused_decode(mode=mode, num_replicas=num_replicas)
                            build.assert_not_called()
                            self.assertIsNone(moe._fused_decode_op)

                    # Valid bind succeeds.
                    moe.prepare_fused_decode(mode=mode, num_replicas=num_replicas)
                build.assert_called_once_with(
                    mode=mode,
                    ep=ep_size,
                    tp=tp_experts,
                    block=block,
                    num_replicas=num_replicas,
                    replica_cost_mode="tiles",
                    chipmate_xor=None,
                )
                expected_arity = 14 if num_replicas > 0 else 9
                for b, decode, sp, want in (
                    (16, True, False, True),
                    (32, True, False, True),
                    (64, True, False, True),
                    (128, True, False, True),
                    (256, True, False, True),
                    (8, True, False, False),
                    (512, True, False, False),
                    (64, False, False, False),
                    (64, True, True, False),
                ):
                    with patch.object(
                        seq,
                        "get_fused_moe_sequence_parallel_plan",
                        return_value=(False, sp, False),
                    ):
                        seen_arity.clear()
                        moe._route.reset_mock()
                        x = torch.empty((b, 8192), dtype=torch.bfloat16, device="meta")
                        if not want and tp_experts == 2:
                            # Fallback on non-converted meta weights runs _combine_slots.
                            moe._combine_slots = Mock(
                                return_value=torch.empty(
                                    (b, 4096), dtype=torch.bfloat16, device="meta"
                                )
                            )
                        y = moe(
                            x,
                            x.float(),
                            return_untransposed_gather=True,
                            decode=decode,
                        )
                        self.assertEqual(tuple(y.shape), (b * world, 8192 // world))
                        self.assertEqual(bool(seen_arity), want)
                        self.assertEqual(moe._route.called, not want)
                        if want:
                            self.assertEqual(seen_arity, [expected_arity])

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_ep_tp2_packed_fp4_mixed_mode_fallback_and_missing_op_guard(self):
        moe, _ = _make_stub_moe(
            world=8, ep_size=4, tp_experts=2, tp_rank=0, block=256, handles_moe_reduction=False
        )
        moe.experts.__dict__["_tpu_ep_tp2_packed_fp4"] = True
        moe._fused_decode_batches = (16, 32, 64, 128, 256)
        moe.register_buffer("_fused_decode_rank", torch.tensor([0], dtype=torch.int32), persistent=False)
        moe.register_buffer(
            "_fused_decode_norm",
            torch.empty(4096, dtype=torch.bfloat16, device="meta"),
            persistent=False,
        )
        moe._route = Mock(
            side_effect=lambda r: SimpleNamespace(
                topk_weights=torch.empty((r.shape[0], 8), dtype=torch.float32, device="meta"),
                topk_ids=torch.empty((r.shape[0], 8), dtype=torch.int32, device="meta"),
            )
        )
        x = torch.empty((8, 8192), dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(RuntimeError, "_ep_tp2_fallback_op"):
            moe(x, x.float(), decode=True)

        # With _ep_tp2_fallback_op bound, both low-batch decode and chunked prefill work.
        chunk_lens = []

        def fake_fb(rank, hs, tw, ti, *weights):
            chunk_lens.append(hs.shape[0])
            return hs

        moe._ep_tp2_fallback_op = fake_fb
        y_small = moe(x, x.float(), decode=True)
        self.assertEqual(tuple(y_small.shape), (8, 8192))
        self.assertEqual(chunk_lens, [8])

        chunk_lens.clear()
        x_prefill = torch.empty((5000, 8192), dtype=torch.bfloat16, device="meta")
        y_prefill = moe(x_prefill, x_prefill.float(), decode=False)
        self.assertEqual(tuple(y_prefill.shape), (5000, 8192))
        self.assertEqual(chunk_lens, [4096, 904])

    def test_postload_dispatch_and_contradictory_flag_guards(self):
        def _run_postload(overrides: dict, *, has_replica_buf: bool = False):
            prepare = Mock()
            mlp = SimpleNamespace(
                prepare_fused_decode=prepare,
                quantized_experts=False,
                _buffers=(
                    {"_fused_decode_replica_meta": object()} if has_replica_buf else {}
                ),
            )
            root = SimpleNamespace(
                _megakernel_enabled=False,
                start_layer=0,
                end_layer=1,
                model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
                _compact_block256_expert_scales=Mock(),
                _probe_block256_expert_scales=Mock(),
                _finalize_decode_replica_banks=Mock(),
            )
            values = {
                "SGLANG_TPU_MUSE_KERNEL_ENABLED": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_EP16": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NATIVE_EP8": False,
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES": False,
                "SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES": False,
                "SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 0,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_COST_MODE": "tiles",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": False,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE": "",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS": 0,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH": "",
            }
            values.update(overrides)
            env = SimpleNamespace(
                **{k: SimpleNamespace(get=lambda v=v: v) for k, v in values.items()}
            )
            with (
                patch("sglang_tpu.models.muse_spark.tpu_envs", env),
                patch(
                    "sglang_tpu.models.muse_spark.get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode="decode"),
                ),
            ):
                MuseSparkForConditionalGeneration.precompile_kernels_after_loading(root)
            return prepare

        # Default-off: zero fused decode binding.
        self.assertEqual(_run_postload({}).call_count, 0)

        # Sparse fuse-all.
        prep = _run_postload({"SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": True})
        prep.assert_called_once_with(
            mode="sparse", num_replicas=0, replica_cost_mode="tiles"
        )

        # Chipmate rebalance.
        prep = _run_postload(
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE": True,
            }
        )
        prep.assert_called_once_with(
            mode="chipmate", num_replicas=0, replica_cost_mode="tiles"
        )

        # EPxTP2 with replicas present vs missing on layer.
        prep = _run_postload(
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 2,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_COST_MODE": "active",
            },
            has_replica_buf=True,
        )
        prep.assert_called_once_with(
            mode="ep_tp2", num_replicas=2, replica_cost_mode="active"
        )
        prep_no_buf = _run_postload(
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 2,
            },
            has_replica_buf=False,
        )
        prep_no_buf.assert_called_once_with(
            mode="ep_tp2", num_replicas=0, replica_cost_mode="tiles"
        )

        # Full-layer decode MoE dispatch + contradictory flags.
        prep_full = _run_postload({"SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE": True})
        prep_full.assert_called_once_with(
            mode="full_moe", num_replicas=0, replica_cost_mode="tiles"
        )
        prep_sa = _run_postload(
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MODE": "stage_a",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 2,
            },
            has_replica_buf=True,
        )
        prep_sa.assert_called_once_with(
            mode="full_stage_a", num_replicas=2, replica_cost_mode="tiles"
        )

        # Contradictory flags fail loudly.
        with self.assertRaisesRegex(ValueError, "CHIPMATE_REBALANCE"):
            _run_postload(
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": True,
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "DECODE_FULL_MOE"):
            _run_postload(
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE": True,
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "FULL_MOE_MODE"):
            _run_postload(
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE": True,
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MODE": "invalid",
                }
            )
        with self.assertRaisesRegex(ValueError, "REPLICA_SLOTS"):
            _run_postload({"SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 3})
        with self.assertRaisesRegex(ValueError, "CALIB_FORWARDS"):
            _run_postload(
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": True,
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": True,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE": "q_block",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS": 4,
                }
            )
        with self.assertRaisesRegex(ValueError, "PREFILL_EPLB_STATS_PATH"):
            _run_postload(
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": True,
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": True,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE": "q_block",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 2,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH": "/tmp/stats.json",
                }
            )
        with self.assertRaisesRegex(ValueError, "DECODE_EP_TP2"):
            _run_postload(
                {
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": True,
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": True,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE": "q_block",
                }
            )
        prep_eplb = _run_postload(
            {
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": True,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE": "q_block",
            }
        )
        prep_eplb.assert_called_once_with(
            mode="sparse", num_replicas=0, replica_cost_mode="tiles"
        )

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_decode_fuse_target_verify_flag_and_norm_layouts(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang_tpu.models.muse_spark import MuseSparkDecoderLayer

        for norm_layout in (0, 5):
            for allow_tv in (False, True):
                layer = MuseSparkDecoderLayer.__new__(MuseSparkDecoderLayer)
                nn.Module.__init__(layer)
                layer._decode_norm_layout = norm_layout
                layer._decode_fuse_target_verify = allow_tv
                layer.input_layernorm = lambda x: x
                layer.self_attn = lambda p, x, fb: x
                layer.post_attention_layernorm = lambda x: x
                layer.post_attention_residual_gate = lambda s, x: s + x
                layer.pre_feedforward_layernorm = lambda x: x
                layer.post_feedforward_layernorm = lambda x: x
                layer.post_feedforward_residual_gate = lambda s, x: s + x
                layer.pre_feedforward_layernorm.gain = lambda: torch.ones(8192)
                layer.pre_feedforward_layernorm.eps = 1e-5
                layer.pre_feedforward_layernorm.scale_input = False
                layer.post_attention_layernorm.eps = 1e-8
                layer.post_attention_residual_gate.alpha = torch.ones(8192)
                layer.post_attention_residual_gate.beta = torch.ones(8192)
                layer.post_feedforward_residual_gate.alpha = torch.ones(8192)
                layer.post_feedforward_residual_gate.beta = torch.ones(8192)
                layer.post_feedforward_layernorm.gain = lambda: torch.ones(8192)
                layer.post_feedforward_layernorm.eps = 1e-8
                layer.post_feedforward_layernorm.scale_input = True
                layer.input_layernorm.gain = lambda: torch.ones(8192)
                layer.input_layernorm.eps = 1e-5
                layer.input_layernorm.scale_input = False
                layer._dummy_gain = torch.ones(8192)
                next_ln = SimpleNamespace(gain=lambda: torch.ones(8192), eps=1e-5)

                seen_calls = []

                def fake_mlp(hs, router_in, **kw):
                    seen_calls.append((hs.shape[0], kw))
                    return hs

                layer.mlp = Mock(
                    side_effect=fake_mlp,
                    tp_size=8,
                    fuses_post_gather=lambda *a, **kw: False,
                )

                fake_decode_ops = SimpleNamespace(
                    get_muse_decode_rmsnorm_op=lambda: (lambda x, *a: x.bfloat16()),
                    get_muse_decode_res_norm_op=lambda: (lambda s, *a: (s, s)),
                    get_muse_decode_res_only_op=lambda: (lambda s, *a: s),
                    get_muse_decode_tp_ag_res_only_op=lambda: (lambda s, *a: s),
                    get_muse_decode_tp_ag_res_norm_op=lambda: (
                        lambda s, *a: (s, s.bfloat16())
                    ),
                )
                with patch.dict(
                    sys.modules,
                    {"sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router": fake_decode_ops},
                ):
                    for mode, bs, expected_decode in (
                        (ForwardMode.DECODE, 64, True),
                        (ForwardMode.TARGET_VERIFY, 128, allow_tv),
                        (ForwardMode.TARGET_VERIFY, 256, allow_tv),
                        (ForwardMode.EXTEND, 128, False),
                    ):
                        seen_calls.clear()
                        x = torch.empty((bs, 8192), dtype=torch.float32, device="meta")
                        pos = torch.empty((bs,), dtype=torch.int64, device="meta")
                        fb = SimpleNamespace(forward_mode=mode)
                        layer(pos, x, fb, next_input_layernorm=next_ln)
                        self.assertEqual(len(seen_calls), 1)
                        n_tok, kw = seen_calls[0]
                        self.assertEqual(n_tok, bs)
                        self.assertEqual(kw["decode"], expected_decode)
                        if norm_layout == 5 and mode != ForwardMode.EXTEND:
                            self.assertTrue(kw["return_untransposed_gather"])

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_decode_allow_prefill_eplb_passes_weights_zero_copy(self):
        for allow_eplb in (False, True):
            moe, _ = _make_stub_moe(world=8, ep_size=8, tp_experts=1, tp_rank=0, block=256)
            moe._eplb_num_replicas = 2
            moe._decode_allow_prefill_eplb = allow_eplb
            moe._fused_decode_batches = (16, 32, 64, 128, 256)
            moe.register_buffer("_fused_decode_rank", torch.tensor([0], dtype=torch.int32), persistent=False)
            moe.register_buffer(
                "_fused_decode_norm",
                torch.empty(4096, dtype=torch.bfloat16, device="meta"),
                persistent=False,
            )
            ex = moe.experts
            fp4 = torch.float4_e2m1fn_x2
            ex.w13_weight = torch.empty((34, 4096, 4096), dtype=fp4, device="meta")
            ex.w13_weight_scale = torch.empty((34, 16, 8192), dtype=torch.bfloat16, device="meta")
            ex.w2_weight = torch.empty((34, 4096, 2048), dtype=fp4, device="meta")
            ex.w2_weight_scale = torch.empty((34, 16, 4096), dtype=torch.bfloat16, device="meta")
            seen_refs = []

            def fake_fused(*args):
                seen_refs.append((args[4], args[5], args[6], args[7]))
                return args[3]

            moe._fused_decode_op = fake_fused
            with patch.object(
                seq, "get_fused_moe_sequence_parallel_plan", return_value=(False, False, False)
            ):
                x = torch.empty((64, 8192), dtype=torch.bfloat16, device="meta")
                moe(x, x.float(), decode=True)
            self.assertEqual(bool(seen_refs), allow_eplb)
            self.assertEqual(moe._route.called, not allow_eplb)
            if allow_eplb:
                self.assertIs(seen_refs[0][0], ex.w13_weight)
                self.assertIs(seen_refs[0][1], ex.w13_weight_scale)
                self.assertIs(seen_refs[0][2], ex.w2_weight)
                self.assertIs(seen_refs[0][3], ex.w2_weight_scale)

    def test_dflash_fp8_draft_max_rows_guard(self):
        from sglang_tpu import fp8_weight_linear as fwl

        for dflash_fp8, max_rows in ((False, 64), (True, 256)):
            layer = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
            fake_op = Mock(side_effect=lambda x2, w8k, sc: x2.float())
            env = SimpleNamespace(
                SGLANG_TPU_DFLASH_FP8_DRAFT=SimpleNamespace(get=lambda v=dflash_fp8: v)
            )
            with (
                patch.object(fwl, "tpu_envs", env),
                patch.object(fwl, "get_fp8_matmul_op", return_value=fake_op),
            ):
                method = fwl.TpuFp8LinearMethod(layer)
                self.assertEqual(method._max_rows, max_rows)

            for m, expect_op in ((64, True), (128, dflash_fp8), (256, dflash_fp8), (512, False)):
                fake_op.reset_mock()
                x = torch.zeros((m, 128), dtype=torch.bfloat16)
                out = method.apply(layer, x)
                self.assertEqual(tuple(out.shape), (m, 128))
                self.assertEqual(fake_op.called, expect_op)

    def test_verify_rpa_fast_flag_unlocks_d4_and_skip_mixed_when_opted_in(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang_tpu import attention_pallas_rpa_v3 as rpa_v3
        from sglang_tpu import decode_qkv_epilogue_tpu as d4

        layer = SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=16,
            tp_k_head_num=2,
            qk_head_dim=64,
            v_head_dim=64,
            logit_cap=0,
        )
        rotary_emb = SimpleNamespace(cos_sin_pairs=object())
        qkv = torch.empty((128, 1280), dtype=torch.bfloat16, device="meta")

        for allow_fast in (False, True):
            env = SimpleNamespace(
                SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST=SimpleNamespace(
                    get=lambda v=allow_fast: v
                )
            )
            backend = rpa_v3.TpuPallasRpaV3AttnBackend.__new__(
                rpa_v3.TpuPallasRpaV3AttnBackend
            )
            backend._dcp = 1
            backend.kv_cache_quant_dtype = torch.float8_e4m3fn
            backend._batched_prefill_op = None
            backend._skip_decode_op = None
            backend._decode_op = object()
            backend._decode_prepared_op = object()
            backend._fused_out_op = object()
            backend._op = object()
            backend.forward_metadata = SimpleNamespace(
                decode_query_size=2, cp_write_segments=None
            )
            fb_tv = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY, attn_backend=backend
            )
            with (
                patch.object(rpa_v3, "envs", env),
                patch.object(d4, "tpu_envs", env),
            ):
                self.assertEqual(
                    backend.can_forward_decode_prepared(layer, fb_tv), allow_fast
                )
                self.assertEqual(
                    d4.can_use_muse_decode_qkv_epilogue(qkv, rotary_emb, layer, fb_tv),
                    allow_fast,
                )
                self.assertIs(
                    backend._kernel_op(fb_tv, use_causal_mask=True),
                    backend._decode_op if allow_fast else backend._fused_out_op,
                )

        # Verify _invoke_v3_kernel sets (decode_query_size, bkv, decode_query_size, bkv).
        captured_kwargs = []

        def fake_jax_rpa(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return args[0], args[3]

        for sw_aligned in (False, True):
            captured_kwargs.clear()
            env_v3 = SimpleNamespace(
                SGLANG_TPU_MUSE_KERNEL_ENABLED=SimpleNamespace(get=lambda: True),
                SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST=SimpleNamespace(get=lambda: True),
                SGLANG_TPU_RPA_V3_DECODE_BLOCK_SIZES=SimpleNamespace(get=lambda: ""),
                SGLANG_TPU_RPA_V3_DECODE_BLOCK_SIZES_SW=SimpleNamespace(get=lambda: ""),
                SGLANG_TPU_RPA_V3_DECODE_SW_WINDOW_ALIGNED=SimpleNamespace(
                    get=lambda v=sw_aligned: v
                ),
                SGLANG_TPU_RPA_V3_DECODE_ZERO_KV_TAIL=SimpleNamespace(get=lambda: True),
                SGLANG_TPU_RPA_V3_DECODE_RING=SimpleNamespace(get=lambda: ""),
                SGLANG_TPU_RPA_V3_DECODE_FAST=SimpleNamespace(get=lambda: "1"),
                SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES=SimpleNamespace(get=lambda: ""),
                SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES_SW=SimpleNamespace(get=lambda: ""),
                SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE=SimpleNamespace(
                    get=lambda: "m0_ring_fuse"
                ),
                SGLANG_TPU_RPA_V3_PREFILL_FOLD_SCALE=SimpleNamespace(get=lambda: False),
                SGLANG_TPU_RPA_V3_PREFILL_MASK_SKIP=SimpleNamespace(get=lambda: False),
                SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE=SimpleNamespace(get=lambda: True),
            )
            kv_fused = SimpleNamespace(shape=(64, 128, 1, 1, 256), dtype=torch.float8_e4m3fn)
            queries = SimpleNamespace(ndim=5, shape=(1, 256, 8, 2, 128), dtype=torch.bfloat16)
            with patch.object(rpa_v3, "envs", env_v3):
                rpa_v3._invoke_v3_kernel(
                    fake_jax_rpa,
                    queries=queries,
                    keys=None,
                    values=None,
                    kv_cache_fused=kv_fused,
                    kv_lens=None,
                    page_indices=None,
                    cu_q_lens=None,
                    cu_kv_lens=None,
                    distribution=None,
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=2048,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    decode_query_size=4,
                    use_causal_mask=True,
                    fused_out_pack=2,
                    prepared_inputs=True,
                    skip_mixed=True,
                )
            self.assertEqual(len(captured_kwargs), 1)
            expected_bkv = 2176 if sw_aligned else 4096
            self.assertEqual(
                captured_kwargs[0]["d_block_sizes"], (4, expected_bkv, 4, expected_bkv)
            )
            self.assertTrue(captured_kwargs[0].get("_skip_mixed"))
            self.assertTrue(captured_kwargs[0].get("_prepared_inputs"))
            self.assertNotIn("m_block_sizes", captured_kwargs[0])

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_full_moe_prepare_and_forward_zero_copy_weights_and_eplb(self):
        bridge = "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op"
        for mode, submode, num_replicas, extra_eplb, use_kmajor in (
            ("full_moe", "full", 0, 0, True),
            ("full_moe", "full", 2, 2, False),
            ("full_stage_a", "stage_a", 2, 0, False),
            ("full_stage_b", "stage_b", 2, 0, True),
        ):
            with self.subTest(
                mode=mode, r=num_replicas, eplb=extra_eplb, kmajor=use_kmajor
            ):
                moe, parallel = _make_stub_moe(
                    world=8, ep_size=8, tp_experts=1, tp_rank=3, block=256, device="cpu"
                )
                moe.pre_expert_proj = Mock(
                    weight=torch.ones((512, 8192), dtype=torch.bfloat16),
                    side_effect=lambda h: (h[:, :4096], None),
                )
                moe.post_expert_proj = Mock(
                    weight=torch.ones((1024, 4096), dtype=torch.bfloat16),
                    side_effect=lambda h: (h[:, :1024], None),
                )
                moe.pre_expert_norm = MuseSparkRMSNorm(
                    4096, eps=1e-5, gain_center=1.0
                )
                moe.pre_expert_norm.weight.data.zero_()
                moe.pre_expert_norm.precompute_gain()
                ex = moe.experts
                if extra_eplb > 0:
                    w_groups = 32 + extra_eplb
                    ex.w13_weight = torch.empty(
                        (w_groups, 4096, 4096), dtype=torch.float4_e2m1fn_x2
                    )
                    ex.w2_weight = torch.empty(
                        (w_groups, 4096, 2048), dtype=torch.float4_e2m1fn_x2
                    )
                    ex.w13_weight_scale = torch.ones((w_groups, 16, 8192), dtype=torch.bfloat16)
                    ex.w2_weight_scale = torch.ones((w_groups, 16, 4096), dtype=torch.bfloat16)
                    ex._tpu_fused_rs_group_offset = torch.tensor(
                        [3 * w_groups], dtype=torch.int32
                    )
                    moe._eplb_num_replicas = extra_eplb
                    moe._decode_allow_prefill_eplb = True

                if num_replicas > 0:
                    moe.attach_preloaded_replicas(
                        rep_w1=torch.empty(
                            (num_replicas, 4096, 4096), dtype=torch.float4_e2m1fn_x2
                        ),
                        rep_s1=torch.ones((num_replicas, 16, 8192), dtype=torch.bfloat16),
                        rep_w2=torch.empty(
                            (num_replicas, 4096, 2048), dtype=torch.float4_e2m1fn_x2
                        ),
                        rep_s2=torch.ones((num_replicas, 16, 4096), dtype=torch.bfloat16),
                        replica_meta=torch.zeros((4, 8 * num_replicas), dtype=torch.int32),
                    )

                captured = []
                sparse_hi_captured = []

                def fake_full_op(*args):
                    captured.append(args)
                    b_tok = args[2].shape[0] if submode != "stage_a" else args[3].shape[0]
                    out_cols = 4096 if submode == "stage_b" else 8192
                    return torch.zeros((b_tok, out_cols), dtype=torch.bfloat16)

                def fake_sparse_hi(*args):
                    sparse_hi_captured.append(args)
                    return torch.zeros((args[3].shape[0], 4096), dtype=torch.bfloat16)

                build_full = Mock(return_value=fake_full_op)
                build_sparse = Mock(return_value=fake_sparse_hi)
                fake_mod = SimpleNamespace(
                    SUPPORTED_FUSE_ALL_BATCHES=(16, 32, 64, 128, 256),
                    get_muse_full_moe_decode_op=build_full,
                    get_muse_fuse_all_decode_op=build_sparse,
                    get_muse_ep_tp2_fallback_op=Mock(),
                )
                graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
                with (
                    tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_KMAJOR.override(
                        use_kmajor
                    ),
                    tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MAX_BS.override(128),
                    patch.dict("os.environ", WORLD_SIZE="8"),
                    patch(
                        "sglang.srt.runtime_context.get_exec",
                        return_value=SimpleNamespace(graph=graph),
                    ),
                    patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch("torch.distributed.get_rank", return_value=3),
                    patch.dict(sys.modules, {bridge: fake_mod}),
                ):
                    moe.prepare_fused_decode(mode=mode, num_replicas=num_replicas)

                build_full.assert_called_once()
                build_sparse.assert_called_once()
                self.assertEqual(int(moe._fused_decode_rank.item()), 3)

                with patch.object(
                    seq,
                    "get_fused_moe_sequence_parallel_plan",
                    return_value=(False, False, False),
                ):
                    self.assertEqual(
                        moe.fuses_post_gather(64, decode=True),
                        mode in ("full_moe", "full_stage_a"),
                    )
                    self.assertEqual(
                        moe.fuses_post_gather(128, decode=True),
                        mode in ("full_moe", "full_stage_a"),
                    )
                    # B=256 > MAX_BS=128 falls through to fused_decode_sparse (fuses_post_gather=False).
                    self.assertFalse(moe.fuses_post_gather(256, decode=True))
                    self.assertFalse(moe.fuses_post_gather(64, decode=False))
                    x = torch.zeros((64, 8192), dtype=torch.bfloat16)
                    y = moe(x, x.float(), return_untransposed_gather=False, decode=True)
                    self.assertEqual(tuple(y.shape), (64, 8192))
                    self.assertEqual(len(captured), 1)
                    call_args = captured[0]
                    if submode in ("full", "stage_b"):
                        self.assertTrue(call_args[3].is_contiguous())
                        self.assertTrue(call_args[5].is_contiguous())
                        if use_kmajor:
                            self.assertEqual(tuple(call_args[3].shape), (8192, 256))
                            self.assertEqual(tuple(call_args[5].shape), (8192, 512))
                        else:
                            self.assertIs(call_args[3], moe.gate.weight)
                            self.assertIs(call_args[5], moe.pre_expert_proj.weight)
                        self.assertFalse(moe.gate.called)
                        self.assertFalse(moe.pre_expert_proj.called)
                    if submode == "full":
                        self.assertTrue(call_args[12].is_contiguous())
                        if use_kmajor:
                            self.assertEqual(tuple(call_args[12].shape), (4096, 1024))
                        else:
                            self.assertIs(call_args[12], moe.post_expert_proj.weight)
                        self.assertFalse(moe.post_expert_proj.called)
                    elif submode == "stage_a":
                        self.assertIs(call_args[9], moe.post_expert_proj.weight)
                        self.assertFalse(moe.post_expert_proj.called)
                    elif submode == "stage_b":
                        self.assertTrue(moe.post_expert_proj.called)

                    # Verify B=256 > MAX_BS=128 dispatches cleanly to _fused_sparse_hi_op!
                    x256 = torch.zeros((256, 8192), dtype=torch.bfloat16)
                    y256 = moe(
                        x256, x256.float(), return_untransposed_gather=True, decode=True
                    )
                    self.assertEqual(tuple(y256.shape), (256 * 8, 1024))
                    self.assertEqual(len(sparse_hi_captured), 1)
                    self.assertEqual(len(captured), 1)


class TestMuseFuseAllRoutingAndPlannerInvariants(CustomTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("jax"), "requires JAX"
    )
    def test_router_exact_eighth_place_tie_and_perturbations(self):
        import jax.numpy as jnp

        if importlib.util.find_spec(
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing"
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
                route_weights,
            )
        else:
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_experts import (
                route_weights,
            )

        logits = np.full((16, 256), -10.0, dtype=np.float32)
        bias = np.zeros((1, 256), dtype=np.float32)
        # Row 0: 7 clear winners (10..16), exact 8th-place tie between experts 50 and 120.
        logits[0, 10:17] = 5.0
        logits[0, 50] = 2.0
        logits[0, 120] = 2.0
        # Row 1: +1e-6 favors 120.
        logits[1] = logits[0]
        logits[1, 120] = 2.0 + 1e-5
        # Row 2: -1e-6 favors 50.
        logits[2] = logits[0]
        logits[2, 120] = 2.0 - 1e-5
        # Row 3: all -inf -> zero weights (padded row).
        logits[3, :] = -np.inf

        w = np.asarray(route_weights(jnp.asarray(logits), jnp.asarray(bias)))
        self.assertEqual(np.flatnonzero(w[0]).tolist(), [*range(10, 17), 50])
        self.assertEqual(np.flatnonzero(w[1]).tolist(), [*range(10, 17), 120])
        self.assertEqual(np.flatnonzero(w[2]).tolist(), [*range(10, 17), 50])
        np.testing.assert_array_equal(w[3], np.zeros(256, dtype=np.float32))
        for r in (0, 1, 2):
            np.testing.assert_allclose(w[r].sum(), 1.0, rtol=1e-6, atol=1e-6)

        # Verify NaN logits are treated safely as -inf across all routing copies.
        logits_nan = logits.copy()
        logits_nan[4, 0] = np.nan
        logits_nan[5, :] = np.nan
        w_nan = np.asarray(route_weights(jnp.asarray(logits_nan), jnp.asarray(bias)))
        self.assertTrue(np.all(np.isfinite(w_nan)))
        self.assertEqual(w_nan[4, 0], 0.0)
        np.testing.assert_allclose(w_nan[4].sum(), 1.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(w_nan[5], np.zeros(256, dtype=np.float32))

    @unittest.skipUnless(
        importlib.util.find_spec(
            "sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner"
        ),
        "activates once Track 3 moe_imbalance_planner lands",
    )
    def test_replica_planner_invariants_and_monotonicity(self):
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            build_preloaded_replica_plan,
            validate_replica_plan,
        )

        rng = np.random.default_rng(1119)
        zipf_counts = np.round(10000.0 / (np.arange(1, 257) ** 1.15)).astype(np.int64)
        rng.shuffle(zipf_counts)

        for num_owners in (4, 8, 16):
            for num_replicas in (1, 2):
                plan = build_preloaded_replica_plan(
                    zipf_counts,
                    num_owners=num_owners,
                    num_replicas_per_owner=num_replicas,
                )
                validate_replica_plan(plan)
                meta = np.asarray(plan.replica_meta, dtype=np.int32)
                self.assertEqual(meta.shape, (4, num_owners * num_replicas))
                e_per_owner = 256 // num_owners
                used_slots = set()
                used_eids = set()
                for k in range(meta.shape[1]):
                    eid, home, holder, slot = (int(meta[i, k]) for i in range(4))
                    if eid < 0:
                        continue
                    self.assertNotIn(eid, used_eids)
                    used_eids.add(eid)
                    self.assertEqual(home, eid // e_per_owner)
                    self.assertNotEqual(home, holder)
                    self.assertTrue(0 <= holder < num_owners)
                    self.assertTrue(0 <= slot < num_replicas)
                    self.assertNotIn((holder, slot), used_slots)
                    used_slots.add((holder, slot))

        # Verify bottleneck_owner R=2 selection + hot_slot0_order intra-owner sorting.
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            build_static_placement,
            sort_placement_intra_owner_hot0,
        )

        owner_to_chip = np.array([r >> 1 for r in range(8)], dtype=np.int32)
        dual_perm = build_static_placement(
            zipf_counts,
            num_owners=8,
            owner_to_chip=owner_to_chip,
            objective="dual",
            min_gain=0.02,
            hot_slot0_order=True,
        )
        bn_plan = build_preloaded_replica_plan(
            zipf_counts,
            num_owners=8,
            num_replicas_per_owner=2,
            placement=dual_perm,
            owner_to_chip=owner_to_chip,
            selection_mode="bottleneck_owner",
        )
        validate_replica_plan(bn_plan)
        hot0_perm = sort_placement_intra_owner_hot0(
            dual_perm,
            zipf_counts,
            num_owners=8,
            replicated_placed_eids=bn_plan.replica_meta[0],
        )
        # Cross-owner assignment must be 100% preserved by intra-owner sorting!
        for o in range(8):
            self.assertEqual(
                set(dual_perm[o * 32 : (o + 1) * 32].tolist()),
                set(hot0_perm[o * 32 : (o + 1) * 32].tolist()),
            )

    def test_fp8_boundary_moe_decode_gated_by_default_and_binds_when_opted_in(self):
        moe, parallel = _make_stub_moe(
            world=8, ep_size=8, tp_experts=1, tp_rank=0, block=256, device="cpu"
        )
        moe.quantized_experts = False
        ex = moe.experts
        ex.weight_block_size = None
        ex.w13_weight = torch.zeros((32, 4096, 8192), dtype=torch.float8_e4m3fn)
        ex.w2_weight = torch.zeros((32, 4096, 4096), dtype=torch.float8_e4m3fn)
        ex.w13_weight_scale = torch.ones((32, 1, 1, 8192), dtype=torch.bfloat16)
        ex.w2_weight_scale = torch.ones((32, 1, 1, 4096), dtype=torch.bfloat16)

        graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
        seen_shapes = []

        def fake_fp8_op(*args):
            seen_shapes.append((tuple(args[5].shape), tuple(args[7].shape)))
            return args[3]

        build = Mock(return_value=fake_fp8_op)
        fake_mod = SimpleNamespace(
            SUPPORTED_FUSE_ALL_BATCHES=(16, 32, 64, 128, 256),
            get_muse_fuse_all_decode_op=build,
            get_muse_ep_tp2_fallback_op=Mock(),
        )
        bridge = "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op"
        with (
            patch.dict("os.environ", WORLD_SIZE="8"),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(graph=graph),
            ),
            patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch.dict(sys.modules, {bridge: fake_mod}),
        ):
            # Default off -> must NOT bind on non-NVFP4 FP8 boundary layer!
            moe.prepare_fused_decode(mode="sparse", num_replicas=0)
            build.assert_not_called()
            self.assertIsNone(moe._fused_decode_op)

            # Opt-in -> binds fp8_sparse and passes 2D (32, 8192) / (32, 4096) scales!
            with tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL.override(True):
                moe.prepare_fused_decode(mode="sparse", num_replicas=0)
            build.assert_called_once_with(
                mode="fp8_sparse",
                ep=8,
                tp=1,
                block=256,
                num_replicas=0,
                replica_cost_mode="tiles",
            )
            self.assertEqual(moe._fused_decode_mode, "fp8_sparse")
            self.assertEqual(moe._fused_decode_batches, (16, 32, 64, 128))
            with patch.object(
                seq,
                "get_fused_moe_sequence_parallel_plan",
                return_value=(False, False, False),
            ):
                x = torch.zeros((64, 8192), dtype=torch.bfloat16)
                y = moe(x, x.float(), decode=True)
                self.assertEqual(tuple(y.shape), (64, 8192))
                self.assertEqual(seen_shapes, [((32, 8192), (32, 4096))])
                x256 = torch.zeros((256, 8192), dtype=torch.bfloat16)
                y256 = moe(x256, x256.float(), decode=True)
                self.assertEqual(tuple(y256.shape), (256, 8192))
                self.assertEqual(len(seen_shapes), 1)


class TestMuseFuseAllTpuTargetMosaicExport(CustomTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch_tpu") and importlib.util.find_spec("jax"),
        "requires TorchTPU and JAX",
    )
    def test_tpu_target_mosaic_export_available_kernels(self):
        from functools import partial

        import jax
        import jax.numpy as jnp
        from jax import export
        from jax.sharding import Mesh, PartitionSpec as P

        from sglang_tpu import pallas_export
        from sglang_tpu.attention.pallas import util as rpa_util
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        devices = jax.devices()

        def _export_and_check(fn, mesh_size, arg_specs, *, tuple_out: bool = False):
            if len(devices) < mesh_size:
                self.skipTest(
                    f"need {mesh_size} host JAX devices (set XLA_FLAGS=--xla_force_host_platform_device_count={mesh_size})"
                )
            mesh = Mesh(np.asarray(devices[:mesh_size]).reshape(1, mesh_size), ("data", "model"))
            mapped = jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=(P(),) * len(arg_specs),
                out_specs=(P(), P()) if tuple_out else P(),
                check_vma=False,
            )
            with (
                patch.object(
                    pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
                ),
                patch.object(
                    rpa_util, "get_torch_tpu_device_kind", return_value="TPU v7"
                ),
            ):
                with tpu_info_for_cpu_export():
                    ir = export.export(jax.jit(mapped), platforms=("tpu",))(
                        *arg_specs
                    ).mlir_module()
            self.assertEqual(ir.count("stablehlo.custom_call"), 1)
            return ir

        S = jax.ShapeDtypeStruct

        def _base_specs(b, e_owner, i_part, block, r=0, num_owners=8):
            n1 = 2 * i_part
            specs = [
                S((1,), jnp.int32),
                S((b, 256), jnp.float32),
                S((1, 256), jnp.float32),
                S((b, 4096), jnp.bfloat16),
                S((e_owner, 4096, n1), jnp.float4_e2m1fn),
                S((e_owner, 4096 // block, n1), jnp.bfloat16),
                S((e_owner, i_part, 4096), jnp.float4_e2m1fn),
                S((e_owner, i_part // block, 4096), jnp.bfloat16),
                S((1, 4096), jnp.bfloat16),
            ]
            if r > 0:
                specs.extend(
                    [
                        S((r, 4096, n1), jnp.float4_e2m1fn),
                        S((r, 4096 // block, n1), jnp.bfloat16),
                        S((r, i_part, 4096), jnp.float4_e2m1fn),
                        S((r, i_part // block, 4096), jnp.bfloat16),
                        S((4, num_owners * r), jnp.int32),
                    ]
                )
            return tuple(specs)

        # Always verify baseline 5c58d1de1e export AND mixed-mode EPxTP2 fallback export.
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_experts import (
            fused_decode_from_logits,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            ep_tp2_fallback_from_topk,
        )

        if len(devices) >= 8:
            _export_and_check(
                partial(fused_decode_from_logits, ep=8),
                8,
                _base_specs(16, 32, 4096, 16),
            )
            for block in (256, 16):
                for tok in (1, 8, 128):
                    with self.subTest(track="ep_tp2_fallback", ep=4, block=block, tok=tok):
                        fb_specs = (
                            S((1,), jnp.int32),
                            S((tok, 4096), jnp.bfloat16),
                            S((tok, 8), jnp.float32),
                            S((tok, 8), jnp.int32),
                            S((64, 4096, 4096), jnp.float4_e2m1fn),
                            S((64, 4096 // block, 4096), jnp.bfloat16),
                            S((64, 2048, 4096), jnp.float4_e2m1fn),
                            S((64, 2048 // block, 4096), jnp.bfloat16),
                            S((4096,), jnp.bfloat16),
                        )
                        _export_and_check(
                            partial(ep_tp2_fallback_from_topk, ep=4, tp=2, chipmate_xor=1),
                            8,
                            fb_specs,
                        )

        # Track 1 sparse fuse-all export (when merged).
        if importlib.util.find_spec(
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse"
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
                fused_decode_sparse_fp8_from_logits,
                fused_decode_sparse_from_logits,
            )

            for ep in (8, 16):
                if len(devices) < ep:
                    continue
                for batch in (64, 128, 256):
                    for block in (256, 16):
                        for r in (0, 2):
                            with self.subTest(
                                track="sparse", ep=ep, batch=batch, block=block, r=r
                            ):
                                _export_and_check(
                                    partial(fused_decode_sparse_from_logits, ep=ep),
                                    ep,
                                    _base_specs(
                                        batch, 256 // ep, 4096, block, r=r, num_owners=ep
                                    ),
                                )
                for b_fp8 in (16, 64, 256):
                    fp8_specs = (
                        S((1,), jnp.int32),
                        S((b_fp8, 256), jnp.float32),
                        S((1, 256), jnp.float32),
                        S((b_fp8, 4096), jnp.bfloat16),
                        S((256 // ep, 4096, 8192), jnp.float8_e4m3fn),
                        S((256 // ep, 8192), jnp.bfloat16),
                        S((256 // ep, 4096, 4096), jnp.float8_e4m3fn),
                        S((256 // ep, 4096), jnp.bfloat16),
                        S((1, 4096), jnp.bfloat16),
                    )
                    with self.subTest(track="fp8_sparse", ep=ep, b=b_fp8):
                        _export_and_check(
                            partial(fused_decode_sparse_fp8_from_logits, ep=ep),
                            ep,
                            fp8_specs,
                        )
            if len(devices) >= 8:
                with self.subTest(track="sparse_eplb_pad", ep=8, e_groups=34):
                    _export_and_check(
                        partial(fused_decode_sparse_from_logits, ep=8),
                        8,
                        _base_specs(64, 34, 4096, 256, r=2, num_owners=8),
                    )

        # TARGET_VERIFY RPA v3 (d=2, 4) with prepared_inputs + skip_mixed Mosaic export.
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        for d_q, b_tok, sw, bkv in (
            (2, 128, 2048, 2176),
            (4, 256, 2048, 2176),
            (2, 128, None, 2048),
            (4, 256, None, 2048),
        ):
            with self.subTest(
                track="verify_rpa", d_q=d_q, b_tok=b_tok, sw=sw, bkv=bkv
            ):
                rpa_specs = (
                    S((1, b_tok, 8, 2, 128), jnp.bfloat16),
                    S((b_tok, 1, 4, 128), jnp.float8_e4m3fn),
                    S((64, 128, 1, 4, 128), jnp.float8_e4m3fn),
                    S((64,), jnp.int32),
                    S((64 * 129,), jnp.int32),
                    S((65,), jnp.int32),
                    S((65,), jnp.int32),
                    S((3,), jnp.int32),
                )
                def _rpa_fn(
                    q, k, kv_c, kvl, pi, cql, ckl, dist,
                    *, _d_q=d_q, _sw=sw, _bkv=bkv
                ):
                    return ragged_paged_attention(
                        q,
                        k,
                        None,
                        kv_c,
                        kvl,
                        pi,
                        cql,
                        ckl,
                        dist,
                        None,
                        sm_scale=0.125,
                        sliding_window=_sw,
                        k_scale=1.0,
                        v_scale=1.0,
                        decode_query_size=_d_q,
                        d_block_sizes=(_d_q, _bkv, _d_q, _bkv),
                        _prepared_inputs=True,
                        _skip_mixed=True,
                        _fused_out_pack=2,
                    )

                _export_and_check(_rpa_fn, 8, rpa_specs)

        # Track 2 symmetric EPxTP2 export (when merged).
        if importlib.util.find_spec(
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2"
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2 import (
                fused_decode_ep_tp2_from_logits,
            )

            for ep in (4, 8):
                world = ep * 2
                if len(devices) < world:
                    continue
                for block in (256, 16):
                    for r in (0, 1):
                        with self.subTest(track="ep_tp2", ep=ep, block=block, r=r):
                            _export_and_check(
                                partial(fused_decode_ep_tp2_from_logits, ep=ep, tp=2),
                                world,
                                _base_specs(64, 256 // ep, 2048, block, r=r, num_owners=ep),
                            )

        # Track 3 chipmate export (when merged).
        if importlib.util.find_spec(
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_chipmate"
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_chipmate import (
                fused_decode_chipmate_from_logits,
            )

            for ep in (8, 16):
                if len(devices) < ep:
                    continue
                for block in (256, 16):
                    for r in (0, 1):
                        with self.subTest(track="chipmate", ep=ep, block=block, r=r):
                            _export_and_check(
                                partial(fused_decode_chipmate_from_logits, ep=ep),
                                ep,
                                _base_specs(64, 256 // ep, 4096, block, r=r, num_owners=ep),
                            )

        # Wave 2 full-layer boundary-fused decode MoE export (Stage A, Stage B, Full MoE, Full FFN Block).
        if importlib.util.find_spec(
            "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_full_layer"
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_full_layer import (
                fused_decode_full_ffn_block,
                fused_decode_full_moe,
                fused_decode_stage_a_from_logits,
                fused_decode_stage_b_moe,
            )

            def _full_moe_specs(b, ep, block, r=0, e_groups=None):
                eg = (256 // ep) if e_groups is None else e_groups
                specs = [
                    S((1,), jnp.int32),
                    S((b, 8192), jnp.bfloat16),
                    S((b, 8192), jnp.float32),
                    S((256, 8192), jnp.float32),
                    S((1, 256), jnp.float32),
                    S((4096 // ep, 8192), jnp.bfloat16),
                    S((1, 4096), jnp.float32),
                    S((eg, 4096, 8192), jnp.float4_e2m1fn),
                    S((eg, 4096 // block, 8192), jnp.bfloat16),
                    S((eg, 4096, 4096), jnp.float4_e2m1fn),
                    S((eg, 4096 // block, 4096), jnp.bfloat16),
                    S((1, 4096), jnp.bfloat16),
                    S((8192 // ep, 4096), jnp.bfloat16),
                ]
                if r > 0:
                    specs.extend(
                        [
                            S((r, 4096, 8192), jnp.float4_e2m1fn),
                            S((r, 4096 // block, 8192), jnp.bfloat16),
                            S((r, 4096, 4096), jnp.float4_e2m1fn),
                            S((r, 4096 // block, 4096), jnp.bfloat16),
                            S((4, ep * r), jnp.int32),
                        ]
                    )
                return tuple(specs)

            for ep in (8, 16):
                if len(devices) < ep:
                    continue
                for block in (256, 16):
                    for b_tok in (16, 64, 128, 256) if ep == 8 else (64, 128):
                        for r in (0, 2) if b_tok in (64, 128) else (2,):
                            with self.subTest(
                                track="full_moe", ep=ep, block=block, b=b_tok, r=r
                            ):
                                _export_and_check(
                                    partial(fused_decode_full_moe, ep=ep),
                                    ep,
                                    _full_moe_specs(b_tok, ep, block, r=r),
                                )
                            jax.clear_caches()

            if len(devices) >= 8:
                # Zero-copy PREFILL_EPLB (e_groups = 34) + Stage A + Stage B + Full FFN Block.
                with self.subTest(track="full_moe_eplb_pad", ep=8, e_groups=34):
                    _export_and_check(
                        partial(fused_decode_full_moe, ep=8),
                        8,
                        _full_moe_specs(64, 8, 256, r=2, e_groups=34),
                    )
                with self.subTest(track="stage_a", ep=8, b=64, r=2):
                    sa_base = list(_base_specs(64, 32, 4096, 256, r=2, num_owners=8))
                    sa_specs = tuple(
                        sa_base[:9] + [S((1024, 4096), jnp.bfloat16)] + sa_base[9:]
                    )
                    _export_and_check(
                        partial(fused_decode_stage_a_from_logits, ep=8),
                        8,
                        sa_specs,
                    )
                with self.subTest(track="stage_b", ep=8, b=64, r=2):
                    fm_s = list(_full_moe_specs(64, 8, 256, r=2))
                    sb_specs = tuple(fm_s[:12] + fm_s[13:])
                    _export_and_check(
                        partial(fused_decode_stage_b_moe, ep=8),
                        8,
                        sb_specs,
                    )
                for block_blk, b_blk in ((256, 64), (256, 256), (16, 256)):
                    with self.subTest(
                        track="full_ffn_block", ep=8, block=block_blk, b=b_blk, r=2
                    ):
                        fm_b = list(_full_moe_specs(b_blk, 8, block_blk, r=2))
                        blk_specs = tuple(
                            [fm_b[0], S((b_blk, 8192), jnp.float32)]
                            + fm_b[2:13]
                            + [
                                S((8192,), jnp.float32),
                                S((8192,), jnp.float32),
                                S((8192,), jnp.float32),
                                S((8192,), jnp.float32),
                            ]
                            + fm_b[13:]
                        )
                        _export_and_check(
                            partial(fused_decode_full_ffn_block, ep=8),
                            8,
                            blk_specs,
                            tuple_out=True,
                        )
                jax.clear_caches()


if __name__ == "__main__":
    unittest.main()
