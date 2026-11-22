# SPDX-License-Identifier: Apache-2.0
"""CPU unit + 16-device TPU7x Mosaic export tests for Wave-7 Decode MoE optimizations."""

import ast
import functools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.models.muse_spark import (
    MuseSparkMoE,
    MuseSparkRMSNorm,
    _format_fp8_raw_to_nvfp4_replica_bank,
    _format_fp8_replica_bank_from_raw,
)
from sglang_tpu.moe.nvfp4_requant import (
    convert_tpu_fp8_moe_to_nvfp4,
    quantize_fp8_to_nvfp4_blocks,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
    SUPPORTED_REPLICA_SLOTS,
    muse_fuse_all_decode_op_name,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_stub_fp8_moe(*, world: int = 8, e_owner: int = 32):
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    nn.Module.__init__(moe)
    moe.tp_size = world
    moe.moe_ep_size = world
    moe.moe_tp_size = 1
    moe.num_experts = 256
    moe.top_k = 8
    moe.expert_hidden_size = 4096
    moe.route_scale = 1.0
    moe.route_eps = 1e-15
    moe.router_multiplier = 0.17677669529663687
    moe._is_hybrid_ep_tp2 = False
    moe._fused_decode_op = None
    moe._fused_decode_batches = (16, 32)
    moe._fused_decode_v2_op = None
    moe._fused_decode_v2_batches = ()
    moe._fused_decode_num_replicas = 0
    moe._fused_decode_mode = ""
    moe._decode_allow_prefill_eplb = False
    moe._decode_sparse_v2 = False
    moe._decode_sparse_fp8 = False
    moe._decode_fused_router = 1
    moe._manual_post_gather = True
    moe.oneshot_gather = False
    moe.quantized_experts = False

    moe.gate = nn.Module()
    moe.gate.weight = nn.Parameter(
        torch.zeros((256, 8192), dtype=torch.float32), requires_grad=False
    )
    moe.e_score_correction_bias = nn.Parameter(
        torch.zeros((256,), dtype=torch.float32), requires_grad=False
    )
    moe.pre_expert_proj = nn.Module()
    moe.pre_expert_proj.weight = nn.Parameter(
        torch.zeros((4096 // world, 8192), dtype=torch.bfloat16), requires_grad=False
    )
    moe.pre_expert_norm = MuseSparkRMSNorm(4096, eps=1e-5, gain_center=1.0)
    moe.pre_expert_norm.weight.data.zero_()
    moe.pre_expert_norm.precompute_gain()
    moe.post_expert_norm = MuseSparkRMSNorm(
        4096, eps=1e-8, gain_center=1.0, scale_input=True
    )
    moe.post_expert_norm.weight.data.zero_()
    moe.post_expert_norm.precompute_gain()
    moe.post_expert_proj = nn.Module()
    moe.post_expert_proj.weight = nn.Parameter(
        torch.zeros((8192 // world, 4096), dtype=torch.bfloat16), requires_grad=False
    )

    ex = nn.Module()
    ex.handles_moe_reduction = True
    ex._tpu_fused_rs_group_offset = torch.tensor([0], dtype=torch.int32)
    ex._tpu_fused_rs_has_post_expert_norm = True
    ex._tpu_fused_rs_post_expert_norm_weight = torch.zeros(4096, dtype=torch.bfloat16)
    ex.tpu_fused_rs_post_expert_norm_weight = lambda: moe.post_expert_norm.weight.data
    ex.supports_tpu_fused_rs_post_expert_norm = True
    ex.w13_weight = nn.Parameter(
        torch.zeros((e_owner, 4096, 8192), dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    ex.w2_weight = nn.Parameter(
        torch.zeros((e_owner, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    ex.w13_weight_scale = nn.Parameter(
        torch.ones((e_owner, 1, 1, 8192), dtype=torch.float32),
        requires_grad=False,
    )
    ex.w2_weight_scale = nn.Parameter(
        torch.ones((e_owner, 1, 1, 4096), dtype=torch.float32),
        requires_grad=False,
    )
    moe.experts = ex
    parallel = SimpleNamespace(
        tp_size=world,
        tp_rank=0,
        moe_ep_size=world,
        moe_ep_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_dp_size=1,
        pp_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        attn_dcp_size=1,
    )
    graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
    return moe, parallel, graph


class TestMuseW7DecodeMoE(CustomTestCase):
    def test_wave7_env_defaults_and_ast_guards(self):
        self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64.get(), 0)
        self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_MODE.get(), "")
        self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_QUANT_BOUNDARY_MOE_FP4.get(), 0)
        self.assertIs(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_RETURN_SHARDED.get(), False)
        self.assertIs(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SYNC_PREFILL_EPLB.get(), False
        )
        self.assertEqual(SUPPORTED_REPLICA_SLOTS, (0, 1, 2, 4, 6, 8))
        for r in (6, 8):
            self.assertEqual(
                muse_fuse_all_decode_op_name(
                    mode="sparse", ep=8, tp=1, block=256, num_replicas=r
                ),
                f"sglang_tpu_pallas::muse_decode_sparse_ep8_k256_r{r}_tiles",
            )
            self.assertEqual(
                muse_fuse_all_decode_op_name(
                    mode="fp8_sparse", ep=8, tp=1, block=256, num_replicas=r
                ),
                f"sglang_tpu_pallas::muse_decode_fp8_sparse_ep8_r{r}_tiles",
            )

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../..")
        )
        custom_op_files = [
            "sglang_tpu/moe/pallas/fused_rs_muse/decode_norm_router.py",
            "sglang_tpu/moe/pallas/fused_rs_muse/fused_decode_op.py",
            "sglang_tpu/moe/pallas/fused_rs_muse/fused_decode_sparse.py",
        ]
        for rel in custom_op_files:
            with open(os.path.join(repo_root, rel)) as fh:
                tree = ast.parse(fh.read(), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    self.fail(f"PEP 563 __future__ import forbidden in {rel}")

    def test_fp8_to_nvfp4_quant_accuracy_and_modes(self):
        g = torch.Generator().manual_seed(42)
        w_f32 = torch.randn((2, 512, 256), generator=g) * 0.08
        s_f32 = w_f32.abs().amax(dim=(1, 2)).clamp(min=1e-12) / 448.0
        w_fp8 = (w_f32 / s_f32[:, None, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        s_4d = s_f32[:, None, None, None].expand(2, 1, 1, 256).contiguous()

        for block, max_rel_l2 in ((256, 0.13), (16, 0.10)):
            codes, sb, err2, ref2 = quantize_fp8_to_nvfp4_blocks(w_fp8, s_4d, block)
            self.assertEqual(tuple(codes.shape), (2, 512, 128))
            self.assertEqual(codes.dtype, torch.uint8)
            self.assertEqual(tuple(sb.shape), (2, 512 // block, 256))
            self.assertEqual(sb.dtype, torch.bfloat16)
            rel_l2 = float((err2 / ref2).sqrt().item())
            self.assertLess(rel_l2, max_rel_l2)

        # Shadow vs in-place conversion on a small mock FP8 expert module
        for in_place in (False, True):
            ex = nn.Module()
            ex.w13_weight = nn.Parameter(w_fp8.clone(), requires_grad=False)
            ex.w13_weight_scale = nn.Parameter(s_4d.clone(), requires_grad=False)
            ex.w2_weight = nn.Parameter(w_fp8.clone(), requires_grad=False)
            ex.w2_weight_scale = nn.Parameter(s_4d.clone(), requires_grad=False)
            shadow, errs = convert_tpu_fp8_moe_to_nvfp4(ex, 256, in_place=in_place)
            self.assertEqual(len(errs), 2)
            if in_place:
                self.assertEqual(shadow, {})
                self.assertEqual(ex.w13_weight.dtype, torch.float4_e2m1fn_x2)
                self.assertEqual(tuple(ex.w13_weight.shape), (2, 512, 128))
                self.assertEqual(tuple(ex.w13_weight_scale.shape), (2, 2, 256))
            else:
                self.assertEqual(ex.w13_weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(shadow["w13_weight"].dtype, torch.float4_e2m1fn_x2)
                self.assertEqual(tuple(shadow["w13_weight"].shape), (2, 512, 128))

    def test_l0_l61_fp8_replica_banks_and_shadow_fp4_binding(self):
        # 1. Verify _format_fp8_replica_bank_from_raw on small BF16 [R, N, K] slices
        raw_small = {
            "w13_weight": torch.randn((4, 128, 64), dtype=torch.bfloat16) * 0.1,
            "w2_weight": torch.randn((4, 64, 64), dtype=torch.bfloat16) * 0.1,
        }
        rw1, rs1, rw2, rs2 = _format_fp8_replica_bank_from_raw(
            raw_small, device=torch.device("cpu"), scale_dtype=torch.float32
        )
        self.assertEqual(rw1.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(rw1.shape), (4, 64, 128))
        self.assertEqual(tuple(rs1.shape), (4, 1, 1, 128))
        self.assertEqual(tuple(rw2.shape), (4, 64, 64))
        self.assertEqual(tuple(rs2.shape), (4, 1, 1, 64))

        q_rw1, q_rs1, q_rw2, q_rs2 = _format_fp8_raw_to_nvfp4_replica_bank(
            raw_small, device=torch.device("cpu"), block=16
        )
        self.assertEqual(q_rw1.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(tuple(q_rw1.shape), (4, 64, 64))
        self.assertEqual(tuple(q_rs1.shape), (4, 4, 128))
        self.assertEqual(tuple(q_rw2.shape), (4, 64, 32))
        self.assertEqual(tuple(q_rs2.shape), (4, 4, 64))

        # 2. Verify full-shape L0/L61 FP8 MoE binds R=6 and R=8 on BOTH fp8_sparse & sparse_fp8
        for r_slots in (4, 6, 8):
            moe, parallel, graph = _make_stub_fp8_moe()
            moe.attach_preloaded_replicas(
                rep_w1=torch.empty((r_slots, 4096, 8192), dtype=torch.float8_e4m3fn, device="meta"),
                rep_s1=torch.ones((r_slots, 1, 1, 8192), dtype=torch.float32),
                rep_w2=torch.empty((r_slots, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
                rep_s2=torch.ones((r_slots, 1, 1, 4096), dtype=torch.float32),
                replica_meta=torch.zeros((4, 8 * r_slots), dtype=torch.int32),
            )
            fake_op = Mock(name=f"fp8_op_r{r_slots}")
            with (
                patch.dict(
                    os.environ,
                    {
                        "WORLD_SIZE": "8",
                        "SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL": "1",
                    },
                    clear=False,
                ),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_rank", return_value=0),
                patch(
                    "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op.get_muse_fuse_all_decode_op",
                    return_value=fake_op,
                ) as mock_get,
            ):
                moe._prepare_fuse_all_decode(
                    mode="sparse",
                    parallel=parallel,
                    graph=graph,
                    ex=moe.experts,
                    num_replicas=r_slots,
                    replica_cost_mode="tiles",
                    chipmate_xor=None,
                )
            self.assertEqual(moe._fused_decode_mode, "fp8_sparse")
            self.assertEqual(moe._fused_decode_num_replicas, r_slots)
            mock_get.assert_called_once_with(
                mode="fp8_sparse",
                ep=8,
                tp=1,
                block=256,
                num_replicas=r_slots,
                replica_cost_mode="tiles",
            )

            # Also verify DECODE_SPARSE_FP8=1 (_decode_sparse_fp8) preserves num_replicas!
            moe2, parallel2, graph2 = _make_stub_fp8_moe()
            moe2._decode_sparse_fp8 = True
            moe2.attach_preloaded_replicas(
                rep_w1=torch.empty((r_slots, 4096, 8192), dtype=torch.float8_e4m3fn, device="meta"),
                rep_s1=torch.ones((r_slots, 1, 1, 8192), dtype=torch.float32),
                rep_w2=torch.empty((r_slots, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
                rep_s2=torch.ones((r_slots, 1, 1, 4096), dtype=torch.float32),
                replica_meta=torch.zeros((4, 8 * r_slots), dtype=torch.int32),
            )
            with (
                patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_rank", return_value=0),
                patch(
                    "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op.get_muse_fuse_all_decode_op",
                    return_value=fake_op,
                ) as mock_get2,
            ):
                moe2._prepare_fuse_all_decode(
                    mode="sparse",
                    parallel=parallel2,
                    graph=graph2,
                    ex=moe2.experts,
                    num_replicas=r_slots,
                    replica_cost_mode="tiles",
                    chipmate_xor=None,
                )
            self.assertEqual(moe2._fused_decode_mode, "sparse_fp8")
            self.assertEqual(moe2._fused_decode_num_replicas, r_slots)
            mock_get2.assert_called_once_with(
                mode="sparse_fp8",
                ep=8,
                tp=1,
                num_replicas=r_slots,
                replica_cost_mode="tiles",
            )

        # 3. Verify decode-only shadow NVFP4 (_decode_fp4_*) binds NVFP4 sparse op while keeping FP8 ex!
        moe3, parallel3, graph3 = _make_stub_fp8_moe()
        moe3.register_buffer(
            "_decode_fp4_w1",
            torch.empty((32, 4096, 4096), dtype=torch.float4_e2m1fn_x2, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_decode_fp4_s1",
            torch.empty((32, 16, 8192), dtype=torch.bfloat16, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_decode_fp4_w2",
            torch.empty((32, 4096, 2048), dtype=torch.float4_e2m1fn_x2, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_decode_fp4_s2",
            torch.empty((32, 16, 4096), dtype=torch.bfloat16, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_fused_decode_rep_w1",
            torch.empty((4, 4096, 4096), dtype=torch.float4_e2m1fn_x2, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_fused_decode_rep_s1",
            torch.empty((4, 16, 8192), dtype=torch.bfloat16, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_fused_decode_rep_w2",
            torch.empty((4, 4096, 2048), dtype=torch.float4_e2m1fn_x2, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_fused_decode_rep_s2",
            torch.empty((4, 16, 4096), dtype=torch.bfloat16, device="meta"),
            persistent=False,
        )
        moe3.register_buffer(
            "_fused_decode_replica_meta",
            torch.zeros((4, 32), dtype=torch.int32),
            persistent=False,
        )
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op.get_muse_fuse_all_decode_op",
                return_value=Mock(),
            ) as mock_get3,
        ):
            moe3._prepare_fuse_all_decode(
                mode="sparse",
                parallel=parallel3,
                graph=graph3,
                ex=moe3.experts,
                num_replicas=4,
                replica_cost_mode="tiles",
                chipmate_xor=None,
            )
        self.assertFalse(moe3.quantized_experts)
        self.assertEqual(moe3.experts.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(moe3._fused_decode_mode, "sparse")
        self.assertEqual(moe3._fused_decode_num_replicas, 4)
        mock_get3.assert_called_once_with(
            mode="sparse",
            ep=8,
            tp=1,
            block=256,
            num_replicas=4,
            replica_cost_mode="tiles",
            chipmate_xor=None,
        )

    def test_wave7_tpu7x_mosaic_exports(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax import export
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        import sglang_tpu.pallas_export as pallas_export
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            default_fp8_sparse_decode_config,
            default_sparse_decode_config,
            fused_decode_sparse_fp8_from_logits,
            fused_decode_sparse_from_logits,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        devs = jax.devices("cpu")
        if len(devs) < 8:
            self.skipTest(f"Requires >= 8 host CPU XLA devices, got {len(devs)}")

        mesh8 = Mesh(np.array(devs[:8]), ("model",))
        s_rep = NamedSharding(mesh8, P())
        s_mod = NamedSharding(mesh8, P("model"))
        S = jax.ShapeDtypeStruct

        b = 64
        ep = 8
        rank_s = S((ep,), jnp.int32, sharding=s_mod)
        lg_s = S((b, 256), jnp.float32, sharding=s_rep)
        bi_s = S((1, 256), jnp.float32, sharding=s_rep)
        x_s = S((b, 4096), jnp.bfloat16, sharding=s_rep)
        nm_s = S((1, 4096), jnp.bfloat16, sharding=s_rep)

        with (
            patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            tpu_info_for_cpu_export(),
        ):
            # 1. K256 FAST_B64 in (1, 2, 3) & R in (4, 6, 8) + K16 FAST_B64=1
            for block, fast_mode, r_slots in (
                (256, 1, 4),
                (256, 1, 6),
                (256, 1, 8),
                (256, 2, 4),
                (256, 3, 4),
                (16, 1, 4),
            ):
                with patch.dict(
                    os.environ,
                    {"SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64": str(fast_mode)},
                    clear=False,
                ):
                    cfg = default_sparse_decode_config(b, ep, block)
                w1_s = S((256, 4096, 8192), jnp.float4_e2m1fn, sharding=s_mod)
                s1_s = S((256, 4096 // block, 8192), jnp.bfloat16, sharding=s_mod)
                w2_s = S((256, 4096, 4096), jnp.float4_e2m1fn, sharding=s_mod)
                s2_s = S((256, 4096 // block, 4096), jnp.bfloat16, sharding=s_mod)
                rw1_s = S((ep * r_slots, 4096, 8192), jnp.float4_e2m1fn, sharding=s_mod)
                rs1_s = S((ep * r_slots, 4096 // block, 8192), jnp.bfloat16, sharding=s_mod)
                rw2_s = S((ep * r_slots, 4096, 4096), jnp.float4_e2m1fn, sharding=s_mod)
                rs2_s = S((ep * r_slots, 4096 // block, 4096), jnp.bfloat16, sharding=s_mod)
                rmeta_s = S((4, ep * r_slots), jnp.int32, sharding=s_rep)

                fn = jax.jit(
                    jax.shard_map(
                        functools.partial(
                            fused_decode_sparse_from_logits,
                            ep=ep,
                            replica_cost_mode="tiles",
                            config=cfg,
                        ),
                        mesh=mesh8,
                        in_specs=(
                            P("model"), P(), P(), P(),
                            P("model"), P("model"), P("model"), P("model"), P(),
                            P("model"), P("model"), P("model"), P("model"), P(),
                        ),
                        out_specs=P(),
                        check_vma=False,
                    )
                )
                exp = export.export(fn, platforms=["tpu"])(
                    rank_s, lg_s, bi_s, x_s, w1_s, s1_s, w2_s, s2_s, nm_s,
                    rw1_s, rs1_s, rw2_s, rs2_s, rmeta_s,
                )
                self.assertIn("stablehlo.custom_call", exp.mlir_module())

            # 2. FP8 w1_k2_w2_1x, w1_k2_w8a8, dense_k2 at R in (4, 8)
            w1_8 = S((256, 4096, 8192), jnp.float8_e4m3fn, sharding=s_mod)
            s1_8 = S((256, 8192), jnp.float32, sharding=s_mod)
            w2_8 = S((256, 4096, 4096), jnp.float8_e4m3fn, sharding=s_mod)
            s2_8 = S((256, 4096), jnp.float32, sharding=s_mod)
            for fp8_m, r_slots in (
                ("w1_k2_w2_1x", 4),
                ("w1_k2_w2_1x", 8),
                ("w1_k2_w8a8", 4),
                ("dense_k2", 4),
            ):
                with patch.dict(
                    os.environ,
                    {"SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_MODE": fp8_m},
                    clear=False,
                ):
                    fcfg = default_fp8_sparse_decode_config(b, ep)
                rw1_8 = S((ep * r_slots, 4096, 8192), jnp.float8_e4m3fn, sharding=s_mod)
                rs1_8 = S((ep * r_slots, 8192), jnp.float32, sharding=s_mod)
                rw2_8 = S((ep * r_slots, 4096, 4096), jnp.float8_e4m3fn, sharding=s_mod)
                rs2_8 = S((ep * r_slots, 4096), jnp.float32, sharding=s_mod)
                rmeta_8 = S((4, ep * r_slots), jnp.int32, sharding=s_rep)
                fn8 = jax.jit(
                    jax.shard_map(
                        functools.partial(
                            fused_decode_sparse_fp8_from_logits,
                            ep=ep,
                            replica_cost_mode="tiles",
                            config=fcfg,
                        ),
                        mesh=mesh8,
                        in_specs=(
                            P("model"), P(), P(), P(),
                            P("model"), P("model"), P("model"), P("model"), P(),
                            P("model"), P("model"), P("model"), P("model"), P(),
                        ),
                        out_specs=P(),
                        check_vma=False,
                    )
                )
                exp8 = export.export(fn8, platforms=["tpu"])(
                    rank_s, lg_s, bi_s, x_s, w1_8, s1_8, w2_8, s2_8, nm_s,
                    rw1_8, rs1_8, rw2_8, rs2_8, rmeta_8,
                )
                self.assertIn("stablehlo.custom_call", exp8.mlir_module())

            # 3. Token-sharded output (return_sharded=True -> [B // ep, 4096]) Mosaic export
            w1_sh = S((256, 4096, 8192), jnp.float4_e2m1fn, sharding=s_mod)
            s1_sh = S((256, 16, 8192), jnp.bfloat16, sharding=s_mod)
            w2_sh = S((256, 4096, 4096), jnp.float4_e2m1fn, sharding=s_mod)
            s2_sh = S((256, 16, 4096), jnp.bfloat16, sharding=s_mod)
            rw1_sh = S((ep * 4, 4096, 8192), jnp.float4_e2m1fn, sharding=s_mod)
            rs1_sh = S((ep * 4, 16, 8192), jnp.bfloat16, sharding=s_mod)
            rw2_sh = S((ep * 4, 4096, 4096), jnp.float4_e2m1fn, sharding=s_mod)
            rs2_sh = S((ep * 4, 16, 4096), jnp.bfloat16, sharding=s_mod)
            rmeta_sh = S((4, ep * 4), jnp.int32, sharding=s_rep)
            fn_sh = jax.jit(
                jax.shard_map(
                    functools.partial(
                        fused_decode_sparse_from_logits,
                        ep=ep,
                        return_sharded=True,
                        replica_cost_mode="tiles",
                        config=default_sparse_decode_config(b, ep, 256),
                    ),
                    mesh=mesh8,
                    in_specs=(
                        P("model"), P(), P(), P(),
                        P("model"), P("model"), P("model"), P("model"), P(),
                        P("model"), P("model"), P("model"), P("model"), P(),
                    ),
                    out_specs=P("model", None),
                    check_vma=False,
                )
            )
            exp_sh = export.export(fn_sh, platforms=["tpu"])(
                rank_s, lg_s, bi_s, x_s, w1_sh, s1_sh, w2_sh, s2_sh, nm_s,
                rw1_sh, rs1_sh, rw2_sh, rs2_sh, rmeta_sh,
            )
            self.assertIn("stablehlo.custom_call", exp_sh.mlir_module())

            # 4. Wave-7B: EPLB-expanded [8 * 36, ...] L0/L61 FP8 slabs + Pallas EPLB router (T=1, 8)
            w1_eplb8 = S((ep * 36, 4096, 8192), jnp.float8_e4m3fn, sharding=s_mod)
            s1_eplb8 = S((ep * 36, 1, 1, 8192), jnp.float32, sharding=s_mod)
            w2_eplb8 = S((ep * 36, 4096, 4096), jnp.float8_e4m3fn, sharding=s_mod)
            s2_eplb8 = S((ep * 36, 1, 1, 4096), jnp.float32, sharding=s_mod)
            rw1_eplb8 = S((ep * 4, 4096, 8192), jnp.float8_e4m3fn, sharding=s_mod)
            rs1_eplb8 = S((ep * 4, 1, 1, 8192), jnp.float32, sharding=s_mod)
            rw2_eplb8 = S((ep * 4, 4096, 4096), jnp.float8_e4m3fn, sharding=s_mod)
            rs2_eplb8 = S((ep * 4, 1, 1, 4096), jnp.float32, sharding=s_mod)
            rmeta_eplb8 = S((4, ep * 4), jnp.int32, sharding=s_rep)
            fn_eplb8 = jax.jit(
                jax.shard_map(
                    functools.partial(
                        fused_decode_sparse_fp8_from_logits,
                        ep=ep,
                        replica_cost_mode="tiles",
                        config=default_fp8_sparse_decode_config(b, ep),
                    ),
                    mesh=mesh8,
                    in_specs=(
                        P("model"), P(), P(), P(),
                        P("model"), P("model"), P("model"), P("model"), P(),
                        P("model"), P("model"), P("model"), P("model"), P(),
                    ),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            exp_eplb8 = export.export(fn_eplb8, platforms=["tpu"])(
                rank_s, lg_s, bi_s, x_s, w1_eplb8, s1_eplb8, w2_eplb8, s2_eplb8, nm_s,
                rw1_eplb8, rs1_eplb8, rw2_eplb8, rs2_eplb8, rmeta_eplb8,
            )
            self.assertIn("stablehlo.custom_call", exp_eplb8.mlir_module())

            from sglang_tpu.kernels.router_topk import router_topk_jax

            for t_tok in (1, 8):
                fn_rt = jax.jit(
                    lambda lg, bi, _t=t_tok: router_topk_jax(
                        lg,
                        bi,
                        8,
                        True,
                        1e-15,
                        1.0,
                        eplb_replicas=4,
                        ep_size=8,
                        num_base_local=32,
                    )
                )
                exp_rt = export.export(fn_rt, platforms=["tpu"])(
                    S((t_tok, 256), jnp.float32), S((256,), jnp.float32)
                )
                self.assertIn("stablehlo.custom_call", exp_rt.mlir_module())

    def test_return_sharded_op_and_bridge_handoff(self):
        for mode, expected in (
            ("sparse", "sglang_tpu_pallas::muse_decode_sparse_ep8_k256_r4_tiles_sh"),
            ("sparse_v2", "sglang_tpu_pallas::muse_decode_sparse_v2_ep8_k256_r4_tiles_sh"),
            ("fp8_sparse", "sglang_tpu_pallas::muse_decode_fp8_sparse_ep8_r4_tiles_sh"),
            ("sparse_fp8", "sglang_tpu_pallas::muse_decode_sparse_fp8_ep8_r4_tiles_sh"),
        ):
            self.assertEqual(
                muse_fuse_all_decode_op_name(
                    mode=mode, ep=8, tp=1, block=256, num_replicas=4, return_sharded=True
                ),
                expected,
            )
        with self.assertRaises(ValueError):
            muse_fuse_all_decode_op_name(
                mode="ep_tp2", ep=4, tp=2, block=256, num_replicas=0, return_sharded=True
            )

        moe, parallel, graph = _make_stub_fp8_moe()
        moe.attach_preloaded_replicas(
            rep_w1=torch.empty((4, 4096, 8192), dtype=torch.float8_e4m3fn, device="meta"),
            rep_s1=torch.ones((4, 1, 1, 8192), dtype=torch.float32),
            rep_w2=torch.empty((4, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
            rep_s2=torch.ones((4, 1, 1, 4096), dtype=torch.float32),
            replica_meta=torch.zeros((4, 32), dtype=torch.int32),
        )
        rep_out = torch.full((64, 4096), 2.0, dtype=torch.bfloat16)
        sh_out = torch.full((8, 4096), 3.0, dtype=torch.bfloat16)
        mock_rep_op = Mock(return_value=rep_out)
        mock_sh_op = Mock(return_value=sh_out)

        def _fake_get_op(**kwargs):
            return mock_sh_op if kwargs.get("return_sharded", False) else mock_rep_op

        with (
            patch.dict(
                os.environ,
                {
                    "WORLD_SIZE": "8",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL": "1",
                },
                clear=False,
            ),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op.get_muse_fuse_all_decode_op",
                side_effect=_fake_get_op,
            ),
        ):
            moe._prepare_fuse_all_decode(
                mode="sparse",
                parallel=parallel,
                graph=graph,
                ex=moe.experts,
                num_replicas=4,
                replica_cost_mode="tiles",
                chipmate_xor=None,
            )
            self.assertFalse(moe.can_return_sharded_moe_out(64, decode=True))
            self.assertTrue(moe.enable_sharded_moe_out())
            self.assertTrue(moe.can_return_sharded_moe_out(64, decode=True))
            self.assertFalse(moe.can_return_sharded_moe_out(64, decode=False))

        moe.gate.forward = Mock(
            side_effect=AssertionError("gate should not run with precomputed_moe_inputs")
        )
        moe.post_expert_proj.forward = Mock(
            side_effect=AssertionError("post_expert_proj should not run with return_sharded_moe_out")
        )
        scaled_lg = torch.zeros((64, 256), dtype=torch.float32)
        exp_in = torch.zeros((64, 4096), dtype=torch.bfloat16)
        with patch(
            "sglang_tpu.fused_rs_sequence_parallel.get_fused_moe_sequence_parallel_plan",
            return_value=(),
        ):
            out = moe(
                None,
                None,
                decode=True,
                precomputed_moe_inputs=(scaled_lg, exp_in),
                return_sharded_moe_out=True,
            )
        self.assertEqual(tuple(out.shape), (8, 4096))
        mock_sh_op.assert_called_once()
        mock_rep_op.assert_not_called()

    def test_wave7b_eplb_router_topk_bit_exactness_and_dispatch(self):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.kernels.router_topk import router_topk_jax
        from sglang_tpu.moe.pallas.fused_rs_muse import decode_norm_router as dnr
        from sglang_tpu.muse_prefill_kernels.eplb import map_torch_fallback_topk_ids

        rng = np.random.default_rng(1120)
        bias_np = (rng.standard_normal(256) * 0.05).astype(np.float32)
        bias_j = jnp.asarray(bias_np)

        # 1. Bit-exactness across T=1 (_topk_kernel_1tok) and T=8,16,64 (_topk_kernel),
        # random + tied logits, f32 + bf16, and R in (1, 2, 4, 8).
        for t_tok in (1, 8, 16, 64):
            lg_np = rng.standard_normal((t_tok, 256)).astype(np.float32) * 0.7
            # Inject exact ties across replica slots (0..7) and base slots (32..39)
            lg_np[:, 1] = lg_np[:, 0]
            lg_np[:, 34] = lg_np[:, 33]
            bias_tied = bias_np.copy()
            bias_tied[1] = bias_tied[0]
            bias_tied[34] = bias_tied[33]
            bias_tied_j = jnp.asarray(bias_tied)

            for dt in (jnp.float32, jnp.bfloat16):
                lg_j = jnp.asarray(lg_np).astype(dt)
                w_ref, ids_ref0 = router_topk_jax(
                    lg_j, bias_tied_j, 8, True, 1e-15, 1.25, interpret=True
                )
                ids_t0 = torch.from_numpy(np.array(ids_ref0, copy=True))
                for r_rep in (1, 2, 4, 8):
                    w_got, ids_got = router_topk_jax(
                        lg_j,
                        bias_tied_j,
                        8,
                        True,
                        1e-15,
                        1.25,
                        interpret=True,
                        eplb_replicas=r_rep,
                        ep_size=8,
                        num_base_local=32,
                    )
                    ids_exp = map_torch_fallback_topk_ids(
                        ids_t0, num_replicas=r_rep, ep_size=8, num_base_local=32
                    ).numpy()
                    np.testing.assert_array_equal(np.asarray(w_got), np.asarray(w_ref))
                    np.testing.assert_array_equal(np.asarray(ids_got), ids_exp)

        # 2. Verify _route_scaled_logits dispatches get_muse_decode_router_topk_op(R, 8)
        # with ZERO torch.topk fallback when _eplb_num_replicas > 0, and 0-arg when R == 0.
        moe, _, _ = _make_stub_fp8_moe()
        moe._eplb_num_replicas = 4
        moe._eplb_ep_size = 8

        fake_eplb_op = Mock(
            side_effect=lambda lg, b, k, eps, sc: (
                torch.full((lg.shape[0], k), 0.125, dtype=torch.float32),
                torch.zeros((lg.shape[0], k), dtype=torch.int32),
            )
        )
        fake_base_op = Mock(
            side_effect=lambda lg, b, k, eps, sc: (
                torch.full((lg.shape[0], k), 0.125, dtype=torch.float32),
                torch.ones((lg.shape[0], k), dtype=torch.int32),
            )
        )

        def _router_getter(*args):
            if not args or args[0] == 0:
                return fake_base_op
            self.assertEqual(args, (4, 8))
            return fake_eplb_op

        with (
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router.get_muse_decode_router_topk_op",
                side_effect=_router_getter,
            ),
            patch(
                "torch.topk",
                side_effect=AssertionError("torch.topk fallback must not run!"),
            ),
        ):
            for n_tok in (1, 8, 64):
                out_r4 = moe._route_scaled_logits(
                    torch.zeros((n_tok, 256), dtype=torch.float32)
                )
                self.assertEqual(tuple(out_r4.topk_ids.shape), (n_tok, 8))
                self.assertTrue(torch.all(out_r4.topk_ids == 0))
            self.assertEqual(fake_eplb_op.call_count, 3)
            fake_base_op.assert_not_called()

            moe._eplb_num_replicas = 0
            out_r0 = moe._route_scaled_logits(torch.zeros((8, 256), dtype=torch.float32))
            self.assertTrue(torch.all(out_r0.topk_ids == 1))
            fake_base_op.assert_called_once()

        # 3. Real registration caches (4, 8) inside _ROUTER_EPLB_OPS for lock-free cold compile.
        op_4_8 = dnr.get_muse_decode_router_topk_op(4, 8)
        self.assertIs(dnr._ROUTER_EPLB_OPS.get((4, 8)), op_4_8)
        self.assertIs(dnr.get_muse_decode_router_topk_op(4, 8), op_4_8)

    def test_wave7b_l0_l61_eplb_slabs_and_replica_sync(self):
        import numpy as np

        from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration
        from sglang_tpu.muse_prefill_kernels.eplb import PrefillEplbManager

        # 1. L0/L61 FP8 MoE with EPLB-expanded [36, ...] slabs binds fp8_sparse & sparse_fp8
        # when DECODE_ALLOW_PREFILL_EPLB=1, and extracts [32, N] base scales + rank.
        for use_v2_fp8, exp_mode in ((False, "fp8_sparse"), (True, "sparse_fp8")):
            moe, parallel, graph = _make_stub_fp8_moe(world=8, e_owner=36)
            parallel.tp_rank = 3
            parallel.moe_ep_rank = 3
            moe.experts._tpu_fused_rs_group_offset = torch.tensor([3 * 36], dtype=torch.int32)
            moe._decode_sparse_fp8 = use_v2_fp8
            moe.attach_preloaded_replicas(
                rep_w1=torch.empty((4, 4096, 8192), dtype=torch.float8_e4m3fn, device="meta"),
                rep_s1=torch.ones((4, 1, 1, 8192), dtype=torch.float32),
                rep_w2=torch.empty((4, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
                rep_s2=torch.ones((4, 1, 1, 4096), dtype=torch.float32),
                replica_meta=torch.zeros((4, 32), dtype=torch.int32),
            )
            env_map = {
                "WORLD_SIZE": "8",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": "1",
            }
            if not use_v2_fp8:
                env_map["SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL"] = "1"
            with (
                patch.dict(os.environ, env_map, clear=False),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_rank", return_value=3),
                patch(
                    "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op.get_muse_fuse_all_decode_op",
                    return_value=Mock(),
                ),
            ):
                moe._decode_allow_prefill_eplb = True
                moe._prepare_fuse_all_decode(
                    mode="sparse",
                    parallel=parallel,
                    graph=graph,
                    ex=moe.experts,
                    num_replicas=4,
                    replica_cost_mode="tiles",
                    chipmate_xor=None,
                )
            self.assertEqual(moe._fused_decode_mode, exp_mode)
            self.assertEqual(moe._fused_decode_num_replicas, 4)
            self.assertEqual(int(moe._fused_decode_rank.item()), 3)
            if not use_v2_fp8:
                self.assertEqual(tuple(moe._fused_decode_fp8_s1.shape), (32, 8192))
                self.assertEqual(tuple(moe._fused_decode_fp8_s2.shape), (32, 4096))

        # 2. DECODE_SYNC_PREFILL_EPLB=True unlocks DECODE_REPLICA_SLOTS>0 + PREFILL_EPLB_STATS_PATH,
        # while default False raises ValueError matching PREFILL_EPLB_STATS_PATH.
        def _run_postload(sync_flag: bool):
            prepare = Mock()
            mlp = SimpleNamespace(
                prepare_fused_decode=prepare,
                quantized_experts=False,
                _buffers={"_fused_decode_replica_meta": object()},
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
                "SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS": 4,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_COST_MODE": "tiles",
                "SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE": False,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB": True,
                "SGLANG_TPU_MUSE_KERNEL_DECODE_SYNC_PREFILL_EPLB": sync_flag,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE": "zeroglue",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS": 0,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH": "/tmp/stats.json",
            }
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

        with self.assertRaisesRegex(ValueError, "PREFILL_EPLB_STATS_PATH"):
            _run_postload(False)
        prep_sync = _run_postload(True)
        prep_sync.assert_called_once_with(
            mode="sparse", num_replicas=4, replica_cost_mode="tiles"
        )

        # 3. Verify PrefillEplbManager._apply_layer_perm_indices & _refresh_decode_replicas
        # refresh _fused_decode_fp8_s1/s2 AND forward selection_mode=bottleneck_owner_b64!
        ex_fp8 = nn.Module()
        ex_fp8.num_local_experts = 36
        ex_fp8._tpu_fused_rs_group_offset = torch.tensor([0], dtype=torch.int32)
        ex_fp8.w13_weight = nn.Parameter(
            torch.zeros((36, 4, 8), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        ex_fp8.w2_weight = nn.Parameter(
            torch.zeros((36, 4, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        s1_vals = torch.arange(36, dtype=torch.float32).view(36, 1, 1, 1).expand(36, 1, 1, 8).clone()
        s2_vals = (torch.arange(36, dtype=torch.float32) + 100.0).view(36, 1, 1, 1).expand(36, 1, 1, 4).clone()
        ex_fp8.w13_weight_scale = nn.Parameter(s1_vals, requires_grad=False)
        ex_fp8.w2_weight_scale = nn.Parameter(s2_vals, requires_grad=False)

        moe_fp8 = nn.Module()
        moe_fp8.experts = ex_fp8
        moe_fp8.gate = nn.Module()
        moe_fp8.gate.weight = nn.Parameter(
            torch.arange(256, dtype=torch.float32).view(256, 1).expand(256, 8).clone(),
            requires_grad=False,
        )
        moe_fp8.e_score_correction_bias = nn.Parameter(
            torch.arange(256, dtype=torch.float32), requires_grad=False
        )
        moe_fp8._fused_decode_num_replicas = 2
        moe_fp8.register_buffer(
            "_fused_decode_fp8_s1", torch.zeros((32, 8), dtype=torch.float32)
        )
        moe_fp8.register_buffer(
            "_fused_decode_fp8_s2", torch.zeros((32, 4), dtype=torch.float32)
        )
        moe_fp8.register_buffer(
            "_fused_decode_rep_w1", torch.zeros((2, 4, 8), dtype=torch.float8_e4m3fn)
        )
        moe_fp8.register_buffer(
            "_fused_decode_rep_s1", torch.zeros((2, 1, 1, 8), dtype=torch.float32)
        )
        moe_fp8.register_buffer(
            "_fused_decode_rep_w2", torch.zeros((2, 4, 4), dtype=torch.float8_e4m3fn)
        )
        moe_fp8.register_buffer(
            "_fused_decode_rep_s2", torch.zeros((2, 1, 1, 4), dtype=torch.float32)
        )
        moe_fp8.register_buffer(
            "_fused_decode_replica_meta", torch.zeros((4, 16), dtype=torch.int32)
        )

        decoder_stub = SimpleNamespace(
            norm=SimpleNamespace(weight=torch.zeros(4)),
            layers=[SimpleNamespace(mlp=moe_fp8)],
        )
        lw_stub = SimpleNamespace(wrouter=torch.zeros((8, 256), dtype=torch.float32))

        def _fake_gather_8(x_chunk: torch.Tensor) -> torch.Tensor:
            # Tile rank-0 chunk across 8 synthetic TP ranks with distinct offsets if float32
            if x_chunk.dtype == torch.float32 and x_chunk.ndim >= 2:
                parts = [x_chunk + float(r * 1000) for r in range(8)]
                return torch.cat(parts, dim=0)
            return x_chunk.repeat(8, *([1] * (x_chunk.dim() - 1)))

        mgr = PrefillEplbManager.__new__(PrefillEplbManager)
        mgr.decoder = decoder_stub
        mgr.weights_list = [lw_stub]
        mgr.gather_fn = _fake_gather_8
        mgr.start = 0
        mgr.end = 1
        mgr.num_layers = 1
        mgr.tp_size = 8
        mgr.tp_rank = 0
        mgr.mode = "zeroglue"
        mgr.num_replicas = 4
        mgr.dual_decode = True
        mgr.decode_counts_by_layer = {}
        mgr.num_base_local = 32
        mgr.num_experts = 256
        mgr.e_local = 36
        mgr.current_col_perms = [np.arange(256, dtype=np.int32)]

        # Swap expert 0 <-> expert 35 (which lives on rank 1 local slot 3 -> +1000 + 3 = 1003!)
        perm = np.arange(256, dtype=np.int64)
        perm[0], perm[35] = 35, 0
        slot_idx = np.concatenate([perm[:32], perm[:4]])
        mgr._apply_layer_perm_indices(0, perm, slot_idx, torch.device("cpu"))
        self.assertAlmostEqual(float(moe_fp8._fused_decode_fp8_s1[0, 0].item()), 1003.0)
        self.assertAlmostEqual(float(moe_fp8._fused_decode_fp8_s2[0, 0].item()), 1103.0)

        counts = np.arange(1, 257, dtype=np.int64) * 10
        seen_sel_modes: list[str] = []
        from sglang_tpu.moe.pallas.fused_rs_muse import moe_imbalance_planner as mip

        orig_build = mip.build_preloaded_replica_plan

        def _spy_build(*a, **kw):
            seen_sel_modes.append(kw.get("selection_mode", ""))
            return orig_build(*a, **kw)

        fake_env = SimpleNamespace(
            SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SELECTION_MODE=SimpleNamespace(
                get=lambda: "bottleneck_owner_b64"
            )
        )
        with (
            patch("sglang_tpu.envs.envs", fake_env),
            patch.object(mip, "build_preloaded_replica_plan", side_effect=_spy_build),
        ):
            mgr._refresh_decode_replicas(
                0, counts, perm.astype(np.int32), torch.device("cpu")
            )
        self.assertEqual(seen_sel_modes, ["bottleneck_owner_b64"])
        self.assertTrue(torch.any(moe_fp8._fused_decode_replica_meta != 0))


if __name__ == "__main__":
    unittest.main()
