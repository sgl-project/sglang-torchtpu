# SPDX-License-Identifier: Apache-2.0
"""CPU unit & TPU7x Mosaic export tests for Wave-2 New Kernel Lab deliverables.

Covers:
  1. ``fused_decode_sparse_v2`` (3/4-slot circular weight ring, zero-SALU
     ``mxu_sparse``, K16 ``unloop64`` / ``vmem_deq`` / ``e4m3`` scales, paired
     ``Coll`` RS/AG, and ``fused_decode_sparse_fp8_from_logits`` for L0/L61).
  2. ``fused_decode_ep_a2a`` (``blocks_to_owners_norm_pallas`` + rank-deduplicated
     peer-VMEM SP+EP8 dispatch/combine).
  3. ``sglang_tpu/muse_megakernel/ops.py`` EP8 serving weight preparation
     (``_ModelGeometry`` ``inter_local`` and ``_prepare_ep_expert_refs_from_served``
     for both NVFP4 ``qblock in {0, 256}`` and channelwise FP8).
"""

from __future__ import annotations

import dataclasses
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class TestMuseNewKernelLab(CustomTestCase):
    def test_megakernel_ep8_geometry_and_expert_prep(self):
        from sglang_tpu.muse_megakernel.common.ep_layout import chunk_cols, kperm
        from sglang_tpu.muse_megakernel.ops import (
            _ModelGeometry,
            _prepare_ep_expert_refs_from_served,
        )

        # 1. Verify _ModelGeometry sets inter_local = 4096 // 8 = 512 under moe_mode="ep"
        stub_model = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=8192,
                moe_intermediate_size=4096,
                rms_norm_eps=1e-5,
                post_norm_eps=1e-8,
                layer_types=("sliding_attention", "full_attention"),
                sliding_window=2048,
                layer_rope_theta=(10000.0, 0.0),
            ),
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            head_dim=64,
                            num_heads=16,
                            num_kv_heads=2,
                            query_scale=5.473,
                        ),
                        mlp=SimpleNamespace(
                            num_experts=256,
                            top_k=8,
                            expert_hidden_size=4096,
                            intermediate_size=4096,
                            router_multiplier=0.1768,
                            route_scale=1.0,
                            route_eps=1e-15,
                            experts=SimpleNamespace(
                                w2_weight=torch.empty((32, 4096, 2048), dtype=torch.uint8)
                            ),
                        ),
                    ),
                    SimpleNamespace(),
                ]
            ),
        )
        geo = _ModelGeometry(stub_model, moe_mode="ep", tp_size=8)
        self.assertEqual(geo.inter_local, 512)
        self.assertEqual(geo.global_layers, (1,))

        # 2. Verify NVFP4 EP8 expert packing for qblock=0 and qblock=256
        rng = torch.Generator().manual_seed(1234)
        w13_u8 = torch.randint(0, 256, (4, 4096, 4096), dtype=torch.uint8, generator=rng)
        w2_u8 = torch.randint(0, 256, (4, 4096, 2048), dtype=torch.uint8, generator=rng)
        s13 = torch.full((4, 256, 8192), 0.0625, dtype=torch.bfloat16)
        s2 = torch.full((4, 256, 4096), 0.0625, dtype=torch.bfloat16)

        fp4_mod = nn.Module()
        fp4_mod.w13_weight = nn.Parameter(
            w13_u8.view(torch.float4_e2m1fn_x2), requires_grad=False
        )
        fp4_mod.w13_weight_scale = nn.Parameter(s13, requires_grad=False)
        fp4_mod.w2_weight = nn.Parameter(
            w2_u8.view(torch.float4_e2m1fn_x2), requires_grad=False
        )
        fp4_mod.w2_weight_scale = nn.Parameter(s2, requires_grad=False)

        for qb, exp_sr in ((0, 256), (256, 16)):
            refs = _prepare_ep_expert_refs_from_served(
                fp4_mod,
                "fp4",
                inter_local=512,
                tp_size=8,
                qblock=qb,
            )
            self.assertEqual(len(refs), 4)
            self.assertEqual(tuple(refs[0].shape), (4, 512, 8192))
            self.assertEqual(refs[0].dtype, torch.int32)
            self.assertEqual(tuple(refs[1].shape), (4, exp_sr, 8192))
            self.assertEqual(refs[1].dtype, torch.bfloat16)
            self.assertEqual(tuple(refs[2].shape), (4, 512, 4096))
            self.assertEqual(refs[2].dtype, torch.int32)
            self.assertEqual(tuple(refs[3].shape), (4, exp_sr, 4096))
            self.assertEqual(refs[3].dtype, torch.bfloat16)

        # 3. Verify channelwise FP8 EP8 expert packing (L0 & L61)
        fp8_mod = nn.Module()
        fp8_mod.w13_weight = nn.Parameter(
            torch.zeros((4, 4096, 8192), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        fp8_mod.w13_weight_scale = nn.Parameter(
            torch.full((4, 1, 1, 8192), 0.01, dtype=torch.bfloat16), requires_grad=False
        )
        fp8_mod.w2_weight = nn.Parameter(
            torch.zeros((4, 4096, 4096), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        fp8_mod.w2_weight_scale = nn.Parameter(
            torch.full((4, 1, 1, 4096), 0.01, dtype=torch.bfloat16), requires_grad=False
        )
        refs_f8 = _prepare_ep_expert_refs_from_served(
            fp8_mod,
            "fp8",
            inter_local=512,
            tp_size=8,
            qblock=0,
        )
        self.assertEqual(tuple(refs_f8[0].shape), (4, 1024, 8192))
        self.assertEqual(tuple(refs_f8[1].shape), (4, 2, 8192))
        self.assertEqual(tuple(refs_f8[2].shape), (4, 1024, 4096))
        self.assertEqual(tuple(refs_f8[3].shape), (4, 2, 4096))

    def test_tpu7x_mosaic_export_all_new_kernels(self):
        try:
            import jax
            from jax import export
            import jax.numpy as jnp
            from jax.sharding import Mesh, PartitionSpec as P
        except ImportError:
            self.skipTest("jax not installed in host Python")

        from sglang_tpu import pallas_export
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_a2a import (
            blocks_to_owners_norm_pallas,
            fused_decode_ep_a2a_from_sharded,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
            default_sparse_decode_v2_config,
            fused_decode_sparse_fp8_from_logits,
            fused_decode_sparse_v2_from_logits,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        devices = jax.devices()
        if len(devices) < 8:
            self.skipTest(
                "need >=8 host JAX devices (set XLA_FLAGS=--xla_force_host_platform_device_count=8)"
            )

        mesh_2d = Mesh(np.asarray(devices[:8]).reshape(1, 8), ("data", "model"))
        mesh_1d = Mesh(np.asarray(devices[:8]), ("model",))
        S = jax.ShapeDtypeStruct

        def _export_2d(fn, arg_specs, out_spec=P()):
            mapped = jax.shard_map(
                fn,
                mesh=mesh_2d,
                in_specs=(P(),) * len(arg_specs),
                out_specs=out_spec,
                check_vma=False,
            )
            with patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
            ):
                with tpu_info_for_cpu_export():
                    ir = export.export(jax.jit(mapped), platforms=("tpu",))(
                        *arg_specs
                    ).mlir_module()
            self.assertEqual(ir.count("stablehlo.custom_call"), 1)

        def _export_1d(fn, arg_specs, in_specs, out_spec):
            mapped = jax.shard_map(
                fn,
                mesh=mesh_1d,
                in_specs=in_specs,
                out_specs=out_spec,
                check_vma=False,
            )
            with patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
            ):
                with tpu_info_for_cpu_export():
                    ir = export.export(jax.jit(mapped), platforms=("tpu",))(
                        *arg_specs
                    ).mlir_module()
            self.assertEqual(ir.count("stablehlo.custom_call"), 1)

        # 1. fused_decode_sparse_v2 across K256 & K16, B in {16, 64, 128, 256}, R in {0, 2}, mxu_sparse
        for block in (256, 16):
            for b in (16, 64, 128, 256):
                for r in (0, 2):
                    with self.subTest(kernel="v2_default", block=block, b=b, r=r):
                        specs = [
                            S((1,), jnp.int32),
                            S((b, 256), jnp.float32),
                            S((1, 256), jnp.float32),
                            S((b, 4096), jnp.bfloat16),
                            S((32, 4096, 8192), jnp.float4_e2m1fn),
                            S((32, 4096 // block, 8192), jnp.bfloat16),
                            S((32, 4096, 4096), jnp.float4_e2m1fn),
                            S((32, 4096 // block, 4096), jnp.bfloat16),
                            S((1, 4096), jnp.bfloat16),
                        ]
                        if r > 0:
                            specs.extend(
                                [
                                    S((r, 4096, 8192), jnp.float4_e2m1fn),
                                    S((r, 4096 // block, 8192), jnp.bfloat16),
                                    S((r, 4096, 4096), jnp.float4_e2m1fn),
                                    S((r, 4096 // block, 4096), jnp.bfloat16),
                                    S((4, 8 * r), jnp.int32),
                                ]
                            )
                        _export_2d(
                            partial(fused_decode_sparse_v2_from_logits, ep=8),
                            tuple(specs),
                        )

        for b in (64, 128):
            with self.subTest(kernel="v2_mxu_sparse", b=b):
                cfg_m = dataclasses.replace(
                    default_sparse_decode_v2_config(b, 8, 256), row_mode="mxu_sparse"
                )
                _export_2d(
                    partial(fused_decode_sparse_v2_from_logits, ep=8, config=cfg_m),
                    (
                        S((1,), jnp.int32),
                        S((b, 256), jnp.float32),
                        S((1, 256), jnp.float32),
                        S((b, 4096), jnp.bfloat16),
                        S((32, 4096, 8192), jnp.float4_e2m1fn),
                        S((32, 16, 8192), jnp.bfloat16),
                        S((32, 4096, 4096), jnp.float4_e2m1fn),
                        S((32, 16, 4096), jnp.bfloat16),
                        S((1, 4096), jnp.bfloat16),
                    ),
                )

        # 2. fused_decode_sparse_fp8_from_logits (L0/L61 circular ring) across B={16,64,256}, R={0,2}
        for b in (16, 64, 256):
            for r in (0, 2):
                with self.subTest(kernel="v2_fp8", b=b, r=r):
                    f8_specs = [
                        S((1,), jnp.int32),
                        S((b, 256), jnp.float32),
                        S((1, 256), jnp.float32),
                        S((b, 4096), jnp.bfloat16),
                        S((32, 4096, 8192), jnp.float8_e4m3fn),
                        S((32, 1, 1, 8192), jnp.bfloat16),
                        S((32, 4096, 4096), jnp.float8_e4m3fn),
                        S((32, 1, 1, 4096), jnp.bfloat16),
                        S((1, 4096), jnp.bfloat16),
                    ]
                    if r > 0:
                        f8_specs.extend(
                            [
                                S((r, 4096, 8192), jnp.float8_e4m3fn),
                                S((r, 1, 1, 8192), jnp.bfloat16),
                                S((r, 4096, 4096), jnp.float8_e4m3fn),
                                S((r, 1, 1, 4096), jnp.bfloat16),
                                S((4, 8 * r), jnp.int32),
                            ]
                        )
                    _export_2d(
                        partial(fused_decode_sparse_fp8_from_logits, ep=8),
                        tuple(f8_specs),
                    )

        # 3. blocks_to_owners_norm_pallas & fused_decode_ep_a2a_from_sharded
        for b in (16, 64, 256):
            with self.subTest(kernel="blocks_to_owners", b=b):
                _export_1d(
                    partial(blocks_to_owners_norm_pallas, ep=8, eps=1e-5),
                    (
                        S((1,), jnp.int32),
                        S((b, 4096), jnp.bfloat16),
                        S((1, 4096), jnp.bfloat16),
                    ),
                    (P(), P(None, "model"), P()),
                    P("model", None),
                )
            for block in (256, 16):
                with self.subTest(kernel="ep_a2a", b=b, block=block):
                    _export_1d(
                        partial(
                            fused_decode_ep_a2a_from_sharded,
                            input_is_col_sharded=True,
                            pre_eps=1e-5,
                            ep=8,
                            return_replicated=False,
                        ),
                        (
                            S((1,), jnp.int32),
                            S((b, 256), jnp.float32),
                            S((b, 4096), jnp.bfloat16),
                            S((1, 256), jnp.float32),
                            S((32, 4096, 8192), jnp.float4_e2m1fn),
                            S((32, 4096 // block, 8192), jnp.bfloat16),
                            S((32, 4096, 4096), jnp.float4_e2m1fn),
                            S((32, 4096 // block, 4096), jnp.bfloat16),
                            S((1, 4096), jnp.bfloat16),
                            S((1, 4096), jnp.bfloat16),
                        ),
                        (
                            P(),
                            P("model", None),
                            P(None, "model"),
                            P(),
                            P(),
                            P(),
                            P(),
                            P(),
                            P(),
                            P(),
                        ),
                        P("model", None),
                    )

    def test_sparse_v2_and_fp8_opt_in_serving_gates(self):
        import os

        import sglang_tpu.fused_rs_sequence_parallel as seq_sp
        import sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op as fused_op_mod
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.models.muse_spark import MuseSparkMoE
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            muse_fuse_all_decode_op_name,
        )

        # 1. Both new serving flags stay strictly default-off.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_V2", None)
            os.environ.pop("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_FP8", None)
            self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_V2.get())
            self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_FP8.get())

        self.assertEqual(
            muse_fuse_all_decode_op_name(mode="sparse_v2", ep=8, block=16, num_replicas=2),
            "sglang_tpu_pallas::muse_decode_sparse_v2_ep8_k16_r2_tiles",
        )
        self.assertEqual(
            muse_fuse_all_decode_op_name(mode="sparse_fp8", ep=8, num_replicas=0),
            "sglang_tpu_pallas::muse_decode_sparse_fp8_ep8_r0",
        )

        # 2. Verify K16 (B>=32), K256 (B=64), and L0/L61 FP8 (B<=128) serving gates.
        def _stub_moe(*, block: int | None, is_fp8: bool):
            moe = MuseSparkMoE.__new__(MuseSparkMoE)
            nn.Module.__init__(moe)
            moe.tp_size = 8
            moe.moe_ep_size = 8
            moe.moe_tp_size = 1
            moe.num_experts = 256
            moe.top_k = 8
            moe.route_scale = 1.0
            moe.route_eps = 1e-15
            moe.router_multiplier = 1.0
            moe.expert_hidden_size = 4096
            moe.quantized_experts = not is_fp8
            moe.oneshot_gather = False
            moe._manual_post_gather = False
            moe._decode_fused_router = False
            moe._fused_decode_op = None
            moe._fused_decode_v2_op = None
            moe._fused_decode_v2_batches = ()
            moe._fused_decode_batches = (16, 32)
            moe._fused_decode_num_replicas = 0
            moe._fused_decode_mode = ""
            moe._decode_allow_prefill_eplb = False
            moe._decode_sparse_v2 = not is_fp8
            moe._decode_sparse_fp8 = is_fp8
            moe.gate = Mock(
                weight=torch.empty((256, 8192), dtype=torch.float32, device="meta"),
                side_effect=lambda r: (
                    torch.empty((r.shape[0], 256), dtype=torch.float32, device="meta"),
                    None,
                ),
            )
            moe.e_score_correction_bias = nn.Parameter(
                torch.empty(256, dtype=torch.float32, device="meta"),
                requires_grad=False,
            )
            moe.pre_expert_proj = Mock(side_effect=lambda h: (h[:, :4096], None))
            moe.pre_expert_norm = Mock(side_effect=lambda h: h)
            moe.post_expert_proj = Mock(side_effect=lambda h: (h, None))
            moe._route = Mock(
                side_effect=lambda r: SimpleNamespace(
                    topk_weights=torch.empty((r.shape[0], 8), dtype=torch.float32, device="meta"),
                    topk_ids=torch.empty((r.shape[0], 8), dtype=torch.int32, device="meta"),
                )
            )
            moe._combine_slots = Mock(
                side_effect=lambda o, w: torch.empty(
                    (o.shape[0], 4096), dtype=torch.bfloat16, device="meta"
                )
            )
            ex = nn.Module()
            ex.handles_moe_reduction = True
            ex.supports_tpu_fused_rs_post_expert_norm = True
            ex._tpu_fused_rs_post_expert_norm_weight = torch.empty(
                4096, dtype=torch.bfloat16, device="meta"
            )
            ex.register_buffer(
                "_tpu_fused_rs_group_offset",
                torch.tensor([0], dtype=torch.int32),
                persistent=False,
            )
            if is_fp8:
                ex.w13_weight = nn.Parameter(
                    torch.empty((32, 4096, 8192), dtype=torch.float8_e4m3fn, device="meta"),
                    requires_grad=False,
                )
                ex.w13_weight_scale = nn.Parameter(
                    torch.empty((32, 1, 1, 8192), dtype=torch.float32, device="meta"),
                    requires_grad=False,
                )
                ex.w2_weight = nn.Parameter(
                    torch.empty((32, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
                    requires_grad=False,
                )
                ex.w2_weight_scale = nn.Parameter(
                    torch.empty((32, 1, 1, 4096), dtype=torch.float32, device="meta"),
                    requires_grad=False,
                )
            else:
                ex.w13_weight = nn.Parameter(
                    torch.empty((32, 4096, 4096), dtype=torch.float4_e2m1fn_x2, device="meta"),
                    requires_grad=False,
                )
                ex.w13_weight_scale = nn.Parameter(
                    torch.empty((32, 4096 // block, 8192), dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                )
                ex.w2_weight = nn.Parameter(
                    torch.empty((32, 4096, 2048), dtype=torch.float4_e2m1fn_x2, device="meta"),
                    requires_grad=False,
                )
                ex.w2_weight_scale = nn.Parameter(
                    torch.empty((32, 4096 // block, 4096), dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                )
            ex.forward = lambda h, r: torch.empty(
                (h.shape[0], 4096), dtype=torch.bfloat16, device="meta"
            )
            moe.experts = ex
            return moe

        parallel = SimpleNamespace(
            tp_rank=0,
            moe_ep_size=8,
            moe_ep_rank=0,
            moe_tp_size=1,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            attn_dcp_size=1,
            moe_dp_size=1,
        )
        graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
        calls: list[str] = []

        def _fake_get_op(**kw):
            m = kw["mode"]
            return lambda *args, _m=m: calls.append(_m) or args[3]

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            patch("sglang.srt.runtime_context.get_exec", return_value=SimpleNamespace(graph=graph)),
            patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch.object(fused_op_mod, "get_muse_fuse_all_decode_op", side_effect=_fake_get_op),
            patch.object(
                seq_sp,
                "get_fused_moe_sequence_parallel_plan",
                return_value=(False, False, False),
            ),
            patch.object(
                seq_sp,
                "get_tpu_fused_rs_post_expert_norm_weight",
                side_effect=lambda ex: ex._tpu_fused_rs_post_expert_norm_weight,
            ),
        ):
            for block, want_by_b in (
                (16, {16: "sparse", 32: "sparse_v2", 64: "sparse_v2", 128: "sparse_v2", 256: "sparse_v2"}),
                (256, {16: "sparse", 32: "sparse", 64: "sparse_v2", 128: "sparse", 256: "sparse"}),
            ):
                moe = _stub_moe(block=block, is_fp8=False)
                moe.prepare_fused_decode(mode="sparse")
                for b, want in want_by_b.items():
                    calls.clear()
                    x = torch.empty((b, 8192), dtype=torch.bfloat16, device="meta")
                    _ = moe(x, x.float(), decode=True)
                    self.assertEqual(calls, [want], msg=f"block={block}, B={b}")

            moe_fp8 = _stub_moe(block=None, is_fp8=True)
            moe_fp8.prepare_fused_decode(mode="sparse")
            for b, want_fp8 in ((16, True), (32, True), (64, True), (128, True), (256, False)):
                calls.clear()
                moe_fp8._route.reset_mock()
                x = torch.empty((b, 8192), dtype=torch.bfloat16, device="meta")
                _ = moe_fp8(x, x.float(), decode=True)
                self.assertEqual(calls, ["sparse_fp8"] if want_fp8 else [], msg=f"fp8 B={b}")
                self.assertEqual(moe_fp8._route.called, not want_fp8, msg=f"fp8 route B={b}")


if __name__ == "__main__":
    unittest.main()
