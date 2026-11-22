# SPDX-License-Identifier: Apache-2.0
"""CPU unit & TPU7x Mosaic export tests for Wave-2 Muse Spark Prefill MoE Imbalance."""

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import jax
from jax import export
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pallas_export
from sglang_tpu.envs import envs
from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
    build_preloaded_replica_plan,
    build_static_placement,
    interleave_hot_expert_tiles_jax,
    pallas_prefill_imbalance_route,
    route_prefill_imbalance_jax,
    slice_replica_weights_for_owner,
    validate_replica_plan,
)
from sglang_tpu.muse_prefill_kernels.eplb import (
    PrefillEplbManager,
    compute_zeroglue_eplb_perm,
    remap_zeroglue_virtual_ids,
    resolve_prefill_eplb_config,
)
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

register_tpu_ci(est_time=35, suite="stage-a-test-tpu-unit", chips=1)


class TestMusePrefillImbalance(CustomTestCase):
    def test_wave2_prefill_env_flags_default_off(self):
        self.assertEqual(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_REPLICA_SLOTS.get(), 0)
        self.assertEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_REPLICA_MODE.get(), "whole"
        )
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE.get())
        self.assertEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE.get(), "whole"
        )
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_HOT_INTERLEAVE.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB.get())

    def test_dual_static_placement_and_bottleneck_first_replica_plan(self):
        rng = np.random.default_rng(1119)
        ranks = np.arange(1, 257, dtype=np.float64)
        pre_counts = (10000.0 / np.power(ranks, 0.85))[rng.permutation(256)]
        dec_counts = (10000.0 / np.power(ranks, 0.95))[rng.permutation(256)]
        owner_to_chip = [r >> 1 for r in range(8)]

        perm_dual = build_static_placement(
            pre_counts,
            num_owners=8,
            owner_to_chip=owner_to_chip,
            tile_m=128,
            objective="dual",
            decode_counts=dec_counts,
        )
        self.assertEqual(sorted(perm_dual.tolist()), list(range(256)))

        id_pre = pre_counts.reshape(8, 32).sum(axis=1)
        du_pre = pre_counts[perm_dual].reshape(8, 32).sum(axis=1)
        self.assertLess(du_pre.max() / du_pre.mean(), id_pre.max() / id_pre.mean())
        pigeonhole_floor = float(
            (pre_counts.max() + np.sort(pre_counts)[:31].sum()) / (pre_counts.sum() / 8.0)
        )
        self.assertLessEqual(du_pre.max() / du_pre.mean(), max(1.05, pigeonhole_floor + 0.02))

        p_dec_id = dec_counts / dec_counts.sum()
        p_dec_du = dec_counts[perm_dual] / dec_counts.sum()
        id_dec = (1.0 - np.exp(-512.0 * p_dec_id)).reshape(8, 32).sum(axis=1)
        du_dec = (1.0 - np.exp(-512.0 * p_dec_du)).reshape(8, 32).sum(axis=1)
        self.assertLess(du_dec.max() / du_dec.mean(), id_dec.max() / id_dec.mean())
        self.assertLessEqual(du_dec.max() / du_dec.mean(), 1.06)

        # Bottleneck-home-first cross-chip replica plan: concentrates slots on heaviest home
        # and guarantees holder_chip != home_chip for every active replica!
        spiky = np.ones(256, dtype=np.float64) * 10.0
        spiky[:10] = 2000.0  # 10 hot experts all sitting on owner 0 (chip 0)
        plan = build_preloaded_replica_plan(
            spiky,
            num_owners=8,
            num_replicas_per_owner=2,
            owner_to_chip=owner_to_chip,
        )
        validate_replica_plan(plan)
        home_owners = plan.replica_meta[1]
        holder_owners = plan.replica_meta[2]
        valid = plan.replica_meta[0] >= 0
        self.assertGreaterEqual(int(np.sum(home_owners[valid] == 0)), 6)
        for hm, hl in zip(home_owners[valid], holder_owners[valid]):
            self.assertNotEqual(int(hm) >> 1, int(hl) >> 1)

    def test_zeroglue_eplb_dual_decode_and_replica_refresh_sync(self):
        for mode_str, r_in, want in (
            ("dyn_holder", 2, ("dyn_holder", 2)),
            ("dyn_split", 4, ("dyn_split", 4)),
            ("zg_dyn_split", 2, ("dyn_split", 2)),
        ):
            self.assertEqual(resolve_prefill_eplb_config(mode_str, r_in), want)

        rng = np.random.default_rng(2026)
        counts = rng.integers(10, 500, size=(256,), dtype=np.int64)
        counts[69] = 18000
        counts[230] = 12000
        r2g, col_perm = compute_zeroglue_eplb_perm(
            counts, ep_size=8, num_local_base=32, num_replicas=2, dual_decode=True
        )
        self.assertEqual(sorted(col_perm.tolist()), list(range(256)))
        self.assertEqual(col_perm[:2].tolist(), [69, 230])
        for r in range(8):
            np.testing.assert_array_equal(r2g[r, :32], col_perm[r * 32 : (r + 1) * 32])
            np.testing.assert_array_equal(r2g[r, 32:], col_perm[:2])

        # Verify dyn_holder and dyn_split conserve every routed token's original expert ID.
        ids = jnp.asarray(
            rng.choice(256, size=(512, 8), p=counts / counts.sum()), dtype=jnp.int32
        )
        slot_to_col = np.zeros(8 * 34, dtype=np.int32)
        for r in range(8):
            slot_to_col[r * 34 : r * 34 + 32] = np.arange(r * 32, (r + 1) * 32)
            slot_to_col[r * 34 + 32 : r * 34 + 34] = [0, 1]
        for emode in ("dyn_holder", "dyn_split"):
            v_glob, v_loc = remap_zeroglue_virtual_ids(
                ids,
                jnp.int32(3),
                ep_size=8,
                num_local_experts=34,
                mode=emode,
                quantum=128,
            )
            np.testing.assert_array_equal(slot_to_col[np.asarray(v_glob)], np.asarray(ids))
            np.testing.assert_array_equal(
                np.asarray(v_loc), np.asarray(v_glob)[3 * 64 : 4 * 64]
            )

        # Verify PrefillEplbManager refreshes _fused_decode_rep_* & _fused_decode_replica_meta
        # bit-exactly even when starting from a non-identity static placement!
        init_perm = rng.permutation(256).astype(np.int32)
        orig_w13 = torch.arange(256 * 8, dtype=torch.uint8).reshape(256, 2, 4).view(
            torch.float4_e2m1fn_x2
        )
        orig_s13 = torch.arange(256 * 4, dtype=torch.bfloat16).reshape(256, 1, 4)
        orig_w2 = (torch.arange(256 * 8, dtype=torch.uint8) + 3).reshape(
            256, 2, 4
        ).view(torch.float4_e2m1fn_x2)
        orig_s2 = (torch.arange(256 * 4, dtype=torch.bfloat16) + 1).reshape(256, 1, 4)
        orig_gate = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
        orig_bias = torch.arange(256, dtype=torch.float32) * 0.1

        rank_experts = []
        rank_moes = []
        for r in range(8):
            sl = torch.tensor(init_perm[r * 32 : (r + 1) * 32], dtype=torch.long)
            ex_r = torch.nn.Module()
            ex_r.w13_weight = torch.nn.Parameter(
                orig_w13.view(torch.uint8)[sl].contiguous().view(torch.float4_e2m1fn_x2),
                requires_grad=False,
            )
            ex_r.w13_weight_scale = torch.nn.Parameter(
                orig_s13[sl].contiguous(), requires_grad=False
            )
            ex_r.w2_weight = torch.nn.Parameter(
                orig_w2.view(torch.uint8)[sl].contiguous().view(torch.float4_e2m1fn_x2),
                requires_grad=False,
            )
            ex_r.w2_weight_scale = torch.nn.Parameter(
                orig_s2[sl].contiguous(), requires_grad=False
            )
            ex_r.num_local_experts = 32
            ex_r._tpu_fused_rs_group_offset = torch.tensor([r * 32], dtype=torch.int32)

            moe_r = torch.nn.Module()
            moe_r.experts = ex_r
            moe_r.gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    orig_gate[torch.tensor(init_perm, dtype=torch.long)].contiguous(),
                    requires_grad=False,
                )
            )
            moe_r.e_score_correction_bias = torch.nn.Parameter(
                orig_bias[torch.tensor(init_perm, dtype=torch.long)].contiguous(),
                requires_grad=False,
            )
            moe_r._fused_decode_num_replicas = 2
            moe_r.register_buffer(
                "_fused_decode_rep_w1",
                torch.zeros((2, 2, 4), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            )
            moe_r.register_buffer(
                "_fused_decode_rep_s1", torch.zeros((2, 1, 4), dtype=torch.bfloat16)
            )
            moe_r.register_buffer(
                "_fused_decode_rep_w2",
                torch.zeros((2, 2, 4), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
            )
            moe_r.register_buffer(
                "_fused_decode_rep_s2", torch.zeros((2, 1, 4), dtype=torch.bfloat16)
            )
            moe_r.register_buffer(
                "_fused_decode_replica_meta", torch.full((4, 16), -1, dtype=torch.int32)
            )
            rank_experts.append(ex_r)
            rank_moes.append(moe_r)

        # Simulate rank 0's PrefillEplbManager with an exact 8-rank gather callback.
        state = {"stage": "prealloc", "cur_tensors": None}

        def gather_rank0(local_t: torch.Tensor) -> torch.Tensor:
            return torch.cat(state["cur_tensors"], dim=0)

        # Test single-rank manager via per-rank instantiation sharing the full 256 snapshot.
        full_snap = {
            "w13_weight": orig_w13.view(torch.uint8)[
                torch.tensor(init_perm, dtype=torch.long)
            ].contiguous(),
            "w13_weight_scale": orig_s13[
                torch.tensor(init_perm, dtype=torch.long)
            ].contiguous(),
            "w2_weight": orig_w2.view(torch.uint8)[
                torch.tensor(init_perm, dtype=torch.long)
            ].contiguous(),
            "w2_weight_scale": orig_s2[
                torch.tensor(init_perm, dtype=torch.long)
            ].contiguous(),
        }

        def make_gather(moe_obj):
            def _g(chunk_t: torch.Tensor) -> torch.Tensor:
                if chunk_t.shape == (1,) and chunk_t.dtype == torch.int32:
                    return chunk_t.repeat(8)
                # Identify matching attribute & chunk offset on rank 0
                ex = moe_obj.experts
                for attr in (
                    "w13_weight",
                    "w2_weight",
                    "w13_weight_scale",
                    "w2_weight_scale",
                ):
                    cur = getattr(ex, attr).data[:32]
                    cur_u = cur.view(torch.uint8) if cur.dtype == torch.float4_e2m1fn_x2 else cur
                    full_a = full_snap[attr]
                    if chunk_t.shape[1:] == cur_u.shape[1:] and chunk_t.dtype == cur_u.dtype:
                        if chunk_t.shape[0] == 32:
                            return full_a
                        c_sz = chunk_t.shape[0]
                        for s0 in range(0, 32, c_sz):
                            if torch.equal(chunk_t, cur_u[s0 : s0 + c_sz]):
                                parts = [
                                    full_a[rr * 32 + s0 : rr * 32 + s0 + c_sz]
                                    for rr in range(8)
                                ]
                                return torch.cat(parts, dim=0)
                raise AssertionError(f"Unmatched gather chunk {chunk_t.shape}")

            return _g

        dec0 = SimpleNamespace(
            norm=SimpleNamespace(weight=torch.ones(4)),
            layers=[SimpleNamespace(mlp=rank_moes[0])],
        )
        lw0 = SimpleNamespace(wrouter=torch.zeros(4, 256, dtype=torch.float32))
        mgr = PrefillEplbManager(
            decoder=dec0,
            weights_list=[lw0],
            gather_fn=make_gather(rank_moes[0]),
            start=0,
            end=1,
            tp_size=8,
            tp_rank=0,
            mode="home",
            num_replicas=2,
            initial_col_perms=[init_perm],
            dual_decode=True,
        )
        self.assertEqual(tuple(rank_experts[0].w13_weight.shape), (34, 2, 4))

        # Update full_snap after rebalance_from_counts Permutes rank 0 -> new_col_perm:
        # During _apply_layer_perm_indices, _gather_selected_slots reads pre-rebalance `full_snap`,
        # then _refresh_decode_replicas reads post-rebalance `full_snap`!
        _, expected_perm = compute_zeroglue_eplb_perm(
            counts, ep_size=8, num_local_base=32, num_replicas=2, dual_decode=True
        )
        orig_apply = mgr._apply_layer_perm_indices

        def wrapped_apply(offset, rel_col_idx, rel_slot_idx, device):
            orig_apply(offset, rel_col_idx, rel_slot_idx, device)
            p_t = torch.tensor(expected_perm, dtype=torch.long)
            full_snap["w13_weight"] = orig_w13.view(torch.uint8)[p_t].contiguous()
            full_snap["w13_weight_scale"] = orig_s13[p_t].contiguous()
            full_snap["w2_weight"] = orig_w2.view(torch.uint8)[p_t].contiguous()
            full_snap["w2_weight_scale"] = orig_s2[p_t].contiguous()

        mgr._apply_layer_perm_indices = wrapped_apply
        mgr.rebalance_from_counts(counts[None, :])

        # Verify router, base [32], prefill replicas [32:34], AND decode replicas!
        exp_t = torch.tensor(expected_perm, dtype=torch.long)
        torch.testing.assert_close(rank_moes[0].gate.weight.data, orig_gate[exp_t])
        torch.testing.assert_close(
            rank_moes[0].e_score_correction_bias.data, orig_bias[exp_t]
        )
        torch.testing.assert_close(
            rank_experts[0].w13_weight.data[:32].view(torch.uint8),
            orig_w13.view(torch.uint8)[exp_t[:32]],
        )
        torch.testing.assert_close(
            rank_experts[0].w13_weight.data[32:34].view(torch.uint8),
            orig_w13.view(torch.uint8)[exp_t[:2]],
        )
        expected_dec_plan = build_preloaded_replica_plan(
            counts,
            num_owners=8,
            num_replicas_per_owner=2,
            placement=expected_perm,
            owner_to_chip=[r >> 1 for r in range(8)],
        )
        np.testing.assert_array_equal(
            rank_moes[0]._fused_decode_replica_meta.numpy(),
            expected_dec_plan.replica_meta,
        )
        for s_i, ckpt_e in enumerate(expected_dec_plan.holder_checkpoint_ids[0]):
            if ckpt_e >= 0:
                torch.testing.assert_close(
                    rank_moes[0]._fused_decode_rep_w1[s_i].view(torch.uint8),
                    orig_w13.view(torch.uint8)[int(ckpt_e)],
                )
                torch.testing.assert_close(
                    rank_moes[0]._fused_decode_rep_s1[s_i], orig_s13[int(ckpt_e)]
                )

    def test_pallas_prefill_imbalance_router_bit_exact_and_hot_interleave_bijection(self):
        rng = np.random.default_rng(77)
        calib = np.ones(256, dtype=np.float64) * 40.0
        calib[[0, 1, 33, 64]] = [5200.0, 3400.0, 2600.0, 1900.0]
        plan_r2 = build_preloaded_replica_plan(
            calib,
            num_owners=8,
            num_replicas_per_owner=2,
            owner_to_chip=[r >> 1 for r in range(8)],
        )
        plan_mh4 = build_preloaded_replica_plan(
            calib,
            num_owners=8,
            num_replicas_per_owner=2,
            max_holders_per_expert=4,
            owner_to_chip=[r >> 1 for r in range(8)],
        )

        ids_np = rng.choice(256, size=(4096, 8), p=calib / calib.sum()).astype(np.int32)
        ids_np[-64:, :] = 256  # SKIP_PAD_MOE_TOKENS sentinel
        ids_j = jnp.asarray(ids_np, dtype=jnp.int32)

        for name, kw in [
            ("cm_whole", dict(do_chipmate=True, chipmate_mode="whole")),
            ("cm_split", dict(do_chipmate=True, chipmate_mode="split")),
            (
                "r2_whole",
                dict(
                    replica_meta=jnp.asarray(plan_r2.replica_meta),
                    num_replicas=2,
                    replica_mode="whole",
                ),
            ),
            (
                "r2_split",
                dict(
                    replica_meta=jnp.asarray(plan_r2.replica_meta),
                    num_replicas=2,
                    replica_mode="split",
                ),
            ),
            (
                "r2_mh4",
                dict(
                    replica_meta=jnp.asarray(plan_mh4.replica_meta),
                    num_replicas=2,
                    replica_mode="split",
                ),
            ),
        ]:
            with self.subTest(mode=name):
                v_ref, cm_ref = route_prefill_imbalance_jax(
                    ids_j, ep_size=8, num_base_local=32, use_pallas=False, **kw
                )
                v_pal, cm_pal = pallas_prefill_imbalance_route(
                    ids_j, ep_size=8, num_base_local=32, interpret=True, **kw
                )
                np.testing.assert_array_equal(np.asarray(v_ref), np.asarray(v_pal))
                if cm_ref is not None:
                    np.testing.assert_array_equal(np.asarray(cm_ref), np.asarray(cm_pal))

        # Section 4.2b hot-expert tile interleaving exact bijection + borrowed-tail invariance
        tiles_per_g = jnp.array([0, 10, 2, 1, 3, 0, 2, 4], dtype=jnp.int32)
        gm_grp = jnp.array(
            [1] * 10 + [2] * 2 + [3] + [4] * 3 + [6] * 2 + [7] * 4 + [7] * 2,
            dtype=jnp.int32,
        )
        perm_gm, hot_slot = interleave_hot_expert_tiles_jax(
            gm_grp, tiles_per_g, num_slots=24, cand_groups=7
        )
        self.assertEqual(int(hot_slot), 1)
        self.assertEqual(sorted(np.asarray(perm_gm).tolist()), list(range(24)))
        # Borrowed group 7 (`>= cand_groups=7`) and padded tail slots stay in place!
        np.testing.assert_array_equal(np.asarray(perm_gm)[18:], np.arange(18, 24))
        reordered_g = np.asarray(gm_grp)[np.asarray(perm_gm)]
        for k in range(8):
            self.assertEqual(reordered_g[2 * k], 1)
            self.assertNotEqual(reordered_g[2 * k + 1], 1)

    def test_tpu7x_mosaic_exports_prefill_router_and_padded_decode(self):
        devices = jax.devices()
        if len(devices) < 8:
            self.skipTest("Requires XLA_FLAGS=--xla_force_host_platform_device_count=8+")

        S = jax.ShapeDtypeStruct
        with patch.object(
            pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
        ):
            with tpu_info_for_cpu_export():
                # 1. 1-op Pallas prefill imbalance router exports
                for tag, fn, specs in [
                    (
                        "cm_split",
                        lambda x: pallas_prefill_imbalance_route(
                            x,
                            ep_size=8,
                            num_base_local=32,
                            do_chipmate=True,
                            chipmate_mode="split",
                        ),
                        (S((8192, 8), jnp.int32),),
                    ),
                    (
                        "dual_r2_cm",
                        lambda x, m: pallas_prefill_imbalance_route(
                            x,
                            ep_size=8,
                            num_base_local=32,
                            replica_meta=m,
                            num_replicas=2,
                            replica_mode="split",
                            do_chipmate=True,
                            chipmate_mode="split",
                        ),
                        (S((8192, 8), jnp.int32), S((4, 16), jnp.int32)),
                    ),
                ]:
                    with self.subTest(export=tag):
                        ir = export.export(jax.jit(fn), platforms=("tpu",))(
                            *specs
                        ).mlir_module()
                        self.assertEqual(ir.count("stablehlo.custom_call"), 1)

                # 2. Zero-copy [34, ...] prefill EPLB slab exports on decode sparse & chipmate
                from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_chipmate import (
                    fused_decode_chipmate_from_logits,
                )
                from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
                    fused_decode_sparse_from_logits,
                )

                mesh8 = Mesh(np.array(devices[:8]), ("model",))
                dec_specs = (
                    S((1,), jnp.int32),
                    S((64, 256), jnp.float32),
                    S((1, 256), jnp.float32),
                    S((64, 4096), jnp.bfloat16),
                    S((34, 4096, 8192), jnp.float4_e2m1fn),
                    S((34, 16, 8192), jnp.bfloat16),
                    S((34, 4096, 4096), jnp.float4_e2m1fn),
                    S((34, 16, 4096), jnp.bfloat16),
                    S((1, 4096), jnp.bfloat16),
                    S((2, 4096, 8192), jnp.float4_e2m1fn),
                    S((2, 16, 8192), jnp.bfloat16),
                    S((2, 4096, 4096), jnp.float4_e2m1fn),
                    S((2, 16, 4096), jnp.bfloat16),
                    S((4, 16), jnp.int32),
                )
                for dtag, dfn in [
                    ("sparse_eplb34_r2", partial(fused_decode_sparse_from_logits, ep=8)),
                    (
                        "chipmate_eplb34_r2",
                        partial(fused_decode_chipmate_from_logits, ep=8),
                    ),
                ]:
                    with self.subTest(export=dtag):
                        mapped = jax.shard_map(
                            dfn,
                            mesh=mesh8,
                            in_specs=(P(),) * len(dec_specs),
                            out_specs=P(),
                            check_vma=False,
                        )
                        ir = export.export(jax.jit(mapped), platforms=("tpu",))(
                            *dec_specs
                        ).mlir_module()
                        self.assertEqual(ir.count("stablehlo.custom_call"), 1)

    def test_w7_dual_tiles_and_r6_r8_empirical_h2_replica_planner(self):
        import json
        import tempfile
        from pathlib import Path
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            _decode_objective_weights,
            load_calibration_counts,
        )

        rng = np.random.default_rng(20261120)
        ranks = np.arange(1, 257, dtype=np.float64)
        pre_counts = (12000.0 / np.power(ranks, 0.92))[rng.permutation(256)]
        dec_counts = (10000.0 / np.power(ranks, 0.78))[rng.permutation(256)]
        owner_to_chip = [r >> 1 for r in range(8)]

        perm_dt = build_static_placement(
            pre_counts,
            num_owners=8,
            owner_to_chip=owner_to_chip,
            tile_m=128,
            objective="dual_tiles",
            decode_counts=dec_counts,
            min_gain=0.005,
            hot_slot0_order=True,
        )
        self.assertEqual(sorted(perm_dt.tolist()), list(range(256)))
        perm_du = build_static_placement(
            pre_counts,
            num_owners=8,
            owner_to_chip=owner_to_chip,
            tile_m=128,
            objective="dual",
            decode_counts=dec_counts,
            min_gain=0.005,
        )
        w_pre_t = _decode_objective_weights(
            pre_counts, objective="prefill_tiles", tile_m=128
        )
        du_loads = w_pre_t[perm_du].reshape(8, 32).sum(axis=1)
        dt_loads = w_pre_t[perm_dt].reshape(8, 32).sum(axis=1)
        pigeonhole_floor = float(
            (w_pre_t.max() + np.sort(w_pre_t)[:31].sum()) / (w_pre_t.sum() / 8.0)
        )
        self.assertLessEqual(
            float(dt_loads.max() / dt_loads.mean()),
            max(1.03, pigeonhole_floor + 0.015),
        )
        self.assertLessEqual(float(dt_loads.max()), float(du_loads.max()) + 1e-6)

        emp_steps = rng.binomial(64, np.clip(8.0 * dec_counts / dec_counts.sum(), 0.0, 0.99), size=(64, 256))
        for r_slots in (6, 8):
            for sm in ("bottleneck_owner_b64", "bottleneck_owner_b64_h2"):
                plan = build_preloaded_replica_plan(
                    dec_counts,
                    num_owners=8,
                    num_replicas_per_owner=r_slots,
                    placement=perm_dt,
                    owner_to_chip=owner_to_chip,
                    selection_mode=sm,
                    empirical_step_counts=emp_steps,
                )
                validate_replica_plan(plan)
                self.assertEqual(plan.replica_meta.shape, (4, 8 * r_slots))
                by_eid: dict[int, list[int]] = {}
                for k in range(plan.replica_meta.shape[1]):
                    eid, hm, hl, _ = (int(plan.replica_meta[i, k]) for i in range(4))
                    if eid >= 0:
                        self.assertNotEqual(hm >> 1, hl >> 1)
                        by_eid.setdefault(eid, []).append(hl >> 1)
                for chs in by_eid.values():
                    self.assertEqual(len(chs), len(set(chs)))

        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "replica_stats.json"
            sp = Path(td) / "replica_stats.steps.npy"
            jp.write_text(json.dumps({"0": dec_counts.tolist()}))
            np.save(sp, emp_steps[:, None, :].astype(np.uint8))
            loaded = load_calibration_counts(jp, num_layers=1, num_experts=256)
            plan_sc = build_preloaded_replica_plan(
                loaded[0],
                num_owners=8,
                num_replicas_per_owner=8,
                placement=perm_dt,
                owner_to_chip=owner_to_chip,
                selection_mode="bottleneck_owner_b64_h2",
            )
            validate_replica_plan(plan_sc)


if __name__ == "__main__":
    unittest.main()
