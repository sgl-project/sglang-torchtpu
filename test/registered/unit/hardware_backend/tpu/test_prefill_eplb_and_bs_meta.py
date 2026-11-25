# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for Muse Spark prefill Zero-Glue EPLB and 1D block-scatter metadata."""

import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.muse_prefill_kernels.eplb import (
    OnlineEplbController,
    PrefillEplbManager,
    build_eplb_moe_op,
    compute_eplb_tile_skew_stats,
    compute_zeroglue_eplb_perm,
    is_eligible_eplb_calib_batch,
    load_eplb_stats_file,
    map_torch_fallback_topk_ids,
    remap_zeroglue_virtual_ids,
    resolve_prefill_eplb_config,
    save_eplb_stats_file,
)
from sglang_tpu.muse_prefill_kernels.integration import (
    MusePrefillKernels,
    _make_eplb_hist_callback,
)
from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
    FUSED_RS_TUNED_BLOCK_SIZES,
    _compute_local_inv_perm,
    compute_block_scatter_routing,
    register_ep16_tuned_block_sizes,
)

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)


class TestPrefillEplbAndBsMeta(CustomTestCase):
    def test_owner_a1_uses_runtime_rank_for_rope_and_fallback_weights(self):
        from sglang_tpu import rank_local

        ns = types.SimpleNamespace
        # The graph is traced on logical rank zero, then replayed with other rank inputs.
        dispatch = Mock(side_effect=StopIteration)
        ctx = ns(
            tp_size=16,
            tp_rank=0,
            tp16_owner_a1=True,
            a1_variant="v1",
            _a1_skip=False,
            ops=ns(a1=dispatch, a1_owner=dispatch),
        )
        weights = ns(
            wqkvg=torch.arange(48).reshape(16, 1, 3),
            wqkvg_scale=torch.arange(48).reshape(16, 3),
            wo=torch.ones(1),
            wo_shape=(1,),
            wo_scale=torch.ones(1),
        )

        def slice_cpu(x, rank, dim, parts, stride):
            return x.tensor_split(parts, dim)[rank.item() // stride]

        self.enterContext(
            patch.object(rank_local, "get_device_assignment", return_value=range(16))
        )
        self.enterContext(patch.object(rank_local.dist, "get_rank", return_value=0))
        self.enterContext(patch.object(rank_local, "_slice_op", return_value=slice_cpu))
        for rank in (0, 1, 15):
            for rows in (512, 256):
                table = torch.arange(16 * rows).reshape(-1, 1)
                with (
                    self.subTest(rank=rank, rows=rows),
                    rank_local.trace_rank(torch.tensor([rank])),
                    self.assertRaises(StopIteration),
                ):
                    MusePrefillKernels._layer(
                        ctx,
                        layer=ns(self_attn=None, mlp=None),
                        weights=weights,
                        rope=(table, -table),
                        stream_shard=None,
                        attn_in=torch.zeros(rows, 1),
                        next_norm=None,
                        forward_batch=None,
                    )
                _, w, scales, cos, sin = dispatch.call_args.args
                if rows == 512:
                    self.assertIs(w, weights.wqkvg)
                    torch.testing.assert_close(cos, table.split(rows)[rank])
                    torch.testing.assert_close(sin, -table.split(rows)[rank])
                else:
                    torch.testing.assert_close(w, weights.wqkvg[rank])
                    torch.testing.assert_close(
                        scales, weights.wqkvg_scale[rank : rank + 1]
                    )
                    torch.testing.assert_close(cos, table)

    def test_native_tp16_prefill_requires_matching_ranks_and_supported_projections(
        self,
    ):
        from sglang_tpu.muse_prefill_kernels.attn_kernels.reference import QUERY_SCALE

        ns = types.SimpleNamespace
        experts = ns(
            handles_moe_reduction=True, w13_weight=ns(dtype=torch.float4_e2m1fn_x2)
        )
        decoder = ns(pp_group=ns(world_size=1), layers=[ns(mlp=ns(experts=experts))])
        cfg = ns(
            num_attention_heads=128,
            num_key_value_heads=16,
            qk_scale_factor=QUERY_SCALE,
            output_multiplier=0.17677669529663687,
            routed_scaling_factor=1.0,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            hidden_size=8192,
            moe_hidden_size=4096,
            num_local_experts=256,
            num_experts_per_tok=8,
            head_dim=64,
        )
        model = ns(model=decoder, text_config=cfg)
        parallel = ns(
            tp_rank=1, tp_size=16, attn_tp_size=16, moe_ep_size=16, attn_dcp_size=1
        )
        self.enterContext(
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel)
        )
        self.enterContext(
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            )
        )
        init = self.enterContext(
            patch.object(MusePrefillKernels, "__init__", return_value=None)
        )
        self.assertIsNone(MusePrefillKernels.from_model(model))
        with envs.SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL.override(True):
            self.assertIsInstance(
                MusePrefillKernels.from_model(model), MusePrefillKernels
            )
            init.assert_called_once_with(decoder, start=0, end=1, tp_size=16, tp_rank=1)
            with envs.SGLANG_TPU_MUSE_KERNEL_A3_VARIANT.override("v16"):
                self.assertIsNone(MusePrefillKernels.from_model(model))
            parallel.moe_ep_size = 8
            self.assertIsNone(MusePrefillKernels.from_model(model))

    def test_dflash_captures_rank_rows_after_requested_layers(self):
        # Use two distinct rank shards so a partial capture or an incorrect
        # row order cannot match the reference.
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        layer = types.SimpleNamespace(
            self_attn=types.SimpleNamespace(rotary_emb=None),
            input_layernorm=torch.nn.Identity(),
        )
        decoder = types.SimpleNamespace(
            layers=[layer, layer, layer],
            rotary_emb=None,
            _capture_slots_by_layer={0: [1, 2], 2: [0]},
        )
        initial = torch.arange(8).reshape(4, 2).float()

        def gather(local_rows):
            return torch.cat((local_rows, local_rows + 10))

        kernels = types.SimpleNamespace(
            start=0,
            end=3,
            weights=[None] * 3,
            tp_rank=0,
            tp_size=2,
            a3_variant="default",
            capture_dflash=False,
            exit_gather_bf16=False,
            a1_ag_under_d=False,
            debug_compare=False,
            eplb_calib_active=False,
            no_next_norm=torch.ones(4),
            _rope_tables=lambda *_: (torch.ones(1), torch.zeros(1)),
            _prepare_valid_tokens=lambda *_: None,
            _prepare_eplb_calib_tokens=lambda *_: None,
            _layer=lambda **kw: (kw["stream_shard"] + 1.125, kw["attn_in"]),
            _gather=gather,
        )
        # Read the weight of the next norm, as a real layer would.
        layer.input_layernorm.weight = torch.nn.Parameter(torch.ones(4))
        batch = types.SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        self.assertFalse(
            MusePrefillKernels.handles(kernels, batch, 8192, capturing=True)
        )
        kernels.capture_dflash = True
        self.assertTrue(
            MusePrefillKernels.handles(kernels, batch, 8192, capturing=True)
        )
        captured = {}
        out = MusePrefillKernels.forward(
            kernels, decoder, torch.arange(4), initial, batch, captured_by_slot=captured
        )
        rows = initial.view(2, 4)
        self.assertTrue(torch.equal(out, gather(rows + 3 * 1.125)))
        for slot, add in ((0, 3 * 1.125), (1, 1.125), (2, 1.125)):
            self.assertEqual(captured[slot].dtype, torch.bfloat16)
            self.assertTrue(
                torch.equal(captured[slot], gather((rows + add).bfloat16()))
            )

    def test_default_off_flags_and_validation(self):
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META.get())
        self.assertEqual(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE.get(), "")
        self.assertEqual(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REPLICAS.get(), 0)
        self.assertEqual(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH.get(), "")
        self.assertEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS.get(), 0
        )
        self.assertEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_SKIP_FORWARDS.get(), 0
        )
        self.assertEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MIN_CALIB_TOKENS.get(), 256
        )
        self.assertEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REBALANCE_INTERVAL.get(), 16
        )
        self.assertAlmostEqual(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_SKEW_THRESHOLD.get(), 1.22
        )
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_MOE_TOKENS.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.get())

        self.assertEqual(resolve_prefill_eplb_config("", 0), ("", 0))
        self.assertEqual(resolve_prefill_eplb_config("perm", 0), ("perm", 0))
        self.assertEqual(resolve_prefill_eplb_config("stripe", 4), ("stripe", 4))
        self.assertEqual(resolve_prefill_eplb_config("home", 4), ("home", 4))
        self.assertEqual(resolve_prefill_eplb_config("q_block", 0), ("q_block", 4))
        with self.assertRaises(ValueError):
            resolve_prefill_eplb_config("perm", 4)
        with self.assertRaises(ValueError):
            resolve_prefill_eplb_config("q_block", 12)
        with self.assertRaises(ValueError):
            resolve_prefill_eplb_config("unknown_mode", 4)

    def test_tuned_block_sizes_registered_for_replicas(self):
        register_ep16_tuned_block_sizes()
        fp4_tbl = FUSED_RS_TUNED_BLOCK_SIZES["TPU7x"]["lhs_bfloat16_rhs_float4_e2m1fn"]
        fp8_tbl = FUSED_RS_TUNED_BLOCK_SIZES["TPU7x"]["lhs_bfloat16_rhs_float8_e4m3fn"]
        for g in (32, 34, 36, 40):
            key = f"k-4096_n-8192_g-{g}_lhs_quant_rhs_bs-16+k-4096_n-4096_g-{g}_lhs_quant_rhs_bs-16"
            self.assertIn(key, fp4_tbl)
            self.assertEqual(fp4_tbl[key]["65536_silu"][0], 128)
            self.assertEqual(fp4_tbl[key]["131072_silu"][0], 256)
            if g > 32:
                self.assertEqual(fp4_tbl[key]["49152_silu"][0], 128)
                self.assertEqual(fp4_tbl[key]["16384_silu"][0], 128)
                for bs_sfx in ("2048", "4096"):
                    fp8_key = (
                        f"k-4096_n-8192_g-{g}_lhs_quant_rhs_bs-{bs_sfx}+"
                        f"k-4096_n-4096_g-{g}_lhs_quant_rhs_bs-{bs_sfx}"
                    )
                    self.assertIn(fp8_key, fp8_tbl)
                    self.assertEqual(fp8_tbl[fp8_key]["131072_silu"][0], 256)

    def test_perm_bijection_and_balance(self):
        rng = np.random.default_rng(42)
        dist_cases = [
            np.zeros(256, dtype=np.int64),
            np.full(256, 256, dtype=np.int64),
            rng.poisson(256, size=256).astype(np.int64),
        ]
        skew = rng.poisson(200, size=256).astype(np.int64)
        skew[[7, 19, 143, 211]] = [8192, 6000, 4500, 3200]
        dist_cases.append(skew)

        for counts in dist_cases:
            for n_rep in (0, 2, 4, 8):
                r2g, col_perm = compute_zeroglue_eplb_perm(
                    counts, ep_size=8, num_local_base=32, num_replicas=n_rep, tile_m=128
                )
                self.assertEqual(col_perm.shape, (256,))
                self.assertEqual(r2g.shape, (8, 32 + n_rep))
                np.testing.assert_array_equal(np.sort(col_perm), np.arange(256))
                np.testing.assert_array_equal(r2g[:, :32].reshape(-1), col_perm)
                for r in range(8):
                    np.testing.assert_array_equal(r2g[r, 32:], col_perm[:n_rep])

    def test_virtual_id_remap_exact_expert_identity_and_pad_spread(self):
        rng = np.random.default_rng(7)
        counts = rng.integers(50, 400, size=256, dtype=np.int64)
        counts[[3, 44, 99, 180]] = [5000, 4200, 3800, 3100]

        for n_rep in (0, 4, 8):
            e_loc = 32 + n_rep
            r2g, col_perm = compute_zeroglue_eplb_perm(
                counts, ep_size=8, num_local_base=32, num_replicas=n_rep, tile_m=128
            )
            for num_tokens, quantum in ((64, 16), (2048, 128), (8192, 128)):
                # Build synthetic ragged batch where last 20% of tokens are identical pad rows
                real_t = int(num_tokens * 0.8)
                real_ids = np.stack(
                    [rng.choice(256, 8, replace=False) for _ in range(real_t)]
                ).astype(np.int32)
                pad_row = np.array([0, 1, 2, 3, 10, 42, 99, 150], dtype=np.int32)
                pad_ids = np.broadcast_to(pad_row, (num_tokens - real_t, 8)).copy()
                col_ids = jnp.asarray(np.concatenate([real_ids, pad_ids], axis=0))

                modes = ("perm",) if n_rep == 0 else ("stripe", "home", "q_block")
                for mode in modes:
                    v_global, v_loc0 = remap_zeroglue_virtual_ids(
                        col_ids,
                        jnp.int32(0),
                        ep_size=8,
                        num_local_experts=e_loc,
                        mode=mode,
                        quantum=quantum,
                    )
                    v_np = np.asarray(v_global)
                    self.assertTrue(np.all((v_np >= 0) & (v_np < 8 * e_loc)))
                    resolved_orig = r2g[v_np // e_loc, v_np % e_loc]
                    expected_orig = col_perm[np.asarray(col_ids)]
                    np.testing.assert_array_equal(resolved_orig, expected_orig)

                    t_loc = num_tokens // 8
                    np.testing.assert_array_equal(np.asarray(v_loc0), v_np[:t_loc])
                    _, v_loc5 = remap_zeroglue_virtual_ids(
                        col_ids,
                        jnp.int32(5),
                        ep_size=8,
                        num_local_experts=e_loc,
                        mode=mode,
                        quantum=quantum,
                    )
                    np.testing.assert_array_equal(
                        np.asarray(v_loc5), v_np[5 * t_loc : 6 * t_loc]
                    )

                if n_rep == 4 and num_tokens == 8192:
                    # Verify q_block spreads replicated experts evenly across ranks despite tail pad spike
                    v_qb, _ = remap_zeroglue_virtual_ids(
                        col_ids,
                        jnp.int32(0),
                        ep_size=8,
                        num_local_experts=e_loc,
                        mode="q_block",
                        quantum=128,
                    )
                    owners_qb = np.asarray(v_qb).reshape(-1) // e_loc
                    per_rank_qb = np.bincount(owners_qb, minlength=8)
                    self.assertLess(per_rank_qb.max() / per_rank_qb.mean(), 1.18)

    def test_end_to_end_router_moe_bit_exactness_and_ties(self):
        rng = np.random.default_rng(123)
        T, H, E, TOPK = 64, 32, 256, 8
        x = rng.standard_normal((T, H)).astype(np.float32)
        wrouter = (rng.standard_normal((H, E)) * 0.05).astype(np.float32)
        bias = (rng.standard_normal((E,)) * 0.05).astype(np.float32)
        expert_vecs = rng.standard_normal((E, H)).astype(np.float32)

        def run_ref(x_in, wr, b):
            scores = 1.0 / (1.0 + np.exp(-(x_in @ wr) * 0.176776695))
            # Exact JAX top_k behavior via jax.lax.top_k
            _, tk_ids = jax.lax.top_k(jnp.asarray(scores + b[None, :]), TOPK)
            tk_ids_np = np.asarray(tk_ids, dtype=np.int32)
            sel = np.take_along_axis(scores, tk_ids_np, axis=-1)
            tk_w = (sel / (sel.sum(axis=-1, keepdims=True) + 1e-15)).astype(np.float32)
            return tk_ids_np, tk_w

        ref_ids, ref_w = run_ref(x, wrouter, bias)
        ref_out = (expert_vecs[ref_ids] * ref_w[:, :, None]).sum(axis=1)

        counts = np.bincount(ref_ids.reshape(-1), minlength=E)
        for n_rep, mode in (
            (0, "perm"),
            (4, "stripe"),
            (4, "home"),
            (4, "q_block"),
            (8, "stripe"),
        ):
            e_loc = 32 + n_rep
            r2g, col_perm = compute_zeroglue_eplb_perm(
                counts, ep_size=8, num_local_base=32, num_replicas=n_rep, tile_m=16
            )
            perm_ids, perm_w = run_ref(x, wrouter[:, col_perm], bias[col_perm])
            v_ids, _ = remap_zeroglue_virtual_ids(
                jnp.asarray(perm_ids),
                jnp.int32(0),
                ep_size=8,
                num_local_experts=e_loc,
                mode=mode,
                quantum=16,
            )
            v_np = np.asarray(v_ids)
            phys_vecs = expert_vecs[r2g.reshape(-1)]
            perm_out = (phys_vecs[v_np] * perm_w[:, :, None]).sum(axis=1)
            np.testing.assert_array_equal(perm_out, ref_out)

        # Torch fallback topk mapping check
        t_ids = torch.from_numpy(perm_ids.copy())
        mapped_t = map_torch_fallback_topk_ids(t_ids, num_replicas=8, ep_size=8).numpy()
        np.testing.assert_array_equal(
            r2g[mapped_t // 40, mapped_t % 40], col_perm[perm_ids]
        )

    def test_sp_off_bf16_split_preserves_virtual_ids_above_256(self):
        # Verify the hi/lo bf16 split in fused_moe.py preserves all virtual IDs 0..319 bit-identically
        all_ids = jnp.arange(320, dtype=jnp.int32)
        hi = jax.lax.shift_right_logical(all_ids, 7).astype(jnp.bfloat16)
        lo = (all_ids & jnp.int32(127)).astype(jnp.bfloat16)
        reconstructed = jax.lax.shift_left(hi.astype(jnp.int32), 7) + lo.astype(
            jnp.int32
        )
        np.testing.assert_array_equal(np.asarray(reconstructed), np.arange(320))

    def test_stats_file_roundtrip(self):
        counts = np.arange(4 * 256, dtype=np.int64).reshape(4, 256)
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in (".json", ".npy", ".pt"):
                p = str(Path(tmpdir) / f"stats{ext}")
                save_eplb_stats_file(p, counts, start_layer=1)
                loaded = load_eplb_stats_file(p, num_layers=4, start_layer=1)
                np.testing.assert_array_equal(loaded, counts)

    def test_skip_pad_sentinel_routing_and_remap(self):
        rng = np.random.default_rng(2026)
        T, EP, TOPK = 256, 8, 8
        t_loc = T // EP
        raw_ids = np.stack(
            [rng.choice(256, TOPK, replace=False) for _ in range(T)]
        ).astype(np.int32)
        for n_rep, mode in ((0, "perm"), (4, "stripe"), (4, "home"), (4, "q_block")):
            e_loc = 32 + n_rep
            num_e = EP * e_loc
            for valid_q in (0, 1, 2, 31, 32, 33, 197, T):
                sentinel_base = EP * 32
                masked = np.where(
                    (np.arange(T)[:, None] < valid_q), raw_ids, sentinel_base
                ).astype(np.int32)
                if n_rep > 0:
                    v_all, _ = remap_zeroglue_virtual_ids(
                        jnp.asarray(masked),
                        jnp.int32(0),
                        ep_size=EP,
                        num_local_experts=e_loc,
                        mode=mode,
                        quantum=16,
                    )
                    v_np = np.asarray(v_all)
                else:
                    v_np = masked
                self.assertTrue(np.all(v_np[valid_q:] == num_e))
                if valid_q > 0:
                    self.assertTrue(
                        np.all((v_np[:valid_q] >= 0) & (v_np[:valid_q] < num_e))
                    )

                total_gs = 0
                for r in range(EP):
                    go_val = jnp.int32(r * e_loc)
                    _, gs, gsa, t_recv = compute_block_scatter_routing(
                        jnp.asarray(v_np),
                        go_val,
                        jnp.int32(r),
                        ep_size=EP,
                        num_local_experts=e_loc,
                        topk=TOPK,
                        stable_sort=True,
                        has_pad_sentinel=True,
                    )
                    gs_np = np.asarray(gs)
                    total_gs += int(gs_np.sum())
                    my_chunk = v_np[r * t_loc : (r + 1) * t_loc]
                    expected_recv = int(
                        np.sum(
                            (my_chunk < num_e)
                            & ((my_chunk < r * e_loc) | (my_chunk >= (r + 1) * e_loc))
                        )
                    )
                    self.assertEqual(int(t_recv), expected_recv)
                    self.assertEqual(int(gsa[1]), expected_recv)
                self.assertEqual(total_gs, valid_q * TOPK)

    def test_online_eplb_controller_dummy_warmup_filtering_and_multi_rebalance(self):
        rng = np.random.default_rng(99)
        num_layers = 3
        ctrl = OnlineEplbController(
            num_layers=num_layers,
            num_experts=256,
            ep_size=8,
            num_local_base=32,
            num_replicas=4,
            mode="stripe",
            calib_forwards=4,
            skip_forwards=1,
            min_calib_tokens=256,
            rebalance_interval=6,
            skew_threshold=1.22,
            tile_m=128,
            topk=8,
        )
        cur_perms = [np.arange(256, dtype=np.int32) for _ in range(num_layers)]

        # 1. Startup precompile / dummy warmup traps must all be rejected with 0% pollution:
        real_like = rng.poisson(210, size=(num_layers, 256)).astype(np.int64)
        # (a) in_warmup_queue=True
        out, _ = ctrl.observe_batch(real_like, cur_perms, in_warmup_queue=True)
        self.assertIsNone(out)
        self.assertEqual(ctrl.last_skip_reason, "warmup_queue")
        # (b) _precompile_extend bs=64, seq_lens=1 -> 64*8 = 512 assignments (< 256 tokens)
        dummy_bs64 = np.zeros((num_layers, 256), dtype=np.int64)
        dummy_bs64[:, :64] = 8
        out, _ = ctrl.observe_batch(dummy_bs64, cur_perms, in_warmup_queue=False)
        self.assertIsNone(out)
        self.assertEqual(ctrl.last_skip_reason, "below_min_calib_tokens")
        # (c) HTTP server warmup probe (10 tokens)
        dummy_http = np.zeros((num_layers, 256), dtype=np.int64)
        dummy_http[:, :8] = 10
        out, _ = ctrl.observe_batch(dummy_http, cur_perms, in_warmup_queue=False)
        self.assertIsNone(out)
        self.assertEqual(ctrl.last_skip_reason, "below_min_calib_tokens")
        # (d) Constant-token benchmark warmup zeros(8192) -> 8 experts get 100%
        dummy_zeros = np.zeros((num_layers, 256), dtype=np.int64)
        dummy_zeros[:, [2, 17, 33, 88, 101, 144, 200, 250]] = 8192
        out, _ = ctrl.observe_batch(dummy_zeros, cur_perms, in_warmup_queue=False)
        self.assertIsNone(out)
        self.assertEqual(ctrl.last_skip_reason, "degenerate_distinct_experts")
        # (e) cp_extend [1, 0, ..., 0] where >=24 distinct experts appear on early tokens
        #     but >90% of assignments still pile onto 8 experts:
        dummy_cp = np.zeros((num_layers, 256), dtype=np.int64)
        dummy_cp[:, :32] = 10  # 320 assignments across 32 experts
        dummy_cp[:, :8] += 8000  # 64,000 assignments on 8 experts (> 99%)
        self.assertFalse(is_eligible_eplb_calib_batch(dummy_cp)[0])
        out, _ = ctrl.observe_batch(dummy_cp, cur_perms, in_warmup_queue=False)
        self.assertIsNone(out)
        self.assertEqual(ctrl.last_skip_reason, "degenerate_top_k_concentration")
        self.assertEqual(ctrl.skipped_dummy_steps, 5)
        self.assertEqual(ctrl.eligible_steps, 0)
        self.assertEqual(int(ctrl.window_orig_counts.sum()), 0)

        # Helper to simulate runtime column-space histogram under current `cur_perms`:
        def make_runtime_hist(orig_profile_2d: np.ndarray) -> np.ndarray:
            rt = np.empty_like(orig_profile_2d)
            for l_i in range(num_layers):
                rt[l_i] = orig_profile_2d[l_i, cur_perms[l_i]]
            return rt

        # Build Domain A (hot original experts [5, 37, 69, 101] all on rank 0/1/2/3 + heavy rank 2)
        dom_a = rng.integers(80, 180, size=(num_layers, 256), dtype=np.int64)
        dom_a[:, 64:80] = 900  # pile onto rank 2
        dom_a[:, [5, 37, 69, 101]] = [4800, 4200, 3900, 3500]

        # First eligible batch is skipped due to skip_forwards=1:
        out, _ = ctrl.observe_batch(make_runtime_hist(dom_a), cur_perms)
        self.assertIsNone(out)
        self.assertEqual(ctrl.skipped_warmup_steps, 1)

        # Next 4 eligible Domain-A batches trigger Rebalance #1 on step 4:
        for step in range(4):
            jitter = rng.integers(-10, 11, size=dom_a.shape, dtype=np.int64)
            batch_orig = np.maximum(1, dom_a + jitter)
            out, eff_tm = ctrl.observe_batch(make_runtime_hist(batch_orig), cur_perms)
            if step < 3:
                self.assertIsNone(out)
            else:
                self.assertIsNotNone(out)
                self.assertEqual(eff_tm, 128 * 4)
                self.assertEqual(ctrl.rebalance_count, 1)
                for l_i in range(num_layers):
                    _, cur_perms[l_i] = compute_zeroglue_eplb_perm(
                        out[l_i],
                        ep_size=8,
                        num_local_base=32,
                        num_replicas=4,
                        tile_m=eff_tm,
                    )
                    self.assertEqual(set(cur_perms[l_i][:4].tolist()), {5, 37, 69, 101})

        # 6 stationary Domain-A batches -> skew stays low (~1.05 <= 1.22), 0 redundant rebalances:
        for _ in range(6):
            jitter = rng.integers(-10, 11, size=dom_a.shape, dtype=np.int64)
            batch_orig = np.maximum(1, dom_a + jitter)
            out, _ = ctrl.observe_batch(make_runtime_hist(batch_orig), cur_perms)
            self.assertIsNone(out)
        self.assertEqual(ctrl.rebalance_count, 1)
        self.assertLessEqual(ctrl.last_observed_skew, 1.22)

        # Shift abruptly to orthogonal Domain B (hot original experts [190, 205, 221, 249]):
        dom_b = rng.integers(80, 180, size=(num_layers, 256), dtype=np.int64)
        dom_b[:, 192:212] = 950
        dom_b[:, [190, 205, 221, 249]] = [5000, 4400, 4000, 3600]
        expected_sum_b = np.zeros_like(dom_b)
        for step in range(6):
            jitter = rng.integers(-10, 11, size=dom_b.shape, dtype=np.int64)
            batch_orig = np.maximum(1, dom_b + jitter)
            expected_sum_b += batch_orig
            # Notice `make_runtime_hist` emits counts in Permuted-A column coordinates!
            out, eff_tm = ctrl.observe_batch(make_runtime_hist(batch_orig), cur_perms)
            if step < 5:
                self.assertIsNone(out)
            else:
                self.assertIsNotNone(out)
                self.assertEqual(eff_tm, 128 * 6)
                self.assertEqual(ctrl.rebalance_count, 2)
                # Canonicalized `out` must equal `expected_sum_b` bit-for-bit despite Permuted-A!
                np.testing.assert_array_equal(out, expected_sum_b)
                for l_i in range(num_layers):
                    _, cur_perms[l_i] = compute_zeroglue_eplb_perm(
                        out[l_i],
                        ep_size=8,
                        num_local_base=32,
                        num_replicas=4,
                        tile_m=eff_tm,
                    )
                    self.assertEqual(
                        set(cur_perms[l_i][:4].tolist()), {190, 205, 221, 249}
                    )

        skew_after_b, _ = compute_eplb_tile_skew_stats(
            expected_sum_b,
            cur_perms,
            ep_size=8,
            num_local_base=32,
            num_replicas=4,
            mode="stripe",
            tile_m=128,
            window_steps=6,
        )
        self.assertLess(skew_after_b, 1.15)

    def test_prefill_eplb_manager_replica_init_warm_and_two_rebalances_exact(self):
        # Simulate 8 TP ranks with 1 FP8 layer + 1 NVFP4 layer and verify PrefillEplbManager
        # preserves exact expert weights at init (before rebalance), after warm_rebalance_kernels,
        # after Rebalance #1, and after Rebalance #2, plus callback_factory pickling.
        tp_size = 8
        num_layers = 2
        num_replicas = 4
        rng = np.random.default_rng(314)

        orig_gate = [
            torch.from_numpy(rng.standard_normal((256, 16)).astype(np.float32))
            for _ in range(num_layers)
        ]
        orig_bias = [
            torch.from_numpy(rng.standard_normal((256,)).astype(np.float32))
            for _ in range(num_layers)
        ]
        # Layer 0: FP8 experts; Layer 1: packed NVFP4 experts
        orig_w13_u8_l1 = torch.from_numpy(
            rng.integers(0, 256, size=(256, 8, 16), dtype=np.uint8)
        )
        orig_w2_u8_l1 = torch.from_numpy(
            rng.integers(0, 256, size=(256, 8, 8), dtype=np.uint8)
        )
        orig_w13_l0 = torch.from_numpy(
            rng.standard_normal((256, 8, 16)).astype(np.float32)
        ).to(torch.float8_e4m3fn)
        orig_w2_l0 = torch.from_numpy(
            rng.standard_normal((256, 8, 8)).astype(np.float32)
        ).to(torch.float8_e4m3fn)
        orig_s13 = [
            torch.arange(256 * 4, dtype=torch.float32).reshape(256, 1, 1, 4)
            + l_i * 1000
            for l_i in range(num_layers)
        ]
        orig_s2 = [
            torch.arange(256 * 2, dtype=torch.float32).reshape(256, 1, 1, 2)
            + l_i * 2000
            for l_i in range(num_layers)
        ]

        rank_decoders = []
        rank_weights = []
        for r in range(tp_size):
            layers = torch.nn.ModuleList()
            wlist = torch.nn.ModuleList()
            lo, hi = r * 32, (r + 1) * 32
            for l_i in range(num_layers):
                layer = torch.nn.Module()
                moe = torch.nn.Module()
                gate = torch.nn.Module()
                gate.weight = torch.nn.Parameter(
                    orig_gate[l_i].clone(), requires_grad=False
                )
                moe.gate = gate
                moe.e_score_correction_bias = torch.nn.Parameter(
                    orig_bias[l_i].clone(), requires_grad=False
                )
                exp = torch.nn.Module()
                if l_i == 0:
                    w13_r = orig_w13_l0[lo:hi].clone()
                    w2_r = orig_w2_l0[lo:hi].clone()
                else:
                    w13_r = torch.empty((32, 8, 16), dtype=torch.float4_e2m1fn_x2)
                    w13_r.view(torch.uint8).copy_(orig_w13_u8_l1[lo:hi])
                    w2_r = torch.empty((32, 8, 8), dtype=torch.float4_e2m1fn_x2)
                    w2_r.view(torch.uint8).copy_(orig_w2_u8_l1[lo:hi])
                exp.w13_weight = torch.nn.Parameter(w13_r, requires_grad=False)
                exp.w2_weight = torch.nn.Parameter(w2_r, requires_grad=False)
                exp.w13_weight_scale = torch.nn.Parameter(
                    orig_s13[l_i][lo:hi].clone(), requires_grad=False
                )
                exp.w2_weight_scale = torch.nn.Parameter(
                    orig_s2[l_i][lo:hi].clone(), requires_grad=False
                )
                exp.num_local_experts = 32
                exp.register_buffer(
                    "_tpu_fused_rs_group_offset",
                    torch.tensor([r * 32], dtype=torch.int32),
                )
                moe.experts = exp
                layer.mlp = moe
                layers.append(layer)
                lw = torch.nn.Module()
                lw.register_buffer(
                    "wrouter", orig_gate[l_i].t().float().contiguous(), persistent=False
                )
                wlist.append(lw)
            dec = torch.nn.Module()
            dec.layers = layers
            dec.norm = torch.nn.Module()
            dec.norm.weight = torch.nn.Parameter(torch.ones(16), requires_grad=False)
            rank_decoders.append(dec)
            rank_weights.append(wlist)

        # Synchronous simulated cross-rank gather over rank_decoders when gathering expert slices:
        # We run each phase across all 8 ranks using a shared barrier-free tensor map.
        active_phase_tensors: list[torch.Tensor] = [None] * tp_size  # type: ignore[list-item]

        def run_lockstep(step_fn):
            # Because each rank calls gather_fn the exact same number of times in order,
            # record per-rank generator coroutines or pre-populate via lockstep threads.
            import threading

            bar = threading.Barrier(tp_size)
            errs: list[BaseException] = []

            def make_gather(r_idx: int):
                def _gather(t: torch.Tensor) -> torch.Tensor:
                    active_phase_tensors[r_idx] = t
                    bar.wait()
                    out = torch.cat(active_phase_tensors, dim=0)
                    bar.wait()
                    return out

                return _gather

            def worker(r_idx: int):
                try:
                    step_fn(r_idx, make_gather(r_idx))
                except BaseException as exc:  # noqa: BLE001
                    errs.append(exc)
                    bar.abort()

            threads = [
                threading.Thread(target=worker, args=(r,)) for r in range(tp_size)
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            if errs:
                raise errs[0]

        managers: list[PrefillEplbManager] = [None] * tp_size  # type: ignore[list-item]

        def init_and_warm(r_idx: int, gather_fn):
            mgr = PrefillEplbManager(
                decoder=rank_decoders[r_idx],
                weights_list=rank_weights[r_idx],
                gather_fn=gather_fn,
                start=0,
                end=num_layers,
                tp_size=tp_size,
                tp_rank=r_idx,
                mode="stripe",
                num_replicas=num_replicas,
            )
            mgr.warm_rebalance_kernels()
            managers[r_idx] = mgr

        run_lockstep(init_and_warm)

        def assert_exact_mapping(expected_r2g_per_layer, expected_perm_per_layer):
            for l_i in range(num_layers):
                exp_perm = expected_perm_per_layer[l_i]
                exp_r2g = expected_r2g_per_layer[l_i]
                for r in range(tp_size):
                    moe = rank_decoders[r].layers[l_i].mlp
                    lw = rank_weights[r][l_i]
                    idx_p = torch.from_numpy(exp_perm).long()
                    idx_s = torch.from_numpy(exp_r2g[r]).long()
                    self.assertTrue(
                        torch.equal(moe.gate.weight.data, orig_gate[l_i][idx_p])
                    )
                    self.assertTrue(
                        torch.equal(
                            moe.e_score_correction_bias.data, orig_bias[l_i][idx_p]
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            lw.wrouter, orig_gate[l_i][idx_p].t().float().contiguous()
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            moe.experts.w13_weight_scale.data, orig_s13[l_i][idx_s]
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            moe.experts.w2_weight_scale.data, orig_s2[l_i][idx_s]
                        )
                    )
                    if l_i == 0:
                        self.assertTrue(
                            torch.equal(
                                moe.experts.w13_weight.data.view(torch.uint8),
                                orig_w13_l0[idx_s].view(torch.uint8),
                            )
                        )
                    else:
                        self.assertTrue(
                            torch.equal(
                                moe.experts.w13_weight.data.view(torch.uint8),
                                orig_w13_u8_l1[idx_s],
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                moe.experts.w2_weight.data.view(torch.uint8),
                                orig_w2_u8_l1[idx_s],
                            )
                        )

        # Phase 0: Before Rebalance #1 (right after __init__ + warm_rebalance_kernels)
        init_perm = np.arange(256, dtype=np.int32)
        init_r2g = np.stack(
            [
                np.concatenate(
                    [
                        np.arange(r * 32, (r + 1) * 32, dtype=np.int32),
                        np.arange(num_replicas, dtype=np.int32),
                    ]
                )
                for r in range(tp_size)
            ]
        )
        assert_exact_mapping([init_r2g, init_r2g], [init_perm, init_perm])

        # Phase 1: Rebalance #1 on Domain A counts
        counts_a = rng.integers(50, 300, size=(num_layers, 256), dtype=np.int64)
        counts_a[0, [11, 77, 140, 203]] = [5000, 4100, 3800, 3200]
        counts_a[1, [29, 91, 155, 240]] = [4900, 4300, 3600, 3100]

        def reb_a(r_idx: int, gather_fn):
            managers[r_idx].gather_fn = gather_fn
            managers[r_idx].rebalance_from_counts(counts_a, tile_m=512)

        run_lockstep(reb_a)
        r2g_a, perm_a = zip(
            *(
                compute_zeroglue_eplb_perm(
                    counts_a[l_i],
                    ep_size=8,
                    num_local_base=32,
                    num_replicas=4,
                    tile_m=512,
                )
                for l_i in range(num_layers)
            )
        )
        assert_exact_mapping(r2g_a, perm_a)

        # Phase 2: Rebalance #2 on orthogonal Domain B counts (starting from Permuted-A weights!)
        counts_b = rng.integers(50, 300, size=(num_layers, 256), dtype=np.int64)
        counts_b[0, [3, 64, 199, 254]] = [5200, 4500, 3900, 3400]
        counts_b[1, [8, 120, 188, 251]] = [5100, 4600, 3700, 3300]

        def reb_b(r_idx: int, gather_fn):
            managers[r_idx].gather_fn = gather_fn
            managers[r_idx].rebalance_from_counts(counts_b, tile_m=768)

        run_lockstep(reb_b)
        r2g_b, perm_b = zip(
            *(
                compute_zeroglue_eplb_perm(
                    counts_b[l_i],
                    ep_size=8,
                    num_local_base=32,
                    num_replicas=4,
                    tile_m=768,
                )
                for l_i in range(num_layers)
            )
        )
        assert_exact_mapping(r2g_b, perm_b)

        # Verify _make_eplb_hist_callback is picklable for SPMDCompiler and binds all 3 shapes
        restored_factory = pickle.loads(pickle.dumps(_make_eplb_hist_callback))
        dummy_pk = MusePrefillKernels.__new__(MusePrefillKernels)
        self.assertEqual(restored_factory(dummy_pk), dummy_pk._consume_eplb_calib_hists)
        dec_wrap = types.SimpleNamespace(prefill_kernels=dummy_pk)
        self.assertEqual(restored_factory(dec_wrap), dummy_pk._consume_eplb_calib_hists)
        top_wrap = types.SimpleNamespace(model=dec_wrap)
        self.assertEqual(restored_factory(top_wrap), dummy_pk._consume_eplb_calib_hists)

    def test_short_prefill_tiles_selection_and_decode_isolation(self):
        from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
            _select_fused_rs_block_sizes,
        )

        mesh = jax.sharding.AbstractMesh(
            (8,),
            ("model",),
            abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
        )
        self.enterContext(jax.sharding.use_abstract_mesh(mesh))
        self.enterContext(
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.tuned_block_sizes.get_pallas_tpu_device_kind",
                return_value="TPU7x",
            )
        )

        def pick(size_m: int, size_group: int = 32):
            bs = _select_fused_rs_block_sizes(
                size_m=size_m,
                size_k1=4096,
                size_n1=8192,
                size_k2=4096,
                size_n2=4096,
                size_group=size_group,
                size_lhs_group=8 * size_group,
                ep_size=8,
                out_dtype=jnp.bfloat16,
                w1_dtype=jnp.float4_e2m1fn,
                w2_dtype=jnp.float4_e2m1fn,
                is_quantized=True,
                quant_block_size=16,
                act_fn="silu",
                fp8_direct_write=False,
            )
            return (
                bs.tile_m,
                bs.tile_k1,
                bs.tile_n1,
                bs.tile_k2,
                bs.tile_n2,
                bs.num_w1_bufs,
                bs.num_w2_bufs,
            )

        # Default-off preserves stock (128,4096,4096,4096,4096,1,1) for M <= 16384:
        for m in (2048, 4096, 8192, 16384):
            self.assertEqual(pick(m, 32), (128, 4096, 4096, 4096, 4096, 1, 1))

        with envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.override(True):
            # Decode M <= 2048 stays untouched:
            self.assertEqual(pick(2048, 32), (128, 4096, 4096, 4096, 4096, 1, 1))
            # Short/medium prefill M in {4096, 8192, 16384} (T=512, 1024, 2048) uses cached 2x2:
            for m in (4096, 8192, 16384):
                for g in (32, 36, 40):
                    self.assertEqual(pick(m, g), (128, 4096, 2048, 4096, 2048, 2, 2))
            # T >= 4096 (M >= 32768) and T=16384 (M=131072) stay unchanged:
            self.assertEqual(pick(32768, 32), (128, 4096, 2048, 4096, 2048, 2, 2))
            self.assertEqual(pick(131072, 32)[0], 256)

    def test_sortfree_scatter_and_mode3_match_sorted_routing(self):
        ep, e_loc, topk, t = 8, 32, 8, 2048
        num_e = ep * e_loc
        rng = np.random.default_rng(7)
        ids = np.stack(
            [rng.choice(num_e, topk, replace=False) for _ in range(t)]
        ).astype(np.int32)
        ids[t - 300 :] = num_e  # padded tail carries the sentinel
        with envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.override(True):
            for r in range(ep):
                outs = []
                for variant in (0, 1, 2):
                    with envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER.override(variant):
                        outs.append(
                            [
                                np.asarray(x)
                                for x in compute_block_scatter_routing(
                                    jnp.asarray(ids),
                                    jnp.int32(r * e_loc),
                                    jnp.int32(r),
                                    ep_size=ep,
                                    num_local_experts=e_loc,
                                    topk=topk,
                                    has_pad_sentinel=True,
                                )
                            ]
                        )
                rel = ids - r * e_loc
                local = (rel >= 0) & (rel < e_loc)
                tok, slot = np.nonzero(local)
                order = np.lexsort((tok, rel[tok, slot]))
                ref_lhs = tok[order]
                for lhs, gs, gsa, t_recv in outs:
                    np.testing.assert_array_equal(lhs[: ref_lhs.size], ref_lhs)
                    np.testing.assert_array_equal(gs, outs[0][1])
                    np.testing.assert_array_equal(gsa, outs[0][2])
                    self.assertEqual(int(t_recv), int(outs[0][3]))

            local_ids = ids[:256].copy()
            local_ids[200:] = num_e
            ref = np.argsort(
                np.argsort(local_ids.reshape(-1), kind="stable"), kind="stable"
            )
            with envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_MODE.override(3):
                inv = np.asarray(
                    _compute_local_inv_perm(jnp.asarray(local_ids[:256]), num_e + 1)
                )
            valid = local_ids.reshape(-1) < num_e
            np.testing.assert_array_equal(inv.reshape(-1)[valid], ref[valid])

    def test_build_eplb_moe_op_sanitizes_multibucket_causal_chunk_spec(self):
        import re
        from jax.sharding import Mesh

        spec = "3072:1024,2048;2048:768,1280;1024:384,640;512:192,320"
        expected_ccs = "_ccs_3072_1024_2048__2048_768_1280__1024_384_640__512_192_320"
        devs = jax.devices()[:8]
        mesh = Mesh(np.array(devs), ("model",))
        recorded_names = []
        import sglang_tpu.pallas_jax_op as _pjo

        orig_export = _pjo.cpu_export_jax_op

        def _spy_export(name, *args, **kwargs):
            recorded_names.append(name)
            return orig_export(name, *args, **kwargs)

        with (
            patch(
                "sglang_tpu.moe.pallas.fused_moe_rs_op._get_mesh",
                return_value=mesh,
            ),
            patch(
                "sglang_tpu.pallas_jax_op.cpu_export_jax_op",
                side_effect=_spy_export,
            ),
        ):
            op_main = build_eplb_moe_op(
                h_q_local=8,
                h_kv_local=1,
                fused_glue=True,
                eplb_mode="perm",
                eplb_calib=False,
                op_extra_tag="_tp8_utest_ccs",
                causal_chunk_stream=True,
                causal_chunk_spec=spec,
            )
            op_agd = build_eplb_moe_op(
                h_q_local=8,
                h_kv_local=1,
                fused_glue=True,
                eplb_mode="perm",
                eplb_calib=False,
                op_extra_tag=f"_tp8_utest_ccs_agd{expected_ccs}",
                next_in_gather_rows_fn=lambda ts: ts // 3,
                causal_chunk_stream=True,
                causal_chunk_spec=spec,
            )

        self.assertIsNotNone(op_main)
        self.assertIsNotNone(op_agd)
        self.assertEqual(len(recorded_names), 2)
        for name in recorded_names:
            self.assertRegex(name, r"^[A-Za-z0-9_]+::[A-Za-z0-9_]+$")
            self.assertEqual(name.count(expected_ccs), 1, name)


if __name__ == "__main__":
    unittest.main()
