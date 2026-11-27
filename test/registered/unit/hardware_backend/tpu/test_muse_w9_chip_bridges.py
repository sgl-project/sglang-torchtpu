# SPDX-License-Identifier: Apache-2.0
"""Unit, Pallas interpret (eager + detect_races), NaN/subnormal, & TPU7x AOT tests for Wave-9 bridges."""

from __future__ import annotations

import ast
import inspect
import os
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _rms_norm_np(
    x: np.ndarray, gain: np.ndarray, eps: float, scale_input: bool
) -> np.ndarray:
    xf = x.astype(np.float32)
    gf = gain.reshape(1, -1).astype(np.float32)
    vin = xf * gf if scale_input else xf
    inv = 1.0 / np.sqrt(np.mean(vin * vin, axis=-1, keepdims=True) + np.float32(eps))
    return (xf * gf) * inv if scale_input else (xf * inv) * gf


def test_w9_env_default_and_lock_free_prewarm():
    """Verify CHIP_SHARDED_BRIDGES defaults off and prewarm covers TP=8 & TP=16 lock-free."""
    from sglang_tpu.envs import envs
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    assert envs.SGLANG_TPU_MUSE_DECODE_CHIP_SHARDED_BRIDGES.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_CHIP_B2_TP2_WPOST.default is False

    for rel_path in (
        "sglang_tpu/models/muse_spark.py",
        "sglang_tpu/decode_qkv_epilogue_tpu.py",
        "sglang_tpu/kernels/decode_collective_bridges.py",
        "sglang_tpu/kernels/decode_collective_bridges_w8.py",
    ):
        tree = ast.parse((_REPO_ROOT / rel_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                src = ast.unparse(node.test)
                for env_name in (
                    "SGLANG_TPU_MUSE_DECODE_CHIP_SHARDED_BRIDGES",
                    "SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS",
                    "SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT",
                    "SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI",
                    "SGLANG_TPU_MUSE_DECODE_CHIP_B2_TP2_WPOST",
                ):
                    if env_name in src and ("envs." in src or "tpu_envs." in src):
                        assert ".get()" in src, f"bare EnvField bool in `{src}`"

    captured = {}

    def _fake_export(
        name,
        fn,
        *,
        mesh=None,
        input_partition_specs=None,
        output_partition_specs=None,
    ):
        sig = inspect.signature(fn)
        captured[name] = (len(sig.parameters), input_partition_specs, output_partition_specs)
        return fn

    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("_W8_LOCK entered after prewarm!")

        def __exit__(self, *args):
            return False

    fake_mesh = object()
    with (
        mock.patch.object(w8, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export),
        mock.patch.dict(w8._EMBED_CHIP_SHARDED_OPS, {}, clear=True),
        mock.patch.dict(w8._ATTN_TO_MOE_CHIP_SH_OPS, {}, clear=True),
        mock.patch.dict(w8._MOE_TO_ATTN_CHIP_SH_OPS, {}, clear=True),
        mock.patch.dict(w8._MOE_POST_CHIP_SH_FINAL_OPS, {}, clear=True),
        mock.patch.dict(w8._AG_CHIP_CAP_OPS, {}, clear=True),
        mock.patch.dict(w8._ATTN_TO_MOE_TOK_SH_OPS, {}, clear=True),
        mock.patch.dict(w8._MOE_TO_ATTN_QKV_EPI_OPS, {}, clear=True),
        mock.patch.dict(w8._DUAL_QKV_EPI_GATE_OPS, {}, clear=True),
    ):
        for tp in (8, 16):
            w8.prewarm_w8_decode_bridge_ops(
                tp,
                3,
                bf16_router=True,
                tok_sharded=(tp == 8),
                chip_sharded=True,
                bridge_qkv_epi=True,
                fuse_final_norm=True,
                pallas_embed=True,
            )
        with mock.patch.object(w8, "_W8_LOCK", _ExplodingLock()):
            assert callable(w8.get_decode_vocab_embed_chip_sharded_op(8))
            for tp in (8, 16):
                assert callable(w8.get_ag_chip_sharded_capture_op(tp))
                for r16 in (False, True):
                    assert callable(
                        w8.get_attn_to_moe_chip_sharded_bridge_op(tp, bf16_router=r16)
                    )
                for sh in (False, True):
                    for ret_st in (False, True):
                        assert callable(
                            w8.get_moe_post_chip_sharded_final_norm_op(
                                tp, sharded_moe_in=sh, return_stream_too=ret_st
                            )
                        )
                    assert callable(
                        w8.get_moe_to_attn_chip_sharded_bridge_op(
                            tp, sharded_moe_in=sh, fuse_qkv_epi=False
                        )
                    )
                    for hr in (False, True):
                        assert callable(
                            w8.get_moe_to_attn_chip_sharded_bridge_op(
                                tp,
                                sharded_moe_in=sh,
                                fuse_qkv_epi=True,
                                has_rope=hr,
                            )
                        )
                        assert callable(
                            w8.get_moe_to_attn_qkv_epi_bridge_op(
                                tp, sharded_moe_in=sh, has_rope=hr
                            )
                        )
                for hr in (False, True):
                    assert callable(w8.get_fp8_dual_qkv_epi_gate_op(hr))

    for op_name, (n_params, in_specs, _out_specs) in captured.items():
        if in_specs is not None:
            assert len(in_specs) == n_params, f"{op_name}: {len(in_specs)} != {n_params}"


def test_muse_spark_chip_sharded_multilayer_relay_tp8_and_tp16():
    """Verify MuseSparkTextModel & MuseSparkDecoderLayer wire chip_sharded_stream on TP=8 & TP=16."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.models.muse_spark import MuseSparkTextModel
    from test.registered.unit.hardware_backend.tpu.test_muse_w7_col_proj import (
        _make_dummy_layer,
        _make_norm,
    )

    H, H_PRE = 1024, 512
    backend = types.SimpleNamespace(
        can_forward_decode_prepared=lambda *_: False,
        forward_decode_prepared_gated=lambda q, kv, g, *_: g.clone(),
    )
    fb = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        attn_backend=backend,
        return_logprob=False,
        capture_hidden_mode=None,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        extend_seq_lens=None,
    )

    for tp, B, expect_sh in ((8, 32, False), (8, 64, True), (16, 64, False), (16, 128, True)):
        n_chips = tp // 2
        r_pair = B // n_chips
        calls: list[str] = []

        def _fake_b1(st, ga, *_rest):
            calls.append(f"chip_b1:{st.shape[0]}:{ga.shape[0]}")
            return (
                st + 1.0,
                torch.zeros((B, 256), dtype=torch.float32),
                torch.zeros((B, H_PRE), dtype=torch.bfloat16),
            )

        def _fake_b2_fn(sh_flag):
            def _fn(st, mo, *_rest):
                calls.append(f"chip_b2:{sh_flag}:{st.shape[0]}:{mo.shape[0]}")
                return (
                    st + 1.0,
                    torch.zeros((B, 1280 // (tp // 8)), dtype=torch.bfloat16),
                    torch.zeros((B, 1024 // (tp // 8)), dtype=torch.bfloat16),
                )

            return _fn

        def _fake_fn_op(sh_flag):
            def _fn(st, mo, *_rest):
                calls.append(f"chip_fn:{sh_flag}:{st.shape[0]}:{mo.shape[0]}")
                return torch.ones((B, H), dtype=torch.bfloat16)

            return _fn

        model = object.__new__(MuseSparkTextModel)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(hidden_size=H, rms_norm_eps=1e-5)
        model.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.layers_to_capture = []
        model._capture_slots_by_layer = {}
        model.prefill_kernels = None
        model.rotary_emb = types.SimpleNamespace(cos_sin_pairs=None)
        model.start_layer = 0
        model.end_layer = 2
        layers = []
        for _ in range(2):
            lyr = _make_dummy_layer(3, H=H, H_PRE=H_PRE)
            lyr.self_attn.o_proj.tp_size = tp
            lyr.mlp.tp_size = tp
            lyr.self_attn.qkv_proj.weight = torch.nn.Parameter(
                torch.randn(1280 // (tp // 8), H, dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            lyr.self_attn.gate_proj.weight = torch.nn.Parameter(
                torch.randn(1024 // (tp // 8), H, dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            lyr.self_attn.o_proj.weight = torch.nn.Parameter(
                torch.randn(H, 1024 // (tp // 8), dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            lyr.self_attn._o_proj_w8k = None
            assert lyr.self_attn.prepare_w7_qkv_gate_fp8()
            assert lyr.self_attn.prepare_w7_o_proj_fp8()
            lyr.mlp.register_buffer(
                "_w8_wpre_pair_w8k",
                torch.zeros((H, H_PRE // 2), dtype=torch.float8_e4m3fn),
                persistent=False,
            )
            lyr.mlp.register_buffer(
                "_w8_spre_pair", torch.ones((H_PRE // 2,), dtype=torch.float32), persistent=False
            )
            lyr.self_attn.forward = lambda pos, ai, fb_, **kw: torch.zeros(
                (B, 1024 // (tp // 8)), dtype=torch.bfloat16
            )
            lyr.mlp.can_return_sharded_moe_out = lambda n_t, *, decode: True
            lyr.mlp.forward = lambda fh, ri, *, decode, return_unprojected_moe_out, return_sharded_moe_out, **kw: (
                torch.zeros(
                    (B // tp if return_sharded_moe_out else B, H_PRE),
                    dtype=torch.bfloat16,
                )
            )
            layers.append(lyr)
        model.layers = torch.nn.ModuleList(layers)
        model.norm = _make_norm(H)
        model._decode_norm_layout = 0
        model._decode_col_bridges = 3
        model._w9_chip_sharded_bridges = True
        model._w9_all_layers_chip_sharded_ready = True
        model._w9_chip_sharded_batches = (32, 64, 128, 256, 384) if tp == 8 else (64, 128, 256, 384)
        model._w9_chip_sharded_tp = tp
        model._w8_pallas_embed_enabled = False
        model._w8_pallas_embed_tp = 0

        embeds = torch.randn(B, H, dtype=torch.bfloat16)
        pos = torch.arange(B, dtype=torch.int32)
        par_stub = types.SimpleNamespace(tp_rank=3, tp_size=tp, attn_cp_size=1)
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": str(tp)}),
            mock.patch("sglang_tpu.models.muse_spark.get_parallel", return_value=par_stub),
            mock.patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=types.SimpleNamespace(sp_active=False),
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_attn_to_moe_chip_sharded_bridge_op",
                return_value=_fake_b1,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_to_attn_chip_sharded_bridge_op",
                side_effect=lambda w, *, sharded_moe_in=True, **kw: _fake_b2_fn(sharded_moe_in),
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_post_chip_sharded_final_norm_op",
                side_effect=lambda w, *, sharded_moe_in=True, **kw: _fake_fn_op(sharded_moe_in),
            ),
        ):
            out = model.forward(
                torch.zeros((B,), dtype=torch.int32),
                pos,
                fb,
                input_embeds=embeds,
            )
        assert out.shape == (B, H)
        exp_m = B // tp if expect_sh else B
        assert calls == [
            f"chip_b1:{r_pair}:{B}",
            f"chip_b2:{expect_sh}:{r_pair}:{exp_m}",
            f"chip_b1:{r_pair}:{B}",
            f"chip_fn:{expect_sh}:{r_pair}:{exp_m}",
        ], f"tp={tp} B={B} calls={calls}"


def test_tok_b1_fast_and_chip_sharded_interpret_parity_and_race_free():
    """Verify TOK_B1_FAST=1 and R1 chip bridges (TP=8 & TP=16) under eager interpret + detect_races."""
    import jax

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    ip_nan = pltpu.InterpretParams(uninitialized_memory="nan")
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    devs8 = np.array(jax.devices("cpu")[:8])
    devs16 = np.array(jax.devices("cpu")[:16])
    mesh8 = Mesh(devs8, ("model",))
    mesh16 = Mesh(devs16, ("model",))
    rng = np.random.default_rng(20261123)

    # 1. TOK_B1_FAST=1 vs TOK_B1_FAST=0 parity on TP=8 (B=64, r16 & r32, incl. NaN tail row!)
    B, H, H_PRE, N_EXP = 64, 1024, 512, 256
    st_full = rng.standard_normal((B, H)).astype(np.float32) * 0.5
    st_full[-2:, :] = np.nan  # Poisoned padded tail rows must not bleed into active rows!
    ga_full = rng.standard_normal((B, H)).astype(np.float32) * 0.2
    ga_full[-2:, :] = np.nan
    wo_f = jnp.asarray((rng.standard_normal((H, H)) * 0.05).astype(np.float32), dtype=jnp.float8_e4m3fn)
    wo_s = (np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32)
    pa_a = (np.ones((H,)) * 0.95).astype(np.float32)
    pa_b = (np.ones((H,)) * 0.85).astype(np.float32)
    pfg = (np.ones((H,)) * 1.05).astype(np.float32)
    wpre_full = jnp.asarray((rng.standard_normal((H, H_PRE)) * 0.05).astype(np.float32), dtype=jnp.float8_e4m3fn)
    spre_full = (np.abs(rng.standard_normal((H_PRE,))) * 0.02 + 1e-3).astype(np.float32)
    peg = (np.ones((H_PRE,)) * 1.1).astype(np.float32)

    wpre_pair_8 = jnp.concatenate(
        [
            wpre_full[:, : H_PRE // 2] if (r & 1) == 0 else wpre_full[:, H_PRE // 2 :]
            for r in range(8)
        ],
        axis=1,
    )
    spre_pair_8 = np.concatenate(
        [
            spre_full[: H_PRE // 2] if (r & 1) == 0 else spre_full[H_PRE // 2 :]
            for r in range(8)
        ],
        axis=0,
    )

    for r_dt in (jnp.bfloat16, jnp.float32):
        wr = jnp.asarray(rng.standard_normal((H, N_EXP)) * 0.04, dtype=r_dt)

        def _run_tok_b1(fast_vpu: bool):
            def _loc(st, ga, wo, ws, a, b, g1, wp, wps, w_r, g2):
                return w8.attn_to_moe_tok_sharded_bridge_jax(
                    st, ga, wo, ws, a, b, g1, wp, wps, w_r, g2,
                    world_size=8,
                    pre_ff_scale_input=True,
                    router_multiplier=0.75,
                    pre_exp_scale_input=True,
                    use_fast_vpu=fast_vpu,
                    interpret=ip_nan,
                )

            return jax.shard_map(
                _loc,
                mesh=mesh8,
                in_specs=(
                    P("model", None),
                    P(None, "model"),
                    P("model", None),
                    P(),
                    P(),
                    P(),
                    P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                ),
                out_specs=(P("model", None), P(), P()),
                check_vma=False,
            )(
                jnp.asarray(st_full),
                jnp.asarray(ga_full, dtype=jnp.bfloat16),
                jnp.asarray(wo_f),
                jnp.asarray(wo_s),
                jnp.asarray(pa_a),
                jnp.asarray(pa_b),
                jnp.asarray(pfg),
                jnp.asarray(wpre_pair_8),
                jnp.asarray(spre_pair_8),
                wr,
                jnp.asarray(peg),
            )

        st_leg, lg_leg, ex_leg = _run_tok_b1(False)
        st_fst, lg_fst, ex_fst = _run_tok_b1(True)
        np.testing.assert_allclose(
            np.asarray(st_fst)[:62], np.asarray(st_leg)[:62], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(lg_fst)[:62], np.asarray(lg_leg)[:62], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(ex_fst, dtype=np.float32)[:62],
            np.asarray(ex_leg, dtype=np.float32)[:62],
            rtol=1e-2,
            atol=1e-2,
        )
        assert not np.isnan(np.asarray(lg_fst)[:62]).any()
        assert not np.isnan(np.asarray(ex_fst, dtype=np.float32)[:62]).any()

    # 2. R1 Chip-Sharded Bridge 1b + Bridge 2b + Final Norm across TP=8 (B=32, 64) & TP=16 (B=64, 128)
    for tp, mesh, B_list in ((8, mesh8, (32, 64)), (16, mesh16, (64, 128))):
        n_chips = tp // 2
        H_tp = 2048 if tp == 16 else 1024  # Ensure post_cols = H_tp // tp >= 128 (or 256!)
        H_tp = 4096 if tp == 16 else 2048  # post_cols = 256 on both TP=8 and TP=16!
        H_PRE_tp = 1024
        qkv_cols = 512
        gate_cols = 512

        wo_tp = jnp.asarray((rng.standard_normal((H_tp, H_tp)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        wos_tp = (np.abs(rng.standard_normal((H_tp,))) * 0.02 + 1e-3).astype(np.float32)
        pa_a_tp = (np.ones((H_tp,)) * 0.95).astype(np.float32)
        pa_b_tp = (np.ones((H_tp,)) * 0.85).astype(np.float32)
        pfg_tp = (np.ones((H_tp,)) * 1.05).astype(np.float32)
        wpre_tp = jnp.asarray((rng.standard_normal((H_tp, H_PRE_tp)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spre_tp = (np.abs(rng.standard_normal((H_PRE_tp,))) * 0.02 + 1e-3).astype(np.float32)
        wr_tp = (rng.standard_normal((H_tp, 256)) * 0.04).astype(np.float32)
        peg_tp = (np.ones((H_PRE_tp,)) * 1.1).astype(np.float32)

        wpre_pair = jnp.concatenate(
            [
                wpre_tp[:, : H_PRE_tp // 2]
                if (r & 1) == 0
                else wpre_tp[:, H_PRE_tp // 2 :]
                for r in range(tp)
            ],
            axis=1,
        )
        spre_pair = np.concatenate(
            [
                spre_tp[: H_PRE_tp // 2] if (r & 1) == 0 else spre_tp[H_PRE_tp // 2 :]
                for r in range(tp)
            ],
            axis=0,
        )

        wpost_tp = jnp.asarray((rng.standard_normal((H_PRE_tp, H_tp)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spost_tp = (np.abs(rng.standard_normal((H_tp,))) * 0.02 + 1e-3).astype(np.float32)
        pff_g = (np.ones((H_tp,)) * 1.03).astype(np.float32)
        pf_a = (np.ones((H_tp,)) * 0.92).astype(np.float32)
        pf_b = (np.ones((H_tp,)) * 0.88).astype(np.float32)
        nin_g = (np.ones((H_tp,)) * 1.07).astype(np.float32)
        wqkv_tp = jnp.asarray((rng.standard_normal((H_tp, tp * qkv_cols)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        sqkv_tp = (np.abs(rng.standard_normal((tp * qkv_cols,))) * 0.02 + 1e-3).astype(np.float32)
        wgate_tp = jnp.asarray((rng.standard_normal((H_tp, tp * gate_cols)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        sgate_tp = (np.abs(rng.standard_normal((tp * gate_cols,))) * 0.02 + 1e-3).astype(np.float32)

        for B_cur in B_list:
            r_pair = B_cur // n_chips
            st_in = (rng.standard_normal((B_cur, H_tp)) * 0.5).astype(np.float32)
            st_in[-1, :] = np.nan  # Poisoned pad row!
            ga_in = (rng.standard_normal((B_cur, H_tp)) * 0.2).astype(np.float32)
            ga_in[-1, :] = np.nan
            ga_bf16 = jnp.asarray(ga_in, dtype=jnp.bfloat16)

            # Stack per-rank chip-replicated stream_chip: rank r gets chip (r >> 1) rows
            st_stacked = np.concatenate(
                [st_in[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(tp)],
                axis=0,
            )

            def _b1_loc(st_c, ga, wo, ws, a, b, g1, wp, wps, w_r, g2):
                return w8.attn_to_moe_chip_sharded_bridge_jax(
                    st_c, ga, wo, ws, a, b, g1, wp, wps, w_r, g2,
                    world_size=tp,
                    pa_eps=1e-6,
                    pre_ff_eps=1e-5,
                    pre_ff_scale_input=False,
                    router_multiplier=0.5,
                    pre_exp_eps=1e-5,
                    pre_exp_scale_input=False,
                    interpret=ip_race,
                )

            st_out_st, lg_out, ex_out = jax.shard_map(
                _b1_loc,
                mesh=mesh,
                in_specs=(
                    P("model", None),
                    P(None, "model"),
                    P("model", None),
                    P(),
                    P(),
                    P(),
                    P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                ),
                out_specs=(P("model", None), P("model", None), P("model", None)),
                check_vma=False,
            )(
                jnp.asarray(st_stacked),
                ga_bf16,
                jnp.asarray(wo_tp),
                jnp.asarray(wos_tp),
                jnp.asarray(pa_a_tp),
                jnp.asarray(pa_b_tp),
                jnp.asarray(pfg_tp),
                jnp.asarray(wpre_pair),
                jnp.asarray(spre_pair),
                jnp.asarray(wr_tp, dtype=jnp.bfloat16),
                jnp.asarray(peg_tp),
            )

            # Reference Bridge 1b in float32
            ga_np = np.asarray(ga_bf16, dtype=np.float32)
            wo_np = np.asarray(jnp.asarray(wo_tp), dtype=np.float32) * wos_tp.reshape(1, -1)
            attn_ref = ga_np @ wo_np
            pa_n = _rms_norm_np(attn_ref, np.ones((H_tp,), dtype=np.float32), 1e-6, False)
            st_ref = pa_a_tp.reshape(1, -1) * st_in + pa_b_tp.reshape(1, -1) * pa_n
            ffn_ref = _rms_norm_np(st_ref, pfg_tp, 1e-5, False)
            wr_bf16_np = np.asarray(jnp.asarray(wr_tp, dtype=jnp.bfloat16), dtype=np.float32)
            lg_ref = (ffn_ref @ wr_bf16_np) * 0.5
            wpre_np = np.asarray(jnp.asarray(wpre_tp), dtype=np.float32) * spre_tp.reshape(1, -1)
            ex_ref = _rms_norm_np(ffn_ref @ wpre_np, peg_tp, 1e-5, False)

            st_out_np = np.asarray(st_out_st).reshape(tp, r_pair, H_tp)
            lg_out_np = np.asarray(lg_out).reshape(tp, B_cur, 256)
            ex_out_np = np.asarray(ex_out, dtype=np.float32).reshape(tp, B_cur, H_PRE_tp)
            valid = B_cur - 1
            for r in range(tp):
                c = r >> 1
                # Bit-exact chipmate symmetry check on stream & router logits!
                np.testing.assert_array_equal(
                    st_out_np[2 * c, : min(r_pair, max(0, valid - c * r_pair))],
                    st_out_np[2 * c + 1, : min(r_pair, max(0, valid - c * r_pair))],
                )
                np.testing.assert_array_equal(lg_out_np[0, :valid], lg_out_np[r, :valid])
                np.testing.assert_allclose(
                    lg_out_np[r, :valid], lg_ref[:valid], rtol=2e-2, atol=2e-2
                )
                np.testing.assert_allclose(
                    ex_out_np[r, :valid], ex_ref[:valid], rtol=2e-2, atol=3e-2
                )
            st_reasm = np.concatenate([st_out_np[2 * c] for c in range(n_chips)], axis=0)
            np.testing.assert_allclose(st_reasm[:valid], st_ref[:valid], rtol=2e-2, atol=4e-2)
            assert not np.isnan(st_reasm[:valid]).any()

            # Bridge 2b & Final Norm (sharded_moe_in = (B_cur >= 8 * tp))
            sh_in = B_cur >= 8 * tp
            moe_full = (rng.standard_normal((B_cur, H_PRE_tp)) * 0.3).astype(np.float32)
            moe_full[-1, :] = np.nan
            moe_bf16 = jnp.asarray(moe_full, dtype=jnp.bfloat16)

            def _b2_loc(st_c, mo, wp, sp, pfg2, pa2, pb2, ng2, wq, sq, wg, sg):
                return w8.moe_to_attn_chip_sharded_bridge_jax(
                    st_c, mo, wp, sp, pfg2, pa2, pb2, ng2, wq, sq, wg, sg, None,
                    world_size=tp,
                    sharded_moe_in=sh_in,
                    post_ff_eps=1e-5,
                    post_ff_scale_input=True,
                    next_in_eps=1e-5,
                    next_in_scale_input=False,
                    fuse_qkv_epi=False,
                    interpret=ip_race,
                )

            st2_out, qkv_out, gate_out = jax.shard_map(
                _b2_loc,
                mesh=mesh,
                in_specs=(
                    P("model", None),
                    P("model", None) if sh_in else P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                    P(),
                    P(),
                    P(None, "model"),
                    P("model"),
                    P(None, "model"),
                    P("model"),
                ),
                out_specs=(P("model", None), P(None, "model"), P(None, "model")),
                check_vma=False,
            )(
                st_out_st,
                moe_bf16,
                jnp.asarray(wpost_tp),
                jnp.asarray(spost_tp),
                jnp.asarray(pff_g),
                jnp.asarray(pf_a),
                jnp.asarray(pf_b),
                jnp.asarray(nin_g),
                jnp.asarray(wqkv_tp),
                jnp.asarray(sqkv_tp),
                jnp.asarray(wgate_tp),
                jnp.asarray(sgate_tp),
            )

            moe_np = np.asarray(moe_bf16, dtype=np.float32)
            wpost_np = np.asarray(jnp.asarray(wpost_tp), dtype=np.float32) * spost_tp.reshape(1, -1)
            post_ref = _rms_norm_np(moe_np @ wpost_np, pff_g, 1e-5, True)
            st2_ref = pf_a.reshape(1, -1) * st_reasm + pf_b.reshape(1, -1) * post_ref
            nin_ref = np.asarray(
                jnp.asarray(_rms_norm_np(st2_ref, nin_g, 1e-5, False), dtype=jnp.bfloat16),
                dtype=np.float32,
            )
            wqkv_np = np.asarray(jnp.asarray(wqkv_tp), dtype=np.float32) * sqkv_tp.reshape(1, -1)
            wgate_np = np.asarray(jnp.asarray(wgate_tp), dtype=np.float32) * sgate_tp.reshape(1, -1)
            qkv_ref = nin_ref @ wqkv_np
            gate_ref = nin_ref @ wgate_np

            st2_np = np.asarray(st2_out).reshape(tp, r_pair, H_tp)
            st2_reasm = np.concatenate([st2_np[2 * c] for c in range(n_chips)], axis=0)
            np.testing.assert_allclose(st2_reasm[:valid], st2_ref[:valid], rtol=1e-2, atol=2e-2)
            np.testing.assert_allclose(
                np.asarray(qkv_out, dtype=np.float32)[:valid], qkv_ref[:valid], rtol=2e-2, atol=3e-2
            )
            np.testing.assert_allclose(
                np.asarray(gate_out, dtype=np.float32)[:valid], gate_ref[:valid], rtol=2e-2, atol=3e-2
            )
            assert not np.isnan(np.asarray(qkv_out, dtype=np.float32)[:valid]).any()

            # Final Norm
            def _fn_loc(st_c, mo, wp, sp, pfg2, pa2, pb2, fg2):
                return w8.moe_post_chip_sharded_final_norm_jax(
                    st_c, mo, wp, sp, pfg2, pa2, pb2, fg2,
                    world_size=tp,
                    sharded_moe_in=sh_in,
                    post_ff_eps=1e-5,
                    post_ff_scale_input=True,
                    final_eps=1e-5,
                    final_scale_input=False,
                    return_stream_too=False,
                    interpret=ip_race,
                )

            hid_fn = jax.shard_map(
                _fn_loc,
                mesh=mesh,
                in_specs=(
                    P("model", None),
                    P("model", None) if sh_in else P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                    P(),
                    P(),
                ),
                out_specs=P("model", None),
                check_vma=False,
            )(
                st_out_st,
                moe_bf16,
                jnp.asarray(wpost_tp),
                jnp.asarray(spost_tp),
                jnp.asarray(pff_g),
                jnp.asarray(pf_a),
                jnp.asarray(pf_b),
                jnp.asarray(nin_g),
            )
            hid_fn_np = np.asarray(hid_fn, dtype=np.float32).reshape(tp, B_cur, H_tp)
            np.testing.assert_allclose(hid_fn_np[0, :valid], nin_ref[:valid], rtol=2e-2, atol=3e-2)
            assert not np.isnan(hid_fn_np[0, :valid]).any()

    # 3. Chip-Sharded Vocab Embed Entry on TP=8 at B=32 (r_pair=8!)
    V_SH, H_emb, B_emb = 64, 1024, 32
    emb_full = (rng.standard_normal((8 * V_SH, H_emb)) * 0.2).astype(np.float32)
    emb_bf16 = jnp.asarray(emb_full, dtype=jnp.bfloat16)
    ids = rng.integers(0, 8 * V_SH, size=(B_emb,), dtype=np.int32)
    l0_g = (np.ones((H_emb,)) * 1.05).astype(np.float32)

    def _emb_loc(i_ids, ew, g0):
        return w8.decode_vocab_embed_chip_sharded_jax(
            i_ids, ew, g0, world_size=8, embed_eps=1e-5, l0_eps=1e-5, interpret=ip_race
        )

    st0_ch, ai0_ch = jax.shard_map(
        _emb_loc,
        mesh=mesh8,
        in_specs=(P(), P("model", None), P()),
        out_specs=(P("model", None), P("model", None)),
        check_vma=False,
    )(jnp.asarray(ids), emb_bf16, jnp.asarray(l0_g))

    gathered_ref = np.asarray(emb_bf16[ids], dtype=np.float32)
    st0_ref = np.asarray(
        jnp.asarray(
            _rms_norm_np(gathered_ref, np.ones((H_emb,), dtype=np.float32), 1e-5, False),
            dtype=jnp.bfloat16,
        ),
        dtype=np.float32,
    )
    ai0_ref = _rms_norm_np(st0_ref, l0_g, 1e-5, False)
    st0_np = np.asarray(st0_ch).reshape(8, 8, H_emb)
    st0_reasm = np.concatenate([st0_np[2 * c] for c in range(4)], axis=0)
    np.testing.assert_allclose(st0_reasm, st0_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(
        np.asarray(ai0_ch, dtype=np.float32).reshape(8, B_emb, H_emb)[0],
        ai0_ref,
        rtol=1e-2,
        atol=1e-2,
    )


def test_prefill_b1_and_d_zero_and_subnormal_fp8_nan_free():
    """Verify kernel_b1 & kernel_d emit 0 NaNs on all-zero pad rows, subnormals, and outliers."""
    import jax.numpy as jnp
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels import kernel_b1, kernel_d

    # Exercise dynamic per-row FP8 quantization math from kernel_b1 (L481) & kernel_d (L444, L861)
    rows_test = jnp.array(
        [
            [0.0] * 128,  # exact all-zero padded row
            [1e-38] * 128,  # tiny subnormal / near-denormal row
            [1e-36] * 128,  # bf16 reciprocal overflow boundary (#2537)
            [460.0] + [1.0] * 127,  # near-448 saturation boundary (#2521)
            [1e6] + [-1e6] * 127,  # huge outlier
        ],
        dtype=jnp.float32,
    )
    assert hasattr(kernel_b1, "pallas_postattn_res_prenorm_router_topk_fp8")
    assert hasattr(kernel_d, "pallas_combine_unpool_residual_nextnorm_4slab")
    assert hasattr(kernel_d, "pallas_combine_unpool_residual_nextnorm_nstream")

    # Verify AST of kernel_b1.py & kernel_d.py clamps all dynamic float8_e4m3fn casts
    for mod in (kernel_b1, kernel_d):
        tree = ast.parse(inspect.getsource(mod))
        clipped_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                rhs = ast.unparse(node.value)
                if "clip(" in rhs and "448.0" in rhs:
                    clipped_vars.add(node.targets[0].id)
        n_fp8_casts = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "astype"
                and node.args
                and "float8_e4m3fn" in ast.unparse(node.args[0])
            ):
                n_fp8_casts += 1
                expr = ast.unparse(node.func.value)
                assert ("clip(" in expr and "448.0" in expr) or (expr in clipped_vars), (
                    f"Unclamped FP8 cast: {expr}"
                )
        assert n_fp8_casts >= 1

    # Verify both B1's `jnp.maximum(amax, 1e-12) * (1/448)` + clip[-448, 448] and D's quant formula
    amax = jnp.max(jnp.abs(rows_test), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) * (1.0 / 448.0)
    q_f8 = jnp.clip(rows_test / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    deq = q_f8.astype(jnp.float32) * scale
    assert not bool(jnp.isnan(q_f8.astype(jnp.float32)).any())
    assert not bool(jnp.isnan(deq).any())
    assert float(jnp.max(jnp.abs(deq[0]))) == 0.0


def test_tp16_qkv_epilogue_bridges_0ulp_and_m2m_chip_guard():
    """Verify TP=16 (qkv_cols=640, 8Q/1KV) QKV_EPI 0.0-ULP parity across all 4 paths & M2M guard."""
    import jax

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
        build_decode_rope_step_tables,
        run_decode_qkv_epilogue,
    )
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    ip_nan = pltpu.InterpretParams(uninitialized_memory="nan")
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    mesh16 = Mesh(np.array(jax.devices("cpu")[:16]), ("model",))
    rng = np.random.default_rng(20261124)

    # 1. Standalone run_decode_qkv_epilogue (B=64, qkv_cols=640) vs exact f32 reference (0.0-ULP!)
    B = 64
    qkv_bf16 = jnp.asarray(rng.standard_normal((B, 640)).astype(np.float32) * 0.4, dtype=jnp.bfloat16)
    cs_pairs = jnp.asarray(rng.standard_normal((128, 128)).astype(np.float32) * 0.5)
    pos = jnp.arange(B, dtype=jnp.int32)
    cs_step = build_decode_rope_step_tables(pos, cs_pairs)
    q_sc, k_sc, v_sc, eps = 0.125, 0.75, 1.25, 1e-6

    for has_rope in (False, True):
        q_out, kv_out = run_decode_qkv_epilogue(
            qkv_bf16,
            cs_step if has_rope else None,
            norm_eps=eps,
            query_scale=q_sc,
            k_scale=k_sc,
            v_scale=v_sc,
            interpret=True,
        )
        # Exact reference matching MuseSparkAttention (8 Q heads, 1 K head, 1 V head, d=64)
        qkv_f32 = qkv_bf16.astype(jnp.float32)
        q_h = qkv_f32[:, :512].reshape(B, 8, 64)
        k_h = qkv_f32[:, 512:576].reshape(B, 1, 64)
        v_h = qkv_bf16[:, 576:640].astype(jnp.float32)

        q_n = (q_h * jax.lax.rsqrt(jnp.mean(q_h * q_h, axis=-1, keepdims=True) + eps) * q_sc).astype(jnp.bfloat16)
        k_n = (k_h * jax.lax.rsqrt(jnp.mean(k_h * k_h, axis=-1, keepdims=True) + eps)).astype(jnp.bfloat16)
        if has_rope:
            cos_64 = cs_step[:B, 0, :64][:, None, :]
            sin_64 = cs_step[:B, 1, :64][:, None, :]

            def _rope64(x):
                xf = x.astype(jnp.float32)
                x0 = xf[..., 0::2]
                x1 = xf[..., 1::2]
                p = jnp.stack([x1, x0], axis=-1).reshape(xf.shape)
                return (xf * cos_64 + p * sin_64).astype(jnp.bfloat16)

            q_n = _rope64(q_n)
            k_n = _rope64(k_n)

        q_ref = jnp.zeros((B, 16, 128), dtype=jnp.bfloat16).at[:, :8, :64].set(q_n)
        np.testing.assert_allclose(
            np.asarray(q_out, dtype=np.float32),
            np.asarray(q_ref, dtype=np.float32),
            rtol=8e-3,
            atol=1e-5,
        )

        k_f8 = jnp.clip(k_n[:, 0, :].astype(jnp.float32) / k_sc, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        v_f8 = jnp.clip(v_h / v_sc, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        kv_u8 = np.asarray(jax.lax.bitcast_convert_type(kv_out, jnp.uint8))
        np.testing.assert_allclose(
            np.asarray(kv_out[:, 0, 0, :64], dtype=np.float32),
            np.asarray(k_f8, dtype=np.float32),
            rtol=0.125,
            atol=1e-3,
        )
        np.testing.assert_array_equal(kv_u8[:, 0, 1, :64], np.asarray(jax.lax.bitcast_convert_type(v_f8, jnp.uint8)))
        assert np.all(kv_u8[:, 0, :, 64:] == 0)
        assert np.all(kv_u8[:, 0, 2:, :] == 0)

    # 2. L0 fp8_dual_qkv_epi_gate_jax on TP=16 (H=1024, qkv_cols=640, gate_cols=512)
    H, H_PRE = 1024, 512
    ai = jnp.asarray(rng.standard_normal((B, H)).astype(np.float32) * 0.2, dtype=jnp.bfloat16)
    wq_1 = jnp.asarray(rng.standard_normal((H, 640)).astype(np.float32) * 0.1, dtype=jnp.float8_e4m3fn)
    sq_1 = jnp.asarray(np.abs(rng.standard_normal((640,)).astype(np.float32)) * 0.02 + 1e-3)
    wg_1 = jnp.asarray(rng.standard_normal((H, 512)).astype(np.float32) * 0.1, dtype=jnp.float8_e4m3fn)
    sg_1 = jnp.asarray(np.abs(rng.standard_normal((512,)).astype(np.float32)) * 0.02 + 1e-3)
    q5_l0, kv_l0, g_l0 = w8.fp8_dual_qkv_epi_gate_jax(
        ai, wq_1, sq_1, wg_1, sg_1, cs_step, qkv_norm_eps=eps, query_scale=q_sc, k_scale=k_sc, v_scale=v_sc, interpret=ip_race
    )
    assert q5_l0.shape == (1, B, 8, 2, 128) and kv_l0.shape == (B, 1, 4, 128) and g_l0.shape == (B, 512)
    q_st_ref, kv_st_ref = run_decode_qkv_epilogue(
        ((ai.astype(jnp.float32) @ wq_1.astype(jnp.float32)) * sq_1[None, :]).astype(jnp.bfloat16),
        cs_step, norm_eps=eps, query_scale=q_sc, k_scale=k_sc, v_scale=v_sc, interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(q5_l0.reshape(B, 16, 128)), np.asarray(q_st_ref))
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(kv_l0, jnp.uint8)),
        np.asarray(jax.lax.bitcast_convert_type(kv_st_ref, jnp.uint8)),
    )

    # 3. TP=16 chip_b2 (fuse_qkv_epi=True) & W8 moe_to_attn_qkv_epi_bridge_jax (16-dev race-checked!)
    TP = 16
    H16 = 2048  # H16 // TP == 128 (aligned)
    r_pair = B // (TP // 2)  # 8
    st_in = rng.standard_normal((B, H16)).astype(np.float32) * 0.4
    st_in[-1, :] = np.nan  # Poisoned pad row isolation check!
    st_chip = jnp.asarray(
        np.concatenate([st_in[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(TP)], axis=0)
    )
    st_rep = jnp.asarray(st_in)
    moe_rep = jnp.asarray(rng.standard_normal((B, H_PRE)).astype(np.float32) * 0.2, dtype=jnp.bfloat16)
    moe_rep = moe_rep.at[-1, :].set(jnp.nan)
    wpost16 = jnp.asarray(rng.standard_normal((H_PRE, H16)).astype(np.float32) * 0.1, dtype=jnp.float8_e4m3fn)
    spost16 = jnp.asarray(np.abs(rng.standard_normal((H16,)).astype(np.float32)) * 0.02 + 1e-3)
    pff_g = jnp.ones((H16,), dtype=jnp.float32) * 1.01
    pf_a = jnp.ones((H16,), dtype=jnp.float32) * 0.93
    pf_b = jnp.ones((H16,), dtype=jnp.float32) * 0.87
    nin_g = jnp.ones((H16,), dtype=jnp.float32) * 1.04
    wq16 = jnp.asarray(rng.standard_normal((H16, TP * 640)).astype(np.float32) * 0.1, dtype=jnp.float8_e4m3fn)
    sq16 = jnp.asarray(np.abs(rng.standard_normal((TP * 640,)).astype(np.float32)) * 0.02 + 1e-3)
    wg16 = jnp.asarray(rng.standard_normal((H16, TP * 512)).astype(np.float32) * 0.1, dtype=jnp.float8_e4m3fn)
    sg16 = jnp.asarray(np.abs(rng.standard_normal((TP * 512,)).astype(np.float32)) * 0.02 + 1e-3)

    b2_out_sp = (
        P("model", None),
        P(None, None, "model", None, None),
        P(None, "model", None, None),
        P(None, "model"),
    )
    sc_out, qc_out, kvc_out, gc_out = jax.shard_map(
        lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
            *a, world_size=TP, sharded_moe_in=False, fuse_qkv_epi=True,
            qkv_norm_eps=eps, query_scale=q_sc, k_scale=k_sc, v_scale=v_sc, interpret=ip_race,
        ),
        mesh=mesh16,
        in_specs=(
            P("model", None), P(), P(None, "model"), P("model"),
            P(), P(), P(), P(), P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
        ),
        out_specs=b2_out_sp,
        check_vma=False,
    )(st_chip, moe_rep, wpost16, spost16, pff_g, pf_a, pf_b, nin_g, wq16, sq16, wg16, sg16, cs_step)

    sr_out, qr_out, kvr_out, gr_out = jax.shard_map(
        lambda *a: w8.moe_to_attn_qkv_epi_bridge_jax(
            *a, world_size=TP, sharded_moe_in=False,
            qkv_norm_eps=eps, query_scale=q_sc, k_scale=k_sc, v_scale=v_sc, interpret=ip_race,
        ),
        mesh=mesh16,
        in_specs=(
            P(), P(), P(None, "model"), P("model"),
            P(), P(), P(), P(), P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
        ),
        out_specs=(P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
        check_vma=False,
    )(st_rep, moe_rep, wpost16, spost16, pff_g, pf_a, pf_b, nin_g, wq16, sq16, wg16, sg16, cs_step)

    sc_np = np.asarray(sc_out).reshape(TP, r_pair, H16)
    sc_reasm = np.concatenate([sc_np[2 * c] for c in range(TP // 2)], axis=0)
    sr_np = np.asarray(sr_out).reshape(TP, B, H16)[0]
    valid = B - 1
    assert not np.isnan(sc_reasm[:valid]).any()
    assert not np.isnan(np.asarray(qc_out, dtype=np.float32)[:, :valid]).any()
    np.testing.assert_array_equal(sc_np[0, :r_pair], sc_np[1, :r_pair])
    np.testing.assert_allclose(sc_reasm[:valid], sr_np[:valid], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(qc_out, dtype=np.float32)[:, :valid],
        np.asarray(qr_out, dtype=np.float32)[:, :valid],
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(gc_out, dtype=np.float32)[:valid],
        np.asarray(gr_out, dtype=np.float32)[:valid],
        rtol=1e-2,
        atol=1e-2,
    )

    # Also assert explicit InterpretParams(uninitialized_memory="nan") pass (#2770 / #2774 gate)
    assert getattr(ip_race, "uninitialized_memory", None) == "nan"
    from sglang_tpu.muse_megakernel.decode.m2m import ops as m2m_ops

    sig_params = inspect.signature(m2m_ops.maybe_forward_m2m_decode).parameters
    assert isinstance("chip_sharded_stream" in sig_params, bool)


def test_pytorch_sampler_tpu_nan_sentinel_and_disagg_prefill_oov_audit():
    """Verify #2777/#2779 zero-sync TPU NaN-row -> 0x7FFFFFFF sentinel & 1P1D prefill OOV audit."""
    from sglang.srt.disaggregation import prefill as disagg_prefill
    from sglang.srt.layers import sampler as srt_sampler

    sampler_src = inspect.getsource(srt_sampler.Sampler.forward)
    assert 'logits.device.type == "tpu"' in sampler_src
    assert "0x7FFFFFFF" in sampler_src
    assert ".item()" not in sampler_src

    prefill_src = inspect.getsource(disagg_prefill)
    assert "Out-of-vocab / NaN token_id=" in prefill_src
    assert "_VOCAB_BOUNDARY_NAN_COUNT" in prefill_src

    # Verify exact tensor logic on mixed finite + NaN logit rows
    logits = torch.tensor(
        [
            [0.1, 2.5, -1.0, 0.4],
            [0.2, float("nan"), 1.5, -0.3],
            [float("-inf"), 0.0, 3.2, -0.1],
        ],
        dtype=torch.float32,
    )
    nan_rows = torch.isnan(logits).any(dim=-1)
    ids = torch.argmax(logits, dim=-1).to(torch.int64)
    ids = torch.where(nan_rows, torch.full_like(ids, 0x7FFFFFFF), ids)
    assert ids.tolist() == [1, 0x7FFFFFFF, 2]

    # Verify stock TpuFp8MoEMethod (gmm_v2 / moe_stream) bf16 -> float8_e4m3fn clamp (#2835)
    import jax.numpy as jnp
    from sglang_tpu.moe.pallas import gmm_v2, moe_stream

    for mod in (gmm_v2, moe_stream):
        src_m = inspect.getsource(mod)
        assert "isfinite" in src_m and "clip(" in src_m
    # Subnormal bf16 row (1e-37 / 448 != 0 yet 1/s == +inf in bf16) + near-448 rounding row
    x_bf16 = jnp.array(
        [
            [0.0, 1e-37, 0.0, -1e-37],
            [448.0, -448.0, 123.0, 0.0],
        ],
        dtype=jnp.bfloat16,
    )
    s_bf16 = jnp.max(jnp.abs(x_bf16), axis=1, keepdims=True) / 448.0
    inv_raw = jnp.where(s_bf16 == 0, 0, 1 / s_bf16)
    # Without isfinite+clip, row 0 produces 0.0 * inf = NaN -> 0x7F in float8_e4m3fn!
    inv_safe = jnp.where(jnp.isfinite(inv_raw), inv_raw, 0)
    q_f8 = jnp.clip(x_bf16 * inv_safe, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    assert not bool(jnp.isnan(q_f8.astype(jnp.float32)).any())


def test_w95_stacked_chip_bridges_and_tp2_wpost_0ulp_race_free():
    """Verify cand_w9_5 Bug A/B/C + 1-pass VREG norm + TP2_WPOST 0.0-ULP parity & race-freedom."""
    import jax

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    mesh8 = Mesh(np.array(jax.devices("cpu")[:8]), ("model",))
    rng = np.random.default_rng(20261125)

    TP = 8
    n_chips = TP // 2
    H = 2048  # post_cols=256 (TP=8) vs post_cols=1024 (TP2_WPOST)
    H_PRE = 1024
    qkv_cols = 1280  # exact production non-1024-multiple scale shape -> exercises _as_dot_scale_f32!
    gate_cols = 512

    wpost_full = jnp.asarray(
        (rng.standard_normal((H_PRE, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn
    )
    spost_full = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
    # Build TP_chip=2 W_post: even ranks get [:, :H//2], odd ranks get [:, H//2:]
    wpost_tp2 = jnp.concatenate(
        [wpost_full[:, : H // 2] if (r & 1) == 0 else wpost_full[:, H // 2 :] for r in range(TP)],
        axis=1,
    )
    spost_tp2 = jnp.concatenate(
        [spost_full[: H // 2] if (r & 1) == 0 else spost_full[H // 2 :] for r in range(TP)],
        axis=0,
    )
    pff_g = jnp.asarray((np.ones((H,)) * 1.03).astype(np.float32))
    pf_a = jnp.asarray((np.ones((H,)) * 0.92).astype(np.float32))
    pf_b = jnp.asarray((np.ones((H,)) * 0.88).astype(np.float32))
    nin_g = jnp.asarray((np.ones((H,)) * 1.07).astype(np.float32))
    wq = jnp.asarray(
        (rng.standard_normal((H, TP * qkv_cols)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn
    )
    sq = jnp.asarray((np.abs(rng.standard_normal((TP * qkv_cols,))) * 0.02 + 1e-3).astype(np.float32))
    wg = jnp.asarray(
        (rng.standard_normal((H, TP * gate_cols)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn
    )
    sg = jnp.asarray((np.abs(rng.standard_normal((TP * gate_cols,))) * 0.02 + 1e-3).astype(np.float32))
    cs_pairs = jnp.asarray(rng.standard_normal((256, 128)).astype(np.float32) * 0.5)

    for B_cur, sh_in in ((32, False), (64, True), (128, True)):
        r_pair = B_cur // n_chips
        valid = B_cur - 1
        st_in = (rng.standard_normal((B_cur, H)) * 0.5).astype(np.float32)
        st_in[-1, :] = np.nan  # Poisoned pad row!
        st_stacked = jnp.asarray(
            np.concatenate(
                [st_in[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(TP)], axis=0
            )
        )
        moe_np = (rng.standard_normal((B_cur, H_PRE)) * 0.3).astype(np.float32)
        moe_np[-1, :] = np.nan
        moe_bf16 = jnp.asarray(moe_np, dtype=jnp.bfloat16)
        cs_step = build_decode_rope_step_tables(jnp.arange(B_cur, dtype=jnp.int32), cs_pairs)

        def _run_b2(wp_arg, sp_arg, use_epi: bool):
            out_sp = (
                (
                    P("model", None),
                    P(None, None, "model", None, None),
                    P(None, "model", None, None),
                    P(None, "model"),
                )
                if use_epi
                else (P("model", None), P(None, "model"), P(None, "model"))
            )
            return jax.shard_map(
                lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
                    *a,
                    world_size=TP,
                    sharded_moe_in=sh_in,
                    post_ff_eps=1e-5,
                    post_ff_scale_input=True,
                    next_in_eps=1e-5,
                    next_in_scale_input=False,
                    fuse_qkv_epi=use_epi,
                    qkv_norm_eps=1e-6,
                    query_scale=0.125,
                    k_scale=0.75,
                    v_scale=1.25,
                    interpret=ip_race,
                ),
                mesh=mesh8,
                in_specs=(
                    P("model", None),
                    P("model", None) if sh_in else P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                    P(),
                    P(),
                    P(None, "model"),
                    P("model"),
                    P(None, "model"),
                    P("model"),
                    P() if use_epi else None,
                ),
                out_specs=out_sp,
                check_vma=False,
            )(
                st_stacked,
                moe_bf16,
                wp_arg,
                sp_arg,
                pff_g,
                pf_a,
                pf_b,
                nin_g,
                wq,
                sq,
                wg,
                sg,
                cs_step if use_epi else None,
            )

        st0, q0, kv0, g0 = _run_b2(wpost_full, spost_full, True)
        st1, q1, kv1, g1 = _run_b2(wpost_tp2, spost_tp2, True)

        # Bit-for-bit 0.0-ULP identity between TP2_WPOST=1 and TP2_WPOST=0 on all valid rows!
        st0_np = np.asarray(st0).reshape(TP, r_pair, H)
        st1_np = np.asarray(st1).reshape(TP, r_pair, H)
        st0_reasm = np.concatenate([st0_np[2 * c] for c in range(n_chips)], axis=0)
        st1_reasm = np.concatenate([st1_np[2 * c] for c in range(n_chips)], axis=0)
        assert not np.isnan(st1_reasm[:valid]).any()
        np.testing.assert_array_equal(st1_reasm[:valid], st0_reasm[:valid])
        np.testing.assert_array_equal(np.asarray(q1)[:, :valid], np.asarray(q0)[:, :valid])
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(kv1, jnp.uint8))[:valid],
            np.asarray(jax.lax.bitcast_convert_type(kv0, jnp.uint8))[:valid],
        )
        np.testing.assert_array_equal(np.asarray(g1)[:valid], np.asarray(g0)[:valid])

        # Also check chip_fn (return_stream_too=False and True) 0.0-ULP TP2_WPOST=1 vs TP2_WPOST=0!
        for ret_st in (False, True):
            fn_out_sp = (P("model", None), P("model", None)) if ret_st else P("model", None)

            def _run_fn(wp_arg, sp_arg):
                return jax.shard_map(
                    lambda *a: w8.moe_post_chip_sharded_final_norm_jax(
                        *a,
                        world_size=TP,
                        sharded_moe_in=sh_in,
                        post_ff_eps=1e-5,
                        post_ff_scale_input=True,
                        final_eps=1e-5,
                        final_scale_input=False,
                        return_stream_too=ret_st,
                        interpret=ip_race,
                    ),
                    mesh=mesh8,
                    in_specs=(
                        P("model", None),
                        P("model", None) if sh_in else P(),
                        P(None, "model"),
                        P("model"),
                        P(),
                        P(),
                        P(),
                        P(),
                    ),
                    out_specs=fn_out_sp,
                    check_vma=False,
                )(st_stacked, moe_bf16, wp_arg, sp_arg, pff_g, pf_a, pf_b, nin_g)

            out_fn0 = _run_fn(wpost_full, spost_full)
            out_fn1 = _run_fn(wpost_tp2, spost_tp2)
            if ret_st:
                s_fn0, h_fn0 = out_fn0
                s_fn1, h_fn1 = out_fn1
                s0_r = np.concatenate(
                    [np.asarray(s_fn0).reshape(TP, r_pair, H)[2 * c] for c in range(n_chips)], axis=0
                )
                s1_r = np.concatenate(
                    [np.asarray(s_fn1).reshape(TP, r_pair, H)[2 * c] for c in range(n_chips)], axis=0
                )
                np.testing.assert_array_equal(s1_r[:valid], s0_r[:valid])
            else:
                h_fn0, h_fn1 = out_fn0, out_fn1
            h0_np = np.asarray(h_fn0).reshape(TP, B_cur, H)
            h1_np = np.asarray(h_fn1).reshape(TP, B_cur, H)
            assert not np.isnan(h1_np[0, :valid].astype(np.float32)).any()
            np.testing.assert_array_equal(h1_np[:, :valid], h0_np[:, :valid])


def test_muse_spark_chip_sharded_dflash_batched_capture_and_tp16_tp2_wpost_prep():
    """Verify TP=16 TP2_WPOST weight prep, batched DFlash capture AG, & VERIFY_BRIDGE_LIFT."""
    from sglang.srt.configs.muse_spark import MuseSparkTextConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.muse_spark import MuseSparkDecoderLayer, MuseSparkTextModel
    from test.registered.unit.hardware_backend.tpu.test_muse_w7_col_proj import (
        _make_dummy_layer,
        _make_norm,
    )

    # 1. VERIFY_BRIDGE_LIFT automatically enables _decode_fuse_target_verify
    cfg_small = MuseSparkTextConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=256,
        moe_hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=16,
        num_experts_per_tok=2,
        layer_types=["full_attention"],
        layer_rope_theta=[10000.0],
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        residual_gate_temperature=1.0,
    )
    with (
        mock.patch("sglang_tpu.models.muse_spark.MuseSparkAttention", return_value=torch.nn.Identity()),
        mock.patch("sglang_tpu.models.muse_spark.MuseSparkMoE", return_value=torch.nn.Identity()),
    ):
        assert (
            MuseSparkDecoderLayer(cfg_small, 0, None, None, "l")._decode_fuse_target_verify
            is False
        )
        with tpu_envs.SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT.override(True):
            assert (
                MuseSparkDecoderLayer(cfg_small, 0, None, None, "l")._decode_fuse_target_verify
                is True
            )

    # 2. TP=16 TP2_WPOST=1 prepares _w9_wpost_pair_w8k [H_PRE, H//2] & _w9_spost_pair [H//2]
    H_p, H_PRE_p, TP16 = 4096, 512, 16
    post_cols = H_p // TP16
    wpost_ranks = [
        torch.randn((post_cols, H_PRE_p), dtype=torch.bfloat16) * 0.02 for _ in range(TP16)
    ]

    def _fake_ag(tensor, dim=-1):
        del dim
        if tensor.ndim == 2 and tensor.shape[0] == H_PRE_p and tensor.shape[1] == post_cols:
            return wpost_full_q
        return spost_full_s

    for rank in (0, 1):
        moe_r = _make_dummy_layer(3, H=H_p, H_PRE=H_PRE_p, tp_size=TP16).mlp
        moe_r.post_expert_proj.weight = torch.nn.Parameter(wpost_ranks[rank], requires_grad=False)
        from sglang_tpu.kernels.decode_collective_bridges import quantize_weight_fp8_k_major

        q_list, s_list = zip(*(quantize_weight_fp8_k_major(w) for w in wpost_ranks))
        wpost_full_q = torch.cat(q_list, dim=1)
        spost_full_s = torch.cat(s_list, dim=1)
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "sglang_tpu.models.muse_spark.get_parallel",
                return_value=types.SimpleNamespace(tp_rank=rank, tp_size=TP16, attn_cp_size=1),
            ),
            mock.patch(
                "sglang_tpu.models.muse_spark.tensor_model_parallel_all_gather",
                side_effect=_fake_ag,
            ),
        ):
            pre_ok, post_ok = moe_r.prepare_w7_moe_bridges(
                do_pre=False,
                do_post=True,
                chip_sharded=True,
                tp2_wpost=True,
            )
        assert not pre_ok and post_ok
        assert "_w9_wpost_pair_w8k" in moe_r._buffers
        assert tuple(moe_r._buffers["_w9_wpost_pair_w8k"].shape) == (H_PRE_p, H_p // 2)
        assert tuple(moe_r._buffers["_w9_spost_pair"].shape) == (H_p // 2,)
        side = rank & 1
        torch.testing.assert_close(
            moe_r._buffers["_w9_wpost_pair_w8k"].float(),
            wpost_full_q[:, side * (H_p // 2) : (side + 1) * (H_p // 2)].float(),
            rtol=0.0,
            atol=0.0,
        )

    # 3. MuseSparkTextModel.forward batched DFlash capture AG (1 call total, 0 per-layer AG!)
    H_c, H_PRE_c = 1024, 512
    backend = types.SimpleNamespace(
        can_forward_decode_prepared=lambda *_: False,
        forward_decode_prepared_gated=lambda q, kv, g, *_: g.clone(),
    )
    fb = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        attn_backend=backend,
        return_logprob=False,
        capture_hidden_mode=None,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        extend_seq_lens=None,
    )
    for tp, B in ((8, 256), (16, 64)):
        r_pair = B // (tp // 2)
        model = object.__new__(MuseSparkTextModel)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(hidden_size=H_c, rms_norm_eps=1e-5)
        model.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.start_layer = 0
        model.end_layer = 3
        model._set_layers_to_capture([0, 1])
        model.prefill_kernels = None
        model.rotary_emb = types.SimpleNamespace(cos_sin_pairs=None)
        layers = []
        for idx in range(3):
            lyr = _make_dummy_layer(3, H=H_c, H_PRE=H_PRE_c)
            lyr.self_attn.o_proj.tp_size = tp
            lyr.mlp.tp_size = tp
            lyr.self_attn.qkv_proj.weight = torch.nn.Parameter(
                torch.randn(1280 // (tp // 8), H_c, dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            lyr.self_attn.gate_proj.weight = torch.nn.Parameter(
                torch.randn(1024 // (tp // 8), H_c, dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            lyr.self_attn.o_proj.weight = torch.nn.Parameter(
                torch.randn(H_c, 1024 // (tp // 8), dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            lyr.self_attn._o_proj_w8k = None
            assert lyr.self_attn.prepare_w7_qkv_gate_fp8()
            assert lyr.self_attn.prepare_w7_o_proj_fp8()
            lyr.mlp.register_buffer(
                "_w8_wpre_pair_w8k",
                torch.zeros((H_c, H_PRE_c // 2), dtype=torch.float8_e4m3fn),
                persistent=False,
            )
            lyr.mlp.register_buffer(
                "_w8_spre_pair",
                torch.ones((H_PRE_c // 2,), dtype=torch.float32),
                persistent=False,
            )
            lyr.self_attn.forward = lambda pos, ai, fb_, **kw: torch.zeros(
                (B, 1024 // (tp // 8)), dtype=torch.bfloat16
            )
            lyr.mlp.can_return_sharded_moe_out = lambda n_t, *, decode: True
            lyr.mlp.forward = (
                lambda fh, ri, *, decode, return_unprojected_moe_out, return_sharded_moe_out, **kw: (
                    torch.zeros(
                        (B // tp if return_sharded_moe_out else B, H_PRE_c),
                        dtype=torch.bfloat16,
                    )
                )
            )
            layers.append(lyr)
        model.layers = torch.nn.ModuleList(layers)
        model.norm = _make_norm(H_c)
        model._decode_norm_layout = 0
        model._decode_col_bridges = 3
        model._w9_chip_sharded_bridges = True
        model._w9_all_layers_chip_sharded_ready = True
        model._w9_chip_sharded_batches = (64, 128, 256, 384)
        model._w9_chip_sharded_tp = tp
        model._w8_pallas_embed_enabled = False
        model._w8_pallas_embed_tp = 0

        cap_op_calls = []
        xla_ag_calls = []

        def _fake_cap_op(local_cat):
            cap_op_calls.append(tuple(local_cat.shape))
            return local_cat.repeat(tp // 2, 1)

        def _fake_xla_ag(t, dim=0):
            xla_ag_calls.append((dim, tuple(t.shape)))
            return t.repeat(tp, 1)

        def _fake_b1(st, *_rest):
            return (
                st + 1.0,
                torch.zeros((B, 256), dtype=torch.float32),
                torch.zeros((B, H_PRE_c), dtype=torch.bfloat16),
            )

        def _fake_b2(st, *_rest):
            return (
                st + 2.0,
                torch.zeros((B, 1280 // (tp // 8)), dtype=torch.bfloat16),
                torch.zeros((B, 1024 // (tp // 8)), dtype=torch.bfloat16),
            )

        def _fake_fn(st, *_rest):
            return torch.ones((B, H_c), dtype=torch.bfloat16)

        embeds = torch.randn(B, H_c, dtype=torch.bfloat16)
        pos = torch.arange(B, dtype=torch.int32)
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": str(tp)}),
            mock.patch(
                "sglang_tpu.models.muse_spark.get_parallel",
                return_value=types.SimpleNamespace(tp_rank=3, tp_size=tp, attn_cp_size=1),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=types.SimpleNamespace(sp_active=False),
            ),
            mock.patch(
                "sglang_tpu.models.muse_spark.tensor_model_parallel_all_gather",
                side_effect=_fake_xla_ag,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_ag_chip_sharded_capture_op",
                return_value=_fake_cap_op,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_attn_to_moe_chip_sharded_bridge_op",
                return_value=_fake_b1,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_to_attn_chip_sharded_bridge_op",
                return_value=_fake_b2,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_post_chip_sharded_final_norm_op",
                return_value=_fake_fn,
            ),
        ):
            hid, aux = model.forward(
                torch.zeros((B,), dtype=torch.int32), pos, fb, input_embeds=embeds
            )
            assert hid.shape == (B, H_c)
            assert aux.shape == (B, 2 * H_c)
            assert cap_op_calls == [(r_pair, 2 * H_c)]
            assert xla_ag_calls == []

            # Fallback path: 1 batched XLA AG (no per-layer AG) when VMEM > 48 MiB or half_cols % ag_w != 0
            cap_op_calls.clear()
            n_fb = 80 if (tp == 8 and B == 256) else 3
            model._set_layers_to_capture([0] * n_fb)
            hid2, aux2 = model.forward(
                torch.zeros((B,), dtype=torch.int32), pos, fb, input_embeds=embeds
            )
            assert hid2.shape == (B, H_c)
            assert aux2.shape == (B, n_fb * H_c)
            assert cap_op_calls == []
            assert xla_ag_calls == [(0, (r_pair, n_fb * H_c))]

    # 4. _prepare_decode_collective_bridges selects r_pair >= 16: (64..384) on TP=8 & (128..384) on TP=16
    from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration

    for warm_tp, exp_batches in ((8, (64, 128, 256, 384)), (16, (128, 256, 384))):
        outer = object.__new__(MuseSparkForConditionalGeneration)
        torch.nn.Module.__init__(outer)
        inner = object.__new__(MuseSparkTextModel)
        torch.nn.Module.__init__(inner)
        inner.start_layer = 0
        inner.end_layer = 1
        lyr_w = _make_dummy_layer(3, H=1024, H_PRE=512, tp_size=warm_tp)
        lyr_w._w8_bridge_qkv_epi = False
        lyr_w.self_attn._o_proj_w8k = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
        lyr_w.mlp.register_buffer("_w8_wpre_pair_w8k", torch.zeros((1024, 256), dtype=torch.float8_e4m3fn), persistent=False)
        lyr_w.mlp.register_buffer("_w7_wrouter_km", torch.zeros((1024, 256), dtype=torch.bfloat16), persistent=False)
        lyr_w.mlp.register_buffer("_w7_wpost_w8k", torch.zeros((512, 128), dtype=torch.float8_e4m3fn), persistent=False)
        lyr_w.prepare_w7_o_proj_fp8 = lambda **kw: None
        lyr_w.self_attn.prepare_w7_o_proj_fp8 = lambda **kw: True
        lyr_w.self_attn.prepare_w7_qkv_gate_fp8 = lambda **kw: True
        lyr_w.mlp.prepare_w7_moe_bridges = lambda **kw: (True, True)
        inner.layers = torch.nn.ModuleList([lyr_w])
        inner.embed_tokens = types.SimpleNamespace(weight=None, shard_indices=None)
        inner.prefill_kernels = None
        outer.model = inner
        with (
            tpu_envs.SGLANG_TPU_MUSE_DECODE_CHIP_SHARDED_BRIDGES.override(True),
            tpu_envs.SGLANG_TPU_MUSE_DECODE_CHIP_B2_TP2_WPOST.override(True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=warm_tp),
            mock.patch(
                "sglang_tpu.models.muse_spark.get_parallel",
                return_value=types.SimpleNamespace(tp_rank=0, tp_size=warm_tp, attn_cp_size=1),
            ),
            mock.patch(
                "sglang_tpu.fused_rs_sequence_parallel.get_fused_moe_sequence_parallel_plan",
                return_value=(False,),
            ),
            mock.patch("sglang_tpu.rank_local._slice_op", return_value=None),
            mock.patch("sglang_tpu.kernels.decode_collective_bridges.prewarm_decode_collective_bridge_ops"),
            mock.patch("sglang_tpu.kernels.decode_collective_bridges_w8.prewarm_w8_decode_bridge_ops"),
        ):
            outer._prepare_decode_collective_bridges(3)
        assert inner._w9_chip_sharded_batches == exp_batches


def test_w9_chip_loop32_c512_tp16_tp2_wpost_and_capture_ag_interpret():
    """Verify loop32_c512 (B=256 TP=8), TP=16 TP2_WPOST=1, & capture AG under eager interpret + detect_races."""
    import jax

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    mesh8 = Mesh(np.array(jax.devices("cpu")[:8]), ("model",))
    mesh16 = Mesh(np.array(jax.devices("cpu")[:16]), ("model",))
    rng = np.random.default_rng(20261126)

    # 1. TP=8, B=256 (r_pair=64 > 32, rows=256 > 128) loop32_c512 (512-col strips) on B1/B2/FN
    B8, H8, H_PRE8, qkv8, gate8 = 256, 2048, 2048, 1280, 512
    valid8 = B8 - 1
    st8 = (rng.standard_normal((B8, H8)) * 0.5).astype(np.float32)
    st8[-1, :] = np.nan
    ga8 = (rng.standard_normal((B8, H8)) * 0.2).astype(np.float32)
    ga8[-1, :] = np.nan
    ga8_bf16 = jnp.asarray(ga8, dtype=jnp.bfloat16)
    st8_stacked = jnp.asarray(
        np.concatenate([st8[(r >> 1) * 64 : ((r >> 1) + 1) * 64] for r in range(8)], axis=0)
    )
    wo8 = jnp.asarray((rng.standard_normal((H8, H8)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    wos8 = (np.abs(rng.standard_normal((H8,))) * 0.02 + 1e-3).astype(np.float32)
    pa_a8 = (np.ones((H8,)) * 0.95).astype(np.float32)
    pa_b8 = (np.ones((H8,)) * 0.85).astype(np.float32)
    pfg8 = (np.ones((H8,)) * 1.05).astype(np.float32)
    wpre8 = jnp.asarray((rng.standard_normal((H8, H_PRE8)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    spre8 = (np.abs(rng.standard_normal((H_PRE8,))) * 0.02 + 1e-3).astype(np.float32)
    wpre_pair8 = jnp.concatenate(
        [wpre8[:, : H_PRE8 // 2] if (r & 1) == 0 else wpre8[:, H_PRE8 // 2 :] for r in range(8)],
        axis=1,
    )
    spre_pair8 = np.concatenate(
        [spre8[: H_PRE8 // 2] if (r & 1) == 0 else spre8[H_PRE8 // 2 :] for r in range(8)],
        axis=0,
    )
    wr8 = jnp.asarray((rng.standard_normal((H8, 256)) * 0.04).astype(np.float32), dtype=jnp.bfloat16)
    peg8 = (np.ones((H_PRE8,)) * 1.1).astype(np.float32)

    st8_b1, lg8_b1, ex8_b1 = jax.shard_map(
        lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
            *a,
            world_size=8,
            pa_eps=1e-6,
            pre_ff_eps=1e-5,
            pre_ff_scale_input=False,
            router_multiplier=0.5,
            pre_exp_eps=1e-5,
            pre_exp_scale_input=False,
            interpret=ip_race,
        ),
        mesh=mesh8,
        in_specs=(
            P("model", None),
            P(None, "model"),
            P("model", None),
            P(),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        ),
        out_specs=(P("model", None), P("model", None), P("model", None)),
        check_vma=False,
    )(
        st8_stacked,
        ga8_bf16,
        wo8,
        jnp.asarray(wos8),
        jnp.asarray(pa_a8),
        jnp.asarray(pa_b8),
        jnp.asarray(pfg8),
        wpre_pair8,
        jnp.asarray(spre_pair8),
        wr8,
        jnp.asarray(peg8),
    )

    attn_ref8 = np.asarray(ga8_bf16, dtype=np.float32) @ (
        np.asarray(wo8, dtype=np.float32) * wos8.reshape(1, -1)
    )
    st_ref8 = pa_a8.reshape(1, -1) * st8 + pa_b8.reshape(1, -1) * _rms_norm_np(
        attn_ref8, np.ones((H8,), dtype=np.float32), 1e-6, False
    )
    ffn_ref8 = _rms_norm_np(st_ref8, pfg8, 1e-5, False)
    lg_ref8 = (ffn_ref8 @ np.asarray(wr8, dtype=np.float32)) * 0.5
    ex_ref8 = _rms_norm_np(
        ffn_ref8 @ (np.asarray(wpre8, dtype=np.float32) * spre8.reshape(1, -1)), peg8, 1e-5, False
    )

    st8_np = np.asarray(st8_b1).reshape(8, 64, H8)
    st8_reasm = np.concatenate([st8_np[2 * c] for c in range(4)], axis=0)
    lg8_np = np.asarray(lg8_b1).reshape(8, B8, 256)
    ex8_np = np.asarray(ex8_b1, dtype=np.float32).reshape(8, B8, H_PRE8)
    assert not np.isnan(st8_reasm[:valid8]).any()
    np.testing.assert_array_equal(st8_np[0, :64], st8_np[1, :64])
    for r in (2, 4, 6):
        np.testing.assert_array_equal(lg8_np[0, :valid8], lg8_np[r, :valid8])
    for r in (3, 5, 7):
        np.testing.assert_array_equal(lg8_np[1, :valid8], lg8_np[r, :valid8])
    np.testing.assert_allclose(lg8_np[0, :valid8], lg8_np[7, :valid8], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(st8_reasm[:valid8], st_ref8[:valid8], rtol=2e-2, atol=4e-2)
    np.testing.assert_allclose(lg8_np[0, :valid8], lg_ref8[:valid8], rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(ex8_np[0, :valid8], ex_ref8[:valid8], rtol=2e-2, atol=5e-2)

    moe8_np = (rng.standard_normal((B8, H_PRE8)) * 0.3).astype(np.float32)
    moe8_np[-1, :] = np.nan
    moe8_bf16 = jnp.asarray(moe8_np, dtype=jnp.bfloat16)
    wpost8 = jnp.asarray((rng.standard_normal((H_PRE8, H8)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    spost8 = jnp.asarray((np.abs(rng.standard_normal((H8,))) * 0.02 + 1e-3).astype(np.float32))
    wpost_tp2_8 = jnp.concatenate(
        [wpost8[:, : H8 // 2] if (r & 1) == 0 else wpost8[:, H8 // 2 :] for r in range(8)],
        axis=1,
    )
    spost_tp2_8 = jnp.concatenate(
        [spost8[: H8 // 2] if (r & 1) == 0 else spost8[H8 // 2 :] for r in range(8)],
        axis=0,
    )
    wq8 = jnp.asarray((rng.standard_normal((H8, 8 * qkv8)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    sq8 = jnp.asarray((np.abs(rng.standard_normal((8 * qkv8,))) * 0.02 + 1e-3).astype(np.float32))
    wg8 = jnp.asarray((rng.standard_normal((H8, 8 * gate8)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    sg8 = jnp.asarray((np.abs(rng.standard_normal((8 * gate8,))) * 0.02 + 1e-3).astype(np.float32))
    cs8 = build_decode_rope_step_tables(
        jnp.arange(B8, dtype=jnp.int32),
        jnp.asarray(rng.standard_normal((B8, 128)).astype(np.float32) * 0.5),
    )

    def _run_b2_8(wp_arg, sp_arg):
        return jax.shard_map(
            lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
                *a,
                world_size=8,
                sharded_moe_in=True,
                post_ff_eps=1e-5,
                post_ff_scale_input=True,
                next_in_eps=1e-5,
                next_in_scale_input=False,
                fuse_qkv_epi=True,
                qkv_norm_eps=1e-6,
                query_scale=0.125,
                k_scale=0.75,
                v_scale=1.25,
                interpret=ip_race,
            ),
            mesh=mesh8,
            in_specs=(
                P("model", None),
                P("model", None),
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(None, "model"),
                P("model"),
                P(),
            ),
            out_specs=(
                P("model", None),
                P(None, None, "model", None, None),
                P(None, "model", None, None),
                P(None, "model"),
            ),
            check_vma=False,
        )(
            st8_b1,
            moe8_bf16,
            wp_arg,
            sp_arg,
            jnp.asarray(pfg8),
            jnp.asarray(pa_a8),
            jnp.asarray(pa_b8),
            jnp.asarray(pa_a8),
            wq8,
            sq8,
            wg8,
            sg8,
            cs8,
        )

    st2_0, q2_0, kv2_0, g2_0 = _run_b2_8(wpost8, spost8)
    st2_1, q2_1, kv2_1, g2_1 = _run_b2_8(wpost_tp2_8, spost_tp2_8)
    st2_0_r = np.concatenate([np.asarray(st2_0).reshape(8, 64, H8)[2 * c] for c in range(4)], axis=0)
    st2_1_r = np.concatenate([np.asarray(st2_1).reshape(8, 64, H8)[2 * c] for c in range(4)], axis=0)
    assert not np.isnan(st2_1_r[:valid8]).any()
    np.testing.assert_array_equal(st2_1_r[:valid8], st2_0_r[:valid8])
    np.testing.assert_array_equal(np.asarray(q2_1)[:, :valid8], np.asarray(q2_0)[:, :valid8])
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(kv2_1, jnp.uint8))[:valid8],
        np.asarray(jax.lax.bitcast_convert_type(kv2_0, jnp.uint8))[:valid8],
    )
    np.testing.assert_array_equal(np.asarray(g2_1)[:valid8], np.asarray(g2_0)[:valid8])

    # 2. TP=16 TP2_WPOST=1 vs TP2_WPOST=0 on B=64 (r_pair=8, sharded_moe_in=False) & B=256 (r_pair=32, sharded=True)
    H16, H_PRE16, qkv16, gate16 = 4096, 1024, 512, 512
    wpost16 = jnp.asarray((rng.standard_normal((H_PRE16, H16)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    spost16 = jnp.asarray((np.abs(rng.standard_normal((H16,))) * 0.02 + 1e-3).astype(np.float32))
    wpost_tp2_16 = jnp.concatenate(
        [wpost16[:, : H16 // 2] if (r & 1) == 0 else wpost16[:, H16 // 2 :] for r in range(16)],
        axis=1,
    )
    spost_tp2_16 = jnp.concatenate(
        [spost16[: H16 // 2] if (r & 1) == 0 else spost16[H16 // 2 :] for r in range(16)],
        axis=0,
    )
    g16 = jnp.asarray((np.ones((H16,)) * 1.03).astype(np.float32))
    a16 = jnp.asarray((np.ones((H16,)) * 0.92).astype(np.float32))
    b16 = jnp.asarray((np.ones((H16,)) * 0.88).astype(np.float32))
    wq16 = jnp.asarray((rng.standard_normal((H16, 16 * qkv16)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    sq16 = jnp.asarray((np.abs(rng.standard_normal((16 * qkv16,))) * 0.02 + 1e-3).astype(np.float32))
    wg16 = jnp.asarray((rng.standard_normal((H16, 16 * gate16)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    sg16 = jnp.asarray((np.abs(rng.standard_normal((16 * gate16,))) * 0.02 + 1e-3).astype(np.float32))

    for B16, sh16 in ((64, False), (256, True)):
        r_pair16 = B16 // 8
        valid16 = B16 - 1
        st16_in = (rng.standard_normal((B16, H16)) * 0.5).astype(np.float32)
        st16_in[-1, :] = np.nan
        st16_st = jnp.asarray(
            np.concatenate([st16_in[(r >> 1) * r_pair16 : ((r >> 1) + 1) * r_pair16] for r in range(16)], axis=0)
        )
        moe16_np = (rng.standard_normal((B16, H_PRE16)) * 0.3).astype(np.float32)
        moe16_np[-1, :] = np.nan
        moe16_bf16 = jnp.asarray(moe16_np, dtype=jnp.bfloat16)

        def _run_b2_16(wp_arg, sp_arg):
            return jax.shard_map(
                lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
                    *a,
                    world_size=16,
                    sharded_moe_in=sh16,
                    post_ff_eps=1e-5,
                    post_ff_scale_input=True,
                    next_in_eps=1e-5,
                    next_in_scale_input=False,
                    fuse_qkv_epi=False,
                    interpret=ip_race,
                ),
                mesh=mesh16,
                in_specs=(
                    P("model", None),
                    P("model", None) if sh16 else P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                    P(),
                    P(),
                    P(None, "model"),
                    P("model"),
                    P(None, "model"),
                    P("model"),
                    None,
                ),
                out_specs=(P("model", None), P(None, "model"), P(None, "model")),
                check_vma=False,
            )(st16_st, moe16_bf16, wp_arg, sp_arg, g16, a16, b16, g16, wq16, sq16, wg16, sg16, None)

        s16_0, q16_0, g16_0 = _run_b2_16(wpost16, spost16)
        s16_1, q16_1, g16_1 = _run_b2_16(wpost_tp2_16, spost_tp2_16)
        s16_0_r = np.concatenate([np.asarray(s16_0).reshape(16, r_pair16, H16)[2 * c] for c in range(8)], axis=0)
        s16_1_r = np.concatenate([np.asarray(s16_1).reshape(16, r_pair16, H16)[2 * c] for c in range(8)], axis=0)
        assert not np.isnan(s16_1_r[:valid16]).any()
        np.testing.assert_array_equal(s16_1_r[:valid16], s16_0_r[:valid16])
        np.testing.assert_array_equal(np.asarray(q16_1)[:valid16], np.asarray(q16_0)[:valid16])
        np.testing.assert_array_equal(np.asarray(g16_1)[:valid16], np.asarray(g16_0)[:valid16])

        def _run_fn_16(wp_arg, sp_arg):
            return jax.shard_map(
                lambda *a: w8.moe_post_chip_sharded_final_norm_jax(
                    *a,
                    world_size=16,
                    sharded_moe_in=sh16,
                    post_ff_eps=1e-5,
                    post_ff_scale_input=True,
                    final_eps=1e-5,
                    final_scale_input=False,
                    return_stream_too=True,
                    interpret=ip_race,
                ),
                mesh=mesh16,
                in_specs=(
                    P("model", None),
                    P("model", None) if sh16 else P(),
                    P(None, "model"),
                    P("model"),
                    P(),
                    P(),
                    P(),
                    P(),
                ),
                out_specs=(P("model", None), P("model", None)),
                check_vma=False,
            )(st16_st, moe16_bf16, wp_arg, sp_arg, g16, a16, b16, g16)

        stf0, hf0 = _run_fn_16(wpost16, spost16)
        stf1, hf1 = _run_fn_16(wpost_tp2_16, spost_tp2_16)
        stf0_r = np.concatenate([np.asarray(stf0).reshape(16, r_pair16, H16)[2 * c] for c in range(8)], axis=0)
        stf1_r = np.concatenate([np.asarray(stf1).reshape(16, r_pair16, H16)[2 * c] for c in range(8)], axis=0)
        np.testing.assert_array_equal(stf1_r[:valid16], stf0_r[:valid16])
        np.testing.assert_array_equal(
            np.asarray(hf1).reshape(16, B16, H16)[:, :valid16],
            np.asarray(hf0).reshape(16, B16, H16)[:, :valid16],
        )

    # 3. ag_chip_sharded_capture_jax 0.0-ULP bit-exact parity across TP=8 B=256 & TP=16 B=64/256 (w/ NaN tail row!)
    for tp, mesh, B_cap, K_H in ((8, mesh8, 256, 4096), (16, mesh16, 64, 4096), (16, mesh16, 256, 2048)):
        n_chips = tp // 2
        r_p = B_cap // n_chips
        cap_np = (rng.standard_normal((B_cap, K_H)) * 0.5).astype(np.float32)
        cap_np[-1, :] = np.nan
        cap_full = jnp.asarray(cap_np, dtype=jnp.bfloat16)
        cap_stacked = jnp.concatenate(
            [cap_full[(r >> 1) * r_p : ((r >> 1) + 1) * r_p] for r in range(tp)], axis=0
        )
        cap_out = jax.shard_map(
            lambda x: w8.ag_chip_sharded_capture_jax(x, world_size=tp, interpret=ip_race),
            mesh=mesh,
            in_specs=(P("model", None),),
            out_specs=P("model", None),
            check_vma=False,
        )(cap_stacked)
        cap_out_np = np.asarray(cap_out).reshape(tp, B_cap, K_H)
        cap_ref_np = np.asarray(cap_full)
        valid_cap = B_cap - 1
        for r in range(tp):
            assert not np.isnan(np.asarray(cap_out_np[r, :valid_cap], dtype=np.float32)).any()
            np.testing.assert_array_equal(cap_out_np[r, :valid_cap], cap_ref_np[:valid_cap])


def test_w9_chip_vmem_core_hooks_and_shared_pool_interpret():
    """Verify reusable _run_chip_b1/b2_vmem_core VMEM aliasing, pooled sems, and early hooks."""
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    devices = jax.devices()
    if len(devices) < 16:
        pytest.skip(f"Need >=16 host devices, got {len(devices)}")

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(1337)

    # 1. Standalone pooled+aliased chain (`b1_vmem_core -> b2_vmem_core`) vs standalone bridges at TP=8 & TP=16 (B=64)
    for tp, H, H_PRE, qkv_c, gate_c, fuse_epi in (
        (8, 2048, 4096, 1280, 512, True),
        (16, 2048, 1024, 512, 512, False),
    ):
        mesh = Mesh(np.array(devices[:tp]), ("model",))
        B = 64
        valid = B - 1
        r_pair = B // (tp // 2)
        half_h = H // 2
        n_chips = tp // 2
        g_dim = H // tp
        k_in = H_PRE
        pre_pair_cols = H_PRE // 2

        st_in_np = (rng.standard_normal((B, H)) * 0.5).astype(np.float32)
        st_in_np[-1, :] = np.nan
        st_stacked = jnp.asarray(
            np.concatenate([st_in_np[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(tp)], axis=0)
        )
        gated_np = (rng.standard_normal((B, tp * g_dim)) * 0.3).astype(np.float32)
        gated_np[-1, :] = np.nan
        gated = jnp.asarray(gated_np, dtype=jnp.bfloat16)

        wo = jnp.asarray((rng.standard_normal((tp * g_dim, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_wo = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
        pa_a = jnp.asarray((np.ones((H,)) * 0.91).astype(np.float32))
        pa_b = jnp.asarray((np.ones((H,)) * 0.87).astype(np.float32))
        g_pre = jnp.asarray((np.ones((H,)) * 1.04).astype(np.float32))
        g_exp = jnp.asarray((np.ones((H_PRE,)) * 0.96).astype(np.float32))
        w_pre_base = jnp.asarray((rng.standard_normal((H, H_PRE)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_pre_base = jnp.asarray((np.abs(rng.standard_normal((H_PRE,))) * 0.02 + 1e-3).astype(np.float32))
        w_pre = jnp.concatenate(
            [w_pre_base[:, :pre_pair_cols] if (r & 1) == 0 else w_pre_base[:, pre_pair_cols:] for r in range(tp)],
            axis=1,
        )
        s_pre = jnp.concatenate(
            [s_pre_base[:pre_pair_cols] if (r & 1) == 0 else s_pre_base[pre_pair_cols:] for r in range(tp)],
            axis=0,
        )
        w_rt = jnp.asarray((rng.standard_normal((H, 256)) * 0.03).astype(np.float32), dtype=jnp.bfloat16 if r_pair == 16 else jnp.float32)

        moe_np = (rng.standard_normal((B, H_PRE)) * 0.3).astype(np.float32)
        moe_np[-1, :] = np.nan
        moe_bf16 = jnp.asarray(moe_np, dtype=jnp.bfloat16)
        wpost_full = jnp.asarray((rng.standard_normal((H_PRE, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spost_full = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
        wpost_tp2 = jnp.concatenate(
            [wpost_full[:, :half_h] if (r & 1) == 0 else wpost_full[:, half_h:] for r in range(tp)],
            axis=1,
        )
        spost_tp2 = jnp.concatenate(
            [spost_full[:half_h] if (r & 1) == 0 else spost_full[half_h:] for r in range(tp)],
            axis=0,
        )
        w_qkv = jnp.asarray((rng.standard_normal((H, tp * qkv_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_qkv = jnp.asarray((np.abs(rng.standard_normal((tp * qkv_c,))) * 0.02 + 1e-3).astype(np.float32))
        w_gate = jnp.asarray((rng.standard_normal((H, tp * gate_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_gate = jnp.asarray((np.abs(rng.standard_normal((tp * gate_c,))) * 0.02 + 1e-3).astype(np.float32))
        cs = (
            build_decode_rope_step_tables(
                jnp.arange(B, dtype=jnp.int32),
                jnp.asarray(rng.standard_normal((B, 128)).astype(np.float32) * 0.5),
            )
            if fuse_epi
            else None
        )

        # Reference via standalone bridges
        ref_st1, ref_log1, ref_exp1 = jax.shard_map(
            lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
                *a,
                world_size=tp,
                pa_eps=1e-8,
                pre_ff_eps=1e-5,
                pre_ff_scale_input=True,
                router_multiplier=0.5,
                pre_exp_eps=1e-5,
                pre_exp_scale_input=False,
                interpret=ip_race,
            ),
            mesh=mesh,
            in_specs=(P("model", None), P(None, "model"), P("model", None), P(), P(), P(), P(), P(None, "model"), P("model"), P(), P()),
            out_specs=(P("model", None), P(None, None), P(None, None)),
            check_vma=False,
        )(st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp)

        ref_b2_res = jax.shard_map(
            lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
                *a,
                world_size=tp,
                sharded_moe_in=False,
                post_ff_eps=1e-5,
                post_ff_scale_input=True,
                next_in_eps=1e-5,
                next_in_scale_input=False,
                fuse_qkv_epi=fuse_epi,
                qkv_norm_eps=1e-6,
                query_scale=0.125,
                k_scale=0.75,
                v_scale=1.25,
                interpret=ip_race,
            ),
            mesh=mesh,
            in_specs=(
                P("model", None), P(), P(None, "model"), P("model"), P(), P(), P(), P(),
                P(None, "model"), P("model"), P(None, "model"), P("model"), P() if fuse_epi else None,
            ),
            out_specs=(
                (P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"))
                if fuse_epi
                else (P("model", None), P(None, "model"), P(None, "model"))
            ),
            check_vma=False,
        )(ref_st1, moe_bf16, wpost_tp2, spost_tp2, g_pre, pa_a, pa_b, g_pre, w_qkv, s_qkv, w_gate, s_gate, cs)
        jax.block_until_ready(ref_b2_res[0])

        # Fused in-VMEM test using shared pool & Mosaic bitcast/slice aliases
        # Also checks callbacks run AFTER out_router_logits_ref is fully all-gathered!
        vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
        v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
        two_pair = 2 * r_pair
        four_pair = 4 * r_pair
        if fuse_epi:
            out_shapes = (
                jax.ShapeDtypeStruct((r_pair, H), jnp.float32),
                jax.ShapeDtypeStruct((B, 256), jnp.float32),
                jax.ShapeDtypeStruct((B, H_PRE), jnp.bfloat16),
                jax.ShapeDtypeStruct((B, 256), jnp.float32),  # snapshot logits inside during_ag_exp_ici_fn
                jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((B, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((B, gate_c), jnp.bfloat16),
            )
        else:
            out_shapes = (
                jax.ShapeDtypeStruct((r_pair, H), jnp.float32),
                jax.ShapeDtypeStruct((B, 256), jnp.float32),
                jax.ShapeDtypeStruct((B, H_PRE), jnp.bfloat16),
                jax.ShapeDtypeStruct((B, 256), jnp.float32),
                jax.ShapeDtypeStruct((B, qkv_c), jnp.bfloat16),
                jax.ShapeDtypeStruct((B, gate_c), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
            )

        def _chain_kernel(*refs):
            (
                st_in_hbm, g_ref, wo_hbm, swo_ref, pa_a_ref, pa_b_ref, gpre_ref,
                wpre_hbm, spre_ref, wrt_ref, gexp_ref, moe_ref, wpost_hbm, spost_ref,
                wq_hbm, sq_ref, wg_hbm, sg_ref, cs_ref,
                out_st_hbm, out_log_ref, out_exp_ref, out_snap_log_ref,
                out_q_ref, out_kv_ref, out_gate_epi_ref,
                w1_pool_fp8, w2_pool_fp8, w_pre_pool_fp8, w_rt_buf, w_gate_buf,
                s_buf, send_half, mate_recv, own_half,
                rs1_a, rs1_b, rs2_a, rs2_b, my_pair_bf16, recv_pair_bf16,
                sq_pa, ffn_pair, router_pair, pre_my, pre_mate, sq_pre, exp_3d,
                next_in_pair, next_in_vmem, qkv_vmem,
                mate_tok, moe_full, post_shard, mate_post_shard,
                chip_cols, post_pair_full, cols_3d, ag_3d,
                dma_sems, d2d_send, d2d_recv, ici_send, ici_recv, log_send, log_recv,
            ) = refs

            # Note: TPU7x LLO compiles u32.bitcast(fp8).reshape(...); CPU interpret DMA to_range only accepts slices.
            wo_alias = w2_pool_fp8
            wpost_alias = w1_pool_fp8
            w_pre_alias = w_pre_pool_fp8
            w_qkv_alias = w_pre_pool_fp8.at[:, pl.ds(0, qkv_c)]

            def _snapshot_logits_in_ici():
                out_snap_log_ref[...] = out_log_ref[...]
                if r_pair >= 16:
                    # Pre-start b2 DMAs into aliased dead buffers while ICI overlaps (tp == 8 test)
                    pltpu.make_async_copy(wq_hbm, w_qkv_alias, dma_sems.at[4]).start()
                    pltpu.make_async_copy(wg_hbm, w_gate_buf, dma_sems.at[5]).start()

            w8._run_chip_b1_vmem_core(
                st_in_hbm, g_ref, wo_hbm, swo_ref, pa_a_ref, pa_b_ref, gpre_ref,
                wpre_hbm, spre_ref, wrt_ref, gexp_ref,
                None, out_log_ref, out_exp_ref,
                wo_alias, w_pre_alias, w_rt_buf, s_buf,
                send_half, mate_recv, own_half,
                rs1_a, rs1_b, rs2_a, rs2_b, my_pair_bf16, recv_pair_bf16,
                sq_pa, ffn_pair, router_pair, pre_my, pre_mate, sq_pre, exp_3d,
                dma_sems.at[0:2], dma_sems.at[2:3], dma_sems.at[3:4],
                dma_sems.at[4:5], dma_sems.at[5:6],
                d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv,
                ici_send, ici_recv,
                ici_send, ici_recv,
                d2d_send.at[0:1], d2d_recv.at[0:1],
                d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv,
                d2d_send.at[0:1], d2d_recv.at[0:1],
                log_send, log_recv,
                world_size=tp,
                pa_eps=1e-8,
                pre_ff_eps=1e-5,
                pre_ff_scale_input=True,
                router_multiplier=0.5,
                pre_exp_eps=1e-5,
                pre_exp_scale_input=False,
                skip_barrier_signal=False,
                skip_barrier_wait=False,
                skip_stream_load=False,
                skip_stream_store=True,
                early_router_and_log_ag=True,
                during_ag_exp_ici_fn=_snapshot_logits_in_ici,
                during_ag_exp_d2d2_fn=None,
            )

            pre_qkv_wg = r_pair >= 16
            if pre_qkv_wg:
                pltpu.make_async_copy(wpost_hbm, wpost_alias, dma_sems.at[0]).start()

            w8._run_chip_b2_vmem_core(
                None, moe_ref, wpost_hbm, spost_ref, gpre_ref, pa_a_ref, pa_b_ref, gpre_ref,
                wq_hbm, sq_ref, wg_hbm, sg_ref, cs_ref,
                out_st_hbm, out_q_ref, out_kv_ref, out_gate_epi_ref,
                wpost_alias, w_qkv_alias, w_gate_buf, s_buf,
                next_in_pair, next_in_vmem, qkv_vmem,
                mate_tok, moe_full, post_shard, mate_post_shard,
                chip_cols, post_pair_full, cols_3d, ag_3d,
                dma_sems.at[0:1], dma_sems.at[4:5], dma_sems.at[5:6],
                dma_sems.at[2:3], dma_sems.at[3:4],
                d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv,
                d2d_send.at[0:1], d2d_recv.at[0:1],
                d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv,
                ici_send, ici_recv,
                d2d_send.at[0:1], d2d_recv.at[0:1],
                world_size=tp,
                sharded_moe_in=False,
                post_ff_eps=1e-5,
                post_ff_scale_input=True,
                next_in_eps=1e-5,
                next_in_scale_input=False,
                fuse_qkv_epi=fuse_epi,
                qkv_norm_eps=1e-6,
                query_scale=0.125,
                k_scale=0.75,
                v_scale=1.25,
                has_rope=fuse_epi,
                interpret_mode=True,
                skip_barrier_signal=True,
                skip_barrier_wait=True,
                skip_stream_load=True,
                skip_stream_store=False,
                prestarted_w_post_dma=pre_qkv_wg,
                prestarted_w_qkv_dma=pre_qkv_wg,
                prestarted_w_gate_dma=pre_qkv_wg,
            )

        in_specs = [
            h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp,
            h_sp, v_sp, h_sp, v_sp, v_sp, h_sp, v_sp,
            h_sp, v_sp, h_sp, v_sp, v_sp if fuse_epi else None,
        ]
        out_specs = (h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, v_sp)
        # Same-dtype pool views for CPU interpret; TPU7x LLO covers u32.bitcast(fp8).reshape(...)
        use_xy = tp == 8 and r_pair >= 16
        q_cols = half_h // 2
        scratch_shapes = [
            vmem((H_PRE, half_h), jnp.float8_e4m3fn),
            vmem((2, g_dim, half_h), jnp.float8_e4m3fn),
            vmem((H, pre_pair_cols), jnp.float8_e4m3fn),
            vmem((H, 256), jnp.bfloat16 if r_pair == 16 else jnp.float32),
            vmem((H, gate_c), jnp.float8_e4m3fn),
            vmem((r_pair, H), jnp.float32),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((two_pair, q_cols) if use_xy else ((four_pair, half_h) if tp == 16 else (8, 128)), jnp.bfloat16),
            vmem((two_pair, q_cols) if use_xy else (two_pair, half_h), jnp.bfloat16),
            vmem((r_pair, q_cols) if use_xy else (r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, q_cols) if use_xy else ((r_pair, half_h) if r_pair == 8 else (8, 128)), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, 128), jnp.float32),
            vmem((r_pair, H), jnp.bfloat16),
            vmem((8, 128) if r_pair == 16 else (r_pair, H), jnp.float32),
            vmem((r_pair, pre_pair_cols), jnp.bfloat16),
            vmem((r_pair, pre_pair_cols) if r_pair == 8 else (r_pair, 128), jnp.bfloat16 if r_pair == 8 else jnp.float32),
            vmem((r_pair, 128), jnp.float32),
            vmem((n_chips, 8, H_PRE) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((8, H) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((B, H), jnp.bfloat16),
            vmem((B, qkv_c) if fuse_epi else (8, 128), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((r_pair, k_in) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, H), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((n_chips, 8, half_h) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((6,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ]
        # Total sems = 6 + 2 + 4*(n_chips-1) = 20 (TP=8) / 36 (TP=16) << 64!
        chain_call = w8._pallas_call_maybe_interpret(
            _chain_kernel,
            interpret=ip_race,
            out_shape=out_shapes,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            compiler_params=pltpu.CompilerParams(
                collective_id=w8._W8_CID_BASE + 99,
                vmem_limit_bytes=60 * 1024**2,
                disable_bounds_checks=True,
            ),
        )
        args = [
            st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp,
            moe_bf16, wpost_tp2, spost_tp2, w_qkv, s_qkv, w_gate, s_gate, cs,
        ]
        in_p_specs = [
            P("model", None), P(None, "model"), P("model", None), P(), P(), P(), P(),
            P(None, "model"), P("model"), P(), P(), P(), P(None, "model"), P("model"),
            P(None, "model"), P("model"), P(None, "model"), P("model"), P() if fuse_epi else None,
        ]
        out_p_specs = (
            (P("model", None), P(None, None), P(None, None), P(None, None),
             P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"))
            if fuse_epi
            else (P("model", None), P(None, None), P(None, None), P(None, None),
                  P(None, "model"), P(None, "model"), P(None, None))
        )
        def _run_chain_sharded(*xs):
            (
                x_st, x_g, x_wo, x_swo, x_paa, x_pab, x_gpre, x_wpre, x_spre,
                x_wrt, x_gexp, x_moe, x_wpost, x_spost, x_wq, x_sq, x_wg, x_sg, x_cs,
            ) = xs
            st_c, lg_c, ex_c, snap_c, q_c, kv_c, g_c = chain_call(
                x_st, x_g, x_wo,
                w8._as_dot_scale_f32(x_swo, H),
                w8._as_2d_f32(x_paa, H),
                w8._as_2d_f32(x_pab, H),
                w8._as_2d_f32(x_gpre, H),
                x_wpre,
                w8._as_dot_scale_f32(x_spre, pre_pair_cols),
                x_wrt,
                w8._as_2d_f32(x_gexp, H_PRE),
                x_moe,
                x_wpost,
                w8._as_dot_scale_f32(x_spost, half_h),
                x_wq,
                w8._as_dot_scale_f32(x_sq, qkv_c),
                x_wg,
                w8._as_dot_scale_f32(x_sg, gate_c),
                x_cs,
            )
            if fuse_epi:
                q_c = q_c.reshape(1, B, 8, 2, 128)
            return st_c, lg_c, ex_c, snap_c, q_c, kv_c, g_c

        got = jax.shard_map(
            _run_chain_sharded,
            mesh=mesh,
            in_specs=tuple(in_p_specs),
            out_specs=out_p_specs,
            check_vma=False,
        )(*args)
        jax.block_until_ready(got[0])

        got_st, got_log, got_exp, got_snap_log, got_q, got_kv, got_g = got
        # Verify 0-ULP bit-exact parity vs standalone bridge sequence + callback saw finalized logits!
        assert not np.isnan(np.asarray(got_log)[:valid]).any()
        assert not np.isnan(np.asarray(got_exp)[:valid]).any()
        np.testing.assert_array_equal(np.asarray(got_log)[:valid], np.asarray(ref_log1)[:valid])
        np.testing.assert_array_equal(np.asarray(got_snap_log)[:valid], np.asarray(ref_log1)[:valid])
        np.testing.assert_array_equal(np.asarray(got_exp)[:valid], np.asarray(ref_exp1)[:valid])

        got_st_r = np.concatenate([np.asarray(got_st).reshape(tp, r_pair, H)[2 * c] for c in range(n_chips)], axis=0)
        ref_st2 = ref_b2_res[0]
        ref_st2_r = np.concatenate([np.asarray(ref_st2).reshape(tp, r_pair, H)[2 * c] for c in range(n_chips)], axis=0)
        assert not np.isnan(got_st_r[:valid]).any()
        np.testing.assert_array_equal(got_st_r[:valid], ref_st2_r[:valid])
        if fuse_epi:
            _, ref_q, ref_kv, ref_g = ref_b2_res
            np.testing.assert_array_equal(np.asarray(got_q)[:, :valid], np.asarray(ref_q)[:, :valid])
            np.testing.assert_array_equal(
                np.asarray(jax.lax.bitcast_convert_type(got_kv, jnp.uint8))[:valid],
                np.asarray(jax.lax.bitcast_convert_type(ref_kv, jnp.uint8))[:valid],
            )
            np.testing.assert_array_equal(np.asarray(got_g)[:valid], np.asarray(ref_g)[:valid])
        else:
            _, ref_qkv, ref_gate = ref_b2_res
            np.testing.assert_array_equal(np.asarray(got_q)[:valid], np.asarray(ref_qkv)[:valid])
            np.testing.assert_array_equal(np.asarray(got_kv)[:valid], np.asarray(ref_gate)[:valid])


def test_chip_b1_direct_3d_out_and_b2_step_c_hooks_parity():
    """Verify _run_chip_b1_vmem_core(out_expert_3d_ref) & _run_chip_b2_vmem_core Step-(c) hooks 0-ULP."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    devices = jax.devices()
    if len(devices) < 16:
        pytest.skip(f"Need >=16 host devices, got {len(devices)}")

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261125)
    H, H_PRE = 2048, 4096
    pre_pair_cols = H_PRE // 2
    half_h = H // 2

    for tp, qkv_c, gate_c in ((8, 1280, 1024), (16, 640, 512)):
        mesh = Mesh(np.array(devices[:tp]), ("model",))
        B, valid = 64, 63
        r_pair = B // (tp // 2)
        n_chips = tp // 2
        g_dim = H // tp
        two_pair, four_pair, q_cols = 2 * r_pair, 4 * r_pair, half_h // 2

        st_in_np = (rng.standard_normal((B, H)) * 0.4).astype(np.float32)
        st_in_np[-1, :] = np.nan
        st_stacked = jnp.asarray(
            np.concatenate([st_in_np[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(tp)], axis=0)
        )
        gated_np = (rng.standard_normal((B, H)) * 0.3).astype(np.float32)
        gated_np[-1, :] = np.nan
        gated = jnp.asarray(gated_np, dtype=jnp.bfloat16)

        wo = jnp.asarray((rng.standard_normal((H, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_wo = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
        pa_a = jnp.full((H,), 0.91, jnp.float32)
        pa_b = jnp.full((H,), 0.87, jnp.float32)
        g_pre = jnp.full((H,), 1.04, jnp.float32)
        g_exp = jnp.full((H_PRE,), 0.96, jnp.float32)
        w_pre_base = jnp.asarray((rng.standard_normal((H, H_PRE)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_pre_base = (np.abs(rng.standard_normal((H_PRE,))) * 0.02 + 1e-3).astype(np.float32)
        w_pre = jnp.concatenate(
            [w_pre_base[:, :pre_pair_cols] if (r & 1) == 0 else w_pre_base[:, pre_pair_cols:] for r in range(tp)],
            axis=1,
        )
        s_pre = jnp.asarray(
            np.concatenate(
                [s_pre_base[:pre_pair_cols] if (r & 1) == 0 else s_pre_base[pre_pair_cols:] for r in range(tp)],
                axis=0,
            )
        )
        w_rt = jnp.asarray((rng.standard_normal((H, 256)) * 0.03).astype(np.float32), dtype=jnp.bfloat16)

        moe_np = (rng.standard_normal((B, H_PRE)) * 0.3).astype(np.float32)
        moe_np[-1, :] = np.nan
        moe_bf16 = jnp.asarray(moe_np, dtype=jnp.bfloat16)
        wpost_full = jnp.asarray((rng.standard_normal((H_PRE, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spost_full = (np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32)
        wpost_tp2 = jnp.concatenate(
            [wpost_full[:, :half_h] if (r & 1) == 0 else wpost_full[:, half_h:] for r in range(tp)],
            axis=1,
        )
        spost_tp2 = jnp.asarray(
            np.concatenate(
                [spost_full[:half_h] if (r & 1) == 0 else spost_full[half_h:] for r in range(tp)],
                axis=0,
            )
        )
        w_qkv = jnp.asarray((rng.standard_normal((H, tp * qkv_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_qkv = jnp.asarray((np.abs(rng.standard_normal((tp * qkv_c,))) * 0.02 + 1e-3).astype(np.float32))
        w_gate = jnp.asarray((rng.standard_normal((H, tp * gate_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        s_gate = jnp.asarray((np.abs(rng.standard_normal((tp * gate_c,))) * 0.02 + 1e-3).astype(np.float32))
        cs = build_decode_rope_step_tables(
            jnp.arange(B, dtype=jnp.int32),
            jnp.asarray(rng.standard_normal((B, 128)).astype(np.float32) * 0.5),
        )

        ref_st1, ref_log1, ref_exp1 = jax.jit(jax.shard_map(
            lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
                *a, world_size=tp, pa_eps=1e-8, pre_ff_eps=1e-5, pre_ff_scale_input=True,
                router_multiplier=0.5, pre_exp_eps=1e-5, pre_exp_scale_input=False, interpret=ip_race,
            ),
            mesh=mesh,
            in_specs=(P("model", None), P(None, "model"), P("model", None), P(), P(), P(), P(), P(None, "model"), P("model"), P(), P()),
            out_specs=(P("model", None), P(None, None), P(None, None)),
            check_vma=False,
        ))(st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp)

        ref_b2 = jax.jit(jax.shard_map(
            lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
                *a, world_size=tp, sharded_moe_in=False, post_ff_eps=1e-5, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=False, fuse_qkv_epi=True,
                qkv_norm_eps=1e-6, query_scale=0.125, k_scale=0.75, v_scale=1.25, interpret=ip_race,
            ),
            mesh=mesh,
            in_specs=(
                P("model", None), P(), P(None, "model"), P("model"), P(), P(), P(), P(),
                P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
            ),
            out_specs=(P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
            check_vma=False,
        ))(ref_st1, moe_bf16, wpost_tp2, spost_tp2, g_pre, pa_a, pa_b, g_pre, w_qkv, s_qkv, w_gate, s_gate, cs)
        jax.block_until_ready(ref_b2[0])

        vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
        v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
        out_shapes = (
            jax.ShapeDtypeStruct((r_pair, H), jnp.float32),
            jax.ShapeDtypeStruct((B, 256), jnp.float32),
            jax.ShapeDtypeStruct((B, 32, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((B, gate_c), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 1, 4, 128), jnp.float8_e4m3fn),
        )

        def _x3d_step_c_kernel(*refs):
            (
                st_in_hbm, g_ref, wo_hbm, swo_ref, pa_a_ref, pa_b_ref, gpre_ref,
                wpre_hbm, spre_ref, wrt_ref, gexp_ref, moe_ref, wpost_hbm, spost_ref,
                wq_hbm, sq_ref, wg_hbm, sg_ref, cs_ref,
                out_st_hbm, out_log_ref, out_exp3d_ref, out_q_ref, out_kv_ref, out_gate_ref,
                out_snap_q_ref, out_snap_kv_ref,
                wo_buf, wpost_buf, w_pre_pool, w_rt_buf, w_gate_buf,
                s_buf, send_half, mate_recv, own_half,
                rs1_a, rs1_b, rs2_a, rs2_b, my_pair_bf16, recv_pair_bf16,
                sq_pa, ffn_pair, router_pair, pre_my, pre_mate, sq_pre, exp_3d,
                next_in_pair, next_in_vmem, qkv_vmem, gate_vmem,
                mate_tok, moe_full, post_shard, mate_post_shard,
                chip_cols, post_pair_full, cols_3d, ag_3d,
                dma_sems, d2d_send, d2d_recv, ici_send, ici_recv, log_send, log_recv,
            ) = refs
            w_qkv_alias = w_pre_pool.at[:, pl.ds(0, qkv_c)]

            w8._run_chip_b1_vmem_core(
                st_in_hbm, g_ref, wo_hbm, swo_ref, pa_a_ref, pa_b_ref, gpre_ref,
                wpre_hbm, spre_ref, wrt_ref, gexp_ref,
                None, out_log_ref, None,
                wo_buf, w_pre_pool, w_rt_buf, s_buf,
                send_half, mate_recv, own_half,
                rs1_a, rs1_b, rs2_a, rs2_b, my_pair_bf16, recv_pair_bf16,
                sq_pa, ffn_pair, router_pair, pre_my, pre_mate, sq_pre, exp_3d,
                dma_sems.at[0:2], dma_sems.at[2:3], dma_sems.at[3:4],
                dma_sems.at[4:5], dma_sems.at[5:6],
                d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv, ici_send, ici_recv, ici_send, ici_recv,
                d2d_send.at[0:1], d2d_recv.at[0:1],
                d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv, d2d_send.at[0:1], d2d_recv.at[0:1], log_send, log_recv,
                world_size=tp, pa_eps=1e-8, pre_ff_eps=1e-5, pre_ff_scale_input=True,
                router_multiplier=0.5, pre_exp_eps=1e-5, pre_exp_scale_input=False,
                skip_barrier_signal=False, skip_barrier_wait=False,
                skip_stream_load=False, skip_stream_store=True,
                early_router_and_log_ag=True, out_expert_3d_ref=out_exp3d_ref,
            )

            def _after_qkv():
                out_snap_q_ref[...] = out_q_ref[...]
                out_snap_kv_ref[...] = out_kv_ref[...]

            w8._run_chip_b2_vmem_core(
                None, moe_ref, wpost_hbm, spost_ref, gpre_ref, pa_a_ref, pa_b_ref, gpre_ref,
                wq_hbm, sq_ref, wg_hbm, sg_ref, cs_ref,
                out_st_hbm, out_q_ref, out_kv_ref, out_gate_ref,
                wpost_buf, w_qkv_alias, w_gate_buf, s_buf,
                next_in_pair, next_in_vmem, qkv_vmem,
                mate_tok, moe_full, post_shard, mate_post_shard,
                chip_cols, post_pair_full, cols_3d, ag_3d,
                dma_sems.at[0:1], dma_sems.at[4:5], dma_sems.at[5:6],
                dma_sems.at[2:3], dma_sems.at[3:4],
                d2d_send.at[0:1], d2d_recv.at[0:1], ici_send, ici_recv,
                d2d_send.at[0:1], d2d_recv.at[0:1], d2d_send.at[0:1], d2d_recv.at[0:1],
                ici_send, ici_recv, ici_send, ici_recv, d2d_send.at[0:1], d2d_recv.at[0:1],
                world_size=tp, sharded_moe_in=False,
                post_ff_eps=1e-5, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=False,
                fuse_qkv_epi=True, qkv_norm_eps=1e-6, query_scale=0.125, k_scale=0.75, v_scale=1.25,
                has_rope=True, interpret_mode=True,
                skip_barrier_signal=True, skip_barrier_wait=True,
                skip_stream_load=True, skip_stream_store=False,
                gate_dst_ref=gate_vmem, after_qkv_epi_fn=_after_qkv, defer_stream_store_wait=True,
            )
            out_gate_ref[...] = gate_vmem[...]
            w8._dma_wait(out_st_hbm, dma_sems.at[3])

        use_xy = tp == 8 and r_pair >= 16
        q_cols = half_h // 2
        scratch_shapes = [
            vmem((2, g_dim, half_h), jnp.float8_e4m3fn),
            vmem((H_PRE, half_h), jnp.float8_e4m3fn),
            vmem((H, pre_pair_cols), jnp.float8_e4m3fn),
            vmem((H, 256), jnp.bfloat16),
            vmem((H, gate_c), jnp.float8_e4m3fn),
            vmem((r_pair, H), jnp.float32),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((two_pair, q_cols) if use_xy else ((four_pair, half_h) if tp == 16 else (8, 128)), jnp.bfloat16),
            vmem((two_pair, q_cols) if use_xy else (two_pair, half_h), jnp.bfloat16),
            vmem((r_pair, q_cols) if use_xy else (r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, q_cols) if use_xy else ((r_pair, half_h) if r_pair == 8 else (8, 128)), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, 128), jnp.float32),
            vmem((r_pair, H), jnp.bfloat16),
            vmem((8, 128), jnp.float32),
            vmem((r_pair, pre_pair_cols), jnp.bfloat16),
            vmem((r_pair, pre_pair_cols) if r_pair == 8 else (r_pair, 128), jnp.bfloat16 if r_pair == 8 else jnp.float32),
            vmem((r_pair, 128), jnp.float32),
            vmem((n_chips, 8, H_PRE) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((8, H) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((B, H), jnp.bfloat16),
            vmem((B, qkv_c), jnp.bfloat16),
            vmem((B, gate_c), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((r_pair, H_PRE) if r_pair == 8 else (8, 128), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, H), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((n_chips, 8, half_h) if r_pair == 8 else (8, 128), jnp.bfloat16),
            sem((6,)), sem((1,)), sem((1,)), sem((n_chips - 1,)), sem((n_chips - 1,)),
            sem((n_chips - 1,)), sem((n_chips - 1,)),
        ]
        call = w8._pallas_call_maybe_interpret(
            _x3d_step_c_kernel, interpret=ip_race, out_shape=out_shapes,
            in_specs=(
                h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp,
                v_sp, h_sp, v_sp, h_sp, v_sp, h_sp, v_sp, v_sp,
            ),
            out_specs=(h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, v_sp, v_sp),
            scratch_shapes=scratch_shapes,
            compiler_params=pltpu.CompilerParams(collective_id=w8._W8_CID_BASE + 101, disable_bounds_checks=True),
        )
        in_p_specs = (
            P("model", None), P(None, "model"), P("model", None), P(), P(), P(), P(),
            P(None, "model"), P("model"), P(), P(), P(), P(None, "model"), P("model"),
            P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
        )
        out_p_specs = (
            P("model", None), P(None, None), P(None, None, None),
            P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"),
            P(None, None, "model", None, None), P(None, "model", None, None),
        )
        args = (
            st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp,
            moe_bf16, wpost_tp2, spost_tp2, w_qkv, s_qkv, w_gate, s_gate, cs,
        )
        got = jax.jit(jax.shard_map(
            lambda *xs: (lambda r: (r[0], r[1], r[2], r[3].reshape(1, B, 8, 2, 128), r[4], r[5], r[6].reshape(1, B, 8, 2, 128), r[7]))(
                call(
                    xs[0], xs[1], xs[2], w8._as_dot_scale_f32(xs[3], H),
                    w8._as_2d_f32(xs[4], H), w8._as_2d_f32(xs[5], H), w8._as_2d_f32(xs[6], H),
                    xs[7], w8._as_dot_scale_f32(xs[8], pre_pair_cols), xs[9], w8._as_2d_f32(xs[10], H_PRE),
                    xs[11], xs[12], w8._as_dot_scale_f32(xs[13], half_h),
                    xs[14], w8._as_dot_scale_f32(xs[15], qkv_c), xs[16], w8._as_dot_scale_f32(xs[17], gate_c), xs[18],
                )
            ),
            mesh=mesh, in_specs=in_p_specs, out_specs=out_p_specs, check_vma=False,
        ))(*args)
        jax.block_until_ready(got[0])
        got_st, got_log, got_exp3d, got_q, got_kv, got_g, snap_q, snap_kv = got
        np.testing.assert_array_equal(np.asarray(got_log)[:valid], np.asarray(ref_log1)[:valid])
        np.testing.assert_array_equal(np.asarray(got_exp3d).reshape(B, H_PRE)[:valid], np.asarray(ref_exp1)[:valid])
        got_st_r = np.concatenate([np.asarray(got_st).reshape(tp, r_pair, H)[2 * c] for c in range(n_chips)], axis=0)
        ref_st_r = np.concatenate([np.asarray(ref_b2[0]).reshape(tp, r_pair, H)[2 * c] for c in range(n_chips)], axis=0)
        np.testing.assert_array_equal(got_st_r[:valid], ref_st_r[:valid])
        np.testing.assert_array_equal(np.asarray(got_q)[:, :valid], np.asarray(ref_b2[1])[:, :valid])
        np.testing.assert_array_equal(np.asarray(snap_q)[:, :valid], np.asarray(ref_b2[1])[:, :valid])
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(got_kv, jnp.uint8))[:valid],
            np.asarray(jax.lax.bitcast_convert_type(ref_b2[2], jnp.uint8))[:valid],
        )
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(snap_kv, jnp.uint8))[:valid],
            np.asarray(jax.lax.bitcast_convert_type(ref_b2[2], jnp.uint8))[:valid],
        )
        np.testing.assert_array_equal(np.asarray(got_g)[:valid], np.asarray(ref_b2[3])[:valid])


def test_rep_b2_vmem_core_b16_hooks_parity():
    """Verify _run_rep_b2_vmem_core at B=16 (TP=8 & TP=16) with Step-(c) hooks matches 0-ULP."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    devices = jax.devices()
    if len(devices) < 16:
        pytest.skip(f"Need >=16 host devices, got {len(devices)}")

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261126)
    B, valid, H, H_PRE = 16, 15, 2048, 4096

    for tp, qkv_c, gate_c in ((8, 1280, 1024), (16, 640, 512)):
        mesh = Mesh(np.array(devices[:tp]), ("model",))
        post_c = H // tp
        n_chips = tp // 2
        st_np = (rng.standard_normal((B, H)) * 0.4).astype(np.float32); st_np[-1, :] = np.nan
        moe_np = (rng.standard_normal((B, H_PRE)) * 0.3).astype(np.float32); moe_np[-1, :] = np.nan
        st, moe = jnp.asarray(st_np), jnp.asarray(moe_np, dtype=jnp.bfloat16)
        wpost = jnp.asarray((rng.standard_normal((H_PRE, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spost = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
        g = jnp.full((H,), 1.02, jnp.float32)
        pa = jnp.full((H,), 0.91, jnp.float32)
        pb = jnp.full((H,), 0.87, jnp.float32)
        wq = jnp.asarray((rng.standard_normal((H, tp * qkv_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        sq = jnp.asarray((np.abs(rng.standard_normal((tp * qkv_c,))) * 0.02 + 1e-3).astype(np.float32))
        wg = jnp.asarray((rng.standard_normal((H, tp * gate_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        sg = jnp.asarray((np.abs(rng.standard_normal((tp * gate_c,))) * 0.02 + 1e-3).astype(np.float32))
        cs = build_decode_rope_step_tables(
            jnp.arange(B, dtype=jnp.int32),
            jnp.asarray(rng.standard_normal((B, 128)).astype(np.float32) * 0.5),
        )

        ref = jax.jit(jax.shard_map(
            lambda *a: w8.moe_to_attn_qkv_epi_bridge_jax(
                *a, world_size=tp, post_ff_eps=1e-5, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=False, sharded_moe_in=False,
                qkv_norm_eps=1e-6, query_scale=0.125, k_scale=0.75, v_scale=1.25, interpret=ip_race,
            ),
            mesh=mesh,
            in_specs=(
                P(), P(), P(None, "model"), P("model"), P(), P(), P(), P(),
                P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
            ),
            out_specs=(P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
            check_vma=False,
        ))(st, moe, wpost, spost, g, pa, pb, g, wq, sq, wg, sg, cs)
        jax.block_until_ready(ref[0])

        vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
        v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
        out_shapes = (
            jax.ShapeDtypeStruct((B, H), jnp.float32),
            jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((B, gate_c), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
        )

        def _k(*refs):
            (
                st_h, moe_r, wp_h, sp_r, pfg_r, pa_r, pb_r, ng_r, wq_h, sq_r, wg_h, sg_r, cs_r,
                ost_h, oq_r, okv_r, og_r, snap_q_r,
                wp_b, wq_b, wg_b, sb_v, nin_v, qkv_v, gate_v, psh, ha, hb, d1, d2,
                sems, s1, s2, i_s, i_r, s3, r3, ts1s, ts1r, ts2s, ts2r,
            ) = refs
            pltpu.make_async_copy(st_h, sb_v, sems.at[3]).start()
            pltpu.make_async_copy(wp_h, wp_b, sems.at[0]).start()
            pltpu.make_async_copy(wq_h, wq_b, sems.at[1]).start()
            pltpu.make_async_copy(wg_h, wg_b, sems.at[2]).start()
            w8._dma_wait(sb_v, sems.at[3])
            w8._run_rep_b2_vmem_core(
                st_h, moe_r, wp_h, sp_r, pfg_r, pa_r, pb_r, ng_r, wq_h, sq_r, wg_h, sg_r, cs_r,
                ost_h, oq_r, okv_r, og_r,
                wp_b, wq_b, wg_b, sb_v, nin_v, qkv_v, psh, ha, hb, d1, d2,
                sems.at[0:1], sems.at[1:2], sems.at[2:3], sems.at[3:4], sems.at[4:5],
                s1, s2, i_s, i_r, s3, r3, ts1s, ts1r, ts2s, ts2r,
                world_size=tp, post_ff_eps=1e-5, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=False, sharded_moe_in=False,
                qkv_norm_eps=1e-6, query_scale=0.125, k_scale=0.75, v_scale=1.25,
                has_rope=True, norm_before_s3=False, interpret_mode=True,
                skip_barrier_signal=False, skip_barrier_wait=False,
                skip_stream_load=True, skip_stream_store=False,
                prestarted_w_post_dma=True, prestarted_w_qkv_dma=True, prestarted_w_gate_dma=True,
                gate_dst_ref=gate_v,
                after_qkv_epi_fn=lambda: snap_q_r.__setitem__(slice(None), oq_r[...]),
                defer_stream_store_wait=True,
            )
            og_r[...] = gate_v[...]
            w8._dma_wait(ost_h, sems.at[4])

        call = w8._pallas_call_maybe_interpret(
            _k, interpret=ip_race, out_shape=out_shapes,
            in_specs=(h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp, v_sp),
            out_specs=(h_sp, v_sp, v_sp, v_sp, v_sp),
            scratch_shapes=(
                vmem((H_PRE, post_c), jnp.float8_e4m3fn),
                vmem((H, qkv_c), jnp.float8_e4m3fn),
                vmem((H, gate_c), jnp.float8_e4m3fn),
                vmem((B, H), jnp.float32),
                vmem((B, H), jnp.bfloat16),
                vmem((B, qkv_c), jnp.bfloat16),
                vmem((B, gate_c), jnp.bfloat16),
                vmem((B, post_c), jnp.bfloat16),
                vmem((B, H), jnp.bfloat16),
                vmem((B, H), jnp.bfloat16),
                vmem((8, 128), jnp.bfloat16),
                vmem((8, 128), jnp.bfloat16),
                sem((5,)), sem((1,)), sem((1,)), sem((n_chips - 1,)), sem((n_chips - 1,)),
                sem((1,)), sem((1,)), sem((1,)), sem((1,)), sem((1,)), sem((1,)),
            ),
            compiler_params=pltpu.CompilerParams(collective_id=w8._W8_CID_BASE + 102, disable_bounds_checks=True),
        )
        got = jax.jit(jax.shard_map(
            lambda *xs: (lambda r: (r[0], r[1].reshape(1, B, 8, 2, 128), r[2], r[3], r[4].reshape(1, B, 8, 2, 128)))(
                call(
                    xs[0], xs[1], xs[2], w8._as_dot_scale_f32(xs[3], post_c),
                    w8._as_2d_f32(xs[4], H), w8._as_2d_f32(xs[5], H), w8._as_2d_f32(xs[6], H), w8._as_2d_f32(xs[7], H),
                    xs[8], w8._as_dot_scale_f32(xs[9], qkv_c), xs[10], w8._as_dot_scale_f32(xs[11], gate_c), xs[12],
                )
            ),
            mesh=mesh,
            in_specs=(P(), P(), P(None, "model"), P("model"), P(), P(), P(), P(), P(None, "model"), P("model"), P(None, "model"), P("model"), P()),
            out_specs=(P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"), P(None, None, "model", None, None)),
            check_vma=False,
        ))(st, moe, wpost, spost, g, pa, pb, g, wq, sq, wg, sg, cs)
        jax.block_until_ready(got[0])
        np.testing.assert_array_equal(np.asarray(got[0])[:valid], np.asarray(ref[0])[:valid])
        np.testing.assert_array_equal(np.asarray(got[1])[:, :valid], np.asarray(ref[1])[:, :valid])
        np.testing.assert_array_equal(np.asarray(got[4])[:, :valid], np.asarray(ref[1])[:, :valid])
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(got[2], jnp.uint8))[:valid],
            np.asarray(jax.lax.bitcast_convert_type(ref[2], jnp.uint8))[:valid],
        )
        np.testing.assert_array_equal(np.asarray(got[3])[:valid], np.asarray(ref[3])[:valid])


def test_rep_b1_b2_b16_vmem_core_chain_and_3d_out_parity():
    """Verify _run_rep_b1_vmem_core(out_expert_3d_ref, during_ag_exp_ici_fn) -> _run_rep_b2_vmem_core."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges as b7
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    assert w8._run_rep_b1_vmem_core is b7._run_rep_b1_vmem_core
    devices = jax.devices()
    if len(devices) < 16:
        pytest.skip(f"Need >=16 host devices, got {len(devices)}")

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261128)

    def _build_chain_call(*, tp: int, B: int, H: int, H_PRE: int, qkv_c: int, gate_c: int, interpret):
        half_h = H // 2
        chunk = H // tp
        o_in = chunk
        pre_c = H_PRE // tp
        post_c = chunk
        n_chips = tp // 2
        use_xy = b7._use_dual_tree_xy(tp, B, H)
        n_ar_sems = 4 if use_xy else (n_chips - 1)
        n_sc_slots = 2 if use_xy else (n_chips - 1)
        use_halfrow = B >= 32 and B % 32 == 0
        pre_buf_rows = B // 2 if use_halfrow else B
        reuse_pre_ab = H_PRE == half_h
        n_t3d = H_PRE // 128

        vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
        v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
        out_shapes = (
            jax.ShapeDtypeStruct((B, H), jnp.float32),
            jax.ShapeDtypeStruct((B, 256), jnp.float32),
            jax.ShapeDtypeStruct((B, n_t3d, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 256), jnp.float32),
            jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((B, gate_c), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
        )

        def _k(*refs):
            (
                st_h, ga_r, wo_h, swo_r, paa_r, pab_r, pfg_r, wpre_h, spre_r, wrt_r, pexp_r,
                moe_r, wpost_h, spost_r, pff_r, pfa_r, pfb_r, nin_r, wq_h, sq_r, wg_h, sg_r, cs_r,
                ost_h, olog_r, oexp3d_r, osnap_log_r, oq_r, okv_r, og_r, osnap_q_r,
                wo_b, wpre_b, wrt_b, wp_b, wq_b, wg_b, sb_v, ffn_v,
                send_h, mate_r, own_h, sc_r, own_c, sq_v, pre_sh, pre_a, pre_b,
                d_rtr, d_inv, d_xfp8, d_xsc,
                nin_v, qkv_v, gate_v, psh, ha, hb, d1, d2,
                b1_sems, ar1as, ar1ar, ar1s, ar1r, ar2s, ar2r, ar2bs, ar2br,
                ag1s, ag1r, ag2s, ag2r, ag3s, ag3r,
            ) = refs

            b7._run_rep_b1_vmem_core(
                st_h, ga_r, wo_h, swo_r, paa_r, pab_r, pfg_r, wpre_h, spre_r, wrt_r, pexp_r,
                ost_h, olog_r, None,
                wo_b, wpre_b, wrt_b, sb_v, d_rtr, d_inv, ffn_v,
                send_h, mate_r, own_h, sc_r, own_c, sq_v, pre_sh, pre_a, pre_b, d_xfp8, d_xsc,
                b1_sems.at[0:2], b1_sems.at[2:3], b1_sems.at[3:4], b1_sems.at[4:5], b1_sems.at[5:6],
                ar1as, ar1ar, ar1s, ar1r, ar2s, ar2r, ar2bs, ar2br,
                ag1s, ag1r, ag2s, ag2r, ag3s, ag3r,
                world_size=tp,
                pa_eps=1e-5,
                pre_ff_eps=1e-5,
                pre_ff_scale_input=True,
                router_multiplier=0.5,
                pre_exp_eps=1e-5,
                pre_exp_scale_input=False,
                use_w8a8=False,
                skip_barrier_signal=False,
                skip_barrier_wait=False,
                skip_stream_load=False,
                skip_stream_store=True,
                during_ag_exp_ici_fn=lambda: osnap_log_r.__setitem__(slice(None), olog_r[...]),
                out_expert_3d_ref=oexp3d_r,
            )

            pltpu.make_async_copy(wpost_h, wp_b, b1_sems.at[0]).start()
            pltpu.make_async_copy(wq_h, wq_b, b1_sems.at[1]).start()
            pltpu.make_async_copy(wg_h, wg_b, b1_sems.at[2]).start()
            me = lax.axis_index("model").astype(jnp.int32)
            half_b = B // 2
            my_r0 = pl.multiple_of((me & jnp.int32(1)) * half_b, 16)
            s_for_b2 = sb_v.at[pl.ds(my_r0, half_b), :] if use_halfrow else sb_v
            w8._run_rep_b2_vmem_core(
                ost_h, moe_r, wpost_h, spost_r, pff_r, pfa_r, pfb_r, nin_r, wq_h, sq_r, wg_h, sg_r, cs_r,
                ost_h, oq_r, okv_r, og_r,
                wp_b, wq_b, wg_b, s_for_b2, nin_v, qkv_v, psh, ha, hb, d1, d2,
                b1_sems.at[0:1], b1_sems.at[1:2], b1_sems.at[2:3], b1_sems.at[3:4], b1_sems.at[4:6],
                ar1as, ar1ar,
                ar1s.at[pl.ds(0, n_chips - 1)], ar1r.at[pl.ds(0, n_chips - 1)],
                ag3s, ag3r,
                ar1as, ar1ar, ar1as, ar1ar,
                world_size=tp,
                post_ff_eps=1e-5,
                post_ff_scale_input=True,
                next_in_eps=1e-5,
                next_in_scale_input=False,
                sharded_moe_in=False,
                qkv_norm_eps=1e-6,
                query_scale=0.125,
                k_scale=0.75,
                v_scale=1.25,
                has_rope=True,
                norm_before_s3=use_halfrow,
                interpret_mode=bool(interpret),
                skip_barrier_signal=True,
                skip_barrier_wait=True,
                skip_stream_load=True,
                skip_stream_store=False,
                prestarted_w_post_dma=True,
                prestarted_w_qkv_dma=True,
                prestarted_w_gate_dma=True,
                gate_dst_ref=gate_v,
                after_qkv_epi_fn=lambda: osnap_q_r.__setitem__(slice(None), oq_r[...]),
                defer_stream_store_wait=True,
            )
            og_r[...] = gate_v[...]
            if not use_halfrow:
                w8._dma_wait(ost_h, b1_sems.at[4])
            else:
                mate_r0 = pl.multiple_of((jnp.int32(1) - (me & jnp.int32(1))) * half_b, 16)
                w8._dma_wait(ost_h.at[pl.ds(my_r0, half_b)], b1_sems.at[4])
                w8._dma_wait(ost_h.at[pl.ds(mate_r0, half_b)], b1_sems.at[5])

        in_specs = (
            h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp,
            v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp, v_sp,
        )
        out_specs = (h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, v_sp, v_sp)
        scratch_shapes = (
            vmem((2, o_in, half_h), jnp.float8_e4m3fn),
            vmem((H, pre_c), jnp.float8_e4m3fn),
            vmem((H, 256), jnp.bfloat16),
            vmem((H_PRE, post_c), jnp.float8_e4m3fn),
            vmem((H, qkv_c), jnp.float8_e4m3fn),
            vmem((H, gate_c), jnp.float8_e4m3fn),
            vmem((B, H), jnp.float32),
            vmem((B, H), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((n_sc_slots, B, chunk), jnp.bfloat16),
            vmem((8, 128) if use_xy else (B, chunk), jnp.bfloat16),
            vmem((B, 128), jnp.float32),
            vmem((B, pre_c), jnp.bfloat16),
            vmem((8, 128) if reuse_pre_ab else (pre_buf_rows, H_PRE), jnp.bfloat16),
            vmem((8, 128) if (reuse_pre_ab or use_halfrow) else (pre_buf_rows, H_PRE), jnp.bfloat16),
            vmem((8, 128), jnp.float32),
            vmem((8, 1), jnp.float32),
            vmem((8, 128), jnp.float8_e4m3fn),
            vmem((8, 1), jnp.float32),
            vmem((B, H), jnp.bfloat16),
            vmem((B, qkv_c), jnp.bfloat16),
            vmem((B, gate_c), jnp.bfloat16),
            vmem((B, post_c), jnp.bfloat16),
            vmem((pre_buf_rows, H), jnp.bfloat16),
            vmem((pre_buf_rows, H), jnp.float32 if use_halfrow else jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            vmem((8, 128), jnp.bfloat16),
            sem((6,)),
            sem((1,)), sem((1,)),
            sem((n_ar_sems,)), sem((n_ar_sems,)),
            sem((n_ar_sems,)), sem((n_ar_sems,)),
            sem((1,)), sem((1,)),
            sem((1,)), sem((1,)),
            sem((n_chips - 1,)), sem((n_chips - 1,)),
            sem((2 if use_halfrow else 1,)), sem((2 if use_halfrow else 1,)),
        )
        return w8._pallas_call_maybe_interpret(
            _k,
            interpret=interpret,
            out_shape=out_shapes,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            compiler_params=pltpu.CompilerParams(
                collective_id=w8._W8_CID_BASE + 103,
                vmem_limit_bytes=60 * 1024**2,
                disable_bounds_checks=True,
            ),
        )

    def _call_chain(call, xs, *, B: int, H: int, H_PRE: int, pre_c: int, post_c: int, qkv_c: int, gate_c: int):
        r = call(
            xs[0], xs[1], xs[2], w8._as_dot_scale_f32(xs[3], H),
            w8._as_2d_f32(xs[4], H), w8._as_2d_f32(xs[5], H), w8._as_2d_f32(xs[6], H),
            xs[7], w8._as_dot_scale_f32(xs[8], pre_c), xs[9], w8._as_2d_f32(xs[10], H_PRE),
            xs[11], xs[12], w8._as_dot_scale_f32(xs[13], post_c),
            w8._as_2d_f32(xs[14], H), w8._as_2d_f32(xs[15], H), w8._as_2d_f32(xs[16], H), w8._as_2d_f32(xs[17], H),
            xs[18], w8._as_dot_scale_f32(xs[19], qkv_c), xs[20], w8._as_dot_scale_f32(xs[21], gate_c), xs[22],
        )
        return (r[0], r[1], r[2], r[3], r[4].reshape(1, B, 8, 2, 128), r[5], r[6], r[7].reshape(1, B, 8, 2, 128))

    in_p_specs = (
        P(), P(None, "model"), P("model", None), P(), P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P(None, "model"), P("model"), P(), P(), P(), P(),
        P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
    )
    out_p_specs = (
        P(), P(None, None), P(None, None, None), P(None, None),
        P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"),
        P(None, None, "model", None, None),
    )

    ip_onwait = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=False,
        uninitialized_memory="nan",
    )
    for tp, B, H, H_PRE, qkv_c, gate_c, ip_mode in (
        (8, 16, 4096, 2048, 1280, 1024, ip_race),
        (16, 16, 8192, 4096, 640, 512, ip_race),
        (16, 32, 4096, 4096, 640, 512, ip_race),
        (8, 64, 4096, 2048, 1280, 1024, ip_onwait),
    ):
        mesh = Mesh(np.array(devices[:tp]), ("model",))
        valid = B - 1
        pre_c = H_PRE // tp
        post_c = H // tp
        st_np = (rng.standard_normal((B, H)) * 0.4).astype(np.float32)
        st_np[-1, :] = np.nan
        ga_np = (rng.standard_normal((B, H)) * 0.3).astype(np.float32)
        ga_np[-1, :] = np.nan
        moe_np = (rng.standard_normal((B, H_PRE)) * 0.3).astype(np.float32)
        moe_np[-1, :] = np.nan
        st = jnp.asarray(st_np)
        ga = jnp.asarray(ga_np, dtype=jnp.bfloat16)
        moe = jnp.asarray(moe_np, dtype=jnp.bfloat16)
        wo = jnp.asarray((rng.standard_normal((H, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        swo = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
        paa, pab, gpre = jnp.full((H,), 0.91), jnp.full((H,), 0.87), jnp.full((H,), 1.04)
        wpre = jnp.asarray((rng.standard_normal((H, H_PRE)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spre = jnp.asarray((np.abs(rng.standard_normal((H_PRE,))) * 0.02 + 1e-3).astype(np.float32))
        wrt = jnp.asarray((rng.standard_normal((H, 256)) * 0.03).astype(np.float32), dtype=jnp.bfloat16)
        gexp = jnp.full((H_PRE,), 0.96)
        wpost = jnp.asarray((rng.standard_normal((H_PRE, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        spost = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
        wq = jnp.asarray((rng.standard_normal((H, tp * qkv_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        sq = jnp.asarray((np.abs(rng.standard_normal((tp * qkv_c,))) * 0.02 + 1e-3).astype(np.float32))
        wg = jnp.asarray((rng.standard_normal((H, tp * gate_c)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
        sg = jnp.asarray((np.abs(rng.standard_normal((tp * gate_c,))) * 0.02 + 1e-3).astype(np.float32))
        cs = build_decode_rope_step_tables(
            jnp.arange(B, dtype=jnp.int32),
            jnp.asarray(rng.standard_normal((B, 128)).astype(np.float32) * 0.5),
        )

        ref_st1, ref_log1, ref_exp1 = jax.jit(jax.shard_map(
            lambda *a: b7.attn_to_moe_bridge_jax(
                *a, world_size=tp, pa_eps=1e-5, pre_ff_eps=1e-5, pre_ff_scale_input=True,
                router_multiplier=0.5, pre_exp_eps=1e-5, pre_exp_scale_input=False, interpret=ip_mode,
            ),
            mesh=mesh,
            in_specs=(P(), P(None, "model"), P("model", None), P(), P(), P(), P(), P(None, "model"), P("model"), P(), P()),
            out_specs=(P(), P(None, None), P(None, None)),
            check_vma=False,
        ))(st, ga, wo, swo, paa, pab, gpre, wpre, spre, wrt, gexp)

        use_halfrow = B >= 32 and B % 32 == 0
        ref_b2 = jax.jit(jax.shard_map(
            lambda *a: w8.moe_to_attn_qkv_epi_bridge_jax(
                *a, world_size=tp, post_ff_eps=1e-5, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=False, sharded_moe_in=False,
                qkv_norm_eps=1e-6, query_scale=0.125, k_scale=0.75, v_scale=1.25,
                norm_before_s3=use_halfrow, interpret=ip_mode,
            ),
            mesh=mesh,
            in_specs=(
                P(), P(), P(None, "model"), P("model"), P(), P(), P(), P(),
                P(None, "model"), P("model"), P(None, "model"), P("model"), P(),
            ),
            out_specs=(P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
            check_vma=False,
        ))(ref_st1, moe, wpost, spost, gpre, paa, pab, gpre, wq, sq, wg, sg, cs)

        call = _build_chain_call(tp=tp, B=B, H=H, H_PRE=H_PRE, qkv_c=qkv_c, gate_c=gate_c, interpret=ip_mode)
        got = jax.jit(jax.shard_map(
            lambda *xs: _call_chain(call, xs, B=B, H=H, H_PRE=H_PRE, pre_c=pre_c, post_c=post_c, qkv_c=qkv_c, gate_c=gate_c),
            mesh=mesh, in_specs=in_p_specs, out_specs=out_p_specs, check_vma=False,
        ))(st, ga, wo, swo, paa, pab, gpre, wpre, spre, wrt, gexp, moe, wpost, spost, gpre, paa, pab, gpre, wq, sq, wg, sg, cs)
        jax.block_until_ready(got[0])
        got_st, got_log, got_exp3d, snap_log, got_q, got_kv, got_g, snap_q = got
        np.testing.assert_array_equal(np.asarray(got_log)[:valid], np.asarray(ref_log1)[:valid])
        np.testing.assert_array_equal(np.asarray(snap_log)[:valid], np.asarray(ref_log1)[:valid])
        np.testing.assert_array_equal(np.asarray(got_exp3d).reshape(B, H_PRE)[:valid], np.asarray(ref_exp1)[:valid])
        np.testing.assert_array_equal(np.asarray(got_st)[:valid], np.asarray(ref_b2[0])[:valid])
        np.testing.assert_array_equal(np.asarray(got_q)[:, :valid], np.asarray(ref_b2[1])[:, :valid])
        np.testing.assert_array_equal(np.asarray(snap_q)[:, :valid], np.asarray(ref_b2[1])[:, :valid])
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(got_kv, jnp.uint8))[:valid],
            np.asarray(jax.lax.bitcast_convert_type(ref_b2[2], jnp.uint8))[:valid],
        )
        np.testing.assert_array_equal(np.asarray(got_g)[:valid], np.asarray(ref_b2[3])[:valid])

    # Offline TPU7x AOT LLO verification on production (H=8192, H_PRE=4096) for B=16 TP=8/16 & B=64 TP=8.
    for tp, topo, B, qkv_c, gate_c in (
        (8, "tpu7x=2x2x1", 16, 1280, 1024),
        (16, "tpu7x=2x2x2", 16, 640, 512),
        (8, "tpu7x=2x2x1", 64, 1280, 1024),
    ):
        H, H_PRE = 8192, 4096
        pre_c, post_c = H_PRE // tp, H // tp
        aot_mesh = Mesh(
            topologies.get_topology_desc(platform="tpu", topology_name=topo).devices,
            ("model",),
        )
        S = lambda sh, dt, sp: jax.ShapeDtypeStruct(sh, dt, sharding=NamedSharding(aot_mesh, sp))
        sds = (
            S((B, H), jnp.float32, P()),
            S((B, H), jnp.bfloat16, P(None, "model")),
            S((H, H), jnp.float8_e4m3fn, P("model", None)),
            S((H,), jnp.float32, P()),
            S((H,), jnp.float32, P()),
            S((H,), jnp.float32, P()),
            S((H,), jnp.float32, P()),
            S((H, H_PRE), jnp.float8_e4m3fn, P(None, "model")),
            S((H_PRE,), jnp.float32, P("model")),
            S((H, 256), jnp.bfloat16, P()),
            S((H_PRE,), jnp.float32, P()),
            S((B, H_PRE), jnp.bfloat16, P()),
            S((H_PRE, H), jnp.float8_e4m3fn, P(None, "model")),
            S((H,), jnp.float32, P("model")),
            S((H,), jnp.float32, P()),
            S((H,), jnp.float32, P()),
            S((H,), jnp.float32, P()),
            S((H,), jnp.float32, P()),
            S((H, tp * qkv_c), jnp.float8_e4m3fn, P(None, "model")),
            S((tp * qkv_c,), jnp.float32, P("model")),
            S((H, tp * gate_c), jnp.float8_e4m3fn, P(None, "model")),
            S((tp * gate_c,), jnp.float32, P("model")),
            S((B, 2, 128), jnp.float32, P()),
        )
        aot_call = _build_chain_call(tp=tp, B=B, H=H, H_PRE=H_PRE, qkv_c=qkv_c, gate_c=gate_c, interpret=False)
        compiled = jax.jit(jax.shard_map(
            lambda *xs: _call_chain(aot_call, xs, B=B, H=H, H_PRE=H_PRE, pre_c=pre_c, post_c=post_c, qkv_c=qkv_c, gate_c=gate_c),
            mesh=aot_mesh, in_specs=in_p_specs, out_specs=out_p_specs, check_vma=False,
        )).lower(*sds).compile()
        assert compiled.memory_analysis().generated_code_size_in_bytes > 0


def test_tp16_b64_prs_and_bug_c_oneshot_interpret_and_aot():
    """Verify Bug C 1-shot + opt-in TP=16 B=64 PRS (0 races, eager interpret, Step (b) AOT)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.envs import envs
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import _E_TOTAL, _FP4_PACK, _H, _I

    devices = jax.devices()
    if len(devices) < 16:
        pytest.skip(f"Need >=16 host devices, got {len(devices)}")

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(2026112602)

    # 1. Verify Bug C 1-shot + PRS parity & during_ag_exp_d2d2_fn on TP=16 B=64 (2D & 3D)
    tp, B, H, H_PRE = 16, 64, 2048, 4096
    valid = B - 1
    r_pair = B // (tp // 2)
    n_chips = tp // 2
    g_dim = H // tp
    pre_pair_cols = H_PRE // 2
    half_h = H // 2
    mesh = Mesh(np.array(devices[:tp]), ("model",))

    st_in_np = (rng.standard_normal((B, H)) * 0.4).astype(np.float32)
    st_in_np[-1, :] = np.nan
    st_stacked = jnp.asarray(
        np.concatenate([st_in_np[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(tp)], axis=0)
    )
    gated_np = (rng.standard_normal((B, H)) * 0.3).astype(np.float32)
    gated_np[-1, :] = np.nan
    gated = jnp.asarray(gated_np, dtype=jnp.bfloat16)
    wo = jnp.asarray((rng.standard_normal((H, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    s_wo = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
    pa_a = jnp.full((H,), 0.91, jnp.float32)
    pa_b = jnp.full((H,), 0.87, jnp.float32)
    g_pre = jnp.full((H,), 1.04, jnp.float32)
    g_exp = jnp.full((H_PRE,), 0.96, jnp.float32)
    w_pre_base = jnp.asarray((rng.standard_normal((H, H_PRE)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    s_pre_base = (np.abs(rng.standard_normal((H_PRE,))) * 0.02 + 1e-3).astype(np.float32)
    w_pre = jnp.concatenate(
        [w_pre_base[:, :pre_pair_cols] if (r & 1) == 0 else w_pre_base[:, pre_pair_cols:] for r in range(tp)],
        axis=1,
    )
    s_pre = jnp.asarray(
        np.concatenate(
            [s_pre_base[:pre_pair_cols] if (r & 1) == 0 else s_pre_base[pre_pair_cols:] for r in range(tp)],
            axis=0,
        )
    )
    w_rt = jnp.asarray((rng.standard_normal((H, 256)) * 0.03).astype(np.float32), dtype=jnp.bfloat16)

    def _run_standalone(parallel_rs: bool, bug_c_oneshot: bool):
        return jax.jit(
            jax.shard_map(
                lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
                    *a,
                    world_size=tp,
                    pa_eps=1e-8,
                    pre_ff_eps=1e-5,
                    pre_ff_scale_input=True,
                    router_multiplier=0.5,
                    pre_exp_eps=1e-5,
                    pre_exp_scale_input=False,
                    parallel_rs=parallel_rs,
                    bug_c_oneshot=bug_c_oneshot,
                    interpret=ip_race,
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None), P(None, "model"), P("model", None),
                    P(), P(), P(), P(), P(None, "model"), P("model"), P(), P(),
                ),
                out_specs=(P("model", None), P(None, None), P(None, None)),
                check_vma=False,
            )
        )(st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp)

    st_ser, log_ser, exp_ser = _run_standalone(False, False)
    st_bc1s, log_bc1s, exp_bc1s = _run_standalone(False, True)
    st_prs, log_prs, exp_prs = _run_standalone(True, True)

    np.testing.assert_array_equal(
        np.asarray(st_bc1s).reshape(tp, r_pair, H)[:, :7],
        np.asarray(st_ser).reshape(tp, r_pair, H)[:, :7],
    )
    np.testing.assert_array_equal(np.asarray(log_bc1s)[:valid], np.asarray(log_ser)[:valid])
    np.testing.assert_array_equal(np.asarray(exp_bc1s)[:valid], np.asarray(exp_ser)[:valid])

    st_ser_np = np.asarray(st_ser).reshape(tp, r_pair, H)
    st_prs_np = np.asarray(st_prs).reshape(tp, r_pair, H)
    for c in range(n_chips):
        r_lim = min(r_pair, max(0, valid - c * r_pair))
        np.testing.assert_array_equal(st_prs_np[2 * c, :r_lim], st_prs_np[2 * c + 1, :r_lim])
    st_ser_reasm = np.concatenate([st_ser_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
    st_prs_reasm = np.concatenate([st_prs_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
    exp_ser_f = np.asarray(exp_ser, dtype=np.float32)[:valid]
    exp_prs_f = np.asarray(exp_prs, dtype=np.float32)[:valid]
    assert np.isfinite(st_prs_reasm).all()
    assert np.isfinite(exp_prs_f).all()
    np.testing.assert_allclose(st_prs_reasm, st_ser_reasm, rtol=3e-2, atol=8e-2)
    np.testing.assert_allclose(exp_prs_f, exp_ser_f, rtol=3e-2, atol=8e-2)
    st_cos = float(np.sum(st_prs_reasm * st_ser_reasm) / (np.linalg.norm(st_prs_reasm) * np.linalg.norm(st_ser_reasm)))
    exp_cos = float(np.sum(exp_prs_f * exp_ser_f) / (np.linalg.norm(exp_prs_f) * np.linalg.norm(exp_ser_f)))
    assert st_cos >= 0.99999
    assert exp_cos >= 0.99999

    # Verify _run_chip_b1_vmem_core with 3D output + during_ag_exp_d2d2_fn + b1_moe_b2 sem aliasing + env overrides!
    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
    four_pair = 4 * r_pair
    two_pair = 2 * r_pair

    def _core_kernel(*refs):
        (
            st_hbm, g_ref, wo_hbm, swo_ref, a_ref, b_ref, gpre_ref, wpre_hbm, spre_ref, wrt_ref, gexp_ref,
            out_st_hbm, out_log_ref, out_exp3d_ref, out_snap_ref,
            wo_buf, wpre_buf, wrt_buf, s_buf, send_half, mate_recv, own_half,
            r1a, r1b, r2a, r2b, my_p, recv_p, sq_pa, ffn_p, rt_p, pre_my, pre_mate, sq_pre, exp_3d,
            sems_local, d2d_s, d2d_r, rs1_s, rs1_r, ici_s, ici_r, log_s, log_r,
        ) = refs
        w8._run_chip_b1_vmem_core(
            st_hbm, g_ref, wo_hbm, swo_ref, a_ref, b_ref, gpre_ref,
            wpre_hbm, spre_ref, wrt_ref, gexp_ref,
            out_st_hbm, out_log_ref, None,
            wo_buf, wpre_buf, wrt_buf, s_buf,
            send_half, mate_recv, own_half,
            r1a, r1b, r2a, r2b, my_p, recv_p, sq_pa, ffn_p, rt_p, pre_my, pre_mate, sq_pre, exp_3d,
            sems_local.at[0:2], sems_local.at[2:3], sems_local.at[3:4],
            sems_local.at[4:5], sems_local.at[5:6],
            d2d_s.at[0:1], d2d_r.at[0:1],
            rs1_s, rs1_r, rs1_s, rs1_r, rs1_s, rs1_r,
            d2d_s.at[0:1], d2d_r.at[0:1], rs1_s, rs1_r,
            ici_s, ici_r, d2d_s.at[0:1], d2d_r.at[0:1], log_s, log_r,
            world_size=tp,
            pa_eps=1e-8,
            pre_ff_eps=1e-5,
            pre_ff_scale_input=True,
            router_multiplier=0.5,
            pre_exp_eps=1e-5,
            pre_exp_scale_input=False,
            early_router_and_log_ag=True,
            during_ag_exp_d2d2_fn=lambda: out_snap_ref.__setitem__(Ellipsis, out_log_ref[...]),
            out_expert_3d_ref=out_exp3d_ref,
        )

    core_call = w8._pallas_call_maybe_interpret(
        _core_kernel,
        interpret=ip_race,
        out_shape=(
            jax.ShapeDtypeStruct((r_pair, H), jnp.float32),
            jax.ShapeDtypeStruct((B, 256), jnp.float32),
            jax.ShapeDtypeStruct((B, 32, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, 256), jnp.float32),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((2, g_dim, half_h), jnp.float8_e4m3fn),
            vmem((H, pre_pair_cols), jnp.float8_e4m3fn),
            vmem((H, 256), jnp.bfloat16),
            vmem((r_pair, H), jnp.float32),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((B, half_h), jnp.bfloat16),
            vmem((four_pair, half_h), jnp.bfloat16),
            vmem((two_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, half_h), jnp.bfloat16),
            vmem((r_pair, 128), jnp.float32),
            vmem((r_pair, H), jnp.bfloat16),
            vmem((8, 128), jnp.float32),
            vmem((r_pair, pre_pair_cols), jnp.bfloat16),
            vmem((r_pair, 128), jnp.float32),
            vmem((r_pair, 128), jnp.float32),
            vmem((n_chips, 8, H_PRE), jnp.bfloat16),
            sem((6,)), sem((1,)), sem((1,)),
            sem((1,)), sem((1,)),
            sem((n_chips - 1,)), sem((n_chips - 1,)),
            sem((n_chips - 1,)), sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=w8._W8_CID_BASE + 109, disable_bounds_checks=True),
    )

    with (
        envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(True),
        envs.SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT.override(True),
    ):
        got_st, got_log, got_exp3d, got_snap = jax.jit(
            jax.shard_map(
                lambda st, ga, w_o, swo, a, b, gp, wp, sp, wr, ge: core_call(
                    st,
                    ga,
                    w_o,
                    w8._as_dot_scale_f32(swo, H),
                    w8._as_2d_f32(a, H),
                    w8._as_2d_f32(b, H),
                    w8._as_2d_f32(gp, H),
                    wp,
                    w8._as_dot_scale_f32(sp, pre_pair_cols),
                    wr,
                    w8._as_2d_f32(ge, H_PRE),
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None), P(None, "model"), P("model", None),
                    P(), P(), P(), P(), P(None, "model"), P("model"), P(), P(),
                ),
                out_specs=(P("model", None), P(None, None), P(None, None, None), P(None, None)),
                check_vma=False,
            )
        )(st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp)
    got_st_np = np.asarray(got_st).reshape(tp, r_pair, H)
    got_st_reasm = np.concatenate([got_st_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
    np.testing.assert_array_equal(got_st_reasm, st_prs_reasm)
    np.testing.assert_array_equal(np.asarray(got_log)[:valid], np.asarray(log_prs)[:valid])
    np.testing.assert_array_equal(np.asarray(got_snap)[:valid], np.asarray(got_log)[:valid])
    np.testing.assert_array_equal(
        np.asarray(got_exp3d).reshape(B, H_PRE)[:valid],
        np.asarray(exp_prs)[:valid],
    )

    # 2. Verify op-cache isolation when SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS / BUG_C_ONESHOT flip
    seen_names: list[str] = []

    def _rec_export(name, fn, *, mesh=None, input_partition_specs=None, output_partition_specs=None):
        seen_names.append(name)
        return fn

    with (
        mock.patch.object(w8, "_get_tp_mesh", return_value=object()),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_rec_export),
        mock.patch.dict(w8._ATTN_TO_MOE_CHIP_SH_OPS, {}, clear=True),
    ):
        w8.get_attn_to_moe_chip_sharded_bridge_op(16, bf16_router=True)
        with envs.SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT.override(True):
            w8.get_attn_to_moe_chip_sharded_bridge_op(16, bf16_router=True)
        with (
            envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(True),
            envs.SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT.override(True),
        ):
            w8.get_attn_to_moe_chip_sharded_bridge_op(16, bf16_router=True)
    assert seen_names == [
        "sglang_tpu_pallas::muse_w9_chip_attn_to_moe_r16_tp16",
        "sglang_tpu_pallas::muse_w9_chip_attn_to_moe_r16_tp16_bc1s",
        "sglang_tpu_pallas::muse_w9_chip_attn_to_moe_r16_tp16_prs_bc1s",
    ]

    # 3. Offline TPU7x 2x2x2 AOT compilation at exact production B=64 TP=16 H=8192 H_PRE=4096!
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import _E_TOTAL, _H, _I
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
    )

    tpu_devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x2").devices
    aot_mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))),
        ("model",),
    )
    prod_H, prod_HPRE, prod_I = 8192, 4096, 4096
    b1_local = (
        jax.ShapeDtypeStruct((8, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((B, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((512, prod_H), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((prod_H, prod_HPRE // 2), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, prod_HPRE // 2), jnp.float32),
        jax.ShapeDtypeStruct((prod_H, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, prod_HPRE), jnp.float32),
    )
    b1_in_sp = tuple(P() for _ in b1_local)
    for prs_flag, bc_flag in ((False, False), (False, True), (True, True)):
        sm_b1 = jax.jit(
            jax.shard_map(
                lambda *a, _p=prs_flag, _b=bc_flag: w8.attn_to_moe_chip_sharded_bridge_jax(
                    *a, world_size=16, parallel_rs=_p, bug_c_oneshot=_b, interpret=False
                ),
                mesh=aot_mesh,
                in_specs=b1_in_sp,
                out_specs=(P(), P(), P()),
                check_vma=False,
            )
        )
        c_b1 = sm_b1.trace(*b1_local).lower(lowering_platforms=("tpu",)).compile()
        assert "tpu_custom_call" in c_b1.as_text()
        assert c_b1.memory_analysis().generated_code_size_in_bytes > 0

    e_loc = _E_TOTAL // tp
    nqb = _H // 256
    stepb_shapes = (
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((8, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((B, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((512, prod_H), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((prod_H, prod_HPRE // 2), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, prod_HPRE // 2), jnp.float32),
        jax.ShapeDtypeStruct((prod_H, _E_TOTAL), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, prod_HPRE), jnp.float32),
        jax.ShapeDtypeStruct((1, _E_TOTAL), jnp.float32),
        jax.ShapeDtypeStruct((e_loc, prod_HPRE, 2 * prod_I), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((e_loc, nqb, 2 * prod_I), jnp.bfloat16),
        jax.ShapeDtypeStruct((e_loc, prod_I, prod_HPRE), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((e_loc, _I // 256, prod_HPRE), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, prod_HPRE), jnp.bfloat16),
        jax.ShapeDtypeStruct((prod_HPRE, prod_HPRE), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, prod_HPRE), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
        jax.ShapeDtypeStruct((prod_H, 640), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, 640), jnp.float32),
        jax.ShapeDtypeStruct((prod_H, 512), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, 512), jnp.float32),
        jax.ShapeDtypeStruct((B, 2, 128), jnp.float32),
    )
    stepb_sp = tuple(P() for _ in stepb_shapes)
    for prs_flag, bc_flag in ((False, False), (False, True), (True, True)):
        with (
            envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(prs_flag),
            envs.SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT.override(bc_flag),
        ):
            sm_stepb = jax.jit(
                jax.shard_map(
                    lambda *a: b1_moe_b2_chip_sharded_jax(*a, world_size=16, fuse_qkv_epi=True, interpret=False),
                    mesh=aot_mesh,
                    in_specs=stepb_sp,
                    out_specs=(P(), P(), P(), P()),
                    check_vma=False,
                )
            )
            c_stepb = sm_stepb.trace(*stepb_shapes).lower(lowering_platforms=("tpu",)).compile()
        assert "tpu_custom_call" in c_stepb.as_text()
        assert c_stepb.memory_analysis().generated_code_size_in_bytes > 0


def test_tp16_parallel_xyz_rs_large_batch_interpret_and_aot():
    """Verify generalized TP=16 parallel XYZ RS across r_pair in (16, 32, 48) (B=128/256/384)."""
    import jax

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.envs import envs
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import _E_TOTAL, _H, _I
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
    )

    devices = jax.devices("cpu")
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    tp = 16
    n_chips = tp // 2
    mesh = Mesh(np.array(devices[:tp]), ("model",))
    rng = np.random.default_rng(20261127)
    H, H_PRE = 1024, 512
    g_dim = H // tp
    pre_pair_cols = H_PRE // 2

    wo = jnp.asarray((rng.standard_normal((H, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    s_wo = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
    pa_a = jnp.full((H,), 0.91, jnp.float32)
    pa_b = jnp.full((H,), 0.87, jnp.float32)
    g_pre = jnp.full((H,), 1.04, jnp.float32)
    g_exp = jnp.full((H_PRE,), 0.96, jnp.float32)
    w_pre_base = jnp.asarray((rng.standard_normal((H, H_PRE)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn)
    s_pre_base = (np.abs(rng.standard_normal((H_PRE,))) * 0.02 + 1e-3).astype(np.float32)
    w_pre = jnp.concatenate(
        [w_pre_base[:, :pre_pair_cols] if (r & 1) == 0 else w_pre_base[:, pre_pair_cols:] for r in range(tp)],
        axis=1,
    )
    s_pre = jnp.asarray(
        np.concatenate(
            [s_pre_base[:pre_pair_cols] if (r & 1) == 0 else s_pre_base[pre_pair_cols:] for r in range(tp)],
            axis=0,
        )
    )
    w_rt = jnp.asarray((rng.standard_normal((H, 256)) * 0.03).astype(np.float32), dtype=jnp.bfloat16)

    assert w8._tp16_parallel_rs_enabled(r_pair=8) is False
    assert w8._tp16_parallel_rs_enabled(r_pair=24) is False
    assert w8._tp16_parallel_rs_enabled(r_pair=40) is False
    for r_p in (16, 32, 48):
        assert w8._tp16_parallel_rs_enabled(r_pair=r_p) is True
    with envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(False):
        for r_p in (8, 16, 24, 32, 48):
            assert w8._tp16_parallel_rs_enabled(r_pair=r_p) is False
    with envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(True):
        for r_p in (8, 16, 24, 32, 48):
            assert w8._tp16_parallel_rs_enabled(r_pair=r_p) is True

    # Negative guard for B=192 (r_pair=24 % 16 != 0): wrapper rejects invalid batch, and use_prs rejects r_pair % 16 != 0.
    assert "r_pair % 16 == 0" in inspect.getsource(w8._run_chip_b1_vmem_core)
    b192_in = (rng.standard_normal((192, H)) * 0.3).astype(np.float32)
    b192_st = jnp.asarray(np.concatenate([b192_in[(r >> 1) * 24 : ((r >> 1) + 1) * 24] for r in range(tp)], axis=0))
    b192_ga = jnp.asarray(rng.standard_normal((192, H)) * 0.2, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="Unsupported chip-sharded Bridge 1b shapes"):
        jax.shard_map(
            lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
                *a, world_size=tp, router_multiplier=0.5, interpret=ip_race
            ),
            mesh=mesh,
            in_specs=(
                P("model", None), P(None, "model"), P("model", None),
                P(), P(), P(), P(), P(None, "model"), P("model"), P(), P(),
            ),
            out_specs=(P("model", None), P(None, None), P(None, None)),
            check_vma=False,
        )(b192_st, b192_ga, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp)

    # Verify standalone op passes tri-state parallel_rs (None when unset, False when =0, True when =1).
    import contextlib

    seen_prs: list[tuple[str, object]] = []

    def _cap_export(name, fn, *, mesh=None, input_partition_specs=None, output_partition_specs=None):
        def _invoke(*args):
            return fn(*args)

        seen_prs.append((name, _invoke))
        return _invoke

    with (
        mock.patch.object(w8, "_get_tp_mesh", return_value=mesh),
        mock.patch("jax.shard_map", side_effect=lambda f, **kw: f),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_cap_export),
        mock.patch.object(w8, "attn_to_moe_chip_sharded_bridge_jax", return_value=("s", "l", "e")) as mock_b1,
        mock.patch.dict(w8._ATTN_TO_MOE_CHIP_SH_OPS, {}, clear=True),
    ):
        for ctx in (contextlib.nullcontext(), envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(False), envs.SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS.override(True)):
            with ctx:
                fn_op = w8.get_attn_to_moe_chip_sharded_bridge_op(16, bf16_router=True)
                fn_op(None, None, None, None, None, None, None, None, None, None, None, 1e-8, 1e-5, False, 1.0, 1e-5, False)
    assert [n for n, _ in seen_prs] == [
        "sglang_tpu_pallas::muse_w9_chip_attn_to_moe_r16_tp16",
        "sglang_tpu_pallas::muse_w9_chip_attn_to_moe_r16_tp16_prs0",
        "sglang_tpu_pallas::muse_w9_chip_attn_to_moe_r16_tp16_prs",
    ]
    assert [c.kwargs["parallel_rs"] for c in mock_b1.call_args_list] == [None, False, True]

    for B_cur in (128, 256, 384):
        r_pair = B_cur // n_chips
        valid = B_cur - 1
        st_in_np = (rng.standard_normal((B_cur, H)) * 0.4).astype(np.float32)
        st_in_np[-1, :] = np.nan
        st_stacked = jnp.asarray(
            np.concatenate([st_in_np[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(tp)], axis=0)
        )
        gated_np = (rng.standard_normal((B_cur, H)) * 0.3).astype(np.float32)
        gated_np[-1, :] = np.nan
        gated = jnp.asarray(gated_np, dtype=jnp.bfloat16)

        def _run(parallel_rs: bool | None):
            return jax.jit(
                jax.shard_map(
                    lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
                        *a,
                        world_size=tp,
                        pa_eps=1e-8,
                        pre_ff_eps=1e-5,
                        pre_ff_scale_input=True,
                        router_multiplier=0.5,
                        pre_exp_eps=1e-5,
                        pre_exp_scale_input=False,
                        parallel_rs=parallel_rs,
                        interpret=ip_race,
                    ),
                    mesh=mesh,
                    in_specs=(
                        P("model", None), P(None, "model"), P("model", None),
                        P(), P(), P(), P(), P(None, "model"), P("model"), P(), P(),
                    ),
                    out_specs=(P("model", None), P(None, None), P(None, None)),
                    check_vma=False,
                )
            )(st_stacked, gated, wo, s_wo, pa_a, pa_b, g_pre, w_pre, s_pre, w_rt, g_exp)

        st_ser, log_ser, exp_ser = _run(False)
        st_prs, log_prs, exp_prs = _run(True)
        if B_cur == 128:
            st_def, log_def, exp_def = _run(None)
            st_def_reasm = np.concatenate(
                [np.asarray(st_def).reshape(tp, r_pair, H)[2 * c] for c in range(n_chips)], axis=0
            )[:valid]
            st_prs_chk = np.concatenate(
                [np.asarray(st_prs).reshape(tp, r_pair, H)[2 * c] for c in range(n_chips)], axis=0
            )[:valid]
            np.testing.assert_array_equal(st_def_reasm, st_prs_chk)
            np.testing.assert_array_equal(np.asarray(log_def)[:valid], np.asarray(log_prs)[:valid])
            np.testing.assert_array_equal(np.asarray(exp_def)[:valid], np.asarray(exp_prs)[:valid])

        st_ser_np = np.asarray(st_ser).reshape(tp, r_pair, H)
        st_prs_np = np.asarray(st_prs).reshape(tp, r_pair, H)
        for c in range(n_chips):
            r_lim = min(r_pair, max(0, valid - c * r_pair))
            np.testing.assert_array_equal(st_prs_np[2 * c, :r_lim], st_prs_np[2 * c + 1, :r_lim])
        st_ser_reasm = np.concatenate([st_ser_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
        st_prs_reasm = np.concatenate([st_prs_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
        exp_ser_f = np.asarray(exp_ser, dtype=np.float32)[:valid]
        exp_prs_f = np.asarray(exp_prs, dtype=np.float32)[:valid]
        assert np.isfinite(st_prs_reasm).all()
        assert np.isfinite(exp_prs_f).all()
        assert np.isfinite(np.asarray(log_prs)[:valid]).all()
        np.testing.assert_allclose(st_prs_reasm, st_ser_reasm, rtol=3e-2, atol=8e-2)
        np.testing.assert_allclose(exp_prs_f, exp_ser_f, rtol=3e-2, atol=8e-2)
        np.testing.assert_allclose(
            np.asarray(log_prs)[:valid], np.asarray(log_ser)[:valid], rtol=3e-2, atol=8e-2
        )
        st_cos = float(np.sum(st_prs_reasm * st_ser_reasm) / (np.linalg.norm(st_prs_reasm) * np.linalg.norm(st_ser_reasm)))
        exp_cos = float(np.sum(exp_prs_f * exp_ser_f) / (np.linalg.norm(exp_prs_f) * np.linalg.norm(exp_ser_f)))
        assert st_cos >= 0.99999
        assert exp_cos >= 0.99999

    tpu_devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x2").devices
    aot_mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))),
        ("model",),
    )
    prod_H, prod_HPRE, prod_I = 8192, 4096, 4096
    for B_cur in (128, 256, 384):
        r_pair = B_cur // n_chips
        b1_local = (
            jax.ShapeDtypeStruct((r_pair, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((B_cur, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((512, prod_H), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, prod_HPRE // 2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, prod_HPRE // 2), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, prod_HPRE), jnp.float32),
        )
        c_b1 = jax.jit(
            jax.shard_map(
                lambda *a: w8.attn_to_moe_chip_sharded_bridge_jax(
                    *a, world_size=16, parallel_rs=True, interpret=False
                ),
                mesh=aot_mesh,
                in_specs=tuple(P() for _ in b1_local),
                out_specs=(P(), P(), P()),
                check_vma=False,
            )
        ).trace(*b1_local).lower(lowering_platforms=("tpu",)).compile()
        assert "tpu_custom_call" in c_b1.as_text()

    e_loc = _E_TOTAL // tp
    nqb = _H // 256
    for B_cur in (128, 256):
        r_pair = B_cur // n_chips
        stepb_shapes = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((r_pair, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((B_cur, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((512, prod_H), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, prod_HPRE // 2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, prod_HPRE // 2), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, _E_TOTAL), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, prod_HPRE), jnp.float32),
            jax.ShapeDtypeStruct((1, _E_TOTAL), jnp.float32),
            jax.ShapeDtypeStruct((e_loc, prod_HPRE, 2 * prod_I), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((e_loc, nqb, 2 * prod_I), jnp.bfloat16),
            jax.ShapeDtypeStruct((e_loc, prod_I, prod_HPRE), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((e_loc, _I // 256, prod_HPRE), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, prod_HPRE), jnp.bfloat16),
            jax.ShapeDtypeStruct((prod_HPRE, prod_HPRE), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, prod_HPRE), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, 640), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 640), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, 512), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 512), jnp.float32),
            jax.ShapeDtypeStruct((B_cur, 2, 128), jnp.float32),
        )
        c_stepb = jax.jit(
            jax.shard_map(
                lambda *a: b1_moe_b2_chip_sharded_jax(*a, world_size=16, fuse_qkv_epi=True, interpret=False),
                mesh=aot_mesh,
                in_specs=tuple(P() for _ in stepb_shapes),
                out_specs=(P(), P(), P(), P()),
                check_vma=False,
            )
        ).trace(*stepb_shapes).lower(lowering_platforms=("tpu",)).compile()
        assert "tpu_custom_call" in c_stepb.as_text()


def test_tp16_chip_b2_large_batch_post_ici_zero_race_and_aot():
    """Verify TP=16 chip_b2 7-peer post_ici across B=64/128/256/384 under detect_races & 2x2x2 AOT."""
    import jax

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    TP = 16
    n_chips = TP // 2
    mesh = Mesh(np.array(jax.devices("cpu")[:TP]), ("model",))
    rng = np.random.default_rng(2026112702)
    H, H_PRE = 2048, 512
    qkv_cols, gate_cols = 640, 512

    wpost_full = jnp.asarray(
        (rng.standard_normal((H_PRE, H)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn
    )
    spost_full = jnp.asarray((np.abs(rng.standard_normal((H,))) * 0.02 + 1e-3).astype(np.float32))
    wpost_tp2 = jnp.concatenate(
        [wpost_full[:, : H // 2] if (r & 1) == 0 else wpost_full[:, H // 2 :] for r in range(TP)],
        axis=1,
    )
    spost_tp2 = jnp.concatenate(
        [spost_full[: H // 2] if (r & 1) == 0 else spost_full[H // 2 :] for r in range(TP)],
        axis=0,
    )
    pff_g = jnp.asarray((np.ones((H,)) * 1.03).astype(np.float32))
    pf_a = jnp.asarray((np.ones((H,)) * 0.92).astype(np.float32))
    pf_b = jnp.asarray((np.ones((H,)) * 0.88).astype(np.float32))
    nin_g = jnp.asarray((np.ones((H,)) * 1.07).astype(np.float32))
    wq = jnp.asarray(
        (rng.standard_normal((H, TP * qkv_cols)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn
    )
    sq = jnp.asarray((np.abs(rng.standard_normal((TP * qkv_cols,))) * 0.02 + 1e-3).astype(np.float32))
    wg = jnp.asarray(
        (rng.standard_normal((H, TP * gate_cols)) * 0.04).astype(np.float32), dtype=jnp.float8_e4m3fn
    )
    sg = jnp.asarray((np.abs(rng.standard_normal((TP * gate_cols,))) * 0.02 + 1e-3).astype(np.float32))
    cs_pairs = jnp.asarray(rng.standard_normal((512, 128)).astype(np.float32) * 0.5)

    for B_cur, sh_in in ((64, False), (128, True), (256, True), (384, True)):
        r_pair = B_cur // n_chips
        valid = B_cur - 1
        st_in = (rng.standard_normal((B_cur, H)) * 0.5).astype(np.float32)
        st_in[-1, :] = np.nan
        st_stacked = jnp.asarray(
            np.concatenate(
                [st_in[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(TP)], axis=0
            )
        )
        moe_np = (rng.standard_normal((B_cur, H_PRE)) * 0.3).astype(np.float32)
        moe_np[-1, :] = np.nan
        moe_bf16 = jnp.asarray(moe_np, dtype=jnp.bfloat16)
        cs_step = build_decode_rope_step_tables(jnp.arange(B_cur, dtype=jnp.int32), cs_pairs)

        def _run_b2(wp_arg, sp_arg):
            return jax.shard_map(
                lambda *a: w8.moe_to_attn_chip_sharded_bridge_jax(
                    *a,
                    world_size=TP,
                    sharded_moe_in=sh_in,
                    post_ff_eps=1e-5,
                    post_ff_scale_input=True,
                    next_in_eps=1e-5,
                    next_in_scale_input=False,
                    fuse_qkv_epi=True,
                    qkv_norm_eps=1e-6,
                    query_scale=0.125,
                    k_scale=0.75,
                    v_scale=1.25,
                    interpret=ip_race,
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None),
                    P("model", None) if sh_in else P(),
                    P(None, "model"),
                    P("model"),
                    P(), P(), P(), P(),
                    P(None, "model"), P("model"),
                    P(None, "model"), P("model"),
                    P(),
                ),
                out_specs=(
                    P("model", None),
                    P(None, None, "model", None, None),
                    P(None, "model", None, None),
                    P(None, "model"),
                ),
                check_vma=False,
            )(st_stacked, moe_bf16, wp_arg, sp_arg, pff_g, pf_a, pf_b, nin_g, wq, sq, wg, sg, cs_step)

        st_ref, q_ref, kv_ref, g_ref = _run_b2(wpost_tp2, spost_tp2)
        st_xyz, q_xyz, kv_xyz, g_xyz = _run_b2(wpost_full, spost_full)

        st_ref_np = np.asarray(st_ref).reshape(TP, r_pair, H)
        st_xyz_np = np.asarray(st_xyz).reshape(TP, r_pair, H)
        for c in range(n_chips):
            r_lim = min(r_pair, max(0, valid - c * r_pair))
            np.testing.assert_array_equal(st_xyz_np[2 * c, :r_lim], st_xyz_np[2 * c + 1, :r_lim])
        st_ref_reasm = np.concatenate([st_ref_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
        st_xyz_reasm = np.concatenate([st_xyz_np[2 * c] for c in range(n_chips)], axis=0)[:valid]
        assert np.isfinite(st_xyz_reasm).all()
        def _cos64(a: np.ndarray, b: np.ndarray) -> float:
            af = a.astype(np.float64)
            bf = b.astype(np.float64)
            return float(np.sum(af * bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))

        assert _cos64(st_xyz_reasm, st_ref_reasm) >= 0.999999
        np.testing.assert_allclose(st_xyz_reasm, st_ref_reasm, rtol=2e-2, atol=2e-2)
        q_xyz_f = np.asarray(q_xyz, dtype=np.float32)[:, :valid]
        q_ref_f = np.asarray(q_ref, dtype=np.float32)[:, :valid]
        assert _cos64(q_xyz_f, q_ref_f) >= 0.999999
        np.testing.assert_allclose(q_xyz_f, q_ref_f, rtol=3e-2, atol=3e-2)
        kv_xyz_f = np.asarray(kv_xyz, dtype=np.float32)[:valid]
        kv_ref_f = np.asarray(kv_ref, dtype=np.float32)[:valid]
        assert _cos64(kv_xyz_f, kv_ref_f) >= 0.999
        np.testing.assert_allclose(kv_xyz_f, kv_ref_f, rtol=2e-1, atol=3e-1)
        g_xyz_f = np.asarray(g_xyz, dtype=np.float32)[:valid]
        g_ref_f = np.asarray(g_ref, dtype=np.float32)[:valid]
        assert _cos64(g_xyz_f, g_ref_f) >= 0.999999
        np.testing.assert_allclose(g_xyz_f, g_ref_f, rtol=3e-2, atol=3e-2)

    tpu_devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x2").devices
    aot_mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))),
        ("model",),
    )
    prod_H, prod_HPRE = 8192, 4096
    for B_cur, sh_in in ((64, False), (128, True), (256, True), (384, True)):
        r_pair = B_cur // n_chips
        in_r = (B_cur // TP) if sh_in else B_cur
        b2_shapes = (
            jax.ShapeDtypeStruct((r_pair, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((in_r, prod_HPRE), jnp.bfloat16),
            jax.ShapeDtypeStruct((prod_HPRE, prod_H // TP), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, prod_H // TP), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((1, prod_H), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, 640), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 640), jnp.float32),
            jax.ShapeDtypeStruct((prod_H, 512), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 512), jnp.float32),
            jax.ShapeDtypeStruct((B_cur, 2, 128), jnp.float32),
        )
        c_b2 = jax.jit(
            jax.shard_map(
                lambda *a, _sh=sh_in: w8.moe_to_attn_chip_sharded_bridge_jax(
                    *a, world_size=16, sharded_moe_in=_sh, fuse_qkv_epi=True, interpret=False
                ),
                mesh=aot_mesh,
                in_specs=tuple(P() for _ in b2_shapes),
                out_specs=(P(), P(), P(), P()),
                check_vma=False,
            )
        ).trace(*b2_shapes).lower(lowering_platforms=("tpu",)).compile()
        assert "tpu_custom_call" in c_b2.as_text()

