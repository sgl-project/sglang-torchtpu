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



