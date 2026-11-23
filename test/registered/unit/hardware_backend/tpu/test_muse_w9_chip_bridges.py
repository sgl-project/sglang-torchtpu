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

    tree = ast.parse((_REPO_ROOT / "sglang_tpu/models/muse_spark.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            src = ast.unparse(node.test)
            if "SGLANG_TPU_MUSE_DECODE_CHIP_SHARDED_BRIDGES" in src and (
                "envs." in src or "tpu_envs." in src
            ):
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
    ):
        for tp in (8, 16):
            w8.prewarm_w8_decode_bridge_ops(
                tp,
                3,
                bf16_router=True,
                tok_sharded=(tp == 8),
                chip_sharded=True,
                bridge_qkv_epi=(tp == 8),
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
                    if tp == 8:
                        for hr in (False, True):
                            assert callable(
                                w8.get_moe_to_attn_chip_sharded_bridge_op(
                                    8,
                                    sharded_moe_in=sh,
                                    fuse_qkv_epi=True,
                                    has_rope=hr,
                                )
                            )

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

    ip_race = pltpu.InterpretParams(dma_execution_mode="eager", detect_races=True)
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
                    interpret=True,
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
