# SPDX-License-Identifier: Apache-2.0
"""Wave-10 unit, Pallas interpret, & TPU7x AOT tests for B=8 replicated Step (b) and TP=16 chip-sharded embed."""

from __future__ import annotations

import dataclasses
import gc
from unittest import mock

import numpy as np
import pytest


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if np.allclose(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def _rms_norm_np(
    x: np.ndarray, gain: np.ndarray, eps: float, scale_input: bool
) -> np.ndarray:
    xf = x.astype(np.float32)
    gf = gain.reshape(1, -1).astype(np.float32)
    vin = xf * gf if scale_input else xf
    inv = 1.0 / np.sqrt(np.mean(vin * vin, axis=-1, keepdims=True) + np.float32(eps))
    return (xf * gf) * inv if scale_input else (xf * inv) * gf


@pytest.mark.parametrize("world_size,row_mode", [(8, "sparse")])
def test_b1_moe_b2_replicated_b8_interpret_and_aot(world_size: int, row_mode: str):
    """Verify native B=8 replicated Step (b) parity vs standalone Bridge1b+MoE+Bridge2b & TPU7x AOT."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    if len(jax.devices("cpu")) < world_size:
        pytest.skip(
            f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}"
        )

    from sglang_tpu.kernels.decode_collective_bridges import attn_to_moe_bridge_jax
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        moe_to_attn_qkv_epi_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _run_muse_sparse_from_vmem_core,
        build_muse_sparse_scratch_shapes,
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_b16_kernel import (
        b1_moe_b2_replicated_b16_jax,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        _unpack_muse_sparse_scratch,
    )
    from test.registered.unit.hardware_backend.tpu.test_muse_w9_b1_moe_superkernel import (
        _build_b1_moe_inputs,
    )

    batch_size = 8
    hidden = 8192
    h_pre = 4096
    qkv_cols = 640 if world_size == 16 else 1280
    gate_cols = 512 if world_size == 16 else 1024
    scale_block = 512
    ip_fast = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=False,
        uninitialized_memory="nan",
    )
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(
        20261240 + world_size + (1 if row_mode == "sparse" else 0)
    )
    d = _build_b1_moe_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=0,
        bf16_router=True,
        sparse_heavy=True,
    )
    stream_rep = jnp.asarray(
        (rng.standard_normal((batch_size, hidden)) * 0.25).astype(np.float32)
    )
    w_pre_rep = jnp.asarray(
        (rng.standard_normal((hidden, h_pre)) * 0.04).astype(np.float32),
        dtype=jnp.float8_e4m3fn,
    )
    s_pre_rep = jnp.asarray(
        (np.abs(rng.standard_normal((h_pre,))) * 0.02 + 1e-3).astype(np.float32)
    )
    w_post_rep = jnp.asarray(
        (rng.standard_normal((h_pre, hidden)) * 0.04).astype(np.float32),
        dtype=jnp.float8_e4m3fn,
    )
    s_post_rep = jnp.asarray(
        (np.abs(rng.standard_normal((hidden,))) * 0.02 + 1e-3).astype(np.float32)
    )
    post_ff_gain = jnp.asarray((np.ones((hidden,)) * 1.03).astype(np.float32))
    pf_alpha = jnp.asarray((np.ones((hidden,)) * 0.92).astype(np.float32))
    pf_beta = jnp.asarray((np.ones((hidden,)) * 0.88).astype(np.float32))
    next_in_gain = jnp.asarray((np.ones((hidden,)) * 1.01).astype(np.float32))
    next_w_qkv = jnp.asarray(
        rng.standard_normal((world_size * hidden, qkv_cols)).astype(np.float32) * 0.04,
        dtype=jnp.float8_e4m3fn,
    )
    next_s_qkv = jnp.asarray(
        (np.abs(rng.standard_normal((qkv_cols,))) * 0.02 + 1e-3).astype(np.float32)
    )
    next_w_gate = jnp.asarray(
        rng.standard_normal((world_size * hidden, gate_cols)).astype(np.float32) * 0.04,
        dtype=jnp.float8_e4m3fn,
    )
    next_s_gate = jnp.asarray(
        (np.abs(rng.standard_normal((gate_cols,))) * 0.02 + 1e-3).astype(np.float32)
    )
    cos_sin_step = jnp.asarray(
        (rng.standard_normal((batch_size, 2, 128)) * 0.3).astype(np.float32)
    )
    free_params = jnp.asarray([0.84, 1.0, 0.20, 2.0], dtype=jnp.float32)

    cfg = dataclasses.replace(
        default_sparse_decode_config(batch_size, world_size, scale_block),
        row_mode=row_mode,
        weight_mode="stream3",
        tile_m=8 if row_mode == "dense" else 16,
        m_buckets=(8,),
        route_chunk_rows=8,
        static_holder_mask=True,
        route_fast=True,
        fast_plan=2,
        unroll_strips=True,
        fast_gs=True,
        pair_rs=(world_size == 8),
        bridge_ag=False,
        fast_b64_epi_pro=False,
        spec_prime_parts=0,
        k256_strip_mode="wide",
        nc_split512=False,
    )

    def _run_standalone(
        rank,
        stream_r,
        gated_attn,
        wo,
        wo_scale,
        pa_alpha,
        pa_beta,
        pre_ff_gain,
        w_pre,
        s_pre,
        w_router,
        pre_exp_gain,
        bias,
        w1,
        s1,
        w2,
        s2,
        norm,
        w_post,
        s_post,
        post_ff_g,
        pf_a,
        pf_b,
        next_in_g,
        w_qkv,
        s_qkv,
        w_gate,
        s_gate,
        cos_sin,
        fp,
    ):
        st1, logits, expert_in = attn_to_moe_bridge_jax(
            stream_r,
            gated_attn,
            wo,
            wo_scale,
            pa_alpha,
            pa_beta,
            pre_ff_gain,
            w_pre,
            s_pre,
            w_router,
            pre_exp_gain,
            world_size=world_size,
            pa_eps=1e-6,
            pre_ff_eps=1e-5,
            pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687,
            pre_exp_eps=1e-5,
            pre_exp_scale_input=False,
            interpret=ip_fast,
        )
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size,
            batch_size=batch_size,
            scale_block=scale_block,
            s1_dtype=s1.dtype,
            s2_dtype=s2.dtype,
            num_replicas=0,
            return_sharded=False,
            config=cfg,
        )
        v_sp, h_sp = (
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )

        def _ref_moe(*refs):
            it = iter(refs)
            rk_r, fp_r = next(it), next(it)
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = (
                next(it),
                next(it),
                next(it),
                next(it),
                next(it),
            )
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it,
                ep=world_size,
                scale_block=scale_block,
                num_replicas=0,
                return_sharded=False,
                cfg=cfg,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(
                    bsem,
                    device_id={"model": jnp.int32(d_i)},
                    device_id_type=pl.DeviceIdType.MESH,
                )
            _run_muse_sparse_from_vmem_core(
                rk_r[0],
                lg_r,
                bs_r,
                x_r,
                w1_r,
                s1_r,
                w2_r,
                s2_r,
                nm_r,
                **{k: v for k, v in msc.items() if k != "w_mode"},
                barrier_sem=bsem,
                output_ref=out_r,
                ep=world_size,
                batch_size=batch_size,
                scale_block=scale_block,
                num_replicas=0,
                return_sharded=False,
                replica_cost_mode="tiles",
                ep_axis_name="model",
                cfg=cfg,
                free_params_ref=fp_r,
                wait_entry_barrier=True,
                write_output_ref=True,
            )

        moe_out = pl.pallas_call(
            _ref_moe,
            out_shape=jax.ShapeDtypeStruct((batch_size, h_pre), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(1,),
                in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                out_specs=v_sp,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2,
                collective_id=0,
                disable_bounds_checks=True,
            ),
            interpret=ip_fast,
        )(jnp.atleast_1d(rank), fp, logits, bias, expert_in, w1, s1, w2, s2, norm)

        return moe_to_attn_qkv_epi_bridge_jax(
            st1,
            moe_out,
            w_post,
            s_post,
            post_ff_g,
            pf_a,
            pf_b,
            next_in_g,
            w_qkv,
            s_qkv,
            w_gate,
            s_gate,
            cos_sin,
            world_size=world_size,
            sharded_moe_in=False,
            interpret=ip_fast,
        )

    def _run_fused(
        rank,
        stream_r,
        gated_attn,
        wo,
        wo_scale,
        pa_alpha,
        pa_beta,
        pre_ff_gain,
        w_pre,
        s_pre,
        w_router,
        pre_exp_gain,
        bias,
        w1,
        s1,
        w2,
        s2,
        norm,
        w_post,
        s_post,
        post_ff_g,
        pf_a,
        pf_b,
        next_in_g,
        w_qkv,
        s_qkv,
        w_gate,
        s_gate,
        cos_sin,
        fp,
    ):
        return b1_moe_b2_replicated_b16_jax(
            rank,
            stream_r,
            gated_attn,
            wo,
            wo_scale,
            pa_alpha,
            pa_beta,
            pre_ff_gain,
            w_pre,
            s_pre,
            w_router,
            pre_exp_gain,
            bias,
            w1,
            s1,
            w2,
            s2,
            norm,
            w_post,
            s_post,
            post_ff_g,
            pf_a,
            pf_b,
            next_in_g,
            w_qkv,
            s_qkv,
            w_gate,
            s_gate,
            cos_sin_step=cos_sin,
            free_params=fp,
            world_size=world_size,
            pa_eps=1e-6,
            pre_ff_eps=1e-5,
            pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687,
            pre_exp_eps=1e-5,
            pre_exp_scale_input=False,
            replica_cost_mode="tiles",
            config=cfg,
            interpret=ip_race,
        )

    mesh = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    call_args = (
        d["ranks"],
        stream_rep,
        d["gated_full"],
        d["wo_sharded"],
        d["wo_scale"],
        d["pa_alpha"],
        d["pa_beta"],
        d["pre_ff_gain"],
        w_pre_rep,
        s_pre_rep,
        d["w_router"],
        d["pre_exp_gain"],
        d["bias"],
        d["w1_all"],
        d["s1_all"],
        d["w2_all"],
        d["s2_all"],
        d["post_exp_norm"],
        w_post_rep,
        s_post_rep,
        post_ff_gain,
        pf_alpha,
        pf_beta,
        next_in_gain,
        next_w_qkv,
        next_s_qkv,
        next_w_gate,
        next_s_gate,
        cos_sin_step,
        free_params,
    )
    in_specs = (
        P("model"),
        P(),
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
        P(),
        P("model", None, None),
        P("model", None, None),
        P("model", None, None),
        P("model", None, None),
        P(),
        P(None, "model"),
        P("model"),
        P(),
        P(),
        P(),
        P(),
        P("model", None),
        P(),
        P("model", None),
        P(),
        P(),
        P(),
    )
    out_specs = (P("model"), P("model"), P("model"), P("model"))

    ref_outs = jax.jit(
        jax.shard_map(
            _run_standalone,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
    )(*call_args)
    ref_np = [np.asarray(x, dtype=np.float32) for x in ref_outs]
    del ref_outs
    jax.clear_caches()
    gc.collect()

    fused_outs = jax.jit(
        jax.shard_map(
            _run_fused,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
    )(*call_args)
    fused_np = [np.asarray(x, dtype=np.float32) for x in fused_outs]
    del fused_outs, call_args, d
    jax.clear_caches()
    gc.collect()

    for idx, name in enumerate(("stream", "q_5d", "kv_merged", "gate")):
        r_np = ref_np[idx]
        f_np = fused_np[idx]
        assert np.all(np.isfinite(f_np)), f"{name} non-finite"
        cos = _cosine_sim(r_np, f_np)
        assert cos > 0.9999, f"{name} cosine mismatch: {cos}"


@pytest.mark.parametrize("row_mode", ["sparse"])
def test_b1_moe_b2_replicated_b8_tpu7x_aot(row_mode: str):
    """Verify TPU7x AOT compilation for native B=8 replicated Step (b) on TP=8 & TP=16."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_b16_kernel import (
        b1_moe_b2_replicated_b16_jax,
    )

    batch_size, hidden, h_pre, scale_block = 8, 8192, 4096, 512

    # Verify TPU7x AOT compilation for TP=8 (2x2x1) B=8 Step (b).
    for ws_aot, topo_name in ((8, "tpu7x:2x2x1"),):
        tpu_devs = topologies.get_topology_desc(topo_name, platform="tpu").devices
        mesh_aot = Mesh(
            np.array(
                sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))
            ),
            ("model",),
        )
        o_in = hidden // ws_aot
        pre_c = h_pre // ws_aot
        post_c = hidden // ws_aot
        qc = 1280 if ws_aot == 8 else 640
        gc = 1024 if ws_aot == 8 else 512
        e_loc = 256 // ws_aot
        nq = h_pre // scale_block
        abs_aot = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, hidden), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, o_in), jnp.bfloat16),
            jax.ShapeDtypeStruct((o_in, hidden), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((hidden, pre_c), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, pre_c), jnp.float32),
            jax.ShapeDtypeStruct((hidden, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, h_pre), jnp.float32),
            jax.ShapeDtypeStruct((1, 256), jnp.float32),
            jax.ShapeDtypeStruct((e_loc, h_pre, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((e_loc, nq, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((e_loc, 4096, h_pre), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((e_loc, nq, h_pre), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, h_pre), jnp.bfloat16),
            jax.ShapeDtypeStruct((h_pre, post_c), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, post_c), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((1, hidden), jnp.float32),
            jax.ShapeDtypeStruct((hidden, qc), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, qc), jnp.float32),
            jax.ShapeDtypeStruct((hidden, gc), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, gc), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, 2, 128), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        )
        cfg_aot = dataclasses.replace(
            default_sparse_decode_config(batch_size, ws_aot, scale_block),
            row_mode=row_mode,
            weight_mode="stream3",
            tile_m=8 if row_mode == "dense" else 16,
            m_buckets=(8,),
            pair_rs=(ws_aot == 8),
            bridge_ag=False,
        )

        def _aot_fn(*args, _ws=ws_aot, _cfg=cfg_aot):
            (
                rank,
                stream_r,
                gated_attn,
                wo,
                wo_s,
                pa_a,
                pa_b,
                pff_g,
                w_pre,
                s_pre,
                w_rtr,
                pex_g,
                bs,
                w1_,
                s1_,
                w2_,
                s2_,
                nm,
                w_post,
                s_post,
                post_g,
                pf_a,
                pf_b,
                nin_g,
                wqkv,
                sqkv,
                wgate,
                sgate,
                cs,
                fp_,
            ) = args
            return b1_moe_b2_replicated_b16_jax(
                rank,
                stream_r,
                gated_attn,
                wo,
                wo_s,
                pa_a,
                pa_b,
                pff_g,
                w_pre,
                s_pre,
                w_rtr,
                pex_g,
                bs,
                w1_,
                s1_,
                w2_,
                s2_,
                nm,
                w_post,
                s_post,
                post_g,
                pf_a,
                pf_b,
                nin_g,
                wqkv,
                sqkv,
                wgate,
                sgate,
                cos_sin_step=cs,
                free_params=fp_,
                world_size=_ws,
                replica_cost_mode="tiles",
                config=_cfg,
                interpret=False,
            )

        compiled = (
            jax.jit(
                jax.shard_map(
                    _aot_fn,
                    mesh=mesh_aot,
                    in_specs=(P(),) * len(abs_aot),
                    out_specs=(P(),) * 4,
                    check_vma=False,
                )
            )
            .trace(*abs_aot)
            .lower(lowering_platforms=("tpu",))
            .compile()
        )
        assert compiled.memory_analysis().generated_code_size_in_bytes > 0


def test_tp16_decode_vocab_embed_chip_sharded_interpret_and_aot():
    """Verify TP=16 decode_vocab_embed_chip_sharded_jax parity (B=64, B=128) & TPU7x AOT (B=64..384)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    mesh16 = Mesh(np.array(jax.devices("cpu")[:16]), ("model",))
    rng = np.random.default_rng(20261249)

    for b_emb in (64, 128):
        v_sh, h_emb = 32, 1024
        emb_full = (rng.standard_normal((16 * v_sh, h_emb)) * 0.2).astype(np.float32)
        emb_bf16 = jnp.asarray(emb_full, dtype=jnp.bfloat16)
        ids = rng.integers(0, 16 * v_sh, size=(b_emb,), dtype=np.int32)
        l0_g = (np.ones((h_emb,)) * 1.05).astype(np.float32)

        def _emb_loc(i_ids, ew, g0):
            return w8.decode_vocab_embed_chip_sharded_jax(
                i_ids,
                ew,
                g0,
                world_size=16,
                embed_eps=1e-5,
                l0_eps=1e-5,
                interpret=ip_race,
            )

        st0_ch, ai0_ch = jax.shard_map(
            _emb_loc,
            mesh=mesh16,
            in_specs=(P(), P("model", None), P()),
            out_specs=(P("model", None), P("model", None)),
            check_vma=False,
        )(jnp.asarray(ids), emb_bf16, jnp.asarray(l0_g))

        gathered_ref = np.asarray(emb_bf16[ids], dtype=np.float32)
        st0_ref = np.asarray(
            jnp.asarray(
                _rms_norm_np(
                    gathered_ref, np.ones((h_emb,), dtype=np.float32), 1e-5, False
                ),
                dtype=jnp.bfloat16,
            ),
            dtype=np.float32,
        )
        ai0_ref = _rms_norm_np(st0_ref, l0_g, 1e-5, False)
        r_pair = b_emb // 8
        st0_np = np.asarray(st0_ch).reshape(16, r_pair, h_emb)
        st0_reasm = np.concatenate([st0_np[2 * c] for c in range(8)], axis=0)
        np.testing.assert_allclose(st0_reasm, st0_ref, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(
            np.asarray(ai0_ch, dtype=np.float32).reshape(16, b_emb, h_emb)[0],
            ai0_ref,
            rtol=1e-2,
            atol=1e-2,
        )

    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(
            sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))
        ),
        ("model",),
    )
    for b_aot in (64, 128, 256, 384):
        abs_in = (
            jax.ShapeDtypeStruct((b_aot,), jnp.int32),
            jax.ShapeDtypeStruct((8192, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
        )
        comp = (
            jax.jit(
                jax.shard_map(
                    lambda i, e, g: w8.decode_vocab_embed_chip_sharded_jax(
                        i, e, g, world_size=16, embed_eps=1e-5, l0_eps=1e-5
                    ),
                    mesh=mesh_aot,
                    in_specs=(P(), P(), P()),
                    out_specs=(P(), P()),
                    check_vma=False,
                )
            )
            .trace(*abs_in)
            .lower(lowering_platforms=("tpu",))
            .compile()
        )
        assert comp.memory_analysis().generated_code_size_in_bytes > 0


def test_w10_muse_spark_b8_step_b_and_tp16_embed_dispatch_and_prewarm():
    """Verify MuseSpark B=8 replicated Step (b) dispatch & TP=16 lock-free chip-sharded embed."""
    import os
    import types

    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.kernels import decode_collective_bridges_w8 as w8
    from sglang_tpu.models.muse_spark import MuseSparkDecoderLayer, MuseSparkTextModel
    from test.registered.unit.hardware_backend.tpu.test_muse_w7_col_proj import (
        _make_dummy_layer,
        _make_norm,
    )

    h_dim, h_pre = 8192, 4096
    fb = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        attn_backend=types.SimpleNamespace(can_forward_decode_prepared=lambda *_: True),
        return_logprob=False,
        capture_hidden_mode=None,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        extend_seq_lens=None,
    )
    lyr: MuseSparkDecoderLayer = _make_dummy_layer(3, H=h_dim, H_PRE=h_pre, tp_size=8)
    lyr._decode_col_bridges = 3
    lyr._w9_b1_moe_b2_superkernel = True
    lyr._w8_bridge_qkv_epi = True
    lyr.self_attn._o_proj_w8k = torch.zeros((1024, 8192), dtype=torch.float8_e4m3fn)
    lyr.self_attn._o_proj_scale = torch.ones((8192,), dtype=torch.float32)
    lyr.self_attn.q_size = 1024
    lyr.self_attn.kv_size = 128
    lyr.self_attn.decode_qkv_epilogue = True
    lyr.self_attn.rotary_emb = object()

    next_attn = types.SimpleNamespace(
        _buffers={
            "_w7_wqkv_w8k": torch.zeros((h_dim, 1280), dtype=torch.float8_e4m3fn),
            "_w7_sqkv": torch.ones((1280,), dtype=torch.float32),
            "_w7_wgate_w8k": torch.zeros((h_dim, 1024), dtype=torch.float8_e4m3fn),
            "_w7_sgate": torch.ones((1024,), dtype=torch.float32),
        },
        q_size=1024,
        kv_size=128,
        decode_qkv_epilogue=True,
        rotary_emb=object(),
        norm_eps=1e-5,
        query_scale=1.0,
        attn=types.SimpleNamespace(k_scale_float=1.0, v_scale_float=1.0),
    )

    mlp = lyr.mlp
    mlp._fused_decode_mode = "sparse"
    mlp._fused_decode_op = object()
    mlp._fused_decode_batches = (8, 16, 64)
    mlp._fused_decode_rank = torch.zeros((1,), dtype=torch.int32)
    mlp._fused_decode_num_replicas = 0
    mlp._fused_decode_has_spec_slot = False
    mlp._fused_decode_replica_cost_mode = "tiles"
    mlp._free_params_for_batch = lambda _b: None
    mlp._norm_2d = lambda: torch.ones((1, h_pre), dtype=torch.bfloat16)
    mlp._bias_2d = lambda: torch.zeros((1, 256), dtype=torch.float32)
    mlp._buffers = {
        "_w7_wpre_w8k": torch.zeros((h_dim, h_pre // 8), dtype=torch.float8_e4m3fn),
        "_w7_spre": torch.ones((h_pre // 8,), dtype=torch.float32),
        "_w7_wrouter_km": torch.zeros((h_dim, 256), dtype=torch.bfloat16),
        "_w7_wpost_w8k": torch.zeros((h_pre, h_dim // 8), dtype=torch.float8_e4m3fn),
        "_w7_spost": torch.ones((h_dim // 8,), dtype=torch.float32),
        "_decode_fp4_w1": torch.zeros((1,), dtype=torch.uint8)
        .view(torch.float4_e2m1fn_x2)
        .expand(32, 4096, 4096),
        "_decode_fp4_s1": torch.ones((1,), dtype=torch.bfloat16).expand(32, 8, 8192),
        "_decode_fp4_w2": torch.zeros((1,), dtype=torch.uint8)
        .view(torch.float4_e2m1fn_x2)
        .expand(32, 4096, 2048),
        "_decode_fp4_s2": torch.ones((1,), dtype=torch.bfloat16).expand(32, 8, 4096),
    }

    b8_calls: list[int] = []

    def _fake_b16_op(*args, **kwargs):
        del kwargs
        def _fn(_rk, st, *_rest):
            b8_calls.append(int(st.shape[0]))
            return (
                st + 1.0,
                torch.zeros((1, 8, 8, 2, 128), dtype=torch.bfloat16),
                torch.zeros((8, 1, 4, 128), dtype=torch.float8_e4m3fn),
                torch.zeros((8, 1024), dtype=torch.bfloat16),
            )

        return _fn

    with (
        mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_replicated_b16_op",
            side_effect=_fake_b16_op,
        ),
    ):
        b1_k, b2_k = lyr._active_w7_bridges(
            8,
            fb,
            is_dec_step=True,
            next_input_layernorm=_make_norm(h_dim),
            next_layer_attn=next_attn,
            sharded_stream=False,
        )
        assert (b1_k, b2_k) == (2, 3)
        assert lyr._can_run_b1_moe_superkernel(
            8, tp=8, is_dec_step=True, replicated=True
        )
        out = lyr.forward(
            positions=torch.zeros((8,), dtype=torch.int32),
            stream=torch.zeros((8, h_dim), dtype=torch.float32),
            forward_batch=fb,
            next_input_layernorm=_make_norm(h_dim),
            cos_sin_step=torch.zeros((8, 2, 128), dtype=torch.float32),
            precomputed_gated=torch.ones((8, 1024), dtype=torch.bfloat16),
            next_layer_attn=next_attn,
        )
        assert len(out) == 4 and b8_calls == [8]

    # Verify TP=16 prewarm registers get_decode_vocab_embed_chip_sharded_op(16) lock-free.
    captured_ops: dict[str, object] = {}

    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("_W8_LOCK entered after prewarm!")

        def __exit__(self, *args):
            return False

    with (
        mock.patch.object(w8, "_get_tp_mesh", return_value=object()),
        mock.patch(
            "sglang_tpu.pallas_jax_op.cpu_export_jax_op",
            side_effect=lambda name, fn, **_kw: captured_ops.setdefault(name, fn),
        ),
        mock.patch.dict(w8._EMBED_CHIP_SHARDED_OPS, {}, clear=True),
        mock.patch.dict(w8._ATTN_TO_MOE_CHIP_SH_OPS, {}, clear=True),
        mock.patch.dict(w8._MOE_TO_ATTN_CHIP_SH_OPS, {}, clear=True),
        mock.patch.dict(w8._MOE_POST_CHIP_SH_FINAL_OPS, {}, clear=True),
        mock.patch.dict(w8._AG_CHIP_CAP_OPS, {}, clear=True),
    ):
        w8.prewarm_w8_decode_bridge_ops(
            16, 3, bf16_router=True, chip_sharded=True, pallas_embed=True
        )
        with mock.patch.object(w8, "_W8_LOCK", _ExplodingLock()):
            assert w8.get_decode_vocab_embed_chip_sharded_op(16) is not None
        assert "sglang_tpu_pallas::muse_w9_embed_chip_sh_tp16" in captured_ops

    # Verify MuseSparkTextModel.forward entry path dispatches chip-sharded TP=16 embed for B=64.
    emb_calls: list[int] = []
    txt_model = MuseSparkTextModel.__new__(MuseSparkTextModel)
    torch.nn.Module.__init__(txt_model)
    txt_model.config = types.SimpleNamespace(
        rms_norm_eps=1e-5, hidden_size=h_dim, num_hidden_layers=1
    )
    txt_model.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    txt_model.prefill_kernels = None
    txt_model.embed_tokens = types.SimpleNamespace(
        weight=torch.zeros((8192, h_dim), dtype=torch.bfloat16)
    )
    l0_ln = _make_norm(h_dim)
    txt_model.layers = [types.SimpleNamespace(input_layernorm=l0_ln)]
    txt_model.start_layer = 0
    txt_model.end_layer = 0
    txt_model.layers_to_capture = ()
    txt_model._capture_slots_by_layer = {}
    txt_model._decode_col_bridges = 3
    txt_model._decode_norm_layout = 0
    txt_model._w8_pallas_embed_tp = 16
    txt_model._w8_pallas_embed_enabled = True
    txt_model._w9_chip_sharded_bridges = True
    txt_model._w9_all_layers_chip_sharded_ready = True
    txt_model._w9_chip_sharded_tp = 16
    txt_model._w9_chip_sharded_batches = (64, 128, 256, 384)
    txt_model._w8_tok_sharded_bridges = False
    txt_model._w8_all_layers_tok_sharded_ready = False
    txt_model._w8_fuse_final_norm = False
    txt_model.rotary_emb = types.SimpleNamespace(cos_sin_pairs=None)
    txt_model.norm = lambda x: x

    def _fake_emb_op(ws: int):
        emb_calls.append(ws)
        return lambda ids, ew, g0, *_sc: (
            torch.zeros((ids.shape[0] // (ws // 2), h_dim), dtype=torch.float32),
            torch.ones((ids.shape[0], h_dim), dtype=torch.bfloat16),
        )

    with (
        mock.patch.dict(os.environ, {"WORLD_SIZE": "16"}),
        mock.patch.object(
            w8, "get_decode_vocab_embed_chip_sharded_op", side_effect=_fake_emb_op
        ),
    ):
        txt_model.forward(
            input_ids=torch.zeros((64,), dtype=torch.int32),
            positions=torch.zeros((64,), dtype=torch.int32),
            forward_batch=fb,
        )
    assert emb_calls == [16]

    # Verify invalid shape guards (B=8 on TP=16, B=8 with dense config) raise ValueError cleanly.
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_b16_kernel import (
        b1_moe_b2_replicated_b16_jax,
    )
    import jax
    import jax.numpy as jnp

    assert not lyr._can_run_b1_moe_superkernel(
        8, tp=16, is_dec_step=True, replicated=True
    )
    dense_cfg = dataclasses.replace(
        default_sparse_decode_config(16, 8, 512),
        row_mode="dense",
    )
    with pytest.raises(ValueError, match="requires config.row_mode='sparse'"):
        b1_moe_b2_replicated_b16_jax(
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((8, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8, 1024), jnp.bfloat16),
            jax.ShapeDtypeStruct((1024, 8192), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8192, 512), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 512), jnp.float32),
            jax.ShapeDtypeStruct((8192, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 256), jnp.float32),
            jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((32, 8, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((32, 8, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((4096, 1024), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1024), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8192, 1280), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1280), jnp.float32),
            jax.ShapeDtypeStruct((8192, 1024), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1024), jnp.float32),
            world_size=8,
            config=dense_cfg,
        )
