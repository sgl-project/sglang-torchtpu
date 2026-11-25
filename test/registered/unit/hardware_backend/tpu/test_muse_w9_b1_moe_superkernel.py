# SPDX-License-Identifier: Apache-2.0
"""CPU Pallas interpret (eager + detect_races + NaN) & TPU7x AOT tests for Step (a) b1+MoE."""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / max(denom, 1e-30))


def _make_u32_bank_fast(
    seed: int,
    rows: int,
    k: int,
    n: int,
    scale_block: int,
):
    """Low-RAM synthetic packed NVFP4 bank ``(jnp.uint32 [rows, K//8, N], bf16 scales)``."""
    import jax
    import jax.numpy as jnp

    base_rows = min(rows, 4)
    rep = rows // base_rows

    @jax.jit
    def _gen():
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 2)
        # Keep nibbles small (bits in 0..3 => +-0.5..1.5) to avoid NaN/Inf in random dots.
        u = jax.random.randint(
            k1, (base_rows, k // 8, n), 0, 0x33333333, dtype=jnp.uint32
        )
        s = jax.random.uniform(
            k2, (base_rows, k // scale_block, n), jnp.bfloat16, 0.02, 0.08
        )
        return jnp.tile(u, (rep, 1, 1)), jnp.tile(s, (rep, 1, 1))

    return _gen()


def _build_b1_moe_inputs(
    rng: np.random.Generator,
    *,
    world_size: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    bf16_router: bool,
    sparse_heavy: bool = False,
):
    import jax.numpy as jnp
    from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
        build_preloaded_replica_plan,
        build_replica_weight_bank_from_sharded,
    )

    hidden = 8192
    h_pre = 4096
    inter = 4096
    n_exp = 256
    e_local = n_exp // world_size
    o_in = hidden // world_size
    pre_pair_cols = h_pre // 2
    n_chips = world_size // 2
    r_pair = batch_size // n_chips

    stream_full = (rng.standard_normal((batch_size, hidden)) * 0.25).astype(np.float32)
    stream_chip_st = jnp.asarray(
        np.concatenate(
            [stream_full[(r >> 1) * r_pair : ((r >> 1) + 1) * r_pair] for r in range(world_size)],
            axis=0,
        )
    )
    gated_full = jnp.asarray(
        (rng.standard_normal((batch_size, hidden)) * 0.25).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    wo_w8k = jnp.asarray(
        (rng.standard_normal((o_in, hidden)) * 0.04).astype(np.float32),
        dtype=jnp.float8_e4m3fn,
    )
    wo_sharded = jnp.asarray(
        np.concatenate(
            [
                np.asarray(
                    jnp.asarray(
                        (rng.standard_normal((o_in, hidden)) * 0.04).astype(np.float32),
                        dtype=jnp.float8_e4m3fn,
                    )
                )
                for _ in range(world_size)
            ],
            axis=0,
        ),
        dtype=jnp.float8_e4m3fn,
    )
    wo_scale = jnp.asarray((np.abs(rng.standard_normal((hidden,))) * 0.02 + 1e-3).astype(np.float32))
    pa_alpha = jnp.asarray((np.ones((hidden,)) * 0.95).astype(np.float32))
    pa_beta = jnp.asarray((np.ones((hidden,)) * 0.85).astype(np.float32))
    pre_ff_gain = jnp.asarray((np.ones((hidden,)) * 1.05).astype(np.float32))

    w_pre_full = jnp.asarray(
        (rng.standard_normal((hidden, h_pre)) * 0.04).astype(np.float32),
        dtype=jnp.float8_e4m3fn,
    )
    s_pre_full = (np.abs(rng.standard_normal((h_pre,))) * 0.02 + 1e-3).astype(np.float32)
    w_pre_pair = jnp.concatenate(
        [w_pre_full[:, :pre_pair_cols] if (r & 1) == 0 else w_pre_full[:, pre_pair_cols:] for r in range(world_size)],
        axis=1,
    )
    s_pre_pair = jnp.asarray(
        np.concatenate(
            [s_pre_full[:pre_pair_cols] if (r & 1) == 0 else s_pre_full[pre_pair_cols:] for r in range(world_size)],
            axis=0,
        )
    )
    w_router = jnp.asarray(
        (rng.standard_normal((hidden, n_exp)) * (0.2 if sparse_heavy else 0.04)).astype(np.float32),
        dtype=jnp.bfloat16 if bf16_router else jnp.float32,
    )
    pre_exp_gain = jnp.asarray((np.ones((h_pre,)) * 1.1).astype(np.float32))
    bias_np = (rng.standard_normal((1, n_exp)) * 0.05).astype(np.float32)
    # Bias 8 experts (1 per rank on EP=8, 1 per 2 ranks on EP=16) so CPU interpret
    # executes a realistic sparse subset (~8 active experts) quickly without OOM.
    bias_np[0, 0:n_exp:32] += 12.0
    bias = jnp.asarray(bias_np)
    post_exp_norm = jnp.asarray((rng.standard_normal((1, h_pre)) * 0.05).astype(np.float32))

    w1_all, s1_all = _make_u32_bank_fast(101 + world_size, n_exp, h_pre, 2 * inter, scale_block)
    w2_all, s2_all = _make_u32_bank_fast(202 + world_size, n_exp, inter, h_pre, scale_block)

    if num_replicas > 0:
        hot_probs = np.ones((n_exp,), dtype=np.float64)
        hot_probs[0:n_exp:32] = 20.0
        rank_to_chip = [r // 2 for r in range(world_size)]
        plan = build_preloaded_replica_plan(
            hot_probs,
            num_owners=world_size,
            num_replicas_per_owner=num_replicas,
            placement=None,
            owner_to_chip=rank_to_chip,
            selection_mode="tiles",
        )
        rw1_l, rs1_l, rw2_l, rs2_l = build_replica_weight_bank_from_sharded(
            [w1_all[r * e_local : (r + 1) * e_local] for r in range(world_size)],
            [s1_all[r * e_local : (r + 1) * e_local] for r in range(world_size)],
            [w2_all[r * e_local : (r + 1) * e_local] for r in range(world_size)],
            [s2_all[r * e_local : (r + 1) * e_local] for r in range(world_size)],
            plan,
        )
        rep_w1 = jnp.concatenate(rw1_l, axis=0)
        rep_s1 = jnp.concatenate(rs1_l, axis=0)
        rep_w2 = jnp.concatenate(rw2_l, axis=0)
        rep_s2 = jnp.concatenate(rs2_l, axis=0)
        replica_meta = jnp.asarray(plan.replica_meta, dtype=jnp.int32)
    else:
        rep_w1 = rep_s1 = rep_w2 = rep_s2 = replica_meta = None

    return dict(
        ranks=jnp.arange(world_size, dtype=jnp.int32),
        stream_chip_st=stream_chip_st,
        gated_full=gated_full,
        wo_sharded=wo_sharded,
        wo_scale=wo_scale,
        pa_alpha=pa_alpha,
        pa_beta=pa_beta,
        pre_ff_gain=pre_ff_gain,
        w_pre_pair=w_pre_pair,
        s_pre_pair=s_pre_pair,
        w_router=w_router,
        pre_exp_gain=pre_exp_gain,
        bias=bias,
        w1_all=w1_all,
        s1_all=s1_all,
        w2_all=w2_all,
        s2_all=s2_all,
        post_exp_norm=post_exp_norm,
        rep_w1=rep_w1,
        rep_s1=rep_s1,
        rep_w2=rep_w2,
        rep_s2=rep_s2,
        replica_meta=replica_meta,
    )


@pytest.mark.parametrize(
    "world_size,num_replicas,spec_parts",
    [
        (8, 0, 0),
        (8, 2, 0),
        (16, 2, 0),
    ],
)
def test_b1_moe_superkernel_interpret_parity(world_size: int, num_replicas: int, spec_parts: int):
    """Verify InterpretParams(eager + detect_races + nan) parity vs chip_b1 -> muse_sparse."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices("cpu")) < world_size:
        pytest.skip(f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}")

    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_chip_sharded_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _run_muse_sparse_from_vmem_core,
        build_muse_sparse_scratch_shapes,
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        _unpack_muse_sparse_scratch,
        b1_moe_chip_sharded_jax,
    )

    batch_size = 64
    scale_block = 512
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    mesh = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    rng = np.random.default_rng(20261125 + world_size + num_replicas + spec_parts)
    d = _build_b1_moe_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        bf16_router=True,
        sparse_heavy=True,
    )

    cfg = dataclasses.replace(
        default_sparse_decode_config(batch_size, world_size, scale_block),
        m_buckets=(16, 32),
        route_chunk_rows=32,
        static_holder_mask=True,
        route_fast=True,
        fast_plan=2,
        unroll_strips=True,
        fast_gs=True,
        pair_rs=True,
        fast_b64_epi_pro=False,
        spec_prime_parts=spec_parts,
        k256_strip_mode="wide",
        nc_split512=False,
        min_p_topk=0.92,
        free_max_rows=4,
        free_max_w=0.20,
        free_min_slot=1,
    )

    bridge_kw = dict(
        world_size=world_size,
        pa_eps=1e-6,
        pre_ff_eps=1e-5,
        pre_ff_scale_input=False,
        router_multiplier=0.17677669529663687,
        pre_exp_eps=1e-5,
        pre_exp_scale_input=False,
    )

    def _run_standalone(
        rank,
        stream_chip,
        gated_attn,
        wo,
        wo_scale,
        pa_alpha,
        pa_beta,
        pre_ff_gain,
        w_pre_pair,
        s_pre_pair,
        w_router,
        pre_exp_gain,
        bias,
        w1,
        s1,
        w2,
        s2,
        norm,
        rep_w1=None,
        rep_s1=None,
        rep_w2=None,
        rep_s2=None,
        meta=None,
    ):
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        st_new, logits, expert_in = attn_to_moe_chip_sharded_bridge_jax(
            stream_chip,
            gated_attn,
            wo,
            wo_scale,
            pa_alpha,
            pa_beta,
            pre_ff_gain,
            w_pre_pair,
            s_pre_pair,
            w_router,
            pre_exp_gain,
            interpret=ip_race,
            **bridge_kw,
        )
        cfg_ref = dataclasses.replace(cfg, spec_prime_parts=0)
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size,
            batch_size=batch_size,
            scale_block=scale_block,
            s1_dtype=s1.dtype,
            s2_dtype=s2.dtype,
            num_replicas=num_replicas,
            return_sharded=True,
            config=cfg_ref,
        )
        v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
        h_sp = pl.BlockSpec(memory_space=pltpu.HBM)

        def _ref_moe_kernel(*all_refs):
            it = iter(all_refs)
            rank_r = next(it)
            meta_r = next(it) if num_replicas > 0 else None
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = next(it), next(it), next(it), next(it), next(it)
            if num_replicas > 0:
                rw1_r, rs1_r, rw2_r, rs2_r = next(it), next(it), next(it), next(it)
            else:
                rw1_r = rs1_r = rw2_r = rs2_r = None
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it,
                ep=world_size,
                scale_block=scale_block,
                num_replicas=num_replicas,
                return_sharded=True,
                cfg=cfg_ref,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(
                    bsem,
                    device_id={"model": jnp.int32(d_i)},
                    device_id_type=pl.DeviceIdType.MESH,
                )
            _run_muse_sparse_from_vmem_core(
                rank_r[0],
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
                num_replicas=num_replicas,
                return_sharded=True,
                replica_cost_mode="active",
                ep_axis_name="model",
                cfg=cfg_ref,
                rep_w1_u32=rw1_r,
                rep_s1_ref=rs1_r,
                rep_w2_u32=rw2_r,
                rep_s2_ref=rs2_r,
                replica_meta_ref=meta_r,
            )

        sc_args = [jnp.atleast_1d(rank).astype(jnp.int32)]
        if num_replicas > 0:
            sc_args.append(meta)
        base_sp = (v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp)
        in_sp = base_sp + ((h_sp, h_sp, h_sp, h_sp) if num_replicas > 0 else ())
        c_args = (logits, bias, expert_in, w1, s1, w2, s2, norm)
        if num_replicas > 0:
            c_args = c_args + (rep_w1, rep_s1, rep_w2, rep_s2)
        moe_out = pl.pallas_call(
            _ref_moe_kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size // world_size, 4096), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(sc_args),
                grid=(1,),
                in_specs=in_sp,
                out_specs=v_sp,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2,
                collective_id=0,
                disable_bounds_checks=True,
            ),
            interpret=ip_race,
        )(*sc_args, *c_args)
        return st_new, moe_out

    def _run_fused(
        rank,
        stream_chip,
        gated_attn,
        wo,
        wo_scale,
        pa_alpha,
        pa_beta,
        pre_ff_gain,
        w_pre_pair,
        s_pre_pair,
        w_router,
        pre_exp_gain,
        bias,
        w1,
        s1,
        w2,
        s2,
        norm,
        rep_w1=None,
        rep_s1=None,
        rep_w2=None,
        rep_s2=None,
        meta=None,
    ):
        return b1_moe_chip_sharded_jax(
            rank,
            stream_chip,
            gated_attn,
            wo,
            wo_scale,
            pa_alpha,
            pa_beta,
            pre_ff_gain,
            w_pre_pair,
            s_pre_pair,
            w_router,
            pre_exp_gain,
            bias,
            w1,
            s1,
            w2,
            s2,
            norm,
            rep_w1,
            rep_s1,
            rep_w2,
            rep_s2,
            meta,
            return_sharded=True,
            config=cfg,
            interpret=ip_race,
            **bridge_kw,
        )

    base_specs = (
        P("model"),  # rank
        P("model", None),  # stream_chip
        P(None, "model"),  # gated_attn
        P("model", None),  # wo
        P(),  # wo_scale
        P(),  # pa_alpha
        P(),  # pa_beta
        P(),  # pre_ff_gain
        P(None, "model"),  # w_pre_pair
        P("model"),  # s_pre_pair
        P(),  # w_router
        P(),  # pre_exp_gain
        P(),  # bias
        P("model", None, None),  # w1
        P("model", None, None),  # s1
        P("model", None, None),  # w2
        P("model", None, None),  # s2
        P(),  # post_exp_norm
    )
    base_args = (
        d["ranks"],
        d["stream_chip_st"],
        d["gated_full"],
        d["wo_sharded"],
        d["wo_scale"],
        d["pa_alpha"],
        d["pa_beta"],
        d["pre_ff_gain"],
        d["w_pre_pair"],
        d["s_pre_pair"],
        d["w_router"],
        d["pre_exp_gain"],
        d["bias"],
        d["w1_all"],
        d["s1_all"],
        d["w2_all"],
        d["s2_all"],
        d["post_exp_norm"],
    )
    if num_replicas > 0:
        in_specs = base_specs + (
            P("model", None, None),
            P("model", None, None),
            P("model", None, None),
            P("model", None, None),
            P(),
        )
        args = base_args + (d["rep_w1"], d["rep_s1"], d["rep_w2"], d["rep_s2"], d["replica_meta"])
    else:
        in_specs = base_specs
        args = base_args

    out_specs = (P("model", None), P("model", None))
    st_ref, moe_ref = jax.jit(
        jax.shard_map(
            _run_standalone, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )
    )(*args)
    st_fus, moe_fus = jax.jit(
        jax.shard_map(
            _run_fused, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )
    )(*args)

    st_ref_np = np.asarray(st_ref, dtype=np.float32)
    st_fus_np = np.asarray(st_fus, dtype=np.float32)
    moe_ref_np = np.asarray(moe_ref, dtype=np.float32)
    moe_fus_np = np.asarray(moe_fus, dtype=np.float32)

    assert not np.isnan(st_fus_np).any()
    assert not np.isnan(moe_fus_np).any()
    np.testing.assert_allclose(st_fus_np, st_ref_np, rtol=1e-5, atol=1e-5)
    cos = _cosine_sim(moe_fus_np, moe_ref_np)
    max_diff = float(np.max(np.abs(moe_fus_np - moe_ref_np)))
    assert cos >= 0.99999, f"cos={cos}, max_diff={max_diff}"
    assert max_diff <= 1e-3, f"max_diff={max_diff}"


@pytest.mark.parametrize(
    "topo_str,world_size,batch_size,num_replicas,bf16_router,spec_parts,use_default_cfg",
    [
        ("tpu7x:2x2x1", 8, 64, 2, True, 0, True),
        ("tpu7x:2x2x1", 8, 64, 2, True, 1, False),
        ("tpu7x:2x2x2", 16, 64, 2, True, 1, False),
        ("tpu7x:2x2x1", 8, 64, 0, False, 0, False),
        ("tpu7x:2x2x1", 8, 128, 2, True, 0, False),
    ],
)
def test_b1_moe_superkernel_tpu7x_aot_compile(
    topo_str: str,
    world_size: int,
    batch_size: int,
    num_replicas: int,
    bf16_router: bool,
    spec_parts: int,
    use_default_cfg: bool,
):
    """Offline TPU7x Mosaic lowering & VMEM/semaphore verification (`0` TPU chips held)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.topologies import get_topology_desc
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        b1_moe_chip_sharded_jax,
    )

    tpu_devs = get_topology_desc(topo_str, platform="tpu").devices
    mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))),
        ("model",),
    )

    hidden = 8192
    h_pre = 4096
    inter = 4096
    scale_block = 512
    e_local = 256 // world_size
    o_in = hidden // world_size
    pre_pair_cols = h_pre // 2
    n_chips = world_size // 2
    r_pair = batch_size // n_chips
    rows_per_rank = batch_size // world_size
    nqb1 = h_pre // scale_block
    nqb2 = inter // scale_block

    cfg = (
        None
        if use_default_cfg
        else dataclasses.replace(
            default_sparse_decode_config(batch_size, world_size, scale_block),
            m_buckets=(16, 32) if batch_size <= 64 else (16, 32, 64),
            route_chunk_rows=32,
            static_holder_mask=True,
            route_fast=True,
            fast_plan=2,
            unroll_strips=True,
            fast_gs=True,
            pair_rs=True,
            fast_b64_epi_pro=False,
            spec_prime_parts=spec_parts,
        )
    )

    shapes = (
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
        jax.ShapeDtypeStruct((batch_size, o_in), jnp.bfloat16),
        jax.ShapeDtypeStruct((o_in, hidden), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((hidden, pre_pair_cols), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, pre_pair_cols), jnp.float32),
        jax.ShapeDtypeStruct((hidden, 256), jnp.bfloat16 if bf16_router else jnp.float32),
        jax.ShapeDtypeStruct((1, h_pre), jnp.float32),
        jax.ShapeDtypeStruct((1, 256), jnp.float32),
        jax.ShapeDtypeStruct((e_local, h_pre, 2 * inter), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((e_local, nqb1, 2 * inter), jnp.bfloat16),
        jax.ShapeDtypeStruct((e_local, inter, h_pre), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((e_local, nqb2, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, h_pre), jnp.bfloat16),
    )
    if num_replicas > 0:
        shapes = shapes + (
            jax.ShapeDtypeStruct((num_replicas, h_pre, 2 * inter), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((num_replicas, nqb1, 2 * inter), jnp.bfloat16),
            jax.ShapeDtypeStruct((num_replicas, inter, h_pre), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((num_replicas, nqb2, h_pre), jnp.bfloat16),
            jax.ShapeDtypeStruct((4, world_size * num_replicas), jnp.int32),
        )

    def _wrapped(*args):
        return b1_moe_chip_sharded_jax(
            *args,
            world_size=world_size,
            return_sharded=True,
            config=cfg,
        )

    sm = jax.jit(
        jax.shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=(P(),) * len(shapes),
            out_specs=(P(), P()),
            check_vma=False,
        )
    )
    comp = sm.trace(*shapes).lower(lowering_platforms=("tpu",)).compile()
    assert "tpu_custom_call" in comp.as_text()
    assert comp.memory_analysis().generated_code_size_in_bytes > 0
    assert rows_per_rank >= 4


@pytest.mark.parametrize(
    "topo_str,world_size,batch_size,num_replicas,scale_block,fuse_qkv_epi,has_rope",
    [
        ("tpu7x:2x2x1", 8, 64, 8, 512, True, True),
        ("tpu7x:2x2x1", 8, 64, 2, 512, True, False),
        ("tpu7x:2x2x2", 16, 64, 8, 512, True, True),
        ("tpu7x:2x2x2", 16, 128, 2, 512, False, False),
        ("tpu7x:2x2x1", 8, 128, 8, 256, True, True),
    ],
)
def test_b1_moe_b2_superkernel_tpu7x_aot_compile(
    topo_str: str,
    world_size: int,
    batch_size: int,
    num_replicas: int,
    scale_block: int,
    fuse_qkv_epi: bool,
    has_rope: bool,
):
    """Offline TPU7x Mosaic lowering for Step (b) [b1 + MoE + b2] (`0` TPU chips held)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.topologies import get_topology_desc
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
    )

    tpu_devs = get_topology_desc(topo_str, platform="tpu").devices
    mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))),
        ("model",),
    )

    hidden = 8192
    h_pre = 4096
    inter = 4096
    e_local = 256 // world_size
    o_in = hidden // world_size
    pre_pair_cols = h_pre // 2
    n_chips = world_size // 2
    r_pair = batch_size // n_chips
    nqb1 = h_pre // scale_block
    nqb2 = inter // scale_block
    qkv_cols = 1280 if world_size == 8 else 640
    gate_cols = 1024 if world_size == 8 else 512

    shapes = (
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
        jax.ShapeDtypeStruct((batch_size, o_in), jnp.bfloat16),
        jax.ShapeDtypeStruct((o_in, hidden), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((hidden, pre_pair_cols), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, pre_pair_cols), jnp.float32),
        jax.ShapeDtypeStruct((hidden, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, h_pre), jnp.float32),
        jax.ShapeDtypeStruct((1, 256), jnp.float32),
        jax.ShapeDtypeStruct((e_local, h_pre, 2 * inter), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((e_local, nqb1, 2 * inter), jnp.bfloat16),
        jax.ShapeDtypeStruct((e_local, inter, h_pre), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((e_local, nqb2, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, h_pre), jnp.bfloat16),
        # Bridge 2b
        jax.ShapeDtypeStruct((h_pre, h_pre), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, h_pre), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((1, hidden), jnp.float32),
        jax.ShapeDtypeStruct((hidden, qkv_cols), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, qkv_cols), jnp.float32),
        jax.ShapeDtypeStruct((hidden, gate_cols), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((1, gate_cols), jnp.float32),
        jax.ShapeDtypeStruct((batch_size, 2, 128), jnp.float32) if has_rope else None,
        jax.ShapeDtypeStruct((num_replicas, h_pre, 2 * inter), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((num_replicas, nqb1, 2 * inter), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_replicas, inter, h_pre), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((num_replicas, nqb2, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, world_size * num_replicas), jnp.int32),
    )

    in_specs = tuple(None if s is None else P() for s in shapes)
    out_specs = (P(), P(), P(), P()) if fuse_qkv_epi else (P(), P(), P())

    def _wrapped(*args):
        return b1_moe_b2_chip_sharded_jax(
            *args,
            world_size=world_size,
            fuse_qkv_epi=fuse_qkv_epi,
        )

    sm = jax.jit(
        jax.shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
    )
    comp = sm.trace(*shapes).lower(lowering_platforms=("tpu",)).compile()
    assert "tpu_custom_call" in comp.as_text()
    assert comp.memory_analysis().generated_code_size_in_bytes > 0


@pytest.mark.parametrize(
    "world_size,num_replicas,fuse_qkv_epi",
    [
        (8, 2, True),
        (16, 2, True),
    ],
)
def test_b1_moe_b2_superkernel_interpret_parity(
    world_size: int,
    num_replicas: int,
    fuse_qkv_epi: bool,
):
    """Verify InterpretParams(eager + detect_races + nan) parity for Step (b) [b1+MoE+b2]."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices("cpu")) < world_size:
        pytest.skip(f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}")

    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_chip_sharded_bridge_jax,
        moe_to_attn_chip_sharded_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _run_muse_sparse_from_vmem_core,
        build_muse_sparse_scratch_shapes,
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        _unpack_muse_sparse_scratch,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
    )

    batch_size = 64
    scale_block = 512
    hidden = 8192
    h_pre = 4096
    qkv_cols = 1280 if world_size == 8 else 640
    gate_cols = 1024 if world_size == 8 else 512
    sharded_moe_in = (batch_size // world_size) >= 8

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    mesh = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    rng = np.random.default_rng(20261125 + 1000 + world_size + num_replicas)
    d = _build_b1_moe_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        bf16_router=True,
        sparse_heavy=True,
    )
    w_post_tp2 = jnp.asarray(
        np.concatenate(
            [
                (rng.standard_normal((h_pre, h_pre)) * 0.04).astype(np.float32)
                for _ in range(world_size)
            ],
            axis=0,
        ),
        dtype=jnp.float8_e4m3fn,
    )
    s_post_tp2 = jnp.asarray((np.abs(rng.standard_normal((h_pre,))) * 0.02 + 1e-3).astype(np.float32))
    post_ff_gain = jnp.asarray((np.ones((hidden,)) * 1.02).astype(np.float32))
    pf_alpha = jnp.asarray((np.ones((hidden,)) * 0.93).astype(np.float32))
    pf_beta = jnp.asarray((np.ones((hidden,)) * 0.88).astype(np.float32))
    next_in_gain = jnp.asarray((np.ones((hidden,)) * 1.04).astype(np.float32))
    next_w_qkv = jnp.asarray(
        np.concatenate(
            [
                (rng.standard_normal((hidden, qkv_cols)) * 0.04).astype(np.float32)
                for _ in range(world_size)
            ],
            axis=0,
        ),
        dtype=jnp.float8_e4m3fn,
    )
    next_s_qkv = jnp.asarray((np.abs(rng.standard_normal((qkv_cols,))) * 0.02 + 1e-3).astype(np.float32))
    next_w_gate = jnp.asarray(
        np.concatenate(
            [
                (rng.standard_normal((hidden, gate_cols)) * 0.04).astype(np.float32)
                for _ in range(world_size)
            ],
            axis=0,
        ),
        dtype=jnp.float8_e4m3fn,
    )
    next_s_gate = jnp.asarray((np.abs(rng.standard_normal((gate_cols,))) * 0.02 + 1e-3).astype(np.float32))
    cos_sin_step = jnp.asarray((rng.standard_normal((batch_size, 2, 128)) * 0.3).astype(np.float32))

    cfg = dataclasses.replace(
        default_sparse_decode_config(batch_size, world_size, scale_block),
        m_buckets=(16, 32),
        route_chunk_rows=32,
        static_holder_mask=True,
        route_fast=True,
        fast_plan=2,
        unroll_strips=True,
        fast_gs=True,
        pair_rs=True,
        fast_b64_epi_pro=False,
        spec_prime_parts=1,
        k256_strip_mode="wide",
        nc_split512=False,
        min_p_topk=0.92,
        free_max_rows=4,
        free_max_w=0.20,
        free_min_slot=1,
    )

    def _run_standalone(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate, cos_sin,
        rep_w1, rep_s1, rep_w2, rep_s2, meta,
    ):
        st1, logits, expert_in = attn_to_moe_chip_sharded_bridge_jax(
            stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            interpret=ip_race,
        )
        cfg_ref = dataclasses.replace(cfg, spec_prime_parts=0)
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size, batch_size=batch_size, scale_block=scale_block,
            s1_dtype=s1.dtype, s2_dtype=s2.dtype, num_replicas=num_replicas,
            return_sharded=sharded_moe_in, config=cfg_ref,
        )
        v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
        h_sp = pl.BlockSpec(memory_space=pltpu.HBM)

        def _ref_moe(*refs):
            it = iter(refs)
            rk_r, mt_r = next(it), next(it)
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = next(it), next(it), next(it), next(it), next(it)
            rw1_r, rs1_r, rw2_r, rs2_r = next(it), next(it), next(it), next(it)
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it, ep=world_size, scale_block=scale_block,
                num_replicas=num_replicas, return_sharded=sharded_moe_in, cfg=cfg_ref,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(bsem, device_id={"model": jnp.int32(d_i)}, device_id_type=pl.DeviceIdType.MESH)
            _run_muse_sparse_from_vmem_core(
                rk_r[0], lg_r, bs_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                **{k: v for k, v in msc.items() if k != "w_mode"},
                barrier_sem=bsem, output_ref=out_r, ep=world_size, batch_size=batch_size,
                scale_block=scale_block, num_replicas=num_replicas, return_sharded=sharded_moe_in,
                replica_cost_mode="tiles", ep_axis_name="model", cfg=cfg_ref,
                rep_w1_u32=rw1_r, rep_s1_ref=rs1_r, rep_w2_u32=rw2_r, rep_s2_ref=rs2_r,
                replica_meta_ref=mt_r, wait_entry_barrier=True, write_output_ref=True,
            )

        out_r = (batch_size // world_size) if sharded_moe_in else batch_size
        moe_out = pl.pallas_call(
            _ref_moe,
            out_shape=jax.ShapeDtypeStruct((out_r, h_pre), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, grid=(1,),
                in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp, h_sp, h_sp, h_sp, h_sp),
                out_specs=v_sp, scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2,
                collective_id=0,
                disable_bounds_checks=True,
            ),
            interpret=ip_race,
        )(jnp.atleast_1d(rank), meta, logits, bias, expert_in, w1, s1, w2, s2, norm, rep_w1, rep_s1, rep_w2, rep_s2)

        return moe_to_attn_chip_sharded_bridge_jax(
            st1, moe_out, w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g,
            w_qkv, s_qkv, w_gate, s_gate, cos_sin,
            world_size=world_size, sharded_moe_in=sharded_moe_in, fuse_qkv_epi=fuse_qkv_epi,
            interpret=ip_race,
        )

    def _run_fused(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate, cos_sin,
        rep_w1, rep_s1, rep_w2, rep_s2, meta,
    ):
        return b1_moe_b2_chip_sharded_jax(
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate, cos_sin,
            rep_w1, rep_s1, rep_w2, rep_s2, meta,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            fuse_qkv_epi=fuse_qkv_epi, replica_cost_mode="tiles", config=cfg, interpret=ip_race,
        )

    call_args = (
        d["ranks"], d["stream_chip_st"], d["gated_full"], d["wo_sharded"], d["wo_scale"],
        d["pa_alpha"], d["pa_beta"], d["pre_ff_gain"],
        d["w_pre_pair"], d["s_pre_pair"], d["w_router"], d["pre_exp_gain"],
        d["bias"], d["w1_all"], d["s1_all"], d["w2_all"], d["s2_all"], d["post_exp_norm"],
        w_post_tp2, s_post_tp2, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv, next_s_qkv, next_w_gate, next_s_gate, cos_sin_step,
        d["rep_w1"], d["rep_s1"], d["rep_w2"], d["rep_s2"], d["replica_meta"],
    )
    in_specs = (
        P("model"), P("model", None), P(None, "model"), P("model", None), P(),
        P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
        P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(), P(),
        P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
    )
    out_specs = (P("model"), P("model"), P("model"), P("model"))

    ref_outs = jax.jit(jax.shard_map(_run_standalone, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)
    fused_outs = jax.jit(jax.shard_map(_run_fused, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)

    for idx, name in enumerate(("stream_chip", "q_5d", "kv_merged", "gate")):
        r_np = np.asarray(ref_outs[idx], dtype=np.float32)
        f_np = np.asarray(fused_outs[idx], dtype=np.float32)
        assert np.all(np.isfinite(f_np)), f"{name} contains non-finite values"
        assert _cosine_sim(r_np, f_np) > 0.9999, f"{name} cosine mismatch: {_cosine_sim(r_np, f_np)}"


def test_b1_moe_ops_env_default_ast_and_lock_free_prewarm():
    """Verify B1_MOE_B2_SUPERKERNEL defaults off, AST .get() hygiene, and lock-free prewarm."""
    import ast
    import inspect
    from pathlib import Path
    from unittest import mock

    from sglang_tpu.envs import envs
    from sglang_tpu.muse_megakernel.decode import b1_moe_ops as sk_ops

    assert envs.SGLANG_TPU_MUSE_DECODE_B1_MOE_B2_SUPERKERNEL.default is False

    repo_root = Path(__file__).resolve().parents[5]
    for rel_path in (
        "sglang_tpu/models/muse_spark.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_ops.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_b2_kernel.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_kernel.py",
    ):
        tree = ast.parse((repo_root / rel_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                src = ast.unparse(node.test)
                if (
                    "SGLANG_TPU_MUSE_DECODE_B1_MOE_B2_SUPERKERNEL" in src
                    and ("envs." in src or "tpu_envs." in src)
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
        captured[name] = (
            len(sig.parameters),
            input_partition_specs,
            output_partition_specs,
        )
        return fn

    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("_B1_MOE_LOCK entered after prewarm!")

        def __exit__(self, *args):
            return False

    layer_cfgs = [
        (0, False, "tiles", True),
        (8, True, "tiles", True),
    ]
    fake_mesh = object()
    with (
        mock.patch.object(sk_ops, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export),
        mock.patch.dict(sk_ops._B1_MOE_B2_OPS, {}, clear=True),
        mock.patch.dict(sk_ops._B1_MOE_OPS, {}, clear=True),
    ):
        for tp in (8, 16):
            sk_ops.prewarm_b1_moe_superkernel_ops(
                tp,
                layer_cfgs,
                bridge_qkv_epi=True,
                has_rope_variants=(False, True),
            )
        with mock.patch.object(sk_ops, "_B1_MOE_LOCK", _ExplodingLock()):
            for tp in (8, 16):
                for rep, sp, cm, r16 in layer_cfgs:
                    assert callable(
                        sk_ops.get_b1_moe_chip_sharded_op(
                            tp,
                            bf16_router=r16,
                            num_replicas=rep,
                            has_spec_slot=sp,
                            replica_cost_mode=cm,
                            return_sharded=True,
                        )
                    )
                    assert callable(
                        sk_ops.get_b1_moe_b2_chip_sharded_op(
                            tp,
                            bf16_router=r16,
                            num_replicas=rep,
                            has_spec_slot=sp,
                            replica_cost_mode=cm,
                            fuse_qkv_epi=False,
                            has_rope=False,
                        )
                    )
                    for hr in (False, True):
                        assert callable(
                            sk_ops.get_b1_moe_b2_chip_sharded_op(
                                tp,
                                bf16_router=r16,
                                num_replicas=rep,
                                has_spec_slot=sp,
                                replica_cost_mode=cm,
                                fuse_qkv_epi=True,
                                has_rope=hr,
                            )
                        )

    assert len(captured) == 16
    for op_name, (n_params, in_specs, out_specs) in captured.items():
        assert in_specs is not None and len(in_specs) == n_params, (
            f"{op_name}: {len(in_specs) if in_specs else None} != {n_params}"
        )
        assert out_specs is not None and len(out_specs) in (2, 3, 4)


def test_muse_spark_b1_moe_b2_superkernel_dispatch_tp8_and_tp16():
    """Verify MuseSparkTextModel/DecoderLayer Step (b) L0..L(N-2) & Step (a) L(N-1) dispatch."""
    import os
    import types
    from unittest import mock

    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.models.muse_spark import MuseSparkTextModel
    from test.registered.unit.hardware_backend.tpu.test_muse_w7_col_proj import (
        _make_dummy_layer,
        _make_norm,
    )

    h_dim, h_pre = 8192, 4096
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

    for tp in (8, 16):
        batch = 64
        r_pair = batch // (tp // 2)
        calls: list[str] = []

        def _fake_step_b(
            _rk, st, *_rest, fuse_epi=False
        ):
            calls.append(f"step_b:{st.shape[0]}:{fuse_epi}")
            qkv = torch.zeros((batch, 1280 // (tp // 8)), dtype=torch.bfloat16)
            gt = torch.zeros((batch, 1024 // (tp // 8)), dtype=torch.bfloat16)
            return st + 1.0, qkv, gt

        def _fake_step_a(_rk, st, *_rest):
            calls.append(f"step_a:{st.shape[0]}")
            return st + 1.0, torch.zeros((batch // tp, h_pre), dtype=torch.bfloat16)

        def _fake_fn_op(sh_flag):
            def _fn(st, mo, *_rest):
                calls.append(f"chip_fn:{sh_flag}:{st.shape[0]}:{mo.shape[0]}")
                return torch.ones((batch, h_dim), dtype=torch.bfloat16)

            return _fn

        def _fake_b1(st, ga, *_rest):
            calls.append(f"chip_b1:{st.shape[0]}:{ga.shape[0]}")
            return (
                st + 1.0,
                torch.zeros((batch, 256), dtype=torch.float32),
                torch.zeros((batch, h_pre), dtype=torch.bfloat16),
            )

        model = object.__new__(MuseSparkTextModel)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(hidden_size=h_dim, rms_norm_eps=1e-5)
        model.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.layers_to_capture = []
        model._capture_slots_by_layer = {}
        model.prefill_kernels = None
        model.rotary_emb = types.SimpleNamespace(cos_sin_pairs=None)
        model.start_layer = 0
        model.end_layer = 2
        layers = []
        for _ in range(2):
            lyr = _make_dummy_layer(3, H=h_dim, H_PRE=h_pre)
            lyr.self_attn.o_proj.tp_size = tp
            lyr.mlp.tp_size = tp
            lyr.self_attn._o_proj_w8k = torch.zeros(
                (1024 // (tp // 8), h_dim), dtype=torch.float8_e4m3fn
            )
            lyr.self_attn._o_proj_scale = torch.ones((h_dim,), dtype=torch.float32)
            lyr.self_attn.register_buffer(
                "_w7_wqkv_w8k",
                torch.zeros((h_dim, 1280 // (tp // 8)), dtype=torch.float8_e4m3fn),
                persistent=False,
            )
            lyr.self_attn.register_buffer(
                "_w7_sqkv", torch.ones((1280 // (tp // 8),), dtype=torch.float32), persistent=False
            )
            lyr.self_attn.register_buffer(
                "_w7_wgate_w8k",
                torch.zeros((h_dim, 1024 // (tp // 8)), dtype=torch.float8_e4m3fn),
                persistent=False,
            )
            lyr.self_attn.register_buffer(
                "_w7_sgate", torch.ones((1024 // (tp // 8),), dtype=torch.float32), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w7_wpre_w8k", torch.zeros((h_dim, h_pre // tp), dtype=torch.float8_e4m3fn), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w8_wpre_pair_w8k", torch.zeros((h_dim, h_pre // 2), dtype=torch.float8_e4m3fn), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w8_spre_pair", torch.ones((h_pre // 2,), dtype=torch.float32), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w7_wrouter_km", torch.zeros((h_dim, 256), dtype=torch.bfloat16), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w7_wpost_w8k", torch.zeros((h_pre, h_dim // tp), dtype=torch.float8_e4m3fn), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w7_spost", torch.ones((h_dim // tp,), dtype=torch.float32), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w9_wpost_pair_w8k", torch.zeros((h_pre, h_pre), dtype=torch.float8_e4m3fn), persistent=False
            )
            lyr.mlp.register_buffer(
                "_w9_spost_pair", torch.ones((h_pre,), dtype=torch.float32), persistent=False
            )
            lyr.mlp.register_buffer("_fused_decode_rank", torch.zeros((1,), dtype=torch.int32), persistent=False)
            lyr.mlp.register_buffer("_fused_decode_bias_2d", torch.zeros((1, 256), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_fused_decode_norm_2d", torch.ones((1, h_pre), dtype=torch.bfloat16), persistent=False)
            lyr.mlp._bias_2d = lambda l=lyr: l.mlp._buffers["_fused_decode_bias_2d"]
            lyr.mlp._norm_2d = lambda l=lyr: l.mlp._buffers["_fused_decode_norm_2d"]
            lyr.mlp._fused_decode_mode = "sparse"
            lyr.mlp._fused_decode_op = object()
            lyr.mlp._fused_decode_batches = (64, 128)
            lyr.mlp._fused_decode_num_replicas = 0
            lyr.mlp._fused_decode_has_spec_slot = False
            lyr.mlp._fused_decode_replica_cost_mode = "tiles"
            lyr.mlp.register_buffer(
                "_decode_fp4_w1", torch.empty((256 // tp, 4096, 4096), dtype=torch.float4_e2m1fn_x2), persistent=False
            )
            lyr.mlp.register_buffer(
                "_decode_fp4_s1", torch.ones((256 // tp, 8, 8192), dtype=torch.bfloat16), persistent=False
            )
            lyr.mlp.register_buffer(
                "_decode_fp4_w2", torch.empty((256 // tp, 4096, 2048), dtype=torch.float4_e2m1fn_x2), persistent=False
            )
            lyr.mlp.register_buffer(
                "_decode_fp4_s2", torch.ones((256 // tp, 8, 4096), dtype=torch.bfloat16), persistent=False
            )
            lyr._w9_b1_moe_b2_superkernel = True
            lyr.self_attn.forward = lambda pos, ai, fb_, **kw: torch.zeros(
                (batch, 1024 // (tp // 8)), dtype=torch.bfloat16
            )
            lyr.mlp.can_return_sharded_moe_out = lambda n_t, *, decode: n_t >= 8 * tp
            lyr.mlp.forward = lambda fh, ri, *, decode, return_unprojected_moe_out, return_sharded_moe_out, **kw: (
                torch.zeros(
                    (batch // tp if return_sharded_moe_out else batch, h_pre),
                    dtype=torch.bfloat16,
                )
            )
            layers.append(lyr)
        model.layers = torch.nn.ModuleList(layers)
        model.norm = _make_norm(h_dim)
        model._decode_norm_layout = 0
        model._decode_col_bridges = 3
        model._w9_chip_sharded_bridges = True
        model._w9_all_layers_chip_sharded_ready = True
        model._w9_chip_sharded_batches = (64, 128, 256, 384)
        model._w9_chip_sharded_tp = tp
        model._w8_pallas_embed_enabled = False
        model._w8_pallas_embed_tp = 0

        embeds = torch.randn(batch, h_dim, dtype=torch.bfloat16)
        pos = torch.arange(batch, dtype=torch.int32)
        par_stub = types.SimpleNamespace(tp_rank=3, tp_size=tp, attn_cp_size=1)
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": str(tp)}),
            mock.patch("sglang_tpu.models.muse_spark.get_parallel", return_value=par_stub),
            mock.patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=types.SimpleNamespace(sp_active=False),
            ),
            mock.patch(
                "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_chip_sharded_op",
                side_effect=lambda *_a, fuse_qkv_epi=False, **_kw: (
                    lambda *args: _fake_step_b(*args, fuse_epi=fuse_qkv_epi)
                ),
            ),
            mock.patch(
                "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_chip_sharded_op",
                return_value=_fake_step_a,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_attn_to_moe_chip_sharded_bridge_op",
                return_value=_fake_b1,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_post_chip_sharded_final_norm_op",
                side_effect=lambda _w, *, sharded_moe_in=True, **_kw: _fake_fn_op(sharded_moe_in),
            ),
        ):
            out = model.forward(
                torch.zeros((batch,), dtype=torch.int32),
                pos,
                fb,
                input_embeds=embeds,
            )
        assert out.shape == (batch, h_dim)
        if tp == 8:
            assert calls == [
                f"step_b:{r_pair}:False",
                f"step_a:{r_pair}",
                f"chip_fn:True:{r_pair}:{batch // tp}",
            ]
        else:
            assert calls == [
                f"step_b:{r_pair}:False",
                f"chip_b1:{r_pair}:{batch}",
                f"chip_fn:False:{r_pair}:{batch}",
            ]


def test_decode_4_to_1_pjrt_dispatch_collapse():
    """Verify CPU positions/kv_lens + compiled int64 lm_head_topk avoid eager TPU ops."""
    import types
    from unittest import mock

    import torch

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend
    from sglang_tpu.host_metadata_cache import HostMetadataCache
    from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration
    from sglang_tpu.sampler import TpuSampler

    # 1. ForwardBatch.init_new computes CPU int64 positions when _is_tpu=True.
    batch = types.SimpleNamespace(
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        seq_lens=torch.tensor([1, 5, 10], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([1, 5, 10], dtype=torch.int64),
        seq_lens_sum=16,
        req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([0, 1, 2], dtype=torch.int64),
        reqs=[],
        out_cache_loc=torch.tensor([0, 1, 2], dtype=torch.int64),
        extend_lens=None,
        prefix_lens=None,
        extend_num_tokens=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        orig_seq_lens=None,
        out_cache_loc_dsv4=None,
        engram_history=None,
        hisparse_coordinator=None,
        decoding_reqs=None,
        engram_staged_rows=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mamba_track_seqlens_cpu=None,
        mamba_prefill_track_mask_cpu=None,
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=None,
        encoder_lens=None,
        encoder_out_cache_loc=None,
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        positions=None,
        padded_spec_info=None,
        sampling_info=types.SimpleNamespace(
            has_watermark_candidates=False,
            grammars=None,
        ),
        return_logprob=False,
        top_logprobs_nums=None,
        custom_mask=None,
        spec_info=None,
        input_ids=torch.tensor([11, 22, 33], dtype=torch.int32),
        has_grammar=False,
        can_run_dp_cuda_graph=False,
        can_run_dp_breakable_cuda_graph=False,
        can_run_decode_cuda_graph=False,
        can_run_dp_prefill_cuda_graph=False,
        can_run_dp_draft_cuda_graph=False,
        dp_prefill_cuda_graph_max_prefix_len=None,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        global_forward_mode=ForwardMode.DECODE,
        is_prefill_only=False,
        spec_algorithm=None,
        tbo_split_seq_index=None,
        token_ids_logprobs=None,
        multimodal_inputs=None,
        mm_input_ids_cpu=None,
        encoder_cached=None,
        encoder_lens_cpu=None,
        dllm_config=None,
        dp_spec_prefill_coordination_applied=False,
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        dp_cooperation_info=None,
    )
    runner = types.SimpleNamespace(
        device="cpu",
        is_draft_worker=False,
        lora_manager=None,
        model_config=types.SimpleNamespace(
            model_is_mrope=False,
            is_matryoshka=False,
            requires_mm_token_modalities=False,
        ),
        ngram_embedding_manager=types.SimpleNamespace(enabled=False),
        runner_ids=types.SimpleNamespace(
            own_plan=lambda _fb: types.SimpleNamespace(bind=lambda *_a, **_kw: None)
        ),
    )
    with (
        mock.patch("sglang.srt.model_executor.forward_batch_info._is_tpu", True),
        mock.patch(
            "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.model_executor.forward_batch_info.get_parallel",
            return_value=types.SimpleNamespace(attn_dcp_size=1),
        ),
        mock.patch(
            "sglang.srt.model_executor.forward_batch_info.clamp_position",
            side_effect=AssertionError("clamp_position must not run on device seq_lens"),
        ),
    ):
        fb = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
    assert fb.positions.device.type == "cpu"
    assert fb.positions.dtype == torch.int64
    assert torch.equal(fb.positions, torch.tensor([0, 4, 9], dtype=torch.int64))

    # 2. BaseTpuPallasRpaAttnBackend._build_metadata_host uploads CPU kv_lens without device cast.
    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend._dcp = 1
    backend._host_metadata_cache = HostMetadataCache()
    backend._host_metadata_cache.get_by_value = (
        lambda name, value, device, **_kw: value.clone()
    )
    backend._host_page_cols = None
    backend._page_stride = 1
    backend._is_hybrid_swa = False
    backend._swa_group_pages = None
    backend._kv_page_size = 512
    backend._layout_metadata = lambda *, build, **_kw: build()
    backend.supports_shared_decode_batch = lambda _bs: False
    backend.shared_decode_group_size = lambda _bs: 0
    backend.token_to_kv_pool = types.SimpleNamespace(full_to_swa_index_mapping=None)
    fake_tpu_dev = types.SimpleNamespace(type="tpu")

    class _NoDeviceCastSeqLens:
        dtype = torch.int64

        def to(self, *_a, **_kw):
            raise AssertionError("forward_batch.seq_lens.to(int32) must not be called")

    rpa_fb = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens=_NoDeviceCastSeqLens(),
        seq_lens_cpu=torch.tensor([12, 40], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([0, 1, 0, 0], dtype=torch.int64),
    )
    req_to_token = torch.zeros((2, 1024), dtype=torch.int32)
    with mock.patch(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False
    ):
        meta = backend._build_metadata_host(
            rpa_fb,
            512,
            fake_tpu_dev,
            4,
            req_to_token,
            2,
            torch.tensor([0, 0, 2], dtype=torch.int32),
            2,
        )
    assert meta.kv_lens.dtype == torch.int32
    assert meta.kv_lens.tolist() == [12, 40, 0, 0]

    # 3. LogitsProcessor.forward (decode) and _forward_pruned_logits return int64 ids
    # and TpuSampler skips .to(torch.int64).
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
    from sglang.srt.model_executor.forward_batch_info import LogitsOutputMode

    lp_stub = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(lp_stub)
    lp_stub.use_tpu_distributed_top_k = True
    lp_stub.compute_input_logprobs = False
    lp_stub.expected_aux_hidden_states = None
    lp_stub.do_tensor_parallel_all_gather = True
    lp_stub.do_tensor_parallel_all_gather_dp_attn = False
    lp_stub.use_attn_tp_group = False
    lp_stub.logit_scale = 1.0
    lp_stub.final_logit_softcapping = None
    lp_stub._get_pruned_states = (
        lambda hs, hs_bn, aux_hs, _lm: (hs, hs_bn, aux_hs, None, None, None)
    )
    lp_stub._get_hidden_states_to_store = lambda *_a: None
    lp_stub._get_last_hidden_states_to_store = lambda *_a: None
    lp_stub._batch_vocab_shards = lambda _fb: None
    lp_stub._get_distributed_gumbel_max = lambda *_a: (
        torch.tensor([[1.5], [2.5]], dtype=torch.float32),
        torch.tensor([[17], [42]], dtype=torch.int32),
    )
    lp_stub._get_distributed_top_k = lambda *_a: (
        torch.tensor([[1.5], [2.5]], dtype=torch.float32),
        torch.tensor([[99], [7]], dtype=torch.int32),
    )
    model_stub = types.SimpleNamespace(
        logits_processor=lp_stub,
        lm_head=object(),
    )
    dummy_meta = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        extend_return_logprob=False,
        mm_input_embeds=None,
    )
    for lmode, exp_ids in (
        (LogitsOutputMode.DISTRIBUTED_GUMBEL_MAX, torch.tensor([[17], [42]], dtype=torch.int64)),
        (LogitsOutputMode.DISTRIBUTED_GREEDY, torch.tensor([[99], [7]], dtype=torch.int64)),
    ):
        fb_dec = object.__new__(ForwardBatch)
        fb_dec.logits_output_mode = lmode
        fb_dec.gumbel_temperatures = torch.ones((2, 1), dtype=torch.float32)
        fb_dec.multi_item_delimiter_indices = None
        fb_dec.token_indices_to_pool = None
        fb_dec.forward_mode = ForwardMode.DECODE
        fb_dec.capture_hidden_mode = CaptureHiddenMode.NULL
        fb_dec.return_logprob = False
        fb_dec.top_logprobs_nums = None
        fb_dec.token_ids_logprobs = None
        fb_dec.extend_seq_lens_cpu = None
        fb_dec.mm_inputs = None
        fb_dec.mm_input_embeds = None
        with (
            mock.patch(
                "sglang_tpu.sampler.distributed_top_k_candidate_width",
                return_value=64,
            ),
            mock.patch.object(
                LogitsMetadata, "from_forward_batch", return_value=dummy_meta
            ),
        ):
            lout_lp = lp_stub.forward(
                torch.zeros((2,), dtype=torch.int32),
                torch.zeros((2, 64), dtype=torch.bfloat16),
                model_stub.lm_head,
                fb_dec,
            )
            lout_pruned = MuseSparkForConditionalGeneration._forward_pruned_logits(
                model_stub,
                torch.zeros((2, 64), dtype=torch.bfloat16),
                fb_dec,
            )
        for lout in (lout_lp, lout_pruned):
            assert lout.next_token_global_top_k_token_ids.dtype == torch.int64
            assert torch.equal(lout.next_token_global_top_k_token_ids, exp_ids)

        class _NoInt64CastTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

            def to(self, *args, **kwargs):
                if args and args[0] == torch.int64:
                    raise AssertionError("TpuSampler should not cast int64 token ids")
                return super().to(*args, **kwargs)

        sampler = types.SimpleNamespace(_sync_token_ids_across_tp=lambda *_a: None)
        s_out = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=lout_lp.next_token_global_top_k_logits,
            next_token_global_top_k_token_ids=_NoInt64CastTensor(
                lout_lp.next_token_global_top_k_token_ids
            ),
        )
        sampled = TpuSampler.forward(
            sampler,
            s_out,
            types.SimpleNamespace(
                sampling_params_on_cpu=False,
                is_all_greedy=True,
                sampling_seed=None,
            ),
            False,
            None,
            None,
            None,
        )
        assert sampled.dtype == torch.int64
        assert torch.equal(sampled, exp_ids.view(-1))

