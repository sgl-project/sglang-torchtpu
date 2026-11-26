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
    assert envs.SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL.default is False

    repo_root = Path(__file__).resolve().parents[5]
    for rel_path in (
        "sglang_tpu/models/muse_spark.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_ops.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_b2_kernel.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_b2_b16_kernel.py",
        "sglang_tpu/muse_megakernel/decode/b1_moe_kernel.py",
    ):
        tree = ast.parse((repo_root / rel_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                src = ast.unparse(node.test)
                if (
                    (
                        "SGLANG_TPU_MUSE_DECODE_B1_MOE_B2_SUPERKERNEL" in src
                        or "SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL" in src
                    )
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
                    for ret_sh in ((True, False) if tp == 16 else (True,)):
                        assert callable(
                            sk_ops.get_b1_moe_chip_sharded_op(
                                tp,
                                bf16_router=r16,
                                num_replicas=rep,
                                has_spec_slot=sp,
                                replica_cost_mode=cm,
                                return_sharded=ret_sh,
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

    assert len(captured) == 18
    for op_name, (n_params, in_specs, out_specs) in captured.items():
        assert in_specs is not None and len(in_specs) == n_params, (
            f"{op_name}: {len(in_specs) if in_specs else None} != {n_params}"
        )
        assert out_specs is not None and len(out_specs) in (2, 3, 4)

    # Verify Step (c) prewarm covers both has_layer_idx in (False, True) and kv_pack_dense2 in (False, True)
    # matching muse_spark._prepare_decode_collective_bridges step_c_cfgs for 1P1D block-major KV.
    ms_src = (repo_root / "sglang_tpu/models/muse_spark.py").read_text()
    assert "(True, False, rpa_m)" in ms_src and "(True, True, rpa_m)" in ms_src
    step_c_cfgs = (
        (False, False, "w8_noz"),
        (False, True, "w8_noz"),
        (True, False, "w8_noz"),
        (True, True, "w8_noz"),
        (False, False, "w8_noz", True),
        (True, False, "w8_noz", True),
    )
    with (
        mock.patch.object(sk_ops, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export),
        mock.patch.dict(sk_ops._B1_MOE_B2_OPS, {}, clear=True),
        mock.patch.dict(sk_ops._B1_MOE_OPS, {}, clear=True),
        mock.patch.dict(sk_ops._B1_MOE_B2_RPAD_OPS, {}, clear=True),
        mock.patch.dict(sk_ops._STEP_C_KV_WRITE_OPS, {}, clear=True),
        mock.patch.dict(sk_ops._STEP_C_KV_WRITE_MULTI_6D_OPS, {}, clear=True),
        mock.patch.dict(sk_ops._STEP_C_KV_WRITE_MULTI_5D_OPS, {}, clear=True),
    ):
        sk_ops.prewarm_b1_moe_superkernel_ops(
            16,
            [(0, False, "tiles", True)],
            bridge_qkv_epi=True,
            has_rope_variants=(True,),
            has_free_params=True,
            step_c_configs=step_c_cfgs,
        )
        with mock.patch.object(sk_ops, "_B1_MOE_LOCK", _ExplodingLock()):
            for sc_item in step_c_cfgs:
                if len(sc_item) == 4:
                    has_lidx, dense2, rpa_m, bd4 = sc_item
                else:
                    has_lidx, dense2, rpa_m = sc_item
                    bd4 = False
                assert (
                    sk_ops.get_step_c_kv_write_op(
                        kv_pack_dense2=bool(dense2),
                        kv_pack_bd4=bool(bd4),
                    )
                    is not None
                )
                if has_lidx:
                    assert (
                        sk_ops.get_step_c_kv_write_multi_6d_op(
                            kv_pack_dense2=bool(dense2),
                            kv_pack_bd4=bool(bd4),
                        )
                        is not None
                    )
                else:
                    for _k5 in range(2, 9):
                        assert (
                            sk_ops.get_step_c_kv_write_multi_5d_op(
                                _k5,
                                kv_pack_dense2=bool(dense2),
                                kv_pack_bd4=bool(bd4),
                            )
                            is not None
                        )
                assert (
                    sk_ops.get_b1_moe_b2_rpad_chip_sharded_op(
                        16,
                        bf16_router=True,
                        num_replicas=0,
                        has_spec_slot=False,
                        replica_cost_mode="tiles",
                        has_rope=True,
                        has_free_params=True,
                        has_layer_idx=has_lidx,
                        kv_pack_dense2=dense2,
                        kv_pack_bd4=bd4,
                        rpa_mode_str=rpa_m,
                    )
                    is not None
                )


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

    for tp, batch in ((8, 64), (16, 64), (16, 256)):
        r_pair = batch // (tp // 2)
        calls: list[str] = []

        def _fake_step_b(
            _rk, st, *_rest, fuse_epi=False
        ):
            calls.append(f"step_b:{st.shape[0]}:{fuse_epi}")
            qkv = torch.zeros((batch, 1280 // (tp // 8)), dtype=torch.bfloat16)
            gt = torch.zeros((batch, 1024 // (tp // 8)), dtype=torch.bfloat16)
            return st + 1.0, qkv, gt

        def _fake_step_a(_rk, st, *_rest, ret_sh=True):
            calls.append(f"step_a:{st.shape[0]}")
            out_r = (batch // tp) if ret_sh else batch
            return st + 1.0, torch.zeros((out_r, h_pre), dtype=torch.bfloat16)

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
            lyr.mlp._fused_decode_batches = (64, 128, 256)
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
                side_effect=lambda *_a, return_sharded=True, **_kw: (
                    lambda *args, _rs=return_sharded: _fake_step_a(*args, ret_sh=_rs)
                ),
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
        if tp == 8 or batch == 256:
            assert calls == [
                f"step_b:{r_pair}:False",
                f"step_a:{r_pair}",
                f"chip_fn:True:{r_pair}:{batch // tp}",
            ]
        else:
            assert calls == [
                f"step_b:{r_pair}:False",
                f"step_a:{r_pair}",
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

    # 1b. #3949 hardening: non-Tensor seq_lens_cpu falls back cleanly to clamp_position,
    # and enable_torch_compile=False uploads CPU positions to runner.device.
    batch.seq_lens_cpu = [1, 5, 10]
    clamp_calls = []
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
            side_effect=lambda s: clamp_calls.append(s) or torch.clamp(s - 1, min=0),
        ),
    ):
        fb_list = ForwardBatch.init_new(
            batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
    assert len(clamp_calls) == 1
    assert torch.equal(fb_list.positions, torch.tensor([0, 4, 9], dtype=torch.int64))

    batch.seq_lens_cpu = torch.tensor([2, 6, 11], dtype=torch.int64)
    runner_eager = types.SimpleNamespace(
        device="cpu",
        is_draft_worker=False,
        lora_manager=None,
        server_args=types.SimpleNamespace(enable_torch_compile=False),
        model_config=runner.model_config,
        ngram_embedding_manager=runner.ngram_embedding_manager,
        runner_ids=runner.runner_ids,
    )
    upload_calls = []
    orig_to = torch.Tensor.to

    def _spy_to(self_t, *args, **kwargs):
        if args and args[0] == "cpu" and kwargs.get("non_blocking") is True:
            upload_calls.append(kwargs)
        return orig_to(self_t, *args, **kwargs)

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
            side_effect=AssertionError("clamp_position must not run when seq_lens_cpu is Tensor"),
        ),
        mock.patch.object(torch.Tensor, "to", _spy_to),
    ):
        fb_eager = ForwardBatch.init_new(
            batch,
            runner_eager,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
    assert len(upload_calls) >= 1
    assert torch.equal(fb_eager.positions, torch.tensor([1, 5, 10], dtype=torch.int64))

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


def _make_step_c_rpad_inputs(
    rng: np.random.Generator,
    *,
    world_size: int,
    batch_size: int,
    sliding_window: int | None,
    pages_per_seq: int = 4,
    page_size: int = 128,
    n_active: int | None = None,
    poison_nan: bool = False,
):
    import jax
    import jax.numpy as jnp

    n_kv = 1
    n_act = batch_size if n_active is None else n_active
    kv_len_max = (sliding_window - 1) if sliding_window is not None else (pages_per_seq * page_size)
    kv_lens_np = rng.integers(16, min(kv_len_max, pages_per_seq * page_size), size=(batch_size,), dtype=np.int32)
    kv_lens_np = np.clip(kv_lens_np, 1, pages_per_seq * page_size - 1)
    if n_act < batch_size:
        kv_lens_np[n_act:] = 0
    num_blocks = batch_size * pages_per_seq + 8
    page_indices_np = np.arange(batch_size * pages_per_seq, dtype=np.int32) % num_blocks
    if poison_nan:
        kv_u8 = np.full((num_blocks, page_size, n_kv, 4, 128), 0x7F, dtype=np.uint8)
        rng.shuffle(page_indices_np)
        pi_2d = page_indices_np.reshape(batch_size, pages_per_seq)
        for seq in range(n_act):
            old_len = int(kv_lens_np[seq]) - 1
            for p_o in range((old_len + page_size - 1) // page_size):
                pg = int(pi_2d[seq, p_o])
                rows_p = min(page_size, old_len - p_o * page_size)
                rand_f8 = jnp.asarray((rng.standard_normal((rows_p, 2, 128)) * 0.25).astype(np.float32), jnp.float8_e4m3fn)
                kv_u8[pg, :rows_p, 0, :2, :] = np.asarray(jax.lax.bitcast_convert_type(rand_f8, jnp.uint8))
        kv_cache = jax.lax.bitcast_convert_type(jnp.asarray(kv_u8, dtype=jnp.uint8), jnp.float8_e4m3fn)
    else:
        kv_cache_np = (rng.standard_normal((num_blocks, page_size, n_kv, 4, 128)) * 0.25).astype(np.float32)
        kv_cache = jnp.asarray(kv_cache_np, dtype=jnp.float8_e4m3fn)
    kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    page_indices = jnp.asarray(page_indices_np, dtype=jnp.int32)
    cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32)
    cu_kv_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * (pages_per_seq * page_size)
    distribution = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
    return kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution


def _run_step_c_parity_and_aot(
    *,
    world_size: int,
    batch_size: int,
    sliding_window: int | None,
    rpa_mode: str,
    seed: int,
    page_size: int = 128,
    n_active: int | None = None,
    poison_nan: bool = False,
    skip_kv_write: bool = False,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices("cpu")) < world_size:
        pytest.skip(f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}")

    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_chip_sharded_bridge_jax,
        moe_to_attn_chip_sharded_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _run_muse_sparse_from_vmem_core,
        build_muse_sparse_scratch_shapes,
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_rpad_chip_sharded_jax,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import _unpack_muse_sparse_scratch

    hidden = 8192
    h_pre = 4096
    qkv_cols = 640 if world_size == 16 else 1280
    gate_cols = 512 if world_size == 16 else 1024
    scale_block = 512
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(seed)
    d = _build_b1_moe_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=0,
        bf16_router=True,
        sparse_heavy=True,
    )
    w_post_tp2 = jnp.asarray(
        rng.standard_normal((world_size * h_pre, h_pre)).astype(np.float32) * 0.04,
        dtype=jnp.float8_e4m3fn,
    )
    s_post_tp2 = jnp.asarray(
        np.abs(rng.standard_normal((h_pre,))).astype(np.float32) * 0.02 + 1e-3
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
        np.abs(rng.standard_normal((qkv_cols,))).astype(np.float32) * 0.02 + 1e-3
    )
    next_w_gate = jnp.asarray(
        rng.standard_normal((world_size * hidden, gate_cols)).astype(np.float32) * 0.04,
        dtype=jnp.float8_e4m3fn,
    )
    next_s_gate = jnp.asarray(
        np.abs(rng.standard_normal((gate_cols,))).astype(np.float32) * 0.02 + 1e-3
    )
    cos_sin_step = jnp.asarray((rng.standard_normal((batch_size, 2, 128)) * 0.3).astype(np.float32))
    kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution = _make_step_c_rpad_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        sliding_window=sliding_window,
        page_size=page_size,
        n_active=n_active,
        poison_nan=poison_nan,
    )
    free_params = jnp.asarray([0.84, 2.0, 0.20, 1.0], dtype=jnp.float32)
    sharded_moe_in = (batch_size // world_size) >= 8 and (batch_size // world_size) % 8 == 0
    cfg = dataclasses.replace(
        default_sparse_decode_config(batch_size, world_size, scale_block),
        weight_mode="stream3",
        pair_rs=True,
        bridge_ag=(world_size == 16),
        m_buckets=(16, 32),
        route_chunk_rows=32,
        static_holder_mask=True,
        route_fast=True,
        fast_plan=2,
        unroll_strips=True,
        fast_gs=True,
        fast_b64_epi_pro=False,
        spec_prime_parts=0,
        k256_strip_mode="wide",
        nc_split512=False,
    )

    def _run_standalone(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
        kv_c, kvl, pg, cq, ckv, dist, cos_sin, fp,
    ):
        st1, logits, expert_in = attn_to_moe_chip_sharded_bridge_jax(
            stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            interpret=ip_race,
        )
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size, batch_size=batch_size, scale_block=scale_block,
            s1_dtype=s1.dtype, s2_dtype=s2.dtype, num_replicas=0,
            return_sharded=sharded_moe_in, config=cfg,
        )
        v_sp, h_sp = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.HBM)

        def _ref_moe(*refs):
            it = iter(refs)
            rk_r, fp_r = next(it), next(it)
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = next(it), next(it), next(it), next(it), next(it)
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it, ep=world_size, scale_block=scale_block,
                num_replicas=0, return_sharded=sharded_moe_in, cfg=cfg,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(bsem, device_id={"model": jnp.int32(d_i)}, device_id_type=pl.DeviceIdType.MESH)
            _run_muse_sparse_from_vmem_core(
                rk_r[0], lg_r, bs_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                **{k: v for k, v in msc.items() if k != "w_mode"},
                barrier_sem=bsem, output_ref=out_r, ep=world_size, batch_size=batch_size,
                scale_block=scale_block, num_replicas=0, return_sharded=sharded_moe_in,
                replica_cost_mode="tiles", ep_axis_name="model", cfg=cfg,
                free_params_ref=fp_r, wait_entry_barrier=True, write_output_ref=True,
            )

        out_r = (batch_size // world_size) if sharded_moe_in else batch_size
        moe_out = pl.pallas_call(
            _ref_moe,
            out_shape=jax.ShapeDtypeStruct((out_r, h_pre), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, grid=(1,),
                in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                out_specs=v_sp, scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2, collective_id=0, disable_bounds_checks=True,
            ),
            interpret=ip_race,
        )(jnp.atleast_1d(rank), fp, logits, bias, expert_in, w1, s1, w2, s2, norm)

        st2, q_5d, kv_m, gate_2d = moe_to_attn_chip_sharded_bridge_jax(
            st1, moe_out, w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g,
            w_qkv, s_qkv, w_gate, s_gate, cos_sin,
            world_size=world_size, sharded_moe_in=sharded_moe_in, fuse_qkv_epi=True,
            interpret=ip_race,
        )
        gated_out, updated_kv = run_decode_fast_rpa(
            q_5d, kv_m, kv_m, kv_c, kvl, pg, cq, ckv, dist,
            gate=gate_2d, sm_scale=0.125, sliding_window=sliding_window,
            k_scale=0.9, v_scale=1.1, mode_str=rpa_mode,
            prepared_inputs=True, fused_out_pack=2, out_norm_eps=1e-5,
            kv_pack_dense2=False, interpret=True,
        )
        return st2, gated_out.reshape(batch_size, gate_cols), updated_kv

    def _run_fused(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
        kv_c, kvl, pg, cq, ckv, dist, cos_sin, fp,
    ):
        st2, ng, kv_out = b1_moe_b2_rpad_chip_sharded_jax(
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            kv_c, kvl, pg, cq, ckv, dist,
            cos_sin_step=cos_sin, free_params=fp,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            rpa_sm_scale=0.125, rpa_sliding_window=sliding_window,
            rpa_k_scale=0.9, rpa_v_scale=1.1, rpa_out_norm_eps=1e-5,
            rpa_mode_str=rpa_mode, kv_pack_dense2=False,
            replica_cost_mode="tiles", config=cfg, interpret=ip_race,
            skip_kv_write=skip_kv_write,
        )
        if skip_kv_write:
            from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
                write_decode_kv_merged_jax,
            )

            kv_out = write_decode_kv_merged_jax(
                kv_out, kv_c, kvl, pg, cq, ckv, dist, kv_pack_dense2=False, interpret=ip_race,
            )
        return st2, ng, kv_out

    mesh = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    call_args = (
        d["ranks"], d["stream_chip_st"], d["gated_full"], d["wo_sharded"], d["wo_scale"],
        d["pa_alpha"], d["pa_beta"], d["pre_ff_gain"],
        d["w_pre_pair"], d["s_pre_pair"], d["w_router"], d["pre_exp_gain"],
        d["bias"], d["w1_all"], d["s1_all"], d["w2_all"], d["s2_all"], d["post_exp_norm"],
        w_post_tp2, s_post_tp2, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv, next_s_qkv, next_w_gate, next_s_gate,
        kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution,
        cos_sin_step, free_params,
    )
    in_specs = (
        P("model"), P("model", None), P(None, "model"), P("model", None), P(),
        P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
        P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(),
        P(), P(), P(), P(), P(), P(), P(), P(),
    )
    out_specs = (P("model"), P("model"), P("model"))

    ref_outs = jax.jit(jax.shard_map(_run_standalone, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)
    fused_outs = jax.jit(jax.shard_map(_run_fused, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)

    for idx, name in enumerate(("stream_chip", "next_gated", "updated_kv")):
        r_raw = np.asarray(ref_outs[idx])
        f_raw = np.asarray(fused_outs[idx])
        if idx == 2 and poison_nan:
            r_u8 = r_raw.view(np.uint8)
            f_u8 = f_raw.view(np.uint8)
            assert np.array_equal(r_u8 == 0x7F, f_u8 == 0x7F), "0x7F NaN poisoning mask changed"
            valid_mask = r_u8 != 0x7F
            r_np = r_raw.astype(np.float32)[valid_mask]
            f_np = f_raw.astype(np.float32)[valid_mask]
        else:
            r_np = r_raw.astype(np.float32)
            f_np = f_raw.astype(np.float32)
        assert np.all(np.isfinite(f_np)), f"{name} non-finite"
        assert _cosine_sim(r_np, f_np) > 0.9999, f"{name} cosine mismatch: {_cosine_sim(r_np, f_np)}"

    # TPU7x AOT compile check (0 hardware execution required).
    topo_name = "tpu7x:2x2x1" if world_size == 8 else "tpu7x:2x2x2"
    tpu_devs = topologies.get_topology_desc(topo_name, platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    abstract = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in call_args)

    def _aot_step_c(*args):
        (
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            kv_c, kvl, pg, cq, ckv, dist, cos_sin, fp,
        ) = args
        return b1_moe_b2_rpad_chip_sharded_jax(
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            kv_c, kvl, pg, cq, ckv, dist,
            cos_sin_step=cos_sin, free_params=fp,
            world_size=world_size, rpa_sliding_window=sliding_window,
            rpa_sm_scale=0.125, rpa_k_scale=0.9, rpa_v_scale=1.1, rpa_out_norm_eps=1e-5,
            rpa_mode_str=rpa_mode, kv_pack_dense2=False,
            replica_cost_mode="tiles", config=cfg, interpret=False,
            skip_kv_write=skip_kv_write,
        )

    compiled = jax.jit(
        jax.shard_map(_aot_step_c, mesh=mesh_aot, in_specs=in_specs, out_specs=out_specs, check_vma=False),
        donate_argnums=(() if skip_kv_write else (28,)),
    ).lower(*abstract).compile()
    assert compiled is not None


@pytest.mark.parametrize(
    "world_size,batch_size,sliding_window",
    [
        (8, 32, 2049),
        (16, 64, 2049),
    ],
)
def test_step_c_b1_moe_b2_rpad_interpret_parity_and_tpu7x_aot(
    world_size: int, batch_size: int, sliding_window: int | None
):
    """Verify Step (c) [chip_b1 + MoE(3d) + chip_b2 + RPAd] Interpret parity + TPU7x AOT."""
    _run_step_c_parity_and_aot(
        world_size=world_size,
        batch_size=batch_size,
        sliding_window=sliding_window,
        rpa_mode="swabatch_tp16" if world_size == 16 else "w8_noz",
        seed=20261130 + world_size + batch_size,
    )


@pytest.mark.parametrize("world_size", [8, 16])
def test_step_c_2scope_batch4_interpret_races_and_tpu7x_aot(world_size: int):
    """Verify 2-scope Step (c) + SWA_b4_s4 (80pg bkv) race-free Interpret parity + TPU7x AOT."""
    _run_step_c_parity_and_aot(
        world_size=world_size,
        batch_size=64,
        sliding_window=2049,
        rpa_mode="w8,s4,batchfull,g4k,foldcol,batch4",
        seed=20261160 + world_size,
        page_size=512,
        n_active=59,
        poison_nan=True,
        skip_kv_write=True,
    )


def test_step_c_rpa_plan_group_sizing_and_fallback():
    """Verify resolve_step_c_rpa_plan 4->2->1 batch grouping fallback and VMEM ceiling checks."""
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import resolve_step_c_rpa_plan

    # 1. Full b4_s4 (4*4*5*512 = 80*512) fits 2-scope Step (c).
    cfg_4, one_blk, slot_pg, use_grp, is_ok = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=512,
        kv_pack_dense2=False,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert (one_blk, slot_pg, use_grp, is_ok) == (True, 20, True, True)
    assert cfg_4.swabatch_group_size == 4

    # 2. Graceful 4 -> 2 fallback when batch tokens only divisible by 2 (B >= 32).
    cfg_2, one_blk_2, slot_pg_2, use_grp_2, is_ok_2 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=512,
        kv_pack_dense2=False,
        b_tokens=34,
        max_num_seqs=34,
    )
    assert (one_blk_2, slot_pg_2, use_grp_2, is_ok_2) == (True, 10, True, True)
    assert cfg_2.swabatch_group_size == 2

    # 3. Graceful fallback to 1 when b_tokens != max_num_seqs or kv_pack_dense2=True.
    cfg_1a, _, slot_pg_1a, use_grp_1a, is_ok_1a = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=512,
        kv_pack_dense2=False,
        b_tokens=32,
        max_num_seqs=64,
    )
    assert (slot_pg_1a, use_grp_1a, is_ok_1a) == (5, False, True)
    assert cfg_1a.swabatch_group_size == 1

    cfg_1b, _, slot_pg_1b, use_grp_1b, is_ok_1b = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4,dense2",
        sliding_window=2049,
        pages_per_seq=2,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert (slot_pg_1b, use_grp_1b, is_ok_1b) == (5, False, True)
    assert cfg_1b.swabatch_group_size == 1

    # 4. Base > 80*512 rows or active shared16 must return is_ok=False.
    _, _, _, _, ok_too_big = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol",
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=512,
        kv_pack_dense2=False,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert ok_too_big is False

    _, _, _, _, ok_shared = resolve_step_c_rpa_plan(
        "shared16",
        sliding_window=None,
        pages_per_seq=4,
        phys_page_rows=512,
        has_shared_prefix=True,
    )
    assert ok_shared is False


@pytest.mark.parametrize("has_layer_idx", [False, True])
def test_step_c_kv_cache_alias_zero_copy_hlo(has_layer_idx: bool):
    """Verify 3-level kv_cache aliasing emits ZERO HLO copy ops on kv_cache."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from unittest import mock

    from sglang_tpu.muse_megakernel.decode import b1_moe_ops
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    world_size = 8
    batch_size = 32
    hidden = 8192
    h_pre = 4096
    r_pair = batch_size // (world_size // 2)
    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )

    captured = {}
    import torch
    orig_custom_op = torch.library.custom_op

    def _capture_custom_op(name, fn, *, mutates_args=(), device_types=None):
        captured["name"] = name
        captured["fn"] = fn
        captured["mutates_args"] = mutates_args
        return orig_custom_op(name, fn, mutates_args=mutates_args, device_types=device_types)

    with (
        mock.patch.object(b1_moe_ops, "_get_tp_mesh", return_value=mesh_aot),
        mock.patch("torch.library.custom_op", side_effect=_capture_custom_op),
    ):
        b1_moe_ops._B1_MOE_B2_RPAD_OPS.clear()
        op = b1_moe_ops.get_b1_moe_b2_rpad_chip_sharded_op(
            world_size,
            bf16_router=True,
            num_replicas=0,
            has_spec_slot=False,
            replica_cost_mode="tiles",
            has_rope=True,
            has_free_params=True,
            has_layer_idx=has_layer_idx,
            kv_pack_dense2=False,
            rpa_mode_str="w8_noz",
        )
    assert captured["mutates_args"] == ()
    wrapped = captured["fn"]
    assert not wrapped.donate_argnums
    assert not wrapped.input_output_aliases

    kv_shape = (64, 2, 128, 1, 4, 128) if has_layer_idx else (128, 128, 1, 4, 128)
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    jax_args = [
        s((1,), jnp.int32),  # rank
        s((r_pair, hidden), jnp.float32),  # stream_chip
        s((batch_size, 1024), jnp.bfloat16),  # gated_attn
        s((1024, hidden), jnp.float8_e4m3fn),  # wo_w8k
        s((hidden,), jnp.float32),  # wo_scale
        s((hidden,), jnp.float32),  # pa_alpha
        s((hidden,), jnp.float32),  # pa_beta
        s((hidden,), jnp.float32),  # pre_ff_gain
        s((hidden, h_pre // 2), jnp.float8_e4m3fn),  # w_pre_pair_w8k
        s((h_pre // 2,), jnp.float32),  # w_pre_pair_scale
        s((hidden, 256), jnp.bfloat16),  # w_router_km
        s((h_pre,), jnp.float32),  # pre_exp_gain
        s((1, 256), jnp.float32),  # bias
        s((32, 512, 8192), jnp.uint32),  # w1
        s((32, 8, 8192), jnp.bfloat16),  # s1
        s((32, 512, 4096), jnp.uint32),  # w2
        s((32, 8, 4096), jnp.bfloat16),  # s2
        s((1, h_pre), jnp.float32),  # post_exp_norm
        s((h_pre, h_pre), jnp.float8_e4m3fn),  # w_post_tp2_w8k
        s((h_pre,), jnp.float32),  # s_post_tp2
        s((hidden,), jnp.float32),  # post_ff_gain
        s((hidden,), jnp.float32),  # pf_alpha
        s((hidden,), jnp.float32),  # pf_beta
        s((hidden,), jnp.float32),  # next_in_gain
        s((hidden, 1280), jnp.float8_e4m3fn),  # next_w_qkv_w8k
        s((1280,), jnp.float32),  # next_s_qkv
        s((hidden, 1024), jnp.float8_e4m3fn),  # next_w_gate_w8k
        s((1024,), jnp.float32),  # next_s_gate
        s(kv_shape, jnp.float8_e4m3fn),  # kv_cache_fused
        s((batch_size,), jnp.int32),  # kv_lens
        s((batch_size * 4,), jnp.int32),  # page_indices
        s((batch_size + 1,), jnp.int32),  # cu_q_lens
        s((batch_size + 1,), jnp.int32),  # cu_kv_lens
        s((3,), jnp.int32),  # distribution
        s((batch_size, 2, 128), jnp.float32),  # cos_sin_step
        s((4,), jnp.float32),  # free_params
        *((s((1,), jnp.int32),) if has_layer_idx else ()),
        1e-6, 1e-5, False, 0.17677669529663687, 1e-5, False,  # b1 scalars
        1e-8, True, 1e-5, False,  # b2 scalars
        1e-5, 1.0, 1.0, 1.0,  # epi scalars
        0.125, 2049, 1.0, 1.0, 1e-5,  # rpa scalars
    ]
    with (
        tpu_info_for_cpu_export(),
        jax._src.config.export_ignore_forward_compatibility(True),
    ):
        lowered = wrapped.jit_fn.lower(*jax_args)
    hlo_text = lowered.compiler_ir("hlo").as_hlo_text()
    # Read-only kv_cache_fused + 32 KiB kv_merged return must emit ZERO copy or alias on kv_cache!
    kv_shape_tok = "f8e4m3fn[64,2,128,1,4,128]" if has_layer_idx else "f8e4m3fn[128,128,1,4,128]"
    kv_copy_lines = [
        ln for ln in hlo_text.splitlines()
        if kv_shape_tok in ln and "copy." in ln
    ]
    assert not kv_copy_lines, f"Unexpected kv_cache copy in HLO: {kv_copy_lines}"
    assert "input_output_alias" not in hlo_text and "output_to_operand_aliasing" not in hlo_text

    # Guard against regression: calling wrapped must NEVER invoke kv_cache.copy_(...) (#4210).
    from sglang_tpu import pallas_jax_op
    from torch_tpu._internal.pallas import pallas as pallas_impl

    class _NoCopyTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, data):
            return torch.Tensor._make_subclass(cls, data)

        def copy_(self, *_a, **_kw):
            raise AssertionError("Step (c) must never invoke kv_cache.copy_()")

    torch_args = [
        _NoCopyTensor(torch.empty(a.shape, dtype=pallas_impl.JAX_TO_TORCH_DTYPE_MAP[a.dtype]))
        if isinstance(a, jax.ShapeDtypeStruct)
        else a
        for a in jax_args
    ]
    with (
        tpu_info_for_cpu_export(),
        mock.patch.object(pallas_impl.tpu_torch_pallas, "register_custom_kernel"),
        mock.patch.object(
            pallas_jax_op,
            "_torch_placeholder",
            side_effect=lambda aval, sharding, mesh: torch.empty(
                pallas_jax_op._convert_partitioned_shape(
                    aval.shape,
                    mesh,
                    getattr(sharding, "spec", None),
                    lambda sz, ax: sz // ax,
                ),
                dtype=pallas_impl.JAX_TO_TORCH_DTYPE_MAP[aval.dtype],
            ),
        ),
        mock.patch.object(
            pallas_impl.tpu_torch_pallas,
            "call_custom_kernel",
            return_value=[
                torch.zeros((r_pair, hidden), dtype=torch.float32),
                torch.zeros((batch_size, 1024), dtype=torch.bfloat16),
                torch.zeros((batch_size, 1, 4, 128), dtype=torch.float8_e4m3fn),
            ],
        ),
    ):
        res = wrapped(*torch_args)
    assert len(res) == 3 and tuple(res[2].shape) == (batch_size, 1, 4, 128)


@pytest.mark.parametrize(
    "world_size,row_mode",
    [
        (8, "sparse"),
        (16, "dense"),
        (16, "sparse"),
    ],
)
def test_b16_replicated_b1_moe_b2_interpret_parity_and_tpu7x_aot(
    world_size: int, row_mode: str
):
    """Verify replicated B=16 Step (b) [rep_b1 + MoE + rep_b2] Interpret parity + TPU7x AOT."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices("cpu")) < world_size:
        pytest.skip(f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}")

    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
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
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import _unpack_muse_sparse_scratch

    batch_size = 16
    hidden = 8192
    h_pre = 4096
    pre_cols = h_pre // world_size
    post_cols = hidden // world_size
    qkv_cols = 640 if world_size == 16 else 1280
    gate_cols = 512 if world_size == 16 else 1024
    scale_block = 512
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        # _run_rep_b1_vmem_core matches decode_collective_bridges.py:L295 (detect_races=False)
        # so 16-core NpVectorClock does not stall CPU interpret for >10m.
        detect_races=False,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261140 + world_size + (1 if row_mode == "sparse" else 0))
    d = _build_b1_moe_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=0,
        bf16_router=True,
        sparse_heavy=True,
    )
    stream_rep = jnp.asarray((rng.standard_normal((batch_size, hidden)) * 0.25).astype(np.float32))
    w_pre_rep = jnp.asarray(
        (rng.standard_normal((hidden, h_pre)) * 0.04).astype(np.float32),
        dtype=jnp.float8_e4m3fn,
    )
    s_pre_rep = jnp.asarray((np.abs(rng.standard_normal((h_pre,))) * 0.02 + 1e-3).astype(np.float32))
    w_post_rep = jnp.asarray(
        (rng.standard_normal((h_pre, hidden)) * 0.04).astype(np.float32),
        dtype=jnp.float8_e4m3fn,
    )
    s_post_rep = jnp.asarray((np.abs(rng.standard_normal((hidden,))) * 0.02 + 1e-3).astype(np.float32))
    post_ff_gain = jnp.asarray((np.ones((hidden,)) * 1.03).astype(np.float32))
    pf_alpha = jnp.asarray((np.ones((hidden,)) * 0.92).astype(np.float32))
    pf_beta = jnp.asarray((np.ones((hidden,)) * 0.88).astype(np.float32))
    next_in_gain = jnp.asarray((np.ones((hidden,)) * 1.01).astype(np.float32))
    next_w_qkv = jnp.asarray(
        rng.standard_normal((world_size * hidden, qkv_cols)).astype(np.float32) * 0.04,
        dtype=jnp.float8_e4m3fn,
    )
    next_s_qkv = jnp.asarray((np.abs(rng.standard_normal((qkv_cols,))) * 0.02 + 1e-3).astype(np.float32))
    next_w_gate = jnp.asarray(
        rng.standard_normal((world_size * hidden, gate_cols)).astype(np.float32) * 0.04,
        dtype=jnp.float8_e4m3fn,
    )
    next_s_gate = jnp.asarray((np.abs(rng.standard_normal((gate_cols,))) * 0.02 + 1e-3).astype(np.float32))
    cos_sin_step = jnp.asarray((rng.standard_normal((batch_size, 2, 128)) * 0.3).astype(np.float32))
    free_params = jnp.asarray([0.84, 1.0, 0.20, 2.0], dtype=jnp.float32)

    cfg = dataclasses.replace(
        default_sparse_decode_config(batch_size, world_size, scale_block),
        row_mode=row_mode,
        weight_mode="stream3",
        tile_m=16,
        m_buckets=(8, 16) if row_mode == "sparse" else (16,),
        route_chunk_rows=16,
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
        rank, stream_r, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre, s_pre, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
        cos_sin, fp,
    ):
        st1, logits, expert_in = attn_to_moe_bridge_jax(
            stream_r, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre, s_pre, w_router, pre_exp_gain,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            interpret=ip_race,
        )
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size, batch_size=batch_size, scale_block=scale_block,
            s1_dtype=s1.dtype, s2_dtype=s2.dtype, num_replicas=0,
            return_sharded=False, config=cfg,
        )
        v_sp, h_sp = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.HBM)

        def _ref_moe(*refs):
            it = iter(refs)
            rk_r, fp_r = next(it), next(it)
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = next(it), next(it), next(it), next(it), next(it)
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it, ep=world_size, scale_block=scale_block,
                num_replicas=0, return_sharded=False, cfg=cfg,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(bsem, device_id={"model": jnp.int32(d_i)}, device_id_type=pl.DeviceIdType.MESH)
            _run_muse_sparse_from_vmem_core(
                rk_r[0], lg_r, bs_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                **{k: v for k, v in msc.items() if k != "w_mode"},
                barrier_sem=bsem, output_ref=out_r, ep=world_size, batch_size=batch_size,
                scale_block=scale_block, num_replicas=0, return_sharded=False,
                replica_cost_mode="tiles", ep_axis_name="model", cfg=cfg,
                free_params_ref=fp_r, wait_entry_barrier=True, write_output_ref=True,
            )

        moe_out = pl.pallas_call(
            _ref_moe,
            out_shape=jax.ShapeDtypeStruct((batch_size, h_pre), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, grid=(1,),
                in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                out_specs=v_sp, scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2, collective_id=0, disable_bounds_checks=True,
            ),
            interpret=ip_race,
        )(jnp.atleast_1d(rank), fp, logits, bias, expert_in, w1, s1, w2, s2, norm)

        return moe_to_attn_qkv_epi_bridge_jax(
            st1, moe_out, w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g,
            w_qkv, s_qkv, w_gate, s_gate, cos_sin,
            world_size=world_size, sharded_moe_in=False, interpret=ip_race,
        )

    def _run_fused(
        rank, stream_r, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre, s_pre, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
        cos_sin, fp,
    ):
        return b1_moe_b2_replicated_b16_jax(
            rank, stream_r, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre, s_pre, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            cos_sin_step=cos_sin, free_params=fp,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            replica_cost_mode="tiles", config=cfg, interpret=ip_race,
        )

    mesh = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    call_args = (
        d["ranks"], stream_rep, d["gated_full"], d["wo_sharded"], d["wo_scale"],
        d["pa_alpha"], d["pa_beta"], d["pre_ff_gain"],
        w_pre_rep, s_pre_rep, d["w_router"], d["pre_exp_gain"],
        d["bias"], d["w1_all"], d["s1_all"], d["w2_all"], d["s2_all"], d["post_exp_norm"],
        w_post_rep, s_post_rep, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv, next_s_qkv, next_w_gate, next_s_gate, cos_sin_step, free_params,
    )
    in_specs = (
        P("model"), P(), P(None, "model"), P("model", None), P(),
        P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
        P(None, "model"), P("model"), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(), P(), P(),
    )
    out_specs = (P("model"), P("model"), P("model"), P("model"))

    ref_outs = jax.jit(jax.shard_map(_run_standalone, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)
    fused_outs = jax.jit(jax.shard_map(_run_fused, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)

    for idx, name in enumerate(("stream", "q_5d", "kv_merged", "gate")):
        r_np = np.asarray(ref_outs[idx], dtype=np.float32)
        f_np = np.asarray(fused_outs[idx], dtype=np.float32)
        assert np.all(np.isfinite(f_np)), f"{name} non-finite"
        assert _cosine_sim(r_np, f_np) > 0.9999, f"{name} cosine mismatch: {_cosine_sim(r_np, f_np)}"

    topo_name = "tpu7x:2x2x1" if world_size == 8 else "tpu7x:2x2x2"
    tpu_devs = topologies.get_topology_desc(topo_name, platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    abstract = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in call_args)

    def _aot_b16(*args):
        (
            rank, stream_r, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre, s_pre, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            cos_sin, fp,
        ) = args
        return b1_moe_b2_replicated_b16_jax(
            rank, stream_r, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre, s_pre, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            cos_sin_step=cos_sin, free_params=fp,
            world_size=world_size, replica_cost_mode="tiles", config=cfg, interpret=False,
        )

    compiled = jax.jit(
        jax.shard_map(_aot_b16, mesh=mesh_aot, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    ).lower(*abstract).compile()
    assert compiled is not None


def test_nonuniform_free_params_zero_compile_and_step_c_dispatch():
    """Verify Non-Uniform FREE buffers + Step (c) skip-next-attn + B=16 superkernel dispatch."""
    import os
    import types
    from unittest import mock

    import torch

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang_tpu.models.muse_spark import MuseSparkTextModel, MuseSparkMoE
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
        get_muse_fuse_all_decode_op,
    )
    from test.registered.unit.hardware_backend.tpu.test_muse_w7_col_proj import (
        _make_dummy_layer,
        _make_norm,
    )

    import jax
    from jax.sharding import Mesh

    mesh_fdo = Mesh(np.array(jax.devices("cpu")[:8]).reshape(1, 8), ("data", "model"))
    with mock.patch(
        "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op._get_mesh",
        return_value=mesh_fdo,
    ):
        # 1. Verify standalone fused_decode_op supports has_free_params cleanly.
        op_fp0 = get_muse_fuse_all_decode_op(mode="sparse", ep=8, block=512, has_free_params=True)
        op_fp1 = get_muse_fuse_all_decode_op(
            mode="sparse", ep=8, block=512, num_replicas=2, has_spec_slot=True, has_free_params=True
        )
        assert op_fp0 is not None and op_fp1 is not None

    # 2. Verify boundary vs middle layers resolve distinct FREE params under one op.
    moe_b = object.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe_b)
    moe_b.layer_idx = 1
    moe_b.num_layers = 62
    moe_m = object.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe_m)
    moe_m.layer_idx = 30
    moe_m.num_layers = 62
    with mock.patch.dict(
        os.environ,
        {
            "SGLANG_TPU_MUSE_DECODE_NONUNIFORM_FREE": "mid_f10p82",
            "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK": "0.86",
            "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS": "8",
        },
    ):
        assert moe_b._prepare_nonuniform_free_buffers(
            ep_owners=8, batches=(16, 64), device=torch.device("cpu")
        )
        assert moe_m._prepare_nonuniform_free_buffers(
            ep_owners=8, batches=(16, 64), device=torch.device("cpu")
        )
    moe_b._fused_decode_has_free_params = True
    moe_m._fused_decode_has_free_params = True
    assert not torch.equal(
        moe_b._free_params_for_batch(64), moe_m._free_params_for_batch(64)
    )
    assert moe_b._free_params_for_batch(64).tolist() == pytest.approx([0.88, 6.0, 0.18, 2.0])
    assert moe_m._free_params_for_batch(64).tolist() == pytest.approx([0.82, 10.0, 0.22, 1.0])
    assert moe_m._free_params_for_batch(16).tolist()[1] == 2.0  # EP=8 B<=16 clamped to <=2

    # 3. Verify Step (c) skips Layer L+1 self_attn + Step (a) consumes precomputed_gated.
    h_dim, h_pre, batch, tp = 8192, 4096, 64, 8
    r_pair = batch // (tp // 2)
    attn_calls = []
    op_calls = []

    def _make_stub_model(*, b: int, chip_sh: bool):
        m = object.__new__(MuseSparkTextModel)
        torch.nn.Module.__init__(m)
        m.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
        m.layers_to_capture = []
        m._capture_slots_by_layer = {}
        m.prefill_kernels = None
        m.config = types.SimpleNamespace(hidden_size=h_dim, rms_norm_eps=1e-5)
        m.rotary_emb = types.SimpleNamespace(cos_sin_pairs=None)
        m.start_layer = 0
        m.end_layer = 2
        lyrs = []
        for l_idx in range(2):
            lyr = _make_dummy_layer(3, H=h_dim, H_PRE=h_pre)
            lyr.self_attn.o_proj.tp_size = tp
            lyr.mlp.tp_size = tp
            lyr.self_attn.decode_qkv_epilogue = True
            lyr.self_attn.decode_attn_gate_fuse = True
            lyr.self_attn.q_size = 1024
            lyr.self_attn.kv_size = 128
            lyr.self_attn.rotary_emb = None
            lyr.self_attn.norm_eps = 1e-5
            lyr.self_attn.query_scale = 1.0
            lyr.self_attn.attn = types.SimpleNamespace(
                layer_id=l_idx,
                scaling=0.125,
                sliding_window_size=2048,
                k_scale_float=1.0,
                v_scale_float=1.0,
                qk_head_dim=64,
                v_head_dim=64,
                tp_q_head_num=16,
                tp_k_head_num=2,
            )
            lyr.self_attn._o_proj_w8k = torch.zeros((1024, h_dim), dtype=torch.float8_e4m3fn)
            lyr.self_attn._o_proj_scale = torch.ones((h_dim,), dtype=torch.float32)
            lyr.self_attn.register_buffer("_w7_wqkv_w8k", torch.zeros((h_dim, 1280), dtype=torch.float8_e4m3fn), persistent=False)
            lyr.self_attn.register_buffer("_w7_sqkv", torch.ones((1280,), dtype=torch.float32), persistent=False)
            lyr.self_attn.register_buffer("_w7_wgate_w8k", torch.zeros((h_dim, 1024), dtype=torch.float8_e4m3fn), persistent=False)
            lyr.self_attn.register_buffer("_w7_sgate", torch.ones((1024,), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_w7_wpre_w8k", torch.zeros((h_dim, h_pre // tp), dtype=torch.float8_e4m3fn), persistent=False)
            lyr.mlp.register_buffer("_w7_spre", torch.ones((h_pre // tp,), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_w8_wpre_pair_w8k", torch.zeros((h_dim, h_pre // 2), dtype=torch.float8_e4m3fn), persistent=False)
            lyr.mlp.register_buffer("_w8_spre_pair", torch.ones((h_pre // 2,), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_w7_wrouter_km", torch.zeros((h_dim, 256), dtype=torch.bfloat16), persistent=False)
            lyr.mlp.register_buffer("_w7_wpost_w8k", torch.zeros((h_pre, h_dim // tp), dtype=torch.float8_e4m3fn), persistent=False)
            lyr.mlp.register_buffer("_w7_spost", torch.ones((h_dim // tp,), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_w9_wpost_pair_w8k", torch.zeros((h_pre, h_pre), dtype=torch.float8_e4m3fn), persistent=False)
            lyr.mlp.register_buffer("_w9_spost_pair", torch.ones((h_pre,), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_fused_decode_rank", torch.zeros((1,), dtype=torch.int32), persistent=False)
            lyr.mlp.register_buffer("_fused_decode_bias_2d", torch.zeros((1, 256), dtype=torch.float32), persistent=False)
            lyr.mlp.register_buffer("_fused_decode_norm_2d", torch.ones((1, h_pre), dtype=torch.bfloat16), persistent=False)
            lyr.mlp._bias_2d = lambda l=lyr: l.mlp._buffers["_fused_decode_bias_2d"]
            lyr.mlp._norm_2d = lambda l=lyr: l.mlp._buffers["_fused_decode_norm_2d"]
            lyr.mlp._free_params_for_batch = lambda _nt: None
            lyr.mlp._fused_decode_mode = "sparse"
            lyr.mlp._fused_decode_op = object()
            lyr.mlp._fused_decode_batches = (16, 64)
            lyr.mlp._fused_decode_num_replicas = 0
            lyr.mlp._fused_decode_has_spec_slot = False
            lyr.mlp._fused_decode_replica_cost_mode = "tiles"
            lyr.mlp.register_buffer("_decode_fp4_w1", torch.empty((256 // tp, 4096, 4096), dtype=torch.float4_e2m1fn_x2), persistent=False)
            lyr.mlp.register_buffer("_decode_fp4_s1", torch.ones((256 // tp, 8, 8192), dtype=torch.bfloat16), persistent=False)
            lyr.mlp.register_buffer("_decode_fp4_w2", torch.empty((256 // tp, 4096, 2048), dtype=torch.float4_e2m1fn_x2), persistent=False)
            lyr.mlp.register_buffer("_decode_fp4_s2", torch.ones((256 // tp, 8, 4096), dtype=torch.bfloat16), persistent=False)
            lyr._w8_bridge_qkv_epi = True
            lyr.self_attn._w8_bridge_qkv_epi = True
            lyr._w9_b1_moe_b2_superkernel = True
            lyr.self_attn.forward = lambda pos, ai, fb_, li=l_idx, **kw: (
                attn_calls.append(li) or torch.zeros((b, 1024), dtype=torch.bfloat16)
            )
            lyrs.append(lyr)
        m.layers = torch.nn.ModuleList(lyrs)
        m.norm = _make_norm(h_dim)
        m._decode_norm_layout = 0
        m._decode_col_bridges = 3
        m._w9_chip_sharded_bridges = chip_sh
        m._w9_all_layers_chip_sharded_ready = chip_sh
        m._w9_chip_sharded_batches = (64, 128) if chip_sh else ()
        m._w9_chip_sharded_tp = tp if chip_sh else 0
        m._w8_tok_sharded_bridges = False
        m._w8_all_layers_tok_sharded_ready = False
        m._w8_fuse_final_norm = True
        m._w8_pallas_embed_enabled = False
        m._w8_pallas_embed_tp = 0
        return m

    backend_stub = types.SimpleNamespace(
        _dcp=1,
        _decode_prepared_gated_op=object(),
        _kv_pack_dense2=False,
        forward_metadata=types.SimpleNamespace(
            decode_query_size=1,
            decode_shared_prefix_pages=None,
            cu_q_lens=torch.zeros((batch + 1,), dtype=torch.int32),
            cu_kv_lens=torch.zeros((batch + 1,), dtype=torch.int32),
        ),
        token_to_kv_pool=types.SimpleNamespace(
            swa_kv_pool=types.SimpleNamespace(kv_pack_dense2=False),
            get_key_buffer=lambda _lid: torch.zeros((16, 128, 2, 4, 128), dtype=torch.float8_e4m3fn),
        ),
        can_forward_decode_prepared=lambda _a, _fb: True,
        _kv_lens=lambda _fb: torch.full((batch,), 64, dtype=torch.int32),
        _layer_page_indices=lambda _lid, is_swa_layer=False: torch.zeros((batch * 16,), dtype=torch.int32),
        _layer_distribution=lambda is_swa_layer=False: torch.tensor([batch, batch, batch], dtype=torch.int32),
    )
    fb_c = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        attn_backend=backend_stub,
    )
    m_c = _make_stub_model(b=batch, chip_sh=True)
    with (
        mock.patch.dict(
            os.environ,
            {
                "WORLD_SIZE": str(tp),
                "SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL": "1",
                "SGLANG_TPU_RPA_V3_DECODE_FAST": "w8_noz",
            },
        ),
        mock.patch("sglang_tpu.models.muse_spark.get_parallel", return_value=types.SimpleNamespace(tp_rank=0, tp_size=tp, attn_cp_size=1)),
        mock.patch("sglang.srt.runtime_context.get_forward", return_value=types.SimpleNamespace(sp_active=False)),
        mock.patch("sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend", return_value=backend_stub),
        mock.patch("sglang_tpu.attention_pallas_rpa_base._fused_kv_operands", return_value=(torch.zeros((16, 128, 2, 4, 128), dtype=torch.float8_e4m3fn), None)),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_rpad_chip_sharded_op",
            side_effect=lambda *_a, **_kw: (
                lambda *_args: (
                    op_calls.append("step_c")
                    or (
                        torch.zeros((r_pair, h_dim), dtype=torch.float32),
                        torch.full((batch, 1024), 3.0, dtype=torch.bfloat16),
                        torch.zeros((batch, 1, 4, 128), dtype=torch.float8_e4m3fn),
                    )
                )
            ),
        ),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_step_c_kv_write_op",
            side_effect=lambda **_kw: (lambda *_args: op_calls.append("write_kv_only")),
        ),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_chip_sharded_op",
            side_effect=lambda *_a, **_kw: (
                lambda *args: (
                    op_calls.append(("step_a", float(args[2][0, 0])))
                    or (
                        torch.zeros((r_pair, h_dim), dtype=torch.float32),
                        torch.zeros((batch // tp, h_pre), dtype=torch.bfloat16),
                    )
                )
            ),
        ),
        mock.patch(
            "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_post_chip_sharded_final_norm_op",
            side_effect=lambda *_a, **_kw: (lambda *_args: torch.zeros((batch, h_dim), dtype=torch.bfloat16)),
        ),
    ):
        out_c = m_c.forward(torch.zeros((batch,), dtype=torch.int32), torch.arange(batch, dtype=torch.int32), fb_c, input_embeds=torch.randn(batch, h_dim, dtype=torch.bfloat16))
    assert out_c.shape == (batch, h_dim)
    assert attn_calls == [0], f"Layer 1 self_attn must be skipped by Step (c), got attn_calls={attn_calls}"
    assert op_calls == ["step_c", ("step_a", 3.0), "write_kv_only"]

    # 3b. Verify TP=16 Step (c) dispatches when kv_pack_bd4=True (bd4 phys_rows=128).
    attn_calls.clear()
    op_calls.clear()
    bd4_calls = []
    backend_bd4 = types.SimpleNamespace(
        _dcp=1,
        _decode_prepared_gated_op=object(),
        _kv_pack_dense2=False,
        _kv_pack_bd4=True,
        forward_metadata=backend_stub.forward_metadata,
        token_to_kv_pool=types.SimpleNamespace(
            swa_kv_pool=types.SimpleNamespace(kv_pack_dense2=False, kv_pack_bd4=True),
            get_key_buffer=lambda _lid: torch.zeros((16, 128, 1, 4, 128), dtype=torch.float8_e4m3fn),
        ),
        can_forward_decode_prepared=lambda _a, _fb: True,
        _kv_lens=backend_stub._kv_lens,
        _layer_page_indices=lambda _lid, is_swa_layer=False: torch.zeros((batch * 4,), dtype=torch.int32),
        _layer_distribution=backend_stub._layer_distribution,
    )
    fb_bd4 = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        attn_backend=backend_bd4,
    )
    with (
        mock.patch.dict(
            os.environ,
            {
                "WORLD_SIZE": str(tp),
                "SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL": "1",
                "SGLANG_TPU_RPA_V3_DECODE_FAST": "w8,s4,batchfull,g4k,foldcol,batch4",
            },
        ),
        mock.patch("sglang_tpu.models.muse_spark.get_parallel", return_value=types.SimpleNamespace(tp_rank=0, tp_size=tp, attn_cp_size=1)),
        mock.patch("sglang.srt.runtime_context.get_forward", return_value=types.SimpleNamespace(sp_active=False)),
        mock.patch("sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend", return_value=backend_bd4),
        mock.patch("sglang_tpu.attention_pallas_rpa_base._fused_kv_operands", return_value=(torch.zeros((16, 128, 1, 4, 128), dtype=torch.float8_e4m3fn), None)),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_rpad_chip_sharded_op",
            side_effect=lambda *_a, **kw: (
                lambda *_args: (
                    bd4_calls.append(("step_c", bool(kw.get("kv_pack_bd4"))))
                    or (
                        torch.zeros((r_pair, h_dim), dtype=torch.float32),
                        torch.full((batch, 1024), 4.0, dtype=torch.bfloat16),
                        torch.zeros((batch, 1, 4, 128), dtype=torch.float8_e4m3fn),
                    )
                )
            ),
        ),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_step_c_kv_write_op",
            side_effect=lambda **kw: (lambda *_args: bd4_calls.append(("write_kv", bool(kw.get("kv_pack_bd4"))))),
        ),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_chip_sharded_op",
            side_effect=lambda *_a, **_kw: (
                lambda *args: (
                    bd4_calls.append(("step_a", float(args[2][0, 0])))
                    or (
                        torch.zeros((r_pair, h_dim), dtype=torch.float32),
                        torch.zeros((batch // tp, h_pre), dtype=torch.bfloat16),
                    )
                )
            ),
        ),
        mock.patch(
            "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_post_chip_sharded_final_norm_op",
            side_effect=lambda *_a, **_kw: (lambda *_args: torch.zeros((batch, h_dim), dtype=torch.bfloat16)),
        ),
    ):
        out_c_bd4 = m_c.forward(torch.zeros((batch,), dtype=torch.int32), torch.arange(batch, dtype=torch.int32), fb_bd4, input_embeds=torch.randn(batch, h_dim, dtype=torch.bfloat16))
    assert out_c_bd4.shape == (batch, h_dim)
    assert attn_calls == [0]
    assert bd4_calls == [("step_c", True), ("step_a", 4.0), ("write_kv", True)]

    # 4. Verify B=16 replicated Step (b) dispatch when chip_sharded is off for B=16.
    attn_calls.clear()
    op_calls.clear()
    m_b16 = _make_stub_model(b=16, chip_sh=True)
    fb_b16 = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        attn_backend=backend_stub,
    )
    with (
        mock.patch.dict(os.environ, {"WORLD_SIZE": str(tp), "SGLANG_TPU_RPA_V3_DECODE_FAST": "w8_noz"}),
        mock.patch("sglang_tpu.models.muse_spark.get_parallel", return_value=types.SimpleNamespace(tp_rank=0, tp_size=tp, attn_cp_size=1)),
        mock.patch("sglang.srt.runtime_context.get_forward", return_value=types.SimpleNamespace(sp_active=False)),
        mock.patch("sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend", return_value=backend_stub),
        mock.patch(
            "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_replicated_b16_op",
            side_effect=lambda *_a, **_kw: (
                lambda *_args: (
                    op_calls.append("rep_b16")
                    or (
                        torch.zeros((16, h_dim), dtype=torch.float32),
                        torch.zeros((1, 16, 8, 2, 128), dtype=torch.bfloat16),
                        torch.zeros((16, 1, 4, 128), dtype=torch.float8_e4m3fn),
                        torch.zeros((16, 1024), dtype=torch.bfloat16),
                    )
                )
            ),
        ),
        mock.patch(
            "sglang_tpu.kernels.decode_collective_bridges.get_attn_to_moe_bridge_op",
            side_effect=lambda *_a, **_kw: (
                lambda *_args: (
                    op_calls.append("rep_b1")
                    or (
                        torch.zeros((16, h_dim), dtype=torch.float32),
                        torch.zeros((16, 256), dtype=torch.float32),
                        torch.zeros((16, h_pre), dtype=torch.bfloat16),
                    )
                )
            ),
        ),
        mock.patch(
            "sglang_tpu.kernels.decode_collective_bridges_w8.get_moe_post_ag_final_norm_op",
            side_effect=lambda *_a, **_kw: (lambda *_args: torch.zeros((16, h_dim), dtype=torch.bfloat16)),
        ),
    ):
        m_b16.layers[1].mlp.can_return_sharded_moe_out = lambda _n, *, decode: True
        m_b16.layers[1].mlp.forward = lambda *_a, **_kw: torch.zeros((2, h_pre), dtype=torch.bfloat16)
        out_16 = m_b16.forward(torch.zeros((16,), dtype=torch.int32), torch.arange(16, dtype=torch.int32), fb_b16, input_embeds=torch.randn(16, h_dim, dtype=torch.bfloat16))
    assert out_16.shape == (16, h_dim)
    assert op_calls == ["rep_b16", "rep_b1"]
    assert attn_calls == [0, 1]


def test_write_decode_kv_merged_packed_smem_and_bd4_interpret_races_and_tpu7x_aot():
    """Verify 1-SMEM-prefetch unpacked + fast VMEM bd4 scatter (q_len==1 & q_len>1 + 6D alias HLO)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import SingleDeviceSharding
    from torch_tpu._internal import pallas as _torch_pallas
    from unittest import mock

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import write_kv_only_rpa_v3
    from sglang_tpu.muse_megakernel.decode import b1_moe_ops
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
        write_decode_kv_merged_jax,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261217)

    # 1. Unpacked: q_len=1 fast path + q_len=2 page-crossing (511->513) with 0x7F NaN poisoning.
    B, page_size = 16, 512
    pages_per_seq = 4
    num_blocks = B * pages_per_seq + 4
    kv_cache_u8 = np.full((num_blocks, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
    kv_cache_5d = jax.lax.bitcast_convert_type(jnp.asarray(kv_cache_u8, jnp.uint8), jnp.float8_e4m3fn)
    kv_lens_np = np.array([513, 17, 512, 1025, 1, 63, 64, 65, 511, 2047, 300, 0, 0, 0, 0, 0], dtype=np.int32)
    q_lens_np = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    T = int(q_lens_np.sum())
    # Pad T to 16 so fast VMEM bd4 test can share q_lens later cleanly.
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens_np)]).astype(np.int32))
    cu_kv = jnp.arange(B + 1, dtype=jnp.int32) * (pages_per_seq * page_size)
    pg_np = rng.permutation(num_blocks)[: B * pages_per_seq].astype(np.int32)
    page_indices = jnp.asarray(pg_np)
    kv_lens = jnp.asarray(kv_lens_np)
    dist = jnp.asarray([11, 11, 11], dtype=jnp.int32)
    kv_new = jnp.asarray((rng.standard_normal((T, 1, 4, 128)) * 0.25).astype(np.float32), dtype=jnp.float8_e4m3fn)

    ref_unp = write_kv_only_rpa_v3(
        kv_new, kv_cache_5d, kv_lens, page_indices, cu_q, cu_kv, dist, interpret=True, kv_pack_dense2=False,
    )
    got_unp = write_decode_kv_merged_jax(
        kv_new, kv_cache_5d, kv_lens, page_indices, cu_q, cu_kv, dist, interpret=ip_race,
    )
    assert np.array_equal(
        np.asarray(jax.lax.bitcast_convert_type(ref_unp, jnp.uint8)),
        np.asarray(jax.lax.bitcast_convert_type(got_unp, jnp.uint8)),
    )

    # 2. bd4 fast VMEM scatter: q_len=1 across all 4 slot4 alignments (0,1,2,3) + q_len=2 cross-page & mid-quad.
    phys_rows = page_size // 4  # 128
    # Initialize valid pre-existing tokens with random FP8 in low 64 lanes + 0x7F in untouched blocks.
    init_u8 = np.full((num_blocks, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    for b_i in range(11):
        old_end = int(kv_lens_np[b_i] - q_lens_np[b_i])
        for tok in range(max(0, old_end - 4), old_end):
            pg = int(pg_np[b_i * pages_per_seq + tok // page_size])
            pr = (tok % page_size) // 4
            s4 = tok & 3
            c_pair = 0 if s4 < 2 else 2
            l0, l1 = (0, 64) if (s4 & 1) == 0 else (64, 128)
            init_u8[pg, pr, 0, c_pair : c_pair + 2, l0:l1] = np.asarray(
                jax.lax.bitcast_convert_type(
                    jnp.asarray((rng.standard_normal((2, 64)) * 0.2).astype(np.float32), jnp.float8_e4m3fn),
                    jnp.uint8,
                )
            )
    kv_cache_bd4 = jax.lax.bitcast_convert_type(jnp.asarray(init_u8, jnp.uint8), jnp.float8_e4m3fn)
    # T=16 case covering q_len=1 (all slot4 in {0,1,2,3}) and q_len in {2,3} multi-token RMW + page cross.
    B_bd4 = 16
    kv_lens_bd4 = jnp.asarray(
        [10, 11, 12, 13, 513, 515, 19, 20, 100, 101, 102, 103, 0, 0, 0, 0],
        dtype=jnp.int32,
    )
    q_lens_bd4 = np.array([1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0], dtype=np.int32)
    T_bd4 = int(q_lens_bd4.sum())  # 16
    assert T_bd4 == 16
    cu_q_bd4 = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens_bd4)]).astype(np.int32))
    kv_new_bd4_u8 = np.full((T_bd4, 1, 4, 128), 0x7F, dtype=np.uint8)
    kv_new_bd4_u8[:, 0, :2, :64] = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray((rng.standard_normal((T_bd4, 2, 64)) * 0.25).astype(np.float32), jnp.float8_e4m3fn),
            jnp.uint8,
        )
    )
    kv_new_bd4 = jax.lax.bitcast_convert_type(jnp.asarray(kv_new_bd4_u8, jnp.uint8), jnp.float8_e4m3fn)
    dist_bd4 = jnp.asarray([12, 12, 12], dtype=jnp.int32)

    ref_bd4 = write_kv_only_rpa_v3(
        kv_new_bd4, kv_cache_bd4, kv_lens_bd4, page_indices, cu_q_bd4, cu_kv, dist_bd4,
        interpret=True, kv_pack_bd4=True,
    )
    got_bd4 = write_decode_kv_merged_jax(
        kv_new_bd4, kv_cache_bd4, kv_lens_bd4, page_indices, cu_q_bd4, cu_kv, dist_bd4,
        kv_pack_bd4=True, interpret=ip_race,
    )
    assert np.array_equal(
        np.asarray(jax.lax.bitcast_convert_type(ref_bd4, jnp.uint8)),
        np.asarray(jax.lax.bitcast_convert_type(got_bd4, jnp.uint8)),
    )

    # 3. Verify resolve_step_c_rpa_plan with kv_pack_bd4=True (512->128 phys_rows) + TPU7x AOT + 6D alias HLO.
    cfg_bd4, one_blk_bd4, slot_pg_bd4, use_grp_bd4, ok_bd4 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert (one_blk_bd4, slot_pg_bd4, use_grp_bd4, ok_bd4) == (True, 20, True, True)
    assert cfg_bd4.swabatch_group_size == 4

    captured = {}
    orig_cjk = _torch_pallas.custom_jax_kernel

    def _cap(fn, **kw):
        captured["fn"] = fn
        return orig_cjk(fn, **kw)

    dev0 = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices[0]
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt, sharding=SingleDeviceSharding(dev0))
    for bd4_flag, kv_sh, lidx_sh in [
        (False, (64, 8, 512, 1, 4, 128), s((), jnp.int32)),
        (True, (64, 128, 1, 4, 128), None),
    ]:
        with mock.patch.object(_torch_pallas, "custom_jax_kernel", side_effect=_cap):
            b1_moe_ops._STEP_C_KV_WRITE_OPS.clear()
            b1_moe_ops.get_step_c_kv_write_op(kv_pack_bd4=bd4_flag)
        with (
            mock.patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            comp = jax.jit(captured["fn"], donate_argnums=(1,), keep_unused=True).lower(
                s((64, 1, 4, 128), jnp.float8_e4m3fn),
                s(kv_sh, jnp.float8_e4m3fn),
                s((64,), jnp.int32),
                s((256,), jnp.int32),
                s((65,), jnp.int32),
                s((65,), jnp.int32),
                s((3,), jnp.int32),
                lidx_sh,
            ).compile()
        hlo = comp.as_text()
        assert "input_output_alias={ {}: (1, {}, may-alias) }" in hlo


def test_step_c_tp16_bd4_interpret_parity_and_tpu7x_aot():
    """Verify full Step (c) on TP=16 with kv_pack_bd4=True + write_decode_kv_merged_jax (Interpret + TPU7x AOT)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P
    from unittest import mock

    world_size = 16
    if len(jax.devices("cpu")) < world_size:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import write_kv_only_rpa_v3
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_chip_sharded_bridge_jax,
        moe_to_attn_chip_sharded_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _run_muse_sparse_from_vmem_core,
        build_muse_sparse_scratch_shapes,
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_rpad_chip_sharded_jax,
        write_decode_kv_merged_jax,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import _unpack_muse_sparse_scratch

    batch_size = 64
    hidden = 8192
    h_pre = 4096
    qkv_cols = 640
    gate_cols = 512
    scale_block = 512
    page_size = 512
    phys_rows = 128
    pages_per_seq = 4
    n_act = 59
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261218)
    d = _build_b1_moe_inputs(
        rng,
        world_size=world_size,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=0,
        bf16_router=True,
        sparse_heavy=True,
    )
    w_post_tp2 = jnp.asarray(rng.standard_normal((world_size * h_pre, h_pre)).astype(np.float32) * 0.04, dtype=jnp.float8_e4m3fn)
    s_post_tp2 = jnp.asarray(np.abs(rng.standard_normal((h_pre,))).astype(np.float32) * 0.02 + 1e-3)
    post_ff_gain = jnp.asarray((np.ones((hidden,)) * 1.03).astype(np.float32))
    pf_alpha = jnp.asarray((np.ones((hidden,)) * 0.92).astype(np.float32))
    pf_beta = jnp.asarray((np.ones((hidden,)) * 0.88).astype(np.float32))
    next_in_gain = jnp.asarray((np.ones((hidden,)) * 1.01).astype(np.float32))
    next_w_qkv = jnp.asarray(rng.standard_normal((world_size * hidden, qkv_cols)).astype(np.float32) * 0.04, dtype=jnp.float8_e4m3fn)
    next_s_qkv = jnp.asarray(np.abs(rng.standard_normal((qkv_cols,))).astype(np.float32) * 0.02 + 1e-3)
    next_w_gate = jnp.asarray(rng.standard_normal((world_size * hidden, gate_cols)).astype(np.float32) * 0.04, dtype=jnp.float8_e4m3fn)
    next_s_gate = jnp.asarray(np.abs(rng.standard_normal((gate_cols,))).astype(np.float32) * 0.02 + 1e-3)
    cos_sin_step = jnp.asarray((rng.standard_normal((batch_size, 2, 128)) * 0.3).astype(np.float32))
    free_params = jnp.asarray([0.84, 2.0, 0.20, 1.0], dtype=jnp.float32)

    # Ensure kv_lens covers all 4 slot4 = (kv_len - 1) & 3 alignments across active B=59 sequences.
    kv_lens_np = np.zeros((batch_size,), dtype=np.int32)
    kv_lens_np[:n_act] = np.array(
        [17 + (i % 4) + 512 * (i % 3) for i in range(n_act)], dtype=np.int32
    )
    num_blocks = batch_size * pages_per_seq + 8
    pg_np = rng.permutation(num_blocks)[: batch_size * pages_per_seq].astype(np.int32)
    # Build bd4 KV cache with 0x7F NaN on unwritten slots and clean pre-populated historical tokens.
    hist_u8 = np.full((num_blocks, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    hist_bd4 = jax.lax.bitcast_convert_type(jnp.asarray(hist_u8, jnp.uint8), jnp.float8_e4m3fn)
    old_lens_np = np.where(kv_lens_np > 0, kv_lens_np - 1, 0).astype(np.int32)
    tot_old = int(old_lens_np.sum())
    old_u8 = np.zeros((tot_old, 1, 4, 128), dtype=np.uint8)
    old_u8[:, 0, :2, :64] = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray((rng.standard_normal((tot_old, 2, 64)) * 0.2).astype(np.float32), jnp.float8_e4m3fn),
            jnp.uint8,
        )
    )
    old_kv = jax.lax.bitcast_convert_type(jnp.asarray(old_u8, jnp.uint8), jnp.float8_e4m3fn)
    hist_bd4 = write_kv_only_rpa_v3(
        old_kv,
        hist_bd4,
        jnp.asarray(old_lens_np, dtype=jnp.int32),
        jnp.asarray(pg_np, dtype=jnp.int32),
        jnp.asarray(np.concatenate([[0], np.cumsum(old_lens_np)]).astype(np.int32)),
        jnp.arange(batch_size + 1, dtype=jnp.int32) * (pages_per_seq * page_size),
        jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
        interpret=True,
        kv_pack_bd4=True,
    )

    kv_cache = hist_bd4
    kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    page_indices = jnp.asarray(pg_np, dtype=jnp.int32)
    cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32)
    cu_kv_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * (pages_per_seq * page_size)
    distribution = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)

    rpa_mode = "w8,s4,batchfull,g4k,foldcol,batch4"
    sharded_moe_in = (batch_size // world_size) >= 8 and (batch_size // world_size) % 8 == 0
    cfg = dataclasses.replace(
        default_sparse_decode_config(batch_size, world_size, scale_block),
        weight_mode="stream3", pair_rs=True, bridge_ag=True, m_buckets=(16, 32),
        route_chunk_rows=32, static_holder_mask=True, route_fast=True, fast_plan=2,
        unroll_strips=True, fast_gs=True, fast_b64_epi_pro=False, spec_prime_parts=0,
        k256_strip_mode="wide", nc_split512=False,
    )

    def _run_standalone(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
        kv_c, kvl, pg, cq, ckv, dist, cos_sin, fp,
    ):
        st1, logits, expert_in = attn_to_moe_chip_sharded_bridge_jax(
            stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            interpret=ip_race,
        )
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size, batch_size=batch_size, scale_block=scale_block,
            s1_dtype=s1.dtype, s2_dtype=s2.dtype, num_replicas=0,
            return_sharded=sharded_moe_in, config=cfg,
        )
        v_sp, h_sp = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.HBM)

        def _ref_moe(*refs):
            it = iter(refs)
            rk_r, fp_r = next(it), next(it)
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = next(it), next(it), next(it), next(it), next(it)
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it, ep=world_size, scale_block=scale_block,
                num_replicas=0, return_sharded=sharded_moe_in, cfg=cfg,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(bsem, device_id={"model": jnp.int32(d_i)}, device_id_type=pl.DeviceIdType.MESH)
            _run_muse_sparse_from_vmem_core(
                rk_r[0], lg_r, bs_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                **{k: v for k, v in msc.items() if k != "w_mode"},
                barrier_sem=bsem, output_ref=out_r, ep=world_size, batch_size=batch_size,
                scale_block=scale_block, num_replicas=0, return_sharded=sharded_moe_in,
                replica_cost_mode="tiles", ep_axis_name="model", cfg=cfg,
                free_params_ref=fp_r, wait_entry_barrier=True, write_output_ref=True,
            )

        out_r = (batch_size // world_size) if sharded_moe_in else batch_size
        moe_out = pl.pallas_call(
            _ref_moe,
            out_shape=jax.ShapeDtypeStruct((out_r, h_pre), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, grid=(1,),
                in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                out_specs=v_sp, scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2, collective_id=0, disable_bounds_checks=True,
            ),
            interpret=ip_race,
        )(jnp.atleast_1d(rank), fp, logits, bias, expert_in, w1, s1, w2, s2, norm)

        st2, q_5d, kv_m, gate_2d = moe_to_attn_chip_sharded_bridge_jax(
            st1, moe_out, w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g,
            w_qkv, s_qkv, w_gate, s_gate, cos_sin,
            world_size=world_size, sharded_moe_in=sharded_moe_in, fuse_qkv_epi=True,
            interpret=ip_race,
        )
        gated_out, updated_kv = run_decode_fast_rpa(
            q_5d, kv_m, kv_m, kv_c, kvl, pg, cq, ckv, dist,
            gate=gate_2d, sm_scale=0.125, sliding_window=2049,
            k_scale=0.9, v_scale=1.1, mode_str=rpa_mode,
            prepared_inputs=True, fused_out_pack=2, out_norm_eps=1e-5,
            kv_pack_bd4=True, force_native_tp16=True, interpret=True,
        )
        return st2, gated_out.reshape(batch_size, gate_cols), updated_kv

    def _run_fused(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
        w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
        kv_c, kvl, pg, cq, ckv, dist, cos_sin, fp,
    ):
        st2, ng, kvm = b1_moe_b2_rpad_chip_sharded_jax(
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            kv_c, kvl, pg, cq, ckv, dist,
            cos_sin_step=cos_sin, free_params=fp,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            rpa_sm_scale=0.125, rpa_sliding_window=2049,
            rpa_k_scale=0.9, rpa_v_scale=1.1, rpa_out_norm_eps=1e-5,
            rpa_mode_str=rpa_mode, kv_pack_bd4=True,
            replica_cost_mode="tiles", config=cfg, interpret=ip_race, skip_kv_write=True,
        )
        kvo = write_decode_kv_merged_jax(
            kvm, kv_c, kvl, pg, cq, ckv, dist, kv_pack_bd4=True, interpret=ip_race,
        )
        return st2, ng, kvo

    mesh = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    call_args = (
        d["ranks"], d["stream_chip_st"], d["gated_full"], d["wo_sharded"], d["wo_scale"],
        d["pa_alpha"], d["pa_beta"], d["pre_ff_gain"],
        d["w_pre_pair"], d["s_pre_pair"], d["w_router"], d["pre_exp_gain"],
        d["bias"], d["w1_all"], d["s1_all"], d["w2_all"], d["s2_all"], d["post_exp_norm"],
        w_post_tp2, s_post_tp2, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv, next_s_qkv, next_w_gate, next_s_gate,
        kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution,
        cos_sin_step, free_params,
    )
    in_specs = (
        P("model"), P("model", None), P(None, "model"), P("model", None), P(),
        P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
        P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(),
        P(), P(), P(), P(), P(), P(), P(), P(),
    )
    out_specs = (P("model"), P("model"), P("model"))

    ref_outs = jax.jit(jax.shard_map(_run_standalone, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)
    fused_outs = jax.jit(jax.shard_map(_run_fused, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))(*call_args)

    for idx, name in enumerate(("stream_chip", "next_gated", "updated_kv")):
        r_raw = np.asarray(ref_outs[idx])
        f_raw = np.asarray(fused_outs[idx])
        if idx == 2:
            r_u8 = r_raw.view(np.uint8)
            f_u8 = f_raw.view(np.uint8)
            assert np.array_equal(r_u8 == 0x7F, f_u8 == 0x7F), "bd4 0x7F NaN mask changed"
            valid_mask = r_u8 != 0x7F
            r_np = r_raw.astype(np.float32)[valid_mask]
            f_np = f_raw.astype(np.float32)[valid_mask]
        else:
            r_np = r_raw.astype(np.float32)
            f_np = f_raw.astype(np.float32)
        assert np.all(np.isfinite(f_np)), f"{name} non-finite"
        assert _cosine_sim(r_np, f_np) > 0.9999, f"{name} cosine mismatch: {_cosine_sim(r_np, f_np)}"

    # TPU7x 2x2x2 AOT compile check for Step (c) + bd4 (incl. batch4 B=64).
    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    abstract = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in call_args)

    def _aot_step_c_bd4(*args):
        (
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            kv_c, kvl, pg, cq, ckv, dist, cos_sin, fp,
        ) = args
        return b1_moe_b2_rpad_chip_sharded_jax(
            rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            bias, w1, s1, w2, s2, norm,
            w_post, s_post, post_ff_g, pf_a, pf_b, next_in_g, w_qkv, s_qkv, w_gate, s_gate,
            kv_c, kvl, pg, cq, ckv, dist,
            cos_sin_step=cos_sin, free_params=fp,
            world_size=world_size, rpa_sliding_window=2049,
            rpa_sm_scale=0.125, rpa_k_scale=0.9, rpa_v_scale=1.1, rpa_out_norm_eps=1e-5,
            rpa_mode_str=rpa_mode, kv_pack_bd4=True,
            replica_cost_mode="tiles", config=cfg, interpret=False, skip_kv_write=True,
        )

    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        compiled = jax.jit(
            jax.shard_map(_aot_step_c_bd4, mesh=mesh_aot, in_specs=in_specs, out_specs=out_specs, check_vma=False),
        ).lower(*abstract).compile()
    assert compiled is not None


def _make_standalone_step_a_ref_fn(*, world_size: int, scale_block: int, hidden: int, h_pre: int, batch_size: int, ret_sh: bool, cfg_ref, interp):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_chip_sharded_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _run_muse_sparse_from_vmem_core,
        build_muse_sparse_scratch_shapes,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        _unpack_muse_sparse_scratch,
    )

    def _run_standalone(
        rank, stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
        w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
        bias, w1, s1, w2, s2, norm,
    ):
        st1, logits, expert_in = attn_to_moe_chip_sharded_bridge_jax(
            stream_chip, gated_attn, wo, wo_scale, pa_alpha, pa_beta, pre_ff_gain,
            w_pre_pair, s_pre_pair, w_router, pre_exp_gain,
            world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            interpret=interp,
        )
        scratch_shapes = build_muse_sparse_scratch_shapes(
            ep=world_size, batch_size=batch_size, scale_block=scale_block,
            s1_dtype=s1.dtype, s2_dtype=s2.dtype, num_replicas=0,
            return_sharded=ret_sh, config=cfg_ref,
        )
        v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
        h_sp = pl.BlockSpec(memory_space=pltpu.HBM)

        def _ref_moe(*refs):
            it = iter(refs)
            rk_r = next(it)
            lg_r, bs_r, x_r = next(it), next(it), next(it)
            w1_r, s1_r, w2_r, s2_r, nm_r = next(it), next(it), next(it), next(it), next(it)
            out_r = next(it)
            msc = _unpack_muse_sparse_scratch(
                it, ep=world_size, scale_block=scale_block,
                num_replicas=0, return_sharded=ret_sh, cfg=cfg_ref,
            )
            bsem = pltpu.get_barrier_semaphore()
            for d_i in range(world_size):
                pl.semaphore_signal(bsem, device_id={"model": jnp.int32(d_i)}, device_id_type=pl.DeviceIdType.MESH)
            _run_muse_sparse_from_vmem_core(
                rk_r[0], lg_r, bs_r, x_r, w1_r, s1_r, w2_r, s2_r, nm_r,
                **{k: v for k, v in msc.items() if k != "w_mode"},
                barrier_sem=bsem, output_ref=out_r, ep=world_size, batch_size=batch_size,
                scale_block=scale_block, num_replicas=0, return_sharded=ret_sh,
                replica_cost_mode="tiles", ep_axis_name="model", cfg=cfg_ref,
                wait_entry_barrier=True, write_output_ref=True,
            )

        out_r = (batch_size // world_size) if ret_sh else batch_size
        moe_out = pl.pallas_call(
            _ref_moe,
            out_shape=jax.ShapeDtypeStruct((out_r, h_pre), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1, grid=(1,),
                in_specs=(v_sp, v_sp, v_sp, h_sp, h_sp, h_sp, h_sp, v_sp),
                out_specs=v_sp, scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=60 * 1024**2,
                collective_id=0,
                disable_bounds_checks=True,
            ),
            interpret=interp,
        )(jnp.atleast_1d(rank), logits, bias, expert_in, w1, s1, w2, s2, norm)
        return st1, moe_out

    return _run_standalone


def test_l61_tp16_b64_rep_interpret_parity():
    """Verify L61 TP=16 B=64 Step (a) return_sharded=False Interpret parity & zero races."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        moe_post_chip_sharded_final_norm_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        b1_moe_chip_sharded_jax,
    )

    world_size, scale_block, hidden, h_pre = 16, 512, 8192, 4096
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager", detect_races=True, uninitialized_memory="nan"
    )
    mesh_cpu = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    rng_64 = np.random.default_rng(20261218)
    d64 = _build_b1_moe_inputs(
        rng_64, world_size=world_size, batch_size=64, scale_block=scale_block,
        num_replicas=0, bf16_router=True, sparse_heavy=False,
    )
    w_post_64 = jnp.asarray(
        np.concatenate([(rng_64.standard_normal((h_pre, h_pre)) * 0.04).astype(np.float32) for _ in range(world_size)], axis=0),
        dtype=jnp.float8_e4m3fn,
    )
    s_post_64 = jnp.asarray((np.abs(rng_64.standard_normal((h_pre,))) * 0.02 + 1e-3).astype(np.float32))
    post_ff_g64 = jnp.asarray((np.ones((hidden,)) * 1.02).astype(np.float32))
    pf_a64 = jnp.asarray((np.ones((hidden,)) * 0.93).astype(np.float32))
    pf_b64 = jnp.asarray((np.ones((hidden,)) * 0.88).astype(np.float32))
    fin_g64 = jnp.asarray((np.ones((hidden,)) * 1.05).astype(np.float32))
    cfg64 = dataclasses.replace(
        default_sparse_decode_config(64, world_size, scale_block),
        pair_rs=True, bridge_ag=True, spec_prime_parts=0,
    )
    ip_fast = pltpu.InterpretParams(dma_execution_mode="eager")
    ref_step_a = _make_standalone_step_a_ref_fn(
        world_size=world_size, scale_block=scale_block, hidden=hidden, h_pre=h_pre,
        batch_size=64, ret_sh=False, cfg_ref=cfg64, interp=ip_fast,
    )

    def _ref_l61(*args):
        st1, moe_rep = ref_step_a(*args[:18])
        return moe_post_chip_sharded_final_norm_jax(
            st1, moe_rep, *args[18:],
            world_size=world_size, sharded_moe_in=False, return_stream_too=False, interpret=ip_fast,
        )

    def _fused_l61(*args):
        st1, moe_rep = b1_moe_chip_sharded_jax(
            *args[:18], world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            return_sharded=False, config=cfg64, interpret=ip_race,
        )
        return moe_post_chip_sharded_final_norm_jax(
            st1, moe_rep, *args[18:],
            world_size=world_size, sharded_moe_in=False, return_stream_too=False, interpret=ip_race,
        )

    l61_args = (
        d64["ranks"], d64["stream_chip_st"], d64["gated_full"], d64["wo_sharded"], d64["wo_scale"],
        d64["pa_alpha"], d64["pa_beta"], d64["pre_ff_gain"],
        d64["w_pre_pair"], d64["s_pre_pair"], d64["w_router"], d64["pre_exp_gain"],
        d64["bias"], d64["w1_all"], d64["s1_all"], d64["w2_all"], d64["s2_all"], d64["post_exp_norm"],
        w_post_64, s_post_64, post_ff_g64, pf_a64, pf_b64, fin_g64,
    )
    l61_specs = (
        P("model"), P("model", None), P(None, "model"), P("model", None), P(),
        P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
        P("model", None), P(), P(), P(), P(), P(),
    )
    ref_hid = np.asarray(jax.jit(jax.shard_map(_ref_l61, mesh=mesh_cpu, in_specs=l61_specs, out_specs=P(), check_vma=False))(*l61_args), dtype=np.float32)
    jax.clear_caches()
    got_hid = np.asarray(jax.jit(jax.shard_map(_fused_l61, mesh=mesh_cpu, in_specs=l61_specs, out_specs=P(), check_vma=False))(*l61_args), dtype=np.float32)
    jax.clear_caches()
    assert np.all(np.isfinite(got_hid)) and _cosine_sim(ref_hid, got_hid) > 0.9999


def test_tp16_b256_step_a_and_b_interpret_parity():
    """Verify 2/3-scope TP=16 B=256 Step (a) & Step (b) Interpret parity & zero races."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices("cpu")) < 16:
        pytest.skip("Requires XLA_FLAGS=--xla_force_host_platform_device_count=16")

    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        moe_to_attn_chip_sharded_bridge_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        b1_moe_chip_sharded_jax,
    )

    world_size, scale_block = 16, 512
    hidden, h_pre, qkv_cols, gate_cols = 8192, 4096, 640, 512
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager", detect_races=True, uninitialized_memory="nan"
    )
    mesh_cpu = Mesh(np.array(jax.devices("cpu")[:world_size]), ("model",))
    rng_256 = np.random.default_rng(20261219)
    d256 = _build_b1_moe_inputs(
        rng_256, world_size=world_size, batch_size=256, scale_block=scale_block,
        num_replicas=0, bf16_router=True, sparse_heavy=False,
    )
    # Balance 1 hot expert per rank across all 16 ranks so 16-CPU Interpret stays fast (~2x).
    b256_np = (rng_256.standard_normal((1, 256)) * 0.05).astype(np.float32)
    b256_np[0, 0:256:16] += 12.0
    d256["bias"] = jnp.asarray(b256_np)
    w_post_256 = jnp.asarray(
        np.concatenate([(rng_256.standard_normal((h_pre, h_pre)) * 0.04).astype(np.float32) for _ in range(world_size)], axis=0),
        dtype=jnp.float8_e4m3fn,
    )
    s_post_256 = jnp.asarray((np.abs(rng_256.standard_normal((h_pre,))) * 0.02 + 1e-3).astype(np.float32))
    post_ff_g = jnp.asarray((np.ones((hidden,)) * 1.02).astype(np.float32))
    pf_a = jnp.asarray((np.ones((hidden,)) * 0.93).astype(np.float32))
    pf_b = jnp.asarray((np.ones((hidden,)) * 0.88).astype(np.float32))
    fin_g = jnp.asarray((np.ones((hidden,)) * 1.05).astype(np.float32))
    w_qkv_256 = jnp.asarray(
        np.concatenate([(rng_256.standard_normal((hidden, qkv_cols)) * 0.04).astype(np.float32) for _ in range(world_size)], axis=0),
        dtype=jnp.float8_e4m3fn,
    )
    s_qkv_256 = jnp.asarray((np.abs(rng_256.standard_normal((qkv_cols,))) * 0.02 + 1e-3).astype(np.float32))
    w_gate_256 = jnp.asarray(
        np.concatenate([(rng_256.standard_normal((hidden, gate_cols)) * 0.04).astype(np.float32) for _ in range(world_size)], axis=0),
        dtype=jnp.float8_e4m3fn,
    )
    s_gate_256 = jnp.asarray((np.abs(rng_256.standard_normal((gate_cols,))) * 0.02 + 1e-3).astype(np.float32))
    cs_256 = jnp.asarray((rng_256.standard_normal((256, 2, 128)) * 0.3).astype(np.float32))
    cfg256 = dataclasses.replace(
        default_sparse_decode_config(256, world_size, scale_block),
        pair_rs=True, bridge_ag=True, spec_prime_parts=0,
    )
    ip_fast = pltpu.InterpretParams(dma_execution_mode="eager")
    ref_step_a = _make_standalone_step_a_ref_fn(
        world_size=world_size, scale_block=scale_block, hidden=hidden, h_pre=h_pre,
        batch_size=256, ret_sh=True, cfg_ref=cfg256, interp=ip_fast,
    )

    def _ref_b256(*args):
        st1, moe_sh = ref_step_a(*args[:18])
        b2_res = moe_to_attn_chip_sharded_bridge_jax(
            st1, moe_sh, *args[18:],
            world_size=world_size, sharded_moe_in=True, fuse_qkv_epi=True, interpret=ip_fast,
        )
        return st1, moe_sh, *b2_res

    def _fused_a256(*args):
        return b1_moe_chip_sharded_jax(
            *args[:18], world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            return_sharded=True, config=cfg256, interpret=ip_race,
        )

    def _fused_b256(*args):
        return b1_moe_b2_chip_sharded_jax(
            *args, world_size=world_size, pa_eps=1e-6, pre_ff_eps=1e-5, pre_ff_scale_input=False,
            router_multiplier=0.17677669529663687, pre_exp_eps=1e-5, pre_exp_scale_input=False,
            fuse_qkv_epi=True, config=cfg256, interpret=ip_race,
        )

    b256_args = (
        d256["ranks"], d256["stream_chip_st"], d256["gated_full"], d256["wo_sharded"], d256["wo_scale"],
        d256["pa_alpha"], d256["pa_beta"], d256["pre_ff_gain"],
        d256["w_pre_pair"], d256["s_pre_pair"], d256["w_router"], d256["pre_exp_gain"],
        d256["bias"], d256["w1_all"], d256["s1_all"], d256["w2_all"], d256["s2_all"], d256["post_exp_norm"],
        w_post_256, s_post_256, post_ff_g, pf_a, pf_b, fin_g,
        w_qkv_256, s_qkv_256, w_gate_256, s_gate_256, cs_256,
    )
    b256_specs = (
        P("model"), P("model", None), P(None, "model"), P("model", None), P(),
        P(), P(), P(),
        P(None, "model"), P("model"), P(), P(),
        P(), P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None), P(),
        P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(), P(),
    )
    ref_out_jax = jax.jit(
        jax.shard_map(_ref_b256, mesh=mesh_cpu, in_specs=b256_specs, out_specs=(P("model"),) * 6, check_vma=False)
    )(*b256_args)
    ref_np = [np.asarray(x, dtype=np.float32) for x in ref_out_jax]
    del ref_out_jax
    jax.clear_caches()

    got_a = jax.jit(
        jax.shard_map(_fused_a256, mesh=mesh_cpu, in_specs=b256_specs[:18], out_specs=(P("model"), P("model")), check_vma=False)
    )(*b256_args[:18])
    got_a_np = [np.asarray(x, dtype=np.float32) for x in got_a]
    del got_a
    jax.clear_caches()
    assert _cosine_sim(ref_np[0], got_a_np[0]) > 0.9999
    assert _cosine_sim(ref_np[1], got_a_np[1]) > 0.9999

    got_b = jax.jit(
        jax.shard_map(_fused_b256, mesh=mesh_cpu, in_specs=b256_specs, out_specs=(P("model"),) * 4, check_vma=False)
    )(*b256_args)
    got_b_np = [np.asarray(x, dtype=np.float32) for x in got_b]
    del got_b
    jax.clear_caches()
    for idx, nm in enumerate(("stream_chip", "q_5d", "kv_merged", "gate")):
        assert np.all(np.isfinite(got_b_np[idx])), f"b256 {nm} non-finite"
        assert _cosine_sim(ref_np[idx + 2], got_b_np[idx]) > 0.9999, f"b256 {nm} cos={_cosine_sim(ref_np[idx + 2], got_b_np[idx])}"


def test_tp16_b256_and_l61_b64_rep_tpu7x_aot():
    """Verify TPU7x 2x2x2 AOT compilation for L61 TP=16 B=64 rep and TP=16 B=256 Step (a)/(b)."""
    import gc
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
        b1_moe_chip_sharded_jax,
    )

    world_size, scale_block = 16, 512
    hidden, h_pre, inter = 8192, 4096, 4096
    e_local, o_in, pre_pair_cols, n_chips = 256 // world_size, hidden // world_size, h_pre // 2, world_size // 2
    nqb1, nqb2, qkv_cols, gate_cols = h_pre // scale_block, inter // scale_block, 640, 512

    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )

    def _base_sds(B: int):
        r_p = B // n_chips
        return (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((r_p, hidden), jnp.float32),
            jax.ShapeDtypeStruct((B, o_in), jnp.bfloat16),
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
        )

    rep8_sds = (
        jax.ShapeDtypeStruct((8, h_pre, 2 * inter), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((8, nqb1, 2 * inter), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, inter, h_pre), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((8, nqb2, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, world_size * 8), jnp.int32),
    )
    b256_aot_sds = _base_sds(256) + (
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
    )
    cs_sds = (jax.ShapeDtypeStruct((256, 2, 128), jnp.float32),)
    fp_sds = (jax.ShapeDtypeStruct((4,), jnp.float32),)
    a256_fp_sds = _base_sds(256) + rep8_sds + fp_sds
    b256_fp_nope_sds = b256_aot_sds + rep8_sds + fp_sds
    b256_fp_rope_sds = b256_aot_sds + cs_sds + rep8_sds + fp_sds

    def _call_b256_fp_nope(*args):
        # args = base(18) + b2(10) + rep(5) + free_params(1)
        return b1_moe_b2_chip_sharded_jax(
            *args[:28], None, *args[28:33], None, args[33], world_size=16, fuse_qkv_epi=True
        )

    def _call_b256_fp_rope(*args):
        # args = base(18) + b2(10) + cs(1) + rep(5) + free_params(1)
        return b1_moe_b2_chip_sharded_jax(
            *args[:34], None, args[34], world_size=16, fuse_qkv_epi=True
        )

    def _call_a256_fp(*args):
        # args = base(18) + rep(5) + free_params(1)
        return b1_moe_chip_sharded_jax(
            *args[:23], None, args[23], world_size=16, return_sharded=True
        )

    for fn_aot, sds_in, out_n in [
        (lambda *a: b1_moe_chip_sharded_jax(*a, world_size=16, return_sharded=False), _base_sds(64) + rep8_sds, 2),
        (_call_a256_fp, a256_fp_sds, 2),
        (_call_b256_fp_nope, b256_fp_nope_sds, 4),
        (_call_b256_fp_rope, b256_fp_rope_sds, 4),
    ]:
        comp = jax.jit(
            jax.shard_map(fn_aot, mesh=mesh_aot, in_specs=(P(),) * len(sds_in), out_specs=(P(),) * out_n, check_vma=False)
        ).trace(*sds_in).lower(lowering_platforms=("tpu",)).compile()
        assert comp.memory_analysis().generated_code_size_in_bytes > 0
        del comp
        jax.clear_caches()
        gc.collect()

