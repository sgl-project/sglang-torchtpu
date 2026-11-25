# SPDX-License-Identifier: Apache-2.0
"""Sparse & Dense-Lowtok Fuse-All Decode MoE Pallas kernel for Muse Spark.

Executes the entire decode MoE layer in a single Pallas launch:
  1. In-kernel tie-exact routing (``fused_decode_routing.py``) with NaN/-inf
     safety, producing replicated ``counts_smem[0, 256]`` and either
     ``dense_w_vmem[B, 256]`` (dense mode for small K256 batches) or
     VPU-compacted local-candidate hits in SMEM (sparse mode, 8x-16x less SALU).
  2. 32-step SMEM execution plan (plus optional §4.5b preloaded replica bank)
     that primes the weight prefetch ring BEFORE prologue zeroing & token pack:
       - ``stream3``: 3-unit (8+8+8 MiB gate / up / down) ring starting compute
         after an 8-MiB cold-start with 16-MiB prefetch depth in 24.4/30 MiB VMEM.
       - ``w1_2x_w2_1x``: 2x whole W1 (32 MiB) + 1x whole W2 (8 MiB) ring issuing
         4 unstrided linear full-matrix HBM bursts per expert.
       - ``half_n``: 2x 2048-col half-N ring.
  3. Collapsed 2-bucket ``lax.switch`` (< 25K bundles, 0 spills), wide K256 NC
     strips (2048 at m<=16, 1024 at m<=64), hoisted-``masked_x`` K16 blockdiag
     for ``r_e <= 8`` + ``jnp.repeat`` vectorized BF16 dequant for ``r_e > 8``,
     fused ``sq_sum`` per-expert RMSNorm, and 4x-unrolled VMEM gather/scatter.
  4. Zero-copy ``(EP - 1)`` contiguous Reduce-Scatter + flat ``(B, 32, 128)``
     All-Gather fan-out reusing ``x_3d_buf`` after the RS causal barrier.
"""

# PEP 563 stays off so TorchTPU / cpu_export_jax_op sees concrete annotations.

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_rs_muse.decode_hbm_breakers import (
    resolve_free_and_min_p_for_batch,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
    DenseRoutingScratch,
    RoutingScratch,
    build_default_exec_plan_smem,
    build_dense_exec_plan_smem,
    build_exec_plan_from_masks_smem,
    dense_routing_scratch_shapes,
    extract_my_holder_candidates_smem,
    extract_my_holder_candidates_tuple,
    pack_active_tokens_smem,
    parse_free_expert_rows_spec,
    route_dense_and_counts_in_kernel,
    route_topk_and_counts_in_kernel,
    routing_scratch_shapes,
    run_replica_planner_smem,
    select_non_replicated_home_slot,
)

_FP4_PACK = 8
_FP4_GROUP = 16
_KC = 256
_H = 4096
_I = 4096
_N1 = 8192
_E_TOTAL = 256
_TOPK = 8


@dataclasses.dataclass(frozen=True)
class SparseDecodeConfig:
    """Static compile-time configuration for ``fused_decode_sparse_from_logits``."""

    row_mode: str = "sparse"  # "sparse" or "dense" (for K256 B=16)
    weight_mode: str = "stream3"  # "stream3", "w1_2x_w2_2x", "w1_2x_w2_1x", "half_n"
    tile_m: int = 32  # max rows per expert row-tile (32, 64, or 128)
    m_buckets: tuple[int, ...] = (16, 32)
    loop32_in_tile: bool = True  # loop 32-row slices inside tile_m > 32 (zero m=64/128 VREG spills!)
    k16_blockdiag_rows: tuple[int, ...] = (1, 2, 4, 8)
    k16_bd_narrow_nc: bool = False  # False: wide bd strips (1024/512/512/256, -137 us at B=64!)
    k16_bd_fp8: bool = False  # True: per-(tok,16-grp) FP8 masked_x -> 4,614 TFLOP/s MXU!
    k16_dequant_mode: str = "repeat"  # "repeat" or "reshape3d"
    k16_strip_nc: int = 512
    k256_strip_mode: str = "wide"  # "wide" (cand_w8_zi default: 2048 at m<=16, 1024 at m<=64), "adaptive", "fixed512"
    k256_dual_mode: str = "hybrid"  # "vstage", "dual", "hybrid" (dual at m<=16, vstage at m>=32)
    nc_split512: bool = False  # opt-in: sub-tile N into <=512-col sub-strips inside _fp4_gmm_chunked
    k512_chain: bool = False  # opt-in: chain K256 MXU FP32 dots across scale_block=512 before 1x VPU scale
    unroll_strips: bool = False  # opt-in: Python-unroll n_strips <= 4 to avoid Ghostfish sbr MXU drains
    route_rb16_mode: int = 0  # 0 = cand_w8_zi rb=64 routing; 1 = 2-pass rb=16; 2 = 2-pass rb=16 unrolled
    route_fast: bool = False  # opt-in: [256, 128] vertical lane-parallel VPU routing (0 cross-lane butterflies!)
    fast_plan: int = 0  # 0 = stock; 1 = branchless 0-scf.if planner; 2 = 1 + unrolled k_rep<=64
    fast_gs: bool = False  # opt-in: 0-sbr branchless <=8-row gather/scatter (+1 trash row) + unrolled prologue/RS
    lhs_fp8: bool = False  # W4A8: online K256 FP8 activation quant
    rs_wire_bf16: bool = False
    pair_rs: bool = False  # 2-stage intra-chip XOR=1 paired Reduce-Scatter (halves off-chip ICI!)
    bridge_ag: bool = False  # 2-stage chipmate-bridge All-Gather (halves ICI fan-out bytes!)
    norm_f32_stage: bool = True
    route_chunk_rows: int = 32
    static_holder_mask: bool = False  # branchless scalar cand_eids + hoisted VREG replica compares
    spec_prime_parts: int = 0  # 0 = off; 1/2/3 weight chunks primed before routing
    spec_home_slots: tuple[int, ...] = ()  # optional per-rank local slot (0..e_local-1, -1=off)
    spec_skip_replicated: bool = True  # step past outgoing-replicated home slots when R > 0
    fast_b64_epi_pro: bool = False  # Wave-7: unrolled prologue/RS + loop-free <=8-row gather/scatter
    hoist_k256_scales: bool = True  # Wave-8: hoist K256 scale strip load outside K-chunk loop
    min_p_topk: float = 0.0  # Wave-8: <0.5 prob floor, 0.5..1.0 cumulative top-8 mass cap
    free_max_rows: int = 0  # Wave-8: batch-coupled free-expert tail pruning max rows
    free_max_w: float = 0.06
    free_min_slot: int = 5
    debug_stop_after: str = ""  # "", "route", "plan", "pack", "experts", "rs"


def _env_bool_fallback(name: str, default: bool) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_sparse_w9_sub_knobs() -> tuple[bool, bool, bool, int, bool, int]:
    import os

    nc_split = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_NC_SPLIT512", False)
    k512_ch = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_K512_CHAIN", False)
    unroll_st = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_UNROLL_STRIPS", False)
    route_fast = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST", False)
    fp_raw = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN")
    if fp_raw is not None and fp_raw.strip() != "":
        s_fp = fp_raw.strip().lower()
        fast_plan = 1 if s_fp in ("true", "yes", "on") else (0 if s_fp in ("false", "no", "off") else int(s_fp))
    else:
        fast_plan = 0
    rb16_raw = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_RB16")
    if rb16_raw is not None and rb16_raw.strip() != "":
        s = rb16_raw.strip().lower()
        rb16 = 1 if s in ("true", "yes", "on") else (0 if s in ("false", "no", "off") else int(s))
    else:
        rb16 = 0
        try:
            from sglang_tpu.envs import envs

            e_rb16 = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_RB16", None)
            if e_rb16 is not None:
                rb16 = int(e_rb16.get())
        except Exception:
            pass
    try:
        from sglang_tpu.envs import envs

        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_NC_SPLIT512") is None:
            e_nc = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_NC_SPLIT512", None)
            if e_nc is not None:
                nc_split = bool(e_nc.get())
        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_K512_CHAIN") is None:
            e_k5 = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_K512_CHAIN", None)
            if e_k5 is not None:
                k512_ch = bool(e_k5.get())
        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_UNROLL_STRIPS") is None:
            e_us = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_UNROLL_STRIPS", None)
            if e_us is not None:
                unroll_st = bool(e_us.get())
        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST") is None:
            e_rf = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST", None)
            if e_rf is not None:
                route_fast = bool(e_rf.get())
        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN") is None:
            e_fp = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN", None)
            if e_fp is not None:
                fast_plan = int(e_fp.get())
    except Exception:
        pass
    return nc_split, k512_ch, unroll_st, rb16, route_fast, int(fast_plan)


def _read_fast_gs_knob() -> bool:
    import os

    val = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_GS", False)
    if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_GS") is None:
        try:
            from sglang_tpu.envs import envs

            ev = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_GS", None)
            if ev is not None:
                val = bool(ev.get())
        except Exception:
            pass
    return val


def _read_spec_prime_fast_knobs() -> tuple[bool, int]:
    import os

    sp_fast = _env_bool_fallback(
        "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST", False
    )
    raw_slot = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_HOME_SLOT")
    home_slot = int(raw_slot) if (raw_slot is not None and raw_slot.strip() != "") else -1
    try:
        from sglang_tpu.envs import envs

        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST") is None:
            e_spf = getattr(
                envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST", None
            )
            if e_spf is not None:
                sp_fast = bool(e_spf.get())
        if raw_slot is None:
            e_shs = getattr(
                envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_HOME_SLOT", None
            )
            if e_shs is not None:
                home_slot = int(e_shs.get())
    except Exception:
        pass
    return sp_fast, home_slot


def _read_verify_w4a8_knobs() -> tuple[bool, bool]:
    import os

    lhs_fp8 = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8", False)
    tm128 = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_TILE_M128", False)
    try:
        from sglang_tpu.envs import envs

        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8") is None:
            e_lf = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8", None)
            if e_lf is not None:
                lhs_fp8 = bool(e_lf.get())
        if os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_TILE_M128") is None:
            e_tm = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_TILE_M128", None)
            if e_tm is not None:
                tm128 = bool(e_tm.get())
    except Exception:
        pass
    return lhs_fp8, tm128


def _read_sparse_env_overrides() -> (
    tuple[bool, bool, int, int, str, bool, bool, bool, float, int, float, int, int, str]
):
    import os

    decode_m8 = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_M8", False)
    hoist_sc = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_HOIST_SCALES", True)
    contig_p = _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS", False)
    min_p = float(os.environ.get("SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK", "0.0") or 0.0)
    free_spec = os.environ.get("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS", "")
    pair_rs_raw = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS")
    pair_rs = int(pair_rs_raw) if (pair_rs_raw is not None and pair_rs_raw.strip() != "") else -1
    strip_mode = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_STRIP_MODE", "").strip()
    try:
        from sglang_tpu.envs import envs

        fast_b64_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64", None)
        fp8_mode_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_MODE", None)
        m8_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_M8", None)
        hoist_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_HOIST_SCALES", None)
        contig_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS", None)
        minp_env = getattr(envs, "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK", None)
        free_env = getattr(envs, "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS", None)
        pair_rs_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS", None)
        if m8_env is not None:
            decode_m8 = bool(m8_env.get())
        if hoist_env is not None:
            hoist_sc = bool(hoist_env.get())
        if contig_env is not None:
            contig_p = bool(contig_env.get())
        if minp_env is not None:
            min_p = float(minp_env.get())
        if free_env is not None:
            free_spec = str(free_env.get())
        if pair_rs_raw is None and pair_rs_env is not None:
            pair_rs = int(pair_rs_env.get())
        f_rows, f_max_w, f_min_s = parse_free_expert_rows_spec(free_spec)
        return (
            bool(envs.SGLANG_TPU_MUSE_KERNEL_DECODE_STATIC_HOLDER_MASK.get()),
            bool(envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_BRIDGE64.get()),
            max(0, min(3, int(envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME.get()))),
            int(fast_b64_env.get()) if fast_b64_env is not None else 0,
            str(fp8_mode_env.get()).strip() if fp8_mode_env is not None else "",
            decode_m8,
            hoist_sc,
            contig_p,
            max(0.0, float(min_p)),
            f_rows,
            f_max_w,
            f_min_s,
            pair_rs,
            strip_mode,
        )
    except Exception:
        f_rows, f_max_w, f_min_s = parse_free_expert_rows_spec(free_spec)
        return (
            _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_STATIC_HOLDER_MASK", False),
            _env_bool_fallback("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_BRIDGE64", False),
            max(0, min(3, int(os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME", "0") or 0))),
            int(os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64", "0") or 0),
            os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_MODE", "").strip(),
            decode_m8,
            hoist_sc,
            contig_p,
            max(0.0, float(min_p)),
            f_rows,
            f_max_w,
            f_min_s,
            pair_rs,
            strip_mode,
        )


def default_sparse_decode_config(
    batch_size: int, ep: int, scale_block: int
) -> SparseDecodeConfig:
    """Return the tuned default configuration for ``(batch_size, ep, scale_block)``."""
    (
        static_mask,
        bridge64,
        spec_parts,
        fast_b64,
        _,
        decode_m8,
        hoist_sc,
        contig_parts,
        min_p_topk,
        free_rows,
        free_max_w,
        free_min_slot,
        pair_rs_ov,
        strip_mode_ov,
    ) = _read_sparse_env_overrides()
    spec_eff = spec_parts if batch_size <= 128 else 0
    use_br64 = bridge64 and (ep in (8, 16)) and (batch_size == 64)
    use_pair_rs = (
        (pair_rs_ov > 0)
        and (ep in (8, 16))
        and (batch_size >= 32 or pair_rs_ov >= 2)
    )
    use_ep16_bridge_ag = use_pair_rs and (ep == 16)
    eff_strip = strip_mode_ov if strip_mode_ov else "wide"
    (
        nc_split512,
        k512_chain,
        unroll_strips,
        route_rb16_mode,
        route_fast,
        fast_plan,
    ) = _read_sparse_w9_sub_knobs()
    spec_prime_fast, spec_home_slot = _read_spec_prime_fast_knobs()
    if route_fast and not spec_prime_fast:
        # Silicon-proven (#2647 / #2767): vertical ROUTE_FAST (~3.5 us) finishes before
        # speculative slot-0 16 MiB DMA (~4.3 us), causing _drain_spec_miss stalls (+4.55 us).
        # Keep spec_eff=0 unless explicitly opted into SPEC_PRIME_FAST=1 (#3035).
        spec_eff = 0
    spec_home_slots = (
        (spec_home_slot,) * ep
        if (spec_eff > 0 and 0 <= spec_home_slot < (_E_TOTAL // ep))
        else ()
    )
    min_p_eff, free_rows_eff, free_max_w_eff, free_min_slot_eff = (
        resolve_free_and_min_p_for_batch(
            batch_size,
            min_p_topk=min_p_topk,
            free_rows=free_rows,
            free_max_w=free_max_w,
            free_min_slot=free_min_slot,
            parse_free_spec_fn=parse_free_expert_rows_spec,
        )
    )
    fast_gs = _read_fast_gs_knob()
    use_verify_lhs_fp8, use_verify_tm128 = _read_verify_w4a8_knobs()
    route_kw = dict(
        hoist_k256_scales=hoist_sc,
        nc_split512=nc_split512,
        k512_chain=k512_chain,
        unroll_strips=unroll_strips,
        route_rb16_mode=route_rb16_mode,
        route_fast=route_fast,
        fast_plan=fast_plan,
        fast_gs=fast_gs,
        spec_home_slots=spec_home_slots,
        min_p_topk=min_p_eff,
        free_max_rows=free_rows_eff,
        free_max_w=free_max_w_eff,
        free_min_slot=free_min_slot_eff,
    )
    if scale_block == 16:
        tile_m = (
            32
            if batch_size <= 64
            else (64 if (batch_size in (128, 384) or ep > 8) else 128)
        )
        return SparseDecodeConfig(
            row_mode="sparse",
            weight_mode="w1_2x_w2_1x" if (fast_b64 == 2 and batch_size <= 64) else "half_n",
            tile_m=tile_m,
            m_buckets=(8, 32) if (decode_m8 and batch_size <= 64) else (16, 32),
            loop32_in_tile=True,
            k16_blockdiag_rows=(1, 2, 4, 8),
            k16_bd_narrow_nc=False,
            k16_bd_fp8=(fast_b64 >= 1 and batch_size <= 64),
            k16_strip_nc=512,
            rs_wire_bf16=(batch_size >= 128) or use_br64,
            pair_rs=use_pair_rs,
            bridge_ag=use_ep16_bridge_ag or (ep == 8 and batch_size >= 128) or use_br64,
            norm_f32_stage=True,
            route_chunk_rows=batch_size if (fast_b64 >= 1 and batch_size <= 64) else min(batch_size, 32),
            static_holder_mask=static_mask,
            spec_prime_parts=spec_eff,
            fast_b64_epi_pro=(fast_b64 >= 1 and batch_size <= 64),
            **route_kw,
        )

    # Compact K256/K512: dense fast-path at B=16 (or B<=64 when fast_b64==3), 2/3-bucket sparse at B in (32, 64),
    # 3-bucket (16, 32, 64) unlooped sparse + BF16 RS wire at B >= 128 (~38 MiB VMEM).
    if batch_size <= 16 or (fast_b64 == 3 and batch_size <= 64):
        return SparseDecodeConfig(
            row_mode="dense",
            weight_mode="stream3_k2" if (contig_parts and scale_block <= 256) else "stream3",
            tile_m=batch_size,
            m_buckets=(batch_size,),
            loop32_in_tile=False,
            k16_blockdiag_rows=(),
            k256_strip_mode=eff_strip,
            k256_dual_mode="vstage",
            rs_wire_bf16=use_br64 if batch_size == 64 else False,
            pair_rs=use_pair_rs,
            bridge_ag=use_ep16_bridge_ag or (use_br64 if batch_size == 64 else False),
            norm_f32_stage=True,
            route_chunk_rows=batch_size,
            static_holder_mask=static_mask,
            spec_prime_parts=spec_eff,
            fast_b64_epi_pro=(fast_b64 >= 1),
            **route_kw,
        )
    if batch_size <= 64:
        w_m = (
            "w1_2x_w2_1x"
            if fast_b64 == 2
            else ("stream3_k2" if (contig_parts and scale_block <= 256) else "stream3")
        )
        m_bkts = (
            (8, 16, 32)
            if fast_b64 in (1, 2)
            else ((8, 32) if decode_m8 else (16, 32))
        )
        return SparseDecodeConfig(
            row_mode="sparse",
            weight_mode=w_m,
            tile_m=32,
            m_buckets=m_bkts,
            loop32_in_tile=False,
            k16_blockdiag_rows=(),
            k256_strip_mode=eff_strip,
            k256_dual_mode="dual" if fast_b64 == 2 else "hybrid",
            rs_wire_bf16=use_br64,
            pair_rs=use_pair_rs,
            bridge_ag=use_ep16_bridge_ag or use_br64,
            norm_f32_stage=True,
            route_chunk_rows=batch_size if fast_b64 >= 1 else 32,
            static_holder_mask=static_mask,
            spec_prime_parts=spec_eff,
            fast_b64_epi_pro=(fast_b64 >= 1),
            **route_kw,
        )
    use_v_tm128 = use_verify_tm128 and batch_size >= 256
    return SparseDecodeConfig(
        row_mode="sparse",
        weight_mode="half_n",
        tile_m=128 if use_v_tm128 else 64,
        m_buckets=(16, 32, 64, 128) if use_v_tm128 else (16, 32, 64),
        loop32_in_tile=False,
        k16_blockdiag_rows=(),
        k256_strip_mode=eff_strip,
        k256_dual_mode="hybrid",
        lhs_fp8=use_verify_lhs_fp8 and batch_size >= 256,
        rs_wire_bf16=True,
        pair_rs=use_pair_rs,
        bridge_ag=(ep == 8) or use_ep16_bridge_ag,
        norm_f32_stage=True,
        route_chunk_rows=32,
        static_holder_mask=static_mask,
        spec_prime_parts=spec_eff,
        **route_kw,
    )


def _write_sharded_owner_out(output_ref, owner_3d, rows_per_rank: int) -> None:
    if rows_per_rank >= 8 and rows_per_rank % 8 == 0:
        step_r = 16 if (rows_per_rank % 16 == 0) else 8
        for r0 in range(0, rows_per_rank, step_r):
            output_ref[pl.ds(r0, step_r), :] = owner_3d[
                pl.ds(r0, step_r), :, :
            ].reshape(step_r, _H)
    else:
        output_ref[...] = owner_3d[...].reshape(rows_per_rank, _H)


def _pad_rows_to(arr: jax.Array, rows: int) -> jax.Array:
    if arr.shape[0] == rows:
        return arr
    return jnp.concatenate(
        [arr, jnp.zeros((rows - arr.shape[0], arr.shape[1]), arr.dtype)], axis=0
    )


def _pick_k256_nc(m_rows: int, max_n: int, strip_mode: str) -> int:
    if strip_mode == "fixed256":
        return min(256, max_n)
    if strip_mode == "fixed512":
        return min(512, max_n)
    if strip_mode == "fixed1024":
        return min(1024, max_n)
    if strip_mode == "fixed2048":
        return min(2048, max_n)
    if strip_mode == "wide":
        # Exact STRIP_NC=-5 schedule: 2048 at m<=16, 1024 at m<=64, 512 above.
        nc = 2048 if m_rows <= 16 else (1024 if m_rows <= 64 else 512)
        return min(nc, max_n)
    # "adaptive"
    nc = 1024 if m_rows <= 16 else 512
    return min(nc, max_n)


def _pick_bd_nc(bd_rows: int, max_n: int, narrow: bool = True) -> int:
    # res is [bd_rows * 16, nc] f32 (2*bd_rows*nc//128 VREGs), acc is half that.
    # With narrow=True, res + acc + s <= 28 VREGs across all bd_rows -> 0 spills!
    if narrow:
        if bd_rows <= 2:
            return min(512, max_n)
        if bd_rows <= 4:
            return min(256, max_n)
        return min(128, max_n)
    if bd_rows == 1:
        return min(1024, max_n)
    if bd_rows <= 4:
        return min(512, max_n)
    if bd_rows <= 8:
        return min(256, max_n)
    return min(128, max_n)


def _prequant_lhs_k256(x_src_ref, lhs_fp8_ref, lhs_sc_ref, m_rows: int) -> None:
    """Online-quantize ``x_src_ref[0:m_rows, 4096]`` per K256 block into FP8 + scale."""
    read_m = 16 if m_rows == 8 else m_rows
    fp8_m = max(32, read_m)
    for b_id in range(_H // _KC):
        c0 = b_id * _KC
        xf = x_src_ref[pl.ds(0, read_m), pl.ds(c0, _KC)].astype(jnp.float32)
        amax = jnp.max(jnp.abs(xf), axis=1, keepdims=True)
        sc = jnp.maximum(amax, 1e-12) * (1.0 / 448.0)
        xq = jnp.clip(xf / sc, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        lhs_fp8_ref[pl.ds(0, fp8_m), pl.ds(c0, _KC)] = _pad_rows_to(xq, fp8_m)
        lhs_sc_ref[pl.ds(0, read_m), pl.ds(b_id * 128, 128)] = jnp.broadcast_to(
            sc, (read_m, 128)
        )


def _fp4_gmm_chunked(
    x_2d_ref,
    w_ref,
    s_ref,
    m_rows: int,
    k_dim: int,
    n0: jax.Array | int,
    nc: int,
    block256: bool,
    scale_block: int,
    *,
    k16_dequant_mode: str = "repeat",
    lhs_sc_ref=None,
    hoist_scales: bool = True,
    k_offset: int = 0,
    init_acc: jax.Array | None = None,
    nc_split512: bool = False,
    k512_chain: bool = False,
) -> jax.Array:
    """Compute ``x[0:m_rows, k_offset:k_offset+k_dim] @ w[:, n0:n0+nc]`` in FP32 over K256 chunks."""
    gpc = _KC // _FP4_GROUP
    read_m = 16 if m_rows == 8 else m_rows
    use_w4a8 = block256 and (lhs_sc_ref is not None)
    if block256 and not use_w4a8 and (nc_split512 or k512_chain):
        eff_sb = scale_block if k512_chain else _KC
        n_sb = k_dim // eff_sb
        n_real_sb = k_dim // scale_block
        sub_nc = min(nc, 512) if nc_split512 else nc
        n_sub = nc // sub_nc
        can_hoist_s = (
            hoist_scales
            and n_real_sb >= 8
            and (n_real_sb % 8 == 0)
            and ((m_rows * sub_nc) // 1024 <= 16 or sub_nc <= 1024 or (k_dim // _KC) <= 8)
        )
        chunks_per_sb = eff_sb // _KC
        accs: list[jax.Array] = []
        for si in range(n_sub):
            col0 = n0 + si * sub_nc
            s_sub = (
                s_ref[pl.ds(0, n_real_sb), pl.ds(col0, sub_nc)]
                if can_hoist_s
                else None
            )
            acc_si = (
                init_acc[:, si * sub_nc : (si + 1) * sub_nc]
                if init_acc is not None
                else None
            )
            for sb_id in range(n_sb):
                c_base = sb_id * eff_sb
                blk_acc = None
                for ci in range(chunks_per_sb):
                    c0 = c_base + ci * _KC
                    x = x_2d_ref[pl.ds(0, read_m), pl.ds(k_offset + c0, _KC)]
                    if read_m != m_rows:
                        x = x[:m_rows, :]
                    w_fp8 = pltpu.bitcast(
                        w_ref[
                            pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK),
                            pl.ds(col0, sub_nc),
                        ],
                        jnp.float4_e2m1fn,
                    ).astype(jnp.float8_e4m3fn)
                    part = jnp.dot(
                        x, w_fp8, preferred_element_type=jnp.float32
                    )
                    blk_acc = part if blk_acc is None else blk_acc + part
                real_sb = c_base // scale_block
                if s_sub is not None:
                    s = s_sub[real_sb : real_sb + 1, :].astype(jnp.float32)
                else:
                    s = s_ref[pl.ds(real_sb, 1), pl.ds(col0, sub_nc)].astype(
                        jnp.float32
                    )
                scaled = blk_acc * s
                acc_si = scaled if acc_si is None else acc_si + scaled
            accs.append(acc_si)
        return accs[0] if n_sub == 1 else jnp.concatenate(accs, axis=1)

    x_read_m = max(32, read_m) if use_w4a8 else read_m
    n_kc = k_dim // _KC
    n_sb = (k_dim // scale_block) if block256 else 0
    can_hoist_s = (
        block256
        and hoist_scales
        and n_sb >= 8
        and (n_sb % 8 == 0)
        and ((m_rows * nc) // 1024 <= 16 or nc <= 1024 or n_kc <= 8)
    )
    s_all = (
        s_ref[pl.ds(0, n_sb), pl.ds(n0, nc)]
        if can_hoist_s
        else None
    )
    acc = init_acc
    b_off = k_offset // _KC
    for b_id in range(n_kc):
        c0 = b_id * _KC
        x = x_2d_ref[pl.ds(0, x_read_m), pl.ds(k_offset + c0, _KC)]
        if x_read_m != m_rows:
            x = x[:m_rows, :]
        w = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK), pl.ds(n0, nc)],
            jnp.float4_e2m1fn,
        )
        if block256:
            res = jnp.dot(
                x, w.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
            )
            if s_all is not None:
                sb_id = c0 // scale_block
                s = s_all[sb_id : sb_id + 1, :].astype(jnp.float32)
            else:
                s = s_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0, nc)].astype(
                    jnp.float32
                )
            if use_w4a8:
                x_sc = lhs_sc_ref[
                    pl.ds(0, read_m), pl.ds((b_off + b_id) * 128, 128)
                ]
                if m_rows == 8:
                    x_sc = x_sc[:8, :]
                res = (res * x_sc[:, 0:1]) * s
            else:
                res = res * s
        else:
            s = s_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0, nc)].astype(
                jnp.bfloat16
            )
            if k16_dequant_mode == "reshape3d":
                w_bf = (
                    w.astype(jnp.bfloat16).reshape(gpc, _FP4_GROUP, nc)
                    * s[:, None, :]
                ).reshape(_KC, nc)
            else:
                w_bf = w.astype(jnp.bfloat16) * jnp.repeat(
                    s, _FP4_GROUP, axis=0
                )
            res = jnp.dot(x, w_bf, preferred_element_type=jnp.float32)
        acc = res if acc is None else acc + res
    return acc


def _fp4_dual_gmm_chunked(
    x_2d_ref,
    w_ref,
    s_ref,
    m_rows: int,
    k_dim: int,
    n0_g: jax.Array | int,
    n0_u: jax.Array | int,
    nc: int,
    block256: bool,
    scale_block: int,
    *,
    k16_dequant_mode: str = "repeat",
    nc_split512: bool = False,
    k512_chain: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Interleaved gate + up K256 matmul reading ``x`` once per K256 chunk."""
    gpc = _KC // _FP4_GROUP
    read_m = 16 if m_rows == 8 else m_rows
    if block256 and (nc_split512 or k512_chain):
        eff_sb = scale_block if k512_chain else _KC
        n_sb = k_dim // eff_sb
        n_real_sb = k_dim // scale_block
        sub_nc = min(nc, 512) if nc_split512 else nc
        n_sub = nc // sub_nc
        can_hoist = n_real_sb >= 8 and (n_real_sb % 8 == 0)
        chunks_per_sb = eff_sb // _KC
        accs_g: list[jax.Array] = []
        accs_u: list[jax.Array] = []
        for si in range(n_sub):
            cg = n0_g + si * sub_nc
            cu = n0_u + si * sub_nc
            sg_sub = s_ref[pl.ds(0, n_real_sb), pl.ds(cg, sub_nc)] if can_hoist else None
            su_sub = s_ref[pl.ds(0, n_real_sb), pl.ds(cu, sub_nc)] if can_hoist else None
            acc_g_si = None
            acc_u_si = None
            for sb_id in range(n_sb):
                c_base = sb_id * eff_sb
                bg = None
                bu = None
                for ci in range(chunks_per_sb):
                    c0 = c_base + ci * _KC
                    x = x_2d_ref[pl.ds(0, read_m), pl.ds(c0, _KC)]
                    if m_rows == 8:
                        x = x[:8, :]
                    wg_fp8 = pltpu.bitcast(
                        w_ref[
                            pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK),
                            pl.ds(cg, sub_nc),
                        ],
                        jnp.float4_e2m1fn,
                    ).astype(jnp.float8_e4m3fn)
                    rg = jnp.dot(x, wg_fp8, preferred_element_type=jnp.float32)
                    bg = rg if bg is None else bg + rg
                    wu_fp8 = pltpu.bitcast(
                        w_ref[
                            pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK),
                            pl.ds(cu, sub_nc),
                        ],
                        jnp.float4_e2m1fn,
                    ).astype(jnp.float8_e4m3fn)
                    ru = jnp.dot(x, wu_fp8, preferred_element_type=jnp.float32)
                    bu = ru if bu is None else bu + ru
                real_sb = c_base // scale_block
                sg = (
                    sg_sub[real_sb : real_sb + 1, :].astype(jnp.float32)
                    if sg_sub is not None
                    else s_ref[pl.ds(real_sb, 1), pl.ds(cg, sub_nc)].astype(jnp.float32)
                )
                su = (
                    su_sub[real_sb : real_sb + 1, :].astype(jnp.float32)
                    if su_sub is not None
                    else s_ref[pl.ds(real_sb, 1), pl.ds(cu, sub_nc)].astype(jnp.float32)
                )
                gs_v = bg * sg
                us_v = bu * su
                acc_g_si = gs_v if acc_g_si is None else acc_g_si + gs_v
                acc_u_si = us_v if acc_u_si is None else acc_u_si + us_v
            accs_g.append(acc_g_si)
            accs_u.append(acc_u_si)
        out_g = accs_g[0] if n_sub == 1 else jnp.concatenate(accs_g, axis=1)
        out_u = accs_u[0] if n_sub == 1 else jnp.concatenate(accs_u, axis=1)
        return out_g, out_u
    acc_g = None
    acc_u = None
    for c0 in range(0, k_dim, _KC):
        x = x_2d_ref[pl.ds(0, read_m), pl.ds(c0, _KC)]
        if m_rows == 8:
            x = x[:8, :]
        wg = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK), pl.ds(n0_g, nc)],
            jnp.float4_e2m1fn,
        )
        wu = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK), pl.ds(n0_u, nc)],
            jnp.float4_e2m1fn,
        )
        if block256:
            rg = jnp.dot(
                x, wg.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
            )
            sg = s_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0_g, nc)].astype(
                jnp.float32
            )
            rg = rg * sg
            ru = jnp.dot(
                x, wu.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
            )
            su = s_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0_u, nc)].astype(
                jnp.float32
            )
            ru = ru * su
        else:
            sg = s_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0_g, nc)].astype(
                jnp.bfloat16
            )
            su = s_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0_u, nc)].astype(
                jnp.bfloat16
            )
            if k16_dequant_mode == "reshape3d":
                wg_bf = (
                    wg.astype(jnp.bfloat16).reshape(gpc, _FP4_GROUP, nc)
                    * sg[:, None, :]
                ).reshape(_KC, nc)
                wu_bf = (
                    wu.astype(jnp.bfloat16).reshape(gpc, _FP4_GROUP, nc)
                    * su[:, None, :]
                ).reshape(_KC, nc)
            else:
                wg_bf = wg.astype(jnp.bfloat16) * jnp.repeat(
                    sg, _FP4_GROUP, axis=0
                )
                wu_bf = wu.astype(jnp.bfloat16) * jnp.repeat(
                    su, _FP4_GROUP, axis=0
                )
            rg = jnp.dot(x, wg_bf, preferred_element_type=jnp.float32)
            ru = jnp.dot(x, wu_bf, preferred_element_type=jnp.float32)
        acc_g = rg if acc_g is None else acc_g + rg
        acc_u = ru if acc_u is None else acc_u + ru
    return acc_g, acc_u


def _build_masked_x_from_3d(
    lhs_3d_ref, masked_x_ref, bd_rows: int, bd_sc_ref=None
) -> None:
    """Build block-diagonal ``[bd_rows * 16, 4096]`` bf16 (or per-group fp8) once per tile."""
    gpc = _KC // _FP4_GROUP
    diag_16 = (
        lax.broadcasted_iota(jnp.int32, (gpc, _KC), 0)
        == lax.broadcasted_iota(jnp.int32, (gpc, _KC), 1) // _FP4_GROUP
    )
    use_fp8 = bd_sc_ref is not None
    if use_fp8:
        zeros_16 = jnp.zeros((gpc, _KC), dtype=jnp.float8_e4m3fn)
        rows_2d = [
            lhs_3d_ref[pl.ds(a, 1), :, :].reshape(1, _H) for a in range(bd_rows)
        ]
        # Native (16, 256) T(8,128) diagonal max + quant — zero sub-128-lane reshape!
        for pair0 in range(0, max(2, bd_rows), 2):
            dst_32 = pl.ds(pair0 * gpc, 2 * gpc)
            for b_id in range(_H // _KC):
                c0 = b_id * _KC
                halves_mx = []
                for sub_a in range(2):
                    a = pair0 + sub_a
                    if a < bd_rows:
                        bcast = jnp.broadcast_to(
                            rows_2d[a][:, c0 : c0 + _KC].astype(jnp.float32),
                            (gpc, _KC),
                        )
                        masked = jnp.where(diag_16, bcast, 0.0)
                        amax = jnp.max(jnp.abs(masked), axis=1, keepdims=True)
                        sc = jnp.maximum(amax, 1e-12) * (1.0 / 448.0)
                        halves_mx.append(
                            (masked / sc).astype(jnp.float8_e4m3fn)
                        )
                        bd_sc_ref[pl.ds(a * gpc, gpc), pl.ds(b_id * 128, 128)] = (
                            jnp.broadcast_to(sc, (gpc, 128))
                        )
                    else:
                        halves_mx.append(zeros_16)
                masked_x_ref[dst_32, pl.ds(c0, _KC)] = jnp.concatenate(
                    halves_mx, axis=0
                )
        return
    zeros_16 = jnp.zeros((gpc, _KC), dtype=jnp.bfloat16)
    for a in range(bd_rows):
        row_2d = lhs_3d_ref[pl.ds(a, 1), :, :].reshape(1, _H)
        dst_rows = pl.ds(a * gpc, gpc)
        for b_id in range(_H // _KC):
            c0 = b_id * _KC
            bcast = jnp.broadcast_to(row_2d[:, c0 : c0 + _KC], (gpc, _KC))
            masked_x_ref[dst_rows, pl.ds(c0, _KC)] = jnp.where(
                diag_16, bcast, zeros_16
            )


def _build_masked_x_from_2d(
    src_2d_ref, masked_x_ref, bd_rows: int, bd_sc_ref=None
) -> None:
    """Build block-diagonal ``[bd_rows * 16, 4096]`` bf16 (or per-group fp8) from 2D."""
    gpc = _KC // _FP4_GROUP
    diag_16 = (
        lax.broadcasted_iota(jnp.int32, (gpc, _KC), 0)
        == lax.broadcasted_iota(jnp.int32, (gpc, _KC), 1) // _FP4_GROUP
    )
    use_fp8 = bd_sc_ref is not None
    if use_fp8:
        zeros_16 = jnp.zeros((gpc, _KC), dtype=jnp.float8_e4m3fn)
        for s in range(4):
            c_base = s * 1024
            strip = src_2d_ref[pl.ds(0, 16), pl.ds(c_base, 1024)]
            for pair0 in range(0, max(2, bd_rows), 2):
                dst_32 = pl.ds(pair0 * gpc, 2 * gpc)
                for sub in range(4):
                    b_id = s * 4 + sub
                    off = sub * _KC
                    halves_mx = []
                    for sub_a in range(2):
                        a = pair0 + sub_a
                        if a < bd_rows:
                            bcast = jnp.broadcast_to(
                                strip[a : a + 1, off : off + _KC].astype(
                                    jnp.float32
                                ),
                                (gpc, _KC),
                            )
                            masked = jnp.where(diag_16, bcast, 0.0)
                            amax = jnp.max(
                                jnp.abs(masked), axis=1, keepdims=True
                            )
                            sc = jnp.maximum(amax, 1e-12) * (1.0 / 448.0)
                            halves_mx.append(
                                (masked / sc).astype(jnp.float8_e4m3fn)
                            )
                            bd_sc_ref[
                                pl.ds(a * gpc, gpc), pl.ds(b_id * 128, 128)
                            ] = jnp.broadcast_to(sc, (gpc, 128))
                        else:
                            halves_mx.append(zeros_16)
                    masked_x_ref[dst_32, pl.ds(c_base + off, _KC)] = (
                        jnp.concatenate(halves_mx, axis=0)
                    )
        return
    zeros_16 = jnp.zeros((gpc, _KC), dtype=jnp.bfloat16)
    for s in range(4):
        c_base = s * 1024
        strip = src_2d_ref[pl.ds(0, 16), pl.ds(c_base, 1024)]
        for a in range(bd_rows):
            dst_rows = pl.ds(a * gpc, gpc)
            for sub in range(4):
                off = sub * _KC
                bcast = jnp.broadcast_to(
                    strip[a : a + 1, off : off + _KC], (gpc, _KC)
                )
                masked_x_ref[dst_rows, pl.ds(c_base + off, _KC)] = jnp.where(
                    diag_16, bcast, zeros_16
                )


def _fp4_blockdiag_hoisted_strip(
    masked_x_ref,
    w_ref,
    s_ref,
    bd_rows: int,
    n0: jax.Array | int,
    nc: int,
    bd_sc_ref=None,
) -> jax.Array:
    """Block-diagonal K16 matmul over ``[n0 : n0+nc]`` using hoisted ``masked_x``."""
    gpc = _KC // _FP4_GROUP
    rows = bd_rows * gpc
    mx_read_rows = max(32, rows) if bd_sc_ref is not None else rows
    acc = None
    for b_id in range(_H // _KC):
        c0 = b_id * _KC
        mx = masked_x_ref[pl.ds(0, mx_read_rows), pl.ds(c0, _KC)]
        if mx_read_rows != rows:
            mx = mx[:rows, :]
        w = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK), pl.ds(n0, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(mx, w, preferred_element_type=jnp.float32)
        if bd_sc_ref is not None:
            x_sc = bd_sc_ref[pl.ds(0, rows), pl.ds(b_id * 128, 128)][:, 0:1]
            res = res * x_sc
        s = s_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0, nc)].astype(jnp.float32)
        part = (
            (res.reshape(bd_rows, 2, 8, nc) * s.reshape(1, 2, 8, nc))
            .sum(axis=1)
            .reshape(bd_rows * 8, nc)
        )
        acc = part if acc is None else acc + part
    return jnp.concatenate(
        [
            jnp.sum(acc[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
            for a in range(bd_rows)
        ],
        axis=0,
    )


def _fp4_blockdiag_hoisted_dual_strip(
    masked_x_ref,
    w_ref,
    s_ref,
    bd_rows: int,
    n0_g: jax.Array | int,
    n0_u: jax.Array | int,
    nc: int,
    bd_sc_ref=None,
) -> tuple[jax.Array, jax.Array]:
    """Dual gate+up block-diagonal K16 strip sharing one ``masked_x`` load per K256."""
    gpc = _KC // _FP4_GROUP
    rows = bd_rows * gpc
    mx_read_rows = max(32, rows) if bd_sc_ref is not None else rows
    acc_g = None
    acc_u = None
    for b_id in range(_H // _KC):
        c0 = b_id * _KC
        mx = masked_x_ref[pl.ds(0, mx_read_rows), pl.ds(c0, _KC)]
        if mx_read_rows != rows:
            mx = mx[:rows, :]
        x_sc = (
            bd_sc_ref[pl.ds(0, rows), pl.ds(b_id * 128, 128)][:, 0:1]
            if bd_sc_ref is not None
            else None
        )
        wg = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK), pl.ds(n0_g, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        rg = jnp.dot(mx, wg, preferred_element_type=jnp.float32)
        if x_sc is not None:
            rg = rg * x_sc
        sg = s_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0_g, nc)].astype(
            jnp.float32
        )
        pg = (
            (rg.reshape(bd_rows, 2, 8, nc) * sg.reshape(1, 2, 8, nc))
            .sum(axis=1)
            .reshape(bd_rows * 8, nc)
        )
        wu = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, _KC // _FP4_PACK), pl.ds(n0_u, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        ru = jnp.dot(mx, wu, preferred_element_type=jnp.float32)
        if x_sc is not None:
            ru = ru * x_sc
        su = s_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0_u, nc)].astype(
            jnp.float32
        )
        pu = (
            (ru.reshape(bd_rows, 2, 8, nc) * su.reshape(1, 2, 8, nc))
            .sum(axis=1)
            .reshape(bd_rows * 8, nc)
        )
        acc_g = pg if acc_g is None else acc_g + pg
        acc_u = pu if acc_u is None else acc_u + pu
    out_g = jnp.concatenate(
        [
            jnp.sum(acc_g[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
            for a in range(bd_rows)
        ],
        axis=0,
    )
    out_u = jnp.concatenate(
        [
            jnp.sum(acc_u[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
            for a in range(bd_rows)
        ],
        axis=0,
    )
    return out_g, out_u


@dataclasses.dataclass(frozen=True)
class _WeightDmaOps:
    """Async HBM->VMEM weight streaming primitives for a sparse decode kernel."""

    w_mode: str
    issue_spec_prime: object
    prime_first_exec: object
    start_s3_part: object
    wait_s3_part: object
    start_s3_down: object
    wait_s3_down: object
    start_w1_slot: object
    wait_w1_slot: object
    start_w2_slot: object
    wait_w2_slot: object
    start_w2_1x: object
    wait_w2_1x: object
    start_w1_half: object
    wait_w1_half: object
    start_w2_half: object
    wait_w2_half: object


@dataclasses.dataclass(frozen=True)
class MuseSparsePlanResult:
    """Routing and exec-plan state returned by ``plan_and_prime_muse_sparse_from_vmem``."""

    num_exec: jax.Array
    spec_slot: jax.Array | None
    spec_enabled: jax.Array
    is_dense: bool
    is_k256: bool
    w_mode: str
    use_bd: bool
    has_replicas: bool
    use_lhs_fp8: bool
    use_pair_rs: bool
    exec_eid_smem: object
    exec_src_smem: object
    exec_slot_smem: object
    dma_ops: _WeightDmaOps


def _make_sparse_weight_dma_ops(
    w1_u32,
    s1_ref,
    w2_u32,
    s2_ref,
    rep_w1_u32,
    rep_s1_ref,
    rep_w2_u32,
    rep_s2_ref,
    w1_buf,
    s1_buf,
    w2_buf,
    s2_buf,
    w1_sem,
    w2_sem,
    *,
    w_mode: str,
    scale_block: int,
    has_replicas: bool,
) -> _WeightDmaOps:
    """Build weight DMA start/wait closures across all streaming schedules."""
    noop = lambda *a, **kw: None
    start_s3_part = wait_s3_part = start_s3_down = wait_s3_down = noop
    start_w1_slot = wait_w1_slot = start_w2_slot = wait_w2_slot = noop
    start_w2_1x = wait_w2_1x = noop
    start_w1_half = wait_w1_half = start_w2_half = wait_w2_half = noop

    if w_mode in ("stream3", "stream3_k2"):
        half_k_u32 = (_H // 2) // _FP4_PACK
        half_nqb = (_H // scale_block) // 2

        def _issue_proj(w_src, s_src, part_id: int, slot):
            if w_mode == "stream3_k2":
                k0 = part_id * half_k_u32
                sk0 = part_id * half_nqb
                pltpu.make_async_copy(
                    w_src.at[slot, pl.ds(k0, half_k_u32), :],
                    w1_buf.at[part_id],
                    w1_sem.at[part_id],
                ).start()
                pltpu.make_async_copy(
                    s_src.at[slot, pl.ds(sk0, half_nqb), :],
                    s1_buf.at[part_id],
                    w1_sem.at[part_id],
                ).start()
            else:
                col = part_id * _I
                pltpu.make_async_copy(
                    w_src.at[slot, :, pl.ds(col, _I)],
                    w1_buf.at[part_id],
                    w1_sem.at[part_id],
                ).start()
                pltpu.make_async_copy(
                    s_src.at[slot, :, pl.ds(col, _I)],
                    s1_buf.at[part_id],
                    w1_sem.at[part_id],
                ).start()

        def _issue_down(w_src, s_src, slot):
            pltpu.make_async_copy(w_src.at[slot], w2_buf, w2_sem.at[0]).start()
            pltpu.make_async_copy(s_src.at[slot], s2_buf, w2_sem.at[0]).start()

        def start_s3_part(part_id: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_proj(w1_u32, s1_ref, part_id, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_proj(rep_w1_u32, rep_s1_ref, part_id, slot)

            else:
                _issue_proj(w1_u32, s1_ref, part_id, slot)

        def start_s3_down(src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_down(w2_u32, s2_ref, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_down(rep_w2_u32, rep_s2_ref, slot)

            else:
                _issue_down(w2_u32, s2_ref, slot)

        def wait_s3_part(part_id: int):
            pltpu.make_async_copy(
                w1_buf.at[part_id], w1_buf.at[part_id], w1_sem.at[part_id]
            ).wait()
            pltpu.make_async_copy(
                s1_buf.at[part_id], s1_buf.at[part_id], w1_sem.at[part_id]
            ).wait()

        def wait_s3_down():
            pltpu.make_async_copy(w2_buf, w2_buf, w2_sem.at[0]).wait()
            pltpu.make_async_copy(s2_buf, s2_buf, w2_sem.at[0]).wait()

    elif w_mode in ("w1_2x_w2_2x", "w1_2x_w2_1x"):

        def _issue_w1_2x(w_src, s_src, b_idx, slot):
            pltpu.make_async_copy(
                w_src.at[slot], w1_buf.at[b_idx], w1_sem.at[b_idx]
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot], s1_buf.at[b_idx], w1_sem.at[b_idx]
            ).start()

        def _issue_w2_2x(w_src, s_src, b_idx, slot):
            pltpu.make_async_copy(
                w_src.at[slot], w2_buf.at[b_idx], w2_sem.at[b_idx]
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot], s2_buf.at[b_idx], w2_sem.at[b_idx]
            ).start()

        def _issue_w2_1x(w_src, s_src, slot):
            pltpu.make_async_copy(w_src.at[slot], w2_buf, w2_sem.at[0]).start()
            pltpu.make_async_copy(s_src.at[slot], s2_buf, w2_sem.at[0]).start()

        def start_w1_slot(b_idx, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w1_2x(w1_u32, s1_ref, b_idx, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w1_2x(rep_w1_u32, rep_s1_ref, b_idx, slot)

            else:
                _issue_w1_2x(w1_u32, s1_ref, b_idx, slot)

        def start_w2_slot(b_idx, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w2_2x(w2_u32, s2_ref, b_idx, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w2_2x(rep_w2_u32, rep_s2_ref, b_idx, slot)

            else:
                _issue_w2_2x(w2_u32, s2_ref, b_idx, slot)

        def start_w2_1x(src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w2_1x(w2_u32, s2_ref, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w2_1x(rep_w2_u32, rep_s2_ref, slot)

            else:
                _issue_w2_1x(w2_u32, s2_ref, slot)

        def wait_w1_slot(b_idx):
            pltpu.make_async_copy(
                w1_buf.at[b_idx], w1_buf.at[b_idx], w1_sem.at[b_idx]
            ).wait()
            pltpu.make_async_copy(
                s1_buf.at[b_idx], s1_buf.at[b_idx], w1_sem.at[b_idx]
            ).wait()

        def wait_w2_slot(b_idx):
            pltpu.make_async_copy(
                w2_buf.at[b_idx], w2_buf.at[b_idx], w2_sem.at[b_idx]
            ).wait()
            pltpu.make_async_copy(
                s2_buf.at[b_idx], s2_buf.at[b_idx], w2_sem.at[b_idx]
            ).wait()

        def wait_w2_1x():
            pltpu.make_async_copy(w2_buf, w2_buf, w2_sem.at[0]).wait()
            pltpu.make_async_copy(s2_buf, s2_buf, w2_sem.at[0]).wait()

    else:
        half_i = _I // 2
        half_h = _H // 2

        def _issue_w1_hn(w_src, s_src, buf_id: int, slot):
            g_col = buf_id * half_i
            u_col = _I + buf_id * half_i
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(g_col, half_i)],
                w1_buf.at[buf_id, :, 0:half_i],
                w1_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(u_col, half_i)],
                w1_buf.at[buf_id, :, half_i:_I],
                w1_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(g_col, half_i)],
                s1_buf.at[buf_id, :, 0:half_i],
                w1_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(u_col, half_i)],
                s1_buf.at[buf_id, :, half_i:_I],
                w1_sem.at[buf_id],
            ).start()

        def _issue_w2_hn(w_src, s_src, buf_id: int, slot):
            c0 = buf_id * half_h
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(c0, half_h)],
                w2_buf.at[buf_id],
                w2_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(c0, half_h)],
                s2_buf.at[buf_id],
                w2_sem.at[buf_id],
            ).start()

        def start_w1_half(buf_id: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w1_hn(w1_u32, s1_ref, buf_id, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w1_hn(rep_w1_u32, rep_s1_ref, buf_id, slot)

            else:
                _issue_w1_hn(w1_u32, s1_ref, buf_id, slot)

        def start_w2_half(buf_id: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w2_hn(w2_u32, s2_ref, buf_id, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w2_hn(rep_w2_u32, rep_s2_ref, buf_id, slot)

            else:
                _issue_w2_hn(w2_u32, s2_ref, buf_id, slot)

        def wait_w1_half(buf_id: int):
            pltpu.make_async_copy(
                w1_buf.at[buf_id], w1_buf.at[buf_id], w1_sem.at[buf_id]
            ).wait()
            pltpu.make_async_copy(
                s1_buf.at[buf_id], s1_buf.at[buf_id], w1_sem.at[buf_id]
            ).wait()

        def wait_w2_half(buf_id: int):
            pltpu.make_async_copy(
                w2_buf.at[buf_id], w2_buf.at[buf_id], w2_sem.at[buf_id]
            ).wait()
            pltpu.make_async_copy(
                s2_buf.at[buf_id], s2_buf.at[buf_id], w2_sem.at[buf_id]
            ).wait()

    def issue_spec_prime(safe_spec_slot, spec_parts: int):
        if w_mode in ("stream3", "stream3_k2"):
            _issue_proj(w1_u32, s1_ref, 0, safe_spec_slot)
            if spec_parts >= 2:
                _issue_proj(w1_u32, s1_ref, 1, safe_spec_slot)
            if spec_parts >= 3:
                _issue_down(w2_u32, s2_ref, safe_spec_slot)
        elif w_mode == "w1_2x_w2_1x":
            _issue_w1_2x(w1_u32, s1_ref, 0, safe_spec_slot)
            if spec_parts >= 3:
                _issue_w2_1x(w2_u32, s2_ref, safe_spec_slot)
        elif w_mode == "w1_2x_w2_2x":
            _issue_w1_2x(w1_u32, s1_ref, 0, safe_spec_slot)
            if spec_parts >= 3:
                _issue_w2_2x(w2_u32, s2_ref, 0, safe_spec_slot)
        else:
            _issue_w1_hn(w1_u32, s1_ref, 0, safe_spec_slot)
            if spec_parts >= 2:
                _issue_w1_hn(w1_u32, s1_ref, 1, safe_spec_slot)
            if spec_parts >= 3:
                _issue_w2_hn(w2_u32, s2_ref, 0, safe_spec_slot)
                _issue_w2_hn(w2_u32, s2_ref, 1, safe_spec_slot)

    def prime_first_exec(
        num_exec,
        exec_src_smem,
        exec_slot_smem,
        *,
        do_spec: bool,
        spec_parts: int,
        spec_slot,
        spec_enabled,
    ):
        if do_spec and w_mode in ("stream3", "stream3_k2") and spec_parts == 1:
            spec_hit = (
                spec_enabled
                & (num_exec > 0)
                & (exec_src_smem[0] == 0)
                & (exec_slot_smem[0] == spec_slot)
            )

            @pl.when(num_exec > 0)
            def _prime_s3_p1_down():
                s0 = exec_src_smem[0]
                sl0 = exec_slot_smem[0]
                start_s3_part(1, s0, sl0)
                start_s3_down(s0, sl0)

            @pl.when(spec_enabled & (~spec_hit))
            def _drain_s3_p0_miss():
                wait_s3_part(0)

            @pl.when((~spec_hit) & (num_exec > 0))
            def _prime_s3_p0_after_miss():
                start_s3_part(0, exec_src_smem[0], exec_slot_smem[0])

        elif do_spec:
            spec_hit = (
                spec_enabled
                & (num_exec > 0)
                & (exec_src_smem[0] == 0)
                & (exec_slot_smem[0] == spec_slot)
            )

            @pl.when(spec_enabled & (~spec_hit))
            def _drain_spec_miss():
                if w_mode in ("stream3", "stream3_k2"):
                    wait_s3_part(0)
                    if spec_parts >= 2:
                        wait_s3_part(1)
                    if spec_parts >= 3:
                        wait_s3_down()
                elif w_mode == "w1_2x_w2_1x":
                    wait_w1_slot(0)
                    if spec_parts >= 3:
                        wait_w2_1x()
                elif w_mode == "w1_2x_w2_2x":
                    wait_w1_slot(0)
                    if spec_parts >= 3:
                        wait_w2_slot(0)
                else:
                    wait_w1_half(0)
                    if spec_parts >= 2:
                        wait_w1_half(1)
                    if spec_parts >= 3:
                        wait_w2_half(0)
                        wait_w2_half(1)

            if spec_parts < 3 or w_mode in ("w1_2x_w2_1x", "w1_2x_w2_2x"):

                @pl.when(spec_hit)
                def _prime_spec_rest():
                    s0 = exec_src_smem[0]
                    sl0 = exec_slot_smem[0]
                    if w_mode in ("stream3", "stream3_k2"):
                        if spec_parts < 2:
                            start_s3_part(1, s0, sl0)
                        start_s3_down(s0, sl0)
                    elif w_mode == "w1_2x_w2_1x":
                        if spec_parts < 3:
                            start_w2_1x(s0, sl0)

                        @pl.when(num_exec >= 2)
                        def _prime_hit_e1():
                            start_w1_slot(1, exec_src_smem[1], exec_slot_smem[1])

                    elif w_mode == "w1_2x_w2_2x":
                        if spec_parts < 3:
                            start_w2_slot(0, s0, sl0)

                        @pl.when(num_exec >= 2)
                        def _prime_hit_e1_2x():
                            start_w1_slot(1, exec_src_smem[1], exec_slot_smem[1])
                            start_w2_slot(1, exec_src_smem[1], exec_slot_smem[1])

                    else:
                        if spec_parts < 2:
                            start_w1_half(1, s0, sl0)
                        start_w2_half(0, s0, sl0)
                        start_w2_half(1, s0, sl0)

            @pl.when((~spec_hit) & (num_exec > 0))
            def _prime_after_miss():
                s0 = exec_src_smem[0]
                sl0 = exec_slot_smem[0]
                if w_mode in ("stream3", "stream3_k2"):
                    start_s3_part(0, s0, sl0)
                    start_s3_part(1, s0, sl0)
                    start_s3_down(s0, sl0)
                elif w_mode == "w1_2x_w2_1x":
                    start_w1_slot(0, s0, sl0)
                    start_w2_1x(s0, sl0)

                    @pl.when(num_exec >= 2)
                    def _prime_miss_e1():
                        start_w1_slot(1, exec_src_smem[1], exec_slot_smem[1])

                elif w_mode == "w1_2x_w2_2x":
                    start_w1_slot(0, s0, sl0)
                    start_w2_slot(0, s0, sl0)

                    @pl.when(num_exec >= 2)
                    def _prime_miss_e1_2x():
                        start_w1_slot(1, exec_src_smem[1], exec_slot_smem[1])
                        start_w2_slot(1, exec_src_smem[1], exec_slot_smem[1])

                else:
                    start_w1_half(0, s0, sl0)
                    start_w1_half(1, s0, sl0)
                    start_w2_half(0, s0, sl0)
                    start_w2_half(1, s0, sl0)

        else:

            @pl.when(num_exec > 0)
            def _prime_ring():
                s0 = exec_src_smem[0]
                sl0 = exec_slot_smem[0]
                if w_mode in ("stream3", "stream3_k2"):
                    start_s3_part(0, s0, sl0)
                    start_s3_part(1, s0, sl0)
                    start_s3_down(s0, sl0)
                elif w_mode == "w1_2x_w2_2x":
                    start_w1_slot(0, s0, sl0)
                    start_w2_slot(0, s0, sl0)

                    @pl.when(num_exec >= 2)
                    def _prime_e1_2x():
                        s1_k = exec_src_smem[1]
                        sl1_k = exec_slot_smem[1]
                        start_w1_slot(1, s1_k, sl1_k)
                        start_w2_slot(1, s1_k, sl1_k)

                elif w_mode == "w1_2x_w2_1x":
                    start_w1_slot(0, s0, sl0)
                    start_w2_1x(s0, sl0)

                    @pl.when(num_exec >= 2)
                    def _prime_e1():
                        start_w1_slot(1, exec_src_smem[1], exec_slot_smem[1])

                else:
                    start_w1_half(0, s0, sl0)
                    start_w1_half(1, s0, sl0)
                    start_w2_half(0, s0, sl0)
                    start_w2_half(1, s0, sl0)

    return _WeightDmaOps(
        w_mode=w_mode,
        issue_spec_prime=issue_spec_prime,
        prime_first_exec=prime_first_exec,
        start_s3_part=start_s3_part,
        wait_s3_part=wait_s3_part,
        start_s3_down=start_s3_down,
        wait_s3_down=wait_s3_down,
        start_w1_slot=start_w1_slot,
        wait_w1_slot=wait_w1_slot,
        start_w2_slot=start_w2_slot,
        wait_w2_slot=wait_w2_slot,
        start_w2_1x=start_w2_1x,
        wait_w2_1x=wait_w2_1x,
        start_w1_half=start_w1_half,
        wait_w1_half=wait_w1_half,
        start_w2_half=start_w2_half,
        wait_w2_half=wait_w2_half,
    )


def _reshape_x_to_3d(
    x_in_ref,
    x_3d_buf,
    batch_size: int,
    cfg: SparseDecodeConfig,
) -> None:
    """Reshape ``x_in_ref [B, 4096]`` into sublane-tiled ``x_3d_buf [B, 32, 128]``."""
    if (cfg.fast_b64_epi_pro and batch_size <= 64) or (
        cfg.fast_gs and batch_size <= 128
    ):
        x_chunk = min(batch_size, 32)
        for r0 in range(0, batch_size, x_chunk):
            for h_idx in range(2):
                x_3d_buf[
                    pl.ds(r0, x_chunk), pl.ds(h_idx * 16, 16), :
                ] = x_in_ref[
                    pl.ds(r0, x_chunk), pl.ds(h_idx * 2048, 2048)
                ].reshape(x_chunk, 16, 128)
        return

    x_rc = 16

    def _reshape_x_chunk(ci, _):
        r0 = pl.multiple_of(ci * x_rc, x_rc)
        for h_idx in range(2):
            x_3d_buf[
                pl.ds(r0, x_rc), pl.ds(h_idx * 16, 16), :
            ] = x_in_ref[
                pl.ds(r0, x_rc), pl.ds(h_idx * 2048, 2048)
            ].reshape(x_rc, 16, 128)
        return _

    lax.fori_loop(0, batch_size // x_rc, _reshape_x_chunk, jnp.int32(0))


def zero_and_pack_muse_sparse_from_vmem(
    plan_res: MuseSparsePlanResult,
    *,
    r_scratch,
    accum_3d,
    io_3d_buf,
    accum_2d=None,
    x_in_ref=None,
    x_3d_buf=None,
    batch_size: int,
    cfg: SparseDecodeConfig,
) -> None:
    """Zero MoE accumulators/staging buffers, optionally reshape x->3D, and pack SMEM buckets."""
    tile_m = cfg.tile_m
    if plan_res.is_dense:
        for s_col in range(0, _H, 1024):
            accum_2d[:, pl.ds(s_col, 1024)] = jnp.zeros(
                (batch_size, 1024), dtype=jnp.float32
            )
        return

    if (cfg.fast_b64_epi_pro and batch_size <= 64) or (
        cfg.fast_gs and batch_size <= 128
    ):
        z_chunk = min(batch_size, 32)
        z_acc = jnp.zeros((z_chunk, 32, 128), dtype=jnp.float32)
        for r0 in range(0, batch_size, z_chunk):
            accum_3d[pl.ds(r0, z_chunk), :, :] = z_acc
        if cfg.fast_gs:
            accum_3d[pl.ds(batch_size, 1), :, :] = jnp.zeros(
                (1, 32, 128), dtype=jnp.float32
            )
        z_io_chunk = min(tile_m, 32)
        z_io = jnp.zeros((z_io_chunk, 32, 128), dtype=jnp.bfloat16)
        for r0 in range(0, tile_m, z_io_chunk):
            io_3d_buf[pl.ds(r0, z_io_chunk), :, :] = z_io
    else:
        zero_rc = min(batch_size, 8)
        zeros_acc = jnp.zeros((zero_rc, 32, 128), dtype=jnp.float32)

        def _zero_acc_chunk(ci, _):
            accum_3d[pl.ds(ci * zero_rc, zero_rc), :, :] = zeros_acc
            return _

        lax.fori_loop(0, batch_size // zero_rc, _zero_acc_chunk, jnp.int32(0))
        if cfg.fast_gs:
            accum_3d[pl.ds(batch_size, 1), :, :] = jnp.zeros(
                (1, 32, 128), dtype=jnp.float32
            )

        zeros_io = jnp.zeros((8, 32, 128), dtype=jnp.bfloat16)

        def _zero_io_chunk(ci, _):
            io_3d_buf[pl.ds(ci * 8, 8), :, :] = zeros_io
            return _

        lax.fori_loop(0, tile_m // 8, _zero_io_chunk, jnp.int32(0))

    # Emit VPU x_in->x_3d_buf relayout before pack_active_tokens_smem's scf.for
    # so VPU/XLU relayout overlaps concurrently with SALU packing and Expert-0 DMA.
    if x_in_ref is not None and x_3d_buf is not None:
        _reshape_x_to_3d(x_in_ref, x_3d_buf, batch_size, cfg)

    pack_active_tokens_smem(
        batch_size,
        r_scratch,
        has_replicas=plan_res.has_replicas,
        fast_gs=cfg.fast_gs,
    )


def plan_and_prime_muse_sparse_from_vmem(
    me,
    logits_ref,
    bias_ref,
    norm_ref,
    w1_u32,
    s1_ref,
    w2_u32,
    s2_ref,
    *,
    d_scratch,
    r_scratch,
    w1_buf,
    s1_buf,
    w2_buf,
    s2_buf,
    w1_sem,
    w2_sem,
    misc_sem,
    gain_f32_buf,
    accum_3d,
    io_3d_buf,
    accum_2d=None,
    rep_w1_u32=None,
    rep_s1_ref=None,
    rep_w2_u32=None,
    rep_s2_ref=None,
    replica_meta_ref=None,
    owner_load_smem=None,
    moved_flag_smem=None,
    local_suppress_smem=None,
    holder_slot_expert_smem=None,
    holder_cand_smem=None,
    spec_slot_ref=None,
    ep: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    replica_cost_mode: str,
    cfg: SparseDecodeConfig,
    stop_stage: str = "",
    early_exit_fn=None,
    zero_and_pack: bool = True,
    x_in_ref=None,
    x_3d_buf=None,
) -> MuseSparsePlanResult | None:
    """Phase 1 of VMEM-resident Muse sparse decode: route, plan, prime Expert-0, & optional zero/pack.

    Everything here depends only on ``logits_ref`` (not ``x_in_ref``), so fused
    ``[chip_b1 + MoE]`` superkernels can invoke this inside ``ag_exp``'s ICI wait
    and call ``zero_and_pack_muse_sparse_from_vmem`` inside its D2D2 wait.
    """
    has_replicas = num_replicas > 0
    is_k256 = scale_block >= 256
    is_dense = cfg.row_mode == "dense"
    w_mode = (
        "stream3"
        if (cfg.weight_mode == "stream3_k2" and (_H // scale_block) < 16)
        else cfg.weight_mode
    )
    use_bd = (not is_dense) and (not is_k256) and bool(cfg.k16_blockdiag_rows)
    use_lhs_fp8 = is_k256 and cfg.lhs_fp8
    use_pair_rs = cfg.pair_rs and (ep in (8, 16))
    tile_m = cfg.tile_m
    e_local = _E_TOTAL // ep

    dma_ops = _make_sparse_weight_dma_ops(
        w1_u32,
        s1_ref,
        w2_u32,
        s2_ref,
        rep_w1_u32,
        rep_s1_ref,
        rep_w2_u32,
        rep_s2_ref,
        w1_buf,
        s1_buf,
        w2_buf,
        s2_buf,
        w1_sem,
        w2_sem,
        w_mode=w_mode,
        scale_block=scale_block,
        has_replicas=has_replicas,
    )

    spec_parts = cfg.spec_prime_parts
    do_spec = (
        (spec_parts > 0)
        and (w_mode in ("stream3", "stream3_k2", "half_n", "w1_2x_w2_1x", "w1_2x_w2_2x"))
        and (stop_stage not in ("route", "plan", "pack"))
    )
    if do_spec:
        if cfg.spec_home_slots:
            pref_slot = sum(
                jnp.where(
                    me == r, jnp.int32(cfg.spec_home_slots[r]), jnp.int32(0)
                )
                for r in range(ep)
            )
            if any(s < 0 for s in cfg.spec_home_slots):
                pref_slot = sum(
                    jnp.where(
                        me == r,
                        jnp.int32(cfg.spec_home_slots[r] + 1),
                        jnp.int32(0),
                    )
                    for r in range(ep)
                ) - jnp.int32(1)
        elif spec_slot_ref is not None:
            pref_slot = spec_slot_ref[0]
        else:
            pref_slot = jnp.int32(0)
        if has_replicas and cfg.spec_skip_replicated:
            non_rep = select_non_replicated_home_slot(
                me,
                e_local,
                replica_meta_ref,
                preferred_slot=jnp.maximum(pref_slot, jnp.int32(0)),
                num_replicas=num_replicas,
            )
            spec_slot = jnp.where(pref_slot >= 0, non_rep, jnp.int32(-1))
        else:
            spec_slot = pref_slot
        spec_enabled = (spec_slot >= 0) & (spec_slot < e_local)
        safe_spec_slot = jnp.where(spec_enabled, spec_slot, jnp.int32(0))

        @pl.when(spec_enabled)
        def _issue_spec_prime():
            dma_ops.issue_spec_prime(safe_spec_slot, spec_parts)

    else:
        spec_slot = None
        spec_enabled = jnp.bool_(False)

    gain_f32_buf[...] = norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0

    # 2. In-kernel tie-exact routing.
    if is_dense:
        route_dense_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            d_scratch,
            misc_sem.at[0],
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
            min_p_topk=cfg.min_p_topk,
            free_max_rows=cfg.free_max_rows,
            free_max_w=cfg.free_max_w,
            free_min_slot=cfg.free_min_slot,
        )
        counts_smem = d_scratch.counts_smem
    else:
        if has_replicas:
            if cfg.static_holder_mask:
                cand_eids = extract_my_holder_candidates_tuple(
                    me, num_replicas, replica_meta_ref
                )
            else:
                extract_my_holder_candidates_smem(
                    me, num_replicas, replica_meta_ref, holder_cand_smem
                )
                cand_eids = tuple(
                    holder_cand_smem[s] for s in range(num_replicas)
                )
        else:
            cand_eids = ()
        route_topk_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            r_scratch,
            misc_sem.at[0],
            my_owner=me,
            e_per_owner=e_local,
            holder_cand_eids=cand_eids,
            hoist_holder_vecs=cfg.static_holder_mask,
            precompute_owner_loads=(has_replicas and cfg.static_holder_mask),
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=tile_m,
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
            min_p_topk=cfg.min_p_topk,
            free_max_rows=cfg.free_max_rows,
            free_max_w=cfg.free_max_w,
            free_min_slot=cfg.free_min_slot,
            route_rb16_mode=getattr(cfg, "route_rb16_mode", None),
            route_fast=getattr(cfg, "route_fast", None),
        )
        counts_smem = r_scratch.counts_smem

    if stop_stage == "route":
        if early_exit_fn is not None:
            early_exit_fn()
        return None

    # 3. Fast SMEM execution plan (home + optional §4.5b replicas).
    if has_replicas:
        run_replica_planner_smem(
            me,
            e_local,
            ep,
            num_replicas,
            counts_smem,
            replica_meta_ref,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=replica_cost_mode,
            tile_m=tile_m,
            moved_flag_smem=moved_flag_smem,
            vpu_precomputed_loads=(not is_dense and cfg.static_holder_mask),
            fast_plan=getattr(cfg, "fast_plan", False),
        )
        if is_dense:
            num_exec = build_dense_exec_plan_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                d_scratch,
                spec_slot=spec_slot,
            )
        else:
            num_exec = build_exec_plan_from_masks_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                r_scratch,
                spec_slot=spec_slot,
                fast_plan=getattr(cfg, "fast_plan", False),
            )
    else:
        if is_dense:
            num_exec = build_dense_exec_plan_smem(
                me, e_local, 0, None, None, d_scratch, spec_slot=spec_slot
            )
        else:
            num_exec = build_default_exec_plan_smem(
                me,
                e_local,
                r_scratch,
                spec_slot=spec_slot,
                fast_plan=bool(getattr(cfg, "fast_plan", False)),
            )

    if stop_stage == "plan":
        if early_exit_fn is not None:
            early_exit_fn()
        return None

    exec_eid_smem = (
        d_scratch.exec_expert_id if is_dense else r_scratch.exec_expert_id
    )
    exec_src_smem = (
        d_scratch.exec_weight_src if is_dense else r_scratch.exec_weight_src
    )
    exec_slot_smem = (
        d_scratch.exec_weight_slot if is_dense else r_scratch.exec_weight_slot
    )

    # 4. IMMEDIATELY prime weight prefetch ring BEFORE prologue zeroing/packing!
    if stop_stage != "pack":
        dma_ops.prime_first_exec(
            num_exec,
            exec_src_smem,
            exec_slot_smem,
            do_spec=do_spec,
            spec_parts=spec_parts,
            spec_slot=spec_slot,
            spec_enabled=spec_enabled,
        )

    plan_res = MuseSparsePlanResult(
        num_exec=num_exec,
        spec_slot=spec_slot,
        spec_enabled=spec_enabled,
        is_dense=is_dense,
        is_k256=is_k256,
        w_mode=w_mode,
        use_bd=use_bd,
        has_replicas=has_replicas,
        use_lhs_fp8=use_lhs_fp8,
        use_pair_rs=use_pair_rs,
        exec_eid_smem=exec_eid_smem,
        exec_src_smem=exec_src_smem,
        exec_slot_smem=exec_slot_smem,
        dma_ops=dma_ops,
    )
    if zero_and_pack:
        zero_and_pack_muse_sparse_from_vmem(
            plan_res,
            r_scratch=r_scratch,
            accum_3d=accum_3d,
            io_3d_buf=io_3d_buf,
            accum_2d=accum_2d,
            x_in_ref=x_in_ref,
            x_3d_buf=x_3d_buf,
            batch_size=batch_size,
            cfg=cfg,
        )
    if stop_stage == "pack":
        if early_exit_fn is not None:
            early_exit_fn()
        return None
    return plan_res


def run_muse_sparse_experts_and_rs_from_vmem(
    me,
    x_in_ref,
    plan_res: MuseSparsePlanResult,
    *,
    d_scratch,
    r_scratch,
    w1_buf,
    s1_buf,
    w2_buf,
    s2_buf,
    w1_acc_buf,
    x_3d_buf,
    io_3d_buf,
    lhs_2d_buf,
    act_2d_buf,
    y_stage_buf,
    lhs_fp8_buf,
    lhs_sc_buf,
    masked_x_buf,
    bd_sc_buf,
    gain_f32_buf,
    accum_3d,
    accum_2d,
    fan_3d_buf,
    rs_send_bf16,
    rs_pair_recv,
    rs_chip_send,
    rs_chip_recv,
    rs_pair_ssem,
    rs_pair_rsem,
    rs_chip_ssem,
    rs_chip_rsem,
    rs_recv_3d,
    rs_send_sem,
    rs_recv_sem,
    owner_3d,
    fan_send_sem,
    fan_recv_sem,
    fan_s2_send_sem,
    fan_s2_recv_sem,
    barrier_sem,
    output_ref,
    ep: int,
    batch_size: int,
    scale_block: int,
    return_sharded: bool,
    ep_axis_name: str,
    cfg: SparseDecodeConfig,
    x_is_3d: bool = False,
    wait_entry_barrier: bool = True,
    write_output_ref: bool = True,
    on_experts_done_fn=None,
    stop_stage: str = "",
    early_exit_fn=None,
) -> None:
    """Phase 2 of VMEM-resident Muse sparse decode: x reshape, expert loop, RS, & fan-out.

    When called inside a ``[chip_b1 + MoE + chip_b2]`` superkernel:
      - Pass ``x_is_3d=True`` if ``x_3d_buf`` already holds ``[B, 32, 128] bf16``.
      - Pass ``on_experts_done_fn`` to immediately launch ``chip_b2`` weight DMAs
        into freed ``w1_buf`` / ``w2_buf`` overlapped with ``PAIR_RS``.
      - Pass ``write_output_ref=False`` to leave the BF16 MoE output in VMEM:
        inside ``owner_3d`` when ``return_sharded=True`` or inside ``fan_3d_buf``
        (``[B, 32, 128]``) when ``return_sharded=False``.
    """
    is_dense = plan_res.is_dense
    is_k256 = plan_res.is_k256
    w_mode = plan_res.w_mode
    use_bd = plan_res.use_bd
    use_lhs_fp8 = plan_res.use_lhs_fp8
    use_pair_rs = plan_res.use_pair_rs
    num_exec = plan_res.num_exec
    exec_eid_smem = plan_res.exec_eid_smem
    exec_src_smem = plan_res.exec_src_smem
    exec_slot_smem = plan_res.exec_slot_smem
    tile_m = cfg.tile_m
    rows_per_rank = batch_size // ep

    start_s3_part = plan_res.dma_ops.start_s3_part
    wait_s3_part = plan_res.dma_ops.wait_s3_part
    start_s3_down = plan_res.dma_ops.start_s3_down
    wait_s3_down = plan_res.dma_ops.wait_s3_down
    start_w1_slot = plan_res.dma_ops.start_w1_slot
    wait_w1_slot = plan_res.dma_ops.wait_w1_slot
    start_w2_slot = plan_res.dma_ops.start_w2_slot
    wait_w2_slot = plan_res.dma_ops.wait_w2_slot
    start_w2_1x = plan_res.dma_ops.start_w2_1x
    wait_w2_1x = plan_res.dma_ops.wait_w2_1x
    start_w1_half = plan_res.dma_ops.start_w1_half
    wait_w1_half = plan_res.dma_ops.wait_w1_half
    start_w2_half = plan_res.dma_ops.start_w2_half
    wait_w2_half = plan_res.dma_ops.wait_w2_half

    if is_dense:
        if use_lhs_fp8:
            _prequant_lhs_k256(x_in_ref, lhs_fp8_buf, lhs_sc_buf, batch_size)
    elif not x_is_3d:
        _reshape_x_to_3d(x_in_ref, x_3d_buf, batch_size, cfg)


    # --- Compute strip helpers ---
    @jax.named_scope("gmm1_proj")
    def _run_proj_only_slice(
        x_src_ref,
        w_ref,
        s_ref,
        width: int,
        m_rows: int,
        is_up: bool,
        act_buf=None,
    ):
        """Used by stream3: run full gate (is_up=False) or full up (is_up=True)."""
        act_dst = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, width)
        )
        write_m = 16 if m_rows == 8 else m_rows
        lhs_eff = lhs_fp8_buf if use_lhs_fp8 else x_src_ref
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(n0, nc)
            res = _fp4_gmm_chunked(
                lhs_eff,
                w_ref,
                s_ref,
                m_rows,
                _H,
                n0,
                nc,
                is_k256,
                scale_block,
                k16_dequant_mode=cfg.k16_dequant_mode,
                lhs_sc_ref=sc_eff,
                hoist_scales=cfg.hoist_k256_scales,
                nc_split512=cfg.nc_split512,
                k512_chain=cfg.k512_chain,
            )
            if not is_up:
                sg_bf = jax.nn.silu(res).astype(jnp.bfloat16)
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    sg_bf, write_m
                )
            else:
                sg_read = act_dst[pl.ds(0, write_m), col_out]
                if m_rows == 8:
                    sg_read = sg_read[:8, :]
                h = (sg_read.astype(jnp.float32) * res).astype(jnp.bfloat16)
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    h, write_m
                )
            return carry

        n_strips = width // nc
        max_unr = 4 if cfg.unroll_strips else 2
        if n_strips <= max_unr:
            for jj in range(n_strips):
                _strip(jj, 0)
        else:
            lax.fori_loop(0, n_strips, _strip, 0)

    @jax.named_scope("gmm1_k2")
    def _run_k2_proj_half(
        x_src_ref,
        w_ref,
        s_ref,
        part_id: int,
        m_rows: int,
        act_buf=None,
        acc_buf=None,
    ):
        """Used by stream3_k2: part 0 accumulates K[0:2048]; part 1 finishes K[2048:4096] + SiLU."""
        act_dst = act_2d_buf if act_buf is None else act_buf
        acc_dst = w1_acc_buf if acc_buf is None else acc_buf
        nc = (
            _pick_k256_nc(m_rows, _I, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, _I)
        )
        write_m = 16 if m_rows == 8 else m_rows
        lhs_eff = lhs_fp8_buf if use_lhs_fp8 else x_src_ref
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None
        half_k = _H // 2
        if part_id == 0:
            def _strip_k0(j, carry):
                n0 = pl.multiple_of(j * nc, nc)
                p0 = _fp4_gmm_chunked(
                    lhs_eff,
                    w_ref,
                    s_ref,
                    m_rows,
                    half_k,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                    lhs_sc_ref=sc_eff,
                    hoist_scales=cfg.hoist_k256_scales,
                    k_offset=0,
                    nc_split512=cfg.nc_split512,
                    k512_chain=cfg.k512_chain,
                )
                acc_dst[pl.ds(0, m_rows), pl.ds(n0, nc)] = p0
                return carry

            n_s0 = _N1 // nc
            if n_s0 <= 4:
                for jj in range(n_s0):
                    _strip_k0(jj, 0)
            else:
                lax.fori_loop(0, n_s0, _strip_k0, 0)
            return

        def _strip_k1(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            u_n0 = _I + n0
            g = _fp4_gmm_chunked(
                lhs_eff,
                w_ref,
                s_ref,
                m_rows,
                half_k,
                n0,
                nc,
                is_k256,
                scale_block,
                k16_dequant_mode=cfg.k16_dequant_mode,
                lhs_sc_ref=sc_eff,
                hoist_scales=cfg.hoist_k256_scales,
                k_offset=half_k,
                init_acc=acc_dst[pl.ds(0, m_rows), pl.ds(n0, nc)],
                nc_split512=cfg.nc_split512,
                k512_chain=cfg.k512_chain,
            )
            u = _fp4_gmm_chunked(
                lhs_eff,
                w_ref,
                s_ref,
                m_rows,
                half_k,
                u_n0,
                nc,
                is_k256,
                scale_block,
                k16_dequant_mode=cfg.k16_dequant_mode,
                lhs_sc_ref=sc_eff,
                hoist_scales=cfg.hoist_k256_scales,
                k_offset=half_k,
                init_acc=acc_dst[pl.ds(0, m_rows), pl.ds(u_n0, nc)],
                nc_split512=cfg.nc_split512,
                k512_chain=cfg.k512_chain,
            )
            h = (
                jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
            ).astype(jnp.bfloat16)
            act_dst[pl.ds(0, write_m), pl.ds(n0, nc)] = _pad_rows_to(h, write_m)
            return carry

        n_s1 = _I // nc
        max_unr1 = 4 if cfg.unroll_strips else 2
        if n_s1 <= max_unr1:
            for jj in range(n_s1):
                _strip_k1(jj, 0)
        else:
            lax.fori_loop(0, n_s1, _strip_k1, 0)

    @jax.named_scope("gmm1")
    def _run_gmm1_slice(
        x_src_ref,
        w_ref,
        s_ref,
        col_offset: int,
        width: int,
        m_rows: int,
        act_buf=None,
    ):
        act_dst = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, width)
        )
        dual_fits_vregs = (m_rows * nc) // 1024 <= 16
        use_dual = (
            is_k256
            and (not use_lhs_fp8)
            and dual_fits_vregs
            and (
                cfg.k256_dual_mode == "dual"
                or (cfg.k256_dual_mode == "hybrid" and m_rows <= 16)
            )
        )
        write_m = 16 if m_rows == 8 else m_rows
        up_base = width
        lhs_eff = lhs_fp8_buf if use_lhs_fp8 else x_src_ref
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            if use_dual:
                g, u = _fp4_dual_gmm_chunked(
                    x_src_ref,
                    w_ref,
                    s_ref,
                    m_rows,
                    _H,
                    n0,
                    up_base + n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                    nc_split512=cfg.nc_split512,
                    k512_chain=cfg.k512_chain,
                )
                h = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
            else:
                g = _fp4_gmm_chunked(
                    lhs_eff,
                    w_ref,
                    s_ref,
                    m_rows,
                    _H,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                    lhs_sc_ref=sc_eff,
                    hoist_scales=cfg.hoist_k256_scales,
                    nc_split512=cfg.nc_split512,
                    k512_chain=cfg.k512_chain,
                )
                sg_bf = jax.nn.silu(g).astype(jnp.bfloat16)
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    sg_bf, write_m
                )
                u = _fp4_gmm_chunked(
                    lhs_eff,
                    w_ref,
                    s_ref,
                    m_rows,
                    _H,
                    up_base + n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                    lhs_sc_ref=sc_eff,
                    hoist_scales=cfg.hoist_k256_scales,
                    nc_split512=cfg.nc_split512,
                    k512_chain=cfg.k512_chain,
                )
                sg_read = act_dst[pl.ds(0, write_m), col_out]
                if m_rows == 8:
                    sg_read = sg_read[:8, :]
                h = (sg_read.astype(jnp.float32) * u).astype(jnp.bfloat16)
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
            return carry

        n_strips = width // nc
        max_unr = 4 if cfg.unroll_strips else 2
        if n_strips <= max_unr:
            for jj in range(n_strips):
                _strip(jj, 0)
        else:
            lax.fori_loop(0, n_strips, _strip, 0)

    @jax.named_scope("gmm2")
    def _run_gmm2_slice(
        w_ref,
        s_ref,
        col_offset: int,
        width: int,
        m_rows: int,
        sq_sum,
        act_buf=None,
    ):
        act_src = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, width)
        )
        write_m = m_rows if cfg.norm_f32_stage else (16 if m_rows == 8 else m_rows)
        act_eff = lhs_fp8_buf if use_lhs_fp8 else act_src
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None
        if use_lhs_fp8 and col_offset == 0:
            _prequant_lhs_k256(act_src, lhs_fp8_buf, lhs_sc_buf, m_rows)

        def _strip(j, sq_acc):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            d = _fp4_gmm_chunked(
                act_eff,
                w_ref,
                s_ref,
                m_rows,
                _I,
                n0,
                nc,
                is_k256,
                scale_block,
                k16_dequant_mode=cfg.k16_dequant_mode,
                lhs_sc_ref=sc_eff,
                hoist_scales=cfg.hoist_k256_scales,
                nc_split512=cfg.nc_split512,
                k512_chain=cfg.k512_chain,
            ).astype(jnp.bfloat16)
            g_col = gain_f32_buf[0:1, col_out]
            if cfg.norm_f32_stage:
                u = d.astype(jnp.float32) * g_col
                sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
                y_stage_buf[pl.ds(0, write_m), col_out] = u
            else:
                u_bf = d * g_col.astype(jnp.bfloat16)
                u_f32 = u_bf.astype(jnp.float32)
                sq_acc = sq_acc + jnp.sum(u_f32 * u_f32, axis=1, keepdims=True)
                y_stage_buf[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    u_bf, write_m
                )
            return sq_acc

        n_strips = width // nc
        max_unr = 4 if cfg.unroll_strips else 2
        if n_strips <= max_unr:
            for jj in range(n_strips):
                sq_sum = _strip(jj, sq_sum)
            return sq_sum
        return lax.fori_loop(0, n_strips, _strip, sq_sum)

    # =========================================================================
    # 6A. DENSE-ROWS EXECUTION LOOP (for K256 B <= 64 — zero pack/gather/switch!)
    # =========================================================================
    if is_dense:
        lane_128 = lax.broadcasted_iota(jnp.int32, (batch_size, 128), 1)

        def _dense_expert_body(k, _):
            g_e = exec_eid_smem[k]
            has_next = (k + 1) < num_exec
            next_k = jnp.where(has_next, k + 1, jnp.int32(0))
            next_src = exec_src_smem[next_k]
            next_slot = exec_slot_smem[next_k]
            has_k2 = (k + 2) < num_exec
            k2_clamped = jnp.where(has_k2, k + 2, jnp.int32(0))

            sq_sum = jnp.zeros((batch_size, 1), dtype=jnp.float32)
            if w_mode in ("stream3", "stream3_k2"):
                wait_s3_part(0)
                if w_mode == "stream3_k2":
                    _run_k2_proj_half(
                        x_in_ref, w1_buf.at[0], s1_buf.at[0], 0, batch_size
                    )
                else:
                    _run_proj_only_slice(
                        x_in_ref, w1_buf.at[0], s1_buf.at[0], _I, batch_size, False
                    )

                @pl.when(has_next)
                def _pf_g():
                    start_s3_part(0, next_src, next_slot)

                wait_s3_part(1)
                if w_mode == "stream3_k2":
                    _run_k2_proj_half(
                        x_in_ref, w1_buf.at[1], s1_buf.at[1], 1, batch_size
                    )
                else:
                    _run_proj_only_slice(
                        x_in_ref, w1_buf.at[1], s1_buf.at[1], _I, batch_size, True
                    )

                @pl.when(has_next)
                def _pf_u():
                    start_s3_part(1, next_src, next_slot)

                wait_s3_down()
                sq_sum = _run_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, batch_size, sq_sum
                )

                @pl.when(has_next)
                def _pf_d():
                    start_s3_down(next_src, next_slot)

            elif w_mode == "w1_2x_w2_2x":
                b_idx = k % 2
                wait_w1_slot(b_idx)
                _run_gmm1_slice(
                    x_in_ref, w1_buf.at[b_idx], s1_buf.at[b_idx], 0, _I, batch_size
                )

                @pl.when(has_k2)
                def _pf_w1_2x():
                    start_w1_slot(
                        b_idx,
                        exec_src_smem[k2_clamped],
                        exec_slot_smem[k2_clamped],
                    )

                wait_w2_slot(b_idx)
                sq_sum = _run_gmm2_slice(
                    w2_buf.at[b_idx], s2_buf.at[b_idx], 0, _H, batch_size, sq_sum
                )

                @pl.when(has_k2)
                def _pf_w2_2x():
                    start_w2_slot(
                        b_idx,
                        exec_src_smem[k2_clamped],
                        exec_slot_smem[k2_clamped],
                    )

            elif w_mode == "w1_2x_w2_1x":
                b_idx = k % 2
                wait_w1_slot(b_idx)
                _run_gmm1_slice(
                    x_in_ref, w1_buf.at[b_idx], s1_buf.at[b_idx], 0, _I, batch_size
                )

                @pl.when(has_k2)
                def _pf_w1_k2():
                    start_w1_slot(
                        b_idx,
                        exec_src_smem[k2_clamped],
                        exec_slot_smem[k2_clamped],
                    )

                wait_w2_1x()
                sq_sum = _run_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, batch_size, sq_sum
                )

                @pl.when(has_next)
                def _pf_w2_next():
                    start_w2_1x(next_src, next_slot)

            else:
                wait_w1_half(0)
                _run_gmm1_slice(
                    x_in_ref, w1_buf.at[0], s1_buf.at[0], 0, _I // 2, batch_size
                )

                @pl.when(has_next)
                def _pf10():
                    start_w1_half(0, next_src, next_slot)

                wait_w1_half(1)
                _run_gmm1_slice(
                    x_in_ref,
                    w1_buf.at[1],
                    s1_buf.at[1],
                    _I // 2,
                    _I // 2,
                    batch_size,
                )

                @pl.when(has_next)
                def _pf11():
                    start_w1_half(1, next_src, next_slot)

                wait_w2_half(0)
                sq_sum = _run_gmm2_slice(
                    w2_buf.at[0], s2_buf.at[0], 0, _H // 2, batch_size, sq_sum
                )

                @pl.when(has_next)
                def _pf20():
                    start_w2_half(0, next_src, next_slot)

                wait_w2_half(1)
                sq_sum = _run_gmm2_slice(
                    w2_buf.at[1],
                    s2_buf.at[1],
                    _H // 2,
                    _H // 2,
                    batch_size,
                    sq_sum,
                )

                @pl.when(has_next)
                def _pf21():
                    start_w2_half(1, next_src, next_slot)

            # Extract column g_e from dense_w_vmem[B, 256] using 128-lane block select.
            blk = pl.multiple_of((g_e >> 7) << 7, 128)
            w_slice = d_scratch.dense_w_vmem[:, pl.ds(blk, 128)]
            w_col = jnp.sum(
                jnp.where((lane_128 + blk) == g_e, w_slice, 0.0),
                axis=1,
                keepdims=True,
            )
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            rc = min(
                batch_size,
                (16 if cfg.norm_f32_stage else 32)
                if cfg.fast_b64_epi_pro
                else (8 if cfg.norm_f32_stage else 16),
            )
            for r0 in range(0, batch_size, rc):
                inv_r = inv_rms[r0 : r0 + rc, :]
                w_r = w_col[r0 : r0 + rc, :]
                for s_col in range(0, _H, 1024):
                    col_s = pl.ds(s_col, 1024)
                    if cfg.norm_f32_stage:
                        u_c = y_stage_buf[pl.ds(r0, rc), col_s]
                        v_bf = (u_c * inv_r).astype(jnp.bfloat16)
                    else:
                        u_c = y_stage_buf[pl.ds(r0, rc), col_s]
                        v_bf = u_c * inv_r.astype(jnp.bfloat16)
                    contrib = jnp.where(
                        w_r > 0.0, v_bf.astype(jnp.float32) * w_r, 0.0
                    )
                    accum_2d[pl.ds(r0, rc), col_s] = (
                        accum_2d[pl.ds(r0, rc), col_s] + contrib
                    )
            return _

        lax.fori_loop(0, num_exec, _dense_expert_body, jnp.int32(0))

        # Convert accum_2d [B, 4096] -> accum_3d [B, 32, 128] once before RS.
        conv_rc = min(batch_size, 32 if cfg.fast_b64_epi_pro else 8)
        for r0 in range(0, batch_size, conv_rc):
            for h_idx in range(2):
                accum_3d[pl.ds(r0, conv_rc), pl.ds(h_idx * 16, 16), :] = (
                    accum_2d[
                        pl.ds(r0, conv_rc), pl.ds(h_idx * 2048, 2048)
                    ].reshape(conv_rc, 16, 128)
                )

    # =========================================================================
    # 6B. SPARSE EXECUTION LOOP (8x-unrolled VPU gather/scatter + 2-bucket switch)
    # =========================================================================
    else:

        @jax.named_scope("gather_rows")
        def _gather_rows_to_3d(base_pos, cur_rows):
            if cfg.fast_gs:
                max_pk = int(r_scratch.packed_tok_ids.shape[0])
                safe_b = jnp.minimum(base_pos, jnp.int32(max_pk - 1))
                ts8 = [
                    jnp.clip(
                        r_scratch.packed_tok_ids[
                            jnp.where(jnp.int32(s) < cur_rows, base_pos + s, safe_b)
                        ],
                        jnp.int32(0),
                        jnp.int32(batch_size - 1),
                    )
                    for s in range(8)
                ]
                io_3d_buf[pl.ds(0, 8), :, :] = jnp.concatenate(
                    [x_3d_buf[pl.ds(t, 1), :, :] for t in ts8], axis=0
                )

                @pl.when(cur_rows > 8)
                def _gather_gt8_fgs():
                    n8 = cur_rows >> 3

                    def _gather_8_hi(q, _):
                        i0 = pl.multiple_of(q << 3, 8)
                        p0 = base_pos + i0
                        ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                        rows = [x_3d_buf[pl.ds(t, 1), :, :] for t in ts]
                        io_3d_buf[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
                        return _

                    lax.fori_loop(1, n8, _gather_8_hi, jnp.int32(0))

                    def _gather_tail_hi(i, _):
                        tok = r_scratch.packed_tok_ids[base_pos + i]
                        io_3d_buf[pl.ds(i, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
                        return _

                    lax.fori_loop(n8 << 3, cur_rows, _gather_tail_hi, jnp.int32(0))

                return

            if cfg.fast_b64_epi_pro:

                @pl.when(cur_rows <= 8)
                def _gather_le8():
                    ts8 = [
                        r_scratch.packed_tok_ids[
                            jnp.where(s < cur_rows, base_pos + s, base_pos)
                        ]
                        for s in range(8)
                    ]
                    io_3d_buf[pl.ds(0, 8), :, :] = jnp.concatenate(
                        [x_3d_buf[pl.ds(t, 1), :, :] for t in ts8], axis=0
                    )

                @pl.when(cur_rows > 8)
                def _gather_gt8():
                    n8 = cur_rows >> 3

                    def _gather_8(q, _):
                        i0 = pl.multiple_of(q << 3, 8)
                        p0 = base_pos + i0
                        ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                        rows = [x_3d_buf[pl.ds(t, 1), :, :] for t in ts]
                        io_3d_buf[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
                        return _

                    lax.fori_loop(0, n8, _gather_8, jnp.int32(0))

                    def _gather_tail(i, _):
                        tok = r_scratch.packed_tok_ids[base_pos + i]
                        io_3d_buf[pl.ds(i, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
                        return _

                    lax.fori_loop(n8 << 3, cur_rows, _gather_tail, jnp.int32(0))

                return

            n8 = cur_rows >> 3

            def _gather_8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                rows = [x_3d_buf[pl.ds(t, 1), :, :] for t in ts]
                io_3d_buf[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
                return _

            lax.fori_loop(0, n8, _gather_8, jnp.int32(0))

            def _gather_tail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                io_3d_buf[pl.ds(i, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
                return _

            lax.fori_loop(n8 << 3, cur_rows, _gather_tail, jnp.int32(0))

        def _copy_io_3d_to_lhs_2d_at(r_base, m_rows: int):
            copy_m = 16 if m_rows == 8 else m_rows
            for r0 in range(0, copy_m, 16):
                rr = r_base + r0
                for h_idx in range(2):
                    lhs_2d_buf[pl.ds(rr, 16), pl.ds(h_idx * 2048, 2048)] = (
                        io_3d_buf[
                            pl.ds(rr, 16), pl.ds(h_idx * 16, 16), :
                        ].reshape(16, 2048)
                    )

        def _finish_rmsnorm_to_io_3d_at(
            r_base, m_rows: int, sq_sum: jax.Array
        ):
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            if cfg.norm_f32_stage:
                rc = 8
                for r0 in range(0, m_rows, rc):
                    inv_r = inv_rms[r0 : r0 + rc, :]
                    for h_idx in range(2):
                        u_c = y_stage_buf[
                            pl.ds(r0, rc), pl.ds(h_idx * 2048, 2048)
                        ]
                        v_bf = (u_c * inv_r).astype(jnp.bfloat16)
                        io_3d_buf[
                            pl.ds(r_base + r0, rc), pl.ds(h_idx * 16, 16), :
                        ] = v_bf.reshape(rc, 16, 128)
            else:
                inv_bf = inv_rms.astype(jnp.bfloat16)
                rc = 8 if m_rows == 8 else 16
                read_rc = 16 if m_rows == 8 else rc
                for r0 in range(0, m_rows, rc):
                    inv_r = inv_bf[r0 : r0 + rc, :]
                    for h_idx in range(2):
                        u_c = y_stage_buf[
                            pl.ds(r0, read_rc), pl.ds(h_idx * 2048, 2048)
                        ]
                        if m_rows == 8:
                            u_c = u_c[:8, :]
                        v_bf = u_c * inv_r
                        io_3d_buf[
                            pl.ds(r_base + r0, rc), pl.ds(h_idx * 16, 16), :
                        ] = v_bf.reshape(rc, 16, 128)

        @jax.named_scope("scatter_acc")
        def _scatter_accumulate(base_pos, cur_rows):
            # All token IDs within a single expert are strictly distinct.
            if cfg.fast_gs:
                max_pk = int(r_scratch.packed_tok_ids.shape[0])
                safe_b = jnp.minimum(base_pos, jnp.int32(max_pk - 1))
                y8_f = io_3d_buf[pl.ds(0, 8), :, :].astype(jnp.float32)
                for s in range(8):
                    valid_s = jnp.int32(s) < cur_rows
                    p_s = jnp.where(valid_s, base_pos + s, safe_b)
                    tok = jnp.where(
                        valid_s,
                        r_scratch.packed_tok_ids[p_s],
                        jnp.int32(batch_size),
                    )
                    w = jnp.where(
                        valid_s,
                        r_scratch.packed_tok_w[p_s],
                        jnp.float32(0.0),
                    )
                    ys = y8_f[s : s + 1, :, :] * w
                    accum_3d[pl.ds(tok, 1), :, :] = (
                        accum_3d[pl.ds(tok, 1), :, :] + ys
                    )

                @pl.when(cur_rows > 8)
                def _acc_gt8_fgs():
                    n8 = cur_rows >> 3

                    def _acc_8_hi(q, _):
                        i0 = pl.multiple_of(q << 3, 8)
                        p0 = base_pos + i0
                        ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                        ws = [r_scratch.packed_tok_w[p0 + s] for s in range(8)]
                        y8_hi = io_3d_buf[pl.ds(i0, 8), :, :].astype(jnp.float32)
                        for s in range(8):
                            ys_hi = y8_hi[s : s + 1, :, :] * ws[s]
                            accum_3d[pl.ds(ts[s], 1), :, :] = (
                                accum_3d[pl.ds(ts[s], 1), :, :] + ys_hi
                            )
                        return _

                    lax.fori_loop(1, n8, _acc_8_hi, jnp.int32(0))

                    def _acc_tail_hi(i, _):
                        tok = r_scratch.packed_tok_ids[base_pos + i]
                        w = r_scratch.packed_tok_w[base_pos + i]
                        y_row = io_3d_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                        accum_3d[pl.ds(tok, 1), :, :] = (
                            accum_3d[pl.ds(tok, 1), :, :] + y_row * w
                        )
                        return _

                    lax.fori_loop(n8 << 3, cur_rows, _acc_tail_hi, jnp.int32(0))

                return

            if cfg.fast_b64_epi_pro:

                @pl.when(cur_rows <= 8)
                def _acc_le8():
                    y8 = io_3d_buf[pl.ds(0, 8), :, :]
                    for s in range(8):

                        @pl.when(s < cur_rows)
                        def _sc_one(_s=s):
                            tok = r_scratch.packed_tok_ids[base_pos + _s]
                            w = r_scratch.packed_tok_w[base_pos + _s]
                            ys = y8[_s : _s + 1, :, :].astype(jnp.float32) * w
                            accum_3d[pl.ds(tok, 1), :, :] = (
                                accum_3d[pl.ds(tok, 1), :, :] + ys
                            )

                @pl.when(cur_rows > 8)
                def _acc_gt8():
                    n8 = cur_rows >> 3

                    def _acc_8(q, _):
                        i0 = pl.multiple_of(q << 3, 8)
                        p0 = base_pos + i0
                        ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                        ws = [r_scratch.packed_tok_w[p0 + s] for s in range(8)]
                        y8 = io_3d_buf[pl.ds(i0, 8), :, :]
                        for s in range(8):
                            ys = y8[s : s + 1, :, :].astype(jnp.float32) * ws[s]
                            accum_3d[pl.ds(ts[s], 1), :, :] = (
                                accum_3d[pl.ds(ts[s], 1), :, :] + ys
                            )
                        return _

                    lax.fori_loop(0, n8, _acc_8, jnp.int32(0))

                    def _acc_tail(i, _):
                        tok = r_scratch.packed_tok_ids[base_pos + i]
                        w = r_scratch.packed_tok_w[base_pos + i]
                        y_row = io_3d_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                        accum_3d[pl.ds(tok, 1), :, :] = (
                            accum_3d[pl.ds(tok, 1), :, :] + y_row * w
                        )
                        return _

                    lax.fori_loop(n8 << 3, cur_rows, _acc_tail, jnp.int32(0))

                return

            n8 = cur_rows >> 3

            def _acc_8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                ws = [r_scratch.packed_tok_w[p0 + s] for s in range(8)]
                y8 = io_3d_buf[pl.ds(i0, 8), :, :]
                for s in range(8):
                    ys = y8[s : s + 1, :, :].astype(jnp.float32) * ws[s]
                    accum_3d[pl.ds(ts[s], 1), :, :] = (
                        accum_3d[pl.ds(ts[s], 1), :, :] + ys
                    )
                return _

            lax.fori_loop(0, n8, _acc_8, jnp.int32(0))

            def _acc_tail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                w = r_scratch.packed_tok_w[base_pos + i]
                y_row = io_3d_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                accum_3d[pl.ds(tok, 1), :, :] = (
                    accum_3d[pl.ds(tok, 1), :, :] + y_row * w
                )
                return _

            lax.fori_loop(n8 << 3, cur_rows, _acc_tail, jnp.int32(0))

        # --- Block-diagonal K16 helpers (for r_e <= max_bd_rows) ---
        def _run_bd_proj_only(w_ref, s_ref, width: int, bd_rows: int, is_up: bool):
            nc = _pick_bd_nc(bd_rows, width, cfg.k16_bd_narrow_nc)

            def _strip(j, carry):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(n0, nc)
                res = _fp4_blockdiag_hoisted_strip(
                    masked_x_buf, w_ref, s_ref, bd_rows, n0, nc, bd_sc_buf
                )
                if not is_up:
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                        jax.nn.silu(res).astype(jnp.bfloat16), 16
                    )
                else:
                    sg = act_2d_buf[pl.ds(0, 16), col_out][:bd_rows, :].astype(
                        jnp.float32
                    )
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                        (sg * res).astype(jnp.bfloat16), 16
                    )
                return carry

            n_strips = width // nc
            if n_strips == 1:
                _strip(0, 0)
            elif n_strips == 2:
                _strip(0, 0)
                _strip(1, 0)
            else:
                lax.fori_loop(0, n_strips, _strip, 0)

        def _run_bd_gmm1_slice(
            w_ref, s_ref, col_offset: int, width: int, bd_rows: int
        ):
            nc = _pick_bd_nc(bd_rows, width, cfg.k16_bd_narrow_nc)
            up_base = width
            use_dual_bd = (bd_rows * nc) // 128 <= 4

            def _strip(j, carry):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(col_offset + n0, nc)
                if use_dual_bd:
                    g, u = _fp4_blockdiag_hoisted_dual_strip(
                        masked_x_buf,
                        w_ref,
                        s_ref,
                        bd_rows,
                        n0,
                        up_base + n0,
                        nc,
                        bd_sc_buf,
                    )
                    h = (
                        jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32)
                        * u
                    ).astype(jnp.bfloat16)
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(h, 16)
                else:
                    g = _fp4_blockdiag_hoisted_strip(
                        masked_x_buf, w_ref, s_ref, bd_rows, n0, nc, bd_sc_buf
                    )
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                        jax.nn.silu(g).astype(jnp.bfloat16), 16
                    )
                    u = _fp4_blockdiag_hoisted_strip(
                        masked_x_buf,
                        w_ref,
                        s_ref,
                        bd_rows,
                        up_base + n0,
                        nc,
                        bd_sc_buf,
                    )
                    sg = act_2d_buf[pl.ds(0, 16), col_out][:bd_rows, :].astype(
                        jnp.float32
                    )
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                        (sg * u).astype(jnp.bfloat16), 16
                    )
                return carry

            n_strips = width // nc
            if n_strips == 1:
                _strip(0, 0)
            elif n_strips == 2:
                _strip(0, 0)
                _strip(1, 0)
            else:
                lax.fori_loop(0, n_strips, _strip, 0)

        def _run_bd_gmm2_slice(
            w_ref, s_ref, col_offset: int, width: int, bd_rows: int, sq_sum
        ):
            nc = _pick_bd_nc(bd_rows, width, cfg.k16_bd_narrow_nc)
            pad_f32 = 16 if bd_rows > 8 else 8

            def _strip(j, sq_acc):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(col_offset + n0, nc)
                d = _fp4_blockdiag_hoisted_strip(
                    masked_x_buf, w_ref, s_ref, bd_rows, n0, nc, bd_sc_buf
                ).astype(jnp.bfloat16)
                g_col = gain_f32_buf[0:1, col_out]
                if cfg.norm_f32_stage:
                    u = d.astype(jnp.float32) * g_col
                    sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
                    y_stage_buf[pl.ds(0, pad_f32), col_out] = _pad_rows_to(
                        u, pad_f32
                    )
                else:
                    u_bf = d * g_col.astype(jnp.bfloat16)
                    u_f32 = u_bf.astype(jnp.float32)
                    sq_acc = sq_acc + jnp.sum(
                        u_f32 * u_f32, axis=1, keepdims=True
                    )
                    y_stage_buf[pl.ds(0, 16), col_out] = _pad_rows_to(u_bf, 16)
                return sq_acc

            n_strips = width // nc
            if n_strips == 1:
                return _strip(0, sq_sum)
            if n_strips == 2:
                sq_sum = _strip(0, sq_sum)
                return _strip(1, sq_sum)
            return lax.fori_loop(0, n_strips, _strip, sq_sum)

        def _finish_bd_rmsnorm_to_io_3d(bd_rows: int, sq_sum: jax.Array):
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            pad_r = 16 if bd_rows > 8 else 8
            if cfg.norm_f32_stage:
                inv_p = _pad_rows_to(inv_rms, pad_r)
                for r0 in range(0, pad_r, 8):
                    inv_8 = inv_p[r0 : r0 + 8, :]
                    for h_idx in range(2):
                        u_c = y_stage_buf[
                            pl.ds(r0, 8), pl.ds(h_idx * 2048, 2048)
                        ]
                        v_bf = (u_c * inv_8).astype(jnp.bfloat16)
                        io_3d_buf[pl.ds(r0, 8), pl.ds(h_idx * 16, 16), :] = (
                            v_bf.reshape(8, 16, 128)
                        )
            else:
                inv_p = _pad_rows_to(inv_rms.astype(jnp.bfloat16), pad_r)
                for r0 in range(0, pad_r, 8):
                    inv_8 = inv_p[r0 : r0 + 8, :]
                    u_full = y_stage_buf[pl.ds(0, 16), :]
                    for h_idx in range(2):
                        u_c = u_full[r0 : r0 + 8, h_idx * 2048 : (h_idx + 1) * 2048]
                        v_bf = u_c * inv_8
                        io_3d_buf[pl.ds(r0, 8), pl.ds(h_idx * 16, 16), :] = (
                            v_bf.reshape(8, 16, 128)
                        )

        bd_rows_tuple = cfg.k16_blockdiag_rows if use_bd else ()
        m_buckets = (
            tuple(b for b in cfg.m_buckets if b > bd_rows_tuple[-1])
            if bd_rows_tuple
            else cfg.m_buckets
        )
        if not m_buckets:
            m_buckets = (tile_m,)

        def _compute_tile_standard(
            m_rows: int,
            cur_rows,
            k_exp,
            is_first_tile,
            is_last_tile,
            has_next,
            next_src,
            next_slot,
            has_k2,
            k2_src,
            k2_slot,
        ):
            do_pf_next = jnp.logical_and(is_last_tile, has_next)
            do_pf_k2 = jnp.logical_and(is_last_tile, has_k2)
            do_loop32 = (
                cfg.loop32_in_tile
                and (m_rows == 32)
                and (tile_m > 32)
                and (not use_lhs_fp8)
            )

            if do_loop32:
                n_m32 = jnp.maximum(jnp.int32(1), (cur_rows + 31) // 32)

                def _cp_lhs_32(mi, _):
                    r0 = pl.multiple_of(mi * 32, 32)
                    _copy_io_3d_to_lhs_2d_at(r0, 32)
                    return _

                lax.fori_loop(0, n_m32, _cp_lhs_32, jnp.int32(0))

                if w_mode in ("stream3", "stream3_k2"):

                    @pl.when(is_first_tile)
                    def _l32_s3_wg():
                        wait_s3_part(0)

                    def _l32_s3_gate(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        if w_mode == "stream3_k2":
                            _run_k2_proj_half(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w1_buf.at[0],
                                s1_buf.at[0],
                                0,
                                32,
                                act_2d_buf.at[pl.ds(r0, 32)],
                                w1_acc_buf.at[pl.ds(r0, 32)],
                            )
                        else:
                            _run_proj_only_slice(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w1_buf.at[0],
                                s1_buf.at[0],
                                _I,
                                32,
                                False,
                                act_2d_buf.at[pl.ds(r0, 32)],
                            )
                        return _

                    lax.fori_loop(0, n_m32, _l32_s3_gate, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_s3_pfg():
                        start_s3_part(0, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_s3_wu():
                        wait_s3_part(1)

                    def _l32_s3_up(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        if w_mode == "stream3_k2":
                            _run_k2_proj_half(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w1_buf.at[1],
                                s1_buf.at[1],
                                1,
                                32,
                                act_2d_buf.at[pl.ds(r0, 32)],
                                w1_acc_buf.at[pl.ds(r0, 32)],
                            )
                        else:
                            _run_proj_only_slice(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w1_buf.at[1],
                                s1_buf.at[1],
                                _I,
                                32,
                                True,
                                act_2d_buf.at[pl.ds(r0, 32)],
                            )
                        return _

                    lax.fori_loop(0, n_m32, _l32_s3_up, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_s3_pfu():
                        start_s3_part(1, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_s3_wd():
                        wait_s3_down()

                    def _l32_s3_down(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_s = _run_gmm2_slice(
                            w2_buf,
                            s2_buf,
                            0,
                            _H,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq_s)
                        return _

                    lax.fori_loop(0, n_m32, _l32_s3_down, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_s3_pfd():
                        start_s3_down(next_src, next_slot)

                elif w_mode == "w1_2x_w2_2x":
                    b_idx = k_exp % 2

                    @pl.when(is_first_tile)
                    def _l32_w22_w1():
                        wait_w1_slot(b_idx)

                    def _l32_w22_g1(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_gmm1_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[b_idx],
                            s1_buf.at[b_idx],
                            0,
                            _I,
                            32,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_w22_g1, jnp.int32(0))

                    @pl.when(do_pf_k2)
                    def _l32_w22_pf1():
                        start_w1_slot(b_idx, k2_src, k2_slot)

                    @pl.when(is_first_tile)
                    def _l32_w22_w2():
                        wait_w2_slot(b_idx)

                    def _l32_w22_g2(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_s = _run_gmm2_slice(
                            w2_buf.at[b_idx],
                            s2_buf.at[b_idx],
                            0,
                            _H,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq_s)
                        return _

                    lax.fori_loop(0, n_m32, _l32_w22_g2, jnp.int32(0))

                    @pl.when(do_pf_k2)
                    def _l32_w22_pf2():
                        start_w2_slot(b_idx, k2_src, k2_slot)

                elif w_mode == "w1_2x_w2_1x":
                    b_idx = k_exp % 2

                    @pl.when(is_first_tile)
                    def _l32_w21_w1():
                        wait_w1_slot(b_idx)

                    def _l32_w21_g1(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_gmm1_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[b_idx],
                            s1_buf.at[b_idx],
                            0,
                            _I,
                            32,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_w21_g1, jnp.int32(0))

                    @pl.when(do_pf_k2)
                    def _l32_w21_pf1():
                        start_w1_slot(b_idx, k2_src, k2_slot)

                    @pl.when(is_first_tile)
                    def _l32_w21_w2():
                        wait_w2_1x()

                    def _l32_w21_g2(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_s = _run_gmm2_slice(
                            w2_buf,
                            s2_buf,
                            0,
                            _H,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq_s)
                        return _

                    lax.fori_loop(0, n_m32, _l32_w21_g2, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_w21_pf2():
                        start_w2_1x(next_src, next_slot)

                else:

                    @pl.when(is_first_tile)
                    def _l32_hn_10():
                        wait_w1_half(0)

                    def _l32_hn_g10(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_gmm1_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[0],
                            s1_buf.at[0],
                            0,
                            _I // 2,
                            32,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_hn_g10, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_hn_pf10():
                        start_w1_half(0, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_hn_11():
                        wait_w1_half(1)

                    def _l32_hn_g11(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_gmm1_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[1],
                            s1_buf.at[1],
                            _I // 2,
                            _I // 2,
                            32,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_hn_g11, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_hn_pf11():
                        start_w1_half(1, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_hn_20():
                        wait_w2_half(0)

                    @pl.when(is_first_tile)
                    def _l32_hn_21():
                        wait_w2_half(1)

                    def _l32_hn_g2(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_s = _run_gmm2_slice(
                            w2_buf.at[0],
                            s2_buf.at[0],
                            0,
                            _H // 2,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        sq_s = _run_gmm2_slice(
                            w2_buf.at[1],
                            s2_buf.at[1],
                            _H // 2,
                            _H // 2,
                            32,
                            sq_s,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq_s)
                        return _

                    lax.fori_loop(0, n_m32, _l32_hn_g2, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_hn_pf2():
                        start_w2_half(0, next_src, next_slot)
                        start_w2_half(1, next_src, next_slot)

                return

            _copy_io_3d_to_lhs_2d_at(0, m_rows)
            if use_lhs_fp8:
                _prequant_lhs_k256(lhs_2d_buf, lhs_fp8_buf, lhs_sc_buf, m_rows)
            sq_sum = jnp.zeros((m_rows, 1), dtype=jnp.float32)

            if w_mode in ("stream3", "stream3_k2"):

                @pl.when(is_first_tile)
                def _s3_wait_g():
                    wait_s3_part(0)

                if w_mode == "stream3_k2":
                    _run_k2_proj_half(
                        lhs_2d_buf, w1_buf.at[0], s1_buf.at[0], 0, m_rows
                    )
                else:
                    _run_proj_only_slice(
                        lhs_2d_buf, w1_buf.at[0], s1_buf.at[0], _I, m_rows, False
                    )

                @pl.when(do_pf_next)
                def _s3_pf_g():
                    start_s3_part(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _s3_wait_u():
                    wait_s3_part(1)

                if w_mode == "stream3_k2":
                    _run_k2_proj_half(
                        lhs_2d_buf, w1_buf.at[1], s1_buf.at[1], 1, m_rows
                    )
                else:
                    _run_proj_only_slice(
                        lhs_2d_buf, w1_buf.at[1], s1_buf.at[1], _I, m_rows, True
                    )

                @pl.when(do_pf_next)
                def _s3_pf_u():
                    start_s3_part(1, next_src, next_slot)

                @pl.when(is_first_tile)
                def _s3_wait_d():
                    wait_s3_down()

                sq_sum = _run_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _s3_pf_d():
                    start_s3_down(next_src, next_slot)

            elif w_mode == "w1_2x_w2_2x":
                b_idx = k_exp % 2

                @pl.when(is_first_tile)
                def _w22_wait_w1():
                    wait_w1_slot(b_idx)

                _run_gmm1_slice(
                    lhs_2d_buf, w1_buf.at[b_idx], s1_buf.at[b_idx], 0, _I, m_rows
                )

                @pl.when(do_pf_k2)
                def _w22_pf_w1():
                    start_w1_slot(b_idx, k2_src, k2_slot)

                @pl.when(is_first_tile)
                def _w22_wait_w2():
                    wait_w2_slot(b_idx)

                sq_sum = _run_gmm2_slice(
                    w2_buf.at[b_idx], s2_buf.at[b_idx], 0, _H, m_rows, sq_sum
                )

                @pl.when(do_pf_k2)
                def _w22_pf_w2():
                    start_w2_slot(b_idx, k2_src, k2_slot)

            elif w_mode == "w1_2x_w2_1x":
                b_idx = k_exp % 2

                @pl.when(is_first_tile)
                def _w21_wait_w1():
                    wait_w1_slot(b_idx)

                _run_gmm1_slice(
                    lhs_2d_buf, w1_buf.at[b_idx], s1_buf.at[b_idx], 0, _I, m_rows
                )

                @pl.when(do_pf_k2)
                def _w21_pf_w1():
                    start_w1_slot(b_idx, k2_src, k2_slot)

                @pl.when(is_first_tile)
                def _w21_wait_w2():
                    wait_w2_1x()

                sq_sum = _run_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _w21_pf_w2():
                    start_w2_1x(next_src, next_slot)

            else:

                @pl.when(is_first_tile)
                def _hn_w10():
                    wait_w1_half(0)

                _run_gmm1_slice(
                    lhs_2d_buf, w1_buf.at[0], s1_buf.at[0], 0, _I // 2, m_rows
                )

                @pl.when(do_pf_next)
                def _hn_pf10():
                    start_w1_half(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _hn_w11():
                    wait_w1_half(1)

                _run_gmm1_slice(
                    lhs_2d_buf,
                    w1_buf.at[1],
                    s1_buf.at[1],
                    _I // 2,
                    _I // 2,
                    m_rows,
                )

                @pl.when(do_pf_next)
                def _hn_pf11():
                    start_w1_half(1, next_src, next_slot)

                @pl.when(is_first_tile)
                def _hn_w20():
                    wait_w2_half(0)

                sq_sum = _run_gmm2_slice(
                    w2_buf.at[0], s2_buf.at[0], 0, _H // 2, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _hn_pf20():
                    start_w2_half(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _hn_w21():
                    wait_w2_half(1)

                sq_sum = _run_gmm2_slice(
                    w2_buf.at[1], s2_buf.at[1], _H // 2, _H // 2, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _hn_pf21():
                    start_w2_half(1, next_src, next_slot)

            _finish_rmsnorm_to_io_3d_at(0, m_rows, sq_sum)

        def _compute_tile_blockdiag(
            bd_rows: int,
            k_exp,
            is_first_tile,
            is_last_tile,
            has_next,
            next_src,
            next_slot,
            has_k2,
            k2_src,
            k2_slot,
        ):
            _build_masked_x_from_3d(io_3d_buf, masked_x_buf, bd_rows, bd_sc_buf)
            sq_sum = jnp.zeros((bd_rows, 1), dtype=jnp.float32)
            do_pf_next = jnp.logical_and(is_last_tile, has_next)
            do_pf_k2 = jnp.logical_and(is_last_tile, has_k2)

            if w_mode == "stream3":

                @pl.when(is_first_tile)
                def _bd_s3_wait_g():
                    wait_s3_part(0)

                _run_bd_proj_only(
                    w1_buf.at[0], s1_buf.at[0], _I, bd_rows, False
                )

                @pl.when(do_pf_next)
                def _bd_s3_pf_g():
                    start_s3_part(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _bd_s3_wait_u():
                    wait_s3_part(1)

                _run_bd_proj_only(
                    w1_buf.at[1], s1_buf.at[1], _I, bd_rows, True
                )

                @pl.when(do_pf_next)
                def _bd_s3_pf_u():
                    start_s3_part(1, next_src, next_slot)

                _build_masked_x_from_2d(
                    act_2d_buf, masked_x_buf, bd_rows, bd_sc_buf
                )

                @pl.when(is_first_tile)
                def _bd_s3_wait_d():
                    wait_s3_down()

                sq_sum = _run_bd_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, bd_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _bd_s3_pf_d():
                    start_s3_down(next_src, next_slot)

            elif w_mode == "w1_2x_w2_2x":
                b_idx = k_exp % 2

                @pl.when(is_first_tile)
                def _bd_w22_w1():
                    wait_w1_slot(b_idx)

                _run_bd_gmm1_slice(
                    w1_buf.at[b_idx], s1_buf.at[b_idx], 0, _I, bd_rows
                )

                @pl.when(do_pf_k2)
                def _bd_w22_pf_w1():
                    start_w1_slot(b_idx, k2_src, k2_slot)

                _build_masked_x_from_2d(
                    act_2d_buf, masked_x_buf, bd_rows, bd_sc_buf
                )

                @pl.when(is_first_tile)
                def _bd_w22_w2():
                    wait_w2_slot(b_idx)

                sq_sum = _run_bd_gmm2_slice(
                    w2_buf.at[b_idx], s2_buf.at[b_idx], 0, _H, bd_rows, sq_sum
                )

                @pl.when(do_pf_k2)
                def _bd_w22_pf_w2():
                    start_w2_slot(b_idx, k2_src, k2_slot)

            elif w_mode == "w1_2x_w2_1x":
                b_idx = k_exp % 2

                @pl.when(is_first_tile)
                def _bd_w21_w1():
                    wait_w1_slot(b_idx)

                _run_bd_gmm1_slice(
                    w1_buf.at[b_idx], s1_buf.at[b_idx], 0, _I, bd_rows
                )

                @pl.when(do_pf_k2)
                def _bd_w21_pf_w1():
                    start_w1_slot(b_idx, k2_src, k2_slot)

                _build_masked_x_from_2d(
                    act_2d_buf, masked_x_buf, bd_rows, bd_sc_buf
                )

                @pl.when(is_first_tile)
                def _bd_w21_w2():
                    wait_w2_1x()

                sq_sum = _run_bd_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, bd_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _bd_w21_pf_w2():
                    start_w2_1x(next_src, next_slot)

            else:

                @pl.when(is_first_tile)
                def _bd_hn_w10():
                    wait_w1_half(0)

                _run_bd_gmm1_slice(
                    w1_buf.at[0], s1_buf.at[0], 0, _I // 2, bd_rows
                )

                @pl.when(do_pf_next)
                def _bd_hn_pf10():
                    start_w1_half(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _bd_hn_w11():
                    wait_w1_half(1)

                _run_bd_gmm1_slice(
                    w1_buf.at[1], s1_buf.at[1], _I // 2, _I // 2, bd_rows
                )

                @pl.when(do_pf_next)
                def _bd_hn_pf11():
                    start_w1_half(1, next_src, next_slot)

                _build_masked_x_from_2d(
                    act_2d_buf, masked_x_buf, bd_rows, bd_sc_buf
                )

                @pl.when(is_first_tile)
                def _bd_hn_w20():
                    wait_w2_half(0)

                sq_sum = _run_bd_gmm2_slice(
                    w2_buf.at[0], s2_buf.at[0], 0, _H // 2, bd_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _bd_hn_pf20():
                    start_w2_half(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _bd_hn_w21():
                    wait_w2_half(1)

                sq_sum = _run_bd_gmm2_slice(
                    w2_buf.at[1], s2_buf.at[1], _H // 2, _H // 2, bd_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _bd_hn_pf21():
                    start_w2_half(1, next_src, next_slot)

            _finish_bd_rmsnorm_to_io_3d(bd_rows, sq_sum)

        def _sparse_expert_body(k, _):
            has_next = (k + 1) < num_exec
            next_k = jnp.where(has_next, k + 1, jnp.int32(0))
            next_src = exec_src_smem[next_k]
            next_slot = exec_slot_smem[next_k]
            has_k2 = (k + 2) < num_exec
            k2_k = jnp.where(has_k2, k + 2, jnp.int32(0))
            k2_src = exec_src_smem[k2_k]
            k2_slot = exec_slot_smem[k2_k]

            row_start = r_scratch.exec_start[k]
            row_end = r_scratch.exec_start[k + 1]
            r_e = row_end - row_start
            n_row_tiles = (r_e + tile_m - 1) // tile_m

            def _row_tile_body(tile_idx, _c):
                if cfg.fast_gs and batch_size <= tile_m:
                    base_pos = row_start
                    cur_rows = r_e
                    is_first_tile = jnp.bool_(True)
                    is_last_tile = jnp.bool_(True)
                else:
                    tile_off = tile_idx * tile_m
                    base_pos = row_start + tile_off
                    cur_rows = jnp.minimum(tile_m, r_e - tile_off)
                    is_first_tile = tile_idx == 0
                    is_last_tile = tile_idx == (n_row_tiles - 1)

                _gather_rows_to_3d(base_pos, cur_rows)

                bd_args = (
                    k,
                    is_first_tile,
                    is_last_tile,
                    has_next,
                    next_src,
                    next_slot,
                    has_k2,
                    k2_src,
                    k2_slot,
                )
                std_args = (cur_rows,) + bd_args
                if use_bd:
                    bd_idx = jnp.int32(0)
                    for _r in bd_rows_tuple[:-1]:
                        bd_idx = bd_idx + (cur_rows > _r).astype(jnp.int32)
                    b_idx = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                    idx = jnp.where(
                        cur_rows <= bd_rows_tuple[-1],
                        bd_idx,
                        len(bd_rows_tuple) + b_idx,
                    )
                    branches = [
                        (
                            lambda r=r: (
                                _compute_tile_blockdiag(r, *bd_args),
                                jnp.int32(0),
                            )[1]
                        )
                        for r in bd_rows_tuple
                    ] + [
                        (
                            lambda b=b: (
                                _compute_tile_standard(b, *std_args),
                                jnp.int32(0),
                            )[1]
                        )
                        for b in m_buckets
                    ]
                    lax.switch(idx, branches)
                elif len(m_buckets) > 1:
                    b_idx = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                    branches = [
                        (
                            lambda b=b: (
                                _compute_tile_standard(b, *std_args),
                                jnp.int32(0),
                            )[1]
                        )
                        for b in m_buckets
                    ]
                    lax.switch(b_idx, branches)
                else:
                    _compute_tile_standard(m_buckets[0], *std_args)

                _scatter_accumulate(base_pos, cur_rows)
                return _c

            if cfg.fast_gs and batch_size <= tile_m:
                _row_tile_body(jnp.int32(0), jnp.int32(0))
            else:
                lax.fori_loop(0, n_row_tiles, _row_tile_body, jnp.int32(0))
            return _

        lax.fori_loop(0, num_exec, _sparse_expert_body, jnp.int32(0))

    if stop_stage == "experts":
        if early_exit_fn is not None:
            early_exit_fn()
        return

    if on_experts_done_fn is not None:
        on_experts_done_fn()

    # 7. Contiguous EP-1 multi-row Reduce-Scatter.
    if cfg.rs_wire_bf16:
        if (cfg.fast_b64_epi_pro and batch_size <= 64) or (
            cfg.fast_gs and batch_size <= 128
        ):
            cast_step = min(batch_size, 32)
            for r0 in range(0, batch_size, cast_step):
                rs_send_bf16[pl.ds(r0, cast_step), :, :] = accum_3d[
                    pl.ds(r0, cast_step), :, :
                ].astype(jnp.bfloat16)
        else:
            cast_rc = 16 if (cfg.fast_gs and batch_size >= 128) else min(batch_size, 8)

            def _cast_rs_chunk(ci, _):
                r0 = ci * cast_rc
                rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                    pl.ds(r0, cast_rc), :, :
                ].astype(jnp.bfloat16)
                return _

            lax.fori_loop(0, batch_size // cast_rc, _cast_rs_chunk, jnp.int32(0))
        rs_src_ref = rs_send_bf16
    else:
        rs_src_ref = accum_3d

    if wait_entry_barrier:
        pl.semaphore_wait(barrier_sem, ep)

    if use_pair_rs:
        # 2-stage intra-chip XOR=1 paired Reduce-Scatter: exchanges partner blocks over
        # the 733 GB/s die-to-die link first, cutting off-chip/inter-host ICI sends by >50%!
        num_chips_rs = ep // 2
        core_rs = me & jnp.int32(1)
        chip_rs = me >> 1
        partner_rs = jnp.bitwise_xor(me, jnp.int32(1))

        p1_ops = []
        for c in range(num_chips_rs):
            dest_for_partner = jnp.int32(2 * c) + (jnp.int32(1) - core_rs)
            op_p = pltpu.make_async_remote_copy(
                rs_src_ref.at[
                    pl.ds(dest_for_partner * rows_per_rank, rows_per_rank), :, :
                ],
                rs_pair_recv.at[c],
                rs_pair_ssem.at[c],
                rs_pair_rsem.at[c],
                device_id={ep_axis_name: partner_rs},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_p.start()
            p1_ops.append(op_p)

        chip_ops = []
        for c in range(num_chips_rs):
            pltpu.make_async_copy(
                rs_pair_recv.at[c], rs_pair_recv.at[c], rs_pair_rsem.at[c]
            ).wait()
            target_rank = jnp.int32(2 * c) + core_rs

            @pl.when(c != chip_rs)
            def _sum_and_send_chip(c=c, target_rank=target_rank):
                r_off = target_rank * rows_per_rank
                pair_f32 = accum_3d[
                    pl.ds(r_off, rows_per_rank), :, :
                ] + rs_pair_recv[c].astype(jnp.float32)
                rs_chip_send[c] = pair_f32.astype(rs_chip_send.dtype)

            op_c = pltpu.make_async_remote_copy(
                rs_chip_send.at[c],
                rs_chip_recv.at[chip_rs],
                rs_chip_ssem.at[c],
                rs_chip_rsem,
                device_id={ep_axis_name: target_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            chip_ops.append((c, op_c))

            @pl.when(c != chip_rs)
            def _start_chip_rs(_op=op_c):
                _op.start()

        pltpu.make_async_copy(
            rs_chip_recv.at[pl.ds(0, num_chips_rs - 1)],
            rs_chip_recv.at[pl.ds(0, num_chips_rs - 1)],
            rs_chip_rsem,
        ).wait()

        red_rc = min(rows_per_rank, 4)

        def _reduce_owner_paired(ci, _):
            r0 = ci * red_rc
            acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
            for c in range(num_chips_rs):
                part_c = jnp.where(
                    c == chip_rs,
                    rs_pair_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                    rs_chip_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                )
                acc = acc + part_c
            owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
            return _

        lax.fori_loop(
            0, rows_per_rank // red_rc, _reduce_owner_paired, jnp.int32(0)
        )

        def _finish_rs_sends():
            for op_p in p1_ops:
                op_p.wait_send()
            for c, op_c in chip_ops:

                @pl.when(c != chip_rs)
                def _wc(_op=op_c):
                    _op.wait_send()

    else:
        rs_copies = []
        for d in range(ep):
            dst_slot = jnp.where(me < d, me, jnp.maximum(me - 1, 0))
            copy_op = pltpu.make_async_remote_copy(
                rs_src_ref.at[pl.ds(d * rows_per_rank, rows_per_rank), :, :],
                rs_recv_3d.at[dst_slot],
                rs_send_sem.at[d],
                rs_recv_sem,
                device_id={ep_axis_name: jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            rs_copies.append((d, copy_op))

            @pl.when(me != d)
            def _start_rs(_op=copy_op):
                _op.start()

        pltpu.make_async_copy(rs_recv_3d, rs_recv_3d, rs_recv_sem).wait()

        if cfg.fast_b64_epi_pro and rows_per_rank in (2, 4, 8):
            acc = accum_3d[pl.ds(me * rows_per_rank, rows_per_rank), :, :]
            for src in range(ep - 1):
                acc = acc + rs_recv_3d[src, :, :, :].astype(jnp.float32)
            owner_3d[...] = acc.astype(jnp.bfloat16)
        elif cfg.fast_gs and rows_per_rank <= 16:
            red_rc = min(rows_per_rank, 4)
            for r0 in range(0, rows_per_rank, red_rc):
                acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
                for src in range(ep - 1):
                    acc = acc + rs_recv_3d[src, pl.ds(r0, red_rc), :, :].astype(
                        jnp.float32
                    )
                owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
        else:
            red_rc = min(rows_per_rank, 4)

            def _reduce_owner_chunk(ci, _):
                r0 = ci * red_rc
                acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
                for src in range(ep - 1):
                    acc = acc + rs_recv_3d[src, pl.ds(r0, red_rc), :, :].astype(
                        jnp.float32
                    )
                owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
                return _

            lax.fori_loop(
                0, rows_per_rank // red_rc, _reduce_owner_chunk, jnp.int32(0)
            )

        def _finish_rs_sends():
            for d, copy_op in rs_copies:

                @pl.when(me != d)
                def _wait_rs(_op=copy_op):
                    _op.wait_send()

    if stop_stage == "rs":
        _finish_rs_sends()
        if write_output_ref and output_ref is not None:
            if return_sharded:
                z_rows = min(rows_per_rank, 16)
            else:
                z_rows = 16
            output_ref[pl.ds(0, z_rows), :] = (
                owner_3d[pl.ds(0, min(rows_per_rank, z_rows)), :, :].reshape(
                    -1, _H
                )[:z_rows, :]
                if rows_per_rank >= z_rows
                else jnp.zeros((z_rows, _H), dtype=jnp.bfloat16)
            )
        return

    if return_sharded:
        if write_output_ref and output_ref is not None:
            _write_sharded_owner_out(output_ref, owner_3d, rows_per_rank)
    elif cfg.bridge_ag:
        # 8A. 2-Stage Chip-Hierarchical Bridge All-Gather (halves off-chip ICI!).
        num_chips = ep // 2
        my_chip = me >> 1
        my_par = me & 1
        mate = jnp.bitwise_xor(me, jnp.int32(1))
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]

        mate_op = pltpu.make_async_remote_copy(
            owner_3d,
            fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
            fan_send_sem.at[0],
            fan_recv_sem,
            device_id={ep_axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        mate_op.start()

        s1_ops = []
        for c in range(num_chips):
            dest = jnp.int32(c * 2) + my_par
            op = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
                fan_send_sem.at[1 + c],
                fan_recv_sem,
                device_id={ep_axis_name: dest},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s1_ops.append((c, op))

            @pl.when(my_chip != c)
            def _start_s1(_op=op):
                _op.start()

        s1_recv_rows = num_chips * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, s1_recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, s1_recv_rows), :, :],
            fan_recv_sem,
        ).wait()

        s2_ops = []
        for c in range(num_chips):
            src_rank = jnp.int32(c * 2) + my_par
            r_off = src_rank * rows_per_rank
            op2 = pltpu.make_async_remote_copy(
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_s2_send_sem.at[c],
                fan_s2_recv_sem,
                device_id={ep_axis_name: mate},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s2_ops.append((c, op2))

            @pl.when(my_chip != c)
            def _start_s2(_op=op2):
                _op.start()

        s2_recv_rows = (num_chips - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, s2_recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, s2_recv_rows), :, :],
            fan_s2_recv_sem,
        ).wait()

        if write_output_ref and output_ref is not None:
            for r0 in range(0, batch_size, 16):
                output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)

        mate_op.wait_send()
        for c, op in s1_ops:

            @pl.when(my_chip != c)
            def _wait_s1(_op=op):
                _op.wait_send()

        for c, op2 in s2_ops:

            @pl.when(my_chip != c)
            def _wait_s2(_op=op2):
                _op.wait_send()
    else:
        # 8B. Flat (B, 32, 128) All-Gather fan-out + static whole-width output write.
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]
        fan_copies = []
        for dest in range(ep):
            fan_op = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
                fan_send_sem.at[dest],
                fan_recv_sem,
                device_id={ep_axis_name: jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            fan_copies.append((dest, fan_op))

            @pl.when(me != dest)
            def _start_fan(_op=fan_op):
                _op.start()

        recv_rows = (ep - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_recv_sem,
        ).wait()

        if write_output_ref and output_ref is not None:
            for r0 in range(0, batch_size, 16):
                output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)

        for dest, fan_op in fan_copies:

            @pl.when(me != dest)
            def _wait_fan(_op=fan_op):
                _op.wait_send()

    _finish_rs_sends()


def _run_muse_sparse_from_vmem_core(
    me,
    logits_ref,
    bias_ref,
    x_in_ref,
    w1_u32,
    s1_ref,
    w2_u32,
    s2_ref,
    norm_ref,
    *,
    d_scratch,
    r_scratch,
    w1_buf,
    s1_buf,
    w2_buf,
    s2_buf,
    w1_sem,
    w2_sem,
    misc_sem,
    w1_acc_buf,
    x_3d_buf,
    io_3d_buf,
    lhs_2d_buf,
    act_2d_buf,
    y_stage_buf,
    lhs_fp8_buf,
    lhs_sc_buf,
    masked_x_buf,
    bd_sc_buf,
    gain_f32_buf,
    accum_3d,
    accum_2d,
    fan_3d_buf,
    rs_send_bf16,
    rs_pair_recv,
    rs_chip_send,
    rs_chip_recv,
    rs_pair_ssem,
    rs_pair_rsem,
    rs_chip_ssem,
    rs_chip_rsem,
    rs_recv_3d,
    rs_send_sem,
    rs_recv_sem,
    owner_3d,
    fan_send_sem,
    fan_recv_sem,
    fan_s2_send_sem,
    fan_s2_recv_sem,
    barrier_sem,
    output_ref,
    ep: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    return_sharded: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    cfg: SparseDecodeConfig,
    rep_w1_u32=None,
    rep_s1_ref=None,
    rep_w2_u32=None,
    rep_s2_ref=None,
    replica_meta_ref=None,
    owner_load_smem=None,
    moved_flag_smem=None,
    local_suppress_smem=None,
    holder_slot_expert_smem=None,
    holder_cand_smem=None,
    spec_slot_ref=None,
    x_is_3d: bool = False,
    wait_entry_barrier: bool = True,
    write_output_ref: bool = True,
    on_experts_done_fn=None,
    stop_stage: str = "",
    early_exit_fn=None,
) -> None:
    """Execute complete VMEM-resident Muse sparse decode (Phase 1 plan/prime + Phase 2 experts/RS)."""
    plan_res = plan_and_prime_muse_sparse_from_vmem(
        me,
        logits_ref,
        bias_ref,
        norm_ref,
        w1_u32,
        s1_ref,
        w2_u32,
        s2_ref,
        d_scratch=d_scratch,
        r_scratch=r_scratch,
        w1_buf=w1_buf,
        s1_buf=s1_buf,
        w2_buf=w2_buf,
        s2_buf=s2_buf,
        w1_sem=w1_sem,
        w2_sem=w2_sem,
        misc_sem=misc_sem,
        gain_f32_buf=gain_f32_buf,
        accum_3d=accum_3d,
        io_3d_buf=io_3d_buf,
        accum_2d=accum_2d,
        rep_w1_u32=rep_w1_u32,
        rep_s1_ref=rep_s1_ref,
        rep_w2_u32=rep_w2_u32,
        rep_s2_ref=rep_s2_ref,
        replica_meta_ref=replica_meta_ref,
        owner_load_smem=owner_load_smem,
        moved_flag_smem=moved_flag_smem,
        local_suppress_smem=local_suppress_smem,
        holder_slot_expert_smem=holder_slot_expert_smem,
        holder_cand_smem=holder_cand_smem,
        spec_slot_ref=spec_slot_ref,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        cfg=cfg,
        stop_stage=stop_stage,
        early_exit_fn=early_exit_fn,
        zero_and_pack=True,
        x_in_ref=None if x_is_3d else x_in_ref,
        x_3d_buf=None if x_is_3d else x_3d_buf,
    )
    if plan_res is None:
        return
    run_muse_sparse_experts_and_rs_from_vmem(
        me,
        x_in_ref,
        plan_res,
        d_scratch=d_scratch,
        r_scratch=r_scratch,
        w1_buf=w1_buf,
        s1_buf=s1_buf,
        w2_buf=w2_buf,
        s2_buf=s2_buf,
        w1_acc_buf=w1_acc_buf,
        x_3d_buf=x_3d_buf,
        io_3d_buf=io_3d_buf,
        lhs_2d_buf=lhs_2d_buf,
        act_2d_buf=act_2d_buf,
        y_stage_buf=y_stage_buf,
        lhs_fp8_buf=lhs_fp8_buf,
        lhs_sc_buf=lhs_sc_buf,
        masked_x_buf=masked_x_buf,
        bd_sc_buf=bd_sc_buf,
        gain_f32_buf=gain_f32_buf,
        accum_3d=accum_3d,
        accum_2d=accum_2d,
        fan_3d_buf=fan_3d_buf,
        rs_send_bf16=rs_send_bf16,
        rs_pair_recv=rs_pair_recv,
        rs_chip_send=rs_chip_send,
        rs_chip_recv=rs_chip_recv,
        rs_pair_ssem=rs_pair_ssem,
        rs_pair_rsem=rs_pair_rsem,
        rs_chip_ssem=rs_chip_ssem,
        rs_chip_rsem=rs_chip_rsem,
        rs_recv_3d=rs_recv_3d,
        rs_send_sem=rs_send_sem,
        rs_recv_sem=rs_recv_sem,
        owner_3d=owner_3d,
        fan_send_sem=fan_send_sem,
        fan_recv_sem=fan_recv_sem,
        fan_s2_send_sem=fan_s2_send_sem,
        fan_s2_recv_sem=fan_s2_recv_sem,
        barrier_sem=barrier_sem,
        output_ref=output_ref,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        return_sharded=return_sharded,
        ep_axis_name=ep_axis_name,
        cfg=cfg,
        x_is_3d=True if not plan_res.is_dense else x_is_3d,
        wait_entry_barrier=wait_entry_barrier,
        write_output_ref=write_output_ref,
        on_experts_done_fn=on_experts_done_fn,
        stop_stage=stop_stage,
        early_exit_fn=early_exit_fn,
    )


def _sparse_decode_kernel(
    *all_refs,
    ep: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    return_sharded: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    cfg: SparseDecodeConfig,
    has_spec_slot_arg: bool = False,
):
    has_replicas = num_replicas > 0
    is_k256 = scale_block >= 256
    is_dense = cfg.row_mode == "dense"
    w_mode = (
        "stream3"
        if (cfg.weight_mode == "stream3_k2" and (_H // scale_block) < 16)
        else cfg.weight_mode
    )
    use_bd = (not is_dense) and (not is_k256) and bool(cfg.k16_blockdiag_rows)
    e_local = _E_TOTAL // ep
    rows_per_rank = batch_size // ep

    it = iter(all_refs)
    rank_ref = next(it)
    replica_meta_ref = next(it) if has_replicas else None
    spec_slot_ref = next(it) if has_spec_slot_arg else None
    logits_ref = next(it)
    bias_ref = next(it)
    x_in_ref = next(it)
    w1_ref = next(it)
    s1_ref = next(it)
    w2_ref = next(it)
    s2_ref = next(it)
    norm_ref = next(it)
    if has_replicas:
        rep_w1_ref = next(it)
        rep_s1_ref = next(it)
        rep_w2_ref = next(it)
        rep_s2_ref = next(it)
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None
    output_ref = next(it)

    if is_dense:
        d_scratch = DenseRoutingScratch(
            *(next(it) for _ in range(len(DenseRoutingScratch._fields)))
        )
        r_scratch = None
    else:
        r_scratch = RoutingScratch(
            *(next(it) for _ in range(len(RoutingScratch._fields)))
        )
        d_scratch = None

    if has_replicas:
        owner_load_smem = next(it)
        moved_flag_smem = next(it)
        local_suppress_smem = next(it)
        holder_slot_expert_smem = next(it)
        holder_cand_smem = (
            next(it) if (not is_dense and not cfg.static_holder_mask) else None
        )
    else:
        owner_load_smem = moved_flag_smem = None
        local_suppress_smem = holder_slot_expert_smem = holder_cand_smem = None

    w1_buf = next(it)
    s1_buf = next(it)
    w2_buf = next(it)
    s2_buf = next(it)
    w1_sem = next(it)
    w2_sem = next(it)
    misc_sem = next(it)
    w1_acc_buf = next(it) if w_mode == "stream3_k2" else None

    use_lhs_fp8 = is_k256 and cfg.lhs_fp8
    use_bd_fp8 = use_bd and cfg.k16_bd_fp8
    if is_dense:
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        gain_f32_buf = next(it)
        accum_2d = next(it)
        accum_3d = next(it)
        fan_3d_buf = next(it) if not return_sharded else None
        x_3d_buf = io_3d_buf = lhs_2d_buf = masked_x_buf = bd_sc_buf = None
    else:
        x_3d_buf = next(it)
        io_3d_buf = next(it)
        lhs_2d_buf = next(it)
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        masked_x_buf = next(it) if use_bd else None
        bd_sc_buf = next(it) if use_bd_fp8 else None
        gain_f32_buf = next(it)
        accum_3d = next(it)
        accum_2d = None
        fan_3d_buf = x_3d_buf if not return_sharded else None

    use_pair_rs = cfg.pair_rs and (ep in (8, 16))
    rs_send_bf16 = next(it) if cfg.rs_wire_bf16 else None
    if use_pair_rs:
        rs_pair_recv = next(it)
        rs_chip_send = next(it)
        rs_chip_recv = next(it)
        rs_pair_ssem = next(it)
        rs_pair_rsem = next(it)
        rs_chip_ssem = next(it)
        rs_chip_rsem = next(it)
        rs_recv_3d = rs_send_sem = rs_recv_sem = None
    else:
        rs_recv_3d = next(it)
        rs_send_sem = next(it)
        rs_recv_sem = next(it)
        rs_pair_recv = rs_chip_send = rs_chip_recv = None
        rs_pair_ssem = rs_pair_rsem = rs_chip_ssem = rs_chip_rsem = None
    owner_3d = next(it)
    if not return_sharded:
        fan_send_sem = next(it)
        fan_recv_sem = next(it)
        fan_s2_send_sem = next(it) if cfg.bridge_ag else None
        fan_s2_recv_sem = next(it) if cfg.bridge_ag else None
    else:
        fan_send_sem = fan_recv_sem = fan_s2_send_sem = fan_s2_recv_sem = None

    barrier_sem = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]
    w1_u32 = w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref.bitcast(jnp.uint32)
    rep_w1_u32 = rep_w1_ref.bitcast(jnp.uint32) if has_replicas else None
    rep_w2_u32 = rep_w2_ref.bitcast(jnp.uint32) if has_replicas else None

    def _early_exit_finish() -> None:
        pl.semaphore_wait(barrier_sem, ep)
        z_rows = min(rows_per_rank, 16) if return_sharded else 16
        output_ref[pl.ds(0, z_rows), :] = jnp.zeros(
            (z_rows, _H), dtype=jnp.bfloat16
        )

    _run_muse_sparse_from_vmem_core(
        me,
        logits_ref,
        bias_ref,
        x_in_ref,
        w1_u32,
        s1_ref,
        w2_u32,
        s2_ref,
        norm_ref,
        d_scratch=d_scratch,
        r_scratch=r_scratch,
        w1_buf=w1_buf,
        s1_buf=s1_buf,
        w2_buf=w2_buf,
        s2_buf=s2_buf,
        w1_sem=w1_sem,
        w2_sem=w2_sem,
        misc_sem=misc_sem,
        w1_acc_buf=w1_acc_buf,
        x_3d_buf=x_3d_buf,
        io_3d_buf=io_3d_buf,
        lhs_2d_buf=lhs_2d_buf,
        act_2d_buf=act_2d_buf,
        y_stage_buf=y_stage_buf,
        lhs_fp8_buf=lhs_fp8_buf,
        lhs_sc_buf=lhs_sc_buf,
        masked_x_buf=masked_x_buf,
        bd_sc_buf=bd_sc_buf,
        gain_f32_buf=gain_f32_buf,
        accum_3d=accum_3d,
        accum_2d=accum_2d,
        fan_3d_buf=fan_3d_buf,
        rs_send_bf16=rs_send_bf16,
        rs_pair_recv=rs_pair_recv,
        rs_chip_send=rs_chip_send,
        rs_chip_recv=rs_chip_recv,
        rs_pair_ssem=rs_pair_ssem,
        rs_pair_rsem=rs_pair_rsem,
        rs_chip_ssem=rs_chip_ssem,
        rs_chip_rsem=rs_chip_rsem,
        rs_recv_3d=rs_recv_3d,
        rs_send_sem=rs_send_sem,
        rs_recv_sem=rs_recv_sem,
        owner_3d=owner_3d,
        fan_send_sem=fan_send_sem,
        fan_recv_sem=fan_recv_sem,
        fan_s2_send_sem=fan_s2_send_sem,
        fan_s2_recv_sem=fan_s2_recv_sem,
        barrier_sem=barrier_sem,
        output_ref=output_ref,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        cfg=cfg,
        rep_w1_u32=rep_w1_u32,
        rep_s1_ref=rep_s1_ref,
        rep_w2_u32=rep_w2_u32,
        rep_s2_ref=rep_s2_ref,
        replica_meta_ref=replica_meta_ref,
        owner_load_smem=owner_load_smem,
        moved_flag_smem=moved_flag_smem,
        local_suppress_smem=local_suppress_smem,
        holder_slot_expert_smem=holder_slot_expert_smem,
        holder_cand_smem=holder_cand_smem,
        spec_slot_ref=spec_slot_ref,
        x_is_3d=False,
        wait_entry_barrier=True,
        write_output_ref=True,
        on_experts_done_fn=None,
        stop_stage=cfg.debug_stop_after,
        early_exit_fn=_early_exit_finish,
    )


def build_muse_sparse_scratch_shapes(
    *,
    ep: int,
    batch_size: int,
    scale_block: int,
    s1_dtype,
    s2_dtype,
    num_replicas: int,
    return_sharded: bool,
    config: SparseDecodeConfig,
) -> list:
    """Build exact Pallas scratch shapes matching ``_sparse_decode_kernel``."""
    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem = pltpu.SemaphoreType.DMA
    e_local = _E_TOTAL // ep
    has_replicas = num_replicas > 0
    is_dense = config.row_mode == "dense"
    max_exec = e_local + num_replicas
    scratch_shapes: list = []
    if is_dense:
        scratch_shapes.extend(
            dense_routing_scratch_shapes(
                batch_size, max_exec, num_experts=_E_TOTAL
            )
        )
    else:
        scratch_shapes.extend(
            routing_scratch_shapes(
                batch_size, max_exec, topk=_TOPK, num_experts=_E_TOTAL
            )
        )

    if has_replicas:
        moved_smem_len = 8 if (not is_dense and config.static_holder_mask) else _E_TOTAL
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((moved_smem_len,), jnp.int32),
                smem((e_local,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )
        if not is_dense and not config.static_holder_mask:
            scratch_shapes.append(smem((num_replicas,), jnp.int32))

    nqb = _H // scale_block
    w_mode = (
        "stream3"
        if (config.weight_mode == "stream3_k2" and nqb < 16)
        else config.weight_mode
    )
    tile_m = config.tile_m
    if w_mode == "stream3":
        w1_buf_shape = vmem((2, _H // _FP4_PACK, _I), jnp.uint32)
        s1_buf_shape = vmem((2, nqb, _I), s1_dtype)
        w2_buf_shape = vmem((_I // _FP4_PACK, _H), jnp.uint32)
        s2_buf_shape = vmem((nqb, _H), s2_dtype)
        w1_sem_shape = sem((2,))
        w2_sem_shape = sem((1,))
    elif w_mode == "stream3_k2":
        w1_buf_shape = vmem((2, (_H // 2) // _FP4_PACK, _N1), jnp.uint32)
        s1_buf_shape = vmem((2, nqb // 2, _N1), s1_dtype)
        w2_buf_shape = vmem((_I // _FP4_PACK, _H), jnp.uint32)
        s2_buf_shape = vmem((nqb, _H), s2_dtype)
        w1_sem_shape = sem((2,))
        w2_sem_shape = sem((1,))
    elif w_mode == "w1_2x_w2_2x":
        w1_buf_shape = vmem((2, _H // _FP4_PACK, _N1), jnp.uint32)
        s1_buf_shape = vmem((2, nqb, _N1), s1_dtype)
        w2_buf_shape = vmem((2, _I // _FP4_PACK, _H), jnp.uint32)
        s2_buf_shape = vmem((2, nqb, _H), s2_dtype)
        w1_sem_shape = sem((2,))
        w2_sem_shape = sem((2,))
    elif w_mode == "w1_2x_w2_1x":
        w1_buf_shape = vmem((2, _H // _FP4_PACK, _N1), jnp.uint32)
        s1_buf_shape = vmem((2, nqb, _N1), s1_dtype)
        w2_buf_shape = vmem((_I // _FP4_PACK, _H), jnp.uint32)
        s2_buf_shape = vmem((nqb, _H), s2_dtype)
        w1_sem_shape = sem((2,))
        w2_sem_shape = sem((1,))
    else:
        w1_buf_shape = vmem((2, _H // _FP4_PACK, _I), jnp.uint32)
        s1_buf_shape = vmem((2, nqb, _I), s1_dtype)
        w2_buf_shape = vmem((2, _I // _FP4_PACK, _H // 2), jnp.uint32)
        s2_buf_shape = vmem((2, nqb, _H // 2), s2_dtype)
        w1_sem_shape = sem((2,))
        w2_sem_shape = sem((2,))

    scratch_shapes.extend(
        [
            w1_buf_shape,
            s1_buf_shape,
            w2_buf_shape,
            s2_buf_shape,
            w1_sem_shape,
            w2_sem_shape,
            sem((1,)),
        ]
    )
    if w_mode == "stream3_k2":
        scratch_shapes.append(
            vmem((batch_size if is_dense else tile_m, _N1), jnp.float32)
        )

    use_bd = (
        (not is_dense)
        and (scale_block == 16)
        and (w_mode != "stream3_k2")
        and bool(config.k16_blockdiag_rows)
    )
    use_lhs_fp8 = (scale_block >= 256) and config.lhs_fp8
    y_dtype = jnp.float32 if config.norm_f32_stage else jnp.bfloat16
    y_stage_rows = (
        max(16, max(config.m_buckets))
        if (config.loop32_in_tile and not use_lhs_fp8)
        else tile_m
    )

    if is_dense:
        scratch_shapes.extend(
            [
                vmem((batch_size, _I), jnp.bfloat16),
                vmem((batch_size, _H), y_dtype),
            ]
        )
        if use_lhs_fp8:
            scratch_shapes.extend(
                [
                    vmem((max(32, batch_size), _H), jnp.float8_e4m3fn),
                    vmem((batch_size, (_H // _KC) * 128), jnp.float32),
                ]
            )
        scratch_shapes.extend(
            [
                vmem((1, _H), jnp.float32),
                vmem((batch_size, _H), jnp.float32),
                vmem((batch_size, 32, 128), jnp.float32),
            ]
        )
        if not return_sharded:
            scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    else:
        scratch_shapes.extend(
            [
                vmem((batch_size, 32, 128), jnp.bfloat16),
                vmem((tile_m, 32, 128), jnp.bfloat16),
                vmem((tile_m, _H), jnp.bfloat16),
                vmem((tile_m, _I), jnp.bfloat16),
                vmem((y_stage_rows, _H), y_dtype),
            ]
        )
        if use_lhs_fp8:
            scratch_shapes.extend(
                [
                    vmem((max(32, tile_m), _H), jnp.float8_e4m3fn),
                    vmem((tile_m, (_H // _KC) * 128), jnp.float32),
                ]
            )
        if use_bd:
            max_bd = max(config.k16_blockdiag_rows)
            mx_dtype = jnp.float8_e4m3fn if config.k16_bd_fp8 else jnp.bfloat16
            mx_rows = (
                max(32, max_bd * 16) if config.k16_bd_fp8 else (max_bd * 16)
            )
            scratch_shapes.append(vmem((mx_rows, _H), mx_dtype))
            if config.k16_bd_fp8:
                scratch_shapes.append(
                    vmem((max_bd * 16, (_H // _KC) * 128), jnp.float32)
                )
        accum_rows = (batch_size + 1) if config.fast_gs else batch_size
        scratch_shapes.extend(
            [
                vmem((1, _H), jnp.float32),
                vmem((accum_rows, 32, 128), jnp.float32),
            ]
        )

    rows_per_rank = batch_size // ep
    use_pair_rs = config.pair_rs and (ep in (8, 16))
    rs_wire_dtype = jnp.bfloat16 if config.rs_wire_bf16 else jnp.float32
    if config.rs_wire_bf16:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    if use_pair_rs:
        num_chips_rs = ep // 2
        scratch_shapes.extend(
            [
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((num_chips_rs,)),
                sem((num_chips_rs,)),
                sem((num_chips_rs,)),
                sem,
                vmem((rows_per_rank, 32, 128), jnp.bfloat16),
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((ep - 1, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((ep,)),
                sem,
                vmem((rows_per_rank, 32, 128), jnp.bfloat16),
            ]
        )
    if not return_sharded:
        if config.bridge_ag:
            num_chips = ep // 2
            scratch_shapes.extend([sem((1 + num_chips,)), sem, sem((num_chips,)), sem])
        else:
            scratch_shapes.extend([sem((ep,)), sem])
    return scratch_shapes


def fused_decode_sparse_from_logits(
    rank: jax.Array,
    logits: jax.Array,
    bias: jax.Array,
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    norm: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    spec_home_slot: jax.Array | None = None,
    *,
    ep: int = 8,
    return_sharded: bool = False,
    replica_cost_mode: str = "active",
    ep_axis_name: str = "model",
    config: SparseDecodeConfig | None = None,
) -> jax.Array:
    """Sparse / Dense-Lowtok Fuse-All Decode MoE from pre-scaled FP32 router logits."""
    if ep not in (8, 16):
        raise ValueError(f"Sparse fused decode supports ep in (8, 16), got {ep}")
    batch_size = x.shape[0]
    if batch_size not in (16, 32, 64, 128, 256, 384):
        raise ValueError(
            f"Sparse fused decode requires B in (16, 32, 64, 128, 256, 384), got {batch_size}"
        )
    if logits.shape != (batch_size, _E_TOTAL):
        raise ValueError(
            f"Expected logits shape ({batch_size}, {_E_TOTAL}), got {logits.shape}"
        )
    e_local = _E_TOTAL // ep
    w_groups = int(w1.shape[0])
    # Allow prefill zero-glue EPLB slabs `[e_local + R_pre, ...]` (`0 <= R_pre <= 8`)
    # directly with zero HBM slice/copy: decode only indexes base slots `[0, e_local)`.
    if (
        not (e_local <= w_groups <= e_local + 8)
        or w1.shape != (w_groups, _H, _N1)
        or w2.shape != (w_groups, _I, _H)
    ):
        raise ValueError(
            f"Unexpected FP4 weight shapes: w1={w1.shape}, w2={w2.shape} for ep={ep}"
        )
    if w1.dtype != jnp.float4_e2m1fn or w2.dtype != jnp.float4_e2m1fn:
        raise ValueError(
            f"Expected float4_e2m1fn weights, got w1={w1.dtype}, w2={w2.dtype}"
        )
    scale_block = _H // s1.shape[1]
    if scale_block not in (16, 256, 512):
        raise ValueError(
            f"Expected K16, K256, or K512 scales, got s1.shape={s1.shape}"
        )
    if s1.shape != (w_groups, _H // scale_block, _N1) or s2.shape != (
        w_groups,
        _I // scale_block,
        _H,
    ):
        raise ValueError(
            f"Mismatched scale shapes: s1={s1.shape}, s2={s2.shape} for block={scale_block}"
        )

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    if has_replicas:
        if any(a is None for a in rep_args):
            raise ValueError(
                "All of rep_w1, rep_s1, rep_w2, rep_s2, replica_meta must be provided together"
            )
        num_replicas = int(rep_w1.shape[0])
        if (
            rep_w1.shape != (num_replicas, _H, _N1)
            or rep_w2.shape != (num_replicas, _I, _H)
            or rep_s1.shape != (num_replicas, _H // scale_block, _N1)
            or rep_s2.shape != (num_replicas, _I // scale_block, _H)
            or replica_meta.shape[0] != 4
        ):
            raise ValueError(
                f"Invalid replica bank shapes: rep_w1={rep_w1.shape}, meta={replica_meta.shape}"
            )
    else:
        num_replicas = 0

    if config is None:
        config = default_sparse_decode_config(batch_size, ep, scale_block)

    bias_2d = bias.reshape(1, _E_TOTAL)
    norm_2d = norm.reshape(1, _H)
    x_2d = x.reshape(batch_size, _H) if x.ndim == 3 else x

    v_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    tile_m = config.tile_m
    rows_per_rank = batch_size // ep
    scratch_shapes = build_muse_sparse_scratch_shapes(
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        s1_dtype=s1.dtype,
        s2_dtype=s2.dtype,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        config=config,
    )

    base_in_specs = (
        v_spec,  # logits
        v_spec,  # bias
        v_spec,  # x
        h_spec,  # w1
        h_spec,  # s1
        h_spec,  # w2
        h_spec,  # s2
        v_spec,  # norm
    )
    has_spec_slot_arg = spec_home_slot is not None
    if has_spec_slot_arg:
        if spec_home_slot.ndim > 1:
            raise ValueError(
                f"Expected 0D or 1D spec_home_slot, got shape {spec_home_slot.shape}"
            )
        spec_slot_arg = jnp.atleast_1d(spec_home_slot).astype(jnp.int32)
        if spec_slot_arg.shape[0] < 1:
            raise ValueError(
                f"Expected non-empty spec_home_slot, got shape {spec_home_slot.shape}"
            )
    else:
        spec_slot_arg = None

    scalar_args = [rank]
    if has_replicas:
        scalar_args.append(replica_meta)
    if has_spec_slot_arg:
        scalar_args.append(spec_slot_arg)
    num_scalar_prefetch = len(scalar_args)

    if has_replicas:
        in_specs = base_in_specs + (h_spec, h_spec, h_spec, h_spec)
        call_args = (
            *scalar_args,
            logits,
            bias_2d,
            x_2d,
            w1,
            s1,
            w2,
            s2,
            norm_2d,
            rep_w1,
            rep_s1,
            rep_w2,
            rep_s2,
        )
    else:
        in_specs = base_in_specs
        call_args = (*scalar_args, logits, bias_2d, x_2d, w1, s1, w2, s2, norm_2d)

    out_rows = rows_per_rank if return_sharded else batch_size
    out_shape = jax.ShapeDtypeStruct((out_rows, _H), jnp.bfloat16)

    kernel_fn = functools.partial(
        _sparse_decode_kernel,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        cfg=config,
        has_spec_slot_arg=has_spec_slot_arg,
    )
    rep_tag = f"_r{num_replicas}" if has_replicas else ""
    out_tag = "_sh" if return_sharded else "_rep"
    w9_tag = (
        f"{'_sm1' if config.static_holder_mask else ''}"
        f"{f'_sp{config.spec_prime_parts}' if config.spec_prime_parts else ''}"
        f"{'_rf1' if config.route_fast else ''}"
        f"{f'_fp{config.fast_plan}' if config.fast_plan else ''}"
        f"{'_us1' if config.unroll_strips else ''}"
        f"{'_fgs1' if config.fast_gs else ''}"
    )
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            grid=(1,),
            in_specs=in_specs,
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=(
            f"muse_sparse_ep{ep}_b{batch_size}_k{scale_block}"
            f"_{config.row_mode}_{config.weight_mode}_tm{tile_m}{rep_tag}{w9_tag}{out_tag}"
        ),
    )(*call_args)


@dataclasses.dataclass(frozen=True)
class Fp8SparseDecodeConfig:
    """Static compile-time configuration for ``fused_decode_sparse_fp8_from_logits``."""

    row_mode: str = "sparse"  # "sparse" or "dense" (for B=16..64)
    weight_mode: str = "stream3_half"  # "stream3_half" (6x8 MiB), "w1_k2_w2_1x" (3x16 MiB K-contig), or "ring3_8m"
    tile_m: int = 32
    m_buckets: tuple[int, ...] = (16, 32)
    fp8_kc: int = 512
    fp8_strip_mode: str = "wide"
    fp8_dual_mode: str = "hybrid"  # "vstage", "dual", "hybrid" (dual at m<=16 w/ stream3_half)
    lhs_fp8: bool = False  # W8A8: per-row FP8 LHS quantization for 2x MXU
    rs_wire_bf16: bool = True
    pair_rs: bool = False
    bridge_ag: bool = True
    norm_f32_stage: bool = True
    route_chunk_rows: int = 32
    static_holder_mask: bool = True
    spec_prime_parts: int = 0  # 0 = off; 1/2/3 chunks primed before routing
    spec_home_slots: tuple[int, ...] = ()
    spec_skip_replicated: bool = True
    fast_b64_epi_pro: bool = False
    min_p_topk: float = 0.0
    free_max_rows: int = 0
    free_max_w: float = 0.06
    free_min_slot: int = 5
    route_rb16_mode: int = 0
    route_fast: bool = False
    fast_plan: int = 0


def default_fp8_sparse_decode_config(
    batch_size: int, ep: int
) -> Fp8SparseDecodeConfig:
    """Return the tuned default FP8 configuration for ``L0`` / ``L61``."""
    (
        _,
        _,
        spec_parts,
        fast_b64,
        fp8_mode,
        decode_m8,
        _,
        contig_parts,
        min_p_topk,
        free_rows,
        free_max_w,
        free_min_slot,
        pair_rs_ov,
        _,
    ) = _read_sparse_env_overrides()
    _, _, _, route_rb16_mode, route_fast, fast_plan = _read_sparse_w9_sub_knobs()
    spec_prime_fast, spec_home_slot = _read_spec_prime_fast_knobs()
    if route_fast and not spec_prime_fast:
        spec_parts = 0
    spec_home_slots = (
        (spec_home_slot,) * ep
        if (spec_parts > 0 and 0 <= spec_home_slot < (_E_TOTAL // ep))
        else ()
    )
    use_k2 = fp8_mode in ("w1_k2_w2_1x", "w1_k2_w8a8", "dense_k2") or contig_parts
    use_w8a8 = fp8_mode == "w1_k2_w8a8"
    use_dense_k2 = fp8_mode == "dense_k2" and batch_size <= 64
    fp8_static_mask = (
        (ep <= 8)
        if _env_bool_fallback(
            "SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_NO_EP16_STATIC_MASK", False
        )
        else True
    )
    fp8_pair_rs = (
        (pair_rs_ov > 0)
        and (ep in (8, 16))
        and (batch_size >= 32 or pair_rs_ov >= 2)
    )
    fp8_bridge_ag = (ep == 8) or (fp8_pair_rs and ep == 16)
    min_p_eff, free_rows_eff, free_max_w_eff, free_min_slot_eff = (
        resolve_free_and_min_p_for_batch(
            batch_size,
            min_p_topk=min_p_topk,
            free_rows=free_rows,
            free_max_w=free_max_w,
            free_min_slot=free_min_slot,
            parse_free_spec_fn=parse_free_expert_rows_spec,
        )
    )
    route_kw = dict(
        min_p_topk=min_p_eff,
        free_max_rows=free_rows_eff,
        free_max_w=free_max_w_eff,
        free_min_slot=free_min_slot_eff,
        route_rb16_mode=route_rb16_mode,
        route_fast=route_fast,
        fast_plan=fast_plan,
        spec_home_slots=spec_home_slots,
        pair_rs=fp8_pair_rs,
    )
    if batch_size <= 16 or use_dense_k2:
        return Fp8SparseDecodeConfig(
            row_mode="dense",
            weight_mode="w1_k2_w2_1x" if use_k2 else "stream3_half",
            tile_m=batch_size,
            m_buckets=(batch_size,),
            fp8_kc=512,
            fp8_strip_mode="wide",
            fp8_dual_mode="hybrid",
            lhs_fp8=use_w8a8,
            rs_wire_bf16=(ep == 8),
            bridge_ag=fp8_bridge_ag,
            norm_f32_stage=True,
            route_chunk_rows=batch_size,
            static_holder_mask=fp8_static_mask,
            spec_prime_parts=spec_parts,
            fast_b64_epi_pro=(fast_b64 >= 1 or use_k2),
            **route_kw,
        )
    if batch_size <= 64:
        m_bkts = (
            (8, 16, 32)
            if (fast_b64 >= 1 or fp8_mode in ("w1_k2_w2_1x", "w1_k2_w8a8"))
            else ((8, 32) if decode_m8 else (16, 32))
        )
        return Fp8SparseDecodeConfig(
            row_mode="sparse",
            weight_mode="w1_k2_w2_1x" if use_k2 else "stream3_half",
            tile_m=32,
            m_buckets=m_bkts,
            fp8_kc=512,
            fp8_strip_mode="wide",
            fp8_dual_mode="hybrid",
            lhs_fp8=use_w8a8,
            rs_wire_bf16=(ep == 8),
            bridge_ag=fp8_bridge_ag,
            norm_f32_stage=True,
            route_chunk_rows=batch_size if (fast_b64 >= 1 or use_k2) else 32,
            static_holder_mask=fp8_static_mask,
            spec_prime_parts=spec_parts,
            fast_b64_epi_pro=(fast_b64 >= 1 or use_k2),
            **route_kw,
        )
    if batch_size <= 128:
        return Fp8SparseDecodeConfig(
            row_mode="sparse",
            weight_mode="w1_k2_w2_1x" if use_k2 else "stream3_half",
            tile_m=32 if use_k2 else 64,
            m_buckets=(16, 32) if use_k2 else (16, 32, 64),
            fp8_kc=512,
            fp8_strip_mode="wide",
            fp8_dual_mode="hybrid",
            lhs_fp8=use_w8a8,
            rs_wire_bf16=True,
            bridge_ag=fp8_bridge_ag,
            norm_f32_stage=True,
            route_chunk_rows=32,
            static_holder_mask=fp8_static_mask,
            spec_prime_parts=spec_parts,
            **route_kw,
        )
    return Fp8SparseDecodeConfig(
        row_mode="sparse",
        weight_mode="ring3_8m",
        tile_m=64,
        m_buckets=(16, 32, 64),
        fp8_kc=512,
        fp8_strip_mode="wide",
        fp8_dual_mode="vstage",
        rs_wire_bf16=True,
        bridge_ag=fp8_bridge_ag,
        norm_f32_stage=True,
        route_chunk_rows=32,
        static_holder_mask=fp8_static_mask,
        spec_prime_parts=0,
        **route_kw,
    )


def _normalize_fp8_scale_2d(
    scale: jax.Array, expected_rows: int, out_dim: int, *, pad_rows_to_8: bool = False
) -> jax.Array:
    """Reshape channelwise FP8 scale ``(E, [1, [1,]] N)`` -> ``(E_padded, N)`` for VMEM."""
    if scale.ndim == 4:
        if scale.shape != (expected_rows, 1, 1, out_dim):
            raise ValueError(
                f"Expected 4D FP8 scale ({expected_rows}, 1, 1, {out_dim}), got {scale.shape}"
            )
        s2d = scale.reshape(expected_rows, out_dim)
    elif scale.ndim == 3:
        if scale.shape != (expected_rows, 1, out_dim):
            raise ValueError(
                f"Expected 3D FP8 scale ({expected_rows}, 1, {out_dim}), got {scale.shape}"
            )
        s2d = scale.reshape(expected_rows, out_dim)
    elif scale.ndim == 2:
        if scale.shape != (expected_rows, out_dim):
            raise ValueError(
                f"Expected 2D FP8 scale ({expected_rows}, {out_dim}), got {scale.shape}"
            )
        s2d = scale
    else:
        raise ValueError(f"Unsupported FP8 scale rank {scale.ndim}: shape={scale.shape}")
    if pad_rows_to_8 and expected_rows < 8:
        s2d = _pad_rows_to(s2d, 8)
    return s2d


def _fp8_gmm_chunked(
    x_2d_ref,
    w_2d_ref,
    m_rows: int,
    k_dim: int,
    n0: jax.Array | int,
    nc: int,
    fp8_kc: int,
) -> jax.Array:
    """Compute unscaled ``x[0:m_rows, 0:k_dim] @ w[:, n0:n0+nc]`` in FP32 on MXU."""
    read_m = 16 if m_rows == 8 else m_rows
    acc = None
    for c0 in range(0, k_dim, fp8_kc):
        x = x_2d_ref[pl.ds(0, read_m), pl.ds(c0, fp8_kc)]
        if read_m != m_rows:
            x = x[:m_rows, :]
        w = w_2d_ref[pl.ds(c0, fp8_kc), pl.ds(n0, nc)]
        res = jnp.dot(x, w, preferred_element_type=jnp.float32)
        acc = res if acc is None else acc + res
    return acc


def _fp8_dual_gmm_chunked(
    x_2d_ref,
    wg_2d_ref,
    wu_2d_ref,
    m_rows: int,
    k_dim: int,
    n0: jax.Array | int,
    nc: int,
    fp8_kc: int,
) -> tuple[jax.Array, jax.Array]:
    """Compute unscaled dual ``(x @ wg, x @ wu)`` reading ``x`` once per ``fp8_kc``."""
    read_m = 16 if m_rows == 8 else m_rows
    acc_g = None
    acc_u = None
    for c0 in range(0, k_dim, fp8_kc):
        x = x_2d_ref[pl.ds(0, read_m), pl.ds(c0, fp8_kc)]
        if read_m != m_rows:
            x = x[:m_rows, :]
        wg = wg_2d_ref[pl.ds(c0, fp8_kc), pl.ds(n0, nc)]
        wu = wu_2d_ref[pl.ds(c0, fp8_kc), pl.ds(n0, nc)]
        rg = jnp.dot(x, wg, preferred_element_type=jnp.float32)
        ru = jnp.dot(x, wu, preferred_element_type=jnp.float32)
        acc_g = rg if acc_g is None else acc_g + rg
        acc_u = ru if acc_u is None else acc_u + ru
    return acc_g, acc_u


def _sparse_fp8_decode_kernel(
    *all_refs,
    ep: int,
    batch_size: int,
    num_replicas: int,
    return_sharded: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    cfg: Fp8SparseDecodeConfig,
):
    has_replicas = num_replicas > 0
    is_dense = cfg.row_mode == "dense"
    w_mode = cfg.weight_mode  # "stream3_half", "w1_k2_w2_1x", or "ring3_8m"
    is_ring3 = w_mode == "ring3_8m"
    is_k2 = w_mode == "w1_k2_w2_1x"
    tile_m = cfg.tile_m
    half_i = _I // 2  # 2048
    half_h = _H // 2  # 2048
    e_local = _E_TOTAL // ep
    rows_per_rank = batch_size // ep

    it = iter(all_refs)
    rank_ref = next(it)
    replica_meta_ref = next(it) if has_replicas else None
    logits_ref = next(it)
    bias_ref = next(it)
    x_in_ref = next(it)
    w1_ref = next(it)
    s1_ref = next(it)
    w2_ref = next(it)
    s2_ref = next(it)
    norm_ref = next(it)
    if has_replicas:
        rep_w1_ref = next(it)
        rep_s1_ref = next(it)
        rep_w2_ref = next(it)
        rep_s2_ref = next(it)
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None
    output_ref = next(it)

    if is_dense:
        d_scratch = DenseRoutingScratch(
            *(next(it) for _ in range(len(DenseRoutingScratch._fields)))
        )
        r_scratch = None
    else:
        r_scratch = RoutingScratch(
            *(next(it) for _ in range(len(RoutingScratch._fields)))
        )
        d_scratch = None

    if has_replicas:
        owner_load_smem = next(it)
        moved_flag_smem = next(it)
        local_suppress_smem = next(it)
        holder_slot_expert_smem = next(it)
        holder_cand_smem = (
            next(it) if (not is_dense and not cfg.static_holder_mask) else None
        )
    else:
        owner_load_smem = moved_flag_smem = None
        local_suppress_smem = holder_slot_expert_smem = holder_cand_smem = None

    if is_ring3:
        w_ring_buf = next(it)
        w_ring_sem = next(it)
        w1_buf = w2_buf = w1_acc_f32 = w1_sem = w2_sem = None
    elif is_k2:
        w1_buf = next(it)  # (2, 2048, 8192) fp8
        w2_buf = next(it)  # (4096, 4096) fp8
        w1_acc_f32 = next(it)  # (batch_size if is_dense else tile_m, 8192) f32
        w1_sem = next(it)  # sem((2,))
        w2_sem = next(it)  # sem((1,))
        w_ring_buf = w_ring_sem = None
    else:
        w1_buf = next(it)
        w2_buf = next(it)
        w1_sem = next(it)
        w2_sem = next(it)
        w1_acc_f32 = w_ring_buf = w_ring_sem = None
    misc_sem = next(it)

    if is_dense:
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        gain_f32_buf = next(it)
        accum_2d = next(it)
        accum_3d = next(it)
        fan_3d_buf = next(it) if not return_sharded else None
        x_3d_buf = io_3d_buf = lhs_2d_buf = None
    else:
        x_3d_buf = next(it)
        io_3d_buf = next(it)
        lhs_2d_buf = next(it)
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        gain_f32_buf = next(it)
        accum_3d = next(it)
        accum_2d = None
        fan_3d_buf = x_3d_buf if not return_sharded else None

    use_pair_rs = cfg.pair_rs and (ep in (8, 16))
    rs_send_bf16 = next(it) if cfg.rs_wire_bf16 else None
    if use_pair_rs:
        rs_pair_recv = next(it)
        rs_chip_send = next(it)
        rs_chip_recv = next(it)
        rs_pair_ssem = next(it)
        rs_pair_rsem = next(it)
        rs_chip_ssem = next(it)
        rs_chip_rsem = next(it)
        rs_recv_3d = rs_send_sem = rs_recv_sem = None
    else:
        rs_recv_3d = next(it)
        rs_send_sem = next(it)
        rs_recv_sem = next(it)
        rs_pair_recv = rs_chip_send = rs_chip_recv = None
        rs_pair_ssem = rs_pair_rsem = rs_chip_ssem = rs_chip_rsem = None
    owner_3d = next(it)
    if not return_sharded:
        fan_send_sem = next(it)
        fan_recv_sem = next(it)
        fan_s2_send_sem = next(it) if cfg.bridge_ag else None
        fan_s2_recv_sem = next(it) if cfg.bridge_ag else None
    else:
        fan_send_sem = fan_recv_sem = fan_s2_send_sem = fan_s2_recv_sem = None

    # 1. Signal entry barrier immediately.
    barrier_sem = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]

    def _read_scale_1d(is_w2: bool, src_kind, slot, col0, width: int) -> jax.Array:
        iota8 = lax.broadcasted_iota(jnp.int32, (8, width), 0)
        h_ref = s2_ref if is_w2 else s1_ref
        blk8 = pl.multiple_of((slot >> 3) << 3, 8)
        rem = slot & jnp.int32(7)
        h_tile = h_ref[pl.ds(blk8, 8), pl.ds(col0, width)].astype(jnp.float32)
        h_val = jnp.sum(
            jnp.where(iota8 == rem, h_tile, 0.0), axis=0, keepdims=True
        )
        if not has_replicas:
            return h_val
        r_ref = rep_s2_ref if is_w2 else rep_s1_ref
        r_tile = r_ref[pl.ds(0, 8), pl.ds(col0, width)].astype(jnp.float32)
        r_val = jnp.sum(
            jnp.where(iota8 == slot, r_tile, 0.0), axis=0, keepdims=True
        )
        return jnp.where(src_kind == 0, h_val, r_val)

    # --- FP8 8-MiB chunk DMA helpers (0 scale DMAs!) ---
    # Stage index s in 0..5:
    #   0: W1 Gate_0 [0:2048], 1: W1 Up_0 [4096:6144],
    #   2: W1 Gate_1 [2048:4096], 3: W1 Up_1 [6144:8192],
    #   4: W2 Down_0 [0:2048], 5: W2 Down_1 [2048:4096]
    def _w1_stage_col(part4: int) -> int:
        return (part4 & 1) * _I + (part4 >> 1) * half_i

    if is_ring3:

        def _issue_r3(w_src, stage6: int, slot):
            r_idx = stage6 % 3
            col = (
                (stage6 - 4) * half_h
                if stage6 >= 4
                else _w1_stage_col(stage6)
            )
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(col, half_i)],
                w_ring_buf.at[r_idx],
                w_ring_sem.at[r_idx],
            ).start()

        def start_r3_stage(stage6: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_r3(w2_ref if stage6 >= 4 else w1_ref, stage6, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_r3(
                        rep_w2_ref if stage6 >= 4 else rep_w1_ref, stage6, slot
                    )

            else:
                _issue_r3(w2_ref if stage6 >= 4 else w1_ref, stage6, slot)

        def wait_r3_slot(r_idx: int):
            pltpu.make_async_copy(
                w_ring_buf.at[r_idx], w_ring_buf.at[r_idx], w_ring_sem.at[r_idx]
            ).wait()

    elif is_k2:

        def _issue_k2_w1(w_src, kh: int, slot):
            pltpu.make_async_copy(
                w_src.at[slot, pl.ds(kh * half_h, half_h), :],
                w1_buf.at[kh],
                w1_sem.at[kh],
            ).start()

        def _issue_full_w2(w_src, slot):
            pltpu.make_async_copy(
                w_src.at[slot], w2_buf, w2_sem.at[0]
            ).start()

        def start_k2_w1(kh: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_k2_w1(w1_ref, kh, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_k2_w1(rep_w1_ref, kh, slot)

            else:
                _issue_k2_w1(w1_ref, kh, slot)

        def start_full_w2(src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_full_w2(w2_ref, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_full_w2(rep_w2_ref, slot)

            else:
                _issue_full_w2(w2_ref, slot)

        def wait_k2_w1(kh: int):
            pltpu.make_async_copy(
                w1_buf.at[kh], w1_buf.at[kh], w1_sem.at[kh]
            ).wait()

        def wait_full_w2():
            pltpu.make_async_copy(w2_buf, w2_buf, w2_sem.at[0]).wait()

    else:

        def _issue_s3h_w1(w_src, part4: int, slot):
            col = _w1_stage_col(part4)
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(col, half_i)],
                w1_buf.at[part4],
                w1_sem.at[part4],
            ).start()

        def _issue_s3h_w2(w_src, part2: int, slot):
            col = part2 * half_h
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(col, half_h)],
                w2_buf.at[part2],
                w2_sem.at[part2],
            ).start()

        def start_s3h_w1(part4: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_s3h_w1(w1_ref, part4, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_s3h_w1(rep_w1_ref, part4, slot)

            else:
                _issue_s3h_w1(w1_ref, part4, slot)

        def start_s3h_w2(part2: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_s3h_w2(w2_ref, part2, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_s3h_w2(rep_w2_ref, part2, slot)

            else:
                _issue_s3h_w2(w2_ref, part2, slot)

        def wait_s3h_w1(part4: int):
            pltpu.make_async_copy(
                w1_buf.at[part4], w1_buf.at[part4], w1_sem.at[part4]
            ).wait()

        def wait_s3h_w2(part2: int):
            pltpu.make_async_copy(
                w2_buf.at[part2], w2_buf.at[part2], w2_sem.at[part2]
            ).wait()

    # 1B. Optional speculative first-expert prime (up to 3x 8-MiB stages = 24 MiB) BEFORE routing!
    spec_parts = max(0, min(3, cfg.spec_prime_parts))
    do_spec = spec_parts > 0
    if do_spec:
        if cfg.spec_home_slots:
            pref_slot = sum(
                jnp.where(
                    me == r,
                    jnp.int32(cfg.spec_home_slots[r] + 1),
                    jnp.int32(0),
                )
                for r in range(ep)
            ) - jnp.int32(1)
        else:
            pref_slot = jnp.int32(0)
        if has_replicas and cfg.spec_skip_replicated:
            non_rep = select_non_replicated_home_slot(
                me,
                e_local,
                replica_meta_ref,
                preferred_slot=jnp.maximum(pref_slot, jnp.int32(0)),
                num_replicas=num_replicas,
            )
            spec_slot = jnp.where(pref_slot >= 0, non_rep, jnp.int32(-1))
        else:
            spec_slot = pref_slot
        spec_enabled = (spec_slot >= 0) & (spec_slot < e_local)
        safe_spec_slot = jnp.where(spec_enabled, spec_slot, jnp.int32(0))

        @pl.when(spec_enabled)
        def _issue_fp8_spec_prime():
            if is_k2:
                _issue_k2_w1(w1_ref, 0, safe_spec_slot)
                if spec_parts >= 2:
                    _issue_k2_w1(w1_ref, 1, safe_spec_slot)
                if spec_parts >= 3:
                    _issue_full_w2(w2_ref, safe_spec_slot)
            else:
                for sp_i in range(spec_parts):
                    if is_ring3:
                        _issue_r3(w1_ref, sp_i, safe_spec_slot)
                    else:
                        _issue_s3h_w1(w1_ref, sp_i, safe_spec_slot)

    else:
        spec_slot = None
        spec_enabled = jnp.bool_(False)

    gain_f32_buf[...] = norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0

    # 2. In-kernel tie-exact routing.
    if is_dense:
        route_dense_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            d_scratch,
            misc_sem.at[0],
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
            min_p_topk=cfg.min_p_topk,
            free_max_rows=cfg.free_max_rows,
            free_max_w=cfg.free_max_w,
            free_min_slot=cfg.free_min_slot,
        )
        counts_smem = d_scratch.counts_smem
    else:
        if has_replicas:
            if cfg.static_holder_mask:
                cand_eids = extract_my_holder_candidates_tuple(
                    me, num_replicas, replica_meta_ref
                )
            else:
                extract_my_holder_candidates_smem(
                    me, num_replicas, replica_meta_ref, holder_cand_smem
                )
                cand_eids = tuple(
                    holder_cand_smem[s] for s in range(num_replicas)
                )
        else:
            cand_eids = ()
        route_topk_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            r_scratch,
            misc_sem.at[0],
            my_owner=me,
            e_per_owner=e_local,
            holder_cand_eids=cand_eids,
            hoist_holder_vecs=cfg.static_holder_mask,
            precompute_owner_loads=(has_replicas and cfg.static_holder_mask),
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=tile_m,
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
            min_p_topk=cfg.min_p_topk,
            free_max_rows=cfg.free_max_rows,
            free_max_w=cfg.free_max_w,
            free_min_slot=cfg.free_min_slot,
            route_rb16_mode=getattr(cfg, "route_rb16_mode", None),
            route_fast=getattr(cfg, "route_fast", None),
        )
        counts_smem = r_scratch.counts_smem

    # 3. Fast 32-step SMEM execution plan.
    if has_replicas:
        run_replica_planner_smem(
            me,
            e_local,
            ep,
            num_replicas,
            counts_smem,
            replica_meta_ref,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=replica_cost_mode,
            tile_m=tile_m,
            moved_flag_smem=moved_flag_smem,
            vpu_precomputed_loads=(not is_dense and cfg.static_holder_mask),
            fast_plan=getattr(cfg, "fast_plan", 0),
        )
        if is_dense:
            num_exec = build_dense_exec_plan_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                d_scratch,
                spec_slot=spec_slot,
            )
        else:
            num_exec = build_exec_plan_from_masks_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                r_scratch,
                spec_slot=spec_slot,
                fast_plan=bool(getattr(cfg, "fast_plan", 0)),
            )
    else:
        if is_dense:
            num_exec = build_dense_exec_plan_smem(
                me, e_local, 0, None, None, d_scratch, spec_slot=spec_slot
            )
        else:
            num_exec = build_default_exec_plan_smem(
                me,
                e_local,
                r_scratch,
                spec_slot=spec_slot,
                fast_plan=bool(getattr(cfg, "fast_plan", False)),
            )

    exec_eid_smem = (
        d_scratch.exec_expert_id if is_dense else r_scratch.exec_expert_id
    )
    exec_src_smem = (
        d_scratch.exec_weight_src if is_dense else r_scratch.exec_weight_src
    )
    exec_slot_smem = (
        d_scratch.exec_weight_slot if is_dense else r_scratch.exec_weight_slot
    )

    def _prime_all_e0(s0, sl0):
        if is_ring3:
            start_r3_stage(0, s0, sl0)
            start_r3_stage(1, s0, sl0)
            start_r3_stage(2, s0, sl0)
        elif is_k2:
            start_k2_w1(0, s0, sl0)
            start_k2_w1(1, s0, sl0)
            start_full_w2(s0, sl0)
        else:
            start_s3h_w1(0, s0, sl0)
            start_s3h_w1(1, s0, sl0)
            start_s3h_w1(2, s0, sl0)
            start_s3h_w1(3, s0, sl0)
            start_s3h_w2(0, s0, sl0)
            start_s3h_w2(1, s0, sl0)

    # 4. Prime FP8 weight ring BEFORE prologue zeroing/packing!
    if do_spec:
        spec_hit = (
            spec_enabled
            & (num_exec > 0)
            & (exec_src_smem[0] == 0)
            & (exec_slot_smem[0] == spec_slot)
        )

        @pl.when(spec_enabled & (~spec_hit))
        def _drain_fp8_spec_miss():
            if is_k2:
                wait_k2_w1(0)
                if spec_parts >= 2:
                    wait_k2_w1(1)
                if spec_parts >= 3:
                    wait_full_w2()
            else:
                for sp_i in range(spec_parts):
                    if is_ring3:
                        wait_r3_slot(sp_i)
                    else:
                        wait_s3h_w1(sp_i)

        @pl.when(spec_hit)
        def _prime_fp8_spec_rest():
            s0 = exec_src_smem[0]
            sl0 = exec_slot_smem[0]
            if is_ring3:
                for stg_i in range(spec_parts, 3):
                    start_r3_stage(stg_i, s0, sl0)
            elif is_k2:
                if spec_parts < 2:
                    start_k2_w1(1, s0, sl0)
                if spec_parts < 3:
                    start_full_w2(s0, sl0)
            else:
                for p4_i in range(spec_parts, 4):
                    start_s3h_w1(p4_i, s0, sl0)
                start_s3h_w2(0, s0, sl0)
                start_s3h_w2(1, s0, sl0)

        @pl.when((~spec_hit) & (num_exec > 0))
        def _prime_fp8_after_miss():
            _prime_all_e0(exec_src_smem[0], exec_slot_smem[0])

    else:

        @pl.when(num_exec > 0)
        def _prime_fp8_ring():
            _prime_all_e0(exec_src_smem[0], exec_slot_smem[0])

    # 5. Overlapped prologue while initial FP8 weights stream from HBM.
    if is_dense:
        for s_col in range(0, _H, 1024):
            accum_2d[:, pl.ds(s_col, 1024)] = jnp.zeros(
                (batch_size, 1024), dtype=jnp.float32
            )
    elif cfg.fast_b64_epi_pro and batch_size <= 64 and not is_ring3:
        z_chunk = min(batch_size, 32)
        z_acc = jnp.zeros((z_chunk, 32, 128), dtype=jnp.float32)
        for r0 in range(0, batch_size, z_chunk):
            accum_3d[pl.ds(r0, z_chunk), :, :] = z_acc
        io_3d_buf[...] = jnp.zeros((tile_m, 32, 128), dtype=jnp.bfloat16)
        x_chunk = min(batch_size, 32)
        for r0 in range(0, batch_size, x_chunk):
            for h_idx in range(2):
                x_3d_buf[pl.ds(r0, x_chunk), pl.ds(h_idx * 16, 16), :] = x_in_ref[
                    pl.ds(r0, x_chunk), pl.ds(h_idx * 2048, 2048)
                ].reshape(x_chunk, 16, 128)
        pack_active_tokens_smem(
            batch_size, r_scratch, has_replicas=has_replicas
        )
    else:
        zero_rc = min(batch_size, 8)
        zeros_acc = jnp.zeros((zero_rc, 32, 128), dtype=jnp.float32)

        def _zero_acc_chunk(ci, _):
            accum_3d[pl.ds(ci * zero_rc, zero_rc), :, :] = zeros_acc
            return _

        lax.fori_loop(0, batch_size // zero_rc, _zero_acc_chunk, jnp.int32(0))

        io_rows_alloc = batch_size if is_ring3 else tile_m
        zeros_io = jnp.zeros((8, 32, 128), dtype=jnp.bfloat16)

        def _zero_io_chunk(ci, _):
            io_3d_buf[pl.ds(ci * 8, 8), :, :] = zeros_io
            return _

        lax.fori_loop(0, io_rows_alloc // 8, _zero_io_chunk, jnp.int32(0))

        x_rc = 16

        def _reshape_x_chunk(ci, _):
            r0 = pl.multiple_of(ci * x_rc, x_rc)
            for h_idx in range(2):
                x_3d_buf[pl.ds(r0, x_rc), pl.ds(h_idx * 16, 16), :] = x_in_ref[
                    pl.ds(r0, x_rc), pl.ds(h_idx * 2048, 2048)
                ].reshape(x_rc, 16, 128)
            return _

        lax.fori_loop(0, batch_size // x_rc, _reshape_x_chunk, jnp.int32(0))

        pack_active_tokens_smem(
            batch_size, r_scratch, has_replicas=has_replicas
        )

    # --- FP8 Compute Strip Helpers ---
    def _run_fp8_proj_half(
        x_src_ref,
        w_half_ref,
        src_kind,
        slot,
        half_idx: int,
        m_rows: int,
        is_up: bool,
        act_dst_ref,
    ):
        nc = _pick_k256_nc(m_rows, half_i, cfg.fp8_strip_mode)
        write_m = 16 if m_rows == 8 else m_rows
        act_col_base = half_idx * half_i
        s1_col_base = (_I if is_up else 0) + act_col_base

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            raw = _fp8_gmm_chunked(
                x_src_ref, w_half_ref, m_rows, _H, n0, nc, cfg.fp8_kc
            )
            s_col = _read_scale_1d(False, src_kind, slot, s1_col_base + n0, nc)
            res = raw * s_col
            col_out = pl.ds(act_col_base + n0, nc)
            if not is_up:
                sg_bf = jax.nn.silu(res).astype(jnp.bfloat16)
                act_dst_ref[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    sg_bf, write_m
                )
            else:
                sg_read = act_dst_ref[pl.ds(0, write_m), col_out]
                if m_rows == 8:
                    sg_read = sg_read[:8, :]
                h = (sg_read.astype(jnp.float32) * res).astype(jnp.bfloat16)
                act_dst_ref[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    h, write_m
                )
            return carry

        n_strips = half_i // nc
        if n_strips == 1:
            _strip(0, 0)
        elif n_strips == 2:
            _strip(0, 0)
            _strip(1, 0)
        else:
            lax.fori_loop(0, n_strips, _strip, 0)

    def _run_fp8_dual_half(
        x_src_ref,
        wg_half_ref,
        wu_half_ref,
        src_kind,
        slot,
        half_idx: int,
        m_rows: int,
        act_dst_ref,
    ):
        nc = min(1024, _pick_k256_nc(m_rows, half_i, cfg.fp8_strip_mode))
        write_m = 16 if m_rows == 8 else m_rows
        act_col_base = half_idx * half_i
        sg_col_base = act_col_base
        su_col_base = _I + act_col_base

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            raw_g, raw_u = _fp8_dual_gmm_chunked(
                x_src_ref,
                wg_half_ref,
                wu_half_ref,
                m_rows,
                _H,
                n0,
                nc,
                cfg.fp8_kc,
            )
            sg = _read_scale_1d(False, src_kind, slot, sg_col_base + n0, nc)
            su = _read_scale_1d(False, src_kind, slot, su_col_base + n0, nc)
            g = raw_g * sg
            u = raw_u * su
            h = (
                jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
            ).astype(jnp.bfloat16)
            act_dst_ref[pl.ds(0, write_m), pl.ds(act_col_base + n0, nc)] = (
                _pad_rows_to(h, write_m)
            )
            return carry

        n_strips = half_i // nc
        if n_strips == 1:
            _strip(0, 0)
        elif n_strips == 2:
            _strip(0, 0)
            _strip(1, 0)
        else:
            lax.fori_loop(0, n_strips, _strip, 0)

    def _run_fp8_down_half(
        act_src_ref,
        w2_half_ref,
        src_kind,
        slot,
        d_idx: int,
        m_rows: int,
        sq_sum,
        y_dst_ref,
    ):
        nc = _pick_k256_nc(m_rows, half_h, cfg.fp8_strip_mode)
        write_m = m_rows if cfg.norm_f32_stage else (16 if m_rows == 8 else m_rows)
        col_base = d_idx * half_h

        def _strip(j, sq_acc):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_base + n0, nc)
            raw = _fp8_gmm_chunked(
                act_src_ref, w2_half_ref, m_rows, _I, n0, nc, cfg.fp8_kc
            )
            s_col = _read_scale_1d(True, src_kind, slot, col_base + n0, nc)
            d = (raw * s_col).astype(jnp.bfloat16)
            g_col = gain_f32_buf[0:1, col_out]
            if cfg.norm_f32_stage:
                u = d.astype(jnp.float32) * g_col
                sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
                y_dst_ref[pl.ds(0, write_m), col_out] = u
            else:
                u_bf = d * g_col.astype(jnp.bfloat16)
                u_f32 = u_bf.astype(jnp.float32)
                sq_acc = sq_acc + jnp.sum(u_f32 * u_f32, axis=1, keepdims=True)
                y_dst_ref[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    u_bf, write_m
                )
            return sq_acc

        n_strips = half_h // nc
        if n_strips == 1:
            return _strip(0, sq_sum)
        if n_strips == 2:
            sq_sum = _strip(0, sq_sum)
            return _strip(1, sq_sum)
        return lax.fori_loop(0, n_strips, _strip, sq_sum)

    def _quant_x_fp8_row(x_bf16: jax.Array, m_rows: int) -> tuple[jax.Array, jax.Array]:
        xf = x_bf16.astype(jnp.float32)
        amax = jnp.max(jnp.abs(xf), axis=1, keepdims=True)
        sc = jnp.maximum(amax, 1e-12) * (1.0 / 448.0)
        xq = jnp.clip(xf / sc, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        return _pad_rows_to(xq, max(32, m_rows)), sc

    def _dot_k2_slice(
        x_src_ref, w_k_ref, k_off: int, k_len: int, m_rows: int, n0: int, nc: int
    ) -> jax.Array:
        read_m = 16 if m_rows == 8 else m_rows
        if cfg.lhs_fp8:
            x_full = x_src_ref[pl.ds(0, read_m), pl.ds(k_off, k_len)]
            if read_m != m_rows:
                x_full = x_full[:m_rows, :]
            xq_pad, x_sc = _quant_x_fp8_row(x_full, m_rows)
            acc = None
            for c0 in range(0, k_len, cfg.fp8_kc):
                w = w_k_ref[pl.ds(c0, cfg.fp8_kc), pl.ds(n0, nc)]
                res = jnp.dot(
                    xq_pad[:, c0 : c0 + cfg.fp8_kc],
                    w,
                    preferred_element_type=jnp.float32,
                )[:m_rows, :]
                acc = res if acc is None else acc + res
            return acc * x_sc
        acc = None
        for c0 in range(0, k_len, cfg.fp8_kc):
            x = x_src_ref[pl.ds(0, read_m), pl.ds(k_off + c0, cfg.fp8_kc)]
            if read_m != m_rows:
                x = x[:m_rows, :]
            w = w_k_ref[pl.ds(c0, cfg.fp8_kc), pl.ds(n0, nc)]
            res = jnp.dot(x, w, preferred_element_type=jnp.float32)
            acc = res if acc is None else acc + res
        return acc

    def _run_fp8_k2_expert_tile(
        x_src_ref,
        m_rows: int,
        cur_src,
        cur_slot,
        is_first_tile,
        do_pf_next,
        next_src,
        next_slot,
        sq_sum,
    ):
        write_m_bf = 16 if m_rows == 8 else m_rows
        write_m_y = m_rows if cfg.norm_f32_stage else write_m_bf

        if is_first_tile is True:
            wait_k2_w1(0)
        else:

            @pl.when(is_first_tile)
            def _wk0():
                wait_k2_w1(0)

        for n0 in range(0, _N1, half_i):
            p0 = _dot_k2_slice(x_src_ref, w1_buf.at[0], 0, half_h, m_rows, n0, half_i)
            w1_acc_f32[pl.ds(0, m_rows), pl.ds(n0, half_i)] = p0

        @pl.when(do_pf_next)
        def _pf_k0():
            start_k2_w1(0, next_src, next_slot)

        if is_first_tile is True:
            wait_k2_w1(1)
        else:

            @pl.when(is_first_tile)
            def _wk1():
                wait_k2_w1(1)

        for h_idx in range(2):
            g_col0 = h_idx * half_i
            u_col0 = _I + g_col0
            g_raw = w1_acc_f32[pl.ds(0, m_rows), pl.ds(g_col0, half_i)] + _dot_k2_slice(
                x_src_ref, w1_buf.at[1], half_h, half_h, m_rows, g_col0, half_i
            )
            u_raw = w1_acc_f32[pl.ds(0, m_rows), pl.ds(u_col0, half_i)] + _dot_k2_slice(
                x_src_ref, w1_buf.at[1], half_h, half_h, m_rows, u_col0, half_i
            )
            sg = _read_scale_1d(False, cur_src, cur_slot, g_col0, half_i)
            su = _read_scale_1d(False, cur_src, cur_slot, u_col0, half_i)
            g = g_raw * sg
            u = u_raw * su
            h = (
                jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
            ).astype(jnp.bfloat16)
            act_2d_buf[pl.ds(0, write_m_bf), pl.ds(g_col0, half_i)] = _pad_rows_to(
                h, write_m_bf
            )

        @pl.when(do_pf_next)
        def _pf_k1():
            start_k2_w1(1, next_src, next_slot)

        if is_first_tile is True:
            wait_full_w2()
        else:

            @pl.when(is_first_tile)
            def _ww2():
                wait_full_w2()

        for d_idx in range(2):
            c_base = d_idx * half_h
            raw_d = _dot_k2_slice(act_2d_buf, w2_buf, 0, _I, m_rows, c_base, half_h)
            s_col = _read_scale_1d(True, cur_src, cur_slot, c_base, half_h)
            d = (raw_d * s_col).astype(jnp.bfloat16)
            col_out = pl.ds(c_base, half_h)
            g_col = gain_f32_buf[0:1, col_out]
            if cfg.norm_f32_stage:
                u_n = d.astype(jnp.float32) * g_col
                sq_sum = sq_sum + jnp.sum(u_n * u_n, axis=1, keepdims=True)
                y_stage_buf[pl.ds(0, write_m_y), col_out] = u_n
            else:
                u_bf = d * g_col.astype(jnp.bfloat16)
                u_f32 = u_bf.astype(jnp.float32)
                sq_sum = sq_sum + jnp.sum(u_f32 * u_f32, axis=1, keepdims=True)
                y_stage_buf[pl.ds(0, write_m_y), col_out] = _pad_rows_to(
                    u_bf, write_m_y
                )

        @pl.when(do_pf_next)
        def _pf_w2f():
            start_full_w2(next_src, next_slot)

        return sq_sum

    # =========================================================================
    # 6A. DENSE-ROWS FP8 LOOP (B = 16)
    # =========================================================================
    if is_dense:
        lane_128 = lax.broadcasted_iota(jnp.int32, (batch_size, 128), 1)
        use_dual_d = (not is_ring3) and (
            cfg.fp8_dual_mode == "dual"
            or (cfg.fp8_dual_mode == "hybrid" and batch_size <= 16)
        )

        def _dense_fp8_expert_body(k, _):
            g_e = exec_eid_smem[k]
            cur_src = exec_src_smem[k]
            cur_slot = exec_slot_smem[k]
            has_next = (k + 1) < num_exec
            next_k = jnp.where(has_next, k + 1, jnp.int32(0))
            next_src = exec_src_smem[next_k]
            next_slot = exec_slot_smem[next_k]
            sq_sum = jnp.zeros((batch_size, 1), dtype=jnp.float32)

            if is_ring3:
                wait_r3_slot(0)
                _run_fp8_proj_half(
                    x_in_ref,
                    w_ring_buf.at[0],
                    cur_src,
                    cur_slot,
                    0,
                    batch_size,
                    False,
                    act_2d_buf,
                )
                start_r3_stage(3, cur_src, cur_slot)

                wait_r3_slot(1)
                _run_fp8_proj_half(
                    x_in_ref,
                    w_ring_buf.at[1],
                    cur_src,
                    cur_slot,
                    0,
                    batch_size,
                    True,
                    act_2d_buf,
                )
                start_r3_stage(4, cur_src, cur_slot)

                wait_r3_slot(2)
                _run_fp8_proj_half(
                    x_in_ref,
                    w_ring_buf.at[2],
                    cur_src,
                    cur_slot,
                    1,
                    batch_size,
                    False,
                    act_2d_buf,
                )
                start_r3_stage(5, cur_src, cur_slot)

                wait_r3_slot(0)
                _run_fp8_proj_half(
                    x_in_ref,
                    w_ring_buf.at[0],
                    cur_src,
                    cur_slot,
                    1,
                    batch_size,
                    True,
                    act_2d_buf,
                )

                @pl.when(has_next)
                def _pf_r3_0():
                    start_r3_stage(0, next_src, next_slot)

                wait_r3_slot(1)
                sq_sum = _run_fp8_down_half(
                    act_2d_buf,
                    w_ring_buf.at[1],
                    cur_src,
                    cur_slot,
                    0,
                    batch_size,
                    sq_sum,
                    y_stage_buf,
                )

                @pl.when(has_next)
                def _pf_r3_1():
                    start_r3_stage(1, next_src, next_slot)

                wait_r3_slot(2)
                sq_sum = _run_fp8_down_half(
                    act_2d_buf,
                    w_ring_buf.at[2],
                    cur_src,
                    cur_slot,
                    1,
                    batch_size,
                    sq_sum,
                    y_stage_buf,
                )

                @pl.when(has_next)
                def _pf_r3_2():
                    start_r3_stage(2, next_src, next_slot)

            elif is_k2:
                sq_sum = _run_fp8_k2_expert_tile(
                    x_in_ref,
                    batch_size,
                    cur_src,
                    cur_slot,
                    True,
                    has_next,
                    next_src,
                    next_slot,
                    sq_sum,
                )

            else:
                for h_idx in range(2):
                    p_g = h_idx * 2
                    p_u = p_g + 1
                    if use_dual_d:
                        wait_s3h_w1(p_g)
                        wait_s3h_w1(p_u)
                        _run_fp8_dual_half(
                            x_in_ref,
                            w1_buf.at[p_g],
                            w1_buf.at[p_u],
                            cur_src,
                            cur_slot,
                            h_idx,
                            batch_size,
                            act_2d_buf,
                        )

                        @pl.when(has_next)
                        def _pf_du(_pg=p_g, _pu=p_u):
                            start_s3h_w1(_pg, next_src, next_slot)
                            start_s3h_w1(_pu, next_src, next_slot)

                    else:
                        wait_s3h_w1(p_g)
                        _run_fp8_proj_half(
                            x_in_ref,
                            w1_buf.at[p_g],
                            cur_src,
                            cur_slot,
                            h_idx,
                            batch_size,
                            False,
                            act_2d_buf,
                        )

                        @pl.when(has_next)
                        def _pf_g(_pg=p_g):
                            start_s3h_w1(_pg, next_src, next_slot)

                        wait_s3h_w1(p_u)
                        _run_fp8_proj_half(
                            x_in_ref,
                            w1_buf.at[p_u],
                            cur_src,
                            cur_slot,
                            h_idx,
                            batch_size,
                            True,
                            act_2d_buf,
                        )

                        @pl.when(has_next)
                        def _pf_u(_pu=p_u):
                            start_s3h_w1(_pu, next_src, next_slot)

                for d_idx in range(2):
                    wait_s3h_w2(d_idx)
                    sq_sum = _run_fp8_down_half(
                        act_2d_buf,
                        w2_buf.at[d_idx],
                        cur_src,
                        cur_slot,
                        d_idx,
                        batch_size,
                        sq_sum,
                        y_stage_buf,
                    )

                    @pl.when(has_next)
                    def _pf_d(_di=d_idx):
                        start_s3h_w2(_di, next_src, next_slot)

            blk = pl.multiple_of((g_e >> 7) << 7, 128)
            w_slice = d_scratch.dense_w_vmem[:, pl.ds(blk, 128)]
            w_col = jnp.sum(
                jnp.where((lane_128 + blk) == g_e, w_slice, 0.0),
                axis=1,
                keepdims=True,
            )
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            rc = min(batch_size, 8 if cfg.norm_f32_stage else 16)
            for r0 in range(0, batch_size, rc):
                inv_r = inv_rms[r0 : r0 + rc, :]
                w_r = w_col[r0 : r0 + rc, :]
                for s_col in range(0, _H, 1024):
                    col_s = pl.ds(s_col, 1024)
                    u_c = y_stage_buf[pl.ds(r0, rc), col_s]
                    v_bf = (
                        (u_c * inv_r).astype(jnp.bfloat16)
                        if cfg.norm_f32_stage
                        else u_c * inv_r.astype(jnp.bfloat16)
                    )
                    contrib = jnp.where(
                        w_r > 0.0, v_bf.astype(jnp.float32) * w_r, 0.0
                    )
                    accum_2d[pl.ds(r0, rc), col_s] = (
                        accum_2d[pl.ds(r0, rc), col_s] + contrib
                    )
            return _

        lax.fori_loop(0, num_exec, _dense_fp8_expert_body, jnp.int32(0))
        conv_rc = min(batch_size, 8)
        for r0 in range(0, batch_size, conv_rc):
            for h_idx in range(2):
                accum_3d[pl.ds(r0, conv_rc), pl.ds(h_idx * 16, 16), :] = (
                    accum_2d[
                        pl.ds(r0, conv_rc), pl.ds(h_idx * 2048, 2048)
                    ].reshape(conv_rc, 16, 128)
                )

    # =========================================================================
    # 6B. SPARSE FP8 EXECUTION LOOP (stream3_half 6x8 MiB OR ring3_8m 3x8 MiB)
    # =========================================================================
    else:

        def _gather_rows_to_3d(base_pos, cur_rows):
            n8 = cur_rows >> 3

            def _gather_8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                rows = [x_3d_buf[pl.ds(t, 1), :, :] for t in ts]
                io_3d_buf[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
                return _

            lax.fori_loop(0, n8, _gather_8, jnp.int32(0))

            def _gather_tail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                io_3d_buf[pl.ds(i, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
                return _

            lax.fori_loop(n8 << 3, cur_rows, _gather_tail, jnp.int32(0))

        def _copy_io_3d_to_lhs_2d_at(r_base, m_rows: int):
            copy_m = 16 if m_rows == 8 else m_rows
            for r0 in range(0, copy_m, 16):
                rr = r_base + r0
                for h_idx in range(2):
                    lhs_2d_buf[pl.ds(rr, 16), pl.ds(h_idx * 2048, 2048)] = (
                        io_3d_buf[
                            pl.ds(rr, 16), pl.ds(h_idx * 16, 16), :
                        ].reshape(16, 2048)
                    )

        def _finish_rmsnorm_to_io_3d_at(
            r_base, m_rows: int, sq_sum: jax.Array
        ):
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            if cfg.norm_f32_stage:
                rc = 8
                for r0 in range(0, m_rows, rc):
                    inv_r = inv_rms[r0 : r0 + rc, :]
                    for h_idx in range(2):
                        u_c = y_stage_buf[
                            pl.ds(r0, rc), pl.ds(h_idx * 2048, 2048)
                        ]
                        v_bf = (u_c * inv_r).astype(jnp.bfloat16)
                        io_3d_buf[
                            pl.ds(r_base + r0, rc), pl.ds(h_idx * 16, 16), :
                        ] = v_bf.reshape(rc, 16, 128)
            else:
                inv_bf = inv_rms.astype(jnp.bfloat16)
                rc = 8 if m_rows == 8 else 16
                read_rc = 16 if m_rows == 8 else rc
                for r0 in range(0, m_rows, rc):
                    inv_r = inv_bf[r0 : r0 + rc, :]
                    for h_idx in range(2):
                        u_c = y_stage_buf[
                            pl.ds(r0, read_rc), pl.ds(h_idx * 2048, 2048)
                        ]
                        if m_rows == 8:
                            u_c = u_c[:8, :]
                        v_bf = u_c * inv_r
                        io_3d_buf[
                            pl.ds(r_base + r0, rc), pl.ds(h_idx * 16, 16), :
                        ] = v_bf.reshape(rc, 16, 128)

        def _scatter_accumulate(base_pos, cur_rows):
            n8 = cur_rows >> 3

            def _acc_8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                ws = [r_scratch.packed_tok_w[p0 + s] for s in range(8)]
                y8 = io_3d_buf[pl.ds(i0, 8), :, :]
                for s in range(8):
                    ys = y8[s : s + 1, :, :].astype(jnp.float32) * ws[s]
                    accum_3d[pl.ds(ts[s], 1), :, :] = (
                        accum_3d[pl.ds(ts[s], 1), :, :] + ys
                    )
                return _

            lax.fori_loop(0, n8, _acc_8, jnp.int32(0))

            def _acc_tail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                w = r_scratch.packed_tok_w[base_pos + i]
                y_row = io_3d_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                accum_3d[pl.ds(tok, 1), :, :] = (
                    accum_3d[pl.ds(tok, 1), :, :] + y_row * w
                )
                return _

            lax.fori_loop(n8 << 3, cur_rows, _acc_tail, jnp.int32(0))

        m_buckets = cfg.m_buckets

        if is_ring3:
            # All r_e <= B rows gathered once per expert; each 8-MiB stage runs across
            # all rows before releasing its 3-slot FIFO buffer (1x weight read even at B=256!).
            def _compute_r3_expert(
                r_e, cur_src, cur_slot, has_next, next_src, next_slot
            ):
                n_m32 = jnp.maximum(jnp.int32(1), (r_e + 31) // 32)

                def _cp_lhs_all(mi, _):
                    r0 = pl.multiple_of(mi * 32, 32)
                    _copy_io_3d_to_lhs_2d_at(r0, 32)
                    return _

                lax.fori_loop(0, n_m32, _cp_lhs_all, jnp.int32(0))

                def _run_proj_stage(w_b_ref, half_idx: int, is_up: bool):
                    def _single_bkt(m_b: int):
                        _run_fp8_proj_half(
                            lhs_2d_buf.at[pl.ds(0, m_b)],
                            w_b_ref,
                            cur_src,
                            cur_slot,
                            half_idx,
                            m_b,
                            is_up,
                            act_2d_buf.at[pl.ds(0, m_b)],
                        )
                        return jnp.int32(0)

                    def _multi_32():
                        def _lp(mi, _):
                            r0 = pl.multiple_of(mi * 32, 32)
                            _run_fp8_proj_half(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w_b_ref,
                                cur_src,
                                cur_slot,
                                half_idx,
                                32,
                                is_up,
                                act_2d_buf.at[pl.ds(r0, 32)],
                            )
                            return _

                        lax.fori_loop(0, n_m32, _lp, jnp.int32(0))
                        return jnp.int32(0)

                    b_idx = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        b_idx = b_idx + (r_e > _b).astype(jnp.int32)
                    if batch_size > m_buckets[-1]:
                        idx = jnp.where(
                            r_e <= m_buckets[-1],
                            b_idx,
                            jnp.int32(len(m_buckets)),
                        )
                        branches = [
                            (lambda b=b: _single_bkt(b)) for b in m_buckets
                        ] + [_multi_32]
                        lax.switch(idx, branches)
                    elif len(m_buckets) > 1:
                        branches = [
                            (lambda b=b: _single_bkt(b)) for b in m_buckets
                        ]
                        lax.switch(b_idx, branches)
                    else:
                        _single_bkt(m_buckets[0])

                wait_r3_slot(0)
                _run_proj_stage(w_ring_buf.at[0], 0, False)
                start_r3_stage(3, cur_src, cur_slot)

                wait_r3_slot(1)
                _run_proj_stage(w_ring_buf.at[1], 0, True)
                start_r3_stage(4, cur_src, cur_slot)

                wait_r3_slot(2)
                _run_proj_stage(w_ring_buf.at[2], 1, False)
                start_r3_stage(5, cur_src, cur_slot)

                wait_r3_slot(0)
                _run_proj_stage(w_ring_buf.at[0], 1, True)

                @pl.when(has_next)
                def _pf0():
                    start_r3_stage(0, next_src, next_slot)

                wait_r3_slot(1)
                wait_r3_slot(2)

                def _down_single(m_b: int):
                    sq = jnp.zeros((m_b, 1), dtype=jnp.float32)
                    sq = _run_fp8_down_half(
                        act_2d_buf.at[pl.ds(0, m_b)],
                        w_ring_buf.at[1],
                        cur_src,
                        cur_slot,
                        0,
                        m_b,
                        sq,
                        y_stage_buf,
                    )
                    sq = _run_fp8_down_half(
                        act_2d_buf.at[pl.ds(0, m_b)],
                        w_ring_buf.at[2],
                        cur_src,
                        cur_slot,
                        1,
                        m_b,
                        sq,
                        y_stage_buf,
                    )
                    _finish_rmsnorm_to_io_3d_at(0, m_b, sq)
                    return jnp.int32(0)

                def _down_multi_32():
                    def _lpd(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq = jnp.zeros((32, 1), dtype=jnp.float32)
                        sq = _run_fp8_down_half(
                            act_2d_buf.at[pl.ds(r0, 32)],
                            w_ring_buf.at[1],
                            cur_src,
                            cur_slot,
                            0,
                            32,
                            sq,
                            y_stage_buf,
                        )
                        sq = _run_fp8_down_half(
                            act_2d_buf.at[pl.ds(r0, 32)],
                            w_ring_buf.at[2],
                            cur_src,
                            cur_slot,
                            1,
                            32,
                            sq,
                            y_stage_buf,
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq)
                        return _

                    lax.fori_loop(0, n_m32, _lpd, jnp.int32(0))
                    return jnp.int32(0)

                b_idx_d = jnp.int32(0)
                for _b in m_buckets[:-1]:
                    b_idx_d = b_idx_d + (r_e > _b).astype(jnp.int32)
                if batch_size > m_buckets[-1]:
                    idx_d = jnp.where(
                        r_e <= m_buckets[-1],
                        b_idx_d,
                        jnp.int32(len(m_buckets)),
                    )
                    branches_d = [
                        (lambda b=b: _down_single(b)) for b in m_buckets
                    ] + [_down_multi_32]
                    lax.switch(idx_d, branches_d)
                elif len(m_buckets) > 1:
                    branches_d = [
                        (lambda b=b: _down_single(b)) for b in m_buckets
                    ]
                    lax.switch(b_idx_d, branches_d)
                else:
                    _down_single(m_buckets[0])

                @pl.when(has_next)
                def _pf12():
                    start_r3_stage(1, next_src, next_slot)
                    start_r3_stage(2, next_src, next_slot)

            def _r3_expert_body(k, _):
                cur_src = exec_src_smem[k]
                cur_slot = exec_slot_smem[k]
                has_next = (k + 1) < num_exec
                next_k = jnp.where(has_next, k + 1, jnp.int32(0))
                next_src = exec_src_smem[next_k]
                next_slot = exec_slot_smem[next_k]
                row_start = r_scratch.exec_start[k]
                row_end = r_scratch.exec_start[k + 1]
                r_e = row_end - row_start
                _gather_rows_to_3d(row_start, r_e)
                _compute_r3_expert(
                    r_e, cur_src, cur_slot, has_next, next_src, next_slot
                )
                _scatter_accumulate(row_start, r_e)
                return _

            lax.fori_loop(0, num_exec, _r3_expert_body, jnp.int32(0))

        else:
            # stream3_half (6 x 8 MiB = 48 MiB resident per expert across row tiles!)
            def _compute_s3h_tile(
                m_rows: int,
                cur_src,
                cur_slot,
                is_first_tile,
                is_last_tile,
                has_next,
                next_src,
                next_slot,
            ):
                do_pf_next = is_last_tile & has_next
                _copy_io_3d_to_lhs_2d_at(0, m_rows)
                sq_sum = jnp.zeros((m_rows, 1), dtype=jnp.float32)
                if is_k2:
                    sq_sum = _run_fp8_k2_expert_tile(
                        lhs_2d_buf,
                        m_rows,
                        cur_src,
                        cur_slot,
                        is_first_tile,
                        do_pf_next,
                        next_src,
                        next_slot,
                        sq_sum,
                    )
                    _finish_rmsnorm_to_io_3d_at(0, m_rows, sq_sum)
                    return
                use_dual = cfg.fp8_dual_mode == "dual" or (
                    cfg.fp8_dual_mode == "hybrid" and m_rows <= 16
                )

                for h_idx in range(2):
                    p_g = h_idx * 2
                    p_u = p_g + 1
                    if use_dual:

                        @pl.when(is_first_tile)
                        def _w_du(_pg=p_g, _pu=p_u):
                            wait_s3h_w1(_pg)
                            wait_s3h_w1(_pu)

                        _run_fp8_dual_half(
                            lhs_2d_buf,
                            w1_buf.at[p_g],
                            w1_buf.at[p_u],
                            cur_src,
                            cur_slot,
                            h_idx,
                            m_rows,
                            act_2d_buf,
                        )

                        @pl.when(do_pf_next)
                        def _pf_du(_pg=p_g, _pu=p_u):
                            start_s3h_w1(_pg, next_src, next_slot)
                            start_s3h_w1(_pu, next_src, next_slot)

                    else:

                        @pl.when(is_first_tile)
                        def _w_g(_pg=p_g):
                            wait_s3h_w1(_pg)

                        _run_fp8_proj_half(
                            lhs_2d_buf,
                            w1_buf.at[p_g],
                            cur_src,
                            cur_slot,
                            h_idx,
                            m_rows,
                            False,
                            act_2d_buf,
                        )

                        @pl.when(do_pf_next)
                        def _pf_g(_pg=p_g):
                            start_s3h_w1(_pg, next_src, next_slot)

                        @pl.when(is_first_tile)
                        def _w_u(_pu=p_u):
                            wait_s3h_w1(_pu)

                        _run_fp8_proj_half(
                            lhs_2d_buf,
                            w1_buf.at[p_u],
                            cur_src,
                            cur_slot,
                            h_idx,
                            m_rows,
                            True,
                            act_2d_buf,
                        )

                        @pl.when(do_pf_next)
                        def _pf_u(_pu=p_u):
                            start_s3h_w1(_pu, next_src, next_slot)

                for d_idx in range(2):

                    @pl.when(is_first_tile)
                    def _w_d(_di=d_idx):
                        wait_s3h_w2(_di)

                    sq_sum = _run_fp8_down_half(
                        act_2d_buf,
                        w2_buf.at[d_idx],
                        cur_src,
                        cur_slot,
                        d_idx,
                        m_rows,
                        sq_sum,
                        y_stage_buf,
                    )

                    @pl.when(do_pf_next)
                    def _pf_d(_di=d_idx):
                        start_s3h_w2(_di, next_src, next_slot)

                _finish_rmsnorm_to_io_3d_at(0, m_rows, sq_sum)

            def _s3h_expert_body(k, _):
                cur_src = exec_src_smem[k]
                cur_slot = exec_slot_smem[k]
                has_next = (k + 1) < num_exec
                next_k = jnp.where(has_next, k + 1, jnp.int32(0))
                next_src = exec_src_smem[next_k]
                next_slot = exec_slot_smem[next_k]

                row_start = r_scratch.exec_start[k]
                row_end = r_scratch.exec_start[k + 1]
                r_e = row_end - row_start
                n_row_tiles = (r_e + tile_m - 1) // tile_m

                def _row_tile_body(tile_idx, _c):
                    tile_off = tile_idx * tile_m
                    base_pos = row_start + tile_off
                    cur_rows = jnp.minimum(tile_m, r_e - tile_off)
                    is_first_tile = tile_idx == 0
                    is_last_tile = tile_idx == (n_row_tiles - 1)

                    _gather_rows_to_3d(base_pos, cur_rows)
                    t_args = (
                        cur_src,
                        cur_slot,
                        is_first_tile,
                        is_last_tile,
                        has_next,
                        next_src,
                        next_slot,
                    )
                    if len(m_buckets) > 1:
                        b_idx = jnp.int32(0)
                        for _b in m_buckets[:-1]:
                            b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                        branches = [
                            (
                                lambda b=b: (
                                    _compute_s3h_tile(b, *t_args),
                                    jnp.int32(0),
                                )[1]
                            )
                            for b in m_buckets
                        ]
                        lax.switch(b_idx, branches)
                    else:
                        _compute_s3h_tile(m_buckets[0], *t_args)

                    _scatter_accumulate(base_pos, cur_rows)
                    return _c

                lax.fori_loop(0, n_row_tiles, _row_tile_body, jnp.int32(0))
                return _

            lax.fori_loop(0, num_exec, _s3h_expert_body, jnp.int32(0))

    # 7. Contiguous EP-1 / 2-Stage Paired Reduce-Scatter + Bridge/Flat All-Gather.
    if cfg.rs_wire_bf16:
        cast_rc = min(batch_size, 8)

        def _cast_rs_chunk(ci, _):
            r0 = ci * cast_rc
            rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                pl.ds(r0, cast_rc), :, :
            ].astype(jnp.bfloat16)
            return _

        lax.fori_loop(0, batch_size // cast_rc, _cast_rs_chunk, jnp.int32(0))
        rs_src_ref = rs_send_bf16
    else:
        rs_src_ref = accum_3d

    pl.semaphore_wait(barrier_sem, ep)

    if use_pair_rs:
        num_chips_rs = ep // 2
        core_rs = me & jnp.int32(1)
        chip_rs = me >> 1
        partner_rs = jnp.bitwise_xor(me, jnp.int32(1))

        p1_ops = []
        for c in range(num_chips_rs):
            dest_for_partner = jnp.int32(2 * c) + (jnp.int32(1) - core_rs)
            op_p = pltpu.make_async_remote_copy(
                rs_src_ref.at[
                    pl.ds(dest_for_partner * rows_per_rank, rows_per_rank), :, :
                ],
                rs_pair_recv.at[c],
                rs_pair_ssem.at[c],
                rs_pair_rsem.at[c],
                device_id={ep_axis_name: partner_rs},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_p.start()
            p1_ops.append(op_p)

        chip_ops = []
        for c in range(num_chips_rs):
            pltpu.make_async_copy(
                rs_pair_recv.at[c], rs_pair_recv.at[c], rs_pair_rsem.at[c]
            ).wait()
            target_rank = jnp.int32(2 * c) + core_rs

            @pl.when(c != chip_rs)
            def _sum_and_send_chip_fp8(c=c, target_rank=target_rank):
                r_off = target_rank * rows_per_rank
                pair_f32 = accum_3d[
                    pl.ds(r_off, rows_per_rank), :, :
                ] + rs_pair_recv[c].astype(jnp.float32)
                rs_chip_send[c] = pair_f32.astype(rs_chip_send.dtype)

            op_c = pltpu.make_async_remote_copy(
                rs_chip_send.at[c],
                rs_chip_recv.at[chip_rs],
                rs_chip_ssem.at[c],
                rs_chip_rsem,
                device_id={ep_axis_name: target_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            chip_ops.append((c, op_c))

            @pl.when(c != chip_rs)
            def _start_chip_rs_fp8(_op=op_c):
                _op.start()

        pltpu.make_async_copy(
            rs_chip_recv.at[pl.ds(0, num_chips_rs - 1)],
            rs_chip_recv.at[pl.ds(0, num_chips_rs - 1)],
            rs_chip_rsem,
        ).wait()

        red_rc = min(rows_per_rank, 4)

        def _reduce_owner_paired_fp8(ci, _):
            r0 = ci * red_rc
            acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
            for c in range(num_chips_rs):
                part_c = jnp.where(
                    c == chip_rs,
                    rs_pair_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                    rs_chip_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                )
                acc = acc + part_c
            owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
            return _

        lax.fori_loop(
            0, rows_per_rank // red_rc, _reduce_owner_paired_fp8, jnp.int32(0)
        )

        def _finish_rs_sends():
            for op_p in p1_ops:
                op_p.wait_send()
            for c, op_c in chip_ops:

                @pl.when(c != chip_rs)
                def _wc_fp8(_op=op_c):
                    _op.wait_send()

    else:
        rs_copies = []
        for d in range(ep):
            dst_slot = jnp.where(me < d, me, jnp.maximum(me - 1, 0))
            copy_op = pltpu.make_async_remote_copy(
                rs_src_ref.at[pl.ds(d * rows_per_rank, rows_per_rank), :, :],
                rs_recv_3d.at[dst_slot],
                rs_send_sem.at[d],
                rs_recv_sem,
                device_id={ep_axis_name: jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            rs_copies.append((d, copy_op))

            @pl.when(me != d)
            def _start_rs(_op=copy_op):
                _op.start()

        pltpu.make_async_copy(rs_recv_3d, rs_recv_3d, rs_recv_sem).wait()

        red_rc = min(rows_per_rank, 4)

        def _reduce_owner_chunk(ci, _):
            r0 = ci * red_rc
            acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
            for src in range(ep - 1):
                acc = acc + rs_recv_3d[src, pl.ds(r0, red_rc), :, :].astype(
                    jnp.float32
                )
            owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
            return _

        lax.fori_loop(
            0, rows_per_rank // red_rc, _reduce_owner_chunk, jnp.int32(0)
        )

        def _finish_rs_sends():
            for d, copy_op in rs_copies:

                @pl.when(me != d)
                def _wait_rs(_op=copy_op):
                    _op.wait_send()

    if return_sharded:
        _write_sharded_owner_out(output_ref, owner_3d, rows_per_rank)
    elif cfg.bridge_ag:
        num_chips = ep // 2
        my_chip = me >> 1
        my_par = me & 1
        mate = jnp.bitwise_xor(me, jnp.int32(1))
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]

        mate_op = pltpu.make_async_remote_copy(
            owner_3d,
            fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
            fan_send_sem.at[0],
            fan_recv_sem,
            device_id={ep_axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        mate_op.start()

        s1_ops = []
        for c in range(num_chips):
            dest = jnp.int32(c * 2) + my_par
            op = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
                fan_send_sem.at[1 + c],
                fan_recv_sem,
                device_id={ep_axis_name: dest},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s1_ops.append((c, op))

            @pl.when(my_chip != c)
            def _start_s1(_op=op):
                _op.start()

        s1_recv_rows = num_chips * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, s1_recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, s1_recv_rows), :, :],
            fan_recv_sem,
        ).wait()

        s2_ops = []
        for c in range(num_chips):
            src_rank = jnp.int32(c * 2) + my_par
            r_off = src_rank * rows_per_rank
            op2 = pltpu.make_async_remote_copy(
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_s2_send_sem.at[c],
                fan_s2_recv_sem,
                device_id={ep_axis_name: mate},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s2_ops.append((c, op2))

            @pl.when(my_chip != c)
            def _start_s2(_op=op2):
                _op.start()

        s2_recv_rows = (num_chips - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, s2_recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, s2_recv_rows), :, :],
            fan_s2_recv_sem,
        ).wait()

        for r0 in range(0, batch_size, 16):
            output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                pl.ds(r0, 16), :, :
            ].reshape(16, _H)

        mate_op.wait_send()
        for c, op in s1_ops:

            @pl.when(my_chip != c)
            def _wait_s1(_op=op):
                _op.wait_send()

        for c, op2 in s2_ops:

            @pl.when(my_chip != c)
            def _wait_s2(_op=op2):
                _op.wait_send()
    else:
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]
        fan_copies = []
        for dest in range(ep):
            fan_op = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
                fan_send_sem.at[dest],
                fan_recv_sem,
                device_id={ep_axis_name: jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            fan_copies.append((dest, fan_op))

            @pl.when(me != dest)
            def _start_fan(_op=fan_op):
                _op.start()

        recv_rows = (ep - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_recv_sem,
        ).wait()

        for r0 in range(0, batch_size, 16):
            output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                pl.ds(r0, 16), :, :
            ].reshape(16, _H)

        for dest, fan_op in fan_copies:

            @pl.when(me != dest)
            def _wait_fan(_op=fan_op):
                _op.wait_send()

    _finish_rs_sends()


def fused_decode_sparse_fp8_from_logits(
    rank: jax.Array,
    logits: jax.Array,
    bias: jax.Array,
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    norm: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    return_sharded: bool = False,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    config: Fp8SparseDecodeConfig | None = None,
) -> jax.Array:
    """Single-launch ``stream3_half`` / ``ring3_8m`` FP8 Decode MoE for ``L0`` & ``L61``."""
    if ep not in (8, 16):
        raise ValueError(f"FP8 sparse fused decode supports ep in (8, 16), got {ep}")
    batch_size = x.shape[0]
    if batch_size not in (16, 32, 64, 128, 256, 384):
        raise ValueError(
            f"FP8 sparse fused decode requires B in (16, 32, 64, 128, 256, 384), got {batch_size}"
        )
    if logits.shape != (batch_size, _E_TOTAL):
        raise ValueError(
            f"Expected logits shape ({batch_size}, {_E_TOTAL}), got {logits.shape}"
        )
    e_local = _E_TOTAL // ep
    w_groups = int(w1.shape[0])
    # Allow prefill zero-glue EPLB slabs `[e_local + R_pre, ...]` (`0 <= R_pre <= 8`)
    # directly with zero HBM slice/copy: decode only indexes base slots `[0, e_local)`.
    if (
        not (e_local <= w_groups <= e_local + 8)
        or w1.shape != (w_groups, _H, _N1)
        or w2.shape != (w_groups, _I, _H)
    ):
        raise ValueError(
            f"Unexpected FP8 weight shapes: w1={w1.shape}, w2={w2.shape} for ep={ep}"
        )
    if w1.dtype != jnp.float8_e4m3fn or w2.dtype != jnp.float8_e4m3fn:
        raise ValueError(
            f"Expected float8_e4m3fn weights, got w1={w1.dtype}, w2={w2.dtype}"
        )
    s1_in = s1[:e_local] if s1.shape[0] == w_groups and w_groups > e_local else s1
    s2_in = s2[:e_local] if s2.shape[0] == w_groups and w_groups > e_local else s2
    s1_2d = _normalize_fp8_scale_2d(s1_in, e_local, _N1)
    s2_2d = _normalize_fp8_scale_2d(s2_in, e_local, _H)

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    if has_replicas:
        if any(a is None for a in rep_args):
            raise ValueError(
                "All of rep_w1, rep_s1, rep_w2, rep_s2, replica_meta must be provided together"
            )
        num_replicas = int(rep_w1.shape[0])
        if (
            num_replicas not in (1, 2, 4, 6, 8)
            or rep_w1.shape != (num_replicas, _H, _N1)
            or rep_w2.shape != (num_replicas, _I, _H)
            or rep_w1.dtype != jnp.float8_e4m3fn
            or rep_w2.dtype != jnp.float8_e4m3fn
            or replica_meta.shape[0] != 4
        ):
            raise ValueError(
                f"Invalid FP8 replica bank shapes: rep_w1={rep_w1.shape}, meta={replica_meta.shape}"
            )
        rep_s1_2d = _normalize_fp8_scale_2d(
            rep_s1, num_replicas, _N1, pad_rows_to_8=True
        )
        rep_s2_2d = _normalize_fp8_scale_2d(
            rep_s2, num_replicas, _H, pad_rows_to_8=True
        )
    else:
        num_replicas = 0
        rep_s1_2d = rep_s2_2d = None

    if config is None:
        config = default_fp8_sparse_decode_config(batch_size, ep)

    bias_2d = bias.reshape(1, _E_TOTAL)
    norm_2d = norm.reshape(1, _H)
    x_2d = x.reshape(batch_size, _H) if x.ndim == 3 else x

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    is_dense = config.row_mode == "dense"
    is_ring3 = config.weight_mode == "ring3_8m"
    is_k2 = config.weight_mode == "w1_k2_w2_1x"
    max_exec = e_local + num_replicas
    scratch_shapes: list = []
    if is_dense:
        scratch_shapes.extend(
            dense_routing_scratch_shapes(
                batch_size, max_exec, num_experts=_E_TOTAL
            )
        )
    else:
        scratch_shapes.extend(
            routing_scratch_shapes(
                batch_size, max_exec, topk=_TOPK, num_experts=_E_TOTAL
            )
        )

    if has_replicas:
        moved_smem_len = 8 if (not is_dense and config.static_holder_mask) else _E_TOTAL
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((moved_smem_len,), jnp.int32),
                smem((e_local,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )
        if not is_dense and not config.static_holder_mask:
            scratch_shapes.append(smem((num_replicas,), jnp.int32))

    tile_m = config.tile_m
    if is_ring3:
        scratch_shapes.extend(
            [
                vmem((3, _H, _I // 2), jnp.float8_e4m3fn),
                sem((3,)),
                sem((1,)),
            ]
        )
    elif is_k2:
        scratch_shapes.extend(
            [
                vmem((2, _H // 2, _N1), jnp.float8_e4m3fn),
                vmem((_I, _H), jnp.float8_e4m3fn),
                vmem((batch_size if is_dense else tile_m, _N1), jnp.float32),
                sem((2,)),
                sem((1,)),
                sem((1,)),
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((4, _H, _I // 2), jnp.float8_e4m3fn),
                vmem((2, _I, _H // 2), jnp.float8_e4m3fn),
                sem((4,)),
                sem((2,)),
                sem((1,)),
            ]
        )
    y_dtype = jnp.float32 if config.norm_f32_stage else jnp.bfloat16
    y_stage_rows = max(16, max(config.m_buckets))
    if is_dense:
        scratch_shapes.extend(
            [
                vmem((batch_size, _I), jnp.bfloat16),
                vmem((batch_size, _H), y_dtype),
                vmem((1, _H), jnp.float32),
                vmem((batch_size, _H), jnp.float32),
                vmem((batch_size, 32, 128), jnp.float32),
            ]
        )
        if not return_sharded:
            scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    else:
        io_rows = batch_size if is_ring3 else tile_m
        scratch_shapes.extend(
            [
                vmem((batch_size, 32, 128), jnp.bfloat16),
                vmem((io_rows, 32, 128), jnp.bfloat16),
                vmem((io_rows, _H), jnp.bfloat16),
                vmem((io_rows, _I), jnp.bfloat16),
                vmem((y_stage_rows if is_ring3 else tile_m, _H), y_dtype),
                vmem((1, _H), jnp.float32),
                vmem((batch_size, 32, 128), jnp.float32),
            ]
        )

    rows_per_rank = batch_size // ep
    use_pair_rs = config.pair_rs and (ep in (8, 16))
    rs_wire_dtype = jnp.bfloat16 if config.rs_wire_bf16 else jnp.float32
    if config.rs_wire_bf16:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    if use_pair_rs:
        num_chips_rs = ep // 2
        scratch_shapes.extend(
            [
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((num_chips_rs,)),
                sem((num_chips_rs,)),
                sem((num_chips_rs,)),
                sem,
                vmem((rows_per_rank, 32, 128), jnp.bfloat16),
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((ep - 1, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((ep,)),
                sem,
                vmem((rows_per_rank, 32, 128), jnp.bfloat16),
            ]
        )
    if not return_sharded:
        if config.bridge_ag:
            num_chips = ep // 2
            scratch_shapes.extend(
                [sem((1 + num_chips,)), sem, sem((num_chips,)), sem]
            )
        else:
            scratch_shapes.extend([sem((ep,)), sem])

    # Notice: s1_2d and s2_2d (0.75 MiB total) are passed via v_spec!
    base_in_specs = (
        v_spec,  # logits
        v_spec,  # bias
        v_spec,  # x
        h_spec,  # w1
        v_spec,  # s1_2d
        h_spec,  # w2
        v_spec,  # s2_2d
        v_spec,  # norm
    )
    if has_replicas:
        in_specs = base_in_specs + (h_spec, v_spec, h_spec, v_spec)
        num_scalar_prefetch = 2
        call_args = (
            rank,
            replica_meta,
            logits,
            bias_2d,
            x_2d,
            w1,
            s1_2d,
            w2,
            s2_2d,
            norm_2d,
            rep_w1,
            rep_s1_2d,
            rep_w2,
            rep_s2_2d,
        )
    else:
        in_specs = base_in_specs
        num_scalar_prefetch = 1
        call_args = (
            rank,
            logits,
            bias_2d,
            x_2d,
            w1,
            s1_2d,
            w2,
            s2_2d,
            norm_2d,
        )

    out_rows = rows_per_rank if return_sharded else batch_size
    out_shape = jax.ShapeDtypeStruct((out_rows, _H), jnp.bfloat16)

    kernel_fn = functools.partial(
        _sparse_fp8_decode_kernel,
        ep=ep,
        batch_size=batch_size,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        cfg=config,
    )
    rep_tag = f"_r{num_replicas}" if has_replicas else ""
    out_tag = "_sh" if return_sharded else "_rep"
    fp8_w9_tag = (
        f"{'_sm1' if config.static_holder_mask else ''}"
        f"{f'_sp{config.spec_prime_parts}' if config.spec_prime_parts else ''}"
        f"{'_rf1' if config.route_fast else ''}"
        f"{f'_fp{config.fast_plan}' if config.fast_plan else ''}"
    )
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            grid=(1,),
            in_specs=in_specs,
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=(
            f"muse_sparse_fp8_ep{ep}_b{batch_size}"
            f"_{config.row_mode}_{config.weight_mode}_tm{tile_m}{rep_tag}{fp8_w9_tag}{out_tag}"
        ),
    )(*call_args)
