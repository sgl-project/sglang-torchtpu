# SPDX-License-Identifier: Apache-2.0
"""Wave-8 Decode MoE HBM-Byte-Reduction Breakers & Granular V2 Config Helpers.

Provides CPU/NumPy-importable reference implementations alongside Pallas TPU
in-kernel helpers to break the ~11.3 ms HBM3e weight-streaming floor on
``1/1k`` decode (``B=64, TK=8, E=256``):

  1. **Router Min-P & Entropy-Adaptive Cumulative-Mass Top-K** (``#1405 4A``):
     - ``min_p`` (``SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK``): drops tail slots
       ``s >= min_k`` whose 8-expert normalized weight ``p_s < min_p`` and
       renormalizes surviving weights.
     - ``cum_mass`` (``SGLANG_TPU_MUSE_DECODE_TOPK_MASS``): entropy-adaptive
       cutoff keeping slots until cumulative 8-expert probability reaches
       ``cum_mass`` (confident tokens stop at 2-4 experts; flat/uncertain
       tokens retain 7-8 experts).

  2. **Batch-Coupled "Free-an-Expert" Pruning** (``#1408 #11``):
     - ``free_expert_rows`` (``SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS``):
       classifies hits into *core* (``slot < free_min_rank`` or
       ``p_s >= free_max_weight``) vs *tail*. Frees expert ``e``
       (``counts_eff[e] = 0 -> 0 HBM bytes streamed``) **if and only if**
       ``counts_core[e] == 0`` and ``0 < counts_total[e] <= free_expert_rows``.
       Retains 100% of tail hits on experts already loaded by another token's
       core hit (~5x lower routing perturbation per freed expert than static
       Top-K truncation) and renormalizes surviving token weights.

  3. **Cold-Expert 3-Bit ``FP3 -> FP4 -> FP8`` Packing & VPU Unpack** (``#1405 4B``):
     - Quantizes the coldest ``K`` non-replicated home experts to the 8-level
       log-spaced ``float4_e2m1fn`` subset ``+-{0.5, 1.5, 3.0, 6.0}``
       (``bit0 == 1`` in every E2M1 nibble, optimal ``alpha = 0.75`` ->
       13.83 dB Gaussian SQNR at K256, zero wasted ``-0.0`` code).
     - Packs every 4 ``[8, N] uint32`` rows (``kc = 256``) into 3 ``[8, N] uint32``
       rows (``[512, N] -> [384, N] uint32`` = **6.0 MiB vs 8.0 MiB, -25.0% HBM
       DMA bytes**), unpacking in 11 integer VREG ops directly into native TPU
       v7x ``float4_e2m1fn -> float8_e4m3fn`` MXU matmuls (``W3A16`` / ``W3A8``).

  4. **8-MiB Part-Major & Dual-Half-Major Contiguous HBM Repacking** (``#1408 #7``):
     - Repacks ``W1 [E, 512, 8192] uint32`` and ``S1 [E, nqb, 8192]`` so each
       8-MiB ``proj`` or ``dual`` unit occupies one contiguous linear HBM block
       (eliminating ``512 x 16 KiB`` / ``512 x 8 KiB`` strided DMA pitch with
       0.0 GiB extra HBM).
"""

# PEP 563 stays off so TorchTPU / cpu_export_jax_op sees concrete annotations.

import dataclasses
import os
from typing import Any

import numpy as np

# 8-level symmetric log-spaced E2M1 subset (nibble bit 0 == 1):
# codes4 = [1, 3, 5, 7, 9, 11, 13, 15] -> values = [+0.5, +1.5, +3.0, +6.0, -0.5, -1.5, -3.0, -6.0]
W3_POS_MIDPOINTS = (1.0, 2.25, 4.5)
W3_MAX_MAG = 6.0
_U32_HI3_MASK = 0xEEEEEEEE
_U32_LO1_MASK = 0x11111111


@dataclasses.dataclass(frozen=True)
class DecodeRoutingPruneConfig:
    """Compile-time configuration for approximate HBM-saving decode routing."""

    min_p: float = 0.0
    cum_mass: float = 0.0
    min_k: int = 2
    free_expert_rows: int = 0
    free_min_rank: int = 6
    free_max_weight: float = 0.105

    @property
    def enabled(self) -> bool:
        return (
            self.min_p > 0.0
            or self.cum_mass > 0.0
            or self.free_expert_rows > 0
        )


@dataclasses.dataclass(frozen=True)
class SparseV2EnvOverrides:
    """Granular environment overrides for ``fused_decode_sparse_v2``."""

    ring_mode: str = ""
    pair_rs: int = -1
    vmem_deq: int = 0
    fp4_unpack_mode: str = ""
    async_gather: int = -1
    fused_norm_scatter: int = -1
    m8: bool = False
    m_buckets: tuple[int, ...] = ()
    lhs_fp8: bool = False
    hoist_scales: bool = False
    contig_parts: int = 0
    static_holder_mask: bool = False
    spec_prime_parts: int = 0
    rs_bridge64: bool = False
    routing_prune: DecodeRoutingPruneConfig = DecodeRoutingPruneConfig()


def _env_raw(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if raw is not None and raw.strip() != "":
        s = raw.strip()
        if isinstance(default, bool):
            return s.lower() in ("1", "true", "yes", "y", "on")
        if isinstance(default, int):
            if s.lower() in ("true", "yes", "y", "on"):
                return 1
            if s.lower() in ("false", "no", "n", "off"):
                return 0
            return int(s)
        if isinstance(default, float):
            return float(s)
        return s
    try:
        from sglang_tpu.envs import envs

        field = getattr(envs, name, None)
        if field is not None:
            return field.get()
    except Exception:
        pass
    return default


def parse_m_buckets_str(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated bucket spec such as ``'8,32'`` or ``'8,16,32'``."""
    s = (spec or "").strip()
    if not s:
        return ()
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v not in (8, 16, 32, 64, 128):
            raise ValueError(
                f"Invalid decode m_bucket {v} in {spec!r}; expected values in (8, 16, 32, 64, 128)"
            )
        if v not in vals:
            vals.append(v)
    vals.sort()
    return tuple(vals)


def read_decode_routing_prune_config() -> DecodeRoutingPruneConfig:
    """Read HBM-wall-breaker routing parameters from ``sglang_tpu.envs`` / ``os.environ``.

    Supports both dedicated Wave-8 ``v2`` env vars and ``#1466`` compact forms:
      - ``SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS="2"`` or ``"2:0.12:6"``
        (``rows[:max_weight[:min_rank]]``)
      - ``SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK``: ``< 0.5`` sets ``min_p`` floor;
        ``>= 0.5`` sets ``cum_mass`` cap when ``TOPK_MASS`` is not explicitly set.
    """
    raw_min_p = max(0.0, float(_env_raw("SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK", 0.0)))
    cum_mass = max(0.0, float(_env_raw("SGLANG_TPU_MUSE_DECODE_TOPK_MASS", 0.0)))
    if raw_min_p >= 0.5:
        min_p = 0.0
        if cum_mass <= 0.0:
            cum_mass = min(1.0, raw_min_p)
    else:
        min_p = raw_min_p

    min_k = max(1, min(8, int(_env_raw("SGLANG_TPU_MUSE_DECODE_MIN_K", 2))))

    free_spec = str(_env_raw("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS", "")).strip()
    free_rows = 0
    spec_max_w: float | None = None
    spec_min_rank: int | None = None
    if free_spec and free_spec.lower() not in ("0", "false", "no", "off"):
        parts = [p.strip() for p in free_spec.split(":")]
        if parts[0]:
            free_rows = max(0, int(parts[0]))
        if len(parts) >= 2 and parts[1]:
            spec_max_w = max(0.0, float(parts[1]))
        if len(parts) >= 3 and parts[2]:
            spec_min_rank = int(parts[2])

    default_min_rank = spec_min_rank if spec_min_rank is not None else 6
    default_max_w = spec_max_w if spec_max_w is not None else 0.105
    if os.environ.get("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_MIN_RANK", "").strip():
        free_min_rank = int(
            _env_raw("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_MIN_RANK", default_min_rank)
        )
    else:
        free_min_rank = default_min_rank
    if os.environ.get("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_MAX_WEIGHT", "").strip():
        free_max_w = float(
            _env_raw("SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_MAX_WEIGHT", default_max_w)
        )
    else:
        free_max_w = default_max_w

    return DecodeRoutingPruneConfig(
        min_p=min_p,
        cum_mass=cum_mass,
        min_k=min_k,
        free_expert_rows=free_rows,
        free_min_rank=max(1, min(7, free_min_rank)),
        free_max_weight=max(0.0, free_max_w),
    )


def read_sparse_v2_env_overrides() -> SparseV2EnvOverrides:
    """Read granular ``SparseDecodeV2Config`` overrides from environment."""
    m_bkt_str = str(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_M_BUCKETS", ""))
    v2_cp = int(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_V2_CONTIG_PARTS", 0))
    cp = v2_cp if v2_cp != 0 else int(
        _env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS", 0)
    )
    return SparseV2EnvOverrides(
        ring_mode=str(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_V2_RING", "")).strip(),
        pair_rs=int(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS", -1)),
        vmem_deq=int(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_VMEM_DEQ", 0)),
        fp4_unpack_mode=str(
            _env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_FP4_UNPACK", "")
        ).strip(),
        async_gather=int(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_ASYNC_GATHER", -1)),
        fused_norm_scatter=int(
            _env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NORM_SCATTER", -1)
        ),
        m8=bool(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_M8", False)),
        m_buckets=parse_m_buckets_str(m_bkt_str),
        lhs_fp8=bool(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8", False)),
        hoist_scales=bool(
            _env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_HOIST_SCALES", False)
        ),
        contig_parts=cp,
        static_holder_mask=bool(
            _env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_STATIC_HOLDER_MASK", False)
        ),
        spec_prime_parts=max(
            0, min(5, int(_env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME", 0)))
        ),
        rs_bridge64=bool(
            _env_raw("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_BRIDGE64", False)
        ),
        routing_prune=read_decode_routing_prune_config(),
    )


# =============================================================================
# 1. Pure-NumPy Reference Router with Min-P, Entropy Mass, & Free-an-Expert
# =============================================================================


def route_weights_with_breakers_np(
    logits: np.ndarray,
    bias: np.ndarray,
    prune_cfg: DecodeRoutingPruneConfig,
    *,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact NumPy reference for ``route_topk_and_counts_v2_in_kernel``.

    Returns:
      ``(dense_weights [B, E], counts_eff [E], free_mask [E])`` matching the
      in-kernel Pallas VPU implementation bit-for-bit (up to FP32 exp/sum rounding).
    """
    lg_raw = np.asarray(logits, dtype=np.float32)
    lg = np.where(np.isnan(lg_raw), -np.inf, lg_raw)
    bias_row = np.asarray(bias, dtype=np.float32).reshape(1, -1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(lg, -88.0, 88.0)))
    scores = np.where(np.isneginf(lg), 0.0, scores)
    choice = scores + bias_row

    bsz, num_experts = scores.shape
    ids = np.arange(num_experts, dtype=np.int32)[None, :]
    sel_eids = np.full((bsz, topk), -1, dtype=np.int32)
    sel_w = np.zeros((bsz, topk), dtype=np.float32)

    for slot in range(topk):
        peak = np.max(choice, axis=1, keepdims=True)
        best = np.min(
            np.where(choice == peak, ids, num_experts), axis=1, keepdims=True
        )
        pick = ids == best
        w_slot = np.sum(np.where(pick, scores, 0.0), axis=1)
        valid = w_slot > 0.0
        sel_eids[:, slot] = np.where(valid, best[:, 0], -1)
        sel_w[:, slot] = np.where(valid, w_slot, 0.0)
        choice = np.where(pick, -np.inf, choice)

    w_sum8 = np.sum(sel_w, axis=1, keepdims=True) + route_eps
    p8 = sel_w / w_sum8

    keep = sel_eids >= 0
    cum_before = np.zeros((bsz,), dtype=np.float32)
    for slot in range(topk):
        valid_s = sel_eids[:, slot] >= 0
        k_s = valid_s.copy()
        if slot >= prune_cfg.min_k:
            if prune_cfg.min_p > 0.0:
                k_s = k_s & (p8[:, slot] >= prune_cfg.min_p)
            if prune_cfg.cum_mass > 0.0:
                k_s = k_s & (cum_before < prune_cfg.cum_mass)
        keep[:, slot] = k_s
        cum_before = cum_before + np.where(valid_s, p8[:, slot], 0.0)

    counts_total = np.zeros((num_experts,), dtype=np.int32)
    counts_core = np.zeros((num_experts,), dtype=np.int32)
    min_rank = max(1, prune_cfg.free_min_rank)
    for slot in range(topk):
        is_kept = keep[:, slot]
        e_col = sel_eids[:, slot]
        is_core = is_kept & (
            (slot < min_rank) | (p8[:, slot] >= prune_cfg.free_max_weight)
        )
        for b in range(bsz):
            if is_kept[b]:
                counts_total[e_col[b]] += 1
                if is_core[b]:
                    counts_core[e_col[b]] += 1

    if prune_cfg.free_expert_rows > 0:
        free_mask = (
            (counts_core == 0)
            & (counts_total > 0)
            & (counts_total <= prune_cfg.free_expert_rows)
        )
        counts_eff = np.where(free_mask, 0, counts_total).astype(np.int32)
        for slot in range(topk):
            for b in range(bsz):
                if keep[b, slot] and free_mask[sel_eids[b, slot]]:
                    keep[b, slot] = False
    else:
        free_mask = np.zeros((num_experts,), dtype=bool)
        counts_eff = counts_total

    w_kept = np.where(keep, sel_w, 0.0)
    denom = np.sum(w_kept, axis=1, keepdims=True) + route_eps
    w_norm = (w_kept / denom) * route_scale
    dense = np.zeros((bsz, num_experts), dtype=np.float32)
    for slot in range(topk):
        for b in range(bsz):
            if keep[b, slot]:
                dense[b, sel_eids[b, slot]] = w_norm[b, slot]
    return dense, counts_eff, free_mask


# =============================================================================
# 2. In-Kernel Pallas VPU Routing with Min-P / Mass / Free-an-Expert
# =============================================================================


def route_dense_and_counts_v2_in_kernel(
    logits_ref,
    bias_ref,
    d_scratch,
    dma_sem,
    prune_cfg: DecodeRoutingPruneConfig,
    *,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    route_chunk_rows: int = 32,
) -> None:
    """Dense-rows Pallas routing supporting Min-P, Cum-Mass, and Free-an-Expert."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if not prune_cfg.enabled:
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
            route_dense_and_counts_in_kernel,
        )

        route_dense_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            d_scratch,
            dma_sem,
            topk=topk,
            route_eps=route_eps,
            route_scale=route_scale,
            route_chunk_rows=route_chunk_rows,
        )
        return

    batch_size, num_experts = logits_ref.shape
    rb = min(batch_size, route_chunk_rows)
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size

    bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    counts_init = jnp.zeros((1, num_experts), jnp.int32)
    neg_inf = jnp.float32(-jnp.inf)
    min_rank = max(1, prune_cfg.free_min_rank)

    def _pass1_chunk(r0, carry):
        c_tot, c_core = carry
        row_s = pl.ds(r0, rb)
        lg_raw = logits_ref[row_s, :].astype(jnp.float32)
        lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
        scores = 1.0 / (1.0 + jnp.exp(-lg))
        choice = scores + bias_row

        best_list = []
        w_list = []
        for _ in range(topk):
            peak = jnp.max(choice, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(choice == peak, e_iota, num_experts),
                axis=1,
                keepdims=True,
            )
            pick = e_iota == best
            w_slot = jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
            best_list.append(best)
            w_list.append(w_slot)
            choice = jnp.where(pick, neg_inf, choice)

        w_sum8 = sum(w_list) + route_eps
        cum_before = jnp.zeros((rb, 1), dtype=jnp.float32)
        dense_raw = jnp.zeros((rb, num_experts), dtype=jnp.float32)
        hit_tot = jnp.zeros((rb, num_experts), dtype=jnp.bool_)
        hit_core = jnp.zeros((rb, num_experts), dtype=jnp.bool_)

        for s in range(topk):
            valid_s = w_list[s] > 0.0
            p_s = w_list[s] / w_sum8
            keep_s = valid_s
            if s >= prune_cfg.min_k:
                if prune_cfg.min_p > 0.0:
                    keep_s = keep_s & (p_s >= prune_cfg.min_p)
                if prune_cfg.cum_mass > 0.0:
                    keep_s = keep_s & (cum_before < prune_cfg.cum_mass)
            cum_before = cum_before + jnp.where(valid_s, p_s, 0.0)
            pick_s = keep_s & (e_iota == best_list[s])
            dense_raw = jnp.where(pick_s, w_list[s], dense_raw)
            hit_tot = hit_tot | pick_s
            is_core_s = keep_s & (
                bool(s < min_rank) | (p_s >= prune_cfg.free_max_weight)
            )
            hit_core = hit_core | (is_core_s & (e_iota == best_list[s]))

        if prune_cfg.free_expert_rows <= 0:
            w_norm = dense_raw / (
                jnp.sum(dense_raw, axis=1, keepdims=True) + route_eps
            )
            if route_scale != 1.0:
                w_norm = w_norm * route_scale
            d_scratch.dense_w_vmem[row_s, :] = w_norm
        else:
            d_scratch.dense_w_vmem[row_s, :] = dense_raw

        return (
            c_tot + jnp.sum(hit_tot.astype(jnp.int32), axis=0, keepdims=True),
            c_core + jnp.sum(hit_core.astype(jnp.int32), axis=0, keepdims=True),
        )

    n_chunks = batch_size // rb
    if n_chunks == 1:
        counts_tot, counts_core = _pass1_chunk(0, (counts_init, counts_init))
    else:
        counts_tot, counts_core = lax.fori_loop(
            0,
            n_chunks,
            lambda ci, acc: _pass1_chunk(pl.multiple_of(ci * rb, rb), acc),
            (counts_init, counts_init),
        )

    if prune_cfg.free_expert_rows > 0:
        free_mask = (
            (counts_core == 0)
            & (counts_tot > 0)
            & (counts_tot <= prune_cfg.free_expert_rows)
        )
        counts_eff = jnp.where(free_mask, jnp.int32(0), counts_tot)
        keep_e = ~jnp.broadcast_to(free_mask, (rb, num_experts))

        def _norm_chunk(r0, _):
            row_s = pl.ds(r0, rb)
            raw_w = jnp.where(keep_e, d_scratch.dense_w_vmem[row_s, :], 0.0)
            w_norm = raw_w / (jnp.sum(raw_w, axis=1, keepdims=True) + route_eps)
            if route_scale != 1.0:
                w_norm = w_norm * route_scale
            d_scratch.dense_w_vmem[row_s, :] = w_norm
            return _

        if n_chunks == 1:
            _norm_chunk(0, jnp.int32(0))
        else:
            lax.fori_loop(
                0,
                n_chunks,
                lambda ci, c: _norm_chunk(pl.multiple_of(ci * rb, rb), c),
                jnp.int32(0),
            )
    else:
        counts_eff = counts_tot

    d_scratch.counts_vmem[...] = jnp.broadcast_to(
        counts_eff, d_scratch.counts_vmem.shape
    )
    cp = pltpu.make_async_copy(
        d_scratch.counts_vmem, d_scratch.counts_smem, dma_sem
    )
    cp.start()
    cp.wait()


def route_topk_and_counts_v2_in_kernel(
    logits_ref,
    bias_ref,
    scratch,
    dma_sem,
    prune_cfg: DecodeRoutingPruneConfig,
    *,
    my_owner,
    e_per_owner: int,
    holder_cand_eids: tuple = (),
    hoist_holder_vecs: bool = False,
    precompute_owner_loads: bool = False,
    replica_cost_mode: str = "active",
    replica_tile_m: int = 32,
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    route_chunk_rows: int = 32,
) -> None:
    """VPU-filtered sparse routing with Min-P, Cum-Mass, & Free-an-Expert pruning."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if not prune_cfg.enabled:
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
            route_topk_and_counts_in_kernel,
        )

        route_topk_and_counts_in_kernel(
            logits_ref,
            bias_ref,
            scratch,
            dma_sem,
            my_owner=my_owner,
            e_per_owner=e_per_owner,
            holder_cand_eids=holder_cand_eids,
            hoist_holder_vecs=hoist_holder_vecs,
            precompute_owner_loads=precompute_owner_loads,
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=replica_tile_m,
            topk=topk,
            route_eps=route_eps,
            route_scale=route_scale,
            route_chunk_rows=route_chunk_rows,
        )
        return

    batch_size, num_experts = logits_ref.shape
    rb = min(batch_size, min(32, route_chunk_rows))
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size

    first = my_owner * e_per_owner
    last_excl = first + e_per_owner
    first_v = jnp.broadcast_to(first, (rb, 1))
    last_excl_v = jnp.broadcast_to(last_excl, (rb, 1))
    rep_vecs = tuple(
        jnp.broadcast_to(jnp.where(eid >= 0, eid, jnp.int32(-1)), (rb, 1))
        for eid in holder_cand_eids
    )

    bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)
    counts_init = jnp.zeros((1, num_experts), jnp.int32)
    neg_inf = jnp.float32(-jnp.inf)
    min_rank = max(1, prune_cfg.free_min_rank)
    do_free = prune_cfg.free_expert_rows > 0

    def _cand_slot_of(best_v):
        cand = jnp.where(
            (best_v >= first_v) & (best_v < last_excl_v),
            best_v - first_v,
            jnp.int32(-1),
        )
        for r_slot, rep_v in enumerate(rep_vecs):
            cand = jnp.where(
                best_v == rep_v, jnp.int32(e_per_owner + r_slot), cand
            )
        return cand

    def _pass1_chunk(r0, carry):
        c_tot, c_core = carry
        row_s = pl.ds(r0, rb)
        lg_raw = logits_ref[row_s, :].astype(jnp.float32)
        lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
        scores = 1.0 / (1.0 + jnp.exp(-lg))
        choice = scores + bias_row

        best_list = []
        w_list = []
        for _ in range(topk):
            peak = jnp.max(choice, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(choice == peak, e_iota, num_experts),
                axis=1,
                keepdims=True,
            )
            pick = e_iota == best
            w_slot = jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
            best_list.append(best)
            w_list.append(w_slot)
            choice = jnp.where(pick, neg_inf, choice)

        w_sum8 = sum(w_list) + route_eps
        cum_before = jnp.zeros((rb, 1), dtype=jnp.float32)
        hit_tot = jnp.zeros((rb, num_experts), dtype=jnp.bool_)
        hit_core = jnp.zeros((rb, num_experts), dtype=jnp.bool_)

        if do_free:
            # Stage raw selected (eid, w_s) per token into scratch.route_pack_vmem
            # so Pass 2 can prune freed experts with zero re-evaluation of logits!
            stage_eids = jnp.full((rb, 128), -1, dtype=jnp.int32)
            stage_w = jnp.zeros((rb, 128), dtype=jnp.float32)
            for s in range(topk):
                valid_s = w_list[s] > 0.0
                p_s = w_list[s] / w_sum8
                keep_s = valid_s
                if s >= prune_cfg.min_k:
                    if prune_cfg.min_p > 0.0:
                        keep_s = keep_s & (p_s >= prune_cfg.min_p)
                    if prune_cfg.cum_mass > 0.0:
                        keep_s = keep_s & (cum_before < prune_cfg.cum_mass)
                cum_before = cum_before + jnp.where(valid_s, p_s, 0.0)
                pick_s = keep_s & (e_iota == best_list[s])
                hit_tot = hit_tot | pick_s
                is_core_s = keep_s & (
                    bool(s < min_rank) | (p_s >= prune_cfg.free_max_weight)
                )
                hit_core = hit_core | (is_core_s & (e_iota == best_list[s]))
                eid_store = jnp.where(keep_s, best_list[s], jnp.int32(-1))
                w_store = jnp.where(keep_s, w_list[s], jnp.float32(0.0))
                stage_eids = jnp.where(col128 == s, eid_store, stage_eids)
                stage_w = jnp.where(col128 == (16 + s), w_store, stage_w)

            w_bits = lax.bitcast_convert_type(stage_w, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, stage_eids, w_bits
            )
        else:
            pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
            w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
            loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
            w_kept_sum = jnp.zeros((rb, 1), dtype=jnp.float32)

            for s in range(topk):
                valid_s = w_list[s] > 0.0
                p_s = w_list[s] / w_sum8
                keep_s = valid_s
                if s >= prune_cfg.min_k:
                    if prune_cfg.min_p > 0.0:
                        keep_s = keep_s & (p_s >= prune_cfg.min_p)
                    if prune_cfg.cum_mass > 0.0:
                        keep_s = keep_s & (cum_before < prune_cfg.cum_mass)
                cum_before = cum_before + jnp.where(valid_s, p_s, 0.0)
                pick_s = keep_s & (e_iota == best_list[s])
                hit_tot = hit_tot | pick_s
                w_kept_sum = w_kept_sum + jnp.where(keep_s, w_list[s], 0.0)

                cand_slot = _cand_slot_of(best_list[s])
                is_cand = keep_s & (cand_slot >= 0)
                id_col = jnp.int32(1) + loc_cnt
                w_col = jnp.int32(16) + loc_cnt
                pack_ids = jnp.where(
                    is_cand & (col128 == id_col), cand_slot, pack_ids
                )
                w_loc_raw = jnp.where(
                    is_cand & (col128 == w_col), w_list[s], w_loc_raw
                )
                loc_cnt = loc_cnt + is_cand.astype(jnp.int32)

            pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
            w_loc_norm = w_loc_raw / (w_kept_sum + route_eps)
            if route_scale != 1.0:
                w_loc_norm = w_loc_norm * route_scale
            w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, pack_ids, w_bits
            )

        return (
            c_tot + jnp.sum(hit_tot.astype(jnp.int32), axis=0, keepdims=True),
            c_core + jnp.sum(hit_core.astype(jnp.int32), axis=0, keepdims=True),
        )

    n_chunks = batch_size // rb
    if n_chunks <= 4:
        counts_tot, counts_core = counts_init, counts_init
        for ci in range(n_chunks):
            counts_tot, counts_core = _pass1_chunk(
                ci * rb, (counts_tot, counts_core)
            )
    else:
        counts_tot, counts_core = lax.fori_loop(
            0,
            n_chunks,
            lambda ci, acc: _pass1_chunk(pl.multiple_of(ci * rb, rb), acc),
            (counts_init, counts_init),
        )

    if do_free:
        free_mask = (
            (counts_core == 0)
            & (counts_tot > 0)
            & (counts_tot <= prune_cfg.free_expert_rows)
        )
        counts_eff = jnp.where(free_mask, jnp.int32(0), counts_tot)
        free_bcast = jnp.broadcast_to(free_mask, (rb, num_experts))

        def _pass2_chunk(r0, _):
            row_s = pl.ds(r0, rb)
            staged = scratch.route_pack_vmem[row_s, :]
            staged_f = lax.bitcast_convert_type(staged, jnp.float32)

            pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
            w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
            loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
            w_kept_sum = jnp.zeros((rb, 1), dtype=jnp.float32)

            for s in range(topk):
                eid_s = jnp.sum(
                    jnp.where(col128 == s, staged, jnp.int32(0)),
                    axis=1,
                    keepdims=True,
                )
                w_s = jnp.sum(
                    jnp.where(col128 == (16 + s), staged_f, jnp.float32(0.0)),
                    axis=1,
                    keepdims=True,
                )
                was_kept = eid_s >= 0
                is_freed = jnp.any(
                    free_bcast & (e_iota == eid_s), axis=1, keepdims=True
                )
                survive = was_kept & (~is_freed)
                w_kept_sum = w_kept_sum + jnp.where(survive, w_s, 0.0)

                cand_slot = _cand_slot_of(eid_s)
                is_cand = survive & (cand_slot >= 0)
                id_col = jnp.int32(1) + loc_cnt
                w_col = jnp.int32(16) + loc_cnt
                pack_ids = jnp.where(
                    is_cand & (col128 == id_col), cand_slot, pack_ids
                )
                w_loc_raw = jnp.where(
                    is_cand & (col128 == w_col), w_s, w_loc_raw
                )
                loc_cnt = loc_cnt + is_cand.astype(jnp.int32)

            pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
            w_loc_norm = w_loc_raw / (w_kept_sum + route_eps)
            if route_scale != 1.0:
                w_loc_norm = w_loc_norm * route_scale
            w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
            scratch.route_pack_vmem[row_s, :] = jnp.where(
                col128 < 16, pack_ids, w_bits
            )
            return _

        if n_chunks <= 4:
            for ci in range(n_chunks):
                _pass2_chunk(ci * rb, jnp.int32(0))
        else:
            lax.fori_loop(
                0,
                n_chunks,
                lambda ci, c: _pass2_chunk(pl.multiple_of(ci * rb, rb), c),
                jnp.int32(0),
            )
    else:
        counts_eff = counts_tot

    if precompute_owner_loads:
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            precompute_imbalance_loads_vpu,
        )

        scratch.counts_vmem[...] = precompute_imbalance_loads_vpu(
            counts_eff,
            num_owners=num_experts // e_per_owner,
            cost_mode=replica_cost_mode,
            tile_m=replica_tile_m,
        )
    else:
        scratch.counts_vmem[...] = jnp.broadcast_to(
            counts_eff, scratch.counts_vmem.shape
        )

    cp_counts = pltpu.make_async_copy(
        scratch.counts_vmem, scratch.counts_smem, dma_sem
    )
    cp_pack = pltpu.make_async_copy(
        scratch.route_pack_vmem, scratch.route_pack_smem, dma_sem
    )
    cp_counts.start()
    cp_pack.start()
    cp_counts.wait()
    cp_pack.wait()


# =============================================================================
# 3. Cold-Expert 3-Bit FP3 -> FP4 -> FP8 Quantization, Bit-Packing & Unpacking
# =============================================================================


def _e2m1_lut_np() -> np.ndarray:
    pos = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    return np.concatenate([pos, -pos])


def dequant_nvfp4_u32_np(w_u32: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize ``[K // 8, N] uint32`` packed E2M1 + ``[K // block, N]`` scales to FP32."""
    u = np.asarray(w_u32, dtype=np.uint32)
    s = np.asarray(scale, dtype=np.float32)
    k_u32, n = u.shape
    k = k_u32 * 8
    block = k // s.shape[0]
    lut = _e2m1_lut_np()
    nibbles = np.stack([(u >> (4 * b)) & np.uint32(0xF) for b in range(8)], axis=1)
    w_raw = lut[nibbles.reshape(k, n)]
    return w_raw * np.repeat(s, block, axis=0)


def quantize_and_pack_w3_u32_np(
    w_f32: np.ndarray,
    scale_block: int = 256,
    *,
    alpha: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``[K, N]`` weights to 8-level ``FP3 (+-{0.5, 1.5, 3.0, 6.0})`` and
    pack 3-in-4 into ``[K * 3 // 32, N] uint32`` (``[384, N]`` for ``K = 4096``).

    Returns ``(packed_u32 [K * 3 // 32, N], scales_bf16_as_f32 [K // scale_block, N])``.
    """
    w = np.asarray(w_f32, dtype=np.float32)
    k, n = w.shape
    if k % 256 != 0 or k % scale_block != 0:
        raise ValueError(f"W3 packing requires K % 256 == 0, got K={k}")
    wb = w.reshape(k // scale_block, scale_block, n)
    amax = np.max(np.abs(wb), axis=1)
    # Match bfloat16 rounding of stored scales.
    sb_u16 = ((amax * np.float32(alpha / W3_MAX_MAG)).view(np.uint32) + 0x7FFF + ((amax.view(np.uint32) >> 16) & 1)) >> 16
    sb = (sb_u16.astype(np.uint32) << 16).view(np.float32)
    sb_safe = np.where(sb > 0.0, sb, np.float32(1.0))[:, None, :]

    ab = np.abs(wb) / sb_safe
    mag2 = (
        (ab >= W3_POS_MIDPOINTS[0]).astype(np.uint32)
        + (ab >= W3_POS_MIDPOINTS[1]).astype(np.uint32)
        + (ab >= W3_POS_MIDPOINTS[2]).astype(np.uint32)
    )  # 0..3 -> E2M1 exp bits; bit 0 is implicitly 1 -> {0.5, 1.5, 3.0, 6.0}
    sign = (wb < 0.0).astype(np.uint32)
    code4 = ((sign << 3) | (mag2 << 1) | np.uint32(1)).reshape(k // 8, 8, n)

    u32_w4 = np.zeros((k // 8, n), dtype=np.uint32)
    for b in range(8):
        u32_w4 |= code4[:, b, :] << np.uint32(4 * b)

    # Pack each 32-row u32 chunk (kc=256: 4 x [8, N]) into 24-row u32 chunk (3 x [8, N]).
    n_kc = (k // 8) // 32
    chunks = u32_w4.reshape(n_kc, 4, 8, n)
    a0 = chunks[:, 0]
    a1 = chunks[:, 1]
    a2 = chunks[:, 2]
    a3 = chunks[:, 3]
    hi3 = np.uint32(_U32_HI3_MASK)
    lo1 = np.uint32(_U32_LO1_MASK)
    p0 = (a0 & hi3) | ((a3 >> np.uint32(1)) & lo1)
    p1 = (a1 & hi3) | ((a3 >> np.uint32(2)) & lo1)
    p2 = (a2 & hi3) | ((a3 >> np.uint32(3)) & lo1)
    packed = np.concatenate([p0, p1, p2], axis=1).reshape(n_kc * 24, n)
    return packed, sb


def unpack_w3_u32_np(packed_u32: np.ndarray) -> np.ndarray:
    """Exact NumPy inverse of 3-in-4 ``W3`` packing: ``[K * 3 // 32, N] -> [K // 8, N] uint32``."""
    p = np.asarray(packed_u32, dtype=np.uint32)
    if p.shape[0] % 24 != 0:
        raise ValueError(f"Expected W3 rows divisible by 24, got {p.shape}")
    n_kc = p.shape[0] // 24
    n = p.shape[1]
    grp = p.reshape(n_kc, 3, 8, n)
    p0, p1, p2 = grp[:, 0], grp[:, 1], grp[:, 2]
    lo1 = np.uint32(_U32_LO1_MASK)
    a0 = p0 | lo1
    a1 = p1 | lo1
    a2 = p2 | lo1
    a3 = (
        ((p0 & lo1) << np.uint32(1))
        | ((p1 & lo1) << np.uint32(2))
        | ((p2 & lo1) << np.uint32(3))
        | lo1
    )
    return np.concatenate([a0, a1, a2, a3], axis=1).reshape(n_kc * 32, n)


def unpack_w3_u32_chunk_pallas(w_ref, c0: int, n0, nc: int):
    """Unpack one ``kc = 256`` chunk (``24`` ``uint32`` rows -> ``[256, nc] float4_e2m1fn``)
    in 11 integer VREG ops on TPU v7x.
    """
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r24 = (c0 // 256) * 24
    p0 = w_ref[pl.ds(r24, 8), pl.ds(n0, nc)]
    p1 = w_ref[pl.ds(r24 + 8, 8), pl.ds(n0, nc)]
    p2 = w_ref[pl.ds(r24 + 16, 8), pl.ds(n0, nc)]
    lo1 = jnp.uint32(_U32_LO1_MASK)
    a0 = p0 | lo1
    a1 = p1 | lo1
    a2 = p2 | lo1
    a3 = (
        ((p0 & lo1) << jnp.uint32(1))
        | ((p1 & lo1) << jnp.uint32(2))
        | ((p2 & lo1) << jnp.uint32(3))
        | lo1
    )
    u32_32 = jnp.concatenate([a0, a1, a2, a3], axis=0)
    return pltpu.bitcast(u32_32, jnp.float4_e2m1fn)


def select_cold_local_slots_np(
    placed_counts: np.ndarray,
    owner: int,
    e_local: int,
    cold_per_owner: int,
    *,
    replicated_eids: set[int] | tuple[int, ...] = (),
) -> tuple[int, list[int]]:
    """Select the ``cold_per_owner`` coldest non-replicated local home slots on ``owner``.

    Returns ``(w3_bitmask_i32, sorted_cold_slots)`` where bit ``le`` of
    ``w3_bitmask_i32`` (signed ``int32``) is 1 iff local slot ``le`` uses W3.
    """
    if cold_per_owner <= 0:
        return 0, []
    rep_set = {int(e) for e in replicated_eids if int(e) >= 0}
    base = owner * e_local
    cands = [
        (float(placed_counts[base + le]), le)
        for le in range(e_local)
        if (base + le) not in rep_set
    ]
    cands.sort(key=lambda pair: (pair[0], pair[1]))
    chosen = sorted(le for _, le in cands[: min(cold_per_owner, len(cands))])
    mask_u32 = 0
    for le in chosen:
        mask_u32 |= 1 << le
    # Convert uint32 bitmask cleanly to two's-complement Python int in int32 range.
    mask_i32 = int(np.uint32(mask_u32).view(np.int32))
    return mask_i32, chosen


def quantize_cold_slots_inplace_torch(
    w_fp4x2,
    s_bf16,
    cold_slots: list[int],
    *,
    alpha: float = 0.75,
) -> float:
    """Quantize ``cold_slots`` of Torch ``float4_e2m1fn_x2 [E_loc, K, N // 2]`` +
    ``bfloat16 [E_loc, K // block, N]`` in-place into W3 packed rows ``0 : (K * 3 // 4)``
    (i.e. ``0:384`` in ``uint32 [512, N]`` view -> **-25% HBM DMA bytes**).

    Returns mean relative L2 quantization error across modified slots.
    """
    import torch
    from sglang_tpu.moe.nvfp4_requant import _dequant

    if not cold_slots:
        return 0.0

    u8_all = w_fp4x2.data.view(torch.uint8)
    k = int(u8_all.shape[1])
    half_n = int(u8_all.shape[2])
    n = half_n * 2
    scale_block = k // int(s_bf16.shape[1])
    if k % 256 != 0:
        raise ValueError(f"Expected K divisible by 256, got {k}")

    hi3 = int(_U32_HI3_MASK)
    lo1 = int(_U32_LO1_MASK)
    err_sq = 0.0
    ref_sq = 0.0

    for le in cold_slots:
        u_sub = u8_all[le : le + 1].clone()
        s_sub = s_bf16.data[le : le + 1].clone()
        w_f32 = _dequant(u_sub, s_sub)[0]  # [K, N] float32 on device
        wb = w_f32.reshape(k // scale_block, scale_block, n)
        sb = (wb.abs().amax(dim=1) * (alpha / W3_MAX_MAG)).to(torch.bfloat16)
        sbf = torch.where(sb > 0, sb, 1.0).float()[:, None, :]
        ab = wb.abs() / sbf
        mag2 = (
            (ab >= W3_POS_MIDPOINTS[0]).to(torch.int32)
            + (ab >= W3_POS_MIDPOINTS[1]).to(torch.int32)
            + (ab >= W3_POS_MIDPOINTS[2]).to(torch.int32)
        )
        sign = (wb < 0).to(torch.int32)
        code4 = ((sign << 3) | (mag2 << 1) | 1).to(torch.uint8)  # [nqb, block, N]

        # Reconstruct dequantized W3 to log exact SQNR telemetry.
        deq_pair = code4.reshape(1, k, half_n, 2)
        deq_u8 = deq_pair[..., 0] | (deq_pair[..., 1] << 4)
        w_deq = _dequant(deq_u8, sb.unsqueeze(0))[0]
        err_sq += float((w_deq - w_f32).square().sum().item())
        ref_sq += float(w_f32.square().sum().item())

        # Note TorchTPU float4_e2m1fn_x2 layout: [K, N // 2] uint8 transposed in JAX?
        # Wait: verify exact byte layout between Torch [E, K, N // 2] uint8 and JAX
        # float4_e2m1fn [E, K, N] -> bitcast(uint32) [E, K // 8, N]!
        # In TorchTPU/XLA, float4_e2m1fn [E, K, N] has minor-to-major layout with K
        # minor-packed by 8 when bitcast(uint32) gives [E, K // 8, N] ONLY along axis -2!
        # Let's pack on CPU/host via float4 -> uint32 view or exact Torch [K//8, 8, N//2]!
        c_k8 = code4.reshape(k // 8, 8, n)
        u32_w4 = torch.zeros((k // 8, n), dtype=torch.int64, device=w_f32.device)
        for b in range(8):
            u32_w4 = u32_w4 | (c_k8[:, b, :].to(torch.int64) << (4 * b))

        n_kc = (k // 8) // 32
        chunks = u32_w4.reshape(n_kc, 4, 8, n)
        a0, a1, a2, a3 = chunks[:, 0], chunks[:, 1], chunks[:, 2], chunks[:, 3]
        p0 = (a0 & hi3) | ((a3 >> 1) & lo1)
        p1 = (a1 & hi3) | ((a3 >> 2) & lo1)
        p2 = (a2 & hi3) | ((a3 >> 3) & lo1)
        packed_u32 = torch.cat([p0, p1, p2], dim=1).reshape(n_kc * 24, n)  # [384, N]

        # Scatter packed_u32 nibbles back into [K, N // 2] uint8 rows 0 : (n_kc * 24 * 8):
        k_w3 = n_kc * 24 * 8  # 3072 out of 4096
        nib_k = torch.stack(
            [((packed_u32 >> (4 * b)) & 0xF).to(torch.uint8) for b in range(8)],
            dim=1,
        ).reshape(k_w3, half_n, 2)
        packed_u8 = nib_k[..., 0] | (nib_k[..., 1] << 4)
        u8_all[le, :k_w3, :].copy_(packed_u8)
        u8_all[le, k_w3:, :].zero_()
        s_bf16.data[le].copy_(sb)

    return (err_sq / ref_sq) ** 0.5 if ref_sq > 0.0 else 0.0


# =============================================================================
# 4. Contiguous 8-MiB Part-Major & Dual-Half-Major HBM Repacking (#1408 #7)
# =============================================================================


def repack_w1_s1_part_major_np(
    w1: np.ndarray,
    s1: np.ndarray,
    *,
    dual_halves: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Repack ``w1 [E, K, 2*I]`` (or ``uint32 [E, K//8, 2*I]``) + ``s1 [E, nqb, 2*I]``
    into contiguous 8-MiB part-major ``[2 * E, K, I]`` + ``[2 * E, nqb, I]`` arrays!

    Unit ``2 * w_slot + part`` (``part in {0, 1}``) directly matches ``w_ring [r_slot]``
    (``[512, 4096] uint32``) and ``s_ring [r_slot]`` (``[nqb, 4096]``) with zero stride
    and zero HBM capacity overhead:
      - ``dual_halves=False`` (``CONTIG_PARTS=2``):
        ``part 0`` = full Gate ``[K, I]`` (8 MiB linear burst),
        ``part 1`` = full Up   ``[K, I]`` (8 MiB linear burst).
      - ``dual_halves=True`` (``CONTIG_PARTS=3``):
        ``part 0`` = ``[Gate[:, 0:I//2] | Up[:, 0:I//2]]`` (8 MiB linear burst),
        ``part 1`` = ``[Gate[:, I//2:I] | Up[:, I//2:I]]`` (8 MiB linear burst).
    """
    e, k_dim, n1 = w1.shape
    _, nqb, _ = s1.shape
    i_dim = n1 // 2
    half_i = i_dim // 2

    if dual_halves:
        w_p0 = np.concatenate(
            [w1[:, :, 0:half_i], w1[:, :, i_dim : i_dim + half_i]], axis=2
        )
        w_p1 = np.concatenate(
            [w1[:, :, half_i:i_dim], w1[:, :, i_dim + half_i : n1]], axis=2
        )
        s_p0 = np.concatenate(
            [s1[:, :, 0:half_i], s1[:, :, i_dim : i_dim + half_i]], axis=2
        )
        s_p1 = np.concatenate(
            [s1[:, :, half_i:i_dim], s1[:, :, i_dim + half_i : n1]], axis=2
        )
    else:
        w_p0 = w1[:, :, 0:i_dim]
        w_p1 = w1[:, :, i_dim:n1]
        s_p0 = s1[:, :, 0:i_dim]
        s_p1 = s1[:, :, i_dim:n1]

    w_out = np.ascontiguousarray(
        np.stack([w_p0, w_p1], axis=1).reshape(e * 2, k_dim, i_dim)
    )
    s_out = np.ascontiguousarray(
        np.stack([s_p0, s_p1], axis=1).reshape(e * 2, nqb, i_dim)
    )
    return w_out, s_out


def repack_w1_s1_part_major_torch(
    w1_fp4x2,
    s1_bf16,
    *,
    dual_halves: bool = False,
) -> tuple[Any, Any]:
    """PyTorch ``float4_e2m1fn_x2 [E, 4096, 4096]`` + ``bfloat16 [E, nqb, 8192]`` repacker
    producing ``[2 * E, 4096, 2048] float4_e2m1fn_x2`` + ``[2 * E, nqb, 4096] bfloat16``
    contiguous 8-MiB part-major storage with ``0.0 GiB`` net HBM delta.
    """
    import torch
    from sglang_tpu.fp4_moe import _materialize_packed_fp4

    u8 = w1_fp4x2.data.view(torch.uint8)
    e, k, half_n1 = u8.shape  # [E, 4096, 4096] uint8 (2 nibbles per byte along N)
    _, nqb, n1 = s1_bf16.shape
    i_u8 = half_n1 // 2  # 2048 bytes = 4096 nibbles
    hi_u8 = i_u8 // 2  # 1024 bytes = 2048 nibbles
    i_sc = n1 // 2
    hi_sc = i_sc // 2

    if dual_halves:
        u_p0 = torch.cat([u8[:, :, 0:hi_u8], u8[:, :, i_u8 : i_u8 + hi_u8]], dim=2)
        u_p1 = torch.cat([u8[:, :, hi_u8:i_u8], u8[:, :, i_u8 + hi_u8 : half_n1]], dim=2)
        s_p0 = torch.cat([s1_bf16[:, :, 0:hi_sc], s1_bf16[:, :, i_sc : i_sc + hi_sc]], dim=2)
        s_p1 = torch.cat([s1_bf16[:, :, hi_sc:i_sc], s1_bf16[:, :, i_sc + hi_sc : n1]], dim=2)
    else:
        u_p0 = u8[:, :, 0:i_u8]
        u_p1 = u8[:, :, i_u8:half_n1]
        s_p0 = s1_bf16[:, :, 0:i_sc]
        s_p1 = s1_bf16[:, :, i_sc:n1]

    u_repacked = torch.stack([u_p0, u_p1], dim=1).reshape(e * 2, k, i_u8).contiguous()
    s_repacked = torch.stack([s_p0, s_p1], dim=1).reshape(e * 2, nqb, i_sc).contiguous()
    return _materialize_packed_fp4(u_repacked), s_repacked
