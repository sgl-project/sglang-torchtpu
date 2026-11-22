# SPDX-License-Identifier: Apache-2.0
"""Causal Chunked Prefill Streaming helpers for Muse Spark (TP=8 / EP=8, TPU7x).

Eliminates exposed A1 SparseCore `all-gather` latency in prefill by sharding the
sequence-parallel residual stream in **chunk-major** order ``[chunks, tp, c_k, H]``
across TP ranks so that:
  1. ``Kernel D[L]`` runs Part 0 (``c_0`` rows), starts ``ag0 = AG(Chunk 0)`` under
     ``Kernel D[L]`` Part 1 (``sum(c_{1..K-1})`` rows), hiding 100% of ``ag0`` when
     ``c_0 <= ~0.3 * t_shard``.
  2. In Layer ``L+1``, ``ag_{k+1} = AG(Chunk k+1)`` starts *before* ``A1(Chunk k)``
     and synchronizes *after* ``Causal RPAm(Chunk k)``, hiding ``ag_{k+1}`` under both
     ``A1(Chunk k)`` and ``RPAm(Chunk k)`` without any MoE weight reload penalty.
  3. After all ``K`` chunks complete, ``out_k`` and ``gate_k`` are re-indexed along
     ``axis=1`` to ``[tp, sum_k c_k, ...]`` so ``A3 v16`` ``all_to_all`` delivers the
     exact chunk-major rows back to each rank.
"""

from __future__ import annotations

from typing import NamedTuple


def resolve_causal_chunks(t_shard: int, spec: str = "") -> tuple[int, ...]:
    """Return per-rank chunk row counts summing to ``t_shard`` (all >=64 & 64-aligned)."""
    spec_s = spec.strip()
    if spec_s:
        raw = tuple(int(x) for x in spec_s.split(",") if x.strip())
        if len(raw) >= 2 and sum(raw) == t_shard and all(c >= 64 and c % 64 == 0 for c in raw):
            return raw
        if len(raw) >= 2 and t_shard > sum(raw) and t_shard % sum(raw) == 0:
            scale = t_shard // sum(raw)
            scaled = tuple(c * scale for c in raw)
            if all(c >= 64 and c % 64 == 0 for c in scaled):
                return scaled
    if t_shard == 3072:
        return (512, 1280, 1280)
    if t_shard == 2048:
        return (512, 768, 768)
    if t_shard == 1024:
        return (256, 384, 384)
    if t_shard == 512:
        return (128, 192, 192)
    if t_shard >= 384 and t_shard % 64 == 0:
        n_blk = t_shard // 64
        c0 = max(1, n_blk // 4)
        rem = n_blk - c0
        c1 = rem // 2
        c2 = rem - c1
        if c0 >= 1 and c1 >= 1 and c2 >= 1:
            return (c0 * 64, c1 * 64, c2 * 64)
    if t_shard >= 128 and t_shard % 128 == 0:
        return (t_shard // 2, t_shard - t_shard // 2)
    return (t_shard,)


def to_chunk_major_perm(tp_size: int, chunks: tuple[int, ...]) -> list[int]:
    """Global token index list mapping contiguous ``[T]`` -> chunk-major ``[tp, sum c_k]``.

    Rank ``r`` owns rows ``[sum(chunks[:k]) .. sum(chunks[:k+1]))`` corresponding to
    contiguous global tokens ``[T_start_k + r * c_k .. T_start_k + (r + 1) * c_k)``.
    """
    t_offs = []
    cur = 0
    for c_k in chunks:
        t_offs.append(cur)
        cur += tp_size * c_k
    perm: list[int] = []
    for r in range(tp_size):
        for t0, c_k in zip(t_offs, chunks):
            start = t0 + r * c_k
            perm.extend(range(start, start + c_k))
    return perm


def from_chunk_major_perm(tp_size: int, chunks: tuple[int, ...]) -> list[int]:
    """Inverse permutation mapping flattened chunk-major ``[tp, sum c_k]`` -> ``[T]``."""
    perm = to_chunk_major_perm(tp_size, chunks)
    inv = [0] * len(perm)
    for dst, src in enumerate(perm):
        inv[src] = dst
    return inv


def chunk_major_global_token_ids_jax(tp_size: int, chunks: tuple[int, ...]):
    """Return ``int32[tp_size, sum(chunks)]`` of original global token IDs per rank row."""
    import jax.numpy as jnp

    parts = []
    t_off = 0
    for c_k in chunks:
        global_c = tp_size * c_k
        parts.append(
            (t_off + jnp.arange(global_c, dtype=jnp.int32)).reshape(tp_size, c_k)
        )
        t_off += global_c
    return jnp.concatenate(parts, axis=1)


def permute_to_chunk_major_jax(x, tp_size: int, chunks: tuple[int, ...]):
    """Permute leading axis ``T`` of ``x`` into flattened ``[tp, sum c_k, ...]`` order."""
    import jax.numpy as jnp

    tail = x.shape[1:]
    parts = []
    off = 0
    for c_k in chunks:
        seg = tp_size * c_k
        parts.append(x[off : off + seg].reshape(tp_size, c_k, *tail))
        off += seg
    return jnp.concatenate(parts, axis=1).reshape(x.shape[0], *tail)


def unpermute_from_chunk_major_jax(x, tp_size: int, chunks: tuple[int, ...]):
    """Inverse of ``permute_to_chunk_major_jax`` restoring contiguous ``[0..T)`` order."""
    import jax.numpy as jnp

    tail = x.shape[1:]
    t_shard = sum(chunks)
    x_3d = x.reshape(tp_size, t_shard, *tail)
    parts = []
    off = 0
    for c_k in chunks:
        parts.append(x_3d[:, off : off + c_k].reshape(tp_size * c_k, *tail))
        off += c_k
    return jnp.concatenate(parts, axis=0)


class ChunkRpaMetaTorch(NamedTuple):
    kv_lens: object
    cu_q_lens: object
    cu_kv_lens: object
    page_indices: object
    swa_page_indices: object
    swa_group_page_indices: object
    distribution: object


def build_chunk_rpa_metas_torch(
    md,
    *,
    tp_size: int,
    chunks: tuple[int, ...],
) -> tuple[ChunkRpaMetaTorch, ...]:
    """Slice ``PallasForwardMetadata`` on-device into compacted per-chunk causal RPA metadata."""
    import torch

    bs = int(md.kv_lens.shape[0])
    q0 = md.cu_q_lens[:-1]
    q1 = md.cu_q_lens[1:]
    page_mat = md.page_indices.view(bs, -1)
    swa_mat = (
        md.swa_page_indices.view(bs, -1)
        if getattr(md, "swa_page_indices", None) is not None
        else None
    )
    swa_g_mat = (
        md.swa_group_page_indices.view(md.swa_group_page_indices.shape[0], bs, -1)
        if getattr(md, "swa_group_page_indices", None) is not None
        else None
    )
    zero_i32 = md.kv_lens.new_zeros((1,), dtype=torch.int32)
    iota = torch.arange(bs, device=md.kv_lens.device, dtype=torch.int32)
    pps_full = page_mat.shape[1]
    combined_pg = (
        torch.cat([page_mat, swa_mat], dim=1) if swa_mat is not None else page_mat
    )
    metas = []
    t_start = 0
    for c_k in chunks:
        c_tok = tp_size * c_k
        t_end = t_start + c_tok
        lo = q0.clamp(min=t_start, max=t_end)
        hi = q1.clamp(min=t_start, max=t_end)
        q_len_k = (hi - lo).clamp(min=0)
        active = q_len_k > 0
        rem_q = (q1 - hi).clamp(min=0)
        kv_len_k = torch.where(active, (md.kv_lens - rem_q).clamp(min=0), 0)
        # Sort-free stable active-first partition via prefix counts + [bs, bs] one-hot map.
        # Pass explicit dtype=torch.int32 to .sum() so PyTorch never promotes int32 -> int64!
        act_i32 = active.to(torch.int32)
        inact_i32 = 1 - act_i32
        n_act = act_i32.sum(dtype=torch.int32)
        dst_pos = torch.where(
            active,
            torch.cumsum(act_i32, dim=0, dtype=torch.int32) - 1,
            n_act + torch.cumsum(inact_i32, dim=0, dtype=torch.int32) - 1,
        )
        perm_oh_i32 = (dst_pos.unsqueeze(0) == iota.unsqueeze(1)).to(torch.int32)
        order = (perm_oh_i32 * iota.unsqueeze(0)).sum(dim=1, dtype=torch.int32).long()
        q_c = (perm_oh_i32 * q_len_k.to(torch.int32).unsqueeze(0)).sum(
            dim=1, dtype=torch.int32
        )
        kv_c = (perm_oh_i32 * kv_len_k.to(torch.int32).unsqueeze(0)).sum(
            dim=1, dtype=torch.int32
        )
        cu_q_c = torch.cat(
            [zero_i32, torch.cumsum(q_c, dim=0, dtype=torch.int32)], dim=0
        )
        dist_c = torch.stack([zero_i32[0], zero_i32[0], n_act]).to(torch.int32)
        pg_sel = combined_pg.index_select(0, order)
        if swa_mat is not None:
            pg_c = pg_sel[:, :pps_full].reshape(-1)
            swa_c = pg_sel[:, pps_full:].reshape(-1)
        else:
            pg_c = pg_sel.reshape(-1)
            swa_c = None
        swa_g_c = (
            swa_g_mat.index_select(1, order).reshape(swa_g_mat.shape[0], -1)
            if swa_g_mat is not None
            else None
        )
        metas.append(
            ChunkRpaMetaTorch(
                kv_lens=kv_c,
                cu_q_lens=cu_q_c,
                cu_kv_lens=md.cu_kv_lens,
                page_indices=pg_c,
                swa_page_indices=swa_c,
                swa_group_page_indices=swa_g_c,
                distribution=dist_c,
            )
        )
        t_start = t_end
    return tuple(metas)


def slice_chunk_rpa_metadata_jax(
    kv_lens,
    cu_q_lens,
    cu_kv_lens,
    page_indices,
    *,
    t_start: int,
    t_end: int,
):
    """Pure-JAX compacted per-chunk ``(kv_lens, cu_q_lens, cu_kv_lens, page_indices, distribution)``."""
    import jax.numpy as jnp

    bs = kv_lens.shape[0]
    q0 = cu_q_lens[:-1]
    q1 = cu_q_lens[1:]
    lo = jnp.clip(q0, t_start, t_end)
    hi = jnp.clip(q1, t_start, t_end)
    q_len_k = jnp.maximum(0, hi - lo)
    active = q_len_k > 0
    rem_q = jnp.maximum(0, q1 - hi)
    kv_len_k = jnp.where(active, jnp.maximum(0, kv_lens - rem_q), 0)
    # Sort-free stable active-first partition (active sequences placed first in original order).
    act_i32 = active.astype(jnp.int32)
    inact_i32 = jnp.int32(1) - act_i32
    n_act = jnp.sum(act_i32, dtype=jnp.int32)
    dst_pos = jnp.where(
        active,
        jnp.cumsum(act_i32, dtype=jnp.int32) - jnp.int32(1),
        n_act + jnp.cumsum(inact_i32, dtype=jnp.int32) - jnp.int32(1),
    )
    iota = jnp.arange(bs, dtype=jnp.int32)
    perm_oh = (dst_pos[None, :] == iota[:, None]).astype(jnp.int32)
    order = jnp.sum(perm_oh * iota[None, :], axis=1, dtype=jnp.int32)
    q_c = jnp.sum(perm_oh * q_len_k.astype(jnp.int32)[None, :], axis=1, dtype=jnp.int32)
    kv_c = jnp.sum(
        perm_oh * kv_len_k.astype(jnp.int32)[None, :], axis=1, dtype=jnp.int32
    )
    cu_q_c = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_c, dtype=jnp.int32)], axis=0
    )
    dist_c = jnp.array([0, 0, n_act], dtype=jnp.int32)
    pg_c = page_indices.reshape(bs, -1)[order].reshape(-1)
    return kv_c, cu_q_c, cu_kv_lens, pg_c, dist_c


def causal_chunk_a1_rpa_step_jax(
    ag0_or_none,
    attn_in_rest,
    wqkvg,
    wqkvg_scale,
    cos_chunks,
    sin_chunks,
    kv_cache,
    chunk_metas,
    rpa_chunk_fn,
    *,
    chunks: tuple[int, ...],
    h_q_local: int,
    h_kv_local: int,
    tp_axis: str = "model",
    use_xor1_ag: bool = False,
):
    """Run causal chunked ``A1 + RPAm`` overlapping Chunk ``k+1``'s AG with Chunk ``k`` compute.

    ``attn_in_rest`` is already static-FP8 quantized ``[t_shard - lead, H]`` when
    ``ag0_or_none is not None``, or full bf16/fp8 ``[t_shard, H]`` when ``ag0_or_none is None``.
    Returns ``(out_for_a3 [T, h_q_local*64], gate_for_a3 [T, h_q_local*64], kv_cache)``.
    """
    import jax.numpy as jnp
    from jax import lax

    from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather
    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

    tp = cos_chunks[0].shape[0]
    q_size = h_q_local * 64
    _fp8_max = 448.0

    def _ag(x_in):
        if use_xor1_ag:
            return xor1_bridge_all_gather(x_in, axis_name=tp_axis, tiled=False)
        return lax.all_gather(x_in, tp_axis, axis=0, tiled=False)

    def _to_fp8(x_in):
        if x_in.dtype == jnp.float8_e4m3fn:
            return x_in
        return jnp.clip(
            x_in.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
            -_fp8_max,
            _fp8_max,
        ).astype(jnp.float8_e4m3fn)

    w_scale = wqkvg_scale * kernel_a1.STATIC_A1_ACT_SCALE
    x_fp8_rest = _to_fp8(attn_in_rest)
    if ag0_or_none is None:
        ag_cur = _ag(x_fp8_rest[: chunks[0]])
        rest_off = chunks[0]
    else:
        ag_cur = ag0_or_none
        rest_off = 0

    outs = []
    gates = []
    cur_kvc = kv_cache
    for k, c_k in enumerate(chunks):
        if k + 1 < len(chunks):
            c_next = chunks[k + 1]
            x_next = x_fp8_rest[rest_off : rest_off + c_next]
            rest_off += c_next
            x_next, ag_cur = lax.optimization_barrier((x_next, ag_cur))
            ag_next = _ag(x_next)

        q_k, gate_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
            ag_cur,
            wqkvg,
            w_scale,
            cos_chunks[k],
            sin_chunks[k],
            None,
            t_shard=c_k,
            row_off=0,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            rpa_layout=True,
            mode="all",
            tp_axis=tp_axis,
            tm=256 if c_k % 256 == 0 else 128,
        )
        c_tok = tp * c_k
        q5_k = q_k.reshape(1, c_tok, h_q_local // 2, 2, 128)
        kv_4d_k = kv_k.reshape(c_tok, 1, 4, 128)
        out_k, cur_kvc = rpa_chunk_fn(q5_k, kv_4d_k, cur_kvc, chunk_metas[k])
        outs.append(out_k.reshape(tp, c_k, q_size))
        gates.append(gate_k.reshape(tp, c_k, q_size))

        if k + 1 < len(chunks):
            ag_cur, cur_kvc, outs[-1], gates[-1] = lax.optimization_barrier(
                (ag_next, cur_kvc, outs[-1], gates[-1])
            )

    t_total = tp * sum(chunks)
    out_for_a3 = jnp.concatenate(outs, axis=1).reshape(t_total, q_size)
    gate_for_a3 = jnp.concatenate(gates, axis=1).reshape(t_total, q_size)
    return out_for_a3, gate_for_a3, cur_kvc
