# SPDX-License-Identifier: Apache-2.0
"""Materialize DFlash draft KV into the fused TPU pool.

Host side of kv_fused_rope_write: project K/V per draft layer, decompose the
destination slots into contiguous spans, gather the rope tables, then hand it
all to the single Pallas launch that rotates and scatters every layer. Extend
and verify differ only in how the spans come out, so one path serves both.
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.dflash import DFlashDraftModel


def dflash_write_draft_kv(
    *,
    draft_model: DFlashDraftModel,
    kv_pool: TpuFusedMHATokenToKVPool,
    ctx_hidden: torch.Tensor,
    ctx_positions: torch.Tensor,
    ctx_cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
) -> None:
    """Write context K/V into the draft pool, covering extend and verify.

    Projections stay per layer (they are GEMMs), but rope and the scatter run
    once for all layers, so K/V crosses HBM a single time.

    Args:
      draft_model: DFlash draft model owning the layers and the projections.
      kv_pool: Fused TPU KV pool the draft layers write into.
      ctx_hidden: Projected context hidden states [num_tokens, hidden].
      ctx_positions: Token position IDs [num_tokens].
      ctx_cache_loc: Flat destination slot per token [num_tokens]; on the
        verify path uncommitted slots are already masked to 0.
      cache_loc_2d: Verify's dense slots [bs, block_size], or None on extend.
        Kept unflattened so the span table can be built from it.
      commit_lens: Verify's accepted counts [bs], or None on extend.
    """
    # kv_fused_rope_write imports jax eagerly; keep the span math and the rope
    # gather above importable without it.
    from sglang.srt.hardware_backend.tpu.kv_fused_rope_write import (
        fused_rope_write_kv,
    )

    dst_starts, src_starts, span_lens, max_span_len = get_contiguous_slot_spans(
        page_size=int(kv_pool.page_size),
        cache_loc=ctx_cache_loc,
        cache_loc_2d=cache_loc_2d,
        commit_lens=commit_lens,
    )
    cos, sin = _rope_tables(
        draft_model=draft_model, positions=ctx_positions, dtype=ctx_hidden.dtype
    )

    attn0 = draft_model.layers[0].self_attn
    k = ctx_hidden.new_empty(
        (
            len(draft_model.layers),
            int(ctx_hidden.shape[0]),
            attn0.num_kv_heads,
            attn0.head_dim,
        )
    )
    v = torch.empty_like(k)
    for layer_idx, layer in enumerate(draft_model.layers):
        attn = layer.self_attn
        layer_ctx_hidden = draft_model.prepare_context_hidden_for_kv(layer, ctx_hidden)
        layer_k, layer_v = attn.kv_proj_only(layer_ctx_hidden)
        k[layer_idx] = attn.apply_k_norm(layer_k).view_as(k[layer_idx])
        v[layer_idx] = layer_v.view_as(v[layer_idx])

    fused_rope_write_kv(
        dst_starts=dst_starts,
        src_starts=src_starts,
        span_lens=span_lens,
        k=k,
        v=v,
        cos=cos,
        sin=sin,
        kv_bufs=kv_pool.kv_buffer,
        max_span_len=max_span_len,
    )


def dflash_write_draft_kv_ref(
    *,
    draft_model: DFlashDraftModel,
    kv_pool: TpuFusedMHATokenToKVPool,
    ctx_hidden: torch.Tensor,
    ctx_positions: torch.Tensor,
    ctx_cache_loc: torch.Tensor,
) -> None:
    """Per-layer equivalent of dflash_write_draft_kv, for tests and benchmarks.

    Independent route: each layer runs its own apply_k_rope and its own eager
    scatter, so neither the fused rotation nor the span addressing is in the
    path. That is what makes it a reference, and why it costs num_layers
    scatters and num_layers K/V passes over HBM.

    Takes flat per-token slots only, so verify's uncommitted tokens land on slot
    0 as garbage; dflash_write_draft_kv skips them instead.

    Args:
      draft_model: DFlash draft model owning the layers and the projections.
      kv_pool: Fused TPU KV pool the draft layers write into.
      ctx_hidden: Projected context hidden states [num_tokens, hidden].
      ctx_positions: Token position IDs [num_tokens].
      ctx_cache_loc: Flat destination slot per token [num_tokens].
    """
    for layer_idx, layer in enumerate(draft_model.layers):
        attn = layer.self_attn
        layer_ctx_hidden = draft_model.prepare_context_hidden_for_kv(layer, ctx_hidden)
        k, v = attn.kv_proj_only(layer_ctx_hidden)
        k = attn.apply_k_norm(k)
        k = attn.apply_k_rope(ctx_positions, k)
        _scatter_layer_kv(
            kv_pool=kv_pool,
            layer_idx=layer_idx,
            loc=ctx_cache_loc,
            k=k.view(-1, attn.num_kv_heads, attn.head_dim),
            v=v.view(-1, attn.num_kv_heads, attn.head_dim),
        )


def _scatter_layer_kv(
    *,
    kv_pool: TpuFusedMHATokenToKVPool,
    layer_idx: int,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Interleave one layer's K/V per head and scatter into its 5D pool buffer.

    Args:
      kv_pool: Fused TPU KV pool.
      layer_idx: Layer index within the pool.
      loc: Flat slot index per token [num_tokens].
      k: Rotated keys [num_tokens, head_num, head_dim].
      v: Values [num_tokens, head_num, v_head_dim].
    """
    buf = kv_pool.get_fused_kv_buffer(layer_idx)
    num_tokens = int(loc.shape[0])
    head_dim = kv_pool.head_dim

    if kv_pool.v_head_dim < head_dim:
        v = torch.nn.functional.pad(v, (0, head_dim - kv_pool.v_head_dim))
    kv = torch.stack([k, v], dim=2).reshape(num_tokens, kv_pool.head_num * 2, head_dim)
    kv = torch.nn.functional.pad(
        kv,
        (
            0,
            kv_pool.padded_head_dim - head_dim,
            0,
            kv_pool.num_kv_heads_x2 - kv_pool.head_num * 2,
        ),
    )
    buf.flatten(0, 1).index_copy_(
        0,
        loc.to(torch.long),
        kv.view(num_tokens, *buf.shape[2:]).to(kv_pool.store_dtype),
    )


def get_contiguous_slot_spans(
    *,
    page_size: int,
    cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Decompose a draft KV write into spans of contiguous destination slots.

    Verify: one span per request, length straight from commit_lens, so
    uncommitted slots go unwritten instead of being masked onto slot 0.

    Extend: spans break at page boundaries, since slots are consecutive only
    within a page -- the same invariant the RPA attention kernel relies on. The
    first span is short when the request resumes mid-page. The span count is
    fixed at the worst case over head alignments so the shape does not vary with
    alignment; unused spans come out empty and cost a no-op DMA.

    Args:
      page_size: Draft KV pool page size.
      cache_loc: Flat destination slot per token [num_tokens].
      cache_loc_2d: Verify's dense slots [bs, block_size], or None.
      commit_lens: Verify's accepted counts [bs], or None.

    Returns:
      Tuple of dst_starts, src_starts, span_lens (each [num_spans]) and the span
      length bound.
    """
    if cache_loc_2d is not None:
        assert commit_lens is not None
        block_size = int(cache_loc_2d.shape[1])
        src_starts = torch.arange(
            0,
            int(cache_loc_2d.shape[0]) * block_size,
            block_size,
            device=cache_loc_2d.device,
        )
        return cache_loc_2d[:, 0], src_starts, commit_lens, block_size

    num_tokens = int(cache_loc.shape[0])
    head = page_size - cache_loc[0] % page_size
    # Worst case is a one-token head, which pushes the tail one page further
    # than a page-aligned write needs.
    num_spans = (num_tokens + page_size - 2) // page_size + 1
    spans = torch.arange(num_spans, device=cache_loc.device)
    src_starts = (head + (spans - 1) * page_size).clamp(min=0, max=num_tokens)
    span_ends = (head + spans * page_size).clamp(max=num_tokens)
    dst_starts = cache_loc[src_starts.clamp(max=num_tokens - 1)]
    return dst_starts, src_starts, span_ends - src_starts, page_size


def _rope_tables(
    *, draft_model: DFlashDraftModel, positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the draft rope cos/sin rows for positions.

    cos_sin_cache is [max_position, rotary_dim], cos first half and sin second
    -- the only layout the fused write kernel rotates.

    Args:
      draft_model: DFlash draft model whose rope tables are gathered.
      positions: Token position IDs [num_tokens].
      dtype: Dtype of the keys the tables multiply.

    Returns:
      Tuple of cos and sin, each [num_tokens, rotary_dim // 2].
    """
    attn = draft_model.layers[0].self_attn
    rotary = attn.rotary_emb
    # mrope indexes by 3D positions; the table variants widen cos_sin_cache past
    # rotary_dim. Both would gather silently wrong rows here.
    assert (
        not isinstance(rotary, MRotaryEmbedding)
        and rotary.is_neox_style
        and rotary.rotary_dim == attn.head_dim
        and int(rotary.cos_sin_cache.shape[-1]) == attn.head_dim
    ), f"DFLASH draft rope must be neox over head_dim, got {type(rotary).__name__}"
    rows = rotary.cos_sin_cache.index_select(0, positions.to(torch.int64)).to(dtype)
    cos, sin = rows.chunk(2, dim=-1)
    return cos, sin
