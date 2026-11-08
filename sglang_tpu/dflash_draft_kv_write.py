# SPDX-License-Identifier: Apache-2.0
"""Materialize DFlash draft KV into the fused or SEQ_ALONG_LANE TPU pool.

Host side of kv_fused_rope_write: project K/V per draft layer, decompose the
destination slots into contiguous spans, gather the rope tables, then hand it
all to the single Pallas launch that rotates and scatters every layer. A
SEQ_ALONG_LANE pool rotates K here and writes through kv_sal_span_write. Extend
and verify differ only in how the spans come out, so one path serves both.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.dflash import DFlashDraftModel
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
from sglang_tpu.models.dflash_mla import DFlashMLAAttention


def dflash_write_draft_kv(
    *,
    draft_model: DFlashDraftModel,
    kv_pool: TpuFusedMHATokenToKVPool,
    ctx_hidden: torch.Tensor,
    ctx_positions: torch.Tensor,
    ctx_cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
    max_num_reqs: int,
) -> None:
    """Write context K/V into the draft pool, covering extend and verify.

    Projections stay per layer (they are GEMMs), but rope and the scatter run
    once for all layers, so K/V crosses HBM a single time.

    Args:
      draft_model: DFlash draft model owning the layers and the projections.
      kv_pool: Fused or SEQ_ALONG_LANE TPU KV pool the draft layers write into.
      ctx_hidden: Projected context hidden states [num_tokens, hidden].
      ctx_positions: Token position IDs [num_tokens].
      ctx_cache_loc: Flat destination slot per token [num_tokens]; unused on
        the verify path, where the spans come from cache_loc_2d.
      cache_loc_2d: Verify's dense slots [bs, block_size], or None on extend.
        Kept unflattened so the span table can be built from it.
      commit_lens: Verify's accepted counts [bs], or None on extend.
      max_num_reqs: Upper bound on requests in one extend batch.
    """
    # Packed pools (SGLANG_TPU_KV_PACK_HEADS) hold a rank's heads side by side in
    # one row, a layout these writers don't produce.
    if kv_pool.kv_pack_heads > 1:
        raise NotImplementedError("DFLASH draft KV writes need an unpacked KV pool")
    # kv_fused_rope_write imports jax eagerly; keep the span math and the rope
    # gather above importable without it.
    from sglang_tpu.kv_fused_rope_write import (
        fused_rope_write_kv,
    )

    dst_starts, src_starts, span_lens, max_span_len = get_contiguous_slot_spans(
        page_size=int(kv_pool.page_size),
        cache_loc=ctx_cache_loc,
        cache_loc_2d=cache_loc_2d,
        commit_lens=commit_lens,
        max_num_reqs=max_num_reqs,
    )
    k, v, rope = _project_draft_kv(
        draft_model=draft_model, ctx_hidden=ctx_hidden, ctx_positions=ctx_positions
    )
    if isinstance(kv_pool, TpuSeqAlongLaneKVPool):
        from sglang_tpu.kv_sal_span_write import sal_span_write_kv

        sal_span_write_kv(
            dst_starts=dst_starts,
            src_starts=src_starts,
            span_lens=span_lens,
            k=k if rope is None else _neox_rotate(k, cos=rope[0], sin=rope[1]),
            v=v,
            kv_bufs=kv_pool.kv_buffer,
        )
        return
    # MLA K comes rotated: hand the fused kernel an identity rotation (exact).
    cos, sin = rope if rope is not None else _identity_rope(k)
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


def _project_draft_kv(
    *,
    draft_model: DFlashDraftModel,
    ctx_hidden: torch.Tensor,
    ctx_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Every layer's normed K and V [num_layers, num_tokens, kv_heads, head_dim].

    Also returns the (cos, sin) rope rows K still needs, or None when the
    projection already rotated it.
    """
    attn0 = draft_model.layers[0].self_attn
    # MLA drafts rope only their k_pe slice, which the fused kernel's neox
    # rotation over the whole head cannot express: rotate in the projection.
    prerotated = isinstance(attn0, DFlashMLAAttention)
    rope = None
    if not prerotated:
        rope = _rope_tables(
            draft_model=draft_model, positions=ctx_positions, dtype=ctx_hidden.dtype
        )
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
        if prerotated:
            layer_k, layer_v = attn.project_context_kv_rotated(
                ctx_positions, layer_ctx_hidden
            )
            k[layer_idx] = layer_k
            # V shares K's slot width in the fused pool.
            v[layer_idx] = torch.nn.functional.pad(
                layer_v, (0, attn.head_dim - attn.v_head_dim)
            )
            continue
        layer_k, layer_v = attn.kv_proj_only(layer_ctx_hidden)
        k[layer_idx] = attn.apply_k_norm(layer_k).view_as(k[layer_idx])
        v[layer_idx] = layer_v.view_as(v[layer_idx])
    return k, v, rope


def _identity_rope(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = k.new_ones((int(k.shape[1]), int(k.shape[-1]) // 2))
    return cos, torch.zeros_like(cos)


def _neox_rotate(
    k: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Neox rotation of [num_layers, num_tokens, heads, head_dim] K, as kv_fused_rope_write does."""
    half = cos.shape[-1]
    cos_b, sin_b = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = k[..., :half], k[..., half:]
    return torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)


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
    # Packed pools (SGLANG_TPU_KV_PACK_HEADS) hold a rank's heads side by side in
    # one row, a layout these writers don't produce.
    if kv_pool.kv_pack_heads > 1:
        raise NotImplementedError("DFLASH draft KV writes need an unpacked KV pool")
    if isinstance(kv_pool, TpuSeqAlongLaneKVPool):
        raise NotImplementedError("The reference scatter writes the fused layout only")
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
    max_num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Decompose a draft KV write into spans of contiguous destination slots.

    Slots are consecutive only within a page -- the invariant the RPA attention
    kernel relies on too -- so every span has to stay inside one page. Both
    branches pad their span count to a static worst case; unused spans come out
    empty and the kernel skips them.

    Args:
      page_size: Draft KV pool page size.
      cache_loc: Flat destination slot per token [num_tokens].
      cache_loc_2d: Verify's dense slots [bs, block_size], or None.
      commit_lens: Verify's accepted counts [bs], or None.
      max_num_reqs: Upper bound on requests in one extend batch.

    Returns:
      Tuple of dst_starts, src_starts, span_lens (each [num_spans]) and the span
      length bound.
    """
    if cache_loc_2d is not None:
        assert commit_lens is not None
        return _verify_spans(
            page_size=page_size, cache_loc_2d=cache_loc_2d, commit_lens=commit_lens
        )
    return _extend_spans(
        page_size=page_size, cache_loc=cache_loc, max_num_reqs=max_num_reqs
    )


def _verify_spans(
    *, page_size: int, cache_loc_2d: torch.Tensor, commit_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split each request's committed prefix at its page boundary.

    A commit is at most block_size tokens and block_size fits in a page, so a
    request needs two spans: what fits in the page its prefix ends on, and the
    remainder, which starts wherever the next page landed. Uncommitted slots go
    unwritten instead of being masked onto slot 0.

    Args:
      page_size: Draft KV pool page size.
      cache_loc_2d: Verify's dense slots [bs, block_size].
      commit_lens: Verify's accepted counts [bs].

    Returns:
      Tuple of dst_starts, src_starts, span_lens (each [2 * bs]) and the span
      length bound.
    """
    bs, block_size = (int(d) for d in cache_loc_2d.shape)
    assert block_size <= page_size, (
        f"DFLASH verify block ({block_size}) must fit in a page ({page_size}) "
        "for the two-span split to cover it"
    )
    # int32 end to end: the TPU backend fails to lower dynamic indexing over
    # int64, and slots fit comfortably.
    slots = cache_loc_2d.to(torch.int32)
    src_starts = torch.arange(
        0, bs * block_size, block_size, dtype=torch.int32, device=slots.device
    )
    head_dst = slots[:, 0]
    lens = commit_lens.to(torch.int32)
    head_lens = torch.minimum(lens, (page_size - head_dst % page_size).to(torch.int32))
    tail_lens = lens - head_lens
    # Where the spill lands, selected without a gather. A commit that stays in
    # its page picks some unwritten slot, or none at all when it fills the row;
    # either way the tail length is zero.
    offsets = torch.arange(block_size, dtype=torch.int32, device=slots.device)
    tail_dst = (slots * (offsets[None, :] == head_lens[:, None])).sum(
        dim=1, dtype=torch.int32
    )
    return (
        torch.cat([head_dst, tail_dst]),
        torch.cat([src_starts, src_starts + head_lens]),
        torch.cat([head_lens, tail_lens]),
        block_size,
    )


def _extend_spans(
    *, page_size: int, cache_loc: torch.Tensor, max_num_reqs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Cut a flat extend write into runs, reading the runs off the slots.

    An extend batch concatenates requests, and each one resumes at its own page
    offset, so the run structure cannot be derived from the first slot's
    alignment. Deriving it from the slots themselves covers page boundaries and
    request boundaries alike. A batch of R requests over N tokens yields at most
    N / page_size + R runs, which bounds the span count.

    Args:
      page_size: Draft KV pool page size.
      cache_loc: Flat destination slot per token [num_tokens].
      max_num_reqs: Upper bound on requests in one extend batch.

    Returns:
      Tuple of dst_starts, src_starts, span_lens (each [num_spans]) and the span
      length bound.
    """
    num_tokens = int(cache_loc.shape[0])
    device = cache_loc.device
    # Stay in int32 end to end: the TPU backend fails to lower a dynamic gather
    # over int64, and both slots and token counts fit comfortably.
    loc = cache_loc.to(torch.int32)
    # Pages handed out back to back are address-adjacent, so a page start ends a
    # run even where the addresses continue -- otherwise a span could outrun
    # max_span_len.
    breaks = (loc[1:] != loc[:-1] + 1) | (loc[1:] % page_size == 0)
    run_ids = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(breaks.to(torch.int32), dim=0, dtype=torch.int32),
        ]
    )
    num_spans = num_tokens // page_size + min(max_num_reqs, num_tokens) + 1
    # Page 0 is the pool's dummy page, so slot 0 only ever marks a token the
    # caller padded in. Park those past the last span rather than writing them.
    run_ids = torch.where(loc != 0, run_ids, num_spans)
    edges = torch.arange(num_spans + 1, device=device, dtype=torch.int32)
    # run_ids is non-decreasing over the real tokens, so counting the tokens
    # below each edge gives every run's source offset.
    bounds = (run_ids[None, :] < edges[:, None]).sum(dim=1, dtype=torch.int32)
    src_starts = bounds[:-1]
    return (
        torch.index_select(loc, 0, src_starts.clamp(max=num_tokens - 1)),
        src_starts,
        bounds[1:] - src_starts,
        page_size,
    )


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
