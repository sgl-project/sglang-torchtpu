"""Pieces of the Raiden pipeline transfer shared with its second engine:
native transfer ids, per-layer byte spans, destination pages, and the decode
state of one native transfer."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional, Sequence

from sglang.srt.disaggregation.utils import compute_mamba_state_slice_blocks
from sglang_tpu.disaggregation.raiden.reshard_schedule import (
    ReshardGeometry,
    head_bytes,
    head_slice_params,
)

# Two per-layer KV page layouts can cross a pipeline transfer:
#   rank 5, the fused pool:   [num_blocks, page_size, kv_heads_packed, kv_packing, head_dim]
#   rank 4, the RPA-seq pool: [num_blocks, 2 * kv_heads, head_dim_lanes, page_size]
# The rank-4 pool keeps each head's K and V rows adjacent and puts the token
# index on the lane axis, so one kv-head is a single contiguous run of a page.
# A block-major bundle (rank 6) holds every layer of a page in one row:
#   [num_blocks, layer_num, page_size, kv_heads_packed, kv_packing, head_dim]
_FUSED_KV_RANK = 5
_HEAD_MAJOR_KV_RANK = 4
_BLOCK_MAJOR_KV_RANK = 6


def _kv_rank(tensor) -> int:
    rank = len(tensor.shape)
    if rank not in (_FUSED_KV_RANK, _HEAD_MAJOR_KV_RANK, _BLOCK_MAJOR_KV_RANK):
        raise ValueError(
            "Raiden pipeline transfer requires rank-5 fused or rank-4 RPA-seq KV, "
            "or a rank-6 block-major bundle"
        )
    return rank


# Pool kinds whose pages have no kv-head axis, so every attention-TP rank holds
# the whole page: the latent and indexer caches of a split sparse-MLA pool
# (TpuSplitMLATokenToKVPool.transfer_page_entries).
_REPLICATED_KINDS = frozenset({"mla.nope", "mla.rope", "dsa.idx", "dsa.raw"})
# Kind prefix of linear-attention (GDN/KDA) state pools: "gdn.conv", "gdn.ssm".
_STATE_KIND_PREFIX = "gdn."


def pool_kind(tag: str) -> str:
    return tag.rsplit(".l", 1)[0]


def is_swa_pool(tag: str) -> bool:
    return pool_kind(tag).endswith("swa")


def is_draft_pool(tag: str) -> bool:
    return pool_kind(tag).startswith("draft.")


def is_replicated_pool(tag: str) -> bool:
    return pool_kind(tag).removeprefix("draft.") in _REPLICATED_KINDS


def is_state_pool(tag: str) -> bool:
    return pool_kind(tag).startswith(_STATE_KIND_PREFIX)


class StateShard(NamedTuple):
    """How one linear-attention state tensor, ``[rows, ...]``, is head-sharded
    over attention TP (MambaPool._iter_transfer_state_entries)."""

    # "conv" or "ssm".
    kind: str
    layer_id: int
    # The head-sharded axis of a row; None for state every rank holds whole.
    axis: Optional[int]
    # Full extents of the sub-blocks of that axis that are sharded one by one
    # (a conv row is [query | key | value]); None when it is sharded as a whole.
    groups: Optional[list[int]] = None


def _producer_tensor_kv_heads(tensor: Any) -> int:
    """KV head count of a TP=1 producer page tensor when the pool omitted head_num."""
    rank = _kv_rank(tensor)
    if rank == _HEAD_MAJOR_KV_RANK:
        return int(tensor.shape[1]) // 2
    if rank == _FUSED_KV_RANK:
        return int(tensor.shape[2]) * int(tensor.shape[3]) // 2
    raise ValueError("Block-major KV bundles do not slice kv-heads")


def source_spans(
    pools: Sequence[dict],
    full: list[int],
    swa: list[int],
    *,
    stripe: tuple[int, int] | None = None,
) -> list[Any]:
    """Describe whole pages in destination byte space for each locally owned layer.

    ``stripe`` is (rank, workers) on a producer whose workers all hold every
    page: page ``i`` of the transfer moves from worker ``i % workers``, so the
    workers' spans together cover each page once.
    """
    rank, workers = stripe or (0, 1)
    result = []
    for pool in pools:
        blocks = swa if is_swa_pool(pool["tag"]) else full
        stride = pool["block_stride_bytes"]
        spans = [
            SimpleNamespace(
                src_block_ordinal=index,
                src_offset_bytes=0,
                dst_block_index=0,
                dst_offset_bytes=index * stride,
                size_bytes=stride,
                src_stride_bytes=0,
                dst_stride_bytes=0,
                count=1,
            )
            for index in range(len(blocks))
            if index % workers == rank
        ]
        result.append(
            SimpleNamespace(
                tag=pool["tag"],
                block_ids=list(blocks),
                declared_bytes=len(spans) * stride,
                dst_space_version=1,
                spans=spans,
            )
        )
    return result


def state_shard_spans(
    pools: Sequence[dict], rows: list[int], *, rank: int, workers: int
) -> list[Any]:
    """Describe a request's state row as one head shard of the full-width row.

    The producer's ``workers`` attention-TP ranks each hold their heads' share
    of a row; an attention-TP 1 destination row holds every head. Each pool's
    ``shard`` (pool_transfer.state_pools) gives the row geometry, and
    compute_mamba_state_slice_blocks, which the other transfer backends
    follow too, says where rank ``rank``'s share of each sub-block goes: one
    span per sub-block, repeated over the rows before the sharded axis (the
    taps of a conv row). The spans are page-indexed: offsets inside the one
    destination row.
    """
    result = []
    for pool in pools:
        shard = pool["shard"]
        dim, unit, outer = shard["dim"], shard["unit_bytes"], shard["outer"]
        blocks = compute_mamba_state_slice_blocks(
            src_dim=dim,
            dst_dim=dim * workers,
            src_attn_tp_size=workers,
            dst_attn_tp_size=1,
            dst_tp_rank_in_group=0,
            local_tp_rank_in_group=rank,
            conv_shard_groups=shard["groups"],
        )
        result.append(
            SimpleNamespace(
                tag=pool["tag"],
                block_ids=list(rows),
                declared_bytes=len(rows) * outer * dim * unit,
                dst_space_version=0,
                spans=[
                    SimpleNamespace(
                        src_block_ordinal=ordinal,
                        src_offset_bytes=src * unit,
                        dst_block_index=ordinal,
                        dst_offset_bytes=dst * unit,
                        size_bytes=size * unit,
                        src_stride_bytes=dim * unit if outer > 1 else 0,
                        dst_stride_bytes=dim * workers * unit if outer > 1 else 0,
                        count=outer,
                    )
                    for ordinal in range(len(rows))
                    for src, dst, size in blocks
                ],
            )
        )
    return result


def state_destination(pools: Sequence[dict], rows: list[int]) -> dict:
    """Destination rows of a state transfer: every state pool lands in the
    request's rows, whole."""
    tags = [pool["tag"] for pool in pools] if rows else []
    return dict(
        transfer_pool_tags=tags,
        dst_device_block_ids=list(rows) * len(tags),
        dst_block_counts=[len(rows)] * len(tags),
        dst_skip_bytes=[0] * len(tags),
    )


def head_slice_spans(
    pools: Sequence[dict],
    tensors: Sequence[Any],
    full: list[int],
    swa: list[int],
    dst_attn_tp: int,
    total_kv_heads: int,
    *,
    draft_total_kv_heads: int | None = None,
    draft_swa_total_kv_heads: int | None = None,
) -> list[Any]:
    """Describe each destination rank's kv-head slice of every locally owned
    page: one strided span per page and rank, routed to that rank only."""
    result = []
    for pool, tensor in zip(pools, tensors):
        tag = pool["tag"]
        blocks = swa if is_swa_pool(tag) else full
        if _kv_rank(tensor) == _BLOCK_MAJOR_KV_RANK:
            raise ValueError(
                "A block-major KV bundle moves whole page rows; a "
                "tensor-parallel consumer needs per-layer KV pools"
            )
        # A separate draft model can have fewer or more KV heads than the target
        # (e.g. Muse Spark 16 KV heads with DFlash2 8 KV heads), while pdhid.*
        # rows are sharded by the target's total_kv_heads.
        if is_draft_pool(tag):
            draft_heads = (
                draft_swa_total_kv_heads
                if is_swa_pool(tag) and draft_swa_total_kv_heads is not None
                else draft_total_kv_heads
            )
            pool_kv_heads = (
                int(draft_heads)
                if draft_heads is not None
                else _producer_tensor_kv_heads(tensor)
            )
        else:
            pool_kv_heads = total_kv_heads
        if _kv_rank(tensor) == _HEAD_MAJOR_KV_RANK:
            result.append(
                _head_major_slice_spans(
                    pool=pool,
                    tensor=tensor,
                    blocks=blocks,
                    dst_attn_tp=dst_attn_tp,
                    total_kv_heads=pool_kv_heads,
                )
            )
            continue
        geometry = ReshardGeometry(
            src_tp=1,
            dst_tp=dst_attn_tp,
            hp_bytes=head_bytes(tensor),
            total_kv_heads=pool_kv_heads,
            kv_packing=int(tensor.shape[3]),
            element_size=tensor.element_size(),
        )
        page_tokens = int(tensor.shape[1])
        spans = []
        for rank in range(dst_attn_tp):
            src_offset, size, src_stride, dst_offset, dst_stride = geometry.entry_bytes(
                0, rank
            )
            spans.extend(
                SimpleNamespace(
                    src_block_ordinal=index,
                    src_offset_bytes=src_offset,
                    dst_block_index=0,
                    dst_offset_bytes=index * dst_stride * page_tokens + dst_offset,
                    size_bytes=size,
                    src_stride_bytes=src_stride,
                    dst_stride_bytes=dst_stride,
                    count=page_tokens,
                    dst_unit_ordinal=rank,
                )
                for index in range(len(blocks))
            )
        result.append(
            SimpleNamespace(
                tag=tag,
                block_ids=list(blocks),
                # Ranks that share a kv-head each receive it, so the spans can
                # sum to more than the source pages hold.
                declared_bytes=sum(span.size_bytes * span.count for span in spans),
                dst_space_version=1,
                spans=spans,
            )
        )
    return result


def _head_major_slice_spans(
    *, pool: dict, tensor: Any, blocks: list[int], dst_attn_tp: int, total_kv_heads: int
) -> Any:
    """Per-rank kv-head slices of rank-4 RPA-seq pages: one span per page and rank.

    ``num_heads`` heads from ``head`` are the contiguous bytes
    ``[head * run, (head + num_heads) * run)`` of a page, with
    ``run = 2 * head_dim_lanes * page_size * element_size``; the destination
    page stores only that rank's heads.
    """
    run = 2 * int(tensor.shape[2]) * int(tensor.shape[3]) * tensor.element_size()
    spans = []
    for rank in range(dst_attn_tp):
        src_head, num_heads, dst_head, _, dst_heads_per_rank = head_slice_params(
            src_tp=1,
            dst_tp=dst_attn_tp,
            src_rank=0,
            dst_rank=rank,
            total_kv_heads=total_kv_heads,
        )
        spans.extend(
            SimpleNamespace(
                src_block_ordinal=index,
                src_offset_bytes=src_head * run,
                dst_block_index=0,
                dst_offset_bytes=(index * dst_heads_per_rank + dst_head) * run,
                size_bytes=num_heads * run,
                src_stride_bytes=0,
                dst_stride_bytes=0,
                count=1,
                dst_unit_ordinal=rank,
            )
            for index in range(len(blocks))
        )
    return SimpleNamespace(
        tag=pool["tag"],
        block_ids=list(blocks),
        # Ranks that share a kv-head each receive it.
        declared_bytes=sum(span.size_bytes * span.count for span in spans),
        dst_space_version=1,
        spans=spans,
    )


# Early parts carry pages that are final before a request's prefill ends -- its
# cached prefix, finished chunks -- as their own native transfers, "<room>.p<i>".
# Stage groups close a request per group of consecutive stages, "<room>.s<g>",
# as soon as those stages' forwards of the final chunk have completed. The
# coordinated closing transfer keeps the plain room id.
def part_req_id(room: int, part: int) -> str:
    return f"{room}.p{part}"


def stage_req_id(room: int, group: int) -> str:
    return f"{room}.s{group}"


def _derived_uuid(req_id: str) -> int:
    """A positive signed 64-bit native transfer UUID distinct from the room's."""
    digest = hashlib.sha256(req_id.encode()).digest()
    return (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1


def part_uuid(room: int, part: int) -> int:
    return _derived_uuid(part_req_id(room, part))


def stage_uuid(room: int, group: int) -> int:
    return _derived_uuid(stage_req_id(room, group))


def parse_req_id(rid: str) -> tuple[int, Any]:
    """(room, key) of a native transfer id: None for the closing transfer, an
    int for an early part, ("s", group) for a stage group."""
    room, _, group = rid.partition(".s")
    if group:
        return int(room), ("s", int(group))
    room, _, part = rid.partition(".p")
    return int(room), (int(part) if part else None)


def stage_group(rank: int, group_size: int) -> int:
    return rank // group_size


def group_ranks(group: int, group_size: int, pp_size: int) -> list[int]:
    return list(range(group * group_size, min((group + 1) * group_size, pp_size)))


def part_spans(
    pools: Sequence[dict],
    tensors: Sequence[Any],
    full: list[int],
    swa: list[int],
    dst_attn_tp: int,
    total_kv_heads: int,
    *,
    draft_total_kv_heads: int | None = None,
    draft_swa_total_kv_heads: int | None = None,
    with_swa: bool,
    stripe: tuple[int, int] | None = None,
) -> list[Any]:
    """Spans of one transfer over the full-attention pools and, when with_swa,
    the sliding-window pools; ``stripe`` as for source_spans."""
    keep = [
        index
        for index, pool in enumerate(pools)
        if with_swa or not is_swa_pool(pool["tag"])
    ]
    pools = [pools[i] for i in keep]
    tensors = [tensors[i] for i in keep]
    if dst_attn_tp == 1:
        return source_spans(pools, full, swa, stripe=stripe)
    if stripe is not None or any(is_replicated_pool(pool["tag"]) for pool in pools):
        raise ValueError(
            "Raiden pipeline transfer moves pages without a kv-head axis, and "
            "pages from a tensor-parallel producer, to an attention-TP 1 "
            f"consumer only; got attention TP {dst_attn_tp}"
        )
    return head_slice_spans(
        pools,
        tensors,
        full,
        swa,
        dst_attn_tp,
        total_kv_heads,
        draft_total_kv_heads=draft_total_kv_heads,
        draft_swa_total_kv_heads=draft_swa_total_kv_heads,
    )


def part_destination_blocks(
    pools: Sequence[dict],
    full: list[int],
    swa: list[int],
    *,
    start: int,
    count: int,
    source_swa_count: int | None,
    only_tags: set[str] | None = None,
) -> dict:
    """Destination pages of one transfer: full-attention pages [start, start +
    count) and, when source_swa_count is given, the window; only_tags limits
    it to the layers of one stage group."""
    tags, blocks, counts, skips = [], [], [], []
    for pool in pools:
        if only_tags is not None and pool["tag"] not in only_tags:
            continue
        is_swa = is_swa_pool(pool["tag"])
        if is_swa and source_swa_count is None:
            continue
        if is_swa:
            local, remote_count = list(swa)[:source_swa_count], source_swa_count
        else:
            local, remote_count = list(full)[start : start + count], count
        if not local:
            continue
        tags.append(pool["tag"])
        blocks.extend(local)
        counts.append(len(local))
        skips.append((remote_count - len(local)) * pool["block_stride_bytes"])
    return dict(
        transfer_pool_tags=tags,
        dst_device_block_ids=blocks,
        dst_block_counts=counts,
        dst_skip_bytes=skips,
    )


def _new_transfer_state() -> dict:
    """Decode-side state of one early part, stage group or window twin;
    armed_until is when a push of it can no longer land on this rank."""
    return dict(
        sources={},
        future=None,
        accepted=False,
        done=False,
        next_attempt=0.0,
        armed_until=0.0,
    )
