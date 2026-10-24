"""Pieces of the Raiden pipeline transfer shared with its sliding-window engine:
native transfer ids, per-layer byte spans, destination pages, and the decode
state of one native transfer."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, Sequence

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
_FUSED_KV_RANK = 5
_HEAD_MAJOR_KV_RANK = 4


def _kv_rank(tensor) -> int:
    rank = len(tensor.shape)
    if rank not in (_FUSED_KV_RANK, _HEAD_MAJOR_KV_RANK):
        raise ValueError(
            "Raiden pipeline transfer requires rank-5 fused or rank-4 RPA-seq KV"
        )
    return rank


def pool_kind(tag: str) -> str:
    return tag.rsplit(".l", 1)[0]


def is_swa_pool(tag: str) -> bool:
    return pool_kind(tag).endswith("swa")


def source_spans(pools: Sequence[dict], full: list[int], swa: list[int]) -> list[Any]:
    """Describe whole pages in destination byte space for each locally owned layer."""
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
        ]
        result.append(
            SimpleNamespace(
                tag=pool["tag"],
                block_ids=list(blocks),
                declared_bytes=len(blocks) * stride,
                dst_space_version=1,
                spans=spans,
            )
        )
    return result


def head_slice_spans(
    pools: Sequence[dict],
    tensors: Sequence[Any],
    full: list[int],
    swa: list[int],
    dst_attn_tp: int,
    total_kv_heads: int,
    total_swa_kv_heads: int,
) -> list[Any]:
    """Describe each destination rank's kv-head slice of every locally owned
    page: one strided span per page and rank, routed to that rank only.
    Sliding-window pools are sliced over ``total_swa_kv_heads``."""
    result = []
    for pool, tensor in zip(pools, tensors):
        is_swa = is_swa_pool(pool["tag"])
        blocks = swa if is_swa else full
        heads = total_swa_kv_heads if is_swa else total_kv_heads
        if _kv_rank(tensor) == _HEAD_MAJOR_KV_RANK:
            result.append(
                _head_major_slice_spans(
                    pool=pool,
                    tensor=tensor,
                    blocks=blocks,
                    dst_attn_tp=dst_attn_tp,
                    total_kv_heads=heads,
                )
            )
            continue
        geometry = ReshardGeometry(
            src_tp=1,
            dst_tp=dst_attn_tp,
            hp_bytes=head_bytes(tensor),
            total_kv_heads=heads,
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
                tag=pool["tag"],
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
    total_swa_kv_heads: int,
    *,
    with_swa: bool,
) -> list[Any]:
    """Spans of one transfer over the full-attention pools and, when with_swa,
    the sliding-window pools."""
    keep = [
        index
        for index, pool in enumerate(pools)
        if with_swa or not is_swa_pool(pool["tag"])
    ]
    pools = [pools[i] for i in keep]
    tensors = [tensors[i] for i in keep]
    if dst_attn_tp == 1:
        return source_spans(pools, full, swa)
    return head_slice_spans(
        pools, tensors, full, swa, dst_attn_tp, total_kv_heads, total_swa_kv_heads
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
