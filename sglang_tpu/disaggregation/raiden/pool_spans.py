"""Pieces of the Raiden pipeline transfer shared with its sliding-window engine:
native transfer ids, per-layer byte spans, destination pages, and the decode
state of one native transfer."""

from __future__ import annotations

import hashlib
import math
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


def pool_kind(tag: str) -> str:
    return tag.rsplit(".l", 1)[0]


def is_swa_pool(tag: str) -> bool:
    return pool_kind(tag).endswith("swa")


def is_draft_pool(tag: str) -> bool:
    return pool_kind(tag).startswith("draft.")


# Request state on the pipeline transfer: one block per request, its committed
# state row (common.pipeline_state_entries names the tensors).
#   gdn.conv  [slots + 1, K - 1, 1, conv_dim], conv_dim = [q | k | v], each
#             sub-block head-sharded across attention TP
#   gdn.ssm   [slots + 1, n_v, d_k, d_v], sharded on n_v
#   ple.sc    [slots + 1, channels, state_len], replicated
#   ple.ngram [slots + 1, context_len], replicated
GDN_CONV_KIND = "gdn.conv"
GDN_SSM_KIND = "gdn.ssm"
PLE_SHORT_CONV_KIND = "ple.sc"
PLE_NGRAM_KIND = "ple.ngram"
STATE_KINDS = (GDN_CONV_KIND, GDN_SSM_KIND, PLE_SHORT_CONV_KIND, PLE_NGRAM_KIND)


def is_state_pool(tag: str) -> bool:
    return pool_kind(tag) in STATE_KINDS


def state_page_shape(tag: str, tensor: Any) -> list[int]:
    """A state row's shape without its attention-TP-sharded axis, so a
    tensor-parallel decode fingerprints like its TP=1 producer."""
    row = [int(n) for n in tensor.shape[1:]]
    kind = pool_kind(tag)
    if kind == GDN_CONV_KIND:
        return row[:-1]
    if kind == GDN_SSM_KIND:
        return row[1:]
    return row


def _state_span(**fields) -> SimpleNamespace:
    return SimpleNamespace(dst_block_index=0, **fields)


def state_slice_spans(
    pools: Sequence[dict],
    tensors: Sequence[Any],
    rows: list[int],
    dst_attn_tp: int,
    conv_shard_groups: Sequence[int] | None,
) -> list[Any]:
    """Spans moving a request's state rows (one per pool) from a TP=1 stage
    to each destination rank's shard of them.

    GDN conv: the rank's [q | k | v] slices, one strided span per sub-block
    over the K - 1 conv taps. GDN recurrent state: the rank's contiguous value
    heads. PLE state is replicated, so every rank receives the whole row.
    """
    from sglang.srt.disaggregation.utils import compute_mamba_state_slice_blocks

    result = []
    for pool, tensor in zip(pools, tensors):
        tag = pool["tag"]
        kind = pool_kind(tag)
        if kind not in STATE_KINDS:
            raise ValueError(f"{tag} is not a request-state pool")
        stride = pool["block_stride_bytes"]
        elem = tensor.element_size()
        spans = []
        if dst_attn_tp == 1 or kind in (PLE_SHORT_CONV_KIND, PLE_NGRAM_KIND):
            ranks = [None] if dst_attn_tp == 1 else list(range(dst_attn_tp))
            for rank in ranks:
                extra = {} if rank is None else {"dst_unit_ordinal": rank}
                spans.extend(
                    _state_span(
                        src_block_ordinal=index,
                        src_offset_bytes=0,
                        dst_offset_bytes=index * stride,
                        size_bytes=stride,
                        src_stride_bytes=0,
                        dst_stride_bytes=0,
                        count=1,
                        **extra,
                    )
                    for index in range(len(rows))
                )
        else:
            if kind == GDN_CONV_KIND:
                src_dim = int(tensor.shape[-1])
                # Bytes of one sharded-axis element, and rows of the row-major
                # prefix the axis repeats over (the conv taps).
                unit = elem
                repeats = math.prod(int(n) for n in tensor.shape[1:-1])
                groups = list(conv_shard_groups) if conv_shard_groups else None
                if groups is None or sum(groups) != src_dim:
                    raise ValueError(
                        f"{tag}: GDN conv state needs its [q | k | v] widths "
                        f"summing to {src_dim}, got {groups}"
                    )
            else:
                src_dim = int(tensor.shape[1])
                unit = math.prod(int(n) for n in tensor.shape[2:]) * elem
                repeats = 1
                groups = None
            if src_dim % dst_attn_tp:
                raise ValueError(
                    f"{tag}: {src_dim} cannot be split over attention TP {dst_attn_tp}"
                )
            dst_dim = src_dim // dst_attn_tp
            dst_row = stride // dst_attn_tp
            for rank in range(dst_attn_tp):
                slices = compute_mamba_state_slice_blocks(
                    src_dim, dst_dim, 1, dst_attn_tp, rank, 0, groups
                )
                for index in range(len(rows)):
                    spans.extend(
                        _state_span(
                            src_block_ordinal=index,
                            src_offset_bytes=src_start * unit,
                            dst_offset_bytes=index * dst_row + dst_start * unit,
                            size_bytes=width * unit,
                            src_stride_bytes=src_dim * unit if repeats > 1 else 0,
                            dst_stride_bytes=dst_dim * unit if repeats > 1 else 0,
                            count=repeats,
                            dst_unit_ordinal=rank,
                        )
                        for src_start, dst_start, width in slices
                    )
        result.append(
            SimpleNamespace(
                tag=tag,
                block_ids=list(rows),
                declared_bytes=sum(span.size_bytes * span.count for span in spans),
                dst_space_version=1,
                spans=spans,
            )
        )
    return result


def state_destination_blocks(
    pools: Sequence[dict], rows: list[int], *, only_tags: set[str] | None = None
) -> dict:
    """Destination state rows of a request: its committed row in every
    request-state pool (only_tags limits it to one stage group's layers)."""
    tags, blocks, counts, skips = [], [], [], []
    for pool in pools:
        if only_tags is not None and pool["tag"] not in only_tags:
            continue
        if not rows:
            continue
        tags.append(pool["tag"])
        blocks.extend(rows)
        counts.append(len(rows))
        skips.append(0)
    return dict(
        transfer_pool_tags=tags,
        dst_device_block_ids=blocks,
        dst_block_counts=counts,
        dst_skip_bytes=skips,
    )


def _producer_tensor_kv_heads(tensor: Any) -> int:
    """KV head count of a TP=1 producer page tensor when the pool omitted head_num."""
    rank = _kv_rank(tensor)
    if rank == _HEAD_MAJOR_KV_RANK:
        return int(tensor.shape[1]) // 2
    if rank == _FUSED_KV_RANK:
        return int(tensor.shape[2]) * int(tensor.shape[3]) // 2
    raise ValueError("Block-major KV bundles do not slice kv-heads")


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
