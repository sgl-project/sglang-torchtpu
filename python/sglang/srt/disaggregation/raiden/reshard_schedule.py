"""Strided KV resharding schedules for heterogeneous attention-TP disaggregation.

When prefill attention-TP > decode attention-TP (e.g. 8TP prefill -> 4TP decode
under DP-attention), a decode rank's kv-head slice spans several prefill ranks.
Which prefill bytes land where depends on ``total_kv_heads``:

  * total_kv_heads >= src_tp (e.g. 8 kv-heads at 8TP): each prefill rank owns
    a DISTINCT kv-head range, and decode rank r CONCATENATES the ranges from its
    ``ratio = src_tp // dst_tp`` prefill ranks (writers land at offsets 0, w, 2w…).
  * total_kv_heads < src_tp (e.g. 4 kv-heads at 8TP): kv-heads are REPLICATED
    across prefill ranks, so a decode rank's head is available whole from any of
    its prefill sources; each writer lands the same head at the same offset
    (redundant but idempotent).

Both are captured by ``head_slice_params`` (the same math as sglang's
``common.staging_buffer.compute_head_slice_params``), reimplemented here so this
module stays free of torch/triton and the byte-math is unit-testable anywhere.

The fused KV pool stores one ``[num_blocks, page_size, kv_heads_packed,
kv_packing, padded_head_dim]`` tensor per layer. One kv-head's K and V are a
single contiguous run (``kv_packing`` is outer to ``padded_head_dim``), so a
per-token head slice is a contiguous ``num_heads * head_pair_bytes`` granule that
lands strided (period ``dst_heads_per_rank * head_pair_bytes``) in the decode
token. These builders turn that geometry into
``raiden_service_pb2.StartTransferRequest`` push schedules.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Sequence, Tuple


class FusedKVLayerTensor(Protocol):
    """Duck-typed fused KV layer tensor. Only ``shape`` and ``element_size`` are
    read, so the byte math stays free of a torch import (and unit-testable with a
    plain stub)."""

    shape: Sequence[int]

    def element_size(self) -> int: ...


def reshard_ratio(src_tp: int, dst_tp: int) -> int:
    """Prefill ranks feeding one decode rank. Requires src_tp a multiple of dst_tp."""
    if src_tp % dst_tp != 0:
        raise ValueError(
            f"src attn TP ({src_tp}) must be a multiple of dst attn TP ({dst_tp})"
        )
    return src_tp // dst_tp


def decode_rank_for(prefill_rank: int, ratio: int) -> int:
    """Decode rank that prefill ``prefill_rank`` contributes to."""
    return prefill_rank // ratio


def writer_index(prefill_rank: int, ratio: int) -> int:
    """Position of ``prefill_rank`` among the ``ratio`` writers of its decode rank."""
    return prefill_rank % ratio


def source_ranks_for(decode_rank: int, ratio: int) -> List[int]:
    """Prefill ranks feeding decode rank ``decode_rank``, in writer order."""
    return list(range(decode_rank * ratio, (decode_rank + 1) * ratio))


def head_pair_bytes(fused_layer_tensor: FusedKVLayerTensor) -> int:
    """Bytes of one kv-head's contiguous K+V run in a fused KV layer tensor.

    Independent of the per-rank kv-head count, so it matches on prefill and
    decode. ``tensor.shape = [num_blocks, page_size, kv_heads_packed,
    kv_packing, padded_head_dim]``.
    """
    shape = fused_layer_tensor.shape
    return int(shape[3]) * int(shape[4]) * fused_layer_tensor.element_size()


def per_rank_head_count(fused_layer_tensor: FusedKVLayerTensor) -> int:
    """kv_heads_packed (this rank's kv-head count) of a fused KV layer tensor."""
    return int(fused_layer_tensor.shape[2])


def head_slice_params(
    src_tp: int,
    dst_tp: int,
    src_rank: int,
    dst_rank: int,
    total_kv_heads: int,
) -> Tuple[int, int, int, int, int]:
    """Head-slice mapping for one (src_rank -> dst_rank) contribution.

    Returns ``(src_head_start, num_heads, dst_head_start, src_heads_per_rank,
    dst_heads_per_rank)`` in kv-head units. Each ``*_head_start`` is a LOCAL
    offset within a rank's own block -- a block stores only that rank's heads,
    indexed from 0 -- not a global head index. The two per-rank counts double as
    the src/dst per-token strides. Mirrors sglang's ``compute_head_slice_params``.
    """
    src_heads_per_rank = max(1, total_kv_heads // src_tp)
    dst_heads_per_rank = max(1, total_kv_heads // dst_tp)
    tp_src_rank = src_rank % src_tp
    tp_dst_rank = dst_rank % dst_tp
    if src_tp > dst_tp:
        # Fan-in: the prefill rank is more sharded, so it sends ALL of its heads;
        # the placement is on the decode side, where the several sources feeding
        # one decode rank interleave into its wider block.
        src_head_start = 0
        num_heads = src_heads_per_rank
        # The same kv-head can live on `replicas_per_head` consecutive prefill
        # ranks; divide them out to get the distinct head-group this rank carries.
        replicas_per_head = max(1, src_tp // total_kv_heads)
        head_group = tp_src_rank // replicas_per_head
        # Fold this group's first head into the decode rank's local window:
        # distinct groups land at distinct slots (the interleave); ranks sharing
        # a group land at the same slot.
        tp_head_start = head_group * src_heads_per_rank
        dst_head_start = tp_head_start % dst_heads_per_rank
    else:
        # Equal / fan-out: the prefill rank holds a superset of this decode rank's
        # heads, so the decode takes a contiguous sub-slice of the source and
        # fills its own block from slot 0.
        tp_head_start = tp_dst_rank * dst_heads_per_rank
        src_head_start = tp_head_start % src_heads_per_rank
        num_heads = dst_heads_per_rank
        dst_head_start = 0
    return (
        src_head_start,
        num_heads,
        dst_head_start,
        src_heads_per_rank,
        dst_heads_per_rank,
    )


class ReshardGeometry:
    """Byte geometry for one (src_tp -> dst_tp) reshard over ``total_kv_heads``."""

    def __init__(self, src_tp: int, dst_tp: int, hp_bytes: int, total_kv_heads: int):
        self.src_tp = src_tp
        self.dst_tp = dst_tp
        self.ratio = reshard_ratio(src_tp, dst_tp)
        self.hp_bytes = hp_bytes
        self.total_kv_heads = total_kv_heads

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride) for the
        prefill_rank -> decode_rank contribution, per token."""
        sh_start, num_heads, dh_start, src_hpr, dst_hpr = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        return (
            sh_start * self.hp_bytes,  # src_offset
            num_heads * self.hp_bytes,  # size
            src_hpr * self.hp_bytes,  # src_stride (src per-token width)
            dh_start * self.hp_bytes,  # dst_offset
            dst_hpr * self.hp_bytes,  # dst_stride (dst per-token width)
        )


def _add_entry(
    schedule: Any,
    *,
    peer: str,
    src_block_id: int,
    dst_block_id: int,
    dst_shard_idx: int,
    src_offset: int,
    dst_offset: int,
    size: int,
    src_stride: int,
    dst_stride: int,
    count: int,
) -> None:
    entry = schedule.entries.add()
    entry.dst_peer = peer
    entry.dst_shard_idx = dst_shard_idx
    entry.src_block_id = src_block_id
    entry.dst_block_id = dst_block_id
    entry.src_offset_bytes = src_offset
    entry.dst_offset_bytes = dst_offset
    entry.size_bytes = size
    entry.src_stride_bytes = src_stride
    entry.dst_stride_bytes = dst_stride
    entry.count = count


def build_sender_request(
    pb2,
    *,
    uuid: int,
    req_id: str,
    peer: str,
    prefill_rank: int,
    decode_rank: int,
    geom: ReshardGeometry,
    src_block_ids: Sequence[int],
    dst_block_ids: Sequence[int],
    live_tokens: Sequence[int],
):
    """Sender-side StartTransferRequest for one prefill rank pushing to one decode peer.

    ``src_block_ids``/``dst_block_ids`` are physical page ids on each side, paired
    by token order; ``live_tokens[i]`` is the token count (``count``) of page ``i``.
    """
    if not (len(src_block_ids) == len(dst_block_ids) == len(live_tokens)):
        raise ValueError(
            "src_block_ids, dst_block_ids, live_tokens must be equal length"
        )
    src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(
        prefill_rank, decode_rank
    )
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = True
    req.dst_mem_type = pb2.MEMORY_TYPE_DRAM  # push reads the sender host staging pool
    req.use_block_chunks = True
    schedule = req.shard_push_schedules[0]
    for src_b, dst_b, n in zip(src_block_ids, dst_block_ids, live_tokens):
        if n <= 0:
            continue
        _add_entry(
            schedule,
            peer=peer,
            src_block_id=int(src_b),
            dst_block_id=int(dst_b),
            dst_shard_idx=0,
            src_offset=src_off,
            dst_offset=dst_off,
            size=size,
            src_stride=src_stride,
            dst_stride=dst_stride,
            count=int(n),
        )
    return req


def build_receiver_request(
    pb2,
    *,
    uuid: int,
    req_id: str,
    peer: str,
    decode_rank: int,
    geom: ReshardGeometry,
    sources: List[Tuple[int, Sequence[int]]],
    dst_block_ids: Sequence[int],
    live_tokens: Sequence[int],
    expected_block_count: Optional[int] = None,
):
    """Receiver-side StartTransferRequest assembling all sources into decode HBM.

    ``sources`` is ``[(prefill_rank, src_block_ids), ...]`` for the prefill ranks
    feeding this decode rank; each gets its own shard schedule with its strided
    dst placement. The receiver selects a source's schedule by the pushing node's
    id (``header.remote_id``), so each schedule is keyed by its ``prefill_rank``
    (the sender engine's ``node_id``), not by writer position. ``dst_mem_type =
    HBM`` makes the receiver auto-H2D the assembled host staging into HBM.
    """
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = False
    req.dst_mem_type = pb2.MEMORY_TYPE_HBM  # auto-H2D host staging -> decode HBM
    req.use_block_chunks = True
    total = 0
    for prefill_rank, src_block_ids in sources:
        src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(
            prefill_rank, decode_rank
        )
        schedule = req.shard_push_schedules[prefill_rank]
        for src_b, dst_b, n in zip(src_block_ids, dst_block_ids, live_tokens):
            if n <= 0:
                continue
            _add_entry(
                schedule,
                peer=peer,
                src_block_id=int(src_b),
                dst_block_id=int(dst_b),
                dst_shard_idx=0,
                src_offset=src_off,
                dst_offset=dst_off,
                size=size,
                src_stride=src_stride,
                dst_stride=dst_stride,
                count=int(n),
            )
            total += 1
    req.expected_block_count = (
        total if expected_block_count is None else int(expected_block_count)
    )
    return req
