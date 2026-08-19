"""Strided KV resharding declarations for heterogeneous attention-TP disaggregation.

When prefill attention-TP > decode attention-TP (e.g. 8TP prefill -> 4TP decode
under DP-attention), a decode rank's kv-head slice spans several prefill ranks.
Which prefill bytes land where depends on ``total_kv_heads``:

  * total_kv_heads >= src_tp (e.g. 8 kv-heads at 8TP): each prefill rank owns
    a DISTINCT kv-head range, and decode rank r CONCATENATES the ranges from its
    ``ratio = src_tp // dst_tp`` prefill ranks (writers land at offsets 0, w, 2w…).
  * total_kv_heads < src_tp (e.g. 4 kv-heads at 8TP): kv-heads are REPLICATED
    across prefill ranks, so a decode rank's head is available whole from any of
    its prefill sources. Only the lowest-numbered replica of each head group
    declares it (``declares_spans``), because the destination bytes must be
    claimed exactly once.

Both are captured by ``head_slice_params`` (the same math as sglang's
``common.staging_buffer.compute_head_slice_params``), reimplemented here so this
module stays free of torch/triton and the byte-math is unit-testable anywhere.

The fused KV pool stores one ``[num_blocks, page_size, kv_heads_packed,
kv_packing, padded_head_dim]`` tensor per layer. One kv-head's K and V are a
single contiguous run (``kv_packing`` is outer to ``padded_head_dim``), so a
per-token head slice is a contiguous ``num_heads * head_pair_bytes`` granule that
lands strided (period ``dst_heads_per_rank * head_pair_bytes``) in the decode
token. These builders turn that geometry into the byte spans each prefill rank
declares to the raiden controller, which plans and arms the transfer from them.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import msgspec


class FusedKVLayerTensor(Protocol):
    """Duck-typed fused KV layer tensor. Only ``shape`` and ``element_size`` are
    read, so the byte math stays free of a torch import (and unit-testable with a
    plain stub)."""

    shape: Sequence[int]

    def element_size(self) -> int: ...


class PoolByteSpan(msgspec.Struct, frozen=True):
    """One uniformly strided byte copy between one source and one destination block.

    ``src_block_ordinal`` indexes the declaring rank's own ``block_ids`` list;
    ``dst_block_index`` indexes the transfer's destination block list. Neither
    is a physical page id -- the controller binds both when it plans.
    """

    src_block_ordinal: int
    src_offset_bytes: int
    dst_block_index: int
    dst_offset_bytes: int
    size_bytes: int
    src_stride_bytes: int = 0
    dst_stride_bytes: int = 0
    count: int = 1


class PoolSpanRegistration(msgspec.Struct, frozen=True):
    """One rank's declared byte contribution to one pool tag."""

    tag: str
    block_ids: tuple[int, ...]
    spans: tuple[PoolByteSpan, ...]
    declared_bytes: int
    dst_space_version: int = 0


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
        self.replicas_per_head = max(1, src_tp // total_kv_heads)

    def declares_spans(self, prefill_rank: int) -> bool:
        """Whether ``prefill_rank`` is the rank that claims its head group's bytes.

        With fewer kv-heads than prefill ranks the same head lives on several
        ranks; the destination bytes must be claimed exactly once, so only the
        lowest-numbered replica declares them.
        """
        return (prefill_rank % self.src_tp) % self.replicas_per_head == 0

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


def build_span_registration(
    *,
    tag: str,
    geom: ReshardGeometry,
    prefill_rank: int,
    decode_rank: int,
    src_block_ids: Sequence[int],
    live_tokens: Sequence[int],
) -> PoolSpanRegistration:
    """Byte spans one prefill rank contributes to one decode rank's pages.

    ``src_block_ids[i]`` is this rank's physical page holding the same tokens as
    the transfer's destination page ``i``, and ``live_tokens[i]`` is how many of
    that page's tokens carry KV. A rank that is a redundant head replica
    declares no spans, so the union over the decode rank's sources covers each
    destination page exactly once.
    """
    if len(src_block_ids) != len(live_tokens):
        raise ValueError("src_block_ids and live_tokens must be equal length")

    spans: List[PoolByteSpan] = []
    declared_bytes = 0
    if geom.declares_spans(prefill_rank):
        src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(
            prefill_rank, decode_rank
        )
        for ordinal, tokens in enumerate(live_tokens):
            if tokens <= 0:
                continue
            spans.append(
                PoolByteSpan(
                    src_block_ordinal=ordinal,
                    src_offset_bytes=src_off,
                    dst_block_index=ordinal,
                    dst_offset_bytes=dst_off,
                    size_bytes=size,
                    src_stride_bytes=src_stride,
                    dst_stride_bytes=dst_stride,
                    count=int(tokens),
                )
            )
            declared_bytes += size * int(tokens)

    return PoolSpanRegistration(
        tag=tag,
        block_ids=tuple(int(b) for b in src_block_ids),
        spans=tuple(spans),
        declared_bytes=declared_bytes,
    )
