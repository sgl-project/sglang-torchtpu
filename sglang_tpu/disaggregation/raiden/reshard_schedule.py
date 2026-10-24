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
kv_packing, padded_head_dim]`` tensor per layer. Native host copies retain
32-bit packing: BF16 packs one head's K/V into each word, while FP8 packs two
heads. Only complete head packs are contiguous, so a head slice must cover
whole packs. The head-major RPA-seq pool stores ``[num_blocks, 2 * kv_heads,
head_dim_lanes, page_size]`` instead: a head's K and V rows are adjacent and
the token index sits on the lane axis, so one kv-head is a contiguous run of
a page and a slice is copied per page rather than per token. These builders
turn either geometry into ``raiden_service_pb2.StartTransferRequest`` push
schedules.

A decode-context-parallel (DCP) decode keeps prefill attention TP but splits
each request's full-attention pages across its DCP group by page interleave:
logical page ``l`` is DCP rank ``l % dcp_size``'s local page ``l // dcp_size``
(``PageInterleavePlacement`` in page units). ``dcp_page_split`` is that filter
over a request's logical page list; each DCP rank then receives its pages as
a ``src_tp -> src_tp // dcp_size`` fan-in of its group's kv heads.
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
    """Prefill ranks feeding one decode rank; 1 when the decode is sharded
    finer. Requires one attention TP to be a multiple of the other."""
    if src_tp % dst_tp != 0 and dst_tp % src_tp != 0:
        raise ValueError(
            f"src attn TP ({src_tp}) and dst attn TP ({dst_tp}) must be multiples"
        )
    return max(1, src_tp // dst_tp)


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


def head_bytes(fused_layer_tensor: FusedKVLayerTensor) -> int:
    """Bytes of one kv-head's K+V values, padding lanes excluded; the per-head
    granule of a ReshardGeometry built with the tensor's kv_packing."""
    shape = fused_layer_tensor.shape
    return 2 * int(shape[4]) * fused_layer_tensor.element_size()


def per_rank_head_count(fused_layer_tensor: FusedKVLayerTensor) -> int:
    """kv_heads_packed (this rank's kv-head count) of a fused KV layer tensor."""
    return int(fused_layer_tensor.shape[2])


def is_head_major(layer_tensor: FusedKVLayerTensor) -> bool:
    """Whether a KV layer tensor has the rank-4 head-major RPA-seq layout."""
    return len(layer_tensor.shape) == 4


def head_major_run_bytes(layer_tensor: FusedKVLayerTensor) -> int:
    """Bytes of one kv-head's whole-page K+V run in a head-major layer tensor
    ``[num_blocks, 2 * kv_heads, head_dim_lanes, page_size]``."""
    shape = layer_tensor.shape
    return 2 * int(shape[2]) * int(shape[3]) * layer_tensor.element_size()


def dcp_page_split(
    pages: Sequence[int], dcp_size: int, dcp_rank: int
) -> Tuple[List[int], List[int]]:
    """Positions of ``pages`` that DCP rank ``dcp_rank`` holds, with their
    local page ids.

    ``pages`` is a request's logical page list in token order, so the
    positions also index the prefill's page list of the same request. With
    ``dcp_size == 1`` every page is local under its own id.
    """
    if dcp_size == 1:
        return list(range(len(pages))), [int(page) for page in pages]
    positions: List[int] = []
    local: List[int] = []
    for position, page in enumerate(pages):
        if int(page) % dcp_size == dcp_rank:
            positions.append(position)
            local.append(int(page) // dcp_size)
    return positions, local


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
        # fills its own block from slot 0. Consecutive decode ranks share a
        # kv-head when the model has fewer kv-heads than decode ranks.
        dst_replication = max(1, dst_tp // total_kv_heads)
        tp_head_start = tp_dst_rank // dst_replication * dst_heads_per_rank
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
    """Byte geometry for one (src_tp -> dst_tp) reshard over ``total_kv_heads``.

    ``hp_bytes`` is the per-head granule: one kv-head's K+V per token in the
    fused layout, or its whole-page run when ``page_tokens`` names a head-major
    page (then an entry copies head runs of a page rather than token granules,
    and ``kv_packing`` only has to satisfy the 32-bit dtype rule).
    """

    def __init__(
        self,
        src_tp: int,
        dst_tp: int,
        hp_bytes: int,
        total_kv_heads: int,
        kv_packing: int = 2,
        element_size: int = 2,
        *,
        page_tokens: Optional[int] = None,
    ):
        self.src_tp = src_tp
        self.dst_tp = dst_tp
        self.ratio = reshard_ratio(src_tp, dst_tp)
        self.hp_bytes = hp_bytes
        self.total_kv_heads = total_kv_heads
        if kv_packing not in (1, 2, 4) or kv_packing * element_size != 4:
            raise ValueError("Raiden fused KV requires native 32-bit dtype packing")
        if hp_bytes <= 0 or hp_bytes % (2 * element_size):
            raise ValueError("Raiden head-pair bytes must contain whole K/V elements")
        if page_tokens is not None and (
            page_tokens <= 0 or hp_bytes % (2 * page_tokens * element_size)
        ):
            raise ValueError("Raiden head-major runs must hold whole K/V rows")
        self.kv_packing = kv_packing
        self.element_size = element_size
        self.page_tokens = page_tokens

    def page_entry(self, prefill_rank: int, decode_rank: int, live_tokens: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride, count)
        of one page's prefill_rank -> decode_rank contribution carrying
        ``live_tokens`` tokens."""
        if self.page_tokens is None:
            return (*self.entry_bytes(prefill_rank, decode_rank), int(live_tokens))
        sh_start, num_heads, dh_start, _, _ = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        run = self.hp_bytes
        if live_tokens == self.page_tokens:
            return (sh_start * run, num_heads * run, 0, dh_start * run, 0, 1)
        # A partial page: the live tokens are the leading lanes of every K/V
        # row of the slice, one strided copy per row on both sides.
        row = self.page_tokens * self.element_size
        return (
            sh_start * run,
            int(live_tokens) * self.element_size,
            row,
            dh_start * run,
            row,
            num_heads * run // row,
        )

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride) for the
        prefill_rank -> decode_rank contribution, per token."""
        if self.page_tokens is not None:
            raise ValueError("head-major pages have no per-token granule")
        sh_start, num_heads, dh_start, src_hpr, dst_hpr = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        # A 32-bit pack holds kv_packing // 2 heads, and a rank's padded last
        # pack still spans its per-token width.
        heads_per_pack = max(1, self.kv_packing // 2)
        if any(n % heads_per_pack for n in (sh_start, dh_start, num_heads)):
            raise ValueError("Partial FP8 head packs are unsupported")
        src_hpr = -(-src_hpr // heads_per_pack) * heads_per_pack
        dst_hpr = -(-dst_hpr // heads_per_pack) * heads_per_pack
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
        src_off, size, src_stride, dst_off, dst_stride, count = geom.page_entry(
            prefill_rank % geom.src_tp, decode_rank, int(n)
        )
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
            count=count,
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

    ``sources`` is ``[(sender_id, src_block_ids), ...]`` for the prefill engines
    feeding this decode rank; each gets its own shard schedule with its strided
    dst placement. The receiver selects a source's schedule by the pushing node's
    id (``header.remote_id``), so each schedule is keyed by the sender engine's
    full id, while the byte geometry uses the engine's attention-TP rank within
    its replica (``sender_id % src_tp``). ``dst_mem_type = HBM`` makes the
    receiver auto-H2D the assembled host staging into HBM.
    """
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = False
    req.dst_mem_type = pb2.MEMORY_TYPE_HBM  # auto-H2D host staging -> decode HBM
    req.use_block_chunks = True
    total = 0
    for sender_id, src_block_ids in sources:
        schedule = req.shard_push_schedules[sender_id]
        for src_b, dst_b, n in zip(src_block_ids, dst_block_ids, live_tokens):
            if n <= 0:
                continue
            src_off, size, src_stride, dst_off, dst_stride, count = geom.page_entry(
                sender_id % geom.src_tp, decode_rank, int(n)
            )
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
                count=count,
            )
            total += 1
    req.expected_block_count = (
        total if expected_block_count is None else int(expected_block_count)
    )
    return req
