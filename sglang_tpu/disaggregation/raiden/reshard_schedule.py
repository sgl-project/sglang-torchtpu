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
whole packs. These builders turn that geometry into
``raiden_service_pb2.StartTransferRequest`` push schedules.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple


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
    """kv_heads_packed of a fused KV layer tensor: 32-bit rows per token, each
    holding ``kv_packing // 2`` heads' K+V. Equals this rank's kv-head count
    only at kv_packing 2; at fp8 it is ``ceil(2 * heads / 4)``."""
    return int(fused_layer_tensor.shape[2])


def swa_geometry_kv_heads(
    full_kv_heads: int,
    swa_kv_heads: int,
    kv_packing: int,
    src_tp: int,
    dst_tp: int,
) -> int:
    """kv-head count for a sliding-window ReshardGeometry whose granule is
    ``head_pair_bytes`` (the push path), or ValueError when no count yields
    byte-correct placement.

    The granule is one kv-head only at kv_packing 2. At fp8 (kv_packing 4) it
    is a 32-bit pack of two heads, so the schedule is right only where it runs
    in pack units: every rank holds one head (a whole padded pack), or the
    count is ``swa_kv_heads // 2`` packs and every rank holds whole packs. The
    full-attention count is exactly that on MiMo-V2-like models (8 SWA / 4
    full heads), which is why they work today.
    TODO(#829): remove when head_pair_bytes uses per-head units (B1).
    """
    if swa_kv_heads == full_kv_heads or kv_packing == 2:
        return swa_kv_heads
    if kv_packing == 4:
        if swa_kv_heads <= min(src_tp, dst_tp):
            return swa_kv_heads
        if (
            swa_kv_heads == 2 * full_kv_heads
            and swa_kv_heads % (2 * max(src_tp, dst_tp)) == 0
        ):
            return full_kv_heads
    raise ValueError(
        f"Raiden reshard of sliding-window KV with {swa_kv_heads} kv-heads "
        f"(full attention: {full_kv_heads}) at attention TP {src_tp}->{dst_tp} "
        f"is unsupported at kv_packing {kv_packing}"
    )


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
    """Byte geometry for one (src_tp -> dst_tp) reshard over ``total_kv_heads``."""

    def __init__(
        self,
        src_tp: int,
        dst_tp: int,
        hp_bytes: int,
        total_kv_heads: int,
        kv_packing: int = 2,
        element_size: int = 2,
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
        self.kv_packing = kv_packing
        self.element_size = element_size

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride) for the
        prefill_rank -> decode_rank contribution, per token."""
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


# tpu-sync accepts at most IOV_MAX iovecs per (layer, block, sender)
# (transport/lib/socket/util.cc); a push entry spends one per token.
IOV_MAX = 1024


def _moved_extent(offset: int, size: int, stride: int, count: int, moved: int) -> int:
    """One past the last byte touched by the first ``moved`` bytes of
    ``count`` chunks of ``size`` bytes placed every ``stride`` bytes."""
    if size <= 0 or moved <= 0:
        return offset
    full, rest = divmod(min(moved, count * size), size)
    end = offset + (full - 1) * stride + size if full else offset
    if rest:
        end = max(end, offset + full * stride + rest)
    return end


def check_entry_bounds(
    entry: Any, src_block_bytes: int, dst_block_bytes: Optional[int]
) -> None:
    """Raise ValueError if ``entry`` moves bytes outside one block.

    Models tpu-sync's legacy push: the sender sends at most its block's bytes
    of the entry's chunks, and the receiver writes exactly what the sender
    sent. Native code validates offsets only against the whole host block
    array, so an entry that leaves its block silently writes a neighbouring
    block (another request's staging). ``dst_block_bytes=None`` checks the
    source side only.
    """
    size, count = entry.size_bytes, entry.count
    moved = min(count * size, src_block_bytes)
    src_end = _moved_extent(
        entry.src_offset_bytes, size, entry.src_stride_bytes, count, moved
    )
    dst_end = _moved_extent(
        entry.dst_offset_bytes, size, entry.dst_stride_bytes, count, moved
    )
    if src_end > src_block_bytes or (
        dst_block_bytes is not None and dst_end > dst_block_bytes
    ):
        raise ValueError(
            "Raiden reshard entry exceeds the block: src bytes "
            f"[{entry.src_offset_bytes}, {src_end}) of {src_block_bytes}, dst "
            f"bytes [{entry.dst_offset_bytes}, {dst_end}) of {dst_block_bytes}"
        )


def check_request_bounds(
    req: Any, block_bytes: Iterable[Tuple[int, Optional[int]]]
) -> None:
    """check_entry_bounds on every entry of a StartTransferRequest, for each
    ``(src_block_bytes, dst_block_bytes)`` it is pushed with: one plan serves
    every layer of its group, and draft layers can have other block sizes."""
    pairs = list(block_bytes)
    for schedule in req.shard_push_schedules.values():
        for entry in schedule.entries:
            for src_bytes, dst_bytes in pairs:
                check_entry_bounds(entry, src_bytes, dst_bytes)


def candidate_total_heads(own_heads: int, own_tp: int, peer_tp: int) -> List[int]:
    """Global kv-head counts consistent with ``own_heads`` per rank at
    attention TP ``own_tp``, for a peer at ``peer_tp``.

    Exact unless one head per rank. On the smaller (decode) side every count
    up to ``own_tp`` then gives the same push schedule, so ``own_tp`` stands in
    for all of them; on the larger side the peer's per-rank count stays open.
    """
    if own_heads > 1:
        return [own_heads * own_tp]
    if own_tp <= peer_tp:
        return [own_tp]
    return [
        n
        for n in range(1, own_tp + 1)
        if own_tp % n == 0 and (peer_tp % n == 0 or n % peer_tp == 0)
    ]


def push_reshard_where(src_tp: int, dst_tp: int) -> str:
    return f"Raiden push-reshard (prefill attention TP {src_tp} -> decode {dst_tp})"


def fused_layout_refusal(src_tp: int, dst_tp: int, ndim: int) -> Optional[str]:
    """The push schedule addresses the 5-D fused MHA layout only; MLA pools
    are 4-D and quantized (MXINT8) pools 3-D."""
    if ndim == 5:
        return None
    return (
        f"{push_reshard_where(src_tp, dst_tp)} needs the 5-D fused MHA KV "
        f"layout; this KV tensor is {ndim}-D (MLA or a quantized pool)"
    )


def _schedule(geom: ReshardGeometry) -> List[Tuple[int, int, int, int, int]]:
    return [geom.entry_bytes(p, p // geom.ratio) for p in range(geom.src_tp)]


def push_reshard_refusal(
    geom: ReshardGeometry,
    shape: Sequence[int],
    element_size: int,
    *,
    total_heads: int,
    own_tp: int,
    page_size: int,
    seq_along_lane: bool,
) -> Optional[str]:
    """Why pushing one KV layer tensor with ``geom``'s schedule would misplace
    bytes, or None when the copy is byte-exact.

    ``shape``/``element_size`` are this side's tensor (attention TP
    ``own_tp``) and ``total_heads`` the global kv-head count it really holds,
    which differs from ``geom.total_kv_heads`` for draft layers and, before
    #1076, sliding-window layers. The push path builds ``geom`` with the
    default unit, so ``head_pair_bytes`` is one 32-bit row (one bf16 head, two
    fp8 heads) and the schedule is exact iff it equals the schedule over the
    tensor's real rows.
    """
    src_tp, dst_tp = geom.src_tp, geom.dst_tp
    where = push_reshard_where(src_tp, dst_tp)
    reason = fused_layout_refusal(src_tp, dst_tp, len(shape))
    if reason is not None:
        return reason
    if page_size > IOV_MAX:
        return f"{where} supports page_size <= {IOV_MAX}, got {page_size}"
    pk = int(shape[3])
    hp = pk * int(shape[4]) * element_size
    if pk * element_size != 4 or hp != geom.hp_bytes:
        return (
            f"{where} moves every layer with one geometry, but its KV layers "
            "differ in head_dim or dtype (e.g. a draft model's KV)"
        )
    src_heads = max(1, total_heads // src_tp)
    dst_heads = max(1, total_heads // dst_tp)
    own_heads = max(1, total_heads // own_tp)
    rows = 2 * own_heads if seq_along_lane else -(-2 * own_heads // pk)
    if int(shape[1 if seq_along_lane else 2]) != rows:
        return (
            f"{where}: a KV tensor of shape {tuple(shape)} does not hold "
            f"{own_heads} kv heads per rank"
        )
    if seq_along_lane:
        # TODO(#829): remove when the SEQ_ALONG_LANE geometry lands. The
        # schedule treats one (kv_packing, page_size) row as a head; at one head
        # per rank that is page_size contiguous chunks from offset 0 on both
        # sides, which the sender's per-block cap turns into a whole-page copy
        # once the chunks cover the page.
        block = math.prod(int(d) for d in shape[1:]) * element_size
        identity = all(e == (0, hp, hp, 0, hp) for e in _schedule(geom))
        if src_heads == dst_heads == 1 and identity and page_size * hp >= block:
            return None
        return (
            f"{where} supports SEQ_ALONG_LANE KV only with one kv head per rank "
            "on both sides and page_size * kv_packing >= 2 * head_dim lanes "
            f"(here {src_heads} -> {dst_heads} heads per rank)"
        )
    if pk == 2 or src_heads == dst_heads == 1:
        # A row is one head, or each rank's single head fills its own row.
        row_heads = total_heads
    elif src_heads % 2 == 0 and dst_heads % 2 == 0:
        # TODO(#829): remove with B1 (per-head units on the push path). Every
        # rank holds whole two-head rows, so the schedule is exact when its
        # head count is the row count -- true for draft or sliding-window
        # layers holding twice the geometry's heads, false for the full pool.
        row_heads = total_heads // 2
    else:
        row_heads = None
    if row_heads is not None:
        if _schedule(geom) == _schedule(ReshardGeometry(src_tp, dst_tp, hp, row_heads)):
            return None
        if pk == 2 or geom.total_kv_heads != total_heads:
            return (
                f"{where} would misplace KV: its schedule assumes "
                f"{geom.total_kv_heads} kv heads, a KV layer holds {total_heads}"
            )
    return (
        f"{where} misplaces fp8 KV unless each decode rank holds one kv head "
        f"(here {src_heads} per prefill rank -> {dst_heads} per decode rank); "
        "use equal attention TP or a bf16 KV cache"
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
        prefill_rank % geom.src_tp, decode_rank
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
        src_off, size, src_stride, dst_off, dst_stride = geom.entry_bytes(
            sender_id % geom.src_tp, decode_rank
        )
        schedule = req.shard_push_schedules[sender_id]
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
