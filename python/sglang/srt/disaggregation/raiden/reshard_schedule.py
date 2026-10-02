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

Two TPU KV layouts exist, and the byte geometry differs between them.

``HEAD_ALONG_SUBLANE`` (NHD, the default) stores one ``[num_blocks, page_size,
kv_heads_packed, kv_packing, padded_head_dim]`` tensor per layer. Tokens are
axis 1, so a head slice lands strided (period = the destination's per-token
row) in the decode token -- ``count`` is the page's live-token count.

``SEQ_ALONG_LANE`` (HND, ``SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1``) stores
``[num_blocks, num_kv_heads * 2, head_dim_lanes // kv_packing, kv_packing,
page_size]``. The kv-head axis is *outermost*, so a rank's heads occupy one
contiguous slab covering every token of the page: the same transfer collapses
to a single run with ``count = 1``. What HND cannot express is a sub-page
*token* slice, which this path never needs because raiden always moves whole
pages (``live_tokens`` only marks a page as non-empty).

The granule is a packing group, not a head pair
-----------------------------------------------
Raiden moves raw bytes, so the smallest slice it can address is whatever the
TPU stores contiguously -- and that is *not* always one kv-head's K+V.

Under NHD the ``kv_packing`` axis is, by construction, "packed in 32-bit lanes"
(see ``kv_pool.TpuFusedMHATokenToKVPool``): its ``kv_packing = 32 //
elem_bits`` slots are interleaved *inside* each 32-bit word. At fp8 the
512-byte group ``[K0, V0, K1, V1]`` is physically 128 words each holding one
byte of each slot, so its first 256 bytes are NOT ``K0, V0`` -- they are the
low half of all four slots. The atomic granule is therefore the whole group
(``kv_packing // 2`` kv-heads: 2 at fp8, 1 at bf16), and a head slice that
starts or ends mid-group cannot be expressed as a strided copy at all. Those
are rejected by ``ReshardGeometry.validate`` rather than silently corrupted.

Under HND that axis is gone: ``num_kv_heads * 2`` is "a plain axis ... rather
than something packed into 32-bit lanes, so it carries no padding" (see
``kv_pool.TpuSeqAlongLaneKVPool``) and the packing rides the *head-dim* axis
instead. Every head pair is a whole number of words, so HND slices at true
per-head granularity for any dtype. That makes HND not merely cheaper to
reshard but strictly more *reshardable*: fp8 + NHD cannot regroup 1 head into
2, fp8 + HND can.

``BlockLayout`` owns this difference. Offsets and sizes come from
``head_span`` (granule units), strides from ``row_bytes`` and the repeat count
from ``count_for``. These builders turn the result into
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


LAYOUT_HEAD_ALONG_SUBLANE = "head_along_sublane"  # NHD, token-major
LAYOUT_SEQ_ALONG_LANE = "seq_along_lane"  # HND, head-major


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def head_pair_bytes(
    fused_layer_tensor: FusedKVLayerTensor,
    layout: str = LAYOUT_HEAD_ALONG_SUBLANE,
) -> int:
    """Bytes one kv-head's K+V occupies in a fused KV layer tensor.

    Independent of the per-rank kv-head count, so prefill and decode agree on
    it. This is a *logical* size: under ``HEAD_ALONG_SUBLANE`` a lone head pair
    is generally not addressable, because the packing group around it is
    interleaved into 32-bit words (see the module docstring). Go through
    ``BlockLayout.head_span`` for offsets and sizes that are real.

    ``HEAD_ALONG_SUBLANE``: ``[num_blocks, page_size, kv_heads_packed,
    kv_packing, padded_head_dim]``. A head owns two slots (K then V) of the
    ``kv_packing`` axis, so the pair spans ``2 * padded_head_dim``.

    ``SEQ_ALONG_LANE``: ``[num_blocks, num_kv_heads * 2, head_dim_lanes //
    kv_packing, kv_packing, page_size]``. A head owns two entries of axis 1,
    each covering every token of the page.
    """
    shape = fused_layer_tensor.shape
    itemsize = fused_layer_tensor.element_size()
    if layout == LAYOUT_SEQ_ALONG_LANE:
        return 2 * int(shape[2]) * int(shape[3]) * int(shape[4]) * itemsize
    if layout != LAYOUT_HEAD_ALONG_SUBLANE:
        raise ValueError(f"unknown KV layout {layout!r}")
    return 2 * int(shape[4]) * itemsize


def per_rank_head_count(
    fused_layer_tensor: FusedKVLayerTensor,
    layout: str = LAYOUT_HEAD_ALONG_SUBLANE,
) -> int:
    """This rank's kv-head count, where the tensor can express it.

    Exact under ``SEQ_ALONG_LANE`` (axis 1 is ``2 * heads``). Under
    ``HEAD_ALONG_SUBLANE`` the tensor only records ``kv_heads_packed`` -- the
    *group* count -- which cannot recover the head count once the group is
    padded (``2 * heads < kv_packing``), so prefer deriving the head count from
    ``total_kv_heads // tp``. Kept for diagnostics and back-compat.
    """
    if layout == LAYOUT_SEQ_ALONG_LANE:
        return int(fused_layer_tensor.shape[1]) // 2
    return int(fused_layer_tensor.shape[2])


class BlockLayout:
    """How one rank's page maps to bytes, for one of the two KV layouts.

    Three things differ between the layouts and all of them live here:

    * ``head_span`` -- where a head slice starts and how long it is, rounded
      out to whole *granules* (the smallest independently addressable run).
    * ``row_bytes`` -- the span of the unit that repeats inside a page: one
      token under NHD, the rank's whole head slab under HND.
    * ``count_for`` -- how many times that unit repeats.
    """

    __slots__ = ("layout", "hp_bytes", "kv_packing", "slot_bytes")

    def __init__(
        self,
        layout: str,
        *,
        hp_bytes: int,
        kv_packing: Optional[int] = None,
        slot_bytes: Optional[int] = None,
    ):
        if layout not in (LAYOUT_HEAD_ALONG_SUBLANE, LAYOUT_SEQ_ALONG_LANE):
            raise ValueError(f"unknown KV layout {layout!r}")
        self.layout = layout
        self.hp_bytes = int(hp_bytes)
        self.kv_packing = kv_packing
        self.slot_bytes = slot_bytes

    @classmethod
    def from_tensor(
        cls, fused_layer_tensor: FusedKVLayerTensor, layout: str
    ) -> BlockLayout:
        shape = fused_layer_tensor.shape
        itemsize = fused_layer_tensor.element_size()
        return cls(
            layout,
            hp_bytes=head_pair_bytes(fused_layer_tensor, layout),
            kv_packing=int(shape[3]),
            slot_bytes=(
                int(shape[2]) * int(shape[3]) * int(shape[4]) * itemsize
                if layout == LAYOUT_SEQ_ALONG_LANE
                else int(shape[4]) * itemsize
            ),
        )

    @property
    def is_seq_along_lane(self) -> bool:
        return self.layout == LAYOUT_SEQ_ALONG_LANE

    @property
    def heads_per_granule(self) -> int:
        """kv-heads in the smallest independently addressable byte run.

        1 under HND, where axis 1 is plain and every head pair is a whole
        number of 32-bit words. Under NHD the ``kv_packing`` slots share their
        words, so the run is the entire group: ``max(2, kv_packing) // 2``
        heads -- 1 at bf16, 2 at fp8. The ``max`` covers fp32
        (``kv_packing == 1``), where nothing is packed and the head pair is
        already the natural unit.
        """
        if self.is_seq_along_lane or self.kv_packing is None:
            return 1
        return max(2, int(self.kv_packing)) // 2

    @property
    def granule_bytes(self) -> int:
        """Byte span of that run, i.e. ``heads_per_granule`` head pairs."""
        return self.heads_per_granule * self.hp_bytes

    def head_span(
        self, head_start: int, num_heads: int, heads_per_rank: int
    ) -> Tuple[int, int]:
        """``(offset_bytes, size_bytes)`` for ``num_heads`` heads at ``head_start``.

        The slice is rounded out to whole granules. That is a no-op under HND
        and under bf16 NHD, and it is what makes a 1-head rank work at all
        under fp8 NHD: the rank's only head sits in a half-empty 2-head group,
        and the pad slots must travel with it because they share its 32-bit
        words. Copying just the head's nominal ``2 * padded_head_dim`` bytes
        there moves the low half of all four slots instead of K and V, which
        corrupts the cache silently -- the regression this rounding fixes.

        Raises when the slice starts mid-granule, or ends mid-granule while a
        further head of the same rank owns the remainder. Neither is a strided
        byte copy: the two heads are interleaved inside the same words.
        """
        hpg = self.heads_per_granule
        if hpg <= 1:
            return head_start * self.hp_bytes, num_heads * self.hp_bytes
        end = head_start + num_heads
        detail = (
            f"kv-head slice [{head_start}, {end}) of a {heads_per_rank}-head "
            f"rank does not align to the {hpg}-head packing group of the "
            f"{self.layout} layout"
        )
        remedy = (
            "The group's slots are interleaved inside 32-bit words, so this "
            "slice is not a byte range. Set SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1 "
            "(the head axis carries no packing there, so any head slice "
            "works), or use a KV dtype whose packing is 2 (bf16), or match the "
            "prefill and decode attention-TP sizes so no regroup is needed."
        )
        if head_start % hpg:
            raise ValueError(f"{detail}: it starts mid-group. {remedy}")
        if end % hpg and end < heads_per_rank:
            raise ValueError(
                f"{detail}: it ends mid-group and head {end} of the same rank "
                f"owns the rest of it. {remedy}"
            )
        first = head_start // hpg
        return (
            first * self.granule_bytes,
            (_ceil_div(end, hpg) - first) * self.granule_bytes,
        )

    def row_bytes(self, heads_per_rank: int) -> int:
        """Repeat-unit span for a rank holding ``heads_per_rank`` kv-heads."""
        if self.is_seq_along_lane:
            # Axis 1 is exactly 2*heads -- no padding is possible.
            return 2 * int(heads_per_rank) * int(self.slot_bytes)
        if self.kv_packing is None or self.slot_bytes is None:
            # Legacy construction: assume the unpadded case.
            return int(heads_per_rank) * self.hp_bytes
        # The 2*heads K/V slots are laid out in ceil(2h/packing) padded groups.
        groups = _ceil_div(2 * int(heads_per_rank), int(self.kv_packing))
        return groups * int(self.kv_packing) * int(self.slot_bytes)

    def count_for(self, live_tokens: int) -> int:
        """Number of strided repeats for a page holding ``live_tokens`` tokens.

        HND keeps a rank's heads contiguous across the whole page, so one run
        moves the page regardless of how many of its lanes are live; the unused
        tail is copied too, which is harmless because those token slots are
        written before they are ever read.
        """
        return 1 if self.is_seq_along_lane else int(live_tokens)


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
    """Byte geometry for one (src_tp -> dst_tp) reshard over ``total_kv_heads``.

    ``block`` carries the KV layout. When omitted the geometry degrades to the
    historical token-major assumption (strides = heads * head_pair_bytes,
    count = live tokens), which stays exact for unpadded ``HEAD_ALONG_SUBLANE``
    blocks -- the only case that existed before HND support.
    """

    def __init__(
        self,
        src_tp: int,
        dst_tp: int,
        hp_bytes: int,
        total_kv_heads: int,
        block: Optional[BlockLayout] = None,
    ):
        self.src_tp = src_tp
        self.dst_tp = dst_tp
        self.ratio = reshard_ratio(src_tp, dst_tp)
        self.hp_bytes = hp_bytes
        self.total_kv_heads = total_kv_heads
        if block is not None and int(block.hp_bytes) != int(hp_bytes):
            raise ValueError(
                f"hp_bytes ({hp_bytes}) disagrees with block layout "
                f"({block.hp_bytes}); they must describe the same tensor"
            )
        self.block = block

    @property
    def is_seq_along_lane(self) -> bool:
        return self.block is not None and self.block.is_seq_along_lane

    def _row_bytes(self, heads_per_rank: int) -> int:
        if self.block is None:
            return int(heads_per_rank) * self.hp_bytes
        return self.block.row_bytes(heads_per_rank)

    def count_for(self, live_tokens: int) -> int:
        """Strided repeat count for a page holding ``live_tokens`` tokens.

        One run per token under NHD; a single run under HND, where a rank's
        heads are contiguous across the whole page.
        """
        if self.block is None:
            return int(live_tokens)
        return self.block.count_for(live_tokens)

    def _head_span(
        self, head_start: int, num_heads: int, heads_per_rank: int
    ) -> Tuple[int, int]:
        if self.block is None:
            return head_start * self.hp_bytes, num_heads * self.hp_bytes
        return self.block.head_span(head_start, num_heads, heads_per_rank)

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride) for the
        prefill_rank -> decode_rank contribution.

        Offsets and size come from ``BlockLayout.head_span``, so they are whole
        granules on both sides. The strides are the span of the repeat unit:
        one token under NHD, the whole head slab under HND.

        A copy moves one byte count, so the two sides must agree on ``size``.
        They can only disagree when one side pads its granule and the other
        fills it -- precisely the regroup ``head_span`` already rejects -- but
        assert it anyway so a future layout cannot fail quietly here.
        """
        sh_start, num_heads, dh_start, src_hpr, dst_hpr = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        src_off, src_size = self._head_span(sh_start, num_heads, src_hpr)
        dst_off, dst_size = self._head_span(dh_start, num_heads, dst_hpr)
        if src_size != dst_size:
            raise ValueError(
                f"reshard {self.src_tp}->{self.dst_tp} rank {prefill_rank}->"
                f"{decode_rank}: source span {src_size}B and destination span "
                f"{dst_size}B differ for the same {num_heads} kv-head(s); the "
                "two sides round to different granule counts and the copy has "
                "no single size"
            )
        return (
            src_off,  # src_offset
            src_size,  # size
            self._row_bytes(src_hpr),  # src_stride
            dst_off,  # dst_offset
            self._row_bytes(dst_hpr),  # dst_stride
        )

    def validate(self) -> None:
        """Raise unless every rank pair in this reshard is expressible.

        ``entry_bytes`` would raise on its own, but only once a room reached
        the handshake, once per room, from a listener thread. Calling this
        where the geometry is built turns an unsupported deployment into a
        single clear error at the first transfer instead.
        """
        for prefill_rank in range(self.src_tp):
            self.entry_bytes(prefill_rank, decode_rank_for(prefill_rank, self.ratio))


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
            count=geom.count_for(n),
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
                count=geom.count_for(n),
            )
            total += 1
    req.expected_block_count = (
        total if expected_block_count is None else int(expected_block_count)
    )
    return req
