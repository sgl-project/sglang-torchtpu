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

The token-major fused KV pool stores one ``[num_blocks, page_size,
granules, kv_packing, padded_head_dim]`` tensor per layer. A token's K/V slots
(K0, V0, K1, V1, ...) are packed ``kv_packing`` per 32-bit word, and the
``padded_head_dim`` words sharing one slot group form a granule: the smallest
contiguous byte run, in the logical layout and in the tiled TPU layout alike.
A bf16 granule holds one head's K+V, an fp8 granule two heads'. The push copies
whole granules, so a contribution is placed directly only when its slots keep
their position within the granule (``KVLayoutGeometry``,
``direct_granule_copy``).

The SEQ_ALONG_LANE pool stores ``[num_blocks, 2 * kv_heads, head_dim //
kv_packing, kv_packing, page_size]``: the packing rides the head dim, and K/V
slot ``2h``/``2h + 1`` is an outer axis, so each head's K+V covers a whole
page as one contiguous run (``head_run_copy``). Any regroup of whole heads is
then one copy per page.

These builders turn that geometry into
``raiden_service_pb2.StartTransferRequest`` push schedules.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import (
    Any,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)


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


def head_bytes(fused_layer_tensor: FusedKVLayerTensor) -> int:
    """Bytes of one kv-head's K+V values, padding lanes excluded; the per-head
    granule of a ReshardGeometry built with the tensor's kv_packing."""
    shape = fused_layer_tensor.shape
    return 2 * int(shape[4]) * fused_layer_tensor.element_size()


TOKEN_MAJOR = "token_major"
SEQ_ALONG_LANE = "seq_along_lane"
# Lanes per (8, 128) tile row: the TPU pads the minor (lane) dim to this.
_LANES = 128


@dataclasses.dataclass(frozen=True)
class KVLayoutGeometry:
    """Byte layout of one rank's KV pages for one layer.

    Token-major (``[num_blocks, page_size, granules, kv_packing, head_dim]``):
    a token is ``granules_per_token`` granules of ``granule_bytes``, each
    holding ``kv_packing`` K/V slots of ``head_dim`` padded lanes; slot ``2h``
    is head ``h``'s K and ``2h + 1`` its V. Pages are ``page_size`` tokens.

    SEQ_ALONG_LANE (``[num_blocks, 2 * heads, head_dim // kv_packing,
    kv_packing, page_size]``): the staged page is a row-major array of 32-bit
    words ``[2 * heads, head_dim // kv_packing, c_pad]`` (token along lanes,
    padded to whole 128-lane tiles), so slot ``j`` is a contiguous
    ``head_dim * element_size * c_pad`` bytes and head ``h`` (slots ``2h``,
    ``2h + 1``) one ``head_run_bytes`` run. ``head_dim`` is the packed lane
    count ``shape[2] * kv_packing``.
    """

    kind: str
    kv_packing: int
    element_size: int
    heads_per_rank: int
    head_dim: int
    page_size: int

    def __post_init__(self):
        if self.kind not in (TOKEN_MAJOR, SEQ_ALONG_LANE):
            raise ValueError(f"unknown KV layout kind {self.kind!r}")
        if self.kv_packing not in (1, 2, 4) or self.kv_packing * self.element_size != 4:
            raise ValueError("Raiden fused KV requires native 32-bit dtype packing")
        if min(self.heads_per_rank, self.head_dim, self.page_size) <= 0:
            raise ValueError(f"degenerate KV layout {self}")

    @classmethod
    def from_tensor(
        cls,
        shape: Sequence[int],
        element_size: int,
        heads_per_rank: int,
        *,
        seq_along_lane: bool,
        page_size: Optional[int] = None,
    ) -> "KVLayoutGeometry":
        """Layout of a 5-D fused KV tensor holding ``heads_per_rank`` kv heads.
        Raises ValueError when the shape cannot hold that many heads."""
        if len(shape) != 5:
            raise ValueError(f"expected a 5-D fused KV tensor, got {tuple(shape)}")
        kv_packing = int(shape[3])
        pages = int(shape[4] if seq_along_lane else shape[1])
        page_size = pages if page_size is None else int(page_size)
        if page_size != pages:
            raise ValueError(f"page size {page_size} != tensor pages {pages}")
        if seq_along_lane:
            rows, want = int(shape[1]), 2 * heads_per_rank
            head_dim = int(shape[2]) * kv_packing
        else:
            rows, want = int(shape[2]), -(-2 * heads_per_rank // kv_packing)
            head_dim = int(shape[4])
        if rows != want:
            raise ValueError(
                f"a KV tensor of shape {tuple(shape)} does not hold "
                f"{heads_per_rank} kv heads per rank"
            )
        return cls(
            kind=SEQ_ALONG_LANE if seq_along_lane else TOKEN_MAJOR,
            kv_packing=kv_packing,
            element_size=int(element_size),
            heads_per_rank=int(heads_per_rank),
            head_dim=head_dim,
            page_size=page_size,
        )

    def with_heads(self, heads_per_rank: int) -> "KVLayoutGeometry":
        """The same layout on a rank holding ``heads_per_rank`` kv heads."""
        return dataclasses.replace(self, heads_per_rank=heads_per_rank)

    def unit(self) -> Tuple[str, int, int, int, int]:
        """Everything but the head count: equal on both sides of a reshard."""
        return (
            self.kind,
            self.kv_packing,
            self.element_size,
            self.head_dim,
            self.page_size,
        )

    def _require(self, kind: str) -> None:
        if self.kind != kind:
            raise ValueError(f"{kind} quantity of a {self.kind} layout")

    @property
    def slots_per_granule(self) -> int:
        self._require(TOKEN_MAJOR)
        return self.kv_packing

    @property
    def heads_per_granule(self) -> Fraction:
        self._require(TOKEN_MAJOR)
        return Fraction(self.kv_packing, 2)

    @property
    def granule_bytes(self) -> int:
        self._require(TOKEN_MAJOR)
        return self.kv_packing * self.head_dim * self.element_size

    @property
    def granules_per_token(self) -> int:
        self._require(TOKEN_MAJOR)
        return -(-2 * self.heads_per_rank // self.kv_packing)

    @property
    def row_bytes(self) -> int:
        """Per-token stride (token-major)."""
        return self.granules_per_token * self.granule_bytes

    @property
    def c_pad(self) -> int:
        """SEQ_ALONG_LANE lanes per staged slot row: ``page_size`` rounded up
        to whole 128-lane tiles (measured on TPU; equal to ``page_size`` at
        every legal SEQ_ALONG_LANE page size, which are multiples of 128)."""
        self._require(SEQ_ALONG_LANE)
        return -(-self.page_size // _LANES) * _LANES

    @property
    def head_run_bytes(self) -> int:
        """SEQ_ALONG_LANE: one head's K+V across the whole page."""
        return 2 * self.head_dim * self.element_size * self.c_pad

    @property
    def block_bytes(self) -> int:
        """Bytes of one staged page (the push's block)."""
        if self.kind == SEQ_ALONG_LANE:
            return self.heads_per_rank * self.head_run_bytes
        return self.page_size * self.row_bytes

    @property
    def copy_unit_bytes(self) -> int:
        """The smallest run a push entry moves: a granule per token, or a
        head across the page."""
        if self.kind == SEQ_ALONG_LANE:
            return self.head_run_bytes
        return self.granule_bytes

    @property
    def head_bytes(self) -> int:
        """Bytes of one kv-head's K+V values per token, padding lanes excluded."""
        return 2 * self.head_dim * self.element_size


class LandRequired(ValueError):
    """A contribution whose bytes no whole-granule copy places correctly; it
    needs a landing buffer and merge (#829)."""


class LandEntry(NamedTuple):
    """Per-token whole-granule copy into a scratch block rather than the
    final page; ``land_merge`` then moves its head into place."""

    src_offset: int
    size: int
    src_stride: int
    dst_offset: int
    dst_stride: int


# The merge, on the logical [blocks, page, granules, kv_packing, head_dim]
# view of an fp8 pool: a landed head sits at scratch slots 0-1 of the granule
# whose final slots 2-3 it owns.
_LAND_PACKING = 4
_LAND_FINAL_SLOTS = slice(2, 4)
_LAND_SCRATCH_SLOTS = slice(0, 2)


def _lands(src: KVLayoutGeometry, num_heads: int, dst_slot: int) -> bool:
    """A one-head fp8 source placed at final slots 2-3 of a granule: the
    class the merge serves. Wider (3-head) sources are not."""
    return (
        src.kv_packing == _LAND_PACKING
        and src.heads_per_rank == num_heads == 1
        and dst_slot % _LAND_PACKING == _LAND_FINAL_SLOTS.start
    )


def direct_granule_copy(
    src: KVLayoutGeometry,
    dst: KVLayoutGeometry,
    src_head_start: int,
    num_heads: int,
    dst_head_start: int,
    *,
    land: bool = False,
) -> Tuple[int, int, int, int, int]:
    """Per-token ``(src_offset, size, src_stride, dst_offset, dst_stride)``
    copying heads ``[src_head_start, +num_heads)`` of ``src`` to
    ``dst_head_start`` of ``dst`` in whole granules, or LandRequired.

    Whole granules move the contribution's slots plus whatever shares their
    first and last granule. That is exact iff every slot keeps its position
    within the granule, the copy stays inside both rows, and no source padding
    slot lands on a destination head (other source heads carried along land
    on their own destination slots: both windows are contiguous global heads).

    With ``land``, a one-head fp8 source that starts mid-granule returns a
    LandEntry: the same granules, aimed at scratch granule
    ``min(dst_slot // g, granules - count)``. A one-head source whose padding
    covers the next destination head copies directly, since that head lands
    and the merge overwrites the padding. Either is exact only as part of a
    whole plan (``land_plan_exact``).
    """
    if src.unit() != dst.unit():
        raise ValueError(f"reshard sides differ in layout: {src} vs {dst}")
    g, gb = src.slots_per_granule, src.granule_bytes
    first_slot, end_slot = 2 * src_head_start, 2 * (src_head_start + num_heads)
    dst_slot = 2 * dst_head_start
    first = first_slot // g
    count = -(-end_slot // g) - first
    lead = first_slot - first * g
    dst_first, misaligned = divmod(dst_slot - lead, g)
    if (
        misaligned
        or dst_first < 0
        or first + count > src.granules_per_token
        or dst_first + count > dst.granules_per_token
    ):
        q = min(dst_slot // g, dst.granules_per_token - count)
        if (
            land
            and _lands(src, num_heads, dst_slot)
            and lead == 0
            and q * g + _LAND_FINAL_SLOTS.start == dst_slot
        ):
            return LandEntry(
                first * gb, count * gb, src.row_bytes, q * gb, dst.row_bytes
            )
        raise LandRequired(
            f"heads [{src_head_start}, {src_head_start + num_heads}) of "
            f"{src.heads_per_rank} cannot land at head {dst_head_start} of "
            f"{dst.heads_per_rank} by whole {g}-slot granules"
        )
    src_pad = 2 * src.heads_per_rank
    if (
        (first + count) * g > src_pad
        and dst_slot + src_pad - first_slot < 2 * dst.heads_per_rank
        and not (land and _lands(src, num_heads, dst_slot + src_pad - first_slot))
    ):
        raise LandRequired(
            f"source padding after head {src.heads_per_rank - 1} would "
            f"overwrite destination head {dst_head_start + num_heads}"
        )
    return (
        first * gb,
        count * gb,
        src.row_bytes,
        dst_first * gb,
        dst.row_bytes,
    )


def head_run_copy(
    src: KVLayoutGeometry,
    dst: KVLayoutGeometry,
    src_head_start: int,
    num_heads: int,
    dst_head_start: int,
) -> Tuple[int, int, int, int, int]:
    """Per-page ``(src_offset, size, src_stride, dst_offset, dst_stride)``
    copying heads ``[src_head_start, +num_heads)`` of a SEQ_ALONG_LANE ``src``
    page to ``dst_head_start`` of ``dst``: one contiguous run, since a head's
    K and V slots are adjacent rows that each span the page. Strides are the
    page sizes (one run per page)."""
    if src.unit() != dst.unit():
        raise ValueError(f"reshard sides differ in layout: {src} vs {dst}")
    src._require(SEQ_ALONG_LANE)
    if (
        min(src_head_start, dst_head_start) < 0
        or num_heads <= 0
        or src_head_start + num_heads > src.heads_per_rank
        or dst_head_start + num_heads > dst.heads_per_rank
    ):
        raise ValueError(
            f"heads [{src_head_start}, {src_head_start + num_heads}) of "
            f"{src.heads_per_rank} do not fit at head {dst_head_start} of "
            f"{dst.heads_per_rank}"
        )
    run = src.head_run_bytes
    return (
        src_head_start * run,
        num_heads * run,
        src.block_bytes,
        dst_head_start * run,
        dst.block_bytes,
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
    """Byte geometry for one (src_tp -> dst_tp) reshard over ``total_kv_heads``.

    ``from_layout`` derives every entry from the pool layout: whole granules
    per token (token-major) or whole head runs per page (SEQ_ALONG_LANE). The
    constructor takes an explicit per-head unit ``hp_bytes`` instead, for the
    pipeline (per-head unit with the tensor's kv_packing).
    """

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
        self.layout: Optional[KVLayoutGeometry] = None
        self.land = False

    @classmethod
    def from_layout(
        cls,
        src_tp: int,
        dst_tp: int,
        layout: KVLayoutGeometry,
        total_kv_heads: int,
        *,
        land: bool = False,
    ) -> "ReshardGeometry":
        """Geometry over ``layout`` (either side's: only its head count
        differs). ``entry_bytes`` raises LandRequired for a token-major
        contribution no whole-granule copy places, or with ``land`` returns
        a LandEntry for one the merge places."""
        geom = cls(
            src_tp,
            dst_tp,
            layout.copy_unit_bytes,
            total_kv_heads,
            kv_packing=layout.kv_packing,
            element_size=layout.element_size,
        )
        geom.layout = layout
        geom.land = land
        return geom

    def with_land(self) -> "ReshardGeometry":
        """This geometry, classifying landing contributions as LandEntry."""
        if self.layout is None:
            return self
        return ReshardGeometry.from_layout(
            self.src_tp, self.dst_tp, self.layout, self.total_kv_heads, land=True
        )

    def count_for(self, live_tokens: int) -> int:
        """Entry repeat count for a page holding ``live_tokens`` tokens: one
        chunk per token, or one run for a SEQ_ALONG_LANE page."""
        if self.layout is not None and self.layout.kind == SEQ_ALONG_LANE:
            return 1
        return int(live_tokens)

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Byte (src_offset, size, src_stride, dst_offset, dst_stride) for the
        prefill_rank -> decode_rank contribution, repeated ``count_for`` times
        per page; a LandEntry targets the scratch block."""
        sh_start, num_heads, dh_start, src_hpr, dst_hpr = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        if self.layout is not None:
            src = self.layout.with_heads(src_hpr)
            dst = self.layout.with_heads(dst_hpr)
            if self.layout.kind == SEQ_ALONG_LANE:
                return head_run_copy(src, dst, sh_start, num_heads, dh_start)
            return direct_granule_copy(
                src, dst, sh_start, num_heads, dh_start, land=self.land
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


def _typed(schedule: Optional[List[Tuple[int, ...]]]) -> Optional[List[Any]]:
    """A schedule compared with its entry kinds: a LandEntry equals the plain
    tuple of its values."""
    if schedule is None:
        return None
    return [(isinstance(e, LandEntry), tuple(e)) for e in schedule]


def land_merge_slots(layout: KVLayoutGeometry) -> Tuple[slice, slice]:
    """``(final, scratch)`` slices of the kv_packing axis that the merge
    moves, on the logical ``[blocks, page, granules, kv_packing, head_dim]``
    view of a token-major fp8 pool: every granule's final slots 2-3 take its
    scratch slots 0-1."""
    layout._require(TOKEN_MAJOR)
    if layout.kv_packing != _LAND_PACKING:
        raise ValueError(f"no landing merge for kv_packing {layout.kv_packing}")
    return _LAND_FINAL_SLOTS, _LAND_SCRATCH_SLOTS


def land_merge(final: Any, scratch: Any, layout: KVLayoutGeometry) -> Any:
    """Merge gathered scratch pages into gathered final pages in place, both
    in the logical ``[..., granules, kv_packing, head_dim]`` layout (numpy or
    torch, any same-size dtype view). Returns ``final``."""
    dst, src = land_merge_slots(layout)
    final[..., dst, :] = scratch[..., src, :]
    return final


def _rank_heads(total: int, tp: int, rank: int) -> List[int]:
    """Global kv heads attention-TP ``rank`` holds; replicated below tp."""
    if total >= tp:
        per_rank = total // tp
        return list(range(rank * per_rank, (rank + 1) * per_rank))
    return [rank // (tp // total)]


def land_plan_exact(geom: ReshardGeometry) -> bool:
    """Whether ``geom``'s landing schedule, followed on every plan holding a
    LandEntry by ``land_merge``, leaves each decode rank exactly its own
    heads. Checked per K/V slot from the global head ownership, independent
    of ``head_slice_params``: a source's padding must be overwritten by the
    merge. An odd head count per decode rank is refused conservatively: its
    last granule's slots 2-3 are padding, and pad slots must not receive
    scratch bytes."""
    lay = geom.layout
    if (
        lay is None
        or lay.kind != TOKEN_MAJOR
        or lay.kv_packing != _LAND_PACKING
        or geom.src_tp <= geom.dst_tp
    ):
        return False
    total, g, gb = geom.total_kv_heads, lay.kv_packing, lay.granule_bytes
    src = lay.with_heads(max(1, total // geom.src_tp))
    dst = lay.with_heads(max(1, total // geom.dst_tp))
    if dst.heads_per_rank % 2:
        return False
    width = dst.granules_per_token * g
    for d in range(geom.dst_tp):
        final: List[Any] = [None] * width
        scratch: List[Any] = [None] * width
        landed = False
        for p in source_ranks_for(d, geom.ratio):
            try:
                e = geom.entry_bytes(p, d)
            except LandRequired:
                return False
            if (
                any(x % gb for x in (e[0], e[1], e[3]))
                or (e[2], e[4]) != (src.row_bytes, dst.row_bytes)
                or e[0] + e[1] > src.row_bytes
                or e[3] + e[1] > dst.row_bytes
            ):
                return False
            slots = [
                (h, kv) for h in _rank_heads(total, geom.src_tp, p) for kv in (0, 1)
            ]
            slots += [None] * (src.granules_per_token * g - len(slots))
            landed |= isinstance(e, LandEntry)
            out = scratch if isinstance(e, LandEntry) else final
            first, q = e[0] // gb * g, e[3] // gb * g
            out[q : q + e[1] // gb * g] = slots[first : first + e[1] // gb * g]
        if landed:
            for k in range(0, width, g):
                final[k + _LAND_FINAL_SLOTS.start : k + _LAND_FINAL_SLOTS.stop] = (
                    scratch[
                        k + _LAND_SCRATCH_SLOTS.start : k + _LAND_SCRATCH_SLOTS.stop
                    ]
                )
        want = [(h, kv) for h in _rank_heads(total, geom.dst_tp, d) for kv in (0, 1)]
        if final[: len(want)] != want:
            return False
    return True


def push_reshard_refusal(
    geom: ReshardGeometry,
    shape: Sequence[int],
    element_size: int,
    *,
    total_heads: int,
    own_tp: int,
    page_size: int,
    seq_along_lane: bool,
    land_ok: bool = False,
) -> Optional[str]:
    """Why pushing one KV layer tensor with ``geom``'s schedule would misplace
    bytes, or None when the copy is byte-exact.

    ``shape``/``element_size`` are this side's tensor (attention TP
    ``own_tp``) and ``total_heads`` the global kv-head count it really holds,
    which differs from ``geom.total_kv_heads`` for draft layers. The copy is
    exact iff ``geom`` plans exactly the schedule of the tensor's own layout:
    whole granules per token (``direct_granule_copy``) or whole head runs per
    page (``head_run_copy``).

    ``land_ok`` also allows a schedule that needs scratch blocks and a merge
    when ``land_plan_exact`` holds; the caller passes it only for groups that
    have a landing arena (not sliding-window).
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
    mixed = (
        f"{where} moves every layer with one geometry, but its KV layers "
        "differ in head_dim or dtype (e.g. a draft model's KV)"
    )
    if pk * element_size != 4 or (geom.layout is None and hp != geom.hp_bytes):
        return mixed
    src_heads = max(1, total_heads // src_tp)
    dst_heads = max(1, total_heads // dst_tp)
    own_heads = max(1, total_heads // own_tp)
    rows = 2 * own_heads if seq_along_lane else -(-2 * own_heads // pk)
    if int(shape[1 if seq_along_lane else 2]) != rows:
        return (
            f"{where}: a KV tensor of shape {tuple(shape)} does not hold "
            f"{own_heads} kv heads per rank"
        )
    layout = KVLayoutGeometry.from_tensor(
        shape,
        element_size,
        own_heads,
        seq_along_lane=seq_along_lane,
        page_size=page_size if seq_along_lane else None,
    )
    if geom.layout is not None and geom.layout.unit() != layout.unit():
        return mixed
    try:
        exact = _schedule(
            ReshardGeometry.from_layout(src_tp, dst_tp, layout, total_heads)
        )
    except LandRequired as e:
        landing = ReshardGeometry.from_layout(
            src_tp, dst_tp, layout, total_heads, land=True
        )
        if not (land_ok and land_plan_exact(landing)):
            packed = "fp8" if pk == 4 else f"kv_packing {pk}"
            return (
                f"{where} cannot place {packed} KV of {src_heads} kv heads per "
                f"prefill rank into {dst_heads} per decode rank by whole 32-bit "
                f"packs ({e}), which needs a landing buffer (#829). Use equal "
                "attention TP or a bf16 KV cache"
            )
        exact = _schedule(landing)
        geom = geom.with_land()
    try:
        planned = _schedule(geom)
    except ValueError:
        planned = None
    if _typed(planned) == _typed(exact):
        return None
    if geom.total_kv_heads != total_heads:
        return (
            f"{where} would misplace KV: its schedule assumes "
            f"{geom.total_kv_heads} kv heads, a KV layer holds {total_heads}"
        )
    return f"{where} would misplace KV: its schedule does not match the KV layout"


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


def _target_blocks(
    entry: Tuple[int, ...],
    dst_block_ids: Sequence[int],
    scratch_block_ids: Optional[Sequence[int]],
) -> Sequence[int]:
    """The decode blocks a contribution writes: the final pages, or for a
    LandEntry the scratch pages paired with them."""
    if not isinstance(entry, LandEntry):
        return dst_block_ids
    if scratch_block_ids is None:
        raise ValueError("a landing contribution needs scratch_block_ids")
    if len(scratch_block_ids) != len(dst_block_ids):
        raise ValueError("scratch_block_ids and dst_block_ids must be equal length")
    return scratch_block_ids


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
    scratch_block_ids: Optional[Sequence[int]] = None,
):
    """Sender-side StartTransferRequest for one prefill rank pushing to one decode peer.

    ``src_block_ids``/``dst_block_ids`` are physical page ids on each side, paired
    by token order; ``live_tokens[i]`` is the token count of page ``i``. A
    landing contribution writes ``scratch_block_ids[i]`` instead of
    ``dst_block_ids[i]``.
    """
    if not (len(src_block_ids) == len(dst_block_ids) == len(live_tokens)):
        raise ValueError(
            "src_block_ids, dst_block_ids, live_tokens must be equal length"
        )
    entry = geom.entry_bytes(prefill_rank % geom.src_tp, decode_rank)
    src_off, size, src_stride, dst_off, dst_stride = entry
    dst_block_ids = _target_blocks(entry, dst_block_ids, scratch_block_ids)
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
    scratch_block_ids: Optional[Sequence[int]] = None,
):
    """Receiver-side StartTransferRequest assembling all sources into decode HBM.

    ``sources`` is ``[(sender_id, src_block_ids), ...]`` for the prefill engines
    feeding this decode rank; each gets its own shard schedule with its strided
    dst placement. The receiver selects a source's schedule by the pushing node's
    id (``header.remote_id``), so each schedule is keyed by the sender engine's
    full id, while the byte geometry uses the engine's attention-TP rank within
    its replica (``sender_id % src_tp``). ``dst_mem_type = HBM`` makes the
    receiver auto-H2D the assembled host staging into HBM. A landing source's
    entries write ``scratch_block_ids``, as its sender's do: the receiver
    matches entries on (dst block, src block).
    """
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = False
    req.dst_mem_type = pb2.MEMORY_TYPE_HBM  # auto-H2D host staging -> decode HBM
    req.use_block_chunks = True
    total = 0
    for sender_id, src_block_ids in sources:
        entry = geom.entry_bytes(sender_id % geom.src_tp, decode_rank)
        src_off, size, src_stride, dst_off, dst_stride = entry
        targets = _target_blocks(entry, dst_block_ids, scratch_block_ids)
        schedule = req.shard_push_schedules[sender_id]
        for src_b, dst_b, n in zip(src_block_ids, targets, live_tokens):
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
