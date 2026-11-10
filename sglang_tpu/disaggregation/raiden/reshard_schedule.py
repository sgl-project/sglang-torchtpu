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

from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple


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
    """Bytes of one kv-head's K+V values, padding slots excluded; the per-head
    granule of a ReshardGeometry built with the tensor's kv_packing."""
    shape = fused_layer_tensor.shape
    return 2 * int(shape[4]) * fused_layer_tensor.element_size()


def per_rank_head_count(fused_layer_tensor: FusedKVLayerTensor) -> int:
    """kv_heads_packed (``shape[2]``): the number of 32-bit packs per token of a
    fused KV layer tensor. That equals this rank's kv-head count only for bf16
    (one head per pack); an fp8 pack holds two heads, so for a kv-head count
    use ``ReshardGeometry.heads_per_rank``."""
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
        *,
        kv_packing: int,
        element_size: int,
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
        # Padded head-dim lanes; one lane of a pack is one 32-bit word.
        self.padded_head_dim = hp_bytes // (2 * element_size)

    def heads_per_rank(self, tp: int) -> int:
        return max(1, self.total_kv_heads // tp)

    def token_bytes(self, tp: int) -> int:
        """Per-token byte width of a rank's fused KV row at attention TP ``tp``:
        its ``2 * heads`` K/V slots rounded up to whole 32-bit packs."""
        packs = -(-2 * self.heads_per_rank(tp) // self.kv_packing)
        return packs * self.kv_packing * self.padded_head_dim * self.element_size

    def check_token_bytes(self, tp: int, actual: int, what: str) -> None:
        """Fail loudly when a rank's real per-token width (``tensor.stride(1) *
        element_size``) is not what this geometry computes for it."""
        expected = self.token_bytes(tp)
        if int(actual) != expected:
            raise ValueError(
                f"Raiden reshard geometry expects a {expected} B per-token KV row "
                f"for {self.heads_per_rank(tp)} kv-head(s) at attention TP {tp} "
                f"(kv_packing={self.kv_packing}, element_size="
                f"{self.element_size}, padded_head_dim={self.padded_head_dim}), but the "
                f"{what} KV tensor's row is {actual} B"
            )

    def entry_bytes(self, prefill_rank: int, decode_rank: int):
        """Return the (src_offset, size, src_stride, dst_offset, dst_stride) 5-tuple for one reshard table entry."""
        sh_start, num_heads, dh_start, src_hpr, dst_hpr = head_slice_params(
            self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
        )
        heads_per_pack = max(1, self.kv_packing // 2)
        whole_packs = num_heads % heads_per_pack == 0 or (
            sh_start + num_heads == src_hpr and dh_start + num_heads == dst_hpr
        )
        if sh_start % heads_per_pack or dh_start % heads_per_pack or not whole_packs:
            raise ValueError(
                f"Partial FP8 head packs are unsupported: {self.src_tp} -> "
                f"{self.dst_tp} splits a {self.kv_packing}-slot pack across ranks "
                f"({src_hpr} src head(s) vs {dst_hpr} dst head(s))."
            )
        packed_heads = -(-num_heads // heads_per_pack) * heads_per_pack
        return self._entry(sh_start, packed_heads, dh_start)

    def _entry(self, src_head: int, num_heads: int, dst_head: int):
        """(src_offset, size, src_stride, dst_offset, dst_stride) of a run of
        ``num_heads`` heads, per token; the strides are the two row widths."""
        return (
            src_head * self.hp_bytes,
            num_heads * self.hp_bytes,
            self.token_bytes(self.src_tp),
            dst_head * self.hp_bytes,
            self.token_bytes(self.dst_tp),
        )

    def check_decode_rank(self, decode_rank: int) -> None:
        """Raise ``entry_bytes``' error for any source of ``decode_rank``, so a
        decode can refuse an unsupported reshard before asking for the KV."""
        for prefill_rank in self.sources_of(decode_rank):
            self.entry_bytes(prefill_rank, decode_rank)

    def sources_of(self, decode_rank: int) -> List[int]:
        """Attention-TP ranks of the prefill that feed ``decode_rank``."""
        if self.src_tp >= self.dst_tp:
            return source_ranks_for(decode_rank % self.dst_tp, self.ratio)
        return [decode_rank % self.dst_tp * self.src_tp // self.dst_tp]

    def slot_bytes(self) -> int:
        """Bytes one K/V slot adds to a token row (a pack row / kv_packing)."""
        return self.padded_head_dim * self.element_size

    def decode_rank_plan(self, decode_rank: int) -> Dict[int, WriterPlan]:
        """Determine whether each prefill source lands DIRECT or STAGED for decode_rank."""
        heads_per_pack = max(1, self.kv_packing // 2)
        src_row = self.token_bytes(self.src_tp)
        dst_row = self.token_bytes(self.dst_tp)
        dst_heads = self.heads_per_rank(self.dst_tp)
        dst_padded_heads = -(-dst_heads // heads_per_pack) * heads_per_pack
        slot_bytes = self.slot_bytes()
        direct: Dict[int, Tuple[int, int, int]] = {}  # rank -> (dh, n, packed)
        staged: List[int] = []
        slices = {}
        for prefill_rank in self.sources_of(decode_rank):
            sh, n, dh, src_hpr, _ = head_slice_params(
                self.src_tp, self.dst_tp, prefill_rank, decode_rank, self.total_kv_heads
            )
            slices[prefill_rank] = (sh, n, dh)
            packed = -(-n // heads_per_pack) * heads_per_pack
            aligned = sh % heads_per_pack == 0 and dh % heads_per_pack == 0
            padded_tail = sh + n == src_hpr and dh + packed <= dst_padded_heads
            if aligned and (n == packed or padded_tail):
                direct[prefill_rank] = (dh, n, packed)
            else:
                staged.append(prefill_rank)
        # Replicas of one kv-head (more prefill ranks than kv-heads) carry the
        # same bytes: they share one staging slot and land identically.
        staged_slices = list(dict.fromkeys(slices[p] for p in staged))
        if len(staged_slices) * src_row > dst_row:
            src_hpr = self.heads_per_rank(self.src_tp)
            dst_hpr = self.heads_per_rank(self.dst_tp)
            raise ValueError(
                f"Partial FP8 head packs are unsupported: staging {len(staged_slices)} "
                f"prefill rows ({len(staged_slices) * src_row} B) exceeds decode row "
                f"({dst_row} B) for {self.src_tp} -> {self.dst_tp} "
                f"({src_hpr} src head(s) vs {dst_hpr} dst head(s))."
            )
        plans: Dict[int, WriterPlan] = {}
        merged = set()
        for prefill_rank in staged:
            sh, n, dh = slices[prefill_rank]
            k = staged_slices.index((sh, n, dh))
            stage_slot = k * (src_row // slot_bytes) + 2 * sh
            merged.update(range(dh, dh + n))
            plans[prefill_rank] = WriterPlan(
                staged=True,
                entry=(0, src_row, src_row, k * src_row, dst_row),
                merges=((stage_slot, 2 * dh, 2 * n),),
            )
        direct_heads, padding = set(), set()
        for prefill_rank, (dh, n, packed) in direct.items():
            direct_heads.update(range(dh, dh + n))
            padding.update(range(dh + n, dh + packed))
            plans[prefill_rank] = WriterPlan(
                staged=False,
                entry=self._entry(slices[prefill_rank][0], packed, dh),
                merges=(),
            )
        # Each decode head's final bytes come from a merge (which runs after
        # every push), or from a direct push no other padded pack overwrites.
        unplaced = [
            h
            for h in range(dst_heads)
            if h not in merged and (h not in direct_heads or h in padding)
        ]
        if unplaced:
            src_hpr = self.heads_per_rank(self.src_tp)
            dst_hpr = self.heads_per_rank(self.dst_tp)
            raise ValueError(
                f"Unsupported reshard layout: {self.src_tp} -> {self.dst_tp} "
                f"splits a {self.kv_packing}-slot pack across ranks "
                f"({src_hpr} src head(s) vs {dst_hpr} dst head(s))."
            )
        return plans

    def writer_plan(self, prefill_rank: int, decode_rank: int) -> WriterPlan:
        return self.decode_rank_plan(decode_rank)[prefill_rank % self.src_tp]

    def merge_spec(self, decode_rank: int) -> Tuple[Tuple[int, int, int], ...]:
        """Return (staging_slot, decode_slot, num_slots) copy specs for staged head merges."""
        plans = self.decode_rank_plan(decode_rank)
        merges = (m for rank in sorted(plans) for m in plans[rank].merges)
        return tuple(dict.fromkeys(merges))  # replicas merge the same slots


class WriterPlan(NamedTuple):
    """How one prefill rank's contribution lands on one decode rank."""

    staged: bool  # lands in the decode's staging pages, not its real pages
    entry: Tuple[int, int, int, int, int]  # per-token schedule entry bytes
    merges: Tuple[Tuple[int, int, int], ...]  # (staging_slot, decode_slot, n)


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
    For a staged writer the decode sent its staging page ids as ``dst_block_ids``.
    """
    if not (len(src_block_ids) == len(dst_block_ids) == len(live_tokens)):
        raise ValueError(
            "src_block_ids, dst_block_ids, live_tokens must be equal length"
        )
    src_off, size, src_stride, dst_off, dst_stride = geom.writer_plan(
        prefill_rank % geom.src_tp, decode_rank
    ).entry
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
    staging_block_ids: Optional[Sequence[int]] = None,
):
    """Receiver-side StartTransferRequest assembling all sources into decode HBM.

    ``sources`` is ``[(sender_id, src_block_ids), ...]`` for the prefill engines
    feeding this decode rank; each gets its own shard schedule with its strided
    dst placement. The receiver selects a source's schedule by the pushing node's
    id (``header.remote_id``), so each schedule is keyed by the sender engine's
    full id, while the byte geometry uses the engine's attention-TP rank within
    its replica (``sender_id % src_tp``). ``dst_mem_type = HBM`` makes the
    receiver auto-H2D the assembled host staging into HBM.

    A staged source (``ReshardGeometry.decode_rank_plan``) lands in
    ``staging_block_ids`` -- decode pages paired one-to-one with
    ``dst_block_ids`` -- instead of the real pages.
    """
    plans = geom.decode_rank_plan(decode_rank)
    req = pb2.StartTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.is_sender = False
    req.dst_mem_type = pb2.MEMORY_TYPE_HBM  # auto-H2D host staging -> decode HBM
    req.use_block_chunks = True
    total = 0
    for sender_id, src_block_ids in sources:
        plan = plans[sender_id % geom.src_tp]
        src_off, size, src_stride, dst_off, dst_stride = plan.entry
        targets = dst_block_ids
        if plan.staged:
            if staging_block_ids is None or len(staging_block_ids) != len(
                dst_block_ids
            ):
                raise ValueError(
                    f"Raiden reshard source {sender_id} lands in staging pages, "
                    f"but got {staging_block_ids!r} for {len(dst_block_ids)} pages"
                )
            targets = staging_block_ids
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
                count=int(n),
            )
            total += 1
    req.expected_block_count = (
        total if expected_block_count is None else int(expected_block_count)
    )
    return req
