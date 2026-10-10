"""TPU KV-cache page transfer primitives for P/D disaggregation.

This module is the *device-specific data plane* for the ``tpu`` transfer
backend. The control plane (bootstrap handshake, ZMQ coordination, the
``KVPoll`` state machine) is inherited from ``disaggregation/common`` and
lives in ``conn.py``; everything here is pure tensor / host-buffer movement
so it can be unit-tested on plain CPU tensors without a TPU attached.

Why a separate data plane at all? GPU backends (mooncake, nixl, mori) RDMA
directly into the *remote* process's KV buffer using raw device pointers.
That is impossible on TPU: PjRt binds one device per process and device
buffer addresses are not meaningful across processes. So the TPU backend
moves bytes the only portable way — copy the prefill pages off-device to
host memory (D2H), ship the bytes over the host network (ZMQ, in conn.py),
and copy them back onto the decode device (H2D) into pages the *decode*
process allocated itself.

The KV pool used by the default Pallas RPA v3 attention backend
(``TpuFusedMHATokenToKVPool``) stores one fused 5D buffer per layer with
shape ``[num_blocks, page_size, num_kv_heads_x2 // kv_packing, kv_packing,
head_dim]``. The disaggregation scheduler hands us *block ids* (the output
of ``kv_to_page_indices`` == ``loc // page_size``), which index dim 0
directly — so a transfer is an ``index_select`` (gather) on the prefill
side and an ``index_copy_`` (scatter) on the decode side, with the bytes in
between treated as an opaque dtype-tagged blob.
"""

from __future__ import annotations

import ctypes
import enum
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs
from sglang_tpu.disaggregation.tpu.host_staging import PinnedHostPool, StageBuf

logger = logging.getLogger(__name__)


def _blob_to_tensor(blob: StageBuf | bytes) -> torch.Tensor:
    if isinstance(blob, StageBuf):
        return blob._t
    return torch.frombuffer(blob, dtype=torch.uint8)


# A KV head occupies 2 packed slots (its K and V lanes) in the kv_packing axis.
_SLOTS_PER_HEAD = 2


def _view_packed_lane(
    blob: StageBuf | bytes,
    num_pages: int,
    page_size: int,
    n_lanes: int,
    kv_packing: int,
) -> torch.Tensor:
    """Views a gathered layer blob as its 5D packed layout for slot-axis slicing.

    Reshaping to (num_pages, page_size, n_lanes, kv_packing, slot_bytes) lets a
    head be sliced on the kv_packing axis instead of the flat page bytes.
    slot_bytes (head_dim * dtype_size) is inferred; a wrong page_size makes the
    view raise rather than silently mis-slice.

    Args:
        blob: Gathered per-layer payload (StageBuf or raw bytes).
        num_pages: Pages in the blob.
        page_size: Tokens per KV page.
        n_lanes: Packed lanes present in the blob.
        kv_packing: Slots per packed lane.

    Returns:
        uint8 view of shape (num_pages, page_size, n_lanes, kv_packing, slot_bytes).
    """
    return _blob_to_tensor(blob).view(num_pages, page_size, n_lanes, kv_packing, -1)


# ``(start, count)`` on the fused buffer's packed-lane axis (dim 2), or None for
# a whole-page transfer (the equal-attn_tp path). Used by the resharding gather
# /scatter helpers.
LaneSlice = Optional[tuple[int, int]]


class ReshardUnsupportedError(ValueError):
    """Unsuppprted KV-head resharding the TPU packed-lane layout."""


class ReshardDirection(enum.Enum):
    """Describe how do we do resharding, standing from decoder's view.

    If one decoder has smaller attention tp, i.e each decoder TP's head
    will be not less than each prefiller TP's head, then it's FAN-IN.
    If one decoder has larger attention tp, i.e each prefiller TP's head will
    not be less thanthan each decoder TP's head, then it's FAN-OUT.

    Current implementation does prune any traffic. So each prefiller TP will
    send it's data to decoder TP even if there are two TP under same DP group
    which happened to have duplicate heads."""

    UNSPECIFIED = "unspecified"
    # Prefill attention tp == Decode attention tp.
    EQUAL = "equal"
    # Prefill attention tp > Decode attention tp.
    FAN_IN = "fan-in"
    # Prefill attention tp < Decode attention tp.
    FAN_OUT = "fan-out"


@dataclass
class PrefillTransferInfo:
    """Prefill-side record for target decode endpoint and resharding state.

    The dataclass will infer exact data spec to start data push. For case like
    FP8 src_attn_tp_size = 1 (Not enough heads for kv packing), minimum 1 lane
    will be sent.

    Attributes:
      room: Disagg transfer room ID.
      decode_endpoint: Destination socket endpoint address.
      num_pages: Total page count to transfer.
      src_tp_rank: Prefiller tensor parallel rank.
      dst_tp_rank: Decoder tensor parallel rank.
      src_attn_tp_size: Attention TP size of prefill engine.
      dst_attn_tp_size: Attention TP size of decode engine.
      total_num_head: Total attention head count.
      kv_packing: Kv packing number, for FP8 it's 4.
      n_layers: Number of model layers.
      page_size: Tokens per KV page (dim 1 of the fused buffer); needed to
        reshape a gathered blob to its packed layout during odd-head reassembly.
      gather_time_ms: Total D2H gather duration in milliseconds.
      push_time_ms: Total socket push duration in milliseconds.
      push_start_time: Monotonic start timestamp of push.
    """

    room: int
    decode_endpoint: str
    num_pages: int
    kv_packing: int
    src_tp_rank: int
    dst_tp_rank: int
    src_attn_tp_size: int
    dst_attn_tp_size: int
    total_num_head: int
    n_layers: int
    page_size: int
    gather_time_ms: float = 0.0
    push_time_ms: float = 0.0
    push_start_time: float = 0.0

    def __post_init__(self) -> None:
        self._reshard_error: str | None = None
        # Neutral geometry for an unsupported reshard: the transfer is failed
        # by check_status, but the sender thread may read these properties
        # before that, so every derived attribute must exist.
        self.src_attn_tp_rank = self.src_tp_rank % self.src_attn_tp_size
        self.dst_attn_tp_rank = self.dst_tp_rank % self.dst_attn_tp_size
        self.src_heads = self.dst_heads = 0
        self.src_replication = self.dst_replication = 1
        self.resharding_direction = ReshardDirection.UNSPECIFIED
        self.is_src_replica = self.is_dst_replica = False
        self.src_head_start = self.dst_head_start = 0
        self.n_head_to_send = 0
        self.src_lane_idx = self.dst_lane_idx = 0
        self.n_lane_to_send = 0
        if self.total_num_head <= 0:
            self._reshard_error = (
                f"total_num_head must be > 0, got {self.total_num_head}"
            )
            return

        self.src_heads = max(1, self.total_num_head // self.src_attn_tp_size)
        self.dst_heads = max(1, self.total_num_head // self.dst_attn_tp_size)
        self.src_replication = max(1, self.src_attn_tp_size // self.total_num_head)
        self.dst_replication = max(1, self.dst_attn_tp_size // self.total_num_head)
        self.resharding_direction = ReshardDirection.UNSPECIFIED
        self.is_src_replica = self.src_tp_rank % self.src_replication != 0
        self.is_dst_replica = self.dst_tp_rank % self.dst_replication != 0

        if self.src_attn_tp_size > self.dst_attn_tp_size:
            if self.src_attn_tp_size % self.dst_attn_tp_size != 0:
                raise ReshardUnsupportedError()
            self.resharding_direction = ReshardDirection.FAN_IN
            self.src_head_start = 0
            self.dst_head_start = (
                self.src_attn_tp_rank // self.src_replication * self.src_heads
            ) % self.dst_heads
            self.n_head_to_send = self.src_heads
        elif self.src_attn_tp_size < self.dst_attn_tp_size:
            if self.dst_attn_tp_size % self.src_attn_tp_size != 0:
                raise ReshardUnsupportedError()
            self.resharding_direction = ReshardDirection.FAN_OUT
            self.src_head_start = (
                self.dst_attn_tp_rank // self.dst_replication * self.dst_heads
            ) % self.src_heads
            self.dst_head_start = 0
            self.n_head_to_send = self.dst_heads
        else:
            self.resharding_direction = ReshardDirection.EQUAL
            self.src_head_start = 0
            self.dst_head_start = 0
            self.n_head_to_send = self.src_heads

        self.src_lane_idx = self.src_head_start * 2 // self.kv_packing
        self.dst_lane_idx = self.dst_head_start * 2 // self.kv_packing
        self.n_lane_to_send = (
            self.n_head_to_send * 2 + self.kv_packing - 1
        ) // self.kv_packing

    @property
    def if_skip_sending(self) -> bool:
        """Skip prefill sending if possible.

        Current only FANIN, multiple prefill -> decode will drop pushing
        replicated chunks.
        """
        return (
            self.resharding_direction == ReshardDirection.FAN_IN and self.is_src_replica
        )

    @property
    def if_send_aux_data(self) -> bool:
        # Why dst_head_start == 0:
        # 1. Single receive: Each decode rank has exactly one prefill source mapped
        #    to dst_head_start == 0, ensuring each decode rank receives aux_data once.
        # 2. Idempotent: Aux metadata is written via C-memcpy to ptr + len*aux_index,
        #    so duplicate writes to the same row are completely safe and idempotent.
        return self.dst_head_start == 0

    @property
    def total_lanes(self) -> int:
        return max(self.src_heads * 2, self.kv_packing) // self.kv_packing

    @property
    def require_slicing(self) -> bool:
        return self.n_lane_to_send != self.total_lanes

    @property
    def gathered_lanes(self) -> int:
        """Lane count actually present in the gathered blob: the sliced count
        when a sub-range was gathered, else this rank's full lane set."""
        return self.n_lane_to_send if self.require_slicing else self.total_lanes


@dataclass
class DecodeTransferState:
    """Decode-side per-request transfer bookkeeping.

    Attributes:
      dst_page_indices: Block IDs on this process.
      aux_index: Aux state slot index.
      dst_lane_per_page_block: Number of lanes per block.
      num_pages: Total transfer page count.
      dst_lanes_total: Total expected lanes to scatter.
      scattered: Number of lanes scattered so far.
    """

    dst_page_indices: np.ndarray
    aux_index: int | None
    dst_lane_per_page_block: int
    num_pages: int = field(init=False)
    dst_lanes_total: int = field(init=False)
    scattered: int = 0

    def __post_init__(self) -> None:
        self.num_pages = len(self.dst_page_indices)
        self.dst_lanes_total = self.num_pages * self.dst_lane_per_page_block


@dataclass
class PendingChunk:
    """Received KV data chunk awaiting main-thread scatter.

    Attributes:
      index_start: Starting page index for the chunk.
      num_pages: Number of pages in the chunk.
      layer_blobs: Per-layer KV buffers; StageBuf for pinned DMA pool or bytes
        for fallback/reassembled.
      aux_blobs: Aux state byte payloads.
      prefill_transfer_info: Prefill transfer info.
      from_assemble: Whether this chunk was produced by reassembly.
    """

    index_start: int
    num_pages: int
    layer_blobs: list[StageBuf | bytes]
    aux_blobs: list[bytes]
    prefill_transfer_info: PrefillTransferInfo
    from_assemble: bool = False

    @property
    def nbytes(self) -> int:
        """Total bytes carried by layer_blobs."""
        return sum(
            b.nbytes if hasattr(b, "nbytes") else len(b) for b in self.layer_blobs
        )

    @property
    def require_reassemble(self) -> bool:
        """Returns True if the prefill or decode kv head * 2 < kv_packing and
        we need resharding."""
        info = self.prefill_transfer_info
        if self.from_assemble:
            return False
        # Same heads-per-rank on both sides => identical packed-lane layout, so a
        # direct whole-lane copy suffices and any extra prefill ranks are pure GQA
        # replicas (idempotent overwrite) -- no host repack. Covers e.g.
        # total_num_head=4, attn_tp 8->4 (1 head each), which is FAN_IN by attn_tp
        # but needs no resharding.
        if info.src_heads == info.dst_heads:
            return False
        if not (
            2 * info.src_heads < info.kv_packing or 2 * info.dst_heads < info.kv_packing
        ):
            return False
        if info.resharding_direction == ReshardDirection.EQUAL:
            return False
        return not (info.src_heads > info.dst_heads and info.src_head_start == 0)

    @property
    def assemble_key(self) -> tuple[int, int]:
        """key used for assemble."""
        return (self.prefill_transfer_info.room, self.index_start)

    @property
    def assemble_min_chunks(self) -> int:
        """How many prefill chunks need to be merged into a single decode chunk."""
        info = self.prefill_transfer_info
        if info.resharding_direction == ReshardDirection.FAN_IN:
            return max(1, info.src_attn_tp_size // info.dst_attn_tp_size)
        return 1

    @staticmethod
    def reassemble_chunks(chunks: list[PendingChunk]) -> PendingChunk:
        """Merges/extracts chunks based on each chunks's prefill_transfer_chunk.

        For FP8 cases, if prefill rank only has 1 head, it will still sends
        decoder 1 lane with half of slots dummy. On the other hand, if decode
        only has 1 head, its final data will be 1 lane with half of slots dummy.

        We only consider 1:2 and 2:1 odd head case here.

        3 cases to be considered here:
        1. FAN-IN (prefill data have 1 heads), we will combine 2 prefill chunks
            into a single contiguous chunk (strip out dummy data) and then
            scatter.
        2. FAN-OUT (Decode data have 1 heads):
           2.1 need first 2 slots from prefill, then directly use
           original chunk.
           2.2 need second half slots from prefill, then we create a new chunk
           on host where first 2 slots are the copied from the prefill chunks
        3. EQUAL (Both sides only have 1 heads):
            No change.

        Args:
            chunks: List of chunks to be assemble to 1 chunk.
        Returns:
            Assembled pending chunk.
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        first_chunk = chunks[0]
        info = first_chunk.prefill_transfer_info
        direction = info.resharding_direction

        # Case 1: FAN-IN (Prefill has 1 head -> Decode has 2 heads)
        if direction == ReshardDirection.FAN_IN:
            sorted_chunks = sorted(
                chunks, key=lambda c: c.prefill_transfer_info.dst_head_start
            )
            num_layers = len(first_chunk.layer_blobs)
            new_layer_blobs: list[bytes] = []
            num_pages, page_size = first_chunk.num_pages, info.page_size
            n_lanes, kvp = info.gathered_lanes, info.kv_packing
            # Assumes exactly 1 real head per source: the slice below strips all
            # but the first slot pair. require_reassemble only routes here when
            # 2*src_heads < kv_packing, which with the fp8 kv_packing=4 forces
            # src_heads==1. Guard so a future kv_packing>4 fails loud instead of
            # silently dropping heads (the fix is slicing :src_heads*_SLOTS_PER_HEAD).
            assert info.src_heads == 1, (
                f"FAN-IN reassembly assumes 1 head/source, got src_heads={info.src_heads}"
            )
            for layer_idx in range(num_layers):
                # Each source carries its real head in the lane's first slot pair
                # (slots [0:2]); concatenate the sources on the kv_packing axis to
                # fill the decode lane, stripping the trailing dummy slots.
                heads = [
                    _view_packed_lane(
                        c.layer_blobs[layer_idx], num_pages, page_size, n_lanes, kvp
                    )[:, :, :, :_SLOTS_PER_HEAD, :]
                    for c in sorted_chunks
                ]
                merged = torch.cat(heads, dim=3)
                new_layer_blobs.append(merged.contiguous().numpy().tobytes())

            for c in sorted_chunks:
                for b in c.layer_blobs:
                    if isinstance(b, StageBuf):
                        b.release()

            aux_blobs = []
            for c in sorted_chunks:
                if c.aux_blobs:
                    aux_blobs = c.aux_blobs
                    break

            new_info = PrefillTransferInfo(
                room=info.room,
                decode_endpoint=info.decode_endpoint,
                num_pages=info.num_pages,
                kv_packing=info.kv_packing,
                # Synthetic chunk, so use dst rank.
                src_tp_rank=info.dst_tp_rank,
                dst_tp_rank=info.dst_tp_rank,
                src_attn_tp_size=info.dst_attn_tp_size,
                dst_attn_tp_size=info.dst_attn_tp_size,
                total_num_head=info.total_num_head,
                n_layers=info.n_layers,
                page_size=info.page_size,
            )

            new_chunk = PendingChunk(
                index_start=first_chunk.index_start,
                num_pages=first_chunk.num_pages,
                layer_blobs=new_layer_blobs,
                aux_blobs=aux_blobs,
                prefill_transfer_info=new_info,
                from_assemble=True,
            )
            return new_chunk

        # Case 2: FAN-OUT (Decode has 1 head -> Prefill sent 2-head full lane)
        elif direction == ReshardDirection.FAN_OUT:
            # Case 2.1: Head 0 is needed -> original chunk already has valid Head 0 in front
            if info.src_head_start == 0:
                return first_chunk

            # Case 2.2: Head 1 is needed -> move its slot pair (slots [2:4]) to the
            # front of the lane, leaving the trailing slots as dummy padding.
            num_layers = len(first_chunk.layer_blobs)
            new_layer_blobs: list[bytes] = []
            num_pages, page_size = first_chunk.num_pages, info.page_size
            n_lanes, kvp = info.gathered_lanes, info.kv_packing

            for layer_idx in range(num_layers):
                lane = _view_packed_lane(
                    first_chunk.layer_blobs[layer_idx],
                    num_pages,
                    page_size,
                    n_lanes,
                    kvp,
                )
                h1 = lane[:, :, :, _SLOTS_PER_HEAD : 2 * _SLOTS_PER_HEAD, :]
                dummy = torch.zeros_like(h1)
                merged = torch.cat([h1, dummy], dim=3)
                new_layer_blobs.append(merged.contiguous().numpy().tobytes())

            for b in first_chunk.layer_blobs:
                if isinstance(b, StageBuf):
                    b.release()

            new_info = PrefillTransferInfo(
                room=info.room,
                decode_endpoint=info.decode_endpoint,
                num_pages=info.num_pages,
                kv_packing=info.kv_packing,
                # Use src as it's the only one.
                src_tp_rank=info.src_tp_rank,
                dst_tp_rank=info.dst_tp_rank,
                src_attn_tp_size=info.dst_attn_tp_size,
                dst_attn_tp_size=info.dst_attn_tp_size,
                total_num_head=info.total_num_head,
                n_layers=info.n_layers,
                page_size=info.page_size,
            )

            new_chunk = PendingChunk(
                index_start=first_chunk.index_start,
                num_pages=first_chunk.num_pages,
                layer_blobs=new_layer_blobs,
                aux_blobs=first_chunk.aux_blobs,
                prefill_transfer_info=new_info,
                from_assemble=True,
            )
            return new_chunk

        # Case 3: EQUAL
        else:
            return first_chunk

    @staticmethod
    def process_incoming_chunk(
        pending_queue: deque[PendingChunk],
        reasm_dict: dict[tuple, list[PendingChunk]],
        chunk: PendingChunk,
        stats: Any | None = None,
    ) -> None:
        """Enqueues or merges an incoming pending chunk into the queue.

        For each decoder, even if itself is replica, it will always receive data
        from prefill. However, within each DPA group. If prefill TP rank is a
        replica in FANIN case, it will never send these chunks.

        Args:
            pending_queue: Queue of ready chunks for scatter.
            reasm_dict: Dictionary mapping assemble_key to pending un-assembled chunks.
            chunk: Newly received chunk awaiting scatter or reassembly.
            stats: Optional stats recorder for measuring assemble duration.
        """
        info = chunk.prefill_transfer_info
        if not chunk.require_reassemble:
            pending_queue.append(chunk)
            return

        min_chunks = chunk.assemble_min_chunks
        if min_chunks <= 1:
            _t0 = time.perf_counter()
            merged_chunk = PendingChunk.reassemble_chunks([chunk])
            if stats is not None:
                stats.record(time.perf_counter() - _t0, merged_chunk.nbytes)
            logger.info(
                "[pd_kv_resharding] process_incoming_chunk (single reassemble): "
                "room=%s index_start=%d input_bytes=%d merged_bytes=%d",
                info.room,
                chunk.index_start,
                chunk.nbytes,
                merged_chunk.nbytes,
            )
            pending_queue.append(merged_chunk)
            return

        key = chunk.assemble_key
        bucket = reasm_dict[key]

        if not any(
            c.prefill_transfer_info.src_tp_rank
            == chunk.prefill_transfer_info.src_tp_rank
            for c in bucket
        ):
            bucket.append(chunk)

        logger.info(
            "[pd_kv_resharding] process_incoming_chunk bucket state: "
            "room=%s index_start=%d src_tp_rank=%d bucket_len=%d/%d chunk_bytes=%d",
            info.room,
            chunk.index_start,
            info.src_tp_rank,
            len(bucket),
            min_chunks,
            chunk.nbytes,
        )

        if len(bucket) >= min_chunks:
            reasm_dict.pop(key, None)
            _t0 = time.perf_counter()
            chunks_to_merge = bucket[:min_chunks]
            input_bytes_sum = sum(c.nbytes for c in chunks_to_merge)
            merged_chunk = PendingChunk.reassemble_chunks(chunks_to_merge)
            if stats is not None:
                stats.record(time.perf_counter() - _t0, merged_chunk.nbytes)
            logger.info(
                "[pd_kv_resharding] process_incoming_chunk (merged %d chunks): "
                "room=%s index_start=%d input_bytes_sum=%d merged_bytes=%d",
                min_chunks,
                info.room,
                chunk.index_start,
                input_bytes_sum,
                merged_chunk.nbytes,
            )
            pending_queue.append(merged_chunk)


# torch_tpu's batched host<->device DMA primitive. One call moves *all* layers'
# shards in a single dispatch with a single sync, instead of a per-layer
# ``index_select``/``.cpu()`` loop that pays a full device-queue sync per layer
# (48 syncs for a 48-layer model). Loaded lazily + cached; absent on plain CPU
# (the unit tests), where we fall back to the per-layer path.
_BATCH_TRANSFER: tuple | None = None  # (d2h_sync, h2d_sync, d2h_async) or False


def _batch_transfer():
    """Return ``(batch_transfer_d2h_sync, batch_transfer_h2d_sync,
    batch_transfer_d2h)`` (last is the async variant returning a future) or
    ``None`` if torch_tpu's batch_transfer is unavailable (CPU-only test env)."""
    global _BATCH_TRANSFER
    if _BATCH_TRANSFER is None:
        try:
            from torch_tpu._internal.batch_transfer import (
                batch_transfer_d2h,
                batch_transfer_d2h_sync,
                batch_transfer_h2d_sync,
            )

            _BATCH_TRANSFER = (
                batch_transfer_d2h_sync,
                batch_transfer_h2d_sync,
                batch_transfer_d2h,
            )
        except Exception as e:  # pragma: no cover - depends on torch_tpu build
            logger.debug("batch_transfer unavailable, using per-layer fallback: %s", e)
            _BATCH_TRANSFER = False
    return _BATCH_TRANSFER or None


def get_layer_page_buffers(kv_pool) -> list[torch.Tensor]:
    """Return the per-layer KV buffers whose dim 0 is block-indexed.

    Supports the fused RPA v3 pool (``kv_pool.kv_buffer`` — a list of 5D
    tensors shaped ``[num_blocks, page_size, ...]``). The token-indexed
    ``k_buffer``/``v_buffer`` layout used by the ``pallas_tpu_flash`` backend
    is intentionally rejected: its dim 0 is per-token, not per-page, so the
    block-id gather/scatter below would be wrong.
    """
    bufs = getattr(kv_pool, "kv_buffer", None)
    if isinstance(bufs, (list, tuple)) and len(bufs) > 0:
        return list(bufs)
    raise NotImplementedError(
        "TPU P/D transfer requires the fused RPA v3 KV pool "
        "(TpuFusedMHATokenToKVPool, attention backend 'pallas_tpu_rpa_v3'); "
        f"got pool {type(kv_pool).__name__} without a `kv_buffer` list. "
        "The token-indexed flash KV layout is not yet supported."
    )


def _narrow_lanes(buf: torch.Tensor, lane_slice: LaneSlice) -> torch.Tensor:
    """Sub selects lanes from KV tensor. Lanes are at dim 2."""
    if lane_slice is None:
        return buf
    start, count = lane_slice
    return buf.narrow(2, start, count)


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a (CPU) tensor to raw little-endian bytes, dtype-agnostic.

    ``numpy()`` cannot represent bfloat16 / float8, so reinterpret the
    contiguous storage as ``uint8`` first. The decode side reconstructs the
    dtype from ``layer_page_specs`` — the bytes carry no dtype tag.
    """
    flat = t.detach().contiguous().reshape(-1)
    return bytes(flat.view(torch.uint8).numpy())


def bytes_to_tensor(blob, dtype: torch.dtype, shape) -> torch.Tensor:
    """Inverse of :func:`tensor_to_bytes`. ``blob`` is wrapped in a
    ``bytearray`` so ``torch.frombuffer`` gets a writable buffer (avoids a
    UserWarning and a defensive copy inside torch)."""
    buf = blob if isinstance(blob, (bytearray, memoryview)) else bytearray(blob)
    flat = torch.frombuffer(buf, dtype=torch.uint8)
    return flat.view(dtype).reshape(shape)


def gather_pages_to_host(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    lane_slice: LaneSlice = None,
) -> list[bytes]:
    """D2H: gather ``page_indices`` (block ids) from every layer buffer and
    return one raw-byte blob per layer. Runs on the prefill side.

    On TPU the D2H of all layers is issued as a single batched
    ``batch_transfer_d2h_sync`` (one dispatch, one device sync) instead of a
    per-layer ``.cpu()`` loop (which syncs the device queue once per layer).
    On CPU (unit tests) it falls back to the simple per-layer path.

    ``lane_slice`` = ``(start, count)`` restricts the gather to a KV-head
    sub-range on the packed-lane axis (dim 2) for MHA resharding; None gathers
    the whole per-rank page (the equal-attn_tp path)."""
    if len(layer_buffers) == 0:
        return []
    device = layer_buffers[0].device
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)

    # Device-side gather is the same either way (index_select per layer); the
    # optional dim-2 narrow selects the KV-head slice before the gather.
    src_shards = [
        _narrow_lanes(buf, lane_slice).index_select(0, idx) for buf in layer_buffers
    ]

    bt = _batch_transfer() if device.type != "cpu" else None
    if bt is not None:
        d2h_sync, _, _ = bt
        host_dsts = [
            torch.empty(s.shape, dtype=s.dtype, device="cpu") for s in src_shards
        ]
        d2h_sync(src_shards, host_dsts)  # single batched D2H, one sync
        return [tensor_to_bytes(h) for h in host_dsts]

    # Per-layer fallback (CPU / batch_transfer unavailable).
    return [tensor_to_bytes(s.to("cpu")) for s in src_shards]


def gather_pages_to_stage(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    pool: PinnedHostPool,
    lane_slice: LaneSlice = None,
) -> list[StageBuf]:
    """TPU zero-copy D2H (#1/#4): gather each layer's pages directly into pooled
    pinned+aligned host buffers via one batched ``batch_transfer_d2h``. Returns
    the list of ``StageBuf``s; the caller sends their ``.mv()`` memoryviews with
    ``copy=False`` and ``.release()``s them once the send completes.

    ``lane_slice`` = ``(start, count)`` restricts the gather to a KV-head
    sub-range (dim 2) for MHA resharding; None gathers the whole page."""
    if len(layer_buffers) == 0:
        return []
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("gather_pages_to_stage requires torch_tpu batch_transfer")
    d2h_sync, _, _ = bt
    device = layer_buffers[0].device
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    src_shards = [
        _narrow_lanes(buf, lane_slice).index_select(0, idx) for buf in layer_buffers
    ]
    stages = [pool.acquire(s.numel() * s.element_size()) for s in src_shards]
    dsts = [st.view(s.dtype, s.shape) for st, s in zip(stages, src_shards)]
    d2h_sync(src_shards, dsts)  # single batched D2H into the pinned buffers
    return stages


def gather_pages_to_stage_async(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    pool: PinnedHostPool,
    lane_slice: LaneSlice = None,
) -> tuple[Any | None, list[StageBuf]]:
    """Async variant of :func:`gather_pages_to_stage` (#7). Issues the batched
    D2H without blocking and returns ``(future, stages)``; the caller waits the
    future (off the scheduler thread) before sending.

    Donation safety: ``index_select`` (the *read* of the donated ``kv_buffer``)
    is issued on the calling (main) thread and is ordered before any subsequent
    forward on torch_tpu's single execution stream, so the gathered ``src_shards``
    are independent tensors by the time the async DMA copies them — the next
    forward's donation cannot race the copy. Still opt-in (default off) until
    validated under real multi-request concurrency on hardware.

    ``lane_slice`` = ``(start, count)`` restricts the gather to a KV-head
    sub-range (dim 2) for MHA resharding; None gathers the whole page."""
    if len(layer_buffers) == 0:
        return None, []
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("gather_pages_to_stage_async requires batch_transfer")
    _, _, d2h_async = bt
    device = layer_buffers[0].device
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    src_shards = [
        _narrow_lanes(buf, lane_slice).index_select(0, idx) for buf in layer_buffers
    ]
    stages = [pool.acquire(s.numel() * s.element_size()) for s in src_shards]
    dsts = [st.view(s.dtype, s.shape) for st, s in zip(stages, src_shards)]
    future = d2h_async(src_shards, dsts)  # non-blocking; .wait() later
    return future, stages


def scatter_stage_from_host(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    layer_blobs: Sequence[StageBuf | bytes],
    pool: PinnedHostPool,
    dst_lane_slice: LaneSlice = None,
) -> None:
    """H2D scatter via pooled pinned staging: batched H2D into device shards,
    then ``index_copy_`` into the KV pool block positions. Decode side.

    Args:
        layer_buffers: Per-layer block-indexed KV tensors to scatter into.
        page_indices: Destination block ids, one per page.
        layer_blobs: Per-layer payload, either pre-staged ``StageBuf``s (recv
            thread already copied the wire — no copy here) or raw bytes /
            memoryviews (staged and copied here). Released to ``pool`` when done.
        pool: Pinned host staging pool to acquire/release staging buffers.
        dst_lane_slice: (lane_start, lane_count) writing the KV-head slice into that
            dim-2 sub-range of each buffer (MHA resharding)
    """
    if len(layer_buffers) == 0:
        return
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("scatter_stage_from_host requires torch_tpu batch_transfer")
    _, h2d_sync, _ = bt
    dsts = [_narrow_lanes(buf, dst_lane_slice) for buf in layer_buffers]
    device = layer_buffers[0].device
    n_pages = len(np.asarray(page_indices))
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    pre_staged = bool(layer_blobs) and isinstance(layer_blobs[0], StageBuf)
    if pre_staged:
        stages = layer_blobs  # recv thread already copied wire -> staging
    else:
        stages = [pool.acquire(len(blob)) for blob in layer_blobs]
        for st, blob in zip(stages, layer_blobs):
            st.copy_bytes(blob)
    try:
        host_srcs = [
            st.view(dst.dtype, (n_pages, *dst.shape[1:]))
            for st, dst in zip(stages, dsts)
        ]
        dev_shards = [
            torch.empty((n_pages, *dst.shape[1:]), dtype=dst.dtype, device=device)
            for dst in dsts
        ]
        h2d_sync(host_srcs, dev_shards)
        for dst, shard in zip(dsts, dev_shards):
            _index_copy_pages(dst, idx, shard)
    finally:
        for st in stages:
            st.release()


def scatter_pages_from_host(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    layer_blobs: Sequence[bytes],
    dst_lane_slice: LaneSlice = None,
) -> None:
    """H2D scatter of per-layer byte blobs into ``page_indices`` (block ids) of
    every layer buffer, onto whatever device the buffers live on. Decode side.

    Args:
        layer_buffers: Per-layer block-indexed KV tensors to scatter into.
        page_indices: Destination block ids, one per page.
        layer_blobs: Per-layer raw byte payloads, one per layer buffer.
        dst_lane_slice: ``(start, count)`` writing the KV-head slice into that
            dim-2 sub-range of each buffer (MHA resharding); None writes the
            whole page. dtype/shape come from the lane-narrowed targets.
    """
    if len(layer_buffers) == 0:
        return
    assert len(layer_blobs) == len(layer_buffers), (
        f"layer count mismatch: {len(layer_blobs)} blobs vs "
        f"{len(layer_buffers)} layer buffers"
    )
    dsts = [_narrow_lanes(buf, dst_lane_slice) for buf in layer_buffers]
    device = layer_buffers[0].device
    n_pages = len(np.asarray(page_indices))
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    host_srcs = [
        bytes_to_tensor(blob, dst.dtype, (n_pages, *dst.shape[1:]))
        for blob, dst in zip(layer_blobs, dsts)
    ]
    bt = _batch_transfer() if device.type != "cpu" else None
    if bt is not None:
        _, h2d_sync, _ = bt
        # Batched H2D into contiguous device shards (one dispatch, one sync),
        # then scatter each shard into its block positions on the main thread
        # (index_copy_ touches the donated kv_buffer — must stay on this thread).
        # batch_transfer requires >=32B-aligned host buffers; ``frombuffer`` over
        # a bytearray is unaligned, so clone into the (64B-aligned) torch CPU
        # allocator first. Phase 2's pinned aligned staging pool removes this copy.
        aligned_srcs = [h.clone() for h in host_srcs]
        dev_shards = [
            torch.empty((n_pages, *dst.shape[1:]), dtype=dst.dtype, device=device)
            for dst in dsts
        ]
        h2d_sync(aligned_srcs, dev_shards)
        for dst, shard in zip(dsts, dev_shards):
            _index_copy_pages(dst, idx, shard)
        return

    # Per-layer fallback (CPU / batch_transfer unavailable).
    for dst, host in zip(dsts, host_srcs):
        _index_copy_pages(dst, idx, host.to(device))


def _index_copy_pages(buf: torch.Tensor, idx: torch.Tensor, src: torch.Tensor) -> None:
    """``buf.index_copy_(0, idx, src)`` with a uint8-view fallback.

    ``index_copy_`` has no kernel for float8 dtypes on CPU (and historically
    on some accelerators). Reinterpreting both tensors as uint8 keeps dim 0
    (the page axis we index) intact and shares ``buf``'s storage, so the
    scatter still lands in the original fp8 buffer.
    """
    try:
        buf.index_copy_(0, idx, src)
    except (NotImplementedError, RuntimeError):
        buf.view(torch.uint8).index_copy_(0, idx, src.contiguous().view(torch.uint8))


# --------------------------------------------------------------------------
# Auxiliary metadata (first output token, logprobs, hidden states, ...).
#
# These live in the CPU-resident ``MetadataBuffers`` (see disaggregation/
# utils.py). ``kv_args.aux_data_ptrs`` are plain host addresses, so we
# read/write them with ctypes — device-agnostic, identical to mooncake's
# AuxDataCodec. One "row" == the slice ``[aux_index]`` of each aux buffer.
# --------------------------------------------------------------------------


def read_aux_row(kv_args: KVArgs, aux_index: int) -> list[bytes]:
    """Read aux buffer row ``aux_index`` from the prefill MetadataBuffers."""
    blobs: list[bytes] = []
    for ptr, item_len in zip(kv_args.aux_data_ptrs, kv_args.aux_item_lens):
        src = ptr + item_len * aux_index
        buf = (ctypes.c_byte * item_len).from_address(src)
        blobs.append(bytes(buf))
    return blobs


def write_aux_row(kv_args: KVArgs, aux_index: int, blobs: Sequence[bytes]) -> None:
    """Write aux buffer row ``aux_index`` into the decode MetadataBuffers."""
    for ptr, item_len, blob in zip(kv_args.aux_data_ptrs, kv_args.aux_item_lens, blobs):
        # Tolerate length drift defensively; both sides build MetadataBuffers
        # identically so this should always be an exact fit.
        # FIXME: change to assert after tests are stabled,
        n = min(item_len, len(blob))
        dst = (ctypes.c_byte * n).from_address(ptr + item_len * aux_index)
        dst[:] = blob[:n]
