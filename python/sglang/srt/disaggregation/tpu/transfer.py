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
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs
from sglang.srt.disaggregation.tpu.host_staging import PinnedHostPool, StageBuf

logger = logging.getLogger(__name__)

# ``(start, count)`` on the fused buffer's packed-lane axis (dim 2), or None for
# a whole-page transfer (the equal-attn_tp path). Used by the resharding gather
# /scatter helpers.
LaneSlice = Optional[Tuple[int, int]]


class ReshardUnsupportedError(ValueError):
    """Unsuppprted KV-head resharding the TPU packed-lane layout."""


class ReshardDirection(enum.Enum):
    """Which way KV heads are redistributed between prefill and decode attn_tp.

    * ``EQUAL``   — same attn_tp: whole per-rank head set, 1:1 (no resharding).
    * ``FAN_IN``  — prefill_tp > decode_tp: many prefill ranks -> one decode rank.
    * ``FAN_OUT`` — prefill_tp < decode_tp: one prefill rank -> many decode ranks.
    """

    EQUAL = "equal"
    FAN_IN = "fan-in"
    FAN_OUT = "fan-out"


# torch_tpu's batched host<->device DMA primitive. One call moves *all* layers'
# shards in a single dispatch with a single sync, instead of a per-layer
# ``index_select``/``.cpu()`` loop that pays a full device-queue sync per layer
# (48 syncs for a 48-layer model). Loaded lazily + cached; absent on plain CPU
# (the unit tests), where we fall back to the per-layer path.
_BATCH_TRANSFER: Optional[Tuple] = None  # (d2h_sync, h2d_sync, d2h_async) or False


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


def get_layer_page_buffers(kv_pool) -> List[torch.Tensor]:
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


@dataclass
class LayerPageSpec:
    """Specification of a single KV page's layout for a specific layer.

    Attributes:
        dtype: The PyTorch data type of the KV tensor for this layer.
        page_block_shape: The shape of a single page tensor. Match
            ``[page_size, lane_count, kv_packing, head_dim]``.
    """

    dtype: torch.dtype
    page_block_shape: Tuple[int, ...]


def extract_layer_page_specs(
    layer_buffers: Sequence[torch.Tensor],
) -> List[LayerPageSpec]:
    """Extracts layer page spec."""
    return [
        LayerPageSpec(dtype=buf.dtype, page_block_shape=tuple(buf.shape[1:]))
        for buf in layer_buffers
    ]


def _adjust_layer_page_specs_lane_count(
    specs: Sequence[LayerPageSpec], dst_lane_slice: LaneSlice
) -> List[LayerPageSpec]:
    """Rebuild each spec so the lane axis reflects the sliced ``count``.
    ``page_block_shape`` == ``(page_size, lanes, kv_packing, head_dim)``, so the
    lane axis is index 1. No-op when ``dst_lane_slice`` is None (whole page)."""
    if dst_lane_slice is None:
        return list(specs)
    _, count = dst_lane_slice
    return [
        LayerPageSpec(
            dtype=s.dtype,
            page_block_shape=(s.page_block_shape[0], count, *s.page_block_shape[2:]),
        )
        for s in specs
    ]


def _head_to_lane(head: int, kv_packing: int) -> int:
    """Converts a KV-head boundary to a packed-lane index on the interleaved
    (x2) head axis. Raises when the boundary is not packing-aligned (only
    possible for fp8, ``kv_packing==4``, with an odd head count)."""
    elem = head * 2  # interleaved K,V per head
    if elem % kv_packing != 0:
        raise ReshardUnsupportedError(
            f"[pd_kv_resharding] head boundary {head} is not packing-aligned "
            f"(kv_packing={kv_packing}); fp8 resharding needs even head counts"
        )
    return elem // kv_packing


@dataclass
class KvTransferLaneSpec:
    """Lane slice specification for a (prefill rank -> decode rank) KV transfer.

    Attributes:
        src_lane_start: Start lane index on the source prefill rank.
        lane_count: Number of packed lanes to transfer.
        dst_lane_start: Start lane index on the destination decode rank.
        num_heads: Number of KV heads being transferred.
        sends_aux: Whether this transfer includes the auxiliary metadata row.
    """

    src_lane_start: int
    lane_count: int
    dst_lane_start: int
    num_heads: int
    sends_aux: bool


def compute_kv_transfer_lane_spec(
    prefill_tp: int,
    decode_tp: int,
    engine_rank: int,
    dst_tp_rank: int,
    total_kv_heads: int,
    kv_packing: int,
) -> KvTransferLaneSpec:
    """Computes kv transfer lane spec."""
    if total_kv_heads <= 0:
        # A 0 here means the decode side never advertised its KV-head count
        # (see DecodePreallocQueue._init_kv_manager). Fail loud instead of
        # guessing a fallback: a wrong total would silently transfer the wrong
        # KV-head slice, which is far worse than a crash.
        raise ReshardUnsupportedError(
            f"[pd_kv_resharding] total_kv_heads={total_kv_heads} is invalid "
            f"(prefill_tp={prefill_tp}, decode_tp={decode_tp}); the decode side "
            "must set kv_args.total_kv_head_num / kv_head_num before resharding."
        )
    src_heads = max(1, total_kv_heads // prefill_tp)
    dst_heads = max(1, total_kv_heads // decode_tp)
    src_replication = max(1, prefill_tp // total_kv_heads)
    dst_replication = max(1, decode_tp // total_kv_heads)
    local_tp_rank = engine_rank % prefill_tp
    dst_in = dst_tp_rank % decode_tp

    if prefill_tp > decode_tp:
        direction = ReshardDirection.FAN_IN
        src_head_start = 0
        is_replica = (local_tp_rank % src_replication) != 0
        num_heads = 0 if is_replica else src_heads
        unique_head_idx = local_tp_rank // src_replication
        dst_head_start = (unique_head_idx * src_heads) % dst_heads
    elif prefill_tp < decode_tp:
        direction = ReshardDirection.FAN_OUT
        unique_dst_head_idx = dst_in // dst_replication
        src_head_start = (unique_dst_head_idx * dst_heads) % src_heads
        num_heads = dst_heads
        dst_head_start = 0
    else:
        direction = ReshardDirection.EQUAL
        src_head_start = 0
        num_heads = src_heads
        dst_head_start = 0

    src_lane_start = _head_to_lane(src_head_start, kv_packing)
    lane_count = _head_to_lane(src_head_start + num_heads, kv_packing) - src_lane_start
    dst_lane_start = _head_to_lane(dst_head_start, kv_packing)
    sends_aux = (dst_head_start == 0) and (num_heads > 0)

    logger.debug(
        "[pd_kv_resharding] compute_lane_slice %s prefill_tp=%d decode_tp=%d "
        "local=%d dst=%d total_heads=%d src_heads=%d dst_heads=%d repl=%d -> "
        "src_lane=%d count=%d dst_lane=%d num_heads=%d aux=%s",
        direction.value,
        prefill_tp,
        decode_tp,
        local_tp_rank,
        dst_in,
        total_kv_heads,
        src_heads,
        dst_heads,
        src_replication,
        src_lane_start,
        lane_count,
        dst_lane_start,
        num_heads,
        sends_aux,
    )
    return KvTransferLaneSpec(
        src_lane_start=src_lane_start,
        lane_count=lane_count,
        dst_lane_start=dst_lane_start,
        num_heads=num_heads,
        sends_aux=sends_aux,
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


def gather_pages_to_stage(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    pool: PinnedHostPool,
    lane_slice: LaneSlice = None,
) -> List[StageBuf]:
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
) -> Tuple[Optional[Any], List[StageBuf]]:
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
    stages: Sequence[StageBuf],
    specs: Sequence[LayerPageSpec],
    dst_lane_slice: LaneSlice = None,
) -> None:
    """H2D scatter via pooled pinned staging: batch_transfer_h2d into device
    shards, then index_copy_ into the KV pool block positions.

    ``stages`` is a list of :class:`StageBuf`s pre-staged by the recv thread.
    Staging buffers are released to the pool when done.

    ``dst_lane_slice`` = ``(start, count)`` writes the received KV-head slice
    into that dim-2 sub-range of each layer buffer (MHA resharding); None writes
    the whole page. The blob shapes are rebuilt from ``count`` so an fp8/bf16
    slice reshapes correctly."""
    if len(layer_buffers) == 0:
        return
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("scatter_stage_from_host requires torch_tpu batch_transfer")
    _, h2d_sync, _ = bt
    specs = _adjust_layer_page_specs_lane_count(specs, dst_lane_slice)
    device = layer_buffers[0].device
    n_pages = len(np.asarray(page_indices))
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    try:
        host_srcs = [
            st.view(spec.dtype, (n_pages, *spec.page_block_shape))
            for st, spec in zip(stages, specs)
        ]
        dev_shards = [
            torch.empty(
                (n_pages, *spec.page_block_shape), dtype=spec.dtype, device=device
            )
            for spec in specs
        ]
        h2d_sync(host_srcs, dev_shards)
        for buf, shard in zip(layer_buffers, dev_shards):
            _index_copy_pages(_narrow_lanes(buf, dst_lane_slice), idx, shard)
    finally:
        for st in stages:
            st.release()


scatter_to_device_via_pinned_staging_pool = scatter_stage_from_host


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


def read_aux_row(kv_args: KVArgs, aux_index: int) -> List[bytes]:
    """Read aux buffer row ``aux_index`` from the prefill MetadataBuffers."""
    blobs: List[bytes] = []
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
        n = min(item_len, len(blob))
        dst = (ctypes.c_byte * n).from_address(ptr + item_len * aux_index)
        dst[:] = blob[:n]
