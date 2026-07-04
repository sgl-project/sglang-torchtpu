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
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

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


def layer_page_specs(layer_buffers: Sequence[torch.Tensor]):
    """Per-layer ``(dtype, tail_shape)`` so the decode side can reshape the
    raw byte blobs back into device tensors. ``tail_shape`` is everything
    after the block dimension (i.e. the shape of a single page)."""
    return [(buf.dtype, tuple(buf.shape[1:])) for buf in layer_buffers]


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
) -> List[bytes]:
    """D2H: gather ``page_indices`` (block ids) from every layer buffer and
    return one raw-byte blob per layer. Runs on the prefill side.

    On TPU the D2H of all layers is issued as a single batched
    ``batch_transfer_d2h_sync`` (one dispatch, one device sync) instead of a
    per-layer ``.cpu()`` loop (which syncs the device queue once per layer).
    On CPU (unit tests) it falls back to the simple per-layer path."""
    if len(layer_buffers) == 0:
        return []
    device = layer_buffers[0].device
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)

    # Device-side gather is the same either way (index_select per layer).
    src_shards = [buf.index_select(0, idx) for buf in layer_buffers]

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


def gather_pages_to_stage(layer_buffers, page_indices, pool):
    """TPU zero-copy D2H (#1/#4): gather each layer's pages directly into pooled
    pinned+aligned host buffers via one batched ``batch_transfer_d2h``. Returns
    the list of ``StageBuf``s; the caller sends their ``.mv()`` memoryviews with
    ``copy=False`` and ``.release()``s them once the send completes."""
    if len(layer_buffers) == 0:
        return []
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("gather_pages_to_stage requires torch_tpu batch_transfer")
    d2h_sync, _, _ = bt
    device = layer_buffers[0].device
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    src_shards = [buf.index_select(0, idx) for buf in layer_buffers]
    stages = [pool.acquire(s.numel() * s.element_size()) for s in src_shards]
    dsts = [st.view(s.dtype, s.shape) for st, s in zip(stages, src_shards)]
    d2h_sync(src_shards, dsts)  # single batched D2H into the pinned buffers
    return stages


def gather_pages_to_stage_async(layer_buffers, page_indices, pool):
    """Async variant of :func:`gather_pages_to_stage` (#7). Issues the batched
    D2H without blocking and returns ``(future, stages)``; the caller waits the
    future (off the scheduler thread) before sending.

    Donation safety: ``index_select`` (the *read* of the donated ``kv_buffer``)
    is issued on the calling (main) thread and is ordered before any subsequent
    forward on torch_tpu's single execution stream, so the gathered ``src_shards``
    are independent tensors by the time the async DMA copies them — the next
    forward's donation cannot race the copy. Still opt-in (default off) until
    validated under real multi-request concurrency on hardware."""
    if len(layer_buffers) == 0:
        return None, []
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("gather_pages_to_stage_async requires batch_transfer")
    _, _, d2h_async = bt
    device = layer_buffers[0].device
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)
    src_shards = [buf.index_select(0, idx) for buf in layer_buffers]
    stages = [pool.acquire(s.numel() * s.element_size()) for s in src_shards]
    dsts = [st.view(s.dtype, s.shape) for st, s in zip(stages, src_shards)]
    future = d2h_async(src_shards, dsts)  # non-blocking; .wait() later
    return future, stages


def scatter_stage_from_host(layer_buffers, page_indices, layer_blobs, specs, pool):
    """H2D scatter via pooled pinned staging: batch_transfer_h2d into device
    shards, then index_copy_ into the KV pool block positions.

    ``layer_blobs`` is either a list of :class:`StageBuf`s (pre-staged by the
    recv thread — no copy here) or raw bytes/memoryviews (acquire staging and
    copy). Staging buffers are released to the pool when done."""
    from sglang.srt.disaggregation.tpu.host_staging import StageBuf

    if len(layer_buffers) == 0:
        return
    bt = _batch_transfer()
    if bt is None:
        raise RuntimeError("scatter_stage_from_host requires torch_tpu batch_transfer")
    _, h2d_sync, _ = bt
    if specs is None:
        specs = layer_page_specs(layer_buffers)
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
            st.view(dtype, (n_pages, *tail)) for st, (dtype, tail) in zip(stages, specs)
        ]
        dev_shards = [
            torch.empty((n_pages, *tail), dtype=dtype, device=device)
            for (dtype, tail) in specs
        ]
        h2d_sync(host_srcs, dev_shards)
        for buf, shard in zip(layer_buffers, dev_shards):
            _index_copy_pages(buf, idx, shard)
    finally:
        for st in stages:
            st.release()


def scatter_pages_from_host(
    layer_buffers: Sequence[torch.Tensor],
    page_indices: np.ndarray,
    layer_blobs: Sequence[bytes],
    specs=None,
) -> None:
    """H2D: scatter per-layer byte blobs into ``page_indices`` (block ids) of
    every layer buffer, writing onto whatever device the buffers live on.
    Runs on the decode side."""
    if len(layer_buffers) == 0:
        return
    assert len(layer_blobs) == len(layer_buffers), (
        f"layer count mismatch: {len(layer_blobs)} blobs vs "
        f"{len(layer_buffers)} layer buffers"
    )
    if specs is None:
        specs = layer_page_specs(layer_buffers)
    device = layer_buffers[0].device
    n_pages = len(np.asarray(page_indices))
    idx = torch.as_tensor(np.asarray(page_indices), dtype=torch.long, device=device)

    host_srcs = [
        bytes_to_tensor(blob, dtype, (n_pages, *tail))
        for blob, (dtype, tail) in zip(layer_blobs, specs)
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
            torch.empty((n_pages, *tail), dtype=dtype, device=device)
            for (dtype, tail) in specs
        ]
        h2d_sync(aligned_srcs, dev_shards)
        for buf, shard in zip(layer_buffers, dev_shards):
            _index_copy_pages(buf, idx, shard)
        return

    # Per-layer fallback (CPU / batch_transfer unavailable).
    for buf, host in zip(layer_buffers, host_srcs):
        _index_copy_pages(buf, idx, host.to(device))


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


def read_aux_row(kv_args, aux_index: int) -> List[bytes]:
    """Read aux buffer row ``aux_index`` from the prefill MetadataBuffers."""
    blobs: List[bytes] = []
    for ptr, item_len in zip(kv_args.aux_data_ptrs, kv_args.aux_item_lens):
        src = ptr + item_len * aux_index
        buf = (ctypes.c_byte * item_len).from_address(src)
        blobs.append(bytes(buf))
    return blobs


def write_aux_row(kv_args, aux_index: int, blobs: Sequence[bytes]) -> None:
    """Write aux buffer row ``aux_index`` into the decode MetadataBuffers."""
    for ptr, item_len, blob in zip(kv_args.aux_data_ptrs, kv_args.aux_item_lens, blobs):
        # Tolerate length drift defensively; both sides build MetadataBuffers
        # identically so this should always be an exact fit.
        n = min(item_len, len(blob))
        dst = (ctypes.c_byte * n).from_address(ptr + item_len * aux_index)
        dst[:] = blob[:n]
