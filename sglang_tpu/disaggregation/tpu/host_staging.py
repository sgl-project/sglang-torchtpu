"""Pinned, aligned, reusable host staging buffers for TPU KV transfer (#1/#4).

Why this exists
---------------
The KV bytes must pass through host RAM (the NIC can't DMA from HBM), so a
host staging buffer is mandatory on both the prefill (D2H -> send) and decode
(recv -> H2D) sides. Two properties make that buffer fast:

* **Aligned** — torch_tpu's ``batch_transfer`` requires >=32B-aligned host
  buffers; torch's CPU allocator gives 64B, so pooled torch tensors qualify
  (a ``frombuffer``'d ``bytearray`` does not, which is why the naive path had
  to ``clone()``).
* **Pinned** — ``mlock(2)`` keeps the pages resident so PJRT's DMA doesn't
  re-fault them mid-copy. mlock only pays off if the buffer is *reused*, so it
  must come from a pool, not a fresh per-transfer allocation.

This pool vends reusable aligned uint8 torch CPU tensors bucketed by exact byte
size (one buffer per layer; KV layer sizes are regular, so few buckets), and
``mlock``s each on allocation. Retention is bounded by a **total-byte budget**
(default a fraction of host RAM, split across the TP pools co-located on a
host) — not a per-bucket count, so a full multi-layer transfer's worth of
buffers can be retained, and memory can never grow without bound. When
retaining a released buffer would exceed the budget it is **``munlock``'d and
dropped** (GC reclaims it), so pinned RAM is released, not leaked. Pinning is
best-effort: if ``mlock`` is unavailable (no ``CAP_IPC_LOCK`` / no libc),
buffers are still aligned and reused, just not locked.

(Contrast: torchtpu-vllm's ``HostKVShmPool`` instead pre-allocates a fixed set
of *max-size* slots and uses a prefix per transfer — no per-size buckets. We
keep per-layer bucketed buffers here and bound them by the byte budget.)
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)

# Aggregate pinned-staging budget as a fraction of total host RAM (split across
# the co-located TP pools — see default_pool_max_bytes).
_POOL_FRACTION = 0.20
_FALLBACK_TOTAL_BYTES = 8 * 1024**3  # used if host RAM can't be probed
_MIN_POOL_BYTES = 64 * 1024**2  # never floor a per-pool budget below this

try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _LIBC.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _LIBC.mlock.restype = ctypes.c_int
    _LIBC.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _LIBC.munlock.restype = ctypes.c_int
except Exception:  # pragma: no cover - exotic libc
    _LIBC = None


def _total_host_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):  # pragma: no cover
        return 0


def default_pool_max_bytes(local_shares: int = 1) -> int:
    """Per-pool pinned budget: ``_POOL_FRACTION`` of host RAM, divided across
    the ``local_shares`` staging pools co-located on this host (one per TP
    rank), so the *aggregate* across ranks stays within the fraction rather
    than ``local_shares`` × it."""
    ram = _total_host_ram_bytes()
    total = int(_POOL_FRACTION * ram) if ram > 0 else _FALLBACK_TOTAL_BYTES
    return max(total // max(1, local_shares), _MIN_POOL_BYTES)


class StageBuf:
    """A pooled aligned (and ideally mlock'd) host uint8 buffer. ``view`` hands
    back a typed/shaped view sharing storage (no copy). ``mv`` is the
    zero-copy memoryview for ``socket.send(..., copy=False)``. ``_locked``
    tracks this buffer's own mlock state (so eviction can munlock exactly the
    pages it locked — no global pointer set that breaks under address reuse)."""

    __slots__ = ("_locked", "_pool", "_t", "nbytes")

    def __init__(self, nbytes: int, pool: PinnedHostPool | None):
        self.nbytes = nbytes
        self._t = torch.empty(nbytes, dtype=torch.uint8)  # 64B-aligned (torch CPU)
        self._pool = pool
        self._locked = False

    def __len__(self) -> int:
        return self.nbytes

    def view(self, dtype: torch.dtype, shape) -> torch.Tensor:
        return self._t.view(dtype).reshape(shape)

    def mv(self) -> memoryview:
        return memoryview(self._t.numpy())

    def copy_bytes(self, blob) -> None:
        """Copy wire bytes into this (aligned, pinned) buffer for H2D."""
        src = blob if isinstance(blob, (bytearray, memoryview)) else bytearray(blob)
        self._t.copy_(torch.frombuffer(src, dtype=torch.uint8))

    def release(self) -> None:
        if self._pool is not None:
            self._pool._release(self)


class PinnedHostPool:
    """Bucketed pool of reusable aligned+pinned host buffers, bounded by a total
    pinned-byte budget. Thread-safe.

    Retention rule: a released buffer is kept (and stays mlock'd) only while the
    total pooled bytes stay within ``max_bytes``; otherwise it is munlock'd and
    dropped. ``max_bytes`` defaults to a fraction of host RAM."""

    def __init__(self, max_bytes: int | None = None):
        self._free: dict[int, list[StageBuf]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max_bytes = (
            max_bytes if max_bytes is not None else default_pool_max_bytes()
        )
        self._pooled_bytes = 0
        logger.debug(
            "PinnedHostPool: retention budget %.1f MiB (mlock=%s)",
            self._max_bytes / 1024**2,
            _LIBC is not None,
        )

    # ---- mlock / munlock (operate on one buffer; no global pointer set) ----

    def _mlock(self, buf: StageBuf) -> None:
        if _LIBC is None or buf._locked:
            return
        rc = _LIBC.mlock(
            ctypes.c_void_p(buf._t.data_ptr()), ctypes.c_size_t(buf.nbytes)
        )
        buf._locked = rc == 0

    def _munlock(self, buf: StageBuf) -> None:
        if _LIBC is None or not buf._locked:
            return
        _LIBC.munlock(ctypes.c_void_p(buf._t.data_ptr()), ctypes.c_size_t(buf.nbytes))
        buf._locked = False

    # ---- acquire / release -------------------------------------------------

    def acquire(self, nbytes: int) -> StageBuf:
        with self._lock:
            bucket = self._free.get(nbytes)
            if bucket:
                buf = bucket.pop()
                self._pooled_bytes -= nbytes
                return buf  # already mlock'd from when it was first allocated
        # Miss: fresh allocation (mlock outside the pool lock — it's a syscall).
        buf = StageBuf(nbytes, self)
        self._mlock(buf)
        return buf

    def _release(self, buf: StageBuf) -> None:
        with self._lock:
            if self._pooled_bytes + buf.nbytes <= self._max_bytes:
                self._free[buf.nbytes].append(buf)
                self._pooled_bytes += buf.nbytes
                return
        # Over budget: don't retain. Release the lock (so pinned RAM is freed,
        # not leaked) and let GC reclaim the tensor. munlock outside the lock.
        self._munlock(buf)

    def clear(self) -> None:
        """Munlock and drop every pooled buffer (clean teardown)."""
        with self._lock:
            buffers = [b for bucket in self._free.values() for b in bucket]
            self._free.clear()
            self._pooled_bytes = 0
        for b in buffers:
            self._munlock(b)
