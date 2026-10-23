# SPDX-License-Identifier: Apache-2.0
"""Return freed host heap pages to the OS after scheduler warmup.

Startup (weight loading, compilation, warmup) allocates large temporary
host buffers. They are freed afterwards, but the allocator keeps the pages,
so each rank stays near its startup peak.

``install`` releases startup file-cache pages before the scheduler-ready
handshake, including for offline Engine and Rust launches. It also wraps
``Scheduler.handle_freeze_gc`` to release pages created by subsequent warmup,
clear export records and JAX caches, collect garbage before freezing GC,
and return freed heap pages.

A release goes through every allocator loaded in the process (tcmalloc,
glibc) because the one serving ``malloc`` cannot be detected reliably, and
an idle allocator ignores the call. ``SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC``
repeats the release on a timer. It is off by default since a release
briefly locks the allocator.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import msgspec
from torch.compiler._cache import CacheArtifactManager

from sglang.srt.runtime_context import get_parallel
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.page_cache import (
    release_startup_page_cache,
    start_checkpoint_page_cache_release,
)

logger = get_logger(__name__)

_GIB = float(1 << 30)
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")

_periodic_thread: Optional[threading.Thread] = None
_periodic_stop = threading.Event()
_jax_get_backend_uncached: Optional[Callable] = None

Release = Tuple[str, Callable[[], object]]


class HeapTrimResult(msgspec.Struct, frozen=True):
    releases: Tuple[str, ...]
    rss_before: int
    rss_after: int
    seconds: float


def _symbol(handle: ctypes.CDLL, name: str):
    try:
        return getattr(handle, name)
    except AttributeError:
        return None


def _find_releases() -> Tuple[Release, ...]:
    """Release calls of every allocator loaded in the process."""
    releases = []
    try:
        process = ctypes.CDLL(None)
    except OSError:
        process = None
    if process is not None:
        release = _symbol(process, "MallocExtension_ReleaseFreeMemory")
        if release is not None:
            release.argtypes = []
            release.restype = None
            releases.append(("tcmalloc", release))
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        trim = _symbol(libc, "malloc_trim")
    except OSError:
        trim = None
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        releases.append(("glibc", lambda: trim(0)))
    return tuple(releases)


_RELEASES = _find_releases()


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


def log_startup_memory() -> None:
    # Missing cgroup diagnostics must not turn this log into a gate.
    try:
        usage = int(_MEMORY_CURRENT.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("TPU startup memory: memory.current unavailable: %s", exc)
        return
    logger.info("TPU startup memory: memory.current=%d bytes", usage)


def trim_host_heap(reason: str) -> Optional[HeapTrimResult]:
    """Release free heap pages. Returns None when no release call exists."""
    if not _RELEASES:
        logger.info("host heap trim skipped, no release call found (%s)", reason)
        return None
    names = tuple(name for name, _ in _RELEASES)
    rss_before = _rss_bytes()
    start = time.perf_counter()
    for _, release in _RELEASES:
        release()
    seconds = time.perf_counter() - start
    rss_after = _rss_bytes()
    logger.info(
        "host heap trim via %s (%s): rss %.2f -> %.2f GiB in %.2fs",
        "+".join(names),
        reason,
        rss_before / _GIB,
        rss_after / _GIB,
        seconds,
    )
    return HeapTrimResult(
        releases=names,
        rss_before=rss_before,
        rss_after=rss_after,
        seconds=seconds,
    )


def _start_periodic(interval_s: float) -> Optional[threading.Thread]:
    global _periodic_thread
    if interval_s <= 0 or _periodic_thread is not None:
        return _periodic_thread
    _periodic_stop.clear()

    def loop() -> None:
        while not _periodic_stop.wait(interval_s):
            trim_host_heap("periodic")

    _periodic_thread = threading.Thread(
        target=loop, name="tpu-host-heap-trim", daemon=True
    )
    _periodic_thread.start()
    logger.info("periodic host heap trim every %.0fs", interval_s)
    return _periodic_thread


def _stop_periodic() -> None:
    global _periodic_thread
    _periodic_stop.set()
    if _periodic_thread is not None:
        _periodic_thread.join(timeout=5)
    _periodic_thread = None


def _get_backend_with_cpu_reinit(platform=None):
    from jax._src import xla_bridge

    # JAX skips backend initialization while any backend remains registered.
    if platform is None or isinstance(platform, str):
        requested = (
            platform or xla_bridge._XLA_BACKEND.value or xla_bridge._PLATFORM_NAME.value
        )
        if requested == "cpu":
            with xla_bridge._backend_lock:
                if xla_bridge._backends and "cpu" not in xla_bridge._backends:
                    xla_bridge._backends["cpu"] = xla_bridge._init_backend("cpu")
    assert _jax_get_backend_uncached is not None
    return _jax_get_backend_uncached(platform)


def _release_cpu_backend() -> bool:
    """Drop registry references at the idle, post-warmup cleanup boundary.

    Call after clearing JAX caches and before collecting/freezing GC. Existing
    arrays or executables keep their clients alive. Preserve CPU when it is the
    default backend, and never initialize a backend just to release it.
    """
    from jax._src import xla_bridge

    global _jax_get_backend_uncached
    with xla_bridge._backend_lock:
        backend = xla_bridge._backends.get("cpu")
        if backend is None or backend is xla_bridge._default_backend:
            return False
        if _jax_get_backend_uncached is None:
            _jax_get_backend_uncached = xla_bridge._get_backend_uncached
        xla_bridge._get_backend_uncached = _get_backend_with_cpu_reinit
        del xla_bridge._backends["cpu"]
        xla_bridge.get_backend.cache_clear()
    return True


def _before_freeze(*, synchronize: bool) -> None:
    release_startup_page_cache(include_checkpoints=False, synchronize=synchronize)
    if tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.get():
        import jax

        CacheArtifactManager.clear()
        # Serving executables are owned by TorchTPU; discard JAX tracing caches.
        jax.clear_caches()
        _release_cpu_backend()
        # gc.freeze() keeps any cycle it sees forever, so collect first.
        gc.collect()


def _after_freeze() -> None:
    if tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.get():
        trim_host_heap("after warmup")
    _start_periodic(tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.get())


def install(scheduler_cls: Optional[type] = None) -> None:
    """Release file caches at scheduler startup and trim after GC freeze."""
    if scheduler_cls is None:
        # Break the scheduler -> model runner -> TPU backend import cycle.
        from sglang.srt.managers.scheduler import Scheduler as scheduler_cls

    orig = scheduler_cls.handle_freeze_gc
    if getattr(orig, "_tpu_host_heap_trim", False):
        return

    init_graphs = scheduler_cls.init_all_cuda_graphs

    def init_all_cuda_graphs(self, *args, **kwargs):
        start_checkpoint_page_cache_release()
        return init_graphs(self, *args, **kwargs)

    orig_get_init_info = scheduler_cls.get_init_info

    def get_init_info(self, *args, **kwargs):
        # Engine and Rust callers need no explicit freeze request for startup
        # file cleanup. All ranks reach this handshake before serving requests.
        release_startup_page_cache()
        get_parallel().world_group.barrier()
        return orig_get_init_info(self, *args, **kwargs)

    def handle_freeze_gc(self, *args, **kwargs):
        # PP stages dispatch the freeze in different loop iterations (the wave
        # sends PD consensus a round ahead), so a world barrier here deadlocks.
        synchronize = get_parallel().pp_size == 1
        _before_freeze(synchronize=synchronize)
        result = orig(self, *args, **kwargs)
        _after_freeze()
        if synchronize:
            # The scheduler reply covers cleanup on every rank.
            get_parallel().world_group.barrier()
        if int(os.environ["LOCAL_RANK"]) == 0:
            log_startup_memory()
        return result

    handle_freeze_gc._tpu_host_heap_trim = True
    scheduler_cls.init_all_cuda_graphs = init_all_cuda_graphs
    scheduler_cls.get_init_info = get_init_info
    scheduler_cls.handle_freeze_gc = handle_freeze_gc
