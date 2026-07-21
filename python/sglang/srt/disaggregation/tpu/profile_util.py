# Emit one ``trace-event`` line to stdout per KV transfer phase (gather / push /
# receive / scatter).
"""Lightweight Perfetto trace-event instrumentation for TPU P/D KV transfer."""

import contextlib
import threading
import time
from typing import Any

class _LazyPerfetto:
    """Lazy initializer/evaluator for _PERFETTO flag using sglang envs."""

    def __bool__(self) -> bool:
        try:
            from sglang.srt.environ import envs
            return bool(envs.SGLANG_ENABLE_PERFETTO_TRACING.get())
        except Exception:
            return False


_PERFETTO = _LazyPerfetto()


def _fmt_bytes(n: float) -> str:
    """Format bytes to readable string: e.g. ``12.3MB``."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0


class _TraceSpan:
    """Trace span objects. Use slots for minimum memory consumption."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


def perfetto_pid(manager: Any) -> str:
    """Perfetto process-track id based on attention dp rank, tp rank and mode."""
    rank = getattr(manager.kv_args, "engine_rank", 0)
    dp_size = getattr(manager, "attn_dp_size", 1)
    if dp_size > 1:
        dp_rank = getattr(manager, "attn_dp_rank", 0)
        return f"tpu-kv-{manager.disaggregation_mode.value}-dp{dp_rank}-tp{rank}"
    return f"tpu-kv-{manager.disaggregation_mode.value}{rank}"


def perfetto_token_shape_str(manager: Any, n_pages: int, num_lanes: int) -> str:
    """Perfetto token shape string."""
    bufs = manager.layer_buffers
    if not bufs or bufs[0].dim() < 5:
        return f"{n_pages}x{num_lanes}"
    s = bufs[0].shape
    return f"{n_pages}x{int(s[1])}x{num_lanes}x{int(s[3])}x{int(s[4])}"


@contextlib.contextmanager
def trace(manager: Any, name: str):
    """Perfetto trace context manager."""
    if not _PERFETTO:
        yield _TraceSpan(name)
        return
    span = _TraceSpan(name)
    t_wall = time.time()
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        ts_us = int(t_wall * 1e6)
        dur_us = int((time.perf_counter() - t0) * 1e6)
        tid = threading.current_thread().name
        pid = getattr(manager, "_trace_pid", None) or perfetto_pid(manager)
        print(
            f"trace-event: name={span.name} ts={ts_us} dur={dur_us} "
            f"pid={pid} tid={tid}",
            flush=True,
        )
