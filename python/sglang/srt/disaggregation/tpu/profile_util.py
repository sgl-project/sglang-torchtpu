# Emit one ``trace-event`` line to stdout per KV transfer phase (gather / push /
# receive / scatter).
"""Lightweight Perfetto trace-event instrumentation for TPU P/D KV transfer.

FIXME: Refactor this module to use OpenTelemetry (OTEL) tracing, matching the
Mooncake disaggregation tracing path.
"""

import contextlib
import logging
import sys
import threading
import time

_trace_logger = logging.getLogger("sglang.perfetto_trace")
_trace_logger.propagate = False
if not _trace_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _trace_logger.addHandler(_handler)


class _LazyPerfetto:
    """Lazy initializer/evaluator for _PERFETTO flag using sglang envs."""

    def __bool__(self) -> bool:
        try:
            from sglang.srt.environ import envs

            return bool(envs.SGLANG_TPU_ENABLE_DISAGGREGATION_PERFETTO_TRACING.get())
        except Exception:
            return False


_PERFETTO = _LazyPerfetto()


def fmt_bytes(n: float) -> str:
    """Format bytes to readable string: e.g. ``12.3MB``."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0


class _TraceSpan:
    """Trace span objects. Use slots for minimum memory consumption."""

    __slots__ = ("args", "name")

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.args = kwargs


class _DummyDict(dict):
    """No-op dictionary for dummy trace span."""

    def __setitem__(self, key, value):
        pass

    def update(self, *args, **kwargs):
        pass


class _DummyTraceSpan:
    """Singleton dummy trace span for zero-allocation disabled tracing."""

    __slots__ = ()
    name = ""
    args = _DummyDict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


_DUMMY_TRACE_SPAN = _DummyTraceSpan()


def _eval_arg(v):
    return v() if callable(v) else v


@contextlib.contextmanager
def _perfetto_trace(pid: str, name: str, **kwargs):
    span = _TraceSpan(name, **kwargs)
    t_wall = time.time()
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        ts_us = int(t_wall * 1e6)
        dur_us = int((time.perf_counter() - t0) * 1e6)
        tid = threading.current_thread().name
        if span.args:
            args_str = " ".join(f"{k}={_eval_arg(v)}" for k, v in span.args.items())
            _trace_logger.info(
                f"trace-event: name={span.name} ts={ts_us} dur={dur_us} "
                f"pid={pid} tid={tid} {args_str}"
            )
        else:
            _trace_logger.info(
                f"trace-event: name={span.name} ts={ts_us} dur={dur_us} "
                f"pid={pid} tid={tid}"
            )


def perfetto_trace(pid: str, name: str, **kwargs):
    """Perfetto trace context manager using process/track ID string."""
    if not _PERFETTO:
        return _DUMMY_TRACE_SPAN
    return _perfetto_trace(pid, name, **kwargs)
