# SPDX-License-Identifier: Apache-2.0
"""Monitor serving-time XLA compilation on TPU.

Reports new XLA programs and their compile durations after precompilation.
Events whose total compile time reaches ``_FORWARD_COMPILE_WARN_MS`` are
logged at WARNING; smaller events are logged at INFO. Persistent-cache
loads are counted separately because their load times are unavailable.

Periodic summaries report cumulative compile durations relative to elapsed
wall time, helping identify the cost of repeated small compilations.

Every report also carries this rank's host memory: the RSS change since the
previous report and the peak host memory the new programs took to compile.
Each TP rank process keeps its own copy of every executable, so a host pays
that growth once per rank (#1429).

Enable with ``SGLANG_TPU_CHECK_RECOMPILATION=1``. Add
``SGLANG_TPU_CHECK_RECOMPILATION_SITES=1`` to also attribute eager compiles on
the scheduler thread to sglang source lines, at a per-op dispatch cost.
"""

import collections
import datetime
import functools
import time
import traceback
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import msgspec
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.environ import envs
from sglang_tpu._logging import get_logger
from sglang_tpu.host_heap_trim import rss_bytes

logger = get_logger(__name__)

_ONE_MICROSECOND = datetime.timedelta(microseconds=1)
_MIB = float(1 << 20)
_GIB = float(1 << 30)
# A forward runs in tens to hundreds of ms, so 100 ms of compile inside one is
# a visible stall.
_FORWARD_COMPILE_WARN_MS = 100
# Arbitrary, kept long so the line stays rare on a busy server.
_SUMMARY_WINDOW_SEC = 300.0
_SITE_FRAMES = 3

# (site description, compiles at that site)
_Sites = tuple[tuple[str, int], ...]


class RecompileEvent(msgspec.Struct, frozen=True):
    """Programs that finished compiling since the previous check."""

    context: str
    durations_ms: tuple[int, ...]  # descending
    total_ms: int
    # Executables loaded from the tier-2/3 persistent cache carry no compile
    # duration. The load stall is real but not measurable from Python.
    load_count: int
    heavy: bool
    # None when site tracing is off.
    sites: _Sites | None = None
    # This rank's RSS change since the previous report, and the largest peak
    # host memory one of the new programs took while compiling. 0 when the
    # reading is unavailable. Each TP rank process holds its own copy of every
    # executable, so a host pays the growth once per rank (#1429).
    host_rss_delta_bytes: int = 0
    peak_compile_host_bytes: int = 0


# (compile duration in microseconds, peak compile host memory in bytes)
_EntryKey = tuple[int, int]


def _entry_key(entry: Any) -> _EntryKey:
    # Microseconds are the pybind's native unit. Older torch_tpu wheels report
    # no peak host memory; that leaves the key duration-only.
    try:
        peak = int(getattr(entry, "peak_compilation_host_memory_bytes", 0) or 0)
    except (TypeError, ValueError):
        peak = 0
    return (entry.compilation_duration // _ONE_MICROSECOND, peak)


def _duration_multiset(stats: Any) -> collections.Counter:
    # The cache key is not bound and entry order is unstable, so the multiset
    # of (duration, peak host memory) is the identity. That makes matching
    # approximate: an evicted entry replaced by one with the same key is
    # invisible, which needs an upstream stable entry id to fix.
    counter: collections.Counter = collections.Counter()
    for entry in stats.per_entry_stats:
        counter[_entry_key(entry)] += 1
    return counter


def _program_count(multiset: collections.Counter) -> int:
    return sum(multiset.values())


def _compile_seconds(multiset: collections.Counter) -> float:
    return sum(us * n for (us, _), n in multiset.items()) / 1e6


def _build_event(
    *,
    new_us: Iterable[int],
    context: str,
    warn_ms: int,
    sites: _Sites | None = None,
    host_rss_delta_bytes: int = 0,
    peak_compile_host_bytes: int = 0,
) -> RecompileEvent:
    new_us = tuple(new_us)
    durations_ms = tuple(
        sorted((round(us / 1000) for us in new_us if us > 0), reverse=True)
    )
    total_ms = sum(durations_ms)
    return RecompileEvent(
        context=context,
        durations_ms=durations_ms,
        total_ms=total_ms,
        load_count=sum(1 for us in new_us if us == 0),
        # One heavy compile or several adding up both stall the step.
        heavy=total_ms >= warn_ms,
        sites=sites,
        host_rss_delta_bytes=host_rss_delta_bytes,
        peak_compile_host_bytes=peak_compile_host_bytes,
    )


def _format_sites(sites: _Sites | None) -> str:
    if sites is None:
        return ""
    if not sites:
        # The tracer skips compiled regions and other threads.
        return "\n  eager compile sites: none on the traced thread (compiled graph?)"
    return "\n  eager compile sites:" + "".join(
        f"\n    {n}x {site}" for site, n in sites
    )


def _format_event(event: RecompileEvent, *, warn_ms: int) -> str:
    return (
        _format_compiles(event, warn_ms=warn_ms)
        + _format_host_memory(event)
        + _format_sites(event.sites)
    )


def _format_host_memory(event: RecompileEvent) -> str:
    """Host memory the new programs cost this rank; "" when nothing was read."""
    if not event.host_rss_delta_bytes and not event.peak_compile_host_bytes:
        return ""
    programs = len(event.durations_ms) + event.load_count
    parts = []
    if event.host_rss_delta_bytes:
        per_program = (
            f", {event.host_rss_delta_bytes / programs / _MIB:.1f} MiB per program"
            if programs and event.host_rss_delta_bytes > 0
            else ""
        )
        parts.append(
            f"host RSS {event.host_rss_delta_bytes / _MIB:+.1f} MiB since the "
            f"previous report{per_program}"
        )
    if event.peak_compile_host_bytes:
        parts.append(
            f"peak compile host memory {event.peak_compile_host_bytes / _MIB:.0f} MiB"
        )
    return (
        " This rank: "
        + "; ".join(parts)
        + ". Each TP rank process holds its own copy of every executable."
    )


def _format_compiles(event: RecompileEvent, *, warn_ms: int) -> str:
    # The shape is a contract: log-based checks parse the "durations_ms=[...]"
    # list and the bracketed context, so keep both.
    load_note = (
        f", plus {event.load_count} loaded from the persistent compile cache "
        "(load time not measured)"
        if event.load_count
        else ""
    )
    if not event.durations_ms:
        return (
            f"[recompilation checker] {event.load_count} XLA program(s) loaded "
            f"from the persistent compile cache at serving time [{event.context}] "
            "(load time not measured)."
        )
    durations = "[" + ", ".join(str(ms) for ms in event.durations_ms) + "]"
    if event.heavy:
        return (
            f"[recompilation checker] {len(event.durations_ms)} new XLA "
            f"program(s) compiled at serving time [{event.context}]: "
            f"durations_ms={durations} (total {event.total_ms} ms){load_note}. "
            "Compiles this slow leave the TPU idle and should have been "
            "covered by precompile."
        )
    return (
        f"[recompilation checker] {len(event.durations_ms)} cheap XLA "
        f"compile(s) at serving time [{event.context}]: durations_ms={durations} "
        f"(total {event.total_ms} ms, below the {warn_ms} ms warning "
        f"threshold){load_note}."
    )


def _short_path(filename: str) -> str:
    # Core and the out-of-tree backend both own serving-time tensor operations.
    start = max(filename.rfind("/sglang/"), filename.rfind("/sglang_tpu/"))
    return filename[start + 1 :]


def _describe_site(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    leaves = ", ".join(
        (
            f"{tuple(leaf.shape)}:{str(leaf.dtype).removeprefix('torch.')}"
            if isinstance(leaf, torch.Tensor)
            else repr(leaf)
        )
        for leaf in pytree.tree_leaves((args, kwargs))
    )
    frames = [
        fs
        for fs in reversed(traceback.extract_stack())
        if ("/sglang/" in fs.filename or "/sglang_tpu/" in fs.filename)
        and fs.filename != __file__
    ][:_SITE_FRAMES]
    where = " <- ".join(f"{_short_path(fs.filename)}:{fs.lineno}" for fs in frames)
    return f"{func}({leaves}) @ {where or '<no sglang frame>'}"


class _CompileSiteTracer(TorchDispatchMode):
    """Records the sglang call site of each eager op that misses the TPU program cache."""

    def __init__(
        self, *, read_misses: Callable[[], int], record: Callable[[str], None]
    ) -> None:
        super().__init__()
        self._read_misses = read_misses
        self._record = record

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # Otherwise Dynamo runs compiled functions eagerly while the mode is active.
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        before = self._read_misses()
        out = func(*args, **kwargs)
        if self._read_misses() != before:
            self._record(_describe_site(func, args, kwargs))
        return out


class RecompilationChecker:
    """Diffs torch_tpu cache-stats snapshots to find serving-time compiles.

    The cache is process-wide, so one checker serves every compiled forward
    in the process (target and draft models alike). Each TPU warmup phase
    refreshes the baseline when it finishes; installed eager forwards call
    ``check()``. The full stats scan only runs when the tier-1 miss counter
    moves; cache-hit checks read that counter and check the summary clock.

    torch_tpu compiles on a background thread and fills in the duration when
    it finishes, so both snapshots first wait for the device. That charges a
    compile to the forward that triggered it, at the cost of a device sync
    on the forwards that compiled something.
    """

    def __init__(
        self,
        *,
        read_misses: Callable[[], int],
        read_stats: Callable[[], Any],
        sync_device: Callable[[], None],
        warn_ms: int,
        clock: Callable[[], float] = time.monotonic,
        read_rss: Callable[[], int] = rss_bytes,
    ) -> None:
        self._read_misses = read_misses
        self._read_stats = read_stats
        self._sync_device = sync_device
        self._warn_ms = warn_ms
        self._clock = clock
        # This rank's RSS in bytes, 0 when unreadable.
        self._read_rss = read_rss
        self._baseline_rss = 0
        self._last_rss = 0
        self._baseline: collections.Counter | None = None
        self._seen_misses: int | None = None
        self._window_start = clock()
        self._window_events = 0
        self._window_compile_ms = 0
        # Lifetime totals. A later snapshot_baseline() resets the window, not
        # these, so they span the lifetime of the process.
        self.serving_compiles: int = 0
        self.serving_compile_ms: int = 0
        self.worst_compile_ms: int = 0
        # Compiled plus loaded from the persistent cache: every program that
        # entered this rank's in-memory cache since serving started.
        self.serving_programs: int = 0
        self._site_tracer: _CompileSiteTracer | None = None
        self._pending_sites: collections.Counter[str] = collections.Counter()

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    def enable_site_tracing(self) -> None:
        """Attach eager compile sites from the calling thread to later events.

        Dispatch modes are thread-local, so ops from other threads are not traced.
        """
        if self._site_tracer is not None:
            return
        self._site_tracer = _CompileSiteTracer(
            read_misses=self._read_misses, record=self._record_site
        )
        self._site_tracer.__enter__()

    def _record_site(self, site: str) -> None:
        self._pending_sites[site] += 1

    def snapshot_baseline(self) -> None:
        """Record the cache contents, compiles before this are not reported."""
        self._sync_device()
        self._baseline = _duration_multiset(self._read_stats())
        self._seen_misses = self._read_misses()
        self._baseline_rss = self._last_rss = self._read_rss()
        self._window_start = self._clock()
        self._window_events = 0
        self._window_compile_ms = 0
        logger.info(
            "[recompilation checker] XLA graph baseline: %d cached programs, "
            "%.1fs total compile time, %d tier-1 misses, host RSS %.2f GiB "
            "(this rank; each TP rank process holds its own executables).",
            _program_count(self._baseline),
            _compile_seconds(self._baseline),
            self._seen_misses,
            self._baseline_rss / _GIB,
        )

    def check(self, *, context_fn: Callable[[], str]) -> RecompileEvent | None:
        """Report programs compiled since the last snapshot, if any."""
        if self._baseline is None:
            return None
        event = self._detect_compiles(context_fn=context_fn)
        # After recording, and on every forward: a compile that spans the
        # deadline belongs to the window it ran in, and a window with no
        # compiles at all still has to close.
        self._maybe_report_window()
        return event

    def _detect_compiles(
        self, *, context_fn: Callable[[], str]
    ) -> RecompileEvent | None:
        misses = self._read_misses()
        if misses == self._seen_misses:
            return None
        self._seen_misses = misses
        sites = self._take_sites()
        self._sync_device()
        current = _duration_multiset(self._read_stats())
        new = current - self._baseline
        # Replace rather than merge so evicted entries do not linger.
        self._baseline = current
        if not new:
            return None
        # Read after the sync, so the new executables are resident. The delta
        # spans every forward since the previous report, not just this one.
        rss = self._read_rss()
        rss_delta = rss - self._last_rss if rss and self._last_rss else 0
        if rss:
            self._last_rss = rss
        event = _build_event(
            new_us=(us for us, _ in new.elements()),
            context=context_fn(),
            warn_ms=self._warn_ms,
            sites=sites,
            host_rss_delta_bytes=rss_delta,
            peak_compile_host_bytes=max(peak for _, peak in new.elements()),
        )
        self._record_event(event)
        log = logger.warning if event.heavy else logger.info
        log(_format_event(event, warn_ms=self._warn_ms))
        return event

    def _take_sites(self) -> _Sites | None:
        if self._site_tracer is None:
            return None
        sites = tuple(self._pending_sites.most_common())
        self._pending_sites.clear()
        return sites

    def _record_event(self, event: RecompileEvent) -> None:
        self._window_events += 1
        self._window_compile_ms += event.total_ms
        self.serving_compiles += len(event.durations_ms)
        self.serving_compile_ms += event.total_ms
        self.serving_programs += len(event.durations_ms) + event.load_count
        if event.durations_ms:
            self.worst_compile_ms = max(self.worst_compile_ms, event.durations_ms[0])

    def _maybe_report_window(self) -> None:
        elapsed = self._clock() - self._window_start
        if elapsed < _SUMMARY_WINDOW_SEC:
            return
        if self._window_events:
            compile_fraction = self._window_compile_ms / (elapsed * 1000)
            logger.info(
                "[recompilation checker] last %.0fs: %d forward(s) compiled, "
                "%d ms of compile time (%.1f%% of wall), %d program(s) and %d "
                "ms since serving started, worst %d ms.%s",
                elapsed,
                self._window_events,
                self._window_compile_ms,
                100 * compile_fraction,
                self.serving_compiles,
                self.serving_compile_ms,
                self.worst_compile_ms,
                self._format_host_growth(),
            )
        # Reset even with no events, so the next burst is measured from here.
        self._window_start = self._clock()
        self._window_events = 0
        self._window_compile_ms = 0

    def _format_host_growth(self) -> str:
        """This rank's RSS growth since the baseline, per program it added."""
        rss = self._read_rss()
        if not rss or not self._baseline_rss:
            return ""
        growth = rss - self._baseline_rss
        per_program = (
            f" ({growth / self.serving_programs / _MIB:.1f} MiB per program)"
            if self.serving_programs and growth > 0
            else ""
        )
        return (
            f" Host RSS of this rank {growth / _MIB:+.0f} MiB since the baseline "
            f"for {self.serving_programs} program(s){per_program}; each TP rank "
            "process holds its own copy of every executable."
        )


@functools.lru_cache(maxsize=1)
def _shared_recompilation_checker() -> RecompilationChecker | None:
    """The process-wide checker, None when torch.tpu lacks the cache-stats pybind."""
    try:
        read_misses = torch.tpu._get_cache_misses
        read_stats = torch.tpu._get_cache_stats
        sync_device = torch.accelerator.synchronize
    except AttributeError:
        return None
    return RecompilationChecker(
        read_misses=read_misses,
        read_stats=read_stats,
        sync_device=sync_device,
        warn_ms=_FORWARD_COMPILE_WARN_MS,
    )


@runtime_checkable
class _RecompilationCheckPolicy(Protocol):
    """Optional monitoring interface implemented by TPU decoder policies."""

    def set_recompilation_checker(
        self, checker: RecompilationChecker, model_label: str = ""
    ) -> None: ...


def install_recompilation_checker(
    *, policies: Iterable[CompilePolicy], model_label: str
) -> None:
    """Baseline after model warmup and attach the shared checker.

    Later TPU warmup phases refresh the same baseline when they finish.
    """
    if not envs.SGLANG_TPU_CHECK_RECOMPILATION.get():
        return
    monitored_policies = [
        policy for policy in policies if isinstance(policy, _RecompilationCheckPolicy)
    ]
    if not monitored_policies:
        return
    checker = _shared_recompilation_checker()
    if checker is None:
        logger.info(
            "[recompilation checker] torch.tpu cache stats unavailable, "
            "serving-time recompilation checks disabled."
        )
        return
    checker.snapshot_baseline()
    for policy in monitored_policies:
        policy.set_recompilation_checker(checker, model_label=model_label)


def snapshot_recompilation_baseline() -> None:
    """Exclude completed TPU warmup from subsequent serving-time checks.

    Speculative-op and KV-offload warmup run after model precompilation.
    Refresh the process-wide baseline at each phase's completion so their
    programs are excluded for both the target and draft checkers.
    """
    if not envs.SGLANG_TPU_CHECK_RECOMPILATION.get():
        return
    checker = _shared_recompilation_checker()
    if checker is not None:
        checker.snapshot_baseline()


def maybe_enable_site_tracing() -> None:
    """Trace eager compile sites on the calling thread. Call after all warmup."""
    if not envs.SGLANG_TPU_CHECK_RECOMPILATION_SITES.get():
        return
    if not envs.SGLANG_TPU_CHECK_RECOMPILATION.get():
        logger.warning(
            "[recompilation checker] SGLANG_TPU_CHECK_RECOMPILATION_SITES needs "
            "SGLANG_TPU_CHECK_RECOMPILATION=1, site tracing disabled."
        )
        return
    checker = _shared_recompilation_checker()
    if checker is None or not checker.has_baseline:
        logger.warning(
            "[recompilation checker] checker not installed, site tracing disabled."
        )
        return
    checker.enable_site_tracing()
    logger.info("[recompilation checker] eager compile site tracing enabled.")
