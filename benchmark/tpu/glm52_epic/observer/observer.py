"""Opt-in benchmark instrumentation; imports no torch/sglang at startup."""

import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import socket
import sys
import time
import warnings
from dataclasses import asdict, dataclass

ENV = "GLM52_EPIC_OBSERVER_DIR"
MARKER = "TORCH_RANK_LOCAL_PREFILL_COMPLETION"
SNAPSHOT_ATTR = "_glm52_epic_observer_snapshot"


@dataclass(frozen=True)
class RequestSnapshot:
    rid: str
    dp: int
    prefix: int
    tokens: int


@dataclass(frozen=True)
class Snapshot:
    bid: int
    forward_step: int
    tp_rank: int
    pp_rank: int
    attn_dp_rank: int
    requests: tuple
    unavailable: str = ""


def runtime_parallel():
    # Live topology, not dp_rank (which has different attention-DP semantics).
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def _normal_overlap(owner, batch, parallel):
    # Fail closed: exclude MIXED, idle, spec/MTP, PP, embedding and PD paths.
    return (
        parallel.pp_size == 1
        and owner.is_generation
        and owner.enable_overlap
        and not getattr(owner, "enable_overlap_mlx", False)
        and owner.disaggregation_mode.value == "null"
        and batch.spec_algorithm.is_none()
        and batch.forward_mode.name == "EXTEND"
    )


def _cpu_int(value):
    # Do not coerce tensors/scalars with int(): it could add a device read.
    if type(value) is not int or value < 0:
        raise ValueError("metadata must contain nonnegative Python ints")
    return value


def _snapshot(owner, batch, parallel):
    step = _cpu_int(owner.forward_ct) + 1
    rank = _cpu_int(parallel.attn_dp_rank)
    reason = ""
    requests = ()
    try:
        if not batch.reqs or not (
            len(batch.reqs) == len(batch.prefix_lens) == len(batch.extend_lens)
        ):
            raise ValueError("missing or misaligned CPU metadata")
        items = []
        for req, prefix, tokens in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens
        ):
            if type(req.rid) is not str or not req.rid:
                raise ValueError("missing string RID")
            items.append(
                RequestSnapshot(req.rid, rank, _cpu_int(prefix), _cpu_int(tokens))
            )
        if len({r.rid for r in items}) != len(items):
            raise ValueError("duplicate RID")
        requests = tuple(items)
    except (AttributeError, TypeError, ValueError):
        reason = "invalid_cpu_metadata"
    return Snapshot(
        id(batch), step, _cpu_int(parallel.tp_rank), _cpu_int(parallel.pp_rank),
        rank, requests, reason,
    )


def wrap_run_batch(original, parallel_getter=runtime_parallel):
    @functools.wraps(original)
    def wrapped(self, batch, *args, **kwargs):
        parallel = parallel_getter()
        if not _normal_overlap(self, batch, parallel):
            return original(self, batch, *args, **kwargs)
        # All mutable scheduler/request metadata is copied BEFORE original runs.
        snapshot = _snapshot(self, batch, parallel)
        result = original(self, batch, *args, **kwargs)
        if getattr(batch, "forward_iter", None) != snapshot.forward_step:
            from dataclasses import replace

            snapshot = replace(snapshot, unavailable="forward_iter_mismatch")
        setattr(result, SNAPSHOT_ATTR, snapshot)
        return result

    wrapped._glm52_epic_wrapped = True
    return wrapped


class CopyDoneProxy:
    """Observe only waits the processor already issues; never initiate one."""

    def __init__(self, event, clock):
        self._event = event
        self._clock = clock
        self.completed_ns = None
        self.wait_failed = False

    def __getattr__(self, name):
        return getattr(self._event, name)

    def synchronize(self, *args, **kwargs):
        try:
            value = self._event.synchronize(*args, **kwargs)
        except BaseException:
            self.wait_failed = True
            raise
        if self.completed_ns is None:
            self.completed_ns = self._clock()
        return value


def wrap_prefill(original, emit, parallel_getter=runtime_parallel,
                 clock=time.perf_counter_ns):
    @functools.wraps(original)
    def wrapped(self, batch, result, *args, **kwargs):
        snapshot = getattr(result, SNAPSHOT_ATTR, None)
        if snapshot is None or not _normal_overlap(self, batch, parallel_getter()):
            return original(self, batch, result, *args, **kwargs)
        # Install at ENTRY, before even free_group_begin/drain_before_wait.
        event = getattr(result, "copy_done", None)
        proxy = None if event is None else CopyDoneProxy(event, clock)
        if proxy is not None:
            result.copy_done = proxy
        try:
            return original(self, batch, result, *args, **kwargs)
        finally:
            if proxy is not None:
                result.copy_done = event
            reason = snapshot.unavailable
            if not reason:
                if proxy is None:
                    reason = "missing_copy_done"
                elif proxy.wait_failed:
                    reason = "copy_done_synchronize_failed"
                elif proxy.completed_ns is None:
                    reason = "existing_wait_not_observed"
            record = asdict(snapshot)
            record.pop("unavailable")
            record.update(
                marker=MARKER,
                schema_version=1,
                scope="rank_local_diagnostic_not_matched_jax",
                boundary="after_existing_copy_done_synchronize",
                clock="process_local_perf_counter_ns",
                status="unavailable" if reason else "observed",
                unavailable_reason=reason or None,
                completed_ns=None if reason else proxy.completed_ns,
            )
            # I/O is deferred until processing exits; timestamp is NOT exit time.
            # Never replace a missing wait with processor-entry/exit time.
            try:
                emit(record)
            except Exception as exc:
                try:
                    warnings.warn(
                        f"{MARKER} JSONL write failed: {exc}", RuntimeWarning
                    )
                except Exception:
                    # A deployment may turn warnings into errors; still do not
                    # mask the original processor exception/return value.
                    pass

    wrapped._glm52_epic_wrapped = True
    return wrapped


class JsonlSink:
    def __init__(self, directory):
        self.directory = Path(directory)

    def __call__(self, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Compute PID on each emission: safe after fork; no shared file handles.
        host = socket.gethostname().replace("/", "_")
        path = self.directory / f"{host}.pid{os.getpid()}.jsonl"
        record = dict(record, hostname=host, pid=os.getpid())
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


TARGETS = {
    "sglang.srt.managers.scheduler": ("Scheduler", "run_batch"),
    "sglang.srt.managers.scheduler_components.batch_result_processor": (
        "SchedulerBatchResultProcessor", "process_batch_result_prefill"
    ),
}


def _patch(module, sink):
    class_name, method_name = TARGETS[module.__name__]
    cls = getattr(module, class_name)
    original = getattr(cls, method_name)
    if getattr(original, "_glm52_epic_wrapped", False):
        return
    wrapper = (wrap_run_batch(original) if method_name == "run_batch"
               else wrap_prefill(original, sink))
    setattr(cls, method_name, wrapper)


class ObserverLoader(importlib.abc.Loader):
    def __init__(self, loader, sink):
        self.loader = loader
        self.sink = sink

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch(module, self.sink)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class ObserverFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sink):
        self.sink = sink

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = ObserverLoader(spec.loader, self.sink)
        return spec


def install():
    directory = os.environ.get(ENV)
    if not directory:
        return False
    if not Path(directory).is_absolute():
        raise ValueError(f"{ENV} must be an explicit absolute output directory")
    if any(isinstance(finder, ObserverFinder) for finder in sys.meta_path):
        return True
    sink = JsonlSink(directory)
    sys.meta_path.insert(0, ObserverFinder(sink))
    for name in TARGETS:
        if name in sys.modules:
            _patch(sys.modules[name], sink)
    return True
