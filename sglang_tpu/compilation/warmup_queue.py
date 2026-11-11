"""Submit and join TPU model and eager-operation warmups during startup."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.utils import LazyImport
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.pending_call import PendingCall

if TYPE_CHECKING:
    from sglang_tpu.compilation.lowered_graph_cache import FileLoweredGraphStore

SPMDCompiler = LazyImport("sglang_tpu.compilation.spmd_compiler", "SPMDCompiler")
create_lowered_graph_store = LazyImport(
    "sglang_tpu.compilation.lowered_graph_cache", "create_lowered_graph_store"
)
logger = get_logger(__name__)


class WarmupQueue:
    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        self.verify = os.environ.get("VERIFY_SPMD_SAFE", "0") == "1"
        self.spmd_compiler: SPMDCompiler | None = None
        self.lowered_graph_store: FileLoweredGraphStore | None = None
        self.calls: list[PendingCall] = []
        # TpuGraphRunner attaches its recompilation checker after every warmup.
        self.completions: list[Callable[[], None]] = []

        if not dist.is_initialized():
            if self.verify:
                raise ValueError(
                    "SPMD verification requires initialized distributed workers"
                )
            return
        if group is None:
            group = get_parallel().tp_group.cpu_group
        graph = get_exec().graph
        save_dir = graph.unsafe_save_torch_compile_bundle
        load_dir = graph.unsafe_load_torch_compile_bundle
        # Under the direct tracer the bundle flags select the store (else they
        # select torch.compile bundles); without one, don't import its module.
        if graph.tpu_compile_tracer == "direct" and (save_dir or load_dir):
            self.lowered_graph_store = create_lowered_graph_store(save_dir, load_dir)
        use_spmd_compile = graph.use_spmd_compile
        # Agree before compiler construction can enter mode-specific collectives.
        mode = (self.verify, use_spmd_compile)
        modes = [None] * dist.get_world_size(group)
        dist.all_gather_object(modes, mode, group=group)
        if any(peer != mode for peer in modes):
            raise ValueError("SPMD compilation flags differ across TPU ranks")
        if self.verify and not use_spmd_compile:
            raise ValueError("SPMD verification requires --use-spmd-compile")
        if use_spmd_compile:
            self.spmd_compiler = SPMDCompiler(group, verify_spmd_safe=self.verify)

    def wait(self) -> None:
        with torch.no_grad():
            # Release completed calls' inputs before executing the next program.
            while self.calls:
                self.calls.pop(0).wait()
        for complete in self.completions:
            complete()
        # Finish the store last: only after every call and completion
        # succeeded, and after anything a completion compiles.
        if self.lowered_graph_store is not None:
            self.lowered_graph_store.finish()


_CURRENT: ContextVar[WarmupQueue | None] = ContextVar("tpu_warmup_queue", default=None)


def current_warmup_queue() -> WarmupQueue | None:
    return _CURRENT.get()


def current_lowered_graph_store() -> FileLoweredGraphStore | None:
    queue = _CURRENT.get()
    return queue.lowered_graph_store if queue is not None else None


@contextmanager
def eager_warmup() -> Iterator[Callable[[torch.Tensor], None]]:
    """Submit the usual D2H boundaries, then join before releasing their buffers."""
    stream = torch.tpu.current_stream()
    host_outputs: list[torch.Tensor] = []

    def materialize(value: torch.Tensor) -> None:
        # D2H materializes views as well as their base buffers, like .cpu().
        # TorchTPU retains the destination pointer; keep its storage until join.
        host_outputs.append(value.to("cpu", non_blocking=True))

    try:
        yield materialize
    finally:
        stream.synchronize()
        host_outputs.clear()


@contextmanager
def collect_warmups(*, group: dist.ProcessGroup | None = None) -> Iterator[WarmupQueue]:
    existing = _CURRENT.get()
    if existing is not None:
        yield existing
        return
    queue = WarmupQueue(group)
    spmd_compiler = queue.spmd_compiler
    token = _CURRENT.set(queue)
    start = time.perf_counter()
    try:
        trace_error: str | None = None
        try:
            yield queue
        except Exception as caught:
            if spmd_compiler is None:
                raise
            logger.exception("SPMD compilation trace failed")
            trace_error = f"{type(caught).__name__}: {caught}"
        submitted = time.perf_counter()
        count = len(queue.calls)
        logger.info("TPU global warmup: submitted %d calls; joining", count)
        # Pending calls retain this queue's context, but must compile locally after join.
        queue.spmd_compiler = None
        if spmd_compiler is not None:
            spmd_compiler.finish(trace_error)
        queue.wait()
        finished = time.perf_counter()
        logger.info(
            "TPU global warmup: pending=%d submit=%.3fs wait=%.3fs total=%.3fs",
            count,
            submitted - start,
            finished - submitted,
            finished - start,
        )
    finally:
        _CURRENT.reset(token)
        queue.calls.clear()
        queue.completions.clear()
        queue.spmd_compiler = None
        queue.lowered_graph_store = None
        if spmd_compiler is not None:
            spmd_compiler.clear()
