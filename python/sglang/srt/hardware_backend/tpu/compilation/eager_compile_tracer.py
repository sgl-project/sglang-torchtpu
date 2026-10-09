# SPDX-License-Identifier: Apache-2.0
"""Debug-only attribution of serving-time torch_tpu eager compiles to sglang source lines."""

from __future__ import annotations

import collections
import time
import traceback
from typing import Any, Callable

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)

_SUMMARY_INTERVAL_S = 60.0
_CALLER_FRAMES = 3

_SiteKey = tuple[str, int, str]


def _sglang_frames() -> list[traceback.FrameSummary]:
    """Innermost-first sglang frames, excluding this module."""
    return [
        fs
        for fs in reversed(traceback.extract_stack())
        if "/sglang/" in fs.filename and fs.filename != __file__
    ]


def _describe_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    return [
        f"{tuple(leaf.shape)}:{str(leaf.dtype).removeprefix('torch.')}"
        if isinstance(leaf, torch.Tensor)
        else repr(leaf)
        for leaf in pytree.tree_leaves((args, kwargs))
    ]


class EagerCompileTracer(TorchDispatchMode):
    """Logs the sglang call site of every eager op that misses the TPU program cache.

    Attributes:
      counts: Compiles per (filename, lineno, op) call site.
    """

    def __init__(self, read_misses: Callable[[], int]) -> None:
        super().__init__()
        self._read_misses = read_misses
        self.counts: collections.Counter[_SiteKey] = collections.Counter()
        self._last_summary = time.monotonic()

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # Otherwise Dynamo runs compiled functions eagerly while the mode is active.
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        before = self._read_misses()
        out = func(*args, **kwargs)
        if self._read_misses() != before:
            self._record(func, args, kwargs)
        return out

    def _record(self, func, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        frames = _sglang_frames()
        site = frames[0] if frames else None
        key = (
            (site.filename, site.lineno or 0, str(func))
            if site
            else ("<no sglang frame>", 0, str(func))
        )
        self.counts[key] += 1
        stack = "\n".join(
            f"    {fs.filename}:{fs.lineno} in {fs.name}\n      {fs.line}"
            for fs in frames[:_CALLER_FRAMES]
        )
        logger.warning(
            "[eager compile] #%d %s args=%s\n%s",
            self.counts[key],
            func,
            _describe_args(args, kwargs),
            stack,
        )
        now = time.monotonic()
        if now - self._last_summary >= _SUMMARY_INTERVAL_S:
            self._last_summary = now
            self.log_summary()

    def log_summary(self) -> None:
        lines = "\n".join(
            f"  {n:6d}  {op}  {filename}:{lineno}"
            for (filename, lineno, op), n in self.counts.most_common(30)
        )
        logger.warning(
            "[eager compile] summary: %d compiles at %d sites\n%s",
            sum(self.counts.values()),
            len(self.counts),
            lines,
        )


_tracer: EagerCompileTracer | None = None


def maybe_install_eager_compile_tracer() -> None:
    """Enter the tracer on the calling (scheduler/forward) thread after warmup.

    Dispatch modes are thread-local, so ops issued from other threads are not traced.
    """
    global _tracer
    if not envs.SGLANG_DEBUG_TPU_EAGER_COMPILE_SITES.get() or _tracer is not None:
        return
    _tracer = EagerCompileTracer(read_misses=torch.tpu._get_cache_misses)
    _tracer.__enter__()
    logger.info("[eager compile] tracer installed on the scheduler thread.")
