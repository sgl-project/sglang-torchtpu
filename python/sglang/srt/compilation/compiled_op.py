"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_tpu

WarmupArgsFactory = Callable[[Any], tuple[Any, ...]]


@dataclass(frozen=True)
class RegisteredCompiledOp:
    """A compiled helper and the recipe for constructing its warm-up inputs."""

    op: Callable
    make_warmup_args: WarmupArgsFactory


_compiled_ops: dict[str, RegisteredCompiledOp] = {}


def get_compiled_ops() -> tuple[RegisteredCompiledOp, ...]:
    """Return helpers and warm-up recipes registered by ``compiled_op``."""
    return tuple(_compiled_ops.values())


def compiled_op(*, warmup: WarmupArgsFactory | None = None):
    """Lazily ``torch.compile`` a pure tensor helper into one fused dispatch.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe. A ``warmup`` factory registers
    the helper for the speculative-op grid. Model-specific owners can instead
    call ``warmup_compiled_op(op, ...)`` with their own representative inputs.
    The internal ``_warmup_async`` keyword returns a pending call without waiting.
    """

    def decorator(fn):
        compiled_fn = None

        @functools.wraps(fn)
        def wrapper(*args, _warmup_async=False, **kwargs):
            nonlocal compiled_fn
            if (
                torch.compiler.is_compiling()
                or not is_tpu()
                or not envs.SGLANG_ENABLE_TORCH_COMPILE.get()
            ):
                return fn(*args, **kwargs)
            if compiled_fn is None:
                compiled_fn = torch.compile(
                    fn,
                    backend="tpu",
                    fullgraph=True,
                    dynamic=False,
                    options=(
                        {"async_compile": True}
                        if envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
                        else None
                    ),
                )

            from torch_tpu._internal.compile import (
                AsyncCompilationSubmitted,
                resolve_compilations,
            )

            from sglang.srt.hardware_backend.tpu.compilation.pending_call import (
                PendingCall,
            )

            try:
                return compiled_fn(*args, **kwargs)
            except AsyncCompilationSubmitted as submitted:
                artifact = submitted.artifact

                def finish():
                    resolve_compilations([artifact])
                    return compiled_fn(*args, **kwargs)

                pending = PendingCall(finish)
                return pending if _warmup_async else pending.wait()

        if warmup is not None:
            op_name = f"{fn.__module__}.{fn.__qualname__}"
            _compiled_ops[op_name] = RegisteredCompiledOp(
                op=wrapper,
                make_warmup_args=warmup,
            )
        return wrapper

    return decorator


def warmup_compiled_op(op: Callable, *args, **kwargs) -> None:
    """Submit output-free warmup to the current queue, or finish immediately."""
    from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall
    from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import (
        current_warmup_queue,
    )

    pending = op(*args, _warmup_async=True, **kwargs)
    if isinstance(pending, PendingCall):
        queue = current_warmup_queue()
        if queue is None:
            pending.wait()
        else:
            queue.calls.append(pending)
