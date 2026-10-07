"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

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

    op: CompiledOp
    make_warmup_args: WarmupArgsFactory


_compiled_ops: dict[str, RegisteredCompiledOp] = {}


class CompiledOp:
    """One helper's serving callable and explicit, output-free warmup entrypoint."""

    def __init__(self, fn: Callable) -> None:
        self.fn = fn
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", type(fn).__name__)
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__module__ = fn.__module__
        self.__doc__ = fn.__doc__
        self._compiled_fn = None

    def _submit(self, *args, **kwargs):
        if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
            return self.fn(*args, **kwargs)
        if self._compiled_fn is None:
            self._compiled_fn = torch.compile(
                self.fn,
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

        from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall

        try:
            return self._compiled_fn(*args, **kwargs)
        except AsyncCompilationSubmitted as submitted:
            artifact = submitted.artifact

            def finish():
                resolve_compilations([artifact])
                return self._compiled_fn(*args, **kwargs)

            return PendingCall(finish)

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.fn(*args, **kwargs)
        from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall

        result = self._submit(*args, **kwargs)
        return result.wait() if isinstance(result, PendingCall) else result

    def warmup_async(self, *args, **kwargs):
        from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall

        result = self._submit(*args, **kwargs)
        return result if isinstance(result, PendingCall) else None

    def warmup(self, *args, **kwargs) -> None:
        """Submit independent inputs, or finish immediately outside a queue."""
        from sglang.srt.hardware_backend.tpu.compilation.warmup_queue import (
            current_warmup_queue,
        )

        pending = self.warmup_async(*args, **kwargs)
        if pending is not None:
            queue = current_warmup_queue()
            if queue is None:
                pending.wait()
            else:
                queue.calls.append(pending)


def get_compiled_ops() -> tuple[RegisteredCompiledOp, ...]:
    """Return helpers and warm-up recipes registered by ``compiled_op``."""
    return tuple(_compiled_ops.values())


def compiled_op(*, warmup: WarmupArgsFactory | None = None):
    """Lazily ``torch.compile`` a pure tensor helper into one fused dispatch.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe. A ``warmup`` factory registers
    the helper for the speculative-op grid. Model-specific owners can instead
    call ``op.warmup`` with their own representative inputs.
    """

    def decorator(fn):
        op = CompiledOp(fn)
        if warmup is not None:
            op_name = f"{fn.__module__}.{fn.__qualname__}"
            _compiled_ops[op_name] = RegisteredCompiledOp(
                op=op,
                make_warmup_args=warmup,
            )
        return op

    return decorator
