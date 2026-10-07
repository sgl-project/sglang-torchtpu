"""Submit torch.compile helper warmups to the shared TPU compilation queue."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from sglang.srt.hardware_backend.tpu.compilation.pending_call import PendingCall


class DynamoCompiledFunction:
    def __init__(self, fn: Callable, *, async_compile: bool) -> None:
        self.fn = fn
        self.async_compile = async_compile
        self.compiled_fn = torch.compile(
            fn,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
            options={"async_compile": True} if async_compile else None,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            return self.fn(*args, **kwargs)
        result = self._submit(*args, **kwargs)
        return result.wait() if isinstance(result, PendingCall) else result

    def warmup_async(self, *args: Any, **kwargs: Any) -> PendingCall | None:
        result = self._submit(*args, **kwargs)
        return result if isinstance(result, PendingCall) else None

    def _submit(self, *args: Any, **kwargs: Any) -> Any:
        if not self.async_compile:
            return self.compiled_fn(*args, **kwargs)

        from torch_tpu._internal.compile import AsyncCompilationSubmitted

        try:
            return self.compiled_fn(*args, **kwargs)
        except AsyncCompilationSubmitted as submitted:
            # Retain inputs until execution, after all shapes have been submitted.
            return PendingCall(partial(self._finish, submitted.artifact, args, kwargs))

    def _finish(self, artifact: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        from torch_tpu._internal.compile import resolve_compilations

        resolve_compilations([artifact])
        return self.compiled_fn(*args, **kwargs)
