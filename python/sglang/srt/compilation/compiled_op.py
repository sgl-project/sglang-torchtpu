"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_tpu

WarmupArgsFactory = Callable[[Any], tuple[Any, ...]]


@dataclass(frozen=True)
class RegisteredCompiledOp:
    """A compiled helper and the recipe for constructing its warm-up inputs."""

    op: Callable
    make_warmup_args: WarmupArgsFactory
    warmup_async: Callable


_compiled_ops: dict[str, RegisteredCompiledOp] = {}


def get_compiled_ops() -> tuple[RegisteredCompiledOp, ...]:
    """Return helpers and warm-up recipes registered by ``compiled_op``."""
    return tuple(_compiled_ops.values())


def compiled_op(*, warmup: WarmupArgsFactory):
    """Lazily ``torch.compile`` a pure tensor helper into one fused dispatch.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe. Every helper must provide a
    callable that builds representative arguments for ahead-of-time warm-up.
    """

    def decorator(fn):
        compiled_fn = None

        def get_compiled_fn():
            nonlocal compiled_fn
            if compiled_fn is None:
                from sglang.srt.hardware_backend.tpu.compilation.dynamo_compile import (
                    DynamoCompiledFunction,
                )

                compiled_fn = DynamoCompiledFunction(
                    fn, async_compile=envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
                )
            return compiled_fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
                return fn(*args, **kwargs)
            return get_compiled_fn()(*args, **kwargs)

        def warmup_async(*args, **kwargs):
            if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
                fn(*args, **kwargs)
                return None
            return get_compiled_fn().warmup_async(*args, **kwargs)

        op_name = f"{fn.__module__}.{fn.__qualname__}"
        _compiled_ops[op_name] = RegisteredCompiledOp(
            op=wrapper,
            make_warmup_args=warmup,
            warmup_async=warmup_async,
        )
        return wrapper

    return decorator
