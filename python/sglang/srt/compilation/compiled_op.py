"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, List

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_tpu


@dataclass(frozen=True)
class CompiledOpWarmup:
    """Arguments generated for one compiled-op warm-up invocation."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


CompiledOpWarmupDataGenerator = Callable[[Any], CompiledOpWarmup]


@dataclass(frozen=True)
class RegisteredCompiledOp:
    """A compiled helper and the recipe for constructing its warm-up inputs."""

    op: Callable
    warmup_data_generator: CompiledOpWarmupDataGenerator


_compiled_ops: dict[str, RegisteredCompiledOp] = {}


def get_compiled_ops() -> tuple[RegisteredCompiledOp, ...]:
    """Return helpers and warm-up recipes registered by ``compiled_op``."""
    return tuple(_compiled_ops.values())


def compiled_op(*, warmup_data_generator: CompiledOpWarmupDataGenerator):
    """Lazily ``torch.compile`` a pure tensor helper into one fused dispatch.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe. Every helper must provide a
    generator that produces representative arguments for ahead-of-time warm-up.
    """

    def decorator(fn):
        compiled_holder: List[Any] = []

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
                return fn(*args, **kwargs)
            if not compiled_holder:
                compiled_holder.append(
                    torch.compile(fn, backend="tpu", fullgraph=False, dynamic=False)
                )
            return compiled_holder[0](*args, **kwargs)

        op_name = f"{fn.__module__}.{fn.__qualname__}"
        _compiled_ops[op_name] = RegisteredCompiledOp(
            op=wrapper,
            warmup_data_generator=warmup_data_generator,
        )
        return wrapper

    return decorator
