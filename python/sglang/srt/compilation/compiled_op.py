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


def compiled_op(*, warmup: WarmupArgsFactory):
    """Lazily ``torch.compile`` a pure tensor helper into one fused dispatch.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe. Every helper must provide a
    callable that builds representative arguments for ahead-of-time warm-up.
    """

    def decorator(fn):
        compiled_fn = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled_fn
            if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
                return fn(*args, **kwargs)
            if compiled_fn is None:
                compiled_fn = torch.compile(
                    fn, backend="tpu", fullgraph=True, dynamic=False
                )
            return compiled_fn(*args, **kwargs)

        op_name = f"{fn.__module__}.{fn.__qualname__}"
        _compiled_ops[op_name] = RegisteredCompiledOp(
            op=wrapper,
            make_warmup_args=warmup,
        )
        return wrapper

    return decorator
