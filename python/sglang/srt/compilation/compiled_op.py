"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
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


def _canonical_tensor(value):
    if not isinstance(value, torch.Tensor):
        return value
    strides, step = [], 1
    for size in reversed(value.shape):
        strides.append(step)
        step *= max(size, 1)
    # placeholder_like preserves even size-1 strides and storage offsets.
    # Warmup and serving must expose the same logical buffer to the executable.
    if value.storage_offset() == 0 and value.stride() == tuple(reversed(strides)):
        return value
    return value.clone(memory_format=torch.contiguous_format)


def _specialization_key(*args, **kwargs):
    from jax import tree_util

    leaves, structure = tree_util.tree_flatten((args, kwargs))
    return structure, tuple(
        (tuple(value.shape), value.dtype, value.device)
        if isinstance(value, torch.Tensor)
        else (type(value), value)
        for value in leaves
    )


def compiled_op(
    *,
    warmup: WarmupArgsFactory | None = None,
    model: torch.nn.Module | None = None,
    uses_rng: bool = False,
):
    """Compile a tensor helper with the selected TPU tracer.

    Active only on TPU with ``--enable-torch-compile``. The caller must pass
    TPU tensors and ``fn`` must be compile-safe. ``--tpu-compile-tracer`` selects
    direct compilation or synchronous ``torch.compile``. For direct compilation,
    tensor shapes, dtypes, devices,
    argument structure, and non-Tensor values determine specialization; closed-over
    Python configuration must stay fixed. Pass ``model`` for captured parameters
    and ``uses_rng=True`` for helpers drawing from the default TPU generator.
    A ``warmup`` factory registers the helper for the speculative-op grid.
    Model-specific owners can instead
    call ``warmup_compiled_op(op, ...)`` with their own representative inputs.
    Direct helper warmups compile without executing; the queue joins compilation
    futures without retaining example inputs. Dynamo warmups still execute to
    trigger tracing and compilation.
    """

    def decorator(fn):
        compiled_fn = None
        use_direct = False

        @functools.wraps(fn)
        def wrapper(*args, _warmup=False, **kwargs):
            nonlocal compiled_fn, use_direct
            if (
                torch.compiler.is_compiling()
                or not is_tpu()
                or not envs.SGLANG_ENABLE_TORCH_COMPILE.get()
            ):
                return fn(*args, **kwargs)
            from jax import tree_util

            args, kwargs = tree_util.tree_map(_canonical_tensor, (args, kwargs))
            if compiled_fn is None:
                use_direct = get_exec().graph.tpu_compile_tracer == "direct"
                if use_direct:
                    from sglang_tpu.compilation.direct_compile import (
                        direct_compile,
                    )

                    compiled_fn = direct_compile(
                        fn,
                        specialization_key=_specialization_key,
                        model=model,
                        uses_rng=uses_rng,
                    )
                else:
                    compiled_fn = torch.compile(
                        fn, backend="tpu", fullgraph=True, dynamic=False
                    )
            if use_direct and _warmup:
                # Repeated storage/layout warmups share one compiled program.
                if _specialization_key(*args, **kwargs) in compiled_fn.cache:
                    return None
                return compiled_fn.warmup(
                    *args,
                    async_compile=envs.SGLANG_DIRECT_COMPILE_ASYNC.get(),
                    **kwargs,
                )
            return compiled_fn(*args, **kwargs)

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
    from sglang_tpu.compilation.pending_call import PendingCall
    from sglang_tpu.compilation.warmup_queue import (
        current_warmup_queue,
    )

    pending = op(*args, _warmup=True, **kwargs)
    if isinstance(pending, PendingCall):
        queue = current_warmup_queue()
        if queue is None:
            pending.wait()
        else:
            queue.calls.append(pending)
