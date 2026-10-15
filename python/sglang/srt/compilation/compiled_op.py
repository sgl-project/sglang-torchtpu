"""Lazily compile a pure tensor helper into a single fused dispatch."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_leaves, tree_map

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
    leaves, structure = tree_flatten((args, kwargs))
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
    trigger tracing and compilation, preserving the generator when ``uses_rng``
    is set.
    """

    def decorator(fn):
        compiled_fn = None
        use_direct = False
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, _warmup_async=False, **kwargs):
            nonlocal compiled_fn, use_direct
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)
            call = fn
            if is_tpu() and envs.SGLANG_ENABLE_TORCH_COMPILE.get():
                # Bind to fn's signature: positional warmups and keyword callers share a key.
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                args, kwargs = tree_map(_canonical_tensor, (bound.args, bound.kwargs))
                if compiled_fn is None:
                    use_direct = get_exec().graph.tpu_compile_tracer == "direct"
                    if use_direct:
                        # Break direct_compile -> eagle_info -> eagle_utils -> compiled_op.
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
                if use_direct and _warmup_async:
                    # Repeated storage/layout warmups share one compiled program.
                    if _specialization_key(*args, **kwargs) in compiled_fn.cache:
                        return None
                    return compiled_fn.precompile(
                        *args,
                        async_compile=envs.SGLANG_DIRECT_COMPILE_ASYNC.get(),
                        **kwargs,
                    )
                call = compiled_fn
            if _warmup_async and uses_rng:
                device = next(
                    value.device
                    for value in tree_leaves((args, kwargs))
                    if isinstance(value, torch.Tensor)
                )
                device_module = (
                    torch.random
                    if device.type == "cpu"
                    else torch.get_device_module(device)
                )
                state = device_module.get_rng_state()
                try:
                    return call(*args, **kwargs)
                finally:
                    device_module.set_rng_state(state)
            return call(*args, **kwargs)

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

    pending = op(*args, _warmup_async=True, **kwargs)
    if isinstance(pending, PendingCall):
        queue = current_warmup_queue()
        if queue is None:
            pending.wait()
        else:
            queue.calls.append(pending)
