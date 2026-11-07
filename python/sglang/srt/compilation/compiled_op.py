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
from sglang.utils import LazyImport

current_warmup_queue = LazyImport(
    "sglang_tpu.compilation.warmup_queue", "current_warmup_queue"
)
direct_compile = LazyImport("sglang_tpu.compilation.direct_compile", "direct_compile")

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


def _specialization_key(spmd_safe, *args, **kwargs):
    leaves, structure = tree_flatten((args, kwargs))
    key = tuple(
        (tuple(value.shape), value.dtype, str(value.device))
        if isinstance(value, torch.Tensor)
        else (type(value).__name__, value)
        for value in leaves
    )
    # spmd_safe keys name shared graphs, so they must serialize to JSON.
    return key if spmd_safe else (structure, key)


def compiled_op(
    *,
    warmup: WarmupArgsFactory | None = None,
    model: torch.nn.Module | None = None,
    uses_rng: bool = False,
    spmd_safe: bool = False,
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
    Direct helper warmups compile without executing and release example inputs
    after submission. Dynamo warmup executes to trigger
    compilation, preserving the generator when ``uses_rng`` is set. Warmup is a
    no-op when compilation is disabled or off TPU.
    ``spmd_safe=True`` names each direct graph after the helper and keys it
    without the argument structure, which binding the signature fixes, so
    ``--use-spmd-compile`` can compile it once and share it across TP ranks. Such
    a helper takes tensors and plain values, passes the configuration its graph
    bakes in as arguments rather than closures, and cannot set ``uses_rng``.
    """
    if spmd_safe and uses_rng:
        raise ValueError("spmd_safe compiled ops cannot draw from the TPU generator")

    def decorator(fn):
        compiled_fn = None
        use_direct = False
        signature = inspect.signature(fn)
        n_params = len(signature.parameters)
        op_name = (
            fn._qualname
            if isinstance(fn, torch.library.CustomOpDef)
            else f"{fn.__module__}.{fn.__qualname__}"
        )
        specialization_key = functools.partial(_specialization_key, spmd_safe)

        @functools.wraps(fn)
        def wrapper(*args, _warmup_async=False, **kwargs):
            nonlocal compiled_fn, use_direct
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)
            if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
                return None if _warmup_async else fn(*args, **kwargs)
            queue = current_warmup_queue() if _warmup_async else None
            if queue is not None and queue.verify and not spmd_safe:
                return None
            # Positional warmups and keyword callers share a specialization.
            if kwargs or len(args) != n_params:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                args, kwargs = bound.args, bound.kwargs
            # Flat tensor arguments skip the pytree walk (per-call host cost).
            args = tuple(_canonical_tensor(a) for a in args)
            if kwargs:
                kwargs = {k: _canonical_tensor(v) for k, v in kwargs.items()}
            if any(isinstance(a, (list, tuple, dict)) for a in args) or any(
                isinstance(v, (list, tuple, dict)) for v in kwargs.values()
            ):
                args, kwargs = tree_map(_canonical_tensor, (args, kwargs))
            if compiled_fn is None:
                use_direct = get_exec().graph.tpu_compile_tracer == "direct"
                if use_direct:
                    compiled_fn = direct_compile(
                        fn,
                        specialization_key=specialization_key,
                        name=op_name if spmd_safe else None,
                        spmd_safe=spmd_safe,
                        model=model,
                        uses_rng=uses_rng,
                    )
                else:
                    compiled_fn = torch.compile(
                        fn, backend="tpu", fullgraph=True, dynamic=False
                    )
            if use_direct and _warmup_async:
                shared = (
                    spmd_safe and queue is not None and queue.spmd_compiler is not None
                )
                # Every shared queue needs participation even when local caches differ.
                if (
                    not shared
                    and specialization_key(*args, **kwargs) in compiled_fn.cache
                ):
                    return None
                return compiled_fn.precompile(
                    *args,
                    async_compile=shared or envs.SGLANG_DIRECT_COMPILE_ASYNC.get(),
                    **kwargs,
                )
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
                    compiled_fn(*args, **kwargs)
                finally:
                    device_module.set_rng_state(state)
                return None
            result = compiled_fn(*args, **kwargs)
            return None if _warmup_async else result

        if warmup is not None:
            _compiled_ops[op_name] = RegisteredCompiledOp(
                op=wrapper,
                make_warmup_args=warmup,
            )
        return wrapper

    return decorator


def warmup_compiled_op(op: Callable, *args, **kwargs) -> None:
    """Submit output-free warmup to the current queue, or finish immediately."""
    pending = op(*args, _warmup_async=True, **kwargs)
    if pending is not None:
        queue = current_warmup_queue()
        if queue is None:
            pending.wait()
        else:
            queue.calls.append(pending)
