"""Trace the dynamo decoder once per graph structure instead of once per bucket.

With ``dynamic=False`` every token / batch bucket re-runs the whole Dynamo
front end (bytecode tracing, guards, AOTAutograd), about 40 s per bucket for
Qwen3-32B TP8. vLLM traces its backbone once with the token dimension
symbolic and specializes the FX graph per compile size in its backend
(``PiecewiseBackend.compile_all_ranges`` / ``create_concrete_args``). This
module does the same for the TPU decoder policy:

* :func:`mark_bucket_dims_dynamic` marks the bucket-sized dimensions of the
  forward inputs with ``maybe_mark_dynamic`` before the compiled call, so the
  first bucket of a structure traces symbolically and later buckets hit that
  cache entry.
* :class:`PerShapeBackend` is the Dynamo backend. A graph with free symbols is
  never handed to the TPU backend as is (it rejects SymInts): on each call it
  reads the concrete symbol values from the graph's SymInt inputs, rebuilds
  static example inputs for them and compiles that static graph once. So each
  bucket still gets its own static executable, exactly as before; only the
  Dynamo traces are shared.

Size-1 dimensions are never marked: Dynamo would specialize them anyway, and
leaving them static keeps broadcasting semantics exact without
``backed_size_oblivious``. Marking is a hint (``maybe_mark_dynamic``): code
that specializes on a size falls back to a static trace for that bucket, which
costs time but not correctness.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import msgspec
import torch
import torch._guards
from torch._dynamo import eval_frame

from sglang_tpu._logging import get_logger

logger = get_logger(__name__)


class SymbolicWarmupStats(msgspec.Struct):
    """Process-wide counters, logged after warmup and read by tests."""

    dynamo_traces: int = 0
    symbolic_traces: int = 0
    static_compiles: int = 0
    trace_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"dynamo_traces={self.dynamo_traces} "
            f"(symbolic={self.symbolic_traces}) "
            f"static_compiles={self.static_compiles}"
        )


STATS = SymbolicWarmupStats()
_lock = threading.Lock()


def _symbol_of(value: Any) -> Any | None:
    """The sympy symbol a SymInt placeholder stands for, if it is a bare one."""
    if not isinstance(value, torch.SymInt):
        return None
    expr = value.node.expr
    return expr if expr.is_Symbol else None


def _concretize(value: Any, sizes: dict[Any, int]) -> int:
    from torch.fx.experimental.symbolic_shapes import is_symbolic

    if not is_symbolic(value):
        return int(value)
    expr = value.node.expr
    return int(expr.subs({s: sizes[s] for s in expr.free_symbols}))


def concrete_example_inputs(
    graph: torch.fx.GraphModule, sizes: dict[Any, int]
) -> list[Any]:
    """Fake example inputs with every symbol replaced by its value in ``sizes``.

    Mirrors vLLM's ``create_concrete_args``, generalized to several symbols
    (an extend graph has a token and a batch dimension).
    """
    from torch._prims_common import compute_required_storage_length
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    fake_mode = FakeTensorMode(shape_env=ShapeEnv())
    args: list[Any] = []
    with fake_mode:
        for node in graph.graph.find_nodes(op="placeholder"):
            value = node.meta["example_value"]
            if isinstance(value, torch.SymInt):
                args.append(_concretize(value, sizes))
            elif isinstance(value, torch.Tensor):
                shape = tuple(_concretize(d, sizes) for d in value.shape)
                stride = tuple(_concretize(s, sizes) for s in value.stride())
                offset = _concretize(value.storage_offset(), sizes)
                storage = compute_required_storage_length(shape, stride, offset)
                tensor = torch.empty(storage, dtype=value.dtype, device=value.device)
                args.append(tensor.as_strided(shape, stride, offset))
            else:
                args.append(value)
    return args


class _PerShapeGraph:
    """Dispatches one symbolic Dynamo graph to a static executable per shape."""

    def __init__(
        self,
        graph: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        compile_static: Callable[[torch.fx.GraphModule, list[Any]], Callable],
    ) -> None:
        self._graph = graph
        self._compile_static = compile_static
        # Dynamo lifts every free symbol as a SymInt graph input.
        self._symbol_inputs = [
            (index, symbol)
            for index, value in enumerate(example_inputs)
            if (symbol := _symbol_of(value)) is not None
        ]
        free = {
            s
            for value in example_inputs
            if isinstance(value, torch.SymInt)
            for s in value.node.expr.free_symbols
        }
        bound = {symbol for _, symbol in self._symbol_inputs}
        if free - bound:
            raise NotImplementedError(
                f"symbolic warmup: symbols {sorted(map(str, free - bound))} are "
                "not graph inputs, so their runtime values are unknown"
            )
        self._executables: dict[tuple[int, ...], Callable[..., Any]] = {}

    def __call__(self, *args: Any) -> Any:
        key = tuple(int(args[index]) for index, _ in self._symbol_inputs)
        executable = self._executables.get(key)
        if executable is None:
            executable = self._compile(key)
        return executable(*args)

    def _compile(self, key: tuple[int, ...]) -> Callable[..., Any]:
        # A new shape after warmup would bypass Dynamo's recompile check, since
        # the Dynamo cache entry already exists. Fail the same way it would.
        if eval_frame._stance.stance == "fail_on_recompile":
            raise RuntimeError(
                "Detected recompile when torch.compile stance is "
                f"'fail_on_recompile'. symbolic warmup shape: {key!r}"
            )
        sizes = {symbol: value for (_, symbol), value in zip(self._symbol_inputs, key)}
        example_inputs = concrete_example_inputs(self._graph, sizes)
        # The TPU backend (like vLLM's adaptor) needs a tracing context whose
        # fake mode matches the example inputs and carries a ShapeEnv.
        fake_mode = next(
            (
                t.fake_mode
                for t in example_inputs
                if isinstance(t, torch._subclasses.fake_tensor.FakeTensor)
            ),
            None,
        )
        context = (
            torch._guards.TracingContext(fake_mode) if fake_mode is not None else None
        )
        started = time.perf_counter()
        with torch._guards.tracing(context):
            executable = self._compile_static(
                copy.deepcopy(self._graph), example_inputs
            )
        with _lock:
            STATS.static_compiles += 1
        logger.info(
            "[symbolic warmup] compiled shape %s in %.1fs",
            dict(zip((str(s) for _, s in self._symbol_inputs), key)),
            time.perf_counter() - started,
        )
        self._executables[key] = executable
        return executable


class PerShapeBackend:
    """Dynamo backend that compiles symbolic graphs once per concrete shape.

    ``inner`` is the static backend (``"tpu"`` or a callable). Its ``options``
    are forwarded unchanged; static graphs go straight to it.
    """

    def __init__(self, inner: str | Callable[..., Any]) -> None:
        self._inner = inner

    def _inner_fn(self) -> Callable[..., Any]:
        if isinstance(self._inner, str):
            from torch._dynamo.backends.registry import lookup_backend

            return lookup_backend(self._inner)
        return self._inner

    def __call__(
        self,
        graph: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        **kwargs: Any,
    ) -> Callable[..., Any]:
        inner = self._inner_fn()
        symbolic = any(isinstance(v, torch.SymInt) for v in example_inputs)
        with _lock:
            STATS.dynamo_traces += 1
            STATS.symbolic_traces += int(symbolic)
        if not symbolic:
            with _lock:
                STATS.static_compiles += 1
            return inner(graph, example_inputs, **kwargs)
        logger.info(
            "[symbolic warmup] dynamo trace #%d is symbolic: %s",
            STATS.dynamo_traces,
            [str(v) for v in example_inputs if isinstance(v, torch.SymInt)],
        )
        return _PerShapeGraph(
            graph,
            example_inputs,
            lambda gm, inputs: inner(gm, inputs, **kwargs),
        )

    # torch.compile compares backends to decide whether to reuse caches.
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PerShapeBackend) and other._inner == self._inner

    def __hash__(self) -> int:
        return hash(("PerShapeBackend", self._inner))


def _tensor_attrs(obj: Any) -> Iterable[tuple[str, torch.Tensor]]:
    # ForwardBatch is a dataclass; its fields are always set (possibly None).
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = ((f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj))
    else:
        items = vars(obj).items()
    return ((k, v) for k, v in items if isinstance(v, torch.Tensor) and v.ndim > 0)


def _is_bucket_sized(size: int, pad_sizes: set[int]) -> bool:
    # Batch-major metadata is sized bs (seq_lens), bs + 1 (cu_q_lens) or
    # bs * pages_per_seq (page tables); token-major fields are num_tokens.
    return any(size in (p, p + 1) or size % p == 0 for p in pad_sizes)


def mark_bucket_dims_dynamic(
    pad_sizes: Iterable[int],
    tensors: Iterable[torch.Tensor | None] = (),
    objects: Iterable[Any] = (),
    last_dim_fields: frozenset[str] = frozenset({"mrope_positions"}),
) -> int:
    """``maybe_mark_dynamic`` every bucket-sized dimension; return the count.

    A dim-0 extent is bucket-sized when it is a padded token / batch count, one
    more than it, or a multiple of it (see ``_is_bucket_sized``). Fields in
    ``last_dim_fields`` ([3, tokens] mRoPE positions) check their last dim.
    Over-marking a dim that is constant across buckets is harmless: its symbol
    takes the same value for every bucket. Size 1 is never marked.
    """
    sizes = {int(s) for s in pad_sizes if int(s) > 1}
    if not sizes:
        return 0
    marked = 0

    def mark(tensor: torch.Tensor, dim: int) -> None:
        nonlocal marked
        size = int(tensor.shape[dim])
        if size > 1 and _is_bucket_sized(size, sizes):
            torch._dynamo.maybe_mark_dynamic(tensor, dim)
            marked += 1

    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            mark(tensor, 0)
    for obj in objects:
        if obj is None:
            continue
        for name, tensor in _tensor_attrs(obj):
            mark(tensor, -1 if name in last_dim_fields else 0)
    return marked


def warmup_objects(forward_batch: Any, attn_backend: Any) -> list[Any]:
    """The objects whose tensors a compiled decoder forward reads per bucket."""
    objects = [forward_batch, forward_batch.sampling_info]
    # Backends keep their per-forward metadata on themselves; hybrid backends
    # hold one sub-backend per layer kind.
    sub_backends = vars(attn_backend).get("attn_backends") or ()
    backends = [attn_backend, *sub_backends]
    for backend in backends:
        metadata = vars(backend).get("forward_metadata")
        if metadata is not None:
            objects.append(metadata)
    return objects
