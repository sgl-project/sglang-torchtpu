"""Dynamic Dynamo/AOT graphs backed by static-shape TPU executables."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from typing import Any, Literal, TypedDict

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
from torch._inductor.output_code import OutputCode

SizeName = Literal["bs", "num_tokens"]


class ShapeSpec(TypedDict):
    num_tokens: int
    bs: int


_SIZE_NAMES_ATTR = "_sglang_tpu_dynamic_size_names"


# Injects metadata through compilation APIs that cannot accept it directly.
# Dispatch keys use runtime arguments; context presence permits warmup compilation.
_compile_context: ContextVar[ShapeSpec | None] = ContextVar(
    "tpu_shape_compilation", default=None
)


def mark_dynamic_size(tensor: torch.Tensor, dim: int, name: SizeName) -> None:
    """Mark a dynamic axis as a named size to specialize in the backend.

    ``shape_specialization_backend`` uses this axis's runtime value to select
    the corresponding static-shape artifact.
    """
    torch._dynamo.mark_dynamic(tensor, dim)
    size_names = dict(getattr(tensor, _SIZE_NAMES_ATTR, {}))
    size_names[dim] = name
    setattr(tensor, _SIZE_NAMES_ATTR, size_names)


def _named_size_expressions(
    graph_module: torch.fx.GraphModule,
) -> dict[SizeName, tuple[Any, ...]]:
    """Recover named symbolic sizes from Dynamo's original graph arguments."""
    expressions: dict[SizeName, list[Any]] = {}
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        graph_arg = node.meta.get("grapharg")
        # GraphArg retains the original tensor; FakeTensor does not copy
        # arbitrary attributes set alongside mark_dynamic().
        tensor = getattr(graph_arg, "example", None)
        if not isinstance(tensor, torch.Tensor):
            continue
        fake_tensor = getattr(graph_arg, "fake_tensor", None)
        if fake_tensor is None:
            continue
        for dim, name in getattr(tensor, _SIZE_NAMES_ATTR, {}).items():
            size = fake_tensor.shape[dim]
            if not isinstance(size, torch.SymInt):
                continue
            expression = size.node.expr
            named_expressions = expressions.setdefault(name, [])
            if expression not in named_expressions:
                named_expressions.append(expression)
    return {name: tuple(values) for name, values in expressions.items()}


def shape_specialization_backend(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    **_: Any,
) -> Callable[..., Any]:
    """Pair dynamic Dynamo capture with static TPU shape specializations.

    Dynamo captures explicitly marked dimensions symbolically. The resulting
    dynamic AOTAutograd graph is compiled into one static TPU artifact for each
    concrete ``ShapeSpec``, which always specifies both ``num_tokens`` and
    ``bs`` even when they are equal or otherwise related.
    """
    shape_spec = _compile_context.get()
    if shape_spec is None:
        raise RuntimeError(
            "shape_specialization_backend requires a _compile_context to be set"
        )
    from torch_tpu._internal.compile import compiler as tpu_compiler
    from torch_tpu._internal.compile import split_compiler

    compiler = split_compiler.SplitCompiler(tpu_compiler.StaticCompiler())
    compiler.execute_pre_grad_passes(graph_module)
    from torch_tpu._internal.compile._backend import _serialization_context

    size_expressions = _named_size_expressions(graph_module)

    def make_dispatcher(
        is_fwd: bool,
        post_aot_graph: torch.fx.GraphModule,
        inputs: Sequence[Any],
    ) -> _ArtifactDispatcher:
        symints = [
            (index, value.node.expr)
            for index, value in enumerate(inputs)
            if isinstance(value, torch.SymInt)
        ]
        # Record which boxed AOT argument carries each named size so
        # _ArtifactDispatcher can look up the specialization key at runtime.
        dimension_sources = []
        for size_name in shape_spec:
            expressions = size_expressions.get(size_name)
            if not expressions:
                raise RuntimeError(
                    f"No symbolic source was marked for dynamic size {size_name}"
                )
            arg_index = next(
                (
                    index
                    for expression in expressions
                    for index, expr in symints
                    if expr == expression
                ),
                None,
            )
            if arg_index is None:
                raise RuntimeError(
                    f"AOTAutograd did not preserve dynamic size {size_name} "
                    f"({expressions}) as a boxed argument"
                )
            dimension_sources.append((size_name, arg_index))
        return _ArtifactDispatcher(
            compiler,
            post_aot_graph,
            is_fwd,
            tuple(dimension_sources),
        )

    with _serialization_context(True) as captured_entry:
        result = aot_autograd(
            fw_compiler=SerializableAOTDispatchCompiler(
                _ArtifactDispatcher, functools.partial(make_dispatcher, True)
            ),
            bw_compiler=functools.partial(make_dispatcher, False),
            keep_inference_input_mutations=False,
        )(graph_module, example_inputs)

    if captured_entry[0] is not None:
        entry = captured_entry[0]
        result.serialize = lambda: entry
    return result


class _DoneCompiling(Exception):
    """Stop a warmup call after its artifact is compiled.

    torch.compile normally invokes the callable returned by its backend as part
    of the original forward. This unwinds that forward without executing the
    newly compiled artifact.
    """


class _ArtifactDispatcher(OutputCode):
    """Dispatch dynamic dimension values to compiled static artifacts.

    ``__call__`` lazily compiles a missing specialization, such as
    ``num_tokens=128, bs=4``.
    """

    _boxed_call = True

    def __init__(
        self,
        compiler: Any,
        graph_module: torch.fx.GraphModule,
        is_fwd: bool,
        dimension_sources: tuple[tuple[SizeName, int], ...],
        artifacts: dict[tuple[int, ...], Any] | None = None,
    ) -> None:
        self._compiler = compiler
        self._graph_module = graph_module
        self._is_fwd = is_fwd
        self._dimension_sources = dimension_sources
        self._artifacts = artifacts if artifacts is not None else {}

    def __call__(self, args: Sequence[Any]) -> Any:
        key = tuple(int(args[source]) for _, source in self._dimension_sources)
        shape_spec = _compile_context.get()

        artifact = self._artifacts.get(key)
        if artifact is not None:
            if shape_spec is not None:
                raise _DoneCompiling()
            return artifact(*args)

        if shape_spec is None:
            raise RuntimeError(
                "You triggered a recompile during server serving phase, this is an "
                "error. All compilations should have finished during server warmup"
            )
        artifact = self._compiler(self._graph_module, args, self._is_fwd)
        self._artifacts[key] = artifact
        raise _DoneCompiling()

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        return (
            _ArtifactDispatcher,
            (
                self._compiler,
                self._graph_module,
                self._is_fwd,
                self._dimension_sources,
                self._artifacts,
            ),
        )

    # OutputCode requires these hooks; TPU artifacts need no post-processing.
    def prepare_for_serialization(self) -> None:
        pass

    def post_compile(
        self, _example_inputs: Sequence[Any], _constants: Any, _graph_kwargs: Any
    ) -> None:
        pass

    def set_triton_bundle(self, _triton_bundle: Any) -> None:
        pass


@contextlib.contextmanager
def delay_aot_cache_writes() -> Iterator[None]:
    """Delay AOT cache writes until dispatchers contain all warmed artifacts."""
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

    writes = []
    original_save = AOTAutogradCache.save

    def delay_save(key: str, entry: Any, remote: bool) -> None:
        writes.append((key, entry, remote))

    AOTAutogradCache.save = staticmethod(delay_save)
    try:
        yield
    finally:
        AOTAutogradCache.save = staticmethod(original_save)
    for key, entry, remote in writes:
        original_save(key, entry, remote)
