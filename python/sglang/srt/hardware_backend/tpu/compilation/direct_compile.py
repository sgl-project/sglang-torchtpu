"""Direct TorchTPU compilation without Dynamo or AOTAutograd."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Hashable
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch
from jax import tree_util
from torch._dynamo.utils import dynamo_timed
from torch.nn.utils.stateless import _reparametrize_module
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    PallasForwardMetadata,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

SpecializationKey = Callable[..., Hashable]
_STATIC_INPUT_TYPES = (bool, int, float, str, Enum)


class KVPool(Protocol):
    """Expose mutable KV-cache tensors for direct compilation.

    ``get_state`` returns every Tensor read through the pool. ``set_state``
    installs a matching list. The order and length must remain fixed.
    """

    def get_state(self) -> list[torch.Tensor]: ...

    def set_state(self, state: list[torch.Tensor]) -> None: ...


# These objects cross the compiled decoder boundary. Registering them lets JAX
# flatten their tensor fields and reconstruct the same objects for tracing.
tree_util.register_dataclass(ForwardBatch)
tree_util.register_dataclass(PallasForwardMetadata)
tree_util.register_dataclass(LogitsProcessorOutput)


def direct_compile(
    fn: Callable[..., Any],
    *,
    specialization_key: SpecializationKey,
    model: torch.nn.Module | None = None,
    kv_pool: KVPool | None = None,
    ignored_argnames: tuple[str, ...] = (),
) -> CompiledFunction:
    """Compile and dispatch ``fn`` directly, without torch.compile's
    Dynamo or AOTAutograd.

    The programming model is similar to JAX tracing:
    `direct_compile` traces out one straightline graph of Tensor operations.
    Any non-Tensor values are baked directly into the graph; all Tensor
    values are graph inputs.
    Then, it lowers it to MLIR and then compiles it with XLA/PJRT.

    This API is specialized for SGLang inference. A more general API adds
    complexity and can be added when needed.

    `specialization_key` is called with `fn`'s args and kwargs and determines
    whether `direct_compile` can reuse an existing artifact. Unlike `jax.jit`,
    the caller supplies this key because computing a generic key can be
    expensive. Use `torch.compile` if you want automatic specialization.

    Caveat: like `jax.jit`, `direct_compile` does not normalize inputs. Calling
    `fn` with an equivalent but differently structured `(args, kwargs)` creates
    a different specialization. Model warmup, where compilation occurs, must
    therefore call `fn` exactly as model serving does.

    `ignored_argnames` names arguments that are not captured as graph inputs.
    This is useful for unused Tensors whose shapes would otherwise cause
    unexpected specializations.

    In SGLang, `fn` is commonly a model forward function that captures Tensors
    that are not explicitly inputs to the function. The only two allowed
    sources are `model` and `kv_pool`; passing them through `direct_compile`
    lifts all of their Tensors into graph inputs.

    `model` assumes all Tensor state is registered as parameters or buffers and
    that their values do not change during inference. `kv_pool` exposes every
    KV-cache Tensor through `get_state`; all are assumed to be mutated and are
    copied back after execution. The KV pool is the only mutable input.

    Common problems/assumptions:

    * Every Tensor reachable from model execution must be an explicit function
      input, a model parameter/buffer, or KV-pool state. An ``nn.Module`` must
      not keep a Tensor as an ordinary attribute; register such Tensors with
      ``register_buffer(persistent=False)``.
    * Registered aliases are supported, including a parameter also registered
      as a non-persistent buffer. Aliases share one graph input and remain tied
      while tracing.
    * A plain helper object owned by a module is not traversed by
      ``named_parameters``/``named_buffers``. These objects must not own Tensor
      attributes; the easiest workaround is to make the Tensor a buffer
      of the module and have the code use the buffer through accessing
      module attributes.
    * ``build_mlir`` reporting ``identified an argument that was not provided:
      DeviceBufferRef`` means tracing captured such an unregistered Tensor.
      The reported shape and dtype are usually the fastest way to find the
      offending model/helper attribute.

    """
    compiled = CompiledFunction(
        fn,
        specialization_key=specialization_key,
        model=model,
        kv_pool=kv_pool,
        ignored_argnames=ignored_argnames,
    )
    return functools.update_wrapper(compiled, fn)


class CompiledFunction:
    """A CompiledFunction is a collection of 0+ CompiledArtifacts.

    It uses specialization_key to determine if it can reuse an existing
    CompiledArtifact.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        specialization_key: SpecializationKey,
        model: torch.nn.Module | None = None,
        kv_pool: KVPool | None = None,
        ignored_argnames: tuple[str, ...] = (),
    ) -> None:
        self.fn = fn
        self.specialization_key = specialization_key
        self.cache: dict[Hashable, _CompiledArtifact] = {}
        self._signature = inspect.signature(fn)
        self._model = model
        self._kv_pool = kv_pool
        self._ignored_argnames = ignored_argnames

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.specialization_key(*args, **kwargs)
        artifact = self.cache.get(key)
        if artifact is None:
            artifact = self._compile(args, kwargs)
            self.cache[key] = artifact
        return artifact((args, kwargs))

    def _compile(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> _CompiledArtifact:
        """Compile one specialization into a PJRT executable."""

        # Step 1: Figure out a flat list of Tensor inputs.
        call = (args, kwargs)
        # Yes, use jax's tree_util API. jax is required for pallas anyways
        # and their tree_util API is pretty nice.
        path_leaves, input_treespec = tree_util.tree_flatten_with_path(call)
        ignored_paths = tuple(
            _selector_path(self._signature, args, kwargs, selector)
            for selector in self._ignored_argnames
        )
        model_names: tuple[str, ...] = ()
        model_inputs: list[torch.Tensor] = []
        if self._model is not None:
            all_names_and_tensors = [
                *self._model.named_parameters(remove_duplicate=False),
                *self._model.named_buffers(remove_duplicate=False),
            ]
            seen_tensor_ids: set[int] = set()
            names_and_tensors = []
            for name, tensor in all_names_and_tensors:
                tensor_id = id(tensor)
                if tensor_id in seen_tensor_ids:
                    continue
                seen_tensor_ids.add(tensor_id)
                names_and_tensors.append((name, tensor))
            model_names = tuple(name for name, _ in names_and_tensors)
            model_inputs.extend(tensor for _, tensor in names_and_tensors)

        kv_pool_inputs = self._kv_pool.get_state() if self._kv_pool is not None else []

        tensor_leaf_indices: list[int] = []
        tensor_inputs_key_paths: list[tree_util.KeyPath] = []
        for leaf_index, (path, leaf) in enumerate(path_leaves):
            # We need to subtract out the ignored_argnames.
            if any(path[: len(prefix)] == prefix for prefix in ignored_paths):
                continue
            if _is_tpu_tensor(leaf):
                tensor_leaf_indices.append(leaf_index)
                tensor_inputs_key_paths.append(path)
            elif isinstance(leaf, _STATIC_INPUT_TYPES):
                continue
            else:
                raise TypeError(
                    f"Unsupported direct_compile input type "
                    f"{type(leaf).__name__} at {tree_util.keystr(path)}; register structured "
                    "inputs as JAX pytrees"
                )

        model_input_count = len(model_inputs)
        kv_pool_input_start = model_input_count
        kv_pool_input_stop = kv_pool_input_start + len(kv_pool_inputs)
        flat_inputs = [*model_inputs, *kv_pool_inputs]
        flat_inputs.extend(path_leaves[index][1] for index in tensor_leaf_indices)

        # Step 2: We've figured out all of our args. Now, we can create
        # placeholders for them. These will all be inputs to the MLIR graph.
        graph_inputs = [tpu_torch_compile.placeholder_like(t) for t in flat_inputs]
        trace_inputs = list(graph_inputs)
        trace_leaves = [leaf for _, leaf in path_leaves]

        # Clone mutated KV inputs; XLA will optimize these away.
        # The strategy for these is that during runtime, we will copy_ the
        # updated values back into the original Tensors. copy_ doesn't launch a
        # kernel, it just swizzles some metadata around.
        for input_index in range(kv_pool_input_start, kv_pool_input_stop):
            trace_inputs[input_index] = graph_inputs[input_index].clone()
        kv_pool_trace_inputs = trace_inputs[kv_pool_input_start:kv_pool_input_stop]
        for input_index, leaf_index in enumerate(
            tensor_leaf_indices, start=kv_pool_input_stop
        ):
            trace_leaves[leaf_index] = trace_inputs[input_index]

        # Step 3: Lower to MLIR via literally running the function against the
        # placeholders. The goal is to get a flat list of output Tensors.
        with (
            dynamo_timed("direct_compile_trace"),
            execution_mode.set_eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL),
            ExitStack() as stack,
        ):
            # Respect torch.compiler.is_compiling() during tracing by returning True.
            previous_is_compiling = torch.compiler._is_compiling_flag
            torch.compiler._is_compiling_flag = True
            stack.callback(
                setattr,
                torch.compiler,
                "_is_compiling_flag",
                previous_is_compiling,
            )
            if self._model is not None:
                parameters_and_buffers = dict(
                    zip(model_names, graph_inputs[:model_input_count])
                )
                # Need to temporarily yeet the placeholder params/buffers onto
                # the nn.Modules
                # Restore in LIFO order so shared module aliases do not retain
                # placeholders.
                stack.enter_context(
                    _reparametrize_module(
                        self._model,
                        parameters_and_buffers,
                        tie_weights=True,
                        stack_weights=True,
                    )
                )
            if self._kv_pool is not None:
                previous_state = self._kv_pool.get_state()
                self._kv_pool.set_state(kv_pool_trace_inputs)
                stack.callback(self._kv_pool.set_state, previous_state)
            trace_arguments = tree_util.tree_unflatten(input_treespec, trace_leaves)
            outputs = self.fn(*trace_arguments[0], **trace_arguments[1])

        output_leaves, output_treespec = tree_util.tree_flatten(outputs)
        if not output_leaves or not all(_is_tpu_tensor(leaf) for leaf in output_leaves):
            raise TypeError("All direct_compile output leaves must be TPU tensors")
        result_tensors = output_leaves + kv_pool_trace_inputs

        # Step 4: Finish lowering to MLIR, then compile it.
        logger.info(
            "[direct_compile] lowering %d inputs, %d outputs, and %d mutations",
            len(graph_inputs),
            len(output_leaves),
            len(kv_pool_inputs),
        )
        with dynamo_timed("direct_compile_build_mlir"):
            module = tpu_torch_compile.build_mlir(result_tensors, graph_inputs)
        with dynamo_timed("direct_compile_compile_mlir"):
            executable = tpu_torch_compile.compile_mlir(module)
        logger.info("[direct_compile] compilation finished")
        return _CompiledArtifact(
            executable=executable,
            model_inputs=tuple(model_inputs),
            kv_pool=self._kv_pool,
            tensor_inputs_key_paths=tuple(tensor_inputs_key_paths),
            output_treespec=output_treespec,
            output_count=len(output_leaves),
        )


@dataclass(frozen=True)
class _CompiledArtifact:
    """Wrap a TorchTPU PJRT executable with input/output/mutation handling."""

    executable: Any
    model_inputs: tuple[torch.Tensor, ...]
    kv_pool: KVPool | None
    tensor_inputs_key_paths: tuple[tree_util.KeyPath, ...]
    output_treespec: Any
    output_count: int

    def __call__(self, call: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
        kv_pool_inputs = self.kv_pool.get_state() if self.kv_pool is not None else []
        inputs = [*self.model_inputs, *kv_pool_inputs]
        # Resolve explicit inputs by path, similar to Dynamo sources.
        inputs.extend(_get_path(call, path) for path in self.tensor_inputs_key_paths)

        outputs = tpu_torch_compile.execute(self.executable, inputs)

        # Copy updated KV state back into the live pool tensors.
        with torch.no_grad():
            for tensor, replacement in zip(
                kv_pool_inputs, outputs[self.output_count :]
            ):
                tensor.copy_(replacement)

        return tree_util.tree_unflatten(
            self.output_treespec, outputs[: self.output_count]
        )


def _get_path(value: Any, path: tree_util.KeyPath) -> Any:
    """Resolve a JAX key path."""
    for key in path:
        if isinstance(key, tree_util.SequenceKey):
            value = value[key.idx]
        elif isinstance(key, tree_util.DictKey):
            value = value[key.key]
        elif isinstance(key, tree_util.GetAttrKey):
            value = getattr(value, key.name)
        else:
            raise TypeError(f"Unsupported JAX key type: {type(key).__name__}")
    return value


def _selector_path(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    selector: str,
) -> tree_util.KeyPath:
    """Map an ignored or mutated selector to a path through ``(args, kwargs)``.

    For ``fn(value, batch)``, ``batch.seq_lens_cpu`` resolves to
    ``(args, kwargs)[0][1].seq_lens_cpu`` when positional, or
    ``(args, kwargs)[1]["batch"].seq_lens_cpu`` when passed by keyword.
    """
    root, *attributes = selector.split(".")
    positional_names = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if root in kwargs:
        path = (tree_util.SequenceKey(1), tree_util.DictKey(root))
    elif root in positional_names and positional_names.index(root) < len(args):
        path = (
            tree_util.SequenceKey(0),
            tree_util.SequenceKey(positional_names.index(root)),
        )
    else:
        raise ValueError(f"Argument selector {selector!r} is not explicitly supplied")
    return (*path, *(tree_util.GetAttrKey(name) for name in attributes))


def _is_tpu_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "tpu"
