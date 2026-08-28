"""Direct TorchTPU compilation without Dynamo or AOTAutograd."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Hashable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Protocol

import torch
from jax import tree_util
from torch._dynamo import eval_frame
from torch._dynamo.utils import dynamo_timed
from torch.nn.utils.stateless import _reparametrize_module
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    PallasForwardMetadata,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
)

logger = logging.getLogger(__name__)

SpecializationKey = Callable[..., Hashable]
_STATIC_INPUT_TYPES = (bool, int, float, str, Enum)


def _resolve_dotted_attr(root: object, path: str) -> tuple[object, str, object]:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(
            f"model_extra_tensor_attrs contains invalid dotted path {path!r}"
        )

    parent = root
    for part in parts[:-1]:
        try:
            parent = getattr(parent, part)
        except AttributeError as error:
            raise AttributeError(
                f"model_extra_tensor_attrs path {path!r} has no component {part!r}"
            ) from error
    attr_name = parts[-1]
    try:
        value = getattr(parent, attr_name)
    except AttributeError as error:
        raise AttributeError(
            f"model_extra_tensor_attrs path {path!r} has no attribute {attr_name!r}"
        ) from error
    return parent, attr_name, value


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
    model_extra_tensor_attrs: Sequence[str] = (),
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

    `direct_compile` needs to know where all the Tensors in `model` are in
    order to make all of them inputs to the captured graph.
    It knows how to find Tensors that are registered parameters and buffers.
    Please pass paths to non-param non-buffer Tensor attributes via
    `model_extra_tensor_attrs` to tell `direct_compile` where to find those.
    `direct_compile` assumes that the values of Tensors on the model do not
    change during inference.

    `kv_pool` exposes every KV-cache Tensor through `get_state`; all of
    them are assumed to be mutated during inference and are copied back
    after execution. The KV pool is the only mutable input.

    Common problems/assumptions:

    * Every Tensor reachable from model execution must be an explicit function
      input, model parameter/buffer, `model_extra_tensor_attrs` entry, or
      KV-pool state.
    * Registered aliases are supported, including a parameter also registered
      as a non-persistent buffer. Aliases share one graph input and remain tied
      while tracing.
    * A plain helper object owned by a module is not traversed by
      ``named_parameters``/``named_buffers``. Please list all Tensor attributes
      via `model_extra_tensor_attrs`.
    * ``build_mlir`` reporting ``identified an argument that was not provided:
      DeviceBufferRef`` means tracing captured such an unregistered Tensor.
      The reported shape and dtype are usually the fastest way to find the
      offending model/helper attribute.
    * ``Unsupported direct_compile input type ...; register structured inputs
      as JAX pytrees`` means an input container must be registered before its
      Tensor fields can become graph inputs.
    * A compiled artifact requires the same input pytree at warmup and runtime,
      including which fields are Tensors versus ``None``. Either exclude
      host-only or preprocessing-only fields with ``ignored_argnames``, or use
      pytree tricks to hide them; only pytree leaves become graph inputs.

    """
    compiled = CompiledFunction(
        fn,
        specialization_key=specialization_key,
        model=model,
        model_extra_tensor_attrs=model_extra_tensor_attrs,
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
        model_extra_tensor_attrs: Sequence[str] = (),
        kv_pool: KVPool | None = None,
        ignored_argnames: tuple[str, ...] = (),
    ) -> None:
        self.fn = fn
        self.specialization_key = specialization_key
        self.cache: dict[Hashable, _CompiledArtifact] = {}
        self._signature = inspect.signature(fn)
        self._model = model
        self._model_extra_tensor_attrs = tuple(model_extra_tensor_attrs)
        if self._model is None and self._model_extra_tensor_attrs:
            raise ValueError("model_extra_tensor_attrs requires model")
        if len(set(self._model_extra_tensor_attrs)) != len(
            self._model_extra_tensor_attrs
        ):
            raise ValueError("model_extra_tensor_attrs contains duplicate paths")
        self._kv_pool = kv_pool
        self._ignored_argnames = ignored_argnames

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.specialization_key(*args, **kwargs)
        artifact = self.cache.get(key)
        if artifact is None:
            # set_stance has no public getter and replaces eval_frame._stance.
            if self.cache and eval_frame._stance.stance == "fail_on_recompile":
                raise RuntimeError(
                    "Detected recompile when torch.compile stance is "
                    "'fail_on_recompile'. direct_compile specialization key: "
                    f"{key!r}"
                )
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
        registered_model_inputs: list[tuple[str, int]] = []
        extra_model_inputs: list[tuple[object, str, int]] = []
        model_inputs: list[torch.Tensor] = []
        if self._model is not None:
            all_names_and_tensors = [
                *self._model.named_parameters(remove_duplicate=False),
                *self._model.named_buffers(remove_duplicate=False),
            ]
            registered_names = {name for name, _ in all_names_and_tensors}
            tensor_indices: dict[int, int] = {}
            for name, tensor in all_names_and_tensors:
                tensor_id = id(tensor)
                if tensor_id in tensor_indices:
                    continue
                tensor_index = len(model_inputs)
                tensor_indices[tensor_id] = tensor_index
                model_inputs.append(tensor)
                registered_model_inputs.append((name, tensor_index))

            for path in self._model_extra_tensor_attrs:
                if path in registered_names:
                    raise ValueError(
                        f"model_extra_tensor_attrs path {path!r} is already a "
                        "registered parameter or buffer"
                    )
                parent, attr_name, tensor = _resolve_dotted_attr(self._model, path)
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"model_extra_tensor_attrs path {path!r} resolved to "
                        f"{type(tensor).__name__}, not Tensor"
                    )
                tensor_id = id(tensor)
                tensor_index = tensor_indices.get(tensor_id)
                if tensor_index is None:
                    tensor_index = len(model_inputs)
                    tensor_indices[tensor_id] = tensor_index
                    model_inputs.append(tensor)
                extra_model_inputs.append((parent, attr_name, tensor_index))

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
                parameters_and_buffers = {
                    name: graph_inputs[input_index]
                    for name, input_index in registered_model_inputs
                }
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
                # We can't pass these directly to _reparametrize_module because
                # it requires every intermediate dotted-path component to be an
                # nn.Module. Explicit extras may live on ordinary helper objects,
                # so swap those attributes directly and restore them before
                # registered model state.
                for parent, attr_name, input_index in extra_model_inputs:
                    previous_value = getattr(parent, attr_name)
                    setattr(parent, attr_name, graph_inputs[input_index])
                    stack.callback(setattr, parent, attr_name, previous_value)
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
        with dynamo_timed("direct_compile_traverse_and_compile"):
            compile_result = tpu_torch_compile.traverse_and_compile(
                result_tensors, graph_inputs
            )
        logger.info("[direct_compile] compilation finished")
        return _CompiledArtifact(
            executable=compile_result.executable,
            model_inputs=tuple(model_inputs),
            kv_pool=self._kv_pool,
            tensor_inputs_key_paths=tuple(tensor_inputs_key_paths),
            tensor_inputs_traced_shapes=tuple(
                tuple(path_leaves[index][1].shape) for index in tensor_leaf_indices
            ),
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
    tensor_inputs_traced_shapes: tuple[tuple[int, ...], ...]
    output_treespec: Any
    output_count: int

    def __call__(self, call: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
        kv_pool_inputs = self.kv_pool.get_state() if self.kv_pool is not None else []
        inputs = [*self.model_inputs, *kv_pool_inputs]
        # Resolve explicit inputs by path, similar to Dynamo sources.
        for path, traced_shape in zip(
            self.tensor_inputs_key_paths, self.tensor_inputs_traced_shapes
        ):
            value = _get_path(call, path)
            if not _is_tpu_tensor(value):
                raise TypeError(
                    "A direct_compile tensor input changed type at runtime: "
                    f"{tree_util.keystr(path)} was traced as a TPU tensor but is "
                    f"now {type(value).__name__}"
                )
            if tuple(value.shape) != traced_shape:
                # A mismatched buffer corrupts the launch (XLA e0102): the
                # specialization key failed to distinguish this call.
                raise ValueError(
                    "A direct_compile tensor input changed shape at runtime: "
                    f"{tree_util.keystr(path)} was traced as {traced_shape} but "
                    f"is now {tuple(value.shape)} ({value.dtype}). The "
                    "specialization key does not cover whatever varies here."
                )
            inputs.append(value)

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


def _register_hidden_states_pytree(cls: type) -> None:
    """Only expose hidden_states for draft model compilation.

    All other fields are unused by the compiled draft model forward.
    """
    field_names = tuple(field.name for field in fields(cls))
    tree_util.register_dataclass(
        cls,
        data_fields=("hidden_states",),
        meta_fields=(),
        drop_fields=tuple(name for name in field_names if name != "hidden_states"),
    )


_register_hidden_states_pytree(EagleDraftInput)
_register_hidden_states_pytree(EagleDraftExtendInput)
_register_hidden_states_pytree(DFlashDraftInputV2)
tree_util.register_dataclass(DFlashVerifyInput)
