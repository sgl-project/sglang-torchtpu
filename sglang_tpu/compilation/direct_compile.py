"""Direct TorchTPU compilation without Dynamo or AOTAutograd."""

from __future__ import annotations

import functools
import inspect
import logging
import pickle
import time
from collections.abc import Callable, Hashable, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Iterator, Protocol

import torch
from jax import tree_util
from torch._dynamo import eval_frame
from torch._dynamo.utils import dynamo_timed
from torch.nn.utils.stateless import _reparametrize_module
from torch_tpu._internal import execution_mode, sync
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.sync.sync import synchronize

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
)
from sglang_tpu.attention_pallas_rpa_base import (
    PallasForwardMetadata,
)
from sglang_tpu.compilation.graph_identity import _key_value, canonical, digest
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.pp_proxy_pytree import (
    register_pp_proxy_pytree,
)
from sglang_tpu.compilation.spmd_compiler import (
    ExecutableInterface,
    InputSpec,
    SPMDCompiler,
)
from sglang_tpu.compilation.warmup_queue import current_warmup_queue
from sglang_tpu.rank_local import trace_rank

logger = logging.getLogger(__name__)

CallbackFactory = Callable[[torch.nn.Module | None], Callable[[torch.Tensor], None]]


@dataclass(frozen=True)
class _InsertedCallback:
    tensor: torch.Tensor
    callback: Callable[[torch.Tensor], None]
    callback_factory: CallbackFactory | None


_ACTIVE_CALLBACKS: ContextVar[list[_InsertedCallback] | None] = ContextVar(
    "tpu_direct_compile_callbacks", default=None
)


@contextmanager
def _collect_callbacks() -> Iterator[list[_InsertedCallback]]:
    callbacks: list[_InsertedCallback] = []
    token = _ACTIVE_CALLBACKS.set(callbacks)
    try:
        yield callbacks
    finally:
        _ACTIVE_CALLBACKS.reset(token)


def insert_callback(
    tensor: torch.Tensor,
    callback: Callable[[torch.Tensor], None],
    *,
    callback_factory: CallbackFactory | None = None,
) -> bool:
    """Register a callback for a tensor produced during direct compilation.

    This is intended for module forward hooks::

        def hook(_module, _args, output):
            if not insert_callback(output, consume_output):
                raise RuntimeError("direct compilation is not active")

    Returns ``True`` while a direct-compile trace is collecting callbacks.
    Otherwise returns ``False`` without invoking ``callback`` eagerly.

    Shared compilation requires ``callback_factory(model)`` to bind a callback
    to each rank's local model without tracing it. The factory must be picklable
    (a module-level function or partial) and capture only tensor-free,
    rank-independent metadata, never a model or recorder instance.
    """
    callbacks = _ACTIVE_CALLBACKS.get()
    if callbacks is None:
        return False
    callbacks.append(_InsertedCallback(tensor, callback, callback_factory))
    return True


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
# Pipeline stages exchange PPProxyTensors as the compiled input and output.
register_pp_proxy_pytree(tree_util)
# EAGLE3 targets return their aux hidden states as this list subclass, which
# tree_util would otherwise treat as one opaque leaf.
tree_util.register_pytree_node(
    AuxHiddenStateList,
    lambda states: (list(states), None),
    lambda _, children: AuxHiddenStateList(children),
)


def direct_compile(
    fn: Callable[..., Any],
    *,
    specialization_key: SpecializationKey,
    lift_rank: bool = False,
    name: str | None = None,
    spmd_safe: bool = False,
    model: torch.nn.Module | None = None,
    model_extra_tensor_attrs: Sequence[str] = (),
    kv_pool: KVPool | None = None,
    ignored_argnames: tuple[str, ...] = (),
    prune_unused_inputs: bool = False,
    uses_rng: bool = False,
) -> CompiledFunction:
    """Compile and dispatch ``fn`` directly, without torch.compile's
    Dynamo or AOTAutograd.

    The programming model is similar to JAX tracing:
    `direct_compile` traces out one straightline graph of Tensor operations.
    Any non-Tensor values are baked directly into the graph; all Tensor
    values are graph inputs.
    Then, it lowers it to MLIR and then compiles it with XLA/PJRT.
    ``lift_rank`` also passes the logical distributed rank as a tensor input,
    so rank-local operations can select their data without specializing the graph.

    ``name`` identifies the compilation boundary independently of rank, such as
    ``"target/model.forward"``. It is required with ``spmd_safe=True`` and combines
    with the complete ``specialization_key`` to identify a graph across ranks.
    ``spmd_safe`` declares eligibility; it neither verifies nor enables sharing.
    Enable coordination with ``--use-spmd-compile`` and add a recipe test using
    ``sglang_tpu.testing.spmd_safe`` to validate identical graphs and input layouts
    on every rank. ``VERIFY_SPMD_SAFE=1`` runs that verification without execution.

    ``CompiledFunction.warmup_async`` submits native asynchronous compilation.
    Its returned ``PendingCall`` executes the warmup when ``wait`` is called.
    ``CompiledFunction.precompile`` only compiles; waiting on its result never
    executes the graph or retains its example runtime inputs.

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

    `prune_unused_inputs` excludes model tensors unused by a specialization
    from its executable and runtime arguments. It defaults to False, which
    retains all model tensors and skips pruning's MLIR inspection and extra
    lowerings. This option is fixed when constructing the compiled function; explicit
    function inputs and KV-pool state are never pruned.

    ``uses_rng`` lifts the default TPU generator state into an input/output pair.
    Tracing leaves the serving generator unchanged; execution advances it.

    `kv_pool` exposes every KV-cache Tensor through `get_state`; all of
    them are assumed to be mutated during inference and are copied back
    after execution. Explicit function inputs must be immutable.

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
        lift_rank=lift_rank,
        name=name,
        spmd_safe=spmd_safe,
        model=model,
        model_extra_tensor_attrs=model_extra_tensor_attrs,
        kv_pool=kv_pool,
        ignored_argnames=ignored_argnames,
        prune_unused_inputs=prune_unused_inputs,
        uses_rng=uses_rng,
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
        lift_rank: bool = False,
        name: str | None = None,
        spmd_safe: bool = False,
        model: torch.nn.Module | None = None,
        model_extra_tensor_attrs: Sequence[str] = (),
        kv_pool: KVPool | None = None,
        ignored_argnames: tuple[str, ...] = (),
        prune_unused_inputs: bool = False,
        uses_rng: bool = False,
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
        self._prune_unused_inputs = prune_unused_inputs
        self._uses_rng = uses_rng
        if spmd_safe and name is None:
            raise ValueError("spmd_safe=True requires passing name= to direct_compile")
        self._name = name
        self._spmd_safe = spmd_safe
        self._lift_rank = lift_rank
        self._rank_input: torch.Tensor | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        artifact = self._get_artifact(args, kwargs)
        return artifact((args, kwargs))

    def warmup_async(self, *args: Any, **kwargs: Any) -> PendingCall:
        queue = current_warmup_queue()
        if self._spmd_safe and queue is not None and queue.spmd_compiler is not None:
            # Snapshot reused containers while retaining their local tensor buffers.
            args, kwargs = tree_util.tree_map(lambda value: value, (args, kwargs))
        artifact = self._get_artifact(args, kwargs, async_compile=True)
        return PendingCall(lambda: artifact((args, kwargs)))

    def precompile(
        self, *args: Any, async_compile: bool = False, **kwargs: Any
    ) -> PendingCall:
        """Compile without executing; shared input binding completes at queue join."""
        queue = current_warmup_queue() if self._spmd_safe else None
        shared = queue is not None and queue.spmd_compiler is not None
        # Shared compilation validates real input layouts on every rank.
        if async_compile and not shared:
            tensors = [
                value
                for value in tree_util.tree_leaves((args, kwargs))
                if _is_tpu_tensor(value)
            ]
            if tensors:
                # Submit eager input preparation without waiting for buffers.
                # Symbolic arguments keep layout queries from blocking compilation.
                synchronize(tensors, wait=False)
                args, kwargs = tree_util.tree_map(
                    lambda value: (
                        tpu_torch_compile.placeholder_like(value)
                        if _is_tpu_tensor(value)
                        else value
                    ),
                    (args, kwargs),
                )
        artifact = self._get_artifact(args, kwargs, async_compile=async_compile)
        return PendingCall(artifact.wait_for_compile)

    def _get_artifact(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        async_compile: bool = False,
    ) -> _CompiledArtifact:
        key = self.specialization_key(*args, **kwargs)
        queue = current_warmup_queue() if async_compile and self._spmd_safe else None
        spmd_compiler = queue.spmd_compiler if queue is not None else None
        graph = (
            f"{self._name}:{canonical(_key_value(key))}"
            if spmd_compiler is not None
            else None
        )
        artifact = self.cache.get(key)
        if spmd_compiler is not None:
            pending = (
                spmd_compiler
                if spmd_compiler.verify_spmd_safe
                else spmd_compiler.submissions.get(graph)
            )
            if (
                pending is None
                or artifact is None
                or artifact.pending_compile is not pending
            ):
                # Each queue must cover every key even if local caches differ across ranks.
                artifact = None
        if artifact is None:
            # set_stance has no public getter and replaces eval_frame._stance.
            if (
                spmd_compiler is None
                and self.cache
                and eval_frame._stance.stance == "fail_on_recompile"
            ):
                raise RuntimeError(
                    "Detected recompile when torch.compile stance is "
                    "'fail_on_recompile'. direct_compile specialization key: "
                    f"{key!r}"
                )
            logger.info(
                "[direct_compile] new specialization %r (async=%s)", key, async_compile
            )
            if spmd_compiler is not None:
                spmd_compiler.enqueue(graph)
            artifact = self._compile(
                args,
                kwargs,
                async_compile=async_compile,
                spmd_compiler=spmd_compiler,
                graph=graph,
            )
            self.cache[key] = artifact
        return artifact

    def _compile(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        async_compile: bool = False,
        spmd_compiler: SPMDCompiler | None = None,
        graph: str | None = None,
    ) -> _CompiledArtifact:
        """Compile one specialization into a PJRT executable."""

        # Step 1: Figure out a flat list of Tensor inputs.
        if spmd_compiler is not None and self._uses_rng:
            raise ValueError("Shared TPU compilation requires explicit RNG inputs")
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

        rank_input_index = None
        if self._lift_rank:
            if self._rank_input is None:
                self._rank_input = torch.tensor(
                    [torch.distributed.get_rank()], dtype=torch.int32, device="tpu"
                )
                # Resolve the rank buffer's physical layout before tracing it.
                self._rank_input.cpu()
            rank_input_index = len(model_inputs)
            model_inputs.append(self._rank_input)

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
        if spmd_compiler is not None:
            sync.synchronize(flat_inputs)

        binding_input_digest = None
        interface = None
        if spmd_compiler is not None and not spmd_compiler.verify_spmd_safe:
            # The digest checks that every rank assembles inputs in the same order.
            binding_input_digest = digest(
                dict(
                    model=registered_model_inputs,
                    extras=[
                        (path, item[2])
                        for path, item in zip(
                            self._model_extra_tensor_attrs,
                            extra_model_inputs,
                            strict=True,
                        )
                    ],
                    rank_input=rank_input_index,
                    model_count=model_input_count,
                    kv_count=len(kv_pool_inputs),
                    tensor_paths=[
                        tree_util.keystr(path) for path in tensor_inputs_key_paths
                    ],
                )
            )
            submission = spmd_compiler.submissions[graph]
            if submission.owner != spmd_compiler.rank:
                artifact = _CompiledArtifact(
                    executable=None,
                    model_inputs=tuple(model_inputs),
                    kv_pool=self._kv_pool,
                    tensor_inputs_key_paths=tuple(tensor_inputs_key_paths),
                    tensor_inputs_traced_shapes=tuple(
                        tuple(path_leaves[index][1].shape)
                        for index in tensor_leaf_indices
                    ),
                    output_treespec=None,
                    output_count=0,
                    pending_compile=submission,
                )

                def bind(owner_interface: ExecutableInterface) -> None:
                    if owner_interface.input_digest != binding_input_digest:
                        raise ValueError(
                            f"Shared TPU input binding changed: graph={graph}, "
                            f"owner={submission.owner}, rank={spmd_compiler.rank}, "
                            f"owner_digest={owner_interface.input_digest}, "
                            f"local_digest={binding_input_digest}"
                        )
                    live_model_inputs = tuple(
                        model_inputs[index] for index in owner_interface.model_indices
                    )
                    selected = [*live_model_inputs, *flat_inputs[model_input_count:]]
                    layouts = []
                    for tensor in selected:
                        layout = None
                        if tensor.device.type == "tpu":
                            layout = (
                                tpu_torch_compile.get_device_layout_if_materialized(
                                    tensor
                                )
                            )
                        layouts.append(layout[0] if layout is not None else [])
                    local_specs = _tensor_specs(selected)
                    if local_specs != owner_interface.input_specs:
                        raise ValueError(
                            f"Shared TPU input views changed: graph={graph}, "
                            f"owner={submission.owner}, rank={spmd_compiler.rank}, "
                            f"owner_specs={owner_interface.input_specs}, "
                            f"local_specs={local_specs}"
                        )
                    if layouts != owner_interface.argument_layouts:
                        raise ValueError(
                            f"Shared TPU input layouts changed: graph={graph}, "
                            f"owner={submission.owner}, rank={spmd_compiler.rank}, "
                            f"owner_layouts={owner_interface.argument_layouts}, "
                            f"local_layouts={layouts}"
                        )
                    artifact.model_inputs = live_model_inputs
                    artifact.output_treespec = owner_interface.output_treespec
                    artifact.output_count = owner_interface.output_treespec.num_leaves
                    artifact.callbacks = tuple(
                        factory(self._model)
                        for factory in pickle.loads(owner_interface.callback_factories)
                    )

                submission.bind = bind
                return artifact

        # Step 2: We've figured out all of our args. Now, we can create
        # placeholders for them. These will all be inputs to the MLIR graph.
        graph_inputs = [tpu_torch_compile.placeholder_like(t) for t in flat_inputs]
        trace_inputs = list(graph_inputs)
        trace_leaves = [leaf for _, leaf in path_leaves]

        # Clone mutated KV inputs; XLA will optimize these away.
        # The strategy for these is that during runtime, we will copy_ the
        # updated values back into the original Tensors. copy_ doesn't launch a
        # kernel, it just swizzles some metadata around (the KV pool buffers
        # are base tensors of their storage dtype; see kv_pool._zeros_kv_buffer).
        for input_index in range(kv_pool_input_start, kv_pool_input_stop):
            trace_inputs[input_index] = graph_inputs[input_index].clone()
        kv_pool_trace_inputs = trace_inputs[kv_pool_input_start:kv_pool_input_stop]
        for input_index, leaf_index in enumerate(
            tensor_leaf_indices, start=kv_pool_input_stop
        ):
            trace_leaves[leaf_index] = trace_inputs[input_index]

        rng_device = flat_inputs[0].device if self._uses_rng else None
        rng_output = None

        # Step 3: Lower to MLIR via literally running the function against the
        # placeholders. The goal is to get a flat list of output Tensors.
        with (
            dynamo_timed("direct_compile_trace"),
            execution_mode.set_eager_mode(
                execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
            ),
            ExitStack() as stack,
        ):
            if rng_device is not None:
                generator = torch.tpu.default_generators[rng_device.index or 0]
                trace_generator = tpu_torch_compile.make_export_mode_device_generator(
                    generator
                )
                tpu_torch_compile.set_export_mode_default_device_generator(
                    trace_generator
                )
                stack.callback(
                    tpu_torch_compile.clear_export_mode_default_device_generators
                )
                graph_inputs.append(
                    tpu_torch_compile.get_device_state_tensor(trace_generator)
                )

            # Respect torch.compiler.is_compiling() during tracing by returning True.
            previous_is_compiling = torch.compiler._is_compiling_flag
            torch.compiler._is_compiling_flag = True
            stack.callback(
                setattr,
                torch.compiler,
                "_is_compiling_flag",
                previous_is_compiling,
            )
            if rank_input_index is not None:
                stack.enter_context(trace_rank(graph_inputs[rank_input_index]))
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
            with _collect_callbacks() as inserted_callbacks:
                outputs = self.fn(*trace_arguments[0], **trace_arguments[1])
            if rng_device is not None:
                rng_output = tpu_torch_compile.get_device_state_tensor(trace_generator)

        output_leaves, output_treespec = tree_util.tree_flatten(outputs)
        if not all(_is_tpu_tensor(leaf) for leaf in output_leaves):
            raise TypeError("All direct_compile output leaves must be TPU tensors")
        result_tensors = output_leaves + kv_pool_trace_inputs
        # Forward hooks see Torch tensors here, not JAX values, and JAX export
        # cannot serialize host callbacks. Preserve callback values as extra
        # executable outputs, then invoke their Python callbacks after execute.
        result_tensors.extend(entry.tensor for entry in inserted_callbacks)
        if rng_output is not None:
            result_tensors.append(rng_output)
        # A mutation-only fn (the DFlash draft KV appender) returns None and has
        # no output leaves; the KV-pool copy-back is then the graph's only effect.
        if not result_tensors:
            raise TypeError(
                "direct_compile requires TPU tensor outputs or KV-pool mutations"
            )

        live_model_indices = tuple(range(model_input_count))
        if self._prune_unused_inputs:
            # Only immutable model state can be dropped. Keep every explicit input
            # (and its runtime shape check), KV input/output, and callback output.
            # Compute liveness after collecting all three kinds of graph roots.
            with dynamo_timed("direct_compile_prune_model_inputs"):
                live_model_indices = _live_model_input_indices(
                    result_tensors, graph_inputs, model_input_count
                )
            retained_indices = (
                *live_model_indices,
                *range(model_input_count, len(graph_inputs)),
            )
            model_inputs = [model_inputs[index] for index in live_model_indices]
            graph_inputs = [graph_inputs[index] for index in retained_indices]
            flat_inputs = [
                flat_inputs[index]
                for index in retained_indices
                if index < len(flat_inputs)
            ]
            logger.info(
                "[direct_compile] retained %d/%d model inputs",
                len(model_inputs),
                model_input_count,
            )

        # Step 4: Finish lowering to MLIR, then compile it.
        logger.info(
            "[direct_compile] lowering %d inputs, %d outputs, %d mutations, "
            "and %d callbacks",
            len(graph_inputs),
            len(output_leaves),
            len(kv_pool_inputs),
            len(inserted_callbacks),
        )
        # Placeholders do not retain the physical layout of their buffers.
        argument_layouts = []
        for tensor in flat_inputs:
            layout = None
            if tensor.device.type == "tpu":
                layout = tpu_torch_compile.get_device_layout_if_materialized(tensor)
            argument_layouts.append(layout[0] if layout is not None else [])
        if rng_device is not None:
            argument_layouts.append([])
        callbacks = tuple(entry.callback for entry in inserted_callbacks)
        if spmd_compiler is not None and not spmd_compiler.verify_spmd_safe:
            factories = tuple(entry.callback_factory for entry in inserted_callbacks)
            if any(factory is None for factory in factories):
                raise ValueError(
                    "Shared TPU callbacks require a callback_factory for local binding"
                )
            # Fail during tracing, before interface collectives, if a factory
            # accidentally captures an unpicklable recorder or local closure.
            callback_factories = pickle.dumps(factories)
            callbacks = tuple(factory(self._model) for factory in factories)
            interface = ExecutableInterface(
                input_digest=binding_input_digest,
                input_specs=_tensor_specs(flat_inputs),
                argument_layouts=argument_layouts,
                model_indices=live_model_indices,
                output_treespec=output_treespec,
                callback_factories=callback_factories,
            )
        pending_compile = None
        executable = None
        start = time.perf_counter()
        if async_compile:
            with dynamo_timed("direct_compile_async_submit"):
                if spmd_compiler is None:
                    pending_compile = tpu_torch_compile.async_traverse_and_compile(
                        result_tensors, graph_inputs, argument_layouts=argument_layouts
                    )
                else:
                    pending_compile = spmd_compiler.compile(
                        result_tensors,
                        graph_inputs,
                        argument_layouts=argument_layouts,
                        interface=interface,
                        graph=graph,
                    )
            logger.info(
                "[direct_compile] native async submission %.3fs",
                time.perf_counter() - start,
            )
        else:
            with dynamo_timed("direct_compile_traverse_and_compile"):
                compile_result = tpu_torch_compile.traverse_and_compile(
                    result_tensors, graph_inputs, argument_layouts=argument_layouts
                )
            executable = compile_result.executable
            logger.info(
                "[direct_compile] synchronous compilation %.3fs",
                time.perf_counter() - start,
            )
        return _CompiledArtifact(
            executable=executable,
            model_inputs=tuple(model_inputs),
            kv_pool=self._kv_pool,
            tensor_inputs_key_paths=tuple(tensor_inputs_key_paths),
            tensor_inputs_traced_shapes=tuple(
                tuple(path_leaves[index][1].shape) for index in tensor_leaf_indices
            ),
            output_treespec=output_treespec,
            output_count=len(output_leaves),
            callbacks=callbacks,
            pending_compile=pending_compile,
            rng_device=rng_device,
        )


def _tensor_specs(tensors: Sequence[torch.Tensor]) -> tuple[InputSpec, ...]:
    """Check logical input compatibility before binding another rank's executable."""
    return tuple(
        InputSpec(
            tuple(tensor.shape),
            str(tensor.dtype),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )
        for tensor in tensors
    )


@dataclass
class _CompiledArtifact:
    """Wrap a TorchTPU PJRT executable with input/output/mutation handling."""

    executable: Any
    model_inputs: tuple[torch.Tensor, ...]
    kv_pool: KVPool | None
    tensor_inputs_key_paths: tuple[tree_util.KeyPath, ...]
    tensor_inputs_traced_shapes: tuple[tuple[int, ...], ...]
    output_treespec: Any
    output_count: int
    callbacks: tuple[Callable[[torch.Tensor], None], ...] = ()
    pending_compile: Any | None = None
    rng_device: torch.device | None = None

    def wait_for_compile(self) -> None:
        if self.pending_compile is not None:
            start = time.perf_counter()
            self.executable = self.pending_compile.get()
            self.pending_compile = None
            logger.info(
                "[direct_compile] native async resolution %.3fs",
                time.perf_counter() - start,
            )

    def __call__(self, call: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
        self.wait_for_compile()
        kv_pool_inputs = self.kv_pool.get_state() if self.kv_pool is not None else []
        inputs = [*self.model_inputs, *kv_pool_inputs]
        # Resolve explicit inputs by path, similar to Dynamo sources.
        for path, traced_shape in zip(
            self.tensor_inputs_key_paths, self.tensor_inputs_traced_shapes
        ):
            try:
                value = _get_path(call, path)
            except (IndexError, KeyError, AttributeError) as exc:
                # The raw error names only the failing subscript, not which
                # input it belongs to, and the traceback stops inside the
                # generic walker. A caller that reshapes the call between
                # tracing and replay cannot act on that.
                raise ValueError(
                    "A direct_compile tensor input disappeared at runtime: "
                    f"{tree_util.keystr(path)}, traced as {traced_shape}, is no "
                    f"longer reachable ({type(exc).__name__}: {exc}). Runtime "
                    f"call: {len(call[0])} positional, kwargs {sorted(call[1])}. "
                    "Traced tensor inputs: "
                    f"{[tree_util.keystr(p) for p in self.tensor_inputs_key_paths]}"
                ) from exc
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

        if self.rng_device is None:
            outputs = tpu_torch_compile.execute(self.executable, inputs)
        else:
            generator = torch.tpu.default_generators[self.rng_device.index or 0]
            with tpu_torch_compile.MultiGeneratorLocker([generator]):
                inputs.append(tpu_torch_compile.get_device_state_tensor(generator))
                outputs = tpu_torch_compile.execute(self.executable, inputs)
                tpu_torch_compile.set_device_state_tensor(generator, outputs[-1])
                outputs = outputs[:-1]

        # Copy updated KV state back into the live pool tensors.
        with torch.no_grad():
            for tensor, replacement in zip(
                kv_pool_inputs, outputs[self.output_count :]
            ):
                tensor.copy_(replacement)

        callback_outputs = outputs[self.output_count + len(kv_pool_inputs) :]
        if len(callback_outputs) != len(self.callbacks):
            raise RuntimeError(
                "direct_compile callback output count changed: "
                f"expected={len(self.callbacks)} actual={len(callback_outputs)}"
            )
        for callback, tensor in zip(self.callbacks, callback_outputs):
            callback(tensor)

        return tree_util.tree_unflatten(
            self.output_treespec, outputs[: self.output_count]
        )


def _live_model_input_indices(
    result_tensors: list[torch.Tensor],
    graph_inputs: list[torch.Tensor],
    model_input_count: int,
) -> tuple[int, ...]:
    """Find model arguments used by the deferred graph, once per specialization.

    TorchTPU's traversal accepts a superset of the graph's inputs, retaining
    unused model weights/buffers as executable arguments. Its MLIR exporter
    preserves that argument order, so argument uses identify pruning candidates.
    Validate those candidates against the original deferred graph: lowering can
    discard operands that traversal still requires. Recompile through the normal
    compile cache; neither the model computation nor the MLIR is rewritten.
    """
    if model_input_count == 0:
        return ()

    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo

    lowered = tpu_torch_compile.build_mlir(result_tensors, graph_inputs)
    # TorchTPU and JAXlib can embed different MLIR/CHLO versions. Native
    # bytecode is not a portable interchange format (e.g. top_k's is_stable
    # property changed its encoding). TorchTPU's versioned export legalizes
    # CHLO before serializing; deserialize it into JAXlib's own dialect version.
    bytecode = tpu_torch_compile.serialize_mlir_portable_artifact(lowered)
    del lowered
    with mlir.make_ir_context() as context:
        module = stablehlo.deserialize_portable_artifact(context, bytecode)
        entry = ir.SymbolTable(module.operation)["main"]
        arguments = entry.regions[0].blocks[0].arguments
        if len(arguments) != len(graph_inputs):
            raise RuntimeError(
                "TorchTPU MLIR input count changed during model-input pruning: "
                f"expected {len(graph_inputs)}, got {len(arguments)}"
            )
        live_model_indices = tuple(
            index for index in range(model_input_count) if any(arguments[index].uses)
        )

    if len(live_model_indices) == model_input_count:
        return live_model_indices

    compact_inputs = [graph_inputs[index] for index in live_model_indices]
    compact_inputs.extend(graph_inputs[model_input_count:])
    try:
        # For example, integer isnan lowers to a constant but its deferred node
        # still references the integer input. Validate before traverse_and_compile,
        # whose argument validation can abort the process on a missing dependency.
        lowered = tpu_torch_compile.build_mlir(result_tensors, compact_inputs)
    except RuntimeError as error:
        if "identified an argument that was not provided" not in str(error):
            raise
        logger.info(
            "[direct_compile] retaining all %d model inputs: deferred traversal "
            "requires an input discarded by MLIR lowering",
            model_input_count,
        )
        return tuple(range(model_input_count))
    del lowered
    return live_model_indices


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
