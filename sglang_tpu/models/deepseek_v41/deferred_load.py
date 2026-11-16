# SPDX-License-Identifier: Apache-2.0
"""Load DeepSeek-V4.1's real weights while the direct-tracer compile runs.

Opt-in: ``SGLANG_TPU_DSV41_DEFERRED_LOAD=1`` with ``--tpu-compile-tracer
direct`` and ``--use-spmd-compile``. The weights are the usual ones:
``--load-format`` (e.g. ``runai_streamer``) from ``--model-path`` (e.g.
``gs://...``).

The direct tracer needs the final parameters' shapes, dtypes and layouts, not
their values, so startup can be reordered:

1. DSV41DeferredDummyLoader (selected by the get_model_loader hook, holding
   the loader chosen for the real format) builds the model (raw parameter specs recorded first), fills it with random values
   except the host-resident tensors (the Engram tables), and runs the post-load
   pass: the parameters take their final TPU layout.
2. The KV pool is built and the decoder graphs are traced and submitted for
   asynchronous compilation, as usual.
3. Once every rank's trace succeeded, SPMDCompiler.finish runs load_pending
   (added to that compiler's before_distribute steps when the V4.1 policy
   compiled the model) before it waits for the compiles: every compiled function releases its model-input tensors, the
   raw parameters are recreated from the recorded specs (freeing the dummy
   final ones), the real checkpoint is streamed and post-processed, and the
   compiled functions re-attach their inputs by name. The warmups then run on
   real weights.
"""

from __future__ import annotations

import logging
from time import perf_counter

import msgspec
import torch
from torch import nn

logger = logging.getLogger(__name__)


class _Deferred(msgspec.Struct):
    """One model built with dummy weights for a deferred real load."""

    # (module, name, shape, dtype, device, attributes, class) of every raw parameter.
    params: list
    # module -> plain (non-tensor, non-module) attributes before the post-load pass.
    attrs: dict
    # Set by register() when the direct tracer compiles this model.
    runner: object = None
    # The CompiledFunctions traced against this model (register()).
    functions: list = msgspec.field(default_factory=list)
    # The loader get_model_loader chose for the real format (defer()).
    loader: object = None


# id(model) -> deferred state, from _initialize_model until the real load succeeds.
_DEFERRED: dict[int, _Deferred] = {}
# Only the first V4.1 target load in a process is deferred; later loads (e.g.
# a weight update) use the real loader.
_deferred_in_process = False


def _plain_attrs(module: nn.Module) -> dict:
    return {
        k: v
        for k, v in vars(module).items()
        if not k.startswith("_")
        and not isinstance(v, (torch.Tensor, nn.Module))
        and not callable(v)
    }


def record_raw_params(model: nn.Module) -> None:
    """Remember the freshly built model's parameters before anything packs them."""
    params, attrs = [], {}
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is not None:
                params.append(
                    (
                        module,
                        name,
                        tuple(param.shape),
                        param.dtype,
                        param.device,
                        dict(vars(param)),
                        type(param),
                    )
                )
        if getattr(module, "quant_method", None) is not None:
            attrs[module] = _plain_attrs(module)
    _DEFERRED[id(model)] = _Deferred(params=params, attrs=attrs)
    logger.info("[deferred-load] recorded %d raw parameters", len(params))


def restore_raw_params(model: nn.Module) -> None:
    """Replace the post-processed parameters with empty raw ones."""
    state = _DEFERRED[id(model)]
    for module, name, shape, dtype, device, attrs, cls in state.params:
        current = module._parameters.get(name)
        if (
            current is not None
            and tuple(current.shape) == shape
            and current.dtype == dtype
            and current.device == device
        ):
            # Same storage is fine, but the post-load pass may have rebound it to
            # a new Parameter without the loader attributes (weight_loader, ...).
            if type(current) is not cls:
                current.__class__ = cls
            vars(current).update(attrs)
            continue
        param = nn.Parameter(
            torch.empty(shape, dtype=dtype, device=device), requires_grad=False
        )
        # sglang's quantized parameters are Parameter subclasses that expose
        # weight_loader as a property over _weight_loader: keep the class.
        if cls is not nn.Parameter:
            param.__class__ = cls
        vars(param).update(attrs)
        module._parameters[name] = param
    # Drop every parameter that did not exist when the model was built, on any
    # module, including submodules created later (post_load_weights fuses some
    # projections; the post-load pass adds repacked scales). The real load's own
    # post-load steps create them again.
    recorded = {(id(module), name) for module, name, *_ in state.params}
    for module in model.modules():
        for name in [n for n in module._parameters if (id(module), n) not in recorded]:
            del module._parameters[name]
    for module, attrs in state.attrs.items():
        for key, value in attrs.items():
            setattr(module, key, value)


def device_only(model: nn.Module):
    """A state_dict view for initialize_dummy_weights that leaves the
    host-resident tensors (the Engram tables, ~183 GB) alone; the real load
    fills them."""

    class _DeviceOnly:
        def state_dict(self):
            return {
                k: v for k, v in model.state_dict().items() if v.device.type != "cpu"
            }

    return _DeviceOnly()


def should_defer(model_config, load_config) -> bool:
    """Whether this model's real weights load while its graphs compile; raises on
    a configuration the deferred load does not support."""
    from sglang_tpu.envs import envs

    if not envs.SGLANG_TPU_DSV41_DEFERRED_LOAD.get() or _deferred_in_process:
        return False
    config = getattr(model_config, "hf_config", None)
    if getattr(config, "model_type", None) != "deepseek_v41":
        return False
    if getattr(model_config, "is_draft_model", False) or (
        getattr(load_config, "draft_model_idx", None) is not None
    ):
        return False  # Drafts load with their own format, as usual.
    load_format = str(
        getattr(load_config.load_format, "value", load_config.load_format)
    )
    if load_format == "dummy":
        raise ValueError(
            "SGLANG_TPU_DSV41_DEFERRED_LOAD defers the --load-format weights; "
            "with dummy there is nothing to load"
        )
    from sglang.srt.environ import envs as core_envs
    from sglang.srt.runtime_context import get_exec

    graph = get_exec().graph
    required = {
        "--enable-torch-compile": graph.enable_torch_compile,
        "--tpu-compile-tracer direct": graph.tpu_compile_tracer == "direct",
        "--use-spmd-compile": graph.use_spmd_compile,
        "SGLANG_DIRECT_COMPILE_ASYNC": core_envs.SGLANG_DIRECT_COMPILE_ASYNC.get(),
    }
    missing = [name for name, ok in required.items() if not ok]
    if missing:
        raise ValueError(
            "SGLANG_TPU_DSV41_DEFERRED_LOAD loads the weights while the direct "
            "tracer's SPMD compiles run, after every rank's trace succeeded; it "
            f"needs {', '.join(missing)}"
        )
    return True


def defer(model: nn.Module, loader) -> None:
    """Record ``model``'s raw parameters (call before the post-load pass); its
    real weights load with ``loader`` once the direct tracer compiled it."""
    global _deferred_in_process
    record_raw_params(model)
    _DEFERRED[id(model)].loader = loader
    _deferred_in_process = True


def register(model_runner, compiled_function) -> None:
    """The direct tracer compiled this runner's model into ``compiled_function``:
    load its real weights before SPMDCompiler.finish and re-attach that
    function's inputs."""
    state = _DEFERRED.get(id(model_runner.model))
    if state is None:
        return
    from sglang_tpu.compilation.warmup_queue import current_warmup_queue

    queue = current_warmup_queue()
    compiler = queue.spmd_compiler if queue is not None else None
    if compiler is None:
        raise RuntimeError(
            "SGLANG_TPU_DSV41_DEFERRED_LOAD: the direct tracer compiled outside an "
            "SPMD warmup scope, so the real weights would never load"
        )
    state.runner = model_runner
    state.functions.append(compiled_function)
    if load_pending not in compiler.before_distribute:
        compiler.before_distribute.append(load_pending)


class _DeferredBarrierGroup:
    """A process group whose barrier() is a no-op, for the shared Engram tables
    during a deferred load: load_pending's final gather is the barrier, so every
    rank runs exactly one collective however far its load got."""

    def __init__(self, group):
        self._group = group

    def barrier(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return getattr(self._group, name)


def _shared_host_tables(model: nn.Module) -> list:
    return [
        module.host_table
        for module in model.modules()
        if getattr(getattr(module, "host_table", None), "layout", None) == "shared"
    ]


def _load(state: _Deferred) -> None:
    from sglang.srt.model_loader.loader import DefaultModelLoader
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    runner = state.runner
    model = runner.model
    start = perf_counter()
    functions = state.functions
    for fn in functions:
        fn.release_model_inputs()
    restore_raw_params(model)
    loader = state.loader
    with set_default_torch_dtype(runner.model_config.dtype):
        DefaultModelLoader.load_weights_and_postprocess(
            model,
            loader._get_all_weights(runner.model_config, model),
            torch.device(runner.device),
        )
    changed = sum(fn.refresh_model_inputs() for fn in functions)
    logger.info(
        "[deferred-load] real weights loaded in %.1f s; %d compiled functions "
        "re-attached %d inputs",
        perf_counter() - start,
        len(functions),
        changed,
    )


def load_pending(gather) -> None:
    """Load every deferred model's real weights, then agree across ranks with
    ``gather`` (value -> per-rank list), the only collective here: the shared
    Engram tables' barriers are deferred to it, so a rank-local failure at any
    point fails every rank instead of leaving the others in a collective."""
    error = None
    try:
        for key, state in list(_DEFERRED.items()):
            if state.runner is None:
                raise RuntimeError(
                    "a model built for SGLANG_TPU_DSV41_DEFERRED_LOAD was never "
                    "compiled by --tpu-compile-tracer direct; its weights are "
                    "still dummy"
                )
            tables = _shared_host_tables(state.runner.model)
            groups = [table.group for table in tables]
            for table in tables:
                table.group = _DeferredBarrierGroup(table.group)
            try:
                _load(state)
            finally:
                for table, group in zip(tables, groups):
                    table.group = group
            del _DEFERRED[key]
    except Exception as caught:
        logger.exception("[deferred-load] failed")
        error = f"{type(caught).__name__}: {caught}"
    # Every rank has finished writing its rows into the shared tables once this
    # returns, which is what the deferred table barriers ensured.
    errors = gather(error)
    failed = [(rank, e) for rank, e in enumerate(errors) if e is not None]
    if failed:
        _DEFERRED.clear()
        raise RuntimeError(f"DeepSeek-V4.1 deferred weight load failed: {failed}")
