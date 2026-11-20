# SPDX-License-Identifier: Apache-2.0
"""Leaner AOTAutograd runtime wrapper for TPU inference graphs.

torch_tpu's ``TpuBackend`` runs AOTAutograd with
``keep_inference_input_mutations=False``: the compiled graph returns the new
value of every mutated input (KV / recurrent state buffers) and AOTAutograd's
runtime wrapper ``copy_``s them back one tensor at a time, after building a
dict of every input it may need in its epilogue. On torch_tpu that ``copy_``
only swaps device buffers, but each call still costs ~3 us of dispatch; a
93-layer Kimi-K3.1 decode step has ~210 such inputs.

For inference graphs whose input mutations are plain data mutations (no
metadata / ``set_`` mutations, no outputs aliasing inputs, no effect tokens)
this module builds an equivalent wrapper that copies all updated inputs back
with one ``torch._foreach_copy_`` (or a tight ``copy_`` loop). Everything else
keeps torch's wrapper. Same ops on the same tensors, so results are identical.

These are patches of torch / torch_tpu internals, installed by
``install_from_env`` when the decoder is compiled:

install (SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER, default off)
  Why: AOTAutograd's runtime epilogue costs ~3 us per mutated input per call.
  How: replaces torch._functorch._aot_autograd.runtime_wrappers.
       _create_runtime_wrapper; ineligible graphs and every graph's first call
       keep torch's wrapper.
install_split_fast_call (SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL, default off)
  Why: a one-partition torch_tpu split executable runs its split FX
       GraphModule's Python forward on every call.
  How: replaces torch_tpu split_compiler._SplitCompiledExecutable.__call__;
       any graph other than placeholders -> one submodule -> getitems keeps it.
install_param_graph_attrs (SGLANG_TPU_DYNAMO_PARAMS_AS_GRAPH_ATTRS, default 0)
  Why: Dynamo's bytecode re-reads every parameter through its attribute chain
       on every call (~3000 on the 93-layer Kimi-K3 decoder).
  How: narrows Dynamo's install_free_tensors test in torch._dynamo.variables.
       builder to module tensors.
Upstream PR or issue: none yet. The first two belong in torch_tpu
  (keep_inference_input_mutations for inference graphs; a direct call for
  single-partition executables).
Remove when: torch_tpu keeps inference input mutations in the graph or calls
  single-partition executables directly; for the third, when Dynamo lifts
  module parameters once by default.
"""

from __future__ import annotations

import logging
import operator
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_INSTALLED: list[Callable[..., Any]] = []


def _eligible(runtime_metadata: Any, trace_joint: bool, keep_input_mutations: bool):
    """Return the runtime indices to copy back, or None to keep torch's wrapper."""
    if trace_joint or keep_input_mutations:
        return None
    md = runtime_metadata
    if md.num_outputs_aliased != 0 or md.num_intermediate_bases != 0:
        return None
    if md.dynamic_outputs or md.grad_enabled_mutation is not None:
        return None
    if len(getattr(md, "tokens", ()) or ()) != 0:
        return None
    stream_indices = getattr(md, "mutated_inp_stream_indices", None)
    if stream_indices is not None and any(s is not None for s in stream_indices):
        return None
    copies = []
    for i, inp_idx in enumerate(md.mutated_inp_runtime_indices):
        meta = md.input_info[inp_idx]
        if not meta.mutates_data and not meta.mutates_metadata:
            continue
        if meta.mutates_metadata or meta.mutates_storage_metadata:
            return None
        copies.append((i, inp_idx))
    return md.num_mutated_inp_runtime_indices, tuple(copies)


def _make_fast_wrapper(
    compiled_fn: Callable[..., Any],
    fallback: Callable[..., Any],
    num_mutated: int,
    copies: tuple[tuple[int, int], ...],
    expected_outs: int,
    disable_amp: bool,
    use_foreach: bool,
) -> Callable[..., Any]:
    from torch._functorch._aot_autograd.utils import make_boxed_func, simple_wraps

    if not getattr(compiled_fn, "_boxed_call", False):
        compiled_fn = make_boxed_func(compiled_fn)
    out_slots = tuple(i for i, _ in copies)
    inp_slots = tuple(j for _, j in copies)
    state = {"first": True}
    set_grad = torch._C._set_grad_enabled
    is_grad = torch.is_grad_enabled
    foreach_copy = torch._foreach_copy_

    @simple_wraps(compiled_fn)
    def runtime_wrapper(args: list[Any]) -> Any:
        if state["first"]:
            # The first call goes through torch's wrapper (custom-op aliasing
            # analysis, full checks).
            state["first"] = False
            return fallback(args)
        dsts = [args[j] for j in inp_slots]
        grad_enabled = is_grad()
        try:
            if grad_enabled:
                set_grad(False)
            if disable_amp:
                with torch._C._DisableAutocast():
                    all_outs = compiled_fn(args)
            else:
                all_outs = compiled_fn(args)
            if not isinstance(all_outs, list):
                all_outs = list(all_outs) if isinstance(all_outs, tuple) else [all_outs]
            if len(all_outs) != expected_outs:
                raise AssertionError(
                    f"expected {expected_outs} outputs, got {len(all_outs)}"
                )
            if dsts:
                srcs = [all_outs[i] for i in out_slots]
                if use_foreach:
                    foreach_copy(dsts, srcs)
                else:
                    for d, s in zip(dsts, srcs):
                        d.copy_(s)
        finally:
            if grad_enabled:
                set_grad(True)
        del args
        return all_outs[num_mutated:] if num_mutated else all_outs

    return runtime_wrapper


def install(use_foreach: bool = True) -> None:
    """Patch AOTAutograd so later compiles/cache loads use the lean wrapper."""
    import torch._functorch.aot_autograd  # noqa: F401 - import order
    from torch._functorch._aot_autograd import runtime_wrappers as rw

    if _INSTALLED:
        return
    original = rw._create_runtime_wrapper
    counts = {"fast": 0, "torch": 0}

    def _create_runtime_wrapper(compiled_fn, *, runtime_metadata, **kwargs):
        fallback = original(compiled_fn, runtime_metadata=runtime_metadata, **kwargs)
        plan = _eligible(
            runtime_metadata, kwargs["trace_joint"], kwargs["keep_input_mutations"]
        )
        if plan is None:
            counts["torch"] += 1
            return fallback
        num_mutated, copies = plan
        md = runtime_metadata
        expected = num_mutated + md.num_outputs + md.num_intermediate_bases
        counts["fast"] += 1
        logger.info(
            "[aot_fast_runtime_wrapper] lean wrapper: %d graph inputs, %d "
            "mutated inputs copied back with %s (%d lean / %d torch wrappers "
            "so far).",
            len(md.input_info),
            len(copies),
            "_foreach_copy_" if use_foreach else "copy_",
            counts["fast"],
            counts["torch"],
        )
        return _make_fast_wrapper(
            compiled_fn,
            fallback,
            num_mutated,
            copies,
            expected,
            kwargs["disable_amp"],
            use_foreach,
        )

    rw._create_runtime_wrapper = _create_runtime_wrapper
    _INSTALLED.append(original)


def _split_fast_plan(split_gm: Any):
    """Plan for calling a one-partition split GraphModule's executable directly.

    Returns (fn, in_idx, unwrap, out_plan, out_type) or None when the split
    graph is anything but placeholders -> one compiled submodule -> getitems.
    """
    from torch_tpu._internal.compile import split_compiler as sc

    graph = split_gm.graph
    ph_index: dict[Any, int] = {}
    call = None
    output = None
    for node in graph.nodes:
        if node.op == "placeholder":
            ph_index[node] = len(ph_index)
        elif node.op == "call_module":
            if call is not None:
                return None
            call = node
        elif node.op == "call_function":
            if not (
                node.target is operator.getitem
                and len(node.args) == 2
                and node.args[0] is call
                and isinstance(node.args[1], int)
                and not node.kwargs
            ):
                return None
        elif node.op == "output":
            output = node
        else:
            return None
    if call is None or output is None or call.kwargs:
        return None
    if not all(a in ph_index for a in call.args):
        return None
    submod = getattr(split_gm, call.target, None)
    if (
        not isinstance(submod, sc._WrapperModule)
        or submod._forward_hooks
        or submod._forward_pre_hooks
    ):
        return None
    in_idx = tuple(ph_index[a] for a in call.args)
    out = output.args[0]
    out_type = type(out)
    if out_type not in (tuple, list):
        return None
    plan = []
    for o in out:
        if o is None:
            plan.append((0, None))
        elif o in ph_index:
            plan.append((1, ph_index[o]))
        elif isinstance(o, torch.fx.Node) and o.op == "call_function":
            plan.append((2, o.args[1]))
        elif o is call:
            plan.append((3, None))
        else:
            return None
    if in_idx == tuple(range(len(ph_index))):
        in_idx = None  # every placeholder, in order: pass args through
    plan = tuple(plan)
    if plan == tuple((2, k) for k in range(len(plan))):
        plan = len(plan)  # getitem 0..n-1 in order
    return submod.submod, in_idx, submod.unwrap_singleton_tuple, plan, out_type


def install_split_fast_call() -> None:
    """Call a one-partition torch_tpu split executable without its FX forward."""
    from torch_tpu._internal.compile import split_compiler as sc

    cls = sc._SplitCompiledExecutable
    if getattr(cls, "_sglang_fast_call", False):
        return
    original_call = cls.__call__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plan = self.__dict__.get("_sglang_fast_plan", False)
        if plan is False:
            try:
                plan = _split_fast_plan(self._split_gm)
            except Exception:  # noqa: BLE001 - any surprise keeps torch's path
                plan = None
            self.__dict__["_sglang_fast_plan"] = plan
            if plan is not None:
                logger.info(
                    "[aot_fast_runtime_wrapper] direct call of a one-partition "
                    "split executable (input map %s, output map %s).",
                    "identity" if plan[1] is None else len(plan[1]),
                    plan[3] if type(plan[3]) is int else len(plan[3]),
                )
        if plan is None or kwargs or self.recompile_fn is not None:
            return original_call(self, *args, **kwargs)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        fn, in_idx, unwrap, out_plan, out_type = plan
        if in_idx is None:
            res = fn(*args)
        else:
            res = fn(*[args[i] for i in in_idx])
        if unwrap and isinstance(res, (tuple, list)):
            res = res[0]
        if type(out_plan) is int:
            # getitem 0..n-1 of the executable's outputs, in order.
            if len(res) != out_plan:
                res = res[:out_plan]
            return list(res) if out_type is list else tuple(res)
        outs = [
            res[v] if k == 2 else args[v] if k == 1 else res if k == 3 else None
            for k, v in out_plan
        ]
        return outs if out_type is list else tuple(outs)

    cls.__call__ = __call__
    cls._sglang_fast_call = True


_ALL_MODULE_TENSORS = [False]


def _is_module_param_or_buffer(source: Any) -> bool:
    """True for ``<nn.Module>._parameters[k]`` / ``<nn.Module>._buffers[k]``
    (mode 1), or for any tensor reached from an nn.Module (mode 2)."""
    from torch._dynamo import source as dsrc

    if _ALL_MODULE_TENSORS[0]:
        return dsrc.is_from_unspecialized_nn_module_source(source)
    if not isinstance(source, (dsrc.DictGetItemSource, dsrc.GetItemSource)):
        return False
    base = source.base
    if not isinstance(base, dsrc.UnspecializedParamBufferSource):
        return False
    if base.member not in ("_parameters", "_buffers"):
        return False
    if not dsrc.is_from_unspecialized_nn_module_source(base):
        return False
    return True


def install_param_graph_attrs(all_module_tensors: bool = False) -> None:
    """Make Dynamo install nn.Module parameters/buffers as graph attributes.

    With ``inline_inbuilt_nn_modules`` every parameter is a graph input that
    Dynamo's generated bytecode re-reads through its module attribute chain
    on every call (~0.1-0.25 us each; ~3000 on the 93-layer decoder).
    Installed as graph attributes instead, AOTAutograd lifts them once
    (``aot_module_simplified`` keeps them in ``params_buffers_flat``) and the
    per-call work is one list extend. The compiled program sees the same
    tensors (inputs reordered: parameters first).

    Uses Dynamo's ``install_free_tensors`` path, narrowed to tensors read
    from ``<module>._parameters`` / ``<module>._buffers`` (not globals,
    nonlocals or plain module attributes). Like the nn-module guard filter,
    a parameter or buffer *object* replaced after compile is not noticed
    (in-place weight updates are fine).
    """
    import torch._dynamo.config as dynamo_config
    from torch._dynamo.variables import builder

    if getattr(builder, "_sglang_param_graph_attrs", False):
        return
    _ALL_MODULE_TENSORS[0] = all_module_tensors

    def _never(source: Any) -> bool:
        return False

    # In builder.py these three names are only read by wrap_tensor's
    # install_free_tensors test; the other install_free_tensors reads are
    # export-only.
    builder.is_from_global_source = _never
    builder.is_from_nonlocal_source = _never
    builder.is_from_unspecialized_nn_module_source = _is_module_param_or_buffer
    dynamo_config.install_free_tensors = True
    builder._sglang_param_graph_attrs = True
    logger.info(
        "[aot_fast_runtime_wrapper] Dynamo installs nn.Module parameters and "
        "buffers as graph attributes (lifted once by AOTAutograd)."
    )


def install_from_env() -> None:
    """Install the host-overhead cuts enabled by their env flags."""
    from sglang_tpu.envs import envs as tpu_envs

    if tpu_envs.SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER.get():
        install()
    if tpu_envs.SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL.get():
        install_split_fast_call()
    mode = tpu_envs.SGLANG_TPU_DYNAMO_PARAMS_AS_GRAPH_ATTRS.get()
    if mode:
        install_param_graph_attrs(all_module_tensors=mode >= 2)
