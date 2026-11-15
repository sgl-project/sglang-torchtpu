# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 under ``--tpu-compile-tracer direct``: tensor discovery and input pruning.

The direct tracer (sglang_tpu.compilation.direct_compile) traces one straight-line
graph whose only tensor sources are the explicit inputs, the model's parameters /
buffers / ``model_extra_tensor_attrs`` and the KV pool's ``get_state``. With
``--use-spmd-compile`` and an ``spmd_safe`` policy one rank compiles each graph and
ships the executable to the others.

Elsewhere: pool state (memory_pool.get_state), metadata pytree and RoPE through
the owner module (attention_backend), staged Engram rows on the metadata
(compile_policy, engram_host), spmd_safe (compile_policy).

Here, run by DeepseekV4DecoderTpuCompilePolicy for V4.1 before its first direct compile,
after the attention backend has built its caches:
  * plain tensor attributes of modules and of helper objects they hold become
    ``model_extra_tensor_attrs``, including aliases of registered tensors (e.g. the
    MoE TopKConfig.correction_bias holding gate.e_score_correction_bias); unlisted,
    the trace would read them as unprovided constants;
  * the policy opts into ``prune_unused_inputs``: the host-resident Engram tables
    (two FP8 [~384M, 256] parameters, ~183 GB) are registered parameters the graph
    never reads, and lifting them would make the first compile request 430 GB.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_LEAF_TYPES = (str, bytes, int, float, bool, type(None))


def discover_extra_tensor_attrs(
    model: torch.nn.Module, device_type: str = "tpu"
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Dotted paths of the plain ``device_type`` tensor attributes of every module
    (not parameters / buffers) and of helper objects a module holds as attributes,
    plus the paths of tensors the tracer cannot be told about (inside lists /
    dicts / tuples), which a trace touching them would capture as constants."""
    found: list[str] = []
    unreachable: list[str] = []
    registered = {id(t) for t in model.parameters()} | {id(t) for t in model.buffers()}

    def on_device(t) -> bool:
        return isinstance(t, torch.Tensor) and t.device.type == device_type

    def visit_value(path: str, value, depth: int) -> None:
        if on_device(value):
            # Every path, aliases included: the tracer swaps attributes by path, so
            # an unlisted alias (of a plain tensor, or of a registered parameter as
            # TopKConfig.correction_bias = gate.e_score_correction_bias) would be
            # read as an unprovided constant. direct_compile gives one tensor one
            # input however many paths reach it.
            found.append(path)
            return
        if isinstance(value, (list, tuple, dict, set)):
            items = value.values() if isinstance(value, dict) else value
            hits = [v for v in items if on_device(v) and id(v) not in registered]
            if hits:
                shapes = ",".join(
                    f"{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
                    for v in hits[:4]
                )
                unreachable.append(f"{path}[{shapes}]")
            return
        if (
            depth > 0
            or isinstance(value, (torch.nn.Module, _LEAF_TYPES, type))
            or callable(value)
            or not hasattr(value, "__dict__")
        ):
            return
        for name, sub in vars(value).items():
            visit_value(f"{path}.{name}", sub, depth + 1)

    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if name in ("_parameters", "_buffers", "_modules") or name.startswith("__"):
                continue
            path = f"{module_name}.{name}" if module_name else name
            visit_value(path, value, 0)
    return tuple(found), tuple(unreachable)


def host_resident_tensors(model) -> list[tuple[str, int]]:
    """(name, bytes) of registered parameters / buffers that are not on a TPU device, largest first."""
    seen, out = set(), []
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if id(tensor) in seen or tensor.device.type == "tpu":
            continue
        seen.add(id(tensor))
        out.append((name, tensor.numel() * tensor.element_size()))
    return sorted(out, key=lambda item: -item[1])


def prepare_direct_compile(policy) -> None:
    """Register the V4.1 model's plain tensor attributes and prune unused inputs
    on ``policy`` before its first direct compile."""
    runner = policy.model_runner
    extras, unreachable = discover_extra_tensor_attrs(runner.model)
    if extras:
        policy.model_extra_tensor_attrs = tuple(
            dict.fromkeys((*policy.get_model_extra_tensor_attrs(), *extras))
        )
    if unreachable:
        logger.warning(
            "DSV4.1 direct tracer: tensors inside containers the trace cannot lift "
            "(a forward reading them captures constants): %s",
            ", ".join(unreachable[:20]),
        )
    # direct_compile lifts every registered tensor unless told to prune; Dynamo
    # never lifts unused ones. The host Engram tables are read only on the host.
    policy.direct_compile_prune_unused_inputs = True
    host = host_resident_tensors(runner.model)
    if host:
        logger.info(
            "DSV4.1 direct tracer: %d registered tensors are not on TPU (%.1f GB, pruned "
            "unless the graph reads them): %s",
            len(host),
            sum(nbytes for _, nbytes in host) / 2**30,
            ", ".join(f"{name} ({nbytes / 2**30:.1f} GB)" for name, nbytes in host[:8]),
        )
    logger.info(
        "DSV4.1 direct tracer: %s spmd_safe=%s, %d extra tensor attrs %s",
        policy.name,
        policy.spmd_safe,
        len(extras),
        list(extras[:20]),
    )
