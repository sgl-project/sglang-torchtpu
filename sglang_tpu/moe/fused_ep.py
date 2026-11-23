# SPDX-License-Identifier: Apache-2.0
"""Fused expert-parallel MoE: which kernel implementation runs, and where.

``SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL`` picks the implementation for every MoE
layer a model provider admits: ``v2`` runs ``fused_ep_moe_v2`` with all-gather
dispatch, ``push`` the same kernel with ``dispatch="push"``, whose knobs come
from the ``SGLANG_TPU_MOE_FUSED_EP_PUSH_*`` envs. Unset, nothing here runs and
every layer keeps its ``--moe-runner-backend``. The envs are read here only,
never inside the kernel package.

``push`` also requires two XLA flags in ``LIBTPU_INIT_ARGS`` at libtpu start:
``--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0`` and
``--xla_mosaic_unsafe_allow_multicore_remote_dma=true`` (see push_fused_moe).

A provider is a module in ``_PROVIDERS`` with
``prepare_model_for_fused_ep(model, *, impl) -> int``: it finds its own MoE
layers, converts their experts, sets ``DP_LOCAL_FORWARD_ATTR`` on each, and
returns how many it prepared. Process-wide admission is shared
(``unmet_global_requirements``); per-layer admission is the provider's.
``run_moe_ffn_dp_sync_around`` (an AROUND hook, so core stays untouched)
routes a prepared layer's MoE step to it.

Importable without torch_tpu or JAX.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import msgspec

logger = logging.getLogger(__name__)

IMPL_ENV = "SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL"
_PUSH_ENV_PREFIX = "SGLANG_TPU_MOE_FUSED_EP_PUSH_"

# Implementation name -> fused_ep_moe_v2's dispatch.
_DISPATCH = {"v2": "all_gather", "push": "push"}

# The push knobs with an env: the ones vllm-torchtpu's DSV4.1 recipe sets
# (scripts/dsv41_bench). Every other push_fused_moe.PUSH_OPTIONS entry keeps
# the kernel default; add one here (and its env) once a measurement needs it.
PUSH_OPTIONS = ("sc_prepush", "tc_exchange")

# Modules that admit MoE layers to the fused path.
_PROVIDERS = ("sglang_tpu.models.deepseek_v41.fused_ep_op",)

# Set by a provider on each prepared MoE block (``layer.mlp``): a callable
# ``(hidden_states, forward_batch, *, input_ids) -> Optional[Tensor]``
# returning this rank's complete MoE output, or None for the ordinary path.
DP_LOCAL_FORWARD_ATTR = "_tpu_fused_ep_dp_local_forward"


class FusedEpImpl(msgspec.Struct, frozen=True):
    """One resolved implementation; hashable, so it can key a cached op."""

    name: str
    dispatch: str
    # Sorted (option, value) pairs of the push envs that are set; () keeps
    # every kernel default and is the only value under all-gather dispatch.
    push_options: tuple[tuple[str, Any], ...] = ()


def push_option_env(option: str) -> str:
    return _PUSH_ENV_PREFIX + option.upper()


def _push_env_field(option: str):
    from sglang_tpu.envs import envs as tpu_envs

    return getattr(tpu_envs, push_option_env(option))


def _set_push_options() -> tuple[tuple[str, Any], ...]:
    options = []
    for option in PUSH_OPTIONS:
        field = _push_env_field(option)
        if field.is_set():
            options.append((option, field.get()))
    return tuple(sorted(options))


def is_fused_ep_requested() -> bool:
    """Whether the impl env is set at all; never raises (hook specs read it)."""
    from sglang_tpu.envs import envs as tpu_envs

    return bool(tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.get())


def resolve_fused_ep_impl() -> Optional[FusedEpImpl]:
    """The selected implementation, or None when the impl env is unset."""
    from sglang_tpu.envs import envs as tpu_envs

    name = tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.get()
    if not name:
        return None
    if name not in _DISPATCH:
        raise ValueError(
            f"{IMPL_ENV}={name!r}: expected one of {sorted(_DISPATCH)}, or unset"
        )
    dispatch = _DISPATCH[name]
    push_options = _set_push_options() if dispatch == "push" else ()
    return FusedEpImpl(name=name, dispatch=dispatch, push_options=push_options)


def check_fused_ep_env() -> None:
    """Refuse a bad selection at startup, before the weights load."""
    from sglang.srt.environ import envs

    impl = resolve_fused_ep_impl()
    if impl is None:
        return
    if not envs.SGLANG_SHARED_EXPERT_TP1.get():
        raise ValueError(
            f"{IMPL_ENV}={impl.name} needs SGLANG_SHARED_EXPERT_TP1=1 (the fused "
            "EP layers add a replicated shared expert on each rank's own rows)."
        )
    if impl.dispatch != "push":
        ignored = [
            push_option_env(o) for o in PUSH_OPTIONS if _push_env_field(o).is_set()
        ]
        if ignored:
            logger.warning(
                "%s=%s ignores the push-only %s.", IMPL_ENV, impl.name, ignored
            )


def unmet_global_requirements(parallel, exec_ctx, a2a_backend) -> list[str]:
    """Process-wide conditions every fused EP layer relies on (empty when all
    hold): each rank brings its own rows and the kernel does the exchange."""
    world = int(parallel.world_group.world_size)
    ep = int(parallel.moe_ep_size)
    moe_args = exec_ctx.moe
    features = exec_ctx.features
    requirements = (
        ("attention TP size 1", int(parallel.attn_tp_size) == 1),
        (f"MoE EP size == world size ({ep} vs {world})", ep == world),
        (
            f"attention DP size == EP size ({parallel.attn_dp_size} vs {ep})",
            int(parallel.attn_dp_size) == ep,
        ),
        ("pipeline parallel size 1", int(parallel.pp_size) == 1),
        ("attention context parallel size 1", int(parallel.attn_cp_size) == 1),
        ("no MoE all-to-all backend", a2a_backend.is_none()),
        (
            "EPLB and redundant experts off",
            not moe_args.enable_eplb and not moe_args.ep_num_redundant_experts,
        ),
        # A fused layer never runs the layer's TopK, so its routing reaches
        # neither the routed-experts capture nor the expert distribution
        # recorder: refuse both rather than return data without these layers.
        (
            "--enable-return-routed-experts off (fused layers capture no routing)",
            not features.enable_return_routed_experts,
        ),
        (
            "--expert-distribution-recorder-mode unset (fused layers record no "
            "routing)",
            moe_args.expert_distribution_recorder_mode is None,
        ),
        # The relayout and the in-kernel gating take physical expert e as
        # logical expert e.
        (
            "--init-expert-location trivial (experts in logical order)",
            moe_args.init_expert_location == "trivial",
        ),
        (
            "decoder SWA bounded replay off (it drops rows per layer)",
            not features.enable_decoder_swa_bounded_replay,
        ),
    )
    return [name for name, met in requirements if not met]


def prepare_model_for_fused_ep(model) -> int:
    """Hand ``model`` to every provider; returns the number of fused layers."""
    impl = resolve_fused_ep_impl()
    if impl is None:
        return 0
    return sum(
        importlib.import_module(path).prepare_model_for_fused_ep(model, impl=impl)
        for path in _PROVIDERS
    )


def run_moe_ffn_dp_sync_around(
    original,
    layer,
    hidden_states,
    forward_batch,
    *,
    input_ids,
    input_ids_global,
    return_moe_output: bool = False,
):
    """AROUND NonCudaDecoderMixin._run_moe_ffn_dp_sync (same signature after
    ``original``): a prepared layer runs on this rank's own rows, with no DP
    gather, scatter or post-experts all-reduce; any other layer, or a None
    from the provider, takes the original path."""
    # Plugin-owned and set only on prepared layers, so core never declares it.
    dp_local_forward = getattr(layer.mlp, DP_LOCAL_FORWARD_ATTR, None)
    if dp_local_forward is not None:
        out = dp_local_forward(hidden_states, forward_batch, input_ids=input_ids)
        if out is not None:
            if return_moe_output:
                # The output is already reduced; a deferred-finalize caller
                # would reduce it again.
                raise RuntimeError(
                    "the fused EP output is complete, but the caller asked for a "
                    "deferred MoEOutput (return_moe_output)"
                )
            return out
    return original(
        layer,
        hidden_states,
        forward_batch,
        input_ids=input_ids,
        input_ids_global=input_ids_global,
        return_moe_output=return_moe_output,
    )
