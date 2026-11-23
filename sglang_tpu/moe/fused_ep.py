# SPDX-License-Identifier: Apache-2.0
"""Fused expert-parallel MoE: which kernel implementation runs, and where.

``SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL`` picks the implementation for every MoE
layer a model provider admits: ``v2`` runs ``fused_ep_moe_v2`` with all-gather
dispatch, ``push`` the same kernel with ``dispatch="push"``, whose knobs come
from the ``SGLANG_TPU_MOE_FUSED_EP_PUSH_*`` envs. Unset, nothing here runs and
every layer keeps its ``--moe-runner-backend``. The envs are read here only,
never inside the kernel package.

A provider is a module in ``_PROVIDERS`` with
``prepare_model_for_fused_ep(model, *, impl) -> int``: it finds its own MoE
layers, converts their experts and sets their ``dp_local_forward``, and returns
how many it prepared. Process-wide admission is shared
(``unmet_global_requirements``); per-layer admission is the provider's.

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

# push_fused_moe.PUSH_OPTIONS, repeated so that reading the envs needs no JAX
# import (checked equal in test_moe_fused_ep.py).
PUSH_OPTIONS = (
    "sc_prepush",
    "tc_exchange",
    "height_rungs_override",
    "tail_barrier",
    "onehot_slots",
    "hbm_staging",
    "dispatch_ahead",
    "matmul_scale_repeat",
    "exchange_row_gather",
    "bf16_sort_counts",
    "slim_prepush_barrier",
    "fused_dispatch_meta",
    "table_kernel",
    "exchange_tables",
    "kernel_plan_tables",
    "light_prepush",
    "single_barrier",
    "early_scale_scatter",
    "fuse_metadata",
)

# Modules that admit MoE layers to the fused path.
_PROVIDERS = ("sglang_tpu.models.deepseek_v41.fused_ep_op",)


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
