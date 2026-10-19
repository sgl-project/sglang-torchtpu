# SPDX-License-Identifier: Apache-2.0
"""Lazy worker/policy bindings shared by TPU serving and compilation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import msgspec

from sglang.srt.environ import envs
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = get_logger(__name__)

_ROLLBACK_ENV = (
    "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=0 and SGLANG_TPU_HOST_PADDING_NO_D2H=0"
)


def _load_class(path: str) -> type:
    module, name = path.split(":")
    return getattr(import_module(module), name)


class SpecExecutionBinding(msgspec.Struct, frozen=True):
    worker_path: str
    policy_path: str | None = None
    host_padded: bool = False
    block_worker: bool = False
    chain_worker: bool = False

    def worker_cls(self, *, default: type) -> type:
        if not self.host_padded:
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            # Compatibility adapts the shared V2 driver only; other defaults stay intact.
            if default is not EAGLEWorkerV2:
                return default
        return _load_class(self.worker_path)

    def compile_policy_mixin(self) -> type | None:
        return _load_class(self.policy_path) if self.policy_path is not None else None


_COMPAT_WORKER = "sglang_tpu.experimental.speculative.eagle.draft_worker:TpuEagleWorker"
_COMPATIBILITY = SpecExecutionBinding(worker_path=_COMPAT_WORKER)
_EAGLE = SpecExecutionBinding(
    worker_path=_COMPAT_WORKER,
    policy_path="sglang_tpu.compilation.spec_decode_compile_policy_eagle3:Eagle3SpecDecodeCompilePolicyMixin",
)
_DFLASH = SpecExecutionBinding(
    worker_path=_COMPAT_WORKER,
    policy_path="sglang_tpu.compilation.spec_decode_compile_policy_dflash:DFlashSpecDecodeCompilePolicyMixin",
)
_BLOCK = SpecExecutionBinding(
    worker_path="sglang_tpu.experimental.speculative.block_spec_worker:TpuBlockSpecWorker",
    policy_path="sglang_tpu.experimental.compilation.block_compile_policy:TpuBlockSpecDecodeCompilePolicyMixin",
    host_padded=True,
    block_worker=True,
)
_CHAIN = SpecExecutionBinding(
    worker_path="sglang_tpu.experimental.speculative.chain_spec_worker:TpuChainSpecWorker",
    policy_path="sglang_tpu.experimental.compilation.chain_compile_policy:TpuChainSpecDecodeCompilePolicyMixin",
    host_padded=True,
    chain_worker=True,
)


def experimental_spec_enabled() -> bool:
    return (
        tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.get()
        or envs.SGLANG_TPU_HOST_PADDING_NO_D2H.get()
    )


def _is_mtp(algorithm: SpeculativeAlgorithm) -> bool:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # Core resolves NEXTN (MTP) to EAGLE; exact, since is_eagle() also matches EAGLE3.
    return algorithm is SpeculativeAlgorithm.EAGLE


def resolve_spec_execution(
    algorithm: SpeculativeAlgorithm, *, pp_size: int
) -> SpecExecutionBinding:
    # Use the configured target width, never draft_pp_context's temporary PP=1 group.
    if pp_size == 1 and experimental_spec_enabled():
        if algorithm.is_dflash_family():
            return _BLOCK
        if algorithm.is_eagle3() or _is_mtp(algorithm):
            return _CHAIN
    if algorithm.is_dflash():
        return _DFLASH
    if algorithm.is_eagle() and not algorithm.is_frozen_kv_mtp():
        return _EAGLE
    return _COMPATIBILITY


def _plugin_requested_explicitly() -> bool:
    flag = tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER
    return (flag.is_set() and flag.get()) or envs.SGLANG_TPU_HOST_PADDING_NO_D2H.get()


def settle_spec_execution(
    server_args: ServerArgs, *, algorithm: SpeculativeAlgorithm
) -> SpecExecutionBinding:
    """Decide before any model loads whether the plugin worker serves this launch.

    The plugin workers check the same conditions again at construction, after the
    target weights have loaded. Without an explicit opt-in, a launch they cannot serve
    goes to the compatibility worker: turning the flag off here also reaches the
    scheduler processes, which inherit this environment, so their worker factory,
    compile policy and batch hooks all resolve the compatibility route.
    """
    from sglang.srt.arg_groups.overrides import resolving_view

    pp_size = resolving_view(server_args).pp_size
    binding = resolve_spec_execution(algorithm, pp_size=pp_size)
    if not binding.host_padded:
        return binding
    reason = _plugin_unsupported_reason(server_args, algorithm=algorithm)
    if reason is None:
        logger.info("Serving %s with the TPU plugin spec worker.", algorithm.name)
        return binding
    if algorithm.is_dspark():
        raise NotImplementedError(
            f"TPU serves DSPARK only with the plugin spec worker, which {reason}."
        )
    if _plugin_requested_explicitly():
        raise ValueError(
            f"The TPU plugin spec worker {reason}; set {_ROLLBACK_ENV} to serve "
            f"{algorithm.name} with the compatibility worker."
        )
    logger.info(
        "Serving %s with the TPU compatibility spec worker: the plugin worker %s.",
        algorithm.name,
        reason,
    )
    tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.set(False)
    return resolve_spec_execution(algorithm, pp_size=pp_size)


def _plugin_unsupported_reason(
    server_args: ServerArgs, *, algorithm: SpeculativeAlgorithm
) -> str | None:
    """What the plugin worker would reject, read from arguments and model configs.

    Mirrors the construction checks of ``TpuSpecWorker``, ``TpuChainSpecWorker``,
    ``TpuBlockSpecWorker`` and their drafters. Speculative fields still unset here
    are resolved later to values the plugin worker accepts.
    """
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    cfg = resolving_view(server_args)
    if cfg.disable_overlap_schedule:
        return "requires overlap scheduling"
    if envs.SGLANG_SIMULATE_ACC_LEN.get() > 0 and (
        envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.get() != "fixed"
    ):
        return "only supports SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed"
    if cfg.attn_dp_size > 1:
        reason = _dp_attention_unsupported_reason(cfg, algorithm=algorithm)
        if reason is not None:
            return reason
    if algorithm.is_dflash_family():
        if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
            return "only supports SGLANG_RAGGED_VERIFY_MODE=static"
        return None
    return _chain_unsupported_reason(server_args, algorithm=algorithm)


def _chain_unsupported_reason(
    server_args: ServerArgs, *, algorithm: SpeculativeAlgorithm
) -> str | None:
    """What TpuChainSpecWorker and its MTP or EAGLE3 drafter reject at construction."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    cfg = resolving_view(server_args)
    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        return "only supports SGLANG_RAGGED_VERIFY_MODE=static"
    if cfg.speculative_draft_window_size is not None:
        return "does not support --speculative-draft-window-size"
    if cfg.speculative_eagle_topk not in (None, 1):
        return "drafts chains only (--speculative-eagle-topk 1)"
    if cfg.enable_multi_layer_eagle or cfg.speculative_adaptive:
        return "does not support multi-layer or adaptive EAGLE"
    if (
        cfg.speculative_num_steps is not None
        and cfg.speculative_num_draft_tokens is not None
        and cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        return "needs --speculative-num-draft-tokens = --speculative-num-steps + 1"
    if algorithm.is_eagle3():
        return _eagle3_draft_unsupported_reason(server_args)
    return None


def _dp_attention_unsupported_reason(
    cfg, *, algorithm: SpeculativeAlgorithm
) -> str | None:
    from sglang.srt.speculative.spec_tp_sync import (
        SpecTpSyncSite,
        parse_spec_tp_sync,
    )

    if algorithm.is_dflash_family():
        return "does not support DP attention"
    if (
        envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        or cfg.speculative_skip_dp_mlp_sync
    ):
        return "does not mix prefill and decode steps across DP ranks"
    # SpecTpSync syncs nothing on a one-rank attention-TP group.
    if cfg.tp_size // cfg.attn_dp_size // cfg.attn_cp_size > 1:
        sites = parse_spec_tp_sync(envs.SGLANG_SPEC_TP_SYNC.get())
        if (SpecTpSyncSite.SPEC_ACCEPT_GREEDY in sites) != (
            SpecTpSyncSite.SPEC_ACCEPT_SAMPLE in sites
        ):
            return (
                "needs SGLANG_SPEC_TP_SYNC to enable both or neither spec-accept "
                "site under DP attention"
            )
    return None


def _eagle3_draft_unsupported_reason(server_args: ServerArgs) -> str | None:
    """What ``TpuEagle3DraftWorker`` rejects in the draft checkpoint."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.configs.model_config import ModelConfig

    cfg = resolving_view(server_args)
    try:
        draft_config = ModelConfig.from_server_args(
            server_args,
            model_path=cfg.speculative_draft_model_path,
            model_revision=cfg.speculative_draft_model_revision,
            is_draft_model=True,
        )
    except Exception:
        # A draft that cannot be read here fails later with its real error.
        return None
    architecture = draft_config.hf_config.architectures[0]
    if architecture != "LlamaForCausalLMEagle3":
        return f"drafts EAGLE3 with LlamaForCausalLMEagle3 only, not {architecture}"
    if draft_config.model_is_mrope:
        return "does not support M-RoPE drafts"
    return None


def is_experimental_spec_worker(
    spec_algorithm: SpeculativeAlgorithm, *, pp_size=1
) -> bool:
    """Whether DFLASH/DSPARK is served by TpuBlockSpecWorker."""
    return resolve_spec_execution(spec_algorithm, pp_size=pp_size).block_worker


def is_experimental_chain(spec_algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether TpuChainSpecWorker serves this algorithm (MTP or EAGLE3)."""
    return resolve_spec_execution(spec_algorithm, pp_size=pp_size).chain_worker


def is_experimental_mtp(spec_algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether MTP is served by TpuChainSpecWorker."""
    return _is_mtp(spec_algorithm) and is_experimental_chain(
        spec_algorithm, pp_size=pp_size
    )


def is_experimental_eagle3(spec_algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether EAGLE3 is served by TpuChainSpecWorker."""
    return spec_algorithm.is_eagle3() and is_experimental_chain(
        spec_algorithm, pp_size=pp_size
    )


def is_experimental_spec(spec_algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether a TpuSpecWorker serves this algorithm (host-padded batches)."""
    return resolve_spec_execution(spec_algorithm, pp_size=pp_size).host_padded
