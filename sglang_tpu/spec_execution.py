# SPDX-License-Identifier: Apache-2.0
"""Lazy worker/policy bindings shared by TPU serving and compilation."""

from __future__ import annotations

import inspect
import os
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

# MTP drafts (as ModelConfig names a draft) TPU serving is exercised on: Qwen3.8 and
# GLM-5.3-Flash. Others the MTP drafter loads keep the compatibility worker by
# default; an explicit opt-in still sends them to the plugin worker.
_SERVED_MTP_DRAFT_ARCHITECTURES = frozenset(
    {"Qwen3_5ForCausalLMMTP", "Glm5NextForConditionalGenerationNextN"}
)


def experimental_spec_enabled() -> bool:
    if tpu_envs.SGLANG_TPU_SPEC_WORKER_FALLBACK.get():
        return False
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
    return _resolve(algorithm, pp_size=pp_size, plugin=experimental_spec_enabled())


def _resolve(
    algorithm: SpeculativeAlgorithm, *, pp_size: int, plugin: bool
) -> SpecExecutionBinding:
    # Use the configured target width, never draft_pp_context's temporary PP=1 group.
    if pp_size == 1 and plugin:
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
    """Decide before any model loads which worker serves this launch.

    The plugin workers check the same conditions again at construction, after the
    target weights have loaded. Without an explicit opt-in, a launch they cannot serve
    goes to the compatibility worker. The fallback is recorded for this launch alone
    in SGLANG_TPU_SPEC_WORKER_FALLBACK, which the scheduler processes inherit, so
    their worker factory, compile policy and batch hooks all resolve the
    compatibility route; the user's own flag is left as set, and the next launch
    in this process decides afresh.
    """
    from sglang.srt.arg_groups.overrides import resolving_view

    fallback = tpu_envs.SGLANG_TPU_SPEC_WORKER_FALLBACK
    fallback.clear()
    cfg = resolving_view(server_args)
    binding = resolve_spec_execution(algorithm, pp_size=cfg.pp_size)
    reason = None
    if binding.host_padded:
        reason = _plugin_rejection(server_args, algorithm=algorithm)
        if reason is not None:
            binding = _resolve(algorithm, pp_size=cfg.pp_size, plugin=False)
    if (
        cfg.tpu_compile_tracer == "direct"
        and not algorithm.is_none()
        and not _direct_compiles(binding, algorithm)
    ):
        detail = f"; the plugin worker {reason}" if reason is not None else ""
        raise ValueError(
            f"--tpu-compile-tracer direct does not support {algorithm.name} on the "
            f"TPU compatibility spec worker{detail}."
        )
    if reason is not None:
        logger.info(
            "Serving %s with the TPU compatibility spec worker: the plugin worker %s.",
            algorithm.name,
            reason,
        )
        fallback.set(True)
    elif binding.host_padded:
        logger.info("Serving %s with the TPU plugin spec worker.", algorithm.name)
    return binding


def _plugin_rejection(
    server_args: ServerArgs, *, algorithm: SpeculativeAlgorithm
) -> str | None:
    """Why the default route leaves the plugin worker; raise where it cannot."""
    explicit = _plugin_requested_explicitly()
    reason = _plugin_unsupported_reason(server_args, algorithm=algorithm)
    if reason is None and not explicit:
        reason = _plugin_unproven_reason(server_args, algorithm=algorithm)
    if reason is None:
        return None
    if algorithm.is_dspark():
        raise NotImplementedError(
            f"TPU serves DSPARK only with the plugin spec worker, which {reason}."
        )
    if explicit:
        raise ValueError(
            f"The TPU plugin spec worker {reason}; set {_ROLLBACK_ENV} to serve "
            f"{algorithm.name} with the compatibility worker."
        )
    return reason


def _direct_compiles(
    binding: SpecExecutionBinding, algorithm: SpeculativeAlgorithm
) -> bool:
    # Every plugin worker is direct-compiled. A compatibility driver takes core's own
    # answer: this plugin's hooks claim the chain algorithms for the chain worker,
    # which is not serving.
    if binding.host_padded:
        return True
    core = inspect.unwrap(type(algorithm).supports_tpu_direct_compile)
    return core(algorithm)


def _plugin_unsupported_reason(
    server_args: ServerArgs, *, algorithm: SpeculativeAlgorithm
) -> str | None:
    """What the plugin worker would reject, read from arguments and model configs.

    Mirrors the construction checks of ``TpuSpecWorker``, ``TpuChainSpecWorker``,
    ``TpuBlockSpecWorker`` and their drafters. Speculative fields still unset here
    are resolved later to values the plugin worker accepts. A model config that
    cannot be read raises here, as loading it would.
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
        if algorithm.is_dspark():
            return _dspark_draft_unsupported_reason(server_args)
        return _dflash_draft_unsupported_reason(server_args)
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


def _eagle3_draft_unsupported_reason(server_args: ServerArgs) -> str | None:
    """What ``TpuEagle3DraftWorker`` rejects in the draft checkpoint."""
    from sglang.srt.arg_groups.overrides import resolving_view

    path = resolving_view(server_args).speculative_draft_model_path
    if path is None:
        # Core's speculative arguments reject an EAGLE3 launch without one.
        return None
    draft_config = _draft_model_config(server_args, path=path)
    architecture = draft_config.hf_config.architectures[0]
    if architecture != "LlamaForCausalLMEagle3":
        return f"drafts EAGLE3 with LlamaForCausalLMEagle3 only, not {architecture}"
    if draft_config.model_is_mrope:
        return "does not support M-RoPE drafts"
    return None


def _plugin_unproven_reason(
    server_args: ServerArgs, *, algorithm: SpeculativeAlgorithm
) -> str | None:
    """Why a launch the plugin worker accepts still defaults to compatibility."""
    if not _is_mtp(algorithm):
        return None
    architecture = _mtp_draft_architecture(server_args)
    if architecture not in _SERVED_MTP_DRAFT_ARCHITECTURES:
        return f"is not yet exercised with {architecture} drafts"
    return None


def _mtp_draft_architecture(server_args: ServerArgs) -> str:
    from sglang.srt.arg_groups.overrides import resolving_view

    # Native MTP drafts may live in the target checkpoint.
    cfg = resolving_view(server_args)
    path = cfg.speculative_draft_model_path or cfg.model_path
    return _draft_model_config(server_args, path=path).hf_config.architectures[0]


def _draft_model_config(server_args: ServerArgs, *, path: str):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.configs.model_config import ModelConfig

    # Built aside rather than memoized on the record: resolution is still running.
    return ModelConfig.from_server_args(
        server_args,
        model_path=path,
        model_revision=resolving_view(server_args).speculative_draft_model_revision,
        is_draft_model=True,
    )


def _target_model_config(server_args: ServerArgs):
    from sglang.srt.configs.model_config import ModelConfig

    return ModelConfig.from_server_args(server_args)


def _dflash_draft_unsupported_reason(server_args: ServerArgs) -> str | None:
    """What ``TpuDflashDraftWorker`` rejects in the draft checkpoint."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

    path = resolving_view(server_args).speculative_draft_model_path
    if path is None:
        # Core's speculative arguments reject a DFlash-family launch without one.
        return None
    hf_config = _draft_model_config(server_args, path=path).hf_config
    if "LiLiCorrDraftModel" in hf_config.architectures:
        return "does not support LiLiCorr drafts"
    draft_config = parse_dflash_draft_config(draft_hf_config=hf_config)
    if draft_config.is_domino:
        return "does not support Domino drafts"
    if os.path.exists(os.path.join(path, "mask_embedding.pt")):
        return "does not merge mask_embedding.pt"
    if draft_config.mask_token_id is None:
        return "needs dflash_config.mask_token_id in the draft config"
    vocab_size = _target_model_config(server_args).vocab_size
    if draft_config.mask_token_id >= vocab_size:
        return f"needs a mask_token_id inside the target vocab size {vocab_size}"
    return None


def _dspark_draft_unsupported_reason(server_args: ServerArgs) -> str | None:
    """What ``TpuDsparkDraftWorker`` rejects in the draft checkpoint."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.speculative.dspark_components.dspark_config import (
        resolve_runtime_config,
    )

    cfg = resolving_view(server_args)
    if cfg.speculative_draft_model_path is None:
        # Core's speculative arguments reject a DFlash-family launch without one.
        return None
    hf_config = _draft_model_config(
        server_args, path=cfg.speculative_draft_model_path
    ).hf_config
    try:
        resolve_runtime_config(
            draft_hf_config=hf_config,
            speculative_num_draft_tokens=cfg.speculative_num_draft_tokens,
            target_vocab_size=_target_model_config(server_args).vocab_size,
        )
    except ValueError as e:
        return f"cannot serve this draft ({e})"
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
