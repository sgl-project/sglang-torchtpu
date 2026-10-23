# SPDX-License-Identifier: Apache-2.0
"""TPU platform backend for SGLang SRT using torch_tpu.

Implements the SRTPlatform interface so that SGLang can run inference
on Google Cloud TPUs via the torch_tpu (PyTorch privateuse1) backend.
"""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Optional, Type

import torch

from sglang.srt.environ import envs
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.utils import LazyImport
from sglang_tpu._logging import get_logger
from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.server_args import ServerArgs

logger = get_logger(__name__)


# Platform discovery runs before worker bootstrap configures torch_tpu.
rank_slice = LazyImport("sglang_tpu.rank_local", "rank_slice")
traced_rank = LazyImport("sglang_tpu.rank_local", "traced_rank")


# Default size for libtpu's premapped (pinned, DMA-mapped) host staging pool.
_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
# Flush torch_tpu's lazy host->device weight-copy graph once this much work accumulates,
# so it materializes in chunks rather than all at once, which may cause HBM OOMs.
_LOAD_WEIGHT_SYNC_BYTES = 2 << 30  # 2 GiB
_LOAD_WEIGHT_SYNC_TENSORS = 64


def _sync_weight_stream(weights):
    """Yield weights and periodically drain torch_tpu pending work.

    ``torch.accelerator.synchronize()`` blocks until async device work and deferred
    operations on the TPU have completed, which lets transient buffers be freed.
    This bounds HBM peak usage only when the loader copies parameters while
    iterating.

    Warning: when ``weights`` is a one-pass iterator/generator, some loaders call
    ``list(weights)`` before loading parameters. That consumes the entire stream
    before any real parameter copy happens, so the periodic syncs cannot bound the
    HBM peak.
    """
    pending_bytes = 0
    pending_tensors = 0
    for name, tensor in weights:
        yield name, tensor
        pending_tensors += 1
        pending_bytes += tensor.numel() * tensor.element_size()
        if (
            pending_bytes >= _LOAD_WEIGHT_SYNC_BYTES
            or pending_tensors >= _LOAD_WEIGHT_SYNC_TENSORS
        ):
            torch.accelerator.synchronize()
            pending_bytes = 0
            pending_tensors = 0


# Page size used when the user does not pick one and the attention backend has
# no floor of its own. Token-granular pages (SGLang's global default of 1) turn
# each KV fetch into many tiny DMAs, which is fine for GPU kernels and wasteful
# here.
_DEFAULT_PAGE_SIZE = 64


def _rpa_kernel_spec(backend: str | None):
    """``backend``'s ``RpaKernelSpec``, or None if it is not an RPA backend.

    The Pallas RPA backends share one kernel shell, so what the platform needs
    to know about them -- the page-size floor, whether they read fp8 KV -- is
    declared on the spec rather than restated here.
    """
    if backend == "pallas_tpu_rpa_batched":
        from sglang_tpu.attention_pallas_rpa_batched import RPA_BATCHED_SPEC as spec
    elif backend == "pallas_tpu_rpa_v3":
        from sglang_tpu.attention_pallas_rpa_v3 import RPA_V3_SPEC as spec
    else:
        return None
    return spec


def _uses_seq_along_lane_kv() -> bool:
    """Ask the backend, so the pool cannot disagree with the kernel."""
    from sglang_tpu.attention_pallas_rpa_batched import (
        uses_seq_along_lane_kv,
    )

    return uses_seq_along_lane_kv()


# Text-only decoders with softmax attention in every layer and no tensor-parallel
# shared expert: the layer types the context-parallel path covers.
_PREFILL_CP_ARCHITECTURES = ("Qwen3MoeForCausalLM", "GptOssForCausalLM")


def _validate_prefill_context_parallel(server_args: ServerArgs) -> None:
    """Reject configurations the TPU prefill context-parallel path cannot run.

    The path shards extend batches by zigzag blocks over the attention
    context-parallel group (= the TP group), all-gathers K/V per layer into
    every rank's own batched RPA pool and reduces the last-token rows before
    the logits step; it exists only inside the compiled forward.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    attn_cp_size = cfg.attn_cp_size
    if attn_cp_size <= 1:
        return
    feature = f"Prefill context parallelism on TPU (--attn-cp-size {attn_cp_size})"
    if not cfg.enable_prefill_cp:
        raise NotImplementedError(f"{feature} requires --enable-prefill-cp.")
    if cfg.cp_strategy != "zigzag":
        raise NotImplementedError(
            f"{feature} requires --cp-strategy zigzag, got {cfg.cp_strategy!r}."
        )
    from sglang.srt.utils.hf_transformers.config import get_config

    hf_config = get_config(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code, revision=cfg.revision
    )
    architectures = list(hf_config.architectures or [])
    if not set(architectures) & set(_PREFILL_CP_ARCHITECTURES):
        raise NotImplementedError(
            f"{feature} supports {' and '.join(_PREFILL_CP_ARCHITECTURES)} models, "
            f"got {architectures}: multimodal, M-RoPE, linear-attention and "
            "state-space models, and MoE layers with a tensor-parallel shared "
            "expert, do not run on the context-parallel path."
        )
    # TODO(yinlin09): lift once gpt-oss with a bf16 KV cache gets through warmup
    # above attention CP 2; on libtpu 0.0.47 a core halts at CP 4 and the first
    # extend stalls at CP 8 (CP 2 and the fp8 cache run).
    if (
        "GptOssForCausalLM" in architectures
        and cfg.kv_cache_dtype in ("bf16", "bfloat16")
        and attn_cp_size > 2
    ):
        raise NotImplementedError(
            f"{feature} with a bf16 KV cache fails in warmup for gpt-oss above "
            "--attn-cp-size 2; keep the default fp8 KV cache or lower the CP size."
        )
    backend = cfg.attention_backend
    if backend is None:
        declare_resolution(
            server_args,
            "tpu.apply_server_args_defaults",
            attention_backend="pallas_tpu_rpa_batched",
        )
    elif backend != "pallas_tpu_rpa_batched":
        raise NotImplementedError(
            f"{feature} requires --attention-backend pallas_tpu_rpa_batched, "
            f"got {backend!r}."
        )
    if _uses_seq_along_lane_kv():
        raise NotImplementedError(
            f"{feature} writes K/V in the token-major fused layout; unset "
            "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV."
        )
    if cfg.speculative_algorithm is not None:
        raise NotImplementedError(f"{feature} does not support speculative decoding.")
    if cfg.attn_dp_size > 1:
        raise NotImplementedError(f"{feature} does not support DP attention.")
    if cfg.moe_dp_size != attn_cp_size:
        raise NotImplementedError(
            f"{feature} requires --moe-dp-size {attn_cp_size} (replicated experts "
            "on each rank's own tokens); expert-parallel MoE over the gathered "
            f"tokens (--ep-size) is not supported, got {cfg.moe_dp_size}."
        )
    if not cfg.enable_torch_compile:
        raise NotImplementedError(f"{feature} requires --enable-torch-compile.")
    if cfg.tpu_compile_tracer != "dynamo":
        raise NotImplementedError(
            f"{feature} requires --tpu-compile-tracer dynamo, got "
            f"{cfg.tpu_compile_tracer!r}."
        )
    if cfg.sampling_backend is None:
        declare_resolution(
            server_args, "tpu.apply_server_args_defaults", sampling_backend="pytorch"
        )
    elif cfg.sampling_backend != "pytorch":
        raise NotImplementedError(
            f"{feature} requires --sampling-backend pytorch (the logits step runs on "
            f"the reduced last-token rows), got {cfg.sampling_backend!r}."
        )
    if envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.get():
        raise NotImplementedError(
            f"{feature} does not support input (prompt) logprobs; unset "
            "SGLANG_TPU_ENABLE_PROMPT_LOGPROB."
        )
    # The layout and the KV slot table are built from the host req_to_token.
    _pin_req_to_token_on_host("prefill context parallelism")
    # TODO(yinlin09): drop once libtpu is fixed. Next to the per-layer K/V
    # all-gathers, the fused MoE's SparseCore gather gives wrong tokens (seen on
    # libtpu 0.0.47; this forward's SparseCore regressions start with 0.0.46).
    _pin_fused_moe_tensorcore_gather("prefill context parallelism")


def _pin_req_to_token_on_host(reason: str) -> None:
    """Force SGLANG_TPU_REQ_TO_TOKEN_ON_HOST on, warning if it was turned off."""
    if not envs.SGLANG_TPU_REQ_TO_TOKEN_ON_HOST.get():
        logger.warning(
            "Overriding SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=0 -> 1: "
            f"{reason} requires host-resident req_to_token."
        )
    envs.SGLANG_TPU_REQ_TO_TOKEN_ON_HOST.set(True)


def _pin_fused_moe_tensorcore_gather(reason: str) -> None:
    """Force SGLANG_TPU_FUSED_MOE_SC_GATHER off, warning if it was turned on."""
    if (
        envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.is_set()
        and envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get()
    ):
        logger.warning(
            "Overriding SGLANG_TPU_FUSED_MOE_SC_GATHER=1 -> 0: "
            f"{reason} gathers the fused-MoE rows on the TensorCore."
        )
    envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.set(False)


def _mha_kv_pool_cls(backend: str | None) -> type:
    """The MHA KV pool class ``backend`` allocates.

    A function of the backend name alone, so server-args validation can ask
    before the runtime context exists; ``get_mha_kv_pool_cls`` delegates here
    so the two cannot disagree. A plugin that routes its backend to its own
    pool patches this function.
    """
    if backend == "pallas_tpu_rpa_batched" and _uses_seq_along_lane_kv():
        # SEQ_ALONG_LANE puts the token index on the lane axis, so the
        # buffer shape differs from the fused one the other backends read.
        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool

        return TpuSeqAlongLaneKVPool

    # RPA v3 and batched use the fused 5D KV layout; pallas_tpu_flash and
    # torch_native read the generic MHA pool.
    if backend in (
        "pallas_tpu_rpa_v3",
        "pallas_tpu_rpa_batched",
    ):
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        return TpuFusedMHATokenToKVPool

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool


def _validate_pipeline_parallel_args(server_args: ServerArgs) -> None:
    """Reject server configurations pipeline parallelism on TPU cannot run,
    and pin the scheduler order the stage hand-off needs.

    Every device is one process, and a stage is one device, a tensor-parallel
    pair (--tp-size 2 with --attn-cp-size 1) or one prefill context-parallel
    group (--tp-size equal to --attn-cp-size). Activations move over
    ICI as one collective-permute launched by every stage in lockstep, so a
    stage's forward waits on launches the other stages issue at the top of
    their next iteration. The scheduler must therefore receive and forward
    the last stage's outputs before it launches a forward, not after: that is
    the async-batch-depth 1 order (depth 0 does the ring exchange after the
    forward and the two waits form a cycle). A stage of several devices runs
    its own collectives over the stage's devices (this needs a torch_tpu that
    compiles a collective for its device group only), while the hand-off
    stays the one collective every stage launches.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    pp = cfg.pp_size
    if cfg.pp_async_batch_depth > 1:
        raise NotImplementedError(
            "Pipeline parallelism on TPU requires --pp-async-batch-depth 1 "
            "so output delivery and activation hand-offs keep the same pace."
        )
    if cfg.pp_async_batch_depth < 1:
        logger.info(
            "Pipeline parallelism on TPU runs with --pp-async-batch-depth 1 "
            "(got %d): the ICI hand-off needs the last stage's outputs "
            "received before each forward.",
            cfg.pp_async_batch_depth,
        )
        declare_resolution(
            server_args, "tpu.apply_server_args_defaults", pp_async_batch_depth=1
        )
    stage_is_tp_pair = (
        cfg.tp_size == 2
        and cfg.attn_cp_size == 1
        and cfg.attn_dp_size == 1
        and cfg.ep_size == 1
    )
    if cfg.tp_size > 1 and cfg.attn_cp_size != cfg.tp_size and not stage_is_tp_pair:
        raise NotImplementedError(
            "Pipeline parallelism on TPU requires --tp-size 1, a tensor-parallel "
            "pair (--tp-size 2 with --attn-cp-size 1, no DP attention or expert "
            "parallelism) or --attn-cp-size equal to --tp-size (got --tp-size "
            f"{cfg.tp_size} with --attn-cp-size {cfg.attn_cp_size} and --pp-size "
            f"{pp})."
        )
    if cfg.dcp_size > 1:
        raise NotImplementedError(
            f"Decode context parallelism (--dcp-size {cfg.dcp_size}) is not "
            f"supported with pipeline parallelism on TPU (--pp-size {pp})."
        )
    if cfg.attn_cp_size > 1:
        if cfg.nnodes > 1:
            # Only because no multi-host run has validated it with standalone
            # stage groups; no technical limit is known.
            raise NotImplementedError(
                "Pipeline stages made of a context-parallel group run on one host "
                f"(got --nnodes {cfg.nnodes})."
            )
        if cfg.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
            raise NotImplementedError(
                "Pipeline stages made of a context-parallel group support only a "
                "bf16 KV cache (--kv-cache-dtype auto, bf16 or bfloat16): with an "
                "fp8 KV cache, Qwen3 served wrong tokens that changed from run to "
                f"run. Got --kv-cache-dtype {cfg.kv_cache_dtype}."
            )
        if cfg.kv_cache_dtype == "auto":
            logger.info(
                "Pipeline stages made of a context-parallel group run with "
                "--kv-cache-dtype bfloat16."
            )
            declare_resolution(
                server_args, "tpu.apply_server_args_defaults", kv_cache_dtype="bfloat16"
            )
    # PD prefill only exports the target's hidden states for the decode node;
    # the draft loop itself never runs here, so the stage relay stays spec-free.
    if cfg.speculative_algorithm is not None and cfg.disaggregation_mode != "prefill":
        raise NotImplementedError(
            "Speculative decoding "
            f"(--speculative-algorithm {cfg.speculative_algorithm}) "
            f"is not supported with pipeline parallelism on TPU (--pp-size {pp}) "
            "outside --disaggregation-mode prefill."
        )
    if cfg.enable_dynamic_chunking:
        raise NotImplementedError(
            "--enable-dynamic-chunking is not supported with pipeline "
            f"parallelism on TPU (--pp-size {pp}): its chunk-latency profiling "
            "relies on a device synchronize that does not wait for TPU "
            "execution."
        )
    if not cfg.enable_torch_compile:
        raise NotImplementedError(
            "Pipeline parallelism on TPU requires --enable-torch-compile "
            f"(--pp-size {pp}): only the compiled forward trims the received "
            "activations to the batch's padded token count."
        )


def _validate_pd_queue_limit(
    *, disaggregation_mode: str, max_queued_requests: int | None
) -> None:
    """Fail the server when a queue length limit is set and PD is enabled."""
    if disaggregation_mode == "null" or max_queued_requests is None:
        return
    raise ValueError(
        "--max-queued-requests is not supported with --disaggregation-mode "
        f"{disaggregation_mode}: the prefill and the decode admit requests "
        "independently, so a queue cap on one side cannot be enforced "
        "consistently (a decode-side refusal would discard a finished "
        "prefill). Remove the flag and shed load at the router."
    )


def _validate_raiden_optimistic_prefill(
    *,
    disaggregation_mode: str,
    transfer_backend: str,
    optimistic_prefill_attempts: int,
) -> None:
    if (
        disaggregation_mode == "null"
        or transfer_backend != "raiden"
        or optimistic_prefill_attempts == 0
    ):
        return
    raise ValueError(
        "--optimistic-prefill-attempts must be 0 with "
        "--disaggregation-transfer-backend raiden "
        f"(got {optimistic_prefill_attempts}). A raiden prefill must not compute "
        "a request before its decode has reserved the KV pages for it. "
        "Optimistic prefill computes it earlier and parks its KV, which under "
        "load fills the prefill pool with requests the decodes have no pages "
        "for and deadlocks prefill and decode."
    )


def _validate_kv_checksum(enable_kv_checksum: bool) -> None:
    if not enable_kv_checksum:
        return
    raise ValueError(
        "--disaggregation-enable-kv-checksum is not supported on TPU: the KV "
        "checksum has only a CUDA implementation. On TPU the server would pass "
        "its health checks and then fail every rank on the first real request. "
        "Remove the flag from both the prefill and the decode engines."
    )


def _apply_page_size_default(server_args) -> None:
    """Resolve ``page_size`` against the attention backend's declared floor.

    A backend whose kernel cannot address small pages -- the SEQ_ALONG_LANE
    path tiles the page axis onto lanes and needs a multiple of the 128-lane
    width -- must not silently receive the generic default, or it fails deep
    inside the Pallas trace with a shape error that reads like a kernel bug.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    # The backend may be a declaration (record left pristine); read it back through the stash.
    cfg = resolving_view(server_args)
    spec = _rpa_kernel_spec(cfg.attention_backend)
    floor = spec.min_page_size if spec is not None else None
    if cfg.page_size is None:
        default = max(_DEFAULT_PAGE_SIZE, floor or 0)
        declare_resolution(
            server_args, "tpu.apply_server_args_defaults", page_size=default
        )
        return
    if floor is not None and cfg.page_size < floor:
        raise ValueError(
            f"--attention-backend {cfg.attention_backend} needs "
            f"--page-size >= {floor}, got {cfg.page_size}."
        )


def _apply_kv_cache_dtype_default(server_args) -> None:
    """Default the KV cache to fp8 for the Pallas RPA/DSA backends.

    Those kernels dequantize KV on read, and at the head counts TPU serving
    runs (tensor parallel leaves one KV head per rank) fp8 halves KV bytes per
    token. That doubles the context or the batch that fits in HBM, which is the
    binding constraint at long context -- decode there is bound by streaming
    KV, not by arithmetic. "auto" (model dtype) is the generic SGLang default
    and gives up both.

    Only those backends: ``pallas_tpu_mla`` accepts fp8 KV but stays opt-in
    until its accuracy is validated, and ``pallas_tpu_flash`` has no
    dequantization scale, so defaulting it to fp8 silently degrades it.
    """
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    # Same as above: the RPA default backend is a declaration, not a field.
    cfg = resolving_view(server_args)
    if cfg.kv_cache_dtype != "auto":
        return
    if (
        _rpa_kernel_spec(cfg.attention_backend) is None
        and cfg.attention_backend != "pallas_tpu_mla_sparse"
    ):
        return
    declare_resolution(
        server_args, "tpu.apply_server_args_defaults", kv_cache_dtype="fp8_e4m3"
    )


def _validate_dsa(server_args) -> None:
    """Reject the sparse-attention (DSA) setups that TPU cannot serve correctly."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.configs.model_config import is_deepseek_dsa
    from sglang.srt.utils.hf_transformers.config import get_config

    cfg = resolving_view(server_args)
    try:
        hf_config = get_config(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            revision=cfg.revision,
            model_override_args=json.loads(cfg.json_model_override_args or "{}"),
        )
    except Exception:
        # A model that cannot be loaded here fails later with its real error.
        return
    if not is_deepseek_dsa(hf_config):
        return
    if cfg.attention_backend != "pallas_tpu_mla_sparse":
        raise ValueError(
            "Sparse attention (DSA) on TPU needs --attention-backend "
            f"pallas_tpu_mla_sparse, got {cfg.attention_backend!r}."
        )
    # With these two settings the GPU indexer always keeps the first
    # index_init_tokens and the last index_local_tokens positions. The TPU
    # top-k kernel cannot do that yet, and ignoring them would change the
    # selection without any error.
    # TODO(sglang-torchtpu#1072): support them on TPU.
    init_tokens = getattr(hf_config, "index_init_tokens", 0)
    local_tokens = getattr(hf_config, "index_local_tokens", 0)
    if init_tokens or local_tokens:
        raise ValueError(
            "DSA on TPU does not support index_init_tokens or index_local_tokens "
            f"yet; the checkpoint sets them to {init_tokens} and {local_tokens}."
        )
    # The direct tracer gets the KV cache buffers from the pool's get_state and
    # set_state. TpuSplitMLATokenToKVPool does not implement them for its own
    # buffers, so the first compile would fail.
    # TODO(sglang-torchtpu#1072): implement them and run DSA with the direct
    # tracer.
    if cfg.tpu_compile_tracer != "dynamo":
        raise ValueError(
            "DSA on TPU needs --tpu-compile-tracer dynamo, got "
            f"{cfg.tpu_compile_tracer!r}."
        )


def _register_attention_backends() -> None:
    """Add the attention backends that only this package knows to the registry."""
    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
        register_draft_attention_backend,
    )
    from sglang_tpu.attention_pallas_rpa_base import (
        TpuPallasRpaMultiStepDraftBackend,
    )

    @register_draft_attention_backend("pallas_tpu_rpa_v3")
    def create_pallas_tpu_rpa_v3_draft_backend(runner, topk, speculative_num_steps):
        from sglang_tpu.attention_pallas_rpa_v3 import TpuPallasRpaV3AttnBackend

        return TpuPallasRpaMultiStepDraftBackend(
            runner, topk, speculative_num_steps, TpuPallasRpaV3AttnBackend
        )

    @register_draft_attention_backend("pallas_tpu_rpa_batched")
    def create_pallas_tpu_rpa_batched_draft_backend(
        runner, topk, speculative_num_steps
    ):
        from sglang_tpu.attention_pallas_rpa_batched import (
            TpuPallasRpaBatchedAttnBackend,
        )

        return TpuPallasRpaMultiStepDraftBackend(
            runner, topk, speculative_num_steps, TpuPallasRpaBatchedAttnBackend
        )

    @register_draft_attention_backend("pallas_tpu_mla")
    def create_pallas_tpu_mla_draft_backend(runner, topk, speculative_num_steps):
        from sglang_tpu.attention_pallas_mla import TpuPallasMlaAttnBackend

        return TpuPallasRpaMultiStepDraftBackend(
            runner, topk, speculative_num_steps, TpuPallasMlaAttnBackend
        )

    @register_draft_attention_backend("pallas_tpu_mla_sparse")
    def create_pallas_tpu_mla_sparse_draft_backend(runner, topk, speculative_num_steps):
        from sglang_tpu.attention_pallas_mla_sparse import (
            TpuPallasSparseMlaAttnBackend,
        )

        return TpuPallasRpaMultiStepDraftBackend(
            runner, topk, speculative_num_steps, TpuPallasSparseMlaAttnBackend
        )

    @register_attention_backend("pallas_tpu_mla_sparse")
    def create_pallas_tpu_mla_sparse_backend(runner):
        from sglang_tpu.attention_pallas_mla_sparse import (
            TpuPallasSparseMlaAttnBackend,
        )

        return TpuPallasSparseMlaAttnBackend(runner)

    @register_attention_backend("dsv4")
    def create_tpu_dsv4_backend(runner):
        from sglang_tpu.dsv4 import DSV4TPUAttnBackend

        return DSV4TPUAttnBackend(runner)


def _register_models() -> None:
    """Add the model implementations that only this package provides to the registry."""
    from sglang.srt.models.registry import ModelRegistry

    # A module that fails to import is logged and skipped, so it can't take the
    # other models down with it; an arch core already serves raises.
    ModelRegistry.register("sglang_tpu.models")


def _resolve_num_hidden_layers(server_args) -> int:
    """Layer count from the model's HF config (lru-cached load), 0 when
    unresolvable — an unloadable model fails later with its real error, and
    the prefetch containment stays off rather than guessing."""
    try:
        from sglang.srt.utils.hf_transformers.config import get_config

        hf_config = get_config(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        return int(getattr(hf_config, "num_hidden_layers", 0))
    except Exception:
        return 0


# The attention backend TPU decode context parallelism runs on.
_DCP_ATTENTION_BACKEND = "pallas_tpu_rpa_batched"

# The chunk handle_gpu_memory_settings declares later for an unset
# --chunked-prefill-size on TPU, whose device memory capacity it does not know.
_UNSET_CHUNKED_PREFILL_SIZE = 4096


def _validate_decode_context_parallel(server_args) -> None:
    """Refuse the server-arg combinations TPU decode context parallelism does
    not implement.

    DCP shards each request's KV by absolute token position across a slice of
    the TP group and merges the per-rank partial attention by log-sum-exp. It
    runs on the batched RPA backend with a host-resident ``req_to_token`` and a
    KV cache no other component reads -- every other attention backend,
    parallelism mode and KV consumer is refused.
    """
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.environ import envs

    cfg = resolving_view(server_args)
    dcp_size = cfg.dcp_size
    if dcp_size <= 1:
        return

    if cfg.attention_backend != _DCP_ATTENTION_BACKEND:
        raise ValueError(
            "--decode-context-parallel-size > 1 on TPU requires "
            f"--attention-backend {_DCP_ATTENTION_BACKEND}, got "
            f"{cfg.attention_backend!r}."
        )
    if cfg.tp_size % dcp_size != 0:
        raise ValueError(
            f"--tensor-parallel-size ({cfg.tp_size}) must be divisible by "
            f"--decode-context-parallel-size ({dcp_size})."
        )
    if cfg.pp_size > 1:
        raise ValueError(
            "--pipeline-parallel-size > 1 is not supported with "
            f"--decode-context-parallel-size > 1 on TPU (got pp_size={cfg.pp_size})."
        )
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "--speculative-algorithm is not supported with "
            "--decode-context-parallel-size > 1 on TPU (got "
            f"{cfg.speculative_algorithm!r})."
        )
    if cfg.enable_mixed_chunk:
        raise ValueError(
            "--enable-mixed-chunk is not supported with "
            "--decode-context-parallel-size > 1 on TPU: the DCP attention runs "
            "plain decode and extend batches only."
        )
    if cfg.attn_dp_size > 1:
        raise ValueError(
            "DP attention (--attn-dp-size > 1) is not supported with "
            "--decode-context-parallel-size > 1 on TPU."
        )
    if cfg.attn_cp_size > 1:
        raise ValueError(
            "--attention-context-parallel-size > 1 is not supported with "
            "--decode-context-parallel-size > 1 on TPU (got "
            f"attn_cp_size={cfg.attn_cp_size})."
        )
    if cfg.enable_prefill_cp:
        raise ValueError(
            "--enable-prefill-cp is not supported with "
            "--decode-context-parallel-size > 1 on TPU."
        )
    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "--disaggregation-mode is not supported with "
            "--decode-context-parallel-size > 1 on TPU (got "
            f"{cfg.disaggregation_mode!r})."
        )
    if cfg.enable_hierarchical_cache:
        raise ValueError(
            "--enable-hierarchical-cache is not supported with "
            "--decode-context-parallel-size > 1 on TPU."
        )
    if cfg.dcp_comm_backend != "ag_rs":
        raise ValueError(
            "--dcp-comm-backend must be 'ag_rs' with "
            "--decode-context-parallel-size > 1 on TPU: the merge gathers the "
            "LSEs and sums the outputs with zero-padded all-reduces, got "
            f"{cfg.dcp_comm_backend!r}."
        )
    if cfg.dcp_replicate_q_proj:
        raise ValueError(
            "--dcp-replicate-q-proj is not supported with "
            "--decode-context-parallel-size > 1 on TPU: it only applies to the "
            "a2a/fi_a2a exchange on MLA attention."
        )
    if not cfg.enable_torch_compile:
        raise ValueError(
            "--enable-torch-compile is required with "
            "--decode-context-parallel-size > 1 on TPU."
        )
    if cfg.tpu_compile_tracer != "dynamo":
        raise ValueError(
            "--tpu-compile-tracer must be 'dynamo' with "
            "--decode-context-parallel-size > 1 on TPU (got "
            f"{cfg.tpu_compile_tracer!r})."
        )
    if not envs.SGLANG_TPU_REQ_TO_TOKEN_ON_HOST.get():
        raise ValueError(
            "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=0 is not supported with "
            "--decode-context-parallel-size > 1 on TPU: the DCP attention "
            "metadata is built from a host-resident req_to_token."
        )
    if _uses_seq_along_lane_kv():
        raise ValueError(
            "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1 is not supported with "
            "--decode-context-parallel-size > 1 on TPU: the DCP KV write "
            "targets the token-major fused KV cache."
        )
    if cfg.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError(
            "--decode-context-parallel-size > 1 on TPU supports only a bf16 KV "
            "cache (--kv-cache-dtype bfloat16), got --kv-cache-dtype "
            f"{cfg.kv_cache_dtype}."
        )
    chunk = cfg.chunked_prefill_size
    if chunk is None:
        chunk = _UNSET_CHUNKED_PREFILL_SIZE
    if 0 < chunk < cfg.page_size * dcp_size:
        raise ValueError(
            f"--chunked-prefill-size ({chunk}) must be at least --page-size x "
            f"--decode-context-parallel-size ({cfg.page_size} x {dcp_size} = "
            f"{cfg.page_size * dcp_size}) on TPU: the DCP allocator hands out "
            "pages of that many tokens and the scheduler cuts prefill chunks at "
            "page boundaries, so a smaller chunk is never scheduled."
        )
    # TODO(yinlin09): lift once DCP4/DCP8 pass the e2e gate on the pinned stack;
    # DCP8 served on torch_tpu 0.1.1.dev20260914 with libtpu 0.0.47.
    if dcp_size > 2:
        raise ValueError(
            "--decode-context-parallel-size must be 1 or 2 on TPU, got "
            f"{dcp_size}: with more ranks the forward stalls in warmup with the "
            "SparseCore gather offloads off and returns wrong first tokens with "
            "them on."
        )


def _experimental_spec_enabled() -> bool:
    # Lazy: sglang_tpu.experimental imports spec_info, which loads after the platform.
    from sglang_tpu.experimental import experimental_spec_enabled

    return experimental_spec_enabled()


class TpuSRTPlatform(SRTPlatform):
    """SGLang SRT platform for Google Cloud TPUs via torch_tpu."""

    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    supported_quantization: list[str] = []

    def get_traced_rank(
        self, rank: int, *, group_size: int | None = None
    ) -> torch.Tensor | None:
        return traced_rank(rank, group_size=group_size)

    def slice_tensor_for_rank(
        self, tensor: torch.Tensor, dim: int, parts: int, rank: int
    ) -> torch.Tensor:
        return rank_slice(tensor, dim, parts, rank)

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args: ServerArgs) -> None:
        # Pre-publish resolution declares rather than writes, so every read below goes
        # through the live declaration view, this hook's own declarations included.
        from sglang.srt.arg_groups.overrides import (
            declare_resolution,
            resolving_view,
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            with_phase,
        )
        from sglang_tpu.kimi_k3 import check_static_fp8_spec_batch, static_fp8_parts
        from sglang_tpu.patches import _prepare_tpu_env

        # Fail on a mistyped SGLANG_TPU_K3_STATIC_FP8 now, not after the
        # weights have loaded (its binder runs post-load).
        parts = static_fp8_parts()
        cfg = resolving_view(server_args)
        if "latent_up" in parts:
            check_static_fp8_spec_batch(
                parts,
                speculative_algorithm=cfg.speculative_algorithm,
                max_running_requests=cfg.max_running_requests,
            )
        if cfg.pp_size > 1:
            _validate_pipeline_parallel_args(server_args)
        _validate_raiden_optimistic_prefill(
            disaggregation_mode=cfg.disaggregation_mode,
            transfer_backend=cfg.disaggregation_transfer_backend,
            optimistic_prefill_attempts=cfg.optimistic_prefill_attempts,
        )
        _validate_pd_queue_limit(
            disaggregation_mode=cfg.disaggregation_mode,
            max_queued_requests=cfg.max_queued_requests,
        )
        _validate_kv_checksum(cfg.disaggregation_enable_kv_checksum)

        # Under Ray the slicebuilder/topology env is built per-rank after the
        # cluster's real node IPs are discovered (multihost: per-actor from Ray
        # IPs in _prepare_tpu_env; single-host: seeded once in
        # RayEngine._launch_scheduler_processes). Building it here -- in the
        # orchestrator, before discovery -- would emit a bogus single-host
        # (localhost) env that never applies to the actual ranks. Still apply
        # the libtpu/env overrides now so actors inherit them before importing
        # torch_tpu.
        # One device per scheduler process across all pipeline stages.
        _prepare_tpu_env(cfg.tp_size * cfg.pp_size, setup_slicebuilder=not cfg.use_ray)

        from sglang_tpu.env_override import (
            apply_decode_cp_sparse_core_override,
            apply_prefill_cp_sparse_core_override,
            apply_small_kv_pool_vmem_prefetch_override,
        )

        apply_small_kv_pool_vmem_prefetch_override(
            max_total_tokens=cfg.max_total_tokens,
            num_hidden_layers=_resolve_num_hidden_layers(cfg),
        )
        apply_decode_cp_sparse_core_override(dcp_size=cfg.dcp_size)

        if cfg.dtype is None or cfg.dtype == "auto":
            declare_resolution(
                server_args, "tpu.apply_server_args_defaults", dtype="bfloat16"
            )

        _validate_prefill_context_parallel(server_args)
        apply_prefill_cp_sparse_core_override(attn_cp_size=cfg.attn_cp_size)
        # Before the default backend below, so the error names what was passed.
        _validate_dsa(server_args)

        # Decode context parallelism runs on one backend only, so an unset
        # backend resolves to it instead of the platform default.
        if cfg.attention_backend is None and cfg.dcp_size > 1:
            declare_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                attention_backend=_DCP_ATTENTION_BACKEND,
            )
        # Decode context parallelism supports only a bf16 KV cache, so an unset
        # KV dtype resolves to it instead of the RPA backends' fp8 default.
        if cfg.dcp_size > 1 and cfg.kv_cache_dtype == "auto":
            declare_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                kv_cache_dtype="bfloat16",
            )
        if cfg.attention_backend is None:
            declare_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                attention_backend=self.get_default_attention_backend(),
            )

        if cfg.sampling_backend is None:
            declare_resolution(
                server_args,
                "tpu.apply_server_args_defaults",
                sampling_backend="tpu",
            )

        # Graph configuration is resolved before platform defaults. Updating the
        # legacy flag alone cannot disable the already-resolved phase runners.
        graph_config = cfg.cuda_graph_config
        for phase in Phase.ALL:
            graph_config = with_phase(graph_config, phase, backend=Backend.DISABLED)
        declare_resolution(
            server_args,
            "tpu.apply_server_args_defaults",
            disable_cuda_graph=True,
            cuda_graph_config=graph_config,
        )
        # ``enable_torch_compile`` is honored on TPU and wired up by
        # ``TpuGraphRunner`` (see ``ModelRunner.init_device_graphs``). Off by
        # default (the SGLang-wide default); opt in with ``--enable-torch-compile``.
        # TPU runs SGLang's regular event-based overlap scheduling, minus the
        # CUDA-style forward/copy/schedule streams: torch_tpu serialises on a
        # single PJRT queue so cross-stream ordering is a no-op (see
        # ``Scheduler.init_overlap``'s ``use_overlap_streams``). Overlap comes
        # from PJRT's async dispatch and a scoped ``torch.tpu.Event`` D2H fence.
        # Leave it enabled by default; users can opt out via
        # ``--disable-overlap-schedule``.

        if cfg.mem_fraction_static is None:
            declare_resolution(
                server_args, "tpu.apply_server_args_defaults", mem_fraction_static=0.40
            )

        # A TPU DMA backend's load-back keeps the reloaded prefix's slot indices
        # on CPU (TpuDMAHiCacheController.load), so req_to_token must stay
        # host-resident: a device-resident req_to_token would put the other
        # nodes' prefix indices on device and mix devices in the prefix cat. Pin
        # SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1 for every engine.
        if is_tpu_dma_hicache_backend(cfg.hicache_io_backend):
            _pin_req_to_token_on_host(f"--hicache-io-backend {cfg.hicache_io_backend}")

        # There is no CUDA coredump on TPU: on a crash the handler would trigger
        # a no-op dump, then sleep SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS
        # (60 s) with the dead server still up. An explicit setting still wins.
        os.environ.setdefault("SGLANG_CUDA_COREDUMP_BEFORE_CRASH", "0")

        # The default TPU backend (Pallas RPA v3) expects a paged KV cache where
        # page_size is the HBM<->VMEM DMA chunk and the fused KV buffer's page
        # axis. SGLang's global page_size=1 default is valid for token-granular
        # GPU kernels, but is inefficient here: it makes page_indices
        # token-granular and turns KV fetches into many tiny DMAs.
        # Builtin TPU speculative decoding supports linear EAGLE/MTP and DFLASH.
        # Registered algorithms own their validation and backend integration.
        spec_algo = cfg.speculative_algorithm
        if spec_algo is not None:
            if spec_algo in ("EAGLE", "EAGLE3", "NEXTN"):
                if cfg.speculative_use_rejection_sampling:
                    raise NotImplementedError(
                        "EAGLE-family rejection sampling is not supported on TPU yet."
                    )
                if (cfg.speculative_eagle_topk or 1) > 1:
                    raise NotImplementedError(
                        "EAGLE-family topk > 1 is not supported on TPU yet."
                    )
            elif spec_algo == "DFLASH":
                pass
            elif spec_algo == "DSPARK" and _experimental_spec_enabled():
                # Served by sglang_tpu.experimental.TpuBlockSpecWorker.
                pass
            else:
                from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

                # Unknown names may be registered by a plugin after platform
                # defaults run. The shared speculative registry validates them
                # later; only reject builtins the TPU backend knows it cannot run.
                if spec_algo.upper() in SpeculativeAlgorithm.__members__:
                    raise NotImplementedError(
                        f"Speculative algorithm {spec_algo!r} is not supported on TPU "
                        "yet (only EAGLE, EAGLE3, NEXTN and DFLASH among builtin algorithms)."
                    )

        # Platform defaults run after handle_deprecated_dp_attention, so the
        # deprecated --dp-size N --enable-dp-attention spelling has already
        # resolved to attn_dp_size=N here.
        attn_dp_size = cfg.attn_dp_size
        if attn_dp_size > 1:
            attn_tp_size = cfg.tp_size // attn_dp_size
            if attn_tp_size > 1 and not cfg.enable_dp_lm_head:
                raise ValueError(
                    f"When attention TP > 1 under DP attention on TPU (attn_tp_size={attn_tp_size} "
                    f"from tp_size={cfg.tp_size} and attn_dp_size={attn_dp_size}), "
                    "--enable-dp-lm-head must be enabled."
                )
            if (
                not cfg.enable_dp_lm_head
                and spec_algo in ("EAGLE", "NEXTN", "DFLASH", "DSPARK")
                and _experimental_spec_enabled()
            ):
                # These drafts share the target lm_head but run in the attention-TP
                # group; a head sharded over the full TP group would draft from one shard.
                raise ValueError(
                    f"The experimental TPU spec worker serves {spec_algo} under DP "
                    "attention only with --enable-dp-lm-head, which shards the shared "
                    "lm_head over the draft's attention-TP group."
                )

        _apply_page_size_default(server_args)
        # After the page-size default: the DCP chunk check reads the page size.
        _validate_decode_context_parallel(server_args)
        _apply_kv_cache_dtype_default(server_args)

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        # Three Pallas backends are registered for TPU:
        #   - "pallas_tpu_rpa_v3" (default): ragged_paged_attention v3 kernel
        #     in `pallas/ragged_paged_attention_v3.py`. Single unified kernel
        #     for prefill + decode, fused KV cache (interleaved K/V, packed),
        #     decode-native batched call.
        #   - "pallas_tpu_rpa_batched": the batched RPA kernel in
        #     `pallas/batched_rpa`, a thin subclass of the same shared base as
        #     v3 (see `attention_pallas_rpa_base`). Same fused KV cache, batches
        #     sequences for higher throughput. Supports EAGLE3 speculative
        #     decoding through the shared RPA draft-backend implementation.
        #   - "pallas_tpu_flash": flash_attention for prefill/extend +
        #     `jax.experimental.pallas.ops.tpu.paged_attention` for decode.
        #     Original implementation from PR #5, works against the flat
        #     `MHATokenToKVPool` layout (no fused-pool swap required).
        # All are listed in `server_args.ATTENTION_BACKEND_CHOICES` and
        # registered in `layers.attention.attention_registry` (same place as
        # `ascend` / `intel_xpu`). Factories use lazy imports so non-TPU runs
        # don't pull in jax/torch_tpu.
        # Override to the reference path with `--attention-backend torch_native`.
        return "pallas_tpu_rpa_v3"

    def get_graph_runner_cls(self) -> type:
        from sglang_tpu.graph_runner.tpu_graph_runner import TpuGraphRunner

        return TpuGraphRunner

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:
        if quantization == "modelopt_fp4":
            from sglang.srt.layers.quantization.modelopt_quant import (
                TpuModelOptCompatibleFp4Config,
            )

            return TpuModelOptCompatibleFp4Config
        if quantization == "modelopt_mixed":
            from sglang.srt.layers.quantization.modelopt_quant import (
                TpuModelOptMixedPrecisionConfig,
            )

            return TpuModelOptMixedPrecisionConfig
        return None

    def get_mha_kv_pool_cls(self) -> type:
        # Pick per the resolved backend: the raw field is None when --attention-backend is omitted.
        from sglang.srt.runtime_context import get_exec

        return _mha_kv_pool_cls(get_exec().kernel.attention_backend)

    def get_mha_kv_bytes_per_token(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        kv_cache_dtype: torch.dtype,
    ) -> Optional[int]:
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        pool_cls = self.get_mha_kv_pool_cls()
        if not issubclass(pool_cls, TpuFusedMHATokenToKVPool):
            return None
        return pool_cls.kv_bytes_per_token_per_layer(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            store_dtype=kv_cache_dtype,
        )

    supports_mamba_extra_buffer = True

    def supports_mamba_prefix_cache(self, model_config) -> bool:
        from sglang.srt.configs.hybrid_arch import hybrid_gdn_config

        # KDA shares GDN metadata setup, but does not pass snapshot tracking
        # to its kernel yet. Only the GDN backend writes reusable checkpoints.
        return hybrid_gdn_config(model_config) is not None

    def validate_mamba_prefix_cache(self, view, hf_config=None) -> None:
        # Mixed-batch merging drops the tracking plan while requests retain
        # checkpoint depths, so the radix tree could cache unwritten states.
        if view.enable_mixed_chunk and not view.disable_radix_cache:
            raise ValueError(
                "TPU GDN prefix caching does not support --enable-mixed-chunk. "
                "Remove --enable-mixed-chunk or use --disable-radix-cache."
            )
        # Lazy slot allocation/release has not been validated with TPU snapshots.
        if view.mamba_radix_cache_strategy == "extra_buffer_lazy":
            raise ValueError(
                "extra_buffer_lazy is not supported for TPU GDN prefix caching. "
                "Use --mamba-radix-cache-strategy extra_buffer."
            )
        # Host and int8 stores expect stacked GPU-layout states; TPU owns
        # independent kernel-layout tensors per layer.
        for flag in ("enable_hierarchical_cache", "enable_int8_mamba_checkpoint"):
            if getattr(view, flag):
                raise ValueError(
                    f"--{flag.replace('_', '-')} is not supported for TPU GDN prefix caching. "
                    "Use device-resident, uncompressed Mamba states."
                )
        if hf_config is not None:
            text_config = getattr(hf_config, "text_config", hf_config)
            assert getattr(text_config, "mamba_chunk_size", 64) == 64, (
                "TPU GDN prefix caching requires mamba_chunk_size=64."
            )

    def get_mamba_pool_cls(self) -> type:
        # Hybrid GDN models (Qwen3.5 / Qwen3-Next) store their recurrent state
        # in a TPU-native, Pallas-kernel-layout pool.
        from sglang_tpu.mamba_pool import TpuMambaPool

        return TpuMambaPool

    def get_hicache_explicit_size_context(self) -> AbstractContextManager[None]:
        from sglang.srt.mem_cache.pool_host.base import (
            host_memory_budget_bytes,
            host_memory_budget_scope,
        )
        from sglang.srt.runtime_context import get_memory

        if not is_tpu_dma_hicache_backend(get_memory().hicache_io_backend):
            return super().get_hicache_explicit_size_context()
        # Full, SWA, and draft pools must fit within one combined RAM budget.
        return host_memory_budget_scope(host_memory_budget_bytes())

    def get_mha_kv_pool_host_cls(self, device_pool) -> type:
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.runtime_context import get_memory
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang_tpu.kv_pool_host import (
            TpuFusedMHATokenToKVPoolHost,
            tpu_dma_host_kv_pool_cls,
        )

        # TPU DMA backends offload through their own host memory instead of the
        # torch-copy fused host pool; native (torch_tpu) keeps the fused pool.
        io_backend = get_memory().hicache_io_backend
        if is_tpu_dma_hicache_backend(io_backend):
            return tpu_dma_host_kv_pool_cls(io_backend)

        if isinstance(device_pool, TpuFusedMHATokenToKVPool):
            return TpuFusedMHATokenToKVPoolHost
        return MHATokenToKVPoolHost

    def get_mla_kv_pool_cls(self) -> type:
        # pallas_tpu_mla uses the paged fused latent layout the MLA v2 kernel
        # addresses; other backends (torch_native reference) keep the flat
        # upstream pool.
        from sglang.srt.runtime_context import get_exec

        if get_exec().kernel.attention_backend == "pallas_tpu_mla":
            from sglang_tpu.kv_pool import (
                TpuFusedMLATokenToKVPool,
            )

            return TpuFusedMLATokenToKVPool

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

        return TpuSplitMLATokenToKVPool

    def get_zigzag_cp_strategy_cls(self) -> type:
        from sglang_tpu.cp_strategy import TpuZigzagCPStrategy

        return TpuZigzagCPStrategy

    def runs_eager_extend_cp(self) -> bool:
        # The compiled forward shards the context-parallel extend.
        return False

    def get_paged_allocator_cls(self) -> type:
        from sglang_tpu.allocator_tpu import (
            TpuPagedTokenToKVPoolAllocator,
        )

        return TpuPagedTokenToKVPoolAllocator

    def get_compile_backend(self, mode: str | None = None) -> str:
        return "tpu"

    def get_batch_input_device(self, batch) -> str | None:
        """Keep eligible prefill inputs on CPU until backend padding."""
        from sglang.srt.layers.cp.base import get_cp_strategy
        from sglang.srt.utils.common import print_warning_once
        from sglang_tpu.allocator_tpu import host_return_allocators
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.host_padding import _can_stage_forward_metadata

        if not tpu_envs.SGLANG_TPU_HOST_PADDING.get():
            return None
        # TBO, hisparse and beam search have no TPU path, so they need no opt-out.
        allocator = batch.token_to_kv_pool_allocator
        checks = (
            ("forward metadata is not staged", _can_stage_forward_metadata(batch)),
            ("context parallelism is on", get_cp_strategy() is None),
            ("the allocator returns device indices", host_return_allocators(allocator)),
            # Page size 1 uses alloc(), which has no host-return contract.
            ("page size is 1", allocator.page_size > 1),
            (
                "req_to_token is on the device",
                batch.req_to_token_pool.req_to_token.device.type == "cpu",
            ),
        )
        for reason, ok in checks:
            if not ok:
                print_warning_once(
                    "SGLANG_TPU_HOST_PADDING is set, but prefill inputs upload at "
                    f"the live length because {reason}. This may compile a program "
                    "per prompt length."
                )
                return None
        return "cpu"

    def supports_torch_compile(self) -> bool:
        return True

    def get_piecewise_backend_cls(self) -> type:
        raise NotImplementedError("TPU piecewise backend not yet implemented.")

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        return False

    def supports_speculative_dp_attention(self, algorithm: str) -> bool:
        # Every experimental TpuSpecWorker serves DP attention; core's DFlash worker does not.
        # Lazy: sglang_tpu.experimental imports spec_info, which loads after the platform.
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.experimental import is_experimental_spec

        return is_experimental_spec(SpeculativeAlgorithm.from_string(algorithm))

    def is_pin_memory_available(self, device=None) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def supports_decode_context_parallel(self) -> bool:
        return True

    def dcp_shards_every_kv_pool(self) -> bool:
        # The batched RPA DCP path addresses every pool through the widened
        # location space and replicates K/V heads within a DCP group.
        return True

    def supports_moe_dp_pipeline_stages(self) -> bool:
        return True

    def keeps_swa_index_mapping_on_host(self) -> bool:
        # SWATokenToKVPoolAllocator keeps the mapping on the host, and the
        # host-resident req_to_token table hands its segment frees host indices.
        return True

    def supports_overlap_streams(self) -> bool:
        # torch_tpu serialises all device work on a single implicit PJRT queue:
        # cross-stream ordering (wait_stream / wait_event) is a no-op and a
        # non-default stream only relabels async-copy fences. Overlap comes from
        # async PJRT dispatch + a scoped torch.tpu.Event D2H fence instead. Flip
        # to True once torch_tpu exposes real multi-stream execution.
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        if getattr(self, "_backend_initialized", False):
            return

        world_size = os.environ.get("WORLD_SIZE")
        if world_size is None:
            raise RuntimeError(
                "TorchTPU backend initialization requires worker WORLD_SIZE. "
                "Initialize it from the scheduler worker bootstrap."
            )

        # Keep c10d_functional collectives inside the compiled graph. torch_tpu
        # defaults to materializing collective tensors by forcing graph breaks, which
        # crashes under torch.compile(fullgraph=True) at TP>1. Setting this to 0/false
        # disables that guardrail for SPMD TP graphs.
        os.environ.setdefault(
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false"
        )
        # Allow legacy torch.tpu device APIs (e.g. set_device, is_available,
        # synchronize) called by SGLang's generic runtime bootstrap.
        os.environ.setdefault("TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS", "1")

        # Size libtpu's premapped (pinned, DMA-mapped) host staging pool. This is
        # read by libtpu at init, so it MUST be set before `import torch_tpu`.
        os.environ.setdefault(
            "TPU_PREMAPPED_BUFFER_SIZE", str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE)
        )
        premapped_bytes = int(os.environ["TPU_PREMAPPED_BUFFER_SIZE"])
        # Log the effective size so it's in hand when debugging KV-offload OOMs.
        logger.info(
            "libtpu premapped host staging pool: %.2f GiB "
            "(TPU_PREMAPPED_BUFFER_SIZE=%d); raise it if KV-offload H2D OOMs.",
            premapped_bytes / (1024 * 1024 * 1024),
            premapped_bytes,
        )

        from sglang_tpu.patches import (
            _prepare_tpu_env,
            apply_tpu_patches,
        )

        # WORLD_SIZE is authoritative only inside a scheduler worker. Validate
        # the inherited launch environment here; any TorchTPU-dependent
        # topology discovery is therefore confined to the worker.
        _prepare_tpu_env(int(world_size), resolve_single_host_topology=True)

        from sglang_tpu.compile_cache import (
            add_tpu_kernel_inputs_to_compile_cache_key,
        )

        # After _prepare_tpu_env: the key must see the final LIBTPU_INIT_ARGS.
        add_tpu_kernel_inputs_to_compile_cache_key()

        import torch_tpu  # noqa: F401

        apply_tpu_patches()
        # Resolve after worker setup, before Dynamo can trace either helper.
        rank_slice._load()
        traced_rank._load()
        _register_attention_backends()
        from sglang_tpu.envs import envs as tpu_envs

        if tpu_envs.SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL.get():
            from sglang_tpu.fp8_linear import register_fp8_linear_small_m_op

            register_fp8_linear_small_m_op()
        # Set before registering: importing the model registry imports model_runner,
        # which calls init_backend again on the OOT path.
        self._backend_initialized = True
        _register_models()

    def on_warmup_complete(self) -> None:
        """Freeze torch.compile so serving-time recompiles raise."""
        from sglang.srt.runtime_context import get_exec, get_spec
        from sglang_tpu.compilation.recompilation_checker import (
            maybe_enable_site_tracing,
        )

        # Dispatch modes are thread-local; this runs on the scheduler thread.
        maybe_enable_site_tracing()
        if not get_exec().graph.enable_torch_compile:
            return
        if (
            get_spec().speculative_algorithm is not None
            and envs.SGLANG_TPU_ALLOW_SPEC_RECOMPILE.get()
        ):
            logger.info(
                "[tpu] Warmup complete; leaving torch.compile recompiles enabled "
                "for speculative decoding because "
                "SGLANG_TPU_ALLOW_SPEC_RECOMPILE is enabled."
            )
            return
        torch.compiler.set_stance("fail_on_recompile")
        logger.info(
            "[tpu] Warmup complete; froze torch.compile via "
            "set_stance('fail_on_recompile')."
        )

    def get_compilation_cache_misses(self) -> Optional[int]:
        """Return tier-1 misses, including compiles and persistent-cache loads."""
        get_cache_misses = getattr(
            getattr(torch, "tpu", None), "_get_cache_misses", None
        )
        return None if get_cache_misses is None else int(get_cache_misses())

    def prepare_model_parallel_groups(self) -> None:
        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.pp_stage_chips import (
            validate_chips_for_pp,
        )
        from sglang_tpu.pp_stage_groups import (
            make_stage_groups_standalone,
        )

        make_stage_groups_standalone(get_parallel())
        validate_chips_for_pp(get_parallel())
        # Before the first collective and before torch.compile traces the model.
        from sglang_tpu.twolevel_allreduce import register_from_parallel_state

        register_from_parallel_state()

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        return "tpu"

    # ------------------------------------------------------------------
    # Device identity & operations (DeviceMixin)
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        # Per-process HBM limit. Each worker is pinned to one core
        # (TPU_VISIBLE_CHIPS=local_rank), so this already reflects v7 megacore
        # sharing (2 cores split a chip's HBM) — no cores-per-chip math needed.
        # Guard against premature PjRt initialization in the parent/tokenizer
        # process before init_backend() runs in worker subprocesses.
        if not getattr(self, "_backend_initialized", False):
            return 0
        _, limit = torch.accelerator.get_memory_info(self.get_device(device_id))
        return int(limit)

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        if not getattr(self, "_backend_initialized", False):
            return 0.0
        if device is None:
            device = self.get_device(0)
        free, total = torch.accelerator.get_memory_info(device)
        return float(total - free)

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device("tpu", device_id)

    def set_device(self, device: torch.device) -> None:
        torch.accelerator.set_device_index(device)

    def get_device_name(self, device_id: int = 0) -> str:
        try:
            from torch_tpu._internal.utils import hardware

            return hardware.get_tpu_device_name()
        except Exception:
            return "tpu"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return f"tpu:{device_id}"

    def get_device_capability(self, device_id: int = 0):
        return None

    def synchronize(self) -> None:
        torch.accelerator.synchronize()

    @contextmanager
    def weight_loading_context(self, weights):
        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.expert_loading import (
            expert_loading_context,
        )

        if get_parallel().pp_size > 1:
            with expert_loading_context():
                yield _sync_weight_stream(weights)
            return

        # No eager drain at load exit: torch.accelerator.synchronize() here deadlocks
        # EAGLE3 draft init under TP>1 -- rank 0 blocks on an async buffer while
        # peers reach the post-load barrier. Let it resolve at the first aligned
        # sync instead. Periodic syncs in _sync_weight_stream still bound HBM
        # during large loads.
        yield _sync_weight_stream(weights)

    def empty_cache(self) -> None:
        pass

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        total = self.get_device_total_memory(device_id)
        used = int(self.get_current_memory_usage())
        return (total - used, total)

    def get_torch_distributed_backend_str(self) -> str:
        return "tpu_dist"

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.tpu.manual_seed_all(seed)
