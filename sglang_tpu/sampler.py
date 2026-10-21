# SPDX-License-Identifier: Apache-2.0
"""TPU sampler with exact top-k reduction before vocabulary communication.
Unsupported sampling features retain the full-vocabulary PyTorch path."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import (
    Sampler,
    _SamplingMaskCapture,
    _select_sampling_mask_rows,
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.model_executor.forward_batch_info import LogitsOutputMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded
from sglang_tpu.top_k import PallasDistributedTopKConfig

logger = logging.getLogger(__name__)

MAX_DISTRIBUTED_TOP_K = 64
_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE = max(
    1, envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE.get()
)
# Kill switch for the sort-free top-k/top-p/min-p filtering path.
_BSEARCH_SAMPLING_ENABLED = envs.SGLANG_TPU_ENABLE_BSEARCH_SAMPLING.get()


class _DistributedTopKState(NamedTuple):
    """Terminal capability result owned by the TPU sampling backend."""

    logits_processor: object
    lm_head: object
    config: PallasDistributedTopKConfig | None
    fallback_reason: str | None


# Retain the objects alongside their ids so a long-lived worker cannot reuse a
# stale entry after Python recycles an id. Serving workers normally own one
# target model, so this process-lifetime cache stays bounded in practice.
_DISTRIBUTED_TOP_K_STATES: dict[int, _DistributedTopKState] = {}


def distributed_top_k_min_batch_size() -> int:
    return _DISTRIBUTED_TOP_K_MIN_BATCH_SIZE


def distributed_top_k_candidate_width(logits_processor, lm_head) -> Optional[int]:
    """Negotiate once, then return the model-static Pallas candidate width.

    TPU mode selection calls this before the model forward, so the collective
    and custom-op registration never occur inside a compiled region. The same
    helper is subsequently safe to call while tracing: it only reads terminal
    TPU-owned state.
    """

    if not getattr(logits_processor, "use_tpu_distributed_top_k", False):
        return None

    state = _DISTRIBUTED_TOP_K_STATES.get(id(logits_processor))
    if state is not None:
        if (
            state.logits_processor is not logits_processor
            or state.lm_head is not lm_head
        ):
            raise RuntimeError(
                "distributed TPU TopK cannot be reinitialized for a different LM head"
            )
        return None if state.config is None else state.config.candidate_width

    from sglang.srt.distributed import model_parallel_is_initialized

    if not model_parallel_is_initialized():
        return None

    group = (
        get_parallel().attn_tp_group
        if getattr(logits_processor, "use_attn_tp_group", False)
        else get_parallel().tp_group
    )
    expected_dtype = _distributed_top_k_output_dtype(logits_processor, lm_head)
    local_fallback_reason = None
    if not getattr(logits_processor, "do_tensor_parallel_all_gather", False) or getattr(
        logits_processor, "do_tensor_parallel_all_gather_dp_attn", False
    ):
        local_fallback_reason = (
            "the logits processor does not use a supported TP gather"
        )

    from sglang_tpu.top_k import (
        negotiate_pallas_distributed_top_k_config,
    )

    negotiation = negotiate_pallas_distributed_top_k_config(
        lm_head,
        group=group,
        expected_dtype=expected_dtype,
        local_fallback_reason=local_fallback_reason,
    )
    config = negotiation.config
    if config is not None:
        from sglang_tpu.distributed_topk_pallas import (
            register_pallas_distributed_top_k_op,
        )

        register_pallas_distributed_top_k_op(config.world_size)

    # Publish only after consensus and registration both succeed. A transport
    # or export failure must remain fatal instead of becoming a local fallback.
    _DISTRIBUTED_TOP_K_STATES[id(logits_processor)] = _DistributedTopKState(
        logits_processor=logits_processor,
        lm_head=lm_head,
        config=config,
        fallback_reason=negotiation.fallback_reason,
    )
    if getattr(group, "rank_in_group", None) == 0:
        if config is None:
            logger.info(
                "Pallas distributed TopK disabled; using full-vocabulary logits: %s",
                negotiation.fallback_reason,
            )
        else:
            logger.info(
                "Pallas distributed TopK enabled: TP%d dtype=%s vocab=%d "
                "local_width=%d candidates=%d",
                config.world_size,
                config.output_dtype,
                config.shard.global_vocab_size,
                config.shard.width,
                config.candidate_width,
            )
    return None if config is None else config.candidate_width


def _distributed_top_k_output_dtype(logits_processor, lm_head) -> torch.dtype | None:
    """Resolve the compile-static dtype produced by this TPU LM head."""

    if getattr(logits_processor, "use_fp32_lm_head", False):
        return torch.float32
    if getattr(logits_processor, "rl_on_policy_target", None) is not None:
        return torch.bfloat16

    weight_dtype = getattr(getattr(lm_head, "weight", None), "dtype", None)
    if weight_dtype in (torch.bfloat16, torch.float32):
        return weight_dtype
    config_dtype = getattr(
        getattr(logits_processor, "config", None), "torch_dtype", None
    )
    if config_dtype in (torch.bfloat16, torch.float32):
        return config_dtype
    return None


def _distributed_top_k_output_head(model):
    """Resolve the output head consumed by the model's logits processor."""

    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if output_head is None:
        output_head = getattr(model, "lm_head", None)
    if output_head is None and getattr(
        getattr(model, "config", None), "tie_word_embeddings", False
    ):
        output_head = getattr(getattr(model, "model", None), "embed_tokens", None)
    return output_head


def distributed_top_k_candidate_width_for_model(model) -> Optional[int]:
    """Return the fixed candidate width supported by a model's output head."""

    output_head = _distributed_top_k_output_head(model)
    if output_head is None:
        return None
    return distributed_top_k_candidate_width(
        getattr(model, "logits_processor", None), output_head
    )


def _use_bsearch_filtering(probs: torch.Tensor) -> bool:
    """The sort-free mask is a TPU custom op; CPU tensors (unit tests, other
    devices) keep the generic torch fallback."""
    return _BSEARCH_SAMPLING_ENABLED and probs.device.type == "tpu"


def _filter_gathered_top_k_probs(
    top_k_probs: torch.Tensor,
    sampling_info: SamplingBatchInfo,
) -> torch.Tensor:
    """Apply top-p/min-p inside a gathered descending [bs, k] block.

    Exact versus full-width filtering: a nucleus narrower than k finds the
    same cutoff in the block, a wider one keeps the whole block (which is
    what ANDing with top-k yields), and min-p's reference max is the block's
    first column. Nothing renormalizes on this path, so gathered
    probabilities stay directly comparable to full-row ones."""

    from sglang_tpu.sampling_mask import tpu_filter_probs

    return tpu_filter_probs(
        top_k_probs,
        sampling_info.top_ks,
        sampling_info.top_ps,
        sampling_info.min_ps,
        need_top_k=False,  # gathering k columns already applied top-k
        need_top_p=sampling_info.need_top_p_sampling,
        need_min_p=sampling_info.need_min_p_sampling,
    )


def _is_speculative_batch(forward_batch) -> bool:
    # Draft and verify forwards carry spec_info even when their spec_algorithm
    # field differs between precompile and serving. Return before reading either
    # spec_algorithm or sampling_info so Dynamo does not guard those fields.
    if getattr(forward_batch, "spec_info", None) is not None:
        return True
    spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
    return spec_algorithm is not None and not spec_algorithm.is_none()


def distributed_top_k_for_batch(
    forward_batch,
    *,
    sampling_backend: str,
) -> Optional[int]:
    """Return k only when pre-gather reduction preserves sampling semantics.
    Full-vocabulary transforms and batches below the configured cutoff fall back."""

    if sampling_backend != "tpu":
        return None
    if forward_batch is None:
        return None
    # Speculative decoding never uses the distributed TopK path. Check this
    # before sampling_info so compiled speculative forwards do not specialize
    # on request-specific sampling state.
    if _is_speculative_batch(forward_batch):
        return None
    if getattr(forward_batch, "sampling_info", None) is None:
        return None
    # Mode selection happens after TPU padding, so use the physical batch size.
    # The logical size must not become a compile-time guard.
    batch_size = getattr(forward_batch, "batch_size", 0)
    if (
        getattr(forward_batch, "return_logprob", False)
        or batch_size < _DISTRIBUTED_TOP_K_MIN_BATCH_SIZE
    ):
        return None

    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_extend_without_speculative()):
        return None

    info = forward_batch.sampling_info
    top_k = info.uniform_top_k
    if (
        top_k is None
        or top_k <= 1
        or info.is_any_greedy
        or info.sampling_seed is not None
        or not info.need_top_k_sampling
        or info.need_top_p_sampling
        or info.need_min_p_sampling
    ):
        return None
    if info.has_custom_logit_processor or info.logit_bias is not None:
        return None
    if any(info.grammars or ()) or info.sampling_mask_batch_indices is not None:
        return None
    if (
        info.acc_additive_penalties is not None
        or info.acc_scaling_penalties is not None
    ):
        return None
    if (
        info.penalizer_orchestrator is not None
        and info.penalizer_orchestrator.is_required
    ):
        return None
    return top_k


def select_logits_output_mode(forward_batch, model) -> LogitsOutputMode:
    """Select full-vocabulary logits or distributed TopK candidates."""

    # Model capability negotiation uses a CPU collective, which speculative
    # forwards never need because they require full-vocabulary logits.
    if _is_speculative_batch(forward_batch):
        return LogitsOutputMode.FULL_VOCAB
    candidate_width = distributed_top_k_candidate_width_for_model(model)
    if candidate_width is None:
        return LogitsOutputMode.FULL_VOCAB
    requested_top_k = distributed_top_k_for_batch(forward_batch, sampling_backend="tpu")
    if requested_top_k is None or requested_top_k > candidate_width:
        return LogitsOutputMode.FULL_VOCAB
    return LogitsOutputMode.DISTRIBUTED_TOP_K


def sample_from_global_top_k_logits(
    global_top_k_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize temperature-scaled global TopK logits and sample a position.
    Returned positions index global TopK token IDs, not the full vocabulary."""

    if top_ks is not None:
        positions = torch.arange(
            global_top_k_logits.shape[-1],
            dtype=top_ks.dtype,
            device=global_top_k_logits.device,
        )
        global_top_k_logits = global_top_k_logits.masked_fill(
            positions.unsqueeze(0) >= top_ks.unsqueeze(1), -torch.inf
        )
    global_top_k_probs = torch.softmax(global_top_k_logits / temperatures, dim=-1)
    return sampling_from_probs_torch(
        global_top_k_probs, unseeded_sampling_fn=sample_exponential_race
    )


def sample_exponential_race(probs: torch.Tensor) -> torch.Tensor:
    """Exact categorical draw per row as argmax(p_i / E_i), E_i ~ Exp(1).

    Elementwise ops plus one reduce, so it stays inside the compiled graph;
    ``torch.multinomial`` runs eagerly on TPU. All-zero rows return 0.
    """
    u = torch.rand_like(probs)
    # -log(u) ~ Exp(1); clamp keeps u=0 from producing inf/nan.
    e = torch.log(u.clamp_min(torch.finfo(probs.dtype).tiny)).neg_()
    return torch.argmax(probs / e, dim=-1).to(torch.int32)


def _capture_sampling_mask(
    weights: torch.Tensor,
    batch_next_token_ids: torch.Tensor,
    batch_indices: torch.Tensor,
    token_ids: Optional[torch.Tensor] = None,
) -> _SamplingMaskCapture:
    """Capture filtered support for the rows requested by the base sampler."""
    capture_weights = _select_sampling_mask_rows(weights, batch_indices)
    capture_tokens = _select_sampling_mask_rows(batch_next_token_ids, batch_indices)
    capture_token_ids = (
        None
        if token_ids is None
        else _select_sampling_mask_rows(token_ids, batch_indices)
    )
    if capture_token_ids is None:
        selected_weight = torch.gather(
            capture_weights,
            1,
            capture_tokens.long().view(-1, 1),
        ).squeeze(1)
    else:
        selected_weight = (
            capture_weights
            * (capture_token_ids == capture_tokens.view(-1, 1)).to(
                capture_weights.dtype
            )
        ).sum(dim=1)
    return _SamplingMaskCapture(
        weights=capture_weights,
        token_ids=capture_token_ids,
        selected_weight=selected_weight,
        batch_rows=batch_indices,
    )


_PARAM_DEFAULTS = {
    "temperatures": 1,
    "top_ps": 1,
    "top_ks": 1,
    "min_ps": 0,
    "sampling_seed": 0,
}


def supports_padded_sampling(info: SamplingBatchInfo) -> bool:
    """Whether all sampling consumers accept physical rows independently of requests.

    Biases, penalties, grammar and observers run before the sampler and still
    carry live-row state. Keep their existing contract until they are bucketed
    together with logits, rather than padding only their sampling parameters.
    """
    return (
        info.sampling_params_on_cpu
        and not info.has_custom_logit_processor
        and info.sampling_mask_batch_indices is None
        and info.sampling_support_logprobs_capture_indices is None
        and not any(info.grammars or ())
        and info.logit_bias is None
        and info.acc_additive_penalties is None
        and info.acc_scaling_penalties is None
        and not (
            info.penalizer_orchestrator is not None
            and info.penalizer_orchestrator.is_required
        )
    )


def sampling_params_to_device(
    info: SamplingBatchInfo, *, padded_size: int | None = None
) -> SamplingBatchInfo:
    """Upload parameters with the same row count as the sampling inputs.

    Leave the scheduler's host snapshot intact while overlap scheduling filters
    or merges subsequent batches. A caller using bucketed logits supplies their
    row count; other callers upload live rows directly, without a device slice.
    Device-backed inputs need no preparation.
    """
    if not info.sampling_params_on_cpu:
        return info
    size = len(info)
    padded_size = size if padded_size is None else padded_size
    return replace(
        info, sampling_params_on_cpu=False, **_upload_params(info, padded_size)
    )


def _upload_params(info: SamplingBatchInfo, padded_size: int) -> dict:
    size = len(info)
    if padded_size < size:
        raise ValueError("Sampling parameter rows cannot be smaller than the batch")
    params = {}
    for name, fill in _PARAM_DEFAULTS.items():
        value = getattr(info, name)
        if value is None:
            continue
        padded = value.new_full((padded_size, *value.shape[1:]), fill)
        padded[:size].copy_(value)
        params[name] = padded.to(info.device, non_blocking=True)
    return params


class _DeviceParamCache:
    """Reuse uploaded sampling parameters while a batch keeps its composition.

    Each upload costs a fixed host launch, so per-step uploads dominate small
    decode steps. filter_batch and merge_batch rebind the host tensors, so tensor
    identity marks a composition change. Two entries cover alternating batches.
    """

    _CAPACITY = 2

    def __init__(self):
        self._entries: list[tuple[tuple, int, dict]] = []

    def materialize(
        self, info: SamplingBatchInfo, padded_size: int
    ) -> SamplingBatchInfo:
        host = tuple(getattr(info, name) for name in _PARAM_DEFAULTS)
        for index, (cached_host, cached_size, params) in enumerate(self._entries):
            if cached_size == padded_size and all(
                a is b for a, b in zip(cached_host, host)
            ):
                self._entries.append(self._entries.pop(index))
                return replace(info, sampling_params_on_cpu=False, **params)
        params = _upload_params(info, padded_size)
        # Holding the host tensors keeps their identities from being reused.
        self._entries = [*self._entries, (host, padded_size, params)][-self._CAPACITY :]
        return replace(info, sampling_params_on_cpu=False, **params)


class TpuSampler(Sampler):
    """Consume a pre-gather global TopK for eligible TPU sampling batches.
    Every other batch delegates to the full-vocabulary sampler implementation."""

    def __init__(self):
        super().__init__()
        self._device_params = _DeviceParamCache()

    def forward(
        self,
        logits_output,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums,
        token_ids_logprobs,
        positions: torch.Tensor,
    ):
        """Sample from exact global TopK when full logits gathering was skipped.
        Full-logit outputs retain the base sampler's processing and return contract."""
        if is_host_padded(sampling_info):
            sampling_info = sampling_info.on_device()
            padded_bs = sampling_info.temperatures.shape[0]
            logits = logits_output.next_token_logits
            # An eager outer logits processor (multimodal wrappers) emits live
            # rows; HOST_PADDED sampling runs at the batch bucket.
            if logits is not None and logits.shape[0] < padded_bs:
                logits_output.next_token_logits = torch.nn.functional.pad(
                    logits, (0, 0, 0, padded_bs - logits.shape[0])
                )

        live_rows = None
        if sampling_info.sampling_params_on_cpu:
            live_rows = len(sampling_info)
            logits = (
                logits_output.next_token_global_top_k_logits
                if logits_output.has_next_token_global_top_k
                else logits_output.next_token_logits
            )
            physical_rows = logits.shape[0]
            if physical_rows != live_rows and (
                return_logprob or not supports_padded_sampling(sampling_info)
            ):
                raise ValueError("This sampling mode requires live-row logits")
            if (
                sampling_info.sampling_seed is not None
                and positions.shape[0] != physical_rows
            ):
                raise ValueError(
                    "Seeded sampling positions must match physical logits rows"
                )
            # The producer retains physical logits and forward metadata. Never
            # slice or re-pad them here. Only CPU parameters need materializing
            # at the producer's extent (prefill can use a fixed batch cap).
            if (
                sampling_info.is_all_greedy
                and not logits_output.has_next_token_global_top_k
            ):
                # Greedy argmax reads no sampling parameters.
                sampling_info = replace(sampling_info, sampling_params_on_cpu=False)
            else:
                sampling_info = self._device_params.materialize(
                    sampling_info, physical_rows
                )

        # No global TopK: sample from the full-vocabulary logits instead.
        if not logits_output.has_next_token_global_top_k:
            token_ids = super().forward(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
            )
            return token_ids

        if return_logprob:
            raise ValueError("distributed TPU top-k cannot return full logprobs")
        sampled_top_k_position = sample_from_global_top_k_logits(
            logits_output.next_token_global_top_k_logits,
            sampling_info.temperatures,
            sampling_info.top_ks,
        )
        batch_next_token_ids = torch.gather(
            logits_output.next_token_global_top_k_token_ids,
            dim=1,
            index=sampled_top_k_position.to(torch.int64).view(-1, 1),
        ).view(-1)
        batch_next_token_ids = batch_next_token_ids.to(torch.int64)
        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
        return batch_next_token_ids

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> tuple[torch.Tensor, Optional[_SamplingMaskCapture]]:
        """Use native XLA TopK after gathering uniform small-k probability rows.
        This avoids a full sort when pre-gather reduction is not selected.

        Keep the base sampler's ``(token_ids, sampling_mask_capture)`` return
        contract. Sampling-mask requests need the filtered support as well as
        the sampled token, even though distributed TopK is disabled for them.
        """

        capture_rows = sampling_info.sampling_mask_batch_indices

        if simple_sampling_case:
            batch_next_token_ids = sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
                unseeded_sampling_fn=sample_exponential_race,
            )
            return batch_next_token_ids, (
                None
                if capture_rows is None
                else _capture_sampling_mask(probs, batch_next_token_ids, capture_rows)
            )
        top_k = sampling_info.uniform_top_k
        needs_top_p_or_min_p = (
            sampling_info.need_top_p_sampling or sampling_info.need_min_p_sampling
        )
        if (
            top_k is not None
            and 1 < top_k <= MAX_DISTRIBUTED_TOP_K
            # A Python k compiles per value; HOST_PADDED stays on the warmed tensor-k path.
            and not is_host_padded(sampling_info)
            and sampling_info.sampling_seed is None
            and sampling_info.need_top_k_sampling
            and (not needs_top_p_or_min_p or _use_bsearch_filtering(probs))
        ):
            from sglang_tpu.top_k import tpu_top_k

            candidate_width = (
                min(MAX_DISTRIBUTED_TOP_K, probs.shape[-1])
                if tpu_envs.SGLANG_TPU_HOST_PADDING.get()
                else top_k
            )
            global_top_k_probs, global_top_k_token_ids = tpu_top_k(
                probs, candidate_width
            )
            if tpu_envs.SGLANG_TPU_HOST_PADDING.get():
                # Keep the candidate shape independent of the requested k.
                # Rank masking retains exactly the requested number of candidates.
                ranks = torch.arange(candidate_width, device=probs.device)
                global_top_k_probs = global_top_k_probs.masked_fill(
                    ranks[None, :] >= sampling_info.top_ks[:, None], 0
                )
            if needs_top_p_or_min_p:
                # Filtering the gathered block replaces ~30 full-vocabulary
                # reductions with ~30 reductions over top_k columns.
                global_top_k_probs = _filter_gathered_top_k_probs(
                    global_top_k_probs, sampling_info
                )
            sampled_top_k_position = sampling_from_probs_torch(
                global_top_k_probs, unseeded_sampling_fn=sample_exponential_race
            )
            batch_next_token_ids = torch.gather(
                global_top_k_token_ids,
                dim=1,
                index=sampled_top_k_position.to(torch.int64).view(-1, 1),
            ).view(-1)
            return batch_next_token_ids, (
                None
                if capture_rows is None
                else _capture_sampling_mask(
                    global_top_k_probs,
                    batch_next_token_ids,
                    capture_rows,
                    global_top_k_token_ids,
                )
            )

        if sampling_info.sampling_seed is None and _use_bsearch_filtering(probs):
            # Sort-free path: the torch fallback below needs a full-vocab sort
            # plus boolean-mask scatters, which run eagerly on TPU at ~415 ms
            # per decode step. The binary-search mask is one compiled program.
            from sglang_tpu.sampling_mask import tpu_filter_probs

            filtered_probs = tpu_filter_probs(
                probs,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                need_top_k=sampling_info.need_top_k_sampling,
                need_top_p=sampling_info.need_top_p_sampling,
                need_min_p=sampling_info.need_min_p_sampling,
            )
            batch_next_token_ids = sample_exponential_race(filtered_probs)
            return batch_next_token_ids, (
                None
                if capture_rows is None
                else _capture_sampling_mask(
                    filtered_probs, batch_next_token_ids, capture_rows
                )
            )

        # Seeded (deterministic) sampling keeps the sort-based reference path.
        sample_result = top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            sampling_info.top_ks,
            sampling_info.top_ps,
            sampling_info.min_ps,
            sampling_info.need_min_p_sampling,
            sampling_info.sampling_seed,
            positions,
            return_filtered_probs=capture_rows is not None,
        )
        if capture_rows is None:
            return sample_result, None

        batch_next_token_ids, filtered_probs, token_ids, _ = sample_result
        return batch_next_token_ids, _capture_sampling_mask(
            filtered_probs, batch_next_token_ids, capture_rows, token_ids
        )
