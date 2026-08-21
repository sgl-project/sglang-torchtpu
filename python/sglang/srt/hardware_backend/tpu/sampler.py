# SPDX-License-Identifier: Apache-2.0
"""TPU sampler with exact top-k reduction before vocabulary communication.
Unsupported sampling features retain the full-vocabulary PyTorch path."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import (
    Sampler,
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

MAX_DISTRIBUTED_TOP_K = 64
_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE = max(
    1, envs.SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE.get()
)


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
    spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
    has_speculative_decode = spec_algorithm is not None and not spec_algorithm.is_none()
    # Speculative decoding never uses the distributed TopK path. Check this
    # before sampling_info so compiled speculative forwards do not specialize
    # on request-specific sampling state.
    if has_speculative_decode:
        return None
    if getattr(forward_batch, "sampling_info", None) is None:
        return None
    batch_size = getattr(
        forward_batch,
        "_tpu_real_b",
        getattr(forward_batch, "batch_size", 0),
    )
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
        or not 1 < top_k <= MAX_DISTRIBUTED_TOP_K
        or info.is_any_greedy
        or info.sampling_seed is not None
        or not info.need_top_k_sampling
        or info.need_top_p_sampling
        or info.need_min_p_sampling
    ):
        return None
    if info.has_custom_logit_processor or info.logit_bias is not None:
        return None
    if any(info.grammars or ()) or getattr(info, "sampling_mask_max_top_k", 0) > 0:
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


def sample_from_global_top_k_logits(
    global_top_k_logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Normalize temperature-scaled global TopK logits and sample a position.
    Returned positions index global TopK token IDs, not the full vocabulary."""

    global_top_k_probs = torch.softmax(global_top_k_logits / temperatures, dim=-1)
    return sampling_from_probs_torch(global_top_k_probs)


class TpuSampler(Sampler):
    """Consume a pre-gather global TopK for eligible TPU sampling batches.
    Every other batch delegates to the full-vocabulary sampler implementation."""

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

        # No global TopK: sample from the full-vocabulary logits instead.
        if not logits_output.has_next_token_global_top_k:
            return super().forward(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
            )

        if return_logprob:
            raise ValueError("distributed TPU top-k cannot return full logprobs")
        sampled_top_k_position = sample_from_global_top_k_logits(
            logits_output.next_token_global_top_k_logits,
            sampling_info.temperatures,
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
    ) -> torch.Tensor:
        """Use native XLA TopK after gathering uniform small-k probability rows.
        This avoids a full sort when pre-gather reduction is not selected."""

        if simple_sampling_case:
            return sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        top_k = sampling_info.uniform_top_k
        if (
            top_k is not None
            and 1 < top_k <= MAX_DISTRIBUTED_TOP_K
            and sampling_info.sampling_seed is None
            and sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_min_p_sampling
        ):
            from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k

            global_top_k_probs, global_top_k_token_ids = tpu_top_k(probs, top_k)
            sampled_top_k_position = sampling_from_probs_torch(global_top_k_probs)
            return torch.gather(
                global_top_k_token_ids,
                dim=1,
                index=sampled_top_k_position.to(torch.int64).view(-1, 1),
            ).view(-1)

        return top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            sampling_info.top_ks,
            sampling_info.top_ps,
            sampling_info.min_ps,
            sampling_info.need_min_p_sampling,
            sampling_info.sampling_seed,
            positions,
        )
