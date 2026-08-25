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
# Kill switch for the sort-free top-k/top-p/min-p filtering path.
_BSEARCH_SAMPLING_ENABLED = envs.SGLANG_TPU_ENABLE_BSEARCH_SAMPLING.get()


def distributed_top_k_min_batch_size() -> int:
    return _DISTRIBUTED_TOP_K_MIN_BATCH_SIZE


def distributed_top_k_candidate_width(logits_processor, lm_head) -> Optional[int]:
    """Return the fixed candidate width supported by a TPU model layout."""

    if (
        not getattr(logits_processor, "use_tpu_distributed_top_k", False)
        or not getattr(logits_processor, "do_tensor_parallel_all_gather", False)
        or getattr(logits_processor, "do_tensor_parallel_all_gather_dp_attn", False)
    ):
        return None

    from sglang.srt.hardware_backend.tpu.top_k import (
        supports_distributed_tpu_top_k,
    )

    candidate_width = min(
        MAX_DISTRIBUTED_TOP_K,
        int(getattr(lm_head, "num_embeddings", 0)),
        int(getattr(lm_head, "num_embeddings_per_partition", 0)),
    )
    if candidate_width <= 1 or not supports_distributed_tpu_top_k(
        lm_head, candidate_width
    ):
        return None
    return candidate_width


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

    from sglang.srt.hardware_backend.tpu.sampling_mask import tpu_filter_probs

    return tpu_filter_probs(
        top_k_probs,
        sampling_info.top_ks,
        sampling_info.top_ps,
        sampling_info.min_ps,
        need_top_k=False,  # gathering k columns already applied top-k
        need_top_p=sampling_info.need_top_p_sampling,
        need_min_p=sampling_info.need_min_p_sampling,
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
        needs_top_p_or_min_p = (
            sampling_info.need_top_p_sampling or sampling_info.need_min_p_sampling
        )
        if (
            top_k is not None
            and 1 < top_k <= MAX_DISTRIBUTED_TOP_K
            and sampling_info.sampling_seed is None
            and sampling_info.need_top_k_sampling
            and (not needs_top_p_or_min_p or _use_bsearch_filtering(probs))
        ):
            from sglang.srt.hardware_backend.tpu.top_k import tpu_top_k

            global_top_k_probs, global_top_k_token_ids = tpu_top_k(probs, top_k)
            if needs_top_p_or_min_p:
                # Filtering the gathered block replaces ~30 full-vocabulary
                # reductions with ~30 reductions over top_k columns.
                global_top_k_probs = _filter_gathered_top_k_probs(
                    global_top_k_probs, sampling_info
                )
            sampled_top_k_position = sampling_from_probs_torch(global_top_k_probs)
            return torch.gather(
                global_top_k_token_ids,
                dim=1,
                index=sampled_top_k_position.to(torch.int64).view(-1, 1),
            ).view(-1)

        if sampling_info.sampling_seed is None and _use_bsearch_filtering(probs):
            # Sort-free path: the torch fallback below needs a full-vocab sort
            # plus boolean-mask scatters, which run eagerly on TPU at ~415 ms
            # per decode step. The binary-search mask is one compiled program;
            # the remaining multinomial is the cheap tier-2 op.
            from sglang.srt.hardware_backend.tpu.sampling_mask import tpu_filter_probs

            filtered_probs = tpu_filter_probs(
                probs,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                need_top_k=sampling_info.need_top_k_sampling,
                need_top_p=sampling_info.need_top_p_sampling,
                need_min_p=sampling_info.need_min_p_sampling,
            )
            return sampling_from_probs_torch(filtered_probs)

        # Seeded (deterministic) sampling keeps the sort-based reference path.
        return top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            sampling_info.top_ks,
            sampling_info.top_ps,
            sampling_info.min_ps,
            sampling_info.need_min_p_sampling,
            sampling_info.sampling_seed,
            positions,
        )
