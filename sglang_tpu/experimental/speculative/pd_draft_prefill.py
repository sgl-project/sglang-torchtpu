"""Decode-side draft prompt pass of SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.

A PD decode receives each request's target KV and its packed target hidden rows
(pd_hidden_pool.py) instead of a draft KV; before the first decode step it runs
the draft over the prompt, in extend-shaped groups cut from the prebuilt batch.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.experimental.speculative.pd_hidden_pool import PdHiddenPool

# (prebuilt ScheduleBatch) -> None.
PdDraftPrefiller = Callable[[ScheduleBatch], None]

_pd_draft_prefiller: PdDraftPrefiller | None = None
_pd_draft_prompt_cap: Callable[[], int] | None = None


def register_pd_draft_prefiller(
    prefiller: PdDraftPrefiller, *, max_prompt_tokens: Callable[[], int]
) -> None:
    """Set the draft prompt pass a PD decode runs on each prebuilt batch, and the
    longest prompt it takes (read once the compile policy exists)."""
    global _pd_draft_prefiller, _pd_draft_prompt_cap
    _pd_draft_prefiller = prefiller
    _pd_draft_prompt_cap = max_prompt_tokens


def pd_draft_prompt_cap() -> int | None:
    """The longest prompt the decode's draft prompt pass takes; None without one."""
    return None if _pd_draft_prompt_cap is None else _pd_draft_prompt_cap()


def get_pd_draft_prefiller() -> PdDraftPrefiller:
    if _pd_draft_prefiller is None:
        raise RuntimeError("No TpuSpecWorker registered a PD draft prefiller.")
    return _pd_draft_prefiller


def draft_feature_size(config: SpecAuxHiddenStateConfig, hidden_size: int) -> int:
    """Width of the target hidden the draft reads: its aux layers, or the final hidden."""
    if config.eagle_use_aux_hidden_state:
        layer_ids = config.eagle_aux_hidden_state_layer_ids
    elif config.dflash_use_aux_hidden_state:
        layer_ids = config.dflash_target_layer_ids
    else:
        return hidden_size
    if layer_ids is None:
        raise NotImplementedError(
            "SGLANG_TPU_PD_DECODE_DRAFT_PREFILL sizes the hidden rows from the "
            "draft's aux layer ids, which this draft does not declare."
        )
    return len(layer_ids) * hidden_size


def pd_hidden_bytes_per_token(
    config: SpecAuxHiddenStateConfig, model_config: ModelConfig, dtype: torch.dtype
) -> int:
    """The hidden pool's HBM per KV token slot on this rank."""
    return PdHiddenPool.bytes_per_token(
        feature_size=draft_feature_size(config, model_config.hidden_size),
        total_kv_heads=model_config.get_total_num_kv_heads(),
        attn_tp_size=get_parallel().attn_tp_size,
        dtype=dtype,
    )


def build_pd_hidden_pool(target_runner: ModelRunner) -> PdHiddenPool:
    """A hidden pool paged like the target's full-attention KV pages."""
    full = get_raiden_kv_caches(target_runner.token_to_kv_pool).full
    parallel = get_parallel()
    return PdHiddenPool(
        num_pages=int(full[0].shape[0]),
        page_size=int(full[0].shape[1]),
        feature_size=draft_feature_size(
            target_runner.spec_aux_config, target_runner.model_config.hidden_size
        ),
        total_kv_heads=target_runner.model_config.get_total_num_kv_heads(),
        attn_tp_size=parallel.attn_tp_size,
        attn_tp_group=parallel.attn_tp_group if parallel.attn_tp_size > 1 else None,
        dtype=target_runner.dtype,
        device=target_runner.device,
    )


def prebuilt_extend_groups(
    batch: ScheduleBatch, *, max_tokens: int, max_reqs: int
) -> list[range]:
    """Consecutive request ranges of at most max_tokens extend tokens and max_reqs requests."""
    groups = []
    start = tokens = 0
    for index, length in enumerate(batch.extend_lens):
        # Admission aborts longer prompts (reject_overlong_pd_draft_prompt).
        assert length <= max_tokens, (length, max_tokens)
        if index > start and (
            tokens + length > max_tokens or index - start == max_reqs
        ):
            groups.append(range(start, index))
            start, tokens = index, 0
        tokens += length
    if start < len(batch.extend_lens):
        groups.append(range(start, len(batch.extend_lens)))
    return groups


def prebuilt_extend_view(
    batch: ScheduleBatch, group: range, *, vocab_size: int
) -> ScheduleBatch:
    """An EXTEND copy of batch's group of requests over their prebuilt KV slots."""
    view = copy.copy(batch)
    reqs = batch.reqs[group.start : group.stop]
    prefix_lens = batch.prefix_lens[group.start : group.stop]
    extend_lens = batch.extend_lens[group.start : group.stop]
    starts = [sum(batch.extend_lens[: group.start])]
    for length in extend_lens:
        starts.append(starts[-1] + length)
    view.reqs = reqs
    view.forward_mode = ForwardMode.EXTEND
    view.input_ids = torch.tensor(
        [
            token
            for req, prefix, length in zip(reqs, prefix_lens, extend_lens)
            for token in req.origin_input_ids[prefix : prefix + length]
        ],
        dtype=torch.int64,
    )
    view.out_cache_loc = batch.out_cache_loc[starts[0] : starts[-1]]
    # Host copies: the shim pads them on host and uploads at the bucket, while a
    # device slice would compile a pad per group size and offset.
    view.req_pool_indices_cpu = batch.req_pool_indices_cpu[group.start : group.stop]
    view.req_pool_indices = view.req_pool_indices_cpu.clone()
    view.seq_lens_cpu = batch.seq_lens_cpu[group.start : group.stop]
    view.seq_lens = view.seq_lens_cpu.clone()
    view.orig_seq_lens = view.seq_lens_cpu.to(torch.int32)
    view.seq_lens_sum = int(view.seq_lens_cpu.sum())
    view.prefix_lens = prefix_lens
    view.extend_lens = extend_lens
    view.extend_num_tokens = starts[-1] - starts[0]
    view.multimodal_inputs = [req.multimodal_inputs for req in reqs]
    view.return_logprob = False
    view.spec_info = None
    view.chunked_req = None
    view.chunked_req_next_prompt_token = None
    view.sampling_info = SamplingBatchInfo.from_schedule_batch(view, vocab_size)
    return view
