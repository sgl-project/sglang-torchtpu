"""Narrow hook adapters for the experimental host-padded spec path.

Why: filter/merge and H2D staging must be intercepted before device operations,
not repaired afterward. All non-experimental calls delegate unchanged.
Upstream seam issue: not filed (local prototype). Before proposing upstream,
replace these hooks with agreed factories/overrides for the extracted operations.
Remove when: the experiment is removed or those upstream seams are available.
"""

from __future__ import annotations

import torch

from sglang_tpu.experimental import get_host_batch_padder, is_experimental_dspark


def uses_host_batch(batch):
    return getattr(batch, "host_pad_bs", None) is not None


def initialize_host_batch(result, *args, **kwargs):
    if is_experimental_dspark(result.spec_algorithm):
        if not result.enable_overlap:
            raise ValueError("Experimental TPU DSpark requires overlap scheduling")
        result.host_pad_bs = get_host_batch_padder()
    return result


def copy_host_batch(result, batch):
    if uses_host_batch(batch):
        result.host_pad_bs = batch.host_pad_bs
    return result


def sampling_from_batch(original, cls, batch, vocab_size):
    if uses_host_batch(batch):
        from sglang_tpu.experimental.sampling.sampling_batch_info import (
            HostPaddedSamplingInfo,
        )

        cls = HostPaddedSamplingInfo
    return original(cls, batch, vocab_size)


def _rebuild_relay_rows(batch):
    assert batch.enable_overlap and not batch.spec_algorithm.is_none()
    assert batch.input_ids is None
    batch.req_pool_indices = batch.req_pool_indices_cpu.to(
        batch.device, non_blocking=True
    )
    # FutureMap resolves these before forward; direct compile ignores orig_seq_lens.
    batch.seq_lens = None
    batch.orig_seq_lens = None


def filter_model_inputs(original, batch, keep_indices_device):
    if not uses_host_batch(batch):
        return original(batch, keep_indices_device)
    _rebuild_relay_rows(batch)


def merge_model_inputs(original, batch, other):
    assert uses_host_batch(batch) == uses_host_batch(other)
    if not uses_host_batch(batch):
        return original(batch, other)
    assert other.input_ids is None
    _rebuild_relay_rows(batch)


def filter_spec_input(original, batch, keep_indices, keep_indices_device):
    if not uses_host_batch(batch):
        return original(batch, keep_indices, keep_indices_device)
    from sglang_tpu.experimental.speculative.dflash_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.filter_batch(
        new_indices=keep_indices_device,
        new_indices_cpu=keep_indices,
        future_indices=batch.req_pool_indices,
    )


def merge_spec_input(original, batch, other):
    if not uses_host_batch(batch):
        return original(batch, other)
    from sglang_tpu.experimental.speculative.dflash_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.merge_batch(other.spec_info, future_indices=batch.req_pool_indices)


def stage_prefill_tokens(original, batch):
    if not uses_host_batch(batch):
        return original(batch)
    assert batch.mix_running_indices is None, (
        "Host-padded spec does not support mixed chunks"
    )
    return batch.prefill_input_ids_cpu


def extend_metadata_device(original, batch, device):
    from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded

    if is_host_padded(batch.sampling_info):
        return "cpu"
    return original(batch, device)


def alloc_extend(
    original,
    tree_cache,
    prefix_lens,
    prefix_lens_cpu,
    seq_lens,
    seq_lens_cpu,
    last_loc,
    extend_num_tokens,
    req_pool_indices=None,
    batch=None,
    out_cache_loc_host=None,
):
    if uses_host_batch(batch) and out_cache_loc_host is None:
        out_cache_loc_host = torch.empty((extend_num_tokens,), dtype=torch.int32)
    return original(
        tree_cache=tree_cache,
        prefix_lens=prefix_lens,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=extend_num_tokens,
        req_pool_indices=req_pool_indices,
        batch=batch,
        out_cache_loc_host=out_cache_loc_host,
    )


def sample_draft_tokens(original, model, hidden_states, input_ids):
    from sglang_tpu.envs import envs

    if not envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.get():
        return original(model, hidden_states, input_ids)
    from sglang_tpu.experimental.models.dspark import sample_draft_tokens as sample

    return sample(model, hidden_states, input_ids)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    schedule = "sglang.srt.managers.schedule_batch.ScheduleBatch."
    return [
        (
            "sglang.srt.models.dspark.DSparkDraftModel.sample_draft_tokens_tpu",
            sample_draft_tokens,
            HookType.AROUND,
        ),
        (schedule + "init_new", initialize_host_batch, HookType.AFTER),
        (schedule + "copy", copy_host_batch, HookType.AFTER),
        (schedule + "_filter_model_inputs", filter_model_inputs, HookType.AROUND),
        (schedule + "_merge_model_inputs", merge_model_inputs, HookType.AROUND),
        (schedule + "_filter_spec_input", filter_spec_input, HookType.AROUND),
        (schedule + "_merge_spec_input", merge_spec_input, HookType.AROUND),
        (
            "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
            sampling_from_batch,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.overlap_utils._stage_prefill_tokens",
            stage_prefill_tokens,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.forward_batch_info._extend_metadata_device",
            extend_metadata_device,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_extend",
            alloc_extend,
            HookType.AROUND,
        ),
    ]
