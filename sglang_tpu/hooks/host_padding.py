"""Defer host-built inputs until the decoder's padding boundary.

Why: uploading before padding produces unnecessary transfers and shape variants.
How: AROUND the sampling factory, input staging, and extend allocation.
Upstream seam issue: not filed (local prototype).
Remove when: upstream provides backend factories for these staging decisions.
"""

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_flags, get_parallel, get_schedule


def _enabled(batch):
    return envs.SGLANG_TPU_HOST_PADDING.get() and batch.spec_algorithm.is_none()


def sampling_from_batch(original, cls, batch, vocab_size, *, params_on_cpu=False):
    params_on_cpu = params_on_cpu or (
        _enabled(batch) and get_exec().kernel.sampling_backend == "tpu"
    )
    return original(cls, batch, vocab_size, params_on_cpu=params_on_cpu)


def _can_stage_prefill(batch):
    parallel = get_parallel()
    # Only the ordinary compiled text path has the upload boundary below.
    # Mixed chunks concatenate device-resident decode tokens before that point.
    return (
        _enabled(batch)
        and get_flags().capture.enable_torch_compile
        and parallel.pp_size == 1
        and parallel.attn_cp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.attn_dcp_size == 1
        and not get_schedule().enable_mixed_chunk
        and batch.mix_running_indices is None
        and batch.model_config.is_generation
        and not batch.model_config.use_ngram_embedding
        and not batch.model_config.use_engram
        and not batch.model_config.is_multimodal
        and not batch.model_config.is_encoder_decoder
        and batch.dllm_config is None
    )


def stage_prefill_tokens(original, batch):
    if _can_stage_prefill(batch):
        return batch.prefill_input_ids_cpu
    return original(batch)


def extend_metadata_device(original, batch, device):
    # Input placement carries the decision across ScheduleBatch -> worker batch.
    if (
        _can_stage_prefill(batch)
        and batch.input_ids is not None
        and batch.input_ids.device.type == "cpu"
    ):
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
    return_host=False,
):
    from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator

    # Wrapped/SWA allocators retain their original return-device contract.
    allocator = tree_cache.token_to_kv_pool_allocator
    if (
        batch is not None
        and type(allocator) is TpuPagedTokenToKVPoolAllocator
        and _can_stage_prefill(batch)
        and batch.req_to_token_pool.req_to_token.device.type == "cpu"
    ):
        return_host = True
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
        return_host=return_host,
    )


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_extend",
            alloc_extend,
            HookType.AROUND,
        ),
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
    ]
