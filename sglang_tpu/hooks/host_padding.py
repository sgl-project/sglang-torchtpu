"""Defer host-built inputs until the decoder's padding boundary.

Why: uploading before padding produces unnecessary transfers and shape variants.
How: AROUND metadata placement, relay factory/entry, result copy, and allocation.
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


def _can_stage_forward_metadata(batch):
    parallel = get_parallel()
    # Ordinary compiled forwards upload metadata after padding, including VL.
    # Parallel and mixed-chunk paths retain their existing placement contracts.
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
        and not batch.model_config.is_encoder_decoder
        and batch.dllm_config is None
    )


def _can_stage_prefill(batch):
    # Multimodal embedding preparation still consumes device token IDs before
    # decoder padding. This restriction does not apply to scheduler metadata.
    return _can_stage_forward_metadata(batch) and not batch.model_config.is_multimodal


def metadata_device(original, batch):
    # Independent of sampling parameter placement. Host allocation must also
    # understand these indices before the forward upload boundary.
    if (
        _can_stage_forward_metadata(batch)
        and batch.req_to_token_pool.req_to_token.device.type == "cpu"
        and batch.hisparse_coordinator is None
        and not any(req.beam_group is not None for req in batch.reqs)
    ):
        return "cpu"
    return original(batch)


def mrope_decode_device(original, batch, device):
    # Unlike token staging, mRoPE construction supports multimodal decode.
    # Parallel/mixed paths have separate metadata preparation contracts.
    parallel = get_parallel()
    if (
        _enabled(batch)
        and get_flags().capture.enable_torch_compile
        and parallel.pp_size == 1
        and parallel.attn_cp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.attn_dcp_size == 1
        and not get_schedule().enable_mixed_chunk
        and batch.forward_mode.is_decode()
    ):
        return "cpu"
    return original(batch, device)


def stage_prefill_tokens(original, batch):
    if _can_stage_prefill(batch):
        return batch.prefill_input_ids_cpu
    return original(batch)


def extend_metadata_device(original, batch, device):
    # VL token IDs reach embedding preparation before the decoder, but extend
    # lengths and offsets are consumed after the decoder's padded upload.
    if _can_stage_forward_metadata(batch):
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
    # VL embedding preparation needs device token IDs, but not KV locations.
    allocator = tree_cache.token_to_kv_pool_allocator
    if (
        batch is not None
        and type(allocator) is TpuPagedTokenToKVPoolAllocator
        and _can_stage_forward_metadata(batch)
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


def create_future_map(original, algorithm, device, req_to_token_pool, *args, **kwargs):
    if (
        envs.SGLANG_TPU_HOST_PADDING.get()
        and algorithm.is_none()
        and getattr(req_to_token_pool, "_tpu_relay_bucketer", None) is not None
    ):
        from sglang_tpu.host_padding_relay import HostPaddingFutureMap

        return HostPaddingFutureMap(
            device, algorithm, req_to_token_pool, *args, **kwargs
        )
    return original(algorithm, device, req_to_token_pool, *args, **kwargs)


def resolve_forward_inputs(original, batch, future_map):
    from sglang_tpu.host_padding_relay import HostPaddingFutureMap

    if (
        isinstance(future_map, HostPaddingFutureMap)
        and _can_stage_forward_metadata(batch)
        and batch.forward_mode.is_decode()
        and batch.prefill_input_ids_cpu is None
        and batch.input_ids is None
        and batch.req_pool_indices.device.type == "cpu"
    ):
        batch.input_ids = future_map.gather(batch.req_pool_indices)
    return original(batch, future_map)


def relay_forward_payload(original, scheduler, batch, future_indices, result):
    from sglang_tpu.host_padding_relay import HostPaddingFutureMap

    padded = (
        isinstance(scheduler.future_map, HostPaddingFutureMap)
        and result.has_sampled_token_ids
        and result.next_draft_input is None
    )
    if padded:
        result._tpu_live_token_rows = len(future_indices)
    value = original(scheduler, batch, future_indices, result)
    if padded and not scheduler.enable_overlap:
        # Synchronous scheduling already consumes the tokens on CPU next.
        result.next_token_ids = result.next_token_ids.cpu()[
            : result._tpu_live_token_rows
        ]
    return value


def copy_result_to_cpu(original, result, *args, **kwargs):
    value = original(result, *args, **kwargs)
    live = getattr(result, "_tpu_live_token_rows", None)
    if live is not None:
        # This is a host view, with no read of pending D2H contents. The original
        # copy_done event still owns completion and consumers wait on it.
        result.next_token_ids = result.next_token_ids[:live]
    return value


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.create_future_map",
            create_future_map,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
            resolve_forward_inputs,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler.Scheduler._relay_forward_payload",
            relay_forward_payload,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.utils.GenerationBatchResult.copy_to_cpu",
            copy_result_to_cpu,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.forward_batch_info._mrope_decode_device",
            mrope_decode_device,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.schedule_batch.ScheduleBatch.metadata_device",
            metadata_device,
            HookType.AROUND,
        ),
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
