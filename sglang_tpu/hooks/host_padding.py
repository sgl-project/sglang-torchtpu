"""Defer host-built inputs until the decoder's padding boundary.

Why: uploading before padding produces unnecessary transfers and shape variants.
How: AROUND metadata placement, relay factory/entry, and result copy.
Upstream seam issue: not filed (local prototype).
Remove when: upstream provides backend factories for these staging decisions.
"""

from sglang.srt.runtime_context import get_exec, get_flags, get_parallel, get_schedule
from sglang_tpu.envs import envs
from sglang_tpu.host_padding import _can_stage_forward_metadata, _enabled


def sampling_from_batch(
    original, cls, batch, vocab_size, *, sampling_params_on_cpu=False
):
    sampling_params_on_cpu = sampling_params_on_cpu or (
        _enabled(batch) and get_exec().kernel.sampling_backend == "tpu"
    )
    return original(
        cls, batch, vocab_size, sampling_params_on_cpu=sampling_params_on_cpu
    )


def metadata_device(original, batch):
    # Independent of sampling parameter placement. Host allocation must also
    # understand these indices before the forward upload boundary.
    if (
        (_can_stage_forward_metadata(batch) or _stages_spec_metadata(batch))
        and batch.req_to_token_pool.req_to_token.device.type == "cpu"
        and batch.hisparse_coordinator is None
        and not any(req.beam_group is not None for req in batch.reqs)
    ):
        return "cpu"
    return original(batch)


def _stages_spec_metadata(batch) -> bool:
    """A non-overlap experimental-worker batch (a PD prefill, PP included) keeps
    its lengths and request rows on host, so filtering and bucket padding never
    compile per batch size on the device. The worker is on whenever
    SGLANG_TPU_HOST_PADDING_NO_D2H or SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER is."""
    # Lazy: sglang_tpu.experimental imports spec_info, which loads after the platform.
    from sglang_tpu.experimental import uses_host_batch

    # Overlap batches resolve seq_lens on the device through the FutureMap.
    return uses_host_batch(batch) and not batch.enable_overlap


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
            "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
            sampling_from_batch,
            HookType.AROUND,
        ),
    ]
