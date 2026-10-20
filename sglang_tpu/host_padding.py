"""Host-padding eligibility shared by the TPU platform and staging hooks."""

from sglang.srt.runtime_context import get_flags, get_parallel, get_schedule
from sglang_tpu.envs import envs


def _enabled(batch):
    return envs.SGLANG_TPU_HOST_PADDING.get() and batch.spec_algorithm.is_none()


def ordinary_compiled_forward(model_config) -> bool:
    """Single-rank generation forward whose inputs only the TPU padder consumes.

    Parallel, mixed-chunk and token-indexing model paths read forward inputs
    on the device before padding, so they keep their own placement contracts.
    """
    parallel = get_parallel()
    if (
        parallel.pp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.attn_dp_size != 1
        or parallel.attn_dcp_size != 1
        or get_schedule().enable_mixed_chunk
    ):
        return False
    return (
        model_config.is_generation
        and not model_config.use_ngram_embedding
        and not model_config.use_engram
        and not model_config.is_encoder_decoder
    )


def _can_stage_forward_metadata(batch):
    return (
        _enabled(batch)
        and get_flags().capture.enable_torch_compile
        and ordinary_compiled_forward(batch.model_config)
        and batch.mix_running_indices is None
        and batch.dllm_config is None
    )


def check_host_padding_config(runner) -> None:
    """Fail startup when SGLANG_TPU_HOST_PADDING is set but can never take effect.

    Per-batch exclusions (speculative, parallel and mixed batches) only warn,
    since the env still applies to sampling there. Raising here covers the
    server configuration itself, where the KV locations could never return
    on the host.
    """
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang_tpu.allocator_tpu import host_return_allocators

    allocator = runner.token_to_kv_pool_allocator
    problems = []
    if not get_flags().capture.enable_torch_compile:
        problems.append("torch.compile is off, so no padder uploads staged inputs")
    if get_cp_strategy() is not None:
        problems.append("prefill context parallelism splits prompts on the device")
    if not host_return_allocators(allocator):
        problems.append(f"{type(allocator).__name__} returns device KV indices")
    if allocator.page_size <= 1:
        problems.append("page size 1 allocates with alloc(), which has no host return")
    if runner.req_to_token_pool.req_to_token.device.type != "cpu":
        problems.append("req_to_token is on the device")
    if problems:
        raise ValueError(
            "SGLANG_TPU_HOST_PADDING is set, but prefill inputs could never stay "
            "on the host:\n- " + "\n- ".join(problems)
        )
