"""Host-padding eligibility shared by the TPU platform and staging hooks."""

from sglang.srt.runtime_context import get_flags, get_parallel, get_schedule


def _enabled(batch):
    return batch.spec_algorithm.is_none()


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
