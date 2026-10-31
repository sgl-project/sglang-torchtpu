"""Host-padding eligibility shared by the TPU platform and staging hooks."""

from abc import ABC

from sglang.srt.runtime_context import get_flags, get_parallel, get_schedule


class HostSamplingAlgorithm(ABC):
    """Algorithm whose worker stages request-owned CPU sampling parameters.

    Backend adapters register their algorithm class after installing a worker
    that materializes a device snapshot before sampling.
    """


def _enabled(batch):
    return batch.spec_algorithm.is_none()


def ordinary_compiled_forward(
    model_config, *, allow_pipeline_parallel: bool = False
) -> bool:
    """Generation forward whose staged inputs the TPU padder consumes.

    Only extend lengths and their derived positions opt into PP here. Other
    parallel paths and token-indexing models retain their placement contracts.
    """
    parallel = get_parallel()
    if (
        (parallel.pp_size != 1 and not allow_pipeline_parallel)
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


def _can_stage_forward_metadata(batch, *, allow_pipeline_parallel: bool = False):
    return (
        _enabled(batch)
        and get_flags().capture.enable_torch_compile
        and ordinary_compiled_forward(
            batch.model_config, allow_pipeline_parallel=allow_pipeline_parallel
        )
        and batch.mix_running_indices is None
        and batch.dllm_config is None
    )
