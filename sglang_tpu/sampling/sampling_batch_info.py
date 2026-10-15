"""Bucketed transfers for host-owned sampling parameters."""

from dataclasses import replace

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

_PARAM_DEFAULTS = {
    "temperatures": 1,
    "top_ps": 1,
    "top_ks": 1,
    "min_ps": 0,
    "sampling_seed": 0,
}


def supports_padded_sampling(info: SamplingBatchInfo) -> bool:
    """Whether all sampling consumers accept physical rows independently of requests.

    Biases, penalties, grammar and observers run before the sampler and still
    carry live-row state. Keep their existing contract until they are bucketed
    together with logits, rather than padding only their sampling parameters.
    """
    return (
        info.params_on_cpu
        and not info.has_custom_logit_processor
        and info.sampling_mask_batch_indices is None
        and info.sampling_support_logprobs_capture_indices is None
        and not any(info.grammars or ())
        and info.logit_bias is None
        and info.acc_additive_penalties is None
        and info.acc_scaling_penalties is None
        and not (
            info.penalizer_orchestrator is not None
            and info.penalizer_orchestrator.is_required
        )
    )


def sampling_params_to_device(
    info: SamplingBatchInfo, *, padded_size: int | None = None
) -> SamplingBatchInfo:
    """Upload parameters with the same row count as the sampling inputs.

    Leave the scheduler's host snapshot intact while overlap scheduling filters
    or merges subsequent batches. A caller using bucketed logits supplies their
    row count; other callers upload live rows directly, without a device slice.
    Device-backed inputs need no preparation.
    """
    if not info.params_on_cpu:
        return info
    size = len(info)
    padded_size = size if padded_size is None else padded_size
    if padded_size < size:
        raise ValueError("Sampling parameter rows cannot be smaller than the batch")
    params = {}
    for name, fill in _PARAM_DEFAULTS.items():
        value = getattr(info, name)
        if value is None:
            continue
        padded = value.new_full((padded_size, *value.shape[1:]), fill)
        padded[:size].copy_(value)
        params[name] = padded.to(info.device, non_blocking=True)
    return replace(info, params_on_cpu=False, **params)
