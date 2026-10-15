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


def sampling_params_to_device(info: SamplingBatchInfo) -> SamplingBatchInfo:
    """Upload bucket shapes and expose only the rows present in the logits.

    Leave the scheduler's host snapshot intact while overlap scheduling filters
    or merges subsequent batches. Device-backed inputs need no preparation.
    """
    if not info.params_on_cpu:
        return info
    size = len(info)
    padded_size = 1 << (max(size, 1) - 1).bit_length()
    params = {}
    for name, fill in _PARAM_DEFAULTS.items():
        value = getattr(info, name)
        if value is None:
            continue
        padded = value.new_full((padded_size, *value.shape[1:]), fill)
        padded[:size].copy_(value)
        params[name] = padded.to(info.device, non_blocking=True)[:size]
    return replace(info, params_on_cpu=False, **params)
