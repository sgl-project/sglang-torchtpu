"""TpuSampler for the experimental worker's HOST_PADDED batches."""

from __future__ import annotations

import torch

from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded
from sglang_tpu.sampler import TpuSampler


class HostPaddedTpuSampler(TpuSampler):
    """TpuSampler that brings live-row logits up to the HOST_PADDED bucket."""

    def forward(self, logits_output, sampling_info, *args, **kwargs):
        if is_host_padded(sampling_info):
            # An eager outer logits processor (multimodal wrappers) emits live
            # rows; the host sampling params already sit at the batch bucket.
            padded_bs = sampling_info.temperatures.shape[0]
            logits = logits_output.next_token_logits
            if logits is not None and logits.shape[0] < padded_bs:
                logits_output.next_token_logits = torch.nn.functional.pad(
                    logits, (0, 0, 0, padded_bs - logits.shape[0])
                )
        return super().forward(logits_output, sampling_info, *args, **kwargs)
