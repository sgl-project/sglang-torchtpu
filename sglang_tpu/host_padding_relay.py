"""Bucketed non-spec token relay with private destinations for padded rows."""

import torch

from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap


class HostPaddingFutureMap(FutureMap):
    def __init__(self, device, spec_algo, req_to_token_pool, *args, **kwargs):
        super().__init__(device, spec_algo, req_to_token_pool, *args, **kwargs)
        self.bucketer = req_to_token_pool._tpu_relay_bucketer
        # Distinct scratch destinations avoid duplicate-index scatter writes.
        # No scratch slot aliases a request, including the reserved request row 0.
        host_tokens = torch.zeros(
            self.req_pool_size + self.bucketer.max_val, dtype=torch.int64
        )
        if overlap_utils._DEBUG_ASSERT:
            host_tokens[: self.req_pool_size] = -1
        self.output_tokens_buf = host_tokens.to(device)

    def _indices(self, indices, width):
        if indices.device.type != "cpu":
            raise ValueError("Bucketed relay requires host-owned request indices")
        if width < len(indices):
            raise ValueError("Relay values cannot have fewer rows than requests")
        padded = torch.arange(
            self.req_pool_size, self.req_pool_size + width, dtype=torch.int64
        )
        padded[: len(indices)].copy_(indices)
        return padded.to(self.device, non_blocking=True)

    def _store(self, buf, indices, values):
        if buf is not self.output_tokens_buf or indices.device.type != "cpu":
            return super()._store(buf, indices, values)
        if values.device.type == "cpu":
            width = self.bucketer.next_bucket(len(values))
            padded = values.new_zeros(width)
            padded[: len(values)].copy_(values)
            values = padded.to(self.device)
        # Supported padded sampling supplies a whole bucket. Other sampling
        # modes retain their live-size contract until their outputs are padded.
        padded_indices = self._indices(indices, len(values))
        buf.index_copy_(0, padded_indices, values.to(buf.dtype))

    def gather(self, indices):
        width = self.bucketer.next_bucket(len(indices))
        device_indices = self._indices(indices, width)
        tokens = self.output_tokens_buf[device_indices]
        if overlap_utils._DEBUG_ASSERT:
            valid = (torch.arange(width) < len(indices)).to(self.device)
            tokens = torch.where(valid, tokens, 0)
            overlap_utils._assert_nonneg_and_invalidate(
                tokens, self.output_tokens_buf, device_indices
            )
        return tokens
