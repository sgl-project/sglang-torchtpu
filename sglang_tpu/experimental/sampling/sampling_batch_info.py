"""Host-padded sampling metadata for the experimental TPU worker."""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils.common import is_pin_memory_available

_PARAM_FILLS = {
    "temperatures": 1.0,
    "top_ps": 1.0,
    "top_ks": 1,
    "min_ps": 0.0,
    "sampling_seed": 0,
}


def is_host_padded(info) -> bool:
    return isinstance(info, HostPaddedSamplingInfo)


def _pad_host_rows(
    value: torch.Tensor, rows: int, fill: float, pin: bool
) -> torch.Tensor:
    out = torch.full((rows, *value.shape[1:]), fill, dtype=value.dtype, pin_memory=pin)
    out[: value.shape[0]] = value
    return out


@dataclasses.dataclass
class _DeviceBiasCache:
    """Device copies of one host logit_bias tensor, by physical row count."""

    source: torch.Tensor | None = None
    by_rows: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class HostPaddedSamplingInfo(SamplingBatchInfo):
    host_pad_bs: Callable[[int], int] | None = None
    real_batch_size: int = 0
    # Shared with the per-forward copies (dataclasses.replace), so a static bias
    # uploads once per bucket. Filtering or merging replaces logit_bias, and
    # nothing writes a bias tensor after it is built, so identity marks a change.
    device_bias: _DeviceBiasCache = dataclasses.field(
        default_factory=_DeviceBiasCache, repr=False
    )

    @classmethod
    def parameter_device(cls, batch):
        return "cpu"

    def adjusted_from_schedule_batch(self, batch, vocab_size):
        self.host_pad_bs = batch.host_pad_bs
        self.real_batch_size = len(batch.reqs)
        self._store_host_params(self._live_host_params(), self.real_batch_size)

    def __len__(self):
        return self.real_batch_size

    def _filter_params(self, keep_indices, keep_indices_device):
        keep = torch.tensor(keep_indices, dtype=torch.long)
        self._store_host_params(
            {name: value[keep] for name, value in self._live_host_params().items()},
            len(keep_indices),
        )

    def _merge_params(self, other):
        assert isinstance(other, HostPaddedSamplingInfo)
        other_live = other._live_host_params()
        self._store_host_params(
            {
                name: torch.cat([value, other_live[name]])
                for name, value in self._live_host_params().items()
            },
            len(self) + len(other),
        )

    def _live_host_params(self) -> dict[str, torch.Tensor]:
        return {
            name: value[: self.real_batch_size]
            for name in _PARAM_FILLS
            if (value := getattr(self, name)) is not None
        }

    def _store_host_params(
        self, live: dict[str, torch.Tensor], batch_size: int
    ) -> None:
        """Pad live host param rows to the bucket of batch_size, in place."""
        assert self.host_pad_bs is not None
        padded_bs = self.host_pad_bs(batch_size)
        pin = is_pin_memory_available(self.device)
        for name, value in live.items():
            fill = _PARAM_FILLS[name]
            setattr(self, name, _pad_host_rows(value, padded_bs, fill, pin))
        self.real_batch_size = batch_size

    def host_params_at(self, rows: int) -> dict[str, torch.Tensor]:
        """Host sampling params padded to rows, at least this info's own bucket."""
        return {
            name: _pad_host_rows(value[: self.real_batch_size], rows, fill, False)
            if value.shape[0] < rows
            else value
            for name, fill in _PARAM_FILLS.items()
            if (value := getattr(self, name)) is not None
        }

    def device_logit_bias(self, rows: int) -> torch.Tensor | None:
        """Upload live bias rows at the forward's physical bucket, with neutral padding."""
        if self.logit_bias is None:
            return None
        cache = self.device_bias
        if cache.source is not self.logit_bias:
            cache.source, cache.by_rows = self.logit_bias, {}
        bias = cache.by_rows.get(rows)
        if bias is None:
            bias = _pad_host_rows(
                self.logit_bias, rows, 0.0, is_pin_memory_available(self.device)
            ).to(self.device, non_blocking=True)
            cache.by_rows[rows] = bias
        return bias

    def _apply_post_grammar_logits_transforms(self, logits):
        # Prefill only, outside compiled forwards: not warmed. Like the other
        # vocabulary-sized live-extent pads, its eager programs compile once per
        # new row count, the first time a biased request needs them.
        bias = self.device_logit_bias(logits.shape[0])
        if bias is not None:
            logits.add_(bias)

    def on_device(self) -> SamplingBatchInfo:
        """This info with the sampling param tensors on self.device; HOST_PADDED keeps its bucket rows."""
        moved = {
            name: value.to(self.device, non_blocking=True)
            for name in _PARAM_FILLS
            if (value := getattr(self, name)) is not None
        }
        return dataclasses.replace(
            self,
            logit_bias=self.device_logit_bias(self.temperatures.shape[0]),
            # The copy's bias is already on the device; keep this info's cache.
            device_bias=_DeviceBiasCache(),
            **moved,
        )
