"""Host-padded sampling metadata for the experimental TPU worker."""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch

from sglang.srt.sampling.penaltylib import repetition_penalty
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
class HostPaddedSamplingInfo(SamplingBatchInfo):
    host_pad_bs: Callable[[int], int] | None = None
    real_batch_size: int = 0

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
        """Host sampling params with exactly the selected bucket's rows."""
        if rows < self.real_batch_size:
            raise ValueError(
                f"sampling bucket {rows} is smaller than live batch {self.real_batch_size}"
            )
        return {
            name: (
                _pad_host_rows(value[: self.real_batch_size], rows, fill, False)
                if value.shape[0] != rows
                else value
            )
            for name, fill in _PARAM_FILLS.items()
            if (value := getattr(self, name)) is not None
        }

    def on_device(self) -> SamplingBatchInfo:
        """This info with the sampling param tensors on self.device; HOST_PADDED keeps its bucket rows."""
        moved = {
            name: value.to(self.device, non_blocking=True)
            for name in _PARAM_FILLS
            if (value := getattr(self, name)) is not None
        }
        return dataclasses.replace(self, **moved)

    def device_penalties(
        self, rows: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(additive, scaling) penalty snapshots at rows, padded neutrally (0 and 1).

        The snapshot holds the live rows. Like the shared penalizer state it comes
        from, its pad is eager and compiles once per new live size.
        """
        return (
            self._pad_penalty_rows(self.acc_additive_penalties, rows, 0.0),
            self._pad_penalty_rows(self.acc_scaling_penalties, rows, 1.0),
        )

    def _apply_pre_grammar_logits_transforms(self, logits: torch.Tensor) -> None:
        # Extend logits keep the bucket rows; the penalty state has the live rows.
        additive, scaling = self.device_penalties(logits.shape[0])
        if additive is not None:
            logits.add_(additive)
        if scaling is not None:
            # Looked up at call time: core wraps it in torch.compile, and the TPU
            # platform unwraps the module attribute only after this import.
            repetition_penalty.apply_scaling_penalties(logits, scaling)
        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            self.penalizer_orchestrator.apply(logits[: len(self)])

    def _pad_penalty_rows(
        self, value: torch.Tensor | None, rows: int, fill: float
    ) -> torch.Tensor | None:
        if value is None:
            return None
        value = value[: self.real_batch_size]
        if value.device.type == "cpu":
            pin = is_pin_memory_available(self.device)
            return _pad_host_rows(value, rows, fill, pin).to(
                self.device, non_blocking=True
            )
        return torch.nn.functional.pad(
            value, (0, 0, 0, rows - value.shape[0]), value=fill
        )
