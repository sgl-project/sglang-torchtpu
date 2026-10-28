import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils import get_compiler_backend, is_npu
from sglang.srt.utils.common import is_pin_memory_available

_is_npu = is_npu()

# Keep the [rows, vocab] penalty state at a power-of-two row count so its filter
# and merge gathers have one shape per bucket pair instead of one per exact
# (old, new) batch-size pair. Backends that compile one program per input shape
# (such as eager TPU) otherwise stall for a compile on every new pair of batch
# sizes. Live rows come first and padded rows are never read, so the live values
# are unchanged.
_BUCKETED_ROWS = envs.SGLANG_BUCKETED_PENALTY_ROWS.get()


def _row_bucket(rows: int) -> int:
    return 1 << max(rows - 1, 0).bit_length()


def _pad_index(index: torch.Tensor, rows: int) -> torch.Tensor:
    if index.shape[0] == rows:
        return index
    return torch.cat([index, index.new_zeros(rows - index.shape[0])])


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their presence in the generated output.
    """

    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        reqs = self.orchestrator.reqs()
        live = len(reqs)
        rows = _row_bucket(live) if _BUCKETED_ROWS else live
        self._live_rows = live
        self.cumulated_repetition_penalties = torch.ones(
            (rows, self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        self.repetition_penalties = (
            torch.tensor(
                data=[req.sampling_params.repetition_penalty for req in reqs]
                + [1.0] * (rows - live),
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        index = output_ids
        if _BUCKETED_ROWS:
            # Padded rows are never read; scattering into them keeps one shape
            # per row bucket.
            index = _pad_index(output_ids, self.cumulated_repetition_penalties.shape[0])
        self.cumulated_repetition_penalties.scatter_(
            dim=1,
            index=index.unsqueeze(1),
            src=self.repetition_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        apply_scaling_penalties(logits, self.get_scaling_penalties())
        return logits

    def get_scaling_penalties(self) -> torch.Tensor:
        if _BUCKETED_ROWS:
            return self.cumulated_repetition_penalties[: self._live_rows]
        return self.cumulated_repetition_penalties

    def _filter(self, keep_indices: torch.Tensor):
        if _BUCKETED_ROWS:
            live = keep_indices.shape[0]
            index = _pad_index(keep_indices, _row_bucket(live))
            self.repetition_penalties = self.repetition_penalties.index_select(0, index)
            self.cumulated_repetition_penalties = (
                self.cumulated_repetition_penalties.index_select(0, index)
            )
            self._live_rows = live
            return
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        if _BUCKETED_ROWS:
            ours_live = self._live_rows
            ours_rows = self.cumulated_repetition_penalties.shape[0]
            theirs_live = their._live_rows
            live = ours_live + theirs_live
            rows = _row_bucket(live)
            index = torch.tensor(
                list(range(ours_live))
                + [ours_rows + i for i in range(theirs_live)]
                + [0] * (rows - live),
                dtype=torch.int64,
            ).to(self.orchestrator.device, non_blocking=True)
            self.repetition_penalties = torch.cat(
                [self.repetition_penalties, their.repetition_penalties], dim=0
            ).index_select(0, index)
            self.cumulated_repetition_penalties = torch.cat(
                [
                    self.cumulated_repetition_penalties,
                    their.cumulated_repetition_penalties,
                ],
                dim=0,
            ).index_select(0, index)
            self._live_rows = live
            return
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):
            if hasattr(self, name):
                delattr(self, name)
