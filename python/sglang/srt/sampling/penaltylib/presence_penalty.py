import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils.common import is_pin_memory_available


class BatchedPresencePenalizer(_BatchedPenalizer):
    """
    Presence penalizer penalizes tokens based on their presence in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.cumulated_presence_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.presence_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.presence_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.cumulated_presence_penalties.scatter_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.presence_penalties,
        )

    def _seed_history(self, rows: list, tokens: list):
        # Set semantics (scatter_ of the row's value): a token present in the
        # history holds the row's value, the same bits however often it occurs.
        device = self.orchestrator.device
        for row, toks in zip(rows, tokens):
            mask = torch.zeros(self.orchestrator.vocab_size, dtype=torch.bool)
            mask[torch.tensor(toks, dtype=torch.int64)] = True
            index = torch.tensor([row], dtype=torch.int64, device=device)
            current = self.cumulated_presence_penalties.index_select(0, index)
            seeded = torch.where(
                mask.to(device).unsqueeze(0),
                self.presence_penalties.index_select(0, index),
                current,
            )
            self.cumulated_presence_penalties.index_copy_(0, index, seeded)

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_presence_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.presence_penalties = self.presence_penalties[keep_indices]
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedPresencePenalizer"):
        self.presence_penalties = torch.cat(
            [self.presence_penalties, their.presence_penalties], dim=0
        )
        self.cumulated_presence_penalties = torch.cat(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("presence_penalties", "cumulated_presence_penalties"):
            if hasattr(self, name):
                delattr(self, name)
