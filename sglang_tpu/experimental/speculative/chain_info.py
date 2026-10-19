"""Chain-drafter (MTP) overlap state: the shared host-padded draft input, relaying only the bonus token."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.spec_info import SpecInputType
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput


@dataclass
class HostPaddedChainDraftInput(HostPaddedDraftInput):
    # The draft hidden lives in TpuChainDraftWorker's req-pool-indexed carry table,
    # so the relay carries no seed; the DFlash type keeps core off the EAGLE routes.
    SPEC_INPUT_TYPE = SpecInputType.DFLASH_DRAFT

    # RelayPayload.from_draft_input reads these for every draft input.
    topk_p: torch.Tensor | None = None
    topk_index: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
