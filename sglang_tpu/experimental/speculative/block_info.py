"""Block-drafter (DFlash/DSpark) overlap state: the shared host-padded draft input, relaying only the bonus token."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.speculative.spec_info import SpecInputType
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput


@dataclass
class HostPaddedBlockDraftInput(HostPaddedDraftInput):
    SPEC_INPUT_TYPE = SpecInputType.DFLASH_DRAFT
