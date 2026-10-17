"""EAGLE3 overlap state on the host-padded verify window."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.spec_info import SpecInputType
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput


@dataclass
class HostPaddedEagleDraftInput(HostPaddedDraftInput):
    """HostPaddedDraftInput carrying the EAGLE3 draft seed.

    EAGLE3 draft steps, verify and draft extend all write inside the shared
    [n, n + N] window. topk_p / topk_index / hidden_states seed the next draft
    chain; FutureMap relays them by pool row for EAGLE3 (RelayPayload).

    The device-accept fields are the TPU EAGLE route FutureMap's TPU resolvers
    read for any EAGLE3 input. They stay None: the window already addresses the
    verify rows, so those resolvers fall back to the plain relay.
    """

    SPEC_INPUT_TYPE = SpecInputType.EAGLE_DRAFT
    RELAYED_FIELDS = HostPaddedDraftInput.RELAYED_FIELDS + (
        "topk_p",
        "topk_index",
        "hidden_states",
    )

    topk_p: torch.Tensor | None = None
    topk_index: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    accept_lens_device: torch.Tensor | None = None
    device_accept_candidate_locs: torch.Tensor | None = None
    device_accept_selected_cache_locs: torch.Tensor | None = None
