"""EAGLE3 overlap state on the host-padded verify window."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_tpu.experimental.speculative.dflash_info import HostPaddedDraftInput


@dataclass
class HostPaddedEagleDraftInput(HostPaddedDraftInput):
    """HostPaddedDraftInput carrying the EAGLE3 draft seed.

    The bucketed verify window, KV reserve and filter/merge are DSpark's: EAGLE3
    draft steps, verify and draft extend all write inside [n, n + N]. The seed of
    the next draft chain is the inherited Eagle-shaped topk_p / topk_index /
    hidden_states fields, which FutureMap relays by pool row for EAGLE3.

    The fields below are the TPU EAGLE device-accept route, which FutureMap's
    TPU resolvers read for any EAGLE3 input. They stay None: the window already
    addresses the verify rows, so those resolvers fall back to the plain relay.
    """

    accept_lens_device: torch.Tensor | None = None
    device_accept_candidate_locs: torch.Tensor | None = None
    device_accept_selected_cache_locs: torch.Tensor | None = None
