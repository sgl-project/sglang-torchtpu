"""Draft forward input of the chain worker: verify-shaped rows plus the hidden the draft reads."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash_info import DFlashVerifyInput


# A dataclass because it extends core's DFlashVerifyInput and registers as a jax pytree.
@dataclass
class ChainDraftForwardInput(DFlashVerifyInput):
    """Per-row draft conditioning, plus the KV length RPA reads during the fused draft.

    Attributes:
      hidden_states: [rows, D] target aux features or the draft's own hidden.
      rpa_draft_kv_lens: [pbs] int32 absolute KV length of the current fused draft
        step; None outside the fused draft, where the attention metadata holds it.
      dsa_seed_topk_capture: [rows, W] int32 buffer an extend-shaped draft forward
        publishes its DSA top-k rows into (IndexTopKShareState.publish); None
        when the draft does not share its indexer top-k.
      dsa_seed_topk_select: [rows] int64 forward rows the capture keeps, or None
        to keep the first rows.
    """

    hidden_states: torch.Tensor | None = None
    rpa_draft_kv_lens: torch.Tensor | None = None
    dsa_seed_topk_capture: torch.Tensor | None = None
    dsa_seed_topk_select: torch.Tensor | None = None


@functools.cache
def register_chain_draft_pytree() -> None:
    """Register ChainDraftForwardInput as a pytree, as direct_compile does DFlashVerifyInput.

    A subclass is its own pytree type, so direct_compile would otherwise not
    trace hidden_states as a graph input. jax is imported here, not at module
    load: CPU runs import this module without jax installed.
    """
    from jax import tree_util

    tree_util.register_dataclass(ChainDraftForwardInput)


def chain_draft_forward_input(
    *,
    hidden_states: torch.Tensor,
    width: int,
    dsa_seed_topk_capture: torch.Tensor | None = None,
    dsa_seed_topk_select: torch.Tensor | None = None,
) -> ChainDraftForwardInput:
    device = hidden_states.device
    return ChainDraftForwardInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=width,
        custom_mask=None,
        hidden_states=hidden_states,
        dsa_seed_topk_capture=dsa_seed_topk_capture,
        dsa_seed_topk_select=dsa_seed_topk_select,
    )


def dsa_seed_topk_buffer(
    *, num_rows: int, width: int | None, device: torch.device | str
) -> torch.Tensor | None:
    """The [num_rows, width] capture a draft forward publishes its DSA top-k into.

    None when width is None (the draft runs its indexer every step). Warmup and
    serving must both build it here: the compiled forward guards on its shape.
    """
    if width is None:
        return None
    return torch.empty((num_rows, width), dtype=torch.int32, device=device)
