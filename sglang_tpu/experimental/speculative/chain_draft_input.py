"""Forward inputs of the chain worker: the target verify rows, and the draft's
verify-shaped rows plus the hidden it reads."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash_info import DFlashVerifyInput


# Dataclasses because they extend core's DFlashVerifyInput and register as jax pytrees.
@dataclass
class ChainVerifyInput(DFlashVerifyInput):
    """Verify rows of the chain worker.

    Attributes:
      verify_kv_lens: [pbs] int32 RPA KV length of a forward over the verify
        rows (select_verify_rows builds it); None derives it from seq_lens.
    """

    verify_kv_lens: torch.Tensor | None = None


@dataclass
class ChainDraftForwardInput(ChainVerifyInput):
    """Per-row draft conditioning, plus the KV length RPA reads during the fused draft.

    Attributes:
      hidden_states: [rows, D] target aux features or the draft's own hidden.
      rpa_draft_kv_lens: [pbs] int32 absolute KV length of the current fused draft
        step; None outside the fused draft, where the attention metadata holds it.
    """

    hidden_states: torch.Tensor | None = None
    rpa_draft_kv_lens: torch.Tensor | None = None


@functools.cache
def register_chain_draft_pytree() -> None:
    """Register the chain inputs as pytrees, as direct_compile does DFlashVerifyInput.

    A subclass is its own pytree type, so direct_compile would otherwise not
    trace its fields as graph inputs. jax is imported here, not at module
    load: CPU runs import this module without jax installed.
    """
    from jax import tree_util

    tree_util.register_dataclass(ChainVerifyInput)
    tree_util.register_dataclass(ChainDraftForwardInput)


def chain_draft_forward_input(
    *,
    hidden_states: torch.Tensor,
    width: int,
    verify_kv_lens: torch.Tensor | None = None,
) -> ChainDraftForwardInput:
    device = hidden_states.device
    return ChainDraftForwardInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=width,
        custom_mask=None,
        hidden_states=hidden_states,
        verify_kv_lens=verify_kv_lens,
    )
