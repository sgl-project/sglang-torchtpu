"""Core's RaggedVerifyLayout plus the RPA metadata a TPU ragged verify precomputes."""

from __future__ import annotations

import torch

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class TpuRaggedVerifyLayout(RaggedVerifyLayout, frozen=True):
    """A ragged TARGET_VERIFY layout whose RPA kv lengths and token count are device outputs.

    The worker's compiled pack op produces them, so building attention metadata runs
    no eager device ops per step.

    Attributes:
      kv_lens: [pbs] int32 seq_lens + verify_lens.
      active_num_tokens: [] int32 sum(verify_lens).
    """

    kv_lens: torch.Tensor | None = None
    active_num_tokens: torch.Tensor | None = None
