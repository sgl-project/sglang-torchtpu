"""Logit rows of a TPU prefill (EXTEND) forward, for models whose compiled
forward can run the last decoder layer's FFN on those rows only (the Watermelon
RMT four-stage path, "LR1").

A prefill forward takes the logits of each request's last row only. With
``SGLANG_TPU_FOUR_STAGE_LOGIT_ROWS=1`` the compiled-forward shim puts the rows
on ``ForwardBatch.logit_rows`` before every EXTEND forward: one int32 ``[6]``
device tensor, ``[row0, row1, row2, row3, count, identity]``, built on the
host. The shape never changes and every EXTEND forward carries the field, so
direct_compile lifts it as one more tensor input of the same EXTEND graph: no
new specialization. ``identity`` set means "run the last layer on every row";
the model reads only the tensor's values, never a Python branch on them.

This module has no TorchTPU dependency, so its rules run in the CPU tests.
"""

from __future__ import annotations

import os
from typing import Any

import torch

ENV = "SGLANG_TPU_FOUR_STAGE_LOGIT_ROWS"
# One row per request: a prefill forward holds at most --prefill-max-requests
# requests (4 in the served recipes); more requests take the identity path.
MAX_LOGIT_ROWS = 4
PACKED_WIDTH = MAX_LOGIT_ROWS + 2  # rows, count, identity


def logit_rows_enabled() -> bool:
    return os.environ.get(ENV) == "1"


def takes_logit_rows(forward_batch: Any) -> bool:
    """Whether this forward carries the field: plain EXTEND forwards only (not
    MIXED, TARGET_VERIFY, draft or split prefill), so no other graph changes."""
    return getattr(forward_batch.forward_mode, "name", None) == "EXTEND"


def identity_reasons(
    forward_batch: Any,
    *,
    wants_input_logprob: bool,
    capture_hidden_mode: Any,
    real_b: int | None,
) -> list[str]:
    """Why this EXTEND forward must keep the full last layer (empty: it can
    prune). Read the batch before the shim pins or rewrites its fields:
    ``wants_input_logprob`` is ``needs_input_logprob(batch)`` and
    ``capture_hidden_mode`` the batch's own mode, not the FULL capture the
    shim presents for SGLANG_TPU_ENABLE_PROMPT_LOGPROB."""
    reasons = []
    if wants_input_logprob:
        # Prompt / echo logprobs and scoring read every prefill row.
        reasons.append("input logprobs")
    if capture_hidden_mode is not None and capture_hidden_mode.is_full():
        # return_hidden_states (full), speculative capture: every row.
        reasons.append("hidden-state capture of every row")
    if getattr(forward_batch, "spec_info", None) is not None:
        reasons.append("speculative input")
    lens = _real_extend_lens(forward_batch, real_b)
    if lens is None:
        reasons.append("no host extend lengths")
    elif len(lens) > MAX_LOGIT_ROWS:
        reasons.append(f"{len(lens)} requests > {MAX_LOGIT_ROWS} logit rows")
    elif any(length <= 0 for length in lens):
        reasons.append("a request with no extend rows")
    return reasons


def build_logit_rows(
    forward_batch: Any, *, identity: bool, real_b: int | None, device: Any
) -> torch.Tensor:
    """The packed int32 ``[6]`` field: each request's last row in the
    forward's rows (``cumsum(extend_seq_lens) - 1``, the rows the logits
    processor selects), padded with row 0, the request count and the identity
    flag. The identity path carries rows 0..3, count 0 and identity 1."""
    lens = None if identity else _real_extend_lens(forward_batch, real_b)
    if identity or not lens:
        packed = list(range(MAX_LOGIT_ROWS)) + [0, 1]
    else:
        rows, end = [], 0
        for length in lens:
            end += int(length)
            rows.append(end - 1)
        packed = rows + [0] * (MAX_LOGIT_ROWS - len(rows)) + [len(rows), 0]
    return torch.tensor(packed, dtype=torch.int32).to(device)


def _real_extend_lens(forward_batch: Any, real_b: int | None) -> list[int] | None:
    """The live requests' extend lengths from the host list (the padded list
    has one entry per padded request)."""
    lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if lens is None:
        return None
    lens = [int(length) for length in lens]
    if real_b is not None:
        if real_b > len(lens):
            return None
        lens = lens[:real_b]
    return lens
