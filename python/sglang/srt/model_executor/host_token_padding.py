"""Pad prefill token tensors on the host to a backend's compile bucket.

Backends that compile one program per input shape (TPU) register the token
bucket their extend graphs use. Host-built prefill tensors are then padded on
the host before their first host-to-device copy, so neither the copy nor a
device-side pad sees an arbitrary prompt length, and no device read is needed.
Without a registered bucket every helper here is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# (num_tokens, num_reqs) -> padded token count, or None to leave unpadded.
_extend_token_bucket: Callable[[int, int], int | None] | None = None
# Fixed request count an EXTEND forward is padded to.
_extend_batch_cap: int | None = None


def register_extend_buckets(
    token_bucket: Callable[[int, int], int | None], batch_cap: int
) -> None:
    global _extend_token_bucket, _extend_batch_cap
    _extend_token_bucket = token_bucket
    _extend_batch_cap = batch_cap


def extend_token_bucket(num_tokens: int, num_reqs: int) -> int | None:
    """Token count a plain (non-mixed) EXTEND forward is padded to, if any."""
    if _extend_token_bucket is None:
        return None
    return _extend_token_bucket(num_tokens, num_reqs)


def extend_batch_bucket(num_reqs: int) -> int | None:
    """Request count a plain (non-mixed) EXTEND forward is padded to, if any."""
    if _extend_batch_cap is None or num_reqs > _extend_batch_cap:
        return None
    return _extend_batch_cap


def pad_host_list(values: list[int], size: int | None) -> list[int]:
    if size is None or size <= len(values):
        return values
    return list(values) + [0] * (size - len(values))


def pad_host_tokens(t: torch.Tensor, size: int | None) -> torch.Tensor:
    """Zero-pad a host tensor along dim 0 to ``size``, keeping pinned memory."""
    if size is None or size <= t.shape[0]:
        return t
    padded = torch.zeros((size, *t.shape[1:]), dtype=t.dtype, pin_memory=t.is_pinned())
    padded[: t.shape[0]] = t
    return padded
