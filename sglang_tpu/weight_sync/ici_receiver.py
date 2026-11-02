# SPDX-License-Identifier: Apache-2.0
"""Receive RL weight buckets over ICI from a trainer sharing the TPU slice.

The trainer (miles, --update-weight-transfer-mode ici) packs each bucket into one
buffer and sends it to every engine TP rank, from the trainer rank the payload names
for that rank. Every rank views the full tensors back out and loads them through the
model's own load_weights, so fused, MoE and MLA parameters load as they do from a
checkpoint. Served as "ici.receive" through /plugin_weight_update, inside the begin/end
weight-update session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.weight_updater import (
        PluginWeightUpdateContext,
    )
    from sglang_tpu.weight_sync.shared_slice import SharedSlice

ICI_RECEIVE_HANDLER = "ici.receive"
# The trainer's send buffer dtype (miles protocols/ici.py); a recv must match it.
_BUFFER_DTYPE = torch.int16


def receive_bucket(
    *,
    context: PluginWeightUpdateContext,
    payload: dict[str, Any],
    shared_slice: SharedSlice,
) -> int:
    """Receive one packed bucket from this TP rank's trainer rank and load it.

    payload: {"src_slice_ranks": the trainer slice rank each TP rank receives
    from, "tag": int, "num_bytes": int,
    "tensors": [[name, dtype, shape, byte_offset], ...]}.
    Returns the number of tensors loaded.
    """
    buffer = torch.empty(
        payload["num_bytes"] // _BUFFER_DTYPE.itemsize,
        dtype=_BUFFER_DTYPE,
        device="tpu",
    )
    # recv rebinds `buffer` to the received storage, so views come only after it
    shared_slice.backend.recv(
        [buffer], payload["src_slice_ranks"][context.tp_rank], payload["tag"]
    ).wait()
    # wait() returns once the receive is posted, not once the data lands; a read
    # waits for it, and freeing the buffer before that crashes the sender
    buffer[:1].cpu()
    # checked only once the bucket has landed: a rank that raised before its recv
    # would leave the trainer's send blocked until the backend times out
    _check_engine_takes_updates(context)
    named_tensors = unpack_bucket(buffer, payload["tensors"])
    context.model.load_weights(named_tensors)
    return len(named_tensors)


def _check_engine_takes_updates(context: PluginWeightUpdateContext) -> None:
    if context.has_draft_model:
        raise ValueError("ICI weight sync does not cover draft models")
    # the session restores post-processed weights before loading, finalizes after
    if not context.session_open:
        raise RuntimeError("ici.receive needs an open begin_weight_update() session")
    # derived-weight caches would keep serving the old weights, as for every other update path
    error = _unsupported_derived_weight_cache_error(context.model)
    if error is not None:
        raise RuntimeError(error)


def unpack_bucket(
    buffer: torch.Tensor, entries: list[list[Any]]
) -> list[tuple[str, torch.Tensor]]:
    """Views of each ``[name, dtype, shape, byte_offset]`` entry's tensor in buffer."""
    named_tensors = []
    for name, dtype_name, shape, byte_offset in entries:
        dtype = getattr(torch, dtype_name)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{name}: {dtype_name!r} is not a torch dtype")
        numel = torch.Size(shape).numel()
        start = byte_offset // buffer.element_size()
        length = -(-numel * dtype.itemsize // buffer.element_size())
        # [:numel] drops the padding byte of an odd-length 1-byte tensor
        region = buffer[start : start + length].view(dtype)[:numel]
        named_tensors.append((name, region.view(shape)))
    return named_tensors
