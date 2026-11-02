# SPDX-License-Identifier: Apache-2.0
"""Receive RL weight buckets over ICI from a trainer sharing the TPU slice.

The trainer (miles, --update-weight-transfer-mode ici) packs each bucket into one
buffer and sends it to every engine rank, from the trainer rank the payload names
for that rank. Every rank views the full tensors back out of it; /plugin_weight_update
then loads them through each selected runner's load_weights, as
/update_weights_from_distributed does, so fused, MoE and MLA parameters load as they
do from a checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.weight_updater import (
        PluginWeightUpdateContext,
    )
    from sglang_tpu.rl.shared_slice import SharedSlice

ICI_RECEIVE_HANDLER = "ici.receive"
# The trainer's send buffer dtype (miles protocols/ici.py); a recv must match it.
_BUFFER_DTYPE = torch.int16


def receive_bucket(
    *,
    context: PluginWeightUpdateContext,
    payload: dict[str, Any],
    shared_slice: SharedSlice,
) -> list[tuple[str, torch.Tensor]]:
    """Receive one packed bucket from this rank's trainer rank.

    payload: {"src_slice_ranks": the trainer slice rank each engine rank (by job
    rank) receives from, "tag": int, "num_bytes": int,
    "tensors": [[name, dtype, shape, byte_offset], ...]}.
    """
    buffer = torch.empty(
        payload["num_bytes"] // _BUFFER_DTYPE.itemsize,
        dtype=_BUFFER_DTYPE,
        device="tpu",
    )
    # recv rebinds `buffer` to the received storage, so views come only after it
    shared_slice.backend.recv(
        [buffer], payload["src_slice_ranks"][shared_slice.job_rank], payload["tag"]
    ).wait()
    # wait() returns once the receive is posted, not once the data lands; a read
    # waits for it, and freeing the buffer before that crashes the sender
    buffer[:1].cpu()
    return unpack_bucket(buffer, payload["tensors"])


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
