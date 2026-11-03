# SPDX-License-Identifier: Apache-2.0
"""Receive RL weight buckets over ICI from a trainer sharing the TPU slice.

The trainer (miles, --update-weight-transfer-mode ici) packs what each of its ranks
holds of a bucket into one buffer per rank and sends it to the engine ranks that rank
serves. The payload lists, per engine rank, the transfers to receive, each with its
source and manifest, so trainer ranks may hold different tensors (their own experts
under expert parallelism). Every engine rank views the tensors back out;
/plugin_weight_update then loads them through each selected runner's load_weights,
as /update_weights_from_distributed does, so fused, MoE and MLA parameters load as
they do from a checkpoint.
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
    """Receive this rank's share of one bucket: every transfer the payload lists for it.

    payload: {"tag": int, "receives": per engine rank (by job rank), a list of
    {"src_slice_rank": int, "num_bytes": int,
     "tensors": [[name, dtype, shape, byte_offset], ...]}}.
    """
    _, target = context.runners[0]
    transfers = payload["receives"][shared_slice.job_rank]
    buffers = []
    try:
        # post every receive first, so no sender waits on another's transfer
        for transfer in transfers:
            buffer = torch.empty(
                transfer["num_bytes"] // _BUFFER_DTYPE.itemsize,
                dtype=_BUFFER_DTYPE,
                device=target.device,
            )
            # recv rebinds `buffer` to the received storage, so views come only after it
            shared_slice.backend.recv(
                [buffer], transfer["src_slice_rank"], payload["tag"]
            ).wait()
            buffers.append(buffer)
    finally:
        # wait() returns once the receive is posted, not once the data lands; a read
        # waits for it, and freeing a posted buffer before that crashes the sender,
        # so every posted buffer is fenced even when a later post failed
        for buffer in buffers:
            buffer[:1].cpu()
    return [
        named_tensor
        for transfer, buffer in zip(transfers, buffers)
        for named_tensor in unpack_bucket(buffer, transfer["tensors"])
    ]


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
