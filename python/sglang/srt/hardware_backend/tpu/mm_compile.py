"""Shared serving and asynchronous warmup boundaries for multimodal glue."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tensor_signature(*tensors: torch.Tensor) -> tuple:
    return tuple(
        (tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors
    )


def merge_mm_rows(
    dest: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    src = src.to(dtype=dest.dtype)
    if src.shape[0] < dest.shape[0]:
        src = F.pad(src, (0, 0, 0, dest.shape[0] - src.shape[0]))
    return torch.where(mask, src.index_select(0, indices), dest)


def gather_prefill_states(
    hidden_states: torch.Tensor, extend_seq_lens: torch.Tensor
) -> torch.Tensor:
    last_index = torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32) - 1
    return hidden_states[last_index]


class MmCompiledOps:
    def __init__(self):
        from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
            direct_compile,
        )

        self.merge = direct_compile(merge_mm_rows, specialization_key=tensor_signature)
        self.gather = direct_compile(
            gather_prefill_states, specialization_key=tensor_signature
        )
