"""Paged buffer that carries the draft's target hidden states from a PD prefill to its decode.

With SGLANG_TPU_PD_DECODE_DRAFT_PREFILL the prefill skips the draft over the prompt
and writes each prompt token's packed target hidden (EAGLE3 aux layers, DFlash
target layers) into this buffer instead; Raiden moves it with the target KV under
the same page ids, and the decode reads it back to build the draft KV and carry.

The buffer is laid out as one extra fused KV layer, [num_pages, page_size,
heads, 2, lanes], so Raiden's PP -> TP reshard moves it unchanged: a token's row
is cut into total_kv_heads equal slices, and each decode attention-TP rank gets
the slices of its own kv heads, which an all-gather over attention TP joins back.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang_tpu.kv_row_write import kv_write_segments
from sglang_tpu.kv_span_write import get_kv_span_write_op

_LANES = 128
# bf16 packs two values per 32-bit word, the fused layout's K/V pair axis.
_PACKING = 2

_pd_hidden_pool: PdHiddenPool | None = None


def register_pd_hidden_pool(pool: PdHiddenPool) -> None:
    global _pd_hidden_pool
    _pd_hidden_pool = pool


def get_pd_hidden_pool() -> PdHiddenPool | None:
    """The process's hidden pool; None unless a PD spec worker built one."""
    return _pd_hidden_pool


def hidden_lanes(*, feature_size: int, total_kv_heads: int) -> int:
    """Lanes per head slice: total_kv_heads slices of _PACKING x lanes hold a row."""
    per_slice = -(-feature_size // (total_kv_heads * _PACKING))
    return -(-per_slice // _LANES) * _LANES


class PdHiddenPool:
    """Per-token hidden rows of in-flight PD requests, addressed by KV slot.

    Attributes:
      buffer: [num_pages, page_size, local_heads, 2, lanes] bf16; local_heads is
        this rank's share of total_kv_heads (all of them on a prefill stage).
      feature_size: Real width of a row; the tail lanes are padding.
    """

    def __init__(
        self,
        *,
        num_pages: int,
        page_size: int,
        feature_size: int,
        total_kv_heads: int,
        attn_tp_size: int,
        attn_tp_group: GroupCoordinator | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if dtype != torch.bfloat16:
            raise NotImplementedError(
                f"The PD hidden pool packs bf16 like the fused KV pool, got {dtype}."
            )
        self.feature_size = feature_size
        self.total_kv_heads = total_kv_heads
        self.attn_tp_size = attn_tp_size
        self._attn_tp_group = attn_tp_group
        self.local_heads = max(1, total_kv_heads // attn_tp_size)
        # Decode ranks beyond the head count replicate a slice (head_slice_params).
        self._replication = max(1, attn_tp_size // total_kv_heads)
        self.lanes = hidden_lanes(
            feature_size=feature_size, total_kv_heads=total_kv_heads
        )
        self.buffer = torch.zeros(
            (num_pages, page_size, self.local_heads, _PACKING, self.lanes),
            dtype=dtype,
            device=device,
        )
        self._row_width = total_kv_heads * _PACKING * self.lanes
        self._span_write: Callable[..., None] | None = None

    @staticmethod
    def bytes_per_token(
        *, feature_size: int, total_kv_heads: int, attn_tp_size: int, dtype: torch.dtype
    ) -> int:
        """HBM one token's row takes on a rank, as the KV budget counts it."""
        local_heads = max(1, total_kv_heads // attn_tp_size)
        lanes = hidden_lanes(feature_size=feature_size, total_kv_heads=total_kv_heads)
        return local_heads * _PACKING * lanes * torch._utils._element_size(dtype)

    def write_segments(
        self, cache_loc_cpu: torch.Tensor, num_rows: int
    ) -> torch.Tensor:
        """Host [3, num_rows] (source row, slot, length) runs of num_rows rows whose
        first len(cache_loc_cpu) are real; unused runs have length 0."""
        slots = np.zeros(num_rows, dtype=np.int64)
        valid = np.zeros(num_rows, dtype=bool)
        live = cache_loc_cpu.numel()
        slots[:live] = cache_loc_cpu.numpy()
        valid[:live] = True
        segments, _ = kv_write_segments(slots, valid)
        return segments

    def write(
        self,
        hidden: torch.Tensor,
        *,
        segments: torch.Tensor,
    ) -> torch.Tensor:
        """Write hidden [rows, feature_size] along the runs; returns a kv_fence_tail."""
        assert self.local_heads == self.total_kv_heads, "prefill stages hold every head"
        rows = torch.nn.functional.pad(
            hidden.to(self.buffer.dtype), (0, self._row_width - hidden.shape[1])
        ).view(hidden.shape[0], self.local_heads, _PACKING, self.lanes)
        if self.buffer.device.type == "tpu":
            # HBM-to-HBM DMAs: a hidden row is too wide for kv_row_write's VMEM staging.
            if self._span_write is None:
                self._span_write = get_kv_span_write_op()
            self._span_write(segments[1], segments[0], segments[2], rows, self.buffer)
        else:
            flat = self.buffer.flatten(0, 1)
            for src, dst, n in segments.T.tolist():
                flat[dst : dst + n] = rows[src : src + n]
        # Read after the write, so copying it waits for the rows to land.
        return self.buffer[0, 0, 0, 0, :1]

    def read(self, cache_loc: torch.Tensor) -> torch.Tensor:
        """[len(cache_loc), feature_size] rows of these slots, joined over attention TP."""
        local = self.buffer.flatten(0, 1).index_select(0, cache_loc)
        local = local.reshape(cache_loc.shape[0], -1)
        # Contiguous: compiled consumers take no views of a wider buffer.
        if self.attn_tp_size == 1:
            return local[:, : self.feature_size].contiguous()
        assert self._attn_tp_group is not None
        gathered = self._attn_tp_group.all_gather(local.unsqueeze(0), dim=0)
        # One rank per distinct head slice, in head order.
        distinct = gathered[:: self._replication]
        rows = distinct.permute(1, 0, 2).reshape(cache_loc.shape[0], -1)
        return rows[:, : self.feature_size].contiguous()
