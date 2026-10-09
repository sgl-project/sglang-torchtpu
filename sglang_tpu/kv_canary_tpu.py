# SPDX-License-Identifier: Apache-2.0
"""KV-canary attachers for the TPU fused KV pools.

The fused buffer (K and V in one buffer per layer) is treated like an MLA pool:
one canary pair per token slot, no separate V half.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patcher.adapters.swa import (
    register_subpool_group_builder,
)
from sglang.srt.kv_canary.pool_patcher.api import register_pool_attacher
from sglang.srt.kv_canary.pool_patcher.buf_info_splice import patch_buf_info_method
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import (
    alloc_canary_buf,
    make_row_source,
)


def is_fused_kv_pool(pool: object) -> bool:
    """A pool that stores K and V in one fused buffer per layer (no k_buffer)."""
    return hasattr(pool, "kv_buffer") and not hasattr(pool, "k_buffer")


def fused_slot_rows(pool: object) -> torch.Tensor:
    """Layer 0 of the fused buffer as ``[num_slots, bytes_per_slot]`` rows."""
    buf = pool.kv_buffer[0]
    num_slots = int(buf.shape[0]) * int(buf.shape[1])
    return buf.reshape(num_slots, -1)


def build_fused_group(
    *,
    pool: object,
    kind: PoolKind,
    device: torch.device,
    read_bytes: int,
    swa_lut: Optional[torch.Tensor],
    kv_token_id_vs_position_offset: int,
) -> CanaryBufferGroup:
    rows = fused_slot_rows(pool)
    num_slots = int(rows.shape[0])
    return CanaryBufferGroup(
        kind=kind,
        k_head=alloc_canary_buf(num_slots=num_slots, device=device),
        k_tail=alloc_canary_buf(num_slots=num_slots, device=device),
        v_head=None,
        v_tail=None,
        real_kv_sources_k=make_row_source(layer_buffer=rows, read_bytes=read_bytes),
        real_kv_sources_v=(),
        swa_index_lut=swa_lut,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )


def attach_tpu_fused(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    group = build_fused_group(
        pool=pool,
        kind=PoolKind.FULL,
        device=device,
        read_bytes=read_bytes,
        swa_lut=None,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    patch_buf_info_method(
        pool,
        method_name="get_contiguous_buf_infos",
        group=group,
        has_v_half=False,
        page_size=pool.page_size,
    )
    return (group,)


def register_tpu_attachers() -> None:
    """Register the TPU fused pool classes with the canary pool patcher."""
    from sglang_tpu.kv_pool import (
        TpuFusedMHATokenToKVPool,
        TpuSeqAlongLaneKVPool,
    )

    register_pool_attacher(TpuFusedMHATokenToKVPool, attach_tpu_fused)
    register_pool_attacher(TpuSeqAlongLaneKVPool, attach_tpu_fused)
    # Also as sub-pools of the hybrid SWA pool.
    register_subpool_group_builder(TpuFusedMHATokenToKVPool, build_fused_group)
    register_subpool_group_builder(TpuSeqAlongLaneKVPool, build_fused_group)
