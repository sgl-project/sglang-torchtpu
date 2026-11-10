# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = full.num_layers`` physical 5D Pallas KV buffers across Full-Attention
(1 physical page per logical page) and Sliding-Window Attention
(``M_swa = ceil(swa.num_layers / G)`` groups) with zero external fragmentation for
models where ``H_swa == H_full`` and ``D_swa == D_full``. Physical page 0 is
reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    _cdiv,
)

logger = logging.getLogger(__name__)


def num_swa_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of SWA groups ``M_swa = ceil(num_swa_layers / num_full_layers)``."""
    if num_full_layers <= 0 or num_swa_layers <= 0:
        raise ValueError(
            f"Unified KV pool requires positive full ({num_full_layers}) and "
            f"SWA ({num_swa_layers}) layer counts."
        )
    return _cdiv(num_swa_layers, num_full_layers)


@dataclass(frozen=True)
class AttnLayerSpec:
    """Per-attention-type geometry for either Full or SWA layers.

    Attributes:
        head_num: KV head count on this TP rank (post-TP partitioning).
        head_dim: Unpadded per-head K/V dimension (must satisfy K == V).
        layer_ids: Global model layer indices belonging to this attention type,
            in ascending model execution order.
    """

    head_num: int
    head_dim: int
    layer_ids: Sequence[int]

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)


@dataclass(frozen=True)
class UnifiedPageLayout:
    """Resolved physical page layout for a unified Full + SWA KV pool.

    Attributes:
        page_head_num: Physical KV heads per page (requires ``swa.head_num == full.head_num``).
        page_head_dim: Physical head dimension per page (requires ``swa.head_dim == full.head_dim``).
        num_shared_buffers: Number of shared 5D physical KV tensors ``G = full.num_layers``.
        num_swa_groups: Number of physical SWA page groups per logical SWA token
            ``M_swa = ceil(swa.num_layers / G)``.
    """

    page_head_num: int
    page_head_dim: int
    num_shared_buffers: int
    num_swa_groups: int


def resolve_unified_page_layout(
    full: AttnLayerSpec,
    swa: AttnLayerSpec,
) -> UnifiedPageLayout:
    """Validate matching Full/SWA KV geometry and compute SWA group counts."""
    if (
        full.head_num <= 0
        or full.head_dim <= 0
        or swa.head_num <= 0
        or swa.head_dim <= 0
    ):
        raise ValueError(
            f"Invalid head geometry: full=({full.head_num}, {full.head_dim}), "
            f"swa=({swa.head_num}, {swa.head_dim})."
        )
    if swa.head_num != full.head_num or swa.head_dim != full.head_dim:
        raise ValueError(
            "TpuUnifiedSWAKVPool requires matching Full and SWA KV head geometry "
            f"(H_swa == H_full and D_swa == D_full), got full=({full.head_num}, "
            f"{full.head_dim}), swa=({swa.head_num}, {swa.head_dim})."
        )
    return UnifiedPageLayout(
        page_head_num=full.head_num,
        page_head_dim=full.head_dim,
        num_shared_buffers=full.num_layers,
        num_swa_groups=num_swa_groups(full.num_layers, swa.num_layers),
    )


class TpuUnifiedSWAKVPool(BaseSWAKVPool):
    """Unified Hybrid Full/SWA KV pool backed by ``G = full.num_layers`` shared 5D buffers."""

    AttnSpec = AttnLayerSpec

    def __init__(
        self,
        size: int,
        *,
        page_size: int,
        dtype: torch.dtype,
        full: AttnLayerSpec,
        swa: AttnLayerSpec,
        device: str,
        token_to_kv_pool_class: type = TpuFusedMHATokenToKVPool,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
    ):
        if tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get():
            raise ValueError(
                "TpuUnifiedSWAKVPool does not support SGLANG_TPU_KV_PACK_HEADS."
            )
        if not (
            isinstance(token_to_kv_pool_class, type)
            and issubclass(token_to_kv_pool_class, TpuFusedMHATokenToKVPool)
        ):
            raise ValueError(
                "TpuUnifiedSWAKVPool requires a TpuFusedMHATokenToKVPool subclass, "
                f"got {token_to_kv_pool_class}."
            )
        self.full = full
        self.swa = swa
        self.layout = resolve_unified_page_layout(full, swa)
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=full.num_layers + swa.num_layers,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
        )
        self.head_num = self.layout.page_head_num
        self.head_dim = self.layout.page_head_dim

        G = self.layout.num_shared_buffers
        num_blocks = size // page_size + 1
        self.swa_group_pages = torch.zeros(
            (max(0, self.layout.num_swa_groups - 1), num_blocks), dtype=torch.int32
        )

        self.kv_pool: TpuFusedMHATokenToKVPool = token_to_kv_pool_class(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=self.layout.page_head_num,
            head_dim=self.layout.page_head_dim,
            layer_num=G,
            device=device,
            enable_memory_saver=enable_memory_saver,
            allocation_label="Unified",
        )
        self._swa_copy_kwargs = (
            {"allow_raiden_backup": False}
            if "allow_raiden_backup"
            in inspect.signature(self.kv_pool.get_cpu_copy).parameters
            else {}
        )

        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        self._swa_layer_group: Dict[int, int] = {}
        for i, gid in enumerate(full.layer_ids):
            self.layers_mapping[gid] = (i, False)
        for j, gid in enumerate(swa.layer_ids):
            self.layers_mapping[gid] = (j, True)
            self._swa_layer_group[gid] = j // G

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            "TpuUnifiedSWAKVPool mem usage: %.2f GB, unified size: %d tokens "
            "(shared buffers=%d, swa_groups=%d)",
            self.mem_usage,
            self.size,
            G,
            self.layout.num_swa_groups,
        )

    def _buffer_idx(self, layer_id: int) -> int:
        idx, is_swa = self.layers_mapping[layer_id]
        return idx % self.layout.num_shared_buffers if is_swa else idx

    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_fused_kv_buffer(self._buffer_idx(layer_id))

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_key_buffer(self._buffer_idx(layer_id))

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_value_buffer(self._buffer_idx(layer_id))

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_pool.get_kv_buffer(self._buffer_idx(layer_id))

    def set_kv_buffer(self, *args, **kwargs) -> None:
        self.kv_pool.set_kv_buffer(*args, **kwargs)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        mapping = self.full_to_swa_index_mapping
        return (
            mapping[kv_indices.to(mapping.device)].to(torch.int32).to(kv_indices.device)
        )

    def swa_page_group(self, layer_id: int) -> int:
        return self._swa_layer_group.get(layer_id, 0)

    def get_state(self) -> list[torch.Tensor]:
        return self.kv_pool.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        self.kv_pool.set_state(state)

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return self.kv_pool.get_kv_size_bytes()

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return [], [], []

    def _map_swa_group_indices(
        self, group: int, swa_indices_cpu: torch.Tensor
    ) -> torch.Tensor:
        if group == 0:
            return swa_indices_cpu
        phys_pages = self.swa_group_pages[
            group - 1, swa_indices_cpu // self.page_size
        ].to(torch.long)
        return phys_pages * self.page_size + (swa_indices_cpu % self.page_size)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        full_kv_cpu = self.kv_pool.get_cpu_copy(
            indices, mamba_indices=mamba_indices, req_pool_index=req_pool_index
        )
        swa_kv_cpu = None
        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            mapping = self.full_to_swa_index_mapping
            swa_indices = mapping[indices.to(mapping.device)]
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                valid_swa = swa_indices[swa_mask].to(device="cpu", dtype=torch.long)
                swa_kv_cpu = [
                    self.kv_pool.get_cpu_copy(
                        self._map_swa_group_indices(g, valid_swa),
                        **self._swa_copy_kwargs,
                    )
                    for g in range(self.layout.num_swa_groups)
                ]
                swa_mask = swa_mask.cpu()
        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        self.kv_pool.load_cpu_copy(
            kv_cache_cpu["full"],
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )
        swa_kv_cpu = kv_cache_cpu["swa"]
        if swa_kv_cpu is None or self.full_to_swa_index_mapping is None:
            return
        mapping = self.full_to_swa_index_mapping
        swa_indices = mapping[indices.to(mapping.device)]
        new_swa_mask = swa_indices > 0
        old_swa_mask = kv_cache_cpu.get("swa_mask")
        if old_swa_mask is not None:
            old_swa_mask = old_swa_mask.to(indices.device)
            row_mask = new_swa_mask[old_swa_mask].cpu()
            swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
        else:
            row_mask = new_swa_mask.cpu()
            swa_indices = swa_indices[new_swa_mask]
        if swa_indices.numel() == 0:
            return
        valid_swa = swa_indices.to(device="cpu", dtype=torch.long)
        need_filter = not bool(torch.all(row_mask).item())
        for g, grp_cpu in enumerate(swa_kv_cpu):
            if need_filter:
                grp_cpu = self.kv_pool.filter_cpu_copy(grp_cpu, row_mask)
            self.kv_pool.load_cpu_copy(
                grp_cpu, self._map_swa_group_indices(g, valid_swa)
            )
