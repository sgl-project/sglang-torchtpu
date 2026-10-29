# SPDX-License-Identifier: Apache-2.0
"""sglang pool classes over the Kimi-K3 programs' unified pools.

sglang builds a hybrid model's pools in two steps: ``HybridReqToTokenPool``
builds the Mamba pool (``get_mamba_pool_cls``) with the state slot count, then
``HybridLinearKVPool`` builds the full-attention pool (``get_mla_kv_pool_cls``)
with the KV token count. Under ``K3_PROGRAM`` both are stand-ins over one
allocation (:mod:`sglang_bridge`): :class:`K3StatePool` records the slot
count, and :class:`K3UnifiedKVPool`, built after it, allocates the unified
pools for both and hands them back. sglang never runs attention or KDA
kernels on them (the program does), so the classes carry only what the
allocators, memory accounting and P/D transfer read:

* KV: ``get_contiguous_buf_infos`` (:func:`sglang_bridge.kv_buf_infos`), the
  24 MLA pools then the DSpark draft's five;
* state: ``get_contiguous_buf_infos`` (:func:`sglang_bridge.state_buf_infos`),
  one entry per KDA layer.

Every rank's blocks hold its own share of each logical block (decode context
parallel), so the KV pool is not an ``MLATokenToKVPool``: sglang treats those
as replicated across ranks, and these are sent rank to rank.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, cache_layout
from sglang_tpu.k3_programs.sglang_config import program_kv_cache_dtype

# The state pool built for the KV pool that follows it (one hybrid pool per
# worker process).
_pending_state_pool: K3StatePool | None = None


def _block_bytes(kv_cache_dtype: str) -> int:
    layout = cache_layout(kv_cache_dtype)
    itemsize = 2 if layout.dtype == "bfloat16" else 1
    return int(torch.Size(layout.block_shape).numel()) * itemsize


def _kv_cache_dtype() -> str:
    from sglang.srt.runtime_context import get_model

    return program_kv_cache_dtype(get_model().kv_cache_dtype)


class K3StatePool:
    """The Mamba pool's stand-in: slot ``i``'s KDA state is its three blocks of
    the unified pools (``UnifiedPoolPlan.state_row``)."""

    slot_allocator_device = "cpu"

    def __init__(
        self, *, size: int, mamba_layer_ids, device: str, **kwargs: Any
    ) -> None:
        global _pending_state_pool
        if sorted(mamba_layer_ids) != sorted(KDA):
            raise ValueError(
                f"K3_PROGRAM: Mamba layers {list(mamba_layer_ids)[:4]}... are not "
                "Kimi-K3's KDA layers"
            )
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.plan: bridge.UnifiedPoolPlan | None = None
        self.caches: dict[str, torch.Tensor] | None = None
        _pending_state_pool = self

    @classmethod
    def cache_bytes_per_request(cls, cache_params) -> int:
        """A slot's share of the unified pools: three blocks of every pool."""
        pools = len(MLA) + bridge.DSPARK_DRAFT_LAYERS
        return bridge.KDA_PER_POOL * pools * _block_bytes(_kv_cache_dtype())

    def bind(self, plan: bridge.UnifiedPoolPlan, caches: dict[str, torch.Tensor]):
        self.plan, self.caches = plan, caches

    def mem_usage_bytes(self) -> int:
        return 0  # counted with the KV pool, which allocates the unified pools

    def get_contiguous_buf_infos(self) -> bridge.BufInfos:
        assert self.plan is not None and self.caches is not None
        return bridge.state_buf_infos(self.caches, self.plan)

    def get_state_layer_ids(self) -> list[int]:
        return list(KDA)

    def get_state_dim_per_tensor(self) -> list[int]:
        return []  # whole blocks, never sliced across ranks

    def get_state_slice_outer_counts(self) -> list[int]:
        return []

    def get_state_conv_shard_groups(self) -> list:
        return []


class K3UnifiedKVPool:
    """The full-attention pool's stand-in: allocates the unified pools, KV
    page ``p`` on block ``p``."""

    def __init__(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        **kwargs: Any,
    ) -> None:
        global _pending_state_pool
        state = _pending_state_pool
        if state is None:
            raise RuntimeError("K3_PROGRAM: the KV pool needs the state pool first")
        if layer_num != len(MLA):
            raise ValueError(f"K3_PROGRAM: {layer_num} MLA layers, not {len(MLA)}")
        layout = cache_layout(_kv_cache_dtype())
        step = bridge.TP_SIZE * layout.tokens_per_block
        if page_size != step:
            raise ValueError(
                f"K3_PROGRAM: --page-size must be one logical block ({step} "
                f"tokens), not {page_size}"
            )
        pages = -(-size // page_size)
        num_blocks = 1 + bridge.KDA_PER_POOL * (state.size + 1) + pages
        self.plan = bridge.UnifiedPoolPlan(num_blocks, state.size, layout)
        self.caches = bridge.allocate_pools(
            self.plan, device, draft_layers=bridge.DSPARK_DRAFT_LAYERS
        )
        state.bind(self.plan, self.caches)
        _pending_state_pool = None
        self.state_pool = state
        self.size = pages * page_size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.layer_num = layer_num
        self.device = device
        self.start_layer = 0
        self.end_layer = layer_num - 1
        self.mem_usage = self.get_kv_size_bytes() / (1 << 30)

    # Read through HybridLinearKVPool: no post-capture resizing, no HiCache.
    post_capture_active = False
    post_capture_backed_bytes = 0

    def host_pool_decls(self) -> list:
        return []

    def raw_pools(self) -> list[torch.Tensor]:
        """The distinct tensors: 24 unified pools, then the draft's."""
        seen: dict[int, torch.Tensor] = {}
        for tensor in self.caches.values():
            seen.setdefault(tensor.data_ptr(), tensor)
        return list(seen.values())

    def get_kv_size_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.raw_pools())

    def get_contiguous_buf_infos(self) -> bridge.BufInfos:
        return bridge.kv_buf_infos(self.caches, self.plan)

    def get_kv_layer_ids(self) -> list[int]:
        """The model layer of each KV entry: the MLA layers, then the draft
        layers in the band sglang numbers draft layers in (after the 93)."""
        from sglang_tpu.k3_programs.kimi_k3.layout import NUM_LAYERS

        draft = range(NUM_LAYERS, NUM_LAYERS + bridge.DSPARK_DRAFT_LAYERS)
        return list(MLA) + list(draft)
