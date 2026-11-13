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

import math
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


def _prefix_tails() -> int:
    from sglang_tpu.envs import envs as tpu_envs

    return int(tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS.get() or 0)


def _colocated_dp_share() -> float:
    """B rows per KV page under colocated attention DP2 (0 otherwise)."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.k3_programs.sglang_program import COLOCATED, server_role

    if (
        tpu_envs.SGLANG_TPU_K3_PROGRAM_ATTENTION_DP.get() != 2
        or server_role() != COLOCATED
    ):
        return 0.0
    return float(tpu_envs.SGLANG_TPU_K3_PROGRAM_DP_ROW_SHARE.get())


def is_k31(hf_config) -> bool:
    """Whether ``hf_config`` is Kimi-K3.1's (a DSA indexer on its MLA layers)."""
    text = getattr(hf_config, "text_config", hf_config)
    return getattr(text, "dsa_config", None) is not None


def _copy_rows(tensors, src: torch.Tensor, dst: torch.Tensor) -> None:
    for pool in tensors:
        device_src, device_dst = src.to(pool.device), dst.to(pool.device)
        pool.index_copy_(0, device_dst, pool.index_select(0, device_src))


# A bound program's block copy, ``(pairs) -> None`` over every unified and
# draft pool (``sglang_program.block_copier``). Eager index copies into the
# pools would stay pending until the program's next op read them, which then
# compiles a new executable for each pending copy; the program's copy op is
# compiled once. Unset (no program, CPU tests): eager index copies.
_block_copier: Any = None


def set_block_copier(copier: Any) -> None:
    global _block_copier
    _block_copier = copier


def _copy_blocks(tensors, src: torch.Tensor, dst: torch.Tensor) -> None:
    if _block_copier is None:
        _copy_rows(tensors, src, dst)
        return
    _block_copier([(int(a), int(b)) for a, b in zip(src.tolist(), dst.tolist())])


def _kv_cache_dtype() -> str:
    from sglang.srt.runtime_context import get_model

    return program_kv_cache_dtype(get_model().kv_cache_dtype)


class K3StatePool:
    """The Mamba pool's stand-in: slot ``i``'s KDA state is its three blocks of
    the unified pools (``UnifiedPoolPlan.state_row``)."""

    # Kimi-K3.1: the index-key pools have rows for the state blocks too
    # (set by hooks.k3_program before sglang sizes the state pool).
    dsa = False

    slot_allocator_device = "cpu"
    # HybridReqToTokenPool resets GDN ReplaySSM rings on fresh slots; KDA has none.
    replayssm_write_pos = None

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
        """A slot's share of the unified pools: three blocks of every pool,
        and its share of the prefix tails' blocks (sglang sizes the state by
        slot count, and the tails sit beside the slots)."""
        from sglang.srt.runtime_context import get_schedule

        dtype = _kv_cache_dtype()
        block = (len(MLA) + bridge.DSPARK_DRAFT_LAYERS) * _block_bytes(dtype)
        if cls.dsa:  # Kimi-K3.1: the slot's rows of the 24 index-key pools
            block += len(MLA) * bridge.index_block_bytes(cache_layout(dtype))
        slot_bytes = bridge.KDA_PER_POOL * block
        tails = _prefix_tails()
        if not tails:
            return slot_bytes
        slots = int(get_schedule().max_mamba_cache_size or 1) + 1
        tail_bytes = (1 + bridge.KDA_PER_POOL) * block * tails
        return slot_bytes + -(-tail_bytes // slots)

    def bind(self, plan: bridge.UnifiedPoolPlan, caches: dict[str, torch.Tensor]):
        self.plan, self.caches = plan, caches

    def _state_rows(self, slots) -> torch.Tensor:
        assert self.plan is not None
        slots = torch.as_tensor(slots, dtype=torch.int64).reshape(-1).cpu()
        first = torch.tensor([self.plan.state_row(int(s)) for s in slots])
        rows = first[:, None] + torch.arange(bridge.KDA_PER_POOL)[None, :]
        return rows.reshape(-1)

    def host_pools(self) -> list[torch.Tensor]:
        """The pools holding KDA state (each hosts three KDA layers)."""
        assert self.caches is not None
        seen: dict[int, torch.Tensor] = {}
        for layer in KDA:
            pool = self.caches[bridge.kda_layer_name(layer)]
            seen.setdefault(pool.data_ptr(), pool)
        return list(seen.values())

    def copy_from(self, src_indices, dst_indices) -> None:
        """Copy slots' KDA state (their three blocks of every hosting pool):
        a prefix hit's cached state into the request's slot, or a chunk's
        state into its track slot for the radix cache."""
        src, dst = self._state_rows(src_indices), self._state_rows(dst_indices)
        if src.numel() != dst.numel():
            raise ValueError(f"copy of {src.numel()} state rows into {dst.numel()}")
        if not src.numel():
            return
        _copy_blocks(self.host_pools(), src, dst)

    def copy_blocks(self, src_blocks, dst_blocks) -> None:
        """Copy state blocks (a tail's slot) in every hosting pool."""
        src = torch.as_tensor(src_blocks, dtype=torch.int64).reshape(-1).cpu()
        dst = torch.as_tensor(dst_blocks, dtype=torch.int64).reshape(-1).cpu()
        if src.numel():
            _copy_blocks(self.host_pools(), src, dst)

    def clear_slots(self, indices) -> None:
        """Nothing to clear: a request without a prefix hit starts from a zero
        state in the program, which never reads its slot first."""

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
        tails = _prefix_tails()
        dp_rows = math.ceil(pages * _colocated_dp_share())
        num_blocks = (
            1
            + bridge.KDA_PER_POOL * (state.size + 1)
            + pages
            + (1 + bridge.KDA_PER_POOL) * tails
            + dp_rows
        )
        self.plan = bridge.UnifiedPoolPlan(
            num_blocks, state.size, layout, tail_slots=tails, dp_rows=dp_rows
        )
        self.caches = bridge.allocate_pools(
            self.plan, device, draft_layers=bridge.DSPARK_DRAFT_LAYERS
        )
        # Kimi-K3.1: the DSA index keys, one pool per MLA layer beside the
        # latent rows (sglang builds this pool as its DSA pool: index_head_dim).
        self.index_pools = (
            bridge.allocate_index_pools(self.plan, device)
            if kwargs.get("index_head_dim") is not None
            else []
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

    def maybe_get_custom_mem_pool(self) -> None:
        # The P/D transfer registers the pools' own buffers.
        return None

    @classmethod
    def bytes_per_token(cls, kv_cache_dtype: str, dsa: bool = False) -> int:
        """A logical token's share of the unified pools on one rank: every
        pool's block holds this rank's share of a logical block's tokens
        (``dsa``: and Kimi-K3.1's index-key pools)."""
        layout = cache_layout(program_kv_cache_dtype(kv_cache_dtype))
        pools = len(MLA) + bridge.DSPARK_DRAFT_LAYERS
        tokens = bridge.TP_SIZE * layout.tokens_per_block
        block = pools * _block_bytes(program_kv_cache_dtype(kv_cache_dtype))
        if dsa:
            block += len(MLA) * bridge.index_block_bytes(layout)
        # Colocated DP2's B rows ride on the KV pages' budget.
        return int(block * (1 + _colocated_dp_share())) // tokens

    def raw_pools(self) -> list[torch.Tensor]:
        """The distinct tensors: 24 unified pools, then the draft's, then
        (Kimi-K3.1) the DSA index-key pools."""
        seen: dict[int, torch.Tensor] = {}
        for tensor in self.caches.values():
            seen.setdefault(tensor.data_ptr(), tensor)
        return list(seen.values()) + list(getattr(self, "index_pools", ()))

    def copy_blocks(self, src_blocks, dst_blocks) -> None:
        """Copy whole blocks of every pool (the 24 unified pools and the
        draft's): a partial logical block into or out of a prefix tail."""
        src = torch.as_tensor(src_blocks, dtype=torch.int64).reshape(-1).cpu()
        dst = torch.as_tensor(dst_blocks, dtype=torch.int64).reshape(-1).cpu()
        if src.numel():
            _copy_blocks(self.raw_pools(), src, dst)

    def get_kv_size_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.raw_pools())

    layer_transfer_counter = None

    def register_layer_transfer_counter(self, counter) -> None:
        """The HiCache load counter (sglang_hicache): the program's worker
        waits on HybridLinearKVPool's, which hands this pool None."""
        self.layer_transfer_counter = counter

    def get_contiguous_buf_infos(self) -> bridge.BufInfos:
        return bridge.kv_buf_infos(self.caches, self.plan, self.index_pools)

    def get_kv_layer_ids(self) -> list[int]:
        """The model layer of each KV entry: the MLA layers, then the draft
        layers in the band sglang numbers draft layers in (after the 93)."""
        from sglang_tpu.k3_programs.kimi_k3.layout import NUM_LAYERS

        draft = range(NUM_LAYERS, NUM_LAYERS + bridge.DSPARK_DRAFT_LAYERS)
        # Kimi-K3.1's index-key pools after the draft band (get_contiguous_buf_infos' order).
        index = range(draft.stop, draft.stop + len(self.index_pools))
        return list(MLA) + list(draft) + list(index)
