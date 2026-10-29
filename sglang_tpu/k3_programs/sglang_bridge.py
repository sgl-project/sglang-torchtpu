# SPDX-License-Identifier: Apache-2.0
"""sglang's side of the vendored Kimi-K3 programs' unified pools.

The programs (``kimi_k3.decode`` / ``kimi_k3.prefill``) read vLLM's unified
pools: 24 raw tensors of ``layout.cache_layout()`` blocks per rank, one per
MLA layer, each also holding three KDA layers' state blocks. Pages are
decode-context parallel over the 32 ranks: page ``k`` of a sequence lives on
rank ``k % 32`` in logical block ``k // 32``, so one block-table entry covers
``32 * tokens_per_block`` tokens (8192 in bf16).

sglang keeps KV pages and recurrent state in two allocators. This module
places both in the same tensors, in disjoint block ranges:

* blocks ``1 .. 3S`` are KDA state blocks: three KDA layers share each pool, so
  a request holds one state block per KDA group (group ``j``: the ``j``-th KDA
  layer of every pool), and sglang's Mamba slot ``i`` owns blocks
  ``3(i - 1) + 1 .. 3i``;
* blocks ``3S + 1 .. N - 1`` are KV logical blocks: sglang's KV allocator runs
  with ``page_size`` = one logical block, and its page ``p`` (``p >= 1``; page
  0 is the padding page) is block ``3S + p``;
* block 0 is the null block the programs read for padding rows.

The programs see one MLA group (the block-table row of the request's KV
pages) and three KDA groups whose ``block_size`` spans the whole context, so
the block each KDA layer imports a request's state from
(``pools.kda_blocks_at_length``) is column 0 of its group: one of the slot's
three blocks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS, cache_layout
from sglang_tpu.k3_programs.layout import KVLayout
from sglang_tpu.k3_programs.program import KVGroup

TP_SIZE = 32
# Kernel KDA layer ``k`` lives in pool ``k // KDA_PER_POOL`` (the last pool none),
# and in KDA group ``k % KDA_PER_POOL``: a group's layers are in distinct pools.
KDA_PER_POOL = 3


def mla_layer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def kda_layer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.kda"


@dataclasses.dataclass(frozen=True)
class UnifiedPoolPlan:
    """Block ranges of the unified pools on every rank."""

    num_blocks: int  # per pool, including the null block
    state_slots: int  # S: sglang Mamba slots, blocks 1 .. 3S
    layout: KVLayout

    def __post_init__(self) -> None:
        if self.state_slots < 1 or self.kv_blocks < 1:
            raise ValueError(
                f"{self.num_blocks} blocks cannot hold {self.state_slots} state "
                "slots, the null block and at least one KV logical block"
            )

    @property
    def state_blocks(self) -> int:
        """Blocks the state slots occupy: one per KDA group per slot."""
        return KDA_PER_POOL * self.state_slots

    @property
    def kv_blocks(self) -> int:
        """KV logical blocks: sglang's KV pages 1 .. kv_blocks."""
        return self.num_blocks - 1 - self.state_blocks

    @property
    def logical_block_tokens(self) -> int:
        """Tokens one block-table entry covers: a page on each of the 32 ranks."""
        return TP_SIZE * self.layout.tokens_per_block

    def kv_block(self, page: int) -> int:
        """The pool block of sglang KV page ``page`` (0: the null block)."""
        if not 0 <= page <= self.kv_blocks:
            raise ValueError(f"KV page {page} is outside 0 .. {self.kv_blocks}")
        return 0 if page == 0 else self.state_blocks + page

    def state_block(self, slot: int, group: int) -> int:
        """KDA group ``group``'s pool block for sglang Mamba slot ``slot``
        (0: the null block)."""
        if not 0 <= slot <= self.state_slots:
            raise ValueError(f"KDA slot {slot} is outside 0 .. {self.state_slots}")
        if not 0 <= group < KDA_PER_POOL:
            raise ValueError(f"KDA group {group} is outside 0 .. {KDA_PER_POOL - 1}")
        return 0 if slot == 0 else KDA_PER_POOL * (slot - 1) + group + 1


def plan_pools(
    pool_bytes: int, state_slots: int, kv_cache_dtype: str = "auto"
) -> UnifiedPoolPlan:
    """The plan for ``pool_bytes`` of HBM per rank over all 24 pools."""
    layout = cache_layout(kv_cache_dtype)
    itemsize = 2 if layout.dtype == "bfloat16" else 1  # float8_e4m3fn
    block_bytes = int(np.prod(layout.block_shape)) * itemsize
    num_blocks = pool_bytes // (block_bytes * len(MLA))
    return UnifiedPoolPlan(int(num_blocks), state_slots, layout)


def allocate_pools(plan: UnifiedPoolPlan, device) -> dict[str, torch.Tensor]:
    """The 24 raw pools, keyed by the layer names the programs bind: each MLA
    layer's pool, shared by the three KDA layers ``3p .. 3p + 2`` it hosts."""
    dtype = getattr(torch, plan.layout.dtype)
    pools = [
        torch.zeros(
            (plan.num_blocks, *plan.layout.block_shape), dtype=dtype, device=device
        )
        for _ in MLA
    ]
    caches = {mla_layer_name(layer): pools[p] for p, layer in enumerate(MLA)}
    for k, layer in enumerate(KDA):
        caches[kda_layer_name(layer)] = pools[k // KDA_PER_POOL]
    return caches


def mla_table_row(
    req_to_token_row: np.ndarray, seq_len: int, plan: UnifiedPoolPlan, width: int
) -> np.ndarray:
    """A request's MLA block-table row: the pool block of each logical block
    its first ``seq_len`` tokens touch, 0-padded to ``width``."""
    step = plan.logical_block_tokens
    used = -(-seq_len // step)
    if used > width:
        raise ValueError(
            f"{seq_len} tokens need {used} blocks; the table holds {width}"
        )
    pages = np.asarray(req_to_token_row[: used * step : step], dtype=np.int64) // step
    if len(pages) != used:
        raise ValueError(
            f"a req_to_token row of {len(req_to_token_row)} slots does not cover "
            f"{seq_len} tokens"
        )
    row = np.zeros(width, np.int32)
    row[:used] = [plan.kv_block(int(p)) for p in pages]
    return row


def kda_table_rows(slot: int, plan: UnifiedPoolPlan) -> tuple[np.ndarray, ...]:
    """A request's rows in the three KDA groups: its slot's block for that
    group, in column 0."""
    return tuple(
        np.array([plan.state_block(slot, group)], np.int32)
        for group in range(KDA_PER_POOL)
    )


def kda_group_layers(group: int) -> tuple[int, ...]:
    """Model layers of KDA group ``group``: the ``group``-th KDA layer of each pool."""
    return tuple(layer for k, layer in enumerate(KDA) if k % KDA_PER_POOL == group)


def kv_groups(
    plan: UnifiedPoolPlan, context_len: int, table_width: int
) -> list[KVGroup]:
    """The groups the programs bind: the MLA layers' pages, then the three KDA
    groups, each with one block spanning the context (column 0: the request's
    state block for that group)."""
    return [
        KVGroup(
            tuple(mla_layer_name(layer) for layer in MLA),
            plan.logical_block_tokens,
            table_width,
        ),
        *(
            KVGroup(
                tuple(kda_layer_name(layer) for layer in kda_group_layers(group)),
                max(context_len, 1),
                1,
            )
            for group in range(KDA_PER_POOL)
        ),
    ]


def block_tables(
    req_to_token_row: np.ndarray,
    seq_len: int,
    slot: int,
    plan: UnifiedPoolPlan,
    table_width: int,
) -> tuple[np.ndarray, ...]:
    """``RequestInput.block_tables`` for the four groups of :func:`kv_groups`."""
    return (
        mla_table_row(req_to_token_row, seq_len, plan, table_width),
        *kda_table_rows(slot, plan),
    )


def layers_covered(groups: Sequence[KVGroup]) -> set[int]:
    """Model layer indices the groups name (every one of the 93)."""
    from sglang_tpu.k3_programs.kimi_k3.pools import layer_index

    return {layer_index(name) for group in groups for name in group.layer_names}


assert len(MLA) + len(KDA) == NUM_LAYERS
