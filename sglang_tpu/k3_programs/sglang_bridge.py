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

* blocks ``0 .. K`` are KV logical blocks: sglang's KV allocator runs with
  ``page_size`` = one logical block, and its page ``p`` is block ``p``. sglang
  never allocates page 0, its padding page: block 0 is the null block the
  programs read for padding rows;
* blocks ``K + 1 .. N - 1`` are KDA state blocks: three KDA layers share each
  pool, so a request holds one state block per KDA group (group ``g``: the
  ``g``-th KDA layer of every pool), and sglang's Mamba slot ``i`` (slot 0
  reserved) owns the three consecutive blocks ``K + 1 + 3i + g``.

The TPU P/D transfer (Raiden) moves whole rows of the pool tensors by index:
a page's row is its page id, a slot's are three rows from
:meth:`UnifiedPoolPlan.state_row`. Pointer-based transfers address the same
blocks as strides from a base inside each tensor (:func:`kv_buf_infos`,
:func:`state_buf_infos`), identically on the prefill and the decode side.

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
from sglang_tpu.k3_programs.kimi_k3.pools import DRAFT_LAYER0
from sglang_tpu.k3_programs.layout import KVLayout
from sglang_tpu.k3_programs.program import KVGroup

TP_SIZE = 32
# Kernel KDA layer ``k`` lives in pool ``k // KDA_PER_POOL`` (the last pool none),
# and in KDA group ``k % KDA_PER_POOL``: a group's layers are in distinct pools.
KDA_PER_POOL = 3
# Kimi-K3 DSpark's MLA draft layers (``num_hidden_layers`` of its checkpoint).
DSPARK_DRAFT_LAYERS = 5


def mla_layer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def kda_layer_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.kda"


def draft_layer_name(index: int) -> str:
    """The ``index``-th DSpark draft layer, numbered after the 93 target layers
    as ``pools.draft_raw_pools`` reads them."""
    return f"model.layers.{DRAFT_LAYER0 + index}.self_attn.attn"


@dataclasses.dataclass(frozen=True)
class UnifiedPoolPlan:
    """Block ranges of the unified pools on every rank."""

    num_blocks: int  # per pool, including the null block
    state_slots: int  # S: sglang Mamba slots
    layout: KVLayout
    # Prefix tails (sglang_tails): a KV block and a KDA state slot each, after
    # the state blocks, outside sglang's allocators.
    tail_slots: int = 0
    # Colocated attention DP2 (sglang_dp.ColoRows): rows after the tails,
    # outside sglang's allocators, each a B row on either DP group's ranks.
    dp_rows: int = 0

    def __post_init__(self) -> None:
        if self.state_slots < 1 or self.kv_blocks < 1:
            raise ValueError(
                f"{self.num_blocks} blocks cannot hold {self.state_slots} state "
                "slots, the reserved slot 0 and at least one KV logical block"
            )

    @property
    def state_blocks(self) -> int:
        """Blocks the state slots occupy, slot 0 included: one per KDA group
        per slot."""
        return KDA_PER_POOL * (self.state_slots + 1)

    @property
    def kv_blocks(self) -> int:
        """KV logical blocks: sglang's KV pages 1 .. kv_blocks."""
        return self.num_blocks - 1 - self.state_blocks - self.tail_blocks - self.dp_rows

    @property
    def tail_blocks(self) -> int:
        """Blocks the prefix tails occupy: a KV block and three state blocks each."""
        return (1 + KDA_PER_POOL) * self.tail_slots

    def tail_kv_block(self, tail: int) -> int:
        """Prefix tail ``tail``'s KV block (its partial logical block)."""
        if not 0 <= tail < self.tail_slots:
            raise ValueError(
                f"prefix tail {tail} is outside 0 .. {self.tail_slots - 1}"
            )
        return self.state_base + self.state_blocks + tail

    def tail_state_block(self, tail: int, group: int) -> int:
        """Prefix tail ``tail``'s KDA state block for group ``group``."""
        if not 0 <= tail < self.tail_slots:
            raise ValueError(
                f"prefix tail {tail} is outside 0 .. {self.tail_slots - 1}"
            )
        return (
            self.state_base
            + self.state_blocks
            + self.tail_slots
            + KDA_PER_POOL * tail
            + group
        )

    @property
    def dp_row_base(self) -> int:
        """The first colocated-DP2 row (after the tails)."""
        return self.state_base + self.state_blocks + self.tail_blocks

    @property
    def state_base(self) -> int:
        """The first state block (slot 0's), after the KV blocks."""
        return self.kv_blocks + 1

    def state_row(self, slot: int) -> int:
        """The first of Mamba slot ``slot``'s three consecutive state blocks."""
        if not 0 <= slot <= self.state_slots:
            raise ValueError(f"KDA slot {slot} is outside 0 .. {self.state_slots}")
        return self.state_base + KDA_PER_POOL * slot

    @property
    def logical_block_tokens(self) -> int:
        """Tokens one block-table entry covers: a page on each of the 32 ranks."""
        return TP_SIZE * self.layout.tokens_per_block

    def kv_block(self, page: int) -> int:
        """The pool block of sglang KV page ``page`` (0: the null block)."""
        if not 0 <= page <= self.kv_blocks:
            raise ValueError(f"KV page {page} is outside 0 .. {self.kv_blocks}")
        return page

    def state_block(self, slot: int, group: int) -> int:
        """KDA group ``group``'s pool block for sglang Mamba slot ``slot``
        (0: the null block)."""
        if not 0 <= slot <= self.state_slots:
            raise ValueError(f"KDA slot {slot} is outside 0 .. {self.state_slots}")
        if not 0 <= group < KDA_PER_POOL:
            raise ValueError(f"KDA group {group} is outside 0 .. {KDA_PER_POOL - 1}")
        return 0 if slot == 0 else self.state_row(slot) + group


def plan_pools(
    pool_bytes: int,
    state_slots: int,
    kv_cache_dtype: str = "auto",
    draft_layers: int = 0,
) -> UnifiedPoolPlan:
    """The plan for ``pool_bytes`` of HBM per rank over all 24 pools and the
    ``draft_layers`` draft pools."""
    layout = cache_layout(kv_cache_dtype)
    itemsize = 2 if layout.dtype == "bfloat16" else 1  # float8_e4m3fn
    block_bytes = int(np.prod(layout.block_shape)) * itemsize
    num_blocks = pool_bytes // (block_bytes * (len(MLA) + draft_layers))
    return UnifiedPoolPlan(int(num_blocks), state_slots, layout)


def allocate_pools(
    plan: UnifiedPoolPlan, device, draft_layers: int = 0
) -> dict[str, torch.Tensor]:
    """The 24 raw pools, keyed by the layer names the programs bind: each MLA
    layer's pool, shared by the three KDA layers ``3p .. 3p + 2`` it hosts.
    ``draft_layers``: also a pool per DSpark draft layer, shaped like the
    target's, which a P/D transfer moves with the target's KV blocks."""
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
    for i in range(draft_layers):
        caches[draft_layer_name(i)] = torch.zeros_like(pools[0])
    return caches


def allocate_index_pools(plan: UnifiedPoolPlan, device) -> list[torch.Tensor]:
    """Kimi-K3.1: the 24 MLA layers' DSA index-key pools (uint8), rows aligned
    with the unified pools' blocks (``layout.index_block_shape``)."""
    from sglang_tpu.k3_programs.kimi_k3.layout import index_block_shape

    shape = index_block_shape(plan.layout)
    return [
        torch.zeros((plan.num_blocks, *shape), dtype=torch.uint8, device=device)
        for _ in MLA
    ]


def index_block_bytes(layout: KVLayout) -> int:
    from sglang_tpu.k3_programs.kimi_k3.layout import index_block_shape

    return int(np.prod(index_block_shape(layout)))


def _block_bytes(pool: torch.Tensor) -> int:
    return pool[0].numel() * pool.element_size()


BufInfos = tuple[list[int], list[int], list[int]]  # (data ptrs, lengths, item lengths)


def kv_buf_infos(
    caches: dict[str, torch.Tensor], plan: UnifiedPoolPlan, index_pools=()
) -> BufInfos:
    """The KV pages as sglang's P/D transfer addresses them: per MLA pool, then
    per draft pool (then, Kimi-K3.1, per DSA index-key pool), page ``p`` at
    ``ptr + p * item_len`` (block ``p``)."""
    from sglang_tpu.k3_programs.kimi_k3.pools import draft_raw_pools

    tensors = [caches[mla_layer_name(layer)] for layer in MLA]
    tensors += draft_raw_pools(caches)
    tensors += list(index_pools)
    ptrs, lens, items = [], [], []
    for pool in tensors:
        block = _block_bytes(pool)
        ptrs.append(pool.data_ptr())
        lens.append((plan.kv_blocks + 1) * block)
        items.append(block)
    return ptrs, lens, items


def state_buf_infos(caches: dict[str, torch.Tensor], plan: UnifiedPoolPlan) -> BufInfos:
    """The KDA state as sglang's P/D transfer addresses it: per kernel KDA
    layer (``KDA`` order), Mamba slot ``i`` at ``ptr + i * item_len`` (its
    group's block ``state_row(i) + g`` of the layer's pool)."""
    ptrs, lens, items = [], [], []
    for k, layer in enumerate(KDA):
        pool = caches[kda_layer_name(layer)]
        block, group = _block_bytes(pool), k % KDA_PER_POOL
        ptrs.append(pool.data_ptr() + (plan.state_base + group) * block)
        lens.append((plan.state_blocks - group) * block)
        items.append(KDA_PER_POOL * block)
    return ptrs, lens, items


def page_starts(req_to_token, plan: UnifiedPoolPlan):
    """The ``req_to_token`` slot of every logical block's first token
    (``[rows, blocks]``, a strided view): all a block-table row reads."""
    return req_to_token[:, :: plan.logical_block_tokens]


def mla_table_row(
    starts: np.ndarray, seq_len: int, plan: UnifiedPoolPlan, width: int
) -> np.ndarray:
    """A request's MLA block-table row from its :func:`page_starts` row: the
    pool block of each logical block its first ``seq_len`` tokens touch,
    0-padded to ``width``."""
    step = plan.logical_block_tokens
    used = -(-seq_len // step)
    if used > width:
        raise ValueError(
            f"{seq_len} tokens need {used} blocks; the table holds {width}"
        )
    pages = np.asarray(starts[:used], dtype=np.int64) // step
    if len(pages) != used:
        raise ValueError(
            f"a req_to_token row of {len(starts)} blocks does not cover "
            f"{seq_len} tokens"
        )
    if not pages.all():
        # sglang never allocates page 0 (its padding page): these tokens have
        # no KV slots, and the program would write them into the null block.
        raise ValueError(
            f"logical blocks {np.flatnonzero(pages == 0).tolist()} of the "
            f"{seq_len} tokens are not allocated"
        )
    if used and int(pages.max()) > plan.kv_blocks:
        raise ValueError(f"KV pages {pages.tolist()} exceed {plan.kv_blocks}")
    row = np.zeros(width, np.int32)
    row[:used] = pages  # KV page p is block p
    return row


def kda_table_rows(
    slot: int, plan: UnifiedPoolPlan, width: int = 1
) -> tuple[np.ndarray, ...]:
    """A request's rows in the three KDA groups: its slot's block for that
    group in every column, so the state a step reads or writes at any length
    (a prefix hit's column included) is the slot's."""
    return tuple(
        np.full(width, plan.state_block(slot, group), np.int32)
        for group in range(KDA_PER_POOL)
    )


def kda_group_layers(group: int) -> tuple[int, ...]:
    """Model layers of KDA group ``group``: the ``group``-th KDA layer of each pool."""
    return tuple(layer for k, layer in enumerate(KDA) if k % KDA_PER_POOL == group)


def kv_groups(
    plan: UnifiedPoolPlan,
    context_len: int,
    table_width: int,
    draft_layers: int = 0,
    kda_block_size: int | None = None,
    mla_block_tokens: int | None = None,
) -> list[KVGroup]:
    """The groups the programs bind: the MLA layers' pages (with the draft
    layers', read through the same tables), then the three KDA groups. A KDA
    group's block spans the whole context (one column: the request's state
    block), or ``kda_block_size`` tokens: the prefill program reads a prefix
    hit's state from column ``hit // block_size - 1`` (vLLM's ``align``
    mode), so a prefill with prefix caching takes a column per page.
    ``mla_block_tokens``: an MLA table entry's tokens (attention DP2's 8192:
    a DCP16 block, sglang_dp) instead of a logical block."""
    kda_size = kda_block_size or max(context_len, 1)
    kda_width = -(-max(context_len, 1) // kda_size)
    return [
        KVGroup(
            tuple(mla_layer_name(layer) for layer in MLA)
            + tuple(draft_layer_name(i) for i in range(draft_layers)),
            mla_block_tokens or plan.logical_block_tokens,
            table_width,
        ),
        *(
            KVGroup(
                tuple(kda_layer_name(layer) for layer in kda_group_layers(group)),
                kda_size,
                kda_width,
            )
            for group in range(KDA_PER_POOL)
        ),
    ]


def block_tables(
    starts: np.ndarray,
    seq_len: int,
    slot: int,
    plan: UnifiedPoolPlan,
    table_width: int,
    kda_width: int = 1,
) -> tuple[np.ndarray, ...]:
    """``RequestInput.block_tables`` for the four groups of :func:`kv_groups`."""
    return (
        mla_table_row(starts, seq_len, plan, table_width),
        *kda_table_rows(slot, plan, kda_width),
    )


def layers_covered(groups: Sequence[KVGroup]) -> set[int]:
    """Model layer indices the groups name (every one of the 93, then the draft's)."""
    from sglang_tpu.k3_programs.kimi_k3.pools import layer_index

    return {layer_index(name) for group in groups for name in group.layer_names}


assert len(MLA) + len(KDA) == NUM_LAYERS
