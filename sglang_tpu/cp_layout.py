# SPDX-License-Identifier: Apache-2.0
"""Host-side token layout for prefill context parallelism on TPU.

Every request's extend tokens are cut into ``2 * cp_size`` zigzag blocks
(the first ``length % (2 * cp_size)`` blocks are one token longer, so a request
shorter than ``2 * cp_size`` tokens leaves its last blocks empty); rank ``r``
owns blocks ``r`` and ``2 * cp_size - 1 - r``. The local token order on a
rank is request-major with the two owned blocks adjacent:

    [req0 block r, req0 block 2cp-1-r, req1 block r, req1 block 2cp-1-r, ...]

followed by padding rows up to the local token bucket. Each non-empty owned
block is one attention row of the ragged-paged-attention kernel (``kv_len`` =
absolute end of the block, ``q_len`` = block length, the request's page table),
so a request contributes up to two rows per rank; the remaining rows are zero
rows at the tail.

The layout is deterministic from CPU lengths, so every rank computes the same
per-rank row counts and the same token bucket without communication. Only
numpy is used here; the torch glue lives in ``cp_forward``.

Ring mode (``SGLANG_TPU_PCP_MODE=ring``) makes the blocks page-aligned and
rank-uniform."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from sglang.srt.utils.common import cdiv, round_up


class TpuCpHostLayout(NamedTuple):
    """Zigzag layout of one extend batch as seen from ``cp_rank``."""

    cp_size: int
    cp_rank: int
    # Real request count; rows for requests >= num_reqs are padding.
    num_reqs: int
    # Request bucket (rows of every per-request array below).
    pad_b: int
    # Local token bucket (rows of ``local_rows``); identical on every rank.
    local_rows_padded: int
    # Local rows carrying a real token on this rank.
    local_len: int
    # Global token index (position in the batch's request-major input_ids) of
    # each local row; -1 for padding rows.
    local_rows: np.ndarray
    # For each row of the rank-major all-gathered [cp_size * local_rows_padded]
    # tensor: request index and offset inside that request's extend range;
    # -1 for padding rows.
    gathered_req: np.ndarray
    gathered_pos: np.ndarray
    # Radix-cache prefix length per request, zero for padded requests.
    prefix_lens: np.ndarray
    # Attention rows ([2 * pad_b], one per non-empty owned block, request-major,
    # early block then late block): absolute KV end and query length, zero
    # past ``num_rows``.
    pseudo_kv_lens: np.ndarray
    pseudo_q_lens: np.ndarray
    # Request whose page row each attention row reads; a zero row at index k
    # reads request k // 2, which is the padded request when no block is empty.
    row_reqs: np.ndarray
    # Attention rows carrying a non-empty block on this rank.
    num_rows: int
    # Local row of each request's last extend token on this rank (the owner of
    # the request's last non-empty block); 0 where this rank does not own it.
    last_rows: np.ndarray
    # 1 for requests whose last token lives on this rank, else 0.
    last_owner_mask: np.ndarray
    # Ring mode only: the fused ring kernel's rows (None in gather mode).
    ring: RingLayout | None = None


class RingLayout(NamedTuple):
    """Extra inputs of the ring attention kernel for one extend batch, on
    ``cp_rank``.

    The kernel runs attention in two phases and merges the results:

    1. New tokens: each request has two query rows on this rank, one per owned
       block. A row attends to all of the request's new tokens, read from the
       all-gathered buffer; ``q_positions`` gives the causal mask.
    2. Cached prefix: each request has one query row covering both blocks. It
       attends to the request's cached prefix, which the ring rotates in from
       the other ranks.

    All lengths and positions count tokens of the whole (unsplit) sequence.
    Only ``q_positions`` differs between ranks.
    """

    # --- Phase 1 (new tokens): one entry per query row, [2 * pad_b] ---------
    # The row's query and key lengths are TpuCpHostLayout.pseudo_q_lens (C)
    # and pseudo_kv_lens (prefix + extend length).
    # Cached prefix length of the row's request.
    kv_cache_lens: np.ndarray
    # Position of the row's first query in its sequence: prefix + block * C.
    q_positions: np.ndarray
    # [2 * pad_b, gathered_pages]: for each page of the request's new tokens,
    # where that page sits in the all-gathered buffer; 0 past the last page.
    new_kv_page_indices: np.ndarray
    # Number of pages in the all-gathered buffer.
    gathered_pages: int
    # --- Phase 2 (cached prefix): one entry per request, [pad_b] -----------
    # Query rows of the request on this rank: 2 * C.
    cache_q_lens: np.ndarray
    # Cached prefix + the row's query length (the kernel subtracts the latter).
    cache_kv_lens: np.ndarray


def zigzag_block_sizes(length: int, cp_size: int) -> list[int]:
    """Sizes of the ``2 * cp_size`` zigzag blocks of one request."""
    num_blocks = 2 * cp_size
    base, rem = divmod(int(length), num_blocks)
    return [base + 1 if block < rem else base for block in range(num_blocks)]


def owned_blocks(cp_size: int, cp_rank: int) -> tuple[int, int]:
    """Block ids owned by ``cp_rank``: the early block and its zigzag partner."""
    return cp_rank, 2 * cp_size - 1 - cp_rank


def local_row_count(extend_lens: Sequence[int], cp_size: int, cp_rank: int) -> int:
    """Number of real local rows on ``cp_rank`` in gather mode."""
    early, late = owned_blocks(cp_size, cp_rank)
    total = 0
    for length in extend_lens:
        blocks = zigzag_block_sizes(length, cp_size)
        total += blocks[early] + blocks[late]
    return total


def gather_max_local_rows(extend_lens: Sequence[int], cp_size: int) -> int:
    """Gather mode: largest real local row count over all ranks (the bucketed
    quantity)."""
    return max(local_row_count(extend_lens, cp_size, rank) for rank in range(cp_size))


# ---------------------------------------------------------------------------
# Shared by both modes
# ---------------------------------------------------------------------------
def _rank_rows(
    extend_lens: Sequence[int],
    block_lens: Sequence[Sequence[int]],
    block_rows: Sequence[Sequence[int]],
    cp_size: int,
    cp_rank: int,
) -> tuple[np.ndarray, list[int]]:
    """Global token indices in local order for ``cp_rank`` and the local row
    offset at which each request starts.

    ``block_lens[i][b]`` is the number of tokens in zigzag block ``b`` of
    request ``i``; ``block_rows[i][b]`` is the number of local rows that block
    takes on its owner. Rows past a block's tokens are -1.
    """
    early, late = owned_blocks(cp_size, cp_rank)
    rows: list[np.ndarray] = []
    req_starts: list[int] = []
    global_start = 0
    local_start = 0
    for i, length in enumerate(extend_lens):
        blocks = block_lens[i]
        block_starts = np.concatenate([[0], np.cumsum(blocks)])
        req_starts.append(local_start)
        for block in (early, late):
            start = global_start + int(block_starts[block])
            end = global_start + int(block_starts[block + 1])
            rows.append(np.arange(start, end, dtype=np.int64))
            reserved = block_rows[i][block]
            rows.append(np.full(reserved - (end - start), -1, dtype=np.int64))
            local_start += reserved
        global_start += length
    if rows:
        return np.concatenate(rows), req_starts
    return np.zeros(0, dtype=np.int64), req_starts


def _pad_rows(rows: np.ndarray, padded_len: int, what: str) -> np.ndarray:
    if rows.shape[0] > padded_len:
        raise ValueError(
            f"{what}: {rows.shape[0]} real rows exceed the local token bucket "
            f"{padded_len}."
        )
    out = np.full(padded_len, -1, dtype=np.int64)
    out[: rows.shape[0]] = rows
    return out


def _check_batch(
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
) -> tuple[list[int], list[int]]:
    """Validate one extend batch; returns its lengths as Python ints."""
    extend_lens = [int(x) for x in extend_lens]
    prefix_lens = [int(x) for x in prefix_lens]
    num_reqs = len(extend_lens)
    if len(prefix_lens) != num_reqs:
        raise ValueError(
            f"prefix_lens has {len(prefix_lens)} entries for {num_reqs} requests."
        )
    if num_reqs > pad_b:
        raise ValueError(f"{num_reqs} requests exceed the request bucket {pad_b}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank {cp_rank} is not in [0, {cp_size}).")
    for length in extend_lens:
        if length < 1:
            raise ValueError(f"extend length {length} has no token to prefill.")
    return extend_lens, prefix_lens


class _BatchRows(NamedTuple):
    """Row bookkeeping both layouts share."""

    local_rows: np.ndarray
    local_len: int
    req_starts: list[int]
    gathered_req: np.ndarray
    gathered_pos: np.ndarray
    prefix_lens: np.ndarray
    last_rows: np.ndarray
    last_owner_mask: np.ndarray


def _batch_rows(
    extend_lens: list[int],
    prefix_lens: list[int],
    block_lens: list[list[int]],
    block_rows: list[list[int]],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
    local_rows_padded: int,
) -> _BatchRows:
    """Local rows, the all-gathered row map, and each request's last-token row
    for blocks of ``block_lens`` laid out as ``_rank_rows`` describes."""
    cu_extend = np.concatenate([[0], np.cumsum(extend_lens, dtype=np.int64)])
    gathered: list[np.ndarray] = []
    req_starts: list[int] = []
    local_len = 0
    for rank in range(cp_size):
        rows, starts = _rank_rows(extend_lens, block_lens, block_rows, cp_size, rank)
        gathered.append(_pad_rows(rows, local_rows_padded, f"rank {rank}"))
        if rank == cp_rank:
            req_starts = starts
            local_len = int((rows >= 0).sum())
    gathered_rows = np.concatenate(gathered)
    valid = gathered_rows >= 0
    gathered_req = np.where(
        valid, np.searchsorted(cu_extend, gathered_rows, side="right") - 1, -1
    ).astype(np.int64)
    gathered_pos = np.where(
        valid, gathered_rows - cu_extend[np.clip(gathered_req, 0, None)], -1
    ).astype(np.int64)

    early, late = owned_blocks(cp_size, cp_rank)
    padded_prefix = np.zeros(pad_b, dtype=np.int64)
    padded_prefix[: len(prefix_lens)] = prefix_lens
    last_rows = np.zeros(pad_b, dtype=np.int64)
    last_owner_mask = np.zeros(pad_b, dtype=np.int64)
    for i, blocks in enumerate(block_lens):
        last_block = max(index for index, n in enumerate(blocks) if n)
        if last_block not in (early, late):
            continue
        # The late block starts after the rows the early block takes.
        offset = block_rows[i][early] if last_block == late else 0
        last_rows[i] = req_starts[i] + offset + blocks[last_block] - 1
        last_owner_mask[i] = 1

    return _BatchRows(
        local_rows=gathered[cp_rank],
        local_len=local_len,
        req_starts=req_starts,
        gathered_req=gathered_req,
        gathered_pos=gathered_pos,
        prefix_lens=padded_prefix,
        last_rows=last_rows,
        last_owner_mask=last_owner_mask,
    )


# ---------------------------------------------------------------------------
# Gather mode: even zigzag blocks
# ---------------------------------------------------------------------------


def build_gather_layout(
    *,
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
    local_rows_padded: int,
) -> TpuCpHostLayout:
    """Gather-mode zigzag layout of one extend batch for ``cp_rank``: even
    blocks packed back to back, one attention row per non-empty owned block."""
    extend_lens, prefix_lens = _check_batch(
        extend_lens, prefix_lens, cp_size, cp_rank, pad_b
    )
    block_lens = [zigzag_block_sizes(length, cp_size) for length in extend_lens]
    rows = _batch_rows(
        extend_lens,
        prefix_lens,
        block_lens,
        block_lens,  # packed: a block takes exactly its tokens' rows
        cp_size,
        cp_rank,
        pad_b,
        local_rows_padded,
    )

    early, late = owned_blocks(cp_size, cp_rank)
    pseudo_kv_lens = np.zeros(2 * pad_b, dtype=np.int64)
    pseudo_q_lens = np.zeros(2 * pad_b, dtype=np.int64)
    row_reqs = np.arange(2 * pad_b, dtype=np.int64) // 2
    num_rows = 0
    for i, blocks in enumerate(block_lens):
        for block in (early, late):
            if blocks[block] == 0:
                continue
            row_reqs[num_rows] = i
            pseudo_kv_lens[num_rows] = prefix_lens[i] + sum(blocks[: block + 1])
            pseudo_q_lens[num_rows] = blocks[block]
            num_rows += 1

    return TpuCpHostLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        num_reqs=len(extend_lens),
        pad_b=pad_b,
        local_rows_padded=local_rows_padded,
        local_len=rows.local_len,
        local_rows=rows.local_rows,
        gathered_req=rows.gathered_req,
        gathered_pos=rows.gathered_pos,
        prefix_lens=rows.prefix_lens,
        pseudo_kv_lens=pseudo_kv_lens,
        pseudo_q_lens=pseudo_q_lens,
        row_reqs=row_reqs,
        num_rows=num_rows,
        last_rows=rows.last_rows,
        last_owner_mask=rows.last_owner_mask,
        ring=None,
    )


# ---------------------------------------------------------------------------
# Ring mode: page-aligned, rank-uniform blocks
# ---------------------------------------------------------------------------


def ring_block_size(length: int, cp_size: int, page_size: int) -> int:
    """Size of each of a request's ``2 * cp_size`` zigzag blocks: its tokens
    split evenly over the blocks, rounded up to whole pages (e.g. 6808 tokens,
    cp 4, page 256: 851 per block -> 1024)."""
    tokens_per_block = cdiv(int(length), 2 * cp_size)
    return round_up(tokens_per_block, page_size)


def ring_block_lens(length: int, cp_size: int, page_size: int) -> list[int]:
    """Tokens in each of the ``2 * cp_size`` blocks, filled in order."""
    block = ring_block_size(length, cp_size, page_size)
    return [
        min(max(int(length) - index * block, 0), block) for index in range(2 * cp_size)
    ]


def ring_max_local_rows(
    extend_lens: Sequence[int], cp_size: int, page_size: int
) -> int:
    """Ring mode: local rows of a batch, the same on every rank (the bucketed
    quantity); every request reserves ``2 * C`` rows."""
    return sum(2 * ring_block_size(n, cp_size, page_size) for n in extend_lens)


def build_ring_layout(
    *,
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
    local_rows_padded: int,
    page_size: int,
) -> TpuCpHostLayout:
    """Ring-mode zigzag layout of one extend batch for ``cp_rank``: every owned
    block reserves ``C`` page-aligned rows (``ring_block_size``), and each is
    one attention row of ``C`` queries over the request's whole chunk (the
    causal mask comes from ``RingLayout.q_positions``)."""
    extend_lens, prefix_lens = _check_batch(
        extend_lens, prefix_lens, cp_size, cp_rank, pad_b
    )
    block_sizes = [ring_block_size(n, cp_size, page_size) for n in extend_lens]
    block_lens = [ring_block_lens(n, cp_size, page_size) for n in extend_lens]
    rows = _batch_rows(
        extend_lens,
        prefix_lens,
        block_lens,
        # Every block takes C rows on its owner, full or not.
        [[size] * (2 * cp_size) for size in block_sizes],
        cp_size,
        cp_rank,
        pad_b,
        local_rows_padded,
    )

    num_reqs = len(extend_lens)
    pseudo_kv_lens = np.zeros(2 * pad_b, dtype=np.int64)
    pseudo_q_lens = np.zeros(2 * pad_b, dtype=np.int64)
    # Rows 2i and 2i + 1 (early and late block) belong to request i.
    row_reqs = np.arange(2 * pad_b, dtype=np.int64) // 2
    for i, (length, prefix) in enumerate(zip(extend_lens, prefix_lens)):
        pseudo_kv_lens[2 * i : 2 * i + 2] = prefix + length
        pseudo_q_lens[2 * i : 2 * i + 2] = block_sizes[i]

    return TpuCpHostLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        num_reqs=num_reqs,
        pad_b=pad_b,
        local_rows_padded=local_rows_padded,
        local_len=rows.local_len,
        local_rows=rows.local_rows,
        gathered_req=rows.gathered_req,
        gathered_pos=rows.gathered_pos,
        prefix_lens=rows.prefix_lens,
        pseudo_kv_lens=pseudo_kv_lens,
        pseudo_q_lens=pseudo_q_lens,
        row_reqs=row_reqs,
        num_rows=2 * num_reqs,
        last_rows=rows.last_rows,
        last_owner_mask=rows.last_owner_mask,
        ring=_ring_layout(
            extend_lens,
            prefix_lens,
            cp_size,
            cp_rank,
            pad_b,
            local_rows_padded,
            page_size,
            rows.req_starts,
        ),
    )


def _ring_layout(
    extend_lens: list[int],
    prefix_lens: list[int],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
    local_rows_padded: int,
    page_size: int,
    req_starts: list[int],
) -> RingLayout:
    """The fused ring kernel's operands on top of the ring-mode host layout."""
    early, late = owned_blocks(cp_size, cp_rank)
    gathered_pages = cp_size * local_rows_padded // page_size
    kv_cache_lens = np.zeros(2 * pad_b, dtype=np.int64)
    q_positions = np.zeros(2 * pad_b, dtype=np.int64)
    new_kv_page_indices = np.zeros((2 * pad_b, gathered_pages), dtype=np.int64)
    cache_q_lens = np.zeros(pad_b, dtype=np.int64)
    cache_kv_lens = np.zeros(pad_b, dtype=np.int64)
    for i, (length, prefix) in enumerate(zip(extend_lens, prefix_lens)):
        block = ring_block_size(length, cp_size, page_size)
        kv_cache_lens[2 * i : 2 * i + 2] = prefix
        q_positions[2 * i : 2 * i + 2] = (prefix + early * block, prefix + late * block)
        cache_q_lens[i] = 2 * block
        cache_kv_lens[i] = prefix + 2 * block
        # Page ``j`` of the chunk lies in block ``j // pages_per_block``, held by
        # the rank that owns that block at this request's local offset (the
        # same on every rank).
        pages_per_block = block // page_size
        for index in range(2 * cp_size):
            owner = index if index < cp_size else 2 * cp_size - 1 - index
            slot = 0 if index < cp_size else 1
            base = owner * local_rows_padded + req_starts[i] + slot * block
            cols = slice(index * pages_per_block, (index + 1) * pages_per_block)
            pages = base // page_size + np.arange(pages_per_block, dtype=np.int64)
            new_kv_page_indices[2 * i : 2 * i + 2, cols] = pages
    return RingLayout(
        kv_cache_lens=kv_cache_lens,
        q_positions=q_positions,
        new_kv_page_indices=new_kv_page_indices,
        gathered_pages=gathered_pages,
        cache_q_lens=cache_q_lens,
        cache_kv_lens=cache_kv_lens,
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def warmup_request_lengths(
    *, local_rows: int, cp_size: int, max_reqs: int, page_size: int = 1
) -> list[int]:
    """Request lengths whose zigzag layout fills exactly ``local_rows`` rows on
    every rank.

    Tokens come in units of ``2 * cp_size * page_size`` (``page_size`` tokens
    per block, two blocks per rank; ``page_size`` 1 in gather mode, the KV page
    size in ring mode, whose blocks are whole pages); the ``local_rows / (2 *
    page_size)`` units are spread over at most ``max_reqs`` requests as evenly
    as possible, so every block is full.
    """
    unit = 2 * page_size
    if local_rows < unit or local_rows % unit:
        raise ValueError(
            f"a context-parallel warmup needs a local token bucket that is a "
            f"positive multiple of {unit} rows (2 x page size), got {local_rows}."
        )
    if max_reqs < 1:
        raise ValueError(f"max_reqs must be positive, got {max_reqs}.")
    units = local_rows // unit
    num_reqs = min(max_reqs, units)
    base, rem = divmod(units, num_reqs)
    return [
        2 * cp_size * page_size * (base + 1 if i < rem else base)
        for i in range(num_reqs)
    ]
