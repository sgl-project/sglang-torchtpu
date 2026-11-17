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
    """Attention rows of one ring-mode extend batch as seen from ``cp_rank``.

    Lengths and positions are global (unsharded) token counts; the kernel
    derives each rank's page stripe from them and ``cp_rank``. Every array has
    the same value on every rank except ``q_positions``.

    New-token phase: two rows per request (early block, then late block), each
    with ``C`` queries -- the block's tokens followed by padding queries. The new keys of a row are the request's whole chunk
    (``new_kv_lens - kv_cache_lens`` tokens, the real ones only), read from the
    rank-major all-gathered buffer through ``new_kv_page_indices``; the causal
    mask comes from ``q_positions``.

    Cache phase: one row per request over the same local query rows (both
    blocks), attending the cached prefix only.
    """

    # --- new-token phase: the host layout's [2 * pad_b] attention rows
    # (``pseudo_q_lens`` C, ``pseudo_kv_lens`` prefix + extend length), plus:
    # Cached prefix of the row's request.
    kv_cache_lens: np.ndarray
    # Global position of the row's first query: prefix + block * C.
    q_positions: np.ndarray
    # [2 * pad_b, gathered_pages] page of the rank-major all-gathered buffer
    # holding each page of the row's chunk; 0 past the chunk's 2 * cp * C rows.
    new_kv_page_indices: np.ndarray
    # Pages of the all-gathered buffer: cp_size * local_rows_padded / page_size.
    gathered_pages: int
    # --- cache phase, [pad_b] rows (num_reqs live) -------------------------
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
    """Number of real local rows on ``cp_rank``."""
    early, late = owned_blocks(cp_size, cp_rank)
    total = 0
    for length in extend_lens:
        blocks = zigzag_block_sizes(length, cp_size)
        total += blocks[early] + blocks[late]
    return total


def max_local_rows(
    extend_lens: Sequence[int], cp_size: int, page_size: int | None = None
) -> int:
    """Largest real local row count over all ranks (the bucketed quantity).
    Ring mode (``page_size``) reserves ``2 * C`` rows per request on every rank."""
    if page_size is not None:
        return sum(2 * ring_block_size(n, cp_size, page_size) for n in extend_lens)
    return max(local_row_count(extend_lens, cp_size, rank) for rank in range(cp_size))


def _block_lens(length: int, cp_size: int, page_size: int | None) -> list[int]:
    """Tokens in each zigzag block: the even split, or ring mode's page-aligned
    blocks filled in order."""
    if page_size is None:
        return zigzag_block_sizes(length, cp_size)
    return ring_block_lens(length, cp_size, page_size)


def _rank_rows(
    extend_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    page_size: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Global token indices in local order for ``cp_rank`` and the local row
    offset at which each request starts. In ring mode every owned block
    reserves ``C`` rows; the rows past its tokens are -1."""
    early, late = owned_blocks(cp_size, cp_rank)
    rows: list[np.ndarray] = []
    req_starts: list[int] = []
    global_start = 0
    local_start = 0
    for length in extend_lens:
        blocks = _block_lens(length, cp_size, page_size)
        block_starts = np.concatenate([[0], np.cumsum(blocks)])
        req_starts.append(local_start)
        for block in (early, late):
            start = global_start + int(block_starts[block])
            end = global_start + int(block_starts[block + 1])
            rows.append(np.arange(start, end, dtype=np.int64))
            local_start += end - start
            if page_size is not None:
                pad = ring_block_size(length, cp_size, page_size) - (end - start)
                rows.append(np.full(pad, -1, dtype=np.int64))
                local_start += pad
        global_start += int(length)
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


def build_tpu_cp_layout(
    *,
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
    local_rows_padded: int,
    page_size: int | None = None,
) -> TpuCpHostLayout:
    """Build the zigzag layout of one extend batch for ``cp_rank``; with
    ``page_size``, ring mode's page-aligned, rank-uniform layout."""
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
    if page_size is not None:
        _check_ring_shapes(page_size, local_rows_padded)

    cu_extend = np.concatenate([[0], np.cumsum(extend_lens, dtype=np.int64)])

    gathered: list[np.ndarray] = []
    for rank in range(cp_size):
        rows, _ = _rank_rows(extend_lens, cp_size, rank, page_size)
        gathered.append(_pad_rows(rows, local_rows_padded, f"rank {rank}"))
    gathered_rows = np.concatenate(gathered)
    valid = gathered_rows >= 0
    gathered_req = np.where(
        valid, np.searchsorted(cu_extend, gathered_rows, side="right") - 1, -1
    ).astype(np.int64)
    gathered_pos = np.where(
        valid, gathered_rows - cu_extend[np.clip(gathered_req, 0, None)], -1
    ).astype(np.int64)

    local_rows_real, req_starts = _rank_rows(extend_lens, cp_size, cp_rank, page_size)
    local_rows = gathered[cp_rank]

    early, late = owned_blocks(cp_size, cp_rank)
    pseudo_kv_lens = np.zeros(2 * pad_b, dtype=np.int64)
    pseudo_q_lens = np.zeros(2 * pad_b, dtype=np.int64)
    row_reqs = np.arange(2 * pad_b, dtype=np.int64) // 2
    last_rows = np.zeros(pad_b, dtype=np.int64)
    last_owner_mask = np.zeros(pad_b, dtype=np.int64)
    padded_prefix = np.zeros(pad_b, dtype=np.int64)
    num_rows = 0
    for i, length in enumerate(extend_lens):
        blocks = _block_lens(length, cp_size, page_size)
        prefix = prefix_lens[i]
        padded_prefix[i] = prefix
        early_rows = blocks[early]
        if page_size is not None:
            # Ring mode: both owned blocks are rows of C queries over the whole
            # chunk (the causal mask comes from RingLayout.q_positions).
            early_rows = ring_block_size(length, cp_size, page_size)
            row_reqs[num_rows : num_rows + 2] = i
            pseudo_kv_lens[num_rows : num_rows + 2] = prefix + length
            pseudo_q_lens[num_rows : num_rows + 2] = early_rows
            num_rows += 2
        for block in (early, late):
            if blocks[block] == 0 or page_size is not None:
                continue
            row_reqs[num_rows] = i
            pseudo_kv_lens[num_rows] = prefix + sum(blocks[: block + 1])
            pseudo_q_lens[num_rows] = blocks[block]
            num_rows += 1
        last_block = max(index for index, n in enumerate(blocks) if n)
        if last_block in (early, late):
            offset = early_rows if last_block == late else 0
            last_rows[i] = req_starts[i] + offset + blocks[last_block] - 1
            last_owner_mask[i] = 1

    return TpuCpHostLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        num_reqs=num_reqs,
        pad_b=pad_b,
        local_rows_padded=local_rows_padded,
        local_len=int((local_rows_real >= 0).sum()),
        local_rows=local_rows,
        gathered_req=gathered_req,
        gathered_pos=gathered_pos,
        prefix_lens=padded_prefix,
        pseudo_kv_lens=pseudo_kv_lens,
        pseudo_q_lens=pseudo_q_lens,
        row_reqs=row_reqs,
        num_rows=num_rows,
        last_rows=last_rows,
        last_owner_mask=last_owner_mask,
        ring=None
        if page_size is None
        else _ring_layout(
            extend_lens,
            prefix_lens,
            cp_size,
            cp_rank,
            pad_b,
            local_rows_padded,
            page_size,
            req_starts,
        ),
    )


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


def _check_ring_shapes(page_size: int, local_rows_padded: int) -> None:
    if page_size < 16 or page_size % 16:
        # The kernel hands the cache phase's softmax state to the new-token
        # phase in 16-row units, so every row must start on one.
        raise ValueError(
            f"Ring-mode context parallelism needs a page size that is a multiple "
            f"of 16, got {page_size}."
        )
    if local_rows_padded % page_size:
        raise ValueError(
            f"The local token bucket {local_rows_padded} is not a multiple of the "
            f"page size {page_size}: ring-mode context parallelism reads the "
            "all-gathered chunk by page."
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
