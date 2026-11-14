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

Ring mode (``SGLANG_TPU_PCP_MODE=ring``, ``build_tpu_cp_ring_layout``) keeps
the block ownership but makes the blocks page-aligned and rank-uniform, because
the fused ring kernel reads the all-gathered chunk by page and every rank must
run the same ring schedule:

* block size ``C = align_up(ceil(length / (2 * cp_size)), page_size)``; the
  blocks are filled in order, so trailing blocks are short or empty;
* every rank reserves ``2 * C`` local rows per request -- its early block at
  the request's local offset and its late block ``C`` rows later -- whether or
  not the blocks hold tokens. Rows past a block's tokens are padding: they are
  attention queries whose output is discarded, and they sit at chunk positions
  at or past the request's length, so no row reads them as keys and they are
  never written to the KV cache (``RingLayout`` below).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


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
    with ``block_sizes[i]`` queries -- the block's tokens followed by padding
    queries. The new keys of a row are the request's whole chunk
    (``new_kv_lens - kv_cache_lens`` tokens, the real ones only), read from the
    rank-major all-gathered buffer through ``new_kv_page_indices``; the causal
    mask comes from ``q_positions``.

    Cache phase: ``cache_rows_per_req`` rows per request over the same local
    query rows (one row holding both blocks, or the two new-token rows), each
    attending the cached prefix only.
    """

    page_size: int
    # [pad_b] page-aligned zigzag block size of each request, 0 when padded.
    block_sizes: np.ndarray
    # [pad_b] local row at which each request's 2 * C reserved rows start.
    req_starts: np.ndarray
    # Whether any request has a cached prefix (the ring then has steps to run).
    has_cache: bool
    # --- new-token phase, [2 * pad_b] rows -------------------------------
    new_q_lens: np.ndarray
    # prefix + extend length of the row's request.
    new_kv_lens: np.ndarray
    # Cached prefix of the row's request.
    kv_cache_lens: np.ndarray
    # Global position of the row's first query: prefix + block * C.
    q_positions: np.ndarray
    # Rows carrying a request (2 * num_reqs).
    new_num_rows: int
    # [2 * pad_b, gathered_pages] page of the rank-major all-gathered buffer
    # holding each page of the row's chunk; 0 past the chunk's 2 * cp * C rows.
    new_kv_page_indices: np.ndarray
    # Pages of the all-gathered buffer: cp_size * local_rows_padded / page_size.
    gathered_pages: int
    # --- cache phase, [cache_rows_per_req * pad_b] rows ------------------
    cache_rows_per_req: int
    cache_q_lens: np.ndarray
    # Cached prefix + the row's query length (the kernel subtracts the latter).
    cache_kv_lens: np.ndarray
    cache_num_rows: int
    # Request whose page-table row each cache row reads.
    cache_row_reqs: np.ndarray


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


def max_local_rows(extend_lens: Sequence[int], cp_size: int) -> int:
    """Largest real local row count over all ranks (the bucketed quantity)."""
    return max(local_row_count(extend_lens, cp_size, rank) for rank in range(cp_size))


def _rank_rows(
    extend_lens: Sequence[int], cp_size: int, cp_rank: int
) -> tuple[np.ndarray, list[int]]:
    """Global token indices in local order for ``cp_rank`` and the local row
    offset at which each request starts."""
    early, late = owned_blocks(cp_size, cp_rank)
    rows: list[np.ndarray] = []
    req_starts: list[int] = []
    global_start = 0
    local_start = 0
    for length in extend_lens:
        blocks = zigzag_block_sizes(length, cp_size)
        block_starts = np.concatenate([[0], np.cumsum(blocks)])
        req_starts.append(local_start)
        for block in (early, late):
            start = global_start + int(block_starts[block])
            end = global_start + int(block_starts[block + 1])
            rows.append(np.arange(start, end, dtype=np.int64))
            local_start += end - start
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
) -> TpuCpHostLayout:
    """Build the zigzag layout of one extend batch for ``cp_rank``."""
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

    cu_extend = np.concatenate([[0], np.cumsum(extend_lens, dtype=np.int64)])

    gathered: list[np.ndarray] = []
    for rank in range(cp_size):
        rows, _ = _rank_rows(extend_lens, cp_size, rank)
        gathered.append(_pad_rows(rows, local_rows_padded, f"rank {rank}"))
    gathered_rows = np.concatenate(gathered)
    valid = gathered_rows >= 0
    gathered_req = np.where(
        valid, np.searchsorted(cu_extend, gathered_rows, side="right") - 1, -1
    ).astype(np.int64)
    gathered_pos = np.where(
        valid, gathered_rows - cu_extend[np.clip(gathered_req, 0, None)], -1
    ).astype(np.int64)

    local_rows_real, req_starts = _rank_rows(extend_lens, cp_size, cp_rank)
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
        blocks = zigzag_block_sizes(length, cp_size)
        prefix = prefix_lens[i]
        padded_prefix[i] = prefix
        for block in (early, late):
            if blocks[block] == 0:
                continue
            row_reqs[num_rows] = i
            pseudo_kv_lens[num_rows] = prefix + sum(blocks[: block + 1])
            pseudo_q_lens[num_rows] = blocks[block]
            num_rows += 1
        last_block = min(length, 2 * cp_size) - 1
        if last_block in (early, late):
            offset = blocks[early] if last_block == late else 0
            last_rows[i] = req_starts[i] + offset + blocks[last_block] - 1
            last_owner_mask[i] = 1

    return TpuCpHostLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        num_reqs=num_reqs,
        pad_b=pad_b,
        local_rows_padded=local_rows_padded,
        local_len=int(local_rows_real.shape[0]),
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
    )


def warmup_request_lengths(
    *, local_rows: int, cp_size: int, max_reqs: int
) -> list[int]:
    """Request lengths whose zigzag layout fills exactly ``local_rows`` rows on
    every rank.

    Tokens come in units of ``2 * cp_size`` (one token per block, two rows
    per rank); ``local_rows / 2`` units are spread over at most ``max_reqs``
    requests as evenly as possible.
    """
    if local_rows < 2 or local_rows % 2:
        raise ValueError(
            f"a context-parallel warmup needs an even local token bucket of at "
            f"least 2 rows, got {local_rows}."
        )
    if max_reqs < 1:
        raise ValueError(f"max_reqs must be positive, got {max_reqs}.")
    units = local_rows // 2
    num_reqs = min(max_reqs, units)
    base, rem = divmod(units, num_reqs)
    return [2 * cp_size * (base + 1 if i < rem else base) for i in range(num_reqs)]


# ---------------------------------------------------------------------------
# Ring mode: page-aligned, rank-uniform blocks
# ---------------------------------------------------------------------------


def ring_block_size(length: int, cp_size: int, page_size: int) -> int:
    """Page-aligned zigzag block size of a request of ``length`` tokens."""
    block = -(-int(length) // (2 * cp_size))
    return -(-block // page_size) * page_size


def ring_block_lens(length: int, cp_size: int, page_size: int) -> list[int]:
    """Tokens in each of the ``2 * cp_size`` blocks, filled in order."""
    block = ring_block_size(length, cp_size, page_size)
    return [
        min(max(int(length) - index * block, 0), block) for index in range(2 * cp_size)
    ]


def ring_local_rows(extend_lens: Sequence[int], cp_size: int, page_size: int) -> int:
    """Local rows a ring-mode batch reserves on every rank (the bucketed
    quantity): ``2 * C`` per request."""
    return sum(
        2 * ring_block_size(length, cp_size, page_size) for length in extend_lens
    )


def build_tpu_cp_ring_layout(
    *,
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    pad_b: int,
    local_rows_padded: int,
    page_size: int,
    cache_rows_per_req: int = 1,
) -> TpuCpHostLayout:
    """Build the ring-mode zigzag layout of one extend batch for ``cp_rank``.

    ``cache_rows_per_req`` is 1 (one cache-phase row per request spanning both
    owned blocks) or 2 (the cache phase reuses the two new-token rows).
    """
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
    if cache_rows_per_req not in (1, 2):
        raise ValueError(
            f"cache_rows_per_req must be 1 or 2, got {cache_rows_per_req}."
        )
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
    for length in extend_lens:
        if length < 1:
            raise ValueError(f"extend length {length} has no token to prefill.")
    reserved = ring_local_rows(extend_lens, cp_size, page_size)
    if reserved > local_rows_padded:
        raise ValueError(
            f"ring-mode context parallelism reserves {reserved} local rows for "
            f"{num_reqs} requests, above the local token bucket {local_rows_padded}."
        )

    early, late = owned_blocks(cp_size, cp_rank)
    gathered_pages = cp_size * local_rows_padded // page_size
    cu_extend = np.concatenate([[0], np.cumsum(extend_lens, dtype=np.int64)])

    # Request and chunk offset of every row of the rank-major all-gathered
    # buffer (-1 for padding), for the owner-restricted KV write.
    gathered_req = np.full(cp_size * local_rows_padded, -1, dtype=np.int64)
    gathered_pos = np.full(cp_size * local_rows_padded, -1, dtype=np.int64)
    local_rows = np.full(local_rows_padded, -1, dtype=np.int64)
    block_sizes = np.zeros(pad_b, dtype=np.int64)
    req_starts = np.zeros(pad_b, dtype=np.int64)
    padded_prefix = np.zeros(pad_b, dtype=np.int64)
    new_q_lens = np.zeros(2 * pad_b, dtype=np.int64)
    new_kv_lens = np.zeros(2 * pad_b, dtype=np.int64)
    kv_cache_lens = np.zeros(2 * pad_b, dtype=np.int64)
    q_positions = np.zeros(2 * pad_b, dtype=np.int64)
    new_kv_page_indices = np.zeros((2 * pad_b, gathered_pages), dtype=np.int64)
    last_rows = np.zeros(pad_b, dtype=np.int64)
    last_owner_mask = np.zeros(pad_b, dtype=np.int64)

    local_len = 0
    local_start = 0
    for i, (length, prefix) in enumerate(zip(extend_lens, prefix_lens)):
        block = ring_block_size(length, cp_size, page_size)
        lens = ring_block_lens(length, cp_size, page_size)
        block_sizes[i] = block
        req_starts[i] = local_start
        padded_prefix[i] = prefix
        for slot, index in enumerate((early, late)):
            row = 2 * i + slot
            start = local_start + slot * block
            first = int(cu_extend[i]) + index * block
            local_rows[start : start + lens[index]] = np.arange(
                first, first + lens[index], dtype=np.int64
            )
            local_len += lens[index]
            new_q_lens[row] = block
            new_kv_lens[row] = prefix + length
            kv_cache_lens[row] = prefix
            q_positions[row] = prefix + index * block
        # Page ``j`` of the chunk lies in block ``j // pages_per_block``, held by
        # the rank that owns that block at this request's local offset.
        pages_per_block = block // page_size
        for index in range(2 * cp_size):
            owner = index if index < cp_size else 2 * cp_size - 1 - index
            slot = 0 if index < cp_size else 1
            base = owner * local_rows_padded + local_start + slot * block
            cols = slice(index * pages_per_block, (index + 1) * pages_per_block)
            pages = base // page_size + np.arange(pages_per_block, dtype=np.int64)
            new_kv_page_indices[2 * i, cols] = pages
            new_kv_page_indices[2 * i + 1, cols] = pages
            gathered_req[base : base + lens[index]] = i
            gathered_pos[base : base + lens[index]] = index * block + np.arange(
                lens[index], dtype=np.int64
            )
        last_block, last_offset = divmod(length - 1, block)
        if last_block in (early, late):
            slot = 0 if last_block == early else 1
            last_rows[i] = local_start + slot * block + last_offset
            last_owner_mask[i] = 1
        local_start += 2 * block

    if cache_rows_per_req == 1:
        cache_q_lens = 2 * block_sizes
        cache_prefix = padded_prefix
        cache_row_reqs = np.arange(pad_b, dtype=np.int64)
    else:
        cache_q_lens = new_q_lens.copy()
        cache_prefix = kv_cache_lens
        cache_row_reqs = np.arange(2 * pad_b, dtype=np.int64) // 2
    # A padded request's rows stay all-zero (no queries, no keys).
    cache_kv_lens = np.where(cache_q_lens > 0, cache_prefix + cache_q_lens, 0)

    ring = RingLayout(
        page_size=page_size,
        block_sizes=block_sizes,
        req_starts=req_starts,
        has_cache=any(prefix > 0 for prefix in prefix_lens),
        new_q_lens=new_q_lens,
        new_kv_lens=new_kv_lens,
        kv_cache_lens=kv_cache_lens,
        q_positions=q_positions,
        new_num_rows=2 * num_reqs,
        new_kv_page_indices=new_kv_page_indices,
        gathered_pages=gathered_pages,
        cache_rows_per_req=cache_rows_per_req,
        cache_q_lens=cache_q_lens,
        cache_kv_lens=cache_kv_lens,
        cache_num_rows=cache_rows_per_req * num_reqs,
        cache_row_reqs=cache_row_reqs,
    )
    return TpuCpHostLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        num_reqs=num_reqs,
        pad_b=pad_b,
        local_rows_padded=local_rows_padded,
        local_len=local_len,
        local_rows=local_rows,
        gathered_req=gathered_req,
        gathered_pos=gathered_pos,
        prefix_lens=padded_prefix,
        pseudo_kv_lens=new_kv_lens,
        pseudo_q_lens=new_q_lens,
        row_reqs=np.arange(2 * pad_b, dtype=np.int64) // 2,
        num_rows=2 * num_reqs,
        last_rows=last_rows,
        last_owner_mask=last_owner_mask,
        ring=ring,
    )


def ring_warmup_request_lengths(
    *, local_rows: int, cp_size: int, page_size: int, max_reqs: int
) -> list[int]:
    """Request lengths whose ring-mode blocks fill ``local_rows`` exactly.

    A request reserves at least ``2 * page_size`` local rows, so the bucket
    holds at most ``local_rows // (2 * page_size)`` requests; the first one
    takes the pages the others leave.
    """
    pairs = local_rows // (2 * page_size)
    if pairs < 1 or local_rows % (2 * page_size):
        raise ValueError(
            f"A ring-mode context-parallel token bucket must be a positive "
            f"multiple of 2 x page size ({2 * page_size}), got {local_rows}."
        )
    num_reqs = max(1, min(max_reqs, pairs))
    block_pages = [1] * num_reqs
    block_pages[0] = pairs - (num_reqs - 1)
    return [2 * cp_size * pages * page_size for pages in block_pages]
