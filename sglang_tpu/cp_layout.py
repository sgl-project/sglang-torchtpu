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
