"""Decode-side offer of a hybrid sliding-window model's held prefix.

A prefill can resume a hybrid model at token M only with full KV for [0, M)
and sliding-window KV for the window ending at M: an SWA layer's KV depends on
the previous layer's window, so the window cannot be recomputed alone. The
decode therefore offers the deepest page-aligned M whose window its device tree
still holds (a multi-turn follow-up finds one about where the previous turn
ended), and locks that path, full pages and window, until the request leaves
the transfer queue, so neither LRU can reuse pages the prefill reads.
"""

from __future__ import annotations

from array import array
from typing import Any, List, Optional, Sequence, Tuple

import msgspec

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import kv_to_page_indices, page_align_floor
from sglang.srt.mem_cache.radix_cache import RadixKey


class WindowPrefixOffer(msgspec.Struct, frozen=True):
    """The decode's pages of prompt prefix [0, M) and of the window ending at M."""

    # Full-attention page ids of [0, M).
    blocks: List[int]
    # SWA-pool page ids of pages [swa_first_page, M / page_size).
    swa_first_page: int
    swa_blocks: List[int]
    # Tree lock on the path to M; released when the request leaves the
    # transfer queue.
    lock_node: Any
    lock_receipt: Any


def window_tokens(*, window_size: int, page_size: int) -> int:
    """The sliding window rounded up to whole pages: a page-aligned M needs
    window KV from M - window_tokens on."""
    return -(-window_size // page_size) * page_size


def deepest_window_prefix(
    *, matched: int, holes: Sequence[Tuple[int, int]], window: int
) -> int:
    """Deepest m <= matched whose window [m - window, m) touches no SWA hole.

    ``holes`` are sorted, disjoint [lo, hi) token ranges without SWA KV; a
    window reaching before token 0 needs every token from the root.
    """
    end = matched
    for lo, hi in reversed(holes):
        if hi <= end - window:
            break
        end = min(end, lo)
    return end


def prefix_radix_key(*, req: Any, num_tokens: int, is_eagle: bool) -> RadixKey:
    """The tree key of the request's first ``num_tokens`` prompt tokens of KV."""
    # An EAGLE tree keys token pairs, so N KV slots need N + 1 tokens.
    return RadixKey(
        token_ids=array("q", req.origin_input_ids[: num_tokens + int(is_eagle)]),
        extra_key=req.extra_key,
        is_bigram=is_eagle,
        cache_salt=req.cache_salt,
    )


def offer_limit(*, req: Any, page_size: int) -> int:
    """Longest prompt prefix worth offering: the prefill computes at least the
    last prompt token."""
    return page_align_floor(len(req.origin_input_ids) - 1, page_size)


def deepest_device_window_prefix(
    *, tree_core: Any, req: Any, window_size: int, page_size: int
) -> int:
    """Deepest page-aligned M whose full KV [0, M) and window ending at M the
    decode's device tree holds, 0 when there is none. Read-only."""
    limit = offer_limit(req=req, page_size=page_size)
    if limit < page_size:
        return 0
    key = prefix_radix_key(req=req, num_tokens=limit, is_eagle=tree_core.is_eagle)
    matched, _, _ = tree_core.match_full_device_prefix(key)
    holes = tree_core.swa_tombstone_ranges(key, 0, matched)
    window = window_tokens(window_size=window_size, page_size=page_size)
    return page_align_floor(
        deepest_window_prefix(matched=matched, holes=holes, window=window),
        page_size,
    )


def lock_window_prefix(
    *, tree_cache: Any, allocator: Any, req: Any, window_size: int
) -> Optional[WindowPrefixOffer]:
    """Lock and describe the deepest prompt prefix whose window this decode's
    device tree holds; None when there is none worth a page.

    Every rank runs this in the same scheduler step on the same tree, so all
    take the same lock.
    """
    page_size = int(allocator.page_size)
    tree_core = tree_cache.tree_core
    prefix_len = deepest_device_window_prefix(
        tree_core=tree_core, req=req, window_size=window_size, page_size=page_size
    )
    if prefix_len < page_size:
        return None
    # The match splits the path at prefix_len, so the lock covers exactly it.
    match = tree_cache.match_prefix(
        MatchPrefixParams(
            key=prefix_radix_key(
                req=req, num_tokens=prefix_len, is_eagle=tree_core.is_eagle
            )
        )
    )
    full = match.device_indices
    if len(full) != prefix_len:
        return None
    receipt = tree_cache.inc_lock_ref(match.last_device_node).to_dec_params()
    window = window_tokens(window_size=window_size, page_size=page_size)
    swa_start = max(0, prefix_len - window)
    swa = allocator.translate_swa_indices_for_transfer(full[swa_start:prefix_len])
    # A freed window slot maps to 0: the window is not whole after all.
    if bool((swa <= 0).any()):
        tree_cache.dec_lock_ref(match.last_device_node, receipt)
        return None
    return WindowPrefixOffer(
        blocks=kv_to_page_indices(full, page_size).tolist(),
        swa_first_page=swa_start // page_size,
        swa_blocks=kv_to_page_indices(swa, page_size).tolist(),
        lock_node=match.last_device_node,
        lock_receipt=receipt,
    )
