"""Attribute a request's prefix misses to a missing SWA window.

Only meaningful when SWA is the cache's only match validator besides full KV;
otherwise a cut in the match cannot be attributed to SWA, and every count is zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult


class SwaPrefixMatch(msgspec.Struct, frozen=True):
    """Match-time prefix lengths, in tokens."""

    full_kv_hit_length: int = 0
    # Deepest prefix every validator accepted, including host-restorable windows and
    # backed-up FULL; a load-back declined past it is a capacity loss, not an SWA miss.
    reusable_length: int = 0


def swa_prefix_match(
    *, tree_cache: BasePrefixCache, match_result: MatchResult
) -> SwaPrefixMatch:
    if not tree_cache.swa_is_only_match_cut():
        return SwaPrefixMatch()
    reusable = max(
        match_result.best_match_length,
        len(match_result.device_indices) + match_result.host_hit_length,
    )
    # Producers that skip the tree walk (streaming sessions) leave the full-KV
    # length at 0; whatever the match reuses has full KV too.
    return SwaPrefixMatch(
        full_kv_hit_length=max(match_result.full_kv_hit_length, reusable),
        reusable_length=reusable,
    )


def swa_miss_len(*, match: SwaPrefixMatch, prefix_len: int) -> int:
    """Prefix tokens with full KV cached but recomputed because the SWA window was
    missing. `prefix_len` is the admitted prefix after load-back, so a window that
    load-back restored is reused, not missed."""
    return max(0, match.full_kv_hit_length - max(prefix_len, match.reusable_length))
