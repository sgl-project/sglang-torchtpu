"""Fail-closed benchmark-only admission inventory validation.

No server transport or pause implementation: callers must obtain snapshots
on scheduler threads while paused. Counts alone are intentionally insufficient.
"""
from collections import Counter


def validate_inventory(expected_routes, snapshots):
    """Validate one snapshot per attention-DP scheduler (PP1/attnTP1 only).

    expected_routes maps explicit RID to integer DP rank. Each snapshot contains
    dp_rank, paused, waiting_rids, running_rids, chunked_rids. Return diagnostic
    reasons rather than granting admission on incomplete/malformed evidence.
    """
    if not expected_routes or any(type(rid) is not str or not rid or
                                  type(rank) is not int or rank < 0
                                  for rid, rank in expected_routes.items()):
        raise ValueError('nonempty explicit RID-to-rank mapping required')
    ranks = set(expected_routes.values())
    if ranks != set(range(max(ranks) + 1)):
        raise ValueError('expected routes must cover contiguous participating ranks')
    reasons, seen_ranks, seen_rids = [], [], []
    for snap in snapshots:
        if not isinstance(snap, dict):
            reasons.append('malformed_snapshot')
            continue
        rank = snap.get('dp_rank')
        if type(rank) is not int or rank not in ranks:
            reasons.append('unexpected_rank')
            continue
        seen_ranks.append(rank)
        if snap.get('paused') is not True:
            reasons.append(f'rank{rank}:not_paused')
        for key in ('waiting_rids', 'running_rids', 'chunked_rids'):
            value = snap.get(key)
            if not isinstance(value, list) or any(type(rid) is not str for rid in value):
                reasons.append(f'rank{rank}:malformed_{key}')
                continue
            if key != 'waiting_rids':
                if value:
                    reasons.append(f'rank{rank}:nonempty_{key}')
                continue
            seen_rids.extend(value)
            if any(expected_routes.get(rid) != rank for rid in value):
                reasons.append(f'rank{rank}:wrong_route_or_unexpected_rid')
    if Counter(seen_ranks) != Counter({rank: 1 for rank in ranks}):
        reasons.append('missing_or_duplicate_rank')
    if Counter(seen_rids) != Counter({rid: 1 for rid in expected_routes}):
        reasons.append('missing_duplicate_or_unexpected_rid')
    return {'ready': not reasons, 'reasons': sorted(set(reasons)),
            'expected_requests': len(expected_routes),
            'observed_waiting_entries': len(seen_rids),
            'scope': 'inventory only; no transport, atomicity or completion timing proof'}
