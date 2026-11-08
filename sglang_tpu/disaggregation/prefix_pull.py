# SPDX-License-Identifier: Apache-2.0
"""Pull the longest prefix from the decode (P/D prefix pull).

The decode holds every prompt token's full KV it received, so the prefill can
pull a prefix the decode cached but the prefill no longer has. A pull point m is
usable only where the decode still holds the SWA window [m - W, m); everything
past m is computed on the prefill.

This module holds the side-effect-free pieces: the pull-point search over a
decode radix path and the per-request gap classification.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import msgspec


@dataclass(frozen=True)
class PathSegment:
    """One radix node on a root-to-leaf path: tokens [start, end)."""

    start: int
    end: int
    # Whether the node's SWA KV is on device.
    swa_resident: bool


def window_start(end: int, window: int, page_size: int) -> int:
    """First token of the page-aligned SWA window ending at end."""
    return (max(0, end - window) // page_size) * page_size


def longest_pullable_len(
    segments: Sequence[PathSegment], limit: int, window: int, page_size: int
) -> int:
    """Longest page-aligned m <= limit whose SWA window lies on resident nodes.

    Returns 0 when no m > 0 qualifies.
    """
    runs: list[list[int]] = []
    prev_resident = False
    for seg in segments:
        if seg.start >= limit:
            break
        if seg.swa_resident and prev_resident:
            runs[-1][1] = seg.end
        elif seg.swa_resident:
            runs.append([seg.start, seg.end])
        prev_resident = seg.swa_resident
    # A deeper run gives a larger m. The window grows toward smaller m, so only
    # each run's largest m can fit.
    for run_start, run_end in reversed(runs):
        m = (min(run_end, limit) // page_size) * page_size
        if m > 0 and window_start(m, window, page_size) >= run_start:
            return m
    return 0


def iter_path_segments(
    tree_core: Any, node_id: Any, component: Any
) -> Iterator[PathSegment]:
    """Segments from the root to node_id, root first, for one SWA component type."""
    nodes = []
    node = tree_core.node_by_id(node_id)
    while node.parent is not None:
        nodes.append(node)
        node = node.parent
    start = 0
    for node in reversed(nodes):
        end = start + len(node.key)
        yield PathSegment(
            start=start,
            end=end,
            swa_resident=node.component_data[component].value is not None,
        )
        start = end


class PrefixGapCase(str, enum.Enum):
    """Where the decode's prefix stands against the prefill's own match."""

    # The decode holds no prefix of the prompt.
    DECODE_MISS = "decode_miss"
    # The prefill already has everything the decode has.
    PREFILL_COVERS = "prefill_covers"
    # The decode is ahead by less than the pull threshold.
    DECODE_AHEAD_SMALL = "decode_ahead_small"
    # The decode is ahead by at least the pull threshold: a pull candidate.
    DECODE_AHEAD_PULL = "decode_ahead_pull"


def classify_prefix_gap(
    *, decode_prefix_len: int, prefill_prefix_len: int, min_pull_tokens: int
) -> tuple[PrefixGapCase, int]:
    """Case and gap tokens max(0, decode_prefix_len - prefill_prefix_len)."""
    gap = max(0, decode_prefix_len - prefill_prefix_len)
    if decode_prefix_len <= 0:
        return PrefixGapCase.DECODE_MISS, 0
    if gap == 0:
        return PrefixGapCase.PREFILL_COVERS, 0
    if gap < min_pull_tokens:
        return PrefixGapCase.DECODE_AHEAD_SMALL, gap
    return PrefixGapCase.DECODE_AHEAD_PULL, gap


def decode_match_cap(prompt_len: int, window: int, page_size: int) -> int:
    """Longest prefix a decode with SWA tail prealloc matches: the window start."""
    return window_start(prompt_len, window, page_size)


class DecodeWindowCase(str, enum.Enum):
    """Whether the decode still holds the SWA window at its match end."""

    NO_PREFIX = "no_prefix"
    # The window ending at decode_prefix_len is resident: all of it is pullable.
    RESIDENT = "resident"
    # Only a shorter prefix has a resident window.
    PARTIAL = "partial"
    # No prefix has a resident window.
    MISSING = "missing"


def classify_decode_window(
    *, decode_prefix_len: int, pullable_len: int
) -> DecodeWindowCase:
    if decode_prefix_len <= 0:
        return DecodeWindowCase.NO_PREFIX
    if pullable_len >= decode_prefix_len:
        return DecodeWindowCase.RESIDENT
    if pullable_len > 0:
        return DecodeWindowCase.PARTIAL
    return DecodeWindowCase.MISSING


class DecodePrefixReport(msgspec.Struct, frozen=True):
    """What a decode holds of a request's prompt, sent with its KV request."""

    # Prefix the decode reuses from its own cache (decode_prefix_len).
    prefix_len: int = 0
    # Longest page-aligned prefix whose SWA window the decode holds on device.
    pullable_len: int = 0
    # Whether the decode locked that prefix for a pull (SGLANG_TPU_PD_PREFIX_PULL).
    pull_held: bool = False

    def to_fields(self) -> dict:
        return dict(
            decode_prefix_len=self.prefix_len,
            decode_pullable_len=self.pullable_len,
            decode_pull_held=self.pull_held,
        )

    @classmethod
    def from_fields(cls, fields: dict) -> Optional[DecodePrefixReport]:
        if "decode_prefix_len" not in fields:
            return None
        return cls(
            prefix_len=int(fields["decode_prefix_len"]),
            pullable_len=int(fields.get("decode_pullable_len", 0)),
            pull_held=bool(fields.get("decode_pull_held", False)),
        )


def agree(
    reports: Sequence[Optional[DecodePrefixReport]],
) -> Optional[DecodePrefixReport]:
    """One report from every decode rank's copy; ranks match the same tree, so
    the minimum only guards a rank that saw less."""
    if not reports or any(r is None for r in reports):
        return None
    return DecodePrefixReport(
        prefix_len=min(r.prefix_len for r in reports),
        pullable_len=min(r.pullable_len for r in reports),
        pull_held=all(r.pull_held for r in reports),
    )


class _PullableByRoom:
    """Decode: pullable prefix found at match time, read when the KV request goes out."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_room: dict[int, tuple[int, bool]] = {}

    def put(self, room: int, pullable_len: int, held: bool) -> None:
        with self._lock:
            self._by_room[room] = (pullable_len, held)

    def report(self, room: int, prefix_len: int) -> DecodePrefixReport:
        """The room's report, consuming its entry."""
        with self._lock:
            pullable_len, held = self._by_room.pop(room, (0, False))
        return DecodePrefixReport(
            prefix_len=prefix_len, pullable_len=pullable_len, pull_held=held
        )


pullable_by_room = _PullableByRoom()
