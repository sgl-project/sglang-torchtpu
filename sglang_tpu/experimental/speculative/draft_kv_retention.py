"""Extend rows whose windowed draft KV can still be read after the extend.

A windowed (all-SWA) block draft keeps its KV in the target's SWA slots and never
runs on a prefill. A row's draft KV is read only if its SWA slot outlives the
extend: the window before the prefill end (the request's own decode) and the
windows the unified radix cache keeps at the points it publishes.
"""

from __future__ import annotations

import enum

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.experimental import runs_draft_prompt_pass


class CachePublishing(enum.Enum):
    """Which SWA windows the prefix cache keeps past the extend."""

    # No prefix sharing: only the prefill end's window, for the request's decode.
    PREFILL_END = enum.auto()
    # Unified radix cache: also every window an insert publishes.
    UNIFIED_RADIX = enum.auto()
    # Retained set not derived for this cache: every row keeps its draft KV.
    EVERY_ROW = enum.auto()


class DraftKvRetention(msgspec.Struct, frozen=True):
    publishing: CachePublishing
    # max(draft window, cache window) in tokens.
    window: int
    page_size: int
    # Radix keys of bigram (EAGLE) trees are one shorter than the tokens they span.
    bigram_shift: int = 0
    # SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE; 0 publishes every chunk.
    chunk_insert_stride: int = 0
    # A PD prefill checkpoints every chunk (never mid_prefill), whatever the stride.
    publishes_every_chunk: bool = False

    def window_floor(self, point: int) -> int:
        """Lowest position whose SWA slot survives a publish (or decode) at point.

        Mirrors free_swa_out_of_window_slots (mem_cache/common.py) at
        pre_len = len(radix key) - 1, as cache_unfinished_req passes it.
        """
        pre_len = point - 1 - self.bigram_shift
        floor = pre_len - max(self.window, self.page_size)
        return max(0, floor // self.page_size * self.page_size)


class ExtendRequest(msgspec.Struct, frozen=True):
    """One request's rows [prefix_len, prefix_len + extend_len) of an extend."""

    prefix_len: int
    extend_len: int
    # Where the prefill ends: the first decode query position.
    prefill_end: int
    swa_branch: int | None = None
    # A radix insert may publish this request's windows.
    publishes: bool = True
    # A streaming session checkpoints through its own path: keep every row.
    keeps_every_row: bool = False


def selects_retained_extend_rows(disagg_mode: str) -> bool:
    """Whether the extend writes only retained rows (host-only selection, local writer)."""
    # A DP-attention batch pads every rank to one writer shape; a PD decode's tree
    # (HiCache sidecar) keeps windows this rule does not derive.
    return (
        disagg_mode != "decode"
        and runs_draft_prompt_pass(disagg_mode)
        and not is_dp_attention_enabled()
    )


def retention_for_cache(
    tree_cache: BasePrefixCache, *, draft_window: int, disagg_mode: str
) -> DraftKvRetention:
    page_size = tree_cache.page_size
    if type(tree_cache) is UnifiedRadixCache:
        cache_window = tree_cache.sliding_window_size
        if (
            cache_window is None
            or tree_cache.supports_mamba()
            # A HiCache draft sidecar backs draft KV up by its own anchoring.
            or tree_cache.sidecar_pool_specs
            or envs.SGLANG_OPT_SWA_KEEP_PREFILL_CHUNK_WINDOWS.get()
            or not envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get()
        ):
            return DraftKvRetention(
                publishing=CachePublishing.EVERY_ROW,
                window=draft_window,
                page_size=page_size,
            )
        return DraftKvRetention(
            publishing=(
                CachePublishing.PREFILL_END
                if tree_cache.disable
                else CachePublishing.UNIFIED_RADIX
            ),
            window=max(draft_window, cache_window),
            page_size=page_size,
            bigram_shift=int(tree_cache.tree_core.is_eagle),
            chunk_insert_stride=envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.get(),
            publishes_every_chunk=disagg_mode == "prefill",
        )
    # Chunk caches never share a prefix (their subclasses only differ in freeing).
    publishing = (
        CachePublishing.PREFILL_END
        if isinstance(tree_cache, ChunkCache)
        else CachePublishing.EVERY_ROW
    )
    return DraftKvRetention(
        publishing=publishing, window=draft_window, page_size=page_size
    )


def extend_request(req: Req, *, prefix_len: int, extend_len: int) -> ExtendRequest:
    return ExtendRequest(
        prefix_len=prefix_len,
        extend_len=extend_len,
        # A mixed-chunk decode row runs one past full_untruncated_fill_ids.
        prefill_end=max(len(req.full_untruncated_fill_ids), prefix_len + extend_len),
        swa_branch=req.swa_branching_seqlen,
        publishes=not req.skip_radix_cache_insert,
        keeps_every_row=req.session_id is not None,
    )


def batch_extend_requests(batch: ScheduleBatch) -> list[ExtendRequest]:
    return [
        extend_request(req, prefix_len=prefix_len, extend_len=extend_len)
        for req, prefix_len, extend_len in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
        )
    ]


def _chunk_publishes(req: ExtendRequest, retention: DraftKvRetention) -> bool:
    """Whether checkpoint_kv_cache (mem_cache/common.py) inserts up to this chunk's end."""
    stride = retention.chunk_insert_stride
    start, end = req.prefix_len, req.prefix_len + req.extend_len
    if retention.publishes_every_chunk or stride <= 0:
        return True
    branch = req.swa_branch
    return end // stride != start // stride or (branch is not None and branch <= end)


def retention_points(req: ExtendRequest, retention: DraftKvRetention) -> list[int]:
    """Positions whose preceding window may be read once this extend is done."""
    end = req.prefix_len + req.extend_len
    points = [req.prefill_end]
    if retention.publishing is not CachePublishing.UNIFIED_RADIX or not req.publishes:
        return points
    if req.swa_branch is not None and req.swa_branch > req.prefix_len:
        points.append(req.swa_branch)
    if end >= req.prefill_end:
        return points
    if _chunk_publishes(req, retention):
        points.append(end)
    stride = retention.chunk_insert_stride
    if stride > 0 and not retention.publishes_every_chunk:
        # A later chunk crossing the next stride multiple publishes an end at or past
        # it; that window can reach back into this chunk.
        next_multiple = (end // stride + 1) * stride
        if next_multiple < req.prefill_end:
            points.append(next_multiple)
    return points


def retained_position_ranges(
    req: ExtendRequest, retention: DraftKvRetention
) -> list[tuple[int, int]]:
    """Sorted disjoint [lo, hi) positions of this extend whose draft KV can be read."""
    start, end = req.prefix_len, req.prefix_len + req.extend_len
    if retention.publishing is CachePublishing.EVERY_ROW or req.keeps_every_row:
        return [(start, end)] if end > start else []
    ranges = sorted(
        (max(start, retention.window_floor(point)), min(end, point))
        for point in retention_points(req, retention)
    )
    merged: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def retained_extend_rows(
    reqs: list[ExtendRequest], retention: DraftKvRetention
) -> torch.Tensor:
    """CPU int64 indices into the extend's flattened rows to write, ascending."""
    pieces: list[torch.Tensor] = []
    row_offset = 0
    for req in reqs:
        for lo, hi in retained_position_ranges(req, retention):
            first = row_offset + lo - req.prefix_len
            pieces.append(torch.arange(first, first + hi - lo, dtype=torch.int64))
        row_offset += req.extend_len
    if not pieces:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(pieces)
