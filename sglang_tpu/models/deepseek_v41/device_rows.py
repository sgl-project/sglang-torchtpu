# SPDX-License-Identifier: Apache-2.0
"""Row metadata of a spec draft / verify forward from its device positions.

A spec step's rows sit at the device sequence lengths, which the host would only
learn by reading them back, and that read waits for the previous verify. The
fields that depend on the positions are elementwise on the positions and cache
slots, so they are built here with device ops; whatever depends only on the
batch layout (row counts, slots) still comes from the host. Every function gives
the same values as its host counterpart in attention_backend / request_window.
"""

from functools import lru_cache
from typing import Tuple

import torch

from sglang_tpu.models.deepseek_v41.request_window import (
    RESERVED_PAGES,
    RING_PAGES,
    SWA_PAGE_TOKENS,
    WINDOW_TOKENS,
    WindowPlan,
    ring_rows,
)


def token_rows(
    *,
    host_rows: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """attention_backend._token_rows with the cache slots on the device.

    ``host_rows``: [T] batch row of each request-contiguous row (-1 past the
    counts) and ``req_pool_indices``: [B], both already on the slots' device.
    """
    valid = (host_rows >= 0) & (out_cache_loc > 0)
    rows = host_rows.masked_fill(~valid, -1)
    req = req_pool_indices[rows.clamp_min(0)].masked_fill(~valid, 0)
    return rows, valid, req


def low_ratio_rows(
    *,
    ratio: int,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    valid: torch.Tensor,
    compress_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """attention_backend._low_ratio_rows in the inputs' dtype (int32 on TPU)."""
    seq = positions + 1
    completes = (seq % ratio == 0) & compress_valid
    out_loc = torch.where(completes, out_cache_loc // ratio, -1)
    return out_loc, torch.where(valid, seq // ratio, 0)


def window_live(slots: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """[B] bool on the host: the request moves window rows (extend_window_pages'
    rule: it has rows, a real slot, and is its slot's first request)."""
    first_of_slot = torch.ones_like(slots, dtype=torch.bool)
    if slots.numel() > 1:
        order = torch.argsort(slots, stable=True)
        sorted_slots = slots[order]
        repeat = torch.zeros_like(first_of_slot)
        repeat[1:] = sorted_slots[1:] == sorted_slots[:-1]
        first_of_slot[order] = ~repeat
    return (counts > 0) & (slots != 0) & first_of_slot


def window_pages(
    *,
    first: torch.Tensor,
    counts: torch.Tensor,
    live: torch.Tensor,
    slots: torch.Tensor,
    minimum: torch.Tensor,
    columns: int,
    base_page: int,
) -> Tuple[torch.Tensor, WindowPlan]:
    """request_window.extend_window_pages with ``first`` on the device.

    ``counts``, ``slots``, ``minimum`` ([B]) and ``live`` ([B] bool: the slot's
    request moves rows) come from the host layout, uploaded to ``first``'s device.
    The workspace capacity and SWA-table bound checks of the host function are
    the caller's: spec rows add at most a few pages per request.
    """
    lane = torch.arange(WINDOW_TOKENS, dtype=first.dtype, device=first.device)[None, :]
    pages = _workspace_pages(
        first=first, counts=counts, columns=columns, base_page=base_page
    )
    live = live[:, None]
    slot = slots[:, None]

    history = first[:, None] - WINDOW_TOKENS + lane
    history_ok = live & (history >= 0) & (history >= minimum[:, None])
    safe = history.clamp(min=0)
    page = pages.gather(1, (safe // SWA_PAGE_TOKENS).clamp(max=columns - 1))
    history_dst = torch.where(
        history_ok, page * SWA_PAGE_TOKENS + safe % SWA_PAGE_TOKENS, -1
    )
    history_src = torch.where(history_ok, ring_rows(slot, safe), 0)

    kept = (first + counts)[:, None] - WINDOW_TOKENS + lane
    kept_ok = live & (kept >= first[:, None])
    # cumsum would promote int32 to int64, which the TPU vector units do not lower.
    row_start = (torch.cumsum(counts, 0, dtype=counts.dtype) - counts)[:, None]
    commit_token = torch.where(kept_ok, row_start + kept - first[:, None], 0)
    commit_dst = torch.where(kept_ok, ring_rows(slot, kept.clamp(min=0)), -1)

    # All int32: casting to int64 on the device does not lower on TPU, so with
    # device rows the host plan is uploaded as int32 too (window_plan_upload).
    # Fresh 1-D tensors, not views of the [B, 128] results: TorchTPU compiles a
    # view input as its base buffer, unlike the host plan's uploads.
    plan = WindowPlan(
        *(
            t.to(torch.int32).reshape(-1).clone()
            for t in (history_src, history_dst, commit_token, commit_dst)
        )
    )
    return pages, plan


def _workspace_pages(
    *, first: torch.Tensor, counts: torch.Tensor, columns: int, base_page: int
) -> torch.Tensor:
    """request_window._assign_workspace_pages without the per-request loop: each
    request with rows takes the next fresh pages over [first - 128, first + count)."""
    has_rows = counts > 0
    low = (first - WINDOW_TOKENS).clamp(min=0) // SWA_PAGE_TOKENS
    high = (first + counts - 1) // SWA_PAGE_TOKENS
    width = torch.where(has_rows, high - low + 1, 0)
    offset = torch.cumsum(width, 0, dtype=width.dtype) - width
    col = torch.arange(columns, dtype=first.dtype, device=first.device)[None, :]
    inside = has_rows[:, None] & (col >= low[:, None]) & (col <= high[:, None])
    return torch.where(inside, base_page + offset[:, None] + col - low[:, None], 0)


# --------------------------------------------------------------------------- #
# Single-program JAX export of _device_row_fields on TPU.
# TorchTPU's DEFER_AND_FUSE does not fuse masked_fill/index/arange ops, so the
# eager path emits ~25 separate executables per decode step. Exporting the
# whole metadata build as one cpu_export_jax_op collapses it to 1 execution.
# --------------------------------------------------------------------------- #
def _ring_rows_jax(slots, positions):
    page = (
        RESERVED_PAGES
        + slots * RING_PAGES
        + (positions // SWA_PAGE_TOKENS) % RING_PAGES
    )
    return page * SWA_PAGE_TOKENS + positions % SWA_PAGE_TOKENS


@lru_cache(maxsize=64)
def _get_fused_fields_op(
    window: bool, columns: int, base_page: int, ratios: Tuple[int, ...]
):
    import jax
    import jax.numpy as jnp

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    rtag = "_".join(str(r) for r in ratios) or "none"

    if window:
        def _fn(
            positions: jax.Array,
            out_cache_loc: jax.Array,
            host_rows: jax.Array,
            req_pool_indices: jax.Array,
            counts: jax.Array,
            live: jax.Array,
            first_row: jax.Array,
            swa_minimum: jax.Array,
        ):
            pos = positions.astype(jnp.int32)
            loc = out_cache_loc.astype(jnp.int32)
            hrows = host_rows.astype(jnp.int32)
            slots = req_pool_indices.astype(jnp.int32)
            cnt = counts.astype(jnp.int32)
            lv = live.astype(jnp.bool_)
            frow = first_row.astype(jnp.int32)
            smin = swa_minimum.astype(jnp.int32)

            valid = (hrows >= 0) & (loc > 0)
            rows = jnp.where(valid, hrows, jnp.int32(-1))
            req = jnp.where(valid, slots[jnp.maximum(rows, jnp.int32(0))], jnp.int32(0))

            first = pos[frow]
            has_rows = cnt > 0
            low = jnp.maximum(first - WINDOW_TOKENS, jnp.int32(0)) // SWA_PAGE_TOKENS
            high = (first + cnt - 1) // SWA_PAGE_TOKENS
            width = jnp.where(has_rows, high - low + 1, jnp.int32(0))
            offset = jnp.cumsum(width, axis=0) - width
            col = jnp.arange(columns, dtype=jnp.int32)[None, :]
            inside = has_rows[:, None] & (col >= low[:, None]) & (col <= high[:, None])
            pages = jnp.where(
                inside, base_page + offset[:, None] + col - low[:, None], jnp.int32(0)
            ).astype(jnp.int32)

            lane = jnp.arange(WINDOW_TOKENS, dtype=jnp.int32)[None, :]
            lv2 = lv[:, None]
            slot2 = slots[:, None]

            history = first[:, None] - WINDOW_TOKENS + lane
            history_ok = lv2 & (history >= 0) & (history >= smin[:, None])
            safe = jnp.maximum(history, jnp.int32(0))
            page_idx = jnp.minimum(safe // SWA_PAGE_TOKENS, columns - 1)
            page = jnp.take_along_axis(pages, page_idx, axis=1)
            history_dst = jnp.where(
                history_ok, page * SWA_PAGE_TOKENS + safe % SWA_PAGE_TOKENS, jnp.int32(-1)
            ).astype(jnp.int32).reshape(-1)
            history_src = jnp.where(
                history_ok, _ring_rows_jax(slot2, safe), jnp.int32(0)
            ).astype(jnp.int32).reshape(-1)

            kept = (first + cnt)[:, None] - WINDOW_TOKENS + lane
            kept_ok = lv2 & (kept >= first[:, None])
            row_start = (jnp.cumsum(cnt, axis=0) - cnt)[:, None]
            commit_token = jnp.where(
                kept_ok, row_start + kept - first[:, None], jnp.int32(0)
            ).astype(jnp.int32).reshape(-1)
            commit_dst = jnp.where(
                kept_ok, _ring_rows_jax(slot2, jnp.maximum(kept, jnp.int32(0))), jnp.int32(-1)
            ).astype(jnp.int32).reshape(-1)

            outs = [
                rows, valid, req,
                pages, history_src, history_dst, commit_token, commit_dst,
            ]
            seq = pos + 1
            for ratio in ratios:
                completes = ((seq % ratio) == 0) & valid
                outs.append(jnp.where(completes, loc // ratio, jnp.int32(-1)).astype(jnp.int32))
                outs.append(jnp.where(valid, seq // ratio, jnp.int32(0)).astype(jnp.int32))
            return tuple(outs)

        name = f"sglang_tpu_pallas::dsv41_dev_rows_w1_c{columns}_b{base_page}_r{rtag}"
        _fn.__annotations__["return"] = tuple[(jax.Array,) * (8 + 2 * len(ratios))]
        op = cpu_export_jax_op(name, _fn)

        def _fake(positions, out_cache_loc, host_rows, req_pool_indices, counts, live, first_row, swa_minimum):
            T = positions.shape[0]
            B = req_pool_indices.shape[0]
            res = [
                positions.new_empty((T,), dtype=torch.int32),
                positions.new_empty((T,), dtype=torch.bool),
                positions.new_empty((T,), dtype=torch.int32),
                positions.new_empty((B, columns), dtype=torch.int32),
                positions.new_empty((B * WINDOW_TOKENS,), dtype=torch.int32),
                positions.new_empty((B * WINDOW_TOKENS,), dtype=torch.int32),
                positions.new_empty((B * WINDOW_TOKENS,), dtype=torch.int32),
                positions.new_empty((B * WINDOW_TOKENS,), dtype=torch.int32),
            ]
            for _ in ratios:
                res.append(positions.new_empty((T,), dtype=torch.int32))
                res.append(positions.new_empty((T,), dtype=torch.int32))
            return tuple(res)

        op.register_fake(_fake)
        return op

    def _fn_nowin(
        positions: jax.Array,
        out_cache_loc: jax.Array,
        host_rows: jax.Array,
        req_pool_indices: jax.Array,
    ):
        pos = positions.astype(jnp.int32)
        loc = out_cache_loc.astype(jnp.int32)
        hrows = host_rows.astype(jnp.int32)
        slots = req_pool_indices.astype(jnp.int32)

        valid = (hrows >= 0) & (loc > 0)
        rows = jnp.where(valid, hrows, jnp.int32(-1))
        req = jnp.where(valid, slots[jnp.maximum(rows, jnp.int32(0))], jnp.int32(0))

        outs = [rows, valid, req]
        seq = pos + 1
        for ratio in ratios:
            completes = ((seq % ratio) == 0) & valid
            outs.append(jnp.where(completes, loc // ratio, jnp.int32(-1)).astype(jnp.int32))
            outs.append(jnp.where(valid, seq // ratio, jnp.int32(0)).astype(jnp.int32))
        return tuple(outs)

    name = f"sglang_tpu_pallas::dsv41_dev_rows_w0_r{rtag}"
    _fn_nowin.__annotations__["return"] = tuple[(jax.Array,) * (3 + 2 * len(ratios))]
    op = cpu_export_jax_op(name, _fn_nowin)

    def _fake_nowin(positions, out_cache_loc, host_rows, req_pool_indices):
        T = positions.shape[0]
        res = [
            positions.new_empty((T,), dtype=torch.int32),
            positions.new_empty((T,), dtype=torch.bool),
            positions.new_empty((T,), dtype=torch.int32),
        ]
        for _ in ratios:
            res.append(positions.new_empty((T,), dtype=torch.int32))
            res.append(positions.new_empty((T,), dtype=torch.int32))
        return tuple(res)

    op.register_fake(_fake_nowin)
    return op


def fused_device_row_fields(
    *,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    host_rows: torch.Tensor,
    req_pool_indices: torch.Tensor,
    window: bool,
    counts: torch.Tensor | None,
    live: torch.Tensor | None,
    first_row: torch.Tensor | None,
    swa_minimum: torch.Tensor | None,
    columns: int,
    base_page: int,
    ratios: Tuple[int, ...],
):
    """Run all of attention_backend._device_row_fields as one compiled TPU op."""
    op = _get_fused_fields_op(window, int(columns), int(base_page), tuple(ratios))
    pos_i32 = positions.to(torch.int32).contiguous()
    loc_i32 = out_cache_loc.to(torch.int32).contiguous()
    hr_i32 = host_rows.to(torch.int32).contiguous()
    rpi_i32 = req_pool_indices.to(torch.int32).contiguous()
    if window:
        return op(
            pos_i32,
            loc_i32,
            hr_i32,
            rpi_i32,
            counts.to(torch.int32).contiguous(),
            live.to(torch.bool).contiguous(),
            first_row.to(torch.int32).contiguous(),
            swa_minimum.to(torch.int32).contiguous(),
        )
    return op(pos_i32, loc_i32, hr_i32, rpi_i32)
