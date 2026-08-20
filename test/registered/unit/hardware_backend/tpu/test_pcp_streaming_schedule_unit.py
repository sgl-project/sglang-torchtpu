# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Host-only unit tests for the PCP streaming RPA schedule stack.

Three layers are exercised, none of which needs a TPU device:

1. :func:`build_pcp_streaming_schedule_inputs_from_metadata_jax` — the compact
   per-tile plan (``TilePlanField``) that the production Pallas kernels
   prefetch from HBM, plus the compact current/history tile indices.
2. :func:`build_runtime_schedule_reference` — the pure-NumPy mirror of the
   per-ring-step rows (``RuntimeScheduleField``) that the kernel stages into
   VMEM out of that tile plan.
3. The ``pcp_layout`` host helpers, which pack query rows rank-major before
   they are moved to the device and therefore have to agree row-for-row with
   the tile plan's ``Q_HBM_OFFSET`` / ``Q_TILE_SIZE`` fields.

The Pallas kernel itself is not run here; kernel-vs-reference parity needs a
TPU host and lives under ``test/manual/tpu/``. The JAX ops used by the
schedule builder are tiny elementwise/gather ops that run fine on the CPU
backend, so this file stays in the CPU CI suite.

Usage::

    cd /path/to/sglang-torchtpu
    python -m unittest \
        test.registered.unit.hardware_backend.tpu.test_pcp_streaming_schedule_unit
"""

from __future__ import annotations

import unittest

import numpy as np

from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.pcp_layout import (
    build_pcp_logits_indices,
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
)
from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.reference import (
    build_runtime_schedule_reference,
)
from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField,
    TilePlanField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _cu_q_lens(q_lens: list[int]) -> np.ndarray:
    """Cumulative query offsets — also the batch-flat owner coordinate."""
    return np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)


def _build_schedule_inputs(
    *,
    kv_lens: list[int],
    q_lens: list[int],
    pcp_size: int,
    page_size: int,
    interleave_size: int,
    q_block_size: int,
    global_bucket_tokens: int,
    pages_per_seq: int | None = None,
    first_page_index: int = 0,
):
    """Build one production tile plan from synthetic attention metadata.

    ``page_indices`` is laid out so each request owns a disjoint, contiguous
    run of ``pages_per_seq`` local KV pages, which makes per-request block
    table lookups easy to assert on.
    """
    num_reqs = len(kv_lens)
    virtual_page_size = page_size * pcp_size
    if pages_per_seq is None:
        pages_per_seq = max(
            1,
            max((kv + virtual_page_size - 1) // virtual_page_size for kv in kv_lens),
        )
    page_indices = (
        np.arange(num_reqs * pages_per_seq, dtype=np.int32) + first_page_index
    )
    return build_pcp_streaming_schedule_inputs_from_metadata_jax(
        kv_lens=np.asarray(kv_lens, dtype=np.int32),
        page_indices=page_indices,
        cu_q_lens=_cu_q_lens(q_lens),
        distribution=np.asarray([0, 0, num_reqs], dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=int(page_indices.max()) + 1,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )


def _logical_plan(tile_plan, pcp_size: int) -> np.ndarray:
    """Unpack the HBM tile plan into ``[tile, consumer_rank, field]``."""
    tile_plan = np.asarray(tile_plan)
    logical_fields = pcp_size * TilePlanField.NUM_FIELDS
    return tile_plan[:, 0, :logical_fields].reshape(
        tile_plan.shape[0], pcp_size, TilePlanField.NUM_FIELDS
    )


def _active_tiles(logical: np.ndarray) -> np.ndarray:
    """Keep only tiles that carry query rows for at least one rank."""
    return logical[np.any(logical[..., TilePlanField.Q_TILE_SIZE] > 0, axis=1)]


def _rank_major_order_from_plan(
    logical: np.ndarray,
    *,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
) -> np.ndarray:
    """Rebuild the packed token order implied by the tile plan's Q fields.

    Row ``j`` of a rank's query tile sits ``(j // interleave_size)`` full
    interleave cycles plus ``(j % interleave_size)`` rows past the tile's
    request-absolute start, and lands at slot ``Q_HBM_OFFSET + j`` inside that
    rank's contiguous local query buffer. Translating the request-absolute
    coordinate back to the batch-flat one gives the packed order that the host
    is expected to have produced.
    """
    cycle = pcp_size * interleave_size
    local_padded_num_tokens = padded_num_tokens // pcp_size
    order = np.full(padded_num_tokens, -1, dtype=np.int64)
    for plan in logical:
        owner_start = int(plan[0, TilePlanField.TOKEN_OWNER_START])
        absolute_start = int(plan[0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START])
        for rank in range(pcp_size):
            tile_size = int(plan[rank, TilePlanField.Q_TILE_SIZE])
            if tile_size == 0:
                continue
            q_global_start = int(plan[rank, TilePlanField.Q_GLOBAL_START])
            hbm_offset = int(plan[rank, TilePlanField.Q_HBM_OFFSET])
            for row in range(tile_size):
                position = (
                    owner_start
                    + (q_global_start - absolute_start)
                    + (row // interleave_size) * cycle
                    + (row % interleave_size)
                )
                slot = rank * local_padded_num_tokens + hbm_offset + row
                if order[slot] != -1:
                    raise AssertionError(f"tile plan writes slot {slot} twice")
                order[slot] = position
    return order


def _build_reference(**kwargs):
    """Expand a freshly built tile plan into the staged runtime rows."""
    tile_plan, block_tables, *_ = _build_schedule_inputs(**kwargs)
    return build_runtime_schedule_reference(
        tile_plan,
        block_tables,
        pcp_size=kwargs["pcp_size"],
        page_size=kwargs["page_size"],
        interleave_size=kwargs["interleave_size"],
    )


def _first_active_tile(num_groups: np.ndarray) -> int:
    return int(np.flatnonzero(np.asarray(num_groups) > 0)[0])


def _active_rows(rows: np.ndarray) -> np.ndarray:
    return rows[..., RuntimeScheduleField.REQ_ID] >= 0


class BatchFlatOwnershipTest(CustomTestCase):
    """Query ownership follows the batch-flat ``cu_q_lens`` coordinate."""

    def test_owner_coordinate_is_batch_flat_not_request_absolute(self) -> None:
        # Two requests, 4 new query rows each, with histories of 10 and 17
        # tokens. pcp_size=2 / interleave_size=2 gives a 4-token cycle.
        #
        # Batch-flat: request 0 owns owner coordinates [0, 4) -> rank 0 takes
        # [0, 2) and rank 1 takes [2, 4), i.e. request-absolute positions 10
        # and 12. Request 1 owns [4, 8) -> the same split at 17 and 19.
        #
        # Request-absolute ownership would instead stripe request 0 from
        # position 10, which is 2 past a cycle boundary, so rank 1 would take
        # positions 10-11 and rank 0 would take 12-13, i.e. [12, 10].
        #
        # Batch-flat is the load-balancing choice: the interleave stripe runs
        # across the concatenated batch exactly once, so the per-rank row
        # counts differ by at most one interleave chunk for the whole forward.
        # Under request-absolute ownership every request re-phases the stripe
        # against its own history length, so the skew compounds per request
        # and ranks fall out of lockstep on their query tiles.
        tile_plan, *_ = _build_schedule_inputs(
            kv_lens=[14, 21],
            q_lens=[4, 4],
            pcp_size=2,
            page_size=4,
            interleave_size=2,
            q_block_size=2,
            global_bucket_tokens=8,
        )
        logical = _active_tiles(_logical_plan(tile_plan, pcp_size=2))

        np.testing.assert_array_equal(logical[:, 0, TilePlanField.REQ_ID], [0, 1])
        np.testing.assert_array_equal(
            logical[..., TilePlanField.Q_GLOBAL_START],
            np.array([[10, 12], [17, 19]], dtype=np.int32),
        )
        # Causality and the KV cache stay request-absolute even though
        # ownership does not.
        np.testing.assert_array_equal(
            logical[..., TilePlanField.REQUEST_ABSOLUTE_QUERY_START],
            np.array([[10, 10], [17, 17]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            logical[..., TilePlanField.TOKEN_OWNER_START],
            np.array([[0, 0], [4, 4]], dtype=np.int32),
        )
        np.testing.assert_array_equal(logical[..., TilePlanField.Q_TILE_SIZE], 2)
        self.assertFalse(
            np.array_equal(
                logical[0, :, TilePlanField.Q_GLOBAL_START],
                np.array([12, 10], dtype=np.int32),
            ),
            msg="request-absolute ownership leaked into the tile plan",
        )

    def test_batch_flat_owner_keeps_short_requests_balanced(self) -> None:
        # Four 2-token continuations at pcp_size=2 / interleave_size=2. Their
        # histories all land on a cycle boundary, so request-absolute
        # ownership would hand every single row to rank 0. The batch-flat
        # coordinate stripes the concatenated batch instead and splits evenly.
        q_lens = [2, 2, 2, 2]
        batch_flat = pcp_local_token_counts(
            q_lens, 2, 2, _cu_q_lens(q_lens)[:-1].tolist()
        )
        request_absolute = pcp_local_token_counts(q_lens, 2, 2, [8, 12, 16, 20])
        np.testing.assert_array_equal(batch_flat, np.array([4, 4], dtype=np.int32))
        np.testing.assert_array_equal(
            request_absolute, np.array([8, 0], dtype=np.int32)
        )

    def test_long_history_keeps_request_absolute_query_positions(self) -> None:
        # Batch-flat ownership must not truncate or wrap the request-absolute
        # positions used for causal masking on very long contexts.
        pcp_size = 4
        page_size = 128
        interleave_size = 32
        q_len = 256
        for history_tokens in (25_600, 128 * 1024):
            with self.subTest(history_tokens=history_tokens):
                tile_plan, *_ = _build_schedule_inputs(
                    kv_lens=[history_tokens + q_len],
                    q_lens=[q_len],
                    pcp_size=pcp_size,
                    page_size=page_size,
                    interleave_size=interleave_size,
                    q_block_size=64,
                    global_bucket_tokens=q_len,
                )
                logical = _active_tiles(_logical_plan(tile_plan, pcp_size=pcp_size))

                self.assertEqual(logical.shape[0], 1)
                np.testing.assert_array_equal(
                    logical[0, :, TilePlanField.Q_GLOBAL_START],
                    history_tokens + np.arange(pcp_size) * interleave_size,
                )
                np.testing.assert_array_equal(
                    logical[0, :, TilePlanField.Q_TILE_SIZE],
                    np.full(pcp_size, 64, dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    logical[..., TilePlanField.REQUEST_ABSOLUTE_QUERY_START],
                    history_tokens,
                )
                self.assertTrue(
                    np.all(logical[..., TilePlanField.HISTORY_NUM_GROUPS] > 0)
                )


class TilePlanStructureTest(CustomTestCase):
    """Compact tile indices and current/history group accounting."""

    def test_compact_indices_track_group_starts(self) -> None:
        (
            tile_plan,
            block_tables,
            current_ids,
            current_starts,
            current_meta,
            history_ids,
            history_starts,
            history_meta,
        ) = _build_schedule_inputs(
            kv_lens=[23, 8],
            q_lens=[8, 0],
            pcp_size=2,
            page_size=4,
            interleave_size=2,
            q_block_size=4,
            global_bucket_tokens=8,
            pages_per_seq=16,
        )
        logical = _logical_plan(tile_plan, pcp_size=2)
        self.assertEqual(
            np.asarray(tile_plan).shape, (6, 1, TilePlanField.HBM_ROW_FIELDS)
        )
        self.assertEqual(np.asarray(block_tables).shape, (2, 16))

        for ids, starts, meta, field in (
            (
                current_ids,
                current_starts,
                current_meta,
                TilePlanField.CURRENT_NUM_GROUPS,
            ),
            (
                history_ids,
                history_starts,
                history_meta,
                TilePlanField.HISTORY_NUM_GROUPS,
            ),
        ):
            groups = logical[:, 0, field]
            expected_ids = np.flatnonzero(groups > 0)
            expected_starts = np.cumsum(groups[expected_ids]) - groups[expected_ids]
            count = int(np.asarray(meta)[0])
            np.testing.assert_array_equal(np.asarray(ids)[0, :count], expected_ids)
            np.testing.assert_array_equal(
                np.asarray(starts)[0, :count], expected_starts
            )
            self.assertEqual(int(np.asarray(meta)[1]), int(groups.sum()))

    def test_partial_history_page_moves_into_the_current_pass(self) -> None:
        pcp_size = 8
        history_tokens = 25_600
        current_tokens = 32_768
        (
            tile_plan,
            _block_tables,
            _current_ids,
            _current_starts,
            current_meta,
            _history_ids,
            _history_starts,
            history_meta,
        ) = _build_schedule_inputs(
            kv_lens=[history_tokens + current_tokens],
            q_lens=[current_tokens],
            pcp_size=pcp_size,
            page_size=256,
            interleave_size=256,
            q_block_size=512,
            global_bucket_tokens=current_tokens,
            pages_per_seq=136,
        )
        logical = _active_tiles(_logical_plan(tile_plan, pcp_size=pcp_size))

        # 25,600 tokens are 12 complete PCP virtual pages (12 * 8 * 256) plus a
        # half-full boundary page. Only the complete pages go through the
        # unmasked history pass; the boundary page becomes the first causal
        # group of every active current tile, which is why CURRENT_NUM_GROUPS
        # starts at 3 rather than 2 and grows by 2 per 512-row query tile.
        np.testing.assert_array_equal(
            logical[:, 0, TilePlanField.HISTORY_NUM_GROUPS],
            np.full(8, 12, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            logical[:, 0, TilePlanField.CURRENT_NUM_GROUPS],
            np.arange(3, 18, 2, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(current_meta), np.array([8, 80], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(history_meta), np.array([8, 96], dtype=np.int32)
        )

    def test_rejects_nonpositive_interleave_size(self) -> None:
        for interleave_size in (0, -2):
            with self.subTest(interleave_size=interleave_size):
                with self.assertRaisesRegex(
                    ValueError, "interleave_size must be positive"
                ):
                    _build_schedule_inputs(
                        kv_lens=[4],
                        q_lens=[4],
                        pcp_size=1,
                        page_size=4,
                        interleave_size=interleave_size,
                        q_block_size=4,
                        global_bucket_tokens=4,
                    )

    def test_rejects_bucket_that_breaks_pcp_lockstep(self) -> None:
        # 12 is divisible by pcp_size=2 but not by pcp_size * q_block_size = 8,
        # so the ranks could not walk their query tiles in lockstep.
        with self.assertRaisesRegex(ValueError, r"pcp_size \* q_block_size"):
            _build_schedule_inputs(
                kv_lens=[12],
                q_lens=[12],
                pcp_size=2,
                page_size=4,
                interleave_size=2,
                q_block_size=4,
                global_bucket_tokens=12,
            )


class TilePlanBatchShapeTest(CustomTestCase):
    """Batch shapes the serving path presents, across pcp_size 2/4/8.

    Every case asserts the strong invariant that ties the device-side tile
    plan to the host-side packing: replaying the plan's Q fields must
    reproduce, slot for slot, the rank-major token order that
    :func:`build_pcp_rank_major_token_order` hands the host packer. A
    disagreement means the kernel would read a rank's query rows from the
    wrong HBM offsets.
    """

    def _assert_plan_matches_host_packing(
        self,
        *,
        kv_lens: list[int],
        q_lens: list[int],
        pcp_size: int,
        page_size: int,
        interleave_size: int,
        q_block_size: int,
        global_bucket_tokens: int,
    ) -> np.ndarray:
        tile_plan, *_ = _build_schedule_inputs(
            kv_lens=kv_lens,
            q_lens=q_lens,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
            global_bucket_tokens=global_bucket_tokens,
        )
        logical = _logical_plan(tile_plan, pcp_size=pcp_size)
        owner_starts = _cu_q_lens(q_lens)[:-1].tolist()
        token_order, _inverse_order = build_pcp_rank_major_token_order(
            q_lens,
            pcp_size,
            interleave_size,
            global_bucket_tokens,
            owner_starts,
        )
        np.testing.assert_array_equal(
            _rank_major_order_from_plan(
                logical,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                padded_num_tokens=global_bucket_tokens,
            ),
            token_order,
        )
        active = _active_tiles(logical)
        # Each rank's total scheduled rows match the host's per-rank counts.
        np.testing.assert_array_equal(
            active[..., TilePlanField.Q_TILE_SIZE].sum(axis=0),
            pcp_local_token_counts(q_lens, pcp_size, interleave_size, owner_starts),
        )
        return active

    def test_single_request_full_prefill_pcp4(self) -> None:
        active = self._assert_plan_matches_host_packing(
            kv_lens=[16],
            q_lens=[16],
            pcp_size=4,
            page_size=4,
            interleave_size=2,
            q_block_size=2,
            global_bucket_tokens=16,
        )
        np.testing.assert_array_equal(active[:, 0, TilePlanField.REQ_ID], 0)
        # A fresh prefill has no complete history page to stream.
        np.testing.assert_array_equal(active[..., TilePlanField.HISTORY_NUM_GROUPS], 0)

    def test_two_requests_uneven_q_lens_pcp4(self) -> None:
        active = self._assert_plan_matches_host_packing(
            kv_lens=[24, 8],
            q_lens=[16, 8],
            pcp_size=4,
            page_size=4,
            interleave_size=2,
            q_block_size=4,
            global_bucket_tokens=32,
        )
        np.testing.assert_array_equal(
            np.unique(active[:, 0, TilePlanField.REQ_ID]), [0, 1]
        )
        # Request 0 continues from 8 cached tokens; request 1 starts fresh.
        np.testing.assert_array_equal(
            np.unique(active[..., TilePlanField.REQUEST_ABSOLUTE_QUERY_START]),
            [0, 8],
        )

    def test_three_requests_mixed_chunked_prefill_pcp2(self) -> None:
        active = self._assert_plan_matches_host_packing(
            kv_lens=[16, 8, 8],
            q_lens=[8, 8, 8],
            pcp_size=2,
            page_size=4,
            interleave_size=2,
            q_block_size=2,
            global_bucket_tokens=24,
        )
        np.testing.assert_array_equal(
            np.unique(active[:, 0, TilePlanField.REQ_ID]), [0, 1, 2]
        )
        # Only the continuation request has a complete virtual history page.
        req_ids = active[:, 0, TilePlanField.REQ_ID]
        history_groups = active[:, 0, TilePlanField.HISTORY_NUM_GROUPS]
        np.testing.assert_array_equal(history_groups[req_ids == 0], 1)
        np.testing.assert_array_equal(history_groups[req_ids != 0], 0)

    def test_decode_shaped_batch_pcp8(self) -> None:
        # Eight one-token continuations. With interleave_size=1 the batch-flat
        # coordinate hands exactly one row to each rank.
        q_lens = [1] * 8
        active = self._assert_plan_matches_host_packing(
            kv_lens=[100, 200, 300, 400, 500, 600, 700, 800],
            q_lens=q_lens,
            pcp_size=8,
            page_size=32,
            interleave_size=1,
            q_block_size=8,
            global_bucket_tokens=64,
        )
        np.testing.assert_array_equal(
            np.unique(active[:, 0, TilePlanField.REQ_ID]), np.arange(8)
        )
        np.testing.assert_array_equal(
            active[..., TilePlanField.Q_TILE_SIZE].sum(axis=0),
            np.ones(8, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            build_pcp_logits_indices(
                q_lens, 8, 1, 64, 0, _cu_q_lens(q_lens)[:-1].tolist()
            ),
            np.arange(8) * 8,
        )

    def test_single_request_full_prefill_pcp8(self) -> None:
        active = self._assert_plan_matches_host_packing(
            kv_lens=[64],
            q_lens=[64],
            pcp_size=8,
            page_size=8,
            interleave_size=8,
            q_block_size=8,
            global_bucket_tokens=64,
        )
        # One 64-row query tile spread as 8 rows per rank.
        self.assertEqual(active.shape[0], 1)
        np.testing.assert_array_equal(active[0, :, TilePlanField.Q_TILE_SIZE], 8)


class RuntimeScheduleReferenceTest(CustomTestCase):
    """Rows staged per ring step, expanded from the compact tile plan."""

    def test_history_boundary_group_is_separated_from_fresh_kv(self) -> None:
        reference = _build_reference(
            kv_lens=[19],
            q_lens=[8],
            pcp_size=2,
            page_size=4,
            interleave_size=2,
            q_block_size=4,
            global_bucket_tokens=8,
            pages_per_seq=8,
            first_page_index=5,
        )
        tile = _first_active_tile(reference.current_num_groups)
        current = reference.current_rows[tile]
        history = reference.history_rows[tile]
        # 11 cached tokens = one complete virtual page (8) plus a partial page,
        # so the current pass runs one boundary group plus two fresh groups.
        self.assertEqual(int(reference.current_num_groups[tile]), 3)
        self.assertEqual(int(reference.history_num_groups[tile]), 1)

        boundary = current[0]
        # A negative KV_HBM_OFFSET selects the paged boundary page instead of
        # the compact fresh-KV buffer, and nothing is captured from it.
        self.assertTrue(
            np.all(
                boundary[..., RuntimeScheduleField.KV_HBM_OFFSET][
                    _active_rows(boundary)
                ]
                == -1
            )
        )
        np.testing.assert_array_equal(
            boundary[..., RuntimeScheduleField.CAPTURE_LEN], 0
        )
        np.testing.assert_array_equal(
            boundary[:, 0, RuntimeScheduleField.KV_PAGE_IDX],
            np.array([6, 6], dtype=np.int32),
        )

        fresh = current[1:3]
        self.assertTrue(
            np.all(
                fresh[..., RuntimeScheduleField.KV_HBM_OFFSET][_active_rows(fresh)] >= 0
            )
        )
        # Every fresh KV row is captured exactly once across consumer ranks.
        np.testing.assert_array_equal(
            fresh[..., RuntimeScheduleField.CAPTURE_LEN].sum(axis=-1),
            fresh[..., 0, RuntimeScheduleField.KV_VALID_LEN],
        )

        self.assertTrue(
            np.all(
                history[..., RuntimeScheduleField.KV_HBM_OFFSET][_active_rows(history)]
                == 0
            )
        )
        np.testing.assert_array_equal(
            history[0, :, 0, RuntimeScheduleField.KV_PAGE_IDX],
            np.array([5, 5], dtype=np.int32),
        )

    def test_single_token_capture_routes_to_the_absolute_cache_owner(self) -> None:
        # One decode row appended at absolute position 301. The cache stripe is
        # request-absolute, so 301 // 256 % 8 == 1 owns the writeback slot even
        # though batch-flat ownership put the query row on rank 0.
        pcp_size = 8
        interleave_size = 256
        reference = _build_reference(
            kv_lens=[302],
            q_lens=[1],
            pcp_size=pcp_size,
            page_size=interleave_size,
            interleave_size=interleave_size,
            q_block_size=interleave_size,
            global_bucket_tokens=pcp_size * interleave_size,
            pages_per_seq=1,
            first_page_index=7,
        )
        tile = _first_active_tile(reference.current_num_groups)
        self.assertEqual(int(reference.current_num_groups[tile]), 2)
        fresh_from_source_0 = reference.current_rows[tile, 1, 0]
        self.assertEqual(
            int(fresh_from_source_0[0, RuntimeScheduleField.KV_GLOBAL_START]), 301
        )
        self.assertEqual(
            int(fresh_from_source_0[0, RuntimeScheduleField.KV_VALID_LEN]), 1
        )
        self.assertEqual(
            int(fresh_from_source_0[1, RuntimeScheduleField.CAPTURE_SRC_OFFSET]), 0
        )
        self.assertEqual(
            int(fresh_from_source_0[1, RuntimeScheduleField.CAPTURE_DST_OFFSET]), 0
        )
        self.assertEqual(
            int(fresh_from_source_0[1, RuntimeScheduleField.CAPTURE_LEN]), 1
        )
        self.assertTrue(
            np.all(
                np.delete(fresh_from_source_0[..., RuntimeScheduleField.CAPTURE_LEN], 1)
                == 0
            )
        )

    def test_capture_splits_at_the_cache_owner_boundary(self) -> None:
        # 32 fresh rows starting at absolute position 301 straddle the cache
        # stripe boundary at 320: rows 0..18 belong to rank 1 and rows 19..31
        # to rank 0, so the capture is emitted as two consumer-local fragments.
        reference = _build_reference(
            kv_lens=[333],
            q_lens=[32],
            pcp_size=2,
            page_size=32,
            interleave_size=32,
            q_block_size=32,
            global_bucket_tokens=64,
            pages_per_seq=8,
        )
        tile = _first_active_tile(reference.current_num_groups)
        self.assertEqual(int(reference.current_num_groups[tile]), 2)
        fresh_from_source_0 = reference.current_rows[tile, 1, 0]
        self.assertEqual(
            int(fresh_from_source_0[0, RuntimeScheduleField.KV_GLOBAL_START]), 301
        )
        self.assertEqual(
            int(fresh_from_source_0[0, RuntimeScheduleField.KV_VALID_LEN]), 32
        )
        np.testing.assert_array_equal(
            fresh_from_source_0[:, RuntimeScheduleField.CAPTURE_SRC_OFFSET],
            np.array([19, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            fresh_from_source_0[:, RuntimeScheduleField.CAPTURE_DST_OFFSET],
            np.array([0, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            fresh_from_source_0[:, RuntimeScheduleField.CAPTURE_LEN],
            np.array([13, 19], dtype=np.int32),
        )

    def test_history_groups_cover_every_complete_virtual_page(self) -> None:
        pcp_size = 4
        page_size = 128
        q_len = 256
        for history_tokens in (25_600, 128 * 1024):
            with self.subTest(history_tokens=history_tokens):
                reference = _build_reference(
                    kv_lens=[history_tokens + q_len],
                    q_lens=[q_len],
                    pcp_size=pcp_size,
                    page_size=page_size,
                    interleave_size=32,
                    q_block_size=64,
                    global_bucket_tokens=q_len,
                )
                tile = _first_active_tile(reference.current_num_groups)
                self.assertEqual(
                    int(reference.history_num_groups[tile]),
                    history_tokens // (pcp_size * page_size),
                )
                current = reference.current_rows[tile]
                self.assertTrue(
                    np.all(
                        current[..., RuntimeScheduleField.Q_GLOBAL_START][
                            _active_rows(current)
                        ]
                        >= history_tokens
                    )
                )

    def test_multi_request_rows_read_their_own_block_table_row(self) -> None:
        pcp_size = 2
        reference = _build_reference(
            kv_lens=[16, 8, 8],
            q_lens=[8, 8, 8],
            pcp_size=pcp_size,
            page_size=4,
            interleave_size=2,
            q_block_size=2,
            global_bucket_tokens=24,
            pages_per_seq=4,
            first_page_index=100,
        )
        tile_plan, block_tables, *_ = _build_schedule_inputs(
            kv_lens=[16, 8, 8],
            q_lens=[8, 8, 8],
            pcp_size=pcp_size,
            page_size=4,
            interleave_size=2,
            q_block_size=2,
            global_bucket_tokens=24,
            pages_per_seq=4,
            first_page_index=100,
        )
        logical = _logical_plan(tile_plan, pcp_size=pcp_size)
        block_tables = np.asarray(block_tables)

        seen_req_ids = set()
        for tile in np.flatnonzero(np.asarray(reference.current_num_groups) > 0):
            req_id = int(logical[tile, 0, TilePlanField.REQ_ID])
            seen_req_ids.add(req_id)
            current = reference.current_rows[
                tile, : int(reference.current_num_groups[tile])
            ]
            np.testing.assert_array_equal(
                np.unique(current[..., RuntimeScheduleField.REQ_ID]), [req_id]
            )
            num_history = int(reference.history_num_groups[tile])
            history_pages = reference.history_rows[
                tile, :num_history, :, 0, RuntimeScheduleField.KV_PAGE_IDX
            ]
            np.testing.assert_array_equal(
                history_pages,
                np.broadcast_to(
                    block_tables[req_id, :num_history, None], history_pages.shape
                ),
            )
        self.assertEqual(seen_req_ids, {0, 1, 2})

    def test_rows_only_use_declared_abi_fields(self) -> None:
        reference = _build_reference(
            kv_lens=[19],
            q_lens=[8],
            pcp_size=2,
            page_size=4,
            interleave_size=2,
            q_block_size=4,
            global_bucket_tokens=8,
            pages_per_seq=8,
        )
        for rows in (reference.current_rows, reference.history_rows):
            self.assertEqual(rows.shape[-1], RuntimeScheduleField.PACKED_NUM_FIELDS)
            self.assertGreater(rows.shape[1], 0)
            np.testing.assert_array_equal(
                rows[..., RuntimeScheduleField.NUM_FIELDS :],
                np.zeros_like(rows[..., RuntimeScheduleField.NUM_FIELDS :]),
            )

    def test_rejects_page_size_indivisible_by_interleave_size(self) -> None:
        tile_plan, block_tables, *_ = _build_schedule_inputs(
            kv_lens=[8],
            q_lens=[8],
            pcp_size=2,
            page_size=4,
            interleave_size=2,
            q_block_size=4,
            global_bucket_tokens=8,
        )
        with self.assertRaisesRegex(
            ValueError, "page_size must be divisible by interleave_size"
        ):
            build_runtime_schedule_reference(
                tile_plan,
                block_tables,
                pcp_size=2,
                page_size=4,
                interleave_size=3,
            )


class PcpLayoutOwnerOffsetTest(CustomTestCase):
    """Host packing helpers keyed on ``token_owner_start_offsets_per_req``."""

    def test_rank_major_order_round_trips_through_inverse_order(self) -> None:
        q_lens = [16, 8, 8]
        owner_starts = _cu_q_lens(q_lens)[:-1].tolist()
        token_order, inverse_order = build_pcp_rank_major_token_order(
            q_lens,
            4,
            2,
            padded_num_tokens=40,
            token_owner_start_offsets_per_req=owner_starts,
        )
        total = sum(q_lens)
        self.assertEqual(inverse_order.size, total)
        np.testing.assert_array_equal(
            token_order[inverse_order], np.arange(total, dtype=np.int64)
        )
        # Padding slots stay sentinel-valued and the real rows land in the
        # per-rank prefix implied by pcp_local_token_counts.
        counts = pcp_local_token_counts(q_lens, 4, 2, owner_starts)
        local_padded = 40 // 4
        for rank, count in enumerate(counts):
            base = rank * local_padded
            self.assertTrue(np.all(token_order[base : base + count] >= 0))
            self.assertTrue(np.all(token_order[base + count : base + local_padded] < 0))

    def test_logits_indices_point_at_each_requests_last_token(self) -> None:
        q_lens = [16, 8, 8]
        owner_starts = _cu_q_lens(q_lens)[:-1].tolist()
        _token_order, inverse_order = build_pcp_rank_major_token_order(
            q_lens, 4, 2, 40, owner_starts
        )
        expected = inverse_order[np.cumsum(q_lens) - 1]
        np.testing.assert_array_equal(
            build_pcp_logits_indices(q_lens, 4, 2, 40, 0, owner_starts), expected
        )
        # token_offset shifts every index into a larger global buffer.
        np.testing.assert_array_equal(
            build_pcp_logits_indices(q_lens, 4, 2, 40, 100, owner_starts),
            expected + 100,
        )

    def test_owner_offsets_shift_ownership_relative_to_zero_offsets(self) -> None:
        # Same three requests, once packed as a batch-flat stream and once as
        # if every request restarted the stripe at 0. The layouts differ, which
        # is exactly why the tile plan carries TOKEN_OWNER_START.
        q_lens = [6, 6, 6]
        batch_flat, _ = build_pcp_rank_major_token_order(
            q_lens, 2, 2, 24, _cu_q_lens(q_lens)[:-1].tolist()
        )
        per_request, _ = build_pcp_rank_major_token_order(q_lens, 2, 2, 24, [0, 0, 0])
        self.assertFalse(np.array_equal(batch_flat, per_request))
        np.testing.assert_array_equal(np.sort(batch_flat), np.sort(per_request))


if __name__ == "__main__":
    unittest.main()
