# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for SGLANG_TPU_DSV41_SWA_PLAN_TRIM: an extend's WindowPlan sized
to the rank's request slots moves exactly the rows of the full [B * 128] plan."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import device_rows
from sglang_tpu.models.deepseek_v41 import request_window as rw

B = 64  # torch_compile_max_bs
COLUMNS = 64
BASE_PAGE = 100


def _moves(plan: rw.WindowPlan):
    """The plan's valid (destination, source) pairs, as sets."""
    history = {
        (int(d), int(s))
        for s, d in zip(plan.history_src.tolist(), plan.history_dst.tolist())
        if d >= 0
    }
    commit = {
        (int(d), int(t))
        for t, d in zip(plan.commit_token.tolist(), plan.commit_dst.tolist())
        if d >= 0
    }
    return history, commit


def _batch(gen, live_requests, num_slots):
    """B batch rows: ``live_requests`` real requests on distinct slots, scattered
    among padding rows (slot 0) and empty rows, plus one repeated slot."""
    slots = torch.zeros(B, dtype=torch.int64)
    counts = torch.zeros(B, dtype=torch.int64)
    first = torch.zeros(B, dtype=torch.int64)
    rows = torch.randperm(B, generator=gen)[:live_requests]
    chosen = torch.randperm(num_slots - 1, generator=gen)[:live_requests] + 1
    for row, slot in zip(rows.tolist(), chosen.tolist()):
        slots[row] = slot
        counts[row] = int(torch.randint(1, 300, (1,), generator=gen))
        first[row] = int(torch.randint(0, 2000, (1,), generator=gen))
    if live_requests:
        # A repeated slot (dummy batches) moves rows for its first request only.
        spare = [r for r in range(B) if r not in set(rows.tolist())]
        slots[spare[0]] = slots[rows[0]]
        counts[spare[0]] = 3
        first[spare[0]] = 7
    minimum = (first - torch.randint(0, 200, (B,), generator=gen)).clamp(min=0)
    return slots, first, counts, minimum


def _host(slots, first, counts, minimum, requests):
    return rw.extend_window_pages(
        slots=slots,
        first=first,
        counts=counts,
        minimum=minimum,
        columns=COLUMNS,
        base_page=BASE_PAGE,
        capacity_pages=10_000,
        requests=requests,
    )


class TestSwaPlanTrim(CustomTestCase):
    def test_trimmed_plan_moves_the_same_rows(self):
        gen = torch.Generator().manual_seed(0)
        for num_slots in (2, 5, 17, 33):
            requests = num_slots - 1
            for live in sorted({0, 1, requests // 2, requests}):
                for _ in range(5):
                    batch = _batch(gen, live, num_slots)
                    pages_full, full = _host(*batch, None)
                    pages_trim, trim = _host(*batch, requests)
                    self.assertTrue(torch.equal(pages_full, pages_trim))
                    self.assertEqual(_moves(full), _moves(trim))
                    for t in trim:
                        self.assertEqual(t.shape, (requests * rw.WINDOW_TOKENS,))

    def test_device_build_matches_host(self):
        gen = torch.Generator().manual_seed(1)
        requests = 16
        for live in (0, 1, 9, 16):
            slots, first, counts, minimum = _batch(gen, live, requests + 1)
            _, host = _host(slots, first, counts, minimum, requests)
            live_rows = device_rows.window_live(slots, counts)
            _, dev = device_rows.window_pages(
                first=first.to(torch.int32),
                counts=counts.to(torch.int32),
                live=live_rows,
                slots=slots.to(torch.int32),
                minimum=minimum.to(torch.int32),
                columns=COLUMNS,
                base_page=BASE_PAGE,
                requests=requests,
            )
            for h, d in zip(host, dev):
                self.assertTrue(torch.equal(h.to(torch.int32), d))

    def test_no_trim_when_batch_fits(self):
        live = torch.tensor([True, False, True])
        self.assertIsNone(rw.live_plan_rows(live, None))
        self.assertIsNone(rw.live_plan_rows(live, 3))
        self.assertEqual(rw.live_plan_rows(live, 2).tolist(), [0, 2])
        live = torch.tensor([False, True, False, True, True])
        self.assertEqual(rw.live_plan_rows(live, 4).tolist(), [1, 3, 4, 0])


if __name__ == "__main__":
    unittest.main()
