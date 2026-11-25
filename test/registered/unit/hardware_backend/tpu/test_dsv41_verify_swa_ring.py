# SPDX-License-Identifier: Apache-2.0
"""A DSpark verify that attends the request-window SWA rings directly must see
exactly the rows the extend workspace plan gives it, over many cycles of partial
acceptance (rejected rows of a cycle stay in the ring until overwritten)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import request_window as rw

DIM = 8
WIDTH = 6  # gamma + 1 verify rows per request
COLUMNS = 64  # SWA table pages: positions below 4096
NUM_SLOTS = 4
WORKSPACE_FIRST = rw.RESERVED_PAGES + rw.RING_PAGES * NUM_SLOTS
WORKSPACE_PAGES = 64


def _table_rows(table, b, positions):
    return table[b, positions // rw.SWA_PAGE_TOKENS] * rw.SWA_PAGE_TOKENS + (
        positions % rw.SWA_PAGE_TOKENS
    )


def _publish(buf, records, kv):
    """csa2.publish_swa: rows below the first page (null page, -1) are dropped."""
    keep = records >= rw.SWA_PAGE_TOKENS
    assert torch.unique(records[keep]).numel() == int(keep.sum()), "duplicate record"
    buf[records[keep]] = kv[keep]


def _forward(buf, *, slots, first, counts, minimum, kv, ring):
    """One forward's SWA write (backend._attention) and the rows each query sees."""
    if ring:
        table, plan = rw.decode_window_pages(slots, COLUMNS), None
    else:
        table, plan = rw.extend_window_pages(
            slots=slots,
            first=first,
            counts=counts,
            minimum=minimum,
            columns=COLUMNS,
            base_page=WORKSPACE_FIRST,
            capacity_pages=WORKSPACE_PAGES,
        )
    records = []
    for b in range(slots.shape[0]):
        pos = first[b] + torch.arange(int(counts[b]))
        records.append(_table_rows(table, b, pos))
    records = torch.cat(records)
    if plan is not None:
        history = buf[plan.history_src]
        kv = torch.cat((kv, history, kv[plan.commit_token]))
        records = torch.cat((records, plan.history_dst.long(), plan.commit_dst.long()))
    _publish(buf, records, kv)
    seen = []
    for b in range(slots.shape[0]):
        for q in range(int(first[b]), int(first[b] + counts[b])):
            low = min(max(q - rw.WINDOW_TOKENS + 1, int(minimum[b]), 0), q + 1)
            seen.append(buf[_table_rows(table, b, torch.arange(low, q + 1))].clone())
    return seen


class TestVerifySwaRing(CustomTestCase):
    def _run(self, seed, *, prompt_lens, minimum):
        gen = torch.Generator().manual_seed(seed)
        slots = torch.tensor([1, 3, 2])
        minimum = torch.tensor(minimum)
        total = (WORKSPACE_FIRST + WORKSPACE_PAGES) * rw.SWA_PAGE_TOKENS
        bufs = {mode: torch.zeros(total, DIM) for mode in (False, True)}
        # Prefill: the extend plan fills each ring with the prompt's last window.
        first = torch.zeros(3, dtype=torch.int64)
        counts = torch.tensor(prompt_lens)
        kv = torch.randn(int(counts.sum()), DIM, generator=gen)
        for mode in (False, True):
            _forward(
                bufs[mode],
                slots=slots,
                first=first,
                counts=counts,
                minimum=minimum,
                kv=kv,
                ring=False,
            )
        seq = counts.clone()
        for _cycle in range(60):
            counts = torch.full((3,), WIDTH)
            kv = torch.randn(3 * WIDTH, DIM, generator=gen)
            seen = {
                mode: _forward(
                    bufs[mode],
                    slots=slots,
                    first=seq,
                    counts=counts,
                    minimum=minimum,
                    kv=kv,
                    ring=mode,
                )
                for mode in (False, True)
            }
            for a, b in zip(seen[False], seen[True], strict=True):
                self.assertTrue(torch.equal(a, b))
            # Same rings, so a later extend's history copy reads the same rows.
            rings = slice(0, WORKSPACE_FIRST * rw.SWA_PAGE_TOKENS)
            self.assertTrue(torch.equal(bufs[False][rings], bufs[True][rings]))
            # Accept 1..WIDTH tokens per request; the rest stay in the ring.
            seq = seq + torch.randint(1, WIDTH + 1, (3,), generator=gen)
        return seq

    def test_ring_matches_the_workspace_plan(self):
        for seed in range(4):
            self._run(seed, prompt_lens=[300, 7, 129], minimum=[0, 0, 0])

    def test_ring_respects_the_replay_floor(self):
        self._run(7, prompt_lens=[500, 260, 64], minimum=[420, 200, 0])


if __name__ == "__main__":
    unittest.main()
