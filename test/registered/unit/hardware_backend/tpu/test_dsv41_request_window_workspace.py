# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 TPU request-window workspace size: it
must hold the precompile's fixed-batch extend at any --max-running-requests
(regression: max_running_requests 8 = 2 request slots per DP rank with 2048-token
chunks failed the warmup with "workspace holds 40 pages, this extend needs 64")."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import request_window as rw

CHUNK = 2048  # tokens per DP rank
MAX_BS = 64  # torch_compile_max_bs: the fixed extend batch of the TPU graphs


def _warmup_extend(bs, capacity_pages):
    """The precompile's EXTEND dummy: bs requests of one token at position 0."""
    return rw._assign_workspace_pages(
        first=torch.zeros(bs, dtype=torch.int64),
        counts=torch.ones(bs, dtype=torch.int64),
        columns=8,
        base_page=1,
        capacity_pages=capacity_pages,
    )


def _chunk_extend(first, count, capacity_pages):
    return rw._assign_workspace_pages(
        first=torch.tensor([first]),
        counts=torch.tensor([count]),
        columns=(first + count) // rw.SWA_PAGE_TOKENS + 2,
        base_page=1,
        capacity_pages=capacity_pages,
    )


class TestRequestWindowWorkspace(CustomTestCase):
    def test_slot_only_size_misses_the_fixed_batch_warmup(self):
        # The failing configuration: 2 request slots per rank, 2048-token chunks.
        cap = rw._workspace_pages(num_slots=2, chunk_tokens=CHUNK)
        self.assertEqual(cap, 40)
        with self.assertRaisesRegex(RuntimeError, "workspace holds 40 pages"):
            _warmup_extend(MAX_BS, cap)

    def test_batch_rows_hold_the_warmup_and_a_full_chunk(self):
        cap = rw._workspace_pages(num_slots=2, chunk_tokens=CHUNK, batch_rows=MAX_BS)
        self.assertEqual(cap, MAX_BS)
        _warmup_extend(MAX_BS, cap)
        _chunk_extend(100_000, CHUNK, cap)

    def test_default_slot_counts_are_unchanged(self):
        # 16 requests per rank (17 slots): the chunk + history size already covers it.
        for slots in (17, 33, 65):
            self.assertEqual(
                rw._workspace_pages(
                    num_slots=slots, chunk_tokens=1024, batch_rows=MAX_BS
                ),
                rw._workspace_pages(num_slots=slots, chunk_tokens=1024),
            )

    def test_window_uses_the_extend_batch(self):
        graph = SimpleNamespace(torch_compile_max_bs=MAX_BS)
        with mock.patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(graph=graph),
        ):
            self.assertEqual(rw.extend_batch_rows(), MAX_BS)
        self.assertEqual(
            rw.window_buffer_pages(num_slots=2, chunk_tokens=CHUNK, batch_rows=MAX_BS)
            - rw.window_buffer_pages(num_slots=2, chunk_tokens=CHUNK),
            MAX_BS - 40,
        )


if __name__ == "__main__":
    unittest.main()
