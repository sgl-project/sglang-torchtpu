"""Checks for the KV-length arithmetic moved into the fused draft loop.

``_fused_draft_steps`` used to index a ``[num_steps - 1, bs]`` stack of
per-step ``kv_lens`` built eagerly by the attention backends. It now derives
step ``i``'s lengths from the base sequence lengths in-graph. These tests pin
the per-step state the loop hands each model call so that move stays
observable, including the two tracer conventions for the hidden-state update.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.eagle_utils_tpu import (
    build_topk1_verify_chain_torch,
)
from sglang_tpu.experimental.speculative.eagle.draft_worker import TpuEagleDraftWorker

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFusedDraftLengths(CustomTestCase):
    def test_step_state_matches_at_page_boundaries(self):
        for tracer in ("dynamo", "direct"):
            for dtype in (torch.int32, torch.int64):
                for steps in (1, 3, 5):
                    with self.subTest(tracer=tracer, dtype=dtype, steps=steps):
                        self.check_step_state(tracer, dtype, steps)

    def check_step_state(self, tracer, dtype, steps):
        # 63/64 straddle a 64-token page boundary, so an off-by-one in the
        # in-graph addition changes the page count the kernel would address.
        seq = torch.tensor([63, 64], dtype=dtype)
        calls = []

        def forward(ids, positions, batch):
            calls.append(
                (
                    ids.clone(),
                    positions.clone(),
                    batch.spec_info.rpa_draft_kv_lens.clone(),
                    batch.out_cache_loc.clone(),
                    batch.spec_info.hidden_states.clone(),
                )
            )
            return SimpleNamespace(
                next_token_logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                hidden_states=batch.spec_info.hidden_states + 1,
            )

        worker = object.__new__(TpuEagleDraftWorker)
        worker.topk = 1
        worker.speculative_num_steps = steps
        worker.server_args = SimpleNamespace(tpu_compile_tracer=tracer)
        # The rebind-vs-copy branch reads the tracer off the policy that
        # compiles this loop, so the fixture has to carry one.
        worker.draft_runner = SimpleNamespace(
            model=SimpleNamespace(forward=forward),
            compile_policy=SimpleNamespace(tracer=tracer),
        )

        batch = SimpleNamespace(
            positions=seq.clone(),
            spec_info=SimpleNamespace(
                hidden_states=torch.zeros(2, 8), rpa_draft_kv_lens=None
            ),
        )
        slots = torch.arange(steps * 2).reshape(steps, 2) + 101

        with patch(
            "sglang_tpu.experimental.speculative.eagle.draft_worker.build_topk1_verify_chain_torch",
            new=build_topk1_verify_chain_torch,
            create=True,
        ):
            result = worker._fused_draft_steps(
                batch,
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.int64),
                torch.zeros(2, 8),
                slots,
                None,
                torch.tensor([99, 98]),
                seq,
            )

        self.assertEqual(len(result), 6)
        # The loop breaks before the last step's model call.
        self.assertEqual(len(calls), steps - 1)
        for i, (ids, positions, kv_lens, out_cache_loc, hidden) in enumerate(calls):
            # Step 0 consumes the seeded topk_index; later steps re-select from
            # the fixed logits above, which always favour column 1 then 0.
            expected_ids = (
                torch.zeros(2, dtype=torch.int64) if i == 0 else torch.tensor([1, 0])
            )
            self.assertTrue(torch.equal(ids, expected_ids))
            self.assertTrue(torch.equal(positions, seq + i))
            # The point of the change: seq_lens + i + 1, as int32, per step.
            self.assertEqual(kv_lens.tolist(), [64 + i, 65 + i])
            self.assertEqual(kv_lens.dtype, torch.int32)
            self.assertTrue(torch.equal(out_cache_loc, slots[i]))
            # Both tracer conventions must advance the hidden state once per
            # step: "direct" rebinds the field, "dynamo" copies into it.
            self.assertTrue(torch.equal(hidden, torch.full((2, 8), float(i))))


if __name__ == "__main__":
    unittest.main()
