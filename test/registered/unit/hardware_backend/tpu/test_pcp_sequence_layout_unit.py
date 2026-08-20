# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the PCP sequence-layout planner.

Cover the pure-Python planner (:func:`plan_pcp_sequence_layout`) and the
host-side reorder helpers (:func:`apply_pcp_local_shard`,
:func:`gather_and_unpack_pcp`) — no TPU / JAX device required.

The tests exercise: padded-num-tokens validation (pcp + interleave), the
identity trivial case (pcp_size=1), rank-major reorder correctness, and
the D->H-sync-free host path introduced when eliminating the eager-side
sync on ``bool(valid.all())``.
"""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch

from sglang.srt.layers.cp.pcp_sequence_layout import (
    PcpSequenceLayoutPlan,
    apply_pcp_local_shard,
    gather_and_unpack_pcp,
    plan_pcp_sequence_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_forward_batch(plan) -> types.SimpleNamespace:
    """The reorder helpers only touch ``forward_batch.pcp_layout_plan``."""
    return types.SimpleNamespace(pcp_layout_plan=plan)


class PlanValidationTest(CustomTestCase):
    def test_pcp_size_zero_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "pcp_size must be >= 1"):
            plan_pcp_sequence_layout(
                num_scheduled_tokens_per_req=np.array([16]),
                prompt_tokens_per_req=np.array([16]),
                computed_tokens_per_req=np.array([0]),
                padded_num_tokens=16,
                pcp_size=0,
                pcp_rank=0,
                interleave_size=1,
            )

    def test_interleave_size_zero_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "interleave_size must be >= 1"):
            plan_pcp_sequence_layout(
                num_scheduled_tokens_per_req=np.array([16]),
                prompt_tokens_per_req=np.array([16]),
                computed_tokens_per_req=np.array([0]),
                padded_num_tokens=16,
                pcp_size=2,
                pcp_rank=0,
                interleave_size=0,
            )

    def test_padded_not_divisible_by_pcp_size_raises(self) -> None:
        # padded_num_tokens=17 with pcp_size=2 fails the divisibility guard.
        with self.assertRaisesRegex(ValueError, "must be divisible by pcp_size=2"):
            plan_pcp_sequence_layout(
                num_scheduled_tokens_per_req=np.array([16]),
                prompt_tokens_per_req=np.array([16]),
                computed_tokens_per_req=np.array([0]),
                padded_num_tokens=17,
                pcp_size=2,
                pcp_rank=0,
                interleave_size=1,
            )

    def test_padded_not_divisible_by_pcp_x_interleave_raises(self) -> None:
        # padded=16 is divisible by pcp_size=2 but NOT by
        # pcp_size*interleave_size = 2*4 = 8? 16 % 8 == 0, so choose a
        # value that fails: padded=12, pcp_size=2, interleave_size=4
        # -> 2*4=8, 12 % 8 == 4 -> fail.
        with self.assertRaisesRegex(
            ValueError, r"must be divisible by pcp_size\*interleave_size=8"
        ):
            plan_pcp_sequence_layout(
                num_scheduled_tokens_per_req=np.array([12]),
                prompt_tokens_per_req=np.array([12]),
                computed_tokens_per_req=np.array([0]),
                padded_num_tokens=12,
                pcp_size=2,
                pcp_rank=0,
                interleave_size=4,
            )


class PlanTrivialTest(CustomTestCase):
    def test_pcp_size_one_identity_layout(self) -> None:
        # cp_size=1 short-circuits to identity; no reorder needed.
        plan = plan_pcp_sequence_layout(
            num_scheduled_tokens_per_req=np.array([4]),
            prompt_tokens_per_req=np.array([4]),
            computed_tokens_per_req=np.array([0]),
            padded_num_tokens=8,
            pcp_size=1,
            pcp_rank=0,
            interleave_size=1,
        )
        # 4 real tokens + 4 padding sentinels.
        np.testing.assert_array_equal(
            plan.token_order, np.array([0, 1, 2, 3, -1, -1, -1, -1])
        )
        self.assertEqual(plan.global_num_tokens, 4)
        self.assertEqual(plan.global_padded_num_tokens, 8)
        self.assertEqual(plan.local_num_tokens, 4)
        self.assertEqual(plan.local_padded_num_tokens, 8)


class ApplyPcpLocalShardTest(CustomTestCase):
    def _plan(self, token_order: list[int], pcp_size: int, pcp_rank: int):
        arr = np.array(token_order, dtype=np.int64)
        assert arr.size % pcp_size == 0
        return PcpSequenceLayoutPlan(
            pcp_size=pcp_size,
            interleave_size=1,
            pcp_rank=pcp_rank,
            token_order=arr,
            inverse_order=np.zeros(arr.size, dtype=np.int64),
            local_token_counts=np.array(
                [arr.size // pcp_size] * pcp_size, dtype=np.int32
            ),
            global_num_tokens=int((arr >= 0).sum()),
            global_padded_num_tokens=arr.size,
            local_num_tokens=arr.size // pcp_size,
            local_padded_num_tokens=arr.size // pcp_size,
        )

    def test_pcp_size_one_returns_input_verbatim(self) -> None:
        # Trivial plan -> apply_pcp_local_shard is a pass-through.
        plan = self._plan(token_order=[0, 1, 2, 3], pcp_size=1, pcp_rank=0)
        fb = _fake_forward_batch(plan)
        x = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 3).contiguous()
        out = apply_pcp_local_shard(x, fb, axis=0)
        self.assertIs(out, x)

    def test_rank_0_of_2_picks_first_half_of_reorder(self) -> None:
        # token_order [2, 0, 3, 1] with pcp_size=2 means rank 0 owns rows
        # [2, 0] and rank 1 owns rows [3, 1] of the request-major batch.
        plan = self._plan(token_order=[2, 0, 3, 1], pcp_size=2, pcp_rank=0)
        fb = _fake_forward_batch(plan)
        x = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 3).contiguous()
        out = apply_pcp_local_shard(x, fb, axis=0)
        # Row 2 then row 0.
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_array_equal(out[0].numpy(), np.full(3, 2.0))
        np.testing.assert_array_equal(out[1].numpy(), np.full(3, 0.0))

    def test_rank_1_of_2_picks_second_half(self) -> None:
        plan = self._plan(token_order=[2, 0, 3, 1], pcp_size=2, pcp_rank=1)
        fb = _fake_forward_batch(plan)
        x = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 3).contiguous()
        out = apply_pcp_local_shard(x, fb, axis=0)
        np.testing.assert_array_equal(out[0].numpy(), np.full(3, 3.0))
        np.testing.assert_array_equal(out[1].numpy(), np.full(3, 1.0))

    def test_padding_sentinels_produce_zero_rows(self) -> None:
        # rank 1 gets [-1, 1] -> the -1 slot is padding and must be zeroed.
        plan = self._plan(token_order=[2, 0, -1, 1], pcp_size=2, pcp_rank=1)
        fb = _fake_forward_batch(plan)
        x = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 3).contiguous()
        out = apply_pcp_local_shard(x, fb, axis=0)
        # Row 0 of this rank's output = the -1 slot -> all zeros.
        np.testing.assert_array_equal(out[0].numpy(), np.zeros(3))
        # Row 1 of this rank's output = row 1 of input.
        np.testing.assert_array_equal(out[1].numpy(), np.full(3, 1.0))

    def test_raises_when_pcp_layout_plan_is_none(self) -> None:
        fb = _fake_forward_batch(plan=None)
        x = torch.zeros(4, 2)
        with self.assertRaisesRegex(RuntimeError, "no pcp_layout_plan"):
            apply_pcp_local_shard(x, fb, axis=0)


class GatherAndUnpackTest(CustomTestCase):
    def test_pcp_size_one_returns_input_verbatim(self) -> None:
        plan = PcpSequenceLayoutPlan(
            pcp_size=1,
            interleave_size=1,
            pcp_rank=0,
            token_order=np.array([0, 1, 2, 3], dtype=np.int64),
            inverse_order=np.array([0, 1, 2, 3], dtype=np.int64),
            local_token_counts=np.array([4], dtype=np.int32),
            global_num_tokens=4,
            global_padded_num_tokens=4,
            local_num_tokens=4,
            local_padded_num_tokens=4,
        )
        fb = _fake_forward_batch(plan)
        x = torch.zeros(4, 3)
        out = gather_and_unpack_pcp(x, fb)
        self.assertIs(out, x)


class InterleaveAndBoundaryCrossingTest(CustomTestCase):
    """Coverage the reference-parity tests don't hit: interleave > 1 and
    the boundary-crossing (prompt/decode mixed in one forward) rejection.
    """

    def test_interleave_gt_1_planner(self) -> None:
        # pcp_size=2, interleave_size=2 → cycle=4. 8 tokens map to
        # {rank 0: positions 0,1,4,5} and {rank 1: positions 2,3,6,7}.
        plan = plan_pcp_sequence_layout(
            num_scheduled_tokens_per_req=np.array([8]),
            prompt_tokens_per_req=np.array([8]),
            computed_tokens_per_req=np.array([0]),
            padded_num_tokens=8,
            pcp_size=2,
            pcp_rank=0,
            interleave_size=2,
        )
        # local_token_counts = 4 per rank (byte-symmetric under this layout).
        np.testing.assert_array_equal(
            plan.local_token_counts, np.array([4, 4], dtype=np.int32)
        )
        self.assertEqual(plan.local_padded_num_tokens, 4)
        # token_order[0:4] = rank 0's slots (positions 0,1,4,5), byte-order.
        np.testing.assert_array_equal(
            plan.token_order[:4], np.array([0, 1, 4, 5], dtype=np.int64)
        )
        # token_order[4:8] = rank 1's slots (positions 2,3,6,7).
        np.testing.assert_array_equal(
            plan.token_order[4:], np.array([2, 3, 6, 7], dtype=np.int64)
        )

    def test_boundary_crossing_rejected(self) -> None:
        # Request 0: prompt_tokens=8, computed=6, scheduled=4 → span [6, 10)
        # crosses the prompt boundary at position 8 (6 < 8 < 10).
        with self.assertRaisesRegex(
            NotImplementedError,
            "prompt/decode boundary-crossing",
        ):
            plan_pcp_sequence_layout(
                num_scheduled_tokens_per_req=np.array([4]),
                prompt_tokens_per_req=np.array([8]),
                computed_tokens_per_req=np.array([6]),
                padded_num_tokens=8,
                pcp_size=2,
                pcp_rank=0,
                interleave_size=1,
            )


class GatherUnpackRoundTripTest(CustomTestCase):
    """Round-trip: apply_pcp_local_shard(x) then gather_and_unpack_pcp of
    the reassembled shards must reproduce x. Guards against off-by-one in
    inverse_order interpretation."""

    def test_pcp_size_2_identity_permutation(self) -> None:
        # Identity token_order for pcp_size=2 with 8 tokens: rank 0 owns
        # positions [0..3], rank 1 owns [4..7]. Round-trip must yield x.
        pcp_size = 2
        num_tokens = 8
        plan = PcpSequenceLayoutPlan(
            pcp_size=pcp_size,
            interleave_size=1,
            pcp_rank=0,
            token_order=np.arange(num_tokens, dtype=np.int64),
            inverse_order=np.arange(num_tokens, dtype=np.int64),
            local_token_counts=np.array([4, 4], dtype=np.int32),
            global_num_tokens=num_tokens,
            global_padded_num_tokens=num_tokens,
            local_num_tokens=4,
            local_padded_num_tokens=4,
        )
        x = torch.arange(num_tokens * 3, dtype=torch.float32).reshape(num_tokens, 3)
        fb = _fake_forward_batch(plan)
        local = apply_pcp_local_shard(x, fb, axis=0)
        # Rank 0's slice = rows [0..3].
        torch.testing.assert_close(local, x[:4])


if __name__ == "__main__":
    unittest.main()
