"""Registered coverage for TPU perfect-balanced expert simulation."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.topk import (
    _simulate_tpu_perfect_balanced_routing,
    _tpu_perfect_balanced_routing_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_tpu_flag_is_ignored_on_other_platforms():
    with (
        patch("sglang.srt.layers.moe.topk._is_tpu", False),
        patch(
            "sglang.srt.layers.moe.topk.envs."
            "SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS.get",
            return_value=True,
        ),
    ):
        assert not _tpu_perfect_balanced_routing_enabled()


def test_tpu_flag_is_enabled_on_tpu():
    with (
        patch("sglang.srt.layers.moe.topk._is_tpu", True),
        patch(
            "sglang.srt.layers.moe.topk.envs."
            "SGLANG_TPU_SIMULATE_PERFECT_BALANCED_EXPERTS.get",
            return_value=True,
        ),
    ):
        assert _tpu_perfect_balanced_routing_enabled()


def test_tpu_perfect_balanced_routing_spreads_assignments_across_ranks():
    num_experts, topk, ep_size = 256, 8, 16
    for num_tokens in (1, 2, 4, 300):
        ids = torch.full((num_tokens, topk), -1, dtype=torch.int32)
        weights = torch.zeros((num_tokens, topk), dtype=torch.float32)

        _simulate_tpu_perfect_balanced_routing(
            ids,
            weights,
            num_experts,
            num_ranks=ep_size,
            layer_id=0,
        )

        counts = torch.bincount(
            ids.flatten().long() // (num_experts // ep_size), minlength=ep_size
        )
        assert int(counts.max() - counts.min()) <= 1
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / topk))


def test_tpu_perfect_balanced_routing_uses_routed_topk_as_assignment_stride():
    num_experts, routed_topk, total_topk, ep_size = 256, 8, 9, 16
    ids = torch.full((2, total_topk), -1, dtype=torch.int32)
    weights = torch.zeros((2, total_topk), dtype=torch.float32)

    _simulate_tpu_perfect_balanced_routing(
        ids,
        weights,
        num_experts,
        num_ranks=ep_size,
        num_routed_topk=routed_topk,
        layer_id=0,
    )

    counts = torch.bincount(
        ids[:, :routed_topk].flatten().long() // (num_experts // ep_size),
        minlength=ep_size,
    )
    assert torch.equal(counts, torch.ones_like(counts))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
