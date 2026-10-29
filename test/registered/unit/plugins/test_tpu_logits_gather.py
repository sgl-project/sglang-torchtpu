# SPDX-License-Identifier: Apache-2.0
"""The TPU logits all-gather must only see lane-aligned vocab shards.

libtpu miscompiles the TP all-gather of vocab-sharded logits whose shard width
is not a multiple of 128 lanes (Qwen3 at TP4: 37984 columns per shard); the
first output token was then sampled from garbage. The gather must see
128-aligned shards and the caller must still get the plain concatenation of the
unpadded shards.
"""

from __future__ import annotations

import unittest
from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import hooks
from sglang_tpu.hooks.logits_gather import gather_lane_aligned_shards

# Pure tensor arithmetic around a TPU-only collective; the collective is faked.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_all_gather(shards: list[torch.Tensor], seen_widths: list[int]):
    """Concatenate every rank's shard, padded the way this rank padded its own."""

    def gather(local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        seen_widths.append(local.shape[-1])
        pad = local.shape[-1] - shards[0].shape[-1]
        torch.testing.assert_close(local[:, : shards[0].shape[-1]], shards[0])
        return torch.cat([torch.nn.functional.pad(s, (0, pad)) for s in shards], -1)

    return gather


def _shards(rows: int, vocab_shard: int, tp_size: int = 4) -> list[torch.Tensor]:
    return [torch.randn(rows, vocab_shard) + 1000 * rank for rank in range(tp_size)]


class TestGatherLaneAlignedShards(unittest.TestCase):
    def _gather(self, rows: int, vocab_shard: int):
        shards = _shards(rows, vocab_shard)
        seen_widths: list[int] = []
        out = gather_lane_aligned_shards(
            shards[0], gather=_fake_all_gather(shards, seen_widths)
        )
        return out, torch.cat(shards, -1), seen_widths

    def test_unaligned_shard_is_gathered_lane_aligned(self):
        out, expected, seen_widths = self._gather(rows=256, vocab_shard=37984)

        self.assertEqual(seen_widths, [38016])
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_zero_rows_keep_the_full_vocab_width(self):
        out, _, _ = self._gather(rows=0, vocab_shard=37984)

        self.assertEqual(tuple(out.shape), (0, 4 * 37984))


class TestLogitsGatherHook(unittest.TestCase):
    """The hook reaches the logits gatherer's fallback all-gather on TPU."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(HookRegistry, "_hooks", defaultdict(list))
        )
        self.stack.enter_context(patch.object(HookRegistry, "_patched", set()))
        self.stack.enter_context(patch.object(hooks, "_registered", False))
        original = triton_symm_mem_ag.MultimemAllGatherer.__call__
        self.stack.enter_context(
            patch.object(triton_symm_mem_ag.MultimemAllGatherer, "__call__", original)
        )

    def _gather_through_hook(self, *, tpu: bool) -> tuple[torch.Tensor, list[int]]:
        platform = SimpleNamespace(is_tpu=lambda: tpu)
        with patch("sglang.srt.platforms.current_platform", platform):
            hooks.register()
        HookRegistry.apply_hooks()
        shards = _shards(rows=4, vocab_shard=37984)
        seen_widths: list[int] = []
        gatherer = triton_symm_mem_ag.MultimemAllGatherer(max_tokens=4, enabled=False)
        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_gather",
            _fake_all_gather(shards, seen_widths),
        ):
            out = gatherer(shards[0])
        torch.testing.assert_close(out, torch.cat(shards, -1), rtol=0, atol=0)
        return out, seen_widths

    def test_tpu_gatherer_pads_the_shard(self):
        _, seen_widths = self._gather_through_hook(tpu=True)

        self.assertEqual(seen_widths, [38016])

    def test_other_platforms_gather_the_raw_shard(self):
        _, seen_widths = self._gather_through_hook(tpu=False)

        self.assertEqual(seen_widths, [37984])


if __name__ == "__main__":
    unittest.main()
