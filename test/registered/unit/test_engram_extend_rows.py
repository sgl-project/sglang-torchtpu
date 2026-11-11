"""Packed extend row dispatch must preserve hash inputs and history ownership."""

import types
import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers import engram
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RemappedHistoryHasher(engram.EngramHasher):
    def prepare_extend_rows(self, lens, starts, *, bs, device, output_size):
        # Deliberately reroute to a different history row so a missing OOT
        # dispatch cannot pass by producing the same values as the native path.
        return torch.tensor([1, 1], dtype=torch.int64, device=device)


class UnblockedLookbackHasher(engram.EngramHasher):
    def prepare_lookback_blocked(self, mask):
        # Deliberately remove barriers so bypassing the external seam changes
        # observable hashes, not merely a mock invocation count.
        return torch.zeros_like(mask)


class TestEngramExtendRows(CustomTestCase):
    @staticmethod
    def hasher(hasher_cls=engram.EngramHasher):
        model = hasher_cls.__new__(hasher_cls)
        torch.nn.Module.__init__(model)
        model.max_ngram_size = 2
        model.pad_id = 0
        model.pad_row = 2
        model.image_token_id = None
        model.history = torch.tensor([[3], [9], [0]], dtype=torch.int32)
        model.token_map = torch.arange(32)
        model.multipliers = torch.tensor([[1, 2]])
        model.primes = torch.tensor([[[101]]])
        model.offsets = torch.tensor([[0]])
        return model

    @staticmethod
    def batch():
        return types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: False,
                is_target_verify=lambda: False,
                is_extend=lambda: True,
            ),
            req_pool_indices=torch.tensor([0, 1]),
            extend_seq_lens=torch.tensor([2, 0]),
            extend_start_loc=torch.tensor([0, 2]),
            extend_seq_lens_cpu=[2, 0],
            positions=torch.tensor([1, 2, 0, 0]),
            engram_history=None,
            out_cache_loc=torch.tensor([1, 2, 0, 0]),
        )

    def test_native_and_external_dispatch_changes_hash_not_history_ownership(self):
        # Native predecessors are [5,3] and [6,5]. Deliberately routing to
        # empty row 1 makes both predecessor pairs [9,9], including the
        # current-token column because both offsets are negative.
        ids = torch.tensor([5, 6, 0, 0])
        for oot, hashes, committed in (
            (False, [3, 12], [6, 9, 6]),
            (True, [27, 27], [9, 9, 9]),
        ):
            with self.subTest(out_of_tree=oot):
                model = self.hasher(RemappedHistoryHasher)
                platform = types.SimpleNamespace(is_out_of_tree=lambda oot=oot: oot)
                with patch.object(engram, "current_platform", platform):
                    result = model(ids, self.batch())
                torch.testing.assert_close(
                    result[:, 0, 0], torch.tensor([*hashes, 0, 0])
                )
                # Commit still uses scheduler request ownership, not row mapping.
                torch.testing.assert_close(
                    model.history,
                    torch.tensor(committed, dtype=torch.int32).reshape(-1, 1),
                )

    def test_image_barriers_use_external_preparer_without_changing_commit(self):
        for oot, expected in ((False, [5, 12, 0, 0]), (True, [3, 12, 0, 0])):
            with self.subTest(out_of_tree=oot):
                model = self.hasher(UnblockedLookbackHasher)
                model.image_token_id = 3
                platform = types.SimpleNamespace(is_out_of_tree=lambda oot=oot: oot)
                with patch.object(engram, "current_platform", platform):
                    result = model(torch.tensor([5, 6, 0, 0]), self.batch())
                # History [3] is an image: native mode blocks that predecessor;
                # the deliberately unblocked external mapping retains it.
                torch.testing.assert_close(result[:, 0, 0], torch.tensor(expected))
                torch.testing.assert_close(
                    model.history, torch.tensor([[6], [9], [6]], dtype=torch.int32)
                )

    def test_default_lookback_preparer_prefix_barriers_and_empty_rows(self):
        model = self.hasher()
        for width in (1, 2, 4, 8):
            masks = torch.tensor(
                [
                    [bool(bits & (1 << i)) for i in range(width)]
                    for bits in range(1 << width)
                ],
                dtype=torch.bool,
            )
            expected = torch.tensor(
                [[any(row[: i + 1]) for i in range(width)] for row in masks.tolist()],
                dtype=torch.bool,
            )
            torch.testing.assert_close(model.prepare_lookback_blocked(masks), expected)
            empty = torch.empty((0, width), dtype=torch.bool)
            torch.testing.assert_close(model.prepare_lookback_blocked(empty), empty)

    def test_default_preparer_zero_length_boundaries_and_unknown_total(self):
        model = self.hasher()
        for lengths in ([0, 2, 0, 3, 0], [0, 0], []):
            lens = torch.tensor(lengths, dtype=torch.int64)
            starts = lens.cumsum(0) - lens
            expected = [i for i, length in enumerate(lengths) for _ in range(length)]
            for total in (sum(lengths), None):
                with self.subTest(lengths=lengths, total=total):
                    rows = model.prepare_extend_rows(
                        lens,
                        starts,
                        bs=len(lengths),
                        device=lens.device,
                        output_size=total,
                    )
                    torch.testing.assert_close(
                        rows, torch.tensor(expected, dtype=torch.int64)
                    )


if __name__ == "__main__":
    unittest.main()
