"""Value and ownership contracts for scheduler row operations."""

import inspect

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import row_ops
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.warmup_queue import collect_warmups
from sglang_tpu.row_ops import (
    _filter_padded_rows,
    _merge_padded_rows,
    filter_rows,
    merge_rows,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("sizes", [(0, 3), (3, 0), (1, 1), (2, 1), (3, 1)])
def test_row_operations_preserve_metadata_and_history(sizes):
    left_rows, right_rows = sizes
    for dtype, tail in (
        (torch.int64, ()),
        (torch.int32, (3,)),
        (torch.float32, ()),
        (torch.float32, (1,)),
    ):
        width = tail[0] if tail else 1
        left = torch.arange(left_rows * width, dtype=dtype).reshape(left_rows, *tail)
        right = (torch.arange(right_rows * width, dtype=dtype) + 100).reshape(
            right_rows, *tail
        )
        expected = torch.cat((left, right))
        assert torch.equal(merge_rows(left, right), expected)
        if not left_rows or not right_rows:
            continue
        rows = left_rows + right_rows
        bucket = 1 << (rows - 1).bit_length()
        padded_left = torch.zeros((bucket, *tail), dtype=dtype)
        padded_right = torch.zeros_like(padded_left)
        padded_left[:left_rows] = left
        padded_right[:right_rows] = right
        actual = inspect.unwrap(_merge_padded_rows)(
            padded_left, padded_right, torch.tensor(left_rows)
        )[:rows]
        assert actual.dtype == dtype
        assert torch.equal(actual, expected)
        actual.fill_(-1)
        assert torch.equal(padded_left[:left_rows], left)
        assert torch.equal(padded_right[:right_rows], right)
        # Filtering a merged batch preserves ordering, duplicates, negative
        # indices and independent output ownership for metadata and history.
        for indices in (
            torch.tensor([rows - 1, 0, -1]),
            torch.empty(0, dtype=torch.int64),
        ):
            expected_selected = expected[indices]
            assert torch.equal(filter_rows(expected, indices), expected_selected)
            selected = inspect.unwrap(_filter_padded_rows)(
                expected, indices, torch.tensor(rows)
            )
            assert torch.equal(selected, expected_selected)
            selected.fill_(-2)
            assert torch.equal(expected, torch.cat((left, right)))


def test_row_precompile_defers_execution_until_global_join(monkeypatch):
    left = torch.zeros(2, dtype=torch.int64)
    right = torch.ones(2, dtype=torch.int64)
    split = torch.tensor(1)
    monkeypatch.setattr(
        row_ops, "_row_bucket_inputs", lambda *args: [(left, right, split)]
    )
    completed = []

    def merge(*args, _warmup_async=False):
        assert _warmup_async, "precompile must not execute the compiled merge"
        return PendingCall(lambda: completed.append(("merge", args[0].shape[0])))

    def gather(*args, _warmup_async=False):
        assert _warmup_async, "precompile must not execute the compiled gather"
        return PendingCall(lambda: completed.append(("filter", args[1].numel())))

    monkeypatch.setattr(row_ops, "_merge_padded_rows", merge)
    monkeypatch.setattr(row_ops, "_filter_padded_rows", gather)
    with collect_warmups() as queue:
        row_ops.precompile_row_ops([left], 2)
        assert len(queue.calls) == 3
        assert not completed
    assert completed == [("merge", 2), ("filter", 1), ("filter", 2)]
