"""CPU contract for the row-selection hooks: the rows of plain indexing and of
concatenation, gathered from rows padded to the request limit."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import row_select
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import row_select as hook


def _hooked(name, *tensors, capacity=16, on_device=True):
    schedule = SimpleNamespace(max_running_requests=capacity)
    with (
        patch.object(hook, "get_schedule", return_value=schedule),
        patch.object(hook, "_on_device", return_value=on_device),
    ):
        return getattr(hook, name)(getattr(row_select, name), *tensors)


@contextlib.contextmanager
def _record_rows(log):
    """Logs the row counts that torch.cat joins and index_select gathers."""
    cat, gather = torch.cat, torch.Tensor.index_select

    def record_cat(tensors, *args, **kwargs):
        log.append(("cat", *(t.shape[0] for t in tensors)))
        return cat(tensors, *args, **kwargs)

    def record_gather(source, dim, index):
        log.append(("gather", source.shape[0], index.shape[0]))
        return gather(source, dim, index)

    with (
        patch.object(torch, "cat", record_cat),
        patch.object(torch.Tensor, "index_select", record_gather),
    ):
        yield


class TestRowSelectHook(CustomTestCase):
    def test_select_gathers_from_rows_padded_to_the_limit(self):
        # 7->4 and 6->4 share the gather program: it sees only (16, kept rows).
        index, log = torch.tensor([0, 2, 5, 3]), []
        for tensor in (
            torch.arange(7, dtype=torch.int32),
            torch.randn(6, 4),
            torch.arange(7) % 3 == 0,
        ):
            with _record_rows(log):
                out = _hooked("select_rows", tensor, index)
            self.assertEqual(out.dtype, tensor.dtype)
            self.assertTrue(torch.equal(out, tensor[index]))
        self.assertEqual(log, [("gather", 16, 4)] * 3)

    def test_concat_joins_two_limit_row_tensors_then_gathers(self):
        log = []
        for first, second in (
            (torch.arange(5, dtype=torch.int32), torch.arange(3, dtype=torch.int32)),
            # int32 running lengths join int64 new lengths: promoted like cat.
            (torch.arange(4, dtype=torch.int32), torch.arange(2) + 9),
            (torch.randn(3, 4), torch.randn(5, 4)),
            (torch.arange(3) % 2 == 0, torch.ones(2, dtype=torch.bool)),
        ):
            with _record_rows(log):
                out = _hooked("concat_rows", first, second)
            expected = torch.cat([first, second])
            self.assertEqual(out.dtype, expected.dtype)
            self.assertTrue(torch.equal(out, expected))
        self.assertEqual(log[::2], [("cat", 16, 16)] * 4)
        self.assertEqual(log[1::2], [("gather", 32, n) for n in (8, 6, 8, 5)])

    def test_concat_adding_one_row_uses_cat_after_one_padded_join(self):
        first, second, log = torch.arange(7), torch.arange(1) + 50, []
        with (
            patch.dict(hook._JOIN_INDEX, clear=True),
            patch.object(hook, "_WARMED", set()),
            _record_rows(log),
        ):
            outs = [_hooked("concat_rows", first, second) for _ in range(2)]
        self.assertEqual(
            log, [("cat", 16, 16), ("gather", 32, 8), ("cat", 7, 1), ("cat", 7, 1)]
        )
        for out in outs:
            self.assertTrue(torch.equal(out, torch.cat([first, second])))

    def test_concat_index_is_built_once_per_pair_and_kept_with_its_copy(self):
        # The device copy is non-blocking (a blocking copy waits for queued
        # device work), so the cache keeps its host source alive.
        sources = []
        real_to = torch.Tensor.to

        def record(tensor, *args, **kwargs):
            if kwargs.get("non_blocking"):
                sources.append(weakref.ref(tensor))
            return real_to(tensor, *args, **kwargs).clone()  # a new tensor

        with (
            patch.dict(hook._JOIN_INDEX, clear=True),
            patch.object(torch.Tensor, "to", record),
        ):
            for _ in range(2):
                out = _hooked("concat_rows", torch.arange(2), torch.arange(3) + 2)
                self.assertTrue(torch.equal(out, torch.arange(5)))
            gc.collect()
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]().tolist(), [0, 1, 16, 17, 18])
            _hooked("concat_rows", torch.arange(3), torch.arange(2))
            self.assertEqual(len(sources), 2)

    def test_plain_off_device_beyond_the_limit_or_mismatched(self):
        seven, two = torch.arange(7), torch.arange(2)
        cases = (
            ("select_rows", (seven, two), {"on_device": False}),
            ("select_rows", (seven, two), {"capacity": None}),
            ("select_rows", (seven, two), {"capacity": 6}),
            ("concat_rows", (seven, two), {"on_device": False}),
            ("concat_rows", (seven, two), {"capacity": None}),
            ("concat_rows", (seven, two), {"capacity": 8}),
        )
        for name, tensors, kwargs in cases:
            with patch.object(hook, "_pad_rows") as pad:
                out = _hooked(name, *tensors, **kwargs)
            pad.assert_not_called()
            self.assertTrue(torch.equal(out, getattr(row_select, name)(*tensors)))
        with patch.object(hook, "_pad_rows") as pad, self.assertRaises(RuntimeError):
            _hooked("concat_rows", torch.zeros(3, 2), torch.zeros(2, 3))
        pad.assert_not_called()

    def test_sampling_filter_and_merge_route_device_params_through_the_seam(self):
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        def info(n, on_cpu=False):
            return SimpleNamespace(
                sampling_params_on_cpu=on_cpu,
                temperatures=torch.arange(float(n)).reshape(n, 1),
                top_ps=torch.ones(n),
                top_ks=torch.arange(n, dtype=torch.int32),
                min_ps=torch.zeros(n),
                sampling_seed=None,
            )

        seam = row_select.select_rows
        for on_cpu, calls in ((False, 4), (True, 0)):
            batch = info(3, on_cpu)
            with patch.object(row_select, "select_rows", wraps=seam) as select:
                SamplingBatchInfo._filter_params(batch, [2, 0], torch.tensor([2, 0]))
            self.assertEqual(select.call_count, calls)
            self.assertEqual(batch.top_ks.tolist(), [2, 0])

        left, seam = info(3), row_select.concat_rows
        with patch.object(row_select, "concat_rows", wraps=seam) as concat:
            SamplingBatchInfo._merge_params(left, info(2))
        self.assertEqual(concat.call_count, 4)
        self.assertEqual(left.top_ks.tolist(), [0, 1, 2, 0, 1])


if __name__ == "__main__":
    unittest.main()
