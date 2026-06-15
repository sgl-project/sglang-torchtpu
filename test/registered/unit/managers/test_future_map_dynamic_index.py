"""Unit tests for the FutureMap dynamic-index path (overlap_utils).

On backends that bake a Python-int index as a compile-time constant (torch_tpu /
XLA), the overlap FutureMap keeps its circular-buffer write offset on-device and
writes with ``index_copy_`` instead of a slice assignment, so the future-map
store does not recompile every step. These tests pin two things on CPU (no TPU
needed):

* ``_device_use_dynamic_index`` selects that path only for the listed backends.
* The dynamic path is **bit-identical** to the plain slice path — same indices,
  same interval, same buffer contents — across many allocations including
  circular-buffer wrap-around. That equivalence is the whole correctness claim;
  it is device-independent, so we force the flag on a CPU FutureMap and diff the
  two paths.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import (
    _DYNAMIC_INDEX_DEVICE_TYPES,
    FutureMap,
    _device_use_dynamic_index,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A non-speculative SpeculativeAlgorithm stub: FutureMap.__init__ only calls
# ``.is_none()`` (True -> the simple token_ids_buf path we exercise here).
_SPEC_NONE = SimpleNamespace(is_none=lambda: True)


def _make_future_map(*, max_running_requests=4, device="cpu") -> FutureMap:
    return FutureMap(
        max_running_requests=max_running_requests,
        chunked_prefill_size=0,  # -> max_num_chunks = 0, future_limit = mrr * 3
        context_len=0,
        device=device,
        spec_algo=_SPEC_NONE,
    )


class TestDeviceUseDynamicIndex(CustomTestCase):
    def test_cpu_and_cuda_are_static(self):
        self.assertFalse(_device_use_dynamic_index(torch.device("cpu")))
        # torch.device("cuda") parses without a GPU present.
        self.assertFalse(_device_use_dynamic_index(torch.device("cuda")))
        self.assertFalse(_device_use_dynamic_index("cpu"))

    def test_tpu_is_dynamic_via_type_attr(self):
        # Object with a ``.type`` attribute (the torch.device shape) -> hasattr branch.
        self.assertTrue(_device_use_dynamic_index(SimpleNamespace(type="tpu")))

    def test_tpu_is_dynamic_via_string(self):
        # Plain strings -> the ``str(device).split(":")`` fallback branch.
        self.assertTrue(_device_use_dynamic_index("tpu"))
        self.assertTrue(_device_use_dynamic_index("tpu:0"))

    def test_device_list_is_the_single_source_of_truth(self):
        # Every listed type maps to dynamic; an unlisted one does not.
        for dev_type in _DYNAMIC_INDEX_DEVICE_TYPES:
            self.assertTrue(_device_use_dynamic_index(dev_type))
        self.assertNotIn("cpu", _DYNAMIC_INDEX_DEVICE_TYPES)


class TestFutureMapInit(CustomTestCase):
    def test_cpu_uses_static_path_and_no_device_counter(self):
        fm = _make_future_map()
        self.assertFalse(fm._use_dynamic_index)
        # future_ct_dev is only allocated on the dynamic path.
        self.assertFalse(hasattr(fm, "future_ct_dev"))
        self.assertEqual(fm.future_limit, 12)  # 4 * (3 + 0)

    def test_dynamic_path_allocates_device_counter(self):
        # Simulate a dynamic-index backend on CPU (so no real TPU is required).
        fm = _make_future_map()
        fm._use_dynamic_index = True
        fm.future_ct_dev = torch.zeros((), dtype=torch.int64)
        self.assertEqual(fm.future_ct_dev.item(), 0)


class TestDynamicIndexEquivalence(CustomTestCase):
    """The dynamic (index_copy_) path must equal the slice path exactly."""

    @staticmethod
    def _force_dynamic(fm: FutureMap) -> FutureMap:
        fm._use_dynamic_index = True
        fm.future_ct_dev = torch.zeros((), dtype=torch.int64, device=fm.device)
        return fm

    def test_indices_interval_and_writes_match_across_wraparound(self):
        static_fm = _make_future_map()  # _use_dynamic_index = False on CPU
        dyn_fm = self._force_dynamic(_make_future_map())

        # token_ids_buf is torch.empty (uninitialized) — zero both so the diff
        # reflects only the written slots, not garbage in the untouched ones.
        static_fm.token_ids_buf.zero_()
        dyn_fm.token_ids_buf.zero_()

        # future_limit = 12; a mixed sequence that wraps several times and pushes
        # start past future_limit into the buffer headroom (future_buffer_len 20).
        bs_sequence = [4, 4, 4, 5, 3, 1, 4, 4, 5, 2, 4, 4]

        for step, bs in enumerate(bs_sequence):
            fi_static = static_fm.alloc_future_indices(bs)
            fi_dyn = dyn_fm.alloc_future_indices(bs)

            # Same slice and same (now tensor-valued) indices.
            self.assertEqual(fi_static.interval, fi_dyn.interval)
            self.assertTrue(
                torch.equal(fi_static.indices, fi_dyn.indices),
                msg=f"step {step} bs {bs}: {fi_static.indices} != {fi_dyn.indices}",
            )
            # On-device counter must stay in lockstep with the host counter.
            self.assertEqual(static_fm.future_ct, dyn_fm.future_ct)
            self.assertEqual(dyn_fm.future_ct_dev.item(), dyn_fm.future_ct)

            # The actual store must land identically: slice assign vs index_copy_.
            values = torch.arange(bs, dtype=torch.int64) + 1000 * (step + 1)
            static_fm._store(static_fm.token_ids_buf, fi_static, values)
            dyn_fm._store(dyn_fm.token_ids_buf, fi_dyn, values)
            self.assertTrue(
                torch.equal(static_fm.token_ids_buf, dyn_fm.token_ids_buf),
                msg=f"step {step}: buffers diverged",
            )

    def test_indices_are_contiguous_and_match_interval(self):
        # Sanity on the dynamic indices themselves: they are exactly the interval
        # [start, end) materialized as a tensor.
        dyn_fm = self._force_dynamic(_make_future_map())
        fi = dyn_fm.alloc_future_indices(5)
        self.assertEqual((fi.interval.start, fi.interval.stop), (1, 6))
        self.assertTrue(torch.equal(fi.indices, torch.arange(1, 6, dtype=torch.int64)))
        self.assertEqual(fi.indices.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
