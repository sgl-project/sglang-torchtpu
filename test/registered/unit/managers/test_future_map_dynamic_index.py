"""Unit tests for the FutureMap dynamic-index store path (overlap_utils).

On backends that bake a buffer WRITE offset as a compile-time constant
(torch_tpu / XLA), the overlap FutureMap routes future-buffer writes through
``_store`` with a tensor-indexed ``index_copy_`` instead of a ``buf[indices] =
values`` setitem, so the compiled store graph shape stays static across overlap
steps (no per-step XLA recompile). These tests pin two things on CPU (no TPU
needed):

* ``_device_use_dynamic_index`` selects that path only for the listed backends.
* The dynamic (``index_copy_``) store is **bit-identical** to the plain setitem
  store — same buffer contents — for the unique-row writes the relay issues.
  That equivalence is the whole correctness claim; it is device-independent, so
  we force the flag on a CPU FutureMap and diff the two paths.

Upstream reworked FutureMap from a circular future-index buffer into a
pool-indexed relay (writes keyed by req_pool_idx via publish/stash); the
dynamic-index ``_store`` seam (fork PR #78) is preserved and is what we test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import (
    _DYNAMIC_INDEX_DEVICE_TYPES,
    FutureMap,
    _device_use_dynamic_index,
)
from sglang_tpu.overlap_utils_tpu import (
    _select_device_accept_cache_locs,
    device_accept_resolved_tpu,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A non-speculative SpeculativeAlgorithm stub. FutureMap.__init__ only stores it
# (spec methods are hit lazily on the first stash, which these tests do not do).
_SPEC_NONE = SimpleNamespace(is_none=lambda: True)


def _make_future_map(*, req_pool_size=16, device="cpu") -> FutureMap:
    # FutureMap sizes its per-req_pool_idx buffers from
    # req_to_token_pool.req_to_token.shape[0].
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((req_pool_size, 1), dtype=torch.int64)
    )
    return FutureMap(
        device=device,
        spec_algo=_SPEC_NONE,
        req_to_token_pool=req_to_token_pool,
        needs_cpu_seq_lens=True,
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
    def test_cpu_uses_static_path(self):
        fm = _make_future_map(req_pool_size=8)
        # CPU is not a dynamic-index backend, so writes use the plain setitem.
        self.assertFalse(fm._use_dynamic_index)
        # Buffers are sized by the req->token pool row count.
        self.assertEqual(fm.output_tokens_buf.shape[0], 8)
        self.assertEqual(fm.new_seq_lens_buf.shape[0], 8)


class TestDynamicIndexEquivalence(CustomTestCase):
    """``_store``'s dynamic (index_copy_) path must equal the static setitem path."""

    def test_store_dynamic_matches_static_across_writes(self):
        static_fm = _make_future_map(
            req_pool_size=16
        )  # _use_dynamic_index=False on CPU
        dyn_fm = _make_future_map(req_pool_size=16)
        dyn_fm._use_dynamic_index = True
        self.assertFalse(static_fm._use_dynamic_index)

        # Unique-row writes (the only kind the relay issues): a mix of contiguous
        # intervals, a singleton, and a set that revisits row 0 in a later step.
        index_sets = [
            [0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11],
            [12],
            [13, 14, 15, 0],
        ]
        sbuf = torch.zeros(16, dtype=torch.int64)
        dbuf = torch.zeros(16, dtype=torch.int64)
        for step, idxs in enumerate(index_sets):
            indices = torch.tensor(idxs, dtype=torch.int64)
            values = torch.arange(len(idxs), dtype=torch.int64) + 1000 * (step + 1)
            static_fm._store(sbuf, indices, values)
            dyn_fm._store(dbuf, indices, values)
            self.assertTrue(
                torch.equal(sbuf, dbuf),
                msg=f"step {step} idxs {idxs}: buffers diverged\n{sbuf}\n{dbuf}",
            )

    def test_store_casts_values_to_buffer_dtype(self):
        # _store casts values to the buffer dtype before writing (main #134: the
        # sampler returns int64 ids that must land in the buffer's dtype).
        fm = _make_future_map(req_pool_size=8)
        fm._use_dynamic_index = True
        buf = torch.zeros(8, dtype=torch.int64)
        fm._store(buf, torch.tensor([1, 2]), torch.tensor([5, 6], dtype=torch.int32))
        self.assertEqual(buf[1].item(), 5)
        self.assertEqual(buf[2].item(), 6)


class TestNonEagleRelay(CustomTestCase):
    def test_dflash_and_dspark_skip_eagle_acceptance_fields(self):
        """Non-EAGLE overlap inputs do not carry EAGLE acceptance state."""
        from sglang.srt.managers import overlap_utils
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        for algorithm in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
            for is_tpu in (False, True):
                with self.subTest(algorithm=algorithm, is_tpu=is_tpu):
                    future_map = _make_future_map()
                    future_map.spec_algo = algorithm
                    future_map.need_topk = False
                    future_map.need_hidden_states = False
                    future_map.output_tokens_buf[1] = 29
                    draft_input = DFlashDraftInputV2.create_idle_input(
                        device=torch.device("cpu")
                    )
                    draft_input.future_indices = torch.tensor([1])
                    batch = SimpleNamespace(
                        spec_info=draft_input,
                    )
                    with (
                        patch.object(overlap_utils, "_is_tpu", is_tpu),
                        patch.object(
                            overlap_utils,
                            "device_accept_resolved_tpu",
                            device_accept_resolved_tpu,
                            create=True,
                        ),
                        patch.object(overlap_utils, "_DEBUG_ASSERT", False),
                        patch.object(torch.Tensor, "record_stream"),
                        patch.object(torch, "get_device_module") as device_module,
                    ):
                        device_module.return_value.current_stream.return_value = None
                        future_map._resolve_spec_extras(batch)
                    self.assertEqual(draft_input.bonus_tokens.tolist(), [29])


class TestDeviceAcceptCacheSelection(CustomTestCase):
    def test_selects_window_from_accepted_length(self):
        # Exercise every acceptance length, including both reservation edges,
        # for short proposals as well as the original four-token window. The
        # device seq_lens is the host shadow plus the accepted length.
        for width in (2, 3, 4):
            with self.subTest(width=width):
                candidates = torch.arange(2 * width - 1).repeat(width, 1)
                host_base = torch.arange(10, 10 + width, dtype=torch.int64)
                accept_lens = torch.arange(1, width + 1, dtype=torch.int32)
                actual = _select_device_accept_cache_locs(
                    candidates,
                    host_base + accept_lens,
                    host_base,
                    width=width,
                )
                expected = torch.tensor(
                    [list(range(start, start + width)) for start in range(width)]
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_padding_rows_select_the_zero_window(self):
        # A bucketed decode step pads with seq_lens 0, host_base 0 and an
        # all-zero candidate row; the start clamps into that row.
        candidates = torch.tensor([[5, 6, 7, 8, 9, 10, 11], [0] * 7])
        actual = _select_device_accept_cache_locs(
            candidates,
            torch.tensor([42, 0]),
            torch.tensor([40, 0]),
            width=4,
        )
        self.assertEqual(actual.tolist(), [[6, 7, 8, 9], [0, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
