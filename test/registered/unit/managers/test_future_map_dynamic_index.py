"""CPU equivalence of shared setitem and plugin dynamic-index relay storage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import (
    FutureMap,
)
from sglang_tpu.future_map import TpuFutureMap
from sglang_tpu.overlap_utils_tpu import (
    _select_device_accept_cache_locs,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A non-speculative SpeculativeAlgorithm stub. FutureMap.__init__ only stores it
# (spec methods are hit lazily on the first stash, which these tests do not do).
_SPEC_NONE = SimpleNamespace(is_none=lambda: True)


def _make_future_map(*, req_pool_size=16, device="cpu", map_cls=FutureMap) -> FutureMap:
    # FutureMap sizes its per-req_pool_idx buffers from
    # req_to_token_pool.req_to_token.shape[0].
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((req_pool_size, 1), dtype=torch.int64)
    )
    return map_cls(
        device=device,
        spec_algo=_SPEC_NONE,
        req_to_token_pool=req_to_token_pool,
        needs_cpu_seq_lens=True,
    )


class TestDynamicIndexEquivalence(CustomTestCase):
    """``_store``'s dynamic (index_copy_) path must equal the static setitem path."""

    def test_store_dynamic_matches_static_across_writes(self):
        static_fm = _make_future_map(req_pool_size=16)
        dyn_fm = _make_future_map(req_pool_size=16, map_cls=TpuFutureMap)

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
        fm = _make_future_map(req_pool_size=8, map_cls=TpuFutureMap)
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
            for map_cls in (FutureMap, TpuFutureMap):
                with self.subTest(algorithm=algorithm, map_cls=map_cls):
                    future_map = _make_future_map(map_cls=map_cls)
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
        # for short proposals as well as the original four-token window.
        for width in (2, 3, 4):
            with self.subTest(width=width):
                candidates = torch.arange(2 * width - 1).repeat(width, 1)
                actual = _select_device_accept_cache_locs(
                    candidates,
                    torch.arange(1, width + 1, dtype=torch.int32),
                    width=width,
                )
                expected = torch.tensor(
                    [list(range(start, start + width)) for start in range(width)]
                )
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
