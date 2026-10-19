"""SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE defers only mid-prefill chunk inserts.

The insert at the end of a prefill must still run: a follow-up turn diverges right
after the prompt, and its match needs the SWA window that insert keeps.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Tree:
    def __init__(self, defers=True):
        self.defers = defers
        self.calls = []

    def defer_chunk_checkpoint(self, req):
        self.calls.append(("defer", req.extend_range.end))
        return self.defers

    def checkpoint(self, req, *, up_to):
        self.calls.append(("checkpoint", up_to))


def _run(stride, mid_prefill, start=4096, end=8192, tree=None):
    req = SimpleNamespace(
        rid="r",
        finished=lambda: False,
        skip_radix_cache_insert=False,
        extend_range=SimpleNamespace(start=start, end=end),
    )
    tree = tree or _Tree()
    with envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride):
        checkpoint_kv_cache(req, tree, mid_prefill=mid_prefill)
    return tree.calls


class TestDeferChunkInsert(unittest.TestCase):
    def test_mid_prefill_chunk_is_deferred(self):
        self.assertEqual(_run(16384, mid_prefill=True), [("defer", 8192)])

    def test_end_of_prefill_insert_runs(self):
        self.assertEqual(_run(16384, mid_prefill=False), [("checkpoint", 8192)])

    def test_zero_stride_inserts_every_chunk(self):
        self.assertEqual(_run(0, mid_prefill=True), [("checkpoint", 8192)])

    def test_stride_inserts_the_chunk_that_reaches_a_multiple(self):
        # 16K stride, 4K chunks: [12K, 16K) and [30K, 34K) reach a multiple,
        # [16K, 20K) does not.
        self.assertEqual(_run(16384, True, 12288, 16384), [("checkpoint", 16384)])
        self.assertEqual(_run(16384, True, 30720, 34816), [("checkpoint", 34816)])
        self.assertEqual(_run(16384, True, 16384, 20480), [("defer", 20480)])

    def test_cache_that_declines_still_inserts(self):
        calls = _run(16384, mid_prefill=True, tree=_Tree(defers=False))
        self.assertEqual(calls, [("defer", 8192), ("checkpoint", 8192)])


def _swa_cache(**overrides):
    attrs = dict(
        disable=False,
        supports_swa=lambda: True,
        supports_mamba=lambda: False,
        page_size=4,
        _components_tuple=(Mock(),),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int32).reshape(2, 32)
        ),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _req(**overrides):
    attrs = dict(
        session_id=None,
        finished=lambda: False,
        full_untruncated_fill_ids=list(range(30)),
        kv=SimpleNamespace(cache_protected_len=0, cache_inserted_len=0, req_pool_idx=1),
        priority=None,
        kv_rotation_base=None,
        prefix_indices=None,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestUnifiedDeferChunkCheckpoint(unittest.TestCase):
    def test_falls_back_outside_swa_only_trees(self):
        for cache, req in (
            (_swa_cache(supports_swa=lambda: False), _req()),
            (_swa_cache(supports_mamba=lambda: True), _req()),
            (_swa_cache(disable=True), _req()),
            (_swa_cache(), _req(session_id="s")),
        ):
            self.assertFalse(UnifiedRadixCache.defer_chunk_checkpoint(cache, req))
            cache._components_tuple[0].free_out_of_window_slots.assert_not_called()
            self.assertIsNone(req.prefix_indices)


if __name__ == "__main__":
    unittest.main()
