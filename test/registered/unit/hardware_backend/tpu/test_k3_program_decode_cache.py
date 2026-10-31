"""The P/D decode's prefix cache under K3_PROGRAM: a unified tree whose Mamba
component holds no KDA states (sglang_decode_cache)."""

import os
import sys
import unittest
from array import array
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mem_cache"))

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.schedule_policy import match_prefix_for_req  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams  # noqa: E402
from sglang.srt.mem_cache.unified_cache.components.base import (  # noqa: E402
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.mem_cache_utils import finish_req  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402
from sglang_tpu.k3_programs import sglang_decode_cache  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDecodePrefixCache(CustomTestCase):
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=8,
        kv_size=64,
        max_context_len=64,
    )

    def setUp(self):
        with mock.patch(
            "test_unified_radix_cache_unittest.get_device", return_value="cpu"
        ):
            built, self.allocator, self.pool = build_fixture(self.cfg)
        params = built.cache_init_params
        sglang_decode_cache.install(params)
        self.cache = UnifiedRadixCache(params=params)

    def _req(self, rid, tokens):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.pool.alloc([req])
        req.output_ids = array("q")
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.lock_receipt = DecLockRefParams()
        req.extra_key = None
        return req

    def test_a_finished_turn_is_the_next_turns_prefix(self):
        free_slots = self.pool.mamba_allocator.available_size()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        req = self._req("turn-1", tokens)
        self.assertLess(self.pool.mamba_allocator.available_size(), free_slots)
        kv = self.allocator.alloc(len(tokens))
        self.pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), kv)
        req.last_node = self.cache.root_node_handle()
        finish_req(self.cache, req, req.owned_kv_len())
        # Its slot is back and the tree keeps no state, only its pages.
        self.assertEqual(self.pool.mamba_allocator.available_size(), free_slots)
        self.assertEqual(self.cache.mamba_evictable_size(), 0)

        nxt = self._req("turn-2", tokens + [9, 10])
        result = match_prefix_for_req(
            self.cache, nxt, nxt.origin_input_ids, cow_mamba=True, include_req=True
        )
        # The whole cached turn matches without a state, and nothing is copied
        # into the request's slot (the transfer delivers its state).
        self.assertEqual(result.device_indices.tolist(), kv.tolist())
        self.assertIsNone(getattr(nxt.kv, "mamba_cow_src_index", None))


if __name__ == "__main__":
    unittest.main()
