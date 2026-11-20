"""The P/D decode's prefix cache under K3_PROGRAM: a unified tree whose Mamba
component holds no KDA states (sglang_decode_cache)."""

import unittest
from array import array
from unittest import mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.mem_cache_utils import finish_req
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_decode_cache

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
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
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
        acquire_rows(self.pool, self.pool.state, [req])
        req.output_ids = array("q")
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.lock = None
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
        match_kv_cache(nxt, self.cache, nxt.origin_input_ids, cow_mamba=True)
        # The whole cached turn matches without a state, and nothing is copied
        # into the request's slot (the transfer delivers its state).
        self.assertEqual(self.cache.prefix_device_indices(nxt).tolist(), kv.tolist())
        self.assertIsNone(getattr(nxt.kv, "mamba_cow_src_index", None))


if __name__ == "__main__":
    unittest.main()
