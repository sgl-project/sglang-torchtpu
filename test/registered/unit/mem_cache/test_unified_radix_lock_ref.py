"""CPU-only tests for UnifiedRadixCache request lock lifecycle."""

import unittest
from array import array
from types import SimpleNamespace
<<<<<<< HEAD
from unittest.mock import MagicMock

import torch

=======
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import release_kv_cache
>>>>>>> upstream/main
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestUnifiedRadixLockRefScenarios(unittest.TestCase):
    def test_no_insert_without_last_node_skips_lock_release(self):
        cache = object.__new__(UnifiedRadixCache)
<<<<<<< HEAD
=======
        cache.cache_controller = None
>>>>>>> upstream/main
        cache.session = MagicMock()
        cache.session.try_cache_finished_req.return_value = False
        cache.disable = False
        cache.req_to_token_pool = MagicMock()
<<<<<<< HEAD
=======
        cache.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
>>>>>>> upstream/main
        cache.req_to_token_pool.req_to_token = torch.arange(8).reshape(1, 8)
        cache.free_kv_row = MagicMock()
        cache._dec_req_lock = MagicMock()
        cache._components_tuple = ()
        cache.enable_session_radix_cache = False

<<<<<<< HEAD
        kv = SimpleNamespace(req_pool_idx=0, cache_protected_len=0)
=======
        kv = ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=3,
            kv_allocated_len=3,
            cache_protected_len=0,
        )
>>>>>>> upstream/main
        req = SimpleNamespace(
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q"),
            kv=kv,
            last_node=None,
            swa_prefix_lock_released=False,
<<<<<<< HEAD
        )

        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=3)

        cache.free_kv_row.assert_called_once_with(kv, [(0, 3)])
        cache._dec_req_lock.assert_not_called()
=======
            owned_kv_len=lambda: 3,
        )

        with (
            patch(
                "sglang.srt.mem_cache.common.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                "sglang.srt.mem_cache.common.get_serving",
                return_value=SimpleNamespace(strip_thinking_cache=False),
            ),
        ):
            release_kv_cache(req, cache, is_insert=False)

        cache.free_kv_row.assert_called_once_with(kv, [(0, 3)])
        cache._dec_req_lock.assert_not_called()
        cache.req_to_token_pool.free.assert_called_once_with(req)
>>>>>>> upstream/main


if __name__ == "__main__":
    unittest.main()
