"""CPU-only unit tests for SGLANG_PD_DECODE_KV_ONLY_CHECKPOINT: a PD decode's
radix cache keeps and matches a request's KV without a recurrent-state
checkpoint, since the prefill sends the state at the end of every prompt."""

import unittest
from array import array
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.capacity import reclaim_for_prealloc
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_disagg
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4
PROMPT = list(range(1, 17))
KV_SIZE = 64


def _build():
    cfg = CacheConfig(
        page_size=PAGE,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        kv_size=KV_SIZE,
        max_context_len=64,
        mamba_cache_size=8,
        hybrid_cache_checkpoints_level=2,
    )
    with patch(
        "sglang.test.kits.unified_radix_cache.fixtures.get_device",
        return_value="cpu",
    ):
        return build_fixture(cfg, mamba_cache_chunk_size=PAGE)


def _req(cache, allocator, pool, tokens, rid="r"):
    """A request whose prompt KV sits in fresh pages and which tracked no
    state while decoding, as a short answer on a decode worker."""
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(pool, pool.state, [req])
    req.output_ids = array("q")
    req.refresh_fill_ids()
    req.kv.kv_committed_len = len(tokens)
    req.kv.kv_allocated_len = len(tokens)
    req.kv.cache_protected_len = 0
    pool.write(
        (req.kv.req_pool_idx, slice(0, len(tokens))), allocator.alloc(len(tokens))
    )
    req.last_node = cache.root_node_handle()
    req.lock = None
    req.extend_end = len(tokens)
    return req


def _finish(cache, req):
    req.finished_reason = FINISH_LENGTH(length=0)
    cache.checkpoint(req, up_to=len(req.origin_input_ids))


def _cached_len(cache, tokens, **params):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)), **params)
    ).device_prefix_len


class TestPdDecodeKvOnlyCheckpoint(CustomTestCase):
    def test_a_decode_keeps_and_matches_kv_without_a_state(self):
        cache, allocator, pool = _build()
        free_slots = pool.mamba_allocator.available_size()
        with (
            envs.SGLANG_PD_DECODE_KV_ONLY_CHECKPOINT.override(True),
            get_disagg().override(disaggregation_mode="decode"),
        ):
            req = _req(cache, allocator, pool, PROMPT)
            _finish(cache, req)
            # The whole prompt is kept, and the request's state slots are freed:
            # the tree holds none.
            self.assertEqual(_cached_len(cache, PROMPT), len(PROMPT))
            self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
            cache.unlock(req.lock)
            cache.sanity_check()

            # The next turn matches the kept pages and copies no state.
            turn = _req(cache, allocator, pool, PROMPT + [99, 98, 97, 96], rid="t")
            result = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(array("q", turn.origin_input_ids)),
                    req=turn,
                    cow_mamba=True,
                )
            )
            self.assertEqual(result.device_prefix_len, len(PROMPT))
            self.assertIsNone(turn.kv.mamba_cow_src_index)

            # A preallocation takes back the unlocked pages it needs
            # (DecodePreallocQueue._reclaim_prealloc_capacity), and no more
            # than the pool has.
            free = allocator.available_size()
            self.assertIsNone(
                reclaim_for_prealloc(
                    cache, allocator, KVDemand(full_tokens=free + len(PROMPT))
                )
            )
            self.assertEqual(allocator.available_size(), free + len(PROMPT))
            self.assertEqual(_cached_len(cache, PROMPT), 0)
            self.assertIn(
                "insufficient",
                reclaim_for_prealloc(
                    cache, allocator, KVDemand(full_tokens=free + len(PROMPT) + PAGE)
                ),
            )
            cache.sanity_check()

    def test_off_or_on_another_role_a_state_is_still_required(self):
        for flag, mode in ((False, "decode"), (True, "prefill"), (True, "null")):
            with self.subTest(flag=flag, mode=mode):
                cache, allocator, pool = _build()
                with (
                    envs.SGLANG_PD_DECODE_KV_ONLY_CHECKPOINT.override(flag),
                    get_disagg().override(disaggregation_mode=mode),
                ):
                    req = _req(cache, allocator, pool, PROMPT)
                    _finish(cache, req)
                    self.assertEqual(_cached_len(cache, PROMPT), 0)
                    cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
