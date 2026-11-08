"""CPU checks of the prefill side of the P/D prefix pull on a real full + SWA
tree and allocator, with a fake reverse transfer."""

import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import test_unified_radix_cache_unittest as radix_fixture
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.prefix_pull import DecodePrefixReport
from sglang_tpu.disaggregation.prefix_pull_prefill import PrefixPullFetcher, pull_plan
from sglang_tpu.disaggregation.raiden import pp_reverse
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.envs import envs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4
WINDOW = 8
ROOM = 9


class _Reverse:
    """The stage's ReversePull, as the fetcher sees it."""

    def __init__(self):
        self.started = None
        self.state = (pp_reverse.PENDING, "")
        self.kind = ""
        self.in_flight_polls = 0
        self.forgotten = []

    def start(self, room, **kw):
        self.started = (room, kw)

    def status(self, room):
        return self.state

    def failure_kind(self, room):
        return self.kind

    def in_flight(self, room):
        return self.in_flight_polls > 0

    def forget(self, room):
        self.forgotten.append(room)


class _Metrics:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda value: self.events.append((name, value))


class TestPullPlan(CustomTestCase):
    def test_window_must_lie_inside_a_worthwhile_range(self):
        kw = dict(window=2048, page_size=256, min_tokens=4096)
        self.assertEqual(pull_plan(decode_end=8192, prefill_match=1024, **kw), 6144)
        self.assertIsNone(pull_plan(decode_end=4096, prefill_match=1024, **kw))
        self.assertIsNone(
            pull_plan(
                decode_end=8192,
                prefill_match=1024,
                window=8192,
                page_size=256,
                min_tokens=0,
            )
        )


class TestPrefixPullFetcher(CustomTestCase):
    def setUp(self):
        cfg = CacheConfig(
            page_size=PAGE,
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=WINDOW,
        )
        with mock.patch.object(radix_fixture, "get_device", return_value="cpu"):
            self.cache, self.allocator, self.pool = build_fixture(
                cfg, tree_core_backend="python"
            )
        self.reverse = _Reverse()
        self.polls = 0

        def poll():
            self.polls += 1
            self.reverse.in_flight_polls -= 1

        pipeline = SimpleNamespace(
            lock=threading.RLock(),
            reverse=self.reverse,
            manager=SimpleNamespace(pp_rank=0),
        )
        self.metrics = _Metrics()
        queue = SimpleNamespace(
            kv_manager=SimpleNamespace(_pipeline=pipeline, _drain_engine=poll),
            scheduler=SimpleNamespace(
                tree_cache=self.cache,
                token_to_kv_pool_allocator=self.allocator,
                req_to_token_pool=self.pool,
            ),
        )
        self.fetcher = PrefixPullFetcher(queue, self.metrics)
        # The paged allocator's extend is a Triton kernel; this CPU stand-in keeps
        # its contract for a page-aligned prefix: whole new full pages, SWA slots
        # for the tail only.
        allocator = self.allocator

        def alloc_extend_swa_tail(*, extend_num_tokens, swa_tail_len, **_):
            full = allocator.full_attn_allocator.alloc(extend_num_tokens)
            swa = allocator.swa_attn_allocator.alloc(swa_tail_len)
            if full is None or swa is None:
                return None
            mapping = allocator.full_to_swa_index_mapping
            mapping[full[-swa_tail_len:]] = swa.to(mapping.dtype)
            return full

        patcher = mock.patch.object(
            allocator, "alloc_extend_swa_tail", side_effect=alloc_extend_swa_tail
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.prompt = list(range(100, 140))  # 40 tokens
        # The prefill holds the first two pages, window included.
        self._insert(self.prompt[:8])
        patcher = envs.SGLANG_TPU_PD_PREFIX_PULL_MIN_TOKENS.override(16)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)

    def _insert(self, tokens):
        full = self.allocator.full_attn_allocator.alloc(len(tokens))
        swa = self.allocator.swa_attn_allocator.alloc(len(tokens))
        mapping = self.allocator.full_to_swa_index_mapping
        mapping[full] = swa.to(mapping.dtype)
        self.cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=full))

    def _req(self, pullable=32, held=True):
        req = Req(
            rid="r",
            origin_input_text="",
            origin_input_ids=array("q", self.prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.bootstrap_room = ROOM
        report = DecodePrefixReport(
            prefix_len=16, pullable_len=pullable, pull_held=held
        )
        req.disagg_kv_sender = Mock(spec=RaidenKVSender)
        req.disagg_kv_sender.decode_prefix_report.return_value = report
        return req

    def _match_len(self, tokens):
        return len(
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).device_indices
        )

    def test_landed_pull_is_matched_with_its_window(self):
        free_before = self.allocator.full_attn_allocator.available_size()
        req = self._req()
        self.assertTrue(self.fetcher.start(req))
        self.assertTrue(self.fetcher.is_fetching(req))
        room, kw = self.reverse.started
        self.assertEqual((room, kw["start"], kw["end"]), (ROOM, 8, 32))
        self.assertEqual(len(kw["full"]), 6)  # pages of [8, 32)
        self.assertEqual(len(kw["swa"]), 2)  # window [24, 32)
        self.assertTrue(req.kv.holds_kv)

        self.reverse.state = (pp_reverse.DONE, "")
        self.fetcher.finish(req)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertEqual(self._match_len(self.prompt), 32)
        self.assertEqual(self.metrics.events, [("pulled", 24)])
        self.assertEqual(self.reverse.forgotten, [ROOM])
        # The pulled prefix stays locked until the request is admitted.
        self.assertIn("r", self.fetcher.prefix_locks)
        self.fetcher.release_prefix_lock(req)
        self.assertEqual(self.fetcher.prefix_locks, {})
        self.assertEqual(
            self.allocator.full_attn_allocator.available_size(), free_before - 24
        )

    def test_strict_failure_frees_the_pages_and_fails(self):
        free_before = self.allocator.full_attn_allocator.available_size()
        req = self._req()
        self.fetcher.start(req)
        self.reverse.state = (pp_reverse.FAILED, "decode rank 3: busy")
        self.reverse.kind = pp_reverse.DECODE_REFUSED
        with envs.SGLANG_TPU_PD_PREFIX_PULL_STRICT.override(True):
            self.assertFalse(self.fetcher.abandon(req))
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(
            self.allocator.full_attn_allocator.available_size(), free_before
        )
        self.assertEqual(self.metrics.events, [("failed", pp_reverse.DECODE_REFUSED)])
        self.assertEqual(self._match_len(self.prompt), 8)

    def test_recovery_waits_for_an_in_flight_transfer(self):
        req = self._req()
        self.fetcher.start(req)
        # Another stage failed; this stage's transfer is still landing.
        self.reverse.in_flight_polls = 3
        with envs.SGLANG_TPU_PD_PREFIX_PULL_STRICT.override(False):
            self.assertTrue(self.fetcher.abandon(req))
        self.assertEqual(self.polls, 3)
        self.assertEqual(self.metrics.events, [("fallback", "peer_stage")])
        self.assertFalse(self.fetcher.is_fetching(req))

    def test_no_pull_without_a_held_or_worthwhile_prefix(self):
        self.assertFalse(self.fetcher.start(self._req(held=False)))
        self.assertFalse(self.fetcher.start(self._req(pullable=20)))
        self.assertIsNone(self.reverse.started)


if __name__ == "__main__":
    unittest.main()
