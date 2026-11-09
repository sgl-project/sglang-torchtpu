import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import hicache_async_ready_counts as mod

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Queue:
    def __init__(self, n):
        self.n = n

    def qsize(self):
        return self.n


class _Cache:
    pass


def _cache(*, acks, queue):
    cc = SimpleNamespace(
        ack_write_queue=[],
        ack_load_queue=[],
        prefetch_hit_queue=_Queue(queue),
        ack_prefetch_queue=_Queue(0),
        ack_backup_queue=_Queue(0),
        host_mem_release_queue=_Queue(0),
        extra_host_mem_release_queues={},
    )
    cache = _Cache()
    cache.cache_controller = cc
    cache.enable_storage = True
    cache.tp_world_size = 8
    cache.pp_size = 1
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.tp_group = object()
    cache.tree_core = SimpleNamespace(write_back_duplicate_reclaim_digest=7)
    cache._count_ready_acks = lambda q: acks
    return cache


class TestDeferredReadyCounts(CustomTestCase):
    def test_a_step_returns_the_counts_reduced_one_step_earlier(self):
        """The reduce is issued async and consumed next step; nothing blocks the step
        that issues it, and the counts handed back never exceed the live queues."""
        works = []

        def all_reduce(tensor, op, group, async_op):
            works.append(_Work())
            return works[-1]

        cache = _cache(acks=3, queue=2)
        with patch.object(torch.distributed, "all_reduce", all_reduce):
            first = mod.deferred_ready_counts(cache)
            self.assertEqual(first[:3], (0, 0, (0, 0, 0, 0)))
            self.assertFalse(works[0].waited)
            second = mod.deferred_ready_counts(cache)
        self.assertTrue(works[0].waited)
        self.assertEqual(second[:3], (3, 3, (2, 0, 0, 0)))
        # The second send carries only what remains after the drained counts.
        self.assertEqual(works and mod._pending[cache][1].tolist()[:2], [0, 0])

    def test_a_blocking_drain_clamps_the_previous_result_to_the_live_queues(self):
        def all_reduce(tensor, op, group, async_op):
            return _Work()

        cache = _cache(acks=3, queue=2)
        with patch.object(torch.distributed, "all_reduce", all_reduce):
            mod.deferred_ready_counts(cache)
            cache._count_ready_acks = lambda q: 1
            second = mod.deferred_ready_counts(cache)
        self.assertEqual(second[:2], (1, 1))

    def test_install_keeps_single_rank_and_pp_on_the_synchronous_path(self):
        calls = []

        class Cache:
            def _sync_hicache_ready_counts(self):
                calls.append("orig")
                return (0, 0, (), ())

            def reset(self):
                calls.append("reset")

        mod.install(Cache)
        mod.install(Cache)  # idempotent
        c = Cache()
        c.cache_controller, c.pp_size, c.tp_world_size = object(), 1, 1
        c._sync_hicache_ready_counts()
        c.pp_size, c.tp_world_size = 2, 8
        c._sync_hicache_ready_counts()
        c.reset()
        self.assertEqual(calls, ["orig", "orig", "reset"])


if __name__ == "__main__":
    unittest.main()
