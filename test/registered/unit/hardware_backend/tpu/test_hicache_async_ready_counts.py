import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import hicache_async_ready_counts as mod

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Cache:
    pass


def _cache(*, acks):
    cc = SimpleNamespace(
        ack_write_queue=[object() for _ in range(acks)],
        ack_load_queue=[object() for _ in range(acks)],
    )
    cache = _Cache()
    cache.cache_controller = cc
    cache.enable_storage = False
    cache.tp_world_size = 8
    cache.pp_size = 1
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.tp_group = object()
    cache.tree_core = SimpleNamespace(write_back_duplicate_reclaim_digest=7)
    cache._count_ready_acks = len
    return cache


class TestDeferredReadyCounts(CustomTestCase):
    def test_a_step_returns_the_counts_reduced_one_step_earlier(self):
        """The reduce is issued async and consumed next step; nothing blocks the step
        that issues it."""
        works = []

        def all_reduce(tensor, op, group, async_op):
            works.append(_Work())
            return works[-1]

        cache = _cache(acks=3)
        with patch.object(torch.distributed, "all_reduce", all_reduce):
            first = mod.deferred_ready_counts(cache, cache.tp_group)
            self.assertEqual(first, (0, 0, (), ()))
            self.assertFalse(works[0].waited)
            second = mod.deferred_ready_counts(cache, cache.tp_group)
        self.assertTrue(works[0].waited)
        self.assertEqual(second, (3, 3, (), ()))
        # The second send carries only what remains after the drained counts.
        self.assertEqual(works and mod._pending[cache][1].tolist()[:2], [0, 0])

    def test_a_blocking_drain_leaves_only_the_counted_acks_it_kept(self):
        def all_reduce(tensor, op, group, async_op):
            return _Work()

        cache = _cache(acks=3)
        cc = cache.cache_controller
        with patch.object(torch.distributed, "all_reduce", all_reduce):
            mod.deferred_ready_counts(cache, cache.tp_group)
            # The drain finishes the first two write acks and every load ack,
            # and a load ack arrives that the reduce never counted.
            del cc.ack_write_queue[:2]
            cc.ack_load_queue[:] = [object()]
            second = mod.deferred_ready_counts(cache, cache.tp_group)
        self.assertEqual(second[:2], (1, 0))

    def test_a_partial_drain_pops_the_same_acks_on_every_rank(self):
        """Two ranks with a MIN reduce: a drain keeps ack A and finishes B and C,
        and ack D is done on rank 0 only when last step's result is spent."""

        class _Ack:
            def __init__(self):
                self.done = [False, False]

        rounds = []

        class _MinWork:
            def __init__(self, tensors):
                self.tensors = tensors

            def wait(self):
                low = torch.stack(self.tensors).min(dim=0).values
                for tensor in self.tensors:
                    tensor.copy_(low)

        def all_reduce(tensor, op, group, async_op):
            if not rounds or len(rounds[-1]) == 2:
                rounds.append([])
            rounds[-1].append(tensor)
            return _MinWork(rounds[-1])

        a, b, c, d = (_Ack() for _ in range(4))
        ranks = []
        for rank in range(2):
            cache = _cache(acks=0)
            cache.cache_controller.ack_write_queue = [a, b, c]
            cache._count_ready_acks = lambda q, r=rank: next(
                (i for i, ack in enumerate(q) if not ack.done[r]), len(q)
            )
            ranks.append(cache)
        for ack in (a, b, c):
            ack.done = [True, True]

        def step():
            pops = [
                mod.deferred_ready_counts(cache, cache.tp_group)[0] for cache in ranks
            ]
            for cache, n in zip(ranks, pops):
                del cache.cache_controller.ack_write_queue[:n]
            return pops

        with patch.object(torch.distributed, "all_reduce", all_reduce):
            self.assertEqual(step(), [0, 0])  # A, B and C counted on both ranks
            for cache in ranks:
                cache.cache_controller.ack_write_queue[:] = [a, d]
            d.done = [True, False]
            self.assertEqual(step(), [1, 1])  # A only: nobody counted D yet
            d.done = [True, True]
            self.assertEqual(step(), [0, 0])
            self.assertEqual(step(), [1, 1])  # D, once both ranks counted it
        for cache in ranks:
            self.assertEqual(cache.cache_controller.ack_write_queue, [])

    def test_install_keeps_single_rank_pp_and_storage_on_the_synchronous_path(self):
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
        c.enable_storage = False
        c._sync_hicache_ready_counts()
        c.pp_size, c.tp_world_size = 2, 8
        c._sync_hicache_ready_counts()
        c.reset()
        self.assertEqual(calls, ["orig", "orig", "reset"])

    def test_only_one_wide_attention_group_reduces_asynchronously(self):
        cp, tp = object(), object()
        cache = _cache(acks=0)
        sizes = {}
        with patch.object(
            torch.distributed, "get_world_size", lambda group: sizes[group]
        ):
            cache.attn_cp_group, cache.attn_tp_group = cp, tp
            sizes.update({cp: 1, tp: 4})
            self.assertIs(mod._reduce_group(cache), tp)
            sizes.update({cp: 2, tp: 4})
            self.assertIsNone(mod._reduce_group(cache))
            cache.attn_cp_group = cache.attn_tp_group = None
            self.assertIs(mod._reduce_group(cache), cache.tp_group)

    def test_a_storage_attach_settles_the_reduce_in_flight(self):
        works = []

        def all_reduce(tensor, op, group, async_op):
            works.append(_Work())
            return works[-1]

        class Cache(_Cache):
            def _sync_hicache_ready_counts(self):
                return (0, 0, (), ())

            def reset(self):
                pass

        mod.install(Cache)
        c = Cache()
        c.__dict__.update(_cache(acks=2).__dict__)
        with patch.object(torch.distributed, "all_reduce", all_reduce):
            c._sync_hicache_ready_counts()
            self.assertIn(c, mod._pending)
            c.enable_storage = True
            c._sync_hicache_ready_counts()
        self.assertTrue(works[0].waited)
        self.assertNotIn(c, mod._pending)


if __name__ == "__main__":
    unittest.main()
