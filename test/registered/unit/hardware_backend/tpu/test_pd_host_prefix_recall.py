"""Edge cases of the recall from decode RAM into the prefill's RAM
(sglang_tpu.pd_host_prefix_recall.HostPrefixPuller): no host room, room made
by reclaiming copies the device still holds, a READY that does not match the
plan, and a read that fails after it started. Every path must leave the
prefill computing the prompt, with no page leaked and no lock held."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_vp = _load_sibling("test_pd_victim_push")

from sglang_tpu.pd_host_prefix_recall import HostPrefixPuller  # noqa: E402

ROOM = 7


def _manager(*, offer_pages: int, ready_count: int):
    mgr = MagicMock()
    mgr.prefix_offer_is_host.return_value = True
    mgr.take_prefix_offer.return_value = (offer_pages, list(range(offer_pages)))
    mgr.prefix_swa_offer.return_value = (0, [])
    mgr.ask_prefix_pull.return_value = True
    mgr.prefix_pull_ready.return_value = (
        f"{ROOM}#px",
        11,
        "10.0.0.2:5",
        0,
        ready_count,
    )
    mgr.prefix_swa_ready.return_value = ("", 0, "", 0, 0)
    mgr.prefix_ready_is_host.return_value = True
    mgr.prefix_read_state.return_value = "unknown"
    return mgr


class TestHostRecallEdgeCases(CustomTestCase):
    def _build(self, *, host_capacity: int, mgr):
        cache, allocator = _vp.full_cache(kv_size=64, host_capacity=host_capacity)
        cache.tree_core.is_write_back = cache.is_write_back = False
        scheduler = SimpleNamespace(
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1)),
            attn_tp_cpu_group=None,
            attn_cp_cpu_group=None,
        )
        puller = HostPrefixPuller(kv_manager=mgr, scheduler=scheduler, swa_window=None)
        puller._min_pages = 1
        puller._max_inflight_tokens = puller._max_inflight_swa_tokens = 1 << 20
        return cache, allocator, puller

    def _fill_host_with_a_device_copy(self, cache, allocator, *, lock: bool):
        tokens = list(range(100, 116))
        _vp.insert_tokens(cache, allocator, tokens)
        node = _vp.root_node(cache, 100)
        host = cache.cache_controller.host
        cache.tree_core.commit_backup(node.id, host.alloc(len(node.key)), {})
        cache.tree_core._update_duplicate_tracking(node)
        if lock:
            cache.inc_host_lock_ref(node.id)
        self.assertEqual(host.available_size(), 0)
        return host

    def _req(self):
        return SimpleNamespace(
            bootstrap_room=ROOM,
            origin_input_ids=list(range(200, 232)),
            extra_key=None,
            cache_salt=None,
        )

    def test_no_host_room_abandons_the_recall_and_computes(self):
        mgr = _manager(offer_pages=16, ready_count=16)
        cache, allocator, puller = self._build(host_capacity=16, mgr=mgr)
        host = self._fill_host_with_a_device_copy(cache, allocator, lock=True)
        req = self._req()
        puller._plan(req)
        plan = puller._plans[ROOM]
        puller._allocate(ROOM, plan)
        self.assertEqual(puller.stats["no_memory"], 1)
        self.assertNotIn(ROOM, puller._plans)
        mgr.resolve_prefix_pull.assert_called_with(ROOM)
        mgr.start_host_prefix_read.assert_not_called()
        self.assertEqual(host.available_size(), 0)  # nothing leaked, nothing taken
        self.assertFalse(puller.has_pending_deferred_releases())

    def test_room_is_made_by_reclaiming_a_copy_the_device_holds(self):
        mgr = _manager(offer_pages=16, ready_count=16)
        cache, allocator, puller = self._build(host_capacity=16, mgr=mgr)
        host = self._fill_host_with_a_device_copy(cache, allocator, lock=False)
        req = self._req()
        puller._plan(req)
        plan = puller._plans[ROOM]
        puller._allocate(ROOM, plan)
        self.assertEqual(puller.stats["no_memory"], 0)
        self.assertIsNotNone(plan.new_indices)
        self.assertEqual(_vp.device_hit(cache, list(range(100, 116))), 16)
        puller._maybe_start(req, plan)
        self.assertEqual(plan.state, "pulling")
        kwargs = mgr.start_host_prefix_read.call_args.args
        self.assertEqual(kwargs[0], f"{ROOM}#px")
        self.assertEqual(len(kwargs[4]), 16)  # landing pages named to raiden
        self.assertTrue(puller.has_pending_deferred_releases())

    def test_ready_that_does_not_match_the_plan_is_abandoned(self):
        mgr = _manager(offer_pages=16, ready_count=8)
        cache, allocator, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        puller._plan(req)
        plan = puller._plans[ROOM]
        puller._allocate(ROOM, plan)
        before = host.available_size()
        puller._maybe_start(req, plan)
        self.assertEqual(puller.stats["failed"], 1)
        self.assertNotIn(ROOM, puller._plans)
        mgr.start_host_prefix_read.assert_not_called()
        self.assertEqual(host.available_size(), before + 16)  # landing freed

    def test_read_that_fails_after_start_frees_the_landing_once_settled(self):
        mgr = _manager(offer_pages=16, ready_count=16)
        cache, allocator, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        puller._plan(req)
        plan = puller._plans[ROOM]
        puller._allocate(ROOM, plan)
        puller._maybe_start(req, plan)
        free_before = host.available_size()
        # The decode leaves early (prealloc done) while raiden may still write.
        mgr.prefix_read_state.return_value = "pending"
        puller.commit(req)
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(host.available_size(), free_before)  # still owned
        self.assertTrue(puller.has_pending_deferred_releases())
        mgr.prefix_read_state.return_value = "failed"
        puller.resolve_deferred_releases()
        self.assertEqual(host.available_size(), free_before + 16)
        self.assertFalse(puller.has_pending_deferred_releases())


if __name__ == "__main__":
    unittest.main()
