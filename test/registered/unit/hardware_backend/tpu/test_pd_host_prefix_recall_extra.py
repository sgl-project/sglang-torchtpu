"""Lower-priority unit tests of the recall from decode RAM (sglang_tpu/pd_host_prefix_recall.py): no host room, room made by reclaiming copies the device still holds, a READY that does not match the plan, a read that fails after it started, and less common lifecycle steps. Fixtures come from test_pd_host_prefix_recall.py."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_main = _load_sibling("test_pd_host_prefix_recall")
ROOM = _main.ROOM
_DecodeQueueHooksFixture = _main._DecodeQueueHooksFixture
_HostRecallEdgeCasesFixture = _main._HostRecallEdgeCasesFixture
_RecallLifecycleFixture = _main._RecallLifecycleFixture
_manager = _main._manager
_vp = _main._vp


class TestHostRecallEdgeCasesExtra(_HostRecallEdgeCasesFixture, CustomTestCase):
    def test_no_host_room_abandons_the_recall_and_computes(self):
        mgr = _manager(offer_pages=16)
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
        mgr = _manager(offer_pages=16)
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

    def test_refused_pull_is_abandoned(self):
        mgr = _manager(offer_pages=16, refused=True)
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
        mgr = _manager(offer_pages=16)
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


class TestRecallLifecycleExtra(
    _RecallLifecycleFixture, _HostRecallEdgeCasesFixture, CustomTestCase
):
    def test_room_settled_on_any_rank_is_resolved_on_all(self):
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        req = self._req()
        puller._plan(req)
        # [offered, settled, done, kept, ready] after the all-rank MIN: some
        # rank settled the room with no plan.
        with patch.object(puller, "_all_ranks_reduce", return_value=[[0, 0, 0, 0, 0]]):
            puller.advance([req])
        self.assertNotIn(ROOM, puller._plans)
        mgr.resolve_prefix_pull.assert_called_with(ROOM)

    def test_inflight_recalls_never_take_more_than_their_share_of_host_ram(self):
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        puller._max_inflight_tokens = 16
        first = self._req()
        puller._plan(first)
        self.assertIn(ROOM, puller._plans)
        second = self._req()
        second.bootstrap_room = ROOM + 1
        puller._plan(second)
        self.assertNotIn(ROOM + 1, puller._plans)
        self.assertEqual(puller.stats["over_budget"], 1)
        mgr.resolve_prefix_pull.assert_called_with(ROOM + 1)


class TestDecodeQueueHooksExtra(_DecodeQueueHooksFixture, CustomTestCase):
    def test_pop_releases_the_offers_of_departed_requests(self):
        queue, _ = self._queue("a", "b")
        calls = []
        upstream = MagicMock(
            side_effect=lambda q, rids: calls.append("pop") or ["transferred"]
        )
        with patch.object(
            _vp.victim_push,
            "release_departed_host_offers",
            side_effect=lambda q: calls.append("release"),
        ):
            result = _vp.victim_push._pop_transferred(upstream, queue, ["b"])
        self.assertEqual(result, ["transferred"])
        self.assertEqual(calls, ["pop", "release"])
        upstream.assert_called_once_with(queue, ["b"])

    def test_empty_queue_still_releases(self):
        queue, _ = self._queue()
        upstream = MagicMock(return_value=[])
        with patch.object(_vp.victim_push, "release_departed_host_offers") as release:
            _vp.victim_push._pop_transferred(upstream, queue)
            # A request can leave by abort while the queue is empty of work.
            release.assert_called_once_with(queue)


if __name__ == "__main__":
    unittest.main()
