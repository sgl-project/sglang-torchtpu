"""Unit tests for the TPU raiden KV-offload host pool and future-event.

These cover the hardware-agnostic logic only (page-granular allocation,
token->page id conversion, the RaidenFuture event adapter). The actual raiden
DMA path requires a TPU + the built raiden extension and is validated on-device.

The host pool is constructed via ``object.__new__`` so its ``__init__`` (which
builds a real ``RaidenKVTransfer`` over a torch_tpu device pool) is bypassed; we
inject a fake transfer that records the page ids it is handed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import inspect
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import BaseKVSender, StateType
from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVManager,
    RaidenKVSender,
    get_raiden_kv_caches,
)
from sglang.srt.hardware_backend.tpu.kv_pool_host import (
    RAIDEN_LAYOUT,
    TpuRaidenHostKVCache,
)
from sglang.srt.hardware_backend.tpu.raiden_cache_controller import (
    TpuRaidenHiCacheController,
)
from sglang.srt.hardware_backend.tpu.raiden_transfer import (
    NOOP_EVENT,
    CombinedFutureEvent,
    RaidenFutureEvent,
)
from sglang.srt.managers.cache_controller import CacheOperation
from sglang.test.test_utils import CustomTestCase


class _FakeTransfer:
    """Records (src, dst) page ids passed to backup/load."""

    def __init__(self):
        self.backups = []
        self.loads = []

    def backup(self, device_pages, host_pages):
        self.backups.append((list(device_pages), list(host_pages)))
        return RaidenFutureEvent(None)  # immediately "ready"

    def load(self, host_pages, device_pages):
        self.loads.append((list(host_pages), list(device_pages)))
        return RaidenFutureEvent(None)


def _make_host_pool(page_size=4, page_num=3):
    pool = object.__new__(TpuRaidenHostKVCache)
    pool.page_size = page_size
    pool.page_num = page_num
    pool.size = page_size * page_num
    pool.layout = RAIDEN_LAYOUT
    import threading

    pool.lock = threading.RLock()
    pool.transfer = _FakeTransfer()
    pool.clear()
    return pool


class TestRaidenHostPoolPaging(CustomTestCase):
    def test_sender_matches_base_api_contract(self):
        for method_name in ("__init__", "send"):
            base_params = inspect.signature(
                getattr(BaseKVSender, method_name)
            ).parameters
            raiden_params = inspect.signature(
                getattr(RaidenKVSender, method_name)
            ).parameters
            self.assertEqual(list(raiden_params), list(base_params))

    def test_uniform_kv_caches_are_returned(self):
        class _Pool:
            start_layer = 0
            layer_num = 2
            buffers = [torch.empty(3, 4), torch.empty(3, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        pool = _Pool()
        groups = get_raiden_kv_caches(pool)
        self.assertEqual(groups.full, pool.buffers)
        self.assertEqual(groups.swa, [])

    def test_mixed_block_counts_require_uniform_swa_memory(self):
        class _Pool:
            start_layer = 0
            layer_num = 2
            buffers = [torch.empty(2, 4), torch.empty(3, 4)]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        with self.assertRaisesRegex(RuntimeError, "uniform full KV block counts"):
            get_raiden_kv_caches(_Pool())

    def test_hybrid_swa_caches_are_grouped_by_page_namespace(self):
        class _Pool:
            start_layer = 0
            layer_num = 4
            layers_mapping = {
                0: (0, False),
                1: (0, True),
                2: (1, False),
                3: (1, True),
            }
            buffers = [
                torch.empty(3, 4),
                torch.empty(2, 4),
                torch.empty(3, 4),
                torch.empty(2, 4),
            ]

            def get_fused_kv_buffer(self, layer_id):
                return self.buffers[layer_id]

        groups = get_raiden_kv_caches(_Pool())
        self.assertEqual(groups.full, [_Pool.buffers[0], _Pool.buffers[2]])
        self.assertEqual(groups.swa, [_Pool.buffers[1], _Pool.buffers[3]])

    def test_hybrid_swa_max_blocks_is_sized_from_sliding_window(self):
        mgr = object.__new__(RaidenKVManager)
        mgr.kv_args = SimpleNamespace(page_size=128, sliding_window_size=4096)

        with patch.dict(
            os.environ,
            {"SGLANG_RAIDEN_SWA_MAX_BLOCKS": "0"},
        ):
            self.assertEqual(mgr._swa_max_request_blocks(), 33)

    def test_hybrid_swa_max_blocks_override_is_independent(self):
        mgr = object.__new__(RaidenKVManager)
        mgr.kv_args = SimpleNamespace(page_size=128, sliding_window_size=4096)

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAIDEN_MAX_BLOCKS": "7",
                "SGLANG_RAIDEN_SWA_MAX_BLOCKS": "11",
            },
        ):
            self.assertEqual(mgr._swa_max_request_blocks(), 11)

    def test_symmetric_pull_rendezvous_works_with_reshard_enabled(self):
        class _Engine:
            def __init__(self):
                self.register_read = Mock()

        room = 17
        blocks = [3, 9]
        for finalize_first in (False, True):
            with self.subTest(finalize_first=finalize_first):
                mgr = object.__new__(RaidenKVManager)
                mgr._state_lock = threading.Lock()
                mgr._push_lock = threading.Lock()
                mgr._send_blocks = {}
                mgr._send_aux = {}
                mgr._decode_addr = {}
                mgr._replied = set()
                mgr._reshard_req = {}
                mgr._reshard_finalized = set()
                mgr._reshard_enabled = True
                mgr._engine = _Engine()
                mgr._swa_engine = None
                mgr._swa_kv_caches = []
                mgr._send_swa_blocks = {}
                mgr.swa_raiden_endpoint = None
                mgr.raiden_endpoint = "127.0.0.1:31020"
                socket = Mock()
                mgr._connect = Mock(return_value=socket)

                mgr.append_send_blocks(room, blocks)
                if finalize_first:
                    mgr.finalize_send(room, None)
                    mgr._engine.register_read.assert_not_called()
                    mgr._decode_addr[room] = ("127.0.0.1", 31002)
                    mgr._maybe_reply(room)
                else:
                    mgr._decode_addr[room] = ("127.0.0.1", 31002)
                    mgr._maybe_reply(room)
                    mgr._engine.register_read.assert_not_called()
                    mgr.finalize_send(room, None)

                mgr._engine.register_read.assert_called_once_with(
                    str(room), room, blocks
                )

    def test_hybrid_swa_rendezvous_registers_both_engines(self):
        room = 23
        full_blocks = [3, 9]
        swa_blocks = [5, 7]
        mgr = object.__new__(RaidenKVManager)
        mgr.kv_args = Mock(state_types=[StateType.SWA])
        mgr._state_lock = threading.Lock()
        mgr._push_lock = threading.Lock()
        mgr._send_blocks = {}
        mgr._send_swa_blocks = {}
        mgr._send_aux = {}
        mgr._decode_addr = {}
        mgr._replied = set()
        mgr._reshard_req = {}
        mgr._reshard_finalized = set()
        mgr._engine = Mock()
        mgr._swa_engine = Mock()
        mgr._swa_kv_caches = [object()]
        mgr.raiden_endpoint = "127.0.0.1:31020"
        mgr.swa_raiden_endpoint = "127.0.0.1:31022"
        mgr._completed_send = set()
        socket = Mock()
        mgr._connect = Mock(return_value=socket)

        swa_from_state = mgr._swa_blocks_from_state_indices([swa_blocks])
        mgr.append_send_blocks(room, full_blocks, swa_from_state)
        mgr._decode_addr[room] = ("127.0.0.1", 31002)
        mgr.finalize_send(room, None)

        mgr._engine.register_read.assert_called_once_with(str(room), room, full_blocks)
        mgr._swa_engine.register_read.assert_called_once_with(
            mgr._swa_req_id(room), room, swa_blocks
        )
        payload = socket.send_multipart.call_args.args[0]
        self.assertEqual(len(payload), 7)
        self.assertEqual(payload[5], b"127.0.0.1:31022")

        self.assertFalse(mgr.is_send_done(room))
        mgr._completed_send.update((str(room), mgr._swa_req_id(room)))
        self.assertTrue(mgr.is_send_done(room))

    def test_hybrid_swa_decode_starts_both_pulls_with_separate_pages(self):
        room = 29
        mgr = object.__new__(RaidenKVManager)
        mgr._state_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = Mock()
        mgr._local_blocks = {room: [8, 9]}
        mgr._local_swa_blocks = {room: [7]}
        mgr._remote_blocks = {room: [1, 2, 3]}
        mgr._remote_swa_blocks = {room: [4, 5]}
        mgr._remote_endpoint = {room: "127.0.0.1:31020"}
        mgr._remote_swa_endpoint = {room: "127.0.0.1:31022"}

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            str(room), room, "127.0.0.1:31020", [2, 3], [8, 9]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            mgr._swa_req_id(room), room, "127.0.0.1:31022", [5], [7]
        )

    def test_hybrid_swa_rejects_asymmetric_tp_before_reshard(self):
        mgr = object.__new__(RaidenKVManager)
        mgr._swa_engine = Mock()
        mgr._reshard_enabled = True
        mgr._register_reshard_request = Mock()

        with self.assertRaisesRegex(
            NotImplementedError,
            "cannot reshard hybrid SWA KV caches.*2 prefill sources",
        ):
            mgr.register_decode_request(
                room=31,
                local_blocks=[8, 9],
                local_swa_blocks=[6, 7],
                aux_index=None,
                bootstrap_infos=[{}, {}],
            )

        mgr._register_reshard_request.assert_not_called()

    def test_alloc_returns_page_aligned_token_runs(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        idx = pool.alloc(8)
        self.assertEqual(idx.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(pool._to_page_ids(idx), [0, 1])
        self.assertEqual(pool.available_size(), 4)  # 1 page of 4 left

    def test_alloc_requires_page_multiple(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        with self.assertRaises(AssertionError):
            pool.alloc(6)

    def test_alloc_returns_none_when_full(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        self.assertIsNone(pool.alloc(16))  # 4 pages requested, only 3 exist

    def test_free_recycles_pages(self):
        pool = _make_host_pool(page_size=4, page_num=3)
        idx = pool.alloc(8)
        self.assertEqual(pool.available_size(), 4)
        pool.free(idx)
        self.assertEqual(pool.available_size(), 12)
        # A fresh full allocation must now succeed again.
        self.assertIsNotNone(pool.alloc(12))

    def test_to_page_ids_after_fragmentation(self):
        # Free a middle page, re-alloc, and confirm page ids stay whole/aligned.
        pool = _make_host_pool(page_size=4, page_num=3)
        a = pool.alloc(4)  # page 0
        b = pool.alloc(4)  # page 1
        pool.free(a)  # page 0 back to the tail of the free list
        c = pool.alloc(4)  # page 2 (front of free list), then page 0 next
        self.assertEqual(pool._to_page_ids(b), [1])
        self.assertEqual(pool._to_page_ids(c), [2])

    def test_backup_converts_token_indices_to_page_ids(self):
        pool = _make_host_pool(page_size=4, page_num=8)
        host_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        device_idx = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int64)
        ev = pool.backup_from_device_all_layer(None, host_idx, device_idx, "raiden")
        self.assertTrue(ev.query())
        dev_pages, host_pages = pool.transfer.backups[0]
        self.assertEqual(dev_pages, [2, 5])  # 8//4, 20//4
        self.assertEqual(host_pages, [0, 1])

    def test_load_all_layer_converts_page_ids(self):
        pool = _make_host_pool(page_size=4, page_num=8)
        host_idx = torch.tensor([12, 13, 14, 15], dtype=torch.int64)
        device_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        pool.load_all_layer(None, host_idx, device_idx)
        host_pages, dev_pages = pool.transfer.loads[0]
        self.assertEqual(host_pages, [3])
        self.assertEqual(dev_pages, [0])

    def test_per_layer_load_is_rejected(self):
        pool = _make_host_pool()
        with self.assertRaises(NotImplementedError):
            pool.load_to_device_per_layer(None, None, None, 0, "raiden")

    def test_l3_seams_raise(self):
        pool = _make_host_pool()
        with self.assertRaises(NotImplementedError):
            pool.get_data_page(0)
        with self.assertRaises(NotImplementedError):
            pool.get_dummy_flat_data_page()
        with self.assertRaises(NotImplementedError):
            pool.set_from_flat_data_page(0, torch.empty(0))


class _FakeFuture:
    def __init__(self, ready_after=0):
        self._calls = 0
        self._ready_after = ready_after

    def is_ready(self):
        ready = self._calls >= self._ready_after
        self._calls += 1
        return ready


class TestRaidenFutureEvent(CustomTestCase):
    def test_noop_event_is_always_ready(self):
        self.assertTrue(NOOP_EVENT.query())
        NOOP_EVENT.synchronize()  # must not block

    def test_query_reflects_future(self):
        fut = _FakeFuture(ready_after=2)
        ev = RaidenFutureEvent(fut)
        self.assertFalse(ev.query())
        self.assertFalse(ev.query())
        self.assertTrue(ev.query())

    def test_synchronize_polls_until_ready(self):
        ev = RaidenFutureEvent(_FakeFuture(ready_after=3))
        ev.synchronize()  # returns once is_ready() flips True

    def test_record_and_wait_are_noops(self):
        ev = RaidenFutureEvent(_FakeFuture())
        ev.record()
        ev.wait()


class TestCombinedFutureEvent(CustomTestCase):
    def test_ready_only_when_all_events_ready(self):
        """The combined event must gate on EVERY underlying DMA: an ack that
        reports ready while the draft pool's transfer is still in flight would
        free/reuse pages mid-DMA (silent corruption)."""
        fast = RaidenFutureEvent(_FakeFuture(ready_after=0))
        slow = RaidenFutureEvent(_FakeFuture(ready_after=2))
        ev = CombinedFutureEvent([fast, slow])
        self.assertFalse(ev.query())
        self.assertFalse(ev.query())
        self.assertTrue(ev.query())
        ev.synchronize()  # must not block once all are ready


class _FakeHostPool:
    """Raiden-host-pool stand-in recording the transfer calls it receives."""

    def __init__(self, page_size=4, page_num=8, event=None):
        self.page_size = page_size
        self.page_num = page_num
        self.size = page_size * page_num
        self._event = event if event is not None else RaidenFutureEvent(None)
        self.backup_calls = []
        self.load_calls = []

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self.backup_calls.append((device_pool, host_indices, device_indices))
        return self._event

    def load_all_layer(self, device_pool, host_indices, device_indices):
        self.load_calls.append((device_pool, host_indices, device_indices))
        return self._event


class _FakeBatchCounter:
    def __init__(self):
        self.events = {}

    def update_producer(self):
        return 0

    def set_event(self, producer_id, event):
        self.events[producer_id] = event


def _make_controller(*, draft_pool=None):
    ctl = object.__new__(TpuRaidenHiCacheController)
    ctl.io_backend = "raiden"
    ctl.write_queue = []
    ctl.load_queue = []
    ctl.ack_write_queue = []
    ctl.ack_load_queue = []
    ctl.enable_storage = False
    ctl.mem_pool_device = object()
    ctl.mem_pool_host = _FakeHostPool()
    ctl.has_draft = False
    ctl.mem_pool_device_draft = None
    ctl.mem_pool_host_draft = None
    ctl.layer_done_counter = _FakeBatchCounter()
    ctl._async_h2d = False
    ctl._load_worker = None
    if draft_pool is not None:
        ctl.has_draft = True
        ctl.mem_pool_device_draft = object()
        ctl.mem_pool_host_draft = draft_pool
    return ctl


class TestRaidenControllerDraftPiggyback(CustomTestCase):
    """Spec decoding stores draft-model KV in a second pool indexed by the same
    token slots as the target pool. Write-back / load-back that moves only the
    target pool leaves draft KV stale on every host-tier hit, which silently
    collapses speculation instead of erroring. These pin the piggyback contract:
    same indices to both pools, one ack gated on both DMAs."""

    def test_write_back_piggybacks_draft_pool(self):
        slow_draft = RaidenFutureEvent(_FakeFuture(ready_after=1))
        draft_pool = _FakeHostPool(event=slow_draft)
        ctl = _make_controller(draft_pool=draft_pool)
        host_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_idx = torch.tensor([8, 9, 10, 11], dtype=torch.int64)
        ctl.write_queue.append(CacheOperation(host_idx, device_idx, node_id=1))

        ctl.start_writing()

        (target_call,) = ctl.mem_pool_host.backup_calls
        (draft_call,) = draft_pool.backup_calls
        self.assertIs(target_call[0], ctl.mem_pool_device)
        self.assertIs(draft_call[0], ctl.mem_pool_device_draft)
        # Identical host/device indices in both pools (1-to-1 slot premise).
        self.assertIs(draft_call[1], target_call[1])
        self.assertIs(draft_call[2], target_call[2])
        # The single ack must not report ready while the draft DMA is in flight.
        (ack,) = ctl.ack_write_queue
        self.assertFalse(ack.finish_event.query())
        self.assertTrue(ack.finish_event.query())

    def test_load_back_piggybacks_draft_pool_and_gates_batch(self):
        draft_pool = _FakeHostPool()
        ctl = _make_controller(draft_pool=draft_pool)
        host_idx = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
        device_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ctl.load_queue.append(CacheOperation(host_idx, device_idx, node_id=2))

        producer_id = ctl.start_loading()

        (target_call,) = ctl.mem_pool_host.load_calls
        (draft_call,) = draft_pool.load_calls
        self.assertIs(target_call[0], ctl.mem_pool_device)
        self.assertIs(draft_call[0], ctl.mem_pool_device_draft)
        self.assertIs(draft_call[1], target_call[1])
        self.assertIs(draft_call[2], target_call[2])
        # The whole-batch gate must hold the event covering BOTH pools' DMAs.
        gate_event = ctl.layer_done_counter.events[producer_id]
        self.assertIsInstance(gate_event, CombinedFutureEvent)
        (ack,) = ctl.ack_load_queue
        self.assertIs(ack.finish_event, gate_event)

    def test_set_draft_kv_pool_rejects_mismatched_paging(self):
        """Piggybacking reuses the primary pool's host page ids verbatim, so a
        draft host pool that pages differently would corrupt addressing; it must
        be rejected at registration, not fail silently at transfer time."""
        ctl = _make_controller()
        with self.assertRaises(ValueError):
            ctl.set_draft_kv_pool(Mock(), _FakeHostPool(page_size=8))
        with self.assertRaises(ValueError):
            ctl.set_draft_kv_pool(Mock(), _FakeHostPool(page_num=4))
        with self.assertRaises(TypeError):
            ctl.set_draft_kv_pool(Mock(), object())
        self.assertFalse(ctl.has_draft)

    def test_set_draft_kv_pool_accepts_matching_paging(self):
        ctl = _make_controller()
        draft_pool = _FakeHostPool()
        draft_device_pool = Mock()
        ctl.set_draft_kv_pool(draft_device_pool, draft_pool)
        self.assertTrue(ctl.has_draft)
        self.assertIs(ctl.mem_pool_host_draft, draft_pool)
        # The draft forward must gate on the shared whole-batch reload event.
        draft_device_pool.register_layer_transfer_counter.assert_called_once_with(
            ctl.layer_done_counter
        )


if __name__ == "__main__":
    unittest.main()
