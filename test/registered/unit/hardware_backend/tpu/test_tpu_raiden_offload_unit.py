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
import sys
import threading
import types
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
from sglang.srt.disaggregation.raiden.host_pool import SlotAdmission
from sglang.srt.disaggregation.raiden.serde import MessageTag, unpack_blocks
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
                mgr._announce_lock = threading.Lock()
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
        mgr._announce_lock = threading.Lock()
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
        mgr._pulling = SlotAdmission(1)
        mgr._engine_max_blocks = 2
        mgr._swa_engine_max_blocks = 1
        mgr._local_blocks = {room: [8, 9]}
        mgr._local_swa_blocks = {room: [7]}
        mgr._remote_blocks = {room: [1, 2, 3]}
        mgr._remote_swa_blocks = {room: [4, 5]}
        mgr._remote_endpoint = {room: "127.0.0.1:31020"}
        mgr._remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr.chunked_transfer = False
        mgr._failed = set()

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            str(room), room, "127.0.0.1:31020", [2, 3], [8, 9]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            mgr._swa_req_id(room), room, "127.0.0.1:31022", [5], [7]
        )

    def test_decode_defers_pull_until_a_staging_slot_frees(self):
        first, second = 29, 30
        mgr = object.__new__(RaidenKVManager)
        mgr.chunked_transfer = False
        mgr._state_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = None
        mgr._pulling = SlotAdmission(1)
        mgr._pulling_cleared = set()
        mgr._engine_max_blocks = 4
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._local_blocks = {first: [8], second: [9]}
        mgr._local_swa_blocks = {}
        mgr._remote_blocks = {first: [1], second: [2]}
        mgr._remote_swa_blocks = {}
        mgr._remote_endpoint = {r: "127.0.0.1:31020" for r in (first, second)}
        mgr._remote_swa_endpoint = {}

        self.assertTrue(mgr.maybe_start_pull(first))
        self.assertFalse(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_count, 1)

        mgr._completed_recv.add(str(first))
        mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        mgr._engine.start_read.assert_called_with(
            str(second), second, "127.0.0.1:31020", [2], [9]
        )

    @staticmethod
    def _decode_mgr(capacity, full_limit, swa_limit=None):
        mgr = object.__new__(RaidenKVManager)
        mgr.chunked_transfer = False
        mgr._state_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._swa_engine = Mock() if swa_limit is not None else None
        mgr._pulling = SlotAdmission(capacity)
        mgr._pulling_cleared = set()
        mgr._engine_max_blocks = full_limit
        mgr._swa_engine_max_blocks = swa_limit or 0
        mgr._completed_send = set()
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._local_blocks = {}
        mgr._local_swa_blocks = {}
        mgr._remote_blocks = {}
        mgr._remote_swa_blocks = {}
        mgr._remote_endpoint = {}
        mgr._remote_swa_endpoint = {}
        mgr.attn_tp_rank = 0
        mgr.record_failure = Mock()
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    @staticmethod
    def _announce(mgr, room, full_pages, swa_pages=0):
        mgr._local_blocks[room] = list(range(100, 100 + full_pages))
        mgr._remote_blocks[room] = list(range(full_pages))
        mgr._remote_endpoint[room] = "127.0.0.1:31020"
        if mgr._swa_engine is not None:
            mgr._local_swa_blocks[room] = list(range(200, 200 + swa_pages))
            mgr._remote_swa_blocks[room] = list(range(swa_pages))
            mgr._remote_swa_endpoint[room] = "127.0.0.1:31022"

    def test_hybrid_swa_checks_each_pool_against_its_own_page_limit(self):
        # gpt-oss at page 64: a 128-token window is 3 pages, the full pool 128.
        room = 31
        mgr = self._decode_mgr(capacity=4, full_limit=128, swa_limit=3)
        self._announce(mgr, room, full_pages=40, swa_pages=3)

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once()
        mgr._swa_engine.start_read.assert_called_once()
        mgr.record_failure.assert_not_called()

        too_wide = 32
        self._announce(mgr, too_wide, full_pages=4, swa_pages=5)
        self.assertFalse(mgr.maybe_start_pull(too_wide))
        self.assertIn(str(too_wide), mgr._failed)
        self.assertIn(
            "SGLANG_RAIDEN_SWA_MAX_BLOCKS", mgr.record_failure.call_args.args[1]
        )
        self.assertEqual(mgr._engine.start_read.call_count, 1)

    def test_cleared_room_keeps_its_slot_until_the_read_settles(self):
        first, second = 33, 34
        mgr = self._decode_mgr(capacity=1, full_limit=8)
        self._announce(mgr, first, full_pages=2)
        self._announce(mgr, second, full_pages=2)
        mgr._send_blocks = {}

        self.assertTrue(mgr.maybe_start_pull(first))
        # The framework aborts the request while raiden still holds the read.
        mgr.clear_room(first)
        self.assertFalse(mgr.maybe_start_pull(second))

        mgr._completed_recv.add(str(first))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertNotIn(str(first), mgr._completed_recv)
        self.assertFalse(mgr._pulling_cleared)

    def test_cleared_settled_room_releases_its_slot_at_once(self):
        first, second = 35, 36
        mgr = self._decode_mgr(capacity=1, full_limit=8)
        self._announce(mgr, first, full_pages=2)
        self._announce(mgr, second, full_pages=2)
        mgr._send_blocks = {}

        self.assertTrue(mgr.maybe_start_pull(first))
        mgr._completed_recv.add(str(first))
        mgr.clear_room(first)
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertNotIn(str(first), mgr._completed_recv)

    def test_hybrid_swa_slot_is_held_until_both_engines_settle(self):
        first, second = 37, 38
        mgr = self._decode_mgr(capacity=1, full_limit=8, swa_limit=3)
        self._announce(mgr, first, full_pages=2, swa_pages=1)
        self._announce(mgr, second, full_pages=2, swa_pages=1)

        self.assertTrue(mgr.maybe_start_pull(first))
        # The full-attention read fails while the sliding-window read is
        # still in flight on the other engine.
        mgr._failed.add(str(first))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertFalse(mgr.maybe_start_pull(second))

        mgr._completed_recv.add(mgr._swa_req_id(first))
        with mgr._state_lock:
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))

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


class TestRaidenChunkedTransfer(CustomTestCase):
    """Chunk announcements and pulls on the symmetric pull path."""

    @staticmethod
    def _prefill():
        mgr = object.__new__(RaidenKVManager)
        mgr._state_lock = threading.Lock()
        mgr._push_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._engine.register_read = Mock(return_value=True)
        mgr._swa_engine = None
        mgr._local_rank = 3
        mgr._chunk_seq = {}
        mgr._uuid_seq = 0
        mgr._held_chunks = {}
        mgr._send_total = {}
        mgr._decode_addr = {}
        mgr._decode_pages = {}
        mgr._send_blocks = {}
        mgr._replied = set()
        mgr._pending_offer = {}
        mgr._announce_pages = 16
        mgr._reshard_req = {}
        mgr._reshard_finalized = set()
        mgr.raiden_endpoint = "127.0.0.1:31020"
        mgr._socket = Mock()
        mgr._connect = Mock(return_value=mgr._socket)
        return mgr

    @staticmethod
    def _pull_req(room, decode_pages):
        return [
            MessageTag.PULL_REQ,
            str(room).encode("ascii"),
            b"10.0.0.2",
            b"31002",
            str(decode_pages).encode("ascii"),
        ]

    def test_prefill_announces_only_the_pages_the_decode_will_pull(self):
        room = 31
        mgr = self._prefill()
        mgr._on_pull_req(self._pull_req(room, 40))
        self.assertEqual(mgr._decode_pages[room], 40)

        # 64 prefill pages; the decode already holds the first 24.
        blocks = list(range(100, 164))
        mgr.offer_chunk(room, 0, blocks[0:16], 64)
        mgr._engine.register_read.assert_not_called()
        mgr._socket.send_multipart.assert_not_called()

        mgr.offer_chunk(room, 16, blocks[16:32], 64)
        mgr._engine.register_read.assert_called_once()
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[24:32])
        msg = mgr._socket.send_multipart.call_args.args[0]
        self.assertEqual(msg[0], MessageTag.CHUNK)
        self.assertEqual(int(msg[4]), 24)
        self.assertEqual(int(msg[5]), 64)
        self.assertEqual(unpack_blocks(msg[6]), blocks[24:32])

        mgr.offer_chunk(room, 32, blocks[32:48], 64)
        self.assertEqual(mgr._engine.register_read.call_count, 2)
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[32:48])

    def test_small_chunks_gather_into_one_announcement(self):
        room = 34
        mgr = self._prefill()
        mgr._on_pull_req(self._pull_req(room, 64))
        blocks = list(range(100, 164))
        mgr.offer_chunk(room, 0, blocks[0:8], 64)
        mgr._engine.register_read.assert_not_called()
        mgr.offer_chunk(room, 8, blocks[8:16], 64)
        mgr._engine.register_read.assert_called_once()
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[0:16])
        self.assertEqual(int(mgr._socket.send_multipart.call_args.args[0][4]), 0)
        self.assertNotIn(room, mgr._pending_offer)

    def test_chunks_offered_before_the_pull_request_are_held_then_filtered(self):
        room = 32
        mgr = self._prefill()
        blocks = list(range(100, 164))
        mgr.offer_chunk(room, 0, blocks[0:16], 64)
        mgr.offer_chunk(room, 16, blocks[16:32], 64)
        mgr._engine.register_read.assert_not_called()

        mgr._on_pull_req(self._pull_req(room, 40))
        self.assertEqual(mgr._engine.register_read.call_count, 1)
        self.assertEqual(mgr._engine.register_read.call_args.args[2], blocks[24:32])
        self.assertNotIn(room, mgr._held_chunks)

    @staticmethod
    def _decode(room, local, capacity):
        mgr = object.__new__(RaidenKVManager)
        mgr._state_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._engine.start_read = Mock(return_value=0)
        mgr._swa_engine = None
        mgr.chunked_transfer = True
        mgr._pulling = SlotAdmission(capacity)
        mgr._engine_max_blocks = 64
        mgr._swa_engine_max_blocks = 2
        mgr.attn_tp_rank = 0
        mgr.record_failure = Mock()
        mgr._local_blocks = {room: local}
        mgr._remote_total = {room: 64}
        mgr._chunk_queue = {}
        mgr._chunk_reads = {}
        mgr._announced_pages = {}
        mgr._final_seen = set()
        mgr._completed_send = set()
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._pulling_cleared = set()
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    def test_closing_read_carries_the_sliding_window_pages(self):
        room = 35
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, capacity=4)
        mgr._swa_engine = Mock()
        mgr._remote_swa_blocks = {room: [4, 5]}
        mgr._local_swa_blocks = {room: [7]}
        mgr._remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr._final_seen.add(room)
        mgr._chunk_queue[room] = [(str(room), room, 48, remote[48:64], endpoint)]

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            "35", room, endpoint, remote[48:64], local[48:64]
        )
        mgr._swa_engine.start_read.assert_called_once_with(
            "swa:35", room, "127.0.0.1:31022", [5], [7]
        )
        mgr._completed_recv.add("35")
        self.assertFalse(mgr.is_recv_done(room))
        mgr._completed_recv.add("swa:35")
        self.assertTrue(mgr.is_recv_done(room))

    def test_decode_keeps_one_chunk_read_in_flight_per_request(self):
        room = 33
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, capacity=4)
        mgr._chunk_queue[room] = [
            ("33#0", 11, 0, remote[0:16], endpoint),
            ("33#1", 12, 16, remote[16:32], endpoint),
        ]

        self.assertTrue(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_called_once_with(
            "33#0", 11, endpoint, remote[0:16], local[0:16]
        )
        # The second chunk waits for the first read even though slots are free.
        self.assertTrue(mgr.maybe_start_pull(room))
        self.assertEqual(mgr._engine.start_read.call_count, 1)
        self.assertEqual(len(mgr._pulling), 1)
        self.assertFalse(mgr.is_recv_done(room))

        with mgr._state_lock:
            mgr._completed_recv.add("33#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(room))
        self.assertEqual(mgr._engine.start_read.call_count, 2)
        self.assertEqual(mgr._engine.start_read.call_args.args[0], "33#1")
        self.assertEqual(mgr._engine.start_read.call_args.args[4], local[16:32])
        self.assertEqual(len(mgr._pulling), 1)

    def test_oversized_chunk_fails_the_request(self):
        room = 43
        local = list(range(200, 264))
        remote = list(range(100, 164))
        mgr = self._decode(room, local, capacity=4)
        mgr._engine_max_blocks = 8
        mgr._chunk_queue[room] = [("43#0", 11, 0, remote[0:16], "127.0.0.1:31020")]

        self.assertFalse(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_failed(room))
        self.assertIn("SGLANG_RAIDEN_MAX_BLOCKS", mgr.record_failure.call_args.args[1])
        self.assertIn("16 pages", mgr.record_failure.call_args.args[1])

    def test_closing_read_checks_the_sliding_window_pages_against_their_limit(self):
        room = 44
        local = list(range(200, 264))
        remote = list(range(100, 164))
        mgr = self._decode(room, local, capacity=4)
        mgr._swa_engine = Mock()
        mgr._remote_swa_blocks = {room: [4, 5, 6]}
        mgr._local_swa_blocks = {room: [7, 8, 9]}
        mgr._remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr._chunk_queue[room] = [
            (str(room), room, 48, remote[48:64], "127.0.0.1:31020")
        ]

        self.assertFalse(mgr.maybe_start_pull(room))
        mgr._engine.start_read.assert_not_called()
        mgr._swa_engine.start_read.assert_not_called()
        self.assertTrue(mgr.is_failed(room))
        self.assertIn(
            "SGLANG_RAIDEN_SWA_MAX_BLOCKS", mgr.record_failure.call_args.args[1]
        )

    def test_announced_reach_is_the_furthest_chunk_end(self):
        from sglang.srt.disaggregation.raiden.serde import pack_blocks

        room = 45
        remote = list(range(100, 164))
        mgr = self._decode(room, list(range(200, 264)), capacity=4)

        def chunk(req_id, start, blocks):
            return [
                MessageTag.CHUNK,
                str(room).encode("ascii"),
                req_id.encode("ascii"),
                b"11",
                str(start).encode("ascii"),
                b"64",
                pack_blocks(blocks),
                b"127.0.0.1:31020",
            ]

        # The second chunk overtakes the first on the wire.
        mgr._on_chunk(chunk("45#1", 16, remote[16:32]))
        mgr._on_chunk(chunk("45#0", 0, remote[0:16]))
        self.assertEqual(mgr._announced_pages[room], 32)

    def test_failed_chunk_read_fails_the_request(self):
        room = 42
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, capacity=4)
        mgr._chunk_queue[room] = [
            ("42#0", 11, 0, remote[0:16], endpoint),
            ("42#1", 12, 16, remote[16:32], endpoint),
        ]
        self.assertTrue(mgr.maybe_start_pull(room))
        self.assertFalse(mgr.is_failed(room))

        with mgr._state_lock:
            mgr._failed.add("42#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.is_failed(room))
        self.assertFalse(mgr.is_failed(420))
        self.assertEqual(len(mgr._pulling), 0)

        mgr.clear_room(room)
        self.assertFalse(mgr._failed)

    def test_cleared_room_keeps_its_slot_until_the_chunk_read_settles(self):
        first, second = 36, 37
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(first, local, capacity=1)
        mgr._local_blocks[second] = local
        mgr._remote_total[second] = 64
        mgr._chunk_queue[first] = [("36#0", 11, 0, remote[0:16], endpoint)]
        mgr._chunk_queue[second] = [("37#0", 21, 0, remote[0:16], endpoint)]

        self.assertTrue(mgr.maybe_start_pull(first))
        mgr._completed_send.add("36#0")
        # The framework aborts the request while raiden still holds the read.
        mgr.clear_room(first)
        self.assertFalse(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_count, 1)

        with mgr._state_lock:
            mgr._completed_recv.add("36#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_args.args[0], "37#0")
        self.assertFalse(mgr._pulling_cleared)
        self.assertNotIn("36#0", mgr._completed_recv)
        self.assertNotIn("36#0", mgr._completed_send)

    def test_clearing_a_room_drops_its_settled_chunk_ids(self):
        room = 38
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, capacity=4)
        mgr._chunk_queue[room] = [
            ("38#0", 11, 0, remote[0:16], endpoint),
            ("38#1", 12, 16, remote[16:32], endpoint),
        ]
        self.assertTrue(mgr.maybe_start_pull(room))
        with mgr._state_lock:
            mgr._completed_recv.add("38#0")
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(room))
        with mgr._state_lock:
            mgr._completed_recv.update(("38#1", "380#0"))
            mgr._release_pull_slots_locked()

        mgr.clear_room(room)
        self.assertEqual(len(mgr._pulling), 0)
        # Only the cleared room's ids go; room 380 shares the prefix digits.
        self.assertEqual(mgr._completed_recv, {"380#0"})

    def test_hybrid_closing_read_holds_its_slot_until_both_engines_settle(self):
        first, second = 39, 40
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(first, local, capacity=1)
        mgr._swa_engine = Mock()
        mgr._remote_swa_blocks = {first: [4, 5], second: [4, 5]}
        mgr._local_swa_blocks = {first: [7], second: [7]}
        mgr._remote_swa_endpoint = {first: "127.0.0.1:31022", second: "127.0.0.1:31022"}
        mgr._local_blocks[second] = local
        mgr._remote_total[second] = 64
        mgr._chunk_queue[first] = [(str(first), first, 48, remote[48:64], endpoint)]
        mgr._chunk_queue[second] = [(str(second), second, 48, remote[48:64], endpoint)]

        self.assertTrue(mgr.maybe_start_pull(first))
        # The full-attention read fails while the sliding-window read is
        # still in flight on the other engine.
        with mgr._state_lock:
            mgr._failed.add(str(first))
            mgr._release_pull_slots_locked()
        self.assertFalse(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_count, 1)

        with mgr._state_lock:
            mgr._completed_recv.add(mgr._swa_req_id(first))
            mgr._release_pull_slots_locked()
        self.assertTrue(mgr.maybe_start_pull(second))
        self.assertEqual(mgr._engine.start_read.call_args.args[0], str(second))
        self.assertEqual(mgr._swa_engine.start_read.call_count, 2)

    def test_cleared_hybrid_room_drops_both_reads_when_they_settle(self):
        room = 41
        local = list(range(200, 264))
        remote = list(range(100, 164))
        endpoint = "127.0.0.1:31020"
        mgr = self._decode(room, local, capacity=1)
        mgr._swa_engine = Mock()
        mgr._remote_swa_blocks = {room: [4, 5]}
        mgr._local_swa_blocks = {room: [7]}
        mgr._remote_swa_endpoint = {room: "127.0.0.1:31022"}
        mgr._chunk_queue[room] = [(str(room), room, 48, remote[48:64], endpoint)]

        self.assertTrue(mgr.maybe_start_pull(room))
        with mgr._state_lock:
            mgr._completed_recv.add(str(room))
            mgr._release_pull_slots_locked()
        mgr.clear_room(room)
        self.assertEqual(len(mgr._pulling), 1)
        self.assertIn(str(room), mgr._completed_recv)

        with mgr._state_lock:
            mgr._completed_recv.add(mgr._swa_req_id(room))
            mgr._release_pull_slots_locked()
        self.assertEqual(len(mgr._pulling), 0)
        self.assertFalse(mgr._pulling_cleared)
        self.assertFalse(mgr._completed_recv)


class _InlineExecutor:
    """Runs submitted pushes on the calling thread."""

    def submit(self, fn, *args):
        fn(*args)


class _ReadyFuture:
    def is_ready(self):
        return True

    def ok(self):
        return True


class _EntryList(list):
    def add(self):
        self.append(types.SimpleNamespace())
        return self[-1]


class _ScheduleMap(dict):
    def __missing__(self, key):
        self[key] = types.SimpleNamespace(entries=_EntryList())
        return self[key]


class _Pb2:
    """Stands in for raiden's protobuf module so the real plan builders run."""

    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return types.SimpleNamespace(shard_push_schedules=_ScheduleMap())


def _pb2_modules():
    rpc = types.ModuleType("tpu_sync.rpc")
    rpc.raiden_service_pb2 = _Pb2
    pkg = types.ModuleType("tpu_sync")
    pkg.rpc = rpc
    return {
        "tpu_sync": pkg,
        "tpu_sync.rpc": rpc,
        "tpu_sync.rpc.raiden_service_pb2": _Pb2,
    }


def _plan_blocks(req, prefill_rank):
    """(src, dst) block ids of one source's schedule in a built plan."""
    entries = req.shard_push_schedules[prefill_rank].entries
    return [e.src_block_id for e in entries], [e.dst_block_id for e in entries]


class TestRaidenChunkedResharding(CustomTestCase):
    """Chunk announcements and pushes on the resharding path."""

    ROOM = 41
    PREFILL_BLOCKS = list(range(100, 164))  # 64 pages, 4 chunks of 16
    DECODE_BLOCKS = list(range(200, 240))  # decode holds the last 40

    @staticmethod
    def _geometry(src_tp, dst_tp):
        from sglang.srt.disaggregation.raiden.reshard_schedule import ReshardGeometry

        return ReshardGeometry(src_tp, dst_tp, 1024, 8)

    def _prefill(self):
        mgr = object.__new__(RaidenKVManager)
        mgr.chunked_transfer = True
        mgr._state_lock = threading.Lock()
        mgr._push_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._reshard_push_lock = threading.Lock()
        mgr._engine = Mock()
        mgr._engine.d2h = Mock(return_value=_ReadyFuture())
        mgr._swa_engine = None
        mgr._kv_caches = [object(), object()]
        mgr.kv_args = Mock(page_size=128)
        mgr.attn_tp_rank = 1
        mgr.attn_tp_size = 8
        mgr._local_rank = 1
        mgr._timeout_s = 5
        mgr._push_executor = _InlineExecutor()
        mgr._make_geometry = self._geometry
        mgr._send_blocks = {}
        mgr._send_swa_blocks = {}
        mgr._send_aux = {}
        mgr._send_total = {}
        mgr._held_chunks = {}
        mgr._decode_addr = {}
        mgr._decode_pages = {}
        mgr._chunk_seq = {}
        mgr._uuid_seq = 0
        mgr._reshard_req = {}
        mgr._reshard_finalized = set()
        mgr._reshard_replied = set()
        mgr._reshard_ready = set()
        mgr._reshard_pushed = set()
        mgr._reshard_chunks = {}
        mgr._reshard_announced = {}
        mgr._reshard_decode_frontier = {}
        mgr._pending_offer = {}
        mgr._announce_pages = 16
        mgr._reshard_pushes_pending = {}
        mgr._reshard_final_pushed = set()
        mgr._completed_send = set()
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._pulling = None
        mgr._pulling_cleared = set()
        mgr._cleared_plans = {}
        mgr._socket = Mock()
        mgr._connect = Mock(return_value=mgr._socket)
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    def _reshard_req(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        return [
            MessageTag.RESHARD_REQ,
            str(self.ROOM).encode("ascii"),
            b"10.0.0.2",
            b"31002",
            b"10.0.0.2:31000",
            pack_ints(self.DECODE_BLOCKS),
            b"4",
        ]

    def _sent(self, mgr, tag):
        return [
            c.args[0]
            for c in mgr._socket.send_multipart.call_args_list
            if c.args[0][0] == tag
        ]

    def test_closing_push_slices_from_the_decode_frontier(self):
        # This source announced one chunk more than the group completed; the
        # decode's echoed frontier, not the local count, decides the closing
        # span so every source pushes the same pages.
        mgr = self._prefill()
        room = self.ROOM
        mgr._reshard_announced[room] = 48
        mgr._on_reshard_ready([MessageTag.RESHARD_READY, b"41", b"32"])
        self.assertEqual(mgr._reshard_decode_frontier[room], 32)

    def test_reshard_chunk_racing_the_closing_message_is_skipped(self):
        mgr = self._prefill()
        room = self.ROOM
        mgr._reshard_req[room] = dict(
            reply_addr=("10.0.0.2", 5555), decode_blocks=list(range(40))
        )
        mgr._reshard_replied.add(room)
        mgr._announce_reshard_chunk(room, 0, list(range(16)), 64)
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK), [])
        self.assertNotIn(room, mgr._reshard_announced)

    def test_chunk_racing_the_closing_reply_is_skipped(self):
        # A chunk announced after the reply would arrive after it and outlive
        # the room's completion; its pages ride the closing transfer instead.
        mgr = self._prefill()
        mgr._ensure_engine = Mock()
        room = self.ROOM
        mgr._decode_addr[room] = ("10.0.0.2", 5555)
        mgr._decode_pages[room] = 40
        mgr._replied = {room}
        mgr._announce_chunk(room, 0, list(range(16)), 64)
        mgr._ensure_engine.assert_not_called()
        mgr._socket.send_multipart.assert_not_called()

    def test_prefill_announces_chunks_and_pushes_them_on_request(self):
        from sglang.srt.disaggregation.raiden.serde import unpack_ints

        room, blocks = self.ROOM, self.PREFILL_BLOCKS
        mgr = self._prefill()
        # Two chunks commit before the decode has asked for the room.
        for start in (0, 16):
            mgr.append_send_blocks(room, blocks[start : start + 16])
            mgr.offer_chunk(room, start, blocks[start : start + 16], 64)
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK), [])

        mgr._on_reshard_req(self._reshard_req())
        # The decode holds the first 24 pages: chunk 0 is never announced and
        # chunk 1 is trimmed to pages 24..32.
        chunks = self._sent(mgr, MessageTag.RESHARD_CHUNK)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(int(chunks[0][2]), 0)  # first announced chunk index
        self.assertEqual(int(chunks[0][3]), 24)
        self.assertEqual(unpack_ints(chunks[0][5]), blocks[24:32])

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk_ready(
                [MessageTag.RESHARD_CHUNK_READY, b"41", b"0", b"777"]
            )
            self.assertEqual(mgr._failed, set())
            mgr._engine.d2h.assert_called_once_with(
                blocks[24:32], blocks[24:32], [1] * 8
            )
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((uuid, req.uuid, req.req_id), (777, 777, "41#0"))
            self.assertEqual(
                _plan_blocks(req, 0), (blocks[24:32], self.DECODE_BLOCKS[0:8])
            )
            self.assertEqual(mgr._engine.push_registered_plan.call_count, 2)
            mgr._engine.unregister_active_plan.assert_called_once_with(777)
            self.assertFalse(mgr.is_send_done(room))

            # The prefill finishes; the closing push moves only the rest.
            mgr.append_send_blocks(room, blocks[32:64])
            mgr.finalize_send(room, None)
            self.assertEqual(len(self._sent(mgr, MessageTag.RESHARD_BLOCKS)), 1)
            mgr._on_reshard_ready([MessageTag.RESHARD_READY, b"41", b"32"])
            self.assertEqual(mgr._failed, set())
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((uuid, req.req_id), (room, "41"))
            self.assertEqual(
                _plan_blocks(req, 0), (blocks[32:64], self.DECODE_BLOCKS[8:40])
            )
        self.assertTrue(mgr.is_send_done(room))

    def _decode(self):
        mgr = object.__new__(RaidenKVManager)
        mgr.chunked_transfer = True
        mgr._state_lock = threading.Lock()
        mgr._push_lock = threading.Lock()
        mgr._announce_lock = threading.Lock()
        mgr._engine = Mock(transfer_address="10.0.0.2:31000")
        mgr._swa_engine = None
        mgr.kv_args = Mock(page_size=128)
        mgr.attn_tp_rank = 0
        mgr.attn_tp_size = 4
        mgr._local_rank = 0
        mgr._uuid_seq = 0
        mgr._make_geometry = self._geometry
        mgr._reshard_meta = {
            self.ROOM: dict(
                local_blocks=list(self.DECODE_BLOCKS),
                source_ranks=[0, 1],
                src_attn_tp=8,
                aux_index=None,
                ready_endpoints=["tcp://10.0.0.10:1", "tcp://10.0.0.11:1"],
            )
        }
        mgr._reshard_sources = {self.ROOM: {}}
        mgr._reshard_registered = set()
        mgr._reshard_chunk_sources = {}
        mgr._reshard_plans = {}
        mgr._cleared_plans = {}
        mgr._pulling = None
        mgr._pulling_cleared = set()
        mgr._chunk_reads = {}
        mgr._chunk_queue = {}
        mgr._final_seen = set()
        mgr._announced_pages = {}
        mgr._remote_total = {}
        mgr._completed_send = set()
        mgr._completed_recv = set()
        mgr._failed = set()
        mgr._socket = Mock()
        mgr._connect = Mock(return_value=mgr._socket)
        # clear_room sweeps the reshard bookkeeping too.
        mgr._reshard_slot_of = {}
        mgr._reshard_slot_rooms = [None]
        return mgr

    def _decode_with_chunk_plan(self):
        """A decode whose sources have both announced chunk 0 of ROOM, so its
        receive plan is registered and the push is in flight."""
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        mgr = self._decode()
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_chunk(
                    [
                        MessageTag.RESHARD_CHUNK,
                        b"41",
                        b"0",
                        b"24",
                        b"64",
                        pack_ints(list(range(124, 132))),
                        str(rank).encode("ascii"),
                    ]
                )
        uuid = mgr._engine.register_active_plan.call_args.args[0]
        self.assertEqual(mgr._reshard_plans, {self.ROOM: {"41#0": uuid}})
        return mgr, uuid

    def test_aborted_room_keeps_its_chunk_plan_until_the_push_settles(self):
        mgr, uuid = self._decode_with_chunk_plan()
        mgr._completed_send.add("41#0")

        # The framework aborts the request while the sources are still pushing.
        mgr.clear_room(self.ROOM)
        mgr._engine.unregister_active_plan.assert_not_called()
        self.assertEqual(mgr._cleared_plans, {"41#0": uuid})
        self.assertNotIn(self.ROOM, mgr._reshard_plans)
        self.assertFalse(mgr._completed_send)

        # An unrelated completion does not touch it.
        mgr._engine.poll_stats = Mock(return_value=([], ["7#0"], []))
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_not_called()

        # The push lands late: the plan goes, exactly once, with its record.
        mgr._engine.poll_stats = Mock(return_value=([], ["41#0"], []))
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._cleared_plans, {})
        self.assertNotIn("41#0", mgr._completed_recv)
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)

    def test_aborted_room_drops_its_chunk_plan_when_the_push_times_out(self):
        mgr, uuid = self._decode_with_chunk_plan()
        mgr.clear_room(self.ROOM)

        # Raiden reports the timeout and has already dropped the plan itself.
        mgr._engine.unregister_active_plan = Mock(
            side_effect=RuntimeError(
                f"KVCacheManager unregister_active_plan failed: Plan with "
                f"UUID {uuid} is not registered"
            )
        )
        mgr._engine.poll_stats = Mock(return_value=([], [], ["41#0"]))
        mgr._drain_engine()
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._cleared_plans, {})
        self.assertFalse(mgr._failed)

    def test_room_is_not_complete_while_closing_registration_is_pending(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        mgr, _ = self._decode_with_chunk_plan()
        mgr._completed_recv.add("41#0")
        seen_complete = []

        def registers_slowly(uuid, req, is_sender):
            # While the closing plan registers, the room must not look done:
            # its closing read is already owed.
            seen_complete.append(mgr.is_recv_done(self.ROOM))

        mgr._engine.register_active_plan = Mock(side_effect=registers_slowly)
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_blocks(
                    [
                        MessageTag.RESHARD_BLOCKS,
                        b"41",
                        pack_ints(list(range(100 + 200 * rank, 164 + 200 * rank))),
                        str(rank).encode("ascii"),
                        b"",
                    ]
                )
        self.assertEqual(seen_complete, [False])
        self.assertFalse(mgr.is_recv_done(self.ROOM))
        mgr._completed_recv.add("41")
        self.assertTrue(mgr.is_recv_done(self.ROOM))

    def test_closing_plan_is_dropped_once_its_push_settles(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        mgr, chunk_uuid = self._decode_with_chunk_plan()
        with patch.dict(sys.modules, _pb2_modules()):
            for rank in (0, 1):
                mgr._on_reshard_blocks(
                    [
                        MessageTag.RESHARD_BLOCKS,
                        b"41",
                        pack_ints(list(range(100 + 200 * rank, 164 + 200 * rank))),
                        str(rank).encode("ascii"),
                        b"",
                    ]
                )
        self.assertEqual(
            mgr._reshard_plans[self.ROOM], {"41#0": chunk_uuid, "41": self.ROOM}
        )
        mgr._completed_recv.update({"41#0", "41"})
        self.assertTrue(mgr.is_recv_done(self.ROOM))
        mgr.clear_room(self.ROOM)
        self.assertEqual(
            sorted(
                c.args[0] for c in mgr._engine.unregister_active_plan.call_args_list
            ),
            sorted([chunk_uuid, self.ROOM]),
        )

    def test_clearing_a_room_drops_its_settled_chunk_plan_at_once(self):
        mgr, uuid = self._decode_with_chunk_plan()
        mgr._completed_recv.add("41#0")
        mgr.clear_room(self.ROOM)
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._cleared_plans, {})
        self.assertFalse(mgr._completed_recv)

    def test_push_asked_for_before_the_room_was_cleared_is_a_noop(self):

        room = self.ROOM
        blocks = self.PREFILL_BLOCKS
        mgr = self._prefill()
        mgr._push_executor = Mock()  # hold the push instead of running it
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_req(self._reshard_req())
            mgr._send_blocks[room] = list(blocks[:32])
            for start in (0, 16):
                mgr.offer_chunk(room, start, blocks[start : start + 16], 64)
            chunk_msg = self._sent(mgr, MessageTag.RESHARD_CHUNK)[0]
            mgr._on_reshard_chunk_ready(
                [MessageTag.RESHARD_CHUNK_READY, b"41", chunk_msg[2], b"7"]
            )
            self.assertEqual(mgr._reshard_pushes_pending[room], 1)
            push = mgr._push_executor.submit.call_args.args

            mgr.clear_room(room)
            push[0](*push[1:])
        mgr._engine.register_active_plan.assert_not_called()
        self.assertFalse(mgr._failed)
        self.assertNotIn(room, mgr._reshard_pushes_pending)
        self.assertFalse(mgr._completed_send)

    def test_schedules_are_keyed_by_the_sender_engine_id(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        room = self.ROOM
        src = {8: list(range(100, 164)), 9: list(range(300, 364))}
        mgr = self._decode()

        def chunk(sender_id):
            return [
                MessageTag.RESHARD_CHUNK,
                b"41",
                b"0",
                b"24",
                b"64",
                pack_ints(src[sender_id][24:32]),
                str(sender_id).encode("ascii"),
            ]

        # The sources sit on the prefill's second attention-DP replica, so
        # their engine ids are 8 and 9 while their attention-TP ranks (what
        # the decode expects announcements from) are 0 and 1.
        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(8))
            mgr._engine.register_active_plan.assert_not_called()
            mgr._on_reshard_chunk(chunk(9))
            self.assertEqual(mgr._failed, set())
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual(set(req.shard_push_schedules), {8, 9})
            self.assertEqual(
                _plan_blocks(req, 8), (src[8][24:32], self.DECODE_BLOCKS[0:8])
            )
            self.assertEqual(
                _plan_blocks(req, 9), (src[9][24:32], self.DECODE_BLOCKS[0:8])
            )

    def test_room_cleared_during_plan_registration_is_not_resurrected(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()

        # clear_room() lands while register_active_plan runs outside the lock.
        def clears_midway(uuid, req, is_sender):
            mgr.clear_room(room)

        mgr._engine.register_active_plan = Mock(side_effect=clears_midway)

        def chunk(rank):
            return [
                MessageTag.RESHARD_CHUNK,
                b"41",
                b"0",
                b"24",
                b"64",
                pack_ints(src[rank][24:32]),
                str(rank).encode("ascii"),
            ]

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(0))
            mgr._on_reshard_chunk(chunk(1))
        # The plan is dropped again, nothing is recreated, no push is asked for.
        uuid = mgr._engine.register_active_plan.call_args.args[0]
        mgr._engine.unregister_active_plan.assert_called_once_with(uuid)
        self.assertEqual(mgr._chunk_reads.get(room, []), [])
        self.assertEqual(mgr._reshard_plans.get(room, {}), {})
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK_READY), [])
        self.assertFalse(mgr.is_failed(room))

    def test_failed_plan_registration_fails_the_room(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()
        mgr.record_failure = Mock()
        mgr._engine.register_active_plan = Mock(
            side_effect=RuntimeError("RESOURCE_EXHAUSTED: no host blocks")
        )

        def chunk(rank):
            return [
                MessageTag.RESHARD_CHUNK,
                b"41",
                b"0",
                b"24",
                b"64",
                pack_ints(src[rank][24:32]),
                str(rank).encode("ascii"),
            ]

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(0))
            mgr._on_reshard_chunk(chunk(1))
        self.assertTrue(mgr.is_failed(room))
        self.assertIn("RESOURCE_EXHAUSTED", mgr.record_failure.call_args.args[1])
        # The chunk is not admitted anywhere a completion could be faked from.
        self.assertEqual(mgr._chunk_reads.get(room, []), [])
        self.assertEqual(mgr._reshard_plans.get(room, {}), {})
        self.assertEqual(self._sent(mgr, MessageTag.RESHARD_CHUNK_READY), [])

    def test_decode_registers_a_plan_per_chunk_once_every_source_announced(self):
        from sglang.srt.disaggregation.raiden.serde import pack_ints

        room = self.ROOM
        src = {0: list(range(100, 164)), 1: list(range(300, 364))}
        mgr = self._decode()

        def chunk(rank):
            return [
                MessageTag.RESHARD_CHUNK,
                b"41",
                b"0",
                b"24",
                b"64",
                pack_ints(src[rank][24:32]),
                str(rank).encode("ascii"),
            ]

        with patch.dict(sys.modules, _pb2_modules()):
            mgr._on_reshard_chunk(chunk(0))
            mgr._engine.register_active_plan.assert_not_called()
            mgr._on_reshard_chunk(chunk(1))
            self.assertEqual(mgr._failed, set())
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((req.uuid, req.req_id), (uuid, "41#0"))
            self.assertEqual(
                _plan_blocks(req, 0), (src[0][24:32], self.DECODE_BLOCKS[0:8])
            )
            self.assertEqual(
                _plan_blocks(req, 1), (src[1][24:32], self.DECODE_BLOCKS[0:8])
            )
            ready = self._sent(mgr, MessageTag.RESHARD_CHUNK_READY)
            self.assertEqual(len(ready), 2)
            self.assertEqual([int(ready[0][2]), int(ready[0][3])], [0, uuid])
            self.assertFalse(mgr.is_recv_done(room))

            # Both sources finish: the closing plan covers only the tail.
            for rank in (0, 1):
                mgr._on_reshard_blocks(
                    [
                        MessageTag.RESHARD_BLOCKS,
                        b"41",
                        pack_ints(src[rank]),
                        str(rank).encode("ascii"),
                        b"",
                    ]
                )
            uuid, req = mgr._engine.register_active_plan.call_args.args
            self.assertEqual((uuid, req.req_id), (room, "41"))
            self.assertEqual(
                _plan_blocks(req, 0), (src[0][32:64], self.DECODE_BLOCKS[8:40])
            )
            self.assertEqual(
                _plan_blocks(req, 1), (src[1][32:64], self.DECODE_BLOCKS[8:40])
            )
            self.assertEqual(len(self._sent(mgr, MessageTag.RESHARD_READY)), 2)

        self.assertEqual(mgr._chunk_reads[room], ["41#0", "41"])
        self.assertFalse(mgr.is_recv_done(room))
        mgr._completed_recv.update({"41#0", "41"})
        self.assertTrue(mgr.is_recv_done(room))


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
