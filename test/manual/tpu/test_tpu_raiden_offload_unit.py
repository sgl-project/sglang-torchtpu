"""Unit tests for the TPU raiden KV-offload host pool and future-event.

These cover the hardware-agnostic logic only (page-granular allocation,
token->page id conversion, the RaidenFuture event adapter). The actual raiden
DMA path requires a TPU + the built raiden extension and is validated on-device.

The host pool is constructed via ``object.__new__`` so its ``__init__`` (which
builds a real ``RaidenKVTransfer`` over a torch_tpu device pool) is bypassed; we
inject a fake transfer that records the page ids it is handed.
"""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.kv_pool_host import (
    RAIDEN_LAYOUT,
    TpuRaidenHostKVCache,
)
from sglang.srt.hardware_backend.tpu.raiden_transfer import (
    NOOP_EVENT,
    RaidenFutureEvent,
)
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


if __name__ == "__main__":
    unittest.main()
