"""A failed raiden retraction-backup DMA falls back to the row path.

The save must not raise into the scheduler. Pages of a timed-out DMA stay out
of the free list until it lands, and pages of a DMA that ended in an error are
never handed out again. Runs on CPU with no TPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang.srt.hardware_backend.tpu.raiden_kv_backup import (
    RaidenKVBackup,
    RaidenKVBackupHandle,
)
from sglang.test.test_utils import CustomTestCase

PAGE = 4
HOST_PAGES = 8


class _Event:
    def __init__(self, sync_exc=None):
        self.sync_exc = sync_exc
        self.ready = sync_exc is None
        self.error = None

    def synchronize(self):
        if self.sync_exc is not None:
            raise self.sync_exc

    def query(self):
        if self.ready and self.error is not None:
            raise RuntimeError(self.error)
        return self.ready


class _Transfer:
    def __init__(self):
        self.next_exc = None
        self.backups = []

    def backup(self, page_ids, host_pages):
        event = _Event(self.next_exc)
        self.backups.append((list(host_pages), event))
        return event


def _backup():
    backup = RaidenKVBackup.__new__(RaidenKVBackup)
    backup._transfer = _Transfer()
    backup._page_size = PAGE
    backup._free = list(range(HOST_PAGES))
    backup._quarantine = []
    backup._num_retired = 0
    return backup


def _rows(num_pages, first_page=0):
    return torch.arange(first_page * PAGE, (first_page + num_pages) * PAGE)


class TestRetractionBackupDmaErrors(CustomTestCase):
    def test_success_stages_pages(self):
        backup = _backup()
        handle = backup.try_save(_rows(2))
        self.assertIsInstance(handle, RaidenKVBackupHandle)
        self.assertEqual(len(backup._free), HOST_PAGES - 2)
        handle.release()
        self.assertEqual(len(backup._free), HOST_PAGES)

    def test_failed_dma_retires_pages(self):
        backup = _backup()
        backup._transfer.next_exc = RuntimeError("raiden DMA failed: boom")
        for _ in range(3):
            self.assertIsNone(backup.try_save(_rows(2)))
        failed_pages = {p for pages, _ in backup._transfer.backups for p in pages}
        self.assertEqual(len(failed_pages), 6)
        self.assertFalse(failed_pages & set(backup._free))
        self.assertEqual(len(backup._free), HOST_PAGES - 6)
        self.assertEqual(backup._num_retired, 6)
        self.assertEqual(backup._quarantine, [])

    def test_timed_out_dma_quarantines_until_landed(self):
        backup = _backup()
        backup._transfer.next_exc = TimeoutError("raiden DMA did not complete")
        self.assertIsNone(backup.try_save(_rows(2)))
        stuck_pages, stuck_event = backup._transfer.backups[-1]

        # While the abandoned DMA is in flight its pages are never reissued.
        backup._transfer.next_exc = None
        handle = backup.try_save(_rows(HOST_PAGES - 2))
        self.assertIsNotNone(handle)
        self.assertFalse(set(stuck_pages) & set(handle.host_pages))
        self.assertIsNone(backup.try_save(_rows(1)))  # pool exhausted
        handle.release()

        stuck_event.ready = True
        backup._alloc_pages(0)
        self.assertEqual(sorted(backup._free), list(range(HOST_PAGES)))
        self.assertEqual(backup._quarantine, [])

    def test_timed_out_dma_landing_with_error_is_retired(self):
        backup = _backup()
        backup._transfer.next_exc = TimeoutError("raiden DMA did not complete")
        self.assertIsNone(backup.try_save(_rows(2)))
        stuck_pages, stuck_event = backup._transfer.backups[-1]
        stuck_event.ready = True
        stuck_event.error = "late failure"
        backup._alloc_pages(0)
        self.assertEqual(backup._quarantine, [])
        self.assertFalse(set(stuck_pages) & set(backup._free))
        self.assertEqual(len(backup._free), HOST_PAGES - 2)
        self.assertEqual(backup._num_retired, 2)

    def test_pool_falls_back_to_row_path(self):
        pool = TpuFusedMHATokenToKVPool(
            size=32,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        backup = _backup()
        backup._transfer.next_exc = RuntimeError("raiden DMA failed: boom")
        with mock.patch.object(pool, "_raiden_backup", return_value=backup):
            payload = pool.get_cpu_copy(_rows(2, first_page=1))
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), pool.layer_num)
        self.assertEqual(backup._num_retired, 2)


if __name__ == "__main__":
    unittest.main()
