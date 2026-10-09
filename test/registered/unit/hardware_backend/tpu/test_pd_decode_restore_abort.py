"""A failed raiden retraction-restore DMA aborts that request on every rank.

Covers ``RaidenKVBackup.restore`` (wait out a slow DMA, release and raise on a
failed one), the per-rank mark, and the resume wrapper's all-reduced abort.
Runs on CPU with no TPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.hardware_backend.tpu import pd_decode_restore_abort, raiden_kv_backup
from sglang.srt.hardware_backend.tpu.raiden_kv_backup import (
    RaidenKVBackup,
    RaidenKVBackupHandle,
    RaidenRestoreError,
)
from sglang.test.test_utils import CustomTestCase

PAGE = 4
HOST_PAGES = 8
ENABLED = {"SGLANG_RAIDEN_RETRACT_HOST_GB": "1"}


class _Event:
    """``synchronize`` raises ``sync_exc``; ``query`` walks ``query_results``."""

    def __init__(self, sync_exc=None, query_results=()):
        self.sync_exc = sync_exc
        self.query_results = list(query_results)

    def synchronize(self):
        if self.sync_exc is not None:
            raise self.sync_exc

    def query(self):
        result = self.query_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _Transfer:
    def __init__(self, event):
        self.event = event

    def load(self, host_pages, page_ids):
        return self.event


def _backup_with_handle(event):
    backup = RaidenKVBackup.__new__(RaidenKVBackup)
    backup._transfer = _Transfer(event)
    backup._page_size = PAGE
    backup._free = list(range(2, HOST_PAGES))
    handle = RaidenKVBackupHandle(backup, [0, 1], first_offset=0, num_rows=2 * PAGE)
    return backup, handle


def _rows():
    return torch.arange(4 * PAGE, 6 * PAGE)


def _req(rid, req_pool_idx):
    return SimpleNamespace(
        rid=rid, return_logprob=False, kv=SimpleNamespace(req_pool_idx=req_pool_idx)
    )


def _queue(pp_size=1):
    return SimpleNamespace(
        gloo_group=None,
        attn_cp_cpu_group=None,
        pp_size=pp_size,
        tree_cache=object(),
        scheduler=SimpleNamespace(output_streamer=mock.Mock()),
    )


class TestRestore(CustomTestCase):
    def test_slow_dma_is_waited_out(self):
        event = _Event(TimeoutError("slow"), query_results=[False, False, True])
        backup, handle = _backup_with_handle(event)
        backup.restore(handle, _rows())
        self.assertEqual(event.query_results, [])
        self.assertEqual(sorted(backup._free), list(range(HOST_PAGES)))

    def test_failed_dma_releases_and_raises(self):
        backup, handle = _backup_with_handle(_Event(RuntimeError("raiden DMA failed")))
        with self.assertRaises(RaidenRestoreError):
            backup.restore(handle, _rows())
        self.assertEqual(sorted(backup._free), list(range(HOST_PAGES)))

    def test_slow_dma_landing_with_error_releases_and_raises(self):
        event = _Event(TimeoutError("slow"), query_results=[False, RuntimeError("x")])
        backup, handle = _backup_with_handle(event)
        with self.assertRaises(RaidenRestoreError):
            backup.restore(handle, _rows())
        self.assertEqual(sorted(backup._free), list(range(HOST_PAGES)))


class TestMarkFailed(CustomTestCase):
    def setUp(self):
        raiden_kv_backup.take_failed_restores()

    def test_failure_is_marked_not_raised(self):
        backup, handle = _backup_with_handle(_Event(RuntimeError("raiden DMA failed")))
        raiden_kv_backup.restore_or_mark_failed(backup, handle, _rows(), 7)
        self.assertEqual(raiden_kv_backup.take_failed_restores(), {7})
        self.assertEqual(raiden_kv_backup.take_failed_restores(), set())

    def test_unattributable_failure_still_raises(self):
        backup, handle = _backup_with_handle(_Event(RuntimeError("raiden DMA failed")))
        with self.assertRaises(RaidenRestoreError):
            raiden_kv_backup.restore_or_mark_failed(backup, handle, _rows(), None)


@mock.patch.object(pd_decode_restore_abort, "prepare_abort")
@mock.patch.object(pd_decode_restore_abort, "release_kv_cache")
class TestAbortFailedRestores(CustomTestCase):
    def setUp(self):
        raiden_kv_backup.take_failed_restores()

    def test_only_the_failed_request_is_aborted(self, release, abort):
        queue, reqs = _queue(), [_req("a", 3), _req("b", 5), _req("c", 9)]
        raiden_kv_backup._failed_restores.add(5)
        with mock.patch.dict(os.environ, ENABLED):
            kept = pd_decode_restore_abort._abort_failed_restores(queue, reqs)
        self.assertEqual([r.rid for r in kept], ["a", "c"])
        release.assert_called_once_with(reqs[1], queue.tree_cache, is_insert=False)
        self.assertEqual(abort.call_args.args[0], reqs[1])
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [reqs[1]], False
        )
        self.assertEqual(raiden_kv_backup.take_failed_restores(), set())

    def test_no_failure_keeps_everything(self, release, abort):
        reqs = [_req("a", 3)]
        with mock.patch.dict(os.environ, ENABLED):
            kept = pd_decode_restore_abort._abort_failed_restores(_queue(), reqs)
        self.assertEqual(kept, reqs)
        release.assert_not_called()
        abort.assert_not_called()

    def test_pipeline_parallel_refuses(self, release, abort):
        raiden_kv_backup._failed_restores.add(3)
        with mock.patch.dict(os.environ, ENABLED), self.assertRaises(RuntimeError):
            pd_decode_restore_abort._abort_failed_restores(
                _queue(pp_size=2), [_req("a", 3)]
            )
        release.assert_not_called()


def _rank_main(rank, init_file, result_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    queue = _queue()
    queue.gloo_group = dist.group.WORLD
    if rank == 0:  # only rank 0's DMA failed
        raiden_kv_backup._failed_restores.add(5)
    with (
        mock.patch.dict(os.environ, ENABLED),
        mock.patch.object(pd_decode_restore_abort, "prepare_abort"),
        mock.patch.object(pd_decode_restore_abort, "release_kv_cache"),
    ):
        kept = pd_decode_restore_abort._abort_failed_restores(
            queue, [_req("a", 3), _req("b", 5)]
        )
    with open(f"{result_file}.{rank}", "w") as f:
        f.write(",".join(r.rid for r in kept))
    dist.destroy_process_group()


class TestRankConsistency(CustomTestCase):
    def test_failure_on_one_rank_aborts_on_all(self):
        with tempfile.TemporaryDirectory() as d:
            init_file, result_file = os.path.join(d, "init"), os.path.join(d, "kept")
            mp.start_processes(
                _rank_main,
                args=(init_file, result_file),
                nprocs=2,
                join=True,
                start_method="fork",
            )
            for rank in range(2):
                with open(f"{result_file}.{rank}") as f:
                    self.assertEqual(f.read(), "a")


class TestInstall(CustomTestCase):
    def test_install_wraps_resume(self):
        upstream = DecodePreallocQueue.resume_retracted_reqs
        try:
            with mock.patch.object(pd_decode_restore_abort, "_is_installed", False):
                pd_decode_restore_abort.install()
            self.assertIsNot(DecodePreallocQueue.resume_retracted_reqs, upstream)
        finally:
            DecodePreallocQueue.resume_retracted_reqs = upstream


if __name__ == "__main__":
    unittest.main()
