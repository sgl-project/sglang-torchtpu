"""Unit tests for the TPU patch that returns freed host heap pages after warmup."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import gc
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import host_heap_trim as tpu
from sglang_tpu.envs import envs as tpu_envs


class _Scheduler:
    def __init__(self):
        self.calls = []

    def handle_freeze_gc(self, recv_req):
        # Like the core handler: freeze without collecting first.
        self.calls.append(recv_req)
        gc.freeze()
        return "forwarded"


def _scheduler_cls():
    return type("Scheduler", (_Scheduler,), {})


class _Node:
    ref = None


class TestTrimHostHeap(CustomTestCase):
    def test_finds_glibc_and_the_preloaded_tcmalloc(self):
        expected = ["glibc"]
        if "tcmalloc" in os.environ.get("LD_PRELOAD", ""):
            expected = ["tcmalloc", "glibc"]
        self.assertEqual([name for name, _ in tpu._find_releases()], expected)

    def test_every_release_runs_once_and_rss_is_reported(self):
        tc, libc = MagicMock(), MagicMock()
        with patch.object(tpu, "_RELEASES", (("tcmalloc", tc), ("glibc", libc))):
            result = tpu.trim_host_heap("test")
        tc.assert_called_once_with()
        libc.assert_called_once_with()
        self.assertEqual(result.releases, ("tcmalloc", "glibc"))
        self.assertGreater(result.rss_before, 0)
        self.assertGreater(result.rss_after, 0)
        self.assertGreaterEqual(result.seconds, 0.0)

    def test_skipped_when_the_process_has_no_release_call(self):
        with patch.object(tpu, "_RELEASES", ()):
            self.assertIsNone(tpu.trim_host_heap("test"))


class TestInstall(CustomTestCase):
    def setUp(self):
        for name, value in {
            "release_startup_page_cache": lambda: None,
            "log_startup_memory": lambda: None,
            "get_parallel": lambda: SimpleNamespace(
                world_group=SimpleNamespace(barrier=lambda: None)
            ),
        }.items():
            mocked = patch.object(tpu, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        local_rank = patch.dict(os.environ, {"LOCAL_RANK": "0"})
        local_rank.start()
        self.addCleanup(local_rank.stop)
        self.cls = _scheduler_cls()
        tpu.install(scheduler_cls=self.cls)

    def tearDown(self):
        tpu._stop_periodic()
        gc.unfreeze()

    def test_trims_after_the_handler_and_keeps_the_forward(self):
        scheduler = self.cls()
        with (
            patch.object(tpu, "trim_host_heap", side_effect=scheduler.calls.append),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(True),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0),
        ):
            self.assertEqual(scheduler.handle_freeze_gc("req"), "forwarded")
        self.assertEqual(scheduler.calls, ["req", "after warmup"])
        self.assertIsNone(tpu._periodic_thread)

    def test_collects_cycles_before_the_handler_freezes_them(self):
        # A cycle still tracked when gc.freeze runs is never collected again.
        gc.disable()
        self.addCleanup(gc.enable)
        node = _Node()
        node.ref = node
        alive = weakref.ref(node)
        del node
        with (
            patch.object(tpu, "trim_host_heap"),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(True),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0),
        ):
            self.cls().handle_freeze_gc("req")
        self.assertIsNone(alive())

    def test_env_turns_the_startup_trim_off(self):
        with (
            patch.object(tpu, "trim_host_heap") as trim,
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(False),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0),
        ):
            self.cls().handle_freeze_gc("req")
        trim.assert_not_called()

    def test_install_wraps_once(self):
        wrapped = self.cls.handle_freeze_gc
        tpu.install(scheduler_cls=self.cls)
        self.assertIs(self.cls.handle_freeze_gc, wrapped)

    def test_cleanup_finishes_before_the_scheduler_reply(self):
        scheduler = self.cls()
        group = SimpleNamespace(barrier=lambda: scheduler.calls.append("barrier"))
        with (
            patch.object(
                tpu,
                "release_startup_page_cache",
                side_effect=lambda: scheduler.calls.append("files"),
            ),
            patch.object(
                tpu,
                "_before_freeze",
                side_effect=lambda: scheduler.calls.append("collect"),
            ),
            patch.object(
                tpu, "_after_freeze", side_effect=lambda: scheduler.calls.append("trim")
            ),
            patch.object(
                tpu, "get_parallel", return_value=SimpleNamespace(world_group=group)
            ),
            patch.object(
                tpu,
                "log_startup_memory",
                side_effect=lambda: scheduler.calls.append("memory"),
            ),
        ):
            result = scheduler.handle_freeze_gc("freeze")
        self.assertEqual(result, "forwarded")
        self.assertEqual(
            scheduler.calls, ["files", "collect", "freeze", "trim", "barrier", "memory"]
        )

    def test_periodic_trim_runs_on_the_configured_interval(self):
        trimmed = threading.Event()
        with (
            patch.object(
                tpu, "trim_host_heap", side_effect=lambda reason: trimmed.set()
            ),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(False),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0.01),
        ):
            self.cls().handle_freeze_gc("req")
            self.assertTrue(trimmed.wait(2.0))
        self.assertIsNotNone(tpu._periodic_thread)


if __name__ == "__main__":
    unittest.main()
