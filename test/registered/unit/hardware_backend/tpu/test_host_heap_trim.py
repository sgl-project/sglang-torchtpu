"""Unit tests for the TPU patch that returns freed host heap pages after warmup."""

import gc
import logging
import os
import threading
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import host_heap_trim as tpu
from sglang_tpu import page_cache
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Scheduler:
    def __init__(self):
        self.calls = []

    def get_init_info(self):
        self.calls.append("ready")
        return {"status": "ready"}

    def handle_freeze_gc(self, recv_req):
        # Like the core handler: freeze without collecting first.
        self.calls.append(recv_req)
        gc.freeze()
        return "forwarded"


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

    def test_memory_current_log(self):
        for content, level, message in (
            ("610000000000\n", logging.INFO, "memory.current=610000000000 bytes"),
            ("invalid", logging.WARNING, "memory.current unavailable"),
            (None, logging.WARNING, "memory.current unavailable"),
        ):
            with self.subTest(content=content), TemporaryDirectory() as directory:
                path = Path(directory) / "memory.current"
                if content is not None:
                    path.write_text(content)
                with (
                    patch.object(tpu, "_MEMORY_CURRENT", path),
                    self.assertLogs(tpu.logger, level=logging.INFO) as logs,
                ):
                    tpu.log_startup_memory()
                self.assertEqual(len(logs.records), 1)
                self.assertEqual(logs.records[0].levelno, level)
                self.assertIn(message, logs.records[0].getMessage())


class TestInstall(CustomTestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        (Path(directory) / "weights.safetensors").write_bytes(b"payload")
        self.enterContext(get_context().override_server_args(model_path=directory))
        self.group = MagicMock()
        self.enterContext(get_parallel().override(world_group=self.group))
        self.enterContext(patch.object(page_cache, "resolve_compilations"))
        self.advise = self.enterContext(patch.object(page_cache.os, "posix_fadvise"))
        self.enterContext(tpu_envs.SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS.override(False))
        self.enterContext(patch.dict(os.environ, {"LOCAL_RANK": "0"}))

        class Scheduler(_Scheduler):
            pass

        self.cls = Scheduler
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
        wrapped_init_info = self.cls.get_init_info
        tpu.install(scheduler_cls=self.cls)
        self.assertIs(self.cls.handle_freeze_gc, wrapped)
        self.assertIs(self.cls.get_init_info, wrapped_init_info)

    def test_startup_releases_files_without_a_freeze_request(self):
        scheduler = self.cls()
        self.assertEqual(scheduler.get_init_info(), {"status": "ready"})
        self.advise.assert_called_once()
        self.assertEqual(scheduler.calls, ["ready"])

    def test_startup_cleanup_failure_prevents_ready_handshake(self):
        scheduler = self.cls()
        self.advise.side_effect = OSError("advice failed")
        with self.assertRaisesRegex(OSError, "advice failed"):
            scheduler.get_init_info()
        self.assertEqual(scheduler.calls, [])

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
