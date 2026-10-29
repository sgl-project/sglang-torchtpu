"""Unit tests for the TPU patch that returns freed host heap pages after warmup."""

import gc
import importlib.util
import logging
import os
import sys
import threading
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu import host_heap_trim as tpu
from sglang_tpu import page_cache
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)


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
        self.jax = MagicMock()
        enter_scope(self, patch.dict(sys.modules, {"jax": self.jax}))
        enter_scope(self, patch.object(tpu, "_release_cpu_backend"))
        directory = enter_scope(self, TemporaryDirectory())
        (Path(directory) / "weights.safetensors").write_bytes(b"payload")
        enter_scope(self, get_context().override_server_args(model_path=directory))
        self.group = MagicMock(local_rank=0)
        enter_scope(self, get_parallel().override(world_group=self.group))
        enter_scope(self, patch.object(page_cache, "resolve_compilations", MagicMock()))
        self.advise = enter_scope(self, patch.object(page_cache.os, "posix_fadvise"))
        enter_scope(self, tpu_envs.SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS.override(False))
        enter_scope(self, patch.dict(os.environ, {"LOCAL_RANK": "0"}))

        class Scheduler(_Scheduler):
            pass

        self.cls = Scheduler
        tpu.install(scheduler_cls=self.cls)
        self.clear = enter_scope(self, patch.object(tpu.CacheArtifactManager, "clear"))

    def tearDown(self):
        tpu._stop_periodic()
        gc.unfreeze()

    def test_trims_after_the_handler_and_keeps_the_forward(self):
        scheduler = self.cls()
        self.clear.side_effect = lambda: scheduler.calls.append("export records")
        with (
            patch.object(
                tpu.gc, "collect", side_effect=lambda: scheduler.calls.append("collect")
            ),
            patch.object(tpu, "trim_host_heap", side_effect=scheduler.calls.append),
            patch.object(
                self.jax,
                "clear_caches",
                side_effect=lambda: scheduler.calls.append("clear JAX"),
            ),
            patch.object(
                tpu,
                "_release_cpu_backend",
                side_effect=lambda: scheduler.calls.append("CPU backend"),
            ),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(True),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0),
        ):
            self.assertEqual(scheduler.handle_freeze_gc("req"), "forwarded")
        self.assertEqual(
            scheduler.calls,
            [
                "export records",
                "clear JAX",
                "CPU backend",
                "collect",
                "req",
                "after warmup",
            ],
        )
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

    def test_env_turns_startup_compiler_cleanup_and_trim_off(self):
        with (
            patch.object(tpu, "trim_host_heap") as trim,
            patch.object(self.jax, "clear_caches") as clear_jax,
            patch.object(tpu, "_release_cpu_backend") as release_cpu,
            patch.object(tpu.gc, "collect") as collect,
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(False),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0),
        ):
            self.cls().handle_freeze_gc("req")
        trim.assert_not_called()
        self.clear.assert_not_called()
        clear_jax.assert_not_called()
        release_cpu.assert_not_called()
        collect.assert_not_called()
        self.advise.assert_called_once()

    def test_failed_cache_save_does_not_clear_records_or_freeze(self):
        scheduler = self.cls()
        with patch.object(
            page_cache,
            "resolve_compilations",
            side_effect=RuntimeError("cache save failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cache save failed"):
                scheduler.handle_freeze_gc("req")
        self.clear.assert_not_called()
        self.assertEqual(scheduler.calls, [])

    def test_install_wraps_once(self):
        wrapped = self.cls.handle_freeze_gc
        wrapped_init_info = self.cls.get_init_info
        tpu.install(scheduler_cls=self.cls)
        self.assertIs(self.cls.handle_freeze_gc, wrapped)
        self.assertIs(self.cls.get_init_info, wrapped_init_info)

    def test_pp_freeze_cleans_up_without_a_world_barrier(self):
        # PP stages dispatch the freeze in different loop iterations.
        with (
            get_parallel().override(pp_size=2),
            patch.object(page_cache, "release_compile_artifacts") as release,
            patch.object(tpu, "trim_host_heap"),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(False),
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC.override(0),
        ):
            self.cls().handle_freeze_gc("req")
        self.group.barrier.assert_not_called()
        self.advise.assert_called_once()
        release.assert_called_once_with(native_only=True)

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


class _Backend:
    def __init__(self, platform):
        self.platform = platform


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
class TestCpuBackendRelease(CustomTestCase):
    def setUp(self):
        from jax._src import xla_bridge

        self.bridge = xla_bridge
        self.tpu_backend = _Backend("tpu")
        enter_scope(
            self,
            patch.object(
                self.bridge,
                "_backends",
                {"cpu": _Backend("cpu"), "tpu": self.tpu_backend},
            ),
        )
        enter_scope(
            self, patch.object(self.bridge, "_default_backend", self.tpu_backend)
        )
        enter_scope(
            self,
            patch.object(
                self.bridge,
                "_get_backend_uncached",
                self.bridge._get_backend_uncached,
            ),
        )
        enter_scope(self, patch.object(tpu, "_jax_get_backend_uncached", None))
        enter_scope(self, self.bridge.config.jax_platforms("tpu,cpu"))
        self.init = enter_scope(
            self,
            patch.object(self.bridge, "_init_backend", side_effect=self.init_backend),
        )
        self.bridge.get_backend.cache_clear()
        self.addCleanup(self.bridge.get_backend.cache_clear)

    def init_backend(self, platform):
        return self.tpu_backend if platform == "tpu" else _Backend(platform)

    def test_release_preserves_tpu_and_recreates_cpu_once(self):
        cpu_ref = weakref.ref(self.bridge.get_backend("cpu"))
        self.assertTrue(tpu._release_cpu_backend())
        gc.collect()
        self.assertIsNone(cpu_ref())
        self.assertIs(self.bridge.get_backend(), self.tpu_backend)
        self.assertIs(self.bridge.get_backend("tpu"), self.tpu_backend)
        self.assertIs(self.bridge.get_backend(self.tpu_backend), self.tpu_backend)
        self.init.assert_not_called()

        cpu = self.bridge.get_backend("cpu")
        self.assertIs(self.bridge.get_backend("cpu"), cpu)
        self.init.assert_called_once_with("cpu")
        wrapper = self.bridge._get_backend_uncached
        self.assertTrue(tpu._release_cpu_backend())
        self.assertIs(self.bridge._get_backend_uncached, wrapper)
        self.assertIsNot(self.bridge.get_backend("cpu"), cpu)
        self.assertEqual(self.init.call_count, 2)

    def test_default_or_absent_cpu_is_preserved_without_initialization(self):
        wrapper = self.bridge._get_backend_uncached
        self.bridge._default_backend = self.bridge._backends["cpu"]
        self.assertFalse(tpu._release_cpu_backend())
        self.assertIs(self.bridge._backends["cpu"], self.bridge._default_backend)
        del self.bridge._backends["cpu"]
        self.assertFalse(tpu._release_cpu_backend())
        self.assertIs(self.bridge._get_backend_uncached, wrapper)
        self.init.assert_not_called()

    def test_empty_registry_uses_normal_backend_initialization(self):
        tpu._release_cpu_backend()
        self.bridge._backends.clear()
        self.bridge._default_backend = None
        self.bridge.get_backend.cache_clear()
        cpu = self.bridge.get_backend("cpu")
        self.assertIs(cpu, self.bridge._backends["cpu"])
        self.assertIs(self.bridge.get_backend(), self.tpu_backend)
        self.assertEqual(self.init.call_count, 2)

    def test_cpu_initialization_failure_propagates_without_losing_tpu(self):
        tpu._release_cpu_backend()
        self.init.side_effect = RuntimeError("CPU initialization failed")
        with self.assertRaisesRegex(RuntimeError, "CPU initialization failed"):
            self.bridge.get_backend("cpu")
        self.assertNotIn("cpu", self.bridge._backends)
        self.assertIs(self.bridge.get_backend("tpu"), self.tpu_backend)


if __name__ == "__main__":
    unittest.main()
