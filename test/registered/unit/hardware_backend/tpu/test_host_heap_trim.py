"""Unit tests for the TPU patch that returns freed host heap pages after warmup."""

import gc
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import host_heap_trim as tpu
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Scheduler:
    def __init__(self, enable_hierarchical_cache=False):
        self.calls = []
        self.enable_hierarchical_cache = enable_hierarchical_cache

    def init_model_worker(self):
        self.init_all_cuda_graphs()
        self.calls.append("sampling warmup")
        return "initialized"

    def init_all_cuda_graphs(self):
        self.calls.append("graphs")

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
            "start_checkpoint_page_cache_release": lambda: None,
            "release_startup_page_cache": lambda: None,
            "log_startup_memory": lambda: None,
            "_release_cpu_backend": lambda: None,
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
        self.clear = self.enterContext(patch.object(tpu.CacheArtifactManager, "clear"))

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
                tpu.jax,
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
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(False),
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
        self.clear.assert_called_once_with()

    def test_failed_cache_save_does_not_clear_records_or_freeze(self):
        scheduler = self.cls()
        with patch.object(
            tpu,
            "release_startup_page_cache",
            side_effect=RuntimeError("cache save failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cache save failed"):
                scheduler.handle_freeze_gc("req")
        self.clear.assert_not_called()
        self.assertEqual(scheduler.calls, [])

    def test_install_wraps_once(self):
        wrapped = self.cls.handle_freeze_gc
        graphs = self.cls.init_all_cuda_graphs
        worker = self.cls.init_model_worker
        tpu.install(scheduler_cls=self.cls)
        self.assertIs(self.cls.handle_freeze_gc, wrapped)
        self.assertIs(self.cls.init_all_cuda_graphs, graphs)
        self.assertIs(self.cls.init_model_worker, worker)
        scheduler = self.cls()
        with patch.object(
            tpu,
            "start_checkpoint_page_cache_release",
            side_effect=lambda: scheduler.calls.append("checkpoint release"),
        ):
            scheduler.init_all_cuda_graphs()
        self.assertEqual(scheduler.calls, ["checkpoint release", "graphs"])

    def test_hicache_waits_for_cleanup_without_freezing_gc(self):
        scheduler = self.cls(enable_hierarchical_cache=True)
        gc.disable()
        self.addCleanup(gc.enable)
        node = _Node()
        node.ref = node
        alive = weakref.ref(node)
        del node
        group = SimpleNamespace(barrier=lambda: scheduler.calls.append("barrier"))
        with (
            patch.object(
                tpu, "get_parallel", return_value=SimpleNamespace(world_group=group)
            ),
            patch.object(
                tpu,
                "release_startup_page_cache",
                side_effect=lambda: scheduler.calls.append("files"),
            ),
            patch.object(tpu.jax, "clear_caches"),
            patch.object(tpu, "trim_host_heap", side_effect=scheduler.calls.append),
            patch.object(tpu.gc, "freeze") as freeze,
            tpu_envs.SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED.override(True),
        ):
            self.assertEqual(scheduler.init_model_worker(), "initialized")
            scheduler.calls.append("allocate HiCache")
        self.assertIsNone(alive())
        freeze.assert_not_called()
        self.clear.assert_called_once_with()
        self.assertEqual(
            scheduler.calls,
            [
                "graphs",
                "sampling warmup",
                "files",
                "before HiCache",
                "barrier",
                "allocate HiCache",
            ],
        )

    def test_early_cleanup_only_runs_for_hicache_and_propagates_failures(self):
        with patch.object(
            tpu, "release_before_hicache", side_effect=RuntimeError("cache save failed")
        ) as cleanup:
            self.assertEqual(self.cls().init_model_worker(), "initialized")
            cleanup.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "cache save failed"):
                self.cls(enable_hierarchical_cache=True).init_model_worker()
        self.clear.assert_not_called()

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


class _Backend:
    def __init__(self, platform):
        self.platform = platform


class TestCpuBackendRelease(CustomTestCase):
    def setUp(self):
        self.bridge = tpu.xla_bridge
        self.tpu_backend = _Backend("tpu")
        self.enterContext(
            patch.object(
                self.bridge,
                "_backends",
                {"cpu": _Backend("cpu"), "tpu": self.tpu_backend},
            )
        )
        self.enterContext(
            patch.object(self.bridge, "_default_backend", self.tpu_backend)
        )
        self.enterContext(
            patch.object(
                self.bridge, "_get_backend_uncached", tpu._jax_get_backend_uncached
            )
        )
        self.enterContext(self.bridge.config.jax_platforms("tpu,cpu"))
        self.init = self.enterContext(
            patch.object(self.bridge, "_init_backend", side_effect=self.init_backend)
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
