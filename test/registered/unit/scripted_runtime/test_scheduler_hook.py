from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime import scheduler_hook
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _yielding_gen():
    yield
    yield


def _empty_gen():
    return
    yield  # pragma: no cover — makes this a generator function


def _raising_gen():
    raise ValueError("scripted-boom")
    yield  # pragma: no cover — makes this a generator function


class TestStartupCleanup(CustomTestCase):
    def test_http_startup_drains_freeze_before_waiting_for_scripts(self):
        for skip_warmup in (False, True):
            for pp_size in (1, 2):
                with self.subTest(skip_warmup=skip_warmup, pp_size=pp_size):
                    cfg = SimpleNamespace(
                        skip_server_warmup=skip_warmup,
                        pp_size=pp_size,
                        pp_async_batch_depth=1,
                    )
                    scheduler = MagicMock()
                    scheduler.is_fully_idle.return_value = True
                    proxy = SimpleNamespace(work_reqs_seen=1, freeze_gc_reqs_seen=0)
                    ctx = SimpleNamespace(
                        scheduler=scheduler, _tokenizer_recv_proxy=proxy
                    )
                    with (
                        patch.object(
                            scheduler_hook, "resolving_view", return_value=cfg
                        ),
                        patch.object(
                            scheduler_hook.envs.SGLANG_RUST_SERVER,
                            "get",
                            return_value=False,
                        ),
                    ):
                        driver = scheduler_hook._drive_engine_through_warmup(ctx)
                        rotations = 2 * (pp_size + cfg.pp_async_batch_depth)
                        # Stay receptive to HTTP cleanup while no script can
                        # start yet: readiness waits for its acknowledgement.
                        for _ in range(rotations + 1):
                            next(driver)
                        proxy.freeze_gc_reqs_seen = 1
                        for _ in range(rotations - 1):
                            next(driver)
                        with self.assertRaises(StopIteration):
                            next(driver)


class TestAdvanceGenerator(CustomTestCase):
    def test_not_done_when_generator_yields(self):
        done, exc_tb = scheduler_hook._advance_generator(_yielding_gen())

        self.assertEqual((done, exc_tb), (False, None))

    def test_done_without_traceback_on_stop_iteration(self):
        done, exc_tb = scheduler_hook._advance_generator(_empty_gen())

        self.assertEqual((done, exc_tb), (True, None))

    def test_done_with_traceback_on_exception(self):
        original = scheduler_hook.logger.exception
        scheduler_hook.logger.exception = MagicMock()
        try:
            done, exc_tb = scheduler_hook._advance_generator(_raising_gen())
            scheduler_hook.logger.exception.assert_called_once()
        finally:
            scheduler_hook.logger.exception = original

        self.assertTrue(done)
        self.assertIsNotNone(exc_tb)
        self.assertIn("ValueError", exc_tb)
        self.assertIn("scripted-boom", exc_tb)


if __name__ == "__main__":
    unittest.main()
