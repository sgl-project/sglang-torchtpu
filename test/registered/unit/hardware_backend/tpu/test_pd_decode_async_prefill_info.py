"""Unit tests for the TPU patch that fetches PD prefill info off the scheduler thread."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import inspect
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation import decode
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.hardware_backend.tpu import pd_decode_async_prefill_info as tpu
from sglang.test.test_utils import CustomTestCase

ADDR = "127.0.0.1:11500"


def _queue(executor, max_retries=15, prefill_info_table=None):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue._max_ensure_retries = max_retries
    queue._ensure_retry_interval = 0
    queue._ensure_retry_count = {}
    queue._ensure_last_attempt_time = {}
    queue.kv_manager = SimpleNamespace(
        prefill_info_table={} if prefill_info_table is None else prefill_info_table,
        try_ensure_parallel_info=MagicMock(),
        _ensure_prefill_recompute_executor=lambda: executor,
    )
    return queue


def _req():
    return SimpleNamespace(req=SimpleNamespace(rid="r"), kv_receiver=MagicMock())


class TestAsyncPrefillInfo(CustomTestCase):
    def test_does_not_block_on_a_hung_fetch(self):
        fetch = Future()
        executor = MagicMock()
        executor.submit.return_value = fetch
        queue = _queue(executor)
        req = _req()

        for _ in range(3):
            self.assertEqual(
                tpu._ensure_prefill_info(queue, {ADDR: [req]}), ({}, [req])
            )
        executor.submit.assert_called_once_with(
            queue.kv_manager.try_ensure_parallel_info, ADDR
        )

        fetch.set_result(True)
        self.assertEqual(
            tpu._ensure_prefill_info(queue, {ADDR: [req]}), ({ADDR: [req]}, [])
        )

    def test_cached_info_resolves_without_a_fetch(self):
        executor = MagicMock()
        queue = _queue(executor, prefill_info_table={ADDR: object()})
        req = _req()

        self.assertEqual(
            tpu._ensure_prefill_info(queue, {ADDR: [req]}), ({ADDR: [req]}, [])
        )
        executor.submit.assert_not_called()

    def test_aborts_after_failed_fetches(self):
        executor = MagicMock()

        def failed_fetch(*_):
            fetch = Future()
            fetch.set_result(False)
            return fetch

        executor.submit.side_effect = failed_fetch
        queue = _queue(executor, max_retries=2)
        req = _req()

        # Each attempt takes two passes: submit, then collect the failed result.
        for _ in range(3):
            self.assertEqual(
                tpu._ensure_prefill_info(queue, {ADDR: [req]}), ({}, [req])
            )
        req.kv_receiver.abort.assert_not_called()

        self.assertEqual(tpu._ensure_prefill_info(queue, {ADDR: [req]}), ({}, []))
        req.kv_receiver.abort.assert_called_once_with()

    def test_reraises_config_mismatch(self):
        fetch = Future()
        fetch.set_exception(RuntimeError("Page size mismatch"))
        executor = MagicMock()
        executor.submit.return_value = fetch
        queue = _queue(executor)
        req = _req()

        tpu._ensure_prefill_info(queue, {ADDR: [req]})
        with self.assertRaisesRegex(RuntimeError, "Page size mismatch"):
            tpu._ensure_prefill_info(queue, {ADDR: [req]})


class TestInstall(CustomTestCase):
    def test_install_replaces_upstream_once(self):
        original = DecodePreallocQueue._ensure_prefill_info
        was_installed = tpu._is_installed
        tpu._is_installed = False
        try:
            tpu.install()
            tpu.install()
            self.assertIs(
                DecodePreallocQueue._ensure_prefill_info, tpu._ensure_prefill_info
            )
        finally:
            DecodePreallocQueue._ensure_prefill_info = original
            tpu._is_installed = was_installed

    def test_upstream_still_fetches_inline(self):
        # If upstream stops calling the blocking fetch inline, this patch may be
        # redundant or out of date: revisit it.
        source = inspect.getsource(decode.DecodePreallocQueue)
        self.assertIn("def _ensure_prefill_info", source)
        self.assertIn(
            "self.kv_manager.try_ensure_parallel_info(bootstrap_addr)", source
        )


if __name__ == "__main__":
    unittest.main()
