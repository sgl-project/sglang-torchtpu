"""Regression checks for the native Raiden adapter boundary, without TPU DMA."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, call, patch

from sglang.srt.disaggregation.raiden.conn import RaidenKVManager
from sglang.srt.disaggregation.raiden.engine import RaidenEngine
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRaidenEngine(unittest.TestCase):
    def test_read_result_is_not_coerced(self):
        # The manager decides how to interpret admission; the adapter must
        # preserve the native result, including a rejected read.
        for result in (False, None, object()):
            with self.subTest(result=result):
                native = Mock()
                native.register_read.return_value = result
                engine = RaidenEngine(native)
                self.assertIs(engine.register_read("7", 7, [1, 2]), result)

    def test_cleanup_errors_reach_the_caller(self):
        # Only selected manager paths tolerate an already-reaped plan.
        for message in ("Plan with uuid 7 not registered", "transport failed"):
            with self.subTest(message=message):
                error = RuntimeError(message)
                native = Mock()
                native.unregister_active_plan.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    RaidenEngine(native).unregister_active_plan(7)
                self.assertIs(caught.exception, error)

    def test_staging_returns_the_native_future(self):
        # Cleanup waits on this exact future before releasing host staging.
        native = Mock()
        future = native.d2h.return_value
        future.is_ready.return_value = False
        returned = RaidenEngine(native).d2h([1], [2], [1])
        self.assertIs(returned, future)
        self.assertFalse(returned.is_ready())
        future.is_ready.return_value = True
        self.assertTrue(returned.is_ready())

    def test_manager_builds_independent_page_namespaces(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                mgr = object.__new__(RaidenKVManager)
                mgr._engine = mgr._swa_engine = None
                mgr._kv_caches = [object()]
                mgr._swa_kv_caches = [object()] if hybrid else []
                mgr._local_rank = 3
                mgr.attn_tp_rank = 1
                mgr._control_port = 31012
                mgr._timeout_s = 120.0
                mgr.local_ip = "127.0.0.1"
                mgr.chunked_transfer = True
                mgr._max_request_blocks = Mock(return_value=8)
                mgr._swa_max_request_blocks = Mock(return_value=2)
                mgr._num_slots = Mock(return_value=3)
                full, swa = Mock(), Mock()
                api = ModuleType("tpu_sync.api.torch.kv_cache_manager")
                api.KVCacheManager = Mock(side_effect=[full, swa])
                with patch.dict(sys.modules, {api.__name__: api}):
                    mgr._build_engine()
                    # A second call must retain the engines and their state.
                    mgr._build_engine()

                expected = [
                    call(
                        kv_caches=mgr._kv_caches,
                        node_id=3,
                        local_control_port=31012,
                        max_blocks=8,
                        num_slots=3,
                        timeout_s=120.0,
                    )
                ]
                self.assertIsInstance(mgr._engine, RaidenEngine)
                self.assertIs(
                    mgr._ensure_engine().transfer_address, full.transfer_address
                )
                self.assertEqual(mgr._pulling.capacity, 24)
                if hybrid:
                    expected.append(
                        call(
                            kv_caches=mgr._swa_kv_caches,
                            node_id=3,
                            local_control_port=31014,
                            max_blocks=2,
                            num_slots=3,
                            timeout_s=120.0,
                        )
                    )
                    self.assertIsInstance(mgr._swa_engine, RaidenEngine)
                    self.assertIs(
                        mgr._swa_engine.transfer_address, swa.transfer_address
                    )
                    self.assertEqual(mgr._pulling.swa_capacity, 6)
                else:
                    self.assertIsNone(mgr._swa_engine)
                    self.assertIsNone(mgr._pulling.swa_capacity)
                self.assertEqual(api.KVCacheManager.call_args_list, expected)


if __name__ == "__main__":
    unittest.main()
