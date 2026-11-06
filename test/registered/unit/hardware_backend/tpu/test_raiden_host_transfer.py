"""Host pull-serve API and completion ownership at the native Raiden boundary."""

from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.raiden_transfer import RaidenKVTransfer
from sglang_tpu.testing.decode_dram_offload import (
    RaidenHostTransferFixture,
)

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestRaidenHostTransfer(RaidenHostTransferFixture, CustomTestCase):
    def test_host_domains_and_named_landings_reach_native_api(self) -> None:
        """Missing source/destination defaults silently invokes device DMA."""
        self.assertTrue(self.transfer.register_read("send", 11, [1, 3], source="host"))
        self.assertEqual(self.native.registered, ("send", 11, [1, 3], "host"))
        self.transfer.start_read(
            "recv",
            12,
            "10.0.0.2:9000",
            [1, 3],
            [5, 6],
            destination="host",
        )
        self.assertEqual(
            self.native.started,
            ("recv", 12, "10.0.0.2:9000", [1, 3], [5, 6], "host", 4),
        )
        self.assertIsNone(self.transfer.poll_transfer("send"))
        self.assertIsNone(self.transfer.poll_transfer("recv"))

    def test_victim_listener_uses_existing_pool_and_positive_port(self) -> None:
        """Zero disables host pull-serve in the native Python constructor."""
        calls = []

        def manager_factory(caches, **kwargs):
            calls.append((caches, kwargs))
            return self.native

        native_module = ModuleType("tpu_sync.api.torch.kv_cache_manager")
        native_module.KVCacheManager = manager_factory
        buffer = SimpleNamespace(shape=(8, 4), nbytes=64)
        pool = SimpleNamespace(
            layer_num=1,
            start_layer=0,
            page_size=64,
            page_buffers=lambda: [buffer],
        )
        with (
            patch.dict(sys.modules, {native_module.__name__: native_module}),
            patch.dict(
                "os.environ", {"SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM": "1"}
            ),
            patch("sglang_tpu.raiden_transfer.get_free_port", return_value=12345),
        ):
            transfer = RaidenKVTransfer(pool, num_host_pages=8)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][0][0], buffer)
        self.assertEqual(calls[0][1]["host_blocks_to_allocate"], 8)
        self.assertEqual(calls[0][1]["local_control_port"], 12345)
        self.assertEqual(calls[0][1]["parallelism"], 4)
        self.assertGreater(calls[0][1]["timeout_s"], 0)
        self.assertEqual(transfer.host_control_address("10.0.0.2"), "10.0.0.2:12345")


if __name__ == "__main__":
    unittest.main()
