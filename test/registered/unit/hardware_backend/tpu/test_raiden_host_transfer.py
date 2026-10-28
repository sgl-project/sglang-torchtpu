"""Host pull-serve API and completion ownership at the native Raiden boundary."""

from __future__ import annotations

import sys
import unittest
from collections import deque
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.raiden_transfer import RaidenKVTransfer

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _NativeManager:
    """External boundary recording the native memory-domain contract."""

    def __init__(self) -> None:
        self.registered: tuple | None = None
        self.started: tuple | None = None
        self.registration_needed = True
        self.reports: deque[SimpleNamespace] = deque()

    def register_read(self, req_id, uuid, block_ids, *, source):
        self.registered = (req_id, uuid, block_ids, source)
        return self.registration_needed

    def start_read(
        self,
        req_id,
        uuid,
        endpoint,
        remote_ids,
        local_ids,
        *,
        parallelism,
        local_host_block_ids,
        destination,
    ):
        self.started = (
            req_id,
            uuid,
            endpoint,
            remote_ids,
            local_ids,
            local_host_block_ids,
            destination,
            parallelism,
        )

    def poll_stats_with_details(self):
        if self.reports:
            return self.reports.popleft()
        return SimpleNamespace(sent=[], received=[], failed=[])


def _report(*, sent=(), received=(), failed=()):
    def details(req_id):
        return SimpleNamespace(
            req_id=req_id, code="UNAVAILABLE", phase="push", reason="peer lost"
        )

    return SimpleNamespace(
        sent=[details(req_id) for req_id in sent],
        received=[details(req_id) for req_id in received],
        failed=[details(req_id) for req_id in failed],
    )


class TestRaidenHostTransfer(CustomTestCase):
    def setUp(self) -> None:
        self.native = _NativeManager()
        self.transfer = RaidenKVTransfer.__new__(RaidenKVTransfer)
        self.transfer._mgr = self.native
        self.transfer._host_transfers = {}
        self.transfer._host_control_port = 12345
        self.transfer._parallelism = 4

    def test_host_domains_and_named_landings_reach_native_api(self) -> None:
        """Missing source/destination defaults silently invokes device DMA."""
        self.assertTrue(self.transfer.register_read("send", 11, [1, 3], source="host"))
        self.assertEqual(self.native.registered, ("send", 11, [1, 3], "host"))
        self.transfer.start_read(
            "recv",
            12,
            "10.0.0.2:9000",
            [1, 3],
            None,
            local_host_block_ids=[5, 6],
            destination="host",
        )
        self.assertEqual(
            self.native.started,
            ("recv", 12, "10.0.0.2:9000", [1, 3], None, [5, 6], "host", 4),
        )
        self.assertIsNone(self.transfer.poll_transfer("send"))
        self.assertIsNone(self.transfer.poll_transfer("recv"))

    def test_poll_keeps_other_victims_terminal_reports(self) -> None:
        """Draining a manager for A must not steal B's one-shot completion."""
        self.transfer.register_read("a", 1, [1], source="host")
        self.transfer.start_read(
            "b", 2, "peer:9000", [1], local_host_block_ids=[2], destination="host"
        )
        self.native.reports.append(_report(sent=["a"], received=["b"]))
        self.assertEqual(self.transfer.poll_transfer("a"), "sent")
        self.assertEqual(self.transfer.poll_transfer("b"), "received")
        self.assertEqual(self.transfer.poll_transfer("a"), "sent")
        self.transfer.forget_transfer("a")
        self.assertNotIn("a", self.transfer._host_transfers)
        self.assertEqual(self.transfer.poll_transfer("b"), "received")

    def test_false_registration_waits_for_native_terminal_report(self) -> None:
        """False can consume an early ACK; it does not revoke source reads."""
        self.native.registration_needed = False
        self.assertFalse(self.transfer.register_read("early", 1, [1], source="host"))
        self.assertIsNone(self.transfer.poll_transfer("early"))
        with self.assertRaisesRegex(RuntimeError, "active Raiden transfer"):
            self.transfer.forget_transfer("early")
        self.native.reports.append(_report(failed=["early"]))
        self.assertEqual(self.transfer.poll_transfer("early"), "failed")
        self.transfer.forget_transfer("early")
        self.assertNotIn("early", self.transfer._host_transfers)

    def test_control_address_uses_advertised_host_and_control_port(self) -> None:
        self.assertEqual(
            self.transfer.host_control_address("10.0.0.2"), "10.0.0.2:12345"
        )
        self.assertEqual(
            self.transfer.host_control_address("2001:db8::1"),
            "[2001:db8::1]:12345",
        )
        self.transfer._host_control_port = 0
        with self.assertRaisesRegex(RuntimeError, "victim push enabled"):
            self.transfer.host_control_address("10.0.0.2")

    def test_victim_listener_uses_existing_pool_and_positive_port(self) -> None:
        """Zero disables host pull-serve in the native Python constructor."""
        calls = []

        def manager_factory(caches, **kwargs):
            calls.append((caches, kwargs))
            return self.native

        native_module = ModuleType("tpu_sync.api.torch.kv_cache_manager")
        native_module.KVCacheManager = manager_factory
        pool = SimpleNamespace(
            layer_num=1,
            start_layer=0,
            page_size=64,
            get_fused_kv_buffer=lambda _: "retained-device-buffer",
        )
        with (
            patch.dict(sys.modules, {native_module.__name__: native_module}),
            patch.dict("os.environ", {"SGLANG_DISAGG_PREFIX_VICTIM_PUSH": "1"}),
            patch("sglang_tpu.raiden_transfer.get_free_port", return_value=12345),
        ):
            transfer = RaidenKVTransfer(pool, num_host_pages=8)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], [["retained-device-buffer"]])
        self.assertEqual(calls[0][1]["host_blocks_to_allocate"], 8)
        self.assertEqual(calls[0][1]["local_control_port"], 12345)
        self.assertEqual(calls[0][1]["parallelism"], 4)
        self.assertGreater(calls[0][1]["timeout_s"], 0)
        self.assertEqual(transfer.host_control_address("10.0.0.2"), "10.0.0.2:12345")


if __name__ == "__main__":
    unittest.main()
