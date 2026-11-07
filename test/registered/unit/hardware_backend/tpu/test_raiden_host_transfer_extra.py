"""Lower-priority unit tests of the host-to-host reads on the HiCache raiden transfer: report bookkeeping across transfers and the control address. Fixtures come from test_raiden_host_transfer.py."""

import importlib.util
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_main = _load_sibling("test_raiden_host_transfer")
_RaidenHostTransferFixture = _main._RaidenHostTransferFixture
_report = _main._report


class TestRaidenHostTransferExtra(_RaidenHostTransferFixture, CustomTestCase):
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
        with self.assertRaisesRegex(RuntimeError, "SGLANG_ENABLE_PREFILL_KV_OFFLOAD"):
            self.transfer.host_control_address("10.0.0.2")


if __name__ == "__main__":
    unittest.main()
