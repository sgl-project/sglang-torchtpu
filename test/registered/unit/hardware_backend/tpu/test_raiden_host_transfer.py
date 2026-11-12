"""Host pull-serve API and completion ownership at the native Raiden boundary."""

from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import raiden_transfer
from sglang_tpu.raiden_transfer import RaidenKVTransfer
from sglang_tpu.testing.decode_dram_offload import (
    RaidenHostTransferFixture,
    native_report,
)

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestRaidenHostTransfer(RaidenHostTransferFixture, CustomTestCase):
    def test_host_domains_and_named_landings_reach_native_api(self) -> None:
        """The native API defaults to device DMA; the wrapper always names
        the host memory domain."""
        self.transfer.register_read("send", 11, [1, 3])
        self.assertEqual(self.native.registered, ("send", 11, [1, 3], "host"))
        self.transfer.start_read("recv", 12, "10.0.0.2:9000", [1, 3], [5, 6])
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
            patch(
                "sglang_tpu.raiden_transfer.host_control_port_candidates",
                lambda: iter([12345]),
            ),
        ):
            transfer = RaidenKVTransfer(pool, num_host_pages=8)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][0][0], buffer)
        self.assertEqual(calls[0][1]["host_blocks_to_allocate"], 8)
        self.assertEqual(calls[0][1]["local_control_port"], 12345)
        self.assertEqual(calls[0][1]["parallelism"], 4)
        self.assertGreater(calls[0][1]["timeout_s"], 0)
        self.assertEqual(transfer.host_control_address("10.0.0.2"), "10.0.0.2:12345")


class TestHostControlPortRace(RaidenHostTransferFixture, CustomTestCase):
    """The native constructor binds the host pool's control port; a port picked
    as free can be taken by then. The engine tries the next candidate instead
    of dying at start (seen live: StartControlServer failed to bind)."""

    def _build(self, candidates, taken=(), fail=None):
        calls = []
        taken = set(taken)

        def manager_factory(caches, **kwargs):
            port = kwargs["local_control_port"]
            calls.append(port)
            if port in taken:
                raise RuntimeError(
                    "INTERNAL: StartControlServer: Failed to bind gRPC control "
                    f"server on [::]:{port}"
                )
            if fail is not None and port == fail:
                raise RuntimeError("something else")
            return self.native

        native_module = ModuleType("tpu_sync.api.torch.kv_cache_manager")
        native_module.KVCacheManager = manager_factory
        buffer = SimpleNamespace(shape=(8, 4), nbytes=64)
        pool = SimpleNamespace(
            layer_num=1, start_layer=0, page_size=64, page_buffers=lambda: [buffer]
        )
        with (
            patch.dict(sys.modules, {native_module.__name__: native_module}),
            patch.dict(
                "os.environ", {"SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM": "1"}
            ),
            patch(
                "sglang_tpu.raiden_transfer.host_control_port_candidates",
                lambda: iter(candidates),
            ),
        ):
            transfer = RaidenKVTransfer(pool, num_host_pages=8)
        return transfer, calls

    def test_a_taken_port_is_skipped_for_the_next_candidate(self) -> None:
        transfer, calls = self._build([31384, 40001], taken={31384})
        self.assertEqual(calls, [31384, 40001])
        self.assertEqual(transfer.host_control_address("10.0.0.2"), "10.0.0.2:40001")

    def test_every_candidate_taken_names_them(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"tried \[1, 2\]"):
            self._build([1, 2], taken={1, 2})

    def test_other_constructor_errors_propagate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "something else"):
            self._build([1, 2], fail=1)

    def test_candidates_start_above_the_raiden_kv_ports_then_free_ports(self) -> None:
        with (
            patch(
                "sglang_tpu.raiden_transfer.get_free_port", side_effect=[50001, 50002]
            ),
            patch(
                "sglang_tpu.raiden_transfer.get_tensor_model_parallel_rank",
                return_value=3,
            ),
            patch.object(raiden_transfer, "_host_pools_built", 0),
        ):
            ports = list(raiden_transfer.host_control_port_candidates())
        base = 31000 + raiden_transfer.HOST_POOL_CONTROL_PORT_OFFSET
        self.assertEqual(ports, [base + 3 * 2, 50001, 50002])
        self.assertGreater(ports[0], 31000 + 8 * 6)
        self.assertLess(base + 7 * 2 + 1, 32768)

    def test_each_host_pool_of_a_rank_gets_its_own_port(self) -> None:
        """The native server binds a port another manager of the process already
        serves; a hybrid model's full and window pools must not share one."""
        with (
            patch("sglang_tpu.raiden_transfer.get_free_port", return_value=50001),
            patch(
                "sglang_tpu.raiden_transfer.get_tensor_model_parallel_rank",
                return_value=1,
            ),
            patch.object(raiden_transfer, "_host_pools_built", 0),
        ):
            firsts = [
                next(raiden_transfer.host_control_port_candidates()) for _ in range(3)
            ]
        base = 31000 + raiden_transfer.HOST_POOL_CONTROL_PORT_OFFSET
        self.assertEqual(firsts, [base + 2, base + 3, 50001])


# ---- Lower-priority cases: less common paths, wire round trips, wording ----


class TestRaidenHostTransferExtra(RaidenHostTransferFixture, CustomTestCase):
    def test_poll_keeps_other_victims_terminal_reports(self) -> None:
        """Draining a manager for A must not steal B's one-shot completion."""
        self.transfer.register_read("a", 1, [1])
        self.transfer.start_read("b", 2, "peer:9000", [1], [2])
        self.native.reports.append(native_report(sent=["a"], received=["b"]))
        self.assertEqual(self.transfer.poll_transfer("a"), "sent")
        self.assertEqual(self.transfer.poll_transfer("b"), "received")
        self.assertEqual(self.transfer.poll_transfer("a"), "sent")
        self.transfer.forget_transfer("a")
        self.assertNotIn("a", self.transfer._host_transfers)
        self.assertEqual(self.transfer.poll_transfer("b"), "received")

    def test_control_address_uses_advertised_host_and_control_port(self) -> None:
        self.assertEqual(
            self.transfer.host_control_address("10.0.0.2"), "10.0.0.2:12345"
        )
        self.assertEqual(
            self.transfer.host_control_address("2001:db8::1"),
            "[2001:db8::1]:12345",
        )


if __name__ == "__main__":
    unittest.main()
