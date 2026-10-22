# SPDX-License-Identifier: Apache-2.0
"""CPU checks for collective policies selected by platform capabilities."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt import platforms
from sglang.srt.distributed import parallel_state
from sglang.srt.layers import dp_attention
from sglang.srt.platforms.interface import PlatformCapabilities
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCollectiveCapabilities(unittest.TestCase):
    def test_uniform_padding_does_not_consult_variable_length_policy(self):
        platform = SimpleNamespace(
            capabilities=PlatformCapabilities(uniform_dp_padding=True)
        )
        with (
            patch.object(platforms, "_current_platform", platform),
            patch.object(
                dp_attention,
                "dp_gather_width",
                side_effect=AssertionError("variable padding"),
            ),
        ):
            self.assertEqual(
                dp_attention.DpPaddingMode.get_dp_padding_mode(True, [0, 7]),
                dp_attention.DpPaddingMode.MAX_LEN,
            )

    def test_opaque_collective_uses_registered_op(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        coordinator.world_size = 2
        coordinator.unique_name = "test-group"
        value = SimpleNamespace(is_cpu=False)
        platform = SimpleNamespace(
            capabilities=PlatformCapabilities(traces_raw_collectives=False)
        )
        with (
            patch.object(platforms, "_current_platform", platform),
            patch.object(parallel_state, "inplace_all_reduce") as reduce,
        ):
            self.assertIs(coordinator.all_reduce(value), value)
        reduce.assert_called_once_with(value, group_name="test-group")

    def test_host_staged_receive_preserves_requested_device_and_keep_on_host(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        coordinator.world_size = 2
        coordinator.rank_in_group = 0
        coordinator.ranks = [0, 1]
        coordinator.device_group = object()
        coordinator.cpu_group = object()
        metadata = parallel_state.TensorMetadata("meta", torch.float32, (2, 3))
        coordinator.recv_object = Mock(return_value=[("value", metadata)])
        platform = SimpleNamespace(capabilities=PlatformCapabilities(device_p2p=False))
        for keep_on_host in (False, True):
            with (
                self.subTest(keep_on_host=keep_on_host),
                patch.object(platforms, "_current_platform", platform),
                patch.object(torch.distributed, "is_initialized", return_value=True),
                patch.object(torch.distributed, "irecv") as receive,
            ):
                result = coordinator.recv_tensor_dict(keep_on_host=keep_on_host)
            received = receive.call_args.args[0]
            self.assertEqual(received.device.type, "cpu")
            self.assertIs(receive.call_args.kwargs["group"], coordinator.cpu_group)
            self.assertEqual(
                result["value"].device.type, "cpu" if keep_on_host else "meta"
            )
            self.assertEqual(result["value"].shape, (2, 3))

    def test_device_p2p_default_receives_on_original_device(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        coordinator.world_size = 2
        coordinator.rank_in_group = 0
        coordinator.ranks = [0, 1]
        coordinator.device_group = object()
        coordinator.cpu_group = object()
        coordinator.recv_object = Mock(
            return_value=[
                ("value", parallel_state.TensorMetadata("meta", torch.float32, (2, 3)))
            ]
        )
        platform = SimpleNamespace(capabilities=PlatformCapabilities())
        with (
            patch.object(platforms, "_current_platform", platform),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "irecv") as receive,
        ):
            result = coordinator.recv_tensor_dict()
        self.assertEqual(result["value"].device.type, "meta")
        self.assertIs(receive.call_args.kwargs["group"], coordinator.device_group)

    def test_rank_uniform_graphs_requires_uniform_padding(self):
        with self.assertRaisesRegex(ValueError, "requires uniform_dp_padding"):
            PlatformCapabilities(rank_uniform_graphs=True)
        caps = PlatformCapabilities(rank_uniform_graphs=True, uniform_dp_padding=True)
        self.assertTrue(caps.rank_uniform_graphs)

    def test_traced_padding_mode_reads_the_lazy_platform(self):
        # Production code reads ``platforms.current_platform`` through the
        # module's lazy ``__getattr__``; patch the backing field so the traced
        # lookup takes that same path.
        platform = SimpleNamespace(
            capabilities=PlatformCapabilities(uniform_dp_padding=True)
        )

        def padded(value):
            mode = dp_attention.DpPaddingMode.get_dp_padding_mode(True, [0, 7])
            return value + int(mode)

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with (
            patch.object(platforms, "_current_platform", platform),
            patch.object(
                dp_attention,
                "dp_gather_width",
                side_effect=AssertionError("variable padding"),
            ),
        ):
            self.assertNotIn("current_platform", vars(platforms))
            compiled = torch.compile(padded, backend="eager", fullgraph=True)
            result = compiled(torch.zeros(1))
        self.assertEqual(result.item(), int(dp_attention.DpPaddingMode.MAX_LEN))


if __name__ == "__main__":
    unittest.main()
