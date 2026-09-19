"""Unit tests for the TPU pipeline-parallel host-channel hand-off.

CPU only: the device tensors are mocks and torch.distributed is patched.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator, TensorMetadata
from sglang.srt.hardware_backend.tpu import pp_transfer
from sglang.test.test_utils import CustomTestCase


def _fake_device_tensor(host_copy: torch.Tensor, device_type: str = "tpu"):
    """A tensor stand-in that reports a non-CPU device and copies to host."""
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = SimpleNamespace(type=device_type)
    tensor.is_cpu = False
    tensor.dtype = host_copy.dtype
    tensor.size.return_value = host_copy.size()
    tensor.numel.return_value = host_copy.numel()
    tensor.contiguous.return_value.cpu.return_value = host_copy
    return tensor


def _coordinator(rank_in_group: int = 0, world_size: int = 2) -> GroupCoordinator:
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.world_size = world_size
    coordinator.rank_in_group = rank_in_group
    coordinator.ranks = list(range(world_size))
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.send_object = MagicMock(return_value=[])
    coordinator.recv_object = MagicMock()
    return coordinator


class TestPpTransferHelpers(CustomTestCase):
    def setUp(self):
        pp_transfer._pinned_alloc_failed = False

    def tearDown(self):
        pp_transfer._pinned_alloc_failed = False

    def test_to_host_returns_cpu_tensor_unchanged(self):
        tensor = torch.arange(4)
        self.assertIs(pp_transfer.to_host(tensor), tensor)

    def test_to_host_copies_device_tensor_contiguously(self):
        host = torch.zeros(2, 3, dtype=torch.bfloat16)
        device = _fake_device_tensor(host)
        self.assertIs(pp_transfer.to_host(device), host)
        device.contiguous.assert_called_once_with()
        device.contiguous.return_value.cpu.assert_called_once_with()

    def test_alloc_recv_buffer_requests_pinned_memory(self):
        pinned = torch.empty(2, 3, dtype=torch.bfloat16)
        with patch.object(pp_transfer.torch, "empty", return_value=pinned) as empty:
            out = pp_transfer.alloc_recv_buffer(torch.Size([2, 3]), torch.bfloat16)
        self.assertIs(out, pinned)
        empty.assert_called_once_with(
            (2, 3), dtype=torch.bfloat16, device="cpu", pin_memory=True
        )

    def test_alloc_recv_buffer_falls_back_to_unpinned_once(self):
        unpinned = torch.empty(2, 3)
        calls = []

        def fake_empty(size, **kwargs):
            calls.append(kwargs)
            if kwargs.get("pin_memory"):
                raise RuntimeError("no pinned allocator")
            return unpinned

        with patch.object(pp_transfer.torch, "empty", side_effect=fake_empty):
            first = pp_transfer.alloc_recv_buffer((2, 3), torch.float32)
            second = pp_transfer.alloc_recv_buffer((2, 3), torch.float32)
        self.assertIs(first, unpinned)
        self.assertIs(second, unpinned)
        # pinned attempt, unpinned fallback, then unpinned directly.
        self.assertEqual(
            [bool(c.get("pin_memory")) for c in calls], [True, False, False]
        )

    def test_alloc_recv_buffer_skips_pinning_for_empty_tensors(self):
        with patch.object(
            pp_transfer.torch, "empty", return_value=torch.empty(0)
        ) as empty:
            pp_transfer.alloc_recv_buffer((0,), torch.int64)
        self.assertNotIn("pin_memory", empty.call_args.kwargs)

    def test_to_device_uploads_non_blocking_to_recorded_device_type(self):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.to.return_value = "uploaded"
        self.assertEqual(pp_transfer.to_device(tensor, "tpu"), "uploaded")
        tensor.to.assert_called_once_with("tpu", non_blocking=True)


class TestSendTensorDictOnTpu(CustomTestCase):
    def test_device_tensor_is_sent_as_host_copy_on_cpu_group(self):
        coordinator = _coordinator()
        host = torch.ones(4, 8, dtype=torch.bfloat16)
        device = _fake_device_tensor(host)
        work = object()
        with (
            patch.object(parallel_state, "_is_tpu", True),
            patch.object(
                parallel_state.torch.distributed, "isend", return_value=work
            ) as isend,
        ):
            p2p_works = coordinator.send_tensor_dict(
                {"hidden_states": device, "__msg_type__": "proxy"}, async_send=True
            )

        isend.assert_called_once()
        sent, dst = isend.call_args.args
        self.assertIs(sent, host)
        self.assertEqual(dst, 1)
        self.assertIs(isend.call_args.kwargs["group"], coordinator.cpu_group)
        # The metadata still records the sender's device type.
        metadata_list = coordinator.send_object.call_args.args[0]
        self.assertEqual(dict(metadata_list)["hidden_states"].device, "tpu")
        self.assertEqual(dict(metadata_list)["__msg_type__"], "proxy")
        # The host copy stays alive with the pending work.
        self.assertEqual(len(p2p_works), 1)
        self.assertIs(p2p_works[0].work, work)
        self.assertIs(p2p_works[0].payload, host)

    def test_cpu_tensor_is_sent_on_cpu_group_without_copy(self):
        coordinator = _coordinator()
        tokens = torch.arange(3)
        with (
            patch.object(parallel_state, "_is_tpu", True),
            patch.object(parallel_state.torch.distributed, "send") as send,
        ):
            coordinator.send_tensor_dict({"next_token_ids": tokens})
        send.assert_called_once()
        self.assertIs(send.call_args.args[0], tokens)
        self.assertIs(send.call_args.kwargs["group"], coordinator.cpu_group)

    def test_off_tpu_device_tensor_still_goes_to_device_group(self):
        coordinator = _coordinator()
        device = _fake_device_tensor(torch.ones(2), device_type="cuda")
        with (
            patch.object(parallel_state, "_is_tpu", False),
            patch.object(parallel_state.torch.distributed, "send") as send,
        ):
            coordinator.send_tensor_dict({"hidden_states": device})
        self.assertIs(send.call_args.args[0], device)
        self.assertIs(send.call_args.kwargs["group"], coordinator.device_group)
        device.contiguous.assert_not_called()


class TestRecvTensorDictOnTpu(CustomTestCase):
    def _recv(self, metadata_list, is_tpu=True):
        coordinator = _coordinator(rank_in_group=1)
        coordinator.recv_object.return_value = metadata_list
        received = []
        uploads = []

        def fake_irecv(tensor, src, group):
            received.append((tensor, src, group))
            return MagicMock()

        def fake_alloc(size, dtype):
            return torch.zeros(tuple(size), dtype=dtype)

        def fake_to_device(tensor, device_type):
            uploads.append((tensor, device_type))
            return ("on", device_type, tensor)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(parallel_state, "_is_tpu", is_tpu),
            patch.object(
                parallel_state.torch.distributed, "irecv", side_effect=fake_irecv
            ),
            patch.object(pp_transfer, "alloc_recv_buffer", side_effect=fake_alloc),
            patch.object(pp_transfer, "to_device", side_effect=fake_to_device),
        ):
            out = coordinator.recv_tensor_dict()
        return coordinator, out, received, uploads

    def test_device_entry_is_received_on_host_and_uploaded(self):
        coordinator, out, received, uploads = self._recv(
            [
                ("__msg_type__", "proxy"),
                ("hidden_states", TensorMetadata("tpu", torch.bfloat16, (4, 8))),
                ("host_only", TensorMetadata("cpu", torch.int64, (3,))),
            ]
        )
        self.assertEqual(len(received), 2)
        for tensor, src, group in received:
            self.assertTrue(tensor.is_cpu)
            self.assertEqual(src, 0)
            self.assertIs(group, coordinator.cpu_group)
        self.assertEqual(received[0][0].shape, (4, 8))
        self.assertEqual(received[0][0].dtype, torch.bfloat16)
        self.assertEqual(len(uploads), 1)
        self.assertIs(uploads[0][0], received[0][0])
        self.assertEqual(uploads[0][1], "tpu")
        self.assertEqual(out["hidden_states"][:2], ("on", "tpu"))
        self.assertIs(out["hidden_states"][2], received[0][0])
        self.assertIs(out["host_only"], received[1][0])
        self.assertEqual(out["__msg_type__"], "proxy")

    def test_empty_device_entry_is_uploaded_without_receive(self):
        _, out, received, uploads = self._recv(
            [("logprobs", TensorMetadata("tpu", torch.float32, (0,)))]
        )
        self.assertEqual(received, [])
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0][0].numel(), 0)
        self.assertEqual(out["logprobs"][1], "tpu")

    def test_off_tpu_host_entry_takes_the_unchanged_path(self):
        coordinator = _coordinator(rank_in_group=1)
        coordinator.recv_object.return_value = [
            ("hidden_states", TensorMetadata("cpu", torch.float32, (2,)))
        ]
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(parallel_state, "_is_tpu", False),
            patch.object(
                parallel_state.torch.distributed, "irecv", return_value=MagicMock()
            ) as irecv,
            patch.object(pp_transfer, "alloc_recv_buffer") as alloc,
            patch.object(pp_transfer, "to_device") as to_device,
        ):
            out = coordinator.recv_tensor_dict()
        alloc.assert_not_called()
        to_device.assert_not_called()
        self.assertIs(irecv.call_args.kwargs["group"], coordinator.cpu_group)
        self.assertEqual(out["hidden_states"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
