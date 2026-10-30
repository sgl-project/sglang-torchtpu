"""Unit tests for the TPU pipeline-parallel host-channel hand-off.

CPU only: the device tensors are mocks and torch.distributed is patched, except
for the two-rank hand-off over a gloo process group.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator, TensorMetadata
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
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
    def _recv(self, metadata_list, is_tpu=True, keep_on_host=False):
        coordinator = _coordinator(rank_in_group=1)
        coordinator.recv_object.return_value = metadata_list
        received = []
        uploads = []

        def fake_irecv(tensor, src, group):
            received.append((tensor, src, group))
            return MagicMock()

        def fake_to(tensor, device_type, non_blocking=False):
            uploads.append((tensor, device_type, non_blocking))
            return ("on", device_type, tensor)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(parallel_state, "_is_tpu", is_tpu),
            patch.object(
                parallel_state.torch.distributed, "irecv", side_effect=fake_irecv
            ),
            patch.object(torch.Tensor, "to", fake_to),
        ):
            out = coordinator.recv_tensor_dict(keep_on_host=keep_on_host)
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
        self.assertEqual(uploads[0][1:], ("tpu", True))
        self.assertEqual(out["hidden_states"][:2], ("on", "tpu"))
        self.assertIs(out["hidden_states"][2], received[0][0])
        self.assertIs(out["host_only"], received[1][0])
        self.assertEqual(out["__msg_type__"], "proxy")

    def test_kept_on_host_device_entry_is_received_on_host(self):
        """A device entry the caller keeps on the host still arrives over the
        cpu_group into a host buffer; receiving into a device buffer would
        take the device group, which the host-copy send never matches."""
        coordinator, out, received, uploads = self._recv(
            [("next_token_ids", TensorMetadata("tpu", torch.int64, (3,)))],
            keep_on_host=True,
        )
        self.assertEqual(len(received), 1)
        tensor, _, group = received[0]
        self.assertTrue(tensor.is_cpu)
        self.assertIs(group, coordinator.cpu_group)
        self.assertEqual(uploads, [])
        self.assertIs(out["next_token_ids"], tensor)

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
            patch.object(torch.Tensor, "to") as to,
        ):
            out = coordinator.recv_tensor_dict()
        to.assert_not_called()
        self.assertIs(irecv.call_args.kwargs["group"], coordinator.cpu_group)
        self.assertEqual(out["hidden_states"].shape, (2,))


def _run_wave_column_hand_off(rank: int, init_file: str) -> None:
    """Rank r is stage r of one device column of a PP2 x TP2 layout with the
    ICI wave active; its TP pair's other device sits in the other column."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        pp_group = GroupCoordinator.__new__(GroupCoordinator)
        pp_group.world_size = 2
        pp_group.rank = pp_group.rank_in_group = rank
        pp_group.ranks = [0, 1]
        pp_group.cpu_group = pp_group.device_group = torch.distributed.group.WORLD
        tp_pair = _coordinator(rank_in_group=0, world_size=2)
        tp_pair.all_gather = MagicMock(
            side_effect=AssertionError("the hand-off all-gathered over the TP pair")
        )
        scheduler = SchedulerPPMixin()
        scheduler._pp_wave = object()
        scheduler.pp_group = pp_group
        scheduler.attn_tp_group = tp_pair
        scheduler.pp_comm_stream = None
        scheduler.pp_comm_stream_ctx = nullcontext()
        scheduler._pp_tensor_dict_inbox = defaultdict(deque)
        hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        tokens = torch.arange(4)
        with patch.object(parallel_state, "_is_tpu", True):
            if rank == 0:
                for tensors, kind in (
                    ({"hidden_states": hidden}, "proxy"),
                    ({"next_token_ids": tokens}, "output"),
                ):
                    for p2p in scheduler._pp_send_dict_to_next_stage(
                        tensors, msg_type=kind
                    ):
                        p2p.work.wait()
            else:
                proxy = scheduler._pp_recv_proxy_tensors()
                outputs, _ = scheduler._pp_recv_dict_from_prev_stage()
                if not torch.equal(proxy["hidden_states"], hidden):
                    raise AssertionError(f"proxy hidden states {proxy.tensors}")
                if not torch.equal(outputs["next_token_ids"], tokens):
                    raise AssertionError(f"output dict {outputs}")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class TestWaveHandOffBetweenTwoRanks(CustomTestCase):
    def test_stage_of_a_tp_pair_hands_on_whole_dicts(self):
        """Over the wave, each device sends its column's next-stage device the
        whole proxy and output dicts; no slice, no all-gather over the TP pair."""
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_wave_column_hand_off,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
