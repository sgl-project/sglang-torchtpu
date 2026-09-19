"""Unit tests for the scheduler PP loop pieces that differ on TPU.

CPU only: the mixin is instantiated bare with the attributes each method reads.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

import unittest  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import torch  # noqa: E402

from sglang.srt.distributed.parallel_state_wrapper import ParallelState  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402


def _single_queue_scheduler(is_last_rank=False) -> SchedulerPPMixin:
    """A scheduler on a backend without forward/copy/schedule streams."""
    scheduler = SchedulerPPMixin()
    scheduler.ps = ParallelState.trivial(pp_size=2, pp_rank=1)
    scheduler.use_overlap_streams = False
    scheduler.forward_stream_ctx = nullcontext()
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.device_module = MagicMock()
    scheduler.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=is_last_rank)
    scheduler.run_batch = MagicMock(
        return_value=SimpleNamespace(can_run_cuda_graph=False)
    )
    return scheduler


class TestStreamFences(CustomTestCase):
    def test_launch_batch_runs_without_schedule_stream(self):
        scheduler = _single_queue_scheduler()
        self.assertFalse(hasattr(scheduler, "schedule_stream"))
        cur_batch = SimpleNamespace(reqs=[])
        mb_metadata = [None, None]
        proxy = PPProxyTensors({})

        result, event = scheduler._pp_launch_batch(
            0, cur_batch, proxy, mb_metadata, last_rank_comm_queue=[]
        )

        scheduler.run_batch.assert_called_once_with(cur_batch, proxy)
        self.assertIs(result, scheduler.run_batch.return_value)
        self.assertFalse(mb_metadata[0].can_run_cuda_graph)
        event.record.assert_called_once()

    def test_launch_batch_still_fences_streams_when_they_exist(self):
        scheduler = _single_queue_scheduler()
        scheduler.use_overlap_streams = True
        scheduler.forward_stream = MagicMock()
        scheduler.schedule_stream = object()

        scheduler._pp_launch_batch(
            0, SimpleNamespace(reqs=[]), None, [None], last_rank_comm_queue=[]
        )

        scheduler.forward_stream.wait_stream.assert_called_once_with(
            scheduler.schedule_stream
        )

    def _recv_outputs(self, scheduler):
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
        )
        received = {"next_token_ids": torch.zeros(2, dtype=torch.int64)}
        scheduler._pp_recv_dict_from_prev_stage = MagicMock(return_value=received)
        scheduler._pp_prep_batch_result = MagicMock(return_value="batch_result")
        scheduler._pp_send_output_to_next_stage = MagicMock(return_value=[])
        return scheduler._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id=0,
            next_mb_id=0,
            mbs=[target],
            mb_metadata=[None],
            last_rank_comm_queue=[],
            pp_outputs=None,
        )

    def test_output_receive_runs_without_copy_stream(self):
        scheduler = _single_queue_scheduler()
        self.assertFalse(hasattr(scheduler, "copy_stream"))

        next_pp_outputs, batch_result, d2h_event, send_work = self._recv_outputs(
            scheduler
        )

        self.assertIsInstance(next_pp_outputs, PPProxyTensors)
        self.assertEqual(batch_result, "batch_result")
        d2h_event.record.assert_called_once()
        self.assertEqual(send_work, [])

    def test_output_receive_still_fences_streams_when_they_exist(self):
        scheduler = _single_queue_scheduler()
        scheduler.use_overlap_streams = True
        scheduler.copy_stream = MagicMock()
        scheduler.schedule_stream = object()

        self._recv_outputs(scheduler)

        scheduler.copy_stream.wait_stream.assert_called_once_with(
            scheduler.schedule_stream
        )


class TestProxyReceive(CustomTestCase):
    def _scheduler(self, received):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_first_rank=False)
        scheduler.require_attn_tp_allgather = False
        scheduler._pp_recv_typed_dict = MagicMock(return_value=received)
        return scheduler

    def test_tpu_proxy_holds_tensors_only(self):
        hidden = torch.zeros(4, 2)
        residual = torch.ones(4, 2)
        scheduler = self._scheduler(
            {"hidden_states": hidden, "residual": residual, "__msg_type__": "proxy"}
        )
        with patch.object(scheduler_pp_mixin, "_is_tpu", True):
            proxy = scheduler._pp_recv_proxy_tensors()
        self.assertEqual(list(proxy.tensors), ["hidden_states", "residual"])
        self.assertIs(proxy["hidden_states"], hidden)
        scheduler._pp_recv_typed_dict.assert_called_once_with(
            expected_kind="proxy", all_gather_group=None
        )

    def test_off_tpu_proxy_keeps_the_received_dict(self):
        received = {"hidden_states": torch.zeros(1), "__msg_type__": "proxy"}
        scheduler = self._scheduler(received)
        with patch.object(scheduler_pp_mixin, "_is_tpu", False):
            proxy = scheduler._pp_recv_proxy_tensors()
        self.assertIs(proxy.tensors, received)

    def test_first_stage_receives_nothing(self):
        scheduler = self._scheduler({})
        scheduler.pp_group = SimpleNamespace(is_first_rank=True)
        self.assertIsNone(scheduler._pp_recv_proxy_tensors())
        scheduler._pp_recv_typed_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
