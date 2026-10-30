"""Exercise the real PD scheduler loops with CPU waves and mocked model work."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardMode,
    PPProxyTensors,
)
from sglang_tpu.pp_wave import PPWave  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class EndSchedule(Exception):
    pass


def batch(mode):
    return SimpleNamespace(forward_mode=mode, reqs=[])


def run_loop(mode, schedule, rank, *, use_wave=True, depth=1):
    """Run every loop statement, stopping only at the next request-receive turn."""
    scheduler = SchedulerPPMixin()
    scheduler.pp_group = SimpleNamespace(
        rank_in_group=rank,
        world_size=4,
        is_first_rank=rank == 0,
        is_last_rank=rank == 3,
    )
    scheduler.pp_group.send_object = Mock(return_value=[])
    scheduler.pp_group.recv_object = Mock(return_value=[])
    scheduler.disaggregation_mode = mode
    scheduler.device_module = Mock()
    scheduler.waiting_queue = []
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(kv_manager=Mock())
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(queue=[])
    scheduler.enable_staging = True
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda x: x
    )
    step = [-1]
    events = []

    def record(name, value=None):
        events.append((step[0], name, value))

    wave = PPWave(
        rank=rank,
        size=4,
        rows=8,
        hidden=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        launcher=lambda x: x,
    )
    wave.begin_iteration = Mock(wraps=wave.begin_iteration)
    wave.receive = Mock(wraps=wave.receive)
    wave.end_iteration = Mock(wraps=wave.end_iteration)
    scheduler._pp_init_wave = Mock(return_value=wave)

    def recv_requests():
        step[0] += 1
        if step[0] == len(schedule):
            raise EndSchedule
        return [f"request-{step[0]}"]

    scheduler.ingest_requests = recv_requests
    scheduler._process_hicache_events = Mock()
    scheduler.process_pending_chunked_abort = Mock()
    scheduler.process_prefill_chunk = Mock()
    scheduler.maybe_prefetch_staging_for_batch = Mock()
    scheduler.on_idle = Mock()

    def plan(running_batch):
        return SimpleNamespace(
            batch_to_run=schedule[step[0]], running_batch=running_batch
        )

    scheduler.get_new_batch_prefill = plan
    scheduler.get_next_disagg_decode_batch_to_run = plan
    scheduler._pp_pd_stage_ready = Mock()
    for name in (
        "_pp_pd_get_bootstrapped_ids",
        "_pp_pd_get_prefill_transferred_ids",
        "_pp_pd_get_retract_ids",
        "_pp_pd_get_prealloc_ids",
        "_pp_pd_get_decode_transferred_ids",
    ):
        setattr(scheduler, name, Mock(return_value=[]))
    for name in (
        "process_bootstrapped_queue",
        "process_retract_queue",
        "process_prealloc_queue",
        "process_decode_transfer_queue",
        "process_disagg_prefill_inflight_queue",
    ):
        setattr(scheduler, name, Mock(return_value=[]))
    scheduler._pp_recv_pyobj_from_prev_stage = Mock(return_value=[])
    scheduler._pp_send_pyobj_to_next_stage = Mock(return_value=[])
    scheduler._pp_recv_proxy_tensors = Mock(return_value=PPProxyTensors({}))
    scheduler._pp_send_dict_to_next_stage = Mock(return_value=[])
    copy_event = Mock()

    def preprocess(*args, before_forward=False):
        record("preprocess", before_forward)
        return None, "host-output", copy_event

    scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = preprocess
    scheduler._pp_send_output_to_next_stage = Mock(
        side_effect=lambda *args: record("deferred") or []
    )

    def launch(mb_id, cur_batch, proxy, metadata, queue):
        record("launch", cur_batch)
        if cur_batch.forward_mode.is_prebuilt():
            assert proxy is None
        elif use_wave and rank != 0:
            assert set(proxy.tensors) == {"hidden_states", "residual"}
        result = SimpleNamespace(
            can_run_cuda_graph=False,
            pp_hidden_states_proxy_tensors=PPProxyTensors(
                {"hidden_states": torch.ones(2, 4), "residual": torch.zeros(2, 4)}
            ),
        )
        if rank == 3:
            queue.append((Mock(), PPProxyTensors({"next_token_ids": torch.zeros(2)})))
        return result, Mock()

    scheduler._pp_launch_batch = Mock(side_effect=launch)
    scheduler._pp_process_batch_result = Mock()
    parallel = SimpleNamespace(
        pp_size=4,
        pp_rank=rank,
        pp_async_batch_depth=depth,
        enable_dsa_prefill_context_parallel=False,
    )
    with (
        patch.object(scheduler_pp_mixin, "_is_tpu", use_wave),
        patch.object(scheduler_pp_mixin, "get_parallel", return_value=parallel),
        patch.object(
            scheduler_pp_mixin,
            "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            ),
        ),
        patch.object(
            scheduler_pp_mixin,
            "ScheduleBatch",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ),
    ):
        loop = (
            scheduler.event_loop_pp_disagg_prefill
            if mode == DisaggregationMode.PREFILL
            else scheduler.event_loop_pp_disagg_decode
        )
        try:
            loop()
        except EndSchedule:
            pass
    return scheduler, wave, copy_event, events


class TestDisaggregatedWave(unittest.TestCase):
    def test_prefill_uses_ici_and_drains_while_processing_transfer_consensus(self):
        schedule = [batch(ForwardMode.EXTEND), None, batch(ForwardMode.EXTEND)] + [
            None
        ] * 17
        for rank in range(4):
            with self.subTest(rank=rank):
                scheduler, wave, copied, _ = run_loop(
                    DisaggregationMode.PREFILL, schedule, rank
                )
                scheduler._pp_init_wave.assert_called_once()
                self.assertEqual(wave.forwards, 2)
                self.assertEqual(wave.begin_iteration.call_count, len(schedule))
                self.assertEqual(wave.end_iteration.call_count, len(schedule))
                self.assertEqual(wave.receive.call_count, 2)
                self.assertEqual(wave.owed, 0)
                self.assertIsNone(wave.pending)
                self.assertEqual(wave.launches, wave.stage0_launches)
                scheduler._pp_recv_proxy_tensors.assert_not_called()
                scheduler._pp_send_dict_to_next_stage.assert_not_called()
                copied.synchronize.assert_called()
                self.assertEqual(scheduler._pp_process_batch_result.call_count, 2)
                self.assertEqual(
                    scheduler.process_pending_chunked_abort.call_count, len(schedule)
                )
                self.assertEqual(
                    scheduler._pp_pd_get_prefill_transferred_ids.call_count,
                    len(schedule),
                )
                scheduler.process_disagg_prefill_inflight_queue.assert_called()
                scheduler.process_bootstrapped_queue.assert_called()
                if rank != 3:
                    forwarded_requests = [
                        call.args[0]
                        for call in scheduler._pp_send_pyobj_to_next_stage.call_args_list
                        if call.args[0] and str(call.args[0][0]).startswith("request-")
                    ]
                    self.assertEqual(len(forwarded_requests), len(schedule))
                scheduler.on_idle.assert_called()

    def test_decode_prebuilt_does_not_exchange_activations_but_advances_drain(self):
        schedule = [
            batch(ForwardMode.PREBUILT),
            batch(ForwardMode.DECODE),
            batch(ForwardMode.PREBUILT),
            None,
            batch(ForwardMode.DECODE),
        ] + [None] * 15
        for rank in range(4):
            with self.subTest(rank=rank):
                scheduler, wave, copied, _ = run_loop(
                    DisaggregationMode.DECODE, schedule, rank
                )
                self.assertEqual(wave.forwards, 2)
                self.assertEqual(
                    [call.args[0] for call in wave.begin_iteration.call_args_list],
                    [False, True, False, False, True] + [False] * 15,
                )
                self.assertEqual(wave.receive.call_count, 2)
                self.assertEqual(wave.end_iteration.call_count, len(schedule))
                self.assertEqual(wave.owed, 0)
                self.assertIsNone(wave.pending)
                self.assertEqual(wave.launches, wave.stage0_launches)
                scheduler._pp_recv_proxy_tensors.assert_not_called()
                scheduler._pp_send_dict_to_next_stage.assert_not_called()
                copied.synchronize.assert_called()
                self.assertEqual(scheduler._pp_launch_batch.call_count, 4)
                self.assertEqual(scheduler._pp_process_batch_result.call_count, 2)
                self.assertEqual(
                    scheduler._pp_pd_get_decode_transferred_ids.call_count,
                    len(schedule),
                )
                scheduler.process_decode_transfer_queue.assert_called()
                scheduler.process_retract_queue.assert_called()
                scheduler.process_prealloc_queue.assert_called()
                scheduler.on_idle.assert_called()

    def test_last_stage_defers_output_until_after_current_forward_in_both_modes(self):
        for mode, forward in (
            (DisaggregationMode.PREFILL, ForwardMode.EXTEND),
            (DisaggregationMode.DECODE, ForwardMode.DECODE),
        ):
            with self.subTest(mode=mode):
                _, _, _, events = run_loop(mode, [batch(forward)] + [None] * 9, 3)
                first = [(name, value) for step, name, value in events if step == 0]
                self.assertEqual(
                    [name for name, _ in first], ["preprocess", "launch", "deferred"]
                )
                self.assertTrue(first[0][1])

    def test_depth_zero_sends_output_after_forward_without_deferred_send(self):
        for mode, forward in (
            (DisaggregationMode.PREFILL, ForwardMode.EXTEND),
            (DisaggregationMode.DECODE, ForwardMode.DECODE),
        ):
            with self.subTest(mode=mode):
                scheduler, _, _, events = run_loop(
                    mode, [batch(forward)] + [None] * 9, 3, depth=0
                )
                first = [name for step, name, _ in events if step == 0]
                self.assertEqual(first, ["launch", "preprocess"])
                scheduler._pp_send_output_to_next_stage.assert_not_called()

    def test_other_backends_retain_proxy_transport_and_output_copy_fence(self):
        for mode, forward in (
            (DisaggregationMode.PREFILL, ForwardMode.EXTEND),
            (DisaggregationMode.DECODE, ForwardMode.DECODE),
        ):
            with self.subTest(mode=mode):
                scheduler, _, copied, events = run_loop(
                    mode, [batch(forward)] + [None] * 9, 1, use_wave=False
                )
                scheduler._pp_init_wave.assert_not_called()
                scheduler._pp_recv_proxy_tensors.assert_called_once()
                scheduler._pp_send_dict_to_next_stage.assert_called_once()
                copied.synchronize.assert_called_once()
                self.assertNotIn("deferred", [name for _, name, _ in events])
                scheduler._pp_send_output_to_next_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
