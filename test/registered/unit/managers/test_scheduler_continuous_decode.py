"""Unit tests for Scheduler TPU delay-sample async copy and continuous decode."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SCHED_MOD = "sglang.srt.managers.scheduler"


class _StopLoop(Exception):
    pass


class TestDelaySampleTpuAsyncCopy(CustomTestCase):
    def test_launch_batch_sample_submits_to_tpu_async_copy_worker(self):
        s = Scheduler.__new__(Scheduler)
        s.use_overlap_streams = False
        s.forward_stream_ctx = nullcontext()
        s.copy_stream_ctx = nullcontext()
        s.enable_async_d2h_copy = False
        s.device_module = MagicMock()
        s.tpu_async_copy_worker = MagicMock()
        s._relay_forward_payload = MagicMock()
        s._copy_result_to_cpu = MagicMock()

        br = SimpleNamespace(
            future_indices=object(),
            copy_done=None,
            logits_output=SimpleNamespace(
                next_token_logits=object(),
                next_token_global_top_k_logits=object(),
                next_token_global_top_k_token_ids=object(),
                next_token_logprobs=object(),
            ),
            copy_to_cpu=MagicMock(),
        )
        br.delay_sample_func = lambda: br
        cur_batch = SimpleNamespace(return_logprob=True, return_hidden_states=False)

        Scheduler.launch_batch_sample_if_needed(s, br, cur_batch)

        self.assertIs(br.copy_done, s.device_module.Event.return_value)
        s.tpu_async_copy_worker.submit.assert_called_once_with(
            br, return_logprob=True, return_hidden_states=False
        )
        s._copy_result_to_cpu.assert_not_called()
        s._relay_forward_payload.assert_called_once_with(
            cur_batch, br.future_indices, br
        )
        self.assertIsNone(br.delay_sample_func)
        self.assertIsNotNone(br.logits_output)
        self.assertIsNone(br.logits_output.next_token_logits)
        self.assertIsNone(br.logits_output.next_token_global_top_k_logits)
        self.assertIsNone(br.logits_output.next_token_global_top_k_token_ids)
        self.assertIsNotNone(br.logits_output.next_token_logprobs)


class TestContinuousDecodeInvariants(CustomTestCase):
    def test_next_output_finishes_by_length_accounts_for_inflight_steps(self):
        req = Req("r", "", [1, 2], SamplingParams(max_new_tokens=10))
        req.output_ids = [10, 11, 12, 13, 14, 15, 16]  # 7 tokens committed

        req.inflight_steps = 0
        self.assertFalse(req.next_output_finishes_by_length())
        req.inflight_steps = 1
        self.assertFalse(req.next_output_finishes_by_length())
        req.inflight_steps = 2
        self.assertFalse(req.next_output_finishes_by_length())
        req.inflight_steps = 3
        self.assertTrue(req.next_output_finishes_by_length())

    def test_batch_record_buf_scales_with_continuous_decode_steps(self):
        s = Scheduler.__new__(Scheduler)
        s.device = "cpu"
        s.forward_stream = object()
        s.enable_overlap = True
        s.spec_algorithm = MagicMock()
        s.draft_worker = None
        s.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(attn_backend=object()))
        s.req_to_token_pool = object()
        s.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=None))
        s.server_args = SimpleNamespace(speculative_algorithm=None)
        s.uocr_runtime = None

        mock_platform = MagicMock()
        mock_platform.supports_overlap_streams.return_value = False
        mock_device_module = MagicMock()
        with (
            patch(
                f"{_SCHED_MOD}.get_schedule",
                return_value=SimpleNamespace(num_continuous_decode_steps=4),
            ),
            patch(f"{_SCHED_MOD}.use_mlx", return_value=False),
            patch(f"{_SCHED_MOD}.current_platform", mock_platform),
            patch(f"{_SCHED_MOD}.decide_needs_cpu_seq_lens", return_value=True),
            patch(f"{_SCHED_MOD}.decide_needs_confidence_relay", return_value=False),
            patch("torch.get_device_module", return_value=mock_device_module),
        ):
            Scheduler.init_overlap(s)

        self.assertEqual(len(s.batch_record_buf), 5)
        self.assertEqual(s.batch_record_ct, 0)

        batches = [SimpleNamespace(id=i) for i in range(5)]
        with patch.object(
            Scheduler, "_schedule_batch_attr_names", return_value=("id",)
        ):
            for b in batches:
                Scheduler.record_batch_in_overlap(s, b)

        # Indices 1..4 hold batches 0..3 and index 0 holds batch 4; all 5 stay pinned.
        pinned_ids = {entry[0].id for entry in s.batch_record_buf}
        self.assertEqual(pinned_ids, {0, 1, 2, 3, 4})


class TestContinuousDecodeBurstLoop(CustomTestCase):
    @staticmethod
    def _make_req(rid: str, max_new_tokens: int = 100) -> SimpleNamespace:
        return SimpleNamespace(
            rid=rid,
            beam_group=None,
            grammar=None,
            return_sampling_mask=False,
            to_finish=None,
            inflight_steps=0,
            sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
            finished=lambda: False,
            next_output_finishes_by_length=lambda: False,
            _tpu_piggybacked_prefill=False,
        )

    @staticmethod
    def _make_batch(reqs):
        b = SimpleNamespace(
            reqs=list(reqs),
            forward_mode=ForwardMode.DECODE,
            is_prefill_only=False,
            has_grammar=False,
            batch_is_full=False,
            return_logprob=False,
            return_hidden_states=False,
            steps_prepared=0,
        )
        b.is_empty = lambda: len(b.reqs) == 0
        b.copy = lambda: SimpleNamespace(
            reqs=list(b.reqs), forward_mode=b.forward_mode, step_idx=b.steps_prepared
        )
        b.filter_batch = lambda reqs_to_exclude=(): [
            b.reqs.remove(r) for r in list(reqs_to_exclude) if r in b.reqs
        ]
        b.check_decode_mem = lambda: True

        def _prepare():
            b.steps_prepared += 1

        b.prepare_for_decode = _prepare
        return b

    @staticmethod
    def _make_scheduler(batch, *, num_steps: int = 4) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.gracefully_exit = False
        s.pause_state = SimpleNamespace(paused=False)
        s.is_generation = True
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.require_mlp_sync = False
        s.dllm_config = None
        s.enable_hisparse = False
        s.uocr_runtime = None
        s.decode_offload_manager = None
        s.chunked_reqs = []
        s.waiting_queue = []
        s.running_batch = batch
        s.last_batch = None
        s.kv_free_fence = None
        s.enable_unified_memory = False
        s._num_continuous_decode_steps = num_steps
        s._sched_idled = False
        s.cur_batch_for_debug = None
        s.forward_ct = 0
        s.enable_skip_finishing_decode = True
        s.new_token_ratio_tracker = MagicMock()
        s.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda b, **kw: b
        )
        s.invariant_checker = MagicMock()
        s.increment_scheduler_iterations = MagicMock()
        s.ingest_requests = MagicMock()
        s.should_process_last_batch_early = MagicMock(return_value=False)
        s.on_idle = MagicMock()
        s.launch_batch_sample_if_needed = MagicMock()
        s._apply_war_barrier = MagicMock()
        s._next_continuous_decode_batch = (
            lambda rb: Scheduler._next_continuous_decode_batch(s, rb)
        )
        s._can_continue_decode_burst = (
            lambda b, bs, **kw: Scheduler._can_continue_decode_burst(s, b, bs, **kw)
        )
        s._reqs_finishing_inflight = (
            lambda b: Scheduler._reqs_finishing_inflight(s, b)
        )
        return s

    def test_burst_queues_k_steps_and_drains_in_fifo_order(self):
        req = self._make_req("r0")
        batch = self._make_batch([req])
        s = self._make_scheduler(batch, num_steps=4)

        executed_steps = []
        processed_steps = []

        def _get_next(running_batch, last_batch):
            if s.ingest_requests.call_count == 1:
                batch.prepare_for_decode()
                return SimpleNamespace(running_batch=batch, batch_to_run=batch)
            # Second outer iteration: stop after draining queued burst.
            return SimpleNamespace(running_batch=batch, batch_to_run=None)

        def _run_batch(b):
            step = b.steps_prepared
            executed_steps.append(step)
            req.inflight_steps += 1
            return SimpleNamespace(step=step)

        def _process_result(b, res):
            processed_steps.append(b.step_idx)
            req.inflight_steps -= 1
            if len(processed_steps) == 4:
                s.gracefully_exit = True

        s.get_next_batch_to_run = _get_next
        s.run_batch = _run_batch
        s.process_batch_result = _process_result

        with (
            patch(f"{_SCHED_MOD}.set_schedule_time_batch"),
            patch(f"{_SCHED_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=1)),
            patch(
                f"{_SCHED_MOD}.get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
                ),
            ),
        ):
            Scheduler.event_loop_overlap(s)

        self.assertEqual(s.ingest_requests.call_count, 2)
        self.assertEqual(executed_steps, [1, 2, 3, 4])
        self.assertEqual(processed_steps, [1, 2, 3, 4])
        self.assertEqual(req.inflight_steps, 0)
        self.assertEqual(len(s.result_queue), 0)

    def test_burst_stops_when_waiting_queue_or_mem_full_or_finishing(self):
        req = self._make_req("r0")
        batch = self._make_batch([req])
        s = self._make_scheduler(batch, num_steps=4)

        with (
            patch(f"{_SCHED_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=1)),
            patch(
                f"{_SCHED_MOD}.get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
                ),
            ),
        ):
            self.assertTrue(Scheduler._can_continue_decode_burst(s, batch, 1))
            s.waiting_queue.append(object())
            self.assertFalse(Scheduler._can_continue_decode_burst(s, batch, 1))
            s.waiting_queue.clear()

            batch.has_grammar = True
            self.assertFalse(Scheduler._can_continue_decode_burst(s, batch, 1))
            batch.has_grammar = False

            s.spec_algorithm = SpeculativeAlgorithm.DFLASH
            self.assertFalse(Scheduler._can_continue_decode_burst(s, batch, 1))
            s.spec_algorithm = SpeculativeAlgorithm.NONE

            # Memory pressure aborts inner burst without retracting.
            batch.check_decode_mem = lambda: False
            self.assertIsNone(Scheduler._next_continuous_decode_batch(s, batch))

            # Request finishing by length after inflight steps is excluded.
            batch.check_decode_mem = lambda: True
            req.next_output_finishes_by_length = lambda: True
            req.inflight_steps = 1
            self.assertIsNone(Scheduler._next_continuous_decode_batch(s, batch))
            self.assertTrue(batch.is_empty())

            # Penalizer orchestrator requirement disables bursting cleanly.
            batch2 = self._make_batch([self._make_req("r1")])
            batch2.sampling_info = SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=True)
            )
            s.running_batch = batch2
            self.assertFalse(Scheduler._can_continue_decode_burst(s, batch2, 1))

    def test_disagg_decode_burst_and_pending_queue_gate(self):
        req = self._make_req("r0")
        batch = self._make_batch([req])
        s = self._make_scheduler(batch, num_steps=3)
        s.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[], prefetch_prefill_dp_rank_queries=MagicMock()
        )
        s.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        s.process_decode_queue = MagicMock(return_value=False)
        s._flush_staging_scatters = MagicMock()

        executed_steps = []
        processed_steps = []

        def _get_next(running_batch):
            if s.ingest_requests.call_count == 1:
                batch.prepare_for_decode()
                return SimpleNamespace(running_batch=batch, batch_to_run=batch)
            return SimpleNamespace(running_batch=batch, batch_to_run=None)

        def _run_batch(b):
            step = b.steps_prepared
            executed_steps.append(step)
            req.inflight_steps += 1
            return SimpleNamespace(step=step)

        def _process_result(b, res):
            processed_steps.append(b.step_idx)
            req.inflight_steps -= 1
            if len(processed_steps) == 3:
                s.gracefully_exit = True

        s.get_next_disagg_decode_batch_to_run = _get_next
        s.run_batch = _run_batch
        s.process_batch_result = _process_result

        with (
            patch(f"{_SCHED_MOD}.set_schedule_time_batch"),
            patch(f"{_SCHED_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=1)),
            patch(
                f"{_SCHED_MOD}.get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
                ),
            ),
        ):
            self.assertTrue(
                Scheduler._can_continue_decode_burst(s, batch, 1, is_disagg_decode=True)
            )
            s.disagg_decode_transfer_queue.queue.append(object())
            self.assertFalse(
                Scheduler._can_continue_decode_burst(s, batch, 1, is_disagg_decode=True)
            )
            s.disagg_decode_transfer_queue.queue.clear()

            s.disagg_decode_prealloc_queue.retracted_queue.append(object())
            self.assertFalse(
                Scheduler._can_continue_decode_burst(s, batch, 1, is_disagg_decode=True)
            )
            s.disagg_decode_prealloc_queue.retracted_queue.clear()

            Scheduler.event_loop_overlap_disagg_decode(s)

        self.assertTrue(s.enable_skip_finishing_decode)
        self.assertEqual(s.ingest_requests.call_count, 2)
        self.assertEqual(executed_steps, [1, 2, 3])
        self.assertEqual(processed_steps, [1, 2, 3])
        self.assertEqual(req.inflight_steps, 0)
        self.assertEqual(s._flush_staging_scatters.call_count, 4)

    def test_disagg_decode_burst_stops_at_max_new_tokens(self):
        # Verify PD decode stops cleanly at max_new_tokens inside a K>1 burst
        # because event_loop_overlap_disagg_decode enables skip_finishing_decode.
        req = Req("r_len", "", [1, 2], SamplingParams(max_new_tokens=3))
        req.output_ids = [10]  # 1 committed token -> 2 remaining steps
        req.inflight_steps = 0
        batch = self._make_batch([req])
        s = self._make_scheduler(batch, num_steps=4)
        s.enable_skip_finishing_decode = False
        s.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[], prefetch_prefill_dp_rank_queries=MagicMock()
        )
        s.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        s.process_decode_queue = MagicMock(return_value=False)
        s._flush_staging_scatters = MagicMock()

        executed_steps = []
        processed_steps = []

        def _get_next(running_batch):
            if s.ingest_requests.call_count == 1:
                batch.prepare_for_decode()
                return SimpleNamespace(running_batch=batch, batch_to_run=batch)
            return SimpleNamespace(running_batch=batch, batch_to_run=None)

        def _run_batch(b):
            step = b.steps_prepared
            executed_steps.append(step)
            req.inflight_steps += 1
            return SimpleNamespace(step=step)

        def _process_result(b, res):
            processed_steps.append(b.step_idx)
            req.output_ids.append(10 + b.step_idx)
            req.inflight_steps -= 1
            if len(processed_steps) == 2:
                s.gracefully_exit = True

        s.get_next_disagg_decode_batch_to_run = _get_next
        s.run_batch = _run_batch
        s.process_batch_result = _process_result

        with (
            patch(f"{_SCHED_MOD}.set_schedule_time_batch"),
            patch(f"{_SCHED_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=1)),
            patch(
                f"{_SCHED_MOD}.get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
                ),
            ),
        ):
            Scheduler.event_loop_overlap_disagg_decode(s)

        self.assertTrue(s.enable_skip_finishing_decode)
        self.assertEqual(executed_steps, [1, 2])
        self.assertEqual(processed_steps, [1, 2])
        self.assertEqual(len(req.output_ids), 3)
        self.assertEqual(req.inflight_steps, 0)

    def test_collect_inflight_batches_and_pause_retract_drain_all_queued_steps(self):
        from collections import deque
        from sglang.srt.managers.io_struct import PauseGenerationReqInput

        req_active = self._make_req("r_active")
        req_excluded = self._make_req("r_excluded")
        b0 = self._make_batch([req_active, req_excluded])
        b1 = self._make_batch([req_active])

        s = self._make_scheduler(b1, num_steps=4)
        s.enable_overlap = True
        s.enable_continuous_input_polling = False
        s.enable_overlap_mlx = False
        s.disaggregation_mode = "null"
        s.last_batch = b1
        s.result_queue = deque([(b0, "res0"), (b1, "res1")])

        with patch(
            f"{_SCHED_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=1)
        ):
            inflight_batches = Scheduler._collect_inflight_batches(s)
        self.assertIn(b0, inflight_batches)
        self.assertIn(b1, inflight_batches)

        processed = []
        s.process_batch_result = lambda batch, res: processed.append((batch, res))
        s.pause_state = MagicMock()
        s._retract_batches = MagicMock()
        s._publish_paused_state = MagicMock()

        with patch(
            f"{_SCHED_MOD}.get_parallel", return_value=SimpleNamespace(pp_size=1)
        ):
            Scheduler.pause_generation(s, PauseGenerationReqInput(mode="retract"))

        self.assertEqual(processed, [(b0, "res0"), (b1, "res1")])
        self.assertEqual(len(s.result_queue), 0)
        self.assertIsNone(s.last_batch)
        s._retract_batches.assert_called_once()
