"""Unit tests for the scheduler PP loop pieces that differ on TPU.

CPU only: the mixin is instantiated bare with the attributes each method reads.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

import unittest  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import torch  # noqa: E402

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (  # noqa: E402
    SpecAuxHiddenStateConfig,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402
from sglang_tpu import pp_scheduler  # noqa: E402
from sglang_tpu.compilation.bucketer import (  # noqa: E402
    Bucketer,
)
from sglang_tpu.compilation.decoder_compile_policy import (  # noqa: E402
    DecoderTpuCompilePolicy,
)


def _single_queue_scheduler(is_last_rank=False) -> SchedulerPPMixin:
    """A scheduler on a backend without forward/copy/schedule streams."""
    scheduler = SchedulerPPMixin()
    scheduler.use_overlap_streams = False
    scheduler.pp_send_done_event = None
    scheduler.pp_proxy_recv_event = None
    scheduler.forward_stream_ctx = nullcontext()
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.device_module = MagicMock()
    scheduler.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=is_last_rank)
    scheduler.run_batch = MagicMock(
        return_value=SimpleNamespace(
            can_run_cuda_graph=False, spec_verify_out_cache_loc=None
        )
    )
    return scheduler


class TestStreamFences(CustomTestCase):
    def test_deferred_prefill_result_copy_waits_on_its_communication_stream(self):
        scheduler = _single_queue_scheduler(is_last_rank=True)
        order = []

        @contextmanager
        def communication_stream():
            order.append("enter-copy-stream")
            yield
            order.append("leave-copy-stream")

        scheduler.pp_comm_stream_ctx = communication_stream()
        scheduler._pp_output_send_deferred = lambda: True
        scheduler.mbs = [SimpleNamespace()]
        ready = MagicMock()
        tokens = torch.tensor([7])
        source = MagicMock()
        source.cpu.side_effect = lambda: order.append("copy-complete") or tokens
        scheduler.last_rank_comm_queue = [
            (ready, PPProxyTensors({"next_token_ids": source}))
        ]
        copy_stream = scheduler.device_module.current_stream.return_value
        copy_stream.wait_event.side_effect = lambda event: order.append(
            "wait-producer-on-stream"
        )
        scheduler._pp_send_deferred_output = MagicMock(
            side_effect=lambda mb: order.append("send")
        )
        manager = scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            kv_manager=MagicMock()
        )
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ):
            scheduler._pp_pd_send_deferred_output(0)
        self.assertEqual(
            order,
            [
                "enter-copy-stream",
                "wait-producer-on-stream",
                "copy-complete",
                "leave-copy-stream",
                "send",
            ],
        )
        copy_stream.wait_event.assert_called_once_with(ready)
        ready.synchronize.assert_not_called()
        manager.kv_manager.on_outputs_copied.assert_called_once_with(0, tokens)
        self.assertIs(
            scheduler.last_rank_comm_queue[0][1].tensors["next_token_ids"], tokens
        )

    def test_launch_batch_runs_without_schedule_stream(self):
        scheduler = _single_queue_scheduler()
        self.assertFalse(hasattr(scheduler, "schedule_stream"))
        cur_batch = SimpleNamespace(reqs=[], spec_algorithm=SpeculativeAlgorithm.NONE)
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
            0,
            SimpleNamespace(reqs=[], spec_algorithm=SpeculativeAlgorithm.NONE),
            None,
            [None],
            last_rank_comm_queue=[],
        )

        scheduler.forward_stream.wait_stream.assert_called_once_with(
            scheduler.schedule_stream
        )

    def _recv_outputs(self, scheduler):
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
        )
        received = {"next_token_ids": torch.zeros(2, dtype=torch.int64)}
        scheduler._pp_recv_dict_from_prev_stage = MagicMock(
            return_value=(received, None)
        )
        self.batch_result = SimpleNamespace(logits_output=None)
        scheduler._pp_prep_batch_result = MagicMock(return_value=self.batch_result)
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
        self.assertIs(batch_result, self.batch_result)
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
        scheduler.attn_tp_group = object()
        scheduler._pp_recv_typed_dict = MagicMock(return_value=(received, None))
        return scheduler

    def test_first_stage_receives_nothing(self):
        scheduler = self._scheduler({})
        scheduler.pp_group = SimpleNamespace(is_first_rank=True)
        self.assertIsNone(scheduler._pp_recv_proxy_tensors())
        scheduler._pp_recv_typed_dict.assert_not_called()


class TestWaveBuffers(CustomTestCase):
    def _wave_args(self, compile_policy, model=None):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_first_rank=False)
        scheduler.tp_group = SimpleNamespace()
        scheduler.world_group = SimpleNamespace()
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                compile_policy=compile_policy,
                model_config=SimpleNamespace(context_len=4096, hidden_size=16),
                spec_aux_config=SpecAuxHiddenStateConfig(),
                pp_group=scheduler.pp_group,
                model=SimpleNamespace() if model is None else model,
                dtype=torch.bfloat16,
                device="cpu",
            )
        )
        compile_policy.model_runner = scheduler.tp_worker.model_runner
        with patch.object(pp_scheduler, "build_pp_wave") as build:
            scheduler._pp_init_wave()
        return build.call_args.kwargs

    def _rows(self, compile_policy):
        return self._wave_args(compile_policy)["rows"]

    def _policy(self, token_buckets, bs_buckets):
        return DecoderTpuCompilePolicy(
            t_bucketer=Bucketer(name="Token", buckets=token_buckets),
            b_bucketer=Bucketer(name="Bs", buckets=bs_buckets),
        )

    def test_buffer_holds_a_token_bucket_above_the_extend_limit(self):
        """A batch padded to a configured token bucket above the extend limit
        (--tpu-compile-tokens 4096 8192 16384 with 12288-token chunks) fits."""
        with get_context().override_server_args(
            chunked_prefill_size=12288, enable_dynamic_chunking=False
        ):
            rows = self._rows(self._policy([4096, 8192, 16384], [1, 32]))
        self.assertEqual(rows, 16384)

    def test_buffer_holds_the_largest_decode_batch(self):
        self.assertEqual(self._rows(self._policy([64, 256], [1, 512])), 512)

    def test_wave_and_warmup_preserve_the_models_residual_dtype(self):
        policy = self._policy([64, 256], [1, 32])
        for model, dtype in (
            (SimpleNamespace(), torch.bfloat16),
            (SimpleNamespace(pp_transfer_dtype=torch.float32), torch.float32),
        ):
            with self.subTest(dtype=dtype):
                self.assertEqual(self._wave_args(policy, model)["dtype"], dtype)
                proxy = policy._pp_warmup_kwargs(64)["pp_proxy_tensors"]
                for name in ("hidden_states", "residual"):
                    self.assertEqual(proxy[name].dtype, dtype)


if __name__ == "__main__":
    unittest.main()
