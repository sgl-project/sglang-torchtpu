"""A spec tail that finishes inside a mixed-verify batch must commit its run before release."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_processor(case) -> SchedulerBatchResultProcessor:
    override = get_context().override_server_args()
    override.install()
    case.addCleanup(override.restore)
    metrics_reporter = Mock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


class _Req:
    def __init__(self, rid, finish_after=None):
        self.rid = rid
        self.inflight_middle_chunks = 0
        self.return_hidden_states = False
        self.hidden_states = []
        self.is_retracted = False
        self.output_ids = []
        self.time_stats = Mock()
        self.return_logprob = False
        self.return_sampling_mask = False
        self.return_routed_experts = False
        self.return_pooled_hidden_states = False
        self.return_indexer_topk = False
        self.grammar = None
        self.require_reasoning = False
        self.customized_info = None
        self.beam_group = None
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0
        self.kv = SimpleNamespace(kv_committed_len=100)
        self._finish_after = finish_after
        self._finished = False

    def finished(self):
        return self._finished

    def update_finish_state(self, new_accept_len=1):
        if (
            self._finish_after is not None
            and len(self.output_ids) >= self._finish_after
        ):
            self._finished = True


class TestMixedTailCommitOnFinish(CustomTestCase):
    def _run(self, flag):
        prefill, tail = _Req("prefill"), _Req("tail", finish_after=3)
        batch = SimpleNamespace(
            reqs=[prefill, tail],
            decoding_reqs=[tail],
            mix_verify_tails=1,
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            return_logprob=False,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=SimpleNamespace(
                hidden_states=None, customized_info=None, sampling_mask_output=None
            ),
            next_token_ids=torch.tensor([7, 0, 0, 0, 11, 12, 13, 0]),
            accept_lens=torch.tensor([1, 3]),
            speculative_num_draft_tokens=4,
            extend_input_len_per_req=[5, 4],
            extend_logprob_start_len_per_req=None,
            grammar_advanced=False,
            can_run_cuda_graph=False,
            skipped_output_comm=False,
        )
        released = {}
        mod = "sglang.srt.managers.scheduler_components.batch_result_processor"
        env = {"SGLANG_MIXED_TAIL_COMMIT_ON_FINISH": "1" if flag else "0"}
        with (
            patch.dict(os.environ, env),
            patch(f"{mod}.checkpoint_kv_cache"),
            patch(
                f"{mod}.release_kv_cache",
                side_effect=lambda req, *a, **k: released.update(
                    {req.rid: req.kv.kv_committed_len}
                ),
            ),
            patch(
                f"{mod}.get_memory", return_value=SimpleNamespace(enable_hisparse=False)
            ),
        ):
            _make_processor(self).process_batch_result_prefill(batch, result)
        self.assertEqual(tail.output_ids, [11, 12, 13])
        return released

    def test_finishing_tail_commits_its_run_before_release(self):
        self.assertEqual(self._run(flag=True), {"tail": 103})

    def test_without_the_flag_the_run_is_not_committed(self):
        self.assertEqual(self._run(flag=False), {"tail": 100})


def _dflash_v2_result_input():
    from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

    return make_draft_input_v2(
        bonus_tokens=torch.zeros(1, dtype=torch.int64),
        new_seq_lens=torch.ones(1, dtype=torch.int64),
    )


def _experimental_result_input():
    from sglang_tpu.experimental.speculative.block_info import (
        HostPaddedBlockDraftInput,
    )

    return HostPaddedBlockDraftInput(
        bonus_tokens=torch.zeros(1), new_seq_lens=torch.zeros(1)
    )


class TestSpecRunCommitBothWorkers(CustomTestCase):
    """A request finishing on a decode run releases with the run committed, so the
    radix insert and session leaf see it, for both TPU DFlash workers' results:
    DFlashWorkerV2 at an adaptive verify width below the block, and the experimental
    TpuBlockSpecWorker at its full num_verify_rows stride."""

    def _run(self, *, stride, draft_input):
        req = _Req("req", finish_after=3)
        req.finished_reason = None
        req.multimodal_inputs = None
        req.session = None
        req.require_reasoning = False
        req.to_finish = None
        req.decode_batch_idx = 0
        req.update_spec_correct_drafts_histogram = lambda n: None
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            has_grammar=False,
            return_logprob=False,
            mamba_track_mask_cpu=None,
            batch_size=lambda: 1,
        )
        next_token_ids = torch.zeros(stride, dtype=torch.int64)
        next_token_ids[:3] = torch.tensor([11, 12, 13])
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=SimpleNamespace(
                hidden_states=None, customized_info=None, sampling_mask_output=None
            ),
            next_token_ids=next_token_ids,
            accept_lens=torch.tensor([3]),
            speculative_output_stride=None,
            speculative_num_draft_tokens=stride,
            num_non_draft_tokens_per_req=1,
            block_accept_lens=None,
            cap_lens=None,
            grammar_advanced=False,
            can_run_cuda_graph=False,
            next_draft_input=draft_input,
            get_num_generated_tokens=lambda bs: 3,
        )
        released = {}
        mod = "sglang.srt.managers.scheduler_components.batch_result_processor"
        processor = _make_processor(self)
        with (
            patch(
                f"{mod}.release_kv_cache",
                side_effect=lambda req, *a, **k: released.update(
                    {req.rid: req.kv.kv_committed_len}
                ),
            ),
            patch.object(
                SchedulerBatchResultProcessor, "snapshot_auxiliary_output_starts"
            ),
            patch.object(
                SchedulerBatchResultProcessor, "materialize_sampling_mask_output"
            ),
            patch.object(
                SchedulerBatchResultProcessor, "_maybe_collect_routed_experts"
            ),
            patch.object(SchedulerBatchResultProcessor, "_maybe_collect_indexer_topk"),
            patch.object(
                SchedulerBatchResultProcessor, "_maybe_collect_customized_info"
            ),
            patch.object(SchedulerBatchResultProcessor, "_mamba_prefix_cache_update"),
        ):
            processor.process_batch_result_decode(batch, result)
        self.assertEqual(req.output_ids, [11, 12, 13])
        return released

    def test_dflash_worker_v2_narrow_verify_width(self):
        self.assertEqual(
            self._run(stride=4, draft_input=_dflash_v2_result_input()), {"req": 103}
        )

    def test_experimental_spec_worker(self):
        self.assertEqual(
            self._run(stride=8, draft_input=_experimental_result_input()),
            {"req": 103},
        )


if __name__ == "__main__":
    unittest.main()
