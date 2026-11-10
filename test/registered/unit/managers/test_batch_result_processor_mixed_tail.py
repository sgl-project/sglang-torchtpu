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


if __name__ == "__main__":
    unittest.main()
