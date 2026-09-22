"""Unit tests for the DP-attention MLP-sync pad/unpad round-trip.

``prepare_mlp_sync_batch`` pads per-request tensors (positions / seq_lens /
req_pool_indices) by appending dummy rows after the real ones so all DP ranks
agree on tensor shapes. ``post_forward_mlp_sync_batch`` must slice them back so
post-forward consumers — seeded sampling (which asserts positions rows ==
sampling rows), ngram token-table updates — never see the padding.

Pure dataclass logic — CPU only.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.model_executor.forward_batch_info as forward_batch_info
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mock_model_runner(seq_len_fill_value: int = 1) -> MagicMock:
    runner = MagicMock()
    runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = (
        seq_len_fill_value
    )
    runner.attn_backend.get_cpu_graph_seq_len_fill_value.return_value = (
        seq_len_fill_value
    )
    return runner


def _run_prepare_mlp_sync_batch(
    fb: ForwardBatch, *, hybrid_ssm: bool, hybrid_decode_joins_extend: bool = True
) -> None:
    """Drive prepare_mlp_sync_batch under MAX_LEN with attention TP 1.

    The compile policy never materialises a rank itself, so an empty rank
    exercises the shared fabricated-row branch.
    """
    runner = _mock_model_runner()
    runner.enable_elastic_ep = False
    runner.is_draft_worker = False
    runner.compile_policy.dp_hybrid_decode_joins_extend = hybrid_decode_joins_extend
    runner.compile_policy.get_dp_padded_token_counts.side_effect = (
        lambda _fb, _runner, counts, logprob_counts, **_: (counts, logprob_counts)
    )
    runner.compile_policy.materialize_dp_idle_extend.return_value = False
    parallel = SimpleNamespace(attn_tp_size=1, attn_dp_rank=0)
    exec_ctx = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=None))
        )
    )
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch.object(forward_batch_info, "get_parallel", return_value=parallel))
        enter(patch.object(forward_batch_info, "get_exec", return_value=exec_ctx))
        enter(patch.object(forward_batch_info, "_is_cpu", True))
        enter(
            patch.object(
                forward_batch_info,
                "mambaish_config",
                return_value=object() if hybrid_ssm else None,
            )
        )
        enter(patch.object(forward_batch_info, "set_dp_buffer_len"))
        enter(patch.object(forward_batch_info, "set_is_extend_in_batch"))
        enter(
            patch.object(
                DpPaddingMode, "get_dp_padding_mode", return_value=DpPaddingMode.MAX_LEN
            )
        )
        enter(
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            )
        )
        enter(patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=False))
        enter(
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=1,
            )
        )
        fb.prepare_mlp_sync_batch(runner)


def _decode_batch_joining_extend_step(num_reqs: int) -> ForwardBatch:
    """A DECODE rank while another DP rank prefills 300 tokens (MAX_LEN)."""
    seq_lens = torch.arange(7, 7 + num_reqs)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE if num_reqs else ForwardMode.IDLE,
        batch_size=num_reqs,
        input_ids=torch.arange(num_reqs),
        req_pool_indices=torch.arange(num_reqs),
        seq_lens=seq_lens,
        out_cache_loc=torch.arange(num_reqs),
        seq_lens_sum=int(seq_lens.sum()),
        positions=seq_lens - 1,
        seq_lens_cpu=seq_lens.clone(),
        orig_seq_lens=seq_lens.clone(),
        is_extend_in_batch=True,
        global_num_tokens_cpu=[num_reqs, 300],
        global_num_tokens_gpu=torch.zeros(2, dtype=torch.int64),
        global_num_tokens_for_logprob_cpu=[num_reqs, 300],
    )


def _logits_output(num_rows: int) -> LogitsProcessorOutput:
    return LogitsProcessorOutput(
        next_token_logits=torch.randn(num_rows, 16), hidden_states=None
    )


def _global_top_k_output(num_rows: int) -> LogitsProcessorOutput:
    return LogitsProcessorOutput(
        next_token_logits=None,
        next_token_global_top_k_logits=torch.randn(num_rows, 8),
        next_token_global_top_k_token_ids=torch.arange(num_rows * 8).reshape(
            num_rows, 8
        ),
        hidden_states=None,
    )


class TestMlpSyncPadUnpad(CustomTestCase):
    def test_init_mlp_sync_metadata_scales_speculative_request_width(self):
        spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=2,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=torch.arange(8),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([5, 6]),
            out_cache_loc=torch.arange(8),
            seq_lens_sum=11,
            positions=torch.arange(8),
            spec_info=spec_info,
        )
        batch = SimpleNamespace(
            global_num_tokens=[2, 0, 3],
            global_num_tokens_for_logprob=[2, 0, 3],
            can_run_decode_cuda_graph=True,
        )

        fb.init_mlp_sync_metadata(batch, torch.device("cpu"))

        self.assertEqual(fb.original_global_num_tokens_cpu, [2, 0, 3])
        self.assertEqual(fb.global_num_tokens_cpu, [8, 0, 12])
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [4, 0, 6])
        torch.testing.assert_close(fb.global_num_tokens_gpu, torch.tensor([8, 0, 12]))
        torch.testing.assert_close(
            fb.global_num_tokens_for_logprob_gpu, torch.tensor([4, 0, 6])
        )
        self.assertTrue(fb.can_run_decode_cuda_graph)

    def test_draft_input_without_hidden_states_can_be_padded(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            hidden_states=None,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([11]),
            req_pool_indices=torch.tensor([5]),
            seq_lens=torch.tensor([7]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=7,
            positions=torch.tensor([6]),
            seq_lens_cpu=torch.tensor([7]),
            lora_ids=[None],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=2, bs=1)

        self.assertIsNone(spec_info.hidden_states)

    def test_dp_cuda_graph_batch_size_uses_raw_request_counts(self):
        fb = SimpleNamespace(original_global_num_tokens_cpu=[3, 11, 7])
        self.assertEqual(DecodeCudaGraphRunner._max_dp_batch_size(fb), 11)

        fb.original_global_num_tokens_cpu = None
        with self.assertRaisesRegex(RuntimeError, "raw per-rank request counts"):
            DecodeCudaGraphRunner._max_dp_batch_size(fb)

    def test_decode_post_forward_unpads_per_request_tensors(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=3,
            input_ids=torch.tensor([11, 12, 13]),
            req_pool_indices=torch.tensor([5, 6, 7]),
            seq_lens=torch.tensor([7, 8, 9]),
            out_cache_loc=torch.tensor([0, 1, 2]),
            seq_lens_sum=24,
            positions=torch.tensor([6, 7, 8]),
            seq_lens_cpu=torch.tensor([7, 8, 9]),
            lora_ids=[None, None, None],
        )
        # Mirror the decode arm of prepare_mlp_sync_batch: record the original
        # batch size, adopt the synced (padded) one, then pad the inputs.
        padded = 5
        fb._original_batch_size = fb.batch_size
        fb.batch_size = padded
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=padded, bs=padded)

        # Padding appends dummy rows after the real ones.
        self.assertEqual(fb.positions.shape[0], padded)
        self.assertEqual(fb.seq_lens.shape[0], padded)
        self.assertEqual(fb.req_pool_indices.shape[0], padded)
        torch.testing.assert_close(fb.positions[:3], torch.tensor([6, 7, 8]))

        logits_output = _logits_output(padded)
        fb.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(fb.batch_size, 3)
        torch.testing.assert_close(fb.positions, torch.tensor([6, 7, 8]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([7, 8, 9]))
        torch.testing.assert_close(fb.req_pool_indices, torch.tensor([5, 6, 7]))
        torch.testing.assert_close(fb.seq_lens_cpu, torch.tensor([7, 8, 9]))
        self.assertEqual(logits_output.next_token_logits.shape[0], 3)
        # Seeded sampling asserts positions rows == sampled (real) rows.
        self.assertEqual(fb.positions.shape[0], fb.batch_size)

        global_top_k_output = _global_top_k_output(padded)
        fb.post_forward_mlp_sync_batch(global_top_k_output)
        self.assertIsNone(global_top_k_output.next_token_logits)
        self.assertEqual(global_top_k_output.next_token_global_top_k_logits.shape[0], 3)
        self.assertEqual(
            global_top_k_output.next_token_global_top_k_token_ids.shape[0], 3
        )

    def test_hybrid_rank_with_live_decode_joins_extend_step_per_request(self):
        """A mamba-hybrid rank holding decode work must not be treated as idle.

        Under MAX_LEN a decode rank is relabelled EXTEND whenever a peer
        prefills. Where the compile policy opts in, it keeps its requests as
        1-token extends like every other model instead of asserting.
        """
        fb = _decode_batch_joining_extend_step(num_reqs=2)

        _run_prepare_mlp_sync_batch(fb, hybrid_ssm=True)

        self.assertIs(fb.forward_mode, ForwardMode.EXTEND)
        self.assertIs(fb._original_forward_mode, ForwardMode.DECODE)
        self.assertEqual(fb.batch_size, 2)
        torch.testing.assert_close(fb.seq_lens, torch.tensor([7, 8]))
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1])
        self.assertEqual(fb.extend_prefix_lens.tolist(), [6, 7])
        self.assertEqual(fb.extend_start_loc.tolist(), [0, 1])
        # Token-level inputs pad to the global max; the request axis does not.
        self.assertEqual(fb.positions.shape[0], 300)
        self.assertEqual(fb.req_pool_indices.shape[0], 2)

    def test_hybrid_rank_keeps_the_empty_rank_assert_without_policy_opt_in(self):
        # The upstream behaviour for backends whose decode bookkeeping does
        # not run on an extend step.
        fb = _decode_batch_joining_extend_step(num_reqs=2)

        with self.assertRaisesRegex(AssertionError, "expects an empty rank"):
            _run_prepare_mlp_sync_batch(
                fb, hybrid_ssm=True, hybrid_decode_joins_extend=False
            )

    def test_empty_hybrid_rank_still_gets_the_fabricated_row(self):
        # The idle-rank conversion stays: one request spanning the padded block,
        # counted as real tokens because the recurrent state update needs them.
        fb = _decode_batch_joining_extend_step(num_reqs=0)

        _run_prepare_mlp_sync_batch(fb, hybrid_ssm=True)

        self.assertIs(fb.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(fb.batch_size, 1)
        self.assertEqual(fb.seq_lens.tolist(), [300])
        self.assertEqual(fb.extend_seq_lens.tolist(), [300])
        self.assertEqual(fb.num_token_non_padded_cpu, 300)

    def test_extend_post_forward_unpads_positions(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(7),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([3, 4]),
            out_cache_loc=torch.arange(7),
            seq_lens_sum=7,
            positions=torch.tensor([0, 1, 2, 0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([3, 4]),
            lora_ids=[None, None],
        )
        # Extend keeps batch_size; only token-level tensors get padded.
        fb._original_batch_size = fb.batch_size
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=10, bs=2)

        self.assertEqual(fb.positions.shape[0], 10)

        logits_output = _logits_output(10)
        fb.post_forward_mlp_sync_batch(logits_output)

        torch.testing.assert_close(fb.positions, torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([3, 4]))
        # sample() derives prefill sampling positions from seq_lens - 1, so the
        # row count must match the real request count.
        self.assertEqual((fb.seq_lens - 1).shape[0], fb.batch_size)
        self.assertEqual(logits_output.next_token_logits.shape[0], 7)

        global_top_k_output = _global_top_k_output(10)
        fb.post_forward_mlp_sync_batch(global_top_k_output)
        self.assertIsNone(global_top_k_output.next_token_logits)
        self.assertEqual(global_top_k_output.next_token_global_top_k_logits.shape[0], 2)
        self.assertEqual(
            global_top_k_output.next_token_global_top_k_token_ids.shape[0], 2
        )

    def test_draft_extend_dummy_request_pads_cpu_and_gpu_lens(self):
        spec_info = MagicMock()
        spec_info.num_tokens_per_req = 4
        spec_info.is_draft_input.return_value = False
        fb = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=1,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            extend_seq_lens=torch.empty(0, dtype=torch.int32),
            extend_prefix_lens=torch.empty(0, dtype=torch.int64),
            extend_seq_lens_cpu=[],
            extend_prefix_lens_cpu=[],
            extend_logprob_start_lens_cpu=[],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=4, bs=1)

        torch.testing.assert_close(
            fb.extend_seq_lens, torch.tensor([4], dtype=torch.int32)
        )
        torch.testing.assert_close(fb.extend_prefix_lens, torch.tensor([0]))
        self.assertEqual(fb.extend_seq_lens_cpu, [4])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0])


if __name__ == "__main__":
    unittest.main()
