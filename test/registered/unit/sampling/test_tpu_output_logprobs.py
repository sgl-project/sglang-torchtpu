"""Returned TopK values keep exact request lengths without TPU row slices."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import OutputLogprobProcessor
from sglang.srt.managers import utils as result_utils
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import output_logprobs

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("requested", [[0, 1, 3, 50], [73, 8, 0, 5], [1, 1, 1, 1]])
def test_top_logprobs_copy_bucket_and_preserve_stable_ties(requested):
    logprobs = torch.arange(4 * 73).reshape(4, 73).remainder(11).float()
    expected_values, expected_indices = torch.sort(
        logprobs, dim=-1, descending=True, stable=True
    )
    with patch.object(torch.Tensor, "cpu", side_effect=lambda x: x.clone()) as copy:
        batched = output_logprobs._get_top_logprobs(logprobs, requested)
        copy.assert_not_called()
        values, indices = batched.copy_to_cpu(torch.Tensor.cpu)

    width = min(1 << (max(requested) - 1).bit_length(), 73)
    assert [tuple(args[0].shape) for args, _ in copy.call_args_list] == [
        (4, width),
        (4, width),
    ]
    for row, k in enumerate(requested):
        assert torch.equal(values[row], expected_values[row, :k])
        assert torch.equal(indices[row], expected_indices[row, :k])
        assert values[row].device.type == indices[row].device.type == "cpu"


@pytest.mark.parametrize("vocab_size", [1, 64, 73])
def test_warmup_sorts_once_and_covers_all_output_widths(vocab_size):
    logprobs = torch.zeros(3, vocab_size)
    with (
        patch.object(torch, "sort", wraps=torch.sort) as sort,
        patch.object(torch.Tensor, "cpu", side_effect=lambda x: x.clone()) as copy,
    ):
        output_logprobs.warmup_output_top_logprobs(logprobs)

    sort.assert_called_once()
    expected_widths = sorted(
        {min(1 << (k - 1).bit_length(), vocab_size) for k in range(1, vocab_size + 1)}
    )
    assert [args[0].shape[-1] for args, _ in copy.call_args_list] == [
        width for width in expected_widths for _ in range(2)
    ]


def test_warmup_and_serving_use_the_same_compiled_sort():
    logprobs = torch.zeros(2, 7)
    with patch.object(
        output_logprobs, "_sort_logprobs", wraps=output_logprobs._sort_logprobs
    ) as sort:
        output_logprobs.warmup_output_top_logprobs(logprobs)
        output_logprobs._get_top_logprobs(logprobs, [3, 1])
    assert sort.call_count == 2
    assert all(args[0] is logprobs for args, _ in sort.call_args_list)


def test_dynamo_model_precompiles_and_serves_the_native_sort():
    logits = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    compiled = Mock()
    with (
        envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
        patch.object(output_logprobs, "_compiled_sort", return_value=compiled),
        patch.object(output_logprobs, "current_warmup_queue", return_value=None),
        patch.object(
            output_logprobs,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(
                    enable_torch_compile=True, tpu_compile_tracer="dynamo"
                )
            ),
        ),
    ):
        output_logprobs.precompile_output_top_logprobs(logits)
        result = output_logprobs._sort_logprobs(logits)
    compiled.precompile.assert_called_once_with(logits, async_compile=True)
    compiled.precompile.return_value.wait.assert_called_once_with()
    compiled.assert_called_once_with(logits)
    assert result is compiled.return_value


def test_processor_preserves_clamping_and_sampled_token_logprobs():
    logprobs = torch.tensor([[-float("inf"), -2.0, -2.0], [-3.0, -1.0, -2.0]])
    token_ids = torch.tensor([0, 2])
    expected = OutputLogprobProcessor().compute_logprobs(
        logprobs.clone(), [0, 0], [None, None], token_ids
    )
    result = output_logprobs.TpuOutputLogprobProcessor().compute_logprobs(
        logprobs, [2, 0], [None, None], token_ids
    )

    assert torch.equal(result.token_logprobs, expected.token_logprobs)
    values, indices = result.batched_top_logprobs.copy_to_cpu(lambda x: x)
    assert indices[0].tolist() == [1, 2]
    assert values[0].tolist() == [-2.0, -2.0]
    assert indices[1].numel() == values[1].numel() == 0


def test_no_requested_top_logprobs_skips_sort():
    with patch.object(torch, "sort", side_effect=AssertionError("unexpected TopK")):
        result = output_logprobs.TpuOutputLogprobProcessor().compute_logprobs(
            torch.zeros(1, 3), [0], [None], torch.tensor([0])
        )
    assert result.top_logprobs_val is None
    assert result.top_logprobs_idx is None
    assert result.batched_top_logprobs is None


def test_score_only_uses_same_bucketed_output():
    logits = torch.tensor([[0.0, 2.0, 2.0, 1.0]])
    result = output_logprobs.TpuOutputLogprobProcessor().compute_logprobs_only(
        logits, [3], [None], lambda x: x
    )
    expected = torch.log_softmax(logits, dim=-1)[0, [1, 2, 3]]
    values, indices = result.batched_top_logprobs.copy_to_cpu(lambda x: x)
    assert indices[0].tolist() == [1, 2, 3]
    assert torch.equal(values[0], expected)


def test_result_copy_submits_batched_transfers_before_recording_event():
    logits_output = LogitsProcessorOutput(next_token_logits=None)
    logprobs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    logprob_result = output_logprobs.TpuOutputLogprobProcessor().compute_logprobs(
        logprobs, [3, 1], [None, None], torch.tensor([3, 0])
    )
    logprob_result.write_output_to(logits_output)
    batched = logits_output.batched_output_top_logprobs
    pending = []
    events = []

    def submit_copy(tensor):
        if tensor is batched.values or tensor is batched.indices:
            host = torch.zeros_like(tensor)
            pending.append((tensor, host))
            events.append("copy")
            return host
        return tensor

    def record():
        events.append("record")
        for source, host in pending:
            host.copy_(source)

    result = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=torch.tensor([3, 0]),
        copy_done=SimpleNamespace(record=record),
    )
    with patch.object(result_utils, "_async_d2h", side_effect=submit_copy):
        result.copy_to_cpu(return_logprob=True)

    assert events == ["copy", "copy", "record"]
    assert logits_output.batched_output_top_logprobs is None
    assert [x.tolist() for x in logits_output.next_token_top_logprobs_val] == [
        [4.0, 3.0, 2.0],
        [4.0],
    ]
    assert [x.tolist() for x in logits_output.next_token_top_logprobs_idx] == [
        [3, 2, 1],
        [0],
    ]


def test_batched_outputs_survive_pipeline_relay_without_host_copy():
    output = LogitsProcessorOutput(next_token_logits=None)
    pending = output_logprobs._get_top_logprobs(torch.zeros(2, 7), [3, 0])
    output.batched_output_top_logprobs = pending
    result = GenerationBatchResult(logits_output=output)
    payload = result_utils.get_logprob_dict_from_result(result)
    restored, _, _ = result_utils.get_logprob_from_pp_outputs(payload)
    assert restored.batched_output_top_logprobs.values is pending.values
    assert restored.batched_output_top_logprobs.indices is pending.indices
    assert restored.batched_output_top_logprobs.lengths == [3, 0]


@pytest.mark.parametrize("prefill", [True, False])
def test_non_overlap_consumer_copies_pending_outputs(prefill):
    output = LogitsProcessorOutput(next_token_logits=None)
    logprobs = torch.tensor([[-1.0, -3.0, -2.0]])
    output_logprobs.TpuOutputLogprobProcessor().compute_logprobs(
        logprobs, [2], [None], torch.tensor([0])
    ).write_output_to(output)
    batch = SimpleNamespace(
        return_logprob=True, spec_algorithm=SimpleNamespace(is_none=lambda: True)
    )
    processor = object.__new__(SchedulerBatchResultProcessor)
    if prefill:
        processor.move_logprobs_to_cpu(batch=batch, logits_output=output)
    else:
        processor._normalize_decode_outputs(
            batch=batch,
            result=GenerationBatchResult(logits_output=output),
            logits_output=output,
            next_token_ids=torch.tensor([0]),
        )
    assert output.batched_output_top_logprobs is None
    assert output.next_token_top_logprobs_val == [[-1.0, -2.0]]
    assert output.next_token_top_logprobs_idx == [[0, 2]]


def test_k_above_vocabulary_still_fails():
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        output_logprobs._get_top_logprobs(torch.zeros(1, 3), [4])
