"""Grammar prefill must align physical logits and host metadata to live rows."""

import sys
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from llguidance import LLTokenizer, grammar_from

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.sampling.sampling_batch_info import HostPaddedSamplingInfo
from sglang_tpu.hooks.experimental_spec import sample_host_padded_grammar
from sglang_tpu.sampler import TpuSampler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "live_rows,logit_rows", [(1, 1), (1, 4), (2, 2), (2, 4), (3, 3), (3, 4), (4, 4)]
)
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("transform_rows", [None, "live", "padded"])
@pytest.mark.parametrize("params_on_cpu", [False, True])
def test_grammar_prefill_matches_live_sampling(
    live_rows, logit_rows, greedy, transform_rows, params_on_cpu
):
    tokenizer = LLTokenizer("byte")
    a, b, x = tokenizer.tokenize_str("abx")
    vocab = tokenizer.vocab_size
    grammars = [
        GuidanceGrammar(tokenizer, grammar_from("regex", "a[0-9]{4}")),
        None,
        GuidanceGrammar(tokenizer, grammar_from("regex", "b+")),
        None,
    ][:live_rows]
    transforms = {}
    if transform_rows is not None:
        rows = live_rows if transform_rows == "live" else 4
        transforms = {
            "acc_additive_penalties": torch.zeros(rows, vocab),
            "acc_scaling_penalties": torch.ones(rows, vocab),
            "logit_bias": torch.zeros(rows, vocab),
        }
        transforms["acc_additive_penalties"][:, x] = -0.5
        transforms["acc_scaling_penalties"][:, x] = 1.1
        transforms["logit_bias"][:, x] = 0.2
    source = HostPaddedSamplingInfo(
        temperatures=torch.full((4, 1), 1.0 if greedy else 0.7),
        top_ps=torch.ones(4),
        top_ks=torch.full((4,), vocab, dtype=torch.int32),
        min_ps=torch.zeros(4),
        is_all_greedy=greedy,
        is_any_greedy=greedy,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab,
        device="cpu",
        real_batch_size=live_rows,
        grammars=grammars,
        sampling_params_on_cpu=params_on_cpu,
        **transforms,
    )
    source.update_regex_vocab_mask()
    original_mask = source.grammar_mask
    original_tensors = {
        field.name: value
        for field in fields(source)
        if isinstance(value := getattr(source, field.name), torch.Tensor)
    }
    reference = SamplingBatchInfo(
        **{
            field.name: getattr(source, field.name)
            for field in fields(SamplingBatchInfo)
        }
    )
    reference = replace(
        reference,
        **source.host_params_at(live_rows),
        **{name: value[:live_rows].clone() for name, value in transforms.items()},
    )
    logits = torch.zeros(logit_rows, vocab)
    logits[:, x] = 10
    hidden = torch.zeros(16, 2)
    output = LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
    reference_output = LogitsProcessorOutput(
        next_token_logits=logits[:live_rows].clone()
    )
    batch = SimpleNamespace(
        sampling_info=source,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        top_logprobs_nums=[0] * live_rows,
        token_ids_logprobs=[None] * live_rows,
        seq_lens=torch.full((live_rows,), 10),
    )
    runner = object.__new__(ModelRunner)
    runner.sampling_observer = None
    runner.ngram_embedding_manager = mock.Mock()
    with (
        get_context().override_server_args(sampling_backend="pytorch"),
        mock.patch(
            "sglang.srt.layers.sampler.get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(device_group=None)),
        ),
        mock.patch.object(TpuSampler, "_sync_token_ids_across_tp"),
        torch.random.fork_rng(),
    ):
        runner.sampler = TpuSampler()
        torch.manual_seed(123)
        expected = ModelRunner.sample(
            runner,
            reference_output,
            SimpleNamespace(**{**vars(batch), "sampling_info": reference}),
        )
        torch.manual_seed(123)
        actual = sample_host_padded_grammar(ModelRunner.sample, runner, output, batch)

    assert torch.equal(actual, expected)
    assert torch.equal(output.next_token_logits, reference_output.next_token_logits)
    assert actual.tolist() == [a, x, b, x][:live_rows]
    assert output.hidden_states is hidden
    assert batch.sampling_info is source
    assert source.grammar_mask is original_mask
    assert source.grammars is grammars
    assert all(
        getattr(source, name) is value for name, value in original_tensors.items()
    )


def test_sampling_restores_snapshot_on_error():
    source = HostPaddedSamplingInfo(
        temperatures=torch.ones(4, 1),
        top_ps=torch.ones(4),
        top_ks=torch.ones(4, dtype=torch.int32),
        min_ps=torch.zeros(4),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=8,
        real_batch_size=3,
        grammars=[object(), None, None],
    )
    batch = SimpleNamespace(sampling_info=source, forward_mode=ForwardMode.EXTEND)
    output = LogitsProcessorOutput(next_token_logits=torch.zeros(4, 8))
    with pytest.raises(RuntimeError, match="sample failed"):
        sample_host_padded_grammar(
            mock.Mock(side_effect=RuntimeError("sample failed")), None, output, batch
        )
    assert batch.sampling_info is source


@pytest.mark.parametrize(
    "host_padded,has_grammar,mode",
    [
        (False, True, ForwardMode.EXTEND),
        (True, False, ForwardMode.EXTEND),
        (True, True, ForwardMode.DECODE),
    ],
)
def test_other_sampling_modes_delegate_unchanged(host_padded, has_grammar, mode):
    cls = HostPaddedSamplingInfo if host_padded else SamplingBatchInfo
    info = cls(
        temperatures=torch.ones(4, 1),
        top_ps=torch.ones(4),
        top_ks=torch.ones(4, dtype=torch.int32),
        min_ps=torch.zeros(4),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=8,
        grammars=[object()] if has_grammar else None,
    )
    batch = SimpleNamespace(sampling_info=info, forward_mode=mode)
    logits = torch.zeros(4, 8)
    output = LogitsProcessorOutput(next_token_logits=logits)
    original = mock.Mock(return_value=mock.sentinel.tokens)
    result = sample_host_padded_grammar(original, mock.sentinel.runner, output, batch)
    original.assert_called_once_with(mock.sentinel.runner, output, batch)
    assert result is mock.sentinel.tokens
    assert batch.sampling_info is info
    assert output.next_token_logits is logits


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
