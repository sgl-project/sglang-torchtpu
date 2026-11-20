"""Prefill warmup must cover live relay rows without changing unwarmed draws."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.compilation.compiled_op import warmup_compiled_op
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import sampler
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.host_padding_warmup import warmup_host_relay
from sglang_tpu.host_padding_relay import HostPaddingFutureMap

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_sampling_warmup_preserves_rng_and_only_admits_matching_inputs():
    warmed = torch.ones(3, 64)
    with (
        patch.object(sampler, "_warmed_sampling_draws", set()),
        patch.object(
            sampler,
            "_compiled_sample_exponential_race",
            wraps=sampler._compiled_sample_exponential_race,
        ) as compiled,
        torch.random.fork_rng(),
    ):
        state = torch.random.get_rng_state()
        warmup_compiled_op(sampler.sample_exponential_race, warmed)
        assert torch.equal(torch.random.get_rng_state(), state)
        for probs in (warmed, torch.ones(20, 64), warmed.double(), torch.ones(3, 32)):
            compiled.reset_mock()
            torch.manual_seed(7)
            expected = sampler.sample_exponential_race.__wrapped__(probs)
            expected_next = torch.rand(8)
            torch.manual_seed(7)
            actual = sampler.sample_exponential_race(probs)
            actual_next = torch.rand(8)
            assert torch.equal(actual, expected)
            assert torch.equal(actual_next, expected_next)
            assert compiled.call_count == int(probs is warmed)


def test_sampling_inside_a_compiled_graph_does_not_nest_compilation():
    probs = torch.ones(3, 64)
    key = (probs.shape, probs.dtype, probs.device)
    with (
        patch.object(sampler, "_warmed_sampling_draws", {key}),
        patch.object(
            sampler,
            "_compiled_sample_exponential_race",
            side_effect=AssertionError("nested compilation"),
        ),
    ):
        draw = torch.compile(
            sampler.sample_exponential_race, backend="eager", fullgraph=True
        )
        assert draw(probs).shape == (3,)


@pytest.mark.parametrize("max_live,extend_cap", [(16, 16), (5, 16), (16, 5)])
def test_relay_warmup_covers_partial_prefill_counts(max_live, extend_cap):
    bucketer = Bucketer("batch", [1, 2, 4, 8, 16])
    runner = SimpleNamespace(
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(max_live + 1, 1, dtype=torch.int32),
            _tpu_relay_bucketer=bucketer,
        ),
    )
    stash = HostPaddingFutureMap.stash
    covered = set()

    def record(relay, indices, payload):
        covered.add((len(payload.bonus_tokens), payload.bonus_tokens.dtype))
        assert len(indices) <= max_live
        return stash(relay, indices, payload)

    with patch.object(HostPaddingFutureMap, "stash", record):
        warmup_host_relay(runner, bucketer, max_live, extend_cap=extend_cap)
    for rows in {
        *bucketer.buckets,
        extend_cap,
        *range(1, min(max_live, extend_cap) + 1),
    }:
        for dtype in (torch.int64, torch.int32):
            assert (rows, dtype) in covered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
