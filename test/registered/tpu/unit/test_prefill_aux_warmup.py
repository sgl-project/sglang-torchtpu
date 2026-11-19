"""Check sampler RNG parity and live-row relay cache reuse on TPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch_tpu._internal import execution_mode

from sglang.srt.compilation import compiled_op as compiled_op_module
from sglang.srt.compilation.compiled_op import warmup_compiled_op
from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu import sampler
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.host_padding_warmup import warmup_host_relay
from sglang_tpu.compilation.warmup_queue import collect_warmups
from sglang_tpu.host_padding_relay import HostPaddingFutureMap

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "compile-tracer"],
)


@pytest.mark.parametrize("width", [64, 1024])
def test_compiled_sampling_preserves_tokens_and_rng(width):
    with (
        execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER),
        envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        patch.object(
            compiled_op_module,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(tpu_compile_tracer="direct")
            ),
        ),
        patch.object(sampler, "_warmed_sampling_draws", set()),
    ):
        for rows in (*range(1, 17), 17, 20, 31, 63):
            probs = torch.softmax(
                torch.randn(rows, width, generator=torch.Generator().manual_seed(7)),
                dim=-1,
            ).to("tpu")
            if rows <= 16:
                state = torch.tpu.get_rng_state()
                with collect_warmups():
                    warmup_compiled_op(sampler.sample_exponential_race, probs)
                assert torch.equal(torch.tpu.get_rng_state(), state)
            torch.tpu.manual_seed(rows)
            expected = sampler.sample_exponential_race.__wrapped__(probs).cpu()
            expected_next = torch.rand(8, device="tpu").cpu()
            torch.tpu.manual_seed(rows)
            before = torch.tpu._get_cache_misses()
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = sampler.sample_exponential_race(probs).cpu()
            assert torch.tpu._get_cache_misses() == before
            actual_next = torch.rand(8, device="tpu").cpu()
            assert torch.equal(actual, expected)
            assert torch.equal(actual_next, expected_next)


def test_prefill_host_relay_uses_warmed_programs():
    bucketer = Bucketer("batch", [1, 2, 4, 8, 16])
    runner = SimpleNamespace(
        device="tpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(17, 1, dtype=torch.int32),
            _tpu_relay_bucketer=bucketer,
        ),
    )
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        warmup_host_relay(runner, bucketer, 16, extend_cap=16)
        relay = HostPaddingFutureMap(
            runner.device, runner.spec_algorithm, runner.req_to_token_pool
        )
        for live in (3, 5, 7, 6, 11, 15, 2, 9, 4, 12, 1, 13, 8, 16, 14, 10):
            indices = torch.arange(16, 16 - live, -1)
            expected = torch.arange(live, dtype=torch.int64) + live * 100
            tokens = expected.to("tpu")
            before = torch.tpu._get_cache_misses()
            relay.stash(indices, RelayPayload(bonus_tokens=tokens))
            actual = relay.gather(indices).cpu()
            assert torch.equal(actual[:live], expected)
            assert torch.tpu._get_cache_misses() == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
