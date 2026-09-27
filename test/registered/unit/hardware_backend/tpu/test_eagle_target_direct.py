"""The target tracer override must not change draft or ordinary policies."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tracer", ["dynamo", "direct"])
@pytest.mark.parametrize(
    "algorithm,draft",
    [
        (SpeculativeAlgorithm.EAGLE3, False),
        (SpeculativeAlgorithm.EAGLE3, True),
        (SpeculativeAlgorithm.NONE, False),
    ],
)
def test_target_override_preserves_other_tracers(enabled, tracer, algorithm, draft):
    runner = SimpleNamespace(spec_algorithm=algorithm, is_draft_worker=draft)
    with patch.object(envs.SGLANG_TPU_EAGLE_TARGET_DIRECT, "get", return_value=enabled):
        policy = DecoderTpuCompilePolicy(model_runner=runner, tracer=tracer)
    expected = "direct" if enabled and algorithm.is_eagle3() and not draft else tracer
    assert policy.tracer == expected


def test_runnerless_policy_keeps_default():
    with patch.object(envs.SGLANG_TPU_EAGLE_TARGET_DIRECT, "get", return_value=True):
        assert DecoderTpuCompilePolicy().tracer == "dynamo"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
