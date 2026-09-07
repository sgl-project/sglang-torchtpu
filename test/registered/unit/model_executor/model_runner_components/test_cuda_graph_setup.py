import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_decode_graph,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def test_model_runner_can_override_decode_graph_runner(monkeypatch):
    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(
            model_impl="auto",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="default")
            ),
        )

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    capture = capture_decode_graph(model_runner=model_runner)

    assert isinstance(capture.runner, CustomGraphRunner)
    assert capture.runner.model_runner is model_runner


@pytest.mark.parametrize("device,expect_runner", [("cuda", False), ("tpu", True)])
def test_pd_prefill_target_skip_is_cuda_only(monkeypatch, device, expect_runner):
    """A speculative PD prefill target skips capture on CUDA but not on TPU.

    TPU has no capture/replay; this hook is where it installs torch.compile and
    precompiles the EXTEND buckets. Skipping it for a PD prefill target left the
    target model running fully eager on the only role that ever runs prefill.
    """

    class StubGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    model_runner = SimpleNamespace(
        is_generation=True,
        device=device,
        gpu_id=0,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: True),
        decode_num_tokens_per_req=lambda: 2,
        server_args=SimpleNamespace(
            model_impl="auto",
            disaggregation_mode="prefill",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="disabled")
            ),
        ),
    )
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(cuda_graph_setup.current_platform, "is_tpu", lambda: True)
    monkeypatch.setattr(
        cuda_graph_setup.current_platform,
        "get_graph_runner_cls",
        lambda: StubGraphRunner,
    )

    with get_flags().capture.override(enable_torch_compile=True):
        capture = capture_decode_graph(model_runner=model_runner)

    assert isinstance(capture.runner, StubGraphRunner) is expect_runner


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
