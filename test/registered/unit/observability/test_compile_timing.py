import sys

import pytest

from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.observability.startup_time import (
    aggregate_scheduler_startup_times,
    build_scheduler_startup_time,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("model_startup", [None, 15.0])
def test_compile_timing_is_added_to_scheduler_startup_time(model_startup) -> None:
    target = CompileTimingRecorder()
    target.record(
        component="target",
        phase="extend",
        policy="DecoderTpuCompilePolicy",
        duration_seconds=3.0,
        num_tokens=128,
    )
    draft = CompileTimingRecorder()
    draft.record(
        component="draft",
        phase="draft_decode",
        policy="SpecDecoderTpuCompilePolicy",
        duration_seconds=2.0,
        num_draft_tokens=3,
    )

    startup_time = build_scheduler_startup_time(
        target_load_weight=10.0,
        draft_load_weight=1.0,
        kv_cache_allocation=4.0,
        scheduler_e2e=20.0,
        target_cuda_graph=None,
        draft_cuda_graph=None,
        target_compile_steps=target.snapshot(),
        draft_compile_steps=draft.snapshot(),
        model_startup=model_startup,
    )

    assert startup_time["scheduler_e2e"] == 20.0
    if model_startup is None:
        assert "model_startup" not in startup_time
    else:
        assert startup_time["model_startup"] == model_startup
    assert "total_seconds" not in startup_time["compile"]
    assert [step["component"] for step in startup_time["compile"]["steps"]] == [
        "target",
        "draft",
    ]


def test_compile_timing_aggregation_uses_rank_critical_path() -> None:
    step = {
        "step_key": "component=vision/phase=vision_encode/policy=VisionPolicy",
        "component": "vision",
        "phase": "vision_encode",
        "policy": "VisionPolicy",
        "duration_seconds": 1.0,
    }
    slower_step = {**step, "duration_seconds": 4.0}
    second_step = {**step, "duration_seconds": 2.0}
    slower_second_step = {**step, "duration_seconds": 5.0}

    result = aggregate_scheduler_startup_times(
        [
            {
                "scheduler_e2e": 100.0,
                "model_startup": 15.0,
                "compile": {
                    "schema_version": 1,
                    "steps": [step, second_step],
                },
            },
            {
                "scheduler_e2e": 50.0,
                "model_startup": 20.0,
                "compile": {
                    "schema_version": 1,
                    "steps": [slower_step, slower_second_step],
                },
            },
        ]
    )

    assert result["scheduler_e2e"] == 100.0
    assert result["model_startup"] == 20.0
    assert "total_seconds" not in result["compile"]
    assert [step["duration_seconds"] for step in result["compile"]["steps"]] == [
        4.0,
        5.0,
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
