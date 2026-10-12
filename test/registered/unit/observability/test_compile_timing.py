import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.observability.startup_time import (
    aggregate_scheduler_startup_times,
    build_scheduler_startup_time,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_compile_timing_is_added_to_scheduler_startup_time() -> None:
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
        initialization=2.0,
        target_load_weight=10.0,
        draft_load_weight=1.0,
        kv_cache_allocation=4.0,
        scheduler_e2e=20.0,
        target_cuda_graph=None,
        draft_cuda_graph=None,
        target_compile_steps=target.snapshot(),
        draft_compile_steps=draft.snapshot(),
    )

    assert startup_time["initialization"] == 2.0
    assert "total_seconds" not in startup_time["compile"]
    assert [step["component"] for step in startup_time["compile"]["steps"]] == [
        "target",
        "draft",
    ]


def test_initialization_stops_before_weight_loading() -> None:
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = SimpleNamespace(
        scheduler_startup_begin=10.0,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(weight_load_start_time=30.0),
            weight_load_time=10.0,
            graph_time_usage={},
            compile_timing_steps=[],
        ),
        draft_worker=None,
        kv_cache_allocation_time=4.0,
    )
    with patch(
        "sglang.srt.managers.scheduler.time.perf_counter", side_effect=[100.0, 200.0]
    ):
        Scheduler.init_startup_timing_summary(scheduler)
        assert scheduler.startup_time["initialization"] == 20.0
        assert scheduler.startup_time["scheduler_e2e"] == 90.0
        Scheduler.init_startup_timing_summary(scheduler)
        assert scheduler.startup_time["initialization"] == 20.0
        assert scheduler.startup_time["scheduler_e2e"] == 190.0


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
                "initialization": 2.0,
                "compile": {
                    "schema_version": 1,
                    "steps": [step, second_step],
                },
            },
            {
                "initialization": 5.0,
                "compile": {
                    "schema_version": 1,
                    "steps": [slower_step, slower_second_step],
                },
            },
        ]
    )

    assert result["initialization"] == 5.0
    assert "total_seconds" not in result["compile"]
    assert [step["duration_seconds"] for step in result["compile"]["steps"]] == [
        4.0,
        5.0,
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
