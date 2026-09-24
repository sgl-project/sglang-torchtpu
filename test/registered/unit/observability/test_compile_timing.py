from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.observability.startup_time import (
    aggregate_scheduler_startup_times,
    build_scheduler_startup_time,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_compile_timing_recorder_preserves_local_steps() -> None:
    recorder = CompileTimingRecorder()
    recorder.record(
        component="target",
        phase="decode",
        policy="DecoderTpuCompilePolicy",
        duration_seconds=1.5,
        batch_size=32,
        num_tokens=32,
    )
    recorder.record(
        component="target",
        phase="decode",
        policy="DecoderTpuCompilePolicy",
        duration_seconds=2.5,
        batch_size=32,
        num_tokens=32,
    )

    assert recorder.snapshot() == [
        {
            "step_key": (
                "component=target/phase=decode/"
                "policy=DecoderTpuCompilePolicy/"
                "batch_size=32/num_tokens=32"
            ),
            "component": "target",
            "phase": "decode",
            "policy": "DecoderTpuCompilePolicy",
            "duration_seconds": 1.5,
            "batch_size": 32,
            "num_tokens": 32,
        },
        {
            "step_key": (
                "component=target/phase=decode/"
                "policy=DecoderTpuCompilePolicy/"
                "batch_size=32/num_tokens=32"
            ),
            "component": "target",
            "phase": "decode",
            "policy": "DecoderTpuCompilePolicy",
            "duration_seconds": 2.5,
            "batch_size": 32,
            "num_tokens": 32,
        },
    ]


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
        target_load_weight=10.0,
        draft_load_weight=1.0,
        kv_cache_allocation=4.0,
        scheduler_e2e=20.0,
        target_cuda_graph=None,
        draft_cuda_graph=None,
        target_compile_steps=target.snapshot(),
        draft_compile_steps=draft.snapshot(),
    )

    assert startup_time["compile"]["total_seconds"] == 5.0
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
                "compile": {
                    "schema_version": 1,
                    "total_seconds": 3.0,
                    "steps": [step, second_step],
                }
            },
            {
                "compile": {
                    "schema_version": 1,
                    "total_seconds": 9.0,
                    "steps": [slower_step, slower_second_step],
                }
            },
        ]
    )

    assert result["compile"]["total_seconds"] == 9.0
    assert [step["duration_seconds"] for step in result["compile"]["steps"]] == [
        4.0,
        5.0,
    ]
