from __future__ import annotations

from collections.abc import Iterable, Mapping

from sglang.srt.model_executor.graph_memory_usage import (
    empty_graph_time_usage,
    merge_graph_time_usage,
)


def build_scheduler_startup_time(
    *,
    target_load_weight: float,
    draft_load_weight: float,
    kv_cache_allocation: float,
    scheduler_e2e: float,
    target_cuda_graph: Mapping[str, float] | None,
    draft_cuda_graph: Mapping[str, float] | None,
    target_compile_steps: Iterable[Mapping] | None = None,
    draft_compile_steps: Iterable[Mapping] | None = None,
    model_startup: float | None = None,
) -> dict:
    result = {
        "load_weight": target_load_weight + draft_load_weight,
        "kv_cache_allocation": kv_cache_allocation,
        "scheduler_e2e": scheduler_e2e,
        "cuda_graph": merge_graph_time_usage(
            target_cuda_graph,
            draft_cuda_graph,
        ),
    }
    if model_startup is not None:
        result["model_startup"] = model_startup
    compile_steps = [
        dict(step)
        for steps in (target_compile_steps, draft_compile_steps)
        for step in steps or ()
    ]
    if compile_steps:
        result["compile"] = {
            "schema_version": 1,
            "steps": compile_steps,
        }
    return result


def aggregate_scheduler_startup_times(
    startup_times: Iterable[Mapping | None],
) -> dict:
    """Return critical-path startup durations across scheduler ranks."""
    result = {
        "load_weight": 0.0,
        "kv_cache_allocation": 0.0,
        "scheduler_e2e": 0.0,
        "cuda_graph": empty_graph_time_usage(),
    }
    rank_critical_compile_steps: dict[tuple[str, int], dict] = {}
    for startup_time in startup_times:
        if not startup_time:
            continue
        result["load_weight"] = max(
            result["load_weight"], float(startup_time.get("load_weight", 0.0))
        )
        result["kv_cache_allocation"] = max(
            result["kv_cache_allocation"],
            float(startup_time.get("kv_cache_allocation", 0.0)),
        )
        result["scheduler_e2e"] = max(
            result["scheduler_e2e"],
            float(startup_time.get("scheduler_e2e", 0.0)),
        )
        if "model_startup" in startup_time:
            result["model_startup"] = max(
                result.get("model_startup", 0.0), float(startup_time["model_startup"])
            )
        for phase, duration in startup_time.get("cuda_graph", {}).items():
            result["cuda_graph"][phase] = max(
                result["cuda_graph"].get(phase, 0.0), float(duration)
            )
        compile_timing = startup_time.get("compile")
        if compile_timing is None:
            continue
        step_occurrences: dict[str, int] = {}
        for raw_step in compile_timing.get("steps", []):
            step = dict(raw_step)
            step_key = str(step["step_key"])
            occurrence = step_occurrences.get(step_key, 0)
            step_occurrences[step_key] = occurrence + 1
            aggregate_key = (step_key, occurrence)
            previous = rank_critical_compile_steps.get(aggregate_key)
            if previous is None:
                rank_critical_compile_steps[aggregate_key] = step
            else:
                rank_critical_compile_steps[aggregate_key] = max(
                    previous,
                    step,
                    key=lambda candidate: float(candidate["duration_seconds"]),
                )
    if rank_critical_compile_steps:
        compile_steps = list(rank_critical_compile_steps.values())
        result["compile"] = {
            "schema_version": 1,
            "steps": compile_steps,
        }
    return result


def build_engine_startup_time(
    scheduler_startup_times: Iterable[Mapping | None],
    *,
    tokenizer_e2e: float,
) -> dict:
    result = aggregate_scheduler_startup_times(scheduler_startup_times)
    result["tokenizer_e2e"] = tokenizer_e2e
    return result
