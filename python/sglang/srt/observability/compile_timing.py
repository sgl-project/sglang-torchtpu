from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class CompileTimingRecorder:
    def __init__(self) -> None:
        self._steps: dict[str, dict[str, Any]] = {}

    def record(
        self,
        *,
        component: str,
        phase: str,
        policy: str,
        duration_seconds: float,
        batch_size: int | None = None,
        num_tokens: int | None = None,
        num_draft_tokens: int | None = None,
        draft_step: int | None = None,
        model_index: int | None = None,
        input_variant: str | None = None,
        logits_output_mode: str | None = None,
        capture_hidden_mode: str | None = None,
        has_spec_info: bool | None = None,
    ) -> None:
        if duration_seconds < 0:
            raise ValueError("compile timing duration must be non-negative")
        identity = {
            "component": component,
            "phase": phase,
            "policy": policy,
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "num_draft_tokens": num_draft_tokens,
            "draft_step": draft_step,
            "model_index": model_index,
            "input_variant": input_variant,
            "logits_output_mode": logits_output_mode,
            "capture_hidden_mode": capture_hidden_mode,
            "has_spec_info": has_spec_info,
        }
        step_key = "/".join(
            f"{key}={value}" for key, value in identity.items() if value is not None
        )
        step = {
            "step_key": step_key,
            "duration_seconds": float(duration_seconds),
            **{key: value for key, value in identity.items() if value is not None},
        }
        previous = self._steps.get(step_key)
        if previous is None or step["duration_seconds"] > previous["duration_seconds"]:
            self._steps[step_key] = step

    def snapshot(self) -> dict[str, Any] | None:
        if not self._steps:
            return None
        return _build_compile_timing_report(self._steps.values())


def merge_compile_timing_reports(
    *reports: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    steps: dict[str, dict[str, Any]] = {}
    for report in reports:
        if report is None:
            continue
        for raw_step in report.get("steps", []):
            step = dict(raw_step)
            step_key = str(step["step_key"])
            previous = steps.get(step_key)
            if previous is None or float(step["duration_seconds"]) > float(
                previous["duration_seconds"]
            ):
                steps[step_key] = step
    if not steps:
        return None
    return _build_compile_timing_report(steps.values())


def _build_compile_timing_report(
    steps: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    indexed_steps = [{**step, "step_index": index} for index, step in enumerate(steps)]
    return {
        "schema_version": 1,
        "total_seconds": sum(float(step["duration_seconds"]) for step in indexed_steps),
        "steps": indexed_steps,
    }
