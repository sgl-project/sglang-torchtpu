from __future__ import annotations

from typing import Any


class CompileTimingRecorder:
    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

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
        self._steps.append(
            {
                "step_key": step_key,
                "duration_seconds": float(duration_seconds),
                **{key: value for key, value in identity.items() if value is not None},
            }
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(step) for step in self._steps]
