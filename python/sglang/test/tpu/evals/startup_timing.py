"""Cold-start precompile timing gate for the TPU nightlies.

A server's startup cost is mostly precompile: every prefill/decode shape bucket
is traced and compiled before the first request. CI checks correctness and
throughput, not how long that takes, so a ~10% precompile regression (#814,
fixed by #1037) was caught only by user reports (#1041).

``startup_timing_runner`` is an ``AccuracyTestParams.runner``: after the
launcher's /health_generate succeeds -- i.e. after the last shape compiled --
it reads ``/server_info -> startup_time`` (sglang.srt.observability.startup_time)
and publishes it:

    load_weight, kv_cache_allocation, scheduler_e2e, tokenizer_e2e
    cuda_graph.<phase>      per-phase "capture" time
    compile.steps[]         one record per compiled shape (step_key, phase,
                            batch_size, num_tokens, duration_seconds, ...)

The headline is ``precompile_s`` = scheduler_e2e - load_weight -
kv_cache_allocation: the scheduler's bring-up wall time minus the two non-compile
phases it reports. On TPU that window is dominated by the shape-bucket
precompile, measured end to end. ``cuda_graph`` is NOT used: TPU installs both
the EXTEND and DECODE policies from the decode-capture hook, which returns
once every compile has been *submitted* -- the async join that waits for them to
finish (~12s of ~87s on gpt-oss-120b) happens after it and is not counted. Per-step
durations are also dumped, but their SUM is not wall time -- native compilation
overlaps across shapes -- so it is informational.

WHAT IT GATES. Like bench_serving_runner, not a time threshold: the dashboard's
trend is the regression detector. It gates that the measurement is COMPLETE --
startup_time present, a positive precompile time, at least one compile step --
which is what separates "slower" from "nothing was measured". An incomplete
measurement dumps a ``smoke_failed`` record (the dashboard's ✕) instead of
numbers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import requests

from sglang.test.test_utils import dump_metric
from sglang.test.tpu.evals.eval_support import emit_result_payload

HEADLINE_METRIC = "precompile_s"
_SCALAR_FIELDS = (
    "load_weight",
    "kv_cache_allocation",
    "scheduler_e2e",
    "tokenizer_e2e",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def extract_startup_metrics(startup_time: Mapping[str, Any] | None) -> dict[str, float]:
    """Flatten ``/server_info``'s ``startup_time`` into ``{metric: seconds}``."""
    startup_time = startup_time or {}
    metrics: dict[str, float] = {}
    for field in _SCALAR_FIELDS:
        if _is_number(startup_time.get(field)):
            metrics[field] = float(startup_time[field])

    graph = {
        phase: float(duration)
        for phase, duration in (startup_time.get("cuda_graph") or {}).items()
        if _is_number(duration)
    }
    for phase, duration in graph.items():
        metrics[f"cuda_graph.{phase}"] = duration
    if graph:
        metrics["cuda_graph.total_s"] = sum(graph.values())
    if "scheduler_e2e" in metrics:
        metrics[HEADLINE_METRIC] = (
            metrics["scheduler_e2e"]
            - metrics.get("load_weight", 0.0)
            - metrics.get("kv_cache_allocation", 0.0)
        )

    steps = (startup_time.get("compile") or {}).get("steps") or []
    durations = [s for s in steps if _is_number(s.get("duration_seconds"))]
    if durations:
        metrics["compile_steps.count"] = float(len(durations))
        metrics["compile_steps.sum_s"] = sum(
            float(s["duration_seconds"]) for s in durations
        )
        by_phase: dict[str, float] = defaultdict(float)
        for step in durations:
            key = f"{step.get('component', 'target')}.{step.get('phase', 'unknown')}"
            by_phase[key] += float(step["duration_seconds"])
        for key, total in sorted(by_phase.items()):
            metrics[f"compile_steps.{key}_s"] = total
    return metrics


def measurement_failures(metrics: Mapping[str, float]) -> list[str]:
    """Why a measurement is incomplete (empty list = publishable)."""
    failures = []
    if "scheduler_e2e" not in metrics:
        failures.append("startup_time has no scheduler_e2e")
    elif metrics.get(HEADLINE_METRIC, 0.0) <= 0.0:
        failures.append(
            f"{HEADLINE_METRIC} is {metrics.get(HEADLINE_METRIC)}: scheduler_e2e "
            "does not exceed load_weight + kv_cache_allocation"
        )
    if metrics.get("compile_steps.count", 0.0) <= 0.0:
        failures.append("startup_time recorded no compile steps")
    return failures


def startup_timing_runner(
    *,
    test_name: str,
    model: str,
    workload: str,
    series_id: str = "",
    display_name: str = "",
    extra_labels: Mapping[str, Any] | None = None,
    timeout: float = 60.0,
):
    """Build an ``AccuracyTestParams.runner`` that publishes startup timings.

    ``workload`` names the series in the dashboard's Workload dropdown (e.g.
    "cold-precompile"); ``series_id``/``display_name`` mirror MODEL_CONFIG's
    ``id``/``display_name``, as for bench_serving_runner. ``extra_labels`` ride
    on the headline record (e.g. the concurrency the server was sized for).
    """

    def routing_labels() -> dict[str, Any]:
        labels: dict[str, Any] = {
            "kind": "performance",
            "model": model or test_name,
            "workload": workload,
            **(extra_labels or {}),
        }
        if series_id:
            labels["id"] = series_id
        if display_name:
            labels["display_name"] = display_name
        return labels

    def run(base_url: str) -> dict[str, Any]:
        failures: list[str] = []
        startup_time: Mapping[str, Any] = {}
        try:
            resp = requests.get(f"{base_url}/server_info", timeout=timeout)
            resp.raise_for_status()
            startup_time = resp.json().get("startup_time") or {}
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"could not read /server_info: {exc}")

        metrics = extract_startup_metrics(startup_time)
        if not failures:
            failures = measurement_failures(metrics)

        if failures:
            dump_metric(
                "smoke_failed",
                1,
                labels={**routing_labels(), "reason": "; ".join(failures)},
            )
        else:
            stem = f"{test_name}.{workload}"
            for name, value in metrics.items():
                if name == HEADLINE_METRIC:
                    labels = {**routing_labels(), "headline": "1"}
                else:
                    # Everything else is archived (kind=info), shown in the
                    # point's details, but not charted -- same split as the sweep.
                    labels = {
                        "kind": "info",
                        "model": model or test_name,
                        "workload_name": workload,
                    }
                    if series_id:
                        labels["id"] = series_id
                labels["direction"] = (
                    "informational" if name.endswith(".count") else "lower_is_better"
                )
                dump_metric(f"{stem}.{name}", value, labels=labels)
            print(
                f"[startup] {test_name}: precompile {metrics[HEADLINE_METRIC]:.1f}s "
                f"(load_weight {metrics.get('load_weight', float('nan')):.1f}s, "
                f"scheduler_e2e {metrics.get('scheduler_e2e', float('nan')):.1f}s, "
                f"{int(metrics.get('compile_steps.count', 0))} compile steps)",
                flush=True,
            )

        emit_result_payload(
            {
                "test": test_name,
                "model": model or test_name,
                "legs": [
                    {
                        "workload": workload,
                        "status": "failed" if failures else "clean",
                        **({"reason": "; ".join(failures)} if failures else {}),
                        "metrics": metrics,
                    }
                ],
                "startup_time": startup_time,
            }
        )
        return {
            "score": metrics.get(HEADLINE_METRIC, 0.0),
            "_passed": not failures,
            "_failures": failures,
        }

    return run
