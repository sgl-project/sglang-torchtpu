# SPDX-License-Identifier: Apache-2.0
"""Serving-time XLA recompilation gate helpers for TPU tests."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _get_json(url: str, *, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: float = 180.0) -> Any:
    body = json.dumps(dict(payload)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_recompilation_states(payload: Any) -> list[dict[str, Any]]:
    """Normalize a ``/get_internal_state`` response into per-rank state dicts."""
    if isinstance(payload, Mapping) and "internal_state" in payload:
        raw = payload["internal_state"]
    else:
        raw = payload

    if isinstance(raw, Mapping):
        states = [dict(raw)]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        states = [dict(item) for item in raw if isinstance(item, Mapping)]
    else:
        raise ValueError(
            f"Unexpected /get_internal_state payload shape: {type(payload)!r}"
        )

    if not states:
        raise ValueError("Empty /get_internal_state rank list")

    normalized: list[dict[str, Any]] = []
    for state in states:
        events = state.get("tpu_recompile_events") or []
        compiles = int(state.get("tpu_serving_compiles", 0) or 0)
        heavy_compiles = int(state.get("tpu_heavy_serving_compiles", 0) or 0)
        raw_eager = state.get("tpu_eager_serving_compiles")
        eager_compiles = (
            int(raw_eager or 0)
            if raw_eager is not None
            else max(0, compiles - heavy_compiles)
        )
        normalized.append(
            {
                "tpu_serving_compiles": compiles,
                "tpu_heavy_serving_compiles": heavy_compiles,
                "tpu_eager_serving_compiles": eager_compiles,
                "tpu_serving_cache_loads": int(
                    state.get("tpu_serving_cache_loads", 0) or 0
                ),
                "tpu_serving_compile_ms": int(
                    state.get("tpu_serving_compile_ms", 0) or 0
                ),
                "tpu_worst_compile_ms": int(state.get("tpu_worst_compile_ms", 0) or 0),
                "tpu_recompile_events": (
                    list(events) if isinstance(events, list) else []
                ),
            }
        )
    return normalized


def fetch_recompilation_snapshot(
    url: str, *, timeout: float = 30.0
) -> list[dict[str, Any]]:
    """Query ``{url}/get_internal_state`` and return normalized per-rank counters."""
    endpoint = f"{url.rstrip('/')}/get_internal_state"
    payload = _get_json(endpoint, timeout=timeout)
    return extract_recompilation_states(payload)


def diff_recompilation_snapshots(
    before: Sequence[Mapping[str, Any]] | None,
    after: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-rank counter deltas between two ``/get_internal_state`` snapshots."""
    after_norm = extract_recompilation_states(list(after))
    if before is None:
        return after_norm
    before_norm = extract_recompilation_states(list(before))
    if len(before_norm) != len(after_norm):
        raise ValueError(
            f"Rank count changed between snapshots: {len(before_norm)} vs {len(after_norm)}"
        )

    deltas: list[dict[str, Any]] = []
    for prev, curr in zip(before_norm, after_norm):
        prev_events = prev["tpu_recompile_events"]
        curr_events = curr["tpu_recompile_events"]
        new_events = (
            curr_events[len(prev_events) :]
            if len(curr_events) >= len(prev_events)
            else list(curr_events)
        )
        worst_ms = max(
            (
                max(ev.get("durations_ms") or [0])
                for ev in new_events
                if isinstance(ev, Mapping)
            ),
            default=0,
        )
        if (
            not new_events
            and curr["tpu_serving_compiles"] > prev["tpu_serving_compiles"]
        ):
            worst_ms = curr["tpu_worst_compile_ms"]
        deltas.append(
            {
                "tpu_serving_compiles": max(
                    0, curr["tpu_serving_compiles"] - prev["tpu_serving_compiles"]
                ),
                "tpu_heavy_serving_compiles": max(
                    0,
                    curr["tpu_heavy_serving_compiles"]
                    - prev["tpu_heavy_serving_compiles"],
                ),
                "tpu_eager_serving_compiles": max(
                    0,
                    curr["tpu_eager_serving_compiles"]
                    - prev["tpu_eager_serving_compiles"],
                ),
                "tpu_serving_cache_loads": max(
                    0,
                    curr["tpu_serving_cache_loads"] - prev["tpu_serving_cache_loads"],
                ),
                "tpu_serving_compile_ms": max(
                    0, curr["tpu_serving_compile_ms"] - prev["tpu_serving_compile_ms"]
                ),
                "tpu_worst_compile_ms": worst_ms,
                "tpu_recompile_events": new_events,
            }
        )
    return deltas


def _format_event_summary(ev: Mapping[str, Any]) -> str:
    summary = (
        f"{ev.get('context', '?')} (compiles={ev.get('compile_count', 0)}, "
        f"loads={ev.get('load_count', 0)}, durations_ms={ev.get('durations_ms', [])}"
    )
    sites = ev.get("sites")
    if isinstance(sites, Sequence) and not isinstance(sites, (str, bytes)) and sites:
        site_strs = [
            f"{s.get('count', 1)}x {s.get('site', '?')}"
            if isinstance(s, Mapping)
            else str(s)
            for s in sites[:4]
        ]
        summary += f", sites=[{', '.join(site_strs)}]"
    return summary + ")"


def check_no_serving_recompilation(
    urls: str | Sequence[str],
    *,
    baseline: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    fail_on_cache_load: bool = True,
    only_heavy: bool = False,
    timeout: float = 30.0,
) -> list[str]:
    """Return failure messages for any serving-time XLA compiles or cache loads."""
    target_urls = [urls] if isinstance(urls, str) else list(urls)
    failures: list[str] = []

    for raw_url in target_urls:
        url = raw_url.rstrip("/")
        try:
            after = fetch_recompilation_snapshot(url, timeout=timeout)
            prev = baseline.get(raw_url) or baseline.get(url) if baseline else None
            ranks = diff_recompilation_snapshots(prev, after)
        except Exception as exc:
            failures.append(
                f"{url}: failed to read /get_internal_state recompilation counters: {exc}"
            )
            continue

        for rank_idx, state in enumerate(ranks):
            compiles = state["tpu_serving_compiles"]
            heavy_compiles = state["tpu_heavy_serving_compiles"]
            eager_compiles = state["tpu_eager_serving_compiles"]
            cache_loads = state["tpu_serving_cache_loads"]
            compile_ms = state["tpu_serving_compile_ms"]
            worst_ms = state["tpu_worst_compile_ms"]
            events = state["tpu_recompile_events"]

            gated_compiles = heavy_compiles if only_heavy else compiles
            triggered = gated_compiles > 0 or (fail_on_cache_load and cache_loads > 0)
            if not triggered:
                continue

            event_summaries = [
                _format_event_summary(ev)
                for ev in events[:8]
                if isinstance(ev, Mapping)
            ]
            events_str = (
                "; ".join(event_summaries) if event_summaries else "none recorded"
            )
            failures.append(
                f"{url} rank {rank_idx}: detected serving-time XLA recompilation "
                f"(tpu_serving_compiles={compiles}, "
                f"tpu_heavy_serving_compiles={heavy_compiles}, "
                f"tpu_eager_serving_compiles={eager_compiles}, "
                f"tpu_serving_cache_loads={cache_loads}, "
                f"tpu_serving_compile_ms={compile_ms}, "
                f"tpu_worst_compile_ms={worst_ms}, "
                f"events=[{events_str}])"
            )
    return failures


def assert_no_serving_recompilation(
    urls: str | Sequence[str],
    *,
    baseline: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    fail_on_cache_load: bool = True,
    only_heavy: bool = False,
    timeout: float = 30.0,
) -> None:
    """Assert that no serving-time XLA recompilation or cache load occurred."""
    failures = check_no_serving_recompilation(
        urls,
        baseline=baseline,
        fail_on_cache_load=fail_on_cache_load,
        only_heavy=only_heavy,
        timeout=timeout,
    )
    if failures:
        raise AssertionError("; ".join(failures))


def _make_prompt(word_count: int, idx: int) -> str:
    tokens = [f"token{(idx + i) % 97}" for i in range(max(1, word_count))]
    return " ".join(tokens)


def run_multi_shape_recompilation_workload(
    base_url: str,
    *,
    prompt_word_counts: Sequence[int] = (8, 96, 320),
    batch_sizes: Sequence[int] = (1, 2, 4),
    max_new_tokens: int = 16,
    timeout: float = 180.0,
) -> int:
    """Drive prefill and decode across multiple prompt lengths and batch sizes."""
    endpoint = f"{base_url.rstrip('/')}/generate"
    total_requests = 0
    for word_count in prompt_word_counts:
        for batch_size in batch_sizes:
            prompts = [
                _make_prompt(word_count, idx=batch_size * 100 + i)
                for i in range(batch_size)
            ]
            payload = {
                "text": prompts[0] if batch_size == 1 else prompts,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            }
            _post_json(endpoint, payload, timeout=timeout)
            total_requests += batch_size
    return total_requests


def recompilation_gate_runner(
    *,
    engine_urls: Sequence[str] | None = None,
    run_workload: bool = False,
    prompt_word_counts: Sequence[int] = (8, 96, 320),
    batch_sizes: Sequence[int] = (1, 2, 4),
    max_new_tokens: int = 16,
    fail_on_cache_load: bool = True,
    only_heavy: bool = False,
    use_baseline_diff: bool = False,
    timeout: float = 30.0,
) -> Callable[[str], dict[str, Any]]:
    """Build an ``AccuracyTestParams.runner`` that gates on zero XLA recompilation."""

    def run(base_url: str) -> dict[str, Any]:
        urls = list(engine_urls) if engine_urls is not None else [base_url]
        baseline: dict[str, list[dict[str, Any]]] | None = None
        if use_baseline_diff:
            baseline = {
                u.rstrip("/"): fetch_recompilation_snapshot(u, timeout=timeout)
                for u in urls
            }
        if run_workload:
            run_multi_shape_recompilation_workload(
                base_url,
                prompt_word_counts=prompt_word_counts,
                batch_sizes=batch_sizes,
                max_new_tokens=max_new_tokens,
            )
        failures = check_no_serving_recompilation(
            urls,
            baseline=baseline,
            fail_on_cache_load=fail_on_cache_load,
            only_heavy=only_heavy,
            timeout=timeout,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run
