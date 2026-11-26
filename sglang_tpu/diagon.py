# SPDX-License-Identifier: Apache-2.0
"""Google Cloud ML Diagnostics (Diagon) runtime adapter for SGLang-TorchTPU.

Isolates all optional ``google-cloud-mldiagnostics`` SDK state, lifecycle
initialization (Seam 1), host-side scheduler scalar metrics emission (Seam 2),
and dual-mode xprof profiling (Seam 3A non-phased + Seam 3B stage-separated
prefill/decode) inside the TPU plugin.

Activation is strictly opt-in via ``DIAGON_GCS_PATH``. When unset, every entry
point short-circuits without importing the SDK so clean runs incur zero
overhead.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_STEP_PREFIX_RE = re.compile(r"^step_\d+", re.IGNORECASE)
_AUTO_PROFILE_ID_RE = re.compile(r"^auto-\d+(?:\.\d+)?$")


def _normalize_region(region: Optional[str]) -> Optional[str]:
    """Strip a trailing GCP zone suffix (e.g. 'us-central1-a' -> 'us-central1')."""
    if not region:
        return None
    region = region.strip()
    parts = region.split("-")
    if len(parts) >= 3 and parts[-1].isalpha():
        return "-".join(parts[:-1])
    return region


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _extract_scheduler_configs(scheduler: Any) -> dict[str, Any]:
    """Collect lightweight, serializable serving configs for MLRun metadata."""
    server_args = getattr(scheduler, "server_args", None)
    if server_args is None:
        return {"backend": "torch_tpu"}
    configs: dict[str, Any] = {
        "backend": "torch_tpu",
        "model_path": str(getattr(server_args, "model_path", "unknown")),
        "device": str(getattr(server_args, "device", "tpu")),
        "dtype": str(getattr(server_args, "dtype", "auto")),
        "tp_size": int(getattr(server_args, "tp_size", 1)),
        "pp_size": int(getattr(server_args, "pp_size", 1)),
        "dp_size": int(getattr(server_args, "dp_size", 1)),
    }
    page_size = getattr(server_args, "page_size", None)
    if page_size is not None:
        configs["page_size"] = int(page_size)
    context_length = getattr(server_args, "context_length", None)
    if context_length is not None:
        configs["context_length"] = int(context_length)
    return configs


def _sanitize_session_token(raw: Optional[str]) -> str:
    if not raw:
        return ""
    cleaned = _NON_ALNUM_RE.sub("_", str(raw).strip()).strip("_")
    return cleaned


def _build_serving_session_id(
    stage: Any = None,
    profile_id: Optional[str] = None,
) -> str:
    """Build an HG10-compliant serving session_id (never 'step_<n>').

    - Seam 3B (phased / stage-separated): ``"prefill"`` or ``"decode"`` (with an
      optional user-supplied ``profile_id`` suffix when not an auto timestamp).
    - Seam 3A (non-phased window): sanitized ``profile_id`` or
      ``"inference_trace"``.
    """
    stage_label: Optional[str] = None
    if stage is not None:
        is_prefill = getattr(stage, "is_prefill", None)
        is_decode = getattr(stage, "is_decode", None)
        if callable(is_prefill) and is_prefill():
            stage_label = "prefill"
        elif callable(is_decode) and is_decode():
            stage_label = "decode"
        else:
            stage_str = str(getattr(stage, "name", stage)).lower()
            if "prefill" in stage_str or "extend" in stage_str:
                stage_label = "prefill"
            elif "decode" in stage_str:
                stage_label = "decode"

    raw_id = (profile_id or "").strip()
    is_auto_id = bool(_AUTO_PROFILE_ID_RE.match(raw_id))
    safe_id = "" if is_auto_id else _sanitize_session_token(raw_id)

    if stage_label is not None:
        # Seam 3B: stage-separated prefill / decode profiling.
        if safe_id and not _STEP_PREFIX_RE.match(safe_id):
            return f"{stage_label}_{safe_id}"
        return stage_label

    # Seam 3A: non-phased /start_profile window.
    if safe_id and not _STEP_PREFIX_RE.match(safe_id):
        return safe_id
    return "inference_trace"


class DiagonRuntimeAdapter:
    """Single-responsibility Diagon SDK adapter for a TPU scheduler process."""

    def __init__(self) -> None:
        self._initialized: bool = False
        self._run: Any = None
        self._canonical_run_id: Optional[str] = None
        self._xprof: Any = None
        self._metrics: Any = None
        self._metric_types: Any = None
        self._prev_decode_tic: Optional[float] = None
        self._last_prefill_emit_ts: Optional[float] = None
        self._prefill_emit_interval_sec: float = 0.5
        self._pending_stage: Any = None
        self._pending_profile_id: Optional[str] = None
        self._active_session_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._initialized and self._run is not None

    @property
    def canonical_run_id(self) -> Optional[str]:
        return self._canonical_run_id

    @property
    def profiling_active(self) -> bool:
        return self._active_session_id is not None

    def reset_for_testing(self) -> None:
        """Reset process-local adapter state (used by hermetic unit/Stage 5A tests)."""
        self._initialized = False
        self._run = None
        self._canonical_run_id = None
        self._xprof = None
        self._metrics = None
        self._metric_types = None
        self._prev_decode_tic = None
        self._last_prefill_emit_ts = None
        self._prefill_emit_interval_sec = 0.5
        self._pending_stage = None
        self._pending_profile_id = None
        self._active_session_id = None

    # ------------------------------------------------------------------
    # Seam 1: MLRun lifecycle anchor (inside device-owning TPU scheduler)
    # ------------------------------------------------------------------
    def init_run(self, scheduler: Any) -> None:
        """Initialize MLRun and on-demand xprof (port 9999) when DIAGON_GCS_PATH is set."""
        gcs_path = os.environ.get("DIAGON_GCS_PATH", "").strip()
        if not gcs_path or self._initialized:
            return

        # Lazy, guarded SDK import (HG12): clean runs never import the SDK.
        try:
            from google_cloud_mldiagnostics import (
                machinelearning_run,
                metric_types,
                metrics,
                xprof,
            )
            from google_cloud_mldiagnostics.custom_types import mlrun_types
        except ImportError as exc:
            logger.warning(
                "DIAGON_GCS_PATH is set but google-cloud-mldiagnostics is not installed: %s",
                exc,
            )
            return

        region = _normalize_region(os.environ.get("DIAGON_REGION"))
        on_demand_xprof = _parse_bool_env("DIAGON_ON_DEMAND_XPROF", default=True)
        model_path = getattr(getattr(scheduler, "server_args", None), "model_path", "sglang")
        default_name = f"sglang-tpu-{Path(str(model_path)).name}"
        run_name = os.environ.get("DIAGON_RUN_NAME", "").strip() or default_name
        run_group = os.environ.get("DIAGON_RUN_GROUP", "").strip() or None
        project = os.environ.get("DIAGON_PROJECT", "").strip() or None

        configs = _extract_scheduler_configs(scheduler)
        run = machinelearning_run(
            name=run_name,
            run_group=run_group,
            configs=configs,
            gcs_path=gcs_path,
            project=project,
            region=region,
            framework=mlrun_types.Framework.PYTORCH,
            serving_engine=mlrun_types.ServingEngine.NONE,
            on_demand_xprof=on_demand_xprof,
            log_system_metrics=False,
        )
        self._run = run
        self._canonical_run_id = getattr(run, "name", None)
        self._xprof = xprof(run=run)
        self._metrics = metrics
        self._metric_types = metric_types
        raw_interval = os.environ.get("DIAGON_PREFILL_METRIC_INTERVAL_SEC", "").strip()
        if raw_interval:
            try:
                self._prefill_emit_interval_sec = max(0.0, float(raw_interval))
            except ValueError:
                pass
        self._initialized = True

        rank = os.environ.get("RANK", "0")
        if rank == "0":
            logger.info(
                "Diagon MLRun initialized: canonical_run_id=%s gcs_path=%s on_demand_xprof=%s",
                self._canonical_run_id,
                gcs_path,
                on_demand_xprof,
            )

    # ------------------------------------------------------------------
    # Seam 2: Host-side scalar metrics sink (outside torch.compile)
    # ------------------------------------------------------------------
    def _emit_metrics(self, entries: list[dict[str, Any]], step: int) -> None:
        if not self.enabled or self._metrics is None:
            return
        finite_entries: list[dict[str, Any]] = []
        for entry in entries:
            val = entry.get("value")
            if isinstance(val, (int, float)) and math.isfinite(val):
                finite_entries.append(entry)
        if finite_entries:
            self._metrics.record_metrics(finite_entries, step=step)

    def record_prefill_stats(self, reporter: Any, prefill_stats: Any) -> None:
        """Record prefill batch metrics from SchedulerMetricsReporter.report_prefill_stats."""
        if not self.enabled or not getattr(reporter, "is_stats_logging_rank", False):
            return
        now_mono = time.monotonic()
        if (
            self._last_prefill_emit_ts is not None
            and (now_mono - self._last_prefill_emit_ts) < self._prefill_emit_interval_sec
        ):
            return
        self._last_prefill_emit_ts = now_mono

        eff_in = int(getattr(prefill_stats, "log_input_tokens", 0)) - int(
            getattr(prefill_stats, "reprocessed_log_input_tokens", 0)
        )
        eff_hit = int(getattr(prefill_stats, "log_hit_tokens", 0)) - int(
            getattr(prefill_stats, "reprocessed_log_hit_tokens", 0)
        )
        total = eff_in + eff_hit
        cache_hit_rate = eff_hit / total if total > 0 else 0.0

        scheduler = reporter.scheduler
        pool_stats = scheduler.pool_stats_observer.get_pool_stats()
        step = int(getattr(scheduler, "forward_ct", 0))
        input_throughput = float(getattr(reporter, "last_input_throughput", 0.0))
        num_running = float(getattr(prefill_stats.num_running_reqs, "total", 0))
        num_queue = float(len(getattr(scheduler, "waiting_queue", ())))
        token_usage = float(pool_stats.get_max_pool_usage())

        self._emit_metrics(
            [
                {
                    "metric_name": self._metric_types.MetricType.THROUGHPUT,
                    "value": input_throughput,
                    "labels": {"stage": "prefill"},
                },
                {"metric_name": "prefill_throughput", "value": input_throughput},
                {"metric_name": "num_running_reqs", "value": num_running},
                {"metric_name": "num_queue_reqs", "value": num_queue},
                {"metric_name": "token_usage", "value": token_usage},
                {"metric_name": "cache_hit_rate", "value": float(cache_hit_rate)},
            ],
            step=step,
        )

    def record_decode_stats(self, reporter: Any, running_batch: Any = None) -> None:
        """Record periodic decode metrics from SchedulerMetricsReporter.report_decode_stats."""
        if not self.enabled or not getattr(reporter, "is_stats_logging_rank", False):
            return

        decode_interval = max(1, int(getattr(reporter, "decode_log_interval", 1)))
        forward_ct_decode = int(getattr(reporter, "forward_ct_decode", 0))
        if forward_ct_decode == 0 or forward_ct_decode % decode_interval != 0:
            return

        now_tic = float(getattr(reporter, "last_decode_stats_tic", 0.0))
        prev_tic = self._prev_decode_tic
        self._prev_decode_tic = now_tic

        scheduler = reporter.scheduler
        batch = running_batch or getattr(scheduler, "running_batch", None)
        num_running = float(len(batch.reqs)) if batch is not None and hasattr(batch, "reqs") else 0.0
        num_queue = float(len(getattr(scheduler, "waiting_queue", ())))
        pool_stats = scheduler.pool_stats_observer.get_pool_stats()
        token_usage = float(pool_stats.get_max_pool_usage())
        gen_throughput = float(getattr(reporter, "last_gen_throughput", 0.0))
        step = int(getattr(scheduler, "forward_ct", 0))

        entries: list[dict[str, Any]] = [
            {
                "metric_name": self._metric_types.MetricType.THROUGHPUT,
                "value": gen_throughput,
                "labels": {"stage": "decode"},
            },
            {"metric_name": "gen_throughput", "value": gen_throughput},
            {"metric_name": "num_running_reqs", "value": num_running},
            {"metric_name": "num_queue_reqs", "value": num_queue},
            {"metric_name": "token_usage", "value": token_usage},
        ]
        if prev_tic is not None and now_tic > prev_tic:
            step_latency_s = (now_tic - prev_tic) / decode_interval
            entries.append(
                {
                    "metric_name": self._metric_types.MetricType.LATENCY,
                    "value": float(step_latency_s),
                    "labels": {"stage": "decode"},
                }
            )
        self._emit_metrics(entries, step=step)

    # ------------------------------------------------------------------
    # Seam 3A (non-phased) & Seam 3B (phased prefill/decode) profiling
    # ------------------------------------------------------------------
    def set_pending_profile_context(
        self,
        *,
        stage: Any = None,
        profile_id: Optional[str] = None,
    ) -> None:
        """Capture stage and profile_id immediately before start_tpu_profile runs."""
        self._pending_stage = stage
        self._pending_profile_id = profile_id

    def start_profile(
        self,
        *,
        stage: Any = None,
        profile_id: Optional[str] = None,
    ) -> str:
        """Start a scoped Diagon xprof session with an HG10-compliant session_id."""
        effective_stage = stage if stage is not None else self._pending_stage
        effective_id = profile_id if profile_id is not None else self._pending_profile_id
        self._pending_stage = None
        self._pending_profile_id = None

        session_id = _build_serving_session_id(
            stage=effective_stage,
            profile_id=effective_id,
        )
        if self._xprof is not None:
            self._xprof.start(session_id=session_id)
            self._active_session_id = session_id
            logger.info("Diagon xprof started: session_id=%s", session_id)
        return session_id

    def stop_profile(self) -> None:
        """Stop the active Diagon xprof session."""
        if self._xprof is not None and self._active_session_id is not None:
            session_id = self._active_session_id
            self._active_session_id = None
            self._xprof.stop()
            logger.info("Diagon xprof stopped: session_id=%s", session_id)


_ADAPTER: Optional[DiagonRuntimeAdapter] = None


def get_diagon_adapter() -> DiagonRuntimeAdapter:
    """Return the process-local DiagonRuntimeAdapter singleton."""
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = DiagonRuntimeAdapter()
    return _ADAPTER
