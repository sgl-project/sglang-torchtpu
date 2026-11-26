# SPDX-License-Identifier: Apache-2.0
"""TPU plugin hooks for Google Cloud ML Diagnostics (Diagon).

Why: SGLang-TorchTPU spawns one TPU-owning Scheduler process per rank and emits
     prefill/decode throughput metrics and stage-separated xprof traces inside
     core Scheduler / SchedulerMetricsReporter / SchedulerProfilerManager.
     Diagon needs opt-in MLRun initialization (Seam 1), host-side scalar metric
     forwarding (Seam 2), and dual-mode non-phased + prefill/decode xprof
     context propagation (Seams 3A/3B) without adding TPU-specific branches or
     SDK imports to core python/sglang/srt/.
How: AFTER Scheduler.__init__ initializes MLRun (including on-demand xprof on
     port 9999) when DIAGON_GCS_PATH is set; AFTER SchedulerMetricsReporter
     report_prefill_stats / report_decode_stats forwards host-side scalar
     metrics on the primary stats rank; BEFORE SchedulerProfilerManager
     _init_profile normalizes gs:// output dirs for local bookkeeping and
     BEFORE _start_profile records the active stage (prefill/decode vs
     non-phased) and profile_id for sglang_tpu.profiler.start_tpu_profile.
Upstream PR or issue: N/A; no upstream core seam required while Diagon remains
     TPU-plugin scoped.
Remove when: core SGLang adds first-class observability/profiler plugin seams
     for external ML diagnostics backends.
"""

from __future__ import annotations

from typing import Any

from sglang_tpu.diagon import get_diagon_adapter

_ADAPTER = get_diagon_adapter()


def after_scheduler_init(result: Any, scheduler: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    _ADAPTER.init_run(scheduler)
    return result


def after_report_prefill_stats(
    result: Any, reporter: Any, *args: Any, **kwargs: Any
) -> Any:
    if _ADAPTER.enabled:
        prefill_stats = kwargs["prefill_stats"] if "prefill_stats" in kwargs else args[1]
        _ADAPTER.record_prefill_stats(reporter, prefill_stats)
    return result


def after_report_decode_stats(
    result: Any, reporter: Any, *args: Any, **kwargs: Any
) -> Any:
    if not _ADAPTER.enabled:
        return result
    interval = reporter.decode_log_interval
    if reporter.forward_ct_decode == 0 or (
        interval > 1 and reporter.forward_ct_decode % interval != 0
    ):
        return result
    running_batch = (
        kwargs["running_batch"]
        if "running_batch" in kwargs
        else (args[1] if len(args) > 1 else None)
    )
    _ADAPTER.record_decode_stats(reporter, running_batch=running_batch)
    return result


def before_init_profile(self: Any, *args: Any, **kwargs: Any):
    """Avoid Path('gs://...').mkdir() when Diagon routes traces to DIAGON_GCS_PATH."""
    if not _ADAPTER.enabled:
        return None
    if "output_dir" in kwargs:
        out_dir = kwargs["output_dir"]
        if isinstance(out_dir, str) and out_dir.startswith("gs://"):
            new_kwargs = dict(kwargs)
            new_kwargs["output_dir"] = "/tmp"
            return (self, *args), new_kwargs
    elif args and isinstance(args[0], str) and args[0].startswith("gs://"):
        return (self, "/tmp", *args[1:]), kwargs
    return None


def before_start_profile(self: Any, stage: Any = None, *args: Any, **kwargs: Any):
    """Propagate stage (prefill/decode or None) and profile_id to start_tpu_profile."""
    del args, kwargs
    if _ADAPTER.enabled:
        _ADAPTER.set_pending_profile_context(
            stage=stage,
            profile_id=getattr(self, "profile_id", None),
        )
    return None


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    reporter = "sglang.srt.managers.scheduler_components.metrics_reporter.SchedulerMetricsReporter"
    profiler = "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager"
    return [
        (
            "sglang.srt.managers.scheduler.Scheduler.__init__",
            after_scheduler_init,
            HookType.AFTER,
        ),
        (
            f"{reporter}.report_prefill_stats",
            after_report_prefill_stats,
            HookType.AFTER,
        ),
        (
            f"{reporter}.report_decode_stats",
            after_report_decode_stats,
            HookType.AFTER,
        ),
        (
            f"{profiler}._init_profile",
            before_init_profile,
            HookType.BEFORE,
        ),
        (
            f"{profiler}._start_profile",
            before_start_profile,
            HookType.BEFORE,
        ),
    ]
