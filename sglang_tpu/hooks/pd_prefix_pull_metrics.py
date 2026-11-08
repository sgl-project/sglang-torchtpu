# SPDX-License-Identifier: Apache-2.0
"""Measure how much prefix a P/D prefill could pull from its decode.

Why: the decode reports the prefix it holds (decode_prefix_len) at bootstrap, and
     the prefill recomputes whatever of it its own tree lost. Before building the
     pull (prefix_pull.py) we need the size of that gap, and how often the decode
     still holds the SWA window at its match end, which a pull needs.
How: prefill: AFTER PrefillBootstrapQueue.finalize_bootstrap reads the decode's
     DecodePrefixReport (Raiden's pipeline request carries it; the prefill keeps
     sending the whole prompt); AFTER PrefillAdder._account_prefill_cache_admission records
     max(0, decode_prefix_len - admitted prefix) and the same past the decode's
     pullable length per admitted request; AFTER
     SchedulerMetricsReporter.report_prefill_stats flushes it into counters and a
     periodic log line. Decode: pd_prefix_pull's match hook notes the decode's
     pullable length (prefix_pull_decode.decode_pullable); AFTER
     DecodePreallocQueue._send_kv_metadata counts it once the prealloc commits.
     AFTER SchedulerMetricsCollector.__init__ creates the counters. No behavior
     change.
Upstream PR or issue: none.
Remove when: the prefix pull ships its own metrics.
"""

from __future__ import annotations

import collections
import logging
import time
import weakref
from dataclasses import dataclass, field

from sglang_tpu.disaggregation.prefix_pull import (
    DecodeWindowCase,
    PrefixGapCase,
    classify_decode_window,
    classify_prefix_gap,
    decode_match_cap,
)

logger = logging.getLogger(__name__)

AT_CAP = "to_window"
INTERIOR = "interior"


@dataclass
class PrefillGapStats:
    """Prefill-side tallies; pending_* are the deltas not yet in Prometheus."""

    reqs: collections.Counter = field(default_factory=collections.Counter)
    gap_tokens: collections.Counter = field(default_factory=collections.Counter)
    decode_prefix_tokens: int = 0
    # Tokens a pull could fetch: max(0, decode_pullable_len - prefill prefix).
    pullable_ahead_tokens: int = 0
    pending_pullable_ahead_tokens: int = 0
    pending_reqs: collections.Counter = field(default_factory=collections.Counter)
    pending_gap_tokens: collections.Counter = field(default_factory=collections.Counter)
    pending_decode_prefix_tokens: int = 0
    last_log_time: float = field(default_factory=time.monotonic)

    def record(
        self,
        *,
        decode_prefix_len: int,
        decode_pullable_len: int,
        prefill_prefix_len: int,
        prompt_len: int,
        window: int,
        page_size: int,
        min_pull_tokens: int,
    ) -> None:
        case, gap = classify_prefix_gap(
            decode_prefix_len=decode_prefix_len,
            prefill_prefix_len=prefill_prefix_len,
            min_pull_tokens=min_pull_tokens,
        )
        cap = decode_match_cap(prompt_len, window, page_size) if window else prompt_len
        # A decode match that reaches its cap holds the whole prompt before the
        # closing window; an interior one stopped at its own cache boundary.
        where = (
            AT_CAP if decode_prefix_len > 0 and decode_prefix_len >= cap else INTERIOR
        )
        for reqs, gaps in (
            (self.reqs, self.gap_tokens),
            (self.pending_reqs, self.pending_gap_tokens),
        ):
            reqs[(case, where)] += 1
            gaps[case] += gap
        self.decode_prefix_tokens += decode_prefix_len
        self.pending_decode_prefix_tokens += decode_prefix_len
        pullable_ahead = max(0, decode_pullable_len - prefill_prefix_len)
        self.pullable_ahead_tokens += pullable_ahead
        self.pending_pullable_ahead_tokens += pullable_ahead

    def log_line(self) -> str:
        total = sum(self.reqs.values())
        by_case = collections.Counter()
        at_cap = 0
        for (case, where), n in self.reqs.items():
            by_case[case] += n
            at_cap += n if where == AT_CAP else 0
        gap = sum(self.gap_tokens.values())
        pull_gap = self.gap_tokens[PrefixGapCase.DECODE_AHEAD_PULL]
        return (
            f"PD prefix gap (cumulative): reqs={total}, "
            + ", ".join(f"{c.value}={by_case[c]}" for c in PrefixGapCase)
            + f", decode_at_window_cap={at_cap}, "
            f"decode_prefix_tokens={self.decode_prefix_tokens}, "
            f"decode_ahead_tokens={gap}, pull_candidate_tokens={pull_gap}, "
            f"pullable_ahead_tokens={self.pullable_ahead_tokens}"
        )


@dataclass
class DecodeWindowStats:
    reqs: collections.Counter = field(default_factory=collections.Counter)
    prefix_tokens: int = 0
    pullable_tokens: int = 0
    last_log_time: float = field(default_factory=time.monotonic)

    def record(self, *, decode_prefix_len: int, pullable_len: int) -> DecodeWindowCase:
        case = classify_decode_window(
            decode_prefix_len=decode_prefix_len, pullable_len=pullable_len
        )
        self.reqs[case] += 1
        self.prefix_tokens += decode_prefix_len
        self.pullable_tokens += pullable_len
        return case

    def log_line(self) -> str:
        total = sum(self.reqs.values())
        share = self.pullable_tokens / self.prefix_tokens if self.prefix_tokens else 0.0
        return (
            f"PD decode prefix window (cumulative): prealloc={total}, "
            + ", ".join(f"{c.value}={self.reqs[c]}" for c in DecodeWindowCase)
            + f", decode_prefix_tokens={self.prefix_tokens}, "
            f"pullable_tokens={self.pullable_tokens} ({share:.1%})"
        )


_prefill_stats = PrefillGapStats()
_decode_stats = DecodeWindowStats()
# Match-time (decode_prefix_len, pullable_len), counted when the prealloc commits.
_pending_decode: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Prefill: the decode's DecodePrefixReport, read when the bootstrap finalizes.
_decode_reports: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _log_due(stats, now: float) -> bool:
    from sglang_tpu.envs import envs

    interval = envs.SGLANG_TPU_PD_PREFIX_GAP_LOG_INTERVAL_S.get()
    if interval <= 0 or now - stats.last_log_time < interval:
        return False
    stats.last_log_time = now
    return True


def _is_counting_prefill_rank() -> bool:
    from sglang.srt.runtime_context import get_disagg, get_parallel

    # PP stages admit the same requests, so count on the first stage only.
    return get_disagg().disaggregation_mode == "prefill" and get_parallel().pp_rank == 0


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def record_decode_report(result, queue, req, *args, **kwargs):
    from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender

    sender = req.disagg_kv_sender
    if result and isinstance(sender, RaidenKVSender):
        report = sender.decode_prefix_report()
        if report is not None:
            _decode_reports[req] = report
    return result


def record_prefill_admission(result, adder, req, prefix_len, *args, **kwargs):
    from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
    from sglang_tpu.envs import envs

    # The decode's prefix is known only after bootstrap; retractions re-admit.
    if (
        not _is_counting_prefill_rank()
        or req.retracted_stain
        or req.pending_bootstrap
        or req.bootstrap_host == FAKE_BOOTSTRAP_HOST
    ):
        return result
    tree_cache = adder.tree_cache
    # Raiden's pipeline path reports the prefix only in its pipeline request;
    # other backends set disagg_decode_prefix_len at bootstrap.
    report = _decode_reports.pop(req, None)
    _prefill_stats.record(
        decode_prefix_len=(
            report.prefix_len if report is not None else req.disagg_decode_prefix_len
        ),
        decode_pullable_len=report.pullable_len if report is not None else 0,
        prefill_prefix_len=prefix_len,
        prompt_len=len(req.origin_input_ids),
        window=tree_cache.sliding_window_size if tree_cache.supports_swa() else 0,
        page_size=adder.page_size,
        min_pull_tokens=envs.SGLANG_TPU_PD_PREFIX_PULL_MIN_TOKENS.get(),
    )
    return result


def flush_prefill_stats(result, reporter, *args, **kwargs):
    stats = _prefill_stats
    if reporter.current_scheduler_metrics_enabled and stats.pending_reqs:
        collector = reporter.metrics_collector
        labels = collector.labels
        for (case, where), n in stats.pending_reqs.items():
            collector.pd_prefill_prefix_gap_reqs_total.labels(
                **labels, case=case.value, decode_match=where
            ).inc(n)
        for case, tokens in stats.pending_gap_tokens.items():
            collector.pd_prefill_decode_ahead_tokens_total.labels(
                **labels, case=case.value
            ).inc(tokens)
        collector.pd_prefill_decode_prefix_tokens_total.labels(**labels).inc(
            stats.pending_decode_prefix_tokens
        )
        collector.pd_prefill_decode_pullable_ahead_tokens_total.labels(**labels).inc(
            stats.pending_pullable_ahead_tokens
        )
    stats.pending_reqs.clear()
    stats.pending_gap_tokens.clear()
    stats.pending_decode_prefix_tokens = 0
    stats.pending_pullable_ahead_tokens = 0
    if (
        reporter.is_stats_logging_rank
        and stats.reqs
        and _log_due(stats, time.monotonic())
    ):
        logger.info(stats.log_line())
    return result


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def note_decode_match(req, *, decode_prefix_len: int, pullable_len: int) -> None:
    """Match-time lengths of req, counted if its prealloc commits."""
    _pending_decode[req] = (decode_prefix_len, pullable_len)


def count_decode_prealloc(result, queue, decode_req, *args, **kwargs):
    entry = _pending_decode.pop(decode_req.req, None)
    # The host-receive path sends no prefix, whatever an earlier match found.
    if entry is None or kwargs.get("decode_prefix_len") != entry[0]:
        return result
    decode_prefix_len, pullable_len = entry
    case = _decode_stats.record(
        decode_prefix_len=decode_prefix_len, pullable_len=pullable_len
    )
    reporter = queue.scheduler.metrics_reporter
    if reporter.current_scheduler_metrics_enabled:
        collector = reporter.metrics_collector
        labels = collector.labels
        collector.pd_decode_prefix_window_reqs_total.labels(
            **labels, case=case.value
        ).inc()
        collector.pd_decode_prefix_tokens_total.labels(**labels).inc(decode_prefix_len)
        collector.pd_decode_pullable_tokens_total.labels(**labels).inc(pullable_len)
    if reporter.is_stats_logging_rank and _log_due(_decode_stats, time.monotonic()):
        logger.info(_decode_stats.log_line())
    return result


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def add_prefix_pull_counters(result, collector, *args, **kwargs):
    from prometheus_client import Counter as _PromCounter

    Counter = collector._counter_cls or _PromCounter
    base = list(collector.labels.keys())
    specs = (
        (
            "pd_prefill_prefix_gap_reqs_total",
            "Admitted P/D prefill requests by how the decode's prefix compares to "
            "the prefill's own match (case) and whether the decode matched up to "
            "the prompt's closing SWA window (decode_match).",
            ["case", "decode_match"],
        ),
        (
            "pd_prefill_decode_ahead_tokens_total",
            "Prompt tokens the decode holds beyond the prefill's own match, "
            "max(0, decode_prefix_len - prefill prefix), by case.",
            ["case"],
        ),
        (
            "pd_prefill_decode_prefix_tokens_total",
            "Sum of decode_prefix_len over admitted P/D prefill requests.",
            [],
        ),
        (
            "pd_prefill_decode_pullable_ahead_tokens_total",
            "Prompt tokens past the prefill's own match up to the decode's "
            "pullable length, max(0, decode_pullable_len - prefill prefix).",
            [],
        ),
        (
            "pd_decode_prefix_window_reqs_total",
            "Committed P/D decode preallocations by whether the SWA window at the "
            "decode's matched prefix end was still on device.",
            ["case"],
        ),
        (
            "pd_decode_prefix_tokens_total",
            "Sum of decode_prefix_len over committed P/D decode preallocations.",
            [],
        ),
        (
            "pd_decode_pullable_tokens_total",
            "Sum of the longest page-aligned decode prefix whose SWA window was on "
            "device, over committed P/D decode preallocations.",
            [],
        ),
    )
    specs += (
        (
            "pd_prefill_prefix_pull_total",
            "P/D prefix pulls by outcome: done, failed (strict) or fallback "
            "(recomputed).",
            ["outcome"],
        ),
        (
            "pd_prefill_prefix_pull_tokens_total",
            "Prompt tokens pulled from the decode instead of computed.",
            [],
        ),
        (
            "pd_prefill_prefix_pull_failed_total",
            "Failed prefix pulls that failed their request "
            "(SGLANG_TPU_PD_PREFIX_PULL_STRICT=1), by reason.",
            ["reason"],
        ),
        (
            "pd_prefill_prefix_pull_fallback_total",
            "Failed prefix pulls whose prefix was recomputed "
            "(SGLANG_TPU_PD_PREFIX_PULL_STRICT=0), by reason.",
            ["reason"],
        ),
        (
            "pd_prefill_prefix_pull_skipped_total",
            "Worthwhile prefix pulls not started, by reason.",
            ["reason"],
        ),
    )
    for name, doc, extra in specs:
        setattr(
            collector,
            name,
            Counter(name=f"sglang:{name}", documentation=doc, labelnames=base + extra),
        )
    return result


class PullOutcomeMetrics:
    """Prefill: what each prefix pull came to, counted on its own PP stage."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def _inc(self, name: str, value: int = 1, **labels) -> None:
        reporter = self.scheduler.metrics_reporter
        if not reporter.current_scheduler_metrics_enabled:
            return
        collector = reporter.metrics_collector
        getattr(collector, name).labels(**collector.labels, **labels).inc(value)

    def pulled(self, tokens: int) -> None:
        self._inc("pd_prefill_prefix_pull_total", outcome="done")
        self._inc("pd_prefill_prefix_pull_tokens_total", tokens)

    def failed(self, reason: str) -> None:
        self._inc("pd_prefill_prefix_pull_total", outcome="failed")
        self._inc("pd_prefill_prefix_pull_failed_total", reason=reason)

    def fallback(self, reason: str) -> None:
        self._inc("pd_prefill_prefix_pull_total", outcome="fallback")
        self._inc("pd_prefill_prefix_pull_fallback_total", reason=reason)

    def skipped(self, reason: str) -> None:
        self._inc("pd_prefill_prefix_pull_skipped_total", reason=reason)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    queue = "sglang.srt.disaggregation.decode.DecodePreallocQueue"
    return [
        (
            "sglang.srt.managers.schedule_policy.PrefillAdder."
            "_account_prefill_cache_admission",
            record_prefill_admission,
            HookType.AFTER,
        ),
        (
            "sglang.srt.managers.scheduler_components.metrics_reporter."
            "SchedulerMetricsReporter.report_prefill_stats",
            flush_prefill_stats,
            HookType.AFTER,
        ),
        (f"{queue}._send_kv_metadata", count_decode_prealloc, HookType.AFTER),
        (
            "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue.finalize_bootstrap",
            record_decode_report,
            HookType.AFTER,
        ),
        (
            "sglang.srt.observability.metrics_collector.SchedulerMetricsCollector.__init__",
            add_prefix_pull_counters,
            HookType.AFTER,
        ),
    ]
