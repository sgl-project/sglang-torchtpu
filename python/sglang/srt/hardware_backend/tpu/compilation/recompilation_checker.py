"""Serving-time XLA recompilation detector for TPU.

Detects new XLA program compilations after the precompilation phase is
complete. New compilations at serving time cause latency spikes (seconds
each) and are hard to diagnose without monitoring.

Opt in via ``SGLANG_TPU_CHECK_RECOMPILATION=1``.
"""

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


def _get_xla_graph_count() -> int:
    """Return the number of compiled XLA programs in the cache."""
    try:
        stats = torch.tpu._get_cache_stats()
        return len(stats.per_entry_stats)
    except (AttributeError, RuntimeError):
        return -1


class RecompilationChecker:
    """Detects serving-time XLA recompilation by comparing graph counts.

    After precompilation, call ``snapshot_baseline()`` to record the
    current XLA graph count. Then call ``check(context)`` on each
    forward to detect (and warn about) any new compilations.
    """

    def __init__(self) -> None:
        self._baseline_graph_count: int = -1

    def snapshot_baseline(self) -> None:
        """Record the current XLA graph count as the baseline."""
        self._baseline_graph_count = _get_xla_graph_count()
        logger.info(
            "[recompilation checker] XLA graph baseline: %d cached programs.",
            self._baseline_graph_count,
        )

    def check(self, context: str = "") -> int:
        """Check if new XLA programs were compiled since the baseline.

        Args:
            context: human-readable description of the current forward
                (e.g. "LlamaForCausalLM EXTEND") for the warning.

        Returns the number of new compilations (0 = no recompilation).
        """
        if self._baseline_graph_count < 0:
            return 0  # cache stats unavailable
        current = _get_xla_graph_count()
        if current < 0:
            return 0
        new_graphs = current - self._baseline_graph_count
        if new_graphs > 0:
            logger.warning(
                "[recompilation checker] %d new XLA program(s) compiled at "
                "serving time [%s] (baseline=%d, current=%d). This causes "
                "latency spikes.",
                new_graphs,
                context,
                self._baseline_graph_count,
                current,
            )
            # Update baseline so we don't warn repeatedly for the same graphs.
            self._baseline_graph_count = current
        return new_graphs
