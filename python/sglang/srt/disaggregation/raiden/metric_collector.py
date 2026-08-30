"""Metrics for TPU Raiden PD KV cache transfer."""

from __future__ import annotations

import threading
from typing import Any, Dict, Literal, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_disagg, get_observability, get_serving

TransferOutcome = Literal["success", "failure", "cancelled"]

LABEL_TRANSFER_MODE = "pull"
LABEL_TRANSFER_TYPES = ("chunked", "whole")
LABEL_TRANSFER_OUTCOMES: tuple[TransferOutcome, ...] = (
    "success",
    "failure",
    "cancelled",
)


class NullTPURaidenMetricsCollector:
    """No-op interface for disabled metrics and non-primary TP ranks."""

    def transfer_started(self, room: int, *, chunked: bool = False) -> None:
        pass

    def transfer_finished(self, room: int, outcome: TransferOutcome) -> None:
        pass


class TPURaidenMetricsCollector(NullTPURaidenMetricsCollector):
    """Count local room outcomes and track active rooms under one lock."""

    @classmethod
    def init_new(
        cls,
        *,
        tp_rank: int,
        dp_rank: int,
        registry: Optional[Any] = None,
    ) -> TPURaidenMetricsCollector | NullTPURaidenMetricsCollector:
        if not get_observability().enable_metrics or tp_rank != 0:
            return NullTPURaidenMetricsCollector()
        labels = {
            "model_name": get_serving().served_model_name,
            "engine_type": DisaggregationMode.to_engine_type(
                get_disagg().disaggregation_mode
            ),
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
        }
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        return cls(
            labels=labels,
            registry=registry,
        )

    def __init__(
        self,
        labels: Dict[str, Any],
        *,
        registry: Optional[Any] = None,
    ) -> None:
        from prometheus_client import Counter, Gauge

        kwargs = {} if registry is None else {"registry": registry}
        self._base_labels = {str(key): str(value) for key, value in labels.items()}
        self._transfer_labels = {
            transfer_type: self._label_values(
                transfer_mode=LABEL_TRANSFER_MODE, transfer_type=transfer_type
            )
            for transfer_type in LABEL_TRANSFER_TYPES
        }
        self._active_transfers: Dict[str, set[int]] = {
            kind: set() for kind in LABEL_TRANSFER_TYPES
        }
        self._lock = threading.Lock()

        self.transfers = Counter(
            name="sglang:tpu_raiden_kv_transfers_total",
            documentation="Raiden KV transfers by local terminal outcome.",
            labelnames=self._label_keys("transfer_mode", "transfer_type", "outcome"),
            **kwargs,
        )
        self.active_transfers = Gauge(
            name="sglang:tpu_raiden_kv_active_transfers",
            documentation="Current number of active Raiden KV transfers on this rank.",
            labelnames=self._label_keys("transfer_mode", "transfer_type"),
            multiprocess_mode="mostrecent",
            **kwargs,
        )
        self._initialize_zero_series()

    def _label_keys(self, *extra: str) -> tuple[str, ...]:
        return (*self._base_labels, *extra)

    def _label_values(self, **extra: str) -> Dict[str, str]:
        return {
            **self._base_labels,
            **{key: str(value) for key, value in extra.items()},
        }

    def _initialize_zero_series(self) -> None:
        for labels in self._transfer_labels.values():
            self.active_transfers.labels(**labels).set(0)
            for outcome in LABEL_TRANSFER_OUTCOMES:
                self.transfers.labels(**labels, outcome=outcome)

    def transfer_started(self, room: int, *, chunked: bool = False) -> None:
        with self._lock:
            if any(room in rooms for rooms in self._active_transfers.values()):
                return
            transfer_type = "chunked" if chunked else "whole"
            rooms = self._active_transfers[transfer_type]
            rooms.add(room)
            self.active_transfers.labels(**self._transfer_labels[transfer_type]).set(
                len(rooms)
            )

    def transfer_finished(self, room: int, outcome: TransferOutcome) -> None:
        with self._lock:
            for transfer_type, rooms in self._active_transfers.items():
                if room not in rooms:
                    continue
                rooms.remove(room)
                labels = self._transfer_labels[transfer_type]
                self.active_transfers.labels(**labels).set(len(rooms))
                self.transfers.labels(**labels, outcome=outcome).inc()
                return
