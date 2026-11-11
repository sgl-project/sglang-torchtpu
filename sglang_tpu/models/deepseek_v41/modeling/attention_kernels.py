"""Per-stage reuse of gathered KV payloads across the layers of one forward stage."""

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

import torch


@dataclasses.dataclass
class _GatheredKVCache:
    """One live gather group; records are immutable within a forward stage."""

    source: Any = None
    records: torch.Tensor | None = None
    layout: tuple = ()
    values: tuple[torch.Tensor, torch.Tensor | None] | None = None
    valid_counts: torch.Tensor | None = None

    def get(
        self, source: Any, records: torch.Tensor, layout: tuple
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if self.source is source and self.records is records and self.layout == layout:
            return self.values
        return None

    def put(
        self,
        source: Any,
        records: torch.Tensor,
        layout: tuple,
        values: tuple[torch.Tensor, torch.Tensor | None],
        *,
        valid_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.source, self.records, self.layout, self.values = (
            source,
            records,
            layout,
            values,
        )
        self.valid_counts = valid_counts
        return values


_active_gathered_kv: _GatheredKVCache | None = None


@contextlib.contextmanager
def gathered_kv_cache_scope() -> Iterator[None]:
    """Reuse KV payloads across layers, never across scheduler steps/stages."""
    global _active_gathered_kv
    previous = _active_gathered_kv
    _active_gathered_kv = _GatheredKVCache()
    try:
        yield
    finally:
        _active_gathered_kv = previous
