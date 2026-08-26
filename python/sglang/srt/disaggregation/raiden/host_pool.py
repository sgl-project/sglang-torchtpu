"""Admission control for raiden's host staging pool.

A raiden engine reserves a fixed number of host staging slots at startup and
borrows one per in-flight transfer.
"""

from __future__ import annotations

from typing import Any, List, Set


class SlotAdmission:
    """Admission control over raiden's host staging slots.

    Raiden borrows one slot per in-flight transfer and rejects a read it cannot
    seat, so a consumer must not issue more concurrent reads than the pool
    holds. Reads that are not admitted stay pending and are retried. Keys are
    whatever identifies one read -- a room, or a room's chunk.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._in_flight: Set[Any] = set()

    def __len__(self) -> int:
        return len(self._in_flight)

    def __contains__(self, key: Any) -> bool:
        return key in self._in_flight

    def admit(self, key: Any) -> bool:
        """Reserves a slot for ``key``. False means try again later."""
        if key in self._in_flight:
            return True
        if len(self._in_flight) >= self.capacity:
            return False
        self._in_flight.add(key)
        return True

    def release(self, key: Any) -> None:
        self._in_flight.discard(key)

    def rooms(self) -> List[Any]:
        return list(self._in_flight)
