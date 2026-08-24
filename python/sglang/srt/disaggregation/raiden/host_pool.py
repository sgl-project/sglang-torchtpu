"""Admission control for raiden's host staging pool.

A raiden engine reserves a fixed number of host staging slots at startup and
borrows one per in-flight transfer.
"""

from __future__ import annotations

from typing import List, Set


class SlotAdmission:
    """Admission control over raiden's host staging slots.

    Raiden borrows one slot per in-flight transfer and rejects a read it cannot
    seat, so a consumer must not issue more concurrent pulls than the pool
    holds. Rooms that are not admitted stay pending and are retried.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._in_flight: Set[int] = set()

    def __len__(self) -> int:
        return len(self._in_flight)

    def admit(self, room: int) -> bool:
        """Reserves a slot for ``room``. False means try again later."""
        if room in self._in_flight:
            return True
        if len(self._in_flight) >= self.capacity:
            return False
        self._in_flight.add(room)
        return True

    def release(self, room: int) -> None:
        self._in_flight.discard(room)

    def rooms(self) -> List[int]:
        return list(self._in_flight)
