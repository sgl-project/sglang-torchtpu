"""TPU shape bucketing utilities for input compilation."""

from __future__ import annotations

import bisect


class Bucketer:
    """Maps runtime dimension sizes to discrete bucket boundaries."""

    def __init__(self, name: str, buckets: list[int]):
        if not buckets:
            raise ValueError("Bucket list cannot be empty.")
        if any(b <= 0 for b in buckets):
            raise ValueError(f"Bucket sizes must be positive for {name!r}: {buckets}.")
        self.name = name
        self.buckets = sorted(set(buckets))
        self.min_val = self.buckets[0]
        self.max_val = self.buckets[-1]

    def next_bucket(self, n: int) -> int:
        """Get the smallest bucket size >= n."""
        if n > self.max_val:
            raise ValueError(
                f"Requested size {n} exceeds the maximum bucket size "
                f"{self.max_val} for {self.name!r}."
            )
        idx = bisect.bisect_left(self.buckets, n)
        return self.buckets[idx]

    def __repr__(self) -> str:
        return f"Bucketer(name={self.name!r}, buckets={self.buckets})"


class PowerBucketer(Bucketer):
    """Bucketer whose boundaries form a geometric progression."""

    def __init__(self, name: str, min_val: int, max_val: int, power: int = 2):
        if min_val < 1:
            raise ValueError(f"min_val must be >= 1, got {min_val}.")
        if max_val < min_val:
            raise ValueError(f"max_val ({max_val}) must be >= min_val ({min_val}).")
        if power < 2:
            raise ValueError(f"power must be >= 2, got {power}.")

        buckets: list[int] = []
        v = min_val
        while v < max_val:
            buckets.append(v)
            v *= power
        buckets.append(max_val)
        super().__init__(name, buckets)
