"""Verify width per decode batch bucket (SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS).

A host-side choice from the padded batch size alone: no device reads, and every
reachable (bucket, width) pair is warmed at startup, so switching never compiles.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import msgspec

from sglang_tpu.envs import envs as tpu_envs


class VerifyWidthTable(msgspec.Struct, frozen=True):
    """Ascending (min_bs, width) steps; a bucket takes the last step at or below it."""

    min_bs: tuple[int, ...]
    widths: tuple[int, ...]

    def width_for(self, padded_bs: int) -> int:
        index = bisect.bisect_right(self.min_bs, padded_bs) - 1
        return self.widths[max(index, 0)]

    def reachable_widths(self, buckets: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted({self.width_for(bs) for bs in buckets}))


def parse_verify_width_table(spec: str, max_width: int) -> VerifyWidthTable:
    """Parse "min_bs:width,..." with every width in [2, max_width]."""
    steps: dict[int, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            min_bs_str, width_str = item.split(":")
            min_bs, width = int(min_bs_str), int(width_str)
        except ValueError as e:
            raise ValueError(
                f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS item {item!r} is not min_bs:width."
            ) from e
        if min_bs < 1 or min_bs in steps:
            raise ValueError(
                f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS needs distinct min_bs >= 1, got {spec!r}."
            )
        if not 2 <= width <= max_width:
            raise ValueError(
                f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS width {width} is outside "
                f"[2, gamma + 1 = {max_width}]."
            )
        steps[min_bs] = width
    if not steps:
        raise ValueError("SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS is empty.")
    ordered = sorted(steps.items())
    return VerifyWidthTable(
        min_bs=tuple(bs for bs, _ in ordered),
        widths=tuple(width for _, width in ordered),
    )


def verify_width_table(max_width: int) -> VerifyWidthTable | None:
    """The configured table, or None to verify max_width rows at every bucket."""
    spec = tpu_envs.SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS.get()
    if spec is None:
        return None
    return parse_verify_width_table(spec, max_width)
