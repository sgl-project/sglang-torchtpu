"""Verify width per decode batch bucket (SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS).

A host-side choice from the padded batch size alone: no device reads, DP-attention
ranks (which agree on the bucket) agree on the width, and every reachable
(bucket, width) pair is warmed at startup, so switching never compiles.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import msgspec

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.envs import envs as tpu_envs


class VerifyWidthTable(msgspec.Struct, frozen=True):
    """Ascending (min_bs, width) steps from min_bs 1; a bucket takes the last step at or below it.

    The threshold is the padded batch bucket, not the live request count: with
    buckets [1, 4] two live requests verify at bucket 4's width, and under DP
    attention every rank pads to the shared bucket.
    """

    min_bs: tuple[int, ...]
    widths: tuple[int, ...]

    def width_for(self, padded_bs: int) -> int:
        return self.widths[bisect.bisect_right(self.min_bs, padded_bs) - 1]

    def shapes(self, buckets: Sequence[int]) -> tuple[tuple[int, int], ...]:
        """(bucket, width) for every batch bucket: the verify shapes serving can run."""
        return tuple((bs, self.width_for(bs)) for bs in buckets)

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
    if 1 not in steps:
        # Every bucket needs a width; spelling out the small-batch one ("1:8,3:4")
        # leaves no default to guess.
        raise ValueError(
            f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS must start at min_bs 1, got {spec!r}."
        )
    ordered = sorted(steps.items())
    return VerifyWidthTable(
        min_bs=tuple(bs for bs, _ in ordered),
        widths=tuple(width for _, width in ordered),
    )


def verify_width_table(
    max_width: int, *, spec_algorithm: SpeculativeAlgorithm
) -> VerifyWidthTable | None:
    """The configured table, or None to verify max_width rows at every bucket."""
    spec = tpu_envs.SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS.get()
    if spec is None:
        return None
    if spec_algorithm != SpeculativeAlgorithm.DFLASH:
        # A narrower verify drafts natively at its width; only the DFlash block
        # (its convolutions and lattice follow the block size) can.
        raise NotImplementedError(
            f"SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS supports DFLASH, not {spec_algorithm.name}."
        )
    return parse_verify_width_table(spec, max_width)


def parse_simulated_accept_lengths(spec: str) -> dict[int, float]:
    """Parse "width:accept_length,..." with accept_length in [1, width]."""
    lengths: dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            width_str, length_str = item.split(":")
            width, length = int(width_str), float(length_str)
        except ValueError as e:
            raise ValueError(
                f"SGLANG_TPU_SIMULATE_ACC_LEN_BY_WIDTH item {item!r} is not "
                "width:accept_length."
            ) from e
        if width in lengths or not 1.0 <= length <= width:
            raise ValueError(
                "SGLANG_TPU_SIMULATE_ACC_LEN_BY_WIDTH needs distinct widths and "
                f"accept lengths in [1, width], got {spec!r}."
            )
        lengths[width] = length
    return lengths


def simulated_accept_lengths(
    widths: Sequence[int], *, full_width: int, full_accept_length: float
) -> dict[int, float]:
    """The simulated accept length of every verify width.

    full_accept_length (SGLANG_SIMULATE_ACC_LEN) stands for full_width. A narrower
    width needs its own value: clamping the full-width one to the narrower width
    would simulate a draft that is always fully correct there.
    """
    spec = tpu_envs.SGLANG_TPU_SIMULATE_ACC_LEN_BY_WIDTH.get()
    by_width = {} if spec is None else parse_simulated_accept_lengths(spec)
    if full_width in by_width:
        raise ValueError(
            f"SGLANG_TPU_SIMULATE_ACC_LEN_BY_WIDTH sets the full {full_width}-token "
            "verify, which SGLANG_SIMULATE_ACC_LEN already does."
        )
    lengths = {full_width: full_accept_length}
    for width in widths:
        if width == full_width:
            continue
        if width not in by_width:
            raise ValueError(
                f"SGLANG_SIMULATE_ACC_LEN={full_accept_length} stands for the full "
                f"{full_width}-token verify; set SGLANG_TPU_SIMULATE_ACC_LEN_BY_WIDTH "
                f"for the {width}-token verify SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS reaches."
            )
        lengths[width] = by_width[width]
    return lengths
