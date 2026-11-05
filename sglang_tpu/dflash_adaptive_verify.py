# SPDX-License-Identifier: Apache-2.0
"""Batch-size-dependent DFlash verify width (SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY).

The draft always proposes a full block; a rule can verify only the block's first
``width`` tokens once the decode batch reaches ``min_bs`` requests, which takes
``bs * (block_size - width)`` rows off the target verify. A wide verify pays off
when few requests decode, a narrow one when many do.

Spec: comma-separated ``MIN_BS:WIDTH[:ACC_LEN]`` rules, e.g. ``3:4:3.06``.
``ACC_LEN`` is the accept length SGLANG_SIMULATE_ACC_LEN stands for at that width;
it is only read when acceptance is simulated, and required then.
"""

from __future__ import annotations

from collections.abc import Sequence

import msgspec


class VerifyWidthRule(msgspec.Struct, frozen=True):
    min_bs: int
    width: int
    simulate_accept_length: float | None


def parse_adaptive_verify(spec: str, block_size: int) -> tuple[VerifyWidthRule, ...]:
    """Parse the rules, ordered by ``min_bs``; widths at or above the block are dropped."""
    rules: list[VerifyWidthRule] = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"adaptive verify rule {item!r}: expected MIN_BS:WIDTH[:ACC_LEN]")
        min_bs, width = int(parts[0]), int(parts[1])
        accept_length = float(parts[2]) if len(parts) == 3 else None
        if min_bs < 1 or width < 2:
            raise ValueError(f"adaptive verify rule {item!r}: need MIN_BS >= 1 and WIDTH >= 2")
        if accept_length is not None and not 1.0 <= accept_length <= width:
            raise ValueError(f"adaptive verify rule {item!r}: ACC_LEN must be in [1, WIDTH]")
        if width < block_size:
            rules.append(
                VerifyWidthRule(min_bs=min_bs, width=width, simulate_accept_length=accept_length)
            )
    rules.sort(key=lambda r: r.min_bs)
    if len({r.min_bs for r in rules}) != len(rules):
        raise ValueError(f"adaptive verify spec {spec!r}: repeated MIN_BS")
    return tuple(rules)


def select_verify_width(
    rules: Sequence[VerifyWidthRule], bs: int
) -> VerifyWidthRule | None:
    """The rule for a decode batch of ``bs`` requests, or None for the full block."""
    chosen = None
    for rule in rules:
        if bs < rule.min_bs:
            break
        chosen = rule
    return chosen


def full_verify_buckets(
    rules: Sequence[VerifyWidthRule], buckets: Sequence[int]
) -> list[int]:
    """The batch buckets serving at least one logical batch that verifies the full block."""
    ordered = sorted(buckets)
    out = []
    for prev, bucket in zip([0] + ordered[:-1], ordered):
        if any(select_verify_width(rules, bs) is None for bs in range(prev + 1, bucket + 1)):
            out.append(bucket)
    return out


def adaptive_verify_shapes(
    rules: Sequence[VerifyWidthRule], buckets: Sequence[int]
) -> dict[int, list[int]]:
    """Narrow widths -> the batch buckets serving logical batches that use them."""
    shapes: dict[int, set[int]] = {}
    ordered = sorted(buckets)
    for bs in range(1, (ordered[-1] if ordered else 0) + 1):
        rule = select_verify_width(rules, bs)
        if rule is not None:
            bucket = next(b for b in ordered if b >= bs)
            shapes.setdefault(rule.width, set()).add(bucket)
    return {w: sorted(b) for w, b in sorted(shapes.items())}
