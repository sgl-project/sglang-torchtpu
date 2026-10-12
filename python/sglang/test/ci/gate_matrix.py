"""Sweep sizing shared by the TPU PR gate and the nightly full-matrix run.

Several TPU tests are shape or rank sweeps where every case is its own Mosaic
compile, so their cost is linear in the number of cases and a handful of them
dominate the gate. The cases are not redundant -- they are just not all needed
on every commit. These helpers let a test state its full sweep and the subset
the gate runs, in one place a reviewer can read.

`SGLANG_TPU_FULL_MATRIX=1` selects the full sweep. `render_jobset.py` sets it
on nightly runs, so a test trimmed here must also be registered for a nightly
suite or its dropped cases stop running anywhere.
"""

import os

_TRUTHY = ("1", "true", "yes", "on")


def full_matrix() -> bool:
    """True when the caller should run its whole sweep."""
    return os.environ.get("SGLANG_TPU_FULL_MATRIX", "").strip().lower() in _TRUTHY


def sweep(full, gate):
    """`full` on a full-matrix run, `gate` otherwise.

    `gate` must be a subset of `full`: the gate narrows coverage, it never
    tests something the nightly does not.
    """
    full = tuple(full)
    gate = tuple(gate)
    unknown = [case for case in gate if case not in full]
    if unknown:
        raise ValueError(f"gate cases missing from the full sweep: {unknown}")
    return full if full_matrix() else gate


def shard_ranks(cp: int):
    """Which ranks of a `cp`-way split to exercise.

    A full run walks every rank. The gate takes both boundaries and the first
    interior rank: rank 0 and rank cp-1 are where the local-position arithmetic
    and the ragged tail differ, and one interior rank covers the general case.
    Below four ranks there is no interior, so the gate runs them all.
    """
    if full_matrix() or cp <= 3:
        return tuple(range(cp))
    return (0, 1, cp - 1)
