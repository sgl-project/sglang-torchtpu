# SPDX-License-Identifier: Apache-2.0
"""Saved pipeline chunk cost fits, keyed per serving shape.

A table file maps a shape key to a fit measured offline. A launch whose shape
and fingerprint match an entry installs the saved fit instead of profiling;
a missing table entry or a mismatched fingerprint falls back to the launch-time
profile, with a warning that names the cause.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
from pathlib import Path

import msgspec
import msgspec.structs

from sglang_tpu.pp_chunks import StepCostModel

# Step costs depend on MoE routing; balanced or simulated routing on dummy
# weights times differently from real serving, so only real fits are saved.
_REQUIRED_CALIBRATION = "real"
# A guard point more than this fraction off the saved prediction rejects the fit.
# Arbitrary: wider than the run-to-run spread of the robust profile (under 1%).
GUARD_TOLERANCE = 0.10
# A smaller slack tightens the tolerance to the slack, since a fit accepted past it
# puts the full no-prefix step over the target, but never below this floor, so a
# slack near zero does not reject every fit on run-to-run noise.
GUARD_TOLERANCE_FLOOR = 0.02


class FitFingerprint(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """What a saved fit was measured against; any difference invalidates it."""

    profile_points: list[tuple[int, int]]
    schedule: tuple[int, int, int] | None
    # Owned by the deployment that ships the table; bumped by any kernel change
    # that moves step costs. The startup guard backstops a missed bump.
    kernel_version: str
    runtime_version: str


class FitCalibration(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    weights: str
    routing: str
    date: str = ""
    samples: int = 0
    command: str = ""


class SavedFit(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    linear_ms: dict[int, float]
    attn_ms_per_token_pair: float
    slack: float
    fingerprint: FitFingerprint
    calibration: FitCalibration
    # (tokens, prefix, ms) of the slowest stage at the guard points when calibrated;
    # the fitted curve is an envelope over stages, so the guard compares these instead.
    guard_points: list[tuple[int, int, float]]


def fit_key(
    *,
    model_type: str,
    num_layers: int,
    hidden_size: int,
    model_shape: str,
    dtype: str,
    quantization: str | None,
    kv_cache_dtype: str,
    attention_backend: str | None,
    moe_runner_backend: str | None,
    moe_a2a_backend: str | None,
    device_kind: str,
    pp_size: int,
    tp_size: int,
    attn_cp_size: int,
    attn_dcp_size: int,
    moe_ep_size: int,
    moe_tp_size: int,
    moe_dense_tp_size: int,
    attn_dp_size: int,
    chunk_tokens: int,
    extend_batch_cap: int,
    page_size: int,
    layer_partition: str,
) -> str:
    """The serving shape a fit belongs to."""
    return "/".join(
        (
            model_type,
            f"L{num_layers}xH{hidden_size}",
            model_shape,
            dtype,
            quantization or "none",
            f"kv{kv_cache_dtype}",
            attention_backend or "default",
            moe_runner_backend or "default",
            f"a2a{moe_a2a_backend or 'none'}",
            device_kind,
            f"pp{pp_size}xtp{tp_size}xcp{attn_cp_size}xdcp{attn_dcp_size}"
            f"xep{moe_ep_size}xmoetp{moe_tp_size}xdensetp{moe_dense_tp_size}"
            f"xdp{attn_dp_size}",
            f"chunk{chunk_tokens}",
            f"extendbs{extend_batch_cap}",
            f"page{page_size}",
            f"split{layer_partition or 'even'}",
        )
    )


def runtime_version() -> str:
    versions = []
    for package in ("torch_tpu", "libtpu"):
        try:
            versions.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{package}=none")
    return ";".join(versions)


def _read_raw_table(table_path: str) -> dict[str, msgspec.Raw]:
    return msgspec.json.decode(
        Path(table_path).read_bytes(), type=dict[str, msgspec.Raw]
    )


def read_saved_fits(table_path: str) -> tuple[dict[str, SavedFit], dict[str, str]]:
    """The table's entries that decode, and the error of each one that does
    not. A bad or newer-schema entry costs only its own key; a file that cannot
    be read or parsed as a JSON object raises."""
    entries, unreadable = {}, {}
    for key, raw in _read_raw_table(table_path).items():
        try:
            entries[key] = msgspec.json.decode(raw, type=SavedFit)
        except msgspec.DecodeError as error:
            unreadable[key] = str(error)
    return entries, unreadable


def _fingerprint_mismatch(saved: FitFingerprint, current: FitFingerprint) -> str:
    current_fields = msgspec.structs.asdict(current)
    for field, value in msgspec.structs.asdict(saved).items():
        if value != current_fields[field]:
            return field
    return ""


def load_saved_fit(
    *, table_path: str, key: str, fingerprint: FitFingerprint, check: bool
) -> tuple[SavedFit | None, str]:
    """The saved fit for ``key``, or None and the reason it cannot be used.
    With the startup check on, the entry must carry guard points to re-time."""
    # Any read failure becomes this rank's vote: raising here would leave the
    # other ranks waiting in the vote's gather.
    try:
        if not Path(table_path).is_file():
            return None, f"no fit table at {table_path}"
        entries, unreadable = read_saved_fits(table_path)
    except (OSError, msgspec.DecodeError) as error:
        return None, f"cannot read fit table {table_path}: {error}"
    if key in unreadable:
        return None, f"saved fit for {key} is unreadable: {unreadable[key]}"
    entry = entries.get(key)
    if entry is None:
        return None, f"no saved fit for {key} in {table_path}"
    calibration = entry.calibration
    if (calibration.weights, calibration.routing) != (
        _REQUIRED_CALIBRATION,
        _REQUIRED_CALIBRATION,
    ):
        return None, (
            f"saved fit for {key} was calibrated with weights={calibration.weights} "
            f"routing={calibration.routing}; only real weights and routing are used"
        )
    mismatch = _fingerprint_mismatch(entry.fingerprint, fingerprint)
    if mismatch:
        return None, f"saved fit for {key} is stale: {mismatch} differs"
    # Every compiled bucket is profiled without a prefix, so the ladder must cover exactly those.
    buckets = sorted(
        {tokens for tokens, prefix in fingerprint.profile_points if prefix == 0}
    )
    if not buckets or sorted(entry.linear_ms) != buckets:
        return (
            None,
            f"saved fit for {key} has buckets {sorted(entry.linear_ms)}, not {buckets}",
        )
    # The fitter clamps terms at zero, so zero is valid; the full step must cost something.
    top = buckets[-1]
    full_step = entry.linear_ms[top] + entry.attn_ms_per_token_pair * top * top / 2
    if (
        min(entry.linear_ms.values()) < 0
        or entry.attn_ms_per_token_pair < 0
        or not 0 < full_step < math.inf
    ):
        return None, f"saved fit for {key} has a negative, zero or unbounded step cost"
    if any(ms <= 0 for _, _, ms in entry.guard_points):
        return None, f"saved fit for {key} has a guard time that is not positive"
    if check:
        saved = {(tokens, prefix) for tokens, prefix, _ in entry.guard_points}
        missing = [
            point
            for point in guard_points(fingerprint.profile_points)
            if point not in saved
        ]
        if missing:
            return None, f"saved fit for {key} has no calibrated time at {missing}"
    return entry, f"saved fit for {key} (calibrated {calibration.date or 'undated'})"


def saved_cost_model(
    entry: SavedFit, *, slack: float, granularity: int
) -> StepCostModel:
    """The cost model of a saved fit. The target comes from the launch's slack,
    so a slack change needs no new entry; the saved slack is provenance only."""
    buckets = sorted(entry.linear_ms)
    top = buckets[-1]
    attn = entry.attn_ms_per_token_pair
    full_step = entry.linear_ms[top] + attn * top * top / 2
    return StepCostModel(
        buckets,
        dict(entry.linear_ms),
        attn,
        full_step * (1.0 + slack),
        int(granularity),
        entry.fingerprint.schedule,
    )


def slowest_stage_points(
    stage_samples: list[list[tuple[int, int, float]]],
) -> list[tuple[int, int, float]]:
    slowest: dict[tuple[int, int], float] = {}
    for samples in stage_samples:
        for tokens, prefix, ms in samples:
            slowest[(tokens, prefix)] = max(slowest.get((tokens, prefix), 0.0), ms)
    return [(tokens, prefix, ms) for (tokens, prefix), ms in sorted(slowest.items())]


def weak_fingerprint_fields(fingerprint: FitFingerprint) -> list[str]:
    """Fingerprint fields that match any table entry vacuously, so leave the
    startup guard as the only drift check."""
    weak = []
    if not fingerprint.kernel_version:
        weak.append("kernel_version is empty")
    if "=none" in fingerprint.runtime_version:
        weak.append(f"runtime_version is {fingerprint.runtime_version}")
    return weak


def guard_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The points the startup guard re-times: the smallest and the largest
    bucket without a prefix, and the largest prefixed point (the largest bucket
    that has a prefix, behind its longest one), so drift at either end of the
    ladder or in the attention cost rejects the fit."""
    buckets = sorted({tokens for tokens, prefix in points if prefix == 0})
    guard = {(buckets[0], 0), (buckets[-1], 0)}
    prefixed = [(tokens, prefix) for tokens, prefix in points if prefix > 0]
    if prefixed:
        guard.add(max(prefixed))
    return sorted(guard)


def guard_report(
    saved: list[tuple[int, int, float]],
    timed: list[tuple[int, int, float]],
    *,
    slack: float,
) -> tuple[list[str], list[str]]:
    """Re-timed slowest-stage guard points against the calibrated ones:
    a line per point, and the lines of the points off by more than the tolerance."""
    tolerance = min(GUARD_TOLERANCE, max(slack, GUARD_TOLERANCE_FLOOR))
    reference = {(tokens, prefix): ms for tokens, prefix, ms in saved}
    report, drifted = [], []
    for tokens, prefix, ms in timed:
        expected = reference.get((tokens, prefix))
        if expected is None:
            line = f"({tokens}, {prefix}) measured {ms:.1f} ms, no calibrated time"
        else:
            line = (
                f"({tokens}, {prefix}) measured {ms:.1f} ms, "
                f"saved {expected:.1f} ms ({ms / expected - 1:+.1%})"
            )
        report.append(line)
        if expected is None or abs(ms - expected) > tolerance * expected:
            drifted.append(line)
    return report, drifted


def saved_fit(
    model: StepCostModel,
    *,
    slack: float,
    fingerprint: FitFingerprint,
    calibration: FitCalibration,
    guard_points: list[tuple[int, int, float]],
) -> SavedFit:
    return SavedFit(
        linear_ms=dict(model.linear_ms),
        attn_ms_per_token_pair=model.attn_ms_per_token_pair,
        slack=slack,
        fingerprint=fingerprint,
        calibration=calibration,
        guard_points=guard_points,
    )


def upsert_saved_fit(table_path: str, key: str, entry: SavedFit) -> None:
    """Write ``entry`` under ``key``, keeping the table canonical (sorted keys).
    Other entries are kept as written, so this also replaces an unreadable entry
    under its own key. For saving a launch's logged fit candidate offline; a
    single writer, so concurrent writers are not locked."""
    # Through a symlink, replace the file it points to, not the link.
    path = Path(os.path.realpath(table_path))
    table = _read_raw_table(str(path)) if path.is_file() else {}
    table[key] = msgspec.Raw(msgspec.json.encode(entry))
    encoded = msgspec.json.encode(dict(sorted(table.items())))
    # Replace atomically: a launch reading the table must never see a partial file.
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_bytes(msgspec.json.format(encoded, indent=2) + b"\n")
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
