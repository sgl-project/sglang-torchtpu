#!/usr/bin/env python3
"""Recompute the nightly accuracy floors from the dashboard's recorded scores.

    python3 .github/tpu-ci/derive_accuracy_floors.py \
        --metrics <tpu-ci-dashboard checkout>/state/metrics.json [--write]

Every entry in accuracy_floors.json names one dashboard series (`id`, gate label)
and how many graded items one night scores (`n_items`: questions x repeats). This
script takes the last `--nights` scores of each series and sets

    floor = mean - 3 * max(sigma_nightly, sigma_binomial) * sqrt(1 + 1/n)

sigma_binomial = sqrt(p (1 - p) / n_items) keeps a short, lucky streak of nights
from producing a floor tighter than the eval's own sampling noise. sqrt(1 + 1/n)
widens it for the error in a mean of n nights, so a series with a single night
is gated too, just more loosely (1.41x vs 1.10x at 5). A series with fewer
nights than asked is marked `provisional`; one with no nights falls back to its
entry's `seed` scores (e.g. a sibling series or a one-off run), and fails if it
has none.

A night is one dashboard date: a retried or re-dispatched run records the same
series again, so only the latest score per run_id, then per date, counts. An
entry may declare `min_floor` (with `min_floor_reason`) when the derived floor
is too loose for the gate to mean anything -- e.g. a 30-question eval at one
repeat; the floor is then max(derived, min_floor).

Without --write it prints the table and the diff, and changes nothing.
"""

import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

FLOORS = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/test/tpu/evals/accuracy_floors.json"
)


def _series(metrics_path: Path) -> dict:
    rows = json.loads(metrics_path.read_text())
    out: dict = {}
    for r in rows:
        labels = r.get("labels") or {}
        if labels.get("kind") != "accuracy" or not labels.get("id"):
            continue
        key = (labels["id"], r["metric_name"])
        out.setdefault(key, []).append((r["timestamp"], r["value"], r["run_id"]))
    for key, v in out.items():
        v.sort()
        # A retry of the same run, or a second dispatch the same night, would
        # otherwise count as an extra night and shrink sigma_nightly.
        by_run = {run: (ts, val, run) for ts, val, run in v}
        by_date = {ts[:10]: (ts, val, run) for ts, val, run in sorted(by_run.values())}
        out[key] = sorted(by_date.values())
    return out


def derive(entry: dict, nights: list, want: int) -> dict:
    recent = nights[-want:]
    values = [v for _, v, _ in recent] or list(entry.get("seed", []))
    if not values:
        raise ValueError("no nightly scores and no seed")
    mean = statistics.fmean(values)
    sigma_nightly = statistics.stdev(values) if len(values) > 1 else 0.0
    sigma_binomial = math.sqrt(max(mean * (1 - mean), 0.0) / entry["n_items"])
    sigma = max(sigma_nightly, sigma_binomial)
    # Prediction interval: the next night's noise plus the error in the mean.
    width = 3 * sigma * math.sqrt(1 + 1 / len(values))
    # Rounded down so a score printed at 3 decimals never sits on the line.
    floor = math.floor((mean - width) * 1000) / 1000
    return {
        "floor": max(floor, entry.get("min_floor", floor)),
        "mean": round(mean, 4),
        "sigma": round(sigma, 4),
        "sigma_source": "nightly" if sigma_nightly >= sigma_binomial else "binomial",
        "nights": len(recent),
        "runs": [run for _, _, run in recent],
        "window": [recent[0][0][:10], recent[-1][0][:10]] if recent else None,
        "provisional": len(recent) < want,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--nights", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    doc = json.loads(FLOORS.read_text())
    series = _series(args.metrics)
    today = datetime.date.today().isoformat()
    changed = False
    for sid, gates in doc["floors"].items():
        for label, entry in gates.items():
            got = derive(entry, series.get((sid, label), []), args.nights)
            old = entry.get("floor")
            flag = " provisional" if got["provisional"] else ""
            print(
                f"{sid:38s} {label:20s} floor {old} -> {got['floor']:.3f} "
                f"(mean {got['mean']:.3f}, sigma {got['sigma']:.4f} "
                f"[{got['sigma_source']}], nights {got['nights']}){flag}"
            )
            # Refresh provenance too: an unchanged floor still has new nights
            # (runs, window, mean) behind it.
            if any(entry.get(k) != v for k, v in got.items()):
                changed = True
                entry.update(got, set_on=today)
    if args.write and changed:
        FLOORS.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {FLOORS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
