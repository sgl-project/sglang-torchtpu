#!/usr/bin/env python3
"""Pick the accuracy legs one accuracy-test-tpu.yml run executes.

accuracy_slots.json assigns every scheduled accuracy leg to one daytime slot:
single-host legs by suite, multi-host legs by discovered group name. The weekly
est_time bot (update_tpu_est_time.py) rebalances the slots from measured
runtimes, so a new leg only needs an entry; any slot will do until then.

  accuracy_slots.py singlehost <selection>            -> matrix JSON
  accuracy_slots.py multihost <selection> < groups    -> filtered groups JSON

A selection is a slot (day-10am, day-5pm), all / full-set, or a model name.

Stdlib only: lint runs .github/tpu-ci/tests with `python -E -S -B`.
"""

import json
import sys
from pathlib import Path

TABLE = Path(__file__).resolve().parent / "accuracy_slots.json"
SLOTS = ("day-10am", "day-5pm")
EVERYTHING = ("all", "full-set")
SUITE_PREFIX = "nightly-tpu-accuracy-"
# JobSet validates the pod name, tpu-ci-<run_name>-<5>-slice-0-0-<5>, against 63.
MAX_RUN_NAME = 34


def load(path: Path = TABLE) -> dict:
    return json.loads(path.read_text())


def run_name(suite: str) -> str:
    name = "accuracy-" + suite.removeprefix(SUITE_PREFIX)
    if len(name) > MAX_RUN_NAME:
        raise ValueError(f"run_name {name} exceeds {MAX_RUN_NAME} chars")
    return name


def singlehost(table: dict, selection: str) -> list:
    """Matrix legs for the single-host accuracy job."""
    legs = []
    for suite, leg in sorted(table["singlehost"].items()):
        if selection in SLOTS:
            keep = leg["slot"] == selection
        elif selection in EVERYTHING:
            keep = True
        else:
            keep = f"-{selection}-" in f"-{suite.removeprefix(SUITE_PREFIX)}-"
        if keep:
            legs.append(
                {
                    "suite": suite,
                    "run_name": run_name(suite),
                    "timeout_per_file": leg["timeout_per_file"],
                }
            )
    return legs


def multihost(table: dict, groups: list, selection: str) -> list:
    """The discovered multi-host groups this selection runs."""
    if selection in EVERYTHING:
        return groups
    if selection not in SLOTS:
        return [g for g in groups if selection.replace("-", "_") in g["name"]]
    missing = [g["name"] for g in groups if g["name"] not in table["multihost"]]
    if missing:
        raise ValueError(f"{missing} have no slot in {TABLE.name}")
    return [g for g in groups if table["multihost"][g["name"]]["slot"] == selection]


def main() -> int:
    kind, selection = sys.argv[1:3]
    table = load()
    if kind == "singlehost":
        out = singlehost(table, selection)
    elif kind == "multihost":
        out = multihost(table, json.load(sys.stdin), selection)
    else:
        sys.exit(f"unknown kind {kind!r}")
    print(json.dumps(out, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
