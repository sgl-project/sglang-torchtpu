"""Trace-cost profiles that balance shared-compilation owners across ranks.

Under --use-spmd-compile one owner rank traces and compiles each shared graph,
and startup waits for the rank with the most tracing. Round-robin owners give
every rank the same number of graphs, but a few graphs (the model forwards)
cost far more than the rest, and adding or dropping any graph moves every later
one to another rank. A profile instead records each graph family's trace cost,
and each graph goes to the rank with the least estimated work so far.

A profile also records each graph's own mean trace time (schema 2). Cost grows
with a forward's token count (a 2048-token EXTEND traced ~5x a 6-token verify on
DSV4.1), so graphs the profile lists are assigned up front, largest first to the
least-loaded rank (``CostProfile.plan``); warmup enqueues cheap graphs first,
which defeats a greedy assignment in enqueue order. Most of a forward's trace is
first-touch kernel lowering for its shapes, cached in the tracing process, so a
graph whose affinity key (its name without the last shape field, e.g. the
decode variant) the same rank already traced costs only ``repeat_fraction`` of
its own cost (~0.13 on DSV4.1); the plan keeps such graphs together when that
balances better. Other graphs fall back to a
family estimate: the graph's name with every number replaced by ``N``, or
``default_seconds`` for unlisted families. Build a profile from the
``[shared_compile]`` lines of a startup log::

    python -m sglang_tpu.compilation.shared_compile_cost LOG... --output PROFILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections.abc import Iterable
from pathlib import Path

import msgspec

from sglang_tpu.envs import envs as tpu_envs

SCHEMA_VERSION = 2
# Profiles shipped with the plugin; the env may name one of these files.
PACKAGED_DIR = Path(__file__).with_name("cost_profiles")
# Schema 1 profiles (families only) still load.
_SUPPORTED_SCHEMAS = (1, 2)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TRACE_LINE = re.compile(r"\[shared_compile\] owner=(\d+) trace=(\S+)s graph=(.+)$")
_SHAPE_LIST = re.compile(r"^(.*:\[.*),[^,\[\]]*\]$")


def graph_family(graph: str) -> str:
    """The family of a shared graph name: its name with numbers replaced by N."""
    return _NUMBER.sub("N", graph.split(":", 1)[0])


def affinity_key(graph: str) -> str:
    """Graphs sharing this key reuse each other's cached kernel lowering on one
    rank: the name without the last field of its shape list."""
    match = _SHAPE_LIST.match(graph)
    return f"{match.group(1)}]" if match else graph


def _seconds(value: object, field: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"shared compile cost profile {field} is invalid: {value!r}")
    return seconds


class CostProfile(msgspec.Struct, frozen=True):
    """Trace-cost estimates of shared graphs, in seconds."""

    default_seconds: float
    families: dict[str, float]
    graphs: dict[str, float] = {}
    # Cost of a graph whose affinity key its rank already traced, as a fraction
    # of its own; 1.0 (no reuse) when the logs showed none.
    repeat_fraction: float = 1.0

    def estimate(self, graph: str) -> float:
        """The graph's own cost, else its family's, else the default."""
        cost = self.graphs.get(graph)
        if cost is None:
            cost = self.families.get(graph_family(graph), self.default_seconds)
        return cost

    def plan(self, size: int) -> tuple[dict[str, int], list[float]]:
        """Owners of the listed graphs and each rank's planned load, counting a
        graph at ``repeat_fraction`` of its cost on a rank that already has its
        affinity key. Largest graph first; under a target makespan a graph takes
        the rank where it costs least among those it fits on, else the rank
        where it ends lowest; the best of a sweep of targets wins. Deterministic
        (ties: lowest rank), so every rank computes the same plan."""
        order = sorted(self.graphs.items(), key=lambda kv: (-kv[1], kv[0]))
        if not order:
            return {}, [0.0] * size
        total = sum(cost for _, cost in order)
        floor = max(order[0][1], total * self.repeat_fraction / size)
        best = self._assign(order, size, math.inf)  # plain largest-first
        if self.repeat_fraction < 1.0:
            ceiling = max(best[1])
            for step in range(41):
                target = floor + (ceiling - floor) * step / 40
                candidate = self._assign(order, size, target)
                if max(candidate[1]) < max(best[1]) - 1e-9:
                    best = candidate
        return best

    def _assign(
        self, order: list[tuple[str, float]], size: int, target: float
    ) -> tuple[dict[str, int], list[float]]:
        loads = [0.0] * size
        keys: list[set[str]] = [set() for _ in range(size)]
        owners: dict[str, int] = {}
        for graph, cost in order:
            key = affinity_key(graph)
            costs = [
                cost * self.repeat_fraction if key in keys[rank] else cost
                for rank in range(size)
            ]
            fits = [r for r in range(size) if loads[r] + costs[r] <= target]
            if fits and target != math.inf:
                rank = min(fits, key=lambda r: (costs[r], loads[r], r))
            else:
                rank = min(range(size), key=lambda r: (loads[r] + costs[r], r))
            owners[graph] = rank
            loads[rank] += costs[rank]
            keys[rank].add(key)
        return owners, loads


def load_cost_profile(path: str | Path) -> CostProfile:
    """The profile a JSON file describes."""
    profile = json.loads(Path(path).read_text())
    if profile.get("schema_version") not in _SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported shared compile cost profile: {path}")
    return CostProfile(
        default_seconds=_seconds(profile["default_seconds"], "default_seconds"),
        families={
            str(family): _seconds(seconds, family)
            for family, seconds in profile["families"].items()
        },
        graphs={
            str(graph): _seconds(seconds, graph)
            for graph, seconds in profile.get("graphs", {}).items()
        },
        repeat_fraction=min(
            1.0, _seconds(profile.get("repeat_fraction", 1.0), "repeat_fraction")
        ),
    )


def profile_from_env() -> tuple[CostProfile | None, str]:
    """The configured profile (a path, or the name of a packaged profile) and
    its content digest.

    ``(None, "")`` when unset: round-robin owners. Ranks compare the digest
    because they assign owners independently.
    """
    path = tpu_envs.SGLANG_TPU_SHARED_COMPILE_COST_PROFILE.get().strip()
    if not path:
        return None, ""
    if not Path(path).exists() and (PACKAGED_DIR / path).exists():
        path = str(PACKAGED_DIR / path)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return load_cost_profile(path), digest


def build_cost_profile(lines: Iterable[str]) -> dict:
    """A profile from startup log lines (one or more boots, in log order): each
    family's and each graph's mean first-touch trace seconds, and the median
    cost of a graph whose affinity key its owner had already traced, relative
    to that key's first-touch cost."""
    families: dict[str, list[float]] = {}
    graphs: dict[str, list[float]] = {}
    repeats: list[tuple[str, float]] = []  # (affinity key, seconds)
    seen_graphs: set[str] = set()
    traced: dict[int, set[str]] = {}  # owner -> affinity keys traced this boot
    for line in lines:
        match = _TRACE_LINE.search(line)
        if match is None:
            continue
        owner, seconds, graph = (
            int(match.group(1)),
            float(match.group(2)),
            match.group(3),
        )
        if graph in seen_graphs:  # graph names are unique within one boot
            seen_graphs.clear()
            traced.clear()
        seen_graphs.add(graph)
        if not math.isfinite(seconds):
            continue
        families.setdefault(graph_family(graph), []).append(seconds)
        key = affinity_key(graph)
        if key in traced.setdefault(owner, set()):
            repeats.append((key, seconds))
        else:
            graphs.setdefault(graph, []).append(seconds)
            traced[owner].add(key)
    if not families:
        raise ValueError("no [shared_compile] trace lines in the logs")
    every = [seconds for values in families.values() for seconds in values]
    mean = lambda values: round(statistics.fmean(values), 3)  # noqa: E731
    first: dict[str, list[float]] = {}
    for graph, values in graphs.items():
        first.setdefault(affinity_key(graph), []).extend(values)
    ratios = [s / statistics.fmean(first[k]) for k, s in repeats if first.get(k)]
    return {
        "schema_version": SCHEMA_VERSION,
        "default_seconds": round(statistics.median(every), 3),
        "families": {family: mean(v) for family, v in sorted(families.items())},
        "graphs": {graph: mean(v) for graph, v in sorted(graphs.items())},
        "repeat_fraction": round(min(1.0, statistics.median(ratios)), 3)
        if ratios
        else 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("logs", nargs="+", type=Path, help="startup logs (rank 0)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    lines = (
        line
        for log in args.logs
        for line in log.read_text(errors="replace").splitlines()
    )
    profile = build_cost_profile(lines)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
