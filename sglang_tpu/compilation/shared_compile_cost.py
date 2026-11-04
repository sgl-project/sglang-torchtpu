"""Trace-cost profiles that balance shared-compilation owners across ranks.

Under --use-spmd-compile one owner rank traces and compiles each shared graph,
and startup waits for the rank with the most tracing. Round-robin owners give
every rank the same number of graphs, but a few graphs (the model forwards)
cost far more than the rest, and adding or dropping any graph moves every later
one to another rank. A profile instead records each graph family's trace cost,
and each graph goes to the rank with the least estimated work so far.

A family is a graph's name with every number replaced by ``N``, so one program's
batch-size and width variants share a cost. Families a profile does not list
cost its ``default_seconds``. Build a profile from the ``[shared_compile]``
lines of a startup log::

    python -m sglang_tpu.compilation.shared_compile_cost LOG... --output PROFILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

from sglang_tpu.envs import envs as tpu_envs

SCHEMA_VERSION = 1
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TRACE_LINE = re.compile(r"\[shared_compile\] owner=\d+ trace=(\S+)s graph=(.+)$")


def graph_family(graph: str) -> str:
    """The family of a shared graph name: its name with numbers replaced by N."""
    return _NUMBER.sub("N", graph.split(":", 1)[0])


def _seconds(value: object, field: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"shared compile cost profile {field} is invalid: {value!r}")
    return seconds


def load_cost_profile(path: str | Path) -> Callable[[str], float]:
    """The per-graph trace-cost estimate a profile JSON describes."""
    profile = json.loads(Path(path).read_text())
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported shared compile cost profile: {path}")
    default = _seconds(profile["default_seconds"], "default_seconds")
    families = {
        str(family): _seconds(seconds, family)
        for family, seconds in profile["families"].items()
    }
    return lambda graph: families.get(graph_family(graph), default)


def estimator_from_env() -> tuple[Callable[[str], float] | None, str]:
    """The configured profile's estimate and content digest.

    ``(None, "")`` when unset: round-robin owners. Ranks compare the digest
    because they assign owners independently.
    """
    path = tpu_envs.SGLANG_TPU_SHARED_COMPILE_COST_PROFILE.get().strip()
    if not path:
        return None, ""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return load_cost_profile(path), digest


def build_cost_profile(lines: Iterable[str]) -> dict:
    """A profile from startup log lines: each family's mean trace seconds."""
    samples: dict[str, list[float]] = {}
    for line in lines:
        match = _TRACE_LINE.search(line)
        if match is None:
            continue
        seconds = float(match.group(1))
        if math.isfinite(seconds):
            samples.setdefault(graph_family(match.group(2)), []).append(seconds)
    if not samples:
        raise ValueError("no [shared_compile] trace lines in the logs")
    every = [seconds for values in samples.values() for seconds in values]
    return {
        "schema_version": SCHEMA_VERSION,
        "default_seconds": round(statistics.median(every), 3),
        "families": {
            family: round(statistics.fmean(values), 3)
            for family, values in sorted(samples.items())
        },
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
