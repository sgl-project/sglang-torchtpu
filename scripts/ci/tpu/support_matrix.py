#!/usr/bin/env python3
"""Generate the TPU support matrix from `register_tpu_ci(...)` markers.

The matrix answers "which model / feature runs on tpu7x, in which topology,
and at which CI tier?" and is derived purely from the test registrations, so a
README claim cannot exist without a test behind it. It is *static analysis*
(stdlib only, no TPU, no sglang import): the same AST registry parser that
`run_suite.py` uses (`ut_parse_one_file` in ci_register.py) is loaded by path.

How a registration becomes a matrix cell
    tier      from the suite name: stage-* -> "CI" (PR gate), nightly-tpu-* and
              tpu-integration -> "Nightly", *-ondemand -> "On-demand",
              external-nightly-* -> "Nightly (external)"; `disabled=` wins.
    topology  from the `singlehost` / `multihost` label; if absent, inferred
              from the suite or file name and reported as inferred.
    model     from the one model label (see support_matrix_vocab.json; aliases
              are folded onto the canonical label).
    kind      accuracy / perf / smoke / infra label, if any. `infra` marks
              plumbing tests that have no model; they get their own table.
    features  every remaining label, resolved through the vocabulary.
    A cell lists every tier present among the tests behind it, best first
    (CI > Nightly > Nightly (external) > On-demand > Disabled), with counts.

Unit suites (stage-a-test-tpu-unit) carry no labels by design and are counted
but not placed in the matrix. Every other non-disabled registration is
expected to carry a topology and a model label; `--strict` fails when one
does not, or when a label is outside the vocabulary.

The committed copy lives at SUPPORT_MATRIX.md in the repo root.

Usage:
    python3 scripts/ci/tpu/support_matrix.py --markdown SUPPORT_MATRIX.md
    python3 scripts/ci/tpu/support_matrix.py --strict --check SUPPORT_MATRIX.md
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DEFAULT_VOCAB = os.path.join(HERE, "support_matrix_vocab.json")
REGISTERED_DIR = os.path.join("test", "registered")
CI_REGISTER_REL = os.path.join("python", "sglang", "test", "ci", "ci_register.py")

TIER_CI = "CI"
TIER_NIGHTLY = "Nightly"
TIER_EXTERNAL = "Nightly (external)"
TIER_ONDEMAND = "On-demand"
TIER_DISABLED = "Disabled"
TIER_UNKNOWN = "Unknown"
# Higher is better; the matrix cell shows the best tier among its tests.
TIER_RANK = {
    TIER_CI: 5,
    TIER_NIGHTLY: 4,
    TIER_EXTERNAL: 3,
    TIER_ONDEMAND: 2,
    TIER_DISABLED: 1,
    TIER_UNKNOWN: 0,
}
UNIT_SUITE_RE = re.compile(r"^stage-a-test-tpu")
# Tests of launch/cluster plumbing rather than a model; listed separately.
KIND_INFRA = "infra"
SINGLE = "singlehost"
MULTI = "multihost"


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def load_ci_register(repo_root: str):
    """Import ci_register.py by path (importing `sglang` would boot the runtime)."""
    path = os.path.join(repo_root, CI_REGISTER_REL)
    spec = importlib.util.spec_from_file_location(
        "tpu_support_matrix_ci_register", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations resolve their module at exec time.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


def collect_tpu_registrations(repo_root: str) -> Tuple[List[dict], List[str]]:
    """Return ([{path, suite, nightly, disabled, labels, est_time}], parse_failures)."""
    ci_register = load_ci_register(repo_root)
    tpu = ci_register.HWBackend.TPU
    root = os.path.join(repo_root, REGISTERED_DIR)
    found, failures = [], []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        try:
            registries, _ = ci_register.ut_parse_one_file(path)
        except Exception as exc:  # SyntaxError, ValueError from the visitor, ...
            failures.append(f"{os.path.relpath(path, repo_root)}: {exc}")
            continue
        for reg in registries:
            if reg.backend is not tpu:
                continue
            found.append(
                {
                    "path": os.path.relpath(path, repo_root).replace(os.sep, "/"),
                    "suite": reg.effective_suite,
                    "nightly": bool(reg.nightly),
                    "disabled": reg.disabled,
                    "labels": list(reg.labels),
                    "est_time": reg.est_time,
                }
            )
    return found, failures


# ---------------------------------------------------------------------------
# Vocabulary and classification
# ---------------------------------------------------------------------------


class Vocab:
    def __init__(self, data: dict):
        self.topologies: Dict[str, str] = dict(data.get("topologies", {}))
        self.kinds: Dict[str, str] = dict(data.get("kinds", {}))
        self.models: Dict[str, str] = {}
        self.features: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        for target, key in ((self.models, "models"), (self.features, "features")):
            for label, spec in data.get(key, {}).items():
                if isinstance(spec, str):
                    target[label] = spec
                    continue
                target[label] = spec.get("name", label)
                for alias in spec.get("aliases", []):
                    self.aliases[alias] = label
        overlap = (
            set(self.topologies) & set(self.models)
            | set(self.models) & set(self.features)
            | set(self.kinds) & set(self.features)
            | set(self.aliases) & (set(self.models) | set(self.features))
        )
        if overlap:
            raise ValueError(
                f"vocabulary labels in more than one class: {sorted(overlap)}"
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


def tier_for(suite: Optional[str], disabled: Optional[str]) -> str:
    if disabled is not None:
        return TIER_DISABLED
    if not suite:
        return TIER_UNKNOWN
    if suite.startswith("stage-"):
        return TIER_CI
    if suite.endswith("-ondemand"):
        return TIER_ONDEMAND
    if suite.startswith("external-nightly-"):
        return TIER_EXTERNAL
    if suite.startswith("nightly-tpu-") or suite == "tpu-integration":
        return TIER_NIGHTLY
    return TIER_UNKNOWN


_MULTI_HINT = re.compile(r"(multihost|2x2x2|2x2x4|2p2d|tp16|tp32)")
_SINGLE_HINT = re.compile(r"(singlehost|1host|4-v7)")


def infer_topology(suite: Optional[str], path: str) -> Optional[str]:
    """Best-effort topology when no label says so. Order: suite, then file name."""
    for text in (suite or "", os.path.basename(path)):
        if _MULTI_HINT.search(text):
            return MULTI
        if _SINGLE_HINT.search(text):
            return SINGLE
    return None


def classify(reg: dict, vocab: Vocab) -> dict:
    """Attach tier/topology/model/kind/features/problems to one registration."""
    out = dict(reg)
    out["tier"] = tier_for(reg["suite"], reg["disabled"])
    out["unit"] = bool(reg["suite"] and UNIT_SUITE_RE.match(reg["suite"]))
    topology, models, kinds, features, unknown = None, [], [], [], []
    for raw in reg["labels"]:
        label = vocab.aliases.get(raw, raw)
        if label in vocab.topologies:
            topology = label
        elif label in vocab.models:
            models.append(label)
        elif label in vocab.kinds:
            kinds.append(label)
        elif label in vocab.features:
            features.append(label)
        else:
            unknown.append(raw)
    out["topology_source"] = "label" if topology else None
    if topology is None:
        topology = infer_topology(reg["suite"], reg["path"])
        out["topology_source"] = "inferred" if topology else None
    out["topology"] = topology
    out["model"] = models[0] if models else None
    out["kind"] = kinds[0] if kinds else None
    out["infra"] = KIND_INFRA in kinds
    out["features"] = sorted(set(features))
    out["unknown_labels"] = unknown

    problems = []
    if out["tier"] == TIER_UNKNOWN:
        problems.append(f"unrecognized suite {reg['suite']!r}")
    if unknown:
        problems.append(f"labels not in vocabulary: {unknown}")
    if len(models) > 1:
        problems.append(f"more than one model label: {models}")
    if not out["unit"] and reg["disabled"] is None:
        if out["model"] is None and not out["infra"]:
            problems.append("no model label")
        if out["topology_source"] != "label":
            problems.append(
                "no topology label"
                + (f" (inferred {topology})" if topology else " and none inferable")
            )
    out["problems"] = problems
    return out


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


def _best(tests: List[dict]) -> Optional[dict]:
    """{tier, count, by_tier} for a cell, or None when nothing backs it.

    `tier` is the best tier present; `by_tier` carries the per-tier counts,
    best first, so a single CI test next to twenty nightly ones is not read
    as twenty CI tests.
    """
    if not tests:
        return None
    by_tier: Dict[str, int] = {}
    for t in sorted(tests, key=lambda t: -TIER_RANK[t["tier"]]):
        by_tier[t["tier"]] = by_tier.get(t["tier"], 0) + 1
    return {"tier": next(iter(by_tier)), "count": len(tests), "by_tier": by_tier}


def build_matrix(
    registrations: List[dict],
    vocab: Vocab,
    parse_failures: List[str],
    commit: Optional[str],
) -> dict:
    tests = [classify(r, vocab) for r in registrations]
    placed = [
        t
        for t in tests
        if not t["unit"] and not t["infra"] and t["model"] and t["topology"]
    ]
    infra = sorted((t for t in tests if t["infra"]), key=lambda t: t["path"])
    topo_keys = list(vocab.topologies)

    models: Dict[str, dict] = {}
    for label in sorted(set(t["model"] for t in placed)):
        mine = [t for t in placed if t["model"] == label]
        models[label] = {
            "name": vocab.models[label],
            "cells": {
                topo: _best([t for t in mine if t["topology"] == topo])
                for topo in topo_keys
            },
            "kinds": sorted(set(t["kind"] for t in mine if t["kind"])),
            "features": sorted(set(f for t in mine for f in t["features"])),
            "tests": sorted(t["path"] for t in mine),
        }

    features: Dict[str, dict] = {}
    for label in sorted(set(f for t in placed for f in t["features"])):
        mine = [t for t in placed if label in t["features"]]
        features[label] = {
            "name": vocab.features[label],
            "cells": {
                topo: _best([t for t in mine if t["topology"] == topo])
                for topo in topo_keys
            },
            "models": sorted(set(t["model"] for t in mine)),
            "tests": sorted(t["path"] for t in mine),
        }

    non_unit = [t for t in tests if not t["unit"]]
    summary = {
        "registrations": len(tests),
        "unit": len(tests) - len(non_unit),
        "placed": len(placed),
        "infra": len(infra),
        "disabled": len([t for t in non_unit if t["disabled"] is not None]),
        "with_problems": len([t for t in tests if t["problems"]]),
        "parse_failures": len(parse_failures),
    }
    return {
        "commit": commit,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "topologies": {k: vocab.topologies[k] for k in topo_keys},
        "summary": summary,
        "models": models,
        "features": features,
        "infra": [t["path"] for t in infra],
        "tests": sorted(tests, key=lambda t: t["path"]),
        "parse_failures": parse_failures,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(cell: Optional[dict]) -> str:
    """Every tier present, best first, with its count; the count is shown only when > 1."""
    if cell is None:
        return "—"
    return " · ".join(
        f"{tier} ×{n}" if n > 1 else tier for tier, n in cell["by_tier"].items()
    )


def _row(cols: List[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cols) + " |"


def render_markdown(matrix: dict, full: bool = False) -> str:
    topo = matrix["topologies"]
    topo_keys = list(topo)
    s = matrix["summary"]
    out: List[str] = []
    out.append("# TPU support matrix")
    out.append("")
    out.append(
        "Generated by `scripts/ci/tpu/support_matrix.py` from the `register_tpu_ci(...)` "
        "markers under `test/registered/`. Do not edit by hand."
    )
    # The commit and timestamp live in the JSON only: stamping them here would
    # make a committed copy stale on every commit and defeat --check.
    out.append("")
    out.append(
        f"Tiers: **{TIER_CI}** = runs on every PR · **{TIER_NIGHTLY}** = nightly on the "
        f"project cluster · **{TIER_EXTERNAL}** = nightly on an external cluster · "
        f"**{TIER_ONDEMAND}** = manual dispatch · **{TIER_DISABLED}** = registered but "
        "switched off · — = no test. A cell lists every tier that backs it, best "
        "first, with the number of tests at that tier."
    )
    out.append("")
    out.append(
        f"{s['registrations']} TPU registrations scanned: {s['unit']} unit (not placed), "
        f"{s['placed']} placed in the matrix, {s['infra']} infrastructure, "
        f"{s['disabled']} disabled, "
        f"{s['with_problems']} with problems, {s['parse_failures']} parse failures."
    )

    out.append("")
    out.append("## Models")
    out.append("")
    out.append(
        _row(["Model"] + [topo[k] for k in topo_keys] + ["Kinds", "Features exercised"])
    )
    out.append(_row(["---"] * (len(topo_keys) + 3)))
    for label, m in matrix["models"].items():
        out.append(
            _row(
                [m["name"]]
                + [_cell(m["cells"][k]) for k in topo_keys]
                + [", ".join(m["kinds"]) or "—", ", ".join(m["features"]) or "—"]
            )
        )

    out.append("")
    out.append("## Features")
    out.append("")
    out.append(_row(["Feature"] + [topo[k] for k in topo_keys] + ["Models"]))
    out.append(_row(["---"] * (len(topo_keys) + 2)))
    for label, f in matrix["features"].items():
        out.append(
            _row(
                [f"{f['name']} (`{label}`)"]
                + [_cell(f["cells"][k]) for k in topo_keys]
                + [", ".join(matrix["models"][m]["name"] for m in f["models"])]
            )
        )

    infra = [t for t in matrix["tests"] if t["infra"]]
    if infra:
        out.append("")
        out.append("## Infrastructure tests")
        out.append("")
        out.append("Launch and cluster plumbing; not tied to one model.")
        out.append("")
        out.append(_row(["Test", "Topology", "Tier", "Features"]))
        out.append(_row(["---"] * 4))
        for t in infra:
            out.append(
                _row(
                    [
                        f"`{t['path']}`",
                        topo.get(t["topology"], "—") if t["topology"] else "—",
                        t["tier"],
                        ", ".join(t["features"]) or "—",
                    ]
                )
            )

    disabled = [
        t for t in matrix["tests"] if t["disabled"] is not None and not t["unit"]
    ]
    if disabled:
        out.append("")
        out.append("## Disabled tests")
        out.append("")
        out.append(
            "Registered but switched off; the reason is in each file's `disabled=`."
        )
        out.append("")
        out.append(_row(["Test", "Suite"]))
        out.append(_row(["---"] * 2))
        for t in disabled:
            out.append(_row([f"`{t['path']}`", t["suite"] or "—"]))

    problems = [t for t in matrix["tests"] if t["problems"]]
    if problems or matrix["parse_failures"]:
        out.append("")
        out.append("## Registrations with problems")
        out.append("")
        out.append(
            "Not placed, or placed on inferred data. Fix the labels to clear a row."
        )
        out.append("")
        out.append(_row(["Test", "Suite", "Problem"]))
        out.append(_row(["---"] * 3))
        for t in problems:
            out.append(
                _row([f"`{t['path']}`", t["suite"] or "—", "; ".join(t["problems"])])
            )
        for failure in matrix["parse_failures"]:
            out.append(_row([f"`{failure.split(':', 1)[0]}`", "—", "parse failure"]))

    if full:
        out.append("")
        out.append("## All tests")
        out.append("")
        out.append(
            _row(["Test", "Suite", "Tier", "Topology", "Model", "Kind", "Features"])
        )
        out.append(_row(["---"] * 7))
        for t in matrix["tests"]:
            if t["unit"]:
                continue
            topo_txt = t["topology"] or "—"
            if t["topology_source"] == "inferred":
                topo_txt += " (inferred)"
            out.append(
                _row(
                    [
                        f"`{t['path']}`",
                        t["suite"] or "—",
                        t["tier"],
                        topo_txt,
                        t["model"] or "—",
                        t["kind"] or "—",
                        ", ".join(t["features"]) or "—",
                    ]
                )
            )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_commit(arg: Optional[str], repo_root: str) -> Optional[str]:
    if arg:
        return arg
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB, help="label vocabulary JSON")
    ap.add_argument(
        "--commit", default=None, help="commit to stamp (default: git HEAD)"
    )
    ap.add_argument("--output", default=None, help="write the JSON matrix here")
    ap.add_argument("--markdown", default=None, help="write the Markdown matrix here")
    ap.add_argument("--full", action="store_true", help="append the per-test appendix")
    ap.add_argument(
        "--check",
        default=None,
        metavar="FILE",
        help="exit 1 if FILE differs from the Markdown this run generates",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 if any registration has problems (missing/unknown labels, bad suite)",
    )
    args = ap.parse_args(argv)

    vocab = Vocab.load(args.vocab)
    registrations, failures = collect_tpu_registrations(args.repo_root)
    matrix = build_matrix(
        registrations, vocab, failures, resolve_commit(args.commit, args.repo_root)
    )
    markdown = render_markdown(matrix, full=args.full)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(markdown)
    if not args.output and not args.markdown and not args.check:
        sys.stdout.write(markdown)

    rc = 0
    problems = [t for t in matrix["tests"] if t["problems"]]
    for t in problems:
        print(f"problem: {t['path']}: {'; '.join(t['problems'])}", file=sys.stderr)
    for failure in failures:
        print(f"parse failure: {failure}", file=sys.stderr)
    if args.strict and (problems or failures):
        print(
            f"--strict: {len(problems)} registrations with problems, {len(failures)} parse failures",
            file=sys.stderr,
        )
        rc = 2

    if args.check:
        try:
            with open(args.check, "r", encoding="utf-8") as f:
                committed = f.read()
        except FileNotFoundError:
            committed = None
        if committed != markdown:
            print(
                f"--check: {args.check} is stale; regenerate with --markdown {args.check}",
                file=sys.stderr,
            )
            rc = rc or 1
        else:
            print(f"--check: {args.check} is up to date", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
