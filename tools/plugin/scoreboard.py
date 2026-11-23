# SPDX-License-Identifier: Apache-2.0
"""Print the plugin migration scoreboard: every debt counter in one table.

Every number trends to zero. Parsed, not imported.

The fork-only rows compare against the upstream commit pinned in
tools/plugin/upstream_pin.txt (fork_only_api.py); they read "n/a" when that
commit is not in the local object store and no --upstream-root is given.

Usage:
    python tools/plugin/scoreboard.py          # markdown table
    python tools/plugin/scoreboard.py --json   # machine-readable
    python tools/plugin/scoreboard.py --upstream-root ~/sglang
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Optional

_TOOLS = os.path.dirname(os.path.abspath(__file__))


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(
        f"_scoreboard_{filename[:-3]}", os.path.join(_TOOLS, filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect(
    repo_root: str, upstream_root: Optional[str] = None
) -> dict[str, Optional[int]]:
    debt_tool = _load("check_tpu_branches.py")
    gate_tool = _load("gate_audit.py")
    inventory_tool = _load("inventory.py")
    api_tool = _load("fork_only_api.py")

    debt = debt_tool.collect(repo_root)
    patches = inventory_tool.collect_patches(repo_root)
    shims = 0
    for path in debt_tool.SHIM_PATHS:
        full = os.path.join(repo_root, path)
        if os.path.isfile(full):
            shims += 1
        elif os.path.isdir(full):
            shims += sum(
                name.endswith(".py") for _, _, names in os.walk(full) for name in names
            )
    try:
        api = api_tool.collect(
            repo_root, api_tool.upstream_tree(repo_root, upstream_root)
        )
        fork_only = [len(api.modules), len(api.symbols), len(api.platform_methods)]
    except api_tool.UpstreamUnavailable as exc:
        print(f"fork-only counters skipped: {exc}", file=sys.stderr)
        fork_only = [None, None, None]
    return {
        "is_tpu references in core": sum(debt.branches.values()),
        "core files with is_tpu": len(debt.branches),
        "sglang_tpu imports in core": sum(debt.imports.values()),
        "core files importing sglang_tpu": len(debt.imports),
        "TPU-named modules in core": len(debt.modules),
        "is_out_of_tree() gates": len(gate_tool.audit_tree(repo_root)),
        "hooks": len(inventory_tool.collect_hooks(repo_root)),
        "runtime patches": len(patches),
        "core attributes patched": sum(
            t.startswith("sglang.") for p in patches for t in p.targets
        ),
        "re-export shim files": shims,
        "fork-only core modules imported by sglang_tpu": fork_only[0],
        "fork-only core symbols imported by sglang_tpu": fork_only[1],
        "fork-only platform methods sglang_tpu overrides": fork_only[2],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(_TOOLS)))
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--upstream-root",
        help="an sgl-project/sglang checkout for the fork-only rows, instead of the pin",
    )
    args = parser.parse_args(argv)

    scores = collect(args.repo_root, args.upstream_root)
    if args.json:
        print(json.dumps(scores, indent=2))
        return 0
    print("| Debt counter | Count |\n|---|---|")
    for name, value in scores.items():
        print(f"| {name} | {'n/a' if value is None else value} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
