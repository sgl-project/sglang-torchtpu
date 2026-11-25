# SPDX-License-Identifier: Apache-2.0
"""Ratchet on what sglang_tpu needs from core: hooks, runtime patches, fork-only API.

check_tpu_branches.py ratchets TPU code inside core; this ratchets the other
direction. Each count may only go down:

- hooks: HookRegistry entries, per target module (inventory.py);
- runtime patches: the functions ``apply_tpu_patches()`` runs and the
  attributes each one assigns (inventory.py);
- fork-only core API (fork_only_api.py), the mode-C blockers: core modules and
  symbols the plugin imports, and SRTPlatform / DeviceMixin methods it
  overrides, that the pinned upstream commit lacks.

Pre-commit runs it with --warn-only for now: growth is reported, not blocked.

Usage:
    python tools/plugin/check_plugin_ratchets.py                  # exit 1 on growth
    python tools/plugin/check_plugin_ratchets.py --warn-only      # report growth, exit 0
    python tools/plugin/check_plugin_ratchets.py --write-baseline # record the current numbers
    python tools/plugin/check_plugin_ratchets.py --upstream-root ~/sglang
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
from typing import NamedTuple, Optional

_TOOLS = os.path.dirname(os.path.abspath(__file__))

#: Relative to the repo root.
BASELINE_PATH = os.path.join("tools", "plugin", "plugin_debt_baseline.json")


def _load(filename: str):
    """Import a sibling tools/plugin/ script without putting tools/ on sys.path."""
    name = f"_plugin_ratchets_{filename[:-3]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_TOOLS, filename)
        )
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


inventory = _load("inventory.py")
fork_only_api = _load("fork_only_api.py")


class PluginDebt(NamedTuple):
    """NamedTuple, not msgspec.Struct: runs on the bare pre-commit interpreter."""

    #: target module -> hooks registered on it
    hooks: dict[str, int] = {}
    #: patch function -> attributes it assigns
    patches: dict[str, list[str]] = {}
    #: None when the upstream tree could not be read
    fork_only: Optional[fork_only_api.ForkOnlyApi] = None
    #: the upstream commit ``fork_only`` was measured against
    upstream: str = ""


def hook_module(target: str, repo_root: str) -> str:
    """The module of a dotted hook target: its longest prefix with a source file.

    A target whose module was deleted lands on the closest package that remains.
    """
    parts = target.split(".")
    for end in range(len(parts) - 1, 0, -1):
        base = "/".join(parts[:end])
        for root in (
            "python",
            "",
        ):  # sglang.* lives under python/, sglang_tpu.* at the root
            for rel in (f"{base}.py", f"{base}/__init__.py"):
                if os.path.isfile(os.path.join(repo_root, root, rel)):
                    return ".".join(parts[:end])
    return ".".join(parts[:-1])


def collect(
    repo_root: str, upstream_root: Optional[str] = None
) -> tuple[PluginDebt, str]:
    """The current debt, and why the fork-only counters were skipped ("" if not)."""
    hooks = collections.Counter(
        hook_module(hook.target, repo_root)
        for hook in inventory.collect_hooks(repo_root)
    )
    patches = {
        patch.name: sorted(set(patch.targets))
        for patch in inventory.collect_patches(repo_root)
    }
    fork_only, upstream, skipped = None, "", ""
    try:
        if upstream_root:
            tree, upstream = fork_only_api.disk_tree(upstream_root), upstream_root
        else:
            upstream = fork_only_api.read_pin(repo_root)
            tree = fork_only_api.git_tree(repo_root, upstream)
        fork_only = fork_only_api.collect(repo_root, tree)
    except fork_only_api.UpstreamUnavailable as exc:
        skipped = str(exc)
    debt = PluginDebt(
        hooks=dict(sorted(hooks.items())),
        patches=dict(sorted(patches.items())),
        fork_only=fork_only,
        upstream=upstream,
    )
    return debt, skipped


def load_baseline(repo_root: str) -> PluginDebt:
    path = os.path.join(repo_root, BASELINE_PATH)
    if not os.path.exists(path):
        return PluginDebt(fork_only=fork_only_api.ForkOnlyApi())
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return PluginDebt(
        hooks=dict(payload.get("hooks", {})),
        patches={name: list(t) for name, t in payload.get("patches", {}).items()},
        fork_only=fork_only_api.ForkOnlyApi(**payload.get("fork_only", {})),
        upstream=payload.get("upstream", ""),
    )


def write_baseline(repo_root: str, debt: PluginDebt) -> str:
    path = os.path.join(repo_root, BASELINE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "hooks": debt.hooks,
        "patches": debt.patches,
        "upstream": debt.upstream,
        "fork_only": debt.fork_only._asdict(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


_FORK_ONLY_KINDS = {
    "modules": "fork-only core module",
    "symbols": "fork-only core symbol",
    "platform_methods": "fork-only platform method",
}


def compare(current: PluginDebt, baseline: PluginDebt) -> tuple[list[str], list[str]]:
    """Return ``(violations, improvements)`` as human-readable lines."""
    violations: list[str] = []
    improvements: list[str] = []

    for module in sorted(current.hooks.keys() | baseline.hooks.keys()):
        count, allowed = current.hooks.get(module, 0), baseline.hooks.get(module, 0)
        if count > allowed:
            violations.append(f"{module}: {count} hook(s), baseline {allowed}.")
        elif count < allowed:
            improvements.append(f"{module}: {allowed} -> {count} hook(s)")

    for name in sorted(current.patches.keys() | baseline.patches.keys()):
        if name not in baseline.patches:
            assigns = ", ".join(current.patches[name]) or "no attribute (registration)"
            violations.append(f"new runtime patch {name}(), assigns {assigns}.")
        elif name not in current.patches:
            improvements.append(f"runtime patch {name}() removed")
        else:
            now, before = set(current.patches[name]), set(baseline.patches[name])
            if now - before:
                violations.append(
                    f"runtime patch {name}() now also assigns "
                    f"{', '.join(sorted(now - before))}."
                )
            if before - now:
                improvements.append(
                    f"runtime patch {name}() no longer assigns "
                    f"{', '.join(sorted(before - now))}"
                )

    if current.fork_only is not None and baseline.fork_only is not None:
        now_api, before_api = current.fork_only._asdict(), baseline.fork_only._asdict()
        for field, kind in _FORK_ONLY_KINDS.items():
            now, before = set(now_api[field]), set(before_api[field])
            violations += [f"{item}: new {kind}." for item in sorted(now - before)]
            improvements += [
                f"{item}: no longer a {kind}" for item in sorted(before - now)
            ]
    return violations, improvements


def _summary(debt: PluginDebt) -> str:
    patched = sum(len(targets) for targets in debt.patches.values())
    line = (
        f"{sum(debt.hooks.values())} hook(s) on {len(debt.hooks)} module(s), "
        f"{len(debt.patches)} runtime patch(es) assigning {patched} attribute(s)"
    )
    if debt.fork_only is not None:
        api = debt.fork_only
        # A pinned commit, or the --upstream-root path.
        upstream = debt.upstream[:10] if os.sep not in debt.upstream else debt.upstream
        line += (
            f"; fork-only vs upstream {upstream}: {len(api.modules)} "
            f"module(s), {len(api.symbols)} symbol(s), {len(api.platform_methods)} "
            f"platform method(s)"
        )
    return line


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(_TOOLS))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=_repo_root())
    parser.add_argument(
        "--upstream-root",
        help="compare against this sgl-project/sglang checkout instead of the "
        "pinned commit's git objects",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current numbers as the new ceiling",
    )
    parser.add_argument(
        "--warn-only", action="store_true", help="report growth but exit 0"
    )
    args = parser.parse_args(argv)
    repo_root = os.path.abspath(args.repo_root)

    if args.write_baseline:
        if args.upstream_root:
            parser.error(
                "--write-baseline measures against the pin; drop --upstream-root"
            )
        debt, skipped = collect(repo_root)
        if skipped:
            print(f"cannot write the baseline: {skipped}", file=sys.stderr)
            return 2
        path = write_baseline(repo_root, debt)
        print(f"wrote {os.path.relpath(path, repo_root)}: {_summary(debt)}")
        return 0

    current, skipped = collect(repo_root, args.upstream_root)
    baseline = load_baseline(repo_root)
    violations, improvements = compare(current, baseline)

    for line in improvements:
        print(f"improved: {line}")
    if improvements:
        print(
            "\nPlugin debt went down. Run "
            "`python tools/plugin/check_plugin_ratchets.py --write-baseline` "
            "to lock the gain in.\n"
        )
    if skipped:
        print(f"note: fork-only counters skipped: {skipped}")
    elif not args.upstream_root and current.upstream != baseline.upstream:
        print(
            f"note: the upstream pin moved ({baseline.upstream[:10]} -> "
            f"{current.upstream[:10]}) since the baseline was written; fork-only "
            f"changes below may come from the sync."
        )

    if violations:
        print("Plugin debt increased:\n", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nHooks and runtime patches wrap core code that no seam reaches yet: "
            "prefer an existing platform method, a registry or a resolution hook, "
            "and send the seam upstream. Fork-only core API (absent from upstream "
            "SGLang) keeps the plugin off a stock upstream install. If the addition "
            "is genuinely unavoidable, rerun with --write-baseline and justify it "
            "in the PR description.",
            file=sys.stderr,
        )
        return 0 if args.warn_only else 1

    print(f"Plugin debt check passed: {_summary(current)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
