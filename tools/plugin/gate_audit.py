# SPDX-License-Identifier: Apache-2.0
"""Inventory the ``is_out_of_tree()`` gates in SGLang's core tree.

Under ``PlatformEnum.OOT`` every gate fires, and a gate may call a platform
method the in-tree TPU platform never implemented (the base class raises). This
tool extracts, statically, the platform methods reachable from each gate; the
contract test asserts the OOT platform answers every one of them.

Usage:
    python tools/plugin/gate_audit.py                 # human-readable report
    python tools/plugin/gate_audit.py --json out.json # machine-readable inventory
    python tools/plugin/gate_audit.py --quiet         # summary counts only

Shallow by design: the gate's taken branch plus one level of same-class
``self.<helper>()`` resolution; dynamic dispatch is invisible.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import warnings
from typing import Iterator, NamedTuple, Optional

# Core files are parsed, not imported; their SyntaxWarnings are not this tool's output.
warnings.simplefilter("ignore", SyntaxWarning)

# Receivers whose `.is_out_of_tree()` counts as a platform gate.
_PLATFORM_RECEIVERS = frozenset({"current_platform", "platform", "self.platform"})

_GATE_METHOD = "is_out_of_tree"
_TPU_METHOD = "is_tpu"

#: Relative to the repo root.
CORE_TREE = os.path.join("python", "sglang", "srt")

#: Backend-owned directories; a gate there is not core drift.
_EXCLUDED_DIRS = (
    os.path.join(CORE_TREE, "hardware_backend"),
    os.path.join(CORE_TREE, "platforms"),
)


class Gate(NamedTuple):
    """One ``is_out_of_tree()`` call site and what it reaches. NamedTuple, not
    msgspec.Struct: runnable from a bare checkout with no sglang install."""

    file: str
    line: int
    #: "pure_oot"   -- the condition turns on OOT alone; the flip changes the branch taken.
    #: "oot_or_tpu" -- reads `is_out_of_tree() or is_tpu()`; already true for in-tree TPU,
    #:                 so flipping the enum is a no-op here.
    #: "unscoped"   -- not the test of an `if`/conditional expression (assigned, returned,
    #:                 passed as an argument); no branch window to scan.
    kind: str
    #: Which arm the OOT platform takes: "body", "orelse" (the gate is negated),
    #: or "" for unscoped sites.
    window: str
    #: `current_platform.<name>()` calls reachable from that arm.
    platform_methods: list[str] = []
    #: Same-class `self.<helper>()` calls that contributed methods, for traceability.
    via_helpers: list[str] = []
    source: str = ""


def _iter_python_files(root: str) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _receiver_name(node: ast.expr) -> str:
    """Dotted source name of an attribute call's receiver, best effort."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_receiver_name(node.value)}.{node.attr}"
    return ""


def _is_platform_call(node: ast.AST, method: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and _receiver_name(node.func.value) in _PLATFORM_RECEIVERS
    )


def _contains_platform_call(node: ast.AST, method: str) -> bool:
    return any(_is_platform_call(n, method) for n in ast.walk(node))


def _build_parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _ancestors(node: ast.AST, parents: dict[int, ast.AST]) -> Iterator[ast.AST]:
    current = parents.get(id(node))
    while current is not None:
        yield current
        current = parents.get(id(current))


def _enclosing_class(
    node: ast.AST, parents: dict[int, ast.AST]
) -> Optional[ast.ClassDef]:
    for ancestor in _ancestors(node, parents):
        if isinstance(ancestor, ast.ClassDef):
            return ancestor
    return None


def _gate_context(
    call: ast.Call, parents: dict[int, ast.AST]
) -> tuple[Optional[ast.AST], bool]:
    """Find the ``if`` whose test holds ``call``, and whether it is negated.

    Returns ``(node, negated)`` where ``node`` is the ``ast.If`` / ``ast.IfExp``
    the gate controls, or ``None`` when the call is not used as a condition.
    ``negated`` is True when the call sits under a ``not``, which means the OOT
    platform takes the *other* arm.
    """
    negated = False
    child: ast.AST = call
    for ancestor in _ancestors(call, parents):
        if isinstance(ancestor, ast.UnaryOp) and isinstance(ancestor.op, ast.Not):
            negated = not negated
        elif isinstance(ancestor, (ast.If, ast.IfExp)):
            # Only a gate if we arrived via the condition, not via a branch body.
            return (ancestor, negated) if child is ancestor.test else (None, False)
        elif isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            return None, False
        child = ancestor
    return None, False


def _window_nodes(node: ast.AST, negated: bool) -> list[ast.AST]:
    """The statements the OOT platform executes at this gate."""
    if isinstance(node, ast.IfExp):
        return [node.orelse if negated else node.body]
    assert isinstance(node, ast.If)
    branch = node.orelse if negated else node.body
    return list(branch)


def _collect_platform_methods(nodes: list[ast.AST]) -> list[str]:
    found: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and _receiver_name(sub.func.value) in _PLATFORM_RECEIVERS
            ):
                found.add(sub.func.attr)
    # The gate predicates themselves are not a capability the platform must add.
    return sorted(found - {_GATE_METHOD, _TPU_METHOD})


def _collect_self_calls(nodes: list[ast.AST]) -> list[str]:
    found: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"
            ):
                found.add(sub.func.attr)
    return sorted(found)


def _class_methods(cls: ast.ClassDef) -> dict[str, ast.AST]:
    return {
        stmt.name: stmt
        for stmt in cls.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def audit_file(path: str, rel_path: str) -> list[Gate]:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source, filename=path)
    except (
        SyntaxError
    ) as exc:  # pragma: no cover - a core file failing to parse is CI's problem
        print(f"warning: skipping unparsable {rel_path}: {exc}", file=sys.stderr)
        return []

    parents = _build_parents(tree)
    source_lines = source.splitlines()
    gates: list[Gate] = []

    for node in ast.walk(tree):
        if not _is_platform_call(node, _GATE_METHOD):
            continue

        branch, negated = _gate_context(node, parents)
        if branch is None:
            gates.append(
                Gate(
                    file=rel_path,
                    line=node.lineno,
                    kind="unscoped",
                    window="",
                    source=source_lines[node.lineno - 1].strip(),
                )
            )
            continue

        guarded_by_tpu = _contains_platform_call(branch.test, _TPU_METHOD)
        window = _window_nodes(branch, negated)
        methods = set(_collect_platform_methods(window))

        # One hop through same-class helpers: `_build_oot_dsa_kv_pool` is where
        # `get_dsa_kv_pool_cls` actually lives, and that is the method whose
        # absence breaks DSA models under OOT.
        helpers_used: list[str] = []
        cls = _enclosing_class(branch, parents)
        if cls is not None:
            defined = _class_methods(cls)
            for name in _collect_self_calls(window):
                helper = defined.get(name)
                if helper is None:
                    continue
                helper_methods = _collect_platform_methods([helper])
                if helper_methods:
                    helpers_used.append(name)
                    methods.update(helper_methods)

        gates.append(
            Gate(
                file=rel_path,
                line=branch.lineno,
                kind="oot_or_tpu" if guarded_by_tpu else "pure_oot",
                window="orelse" if negated else "body",
                platform_methods=sorted(methods),
                via_helpers=sorted(helpers_used),
                source=source_lines[branch.lineno - 1].strip(),
            )
        )

    return sorted(gates, key=lambda g: (g.file, g.line))


def audit_tree(repo_root: str) -> list[Gate]:
    """Every OOT gate in the core tree, excluding backend-owned directories."""
    core = os.path.join(repo_root, CORE_TREE)
    if not os.path.isdir(core):
        raise FileNotFoundError(f"core tree not found: {core}")

    excluded = tuple(os.path.join(repo_root, d) for d in _EXCLUDED_DIRS)
    gates: list[Gate] = []
    for path in _iter_python_files(core):
        if path.startswith(excluded):
            continue
        rel = os.path.relpath(path, repo_root)
        gates.extend(audit_file(path, rel))
    return sorted(gates, key=lambda g: (g.file, g.line))


def required_platform_methods(gates: list[Gate]) -> dict[str, list[str]]:
    """Method -> the gate sites that reach it, across every gate kind."""
    reached: dict[str, list[str]] = {}
    for gate in gates:
        for method in gate.platform_methods:
            reached.setdefault(method, []).append(f"{gate.file}:{gate.line}")
    return {name: sorted(sites) for name, sites in sorted(reached.items())}


def build_inventory(repo_root: str) -> dict:
    gates = audit_tree(repo_root)
    pure = [g for g in gates if g.kind == "pure_oot"]
    guarded = [g for g in gates if g.kind == "oot_or_tpu"]
    unscoped = [g for g in gates if g.kind == "unscoped"]
    return {
        "core_tree": CORE_TREE,
        "counts": {
            "total": len(gates),
            "pure_oot": len(pure),
            "oot_or_tpu": len(guarded),
            "unscoped": len(unscoped),
            "files": len({g.file for g in gates}),
        },
        "required_platform_methods": required_platform_methods(gates),
        "required_platform_methods_pure_oot": required_platform_methods(pure),
        "gates": [g._asdict() for g in gates],
    }


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=_repo_root())
    parser.add_argument("--json", dest="json_path", help="write the inventory here")
    parser.add_argument("--quiet", action="store_true", help="summary counts only")
    args = parser.parse_args(argv)

    inventory = build_inventory(args.repo_root)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(inventory, handle, indent=2, sort_keys=False)
            handle.write("\n")
        print(f"wrote {args.json_path}")

    counts = inventory["counts"]
    print(
        f"{counts['total']} is_out_of_tree() gates in {counts['files']} files: "
        f"{counts['pure_oot']} pure-OOT, {counts['oot_or_tpu']} `or is_tpu()`-guarded, "
        f"{counts['unscoped']} unscoped"
    )

    if args.quiet:
        return 0

    print("\nPlatform methods reachable from a gate:")
    for method, sites in inventory["required_platform_methods"].items():
        print(f"  {method}")
        for site in sites:
            print(f"      {site}")

    print(
        "\nPure-OOT gates (these change which branch is taken under PlatformEnum.OOT):"
    )
    for gate in inventory["gates"]:
        if gate["kind"] != "pure_oot":
            continue
        arm = "" if gate["window"] == "body" else f" [negated -> {gate['window']}]"
        print(f"  {gate['file']}:{gate['line']}{arm}")
        print(f"      {gate['source']}")
        if gate["platform_methods"]:
            via = (
                f" (via {', '.join(gate['via_helpers'])})"
                if gate["via_helpers"]
                else ""
            )
            print(f"      calls: {', '.join(gate['platform_methods'])}{via}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
