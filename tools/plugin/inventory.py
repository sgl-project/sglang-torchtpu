# SPDX-License-Identifier: Apache-2.0
"""List the plugin's HookRegistry hooks and runtime patches, as markdown on stdout.

Both counts are migration debt and only go down: a hook or a patch stands in
for a seam core does not have yet. Parsed, not imported, so this runs without
the runtime installed.

Usage:
    python tools/plugin/inventory.py
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import NamedTuple, Optional

HOOKS_SOURCE = "sglang_tpu/hooks/__init__.py"
PATCHES_SOURCE = "sglang_tpu/patches.py"

#: The function init_backend() calls; every function it calls is one patch.
PATCH_ENTRY = "apply_tpu_patches"


class Patch(NamedTuple):
    name: str
    summary: str
    #: Dotted attributes the patch assigns, as resolved from its imports.
    targets: tuple[str, ...]


class Hook(NamedTuple):
    target: str
    hook: str
    hook_type: str


def _summary(func: ast.FunctionDef) -> str:
    doc = ast.get_docstring(func) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _import_aliases(
    scope: ast.AST, base: Optional[dict[str, str]] = None
) -> dict[str, str]:
    aliases: dict[str, str] = dict(base or {})
    nodes = scope.body if isinstance(scope, ast.Module) else ast.walk(scope)
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    root = alias.name.split(".", 1)[0]
                    aliases[root] = root
    return aliases


def _dotted(node: ast.expr, aliases: dict[str, str]) -> Optional[str]:
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in aliases:
        return None
    return ".".join([aliases[node.id], *reversed(attrs)])


def _literal_pairs(node: ast.expr) -> list[tuple[str, str]]:
    """(module, attr) pairs of a literal list or tuple of 2-string tuples."""
    try:
        value = ast.literal_eval(node)
    except ValueError:
        return []
    return [tuple(item) for item in value if isinstance(item, tuple) and len(item) == 2]


def _assigned_targets(func: ast.FunctionDef, module: ast.Module) -> list[str]:
    module_consts = _module_consts(module)
    aliases = _import_aliases(func, _import_aliases(module))
    found: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    dotted = _dotted(target, aliases)
                    if dotted:
                        found.append(dotted)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            owner = _dotted(node.args[0], aliases)
            if owner:
                found.append(f"{owner}.{node.args[1].value}")
        elif (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 3
            and isinstance(node.elts[1], ast.Constant)
            and isinstance(node.elts[1].value, str)
        ):
            # A patch table entry: (owner, "attribute", replacement).
            owner = _dotted(node.elts[0], aliases)
            if owner:
                found.append(f"{owner}.{node.elts[1].value}")
        elif (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in module_consts
            and any(
                isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setattr"
                for n in ast.walk(node)
            )
        ):
            # for module_path, name in TARGETS: setattr(import(module_path), name, ...)
            found += [
                ".".join(pair) for pair in _literal_pairs(module_consts[node.iter.id])
            ]
    return found


def _module_tree(repo_root: str, relpath: str) -> ast.Module:
    with open(os.path.join(repo_root, relpath), encoding="utf-8") as f:
        return ast.parse(f.read())


def _module_consts(tree: ast.Module) -> dict[str, ast.expr]:
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            consts[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = node.value
    return consts


def _patch_targets(
    repo_root: str, func: ast.FunctionDef, module: ast.Module
) -> tuple[str, ...]:
    """Targets the function assigns, plus those of plugin functions it delegates to."""
    targets = _assigned_targets(func, module)
    aliases = _import_aliases(func, _import_aliases(module))
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            continue
        owner = aliases.get(node.func.value.id, "")
        relpath = owner.replace(".", "/") + ".py"
        if not owner.startswith("sglang_tpu.") or not os.path.exists(
            os.path.join(repo_root, relpath)
        ):
            continue
        callee_module = _module_tree(repo_root, relpath)
        funcs = {
            f.name: f for f in callee_module.body if isinstance(f, ast.FunctionDef)
        }
        callee = funcs.get(node.func.attr)
        if callee is None:
            continue
        # The delegate and the same-module functions it calls, one level deep.
        scanned = [callee] + [
            funcs[n.func.id]
            for n in ast.walk(callee)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in funcs
        ]
        for scanned_func in scanned:
            targets += _assigned_targets(scanned_func, callee_module)
    return tuple(dict.fromkeys(targets))


def collect_patches(repo_root: str) -> list[Patch]:
    tree = _module_tree(repo_root, PATCHES_SOURCE)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    patches = []
    for stmt in funcs[PATCH_ENTRY].body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id in funcs
        ):
            func = funcs[stmt.value.func.id]
            patches.append(
                Patch(func.name, _summary(func), _patch_targets(repo_root, func, tree))
            )
    return patches


def collect_hooks(repo_root: str) -> list[Hook]:
    for node in _module_tree(repo_root, HOOKS_SOURCE).body:
        value = None
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "HOOKS"
        ):
            value = node.value
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "HOOKS" for t in node.targets
        ):
            value = node.value
        if value is not None:
            return [
                Hook(*(ast.unparse(part) for part in entry.elts))
                for entry in value.elts
            ]
    raise ValueError(f"no HOOKS list in {HOOKS_SOURCE}")


def render_patches(patches: list[Patch]) -> str:
    lines = [
        "## Runtime patches\n",
        f"{len(patches)} patch function(s), applied in order by `{PATCH_ENTRY}()` from",
        "`init_backend()`. Each one overwrites code SGLang or torch_tpu owns; the",
        "count goes down as seams land upstream.\n",
        "| # | Patch | Assigns | What it does |",
        "|---|---|---|---|",
    ]
    for i, patch in enumerate(patches, 1):
        targets = "<br>".join(f"`{t}`" for t in patch.targets) or "(registration)"
        summary = patch.summary.replace("|", "\\|") or "(no docstring)"
        lines.append(f"| {i} | `{patch.name}` | {targets} | {summary} |")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_hooks(hooks: list[Hook]) -> str:
    lines = [
        "## HookRegistry hooks\n",
        f"{len(hooks)} hook(s) in `sglang_tpu.hooks.HOOKS`.\n",
    ]
    if hooks:
        lines += ["| Target | Hook | Type |", "|---|---|---|"]
        lines += [f"| {h.target} | `{h.hook}` | {h.hook_type} |" for h in hooks]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        default=os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
    )
    args = parser.parse_args(argv)
    print(render_hooks(collect_hooks(args.repo_root)))
    print(render_patches(collect_patches(args.repo_root)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
