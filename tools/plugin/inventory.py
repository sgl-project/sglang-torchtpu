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
HOOKS_MODULE = "sglang_tpu.hooks"
PATCHES_SOURCE = "sglang_tpu/patches.py"

#: The function init_backend() calls; every function it calls is one patch.
PATCH_ENTRY = "apply_tpu_patches"


class Patch(NamedTuple):
    name: str
    summary: str
    #: Dotted attributes the patch assigns, as resolved from its imports.
    targets: tuple[str, ...]


class Hook(NamedTuple):
    #: Dotted path the hook wraps, as registered.
    target: str
    hook: str
    hook_type: str
    #: Plugin file the entry is written in.
    source: str = HOOKS_SOURCE


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


def _string_value(node: ast.expr, names: dict[str, str]) -> Optional[str]:
    """A string built from literals, f-strings, ``+`` and string constants in ``names``."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left, names)
        right = _string_value(node.right, names)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if value.conversion != -1 or value.format_spec is not None:
                    return None
                value = value.value
            part = _string_value(value, names)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _string_names(body: list[ast.stmt], base: dict[str, str]) -> dict[str, str]:
    """``name = <string>`` bindings among ``body``'s statements, on top of ``base``."""
    names = dict(base)
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            value = _string_value(stmt.value, names)
            if value is not None:
                names[stmt.targets[0].id] = value
    return names


def _is_hook_entry(node: ast.AST) -> bool:
    """A ``(target, hook, HookType.X)`` triple."""
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == 3
        and isinstance(node.elts[2], ast.Attribute)
        and isinstance(node.elts[2].value, ast.Name)
        and node.elts[2].value.id == "HookType"
    )


def _hook_entries(scope: ast.AST, names: dict[str, str], source: str) -> list[Hook]:
    entries = sorted(
        (node for node in ast.walk(scope) if _is_hook_entry(node)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    hooks = []
    for entry in entries:
        target, hook, hook_type = entry.elts
        resolved = _string_value(target, names)
        hooks.append(
            Hook(
                target=resolved if resolved is not None else ast.unparse(target),
                hook=ast.unparse(hook),
                hook_type=ast.unparse(hook_type),
                source=source,
            )
        )
    return hooks


def _plugin_module_path(repo_root: str, module: str) -> str:
    base = module.replace(".", "/")
    for relpath in (f"{base}.py", f"{base}/__init__.py"):
        if os.path.exists(os.path.join(repo_root, relpath)):
            return relpath
    raise ValueError(f"no source file for plugin module {module}")


def _provider_hooks(repo_root: str, qualname: str) -> list[Hook]:
    """The entries a provider such as ``sglang_tpu.hooks.dsv4.hook_specs`` returns.

    Entries behind a runtime condition (an env flag) are counted too: the code
    that would register them is debt whether or not a deployment turns it on.
    """
    module, _, func_name = qualname.rpartition(".")
    relpath = _plugin_module_path(repo_root, module)
    tree = _module_tree(repo_root, relpath)
    for func in tree.body:
        if isinstance(func, ast.FunctionDef) and func.name == func_name:
            names = _string_names(func.body, _string_names(tree.body, {}))
            return _hook_entries(func, names, relpath)
    raise ValueError(f"hook provider {qualname} not found in {relpath}")


def _hooks_list_value(node: ast.stmt) -> Optional[ast.expr]:
    """The value of ``HOOKS = [...]`` / ``HOOKS: list = [...]``."""
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return None
    if any(isinstance(t, ast.Name) and t.id == "HOOKS" for t in targets):
        return node.value
    return None


def _hooks_addition(node: ast.stmt) -> Optional[ast.expr]:
    """The argument of ``HOOKS.extend(x)`` / ``HOOKS.append(x)`` / ``HOOKS += x``."""
    if (
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "HOOKS"
    ):
        return node.value
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in ("extend", "append")
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "HOOKS"
        and len(node.value.args) == 1
    ):
        return node.value.args[0]
    return None


def collect_hooks(repo_root: str) -> list[Hook]:
    """``HOOKS``: the literal list plus what each ``HOOKS.extend(provider())`` adds."""
    module = _module_tree(repo_root, HOOKS_SOURCE)
    aliases = _import_aliases(module)
    names = _string_names(module.body, {})
    hooks: Optional[list[Hook]] = None
    for node in module.body:
        value = _hooks_list_value(node)
        if value is not None:
            hooks = _hook_entries(value, names, HOOKS_SOURCE)
            continue
        added = _hooks_addition(node)
        if hooks is None or added is None:
            continue
        if isinstance(added, ast.Call) and isinstance(added.func, ast.Name):
            local = f"{HOOKS_MODULE}.{added.func.id}"
            hooks += _provider_hooks(repo_root, aliases.get(added.func.id, local))
        else:
            hooks += _hook_entries(added, names, HOOKS_SOURCE)
    if hooks is None:
        raise ValueError(f"no HOOKS list in {HOOKS_SOURCE}")
    return hooks


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
        lines += [f"| `{h.target}` | `{h.hook}` | {h.hook_type} |" for h in hooks]
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
