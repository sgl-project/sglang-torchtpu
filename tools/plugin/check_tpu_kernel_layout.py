# SPDX-License-Identifier: Apache-2.0
"""Ratchet on the TPU kernel layout: kernels stay self-contained in sglang_tpu/kernels.

Three rules, each with a baseline of known exceptions that may only shrink:

* Pallas kernels (``pallas_call``, ``pl.kernel(``, ``core_map``) are launched only
  under ``sglang_tpu/kernels/``.
* A kernel module imports no SGLang code but its own group, the single-file
  helpers at the top of ``sglang_tpu/kernels/`` (``vmem_limit``), and the shared
  helpers (``sglang_tpu.envs``, ``pallas_export``, ``pallas_jax_op``,
  ``_logging``): no ``sglang.srt``, no other ``sglang_tpu``, no other group. A
  group is ``ops/<group>/``, or any other subpackage of ``sglang_tpu/kernels/``
  (such as ``deepseek_v41/``, not yet split into ``ops/``).
* Tests under ``test/registered/tpu/kernels/`` follow the same rule for the group
  their path names, plus ``sglang.test``.

The shared helpers are held to the same rule, so their outside imports are in the
baseline too. With these holding, a kernel test can only break when its group, the
shared helpers, a baselined outside import, the test itself, or the toolchain
changes; CI can select kernel tests by group path. An import-time break anywhere in
``sglang`` still breaks every test, so the always-run suites catch it. Runtime integration (custom-op registration, layers, backends) lives outside
``sglang_tpu/kernels/`` and calls in.

Usage:
    python tools/plugin/check_tpu_kernel_layout.py                  # check against baseline
    python tools/plugin/check_tpu_kernel_layout.py --write-baseline # record current exceptions
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings
from typing import NamedTuple, Optional

warnings.simplefilter("ignore", SyntaxWarning)

#: Paths are relative to the repo root throughout, with forward slashes.
BASELINE_PATH = os.path.join("tools", "plugin", "tpu_kernel_layout_baseline.json")
PLUGIN_ROOT = "sglang_tpu"
KERNEL_ROOT = "sglang_tpu/kernels/"
KERNEL_TEST_ROOT = "test/registered/tpu/kernels/"

#: Shared helpers kernels may import besides the kernel package itself.
_KERNEL_HELPERS = (
    "sglang_tpu.envs",
    "sglang_tpu.pallas_export",
    "sglang_tpu.pallas_jax_op",
    "sglang_tpu._logging",
)
_HELPERS_RE = re.compile(
    r"^(" + "|".join(re.escape(m) for m in _KERNEL_HELPERS) + r")(\.|$)"
)
_TEST_HELPERS_RE = re.compile(r"^sglang\.test(\.|$)")
#: TensorCore kernels launch through pallas_call, SparseCore ones through
#: pl.kernel(mesh=...) or core_map.
_KERNEL_LAUNCH = re.compile(r"\b(pallas_call|core_map)\b|\bpl\.kernel\(")


def _group(parts: list[str], root: str) -> Optional[str]:
    """Group of a path below ``sglang_tpu/kernels`` given as components, or None
    for the package itself, the ``ops`` namespace, and single-file helpers."""
    if not parts:
        return None
    if parts[0] == "ops":
        return (
            f"ops/{parts[1]}"
            if len(parts) > 2
            or (
                len(parts) == 2
                and os.path.isdir(os.path.join(root, KERNEL_ROOT, "ops", parts[1]))
            )
            else None
        )
    return (
        parts[0] if os.path.isdir(os.path.join(root, KERNEL_ROOT, parts[0])) else None
    )


def _file_group(rel: str, root: str) -> Optional[str]:
    for prefix in (KERNEL_ROOT, KERNEL_TEST_ROOT):
        if rel.startswith(prefix):
            return _group(rel[len(prefix) :].split("/")[:-1], root)
    return None


def _allowed(module: str, group: Optional[str], is_test: bool, root: str) -> bool:
    """Whether a file of kernel ``group`` may import ``module``."""
    if _HELPERS_RE.search(module) or (is_test and _TEST_HELPERS_RE.search(module)):
        return True
    parts = module.split(".")
    if parts[:2] != ["sglang_tpu", "kernels"]:
        return False
    imported = _group(parts[2:], root)
    return imported is None or imported == group


class Layout(NamedTuple):
    kernels_outside: list[str]
    kernel_imports: dict[str, list[str]]
    test_imports: dict[str, list[str]]


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _py_files(root: str, sub: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, sub)):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
    return sorted(out)


def sglang_imports(source: str, package: str = "") -> set[str]:
    """``sglang`` / ``sglang_tpu`` modules a file imports, at any depth.

    ``from pkg import name`` yields ``pkg.name``, so importing a submodule by name
    is judged by the submodule. Relative imports resolve against ``package``, the
    dotted package of the file.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - (node.level - 1)]
                module = ".".join(parts + ([module] if module else []))
            names = [f"{module}.{alias.name}" for alias in node.names]
        else:
            continue
        found.update(n for n in names if n.split(".")[0] in ("sglang", PLUGIN_ROOT))
    return found


def _module_of(name: str, root: str) -> str:
    """Longest prefix of ``name`` below the top package that is a module or package;
    the rest are attributes. Unresolvable names are kept whole."""
    parts = name.split(".")
    base = os.path.join(root, "python") if parts[0] == "sglang" else root
    for i in range(len(parts), 1, -1):
        path = os.path.join(base, *parts[:i])
        if os.path.isfile(path + ".py") or os.path.isdir(path):
            return ".".join(parts[:i])
    return name


def collect(root: str) -> Layout:
    kernels_outside = []
    for path in _py_files(root, PLUGIN_ROOT):
        rel = _rel(path, root)
        if rel.startswith(KERNEL_ROOT):
            continue
        with open(path, encoding="utf-8") as handle:
            if _KERNEL_LAUNCH.search(handle.read()):
                kernels_outside.append(rel)

    def disallowed(sub: str, is_test: bool) -> dict[str, list[str]]:
        out = {}
        paths = _py_files(root, sub)
        if not is_test:
            helpers = [
                os.path.join(root, *m.split(".")) + ".py" for m in _KERNEL_HELPERS
            ]
            paths += [h for h in helpers if os.path.isfile(h)]
        for path in paths:
            rel = _rel(path, root)
            group = _file_group(rel, root)
            package = os.path.dirname(rel).replace("/", ".")
            with open(path, encoding="utf-8") as handle:
                imported = sglang_imports(handle.read(), package)
            modules = {_module_of(m, root) for m in imported}
            bad = sorted(m for m in modules if not _allowed(m, group, is_test, root))
            if bad:
                out[rel] = bad
        return out

    return Layout(
        kernels_outside=kernels_outside,
        kernel_imports=disallowed(KERNEL_ROOT, is_test=False),
        test_imports=disallowed(KERNEL_TEST_ROOT, is_test=True),
    )


def load_baseline(root: str) -> Layout:
    path = os.path.join(root, BASELINE_PATH)
    if not os.path.exists(path):
        return Layout([], {}, {})
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return Layout(
        data.get("kernels_outside", []),
        data.get("kernel_imports", {}),
        data.get("test_imports", {}),
    )


def write_baseline(root: str, layout: Layout) -> str:
    path = os.path.join(root, BASELINE_PATH)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(layout._asdict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def compare(current: Layout, baseline: Layout) -> tuple[list[str], list[str]]:
    """Return ``(violations, improvements)`` as human-readable lines."""
    violations, improvements = [], []
    for rel in sorted(set(current.kernels_outside) - set(baseline.kernels_outside)):
        violations.append(
            f"{rel}: launches a Pallas kernel outside {KERNEL_ROOT}. Put it in "
            f"{KERNEL_ROOT}ops/<group>/ and call it from here."
        )
    for rel in sorted(set(baseline.kernels_outside) - set(current.kernels_outside)):
        improvements.append(f"{rel}: no longer defines a kernel outside {KERNEL_ROOT}")

    for name, cur, base, rule in (
        (
            "kernel",
            current.kernel_imports,
            baseline.kernel_imports,
            "kernels import only their own group and the shared helpers",
        ),
        (
            "test",
            current.test_imports,
            baseline.test_imports,
            "kernel tests import only their group, the shared helpers, and sglang.test",
        ),
    ):
        for rel in sorted(set(cur) | set(base)):
            new = sorted(set(cur.get(rel, [])) - set(base.get(rel, [])))
            gone = sorted(set(base.get(rel, [])) - set(cur.get(rel, [])))
            if new:
                violations.append(f"{rel}: imports {', '.join(new)}; {rule}.")
            if gone:
                improvements.append(
                    f"{rel}: dropped {name} import(s) {', '.join(gone)}"
                )
    return violations, improvements


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=_repo_root())
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current exceptions as the new baseline",
    )
    args = parser.parse_args(argv)
    root = os.path.abspath(args.repo_root)

    current = collect(root)
    if args.write_baseline:
        path = write_baseline(root, current)
        print(
            f"wrote {_rel(path, root)}: {len(current.kernels_outside)} kernel file(s) "
            f"outside {KERNEL_ROOT}, {len(current.kernel_imports)} kernel module(s) and "
            f"{len(current.test_imports)} kernel test(s) with outside imports"
        )
        return 0

    violations, improvements = compare(current, load_baseline(root))
    if improvements:
        print("TPU kernel layout improved:")
        for line in improvements:
            print(f"  {line}")
        print(
            "\nRun `python tools/plugin/check_tpu_kernel_layout.py --write-baseline` "
            "to lock the gain in.\n"
        )
    if violations:
        print("TPU kernel layout broken:\n", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nIf the exception is genuinely unavoidable, rerun with --write-baseline "
            "and justify it in the PR description.",
            file=sys.stderr,
        )
        return 1
    print("TPU kernel layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
