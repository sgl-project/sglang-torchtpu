#!/usr/bin/env python3
"""
Pre-commit hook: validate CI registry calls under test/registered/.

Reuses ut_parse_one_file() from ci_register.py (AST-based parsing) to match
run_suite.py's collect_tests().
"""

import ast
import glob
import importlib.util
import os
import re
import subprocess
import sys

# Exactly what stage=/runner_config= produces, so a legacy suite= of this shape
# is always expressible the modern way.
_MODERN_SHAPE = re.compile(r"^(.+)-test-(.+)$")

# The only CUDA family still allowed on legacy `suite=`; anything else resolves
# to a suite no workflow invokes and the test silently never runs.
_LEGACY_CUDA_PREFIXES = ("stress",)

<<<<<<< HEAD
_TEST_KINDS = {"unit", "kernel", "e2e", "accuracy", "perf", "stress"}

# Flat vendor trees. Vendor-only coverage fits no kind above: no XPU/NPU suite
# carries the `-kernel-` infix `kernel` needs, and these launch device work.
_VENDOR_DIRS = {"amd", "mlx", "musa", "npu", "xpu"}
=======
_KERNEL_LAYOUT = "test/registered/kernels/{ops,benchmark}/<group>/"
>>>>>>> upstream/main


def _defines_testcase(tree: ast.AST) -> bool:
    """True if the file defines unittest classes, statically or via type()."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any("TestCase" in ast.unparse(b) for b in node.bases):
                return True
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "type"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Tuple)
                and any("TestCase" in ast.unparse(e) for e in node.args[1].elts)
            ):
                return True
    return False


def _main_runs_tests(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if not (
            isinstance(stmt, ast.If)
            and ast.unparse(stmt.test).replace("'", '"') == '__name__ == "__main__"'
        ):
            continue
        body = ast.unparse(ast.Module(body=stmt.body, type_ignores=[]))
        if "unittest.main" in body or "pytest.main" in body:
            return True
    return False


<<<<<<< HEAD
def _git_lines(*args: str) -> list[str] | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def _registered_destinations(lines: list[str]) -> set[str]:
    selected = set()
    for line in lines:
        destination = line.split("\t")[-1]
        if destination.startswith("test/registered/") and destination.endswith(".py"):
            selected.add(destination)
    return selected


def _changed_registered_files() -> set[str]:
    """Return locally added, copied, or renamed registered-test destinations."""

    lines = _git_lines("diff", "--cached", "--name-status", "--diff-filter=ACR")
    if lines:
        return _registered_destinations(lines)

    base_ref = os.environ.get("GITHUB_BASE_REF", "main")
    for candidate in (f"origin/{base_ref}", base_ref):
        if _git_lines("rev-parse", "--verify", candidate) is None:
            continue
        merge_base = _git_lines("merge-base", candidate, "HEAD")
        if not merge_base:
            continue
        lines = _git_lines(
            "diff",
            "--name-status",
            "--diff-filter=ACR",
            merge_base[0],
            "HEAD",
        )
        selected = _registered_destinations(lines or [])

        # A sync PR merges another repository's history. Files already present
        # in that merged parent are imported, not newly admitted by this branch,
        # so leave them to the source repository's incremental taxonomy ratchet.
        merges = _git_lines(
            "rev-list",
            "--first-parent",
            "--merges",
            "--parents",
            f"{merge_base[0]}..HEAD",
        )
        for merge in merges or []:
            for merged_parent in merge.split()[2:]:
                imported = _git_lines(
                    "ls-tree",
                    "-r",
                    "--name-only",
                    merged_parent,
                    "--",
                    "test/registered",
                )
                selected.difference_update(imported or [])
        return selected

    return set()


=======
>>>>>>> upstream/main
def _contains_call(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
        for node in ast.walk(tree)
    )


<<<<<<< HEAD
def taxonomy_errors(path: str, registries: list, tree: ast.AST) -> list[str]:
    """Validate the kind/subsystem contract for a newly admitted path."""

    parts = path.split("/")
    relative_parts = parts[2:] if parts[:2] == ["test", "registered"] else []
    if relative_parts and relative_parts[0] in _VENDOR_DIRS:
        return []
    if len(relative_parts) < 3 or relative_parts[0] not in _TEST_KINDS:
        return [
            f"{path}: registered tests must live under "
            "test/registered/<kind>/<subsystem>/; kind must be one of "
            + ", ".join(sorted(_TEST_KINDS))
        ]

    kind = relative_parts[0]
    errors = []
    if kind == "unit":
        invalid = [
            r
            for r in registries
            if r.backend.name != "CPU" and "-unit-" not in (r.effective_suite or "")
        ]
        if invalid:
            errors.append(f"{path}: unit tests must use CPU or dedicated unit suites")
        if any(r.est_time > 60 for r in registries):
            errors.append(f"{path}: unit test est_time must be <= 60 seconds")
        if _contains_call(tree, "popen_launch_server"):
            errors.append(f"{path}: unit tests may not launch a server")
    elif kind == "kernel":
        if any("-kernel-" not in (r.effective_suite or "") for r in registries):
            errors.append(f"{path}: kernel tests must use a *-kernel-* suite")
    elif kind in {"accuracy", "perf"}:
        invalid = [
            r
            for r in registries
            if not (r.effective_suite or "").startswith(("nightly-", "weekly-"))
        ]
        if invalid:
            errors.append(f"{path}: {kind} tests must use nightly/weekly suites")
    elif kind == "stress":
        invalid = [
            r
            for r in registries
            if (r.effective_suite or "") != "stress"
            and not (r.effective_suite or "").startswith("weekly-")
        ]
        if invalid:
            errors.append(f"{path}: stress tests must use stress/weekly suites")
    return errors
=======
def taxonomy_errors(path: str, tree: ast.AST) -> list[str]:
    parts = path.split("/")
    if parts[:2] != ["test", "registered"] or len(parts) < 3:
        return []
    relative_parts = parts[2:]
    root = relative_parts[0]

    if root in ("kernel", "kernels"):
        canonical = (
            root == "kernels"
            and len(relative_parts) >= 4
            and relative_parts[1] in ("ops", "benchmark")
        )
        return (
            [] if canonical else [f"{path}: kernel tests live under {_KERNEL_LAYOUT}"]
        )

    if root != "unit":
        return []
    if _contains_call(tree, "popen_launch_server"):
        return [f"{path}: unit tests may not launch a server"]
    return []
>>>>>>> upstream/main


def main() -> int:
    # Import ci_register directly to avoid pulling in all of sglang
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join("python", "sglang", "test", "ci", "ci_register.py"),
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)
    cuda = ci_register.HWBackend.CUDA

    # Same exclusion as run_suite.py: pytest+package structure files.
    files = sorted(
        f
        for f in glob.glob("test/registered/**/*.py", recursive=True)
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    )
    if not files:
        return 0

    missing = []
    legacy_shape = []  # (file, suite, stage, runner_config) -- has a -test- split
    non_dispatchable = []  # (file, suite) -- legacy CUDA suite no workflow invokes
    dead_tests = []  # (file) -- TestCase classes that `python3 file.py` never runs
    taxonomy_violations = []
<<<<<<< HEAD
    changed_files = _changed_registered_files()
=======
>>>>>>> upstream/main
    for f in files:
        try:
            registries, _has_main_entry = ci_register.ut_parse_one_file(f)
        except Exception:
            continue
        if len(registries) == 0:
            missing.append(f)
            continue
        # TestCase classes are dead unless __main__ runs them; CI runs the
        # registered file as `python3 file.py`.
        with open(f, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=f)
<<<<<<< HEAD
        if f in changed_files:
            taxonomy_violations.extend(taxonomy_errors(f, registries, tree))
=======
        taxonomy_violations.extend(taxonomy_errors(f, tree))
>>>>>>> upstream/main
        if _defines_testcase(tree) and not _main_runs_tests(tree):
            dead_tests.append(f)
        for r in registries:
            # Pure legacy form on a CUDA registry: suite set, stage/runner unset.
            if not (
                r.backend == cuda
                and r.suite is not None
                and r.stage is None
                and r.runner_config is None
            ):
                continue
            if r.suite.split("-", 1)[0] in _LEGACY_CUDA_PREFIXES:
                continue
            m = _MODERN_SHAPE.match(r.suite)
            if m:
                legacy_shape.append((f, r.suite, m.group(1), m.group(2)))
            else:
                non_dispatchable.append((f, r.suite))

    exit_code = 0
    if missing:
        print("ERROR: Files in test/registered/ missing CI registry call:")
        print("  Move manual-only tests to test/manual/.\n")
        for f in missing:
            print(f"  {f}")
        print()
        exit_code = 1
    if legacy_shape:
        print(
            "ERROR: CUDA test(s) register a `{stage}-test-{runner_config}`-shaped "
            'suite via the legacy `suite="..."` form, which is not dispatchable '
            "via /rerun-test. Switch to the modern `stage=`/`runner_config=` form "
            "(same stage, same runner):\n"
        )
        for f, suite, stage, runner_config in legacy_shape:
            print(
                f"  {f}\n"
                f'    suite="{suite}"'
                f'  ->  stage="{stage}", runner_config="{runner_config}"'
            )
        print()
        exit_code = 1
    if non_dispatchable:
        print(
            'ERROR: CUDA test(s) register a legacy `suite="..."` that is neither a '
            "nightly/stress/weekly suite nor the modern `stage=`/`runner_config=` "
            "form. This name matches no suite the PR-test workflows invoke, so the "
            "test silently never runs. Switch to the modern form:\n"
        )
        for f, suite in non_dispatchable:
            print(f'  {f}\n    suite="{suite}"  ->  stage="...", runner_config="..."')
        print()
        exit_code = 1
    if dead_tests:
        print(
            "ERROR: Test file(s) define TestCase classes that CI never runs: "
            "the registered file is executed as `python3 file.py`, but its "
            '`if __name__ == "__main__"` block does not call unittest.main() '
            "or pytest.main(), so the classes are silently skipped while the "
            "file reports success. Make __main__ run the tests (put any CLI "
            "entry point behind an explicit flag):\n"
        )
        for f in dead_tests:
            print(f"  {f}")
        print()
        exit_code = 1
    if taxonomy_violations:
        print("ERROR: Registered-test taxonomy violations:")
        for error in taxonomy_violations:
            print(f"  {error}")
        print()
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
