"""Core diffusion test files the TPU path filter matches run in the TPU lane.

Fork CI runs python/sglang/multimodal_gen/test/unit only through the
test/registered wrappers, which import test classes by name: a file nobody
wraps, or a class added to a wrapped module, runs nowhere until a wrapper
imports it. Every core test file the TPU path filter matches is wrapped or
exempted with a reason, and every class of a wrapped module is imported unless
the module is only partly wired. Reads source only.
"""

import ast
import fnmatch
import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

# The lane that runs for diffusion PRs. base-a-test-cpu runs from the checkout,
# which has the workflow the path-filter case reads; the TPU image does not.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

_REPO = Path(__file__).resolve().parents[4]
_REGISTERED = _REPO / "test" / "registered"
_TPU_WRAPPERS = _REGISTERED / "tpu" / "unit"
_UNIT_PACKAGE = "sglang.multimodal_gen.test.unit"
_UNIT_DIR = _REPO / "python" / Path(*_UNIT_PACKAGE.split("."))

_TPU_PATH_FILTER = _REPO / ".github" / "workflows" / "pr-test-tpu.yml"

# Upstream modules a TPU wrapper takes only some classes from, with why the rest
# stay off TPU. The wrappers' imports declare the wired classes and fail if one
# is renamed or removed; classes an upstream sync adds here need no edit.
_PARTLY_WIRED = {
    "test_server_args": "upstream ServerArgs cases; the rest mock a CUDA platform "
    "or have no platform branch",
    "test_attention_backend_selector": "upstream component-loader scoping has no "
    "platform branch",
    "test_platform_detection": "the rest are upstream CUDA detection",
    "test_minimax_h3_encoder_device": "the rest check the encoder device under "
    "CPU offload on a mocked CUDA device",
    "test_wan_temb_table_slices": "the rest build Wan/LongLive2 DiTs under the unit "
    "package's conftest model-parallel fixture, which the TPU wrappers do not load",
}

# Core test files the TPU path filter matches that no TPU wrapper runs, with why.
_NOT_WRAPPED = {
    "test_usp_ipc_a2a_guard": "the CUDA IPC all-to-all staging path, which TPU "
    "never takes",
    "test_usp_packed_qkv_a2a": "every case needs CUDA (skipUnless cuda)",
}


def _path_filtered_modules() -> set[str]:
    """Core test modules a glob in the TPU workflow's path filter matches."""
    prefix = "python/sglang/multimodal_gen/test/unit/"
    globs = re.findall(rf"- '({re.escape(prefix)}[^']+)'", _TPU_PATH_FILTER.read_text())
    return {
        path.stem
        for path in _UNIT_DIR.glob("*.py")
        if any(fnmatch.fnmatch(prefix + path.name, glob) for glob in globs)
    }


def _unit_imports(path: Path) -> dict[str, set[str]]:
    """The names a file imports from each core test module."""
    source = path.read_text()
    imports = {}
    if _UNIT_PACKAGE not in source:
        return imports
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            _UNIT_PACKAGE + "."
        ):
            module = node.module.removeprefix(_UNIT_PACKAGE + ".")
            imports.setdefault(module, set()).update(a.name for a in node.names)
    return imports


def _test_classes(module: str) -> set[str]:
    """The TestCase subclasses with test methods that the module defines."""
    path = _UNIT_DIR / f"{module.replace('.', '/')}.py"
    classes = {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    }

    def bases(name):
        return [ast.unparse(b).rsplit(".", 1)[-1] for b in classes[name].bases]

    # `seen` stops at a base that resolves back to a class on the walk, e.g.
    # `class Module(torch.nn.Module)`.
    def is_test_case(name, seen=frozenset()):
        return (
            name in classes
            and name not in seen
            and any(
                base.endswith("TestCase") or is_test_case(base, seen | {name})
                for base in bases(name)
            )
        )

    def has_tests(name, seen=frozenset()):
        return (
            name in classes
            and name not in seen
            and (
                any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name.startswith("test")
                    for item in classes[name].body
                )
                or any(has_tests(base, seen | {name}) for base in bases(name))
            )
        )

    return {name for name in classes if is_test_case(name) and has_tests(name)}


class TestDiffusionUnitWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapped = set()
        cls.imported = {}
        for path in sorted(_REGISTERED.rglob("*.py")):
            for module, names in _unit_imports(path).items():
                cls.imported.setdefault(module, set()).update(names)
                if path.parent == _TPU_WRAPPERS:
                    cls.wrapped.add(module)

    @unittest.skipUnless(
        _TPU_PATH_FILTER.exists(), "the TPU image ships without .github"
    )
    def test_every_path_filtered_file_is_wrapped_or_exempt(self):
        missing = sorted(_path_filtered_modules() - self.wrapped - set(_NOT_WRAPPED))
        self.assertEqual(
            missing,
            [],
            "the TPU path filter matches these core test files but no TPU wrapper "
            "imports them: wrap each, or add it to _NOT_WRAPPED with why",
        )

    def test_every_test_class_of_a_wrapped_module_is_imported(self):
        unwired = sorted(
            f"{module}.{name}"
            for module in self.wrapped - set(_PARTLY_WIRED)
            for name in _test_classes(module) - self.imported[module]
        )
        self.assertEqual(
            unwired,
            [],
            "no registered file imports these: add each to a TPU wrapper, or "
            "the module to _PARTLY_WIRED with why",
        )

    @unittest.skipUnless(
        _TPU_PATH_FILTER.exists(), "the TPU image ships without .github"
    )
    def test_exemptions_are_current(self):
        stale = sorted(
            [f"_PARTLY_WIRED: {m}" for m in _PARTLY_WIRED if m not in self.wrapped]
            + [
                f"_NOT_WRAPPED: {m}"
                for m in _NOT_WRAPPED
                if m in self.wrapped or m not in _path_filtered_modules()
            ]
        )
        self.assertEqual(stale, [], "drop these: no longer partly wired or exempt")


if __name__ == "__main__":
    unittest.main()
