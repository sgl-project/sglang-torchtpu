"""Core diffusion test modules a TPU wrapper imports have every class wired.

Fork CI runs python/sglang/multimodal_gen/test/unit only through the
test/registered wrappers, which import test classes by name: a class added to a
wrapped module runs nowhere until a wrapper imports it. Reads source only.
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci

# The lane that runs for diffusion PRs; base-a-test-cpu skips multimodal_gen paths.
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

_REPO = Path(__file__).resolve().parents[4]
_REGISTERED = _REPO / "test" / "registered"
_TPU_WRAPPERS = _REGISTERED / "tpu" / "unit"
_UNIT_PACKAGE = "sglang.multimodal_gen.test.unit"
_UNIT_DIR = _REPO / "python" / Path(*_UNIT_PACKAGE.split("."))

# Test classes of wrapped modules that no registered file imports, with why.
_NOT_WIRED = {
    # Upstream ServerArgs cases; each mocks a CUDA platform or has no platform
    # branch.
    "test_server_args": {
        "TestDiffusionModelDetection",
        "TestDirectGpuWeightLoading",
        "TestDisaggTimeoutArgs",
        "TestDisaggTransferBackendArgs",
        "TestFSDPShardConditions",
        "TestKVGatherDegree",
        "TestLayerwiseResidencyLifetime",
        "TestMiniMaxH3Routing",
        "TestModelIdResolution",
        "TestNcclNvlsArgs",
        "TestOffloadDefaults",
        "TestPerRoleParallelism",
        "TestPipelineResolutionCliOverride",
        "TestPlatformLifecycleHooks",
        "TestSchedulerRpcTimeoutArgs",
        "TestServerArgsPathExpansion",
        "TestWarmupImageIsModelValid",
        "TestWarmupModeNormalization",
    },
    # Upstream component-loader scoping; no platform branch.
    "test_attention_backend_selector": {"TestComponentAttentionBackendScope"},
    # Upstream CUDA detection.
    "test_platform_detection": {"TestCudaPlatformDetection"},
    # Upstream encoder device under CPU offload, on a mocked CUDA device.
    "test_minimax_h3_encoder_device": {"TestMiniMaxH3EncoderDevice"},
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

    def test_every_test_class_of_a_wrapped_module_is_imported(self):
        unwired = sorted(
            f"{module}.{name}"
            for module in self.wrapped
            for name in _test_classes(module)
            - self.imported[module]
            - _NOT_WIRED.get(module, set())
        )
        self.assertEqual(
            unwired,
            [],
            "no registered file imports these: add each to a TPU wrapper, or "
            "to _NOT_WIRED with why",
        )

    def test_not_wired_lists_only_unimported_test_classes(self):
        stale = sorted(
            f"{module}.{name}"
            for module, names in _NOT_WIRED.items()
            for name in names
            - (_test_classes(module) - self.imported.get(module, set()))
        )
        self.assertEqual(stale, [], "imported or gone: drop these from _NOT_WIRED")


if __name__ == "__main__":
    unittest.main()
