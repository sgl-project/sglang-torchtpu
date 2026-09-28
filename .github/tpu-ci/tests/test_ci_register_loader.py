"""Check discovery imports in a Python process without site-packages."""

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

CI_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path("python/sglang/test/ci/ci_register.py")


class TestCIRegisterLoader(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.scripts = self.root / ".github/tpu-ci"
        self.scripts.mkdir(parents=True)
        for name in ("ci_register_loader.py", "discover_tests.py", "test_names.py"):
            shutil.copyfile(CI_DIR / name, self.scripts / name)
        self.registry = self.root / REGISTRY_PATH
        self.registry.parent.mkdir(parents=True)
        shutil.copyfile(CI_DIR.parents[1] / REGISTRY_PATH, self.registry)

    def run_python(self, script, *args):
        result = subprocess.run(
            [sys.executable, "-E", "-S", "-B", str(script), *args],
            cwd=self.root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def run_probe(self, source):
        probe = self.scripts / "probe.py"
        probe.write_text(textwrap.dedent(source))
        return self.run_python(probe)

    def test_discovery_without_runtime_dependencies(self):
        tests = self.root / "test/registered"
        tests.mkdir(parents=True)
        (tests / "test_example.py").write_text(
            'raise RuntimeError("Discovery must not execute test files")\n'
            'register_tpu_ci(60, "example", weights_pvc="weights")\n'
        )
        rows = json.loads(
            self.run_python(self.scripts / "discover_tests.py", "--suite", "example")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file"], "test/registered/test_example.py")
        self.assertEqual(rows[0]["weights_pvc"], "weights")

    def test_postponed_annotations_and_class_serialization(self):
        self.registry.write_text(
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Registry:\n"
            "    value: int\n"
        )
        self.run_probe(
            """
            import inspect
            import pickle
            import sys
            from typing import get_type_hints

            original_path = sys.path.copy()
            from ci_register_loader import ci_register

            assert sys.path == original_path
            assert "sglang" not in sys.modules
            assert inspect.getmodule(ci_register.Registry) is ci_register
            assert get_type_hints(ci_register.Registry) == {"value": int}
            value = ci_register.Registry(7)
            assert pickle.loads(pickle.dumps(value)) == value
            """
        )

    def test_failed_import_can_be_retried(self):
        self.registry.write_text('raise RuntimeError("registration failed")\n')
        self.run_probe(
            f"""
            import sys
            from pathlib import Path

            try:
                import ci_register_loader
            except RuntimeError as error:
                assert str(error) == "registration failed"
            else:
                raise AssertionError("Import failure was hidden")
            assert "tpu_ci_register" not in sys.modules
            assert "ci_register_loader" not in sys.modules

            Path({str(self.registry)!r}).write_text("READY = True\\n")
            from ci_register_loader import ci_register
            assert ci_register.READY
            """
        )


if __name__ == "__main__":
    unittest.main()
