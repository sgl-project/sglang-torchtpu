"""Check shard planning for the sharded TPU unit suite."""

from __future__ import annotations

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


class TestPlanPartitions(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.scripts = self.root / ".github/tpu-ci"
        self.scripts.mkdir(parents=True)
        for name in ("ci_register_loader.py", "plan_partitions.py"):
            shutil.copyfile(CI_DIR / name, self.scripts / name)
        registry = self.root / REGISTRY_PATH
        registry.parent.mkdir(parents=True)
        shutil.copyfile(CI_DIR.parents[1] / REGISTRY_PATH, registry)
        self.tests = self.root / "test/registered"
        self.tests.mkdir(parents=True)

    def register(
        self, name: str, est_time: int, suite: str = "unit", extra: str = ""
    ) -> None:
        (self.tests / name).write_text(
            f'register_tpu_ci(est_time={est_time}, suite="{suite}"{extra})\n'
        )

    def run_python(self, script: Path, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, "-E", "-S", "-B", str(script), *args],
            cwd=self.root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def plan(self, *args: str) -> list[dict[str, int | str]]:
        out = self.run_python(
            self.scripts / "plan_partitions.py", "--suite", "unit", *args
        )
        return json.loads(out)["include"]

    def test_size_scales_with_est_time_and_is_capped(self) -> None:
        for i in range(10):
            self.register(f"test_{i}.py", 600)
        self.register("test_other_suite.py", 99999, suite="other")
        self.register("test_nightly.py", 99999, extra=", nightly=True")
        self.register("test_disabled.py", 99999, extra=', disabled="flaky"')

        self.assertEqual(len(self.plan("--target-seconds", "100000")), 1)
        self.assertEqual(len(self.plan("--target-seconds", "2000")), 3)
        self.assertEqual(len(self.plan("--target-seconds", "1")), 4)
        rows = self.plan("--target-seconds", "1", "--max-partitions", "2")
        self.assertEqual(
            rows,
            [
                {"id": 0, "size": 2, "label": "1-of-2"},
                {"id": 1, "size": 2, "label": "2-of-2"},
            ],
        )

    def test_shards_cover_suite_once_and_are_stable(self) -> None:
        for i, est in enumerate([900, 400, 400, 120, 120, 120, 60, 10, 10]):
            self.register(f"test_{i}.py", est)
        probe = self.scripts / "probe.py"
        probe.write_text(
            textwrap.dedent(
                """
                from ci_register_loader import ci_register
                from plan_partitions import suite_tests

                def shards():
                    tests = suite_tests("unit")
                    return [
                        sorted(t.filename for t in ci_register.auto_partition(tests, r, 3))
                        for r in range(3)
                    ]

                first = shards()
                assert first == shards()
                files = [f for shard in first for f in shard]
                assert len(files) == len(set(files)) == 9
                assert all(first)
                """
            )
        )
        self.run_python(probe)

    def test_empty_suite_fails(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-E",
                "-S",
                "-B",
                str(self.scripts / "plan_partitions.py"),
                "--suite",
                "unit",
            ],
            cwd=self.root,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no enabled TPU tests", result.stderr)


if __name__ == "__main__":
    unittest.main()
