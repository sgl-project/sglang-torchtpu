"""A filtered external run must execute only matching files within bundles."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/nightly-test-external-tpu.yml"


class TestExternalFilter(unittest.TestCase):
    def discover(self, test_filter):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        script = next(
            step["run"]
            for step in workflow["jobs"]["discover"]["steps"]
            if step.get("id") == "d"
        )
        rows = [
            {"name": "slice-0", "groups": [{"files": "other.py gpt.py qwen.py"}]},
            {"name": "slice-1", "groups": [{"files": "unrelated.py"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.json").write_text(json.dumps(rows))
            python = root / "python3"
            python.write_text(
                '#!/bin/sh\nif [ "$1" = ".github/tpu-ci/external/render_external_jobset.py" ]; '
                'then cat "$FILTER_INPUT"; fi\n'
            )
            python.chmod(0o755)
            output = root / "output"
            output.touch()
            result = subprocess.run(
                ["bash", "-c", script],
                env=dict(
                    os.environ,
                    TEST_FILTER=test_filter,
                    GITHUB_OUTPUT=str(output),
                    FILTER_INPUT=str(root / "rows.json"),
                    PATH=str(root) + os.pathsep + os.environ["PATH"],
                ),
                capture_output=True,
                text=True,
            )
            values = dict(
                line.split("=", 1) for line in output.read_text().splitlines()
            )
        return result, values

    def test_filter_trims_files_inside_a_bundle(self):
        result, values = self.discover("gpt.py qwen.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(values["slices"]),
            [{"name": "slice-0", "groups": [{"files": "gpt.py qwen.py"}]}],
        )
        self.assertEqual(json.loads(values["pairs"]), [])

    def test_unknown_filter_fails(self):
        result, _ = self.discover("missing.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test_filter matched no external tests", result.stdout)


if __name__ == "__main__":
    unittest.main()
