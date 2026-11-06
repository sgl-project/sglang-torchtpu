"""CI prefetch must reach every pod, including both P/D roles."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SETTING = "CI_WEIGHT_LOADER_PREFETCH"


class TestCheckpointPrefetch(unittest.TestCase):
    def pod_settings(self, renderer, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / renderer), "--image", "test:image", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        document = yaml.safe_load(result.stdout)
        values = []
        for role in document["spec"]["replicatedJobs"]:
            pod = role["template"]["spec"]["template"]["spec"]
            script = pod["containers"][0]["args"][0]
            exports = "\n".join(
                line
                for line in script.splitlines()
                if line.strip().startswith("export ") and SETTING in line
            )
            env = dict(os.environ)
            env.pop(SETTING, None)
            output = subprocess.check_output(
                ["bash", "-uc", exports + '\nprintf "%s" "${' + SETTING + ':-}"'],
                env=env,
                text=True,
            )
            values.append(output)
        return values

    def test_in_project_nightly_and_pr_gate(self):
        renderer = ".github/tpu-ci/render_jobset.py"
        modes = (
            ("--suite", "nightly-tpu-4-v7"),
            ("--launch-script", "recipe/test.sh", "--files", "test.py", "--hosts", "2"),
            (
                "--exec",
                "true",
                "--files",
                "test.py",
                "--pd-prefill-hosts",
                "1",
                "--pd-decode-hosts",
                "1",
            ),
        )
        for mode in modes:
            for nightly in ((), ("--nightly",), ("--lane", "nightly")):
                with self.subTest(mode=mode, nightly=bool(nightly)):
                    values = self.pod_settings(renderer, *mode, *nightly)
                    self.assertEqual(values, ["1"] * len(values))
                    self.assertEqual(len(values), 2 if "--exec" in mode else 1)

    def test_external_nightly_all_launch_modes(self):
        renderer = ".github/tpu-ci/external/render_external_jobset.py"
        for mode in ("self", "launch", "exec", "pd"):
            group = {
                "suite": "external-nightly-1host"
                if mode == "self"
                else "external-nightly-2x2x2",
                "mode": mode,
                "run_name": "prefetch-test",
                "files": "test.py",
                "timeout_per_file": 3600,
                "launch_script": "recipe/test.sh",
                "prefill_hosts": 1,
                "decode_hosts": 1,
            }
            with self.subTest(mode=mode):
                values = self.pod_settings(
                    renderer, "--nodepool", "test-pool", "--group", json.dumps(group)
                )
                self.assertEqual(values, ["1", "1"] if mode == "pd" else ["1"])


if __name__ == "__main__":
    unittest.main()
