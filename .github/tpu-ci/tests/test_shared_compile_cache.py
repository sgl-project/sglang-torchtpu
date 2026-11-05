"""The bucket-backed tier-3 compile cache is opt-in per single-host job."""

import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / ".github/tpu-ci/render_jobset.py"


def _pod_script(*args: str) -> str:
    """Render a single-host JobSet and return its container script."""
    rendered = subprocess.run(
        [sys.executable, str(RENDERER), "--image", "test:image", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pod = yaml.safe_load(rendered)["spec"]["replicatedJobs"][0]["template"]["spec"]
    return pod["template"]["spec"]["containers"][0]["args"][0]


class TestSharedCompileCache(unittest.TestCase):
    def test_off_by_default(self):
        script = _pod_script("--suite", "stage-a-test-tpu-unit")
        self.assertIn('if [ "0" = 1 ]; then', script)

    def test_opt_in_exports_bucket_root(self):
        script = _pod_script(
            "--suite", "stage-b-test-tpu-4-v7", "--shared-compile-cache"
        )
        self.assertIn('if [ "1" = 1 ]; then', script)
        self.assertIn('export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$t3"', script)
        self.assertIn('t3="/compile-cache/torch_tpu_tier3/$rt"', script)


if __name__ == "__main__":
    unittest.main()
