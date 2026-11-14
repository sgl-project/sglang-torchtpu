"""SenseNova-U1.5 diffusion recipe's LIBTPU_INIT_ARGS (CPU only, no server)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

RECIPES = Path(__file__).resolve().parents[5] / "benchmark" / "tpu" / "diffusion"


def _launch(script, libtpu_init_args=None):
    """Run a recipe with a stub python3; return its LIBTPU_INIT_ARGS and args."""
    with tempfile.TemporaryDirectory() as stub_dir:
        stub = Path(stub_dir) / "python3"
        stub.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "${LIBTPU_INIT_ARGS}" "$*"\n'
        )
        stub.chmod(0o755)
        env = {"PATH": f"{stub_dir}:{os.environ['PATH']}", "MODEL": "model"}
        if libtpu_init_args is not None:
            env["LIBTPU_INIT_ARGS"] = libtpu_init_args
        out = subprocess.run(
            ["bash", str(RECIPES / script)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    libtpu, args = out.splitlines()
    return libtpu, args


class TestSenseNovaRecipe(unittest.TestCase):
    def test_adds_the_clock_and_scoped_vmem_defaults(self):
        libtpu, args = _launch("sensenova_u1.sh")
        self.assertEqual(
            libtpu, "--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65536"
        )
        self.assertIn("--model-path model --num-gpus 1", args)
        self.assertNotIn("--attention-backend", args)

    def test_libtpu_flags_match_launch_sh(self):
        for preset in (
            None,
            "",
            "--xla_tpu_dvfs_p_state=5",
            "--xla_foo=1 --xla_tpu_scoped_vmem_limit_kib=32768",
            "--xla_foo=1",
        ):
            with self.subTest(preset=preset):
                self.assertEqual(
                    _launch("sensenova_u1.sh", preset)[0],
                    _launch("launch.sh", preset)[0],
                )


if __name__ == "__main__":
    unittest.main()
