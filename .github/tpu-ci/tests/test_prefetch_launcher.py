"""CPU checks for checkpoint settings applied by nightly test launchers."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.tpu.ci_checkpoint_prefetch import configure_checkpoint_prefetch


class TestCheckpointPrefetch(unittest.TestCase):
    def test_nightly_overrides_preserve_unrelated_arguments(self):
        extra = '{"num_threads":2,"name":"space and quote\'"}'
        args = [
            "--model-path",
            "checkpoint",
            "--tp",
            "8",
            "--weight-loader-disable-mmap",
            "--weight-loader-drop-cache-after-load",
            "--weight-loader-prefetch-num-threads=4",
            "--model-loader-extra-config",
            extra,
        ]
        original = list(args)
        env = {"CI_WEIGHT_LOADER_PREFETCH": "1", "LOCAL_SIZE": "99"}
        command, child, _ = configure_checkpoint_prefetch(args, env)
        self.assertIn("--weight-loader-prefetch-checkpoints", command)
        self.assertEqual(
            command[command.index("--weight-loader-prefetch-num-threads") + 1], "1"
        )
        self.assertNotIn("--weight-loader-disable-mmap", command)
        self.assertNotIn("--weight-loader-drop-cache-after-load", command)
        self.assertEqual(
            command[command.index("--model-loader-extra-config") + 1], extra
        )
        self.assertEqual(child["LOCAL_SIZE"], "8")
        self.assertEqual(args, original)
        self.assertEqual(env["LOCAL_SIZE"], "99")

    def test_local_size_counts_tp_and_pp_per_host(self):
        for tp, pp, nodes, expected in (
            (1, 1, 1, "1"),
            (8, 1, 1, "8"),
            (16, 1, 2, "8"),
            (1, 8, 1, "8"),
        ):
            with self.subTest(tp=tp, pp=pp, nodes=nodes):
                _, env, _ = configure_checkpoint_prefetch(
                    [
                        "--tensor-parallel-size",
                        str(tp),
                        "--pipeline-parallel-size",
                        str(pp),
                        "--nnodes",
                        str(nodes),
                    ],
                    {"CI_WEIGHT_LOADER_PREFETCH": "1"},
                )
                self.assertEqual(env["LOCAL_SIZE"], expected)

    def test_ordinary_ray_and_offset_launches_keep_their_settings(self):
        for args, env in (
            (["--tp-size", "8"], {}),
            (["--use-ray", "--nnodes", "0"], {"CI_WEIGHT_LOADER_PREFETCH": "1"}),
            (
                ["--tp-size", "8"],
                {"CI_WEIGHT_LOADER_PREFETCH": "1", "SGLANG_USE_RAY": "1"},
            ),
            (["--base-gpu-id", "4"], {"CI_WEIGHT_LOADER_PREFETCH": "1"}),
            (["--gpu-id-step", "2"], {"CI_WEIGHT_LOADER_PREFETCH": "1"}),
            (
                ["--tp-size", "4"],
                {"CI_WEIGHT_LOADER_PREFETCH": "1", "SGLANG_TPU_PD_COLOCATED": "1"},
            ),
        ):
            with self.subTest(args=args, env=env):
                env = {**env, "LOCAL_SIZE": "99"}
                original = [*args, "--weight-loader-disable-mmap"]
                command, child, overrides = configure_checkpoint_prefetch(original, env)
                self.assertEqual(command, original)
                self.assertEqual(child, env)
                self.assertEqual(overrides, {})

    def test_shell_setup_quotes_arguments(self):
        script = (
            Path(__file__).resolve().parents[3]
            / "benchmark/tpu/ci_checkpoint_prefetch.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "unexpected"
            payload = json.dumps({"value": f"quote' $(touch {marker})"})
            setup = subprocess.check_output(
                [
                    sys.executable,
                    str(script),
                    "--",
                    "-m",
                    "sglang.launch_server",
                    "--tp",
                    "8",
                    "--model-loader-extra-config",
                    payload,
                ],
                env={**os.environ, "CI_WEIGHT_LOADER_PREFETCH": "1"},
                text=True,
            )
            output = (
                subprocess.check_output(
                    [
                        "bash",
                        "-eu",
                        "-c",
                        'eval "$SETUP"; printf "%s\\0" "$LOCAL_SIZE" "${args[@]}"',
                    ],
                    env={**os.environ, "SETUP": setup},
                )
                .decode()
                .split("\0")
            )
            self.assertEqual(output[0], "8")
            self.assertIn(payload, output)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
