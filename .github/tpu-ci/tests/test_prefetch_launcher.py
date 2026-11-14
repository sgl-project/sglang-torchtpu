"""CPU checks for checkpoint settings applied by CI test launchers."""

import ast
import builtins
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from benchmark.tpu.ci_checkpoint_prefetch import configure_checkpoint_prefetch


class TestCheckpointPrefetch(unittest.TestCase):
    def test_shared_launcher_prefetch_does_not_require_benchmark_package(self):
        root = Path(__file__).resolve().parents[3]
        source = root / "python/sglang/test/test_utils.py"
        launcher = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "popen_launch_server"
        )
        module = ast.parse("from __future__ import annotations")
        module.body.append(launcher)
        process = object()
        launch = Mock(return_value=process)
        namespace = {
            "os": os,
            "sys": sys,
            "shlex": shlex,
            "_launch_server_process": launch,
            "_wait_for_server_health": lambda *args: (True, None),
        }
        exec(compile(module, str(source), "exec"), namespace)
        packages = {}
        for name, directory in (
            ("sglang", root / "python/sglang"),
            ("sglang.test", root / "python/sglang/test"),
        ):
            packages[name] = ModuleType(name)
            packages[name].__path__ = [str(directory)]
        packages["sglang.utils"] = ModuleType("sglang.utils")
        packages["sglang.utils"].is_in_ci = lambda: False
        original_import = builtins.__import__

        def import_without_benchmark(name, *args, **kwargs):
            if name == "benchmark" or name.startswith("benchmark."):
                raise ModuleNotFoundError("The benchmark package is not installed")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(sys.modules, packages),
            patch("builtins.__import__", side_effect=import_without_benchmark),
            patch.dict(os.environ, {}, clear=True),
            redirect_stdout(io.StringIO()),
        ):
            result = namespace["popen_launch_server"](
                "checkpoint",
                "http://localhost:30000",
                timeout=1,
                other_args=["--tp-size", "8"],
                env={"CI_WEIGHT_LOADER_PREFETCH": "1"},
                device="tpu",
            )
        self.assertIs(result, process)
        command, env, _, _ = launch.call_args.args
        self.assertIn("--weight-loader-prefetch-checkpoints", command)
        self.assertEqual(env["LOCAL_SIZE"], "8")

    def test_legacy_shell_recipe_receives_prefetch_settings(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            interpreter = Path(directory) / "python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:4] == ['-u', '-m', 'sglang.launch_server']:\n"
                "    print('TEST_SERVER=' + json.dumps({'args': sys.argv[1:], "
                "'local_size': os.environ.get('LOCAL_SIZE')}))\n"
                "else:\n"
                "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
            )
            interpreter.chmod(0o755)
            output = subprocess.check_output(
                ["bash", str(root / "benchmark/tpu/recipe/gpt_oss_120b/server.sh")],
                env={
                    **os.environ,
                    "PYTHON": str(interpreter),
                    "MODEL": "checkpoint",
                    "CI_WEIGHT_LOADER_PREFETCH": "1",
                    "LOCAL_SIZE": "99",
                    "FORCE_SINGLE_HOST": "1",
                    "SGLANG_USE_RAY": "0",
                    "PP_SIZE": "1",
                    "CONCUR": "1",
                    "DUMP_META": "0",
                    "ENABLE_COMPILE_CACHE": "0",
                    "ENABLE_PROFILER_ENDPOINT": "0",
                    "ENABLE_HLO_DUMP": "0",
                    "ENABLE_TORCH_COMPILE_DEBUG": "0",
                    "RUN_DIR": directory,
                },
                cwd=root,
                stderr=subprocess.STDOUT,
                text=True,
            )
        server = next(
            line for line in output.splitlines() if line.startswith("TEST_SERVER=")
        )
        result = json.loads(server.removeprefix("TEST_SERVER="))
        self.assertIn("--weight-loader-prefetch-checkpoints", result["args"])
        self.assertEqual(result["local_size"], "8")

    def test_ci_overrides_preserve_unrelated_arguments(self):
        extra = '{"num_threads":2,"name":"space and quote\'"}'
        args = [
            "--model-path",
            "checkpoint",
            "--tp",
            "8",
            "--weight-loader-drop-cache-after-load",
            "--weight-loader-prefetch-num-threads=2",
            "--model-loader-extra-config",
            extra,
        ]
        original = list(args)
        env = {"CI_WEIGHT_LOADER_PREFETCH": "1", "LOCAL_SIZE": "99"}
        command, child, _ = configure_checkpoint_prefetch(args, env)
        self.assertIn("--weight-loader-prefetch-checkpoints", command)
        self.assertEqual(
            command[command.index("--weight-loader-prefetch-num-threads") + 1], "4"
        )
        self.assertNotIn("--weight-loader-disable-mmap", command)
        self.assertNotIn("--weight-loader-drop-cache-after-load", command)
        self.assertEqual(
            command[command.index("--model-loader-extra-config") + 1], extra
        )
        self.assertEqual(child["LOCAL_SIZE"], "8")
        self.assertEqual(args, original)
        self.assertEqual(env["LOCAL_SIZE"], "99")

    def test_explicit_mmap_opt_out_keeps_the_launch_unchanged(self):
        args = [
            "--tp-size",
            "8",
            "--weight-loader-disable-mmap",
            "--weight-loader-drop-cache-after-load",
            "--weight-loader-prefetch-num-threads=4",
        ]
        env = {"CI_WEIGHT_LOADER_PREFETCH": "1", "LOCAL_SIZE": "99"}
        self.assertEqual(configure_checkpoint_prefetch(args, env), (args, env, {}))

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
                original = [*args, "--weight-loader-drop-cache-after-load"]
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
