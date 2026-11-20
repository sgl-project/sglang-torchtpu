"""Timed-out cache probes must release worker resources before a retry."""

import ast
import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]


class TestPallasCacheProcessCleanup(unittest.TestCase):
    def setUp(self):
        source = ROOT / "test/registered/e2e/tpu/test_pallas_tier3_cache.py"
        tree = ast.parse(source.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_worker_stage"
        ]
        self.assertEqual(
            len(functions), 1, "cache probe needs timeout-safe worker cleanup"
        )
        namespace = {"subprocess": subprocess, "kill_process_tree": kill_process_tree}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.run_stage = namespace["_run_worker_stage"]

    def test_preserves_output_and_exit_code(self):
        result = self.run_stage(
            [sys.executable, "-S", "-c", "print('worker output'); raise SystemExit(7)"],
            env=os.environ.copy(),
            timeout=5,
        )
        self.assertEqual(result.returncode, 7)
        self.assertIn("worker output", result.stdout)

    def test_timeout_releases_a_detached_workers_lock(self):
        child = (
            "import fcntl, os, sys, time; "
            "lock = open(sys.argv[1], 'a'); "
            "fcntl.flock(lock, fcntl.LOCK_EX); "
            "open(sys.argv[2], 'w').write(str(os.getpid())); "
            "print('worker ready', flush=True); time.sleep(60)"
        )
        launcher = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-S', '-c', *sys.argv[1:]], "
            "start_new_session=True); time.sleep(60)"
        )
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "device.lock"
            pid_path = Path(directory) / "worker.pid"
            try:
                result = self.run_stage(
                    [
                        sys.executable,
                        "-S",
                        "-c",
                        launcher,
                        child,
                        str(lock_path),
                        str(pid_path),
                    ],
                    env=os.environ.copy(),
                    timeout=2,
                )
                self.assertEqual(result.returncode, 124)
                self.assertIn("worker ready", result.stdout)
                self.assertIn("timed out", result.stderr)
                # A retry must be able to acquire the resource immediately.
                with lock_path.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                # Keep the test safe if cleanup regresses and leaves the child alive.
                if pid_path.exists():
                    try:
                        worker = psutil.Process(int(pid_path.read_text()))
                        if worker.cmdline()[-2:] == [str(lock_path), str(pid_path)]:
                            kill_process_tree(worker.pid, wait_timeout=5)
                    except psutil.NoSuchProcess:
                        pass


if __name__ == "__main__":
    unittest.main()
