import importlib.util
import json
import os
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "ci_timing.py"


class CiTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("ci_timing", SCRIPT)
        cls.timing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.timing)

    def test_cancellation_interrupts_long_waits(self):
        source = (SCRIPT.parents[1] / "workflows" / "_tpu-test.yml").read_text()
        start = source.index("          release_now() {")
        stop = source.index("          trap cleanup EXIT", start) + len(
            "          trap cleanup EXIT"
        )
        functions = textwrap.dedent(source[start:stop])
        start = source.index('            kubectl -n tpu-ci logs -f -l "$sel"')
        stop = source.index('            since="$(date', start)
        stream = textwrap.dedent(source[start:stop])
        # Use the actual polling sleep, including its interruptible wrapper.
        poll = next(line.strip() for line in source.splitlines() if "sleep 10" in line)
        for phase, body in (("stream", stream), ("cleanup", "exit 0"), ("poll", poll)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                started, deleted = root / "started", root / "deleted"
                stubs = {
                    "kubectl": 'if [[ " $* " != *" logs "* ]]; then printf "%s\\n" "$*" >> "$DELETE_LOG"; exit; fi\n',
                    "sleep": "",
                    "timeout": 'shift; exec "$@"\n',
                }
                for name, stub in stubs.items():
                    if name != "timeout":
                        stub += 'touch "$STARTED"; exec /bin/sleep 60\n'
                    path = root / name
                    path.write_text("#!/bin/bash\n" + stub)
                    path.chmod(0o755)
                script = (
                    "set -euo pipefail\nname=jobset/test\nsel=test\nwl_sel=test\n"
                    "range=(--tail=-1)\n" + functions + "\n" + body
                )
                process = subprocess.Popen(
                    ["bash", "-c", script],
                    env={
                        **os.environ,
                        "PATH": directory + os.pathsep + os.environ["PATH"],
                        "RUNNER_TEMP": directory,
                        "STARTED": str(started),
                        "DELETE_LOG": str(deleted),
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 3
                    while not started.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(started.exists(), "blocking command never started")
                    # Signal only Bash, as the runner does before killing its children.
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        self.fail(
                            "cancellation was deferred until the command finished"
                        )
                    self.assertEqual(process.returncode, 143)
                    self.assertIn("delete jobset/test", deleted.read_text())
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate(timeout=2)

    def report(self, pods=None, workloads=None, log="", saved_admission=None):
        return self.timing.build_report(
            {
                "metadata": {"creationTimestamp": "2026-11-06T01:00:00Z"},
                "spec": {
                    "replicatedJobs": [
                        {
                            "replicas": 1,
                            "template": {"spec": {"completions": 2, "parallelism": 2}},
                        }
                    ]
                },
            },
            workloads
            if workloads is not None
            else {
                "items": [
                    {
                        "status": {
                            "conditions": [
                                {
                                    "type": "QuotaReserved",
                                    "status": "True",
                                    "lastTransitionTime": "2026-11-06T01:00:10Z",
                                },
                                {
                                    "type": "Admitted",
                                    "status": "True",
                                    "lastTransitionTime": "2026-11-06T01:02:00Z",
                                },
                            ]
                        }
                    }
                ]
            },
            pods
            if pods is not None
            else {
                "items": [
                    self.pod("01:03:00", "01:05:30"),
                    self.pod("01:03:10", "01:06:00"),
                ]
            },
            log,
            job_name="nightly / test",
            suite="nightly-tpu",
            run_id="123",
            attempt="2",
            saved_admission=saved_admission,
        )

    @staticmethod
    def pod(start, finish=None):
        state = (
            {
                "terminated": {
                    "startedAt": f"2026-11-06T{start}Z",
                    "finishedAt": f"2026-11-06T{finish}Z",
                }
            }
            if finish
            else {"running": {"startedAt": f"2026-11-06T{start}Z"}}
        )
        return {
            "status": {
                "containerStatuses": [
                    {"name": "test", "state": state},
                    {
                        "name": "sidecar",
                        "state": {
                            "terminated": {
                                "startedAt": "2026-11-06T00:00:00Z",
                                "finishedAt": "2026-11-06T03:00:00Z",
                            }
                        },
                    },
                ]
            }
        }

    def test_measures_real_admission_and_all_test_containers(self):
        report = self.report()
        self.assertEqual(
            report["tpu"],
            {
                "created_at": "2026-11-06T01:00:00Z",
                "admitted_at": "2026-11-06T01:02:00Z",
                "started_at": "2026-11-06T01:03:00Z",
                "test_started_at": None,
                "finished_at": "2026-11-06T01:06:00Z",
                "queue_seconds": 120.0,
                "startup_seconds": 60.0,
                "execution_seconds": 180.0,
            },
        )
        self.assertEqual(
            {
                key: report[key]
                for key in ("schema_version", "job_name", "suite", "run_id", "attempt")
            },
            {
                "schema_version": 1,
                "job_name": "nightly / test",
                "suite": "nightly-tpu",
                "run_id": "123",
                "attempt": "2",
            },
        )

    def test_saved_admission_survives_readmission_or_eviction(self):
        for status in ("True", "False"):
            with self.subTest(status=status):
                report = self.report(
                    saved_admission="2026-11-06T01:01:00Z",
                    workloads={
                        "items": [
                            {
                                "status": {
                                    "conditions": [
                                        {
                                            "type": "Admitted",
                                            "status": status,
                                            "lastTransitionTime": "2026-11-06T01:02:00Z",
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                )
                self.assertEqual(report["tpu"]["admitted_at"], "2026-11-06T01:01:00Z")
                self.assertEqual(report["tpu"]["queue_seconds"], 60.0)
        self.assertEqual(
            self.report(saved_admission="invalid")["tpu"]["queue_seconds"], 120.0
        )

    def test_admission_observation_is_saved_once_and_used_by_cleanup(self):
        workload = {
            "items": [
                {
                    "status": {
                        "conditions": [
                            {
                                "type": "Admitted",
                                "status": "True",
                                "lastTransitionTime": "2026-11-06T01:01:00Z",
                            }
                        ]
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "admission.txt"
            output = Path(directory) / "timing.json"
            args = ["--record-admission", str(saved), "--workload-selector", "uid=a"]
            with patch.object(self.timing, "kubectl_json", return_value=workload):
                self.timing.main(args)
            self.assertEqual(saved.read_text().strip(), "2026-11-06T01:01:00Z")
            with patch.object(self.timing, "kubectl_json", return_value={"items": []}):
                self.timing.main(args)
                self.timing.main(
                    [
                        "--output",
                        str(output),
                        "--admission-file",
                        str(saved),
                        "--workload-selector",
                        "uid=a",
                    ]
                )
            self.assertEqual(saved.read_text().strip(), "2026-11-06T01:01:00Z")
            report = json.loads(output.read_text())
            self.assertEqual(report["tpu"]["admitted_at"], "2026-11-06T01:01:00Z")

    def test_unavailable_cluster_cannot_save_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "admission.txt"
            with patch.object(
                self.timing.subprocess, "run", side_effect=FileNotFoundError("kubectl")
            ):
                self.timing.main(
                    [
                        "--record-admission",
                        str(saved),
                        "--workload-selector",
                        "uid=a",
                    ]
                )
            self.assertFalse(saved.exists())

    def test_actual_test_start_is_not_container_start(self):
        report = self.report(
            log="[pod/x/test] TPU_CI_TEST_STARTED_AT 2026-11-06T01:03:20Z\n[pod/y/test] TPU_CI_TEST_STARTED_AT 2026-11-06T01:03:30Z"
        )
        self.assertEqual(report["tpu"]["test_started_at"], "2026-11-06T01:03:20Z")
        self.assertIsNone(self.report()["tpu"]["test_started_at"])

    def test_partial_or_missing_lifecycle_is_unknown(self):
        report = self.report(
            pods={"items": [self.pod("01:03:00")]}, workloads={"items": []}
        )
        self.assertIsNone(report["tpu"]["queue_seconds"])
        self.assertIsNone(report["tpu"]["startup_seconds"])
        self.assertIsNone(report["tpu"]["execution_seconds"])
        self.assertIsNone(report["tpu"]["finished_at"])
        self.assertIsNone(self.report(pods={"items": []})["tpu"]["execution_seconds"])

    def test_missing_multihost_pod_cannot_report_completed_execution(self):
        report = self.report(pods={"items": [self.pod("01:03:00", "01:05:00")]})
        self.assertIsNone(report["tpu"]["execution_seconds"])
        self.assertIsNone(report["tpu"]["finished_at"])

    def test_missing_container_start_cannot_report_completed_execution(self):
        pod = self.pod("01:03:00", "01:05:00")
        del pod["status"]["containerStatuses"][0]["state"]["terminated"]["startedAt"]
        report = self.report(pods={"items": [pod, self.pod("01:03:00", "01:05:00")]})
        self.assertIsNone(report["tpu"]["execution_seconds"])

    def test_rejects_false_admission_and_negative_intervals(self):
        report = self.report(
            workloads={
                "items": [
                    {
                        "status": {
                            "conditions": [
                                {
                                    "type": "Admitted",
                                    "status": "False",
                                    "lastTransitionTime": "2026-11-06T01:01:00Z",
                                },
                                {
                                    "type": "Admitted",
                                    "status": "True",
                                    "lastTransitionTime": "2026-11-06T00:59:00Z",
                                },
                            ]
                        }
                    }
                ]
            }
        )
        self.assertIsNone(report["tpu"]["queue_seconds"])

    def test_parses_failed_timings_and_deduplicates_reattached_log(self):
        records = [
            {"file": "test/a.py", "elapsed": 12.5, "passed": True},
            {"file": "test/b.py", "elapsed": 2, "passed": False},
            {"file": "test/bad.py", "elapsed": -2, "passed": True},
            {"file": "test/nan.py", "elapsed": float("nan"), "passed": True},
            {"file": "test/string.py", "elapsed": 1, "passed": "false"},
        ]
        block = (
            "[pod/a/test] ========== TIMINGS BEGIN ==========\n"
            + "\n".join(
                "2026-11-06T01:00:00Z [pod/a/test] " + json.dumps(record)
                for record in records
            )
            + "\n[pod/a/test] ========== TIMINGS END ==========\n"
        )
        log = (
            json.dumps({"file": "outside.py", "elapsed": 1, "passed": True})
            + "\n"
            + block
            + block
        )
        self.assertEqual(self.report(log=log)["tests"], records[:2])

    def test_collection_survives_unavailable_kubectl(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with patch.object(
                self.timing.subprocess, "run", side_effect=FileNotFoundError("kubectl")
            ):
                self.timing.main(
                    [
                        "--jobset",
                        "jobset/a",
                        "--pod-selector",
                        "x=a",
                        "--workload-selector",
                        "uid=a",
                        "--output",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text())
            self.assertIsNone(report["tpu"]["queue_seconds"])
            self.assertEqual(report["tests"], [])


if __name__ == "__main__":
    unittest.main()
