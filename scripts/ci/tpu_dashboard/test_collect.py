import io
import json
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import collect as collector
from collect import (
    GitHub,
    SafeRedirect,
    collect,
    merge_history,
    normalize_run,
    parse_log_timing,
    parse_tests,
    parse_workflow_selection,
    refresh_run_metadata,
    seconds,
)


class CollectionTests(unittest.TestCase):
    def test_invalid_days_fail_before_any_api_request(self):
        for days in ("0", "-3", " 500", "91", "18446744073709551617"):
            with self.subTest(days=days), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "data.json"
                with (
                    patch.object(collector, "GitHub") as api,
                    patch.object(
                        sys,
                        "argv",
                        ["collect", "--days", days, "--output", str(output)],
                    ),
                    patch.object(sys, "stderr", io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        collector.main()
                    self.assertEqual(error.exception.code, 2)
                    api.assert_not_called()
                    self.assertFalse(output.exists())

    def test_history_keeps_only_the_longest_page_range(self):
        def days_ago(days):
            return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        history = {
            "runs": [
                {"id": 1, "attempt": 1, "created_at": days_ago(89)},
                {"id": 2, "attempt": 1, "created_at": days_ago(91)},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.json"
            output = Path(directory) / "site" / "data.json"
            history_path.write_text(json.dumps(history))
            argv = ["collect", "--history", str(history_path), "--output", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(collector, "collect", return_value=([], [])),
            ):
                collector.main()
            retained = json.loads(output.read_text())["runs"]
        self.assertEqual([r["id"] for r in retained], [1])

    def collect_artifacts(self, timestamp=None):
        api = MagicMock()
        run = {
            "id": 1,
            "run_attempt": 3,
            "status": "completed",
            "conclusion": "success",
        }
        job = {"id": 7, "name": "test", "status": "completed", "conclusion": "success"}
        artifacts = [
            {
                "id": attempt * 10 + index,
                "name": f"tpu-ci-timing-{suite}-hash-1-{attempt}",
                "size_in_bytes": 1000,
            }
            for attempt in range(1, 4)
            for index, suite in enumerate(("unit", "e2e"))
        ]

        def pages(path, key):
            if key == "workflow_runs":
                return [run] if "pr-test-tpu.yml" in path else []
            return [job] if key == "jobs" else artifacts

        def get(path, **kwargs):
            if "/artifacts/" not in path:
                return dict(run, run_attempt=int(path.rsplit("/", 1)[1]))
            artifact_id = int(path.split("/")[-2])
            data = {
                "schema_version": 1,
                "run_id": 1,
                "attempt": artifact_id // 10,
                "job_name": "test",
                "tpu": {"test_started_at": timestamp},
                "tests": [],
            }
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("timing.json", json.dumps(data))
            return buffer.getvalue()

        api.pages.side_effect = pages
        api.get.side_effect = get
        records, warnings = collect(api, 1, [])
        return records, warnings, api

    def test_malformed_artifact_timestamp_stays_unknown(self):
        records, warnings, _ = self.collect_artifacts(timestamp=True)
        self.assertEqual(warnings, [])
        self.assertEqual(len(records), 3)
        self.assertTrue(
            all(r["jobs"][0]["tpu"]["test_started_at"] is None for r in records)
        )

    def test_each_attempt_downloads_only_its_artifacts(self):
        _, warnings, api = self.collect_artifacts()
        downloads = [
            call.args[0]
            for call in api.get.call_args_list
            if "/artifacts/" in call.args[0]
        ]
        self.assertEqual(warnings, [])
        self.assertEqual(len(downloads), 6)
        self.assertEqual(len(set(downloads)), 6)

    def test_invalid_timestamp_types_and_empty_test_names(self):
        for value in (True, 1, {"bad": 1}, ["bad"]):
            self.assertIsNone(seconds(value, value))
        self.assertEqual(
            parse_tests(
                'TIMINGS BEGIN\n{"file":"","passed":true,"elapsed":1}\nTIMINGS END'
            ),
            [],
        )

    def test_rate_limits_save_progress_but_permission_errors_still_raise(self):
        for code, headers in (
            (403, {"X-RateLimit-Remaining": "0"}),
            (429, {}),
            (403, {"X-RateLimit-Remaining": "1"}),
        ):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                api = MagicMock()
                run = {
                    "id": 1,
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                def pages(path, key):
                    if key == "workflow_runs":
                        if "pr-test-tpu.yml" in path:
                            return [run]
                        raise HTTPError(path, code, "rate limited", headers, None)
                    return []

                api.pages.side_effect = pages
                api.get.return_value = run
                output = Path(directory) / "data.json"
                with (
                    patch.object(collector, "GitHub", return_value=api),
                    patch.object(sys, "argv", ["collect.py", "--output", str(output)]),
                ):
                    if headers.get("X-RateLimit-Remaining") == "1":
                        with self.assertRaises(HTTPError):
                            collector.main()
                        self.assertFalse(output.exists())
                        continue
                    collector.main()
                data = json.loads(output.read_text())
                self.assertEqual([r["id"] for r in data["runs"]], [1])
                self.assertTrue(
                    any("rate limit" in warning for warning in data["warnings"])
                )

    def test_selection_uses_executed_timestamped_output_only(self):
        self.assertIsNone(parse_workflow_selection('echo "selection: day-10am"'))
        self.assertIsNone(
            parse_workflow_selection('2026-11-06T10:00:00Z echo "selection: day-10am"')
        )
        for selection in ("day-10am", "day-5pm"):
            self.assertEqual(
                parse_workflow_selection(
                    f"2026-11-06T18:20:00.123Z selection: {selection}"
                ),
                selection,
            )

    def test_normalization_retains_event(self):
        self.assertEqual(
            normalize_run({"id": 1, "event": "schedule"}, [], "nightly")["event"],
            "schedule",
        )

    def test_cached_complete_record_gets_metadata_without_timing_download(self):
        api = MagicMock()
        run = {
            "id": 1,
            "event": "schedule",
            "name": "Accuracy Test (TPU)",
            "workflow_id": 22,
        }
        cached = {
            "id": 1,
            "attempt": 1,
            "status": "completed",
            "collection_complete": True,
            "timing_version": 3,
            "classification": "executed",
            "jobs": [{"id": 42, "name": "plan"}],
        }
        api.pages.side_effect = lambda path, key: (
            [run] if "accuracy-test-tpu.yml" in path else []
        )
        api.get.return_value = b"2026-11-06T18:20:00Z selection: day-10am"
        records, warnings = collect(api, 1, [cached])
        self.assertEqual(warnings, [])
        self.assertEqual(cached["event"], "schedule")
        self.assertEqual(cached["workflow_name"], "Accuracy Test (TPU)")
        self.assertEqual(cached["workflow_selection"], "day-10am")
        api.get.assert_called_once_with("/actions/jobs/42/logs", raw=True)
        self.assertEqual(records, [])

    def test_selection_missing_or_unavailable_stays_unknown(self):
        api = MagicMock()
        api.get.side_effect = ValueError("too large")
        record = {"id": 1, "jobs": [{"id": 42, "name": "plan"}]}
        warnings = []
        refresh_run_metadata(
            api,
            record,
            {"event": "workflow_dispatch"},
            "accuracy-test-tpu.yml",
            warnings,
        )
        self.assertIsNone(record["workflow_selection"])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(record["event"], "workflow_dispatch")

    def test_historical_wait_estimates_exclude_shell_source_and_keep_queue_unknown(
        self,
    ):
        log = "\n".join(
            [
                '2026-11-06T01:00:00Z echo "Created jobset.jobset.x-k8s.io/fake"',
                "2026-11-06T01:00:10Z Created jobset.jobset.x-k8s.io/real",
                "2026-11-06T01:02:10Z real-slice-0-0-abc 2/2 Running 0 2m node",
                "2026-11-06T01:02:20Z [pod/real-slice-0-0-abc/test] ✅ Enabled 8 test(s) (est total 300s):",
                "2026-11-06T01:07:20Z [pod/real-slice-0-0-abc/test] ========== TIMINGS END ==========",
            ]
        )
        result = parse_log_timing(log)
        self.assertEqual(result["test_started_at_estimate"], "2026-11-06T01:02:20Z")
        self.assertEqual(result["execution_seconds_estimate"], 300)
        self.assertEqual(result["cluster_wait_startup_estimate_seconds"], 120)
        self.assertIsNone(result["queue_seconds"])
        self.assertIsNone(parse_log_timing("no timestamps")["test_started_at_estimate"])

    def test_prefixed_failed_test_and_duplicate(self):
        log = "\n".join(
            [
                "TIMINGS BEGIN",
                '[pod/x/test] {"file":"x.py","passed":false,"elapsed":12}',
                '[pod/x/test] {"file":"x.py","passed":true,"elapsed":13}',
                "TIMINGS END",
            ]
        )
        self.assertEqual(
            parse_tests(log), [{"file": "x.py", "passed": True, "elapsed": 13}]
        )

    def test_trigger_gate_and_unknown_queue(self):
        run = {
            "id": 1,
            "run_attempt": 2,
            "created_at": "2026-11-06T00:00:00Z",
            "run_started_at": "2026-11-06T01:00:00Z",
        }
        jobs = [
            {
                "id": 2,
                "name": "decide",
                "conclusion": "failure",
                "steps": [
                    {"name": "Require an explicit trigger", "conclusion": "failure"}
                ],
            }
        ]
        result = normalize_run(run, jobs, "pr")
        self.assertEqual(result["classification"], "awaiting_trigger")
        self.assertIsNone(result["jobs"][0]["github_wait_seconds"])
        self.assertEqual(result["started_at"], run["run_started_at"])

    def test_parallel_jobs_and_history_attempts(self):
        r = {"id": 1, "run_attempt": 1, "created_at": "2026-11-06T00:00:00Z"}
        jobs = [
            {
                "id": i,
                "started_at": "2026-11-06T00:01:00Z",
                "completed_at": "2026-11-06T00:11:00Z",
            }
            for i in (1, 2)
        ]
        result = normalize_run(r, jobs, "nightly")
        self.assertEqual(result["completed_at"], "2026-11-06T00:11:00Z")
        newer = dict(result, attempt=2)
        self.assertEqual(
            len(merge_history([result], [newer], "2026-11-05T00:00:00Z")), 2
        )

    def test_incomplete_run_has_no_completion(self):
        result = normalize_run(
            {"id": 1, "status": "in_progress"},
            [{"completed_at": "2026-11-06T00:11:00Z"}],
            "pr",
        )
        self.assertIsNone(result["completed_at"])

    def test_submission_timestamp_uses_current_attempt(self):
        original = {
            "id": 1,
            "run_attempt": 1,
            "created_at": "2026-11-05T00:00:00Z",
            "run_started_at": "2026-11-06T01:00:00Z",
        }
        self.assertEqual(
            normalize_run(original, [], "pr")["submitted_at"], original["created_at"]
        )
        original["run_attempt"] = 2
        original["created_at"] = "2026-11-06T01:00:01Z"
        self.assertEqual(
            normalize_run(original, [], "pr")["submitted_at"], original["created_at"]
        )

    def test_build_failure_is_an_execution(self):
        jobs = [
            {
                "name": "build / build",
                "conclusion": "failure",
                "steps": [{"name": "Build", "started_at": "2026-11-06T00:00:00Z"}],
            }
        ]
        self.assertEqual(
            normalize_run({"id": 1}, jobs, "pr")["classification"], "executed"
        )

    def test_cross_host_redirect_strips_token(self):
        req = urllib.request.Request(
            "https://api.github.com/logs", headers={"Authorization": "Bearer private"}
        )
        redirected = SafeRedirect().redirect_request(
            req, None, 302, "", {}, "https://storage.example/logs"
        )
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_response_size_limit_applies_before_decoding(self):
        api = GitHub("owner/repo")
        response = MagicMock()
        response.read.return_value = b"123456"
        api.opener = MagicMock()
        api.opener.open.return_value.__enter__.return_value = response
        with self.assertRaisesRegex(ValueError, "download limit"):
            api.get("/actions/artifacts/1/zip", raw=True, max_bytes=5)
        response.read.assert_called_once_with(6)

    def test_invalid_test_duration_rejected(self):
        self.assertEqual(
            parse_tests(
                'TIMINGS BEGIN\n{"file":"x","elapsed":-1,"passed":true}\nTIMINGS END'
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
