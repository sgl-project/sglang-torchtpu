"""The remote-wheel baseline must be a completed scheduled nightly run."""

import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resolve_nightly_baseline  # noqa: E402

IMAGE = "us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu"
BUILD = "build / build"


def run(run_id, event, status, conclusion):
    return {"id": run_id, "event": event, "status": status, "conclusion": conclusion}


# Newest first, as the runs API lists them.
RUNS = [
    run(3, "schedule", "in_progress", None),  # today's nightly, still running
    run(2, "workflow_dispatch", "completed", "success"),  # manual perf-only run
    run(1, "schedule", "completed", "failure"),  # yesterday's nightly
]
JOBS = {
    3: {BUILD: "success", "perf": "success", "sh": None, "mh": None},
    2: {BUILD: "success", "perf": "success"},
    1: {BUILD: "success", "perf": "success", "sh": "success", "mh": "failure"},
}


def fake_gh_api(path, token=None, allow_escape=False):
    url = urlsplit(path)
    query = {key: values[0] for key, values in parse_qs(url.query).items()}
    if url.path.endswith("/runs"):
        # The REST API matches `status` against a run's status or its conclusion.
        runs = [
            r
            for r in RUNS
            if query.get("event", r["event"]) == r["event"]
            and query.get("status", r["status"]) in (r["status"], r["conclusion"])
        ]
        return json.dumps({"workflow_runs": runs}).encode()
    # Every job carries its run's id, so the build log maps back to the run.
    run_id = int(re.search(r"/(\d+)/(jobs|logs)$", url.path).group(1))
    if url.path.endswith("/jobs"):
        jobs = [
            {"id": run_id, "name": name, "conclusion": conclusion}
            for name, conclusion in JOBS[run_id].items()
        ]
        return json.dumps({"jobs": jobs}).encode()
    return f"pushed {IMAGE}@sha256:{run_id:064x}\n".encode()


class TestResolveBaseline(unittest.TestCase):
    def test_skips_running_and_manually_dispatched_runs(self):
        with patch.object(resolve_nightly_baseline, "_gh_api", fake_gh_api):
            baseline = resolve_nightly_baseline.resolve_baseline("owner/repo")
        self.assertEqual(baseline["run_id"], 1)
        self.assertEqual(baseline["image"], f"{IMAGE}@sha256:{1:064x}")
        self.assertEqual(baseline["jobs"], JOBS[1])


if __name__ == "__main__":
    unittest.main()
