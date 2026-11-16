#!/usr/bin/env python3
"""Collect TPU CI metadata and timing records. Python standard library only."""

import argparse
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Keep telemetry and historical collection on the same validation rules.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github" / "tpu-ci"))
from ci_timing import parse_tests, seconds

WORKFLOWS = {
    "pr-test-tpu.yml": "pr",
    "nightly-test-tpu.yml": "nightly",
    "feature-validation-tpu.yml": "nightly",
    "accuracy-test-tpu.yml": "nightly",
    "perf-test-tpu.yml": "nightly",
}
RETENTION_DAYS = 90


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_log_timing(log):
    """Old runner logs timestamp observation, not the original Kubernetes transition."""
    created = running = observed_test = exact_test = observed_finish = None
    for raw in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        match = re.match(r"^(\d{4}-\d\d-\d\dT\S+Z)\s+(.*)$", line)
        if not match:
            continue
        timestamp, text = match.groups()
        if created is None and re.match(r"Created jobset\.[^/]+/[a-z0-9-]+$", text):
            created = timestamp
        if (
            created
            and running is None
            and re.match(r"[a-z0-9-]+\s+\d+/\d+\s+Running\s", text)
        ):
            running = timestamp
        if re.search(r"\[pod/[^\]]+/test\]", text):
            if observed_finish is None and "TIMINGS END" in text:
                observed_finish = timestamp
            marker = re.search(r"TPU_CI_TEST_STARTED_AT (\S+)", text)
            if marker and seconds(marker[1], marker[1]) == 0:
                exact_test = exact_test or marker[1]
            if observed_test is None and (
                re.search(r"Enabled [1-9]\d* test\(s\)", text)
                or re.search(r"== test files: .+ ==", text)
            ):
                observed_test = timestamp
    return {
        "queue_seconds": None,
        "test_started_at": exact_test,
        "test_started_at_estimate": observed_test,
        "pod_started_at_estimate": running,
        "cluster_wait_startup_estimate_seconds": seconds(created, running),
        "execution_seconds_estimate": seconds(observed_test, observed_finish),
    }


def parse_workflow_selection(log):
    """Use actual plan output, never echoed shell source or delayed start time."""
    for raw in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        match = re.fullmatch(
            r"\d{4}-\d\d-\d\dT\S+Z\s+selection: ([a-zA-Z0-9_-]+)\s*", line
        )
        if match:
            return match[1]
    return None


def refresh_run_metadata(api, record, run, workflow, warnings):
    for source, target in (
        ("event", "event"),
        ("name", "workflow_name"),
        ("workflow_id", "workflow_id"),
    ):
        if run.get(source) is not None:
            record[target] = run[source]
    record.setdefault("workflow_selection", None)
    if workflow != "accuracy-test-tpu.yml" or record["workflow_selection"]:
        return
    plan = next(
        (job for job in record.get("jobs", []) if job.get("name") == "plan"), None
    )
    if plan is None:
        return
    try:
        log = api.get(f"/actions/jobs/{plan['id']}/logs", raw=True).decode(
            "utf-8", errors="replace"
        )
        record["workflow_selection"] = parse_workflow_selection(log)
        if record["workflow_selection"] is None:
            warnings.append(f"Run {record['id']}: accuracy plan selection unavailable")
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        warnings.append(
            f"Run {record['id']}: accuracy plan metadata unavailable "
            f"({getattr(error, 'code', type(error).__name__)})"
        )


def normalize_run(run, jobs, lane):
    normalized = []
    for job in jobs:
        normalized.append(
            {
                **{
                    k: job.get(k)
                    for k in (
                        "id",
                        "name",
                        "status",
                        "conclusion",
                        "created_at",
                        "started_at",
                        "completed_at",
                    )
                },
                "url": job.get("html_url"),
                "github_wait_seconds": None
                if job.get("conclusion") == "skipped"
                else seconds(job.get("created_at"), job.get("started_at")),
                "runtime_seconds": None
                if job.get("conclusion") == "skipped"
                else seconds(job.get("started_at"), job.get("completed_at")),
                "is_tpu_test": any(
                    s.get("name") == "Render + run on TPU" and s.get("started_at")
                    for s in job.get("steps", [])
                ),
                "tpu": None,
                "tests": [],
            }
        )
    awaiting = any(
        s.get("name") == "Require an explicit trigger"
        and s.get("conclusion") == "failure"
        for j in jobs
        for s in j.get("steps", [])
    )
    executed = any(
        s.get("started_at")
        for j in jobs
        if j.get("name") not in ("decide", "pr-test-tpu-finish")
        for s in j.get("steps", [])
    ) or any(
        j.get("conclusion") in ("failure", "timed_out", "action_required") for j in jobs
    )
    pr = next(iter(run.get("pull_requests") or []), {})
    return {
        "id": run["id"],
        "attempt": run.get("run_attempt", 1),
        "lane": lane,
        "name": run.get("display_title") or run.get("name"),
        "workflow_name": run.get("name"),
        "event": run.get("event"),
        "workflow_selection": None,
        "workflow_id": run.get("workflow_id"),
        "pr": pr.get("number"),
        "sha": run.get("head_sha"),
        "url": run.get("html_url"),
        "created_at": run.get("created_at"),
        "submitted_at": run.get("created_at"),
        "started_at": run.get("run_started_at"),
        "completed_at": max(
            (j.get("completed_at") for j in jobs if j.get("completed_at")), default=None
        )
        if run.get("status", "completed") == "completed"
        else None,
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "classification": "awaiting_trigger"
        if awaiting
        else ("executed" if executed else "skipped"),
        "jobs": normalized,
    }


def merge_history(old, new, cutoff):
    records = {(r["id"], r["attempt"]): r for r in old}
    records.update({(r["id"], r["attempt"]): r for r in new})
    return sorted(
        (
            r
            for r in records.values()
            if (
                r.get("completed_at")
                or r.get("started_at")
                or r.get("created_at")
                or ""
            )
            >= cutoff
        ),
        key=lambda r: r.get("started_at") or r.get("created_at") or "",
        reverse=True,
    )


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            redirected
            and urllib.parse.urlparse(req.full_url).netloc
            != urllib.parse.urlparse(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    def __init__(self, repo):
        self.base = f"https://api.github.com/repos/{repo}"
        self.token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        self.opener = urllib.request.build_opener(SafeRedirect())

    def get(self, path, raw=False, max_bytes=64_000_000):
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "tpu-ci-duration-dashboard",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(3):
            try:
                with self.opener.open(
                    urllib.request.Request(self.base + path, headers=headers),
                    timeout=60,
                ) as response:
                    payload = response.read(max_bytes + 1)
                    if len(payload) > max_bytes:
                        raise ValueError("GitHub response exceeds download limit")
                return payload if raw else json.loads(payload)
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == 2:
                    raise
                time.sleep(2**attempt)

    def pages(self, path, key):
        for page in range(1, 1001):
            rows = self.get(
                path + ("&" if "?" in path else "?") + f"per_page=100&page={page}"
            )[key]
            yield from rows
            if len(rows) < 100:
                break


def collect(api, days, old, run_id=None):
    result, warnings = [], []
    known = {(r["id"], r["attempt"]): r for r in old}
    try:
        for workflow, lane in WORKFLOWS.items():
            window_days = max(days, 3) if lane == "nightly" else days
            since = (
                (datetime.now(timezone.utc) - timedelta(days=window_days))
                .isoformat()
                .replace("+00:00", "Z")
            )
            # updated_at is not a completion timestamp; attempt/job timestamps are used below.
            runs = list(
                api.pages(
                    f"/actions/workflows/{workflow}/runs?created=%3E%3D{since}",
                    "workflow_runs",
                )
            )
            if run_id:
                triggered = api.get(f"/actions/runs/{run_id}")
                if triggered.get("path", "").split("@")[0].endswith(
                    "/" + workflow
                ) and all(r["id"] != triggered["id"] for r in runs):
                    runs.append(triggered)
            for run in runs:
                if lane == "nightly" and run.get("event") == "pull_request":
                    continue
                for attempt in range(1, run.get("run_attempt", 1) + 1):
                    key = (run["id"], attempt)
                    if key in known:
                        refresh_run_metadata(api, known[key], run, workflow, warnings)
                    if (
                        key in known
                        and known[key].get("status") == "completed"
                        and known[key].get("collection_complete")
                        and not (
                            known[key].get("classification") == "executed"
                            and known[key].get("timing_version") != 3
                        )
                    ):
                        continue
                    specific = api.get(f"/actions/runs/{run['id']}/attempts/{attempt}")
                    jobs = list(
                        api.pages(
                            f"/actions/runs/{run['id']}/attempts/{attempt}/jobs", "jobs"
                        )
                    )
                    record = normalize_run(specific, jobs, lane)
                    refresh_run_metadata(api, record, run, workflow, warnings)
                    complete = True
                    for job in record["jobs"]:
                        if job.get("runtime_seconds") is None or not any(
                            s.get("name") == "Render + run on TPU"
                            and s.get("started_at")
                            for j in jobs
                            if j.get("id") == job["id"]
                            for s in j.get("steps", [])
                        ):
                            continue
                        try:
                            log = api.get(
                                f"/actions/jobs/{job['id']}/logs", raw=True
                            ).decode("utf-8", errors="replace")
                            job["tests"] = parse_tests(log)
                            job["tpu"] = parse_log_timing(log)
                        except (urllib.error.HTTPError, ValueError) as error:
                            complete = False
                            warnings.append(
                                f"Job {job['id']}: logs unavailable ({getattr(error, 'code', type(error).__name__)})"
                            )
                    # Artifacts contain data only, never executable content. Read in-memory, bounded.
                    for artifact in api.pages(
                        f"/actions/runs/{run['id']}/artifacts", "artifacts"
                    ):
                        if (
                            not artifact["name"].startswith("tpu-ci-timing-")
                            or not artifact["name"].endswith(f"-{run['id']}-{attempt}")
                            or artifact.get("expired")
                        ):
                            continue
                        try:
                            if artifact.get("size_in_bytes", 0) > 10_000_000:
                                raise ValueError("Timing artifact exceeds size limit")
                            payload = api.get(
                                f"/actions/artifacts/{artifact['id']}/zip",
                                raw=True,
                                max_bytes=10_000_000,
                            )
                            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                                members = archive.infolist()
                                if (
                                    len(members) > 20
                                    or sum(m.file_size for m in members) > 4_000_000
                                ):
                                    raise ValueError(
                                        "Timing archive exceeds expansion limit"
                                    )
                                for member in members:
                                    if (
                                        member.file_size > 2_000_000
                                        or not member.filename.endswith(".json")
                                    ):
                                        continue
                                    timing = json.loads(archive.read(member))
                                    if (
                                        not isinstance(timing, dict)
                                        or timing.get("schema_version") != 1
                                    ):
                                        continue
                                    if (
                                        str(timing.get("run_id")) != str(run["id"])
                                        or int(timing.get("attempt", 0)) != attempt
                                    ):
                                        continue
                                    for job in record["jobs"]:
                                        if job.get("name") == timing.get(
                                            "job_name"
                                        ) or (job.get("name") or "").endswith(
                                            " / " + str(timing.get("job_name"))
                                        ):
                                            tpu = timing.get("tpu") or {}
                                            if not isinstance(tpu, dict):
                                                continue
                                            log_timing = job["tpu"] or {}
                                            job["tpu"] = {
                                                k: v
                                                if type(v) in (int, float)
                                                and math.isfinite(v)
                                                and v >= 0
                                                else None
                                                for k, v in (
                                                    (k, tpu.get(k))
                                                    for k in (
                                                        "queue_seconds",
                                                        "startup_seconds",
                                                        "execution_seconds",
                                                    )
                                                )
                                            }
                                            job["tpu"].update(
                                                {
                                                    k: log_timing.get(k)
                                                    for k in (
                                                        "test_started_at_estimate",
                                                        "cluster_wait_startup_estimate_seconds",
                                                        "execution_seconds_estimate",
                                                        "pod_started_at_estimate",
                                                    )
                                                }
                                            )
                                            test_started = tpu.get(
                                                "test_started_at"
                                            ) or log_timing.get("test_started_at")
                                            job["tpu"]["test_started_at"] = (
                                                test_started
                                                if seconds(test_started, test_started)
                                                == 0
                                                else None
                                            )
                                            job["tpu"]["started_at"] = (
                                                tpu.get("started_at")
                                                if seconds(
                                                    tpu.get("started_at"),
                                                    tpu.get("started_at"),
                                                )
                                                == 0
                                                else None
                                            )
                                            job["tests"] = (
                                                parse_tests(
                                                    "TIMINGS BEGIN\n"
                                                    + "\n".join(
                                                        json.dumps(t)
                                                        for t in timing.get("tests", [])
                                                    )
                                                    + "\nTIMINGS END"
                                                )
                                                or job["tests"]
                                            )
                        except (
                            urllib.error.HTTPError,
                            ValueError,
                            KeyError,
                            zipfile.BadZipFile,
                            TypeError,
                        ) as error:
                            complete = False
                            warnings.append(
                                f"Artifact {artifact['id']}: {type(error).__name__}"
                            )
                    print(
                        f"{workflow}: {run['id']} attempt {attempt}, {len(jobs)} jobs",
                        flush=True,
                    )
                    record["collection_complete"] = complete
                    record["timing_version"] = 3
                    result.append(record)
    except urllib.error.HTTPError as error:
        headers = error.headers or {}
        if error.code != 429 and not (
            error.code == 403
            and (
                headers.get("X-RateLimit-Remaining") == "0"
                or headers.get("Retry-After")
            )
        ):
            raise
        warnings.append(
            "GitHub API rate limit reached. Completed attempts were saved. Retry after the quota resets."
        )
    return result, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="sgl-project/sglang-torchtpu")
    parser.add_argument(
        "--run-id",
        type=int,
        help="Always reconcile this run, including reruns of old PRs",
    )
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.days <= RETENTION_DAYS:
        parser.error(f"--days must be between 1 and {RETENTION_DAYS}")
    old = (
        json.loads(args.history.read_text())["runs"]
        if args.history and args.history.exists()
        else []
    )
    records, warnings = collect(GitHub(args.repo), args.days, old, args.run_id)
    # 90 days is the longest range the page offers and keeps data.json (about
    # 20 KB per executed attempt, fetched whole by every page view and committed
    # to ci-metrics) well under GitHub's 100 MB push limit.
    cutoff = (
        (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS))
        .isoformat()
        .replace("+00:00", "Z")
    )
    data = {
        "schema_version": 1,
        "generated_at": iso_now(),
        "runs": merge_history(old, records, cutoff),
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        f"Collected {len(records)} attempts; {len(data['runs'])} retained; {len(warnings)} warnings"
    )


if __name__ == "__main__":
    main()
