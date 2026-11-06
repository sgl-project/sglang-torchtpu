#!/usr/bin/env python3
"""Best-effort TPU phase telemetry, collected before the JobSet is deleted."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else None
    except (AttributeError, TypeError, ValueError):
        return None


def seconds(start, finish):
    first, last = timestamp(start), timestamp(finish)
    if first is None or last is None:
        return None
    elapsed = (last - first).total_seconds()
    return elapsed if math.isfinite(elapsed) and elapsed >= 0 else None


def parse_tests(log):
    tests = {}
    in_timings = False
    for line in log.splitlines():
        if "TIMINGS BEGIN" in line:
            in_timings = True
        elif "TIMINGS END" in line:
            in_timings = False
        elif in_timings:
            try:
                record = json.loads(line[line.index("{") :])
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            elapsed = record.get("elapsed")
            if (
                isinstance(record.get("file"), str)
                and record["file"]
                and isinstance(record.get("passed"), bool)
                and type(elapsed) in (int, float)
                and math.isfinite(elapsed)
                and elapsed >= 0
            ):
                # Re-attaching a stream can replay the same completed TIMINGS block.
                tests[record["file"]] = {
                    key: record[key] for key in ("file", "elapsed", "passed")
                }
    return list(tests.values())


def first_admission(workloads):
    admissions = [
        condition.get("lastTransitionTime")
        for workload in workloads.get("items", [])
        for condition in workload.get("status", {}).get("conditions", [])
        if condition.get("type") == "Admitted"
        and condition.get("status") == "True"
        and timestamp(condition.get("lastTransitionTime")) is not None
    ]
    return min(admissions, key=timestamp) if admissions else None


def read_admission(path):
    try:
        value = Path(path).read_text().strip() if path else None
        return value if timestamp(value) is not None else None
    except OSError:
        return None


def build_report(
    jobset,
    workloads,
    pods,
    log,
    *,
    job_name,
    suite,
    run_id,
    attempt,
    saved_admission=None,
):
    test_starts = [
        value
        for value in re.findall(r"TPU_CI_TEST_STARTED_AT (\S+)", log)
        if timestamp(value) is not None
    ]
    test_started = min(test_starts, key=timestamp) if test_starts else None
    created = jobset.get("metadata", {}).get("creationTimestamp")
    admitted = (
        saved_admission
        if timestamp(saved_admission) is not None
        else first_admission(workloads)
    )
    starts, finishes = [], []
    expected_pods = sum(
        job.get("replicas", 1)
        * job.get("template", {}).get("spec", {}).get("completions", 1)
        for job in jobset.get("spec", {}).get("replicatedJobs", [])
    )
    complete = expected_pods > 0 and len(pods.get("items", [])) >= expected_pods
    for pod in pods.get("items", []):
        containers = [
            container
            for container in pod.get("status", {}).get("containerStatuses", [])
            if container.get("name") == "test"
        ]
        if not containers:
            complete = False
        for container in containers:
            state = container.get("state", {})
            lifecycle = state.get("terminated") or state.get("running") or {}
            start, finish = lifecycle.get("startedAt"), lifecycle.get("finishedAt")
            if timestamp(start) is not None:
                starts.append(start)
            else:
                complete = False
            if timestamp(finish) is not None:
                finishes.append(finish)
            else:
                complete = False
    started = min(starts, key=timestamp) if starts else None
    finished = max(finishes, key=timestamp) if finishes and complete else None
    return {
        "schema_version": 1,
        "job_name": job_name,
        "run_id": str(run_id),
        "attempt": str(attempt),
        "suite": suite,
        "tpu": {
            "created_at": created if timestamp(created) is not None else None,
            "admitted_at": admitted,
            "test_started_at": test_started,
            "started_at": started,
            "finished_at": finished,
            "queue_seconds": seconds(created, admitted),
            "startup_seconds": seconds(admitted, started),
            "execution_seconds": seconds(started, finished),
        },
        "tests": parse_tests(log),
    }


def kubectl_json(*args):
    try:
        result = subprocess.run(
            [
                "kubectl",
                "--request-timeout=10s",
                "-n",
                "tpu-ci",
                "get",
                *args,
                "-o",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        value = json.loads(result.stdout)
        return value if isinstance(value, dict) else {}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"TPU timing unavailable: {error}", file=sys.stderr)
        return {}


def summary(report):
    print("### TPU timing\n")
    print("| Phase | Seconds |\n|---|---:|")
    for phase in ("queue", "startup", "execution"):
        value = report["tpu"][f"{phase}_seconds"]
        print(
            f"| {phase} | {value:.1f} |"
            if value is not None
            else f"| {phase} | unknown |"
        )
    if report["tests"]:
        print("\n| Test file | Seconds | Passed |\n|---|---:|---|")
        for test in report["tests"]:
            name = (
                test["file"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            )
            print(f"| {name} | {test['elapsed']:.1f} | {test['passed']} |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobset", default="")
    parser.add_argument("--pod-selector", default="")
    parser.add_argument("--workload-selector", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--output")
    parser.add_argument("--admission-file", default="")
    parser.add_argument(
        "--record-admission", help="Save the first observed true admission"
    )
    parser.add_argument(
        "--summary", help="Print a previously collected report as Markdown"
    )
    parser.add_argument("--job-name", default=os.environ.get("TPU_CI_JOB_NAME", "test"))
    parser.add_argument("--suite", default=os.environ.get("TPU_CI_SUITE", ""))
    args = parser.parse_args(argv)
    if args.summary:
        summary(json.loads(Path(args.summary).read_text()))
        return
    if args.record_admission:
        if read_admission(args.record_admission) is None and args.workload_selector:
            admitted = first_admission(
                kubectl_json("workloads", "-l", args.workload_selector)
            )
            if admitted is not None:
                Path(args.record_admission).write_text(admitted + "\n")
        return
    if not args.output:
        parser.error("--output is required when collecting")
    jobset = kubectl_json(args.jobset) if args.jobset else {}
    saved_admission = read_admission(args.admission_file)
    workloads = (
        kubectl_json("workloads", "-l", args.workload_selector)
        if args.workload_selector and saved_admission is None
        else {}
    )
    pods = kubectl_json("pods", "-l", args.pod_selector) if args.pod_selector else {}
    try:
        log = Path(args.log).read_text(errors="replace") if args.log else ""
    except OSError:
        log = ""
    report = build_report(
        jobset,
        workloads,
        pods,
        log,
        job_name=args.job_name,
        suite=args.suite,
        run_id=os.environ.get("GITHUB_RUN_ID", ""),
        attempt=os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        saved_admission=saved_admission,
    )
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
