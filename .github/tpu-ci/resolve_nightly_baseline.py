#!/usr/bin/env python3
"""Resolve the latest nightly run on main: extracts its container image digest and job-level pass/fail results."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys


def _gh_api(path: str, token: str | None = None, allow_escape: bool = False) -> bytes:
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    cmd = ["gh", "api", path]
    if allow_escape:
        cmd.append("--allow-escape-sequences")
    try:
        return subprocess.check_output(cmd, env=env, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        if b"unknown flag" in e.stderr and allow_escape:
            return subprocess.check_output(["gh", "api", path], env=env)
        if b"--allow-escape-sequences" in e.stderr:
            return subprocess.check_output(
                ["gh", "api", path, "--allow-escape-sequences"], env=env
            )
        raise


def _image_digest_pattern() -> str:
    """Digest-pinned image reference printed by the TPU build job."""
    image = os.environ.get("TPU_AR_IMAGE")
    if image:
        return re.escape(image) + r"@sha256:[a-f0-9]+"
    return r"[a-z0-9-]+-docker\.pkg\.dev/[\w.-]+/[\w.-]+/sglang-tpu@sha256:[a-f0-9]+"


def resolve_baseline(
    repo: str,
    token: str | None = None,
    branch: str = "main",
    workflow_file: str = "nightly-test-tpu.yml",
) -> dict:
    runs_path = f"repos/{repo}/actions/workflows/{workflow_file}/runs?branch={branch}&per_page=10"
    runs_raw = _gh_api(runs_path, token).decode("utf-8")
    data = json.loads(runs_raw)
    runs = data.get("workflow_runs", [])

    for run in runs:
        run_id = run.get("id")
        jobs_path = f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
        try:
            jobs_raw = _gh_api(jobs_path, token).decode("utf-8")
            jobs_data = json.loads(jobs_raw)
            jobs = jobs_data.get("jobs", [])
            build_job = next(
                (
                    j
                    for j in jobs
                    if j.get("name") == "build / build"
                    and j.get("conclusion") == "success"
                ),
                None,
            )
            if not build_job:
                continue

            # Extract image digest from build logs
            job_id = build_job.get("id")
            log_path = f"repos/{repo}/actions/jobs/{job_id}/logs"
            log_text = _gh_api(log_path, token, allow_escape=True).decode(
                "utf-8", errors="ignore"
            )
            pattern = _image_digest_pattern()
            matches = re.findall(pattern, log_text)
            if not matches:
                continue

            image_digest = matches[-1]

            # Build map of job conclusions
            job_results = {}
            for j in jobs:
                name = j.get("name")
                conclusion = j.get("conclusion")
                if name and conclusion and conclusion != "skipped":
                    job_results[name] = conclusion

            return {
                "run_id": run_id,
                "run_url": run.get(
                    "html_url", f"https://github.com/{repo}/actions/runs/{run_id}"
                ),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha"),
                "branch": branch,
                "image": image_digest,
                "jobs": job_results,
            }
        except Exception as e:
            print(f"debug: skipping run {run_id}: {e}", file=sys.stderr)
            continue

    raise RuntimeError(
        f"Could not find any recent successful build image in {repo} for {workflow_file} on branch {branch}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang-torchtpu"),
        help="GitHub repository (owner/repo)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        help="GitHub API token",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to inspect (default: main)",
    )
    parser.add_argument(
        "--workflow",
        default="nightly-test-tpu.yml",
        help="Workflow file name to search",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Path to save baseline JSON data",
    )
    args = parser.parse_args()

    try:
        baseline = resolve_baseline(args.repo, args.token, args.branch, args.workflow)
        if args.output_json:
            with open(args.output_json, "w") as fp:
                json.dump(baseline, fp, indent=2)
            print(f"Saved baseline metadata to {args.output_json}", file=sys.stderr)

        # Print image to stdout so workflow step can capture it
        print(baseline["image"])
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
