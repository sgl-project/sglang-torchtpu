#!/usr/bin/env python3
"""Resolve the latest successfully built SGLang TPU container image from nightly CI runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys


def _gh_api(path: str, token: str | None = None) -> bytes:
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    return subprocess.check_output(["gh", "api", path], env=env)


def _image_digest_pattern() -> str:
    """Digest-pinned image reference printed by the TPU build job."""
    image = os.environ.get("TPU_AR_IMAGE")
    if image:
        return re.escape(image) + r"@sha256:[a-f0-9]+"
    return r"[a-z0-9-]+-docker\.pkg\.dev/[\w.-]+/[\w.-]+/sglang-tpu@sha256:[a-f0-9]+"


def resolve_image(
    repo: str, token: str | None = None, workflow_file: str = "nightly-test-tpu.yml"
) -> str:
    runs_path = f"repos/{repo}/actions/workflows/{workflow_file}/runs?per_page=10"
    runs_raw = _gh_api(runs_path, token).decode("utf-8")
    data = json.loads(runs_raw)
    runs = data.get("workflow_runs", [])

    for run in runs:
        run_id = run.get("id")
        jobs_path = f"repos/{repo}/actions/runs/{run_id}/jobs"
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

            job_id = build_job.get("id")
            log_path = f"repos/{repo}/actions/jobs/{job_id}/logs"
            log_text = _gh_api(log_path, token).decode("utf-8", errors="ignore")
            pattern = _image_digest_pattern()
            matches = re.findall(pattern, log_text)
            if matches:
                return matches[-1]
        except Exception as e:
            print(f"debug: skipping run {run_id}: {e}", file=sys.stderr)
            continue

    raise RuntimeError(
        f"Could not find any recent successful build image in {repo} for {workflow_file}"
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
        "--workflow",
        default="nightly-test-tpu.yml",
        help="Workflow file name to search",
    )
    args = parser.parse_args()

    try:
        image = resolve_image(args.repo, args.token, args.workflow)
        print(image)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
