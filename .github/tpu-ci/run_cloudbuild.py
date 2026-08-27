#!/usr/bin/env python3
"""Create and run a Cloud Build via the REST API, print the pushed image digest.

`gcloud builds submit` runs a client-side staging / service-usage pre-flight that is
incompatible with Workload Identity Federation credentials: it fails accessing the
``<project>_cloudbuild`` bucket even though the service account can write it (proven
by `gcloud storage cp` succeeding with the same token). This talks to the Cloud Build
REST API directly instead -- the source is pre-uploaded to GCS by the caller and we
POST a Build that points at it, so that broken pre-flight never runs.

Progress goes to stderr; the bare image digest (``sha256:...``) is the ONLY thing
printed to stdout, so callers can capture it with ``digest="$(... )"``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

API = "https://cloudbuild.googleapis.com/v1"
_TERMINAL = {"SUCCESS", "FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED", "EXPIRED"}


def _log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def _load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:  # not guaranteed on the runner
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True
        )
        import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--config", required=True, help="Cloud Build config YAML")
    ap.add_argument("--bucket", required=True, help="GCS bucket of the staged source")
    ap.add_argument("--object", required=True, help="GCS object of the staged source")
    ap.add_argument("--tag", required=True, help="value for the _TAG substitution")
    ap.add_argument(
        "--substitutions",
        nargs="*",
        default=[],
        help="additional substitutions in KEY=VALUE format",
    )
    ap.add_argument("--token-file", required=True, help="file holding a bearer token")
    ap.add_argument("--timeout", default="3600s", help="build timeout")
    args = ap.parse_args()

    with open(args.token_file) as f:
        token = f.read().strip()

    def call(method: str, url: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.load(e)
            except Exception:
                return e.code, {"error": e.read().decode()[:500]}

    cfg = _load_config(args.config)
    subs = dict(cfg.get("substitutions", {}))
    subs["_TAG"] = args.tag
    for item in args.substitutions:
        if "=" in item:
            k, v = item.split("=", 1)
            subs[k] = v
    build = {
        "source": {"storageSource": {"bucket": args.bucket, "object": args.object}},
        "steps": cfg["steps"],
        "images": cfg.get("images", []),
        "options": cfg.get("options", {}),
        "substitutions": subs,
        "timeout": args.timeout,
    }
    if "serviceAccount" in cfg:  # run the build as the configured SA (tpu-ci)
        build["serviceAccount"] = cfg["serviceAccount"]

    _log(f"builds.create (source gs://{args.bucket}/{args.object}, _TAG={args.tag})")
    code, resp = call("POST", f"{API}/projects/{args.project}/builds", build)
    if code != 200:
        _log(f"builds.create failed: HTTP {code}")
        _log(json.dumps(resp, indent=2)[:2000])
        sys.exit(1)
    build_id = resp["metadata"]["build"]["id"]
    _log(
        f"build {build_id} created; logs: {resp['metadata']['build'].get('logUrl', '?')}"
    )

    status = last = None
    b: dict = {}
    for _ in range(240):  # 240 * 15s = up to 60 min
        code, b = call("GET", f"{API}/projects/{args.project}/builds/{build_id}")
        if code != 200:
            _log(f"builds.get HTTP {code}: {json.dumps(b)[:300]}")
            time.sleep(15)
            continue
        status = b.get("status")
        if status != last:
            _log(f"  status: {status}")
            last = status
        if status in _TERMINAL:
            break
        time.sleep(15)

    if status != "SUCCESS":
        _log(f"BUILD {status}")
        if b.get("failureInfo"):
            _log("failureInfo: " + json.dumps(b["failureInfo"]))
        _log("logUrl: " + b.get("logUrl", "?"))
        sys.exit(1)

    images = b.get("results", {}).get("images", [])
    digest = images[0]["digest"] if images else ""
    if not digest:
        _log("build SUCCESS but no image digest in results")
        sys.exit(1)
    _log(f"SUCCESS: {images[0].get('name', '')}@{digest}")
    print(digest)  # stdout: bare digest only


if __name__ == "__main__":
    main()
