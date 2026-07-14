"""Emit a reproducibility metadata record for a TPU launch/bench run.

Shared by launch_server.sh (kind=launch) and bench_serving.sh (kind=bench).
Writing this in Python (not bash) keeps JSON escaping correct and lets us read
package versions straight from the interpreter that actually serves/benches --
so the record reflects the real environment, not a hand-maintained list.

The output is a single JSON object with a stable, versioned schema so a CI
dashboard can ingest launch_meta.json + <tag>.meta.json + the bench_serving
metrics <tag>.jsonl and join them by (hostname, model, git commit, timestamp).

Usage:
  dump_meta.py --kind launch --out PATH \
      --set model=Qwen/Qwen3-32B --set tp_size=8 ... \
      --json server_argv='["-m","sglang.launch_server",...]' \
      --env LIBTPU_INIT_ARGS --env XLA_FLAGS --env-prefix SGLANG_

  dump_meta.py --kind bench --out PATH \
      --set concurrency=64 --set isl=1024 ... \
      --json bench_argv='[...]' \
      --server-info-url http://127.0.0.1:30000

--set values are strings (ints/floats/bools coerced); --json values are parsed
as JSON (for arrays/objects). Nothing here is allowed to abort the run: every
enrichment step is best-effort and records an "_error" note on failure.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys

SCHEMA_VERSION = 1


def _coerce(value: str):
    """Turn a --set string into int/float/bool/None when it looks like one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("", "none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _git_info(start_path: str):
    """Return {commit, short, branch, dirty, remote} for the repo containing
    start_path, or None if start_path is not inside a git checkout."""
    if not start_path:
        return None
    d = start_path if os.path.isdir(start_path) else os.path.dirname(start_path)

    def _git(*args):
        return subprocess.run(
            ["git", "-C", d, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )

    root = _git("rev-parse", "--show-toplevel")
    if root.returncode != 0:
        return None
    info = {"root": root.stdout.strip()}
    commit = _git("rev-parse", "HEAD")
    if commit.returncode == 0:
        info["commit"] = commit.stdout.strip()
        info["short"] = info["commit"][:12]
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch.returncode == 0:
        info["branch"] = branch.stdout.strip()
    status = _git("status", "--porcelain")
    info["dirty"] = bool(status.returncode == 0 and status.stdout.strip())
    remote = _git("config", "--get", "remote.origin.url")
    if remote.returncode == 0 and remote.stdout.strip():
        info["remote"] = remote.stdout.strip()
    return info


def _module_git(module_name: str):
    """git_info for an installed package's source dir (editable/dev installs)."""
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    return _git_info(getattr(mod, "__file__", "") or "")


def _pkg_version(name: str):
    """Best-effort installed version for a distribution name."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _collect_git():
    git = {}
    # This repo, located from the script itself (benchmark/tpu/ -> repo root).
    repo = _git_info(os.path.abspath(__file__))
    if repo:
        git["sglang_torchtpu"] = repo
    # torch_tpu is a separate package; capture its commit only if dev-installed.
    tt = _module_git("torch_tpu")
    if tt and tt.get("root") != (repo or {}).get("root"):
        git["torch_tpu"] = tt
    return git


def _collect_versions():
    versions = {"python": platform.python_version()}
    # Import-based (authoritative for the running interpreter).
    for mod_name, key in (
        ("torch", "torch"),
        ("torch_tpu", "torch_tpu"),
        ("sglang", "sglang"),
        ("jax", "jax"),
        ("jaxlib", "jaxlib"),
        ("ml_dtypes", "ml_dtypes"),
        ("transformers", "transformers"),
    ):
        try:
            versions[key] = getattr(__import__(mod_name), "__version__", None)
        except Exception:
            versions[key] = None
    # Distribution-based (no import side effects; libtpu has no __version__).
    for dist, key in (
        ("libtpu", "libtpu"),
        ("libtpu-nightly", "libtpu_nightly"),
        ("sgl-kernel", "sgl_kernel"),
    ):
        v = _pkg_version(dist)
        if v is not None:
            versions[key] = v
    return versions


def _fetch_server_info(url: str):
    """Pull the live server config from /server_info (falls back to the
    deprecated /get_server_info). This is the authoritative record of what the
    server is actually running, independent of the launch script's view."""
    import urllib.request

    base = url.rstrip("/")
    for path in ("/server_info", "/get_server_info"):
        try:
            with urllib.request.urlopen(base + path, timeout=15) as resp:
                if resp.status != 200:
                    continue
                data = json.loads(resp.read().decode())
            # internal_states is large per-DP runtime state, not config; drop it.
            if isinstance(data, dict):
                data.pop("internal_states", None)
            return data
        except Exception:
            continue
    return {"_error": f"could not reach {base}/server_info"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", required=True, choices=["launch", "bench"])
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="scalar config field (repeatable)",
    )
    ap.add_argument(
        "--json",
        action="append",
        default=[],
        metavar="KEY=JSON",
        dest="json_fields",
        help="config field whose value is parsed as JSON (repeatable)",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME",
        help="snapshot this env var if set (repeatable)",
    )
    ap.add_argument(
        "--env-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="snapshot all env vars with this prefix (repeatable)",
    )
    ap.add_argument(
        "--server-info-url",
        default=None,
        help="bench: pull live config from this server's /server_info",
    )
    args = ap.parse_args()

    config = {}
    for item in args.set:
        key, _, value = item.partition("=")
        config[key] = _coerce(value)
    for item in args.json_fields:
        key, _, value = item.partition("=")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            config[key] = {"_error": f"invalid json: {exc}"}

    env = {}
    for name in args.env:
        if name in os.environ:
            env[name] = os.environ[name]
    for prefix in args.env_prefix:
        for name, value in os.environ.items():
            if name.startswith(prefix):
                env[name] = value

    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": args.kind,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "hostname": socket.gethostname(),
        "git": _collect_git(),
        "versions": _collect_versions(),
        "config": config,
        "env": env,
    }
    if args.server_info_url:
        record["server_info"] = _fetch_server_info(args.server_info_url)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"[dump_meta] wrote {args.kind} metadata -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
