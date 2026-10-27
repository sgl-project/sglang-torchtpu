#!/usr/bin/env python3
"""Recalibrate TPU est_time literals from recent nightly runs.

The weekly est_time bot (scripts/ci/update_est_time.py) reads upstream's
sglang-ci-stats, which has no TPU data, so TPU estimates only ever move by
hand. They matter more here than on GPU: _tpu-test.yml books the runtime from
est_time (or a literal timeout_per_file) and queues for a slice with whatever
is left of the runner's 6h, so an over-stated estimate turns into NOT ADMITTED.

Samples come from successful jobs of recent scheduled runs on main:
  - run_suite's TIMINGS block: {"file": ..., "passed": true, "elapsed": N}
  - multi-host live-server segments: one "== live-server test: <file> ==" marker
    to the next (or the last pod line), server bring-up excluded.

New estimate = p90 x MARGIN, rounded up to 10s, applied only when it
moves by at least MIN_DELTA_S and MIN_DELTA_REL. A literal timeout_per_file
moves with it only where it equals est_time today; anything else is a
deliberate figure (e.g. the bench matrix's budget chain) and is left alone.

Stdlib only: lint runs .github/tpu-ci/tests with `python -E -S -B`.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO = os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang-torchtpu")
# file[:event]; schedule samples main only, pull_request samples the PR gate's branches.
WORKFLOWS = (
    "nightly-test-tpu.yml:schedule",
    "feature-validation-tpu.yml:schedule",
    "accuracy-test-tpu.yml:schedule",
    "perf-test-tpu.yml:schedule",
    "pr-test-tpu.yml:pull_request",
)
MARGIN = 1.25
MIN_SAMPLES = 3
MIN_DELTA_S = 60
MIN_DELTA_REL = 0.10

_TIMINGS_BEGIN = "========== TIMINGS BEGIN =========="
_TIMINGS_END = "========== TIMINGS END =========="
_LIVE_SERVER_RE = re.compile(r"== live-server test: (\S+) ==")
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
_POD_TEST_LINE = re.compile(r"\[pod/[^\]]+/test\]")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# --- log parsing -------------------------------------------------------------
def _ts(line: str) -> Optional[datetime]:
    m = _TS_RE.match(line)
    return datetime.fromisoformat(m.group(1)) if m else None


def timings_samples(log: str) -> List[Tuple[str, int]]:
    """(file, elapsed) for every passed file in run_suite's TIMINGS block."""
    out: Dict[str, int] = {}
    inside = False
    for raw in _ANSI_RE.sub("", log).splitlines():
        line = raw[raw.find("{") :] if "{" in raw else raw
        if _TIMINGS_BEGIN in raw:
            inside = True
        elif _TIMINGS_END in raw:
            inside = False
        elif inside and line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("passed") and "file" in rec and "elapsed" in rec:
                out[rec["file"]] = int(rec["elapsed"])  # later attempts win
    return sorted(out.items())


def live_server_samples(log: str) -> List[Tuple[str, int]]:
    """(file, seconds) per multi-host live-server segment of a successful job."""
    lines = _ANSI_RE.sub("", log).splitlines()
    marks = []
    for line in lines:
        m = _LIVE_SERVER_RE.search(line)
        ts = _ts(line)
        if m and ts:
            marks.append((m.group(1), ts))
    if not marks:
        return []
    last = None
    for line in reversed(lines):
        if _POD_TEST_LINE.search(line) and _ts(line):
            last = _ts(line)
            break
    out = []
    for i, (path, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else last
        if end and end > start:
            out.append((path, int((end - start).total_seconds())))
    return out


# --- estimates -----------------------------------------------------------------
def p90(samples: List[int]) -> int:
    s = sorted(samples)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def propose(old: int, samples: List[int]) -> Optional[int]:
    """New est_time, or None when the data is thin or the move is noise."""
    if len(samples) < MIN_SAMPLES:
        return None
    new = max(10, math.ceil(p90(samples) * MARGIN / 10) * 10)
    if abs(new - old) < max(MIN_DELTA_S, MIN_DELTA_REL * old):
        return None
    return new


# --- source rewriting -----------------------------------------------------------
_CALL_RE = re.compile(r"\bregister_tpu_ci\s*\(")


def _call_span(src: str, start: int) -> Optional[Tuple[int, int]]:
    """(open, close) paren indices of the call that starts at `start`."""
    open_i = src.index("(", start)
    depth = 0
    for i in range(open_i, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return open_i, i
    return None


def current_est(src: str) -> Optional[int]:
    """est_time of the file's single register_tpu_ci call."""
    hits = [m.start() for m in _CALL_RE.finditer(src)]
    if len(hits) != 1:
        return None  # several suites share the file: a sample can't pick one
    span = _call_span(src, hits[0])
    if not span:
        return None
    m = re.search(r"\best_time\s*=\s*(\d+)", src[span[0] : span[1]])
    return int(m.group(1)) if m else None


def rewrite(src: str, new: int) -> str:
    """Set est_time, and a timeout_per_file that mirrors it, to `new`."""
    old = current_est(src)
    if old is None:
        return src
    hit = _CALL_RE.search(src).start()
    a, b = _call_span(src, hit)
    call = re.sub(r"(\best_time\s*=\s*)\d+", rf"\g<1>{new}", src[a:b], count=1)
    src = src[:a] + call + src[b:]
    return re.sub(rf'("timeout_per_file"\s*:\s*){old}\b', rf"\g<1>{new}", src, count=1)


# --- GitHub ----------------------------------------------------------------------
def _get(url: str, token: str, raw: bool = False):
    req = urllib.request.Request(url)
    # Unredirected: the log endpoint 302s to signed blob storage, which rejects
    # a forwarded Authorization header.
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(5 * 2**attempt)
    return body.decode("utf-8", "replace") if raw else json.loads(body)


def collect(
    token: str, runs_per_workflow: int, workflows: Iterable[str] = WORKFLOWS
) -> Dict[str, List[int]]:
    api = f"https://api.github.com/repos/{REPO}/actions"
    samples: Dict[str, List[int]] = defaultdict(list)
    for spec in workflows:
        wf, _, event = spec.partition(":")
        event = event or "schedule"
        branch = "&branch=main" if event == "schedule" else ""
        runs = _get(
            f"{api}/workflows/{wf}/runs?event={event}{branch}"
            f"&status=completed&per_page={runs_per_workflow}",
            token,
        ).get("workflow_runs", [])
        for run in runs:
            page = 1
            while True:
                jobs = _get(
                    f"{api}/runs/{run['id']}/jobs?per_page=100&page={page}", token
                )["jobs"]
                for job in jobs:
                    if job.get("conclusion") != "success":
                        continue
                    try:
                        log = _get(f"{api}/jobs/{job['id']}/logs", token, raw=True)
                    except Exception as e:  # one unreadable log is not fatal
                        print(f"  skip job {job['id']}: {e}", file=sys.stderr)
                        continue
                    for path, sec in timings_samples(log) + live_server_samples(log):
                        samples[path].append(sec)
                if len(jobs) < 100:
                    break
                page += 1
        print(f"{spec}: {len(runs)} runs scanned", file=sys.stderr)
    return samples


def plan(
    samples: Dict[str, List[int]], root: Path, only: Iterable[str] = ()
) -> List[Tuple[str, int, int, List[int]]]:
    only = set(only)
    out = []
    for path, secs in sorted(samples.items()):
        f = root / path
        if (only and path not in only) or not f.is_file():
            continue
        old = current_est(f.read_text())
        new = propose(old, secs) if old is not None else None
        if new is not None:
            out.append((path, old, new, secs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=15, help="recent runs per workflow")
    ap.add_argument(
        "--workflows", nargs="+", default=WORKFLOWS, help="workflow file[:event]"
    )
    ap.add_argument("--root", default=".", help="repo root to rewrite")
    ap.add_argument("--only", nargs="*", default=(), help="limit to these paths")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-file", default="")
    ap.add_argument("--samples", default="", help="JSON cache of collected samples")
    a = ap.parse_args()

    if a.samples and Path(a.samples).is_file():
        samples = json.loads(Path(a.samples).read_text())
    else:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            sys.exit("GITHUB_TOKEN (or GH_TOKEN) is required to read run logs")
        samples = collect(token, a.runs, a.workflows)
        if a.samples:
            Path(a.samples).write_text(json.dumps(samples, indent=1, sort_keys=True))

    root = Path(a.root)
    changes = plan(samples, root, a.only)
    rows = []
    for path, old, new, secs in changes:
        rows.append(
            f"| `{Path(path).name}` | {old} | {new} | {p90(secs)} | {len(secs)} |"
        )
        print(f"{path}: est_time {old} -> {new} (p90 {p90(secs)}s, n={len(secs)})")
        if not a.dry_run:
            f = root / path
            f.write_text(rewrite(f.read_text(), new))
    if a.summary_file and rows:
        Path(a.summary_file).write_text(
            f"### TPU est_time ({len(rows)} files; p90 x {MARGIN} of passing runs)\n\n"
            "| File | Old (s) | New (s) | p90 (s) | n |\n"
            "| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(rows) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
