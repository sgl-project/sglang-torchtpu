"""Retain one bounded allocation for the complete application and explicit recovery."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmark.tpu.glm52_baseline.session import atomic


def run(command=None, *, seconds=None, poll_seconds=2):
    root = Path(os.environ["ARTIFACT_LOCAL_DIR"])
    rank = int(os.environ["FALCON_RANK"])
    control = root / "allocation"
    control.mkdir(parents=True, exist_ok=True)
    status_file = control / f"rank-{rank}.json"
    deadline = time.monotonic() + (seconds or 6900)
    expires_at = time.time() + (seconds or 6900)
    command = command or ["bash", str(Path(__file__).with_name("falcon_runner.sh"))]
    generation = 1
    source = os.environ["SOURCE_COMMIT"]
    child = None
    handlers = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        handlers = {
            s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)
        }
        while time.monotonic() < deadline:
            attempt = root / "application-attempts" / str(generation)
            attempt.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                ARTIFACT_LOCAL_DIR=str(attempt),
                SOURCE_COMMIT=source,
                GLM52_RUN_ID=f"{os.environ.get('FALCON_EXP_ID', 'baseline')}-{generation}",
                GLM52_SESSION_SECONDS=str(max(1, deadline - time.monotonic() - 60)),
                GLM52_ALLOCATION_EXPIRES_AT=str(expires_at),
                GLM52_COMPILE_CACHE_ROOT=env.get(
                    "GLM52_COMPILE_CACHE_ROOT", str(root / "compile-cache")
                ),
            )
            status = dict(
                status="RUNNING",
                generation=generation,
                source=source,
                attempt_root=str(attempt),
                rank=rank,
                started_at=time.time(),
                expires_at=expires_at,
            )
            atomic(status_file, status)
            with (attempt / f"rank-{rank}-application.log").open(
                "ab", buffering=0
            ) as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while child.poll() is None and time.monotonic() < deadline:
                    time.sleep(poll_seconds)
            if child.poll() is None:
                break
            result_path = attempt / "glm52-e2e" / f"rank-{rank}" / "result.json"
            try:
                result = json.loads(result_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                result = {}
            if child.returncode == 0 and result.get("status") == "SUCCEEDED":
                outcome = {**status, "status": "SUCCEEDED", "result": result}
                atomic(attempt / f"rank-{rank}-outcome.json", outcome)
                atomic(status_file, outcome)
                return 0
            outcome = {
                **status,
                "status": "ATTENTION",
                "returncode": child.returncode,
                "result": result,
                "recovery": "fix the recorded failure, then write allocation/retry.json with a larger generation and exact source_commit",
            }
            atomic(attempt / f"rank-{rank}-outcome.json", outcome)
            atomic(status_file, outcome)
            # Failure retains the machine. Only an explicit recovery request starts
            # another application attempt; there is no automatic retry loop.
            while time.monotonic() < deadline:
                try:
                    retry = json.loads((control / "retry.json").read_text())
                    if int(retry["generation"]) > generation:
                        generation = int(retry["generation"])
                        source = retry["source_commit"]
                        break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                time.sleep(poll_seconds)
            else:
                break
        atomic(
            status_file,
            dict(
                status="TIMED_OUT",
                rank=rank,
                generation=generation,
                application_succeeded=False,
            ),
        )
        return 124
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=50)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(run())
