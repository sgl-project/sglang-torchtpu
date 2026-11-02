"""Check verification completion and process cleanup without TPU hardware."""

import json
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.testing.spmd_safe import verify_startup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_PUBLISH_RESULTS = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

mode = sys.argv[1]
directory = Path(os.environ["VERIFY_SPMD_SAFE_DUMP_DIR"])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
(directory / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
if mode == "crash":
    raise SystemExit(2)
for rank in range(2):
    record = {
        "rank": 0 if mode == "duplicate" else rank,
        "world_size": 2,
        "keys": ["different" if mode == "keys" and rank == 1 else "model"],
        "passed": mode != "mismatch",
        "errors": ["MLIR differs across ranks"] if mode == "mismatch" else [],
    }
    marker = directory / f"worker-{rank}.json.tmp"
    marker.write_text("{")
    # An incomplete temporary result must never authorize launcher cleanup.
    time.sleep(0.2)
    marker.write_text(json.dumps(record))
    if mode == "missing" and rank == 1:
        break
    marker.replace(directory / f"worker-{rank}.json")
time.sleep(30)
"""


def _running(pid: int) -> bool:
    try:
        # An orphaned child can remain a zombie until the container's init reaps it.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize(
    "mode,expected_error",
    [
        ("matching", None),
        ("mismatch", "MLIR differs across ranks"),
        ("keys", "inconsistent results"),
        ("duplicate", "distinct local ranks"),
        ("missing", "Verification startup exceeded"),
        ("crash", "Server launcher exited with code 2"),
    ],
)
def test_runner_finishes_on_results_and_cleans_up(tmp_path, mode, expected_error):
    with (
        pytest.raises(RuntimeError, match="SPMD verification failed")
        if expected_error
        else nullcontext()
    ):
        verify_startup(
            [sys.executable, "-S", "-c", _PUBLISH_RESULTS, mode],
            world_size=2,
            local_world_size=2,
            timeout=2,
            dump_dir=tmp_path,
        )
    directory = next(tmp_path.glob("spmd-*"))
    result = json.loads((directory / "result.json").read_text())
    assert result["passed"] == (expected_error is None)
    if expected_error:
        assert expected_error in " ".join(result["errors"])
    else:
        assert result["ranks"] == [0, 1]
        assert result["keys"] == ["model"]
    assert result["terminated_after_verification"] == (mode not in ("missing", "crash"))
    assert result["launcher_exit_code"] == (2 if mode == "crash" else -signal.SIGKILL)
    pids = json.loads((directory / "pids.json").read_text())
    deadline = time.monotonic() + 2
    while any(_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not any(_running(pid) for pid in pids)
