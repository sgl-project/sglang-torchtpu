"""CPU-only retained-allocation recovery and false-success regression checks."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from benchmark.tpu.glm52_baseline.allocation import run

if "--child" in sys.argv:
    raise SystemExit(
        run(
            [sys.executable, os.environ["STUB"]],
            seconds=float(os.environ.get("SECONDS_TEST", "10")),
            poll_seconds=0.02,
        )
    )
with tempfile.TemporaryDirectory(prefix="glm52-allocation-preflight-") as d:
    root = pathlib.Path(d)
    stub = root / "app.py"
    stub.write_text("""import os,json,pathlib,sys
n=int(os.environ['GLM52_RUN_ID'].rsplit('-',1)[1])
if n==1:sys.exit(0)
if n==2:sys.exit(3)
p=pathlib.Path(os.environ['ARTIFACT_LOCAL_DIR'])/'glm52-e2e/rank-0/result.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'SUCCEEDED'}))
""")
    env = {
        **os.environ,
        "ARTIFACT_LOCAL_DIR": str(root),
        "FALCON_RANK": "0",
        "SOURCE_COMMIT": "test-source",
        "FALCON_EXP_ID": "fixture",
        "STUB": str(stub),
    }
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmark.tpu.glm52_baseline.preflight_allocation",
            "--child",
        ],
        env=env,
    )
    control = root / "allocation"
    status = control / "rank-0.json"
    for generation, code in [(1, 0), (2, 3)]:
        end = time.monotonic() + 5
        while True:
            try:
                s = json.loads(status.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                s = {}
            if s.get("status") == "ATTENTION" and s.get("generation") == generation:
                break
            assert time.monotonic() < end
            sleep = 0.02
            time.sleep(sleep)
        assert p.poll() is None and s["returncode"] == code
        tmp = control / "retry.tmp"
        tmp.write_text(
            json.dumps({"generation": generation + 1, "source_commit": "test-source"})
        )
        tmp.replace(control / "retry.json")
    assert p.wait(timeout=5) == 0
    assert json.loads(status.read_text())["status"] == "SUCCEEDED"
    for g in (1, 2):
        assert (
            json.loads(
                (root / f"application-attempts/{g}/rank-0-outcome.json").read_text()
            )["status"]
            == "ATTENTION"
        )
    # No application result means no success, even if its process exits zero.
    root2 = root / "timeout"
    env.update(ARTIFACT_LOCAL_DIR=str(root2), SECONDS_TEST=".3")
    q = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.tpu.glm52_baseline.preflight_allocation",
            "--child",
        ],
        env=env,
    )
    assert q.returncode == 124
    assert (
        json.loads((root2 / "allocation/rank-0.json").read_text())["status"]
        == "TIMED_OUT"
    )
    print(
        "ALLOCATION_PREFLIGHT_PASS: false zero exit retained, setup failure retained, explicit recovery in same allocation, genuine result required, bounded timeout"
    )
