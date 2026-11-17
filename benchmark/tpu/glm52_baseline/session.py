"""Supervise both native server launchers within one bounded Falcon allocation."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path


def atomic(path, data):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def benchmark_identity():
    root = Path(__file__).resolve().parent
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".sh")
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(kind="source-tree", sha256=digest, files=files)


def run_client(proc, command, out, art, shared, deadline):
    """A client failure retains the live server until an explicit retry or deadline."""
    attempt = 1
    request = shared / "retry-client.json"
    last_request = None
    while True:
        destination = out / "measurements" / f"attempt-{attempt}"
        with (art / f"client-attempt-{attempt}.log").open("wb", buffering=0) as log:
            try:
                subprocess.run(
                    [*command, "--output", str(destination)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(1, deadline - time.monotonic()),
                )
                summary = json.loads((destination / "summary.json").read_text())
                smoke = json.loads((destination / "smoke-summary.json").read_text())
                assert len(summary["rounds"]) == 6 and smoke["status"] == "PASS"
                assert {row["dp_rank"] for row in smoke["checks"]} == set(range(16))
                atomic(
                    shared / "client-attention.json",
                    dict(
                        status="RESOLVED",
                        attempt=attempt,
                        measurements=str(destination),
                    ),
                )
                return destination
            except Exception as error:
                atomic(
                    shared / "client-attention.json",
                    dict(
                        status="ATTENTION",
                        attempt=attempt,
                        returncode=getattr(error, "returncode", None),
                        error=str(error),
                        measurements=str(destination),
                        server_pid=proc.pid,
                        recovery="write retry-client.json with a new id after fixing the client",
                    ),
                )
        while True:
            assert proc.poll() is None, (
                f"server exited while retaining client failure: {proc.returncode}"
            )
            assert time.monotonic() < deadline, "client recovery deadline exceeded"
            try:
                retry = json.loads(request.read_text())
                if retry["id"] != last_request:
                    last_request = retry["id"]
                    attempt += 1
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(1)


def run():
    rank = int(os.environ["FALCON_RANK"])
    root = Path(os.environ.get("GLM52_RUN_ROOT", "/tmp/glm52-baseline-runtime"))
    out = Path(os.environ["ARTIFACT_LOCAL_DIR"]) / "glm52-e2e"
    identity = benchmark_identity()
    benchmark_sha = identity["sha256"]
    shared = out / ("session-" + os.environ.get("GLM52_RUN_ID", benchmark_sha[:12]))
    shared.mkdir(parents=True, exist_ok=True)
    art = out / f"rank-{rank}"
    art.mkdir(exist_ok=True)
    atomic(art / "benchmark-source.json", identity)
    deadline = time.monotonic() + float(os.environ.get("GLM52_SESSION_SECONDS", "6000"))
    if "GLM52_ALLOCATION_EXPIRES_AT" in os.environ:
        deadline = min(
            deadline,
            time.monotonic()
            + float(os.environ["GLM52_ALLOCATION_EXPIRES_AT"])
            - time.time()
            - 60,
        )
    peer_deadline = time.monotonic() + 300
    proc = None
    result = dict(status="FAILED", rank=rank)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    handlers = {
        s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        assert Path(root / "SETUP_READY").exists()
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        assert source == os.environ["SOURCE_COMMIT"]
        ip = socket.gethostbyname(socket.gethostname())
        atomic(
            shared / f"peer-{rank}.json",
            dict(rank=rank, ip=ip, source=source, benchmark_sha256=benchmark_sha),
        )
        while True:
            try:
                peers = [
                    json.loads((shared / f"peer-{r}.json").read_text())
                    for r in range(2)
                ]
                assert [p["rank"] for p in peers] == [0, 1]
                assert all(
                    p["source"] == source and p["benchmark_sha256"] == benchmark_sha
                    for p in peers
                )
                assert peers[0]["ip"] != peers[1]["ip"]
                break
            except (FileNotFoundError, json.JSONDecodeError):
                assert time.monotonic() < peer_deadline, "peer bootstrap exceeded 300s"
                time.sleep(1)
        hosts = ",".join(p["ip"] for p in peers)
        os.environ.update(
            NODE_RANK=str(rank),
            TPU_WORKER_ID=str(rank),
            TPU_WORKER_HOSTNAMES=hosts,
            RANK0_HOST=peers[0]["ip"],
            DIST_INIT_ADDR=peers[0]["ip"] + ":10011",
            TORCH_TPU_TOPOLOGY="2,2,2,2",
            TORCH_TPU_DEVICES_PER_HOST="8",
            TORCH_TPU_BASE_PORT="8070",
            SGLANG_USE_RAY="0",
            GLM52_BENCHMARK_SHA256=benchmark_sha,
            XDG_CACHE_HOME=str(root / "cache"),
            TMPDIR=str(root / "tmp"),
            JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache"),
            TORCH_TPU_TIER2_NAME="glm52_block_channel_" + source[:12],
            TORCH_TPU_TIER3_ROOT=os.environ.get(
                "GLM52_COMPILE_CACHE_ROOT", str(out / "compile-cache")
            ),
            PYTHONPATH=str(Path("benchmark/tpu/glm52_baseline/runtime_probe").resolve())
            + os.pathsep
            + str(Path.cwd()),
            GLM52_RUNTIME_DIR=str(art / "actual-process-runtime"),
        )
        for key in ("XDG_CACHE_HOME", "TMPDIR", "JAX_COMPILATION_CACHE_DIR"):
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
        config = root / "recipe/server.json"
        with (art / "cli-validation.log").open("wb") as validation_log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.tpu.glm52_baseline.validate",
                    str(config),
                    str(art / "resolved-args.json"),
                ],
                stdout=validation_log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        # Reuse the successful weekend host preparation before starting the
        # native normalization timeout. Physical host order can differ from Falcon order.
        query = """import os
from sglang_tpu.node_normalization import TopologyConfig, query_node_rank
cfg=TopologyConfig(hosts=tuple(os.environ['TPU_WORKER_HOSTNAMES'].split(',')),topology=(2,2,2,2))
print('PHYSICAL_RANK='+str(query_node_rank(cfg,int(os.environ['FALCON_RANK']),180)))
"""
        probe = subprocess.run(
            [sys.executable, "-c", query],
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
        (art / "physical-rank.log").write_text(probe.stdout + probe.stderr)
        probe.check_returncode()
        physical_rank = int(
            next(
                line.split("=", 1)[1]
                for line in probe.stdout.splitlines()
                if line.startswith("PHYSICAL_RANK=")
            )
        )
        recipe = json.loads(config.read_text())["server_args"]
        model = Path(recipe["model_path"])
        experts = json.loads((model / "config.json").read_text())["n_routed_experts"]
        assert experts == 256 and physical_rank in (0, 1)
        with (art / "host-preparation.log").open("wb") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "benchmark.tpu.glm52_baseline.prepare_host_checkpoint",
                    str(model),
                    "--expert-start",
                    str(physical_rank * 128),
                    "--expert-end",
                    str((physical_rank + 1) * 128),
                    "--execute",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        # The native draft loader reads every tensor in its five shards, so
        # prepare their full host union once rather than cold-reading the other
        # host's expert half independently in each of eight draft processes.
        draft_path = json.loads(config.read_text())["server_args"][
            "speculative_draft_model_path"
        ]
        with (art / "mtp-host-preparation.log").open("wb") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "benchmark.tpu.glm52_baseline.prepare_host_checkpoint",
                    draft_path,
                    "--expert-start",
                    "0",
                    "--expert-end",
                    "256",
                    "--execute",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        atomic(
            shared / f"prepared-{rank}.json",
            dict(rank=rank, physical_rank=physical_rank),
        )
        preparation_deadline = time.monotonic() + 930
        while True:
            try:
                prepared = [
                    json.loads((shared / f"prepared-{r}.json").read_text())
                    for r in range(2)
                ]
                assert {p["physical_rank"] for p in prepared} == {0, 1}
                break
            except (FileNotFoundError, json.JSONDecodeError):
                assert time.monotonic() < preparation_deadline, (
                    "peer preparation timed out"
                )
                time.sleep(2)
        config = root / "recipe/server.json"
        atomic(art / "recipe.json", json.loads(config.read_text()))
        atomic(
            art / "session-env.json",
            {
                k: v
                for k, v in os.environ.items()
                if k.startswith(
                    (
                        "TPU_",
                        "TORCH_TPU",
                        "LIBTPU",
                        "SGLANG",
                        "GLM52",
                        "FALCON_JAX",
                        "XLA_",
                    )
                )
            },
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "benchmark.tpu.launch_server",
            "--config",
            "@" + str(config),
        ]
        with (art / "dry-run.json").open("wb") as log:
            subprocess.run(
                [*command, "--dry-run"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        with (art / "server.log").open("wb", buffering=0) as log:
            proc = subprocess.Popen(
                [*command, "--shutdown-grace", "25"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            atomic(
                art / "launcher.json",
                dict(
                    pid=proc.pid,
                    command=command,
                    executable=sys.executable,
                    source=source,
                    benchmark_sha256=benchmark_sha,
                    peers=peers,
                ),
            )
            if rank == 0:
                while True:
                    assert proc.poll() is None, f"launcher exited {proc.returncode}"
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{recipe['port']}/health", timeout=5
                        ) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        pass
                    assert time.monotonic() < deadline, "server readiness timed out"
                    peer_failure = shared / "result-1.json"
                    if peer_failure.exists():
                        raise RuntimeError(peer_failure.read_text())
                    time.sleep(5)
                client = [
                    sys.executable,
                    "-u",
                    "-m",
                    "benchmark.tpu.glm52_baseline.client",
                    "--base-url",
                    f"http://127.0.0.1:{recipe['port']}",
                    "--model",
                    str(model),
                    "--decode-prefix",
                    "shared",
                ]
                measurements = run_client(proc, client, out, art, shared, deadline)
                result.update(status="SUCCEEDED", measurements=str(measurements))
            else:
                while not (shared / "result-0.json").exists():
                    assert proc.poll() is None, f"launcher exited {proc.returncode}"
                    assert time.monotonic() < deadline, "peer client timed out"
                    time.sleep(5)
                peer = json.loads((shared / "result-0.json").read_text())
                assert peer["status"] == "SUCCEEDED", peer
                result.update(status="SUCCEEDED")
    except Exception as e:
        result.update(error=str(e), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        atomic(shared / f"result-{rank}.json", result)
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                result.update(
                    status="FAILED", shutdown_error="launcher exceeded 35s shutdown"
                )
        atomic(art / "result.json", result)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(run())
