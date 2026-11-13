# SPDX-License-Identifier: Apache-2.0
"""Launching and bringing up a megakernel program without vLLM.

:func:`launch` (on the login node) re-runs the tool as one process per TPU
device of an existing allocation, 8 per host, the way vLLM's workers are laid
out. In each process :func:`init` reproduces the environment the Ray executor
gives a worker (``TORCH_TPU_SLICEBUILDER_ADDRESSES``, ``TORCH_TPU_TOPOLOGY``,
torchrun-style ranks) and opens the ``tpu_dist`` process group; programs then
build their ops exactly as they do inside the server.
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
REPO = Path(__file__).resolve().parents[2]
LOCAL_WORLD = 8
WORKER_ENV = "MEGAKERNEL_TOOL_WORKER"


def is_worker() -> bool:
    return os.environ.get(WORKER_ENV) == "1"


def rank() -> int:
    return int(os.environ.get("SLURM_PROCID", "0"))


def log_host(msg: str) -> None:
    """A line from every process, tagged with its host (host-side tools)."""
    print(
        f"[{time.perf_counter() - T0:7.1f}s host "
        f"{os.environ.get('SLURM_NODEID', '0')}] {msg}",
        flush=True,
    )


def log(msg: str) -> None:
    if rank() == 0:
        print(f"[{time.perf_counter() - T0:7.1f}s r{rank():02d}] {msg}", flush=True)


def in_pod() -> bool:
    """A CDK pod of a 2x2x4 slice: no Slurm allocation, and the GKE TPU webhook
    gives every pod the slice's hosts and its own index."""
    return "TPU_WORKER_HOSTNAMES" in os.environ and not os.environ.get("SLURM_JOB_ID")


def resolve_hosts(hosts: list[str], timeout: float = 1800.0) -> list[str]:
    """The slice's host IPs. A pod's DNS record appears once it is scheduled,
    and the pods of a slice start minutes apart, so wait for all of them."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return [socket.gethostbyname(h) for h in hosts]
        except socket.gaierror as e:
            if time.monotonic() > deadline:
                raise SystemExit(
                    f"TPU hosts {hosts} unresolved after {timeout:.0f}s: {e}"
                )
            time.sleep(10)


def _topology(hosts: list[str]):
    from sglang_tpu.node_normalization import TopologyConfig

    return TopologyConfig(hosts=tuple(hosts), topology=(2, 2, 4, 2))


def topological_node(hosts: list[str], host_index: int) -> int:
    """This pod's node rank in physical (z, y, x) host order, which device
    ids, and so the programs' kernel ranks, follow: its devices are kernel
    ranks ``8 * node .. 8 * node + 7``. Pod indices need not follow it."""
    from sglang_tpu.node_normalization import query_node_rank

    # The coordinate query brings up the TPU runtime, which waits for every
    # host of the slice: pods can start minutes apart.
    return query_node_rank(_topology(hosts), host_index, timeout=3600)


def physical_host_order(hosts: list[str], host_index: int) -> list[int]:
    """The pods in physical host order (all pods call it together), as
    sglang's node normalization orders host ranks and SliceBuilder endpoints."""
    from sglang_tpu.node_normalization import discover_host_order

    return discover_host_order(_topology(hosts), host_index, timeout=3600)


def launch_pod(argv: list[str], extra_env: dict[str, str], per_host: int) -> int:
    """Run ``argv`` as this pod's ``per_host`` workers, with the Slurm variables
    the workers read (node index ``TPU_WORKER_ID``). Every pod of the slice runs
    the same command; the master port and session must agree, so they are
    fixed rather than drawn."""
    hosts = os.environ["TPU_WORKER_HOSTNAMES"].split(",")
    if len(hosts) != 4:
        raise SystemExit(f"{len(hosts)} TPU hosts; the K3 programs need 4 (32 devices)")
    pod = int(os.environ.get("TPU_WORKER_ID", "0"))
    # Host ranks follow physical host order: the pre-shard names its ranks by
    # it, and torch_tpu's SliceBuilder endpoints must follow it.
    order = physical_host_order(hosts, pod)
    hosts = [hosts[i] for i in order]
    node = order.index(pod)
    log_host(
        f"pod {pod} is topological node {node} (kernel ranks {8 * node}..{8 * node + 7})"
    )
    env = dict(os.environ)
    env.update(extra_env)
    env.update(
        {
            WORKER_ENV: "1",
            "TP32_HOST_IPS": ",".join(resolve_hosts(hosts)),
            "MASTER_PORT": env.get("MASTER_PORT", "20111"),
            "TP32_SESSION": env.get("TP32_SESSION", "cdk"),
            "HF_HUB_OFFLINE": "1",
            "SLURM_NODEID": str(node),
            "SLURM_JOB_ID": "cdk",
            # Every pod pre-shards onto its own volume: each writes the files
            # one host writes on a shared Slurm file system.
            "K3_PRESHARD_EVERY_HOST": "1",
        }
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-m", "sglang_tpu.k3_programs.tools", *argv],
            env=dict(env, SLURM_LOCALID=str(i), SLURM_PROCID=str(node * per_host + i)),
        )
        for i in range(per_host)
    ]
    codes = [worker.wait() for worker in workers]
    return next((code for code in codes if code), 0)


def launch(
    argv: list[str],
    jobid: str,
    venv: str,
    extra_env: dict[str, str],
    per_host: int = LOCAL_WORLD,
) -> int:
    """Run ``python -m tools.megakernel argv`` on the four hosts of ``jobid``,
    ``per_host`` processes each (default one per TPU device). In a CDK pod
    (no ``jobid``), run this pod's share (:func:`launch_pod`)."""
    if not jobid and in_pod():
        return launch_pod(argv, extra_env, per_host)
    nodelist = subprocess.run(
        ["squeue", "-j", jobid, "-h", "-o", "%N"], capture_output=True, text=True
    ).stdout.strip()
    if not nodelist:
        raise SystemExit(f"job {jobid} is not running")
    hosts = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True
    ).stdout.split()
    if len(hosts) != 4:
        raise SystemExit(
            f"job {jobid} holds {len(hosts)} hosts; the K3 programs need 4 (32 devices)"
        )
    ips = [
        subprocess.run(
            ["getent", "hosts", h], capture_output=True, text=True
        ).stdout.split()[0]
        for h in hosts
    ]
    env = dict(os.environ)
    env.update(extra_env)
    env.update(
        {
            WORKER_ENV: "1",
            "PYTHONPATH": f"{REPO / 'src'}:{REPO}:{env.get('PYTHONPATH', '')}",
            "TP32_HOST_IPS": ",".join(ips),
            "MASTER_PORT": env.get("MASTER_PORT", str(20000 + random.randrange(20000))),
            "TP32_SESSION": str(time.time_ns()),
            "HF_HUB_OFFLINE": "1",
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", ""),
        }
    )
    command = [
        "srun",
        f"--jobid={jobid}",
        "--overlap",
        "--nodes=4",
        f"--ntasks={4 * per_host}",
        f"--ntasks-per-node={per_host}",
        "--kill-on-bad-exit=1",
        "--unbuffered",
        "--export=ALL",
        f"--chdir={REPO}",
        f"{venv}/bin/python",
        "-m",
        "sglang_tpu.k3_programs.tools",
        *argv,
    ]
    return subprocess.call(command, env=env)


def init() -> None:
    """Bring up torch_tpu and the ``tpu_dist`` process group in a worker."""
    host_ips = os.environ["TP32_HOST_IPS"].split(",")
    base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
    os.environ.update(
        {
            "RANK": str(rank()),
            "LOCAL_RANK": os.environ["SLURM_LOCALID"],
            "WORLD_SIZE": str(len(host_ips) * LOCAL_WORLD),
            "LOCAL_WORLD_SIZE": str(LOCAL_WORLD),
            "NNODES": str(len(host_ips)),
            "NODE_RANK": os.environ["SLURM_NODEID"],
            "TPU_NUM_HOSTS": str(len(host_ips)),
            "MASTER_ADDR": host_ips[0],
            "TORCH_TPU_TOPOLOGY": "2,2,4,2",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
                f"{ip}:{base_port + i}" for ip in host_ips for i in range(LOCAL_WORLD)
            ),
        }
    )
    os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", os.environ["TP32_SESSION"])
    import torch
    import torch_tpu  # noqa: F401

    log("torch_tpu imported")
    torch.distributed.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://{host_ips[0]}:{os.environ['MASTER_PORT']}",
        world_size=len(host_ips) * LOCAL_WORLD,
        rank=rank(),
    )
    log("tpu_dist process group up")


def sync() -> None:
    import torch

    torch.distributed.barrier()


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out.stdout.strip() + ("+dirty" if dirty else "")


class Records:
    """One JSON record per result (rank 0), with what produced it."""

    def __init__(self, path: str | None, command: str, config) -> None:
        self.path, self.command = path, command
        self.context = {
            "git": git_revision(),
            "config": config.key(),
            "jobid": os.environ.get("SLURM_JOB_ID", ""),
        }

    def add(self, **fields) -> None:
        if rank() != 0 or not self.path:
            return
        record = {
            "command": self.command,
            **fields,
            **self.context,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def fail(message: str) -> None:
    log(f"FAIL: {message}")
    sys.exit(1)
