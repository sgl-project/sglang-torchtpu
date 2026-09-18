"""CI caller for Python server configs and the benchmark matrix's P/D groups.

The CI JobSet runs this on every host of the slice; what a host does depends on
the recipe's GROUP and its TPU_WORKER_ID. A 1host_* recipe serves a single tp=8
engine on rank 0 (the registered suite's smallest slice is 2 hosts, so a 1-host
recipe deliberately strands the second); the pd_2p2d recipe runs four tp=8
engines (ranks 0-1 prefill, 2-3 decode, KV over raiden) with rank 0 fronting
them with the mini-LB on :30000, the port the JobSet health-gates and benches.

Children must die with this process: the JobSet template kills the recipe
process on teardown, and a surviving engine would hold the slice's chips
against the next group.
"""

import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

TPU_ROOT = Path(__file__).resolve().parents[2]
# Under Bazel the configs are not installed packages; they are runfiles beside
# this file. Put the repo root (or runfiles root) on sys.path so the
# `benchmark.tpu...` imports below and the `--config` module resolve.
for _candidate in (TPU_ROOT.parent.parent, Path.cwd()):
    if (_candidate / "benchmark" / "tpu" / "recipe").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli

SLICE_ENV = (
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_PROCESS_ADDRESSES",
    "TPU_TOPOLOGY",
    "TPU_TOPOLOGY_WRAP",
    "TPU_TOPOLOGY_ALT",
    "TPU_HOST_BOUNDS",
    "TPU_ACCELERATOR_TYPE",
    "TPU_VISIBLE_CHIPS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
)


def load_recipe(name):
    """Import a config selected by a CI matrix row, relative to benchmark/tpu."""
    name = name.removesuffix(".py").replace("/", ".")
    if name.startswith("recipe."):
        name = "benchmark.tpu." + name
    if not name.startswith("benchmark.tpu.recipe."):
        raise ValueError("Expected a module under benchmark.tpu.recipe")
    return importlib.import_module(name)


def wait_healthy(url, timeout, process=None, interval=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(
                url.rstrip("/") + "/health", timeout=10
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server did not become healthy: {url}")


def stop_process(process, grace=60):
    # Children are started in their own sessions. Stop their entire process group,
    # including workers left behind if the Python parent already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@contextmanager
def child_processes():
    children = []

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield children
    finally:
        for child in reversed(children):
            stop_process(child)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def server_command(config, inherited, *, single_host=False):
    settings = deepcopy(config["server_args"])
    env = {**inherited, **config["runtime_env"]}
    if single_host:
        for key in list(env):
            if key in SLICE_ENV or key.startswith("MEGASCALE"):
                env.pop(key, None)
        settings["nnodes"] = 1
        settings.pop("node_rank", None)
        settings.pop("dist_init_addr", None)
    elif inherited.get("SGLANG_USE_RAY") == "1" or settings.get("use_ray"):
        settings["use_ray"] = True
        settings.pop("node_rank", None)
        settings.pop("dist_init_addr", None)
    else:
        nodes = settings.get("nnodes", 1)
        if nodes > 1:
            rank = int(inherited.get("NODE_RANK", inherited.get("TPU_WORKER_ID", "0")))
            if not 0 <= rank < nodes:
                raise ValueError("node rank outside the recipe's nnodes")
            settings["node_rank"] = rank
            hosts = inherited.get("TPU_WORKER_HOSTNAMES", "").split(",")
            address = inherited.get("DIST_INIT_ADDR") or (
                "0.0.0.0" if rank == 0 else inherited.get("RANK0_HOST") or hosts[0]
            )
            if not address:
                raise ValueError(
                    "Nonzero ranks require RANK0_HOST or TPU_WORKER_HOSTNAMES"
                )
            settings["dist_init_addr"] = (
                address
                if ":" in address
                else address + ":" + inherited.get("DIST_PORT", "10011")
            )
    for name, flag in (("MODEL", "model_path"), ("HOST", "host"), ("PORT", "port")):
        if inherited.get(name):
            settings[flag] = inherited[name]
    if settings.get("tp_size", 1) > 1:
        for key in (
            "TPU_VISIBLE_CHIPS",
            "TPU_CHIPS_PER_HOST_BOUNDS",
            "TPU_HOST_BOUNDS",
        ):
            env.pop(key, None)
    command = [
        sys.executable,
        "-u",
        "-m",
        "sglang.launch_server",
        *server_args_to_cli(settings),
    ]
    return command, env


def dump_launch(config, command, env):
    if env.get("DUMP_META", "1") == "0":
        return
    destination = (
        Path(
            env.get("RUN_DIR", str(TPU_ROOT / "profile" / "recipe" / str(os.getpid())))
        )
        / "launch_meta.json"
    )
    args = [
        sys.executable,
        str(TPU_ROOT / "dump_meta.py"),
        "--kind",
        "launch",
        "--out",
        str(destination),
        "--json",
        "server_argv=" + json.dumps(command[1:]),
        "--set",
        "model=" + config["server_args"]["model_path"],
    ]
    for key, value in config["server_args"].items():
        args.extend(["--json", key + "=" + json.dumps(value)])
    for prefix in ("SGLANG_", "TORCH_TPU_", "TPU_"):
        args.extend(["--env-prefix", prefix])
    for key in ("LIBTPU_INIT_ARGS", "XLA_FLAGS", "TORCH_LOGS"):
        args.extend(["--env", key])
    try:
        subprocess.run(args, env=env, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[recipe] metadata unavailable: {exc}", file=sys.stderr)


def serve(module, inherited=None, dry_run=False):
    inherited = dict(os.environ if inherited is None else inherited)
    group = getattr(module, "GROUP", None)
    rank = int(inherited.get("TPU_WORKER_ID", "0"))
    hosts = inherited.get("TPU_WORKER_HOSTNAMES", socket.gethostname()).split(",")
    if group and not group["prefill_engines"] and rank != 0:
        return
    pd = bool(group and group["prefill_engines"])
    if pd and len(hosts) != group["hosts"]:
        raise ValueError(f"P/D recipe needs {group['hosts']} hosts, got {len(hosts)}")
    config = deepcopy(
        module.CONFIG
        if not pd
        else (
            module.PREFILL_CONFIG
            if rank < group["prefill_engines"]
            else module.DECODE_CONFIG
        )
    )
    if pd:
        inherited.pop("HOST", None)
        inherited.pop("PORT", None)
        config["runtime_env"]["SGLANG_HOST_IP"] = inherited.get(
            "SGLANG_HOST_IP"
        ) or socket.gethostbyname(socket.gethostname())
    command, env = server_command(config, inherited, single_host=bool(group))
    if dry_run:
        print(json.dumps({"command": command, "config": config}, indent=2))
        return
    dump_launch(config, command, env)
    if not pd:
        os.execvpe(command[0], command, env)
    Path("/tmp/bench_matrix_phys_hosts").write_text("\n".join(hosts) + "\n")
    with child_processes() as children:
        engine = subprocess.Popen(command, env=env, start_new_session=True)
        children.append(engine)
        if rank == 0:
            prefill_port = module.PREFILL_CONFIG["server_args"]["port"]
            decode_port = module.DECODE_CONFIG["server_args"]["port"]
            bootstrap = module.PREFILL_CONFIG["server_args"][
                "disaggregation_bootstrap_port"
            ]
            timeout = int(inherited.get("HEALTH_TIMEOUT", "3600"))
            urls = [
                f"http://{host}:{prefill_port if i < 2 else decode_port}"
                for i, host in enumerate(hosts)
            ]
            for i, url in enumerate(urls):
                wait_healthy(url, timeout, engine if i == 0 else None)
            router = [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
            ]
            for url in urls[:2]:
                router.extend(["--prefill", url, str(bootstrap)])
            for url in urls[2:]:
                router.extend(["--decode", url])
            router.extend(
                ["--host", "0.0.0.0", "--port", inherited.get("LB_PORT", "30000")]
            )
            children.append(subprocess.Popen(router, env=env, start_new_session=True))
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError("An engine or router exited before the group was stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Importable recipe module or recipe/name.py"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    serve(load_recipe(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
