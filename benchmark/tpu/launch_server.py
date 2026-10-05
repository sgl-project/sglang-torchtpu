"""Launch TPU servers and P/D groups from Python recipes or JSON configs.

Run with ``python -m benchmark.tpu.launch_server`` or the Bazel target of the
same name on every participating host. This launcher resolves host ranks and
P/D roles, starts servers and the router, and supervises their process groups.
Only children import SGLang, after runtime_envs has been applied.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli

TPU_ROOT = Path(__file__).absolute().parent
# Clear cross-host rendezvous when each P/D engine owns a separate host. Keep
# hardware metadata (notably TPU_TOPOLOGY) available to libtpu and diagnostics.
MULTIHOST_ENV = (
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_PROCESS_ADDRESSES",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
)
# The name torch_tpu itself reads for the tier-2 cache; the launcher's
# TORCH_TPU_INTERNAL_* spelling is mirrored onto it by the TPU env override.
TIER2_CACHE_ENV = "TORCH_TPU_TIER2_COMPILATION_CACHE"


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _envs(value: Any) -> dict[str, str | None]:
    result = _object(value, "runtime_envs")
    for name, entry in result.items():
        if not isinstance(name, str) or not name or "=" in name or "\0" in name:
            raise ValueError(f"Invalid environment variable name: {name!r}")
        if entry is not None and (not isinstance(entry, str) or "\0" in entry):
            raise ValueError(f"Environment variable {name} must be a string or null")
    return dict(result)


@dataclass(frozen=True)
class LaunchConfig:
    server_args: dict[str, Any]
    runtime_envs: dict[str, str | None]
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict, runtime_envs: dict | None = None) -> LaunchConfig:
        value = _object(value, "config")
        unknown = value.keys() - {
            "server_args",
            "runtime_envs",
            "runtime_env",
            "extra_args",
        }
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        if "runtime_env" in value and "runtime_envs" in value:
            raise ValueError("Use only one of runtime_envs and runtime_env")
        settings = deepcopy(_object(value.get("server_args"), "server_args"))
        if (
            not isinstance(settings.get("model_path"), str)
            or not settings["model_path"]
        ):
            raise ValueError("server_args.model_path must be a nonempty string")
        for name in settings:
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Expected a server argument in snake_case: {name!r}")
        # Reject NaN/Infinity and non-JSON Python values before starting a process.
        json.dumps(settings, allow_nan=False)
        if any("\0" in arg for arg in server_args_to_cli(settings)):
            raise ValueError("Server arguments must not contain NUL characters")
        env = _envs(value.get("runtime_envs", value.get("runtime_env", {})))
        if runtime_envs is not None:
            env.update(_envs(runtime_envs))
        extra_args = value.get("extra_args", [])
        if not isinstance(extra_args, list) or any(
            not isinstance(arg, str) or "\0" in arg for arg in extra_args
        ):
            raise ValueError("extra_args must be a list of strings")
        return cls(settings, env, list(extra_args))


def child_env(inherited, runtime_envs):
    env = dict(inherited)
    for name, value in runtime_envs.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


def resolve_server(config: LaunchConfig, inherited, *, single_host=False):
    """Resolve topology and the environment overrides used by shell launchers."""
    settings = deepcopy(config.server_args)
    env = child_env(inherited, config.runtime_envs)
    if single_host or env.get("FORCE_SINGLE_HOST") == "1":
        for key in list(env):
            if key in MULTIHOST_ENV or key.startswith("MEGASCALE"):
                env.pop(key)
        settings["nnodes"] = 1
        settings.pop("node_rank", None)
        settings.pop("dist_init_addr", None)
        settings.pop("use_ray", None)
        env.pop("SGLANG_USE_RAY", None)
        # Keep an explicitly configured single-chip recipe usable in a slice.
        for key in (
            "TPU_VISIBLE_CHIPS",
            "TPU_CHIPS_PER_HOST_BOUNDS",
            "TPU_HOST_BOUNDS",
        ):
            env.pop(key, None)
            if (
                settings.get("tp_size", 1) == 1
                and config.runtime_envs.get(key) is not None
            ):
                env[key] = config.runtime_envs[key]
    elif env.get("SGLANG_USE_RAY") == "1" or settings.get("use_ray"):
        settings["use_ray"] = True
        settings.pop("node_rank", None)
        settings.pop("dist_init_addr", None)
    elif settings.get("nnodes", 1) > 1:
        rank = int(
            settings.get(
                "node_rank", env.get("NODE_RANK", env.get("TPU_WORKER_ID", "0"))
            )
        )
        if not 0 <= rank < settings["nnodes"]:
            raise ValueError("node rank outside the config's nnodes")
        settings["node_rank"] = rank
        hosts = env.get("TPU_WORKER_HOSTNAMES", "").split(",")
        address = (
            settings.get("dist_init_addr")
            or env.get("DIST_INIT_ADDR")
            or ("0.0.0.0" if rank == 0 else env.get("RANK0_HOST") or hosts[0])
        )
        if not address:
            raise ValueError(
                "Nonzero ranks require dist_init_addr, RANK0_HOST or TPU_WORKER_HOSTNAMES"
            )
        settings["dist_init_addr"] = (
            address if ":" in address else address + ":" + env.get("DIST_PORT", "10011")
        )
    for name, flag in (("MODEL", "model_path"), ("HOST", "host"), ("PORT", "port")):
        if env.get(name):
            settings[flag] = int(env[name]) if flag == "port" else env[name]
    if (
        settings.get("enable_torch_compile")
        and env.get("ENABLE_COMPILE_CACHE", "1") == "1"
    ):
        # Match the shell launcher's per-model cache defaults after MODEL is
        # resolved. Explicit runtime_envs nulls must still remove a variable.
        model_slug = re.sub(r"[^a-z0-9]+", "_", settings["model_path"].lower()).strip(
            "_"
        )
        cache_defaults = {
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": env.get(
                "TORCH_TPU_TIER2_NAME", f"torchtpu_{model_slug}"
            ),
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": env.get(
                "TORCH_TPU_TIER3_ROOT",
                str(
                    Path(env.get("HOME", str(Path.home())))
                    / ".cache/torch_tpu_compile_cache"
                ),
            ),
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "1",
        }
        for name, value in cache_defaults.items():
            if name not in config.runtime_envs:
                env.setdefault(name, value)
    elif env.get("ENABLE_COMPILE_CACHE", "1") != "1":
        # Leaving the names unset is not enough: torch_tpu then falls back to
        # an implicit tier-2 cache (/dev/shm/torch_tpu_cache/default) whenever
        # WORLD_SIZE > 1. "disabled" is its sentinel for really off.
        if TIER2_CACHE_ENV not in config.runtime_envs:
            env.setdefault(TIER2_CACHE_ENV, "disabled")
    if settings.get("tp_size", 1) > 1:
        for key in (
            "TPU_VISIBLE_CHIPS",
            "TPU_CHIPS_PER_HOST_BOUNDS",
            "TPU_HOST_BOUNDS",
        ):
            env.pop(key, None)
    if settings.get("disaggregation_mode") in ("prefill", "decode"):
        env["SGLANG_HOST_IP"] = env.get("SGLANG_HOST_IP") or socket.gethostbyname(
            socket.gethostname()
        )
    return LaunchConfig(settings, config.runtime_envs, config.extra_args), env


def server_command(
    config: LaunchConfig, inherited: Mapping[str, str] | None = None
) -> tuple[list[str], dict[str, str]]:
    """Build argv/env without changing the caller's config or environment."""
    env = child_env(os.environ if inherited is None else inherited, config.runtime_envs)
    command = [
        *python_command("sglang.launch_server"),
        *server_args_to_cli(config.server_args),
        *config.extra_args,
    ]
    return command, env


def python_command(module: str) -> list[str]:
    """Use declared Bazel executables, or the caller's installed Python."""
    if not (os.environ.get("RUNFILES_DIR") or os.environ.get("RUNFILES_MANIFEST_FILE")):
        return [sys.executable, "-u", "-m", module]

    from python.runfiles import runfiles

    binaries = {
        "sglang.launch_server": "sglang_server",
        "sglang_router.launch_router": "sglang_router",
    }
    path = runfiles.Create().Rlocation(
        f"sglang_torchtpu/benchmark/tpu/{binaries[module]}"
    )
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing Bazel executable for {module}: {path}")
    return [path]


def router_command(
    prefill_urls, decode_urls, *, bootstrap_port=8998, host="0.0.0.0", port=30000
):
    command = [
        *python_command("sglang_router.launch_router"),
        "--pd-disaggregation",
        "--mini-lb",
    ]
    for url in prefill_urls:
        command.extend(["--prefill", url, str(bootstrap_port)])
    for url in decode_urls:
        command.extend(["--decode", url])
    command.extend(["--host", host, "--port", str(port)])
    return command


def _kill_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def wait_healthy(url, timeout, check, interval=1, *, endpoint="/health", tag=None):
    """Wait for an endpoint while watching local children and termination signals."""
    start = last_note = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        check()
        try:
            with urllib.request.urlopen(
                url.rstrip("/") + endpoint,
                timeout=min(10, max(0.01, deadline - time.monotonic())),
            ) as response:
                if response.status == 200:
                    check()
                    if tag:
                        print(
                            f"[{tag}] server healthy after {time.monotonic() - start:.0f}s",
                            flush=True,
                        )
                    return
        except (OSError, urllib.error.URLError):
            pass
        if tag and time.monotonic() - last_note >= 60:
            last_note = time.monotonic()
            print(
                f"[{tag}] waiting for {url}{endpoint} ({last_note - start:.0f}s elapsed)",
                flush=True,
            )
        until = min(deadline, time.monotonic() + interval)
        while time.monotonic() < until:
            check()
            time.sleep(min(0.2, max(0, until - time.monotonic())))
    raise TimeoutError(f"Server did not become healthy: {url}{endpoint}")


def stop_processes(processes, grace=30):
    """Stop owned process groups, including workers of already-exited parents."""
    processes = list(processes)
    for process in reversed(processes):
        _kill_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        finally:
            _kill_group(process, signal.SIGKILL)
            process.wait()


def start_process(command, env, *, cwd=None):
    """Start an owned process group after applying its runtime environment."""
    env = prepare_runtime_env(env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(command, env=env, start_new_session=True, cwd=cwd)


def run_server(
    command: list[str],
    env: dict[str, str],
    grace: float = 30,
    *,
    router=None,
    health_urls=(),
    health_timeout=3600,
    pd=False,
) -> int:
    """One supervisor for a server and its optional P/D router."""
    received_signal = None

    def interrupted(signum, frame):
        nonlocal received_signal
        received_signal = signum

    handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    children = []

    def check():
        if received_signal is not None:
            raise SystemExit(128 + received_signal)
        for child in children:
            if child.poll() is not None:
                raise RuntimeError(
                    f"An engine or router exited with {child.returncode}"
                )

    try:
        children.append(start_process(command, env))
        if router:
            for url in health_urls:
                wait_healthy(url, health_timeout, check)
            check()
            children.append(start_process(router, env))
        if pd or router:
            # Engines are long-lived services; the test process owns the verdict.
            # Any unsolicited engine/router exit fails the group, even exit 0.
            # SIGTERM/SIGINT to this supervisor instead trigger graceful cleanup.
            while received_signal is None:
                check()
                time.sleep(0.2)
        else:
            while received_signal is None:
                try:
                    code = children[0].wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    continue
                return code if code >= 0 else 128 - code
        return 128 + received_signal
    finally:
        try:
            stop_processes(children, grace)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def load_recipe(name):
    """Load a recipe, or one P/D engine via module:PREFILL_CONFIG/DECODE_CONFIG.

    Selecting an engine runs it on the local host without a router. A decode
    service can therefore start without knowing the prefill pod's address.
    """
    name, _, export = name.partition(":")
    name = name.removesuffix(".py").replace("/", ".")
    if name.startswith("recipe."):
        name = "benchmark.tpu." + name
    if not name.startswith("benchmark.tpu.recipe."):
        raise ValueError("Expected a module under benchmark.tpu.recipe")
    module = importlib.import_module(name)
    group = getattr(module, "GROUP", None)
    if export:
        if (
            not group
            or not group.get("prefill_engines")
            or export not in ("PREFILL_CONFIG", "DECODE_CONFIG")
        ):
            raise ValueError("Select PREFILL_CONFIG or DECODE_CONFIG from a P/D recipe")
        config = deepcopy(getattr(module, export))
        config["server_args"].setdefault(
            "disaggregation_mode", export.removesuffix("_CONFIG").lower()
        )
        config["server_args"].setdefault("host", "0.0.0.0")
        config["group"] = {"hosts": 1}
        if "runtime_env" in config:
            if "runtime_envs" in config:
                raise ValueError("Use only one of runtime_envs and runtime_env")
            config["runtime_envs"] = config.pop("runtime_env")
        env = config.setdefault("runtime_envs", {})
        # These JobSet variables describe the public router, not its engines.
        env.update(HOST=None, PORT=None)
        return config
    if group and group.get("prefill_engines", 0):
        return {
            "group": deepcopy(group),
            "prefill": deepcopy(module.PREFILL_CONFIG),
            "decode": deepcopy(module.DECODE_CONFIG),
        }
    config = deepcopy(module.CONFIG)
    if group:
        config["group"] = deepcopy(group)
    return config


@dataclass
class LaunchPlan:
    config: LaunchConfig
    command: list[str]
    env: dict[str, str]
    hosts: list[str]
    router: list[str] | None = None
    health_urls: tuple[str, ...] = ()


def _count(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _plan_pd_group(config, group, topology, common_env, overrides, *, inherited, rank):
    """Plan one host's P/D role and the rank-zero router after group validation."""
    host_count = group["hosts"]
    prefill_count = group["prefill_engines"]
    router = None
    hosts = [
        host.strip()
        for host in topology.get(
            "SGLANG_LAUNCHER_HOSTNAMES",
            topology.get("TPU_WORKER_HOSTNAMES", ""),
        ).split(",")
        if host.strip()
    ]
    decode_count = _count(
        group.get("decode_engines", host_count - prefill_count),
        "decode_engines",
    )
    if prefill_count + decode_count != host_count or len(hosts) != host_count:
        raise ValueError(
            "P/D group needs one host per engine and a matching TPU_WORKER_HOSTNAMES list"
        )
    if rank >= host_count:
        raise ValueError("TPU_WORKER_ID outside the P/D group")
    unknown = config.keys() - {
        "prefill",
        "decode",
        "runtime_envs",
        "runtime_env",
    }
    if unknown:
        raise ValueError(f"Unknown P/D config fields: {sorted(unknown)}")
    roles = {}
    for role in ("prefill", "decode"):
        role_config = LaunchConfig.from_dict(config.get(role))
        if role_config.server_args.get("nnodes", 1) != 1 or role_config.server_args.get(
            "use_ray"
        ):
            raise ValueError("P/D groups use one single-host server per engine")
        if role_config.server_args.get("disaggregation_mode", role) != role:
            raise ValueError(f"{role} config has a conflicting disaggregation_mode")
        role_config.server_args["disaggregation_mode"] = role
        role_config.server_args.setdefault("host", "0.0.0.0")
        role_config.runtime_envs.update(
            {**common_env, **role_config.runtime_envs, **overrides}
        )
        roles[role] = role_config
    selected = roles["prefill" if rank < prefill_count else "decode"]
    # HOST/PORT describe the public endpoint in JobSets, not P/D engines.
    inherited.pop("HOST", None)
    inherited.pop("PORT", None)
    for role_config in roles.values():
        role_config.runtime_envs.pop("HOST", None)
        role_config.runtime_envs.pop("PORT", None)
    env = child_env(inherited, selected.runtime_envs)
    selected.runtime_envs["SGLANG_HOST_IP"] = env.get(
        "SGLANG_HOST_IP"
    ) or socket.gethostbyname(socket.gethostname())
    prefill_port = roles["prefill"].server_args.get("port", 30000)
    decode_port = roles["decode"].server_args.get("port", 30000)
    bootstrap = roles["prefill"].server_args.get("disaggregation_bootstrap_port", 8998)
    urls = tuple(
        f"http://{host}:{prefill_port if i < prefill_count else decode_port}"
        for i, host in enumerate(hosts)
    )
    if rank == 0:
        lb_port = env.get("LB_PORT", "30000")
        if int(lb_port) == int(prefill_port):
            raise ValueError(
                "LB_PORT must differ from the rank-zero prefill server port"
            )
        router = router_command(
            urls[:prefill_count],
            urls[prefill_count:],
            bootstrap_port=bootstrap,
            port=lb_port,
        )
    return selected, hosts, router, urls


def plan_launch(config, inherited=None, runtime_envs=None):
    """Resolve a recipe/JSON config once, for both dry-run and real execution."""
    config = deepcopy(_object(config, "config"))
    inherited = dict(os.environ if inherited is None else inherited)
    overrides = _envs({} if runtime_envs is None else runtime_envs)
    group = config.pop("group", None)
    router = None
    urls = ()
    hosts = []
    if group is not None:
        group = _object(group, "group")
        unknown = group.keys() - {"hosts", "prefill_engines", "decode_engines"}
        if unknown:
            raise ValueError(f"Unknown group fields: {sorted(unknown)}")
        if "runtime_envs" in config and "runtime_env" in config:
            raise ValueError("Use only one of runtime_envs and runtime_env")
        common_env = _envs(config.get("runtime_envs", config.get("runtime_env", {})))
        # Config and CLI runtime_envs can supply topology in hermetic jobs.
        topology = child_env(child_env(inherited, common_env), overrides)
        # A P/D group may span separate slices whose local ranks both start at
        # zero. Keep group membership separate from the TPU runtime's topology.
        rank = int(
            topology.get("SGLANG_LAUNCHER_RANK", topology.get("TPU_WORKER_ID", "0"))
        )
        prefill_count = _count(group.get("prefill_engines", 0), "prefill_engines", 0)
        host_count = _count(group.get("hosts"), "group.hosts")
        if rank < 0:
            raise ValueError("TPU_WORKER_ID must be nonnegative")
        if prefill_count:
            selected, hosts, router, urls = _plan_pd_group(
                config,
                group,
                topology,
                common_env,
                overrides,
                inherited=inherited,
                rank=rank,
            )
        else:
            if host_count != 1 or group.get("decode_engines", 0) != 0:
                raise ValueError(
                    "A non-P/D group must describe one host and no decode engines"
                )
            selected = LaunchConfig.from_dict(config, overrides)
            if rank != 0:
                return None
    else:
        selected = LaunchConfig.from_dict(config, overrides)
    resolved, env = resolve_server(selected, inherited, single_host=group is not None)
    # resolve_server already applied overrides and topology cleanup. Do not
    # apply runtime_envs again when building the final child command.
    command, env = server_command(
        LaunchConfig(resolved.server_args, {}, resolved.extra_args), env
    )
    return LaunchPlan(resolved, command, env, hosts, router, urls)


def dump_launch(config, command, env):
    """Write the existing launch_meta.json schema using the child's environment."""
    if env.get("DUMP_META", "1") == "0":
        return
    destination = (
        Path(
            env.get(
                "RUN_DIR", str(Path.cwd() / "profile" / "recipe" / str(os.getpid()))
            )
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
    ]
    for key, value in config["server_args"].items():
        args.extend(["--json", key + "=" + json.dumps(value)])
    args.extend(["--set", "model=" + config["server_args"]["model_path"]])
    for prefix in ("SGLANG_", "TORCH_TPU_", "TPU_"):
        args.extend(["--env-prefix", prefix])
    for key in ("LIBTPU_INIT_ARGS", "XLA_FLAGS", "TORCH_LOGS"):
        args.extend(["--env", key])
    try:
        subprocess.run(args, env=env, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[launcher] metadata unavailable: {exc}", file=sys.stderr)


def telemetry_stub(env):
    """Use Bazel's built library, or compile it for a source checkout."""
    stub = Path(
        env.get("LIBTPU_TELEMETRY_STUB", str(TPU_ROOT / "libtpu_telemetry_stub.so"))
    )
    if stub.is_file():
        return stub.absolute()
    if (
        "LIBTPU_TELEMETRY_STUB" in env
        or env.get("RUNFILES_DIR")
        or env.get("RUNFILES_MANIFEST_FILE")
    ):
        raise FileNotFoundError(f"Missing telemetry library: {stub}")

    # Match common_env.sh for direct Python runs in the existing TPU image.
    stub = Path(tempfile.gettempdir()) / "libtpu_telemetry_stub.so"
    if not stub.is_file():
        compiler = shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            raise FileNotFoundError(
                "Source launches need gcc or clang for the telemetry library; "
                "alternatively set LIBTPU_TELEMETRY_STUB to a built library"
            )
        with tempfile.TemporaryDirectory(
            dir=stub.parent, prefix="sglang-telemetry-"
        ) as directory:
            output = Path(directory) / stub.name
            subprocess.run(
                [
                    compiler,
                    "-O2",
                    "-fPIC",
                    "-shared",
                    str(TPU_ROOT / "libtpu_telemetry_stub.c"),
                    "-o",
                    str(output),
                ],
                env=env,
                check=True,
            )
            output.replace(stub)
    return stub


def prepare_runtime_env(env):
    """Materialize explicitly configured runtime artifacts before importing TPU."""
    env = dict(env)
    if env.get("SGLANG_TPU_PRELOAD_TELEMETRY_STUB") == "1":
        stub = telemetry_stub(env)
        preload = env.get("LD_PRELOAD")
        preloads = preload.split(":") if preload else []
        if str(stub.absolute()) not in preloads:
            preloads.insert(0, str(stub.absolute()))
        env["LD_PRELOAD"] = ":".join(preloads)
    for name in (
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
        "SGLANG_TORCH_PROFILER_DIR",
        "TORCH_COMPILE_DEBUG_DIR",
    ):
        if env.get(name):
            Path(env[name]).mkdir(parents=True, exist_ok=True)
    if env.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"):
        try:
            (
                Path("/dev/shm/torch_tpu_cache")
                / env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"]
            ).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # The tier-2 cache is optional, as in the shell launcher.
    return env


def serve(plan, *, dry_run=False, grace=30):
    if plan is None:
        if dry_run:
            print(json.dumps({"idle": True}))
        return 0
    if dry_run:
        print(
            json.dumps(
                {
                    "command": plan.command,
                    "server_args": plan.config.server_args,
                    "runtime_env_keys": sorted(plan.config.runtime_envs),
                    "router_command": plan.router,
                    "health_urls": plan.health_urls,
                },
                indent=2,
            )
        )
        return 0
    plan.env = prepare_runtime_env(plan.env)
    dump_launch({"server_args": plan.config.server_args}, plan.command, plan.env)
    if plan.hosts:
        Path("/tmp/bench_matrix_phys_hosts").write_text("\n".join(plan.hosts) + "\n")
    return run_server(
        plan.command,
        plan.env,
        grace,
        router=plan.router,
        health_urls=plan.health_urls,
        health_timeout=float(plan.env.get("HEALTH_TIMEOUT", "3600")),
        pd=bool(plan.hosts),
    )


def _json_object(argument: str) -> dict:
    """Accept inline JSON or @path, with no evaluation of Python or shell code."""
    try:
        source = (
            Path(argument[1:]).read_text(encoding="utf-8")
            if argument.startswith("@")
            else argument
        )
        return _object(json.loads(source), "JSON input")
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _grace(argument: str) -> float:
    value = float(argument)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            "shutdown grace must be finite and nonnegative"
        )
    return value


def _config(argument):
    if argument.startswith(("recipe/", "recipe.", "benchmark.tpu.recipe.")):
        try:
            return load_recipe(argument)
        except (ImportError, AttributeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return _json_object(argument)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=_config, help="Recipe module, JSON or @file"
    )
    parser.add_argument(
        "--runtime-envs", type=_json_object, help="JSON or @file; overrides config envs"
    )
    parser.add_argument("--shutdown-grace", type=_grace, default=30)
    parser.add_argument(
        "--working-directory", help="Working directory for child servers"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.working_directory:
        os.chdir(args.working_directory)
    try:
        plan = plan_launch(args.config, runtime_envs=args.runtime_envs)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return serve(plan, dry_run=args.dry_run, grace=args.shutdown_grace)


if __name__ == "__main__":
    raise SystemExit(main())
