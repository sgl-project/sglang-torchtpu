"""Sweep a live benchmark-matrix server and collect its results."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.tpu.recipe.bench_matrix.bench_leg import (
    TOOLS,
    run_leg,
    selected_recipe,
    workload_settings,
)
from benchmark.tpu.recipe.bench_matrix.serve_group import child_processes, wait_healthy


def sweep(config, settings, env, output, *, nightly=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        wait_healthy(env["BASE_URL"], settings["health_timeout"])
    except (TimeoutError, RuntimeError):
        (output / "MODE_STATUS").write_text("failed:server\n")
        raise
    for bs in settings["sweep_bs"]:
        for long in settings["sweep_long"]:
            if long > bs:
                (output / f"c{bs}-long{long}.status").write_text("skipped:long>bs\n")
                continue
            try:
                run_leg(
                    config,
                    settings,
                    {
                        **env,
                        "BS": str(bs),
                        "NUM_LONG": str(long),
                        "OUTPUT_DIR": str(output),
                    },
                    nightly=nightly,
                )
            except (RuntimeError, TimeoutError, OSError) as exc:
                print(f"[sweep] c{bs}-long{long} failed: {exc}", file=sys.stderr)
    (output / "MODE_STATUS").write_text("ok\n")


def summarize(root, settings, mode, *, matrix=False):
    command = [
        sys.executable,
        str(TOOLS / "summarize.py"),
        "--run-root",
        str(root),
        "--json-out",
        str(root / "summary.json"),
        "--md-out",
        str(root / "summary.md"),
        "--profile",
        settings["profile"],
        "--mode",
        mode,
    ]
    if matrix:
        label = "-".join(
            settings[k] for k in ("bench_model", "bench_recipe", "workload")
        )
        command.extend(["--benchmark-label", label, "--title", label])
    result = subprocess.run(command, check=False)
    if (root / "summary.md").exists():
        print((root / "summary.md").read_text())
    return result.returncode


def main():
    env = dict(os.environ)
    module = selected_recipe(env)
    settings = workload_settings(module.BENCHMARK, env)
    root = Path(
        env.get(
            "RUN_ROOT",
            str(
                Path(__file__).parent
                / "results"
                / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
            ),
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.log").write_text(
        json.dumps({**module.CONFIG, "benchmark": settings}, indent=2)
    )
    pd = bool(module.GROUP["prefill_engines"])
    if pd:
        host_file = Path("/tmp/bench_matrix_phys_hosts")
        hosts = (
            host_file.read_text().splitlines()
            if host_file.exists()
            else env.get("TPU_WORKER_HOSTNAMES", "").split(",")
        )
        env.setdefault(
            "PROBE_URL",
            f"http://{hosts[0] or '127.0.0.1'}:{settings['pd_prefill_port']}",
        )
        if len(hosts) >= 3:
            env.setdefault(
                "PROBE_URL_2", f"http://{hosts[2]}:{settings['pd_decode_port']}"
            )
    sweep(module.CONFIG, settings, env, root / ("pd" if pd else "colocated"))
    return summarize(root, settings, "external", matrix=True)


if __name__ == "__main__":
    with child_processes():
        sys.exit(main())
