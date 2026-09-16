"""Run the agentic nightly against a colocated, external, or P/D server."""

import json
import os
import shlex
import socket
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from benchmark.tpu.recipe.bench_matrix.bench_leg import run_logged, workload_settings
from benchmark.tpu.recipe.bench_matrix.run_sweep import summarize, sweep
from benchmark.tpu.recipe.bench_matrix.serve_group import (
    child_processes,
    dump_launch,
    server_command,
    wait_healthy,
)
from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli
from benchmark.tpu.recipe.configs.qwen3_coder_480b.nightly.config import BENCHMARK
from benchmark.tpu.recipe.configs.qwen3_coder_480b.nightly.serve import (
    CONFIG,
    DECODE_CONFIG,
    PREFILL_CONFIG,
)


def warmup(config, settings, env, url, log):
    if env.get("SKIP_WARMUP", "0") == "1":
        return
    settings = {
        "backend": "sglang",
        "model": config["server_args"]["model_path"],
        "base_url": url,
        "dataset_name": "random-ids",
        "tokenize_prompt": True,
        "random_range_ratio": 1.0,
        "random_input_len": settings["chunked_prefill_per_rank"],
        "random_output_len": 32,
        "max_concurrency": settings["max_concurrency"],
        "num_prompts": settings["max_concurrency"] * 2,
        "warmup_requests": 0,
    }
    rc = run_logged(
        [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
            *server_args_to_cli(settings),
        ],
        log,
        env=env,
    )
    if rc:
        print(f"[nightly] warmup failed ({rc}); continuing", file=sys.stderr)


def main():
    env = dict(os.environ)
    settings = workload_settings(BENCHMARK, env, nightly=True)
    config = deepcopy(CONFIG)
    args = config["server_args"]
    args["model_path"] = env.get("MODEL", args["model_path"])
    dp = int(env.get("COLOCATED_DP", args["dp_size"]))
    args.update(
        dp_size=dp,
        max_running_requests=settings["max_concurrency"] * dp,
        torch_compile_max_bs=settings["max_concurrency"],
        chunked_prefill_size=settings["chunked_prefill_per_rank"] * dp,
    )
    if settings["profile"] == "ci":
        args.update(tpu_compile_tokens=[4096], tpu_compile_bs=[1, 8, 16])
    if env.get("DRAFT_MODEL"):
        args.update(
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=env["DRAFT_MODEL"],
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
    mode = env.get("MODE", "colocated")
    if mode not in ("colocated", "external", "pd") or (
        mode == "pd" and env.get("ALLOW_PD") != "1"
    ):
        raise ValueError("MODE must be colocated or external; pd requires ALLOW_PD=1")
    root = Path(
        env.get(
            "RUN_ROOT",
            str(
                Path(__file__).parent
                / "results/nightly"
                / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
            ),
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.log").write_text(
        json.dumps({**config, "benchmark": settings}, indent=2)
    )
    label = env.get("EXTERNAL_MODE_LABEL", "colocated") if mode == "external" else mode
    output = root / label
    output.mkdir(parents=True, exist_ok=True)
    mode_rc = 0
    with child_processes() as children:
        logs = []
        try:
            if mode == "external":
                url = env["BASE_URL"]
                wait_healthy(url, settings["health_timeout"])
            elif mode == "colocated":
                url = f"http://127.0.0.1:{env.get('PORT', args['port'])}"
                command, child_env = server_command(config, env)
                child_env["RUN_DIR"] = str(output)
                dump_launch(config, command, child_env)
                log = (output / "server.log").open("a")
                logs.append(log)
                server = subprocess.Popen(
                    command,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append(server)
                wait_healthy(url, settings["health_timeout"], server)
            else:
                urls = []
                external = bool(
                    env.get("PREFILL_BASE_URL") and env.get("DECODE_BASE_URL")
                )
                if not external and not (
                    env.get("PD_PREFILL_SSH") or env.get("PD_DECODE_SSH")
                ):
                    raise ValueError(
                        "P/D needs separate hosts or prestarted PREFILL_BASE_URL and DECODE_BASE_URL"
                    )
                for role, base in (
                    ("prefill", PREFILL_CONFIG),
                    ("decode", DECODE_CONFIG),
                ):
                    if external:
                        url = env[role.upper() + "_BASE_URL"]
                        process = None
                    else:
                        role_config = deepcopy(config)
                        role_config["server_args"].update(
                            {
                                key: value
                                for key, value in base["server_args"].items()
                                if key
                                in ("nnodes", "tp_size", "ep_size", "dp_size", "port")
                                or key.startswith("disaggregation_")
                            }
                        )
                        role_config["server_args"].update(
                            model_path=args["model_path"],
                            max_running_requests=settings["max_concurrency"] * 2,
                            torch_compile_max_bs=settings["max_concurrency"],
                        )
                        port = settings[f"pd_{role}_port"]
                        role_config["server_args"]["port"] = port
                        target = env.get(f"PD_{role.upper()}_SSH")
                        role_config["runtime_env"]["SGLANG_HOST_IP"] = (
                            socket.gethostbyname(
                                target.rsplit("@", 1)[-1]
                                if target
                                else socket.gethostname()
                            )
                        )
                        if role == "prefill":
                            role_config["server_args"][
                                "disaggregation_bootstrap_port"
                            ] = settings["pd_bootstrap_port"]
                        command, child_env = server_command(
                            role_config, env, single_host=True
                        )
                        if target:
                            # The remote command execs the existing server CLI; the SSH
                            # terminal owns it and propagates hangup on teardown.
                            assignments = [
                                f"{k}={v}"
                                for k, v in role_config["runtime_env"].items()
                            ]
                            command = [
                                "ssh",
                                "-tt",
                                "-o",
                                "BatchMode=yes",
                                target,
                                "exec " + shlex.join(["env", *assignments, *command]),
                            ]
                        log = (output / f"server-{role}.log").open("a")
                        logs.append(log)
                        process = subprocess.Popen(
                            command,
                            env=child_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        children.append(process)
                        url = f"http://{target.rsplit('@', 1)[-1] if target else '127.0.0.1'}:{port}"
                    wait_healthy(url, settings["health_timeout"], process)
                    urls.append(url)
                url = env.get("ROUTER_BASE_URL")
                if not url:
                    host = env.get("ROUTER_HOST", "127.0.0.1")
                    port = settings["pd_router_port"]
                    command = [
                        sys.executable,
                        "-m",
                        "sglang_router.launch_router",
                        "--pd-disaggregation",
                        "--prefill",
                        urls[0],
                        str(settings["pd_bootstrap_port"]),
                        "--decode",
                        urls[1],
                        "--host",
                        host,
                        "--port",
                        str(port),
                    ]
                    log = (output / "router.log").open("a")
                    logs.append(log)
                    router = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    children.append(router)
                    url = f"http://{host}:{port}"
                    wait_healthy(url, 300, router)
                else:
                    wait_healthy(url, 300)
            warmup(config, settings, env, url, output / "warmup.log")
            sweep(config, settings, {**env, "BASE_URL": url}, output, nightly=True)
        except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
            mode_rc = 1
            (output / "MODE_STATUS").write_text("failed:server\n")
            print(f"[nightly] {exc}", file=sys.stderr)
        finally:
            for log in logs:
                log.close()
    rc = summarize(root, settings, mode)
    return int(bool(rc or (env.get("STRICT") == "1" and mode_rc)))


if __name__ == "__main__":
    sys.exit(main())
