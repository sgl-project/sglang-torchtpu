"""Run one agentic benchmark leg and record its metrics and status."""

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path

from benchmark.tpu.recipe.bench_matrix.serve_group import (
    TPU_ROOT,
    child_processes,
    load_recipe,
    stop_process,
    wait_healthy,
)
from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli

TOOLS = TPU_ROOT / "recipe/qwen3_coder_480b/nightly"


def selected_recipe(env):
    return load_recipe(
        "recipe/bench_matrix/" + env["BENCH_MODEL"] + "_" + env["BENCH_RECIPE"] + ".py"
    )


def workload_settings(defaults, env, *, nightly=False):
    settings = deepcopy(defaults)
    profile = env.get("PROFILE", settings["profile"])
    workload = env.get("WORKLOAD", settings["workload"])
    if nightly:
        profiles = {
            "full": {},
            "nightly": {"long_ramp_tokens": 16384},
            "ci": {
                "long_ramp_tokens": 16384,
                "sweep_bs": [1, 16],
                "sweep_long": [0, 1],
                "target_context": 30720,
                "long_target_context": 65536,
                "max_leg_seconds": 900,
            },
            "smoke": {
                "sweep_bs": [1, 2],
                "sweep_long": [0, 1],
                "long_target_context": 65536,
                "max_leg_seconds": 1800,
            },
        }
    else:
        profiles = {
            "ci": {
                "sweep_bs": [1, 16, 32],
                "sweep_long": [1, 2, 4],
                "max_leg_seconds": 900,
            },
            "e2e": {"sweep_bs": [32], "sweep_long": [4], "max_leg_seconds": 1800},
            "smoke": {"sweep_bs": [1, 2], "sweep_long": [0, 1], "max_leg_seconds": 900},
        }
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}")
    settings.update(profiles[profile])
    if workload == "8k1k":
        settings.update(
            turn_input_tokens=8192,
            turn_output_tokens=1024,
            target_context=8192,
            sweep_long=[0],
            long_ramp_tokens=settings["long_target_context"],
        )
    elif workload != "agentic":
        raise ValueError(f"Unknown workload: {workload}")
    # Compatibility for the existing benchmark callers; configs themselves never
    # read the environment. New Python callers can update the dictionaries directly.
    for key, value in list(settings.items()):
        override = env.get(key.upper())
        if nightly and key in profiles[profile]:
            override = env.get(key.upper() + "_OVERRIDE")
        if override is not None:
            if isinstance(value, list):
                settings[key] = [int(v) for v in override.split()]
            elif isinstance(value, int):
                settings[key] = int(override)
            elif isinstance(value, float):
                settings[key] = float(override)
            else:
                settings[key] = override
    settings.update(profile=profile, workload=workload)
    if not settings["sweep_bs"] or any(v < 1 for v in settings["sweep_bs"]):
        raise ValueError("sweep_bs must contain positive integers")
    if any(v < 0 for v in settings["sweep_long"]):
        raise ValueError("sweep_long must contain nonnegative integers")
    settings["max_concurrency"] = max(settings["sweep_bs"])
    settings["sessions_per_slot"] = env.get(
        "SESSIONS_PER_SLOT", env.get("NORMAL_REPLAYS", settings["sessions_per_slot"])
    )
    settings["max_sessions_per_slot"] = int(
        env.get(
            "MAX_SESSIONS_PER_SLOT",
            env.get("MAX_NORMAL_REPLAYS", settings["max_sessions_per_slot"]),
        )
    )
    return settings


def validate_workload(settings, context_length, *, nightly=False):
    if nightly:
        final_context = (
            settings["long_target_context"]
            + (settings["long_ramp_tokens"] + settings["turn_output_tokens"]) // 2
        )
    elif max(settings["sweep_long"], default=0) > 0:
        growth = settings["long_ramp_tokens"] + settings["turn_output_tokens"]
        rounds = max(1, (settings["long_target_context"] + growth // 2) // growth)
        final_context = rounds * (growth + 32)
    else:
        return
    if final_context > context_length:
        raise ValueError(
            f"Long sessions reach approximately {final_context} tokens, exceeding context_length={context_length}"
        )


def run_logged(command, log, *, env=None, timeout=None, grace=60):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as output:
        output.write(shlex.join(command) + "\n")
        output.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(process, grace)
            return 124
        finally:
            if process.poll() is None:
                stop_process(process, grace)


def run_leg(config, settings, env, *, nightly=False):
    bs, num_long = int(env["BS"]), int(env.get("NUM_LONG", "0"))
    if bs < 1 or not 0 <= num_long <= bs:
        raise ValueError("A leg requires 0 <= NUM_LONG <= BS and BS > 0")
    output = Path(env["OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    tag = f"c{bs}-long{num_long}"
    status_path = output / f"{tag}.status"
    log = output / f"{tag}.bench.log"
    args = config["server_args"]
    validate_workload(settings, args["context_length"], nightly=nightly)
    model = env.get("MODEL", args["model_path"])
    url = env.get("BASE_URL", f"http://127.0.0.1:{args.get('port', 30000)}")
    probe = env.get("PROBE_URL", url)
    normal_growth = settings["turn_input_tokens"] + settings["turn_output_tokens"]
    long_growth = settings["long_ramp_tokens"] + settings["turn_output_tokens"]
    normal_rounds = max(
        1, (settings["target_context"] + normal_growth // 2) // normal_growth
    )
    long_rounds = max(
        1, (settings["long_target_context"] + long_growth // 2) // long_growth
    )
    replays = settings["sessions_per_slot"]
    if replays == "auto":
        replays = (
            max(
                1,
                min(
                    settings["max_sessions_per_slot"],
                    (long_rounds + normal_rounds - 1) // normal_rounds,
                ),
            )
            if num_long
            else 1
        )
        if nightly and settings["profile"] == "ci" and bs > 1:
            replays = min(settings["max_sessions_per_slot"], max(3, replays))
    replays = int(replays)
    if replays < 1:
        raise ValueError("sessions_per_slot must be positive")
    sessions = max(bs, num_long + (bs - num_long) * replays)
    trace_bs = max(bs, settings["max_concurrency"])
    trace_sessions = max(trace_bs, num_long + (trace_bs - num_long) * replays)
    cache = Path(
        env.get(
            "TRACE_DIR",
            env.get(
                "TRACE_CACHE_DIR",
                str(
                    Path(env.get("XDG_CACHE_HOME", "/tmp"))
                    / "sglang_tpu_agentic_traces"
                ),
            ),
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    seed = int(env.get("SEED", "1"))
    # Include model and shape in the cache key: tokenization depends on the model.
    model_slug = "".join(c if c.isalnum() else "_" for c in model)
    trace = Path(
        env.get(
            "TRACE",
            str(
                cache
                / f"{model_slug}-s{trace_sessions}-long{num_long}-in{settings['turn_input_tokens']}-out{settings['turn_output_tokens']}-ctx{settings['target_context']}-lctx{settings['long_target_context']}-ramp{settings['long_ramp_tokens']}-seed{seed}.json"
            ),
        )
    )
    if not trace.exists() or not trace.stat().st_size:
        temporary = trace.with_name(trace.name + f".tmp.{os.getpid()}")
        trace.parent.mkdir(parents=True, exist_ok=True)
        generator = {
            "out": str(temporary),
            "tokenizer": env.get("TOKENIZER", model),
            "num_sessions": trace_sessions,
            "num_long": num_long,
            "turn_input_tokens": settings["turn_input_tokens"],
            "long_turn_input_tokens": settings["long_ramp_tokens"],
            "turn_output_tokens": settings["turn_output_tokens"],
            "target_context": settings["target_context"],
            "long_target_context": settings["long_target_context"],
            "seed": seed,
        }
        try:
            rc = run_logged(
                [
                    sys.executable,
                    str(TOOLS / "gen_agentic_workload.py"),
                    *server_args_to_cli(generator),
                ],
                output / f"{tag}.trace.log",
                env=env,
            )
            if rc or not temporary.exists() or not temporary.stat().st_size:
                status_path.write_text("failed:trace-gen\n")
                return
            os.replace(temporary, trace)
        finally:
            temporary.unlink(missing_ok=True)
    try:
        wait_healthy(url, settings["health_timeout"])
    except TimeoutError:
        status_path.write_text("failed:health\n")
        raise
    if env.get("FLUSH_CACHE", "1") == "1":
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url + "/flush_cache", method="POST"), timeout=60
            ):
                pass
        except (OSError, urllib.error.URLError) as exc:
            print(f"[leg {tag}] cache flush failed: {exc}", file=sys.stderr)
        time.sleep(2)
    probes = [(probe, "")]
    if env.get("PROBE_URL_2"):
        probes.append((env["PROBE_URL_2"], ".decode"))

    def snapshot(phase):
        for endpoint, suffix in probes:
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "metrics_probe.py"),
                    "snapshot",
                    "--url",
                    endpoint,
                    "--out",
                    str(output / f"{tag}{suffix}.metrics.{phase}.json"),
                ],
                env=env,
                check=False,
            )

    snapshot("pre")
    bench = {
        "backend": "sglang-oai-chat",
        "base_url": url,
        "model": model,
        "dataset_name": "agentic-trace",
        "dataset_path": str(trace),
        "sharegpt_output_len": settings["turn_output_tokens"],
        "num_prompts": sessions,
        "max_concurrency": bs,
        "warmup_requests": 0,
        "seed": seed,
        "output_file": str(output / f"{tag}.jsonl"),
        "output_details": True,
        "cache_report": True,
    }
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        *server_args_to_cli(bench),
        *shlex.split(env.get("EXTRA_BENCH_ARGS", "")),
    ]
    rc = run_logged(
        command,
        log,
        env=env,
        timeout=settings["max_leg_seconds"],
        grace=settings.get("leg_kill_grace", 60),
    )
    status = "ok" if rc == 0 else "timeout" if rc in (124, 137) else f"failed:{rc}"
    status_path.write_text(status + "\n")
    snapshot("post")
    diff = [
        sys.executable,
        str(TOOLS / "metrics_probe.py"),
        "diff",
        "--pre",
        str(output / f"{tag}.metrics.pre.json"),
        "--post",
        str(output / f"{tag}.metrics.post.json"),
        "--out",
        str(output / f"{tag}.metrics.delta.json"),
        "--expect-context",
        str(
            settings["long_target_context"] if num_long else settings["target_context"]
        ),
        "--expect-concurrency",
        str(bs),
    ]
    if nightly:
        diff.extend(
            ["--min-spec-accept-length", str(settings["min_spec_accept_length"])]
        )
    run_logged(diff, log, env=env)
    metadata = {
        **settings,
        "model": model,
        "model_slug": model_slug,
        "base_url": url,
        "backend": "sglang-oai-chat",
        "dataset_name": "agentic-trace",
        "dataset_path": str(trace),
        "run_profile": settings["profile"],
        "leg_tag": tag,
        "concurrency": bs,
        "num_long_sessions": num_long,
        "num_sessions": sessions,
        "trace_sessions": trace_sessions,
        "sessions_per_slot": replays,
        "normal_replays": replays,
        "rounds_per_long_session": long_rounds,
        "rounds_per_normal_session": normal_rounds,
        "server_context_len": args["context_length"],
        "chunked_prefill_per_rank": settings.get(
            "chunked_prefill_per_rank", args["chunked_prefill_size"]
        ),
        "prefix_cache": int(not args.get("disable_radix_cache", False)),
        "kv_offload": int(args.get("enable_hierarchical_cache", False)),
        "spec_decode": int(bool(args.get("speculative_algorithm"))),
        "pd_disagg": int(settings.get("bench_recipe", "").startswith("pd_")),
        "pd_prefill_engines": (
            2 if settings.get("bench_recipe", "").startswith("pd_") else 1
        ),
        "hicache_ratio": args.get("hicache_ratio", 1),
        "seed": seed,
        "status": status,
        "summary_file": f"{tag}.jsonl",
        "bench_argv": command[1:],
    }
    meta_command = [
        sys.executable,
        str(TPU_ROOT / "dump_meta.py"),
        "--kind",
        "bench",
        "--out",
        str(output / f"{tag}.meta.json"),
        "--server-info-url",
        probe,
    ]
    for key, value in metadata.items():
        meta_command.extend(["--json", key + "=" + json.dumps(value)])
    subprocess.run(meta_command, env=env, check=False)
    print(f"[leg {tag}] {status}: {output}", flush=True)


def main():
    env = dict(os.environ)
    module = selected_recipe(env)
    run_leg(module.CONFIG, workload_settings(module.BENCHMARK, env), env)


if __name__ == "__main__":
    with child_processes():
        main()
