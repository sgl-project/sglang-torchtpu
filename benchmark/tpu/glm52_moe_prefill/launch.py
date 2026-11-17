# SPDX-License-Identifier: Apache-2.0
"""Bounded EP16 MoE prefill benchmark using the proven native launcher."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def write_result(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"result.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(directory / "result.json")


def run_worker():
    import faulthandler
    from worker import main
    faulthandler.dump_traceback_later(int(os.environ["GLM_PREFLIGHT_WORKER_SECONDS"]), exit=True)
    main()
    faulthandler.cancel_dump_traceback_later()


def run_launcher(args):
    started = time.monotonic()
    deadline = started + args.seconds
    directory = Path(args.output_dir)
    process = None
    result = dict(status="FAILED", source=None, host_index=args.host_index)
    try:
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if args.source_commit and source != args.source_commit:
            raise ValueError(f"source mismatch: {source} != {args.source_commit}")
        result["source"] = source
        hosts = tuple(args.hosts.split(","))
        if len(hosts) != 2:
            raise ValueError("EP16 preflight requires exactly two host addresses")
        from sglang_tpu.patches import _prepare_tpu_env

        _prepare_tpu_env(world_size=16, setup_slicebuilder=False)
        from sglang_tpu.node_normalization import TopologyConfig, discover_host_order
        from sglang_tpu.moe.pallas.glm_moe_v2.op import validate_source_environment

        validate_source_environment()
        config = TopologyConfig(
            hosts=hosts,
            topology=(2, 2, 2, 2),
            devices_per_host=8,
            bootstrap_port=args.bootstrap_port,
            master_port=args.master_port,
            slicebuilder_port=args.slicebuilder_port,
        )
        order = discover_host_order(
            config,
            args.host_index,
            timeout=min(180, max(1, int(deadline - time.monotonic()) - 30)),
        )
        config = replace(config, hosts=tuple(config.hosts[index] for index in order))
        node_rank = order.index(args.host_index)
        env = dict(os.environ)
        env.update(config.environment(node_rank))
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
        ):
            env.pop(name, None)
        remaining = int(deadline - time.monotonic()) - 30
        if remaining <= 0:
            raise TimeoutError("normalization exhausted the preflight deadline")
        flags = re.sub(
            r"(?:^|\s)--xla_force_host_platform_device_count=\S+",
            "",
            env.get("XLA_FLAGS", ""),
        ).strip()
        env.update(
            TPU_WORKER_ID=str(node_rank),
            JAX_PLATFORMS="cpu",
            XLA_FLAGS=(flags + " --xla_force_host_platform_device_count=16").strip(),
            SOURCE_COMMIT=source,
            TF_CPP_MIN_LOG_LEVEL="3",
            GLM_PREFLIGHT_WORKER_SECONDS=str(max(1, remaining - 30)),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc_per_node=8",
            f"--node_rank={node_rank}",
            f"--master_addr={config.hosts[0]}",
            f"--master_port={config.master_port}",
            str(Path(__file__).resolve()),
            "--worker",
        ]
        directory.mkdir(parents=True, exist_ok=True)
        launch = dict(
            source=source,
            executable=sys.executable,
            original_hosts=hosts,
            normalized_hosts=config.hosts,
            normalized_node_rank=node_rank,
            command=command,
            deadline_seconds=args.seconds,
            environment={
                k: v
                for k, v in env.items()
                if k.startswith(("TORCH_TPU_", "TPU_", "LIBTPU", "XLA_", "JAX_"))
            },
        )
        (directory / "launch.json").write_text(json.dumps(launch, indent=2))
        print(json.dumps(dict(event="ep16_preflight_launch", **launch)), flush=True)
        process = subprocess.Popen(command, env=env, cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"EP16 preflight exceeded {args.seconds}s") from None
        if code != 0:
            raise RuntimeError(f"EP16 worker group exited {code}")
        result.update(status="SUCCEEDED", normalized_node_rank=node_rank)
        return 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(dict(event="ep16_preflight_failure", **result)), flush=True)
        return 1
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        result["elapsed_seconds"] = time.monotonic() - started
        write_result(directory, result)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        run_worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--hosts", default=os.environ.get("TPU_WORKER_HOSTNAMES"), required=False
        )
        parser.add_argument(
            "--host-index", type=int, default=int(os.environ.get("FALCON_RANK", "-1"))
        )
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--source-commit", default=os.environ.get("SOURCE_COMMIT"))
        parser.add_argument("--seconds", type=int, default=1500)
        parser.add_argument("--master-port", type=int, default=10021)
        parser.add_argument("--bootstrap-port", type=int, default=10022)
        parser.add_argument("--slicebuilder-port", type=int, default=8070)
        options = parser.parse_args()
        if (
            not options.hosts
            or options.host_index not in (0, 1)
            or not 60 <= options.seconds <= 1500
        ):
            parser.error("provide two hosts, host-index 0/1, and a 60..1500s deadline")
        sys.exit(run_launcher(options))
