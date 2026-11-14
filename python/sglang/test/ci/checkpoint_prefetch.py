"""Apply checkpoint prefetch settings at CI test launch boundaries."""

import argparse
import shlex
import sys

PREFETCH_SETTINGS = {
    "weight_loader_prefetch_checkpoints": True,
    "weight_loader_prefetch_num_threads": 4,
    "weight_loader_disable_mmap": False,
    "weight_loader_drop_cache_after_load": False,
}


def _local_size(settings, env):
    if env.get("CI_WEIGHT_LOADER_PREFETCH") != "1":
        return None
    # Explicit mmap opt-outs protect recipes that cannot safely map weights.
    # Ray workers need their own environment. Offset engines need a loader fix.
    if (
        settings.get("weight_loader_disable_mmap")
        or settings.get("use_ray")
        or env.get("SGLANG_USE_RAY") == "1"
        or env.get("SGLANG_TPU_PD_COLOCATED") == "1"
        or settings.get("base_gpu_id", 0) != 0
        or settings.get("gpu_id_step", 1) != 1
    ):
        return None
    ranks = settings.get("tp_size", 1) * settings.get("pp_size", 1)
    nodes = settings.get("nnodes", 1)
    if nodes < 1 or ranks < nodes or ranks % nodes:
        return None
    return str(ranks // nodes)


def configure_checkpoint_prefetch(argv, env):
    """Return final argv, child env, and settings for launch metadata."""
    argv, env = list(argv), dict(env)
    if env.get("CI_WEIGHT_LOADER_PREFETCH") != "1":
        return argv, env, {}
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tp-size", "--tensor-parallel-size", "--tp", type=int)
    parser.add_argument("--pp-size", "--pipeline-parallel-size", type=int)
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--base-gpu-id", type=int)
    parser.add_argument("--gpu-id-step", type=int)
    parser.add_argument("--use-ray", action="store_true")
    for name, value in PREFETCH_SETTINGS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true")
        else:
            parser.add_argument(flag, type=int)
    known, remaining = parser.parse_known_args(argv)
    settings = {name: value for name, value in vars(known).items() if value is not None}
    local_size = _local_size(settings, env)
    if local_size is None:
        return argv, env, {}
    for name in ("tp_size", "pp_size", "nnodes", "base_gpu_id", "gpu_id_step"):
        if name in settings:
            remaining.extend(["--" + name.replace("_", "-"), str(settings[name])])
    remaining.extend(
        [
            "--weight-loader-prefetch-checkpoints",
            "--weight-loader-prefetch-num-threads",
            "4",
        ]
    )
    env["LOCAL_SIZE"] = local_size
    return remaining, env, dict(PREFETCH_SETTINGS)


if __name__ == "__main__":
    import os

    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    arguments, child_env, _ = configure_checkpoint_prefetch(arguments, os.environ)
    print("args=(" + shlex.join(arguments) + ")")
    if child_env.get("LOCAL_SIZE") != os.environ.get("LOCAL_SIZE"):
        print("export LOCAL_SIZE=" + shlex.quote(child_env["LOCAL_SIZE"]))
