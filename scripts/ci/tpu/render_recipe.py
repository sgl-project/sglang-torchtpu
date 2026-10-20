#!/usr/bin/env python3
"""Render a TPU serving recipe as a cookbook launch command, and check pages.

A cookbook page shows the command a reader pastes on their own host. That
command is the recipe, so it is generated rather than typed twice:

    python3 scripts/ci/tpu/render_recipe.py \\
        --config recipe/configs/gpt_oss_120b/server.py --profile long_8k1k

A page marks each generated region with a pair of comments naming what produced
it; `--write` fills them in, `--check` fails when they have drifted. They are MDX
comments: the pages are .mdx, where an HTML comment is a syntax error.

    {/* render_recipe config=recipe/configs/gpt_oss_120b/server.py profile=long_8k1k */}
    ```bash
    ...
    ```
    {/* /render_recipe */}

    python3 scripts/ci/tpu/render_recipe.py --check docs/cookbook/.../GPT-OSS.mdx

Only the command region is generated. Measured numbers live outside the
markers: a number belongs to the run that produced it, with its date and
commit, and regenerating must not restate it.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.tpu.launch_server import load_recipe  # noqa: E402
from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG  # noqa: E402
from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli  # noqa: E402
from benchmark.tpu.recipe.configs.profiles import PROFILES, apply_profile  # noqa: E402

# Settings a reader supplies from their own deployment, not from the recipe.
_DEPLOYMENT_ENVS = (
    "TPU_PREMAPPED_BUFFER_SIZE",
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
)

MARKER = re.compile(
    r"(?P<open>\{/\*\s*render_recipe\s+(?P<attrs>[^\n]*?)\s*\*/\}\n)"
    r"(?P<body>.*?)"
    r"(?P<close>\{/\*\s*/render_recipe\s*\*/\})",
    re.DOTALL,
)


def hardware_line(server_args: dict) -> str:
    """One line naming the slice the command below assumes."""
    tp_size = server_args["tp_size"]
    # One tpu7x host is 4 chips x 2 TensorCores; a rank is a core.
    hosts, remainder = divmod(tp_size, 8)
    if remainder or hosts == 0:
        slice_text = f"{tp_size} TensorCores"
    elif hosts == 1:
        slice_text = "one tpu7x host (4 chips x 2 TensorCores)"
    else:
        slice_text = f"{hosts} tpu7x hosts ({tp_size} TensorCores)"
    parts = [f"tp{tp_size}"]
    if server_args.get("ep_size"):
        parts.append(f"ep{server_args['ep_size']}")
    if server_args.get("dp_size", 1) > 1:
        parts.append(f"dp{server_args['dp_size']}")
    return f"**Hardware**: {slice_text}, {'/'.join(parts)}."


def model_id(config_module: str, server_args: dict) -> str:
    """The Hugging Face id to show, never a CI mount path."""
    path = server_args["model_path"]
    if path.startswith("/"):
        raise SystemExit(
            f"{config_module}: model_path {path!r} is a mount path; a cookbook "
            "command needs a Hugging Face id in the recipe's server.py"
        )
    return path


def extra_runtime_env(runtime_env: dict) -> dict:
    """The env a reader must set: this model's own, plus the deployment ones.

    Everything base.py sets is CI hygiene -- telemetry, offline flags, logging.
    """
    base = _FP8_BASE_CONFIG["runtime_env"]
    extra = {
        name: value
        for name, value in runtime_env.items()
        if value is not None and base.get(name) != value
    }
    for name in _DEPLOYMENT_ENVS:
        if name in runtime_env and runtime_env[name] is not None:
            extra[name] = runtime_env[name]
    return extra


def render(config_module: str, profile_name: str) -> str:
    """The generated region's contents for one (config, profile)."""
    if profile_name not in PROFILES:
        raise SystemExit(
            f"unknown profile {profile_name!r}; known: {', '.join(sorted(PROFILES))}"
        )
    config = apply_profile(load_recipe(config_module), PROFILES[profile_name])
    server_args = dict(config["server_args"])
    server_args["model_path"] = model_id(config_module, server_args)
    profile = PROFILES[profile_name]

    lines = [
        hardware_line(server_args),
        "",
        f"**Workload**: {profile['input_tokens']} in / "
        f"{profile['output_tokens']} out at concurrency {profile['concurrency']}.",
        "",
        "```bash",
    ]
    env = extra_runtime_env(config.get("runtime_env", {}))
    for name, value in sorted(env.items()):
        lines.append(f"export {name}={value}")
    if env:
        lines.append("")
    lines.append("python3 -m sglang.launch_server \\")
    flags = server_args_to_cli(server_args)
    rendered = []
    index = 0
    while index < len(flags):
        chunk = [flags[index]]
        index += 1
        while index < len(flags) and not flags[index].startswith("--"):
            chunk.append(flags[index])
            index += 1
        rendered.append(" ".join(chunk))
    for position, flag in enumerate(rendered):
        suffix = "" if position == len(rendered) - 1 else " \\"
        lines.append(f"    {flag}{suffix}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def parse_attrs(text: str) -> dict:
    attrs = dict(re.findall(r"(\w+)=(\S+)", text))
    missing = {"config", "profile"} - set(attrs)
    if missing:
        raise SystemExit(f"render_recipe marker is missing {sorted(missing)}: {text!r}")
    return attrs


def process_page(path: Path, write: bool) -> int:
    """Check (or rewrite) every generated region in one page."""
    original = path.read_text()
    stale = []

    def replace(match: re.Match) -> str:
        attrs = parse_attrs(match.group("attrs"))
        fresh = render(attrs["config"], attrs["profile"])
        if match.group("body") != fresh:
            stale.append(f"{attrs['config']} / {attrs['profile']}")
        return match.group("open") + fresh + match.group("close")

    updated = MARKER.sub(replace, original)
    if not MARKER.search(original):
        print(f"{path}: no render_recipe region", file=sys.stderr)
        return 1
    if write:
        if updated != original:
            path.write_text(updated)
            print(f"{path}: updated {len(stale)} region(s)", file=sys.stderr)
        return 0
    if stale:
        for region in stale:
            print(f"{path}: stale region ({region})", file=sys.stderr)
        print(
            f"--check: regenerate with --write {path}",
            file=sys.stderr,
        )
        return 1
    print(f"--check: {path} is up to date", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", help="recipe module, e.g. recipe/configs/x/server.py"
    )
    parser.add_argument("--profile", help=f"one of: {', '.join(sorted(PROFILES))}")
    parser.add_argument(
        "--check",
        metavar="PAGE",
        action="append",
        default=[],
        help="exit 1 if PAGE's generated regions differ from what the recipes render",
    )
    parser.add_argument(
        "--write",
        metavar="PAGE",
        action="append",
        default=[],
        help="rewrite PAGE's generated regions in place",
    )
    args = parser.parse_args(argv)

    if args.check or args.write:
        if args.config or args.profile:
            parser.error("--check/--write take the page, not --config/--profile")
        return max(
            [process_page(Path(p), write=False) for p in args.check]
            + [process_page(Path(p), write=True) for p in args.write]
        )
    if not (args.config and args.profile):
        parser.error("give --config and --profile, or --check/--write a page")
    sys.stdout.write(render(args.config, args.profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
