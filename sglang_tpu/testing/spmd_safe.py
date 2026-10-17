"""Deployment-independent runner for SPMD-safe recipe verification."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path


def verify_startup(
    command: Sequence[str],
    *,
    world_size: int,
    local_world_size: int,
    timeout: int,
    dump_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Launch one host's server in verification mode and check its worker results."""
    if not command or not 0 < local_world_size <= world_size or timeout <= 0:
        raise ValueError(
            "Verification requires a command, valid rank counts and timeout"
        )
    environment = {**os.environ, **(env or {})}
    root = (
        dump_dir
        if dump_dir is not None
        else Path(environment.get("VERIFY_SPMD_SAFE_DUMP_DIR", "/tmp/sglang-spmd-safe"))
    )
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="spmd-", dir=root))
    errors = []
    with subprocess.Popen(
        command,
        env={
            **environment,
            "VERIFY_SPMD_SAFE": "1",
            "VERIFY_SPMD_SAFE_DUMP_DIR": str(directory.resolve()),
        },
        start_new_session=True,
    ) as process:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            errors.append(f"Verification startup exceeded {timeout}s")
        finally:
            # Include descendants when a recipe uses a shell or launcher process.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    # A normal serving launcher may exit 1 on EOF when verification workers stop
    # before readiness. Only complete successful worker records admit that exit.
    if process.returncode not in (0, 1):
        errors.append(f"Server launcher exited with code {process.returncode}")
    results = [json.loads(path.read_text()) for path in directory.glob("worker-*.json")]
    errors.extend(dict.fromkeys(error for item in results for error in item["errors"]))
    ranks = sorted(item["rank"] for item in results)
    if not len(ranks) == len(set(ranks)) == local_world_size or any(
        not 0 <= rank < world_size for rank in ranks
    ):
        errors.append(f"Expected {local_world_size} distinct local ranks, got {ranks}")
    keys = results[0]["keys"] if results else []
    if not keys or any(
        not item["passed"] or item["world_size"] != world_size or item["keys"] != keys
        for item in results
    ):
        errors.append("SPMD verification failed or reported inconsistent results")
    result = {
        "passed": not errors,
        "ranks": ranks,
        "world_size": world_size,
        "keys": keys,
        "artifacts": str(directory),
        "errors": errors,
        "launcher_exit_code": process.returncode,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    # Publish the completion marker only after all its contents are visible.
    marker = directory / "result.json.tmp"
    marker.write_text(payload + "\n")
    marker.replace(directory / "result.json")
    print("SPMD_VERIFICATION_RESULT " + payload, flush=True)
    if errors:
        raise RuntimeError(f"SPMD verification failed: {directory}: {errors}")


def wait_for_verification(
    collect: Callable[[], Sequence[Path] | None], *, timeout: int
) -> None:
    """Collect one result per host; the adapter returns None until all hosts exit.

    The adapter raises on failed or restarted hosts and collects their artifacts.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result_files = collect()
        if result_files is not None:
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"SPMD verification did not finish within {timeout}s")

    if not result_files:
        raise RuntimeError("No SPMD verification results")
    results = []
    for path in result_files:
        result = json.loads(path.read_text())
        if not result["passed"]:
            raise RuntimeError(f"{path}: {result['errors']}")
        results.append(result)
    reference = results[0]
    ranks = sorted(rank for result in results for rank in result["ranks"])
    if ranks != list(range(reference["world_size"])):
        raise RuntimeError(f"Incomplete or duplicate verification ranks: {ranks}")
    if not reference["keys"] or any(
        result["world_size"] != reference["world_size"]
        or result["keys"] != reference["keys"]
        for result in results
    ):
        raise RuntimeError("Verification hosts reported inconsistent graph keys")
    print(
        f"SPMD verification passed: {len(reference['keys'])} graphs, {len(ranks)} ranks",
        flush=True,
    )


def run_recipes(
    recipes: Sequence[str],
    launch_recipe: Callable[[str, Path], None],
    *,
    output: Path | None = None,
) -> int:
    """Run recipes with an adapter receiving each recipe and artifact directory.

    The adapter invokes ``verify_startup`` per host, waits for completion, and
    cleans up even on failure. Remote adapters collect artifacts locally and use
    ``wait_for_verification`` to check completion across hosts.
    """
    if not recipes:
        raise ValueError("At least one SPMD-safe recipe is required")
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="tpu-spmd-safe-"))
    failed = []
    for index, recipe in enumerate(recipes):
        directory = output / f"{index}-{Path(recipe).stem}"
        directory.mkdir(parents=True)
        try:
            launch_recipe(recipe, directory)
        except Exception as error:
            failed.append(recipe)
            print(
                f"SPMD verification failed for {recipe}: {error}",
                file=sys.stderr,
                flush=True,
            )
    if failed:
        print(f"Failed recipes: {failed}; artifacts: {output}", file=sys.stderr)
    return int(bool(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one host's SPMD-safe startup")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    verify_startup(
        command,
        world_size=args.world_size,
        local_world_size=args.local_world_size,
        timeout=args.timeout,
        dump_dir=args.dump_dir,
    )


if __name__ == "__main__":
    main()
