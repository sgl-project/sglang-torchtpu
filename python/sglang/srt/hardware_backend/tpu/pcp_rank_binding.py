# SPDX-License-Identifier: Apache-2.0
"""PCP LOCAL_RANK remap — parent-side subprocess probe + spawn wiring.

Opt-in via ``SGLANG_TPU_PCP_LOCAL_RANK_REMAP`` (default off) because on
the current 8-chip v7 target the remap is a no-op (JAX device enumeration
already matches process rank order) and enabling it unconditionally would
add ~1-2s startup latency for the subprocess probe on every launch. Data
containers use ``msgspec.Struct`` per the no-dataclasses rule.

Why: when ``pcp_size > 1``, the PCP kernel's ``pltpu.make_async_remote_copy``
ring schedule assumes CP rank ``r`` is on ``jax.devices()[r]``. If
sglang spawns a worker with LOCAL_RANK=k that binds to a different chip,
the CP ring traverses non-adjacent ICI links (slow) or racy links (may
scramble KV writes). This module computes the LOCAL_RANK -> physical
chip mapping via a one-shot subprocess probe and rewrites LOCAL_RANK on
each real worker so ``native_local_rank=k`` binds to ``jax.devices()[k]``.

On a single-host v7 slice this is normally a no-op — the probe just
verifies alignment. On multi-slice / older TPU generations where JAX
device order != torch_tpu bind order it is load-bearing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


_PCP_LOCAL_RANK_REMAPS: dict[int, tuple[tuple[int, ...], str]] = {}
_PCP_REMAP_PROBE_PREFIX = "SGLANG_TPU_PCP_REMAP_PROBE "
_PCP_REMAP_PROBE_TIMEOUT_S = 120.0
_PCP_REMAP_PROBE_SCRIPT = r"""
import json
import os
import sys
import traceback

prefix = "SGLANG_TPU_PCP_REMAP_PROBE "
payload = {
    "rank": int(os.environ["RANK"]),
    "local_rank": int(os.environ["LOCAL_RANK"]),
}
try:
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.distributed import tpu_distributed

    init_method = (
        f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="tpu_dist",
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    # Touch TPU before querying binding metadata.
    torch.empty((1,), device="tpu").cpu()
    import jax
    payload["global_device_id"] = int(tpu_distributed.global_device_id())
    payload["jax_device_ids"] = [int(device.id) for device in jax.devices()]

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)
except BaseException as exc:
    payload["error"] = repr(exc)
    traceback.print_exc()
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)
    sys.exit(1)
"""


def _validate_remap(world_size: int, remap: Sequence[int]) -> tuple[int, ...]:
    """Ensure ``remap`` is a permutation of ``0..world_size-1``."""
    remap_tuple = tuple(int(item) for item in remap)
    if len(remap_tuple) != world_size or sorted(remap_tuple) != list(range(world_size)):
        raise ValueError(
            "PCP local rank remap must be a permutation of "
            f"0..{world_size - 1}, got {remap_tuple}."
        )
    return remap_tuple


def compute_pcp_local_rank_remap(
    local_rank_to_device_id: Sequence[int],
    jax_device_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return native PCP rank -> TorchTPU LOCAL_RANK remap.

    ``local_rank_to_device_id[k]`` == global PjRt device id that TorchTPU
    binds to when the child spawns with LOCAL_RANK=k.
    ``jax_device_ids[k]`` == the k-th device returned by ``jax.devices()``.
    Result: ``remap[k]`` = the LOCAL_RANK value that binds to
    ``jax.devices()[k]``.
    """
    local_ids = tuple(int(d) for d in local_rank_to_device_id)
    jax_ids = tuple(int(d) for d in jax_device_ids)
    if len(local_ids) != len(jax_ids):
        raise RuntimeError(
            "PCP remap probe produced mismatched result lengths: "
            f"local_rank_to_device_id={local_ids}, jax_device_ids={jax_ids}."
        )
    if len(set(local_ids)) != len(local_ids):
        raise RuntimeError(
            f"PCP remap probe produced duplicate TorchTPU device ids: {local_ids}."
        )
    if len(set(jax_ids)) != len(jax_ids):
        raise RuntimeError(
            f"PCP remap probe produced duplicate JAX device ids: {jax_ids}."
        )
    device_to_local_rank = {
        device_id: local_rank for local_rank, device_id in enumerate(local_ids)
    }
    missing = tuple(d for d in jax_ids if d not in device_to_local_rank)
    if missing:
        raise RuntimeError(
            "PCP remap probe JAX device ids missing from TorchTPU bindings: "
            f"missing={missing}, local_rank_to_device_id={local_ids}, "
            f"jax_device_ids={jax_ids}."
        )
    return tuple(device_to_local_rank[d] for d in jax_ids)


def probe_pcp_local_rank_remap(
    world_size: int, *, get_topology: Callable[[int], str]
) -> tuple[int, ...]:
    """Spawn ``world_size`` short-lived TPU probes; compute the remap.

    Each probe runs the payload above with a distinct LOCAL_RANK and
    reports back ``tpu_distributed.global_device_id()`` +
    ``[d.id for d in jax.devices()]``. The remap is then the permutation
    that aligns native rank k with jax.devices()[k].
    """
    import portpicker

    master_port = portpicker.pick_unused_port()
    sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    sb_addresses = ",".join(f"localhost:{port}" for port in sb_ports)
    topology = os.environ.get("TORCH_TPU_TOPOLOGY") or get_topology(world_size)

    logger.info(
        "[PCP] probing LOCAL_RANK remap for world_size=%d topology=%s.",
        world_size,
        topology,
    )
    with tempfile.TemporaryDirectory(prefix="sglang-tpu-pcp-remap-probe-") as tmpdir:
        procs: list[tuple[int, subprocess.Popen, object, str]] = []
        for rank in range(world_size):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_WORLD_SIZE": str(world_size),
                    "GROUP_RANK": "0",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(master_port),
                    "TORCH_TPU_TOPOLOGY": topology,
                    "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
                    "TORCH_TPU_XPROF_SESSION_ID": str(time.time_ns()),
                }
            )
            log_path = os.path.join(tmpdir, f"rank-{rank}.log")
            log_file = open(log_path, "w+", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                [sys.executable, "-c", _PCP_REMAP_PROBE_SCRIPT],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            procs.append((rank, proc, log_file, log_path))

        deadline = time.monotonic() + _PCP_REMAP_PROBE_TIMEOUT_S
        timed_out = False
        while True:
            if all(proc.poll() is not None for _, proc, _, _ in procs):
                break
            if time.monotonic() >= deadline:
                timed_out = True
                for _, proc, _, _ in procs:
                    if proc.poll() is None:
                        proc.kill()
                break
            time.sleep(0.1)

        payloads: dict[int, dict] = {}
        logs: list[str] = []
        failed_returncodes: dict[int, int] = {}
        for rank, proc, log_file, log_path in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                timed_out = True
                if proc.poll() is None:
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            log_file.seek(0)
            log_text = log_file.read()
            log_file.close()
            logs.append(f"rank {rank} log ({log_path}):\n{log_text}")
            for line in log_text.splitlines():
                if line.startswith(_PCP_REMAP_PROBE_PREFIX):
                    payload = json.loads(line[len(_PCP_REMAP_PROBE_PREFIX) :])
                    payloads[int(payload["local_rank"])] = payload
            if proc.returncode is None:
                timed_out = True
            elif proc.returncode != 0:
                failed_returncodes[rank] = int(proc.returncode)
        if timed_out or any(code == -9 for code in failed_returncodes.values()):
            raise RuntimeError(
                "PCP LOCAL_RANK remap probe timed out.\n" + "\n".join(logs)
            )
        if failed_returncodes:
            raise RuntimeError(
                "PCP LOCAL_RANK remap probe failed with returncodes "
                f"{failed_returncodes}.\n" + "\n".join(logs)
            )
        expected_ranks = set(range(world_size))
        if set(payloads) != expected_ranks:
            raise RuntimeError(
                "PCP LOCAL_RANK remap probe did not report every rank: "
                f"expected={sorted(expected_ranks)}, got={sorted(payloads)}."
            )
        errored = [p for p in payloads.values() if "error" in p]
        if errored:
            raise RuntimeError(
                f"PCP LOCAL_RANK remap probe reported errors: {errored}."
            )

        ordered = [payloads[local_rank] for local_rank in range(world_size)]
        local_rank_to_device_id = tuple(int(p["global_device_id"]) for p in ordered)
        first_jax_device_ids = tuple(int(d) for d in ordered[0]["jax_device_ids"])
        for p in ordered[1:]:
            jax_device_ids = tuple(int(d) for d in p["jax_device_ids"])
            if jax_device_ids != first_jax_device_ids:
                raise RuntimeError(
                    "PCP LOCAL_RANK remap probe saw inconsistent JAX device "
                    f"orders across ranks: rank0={first_jax_device_ids}, "
                    f"rank{p['local_rank']}={jax_device_ids}."
                )
        return compute_pcp_local_rank_remap(
            local_rank_to_device_id, first_jax_device_ids[:world_size]
        )


def ensure_pcp_local_rank_remap(
    world_size: int, *, get_topology: Callable[[int], str]
) -> tuple[int, ...]:
    """Return the cached remap or compute-and-cache it. Cached per world_size."""
    entry = _PCP_LOCAL_RANK_REMAPS.get(int(world_size))
    if entry is not None:
        return entry[0]
    remap = probe_pcp_local_rank_remap(world_size, get_topology=get_topology)
    _PCP_LOCAL_RANK_REMAPS[int(world_size)] = (
        _validate_remap(int(world_size), remap),
        f"dynamic probe world_size={world_size}",
    )
    logger.info(
        "[PCP] LOCAL_RANK remap for world_size=%d: %s (identity=%s).",
        world_size,
        remap,
        remap == tuple(range(world_size)),
    )
    return _PCP_LOCAL_RANK_REMAPS[int(world_size)][0]


def get_default_tpu_topology(world_size: int) -> str:
    """Fallback topology getter — uses torch_tpu's shape probe."""
    from torch_tpu._internal.utils import hardware

    topology = hardware.get_tpu_topology(world_size)
    if topology is None:
        raise ValueError("No TPU devices found for topology probe.")
    return topology


def pcp_local_rank_remap_enabled() -> bool:
    """True iff the caller should apply the PCP local-rank remap."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_TPU_PCP_LOCAL_RANK_REMAP.get())


def clear_pcp_local_rank_remaps() -> None:
    """Drop cached remaps — for tests."""
    _PCP_LOCAL_RANK_REMAPS.clear()
