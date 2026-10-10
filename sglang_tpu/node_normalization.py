"""Normalize TPU node order before the existing Engine launches its workers.

Host ranks, the rendezvous leader, and SliceBuilder endpoints follow physical
host tiles in (z, y, x) order. Native within-host device order is unchanged.
The compilation cache must still check the full rank-to-device assignment.

Enabled by default through --tpu-node-normalization; use
--no-tpu-node-normalization to retain the original launcher behavior.
HTTP serving follows normalized rank zero; fixed-leader routing must follow it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import timedelta

from sglang_tpu.patches import build_slicebuilder_addresses

logger = logging.getLogger(__name__)


def dimensions(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    if not result or any(part <= 0 for part in result):
        raise ValueError(f"Expected positive topology dimensions, got {value!r}")
    return result


@dataclass(frozen=True)
class TopologyConfig:
    hosts: tuple[str, ...]
    topology: tuple[int, ...]
    devices_per_host: int = 8
    chips_per_host: tuple[int, ...] = (2, 2, 1)
    bootstrap_port: int = 29590
    master_port: int = 29500
    slicebuilder_port: int = 8070

    def __post_init__(self) -> None:
        if not self.hosts or len(set(self.hosts)) != len(self.hosts):
            raise ValueError("Provide a nonempty list of distinct worker hosts")
        if any(
            not host or ":" in host or any(c.isspace() for c in host)
            for host in self.hosts
        ):
            raise ValueError("Use worker DNS names or IPv4 addresses without ports")
        if len(self.topology) not in (3, 4) or len(self.chips_per_host) != 3:
            raise ValueError("Topology must have 3 or 4 dimensions; chip bounds need 3")
        if any(
            type(n) is not int or n <= 0
            for n in (*self.topology, *self.chips_per_host, self.devices_per_host)
        ):
            raise ValueError(
                "Topology, chip bounds, and device counts must be positive integers"
            )
        if any(
            size % tile for size, tile in zip(self.topology[:3], self.chips_per_host)
        ):
            raise ValueError("Only complete, uniformly tiled hosts are supported")
        host_count = math.prod(
            size // tile for size, tile in zip(self.topology[:3], self.chips_per_host)
        )
        if (
            host_count != len(self.hosts)
            or math.prod(self.topology) != len(self.hosts) * self.devices_per_host
        ):
            raise ValueError("Topology does not match the host and device counts")
        ports = (
            self.bootstrap_port,
            self.master_port,
            *range(
                self.slicebuilder_port, self.slicebuilder_port + self.devices_per_host
            ),
        )
        if len(set(ports)) != len(ports) or any(
            not 1 <= port <= 65535 for port in ports
        ):
            raise ValueError(
                "Bootstrap, master, and SliceBuilder ports must be distinct valid ports"
            )

    @property
    def slicebuilder_addresses(self) -> str:
        return build_slicebuilder_addresses(
            self.hosts, self.slicebuilder_port, self.devices_per_host
        )

    def environment(self, node_rank: int) -> dict[str, str]:
        if not 0 <= node_rank < len(self.hosts):
            raise ValueError("Invalid TPU node rank")
        return {
            "NODE_RANK": str(node_rank),
            "NNODES": str(len(self.hosts)),
            "MASTER_ADDR": self.hosts[0],
            "MASTER_PORT": str(self.master_port),
            "TPU_WORKER_HOSTNAMES": ",".join(self.hosts),
            "TPU_TOPOLOGY": "x".join(map(str, self.topology[:3])),
            "TORCH_TPU_TOPOLOGY": ",".join(map(str, self.topology)),
            "TORCH_TPU_DEVICES_PER_HOST": str(self.devices_per_host),
            "TPU_CHIPS_PER_HOST_BOUNDS": ",".join(map(str, self.chips_per_host)),
            "SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT": str(self.bootstrap_port),
            "TORCH_TPU_BASE_PORT": str(self.slicebuilder_port),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": self.slicebuilder_addresses,
        }


def canonical_host_order(ranks: list[int]) -> list[int]:
    """Invert the complete host-to-rank assignment returned by TorchTPU."""
    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(
        range(len(ranks))
    ):
        raise ValueError(f"Duplicate or incomplete physical host placement: {ranks}")
    return sorted(range(len(ranks)), key=ranks.__getitem__)


def _coordinate_worker(local_rank, config, host_index, timeout, writer):
    # As with scheduler workers, set rank metadata before initializing PJRT.
    os.environ.update(config.environment(host_index))
    os.environ.update(
        RANK=str(host_index * config.devices_per_host + local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(len(config.hosts) * config.devices_per_host),
        LOCAL_WORLD_SIZE=str(config.devices_per_host),
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import torch.distributed as dist
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.distributed.launchers.multihost_wrapper import (
        calculate_topological_node_rank,
    )

    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()
    dist.init_process_group("gloo", timeout=timedelta(seconds=timeout))
    try:
        coords = tpu_distributed.local_device_coords()
        if local_rank == 0:
            # TorchTPU accepts partial coordinates; discovery requires an exact
            # valid chip location and must never fall back to the pod index.
            if len(coords) != 3 or any(
                type(c) is not int or not 0 <= c < bound
                for c, bound in zip(coords, config.topology)
            ):
                raise ValueError(f"Invalid slice-relative TPU coordinates: {coords}")
            writer.send(
                calculate_topological_node_rank(
                    coords,
                    ",".join(map(str, config.topology)),
                    fallback_index=None,
                    chips_per_host=config.chips_per_host,
                )
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
        writer.close()


def _exit_without_native_teardown(local_rank, error_dir, worker, *args):
    # libtpu's atexit teardown sends vBAR StopSession, which can hang forever
    # on GKE. Exit like a killed server instead: closing the device files
    # releases the chips for the scheduler workers that start next. Failures
    # too: mp.spawn would record them and then run the teardown via sys.exit.
    status = 0
    try:
        worker(local_rank, *args)
    except Exception:
        with open(os.path.join(error_dir, str(local_rank)), "w") as file:
            file.write(traceback.format_exc())
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def query_node_rank(config: TopologyConfig, host_index: int, timeout: int) -> int:
    # GKE needs a full SliceBuilder world before local_device_coords() works.
    # Use disposable spawn workers: discovery must release PJRT before the
    # existing scheduler launcher starts model workers. No torchrun/server CLI.
    import torch.multiprocessing as mp

    reader, writer = mp.get_context("spawn").Pipe(duplex=False)
    error_dir = tempfile.mkdtemp(prefix="sglang-tpu-discovery-")
    context = None
    deadline = time.monotonic() + timeout
    try:
        context = mp.spawn(
            _exit_without_native_teardown,
            args=(error_dir, _coordinate_worker, config, host_index, timeout, writer),
            nprocs=config.devices_per_host,
            join=False,
        )
        writer.close()
        try:
            while not context.join(timeout=max(0, deadline - time.monotonic())):
                if time.monotonic() >= deadline:
                    raise TimeoutError("TPU coordinate discovery timed out")
        except mp.ProcessExitedException:
            # One failure breaks its peers' collectives, often before the
            # failing worker records its own error. Let peers exit briefly,
            # then report each distinct error rather than whichever landed first.
            grace = time.monotonic() + 5
            for process in context.processes:
                process.join(timeout=max(0, min(grace, deadline) - time.monotonic()))
            errors = {}
            for name in sorted(os.listdir(error_dir), key=int):
                with open(os.path.join(error_dir, name)) as file:
                    text = file.read()
                # Peer addresses and ports differ; the failure kind does not.
                last = text.strip().splitlines()[-1]
                kind = re.sub(r"\d+", "N", re.sub(r"\[[^\]/]*\]", "[peer]", last))
                errors.setdefault(kind, f"worker {name}: {text}")
            if not errors:
                raise
            raise RuntimeError(
                "TPU coordinate workers failed:\n" + "\n".join(errors.values())
            ) from None
        rank = reader.recv()
        if type(rank) is not int or not 0 <= rank < len(config.hosts):
            raise ValueError(f"TorchTPU returned invalid node rank: {rank}")
        return rank
    finally:
        if context is not None:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            for process in context.processes:
                process.join()
        reader.close()
        writer.close()
        shutil.rmtree(error_dir, ignore_errors=True)


def discover_host_order(
    config: TopologyConfig, host_index: int, timeout: int
) -> list[int]:
    """Exchange probe results on a CPU-only store before assigning model ranks."""
    import torch.distributed as dist

    if not 0 <= host_index < len(config.hosts) or timeout <= 0:
        raise ValueError("Invalid host index or timeout")
    store = dist.TCPStore(
        config.hosts[0],
        config.bootstrap_port,
        len(config.hosts),
        host_index == 0,
        timeout=timedelta(seconds=timeout),
    )
    record = {"config": asdict(config), "rank": None, "error": None}
    try:
        record["rank"] = query_node_rank(config, host_index, timeout)
    except Exception as error:
        # Publish discovery failures so every host rejects the launch.
        record["error"] = f"{type(error).__name__}: {error}"
    store.set(f"host/{host_index}", json.dumps(record))
    records = [
        json.loads(store.get(f"host/{index}")) for index in range(len(config.hosts))
    ]
    # Acknowledge reads before validation too: the owner must not close the
    # store while a peer is still retrieving the shared failure report.
    store.set(f"done/{host_index}", "1")
    if host_index == 0:
        store.wait([f"done/{index}" for index in range(len(config.hosts))])
    errors = [entry["error"] for entry in records]
    if any(errors):
        raise RuntimeError(
            f"TPU coordinate discovery failed; refusing pod-index fallback: {errors}"
        )
    if any(entry["config"] != records[0]["config"] for entry in records):
        raise ValueError("Hosts disagree on the TPU launch configuration")
    return canonical_host_order([entry["rank"] for entry in records])


def prepare_tpu_environment(server_args) -> None:
    """Normalize multi-host TPU launches before publishing config or spawning.

    Called by the existing Engine startup path, not a replacement launcher.
    Constructing or resolving ServerArgs alone never opens a rendezvous.

    Launch on every host with the same TPU_WORKER_HOSTNAMES and full
    TORCH_TPU_TOPOLOGY, plus --nnodes, --node-rank (input host index), and
    --dist-init-addr. --tpu-node-normalization defaults to on. Discovery uses
    dist-init-addr's port plus one for its CPU store, unless
    SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT is set. --dist-timeout applies to
    discovery too (default 120 seconds).

    Single-host TPU and other devices are unchanged. Multi-host Ray, partial
    slices, and external distributed-init overrides are unsupported. HTTP
    serving follows normalized rank zero, which may move to another pod. The
    input leader forwards its HTTP and PD-bootstrap ports to that host; bind
    serving to a pod-reachable address on every host. Cache checks remain
    required even with normalized ranks.
    """
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    cfg = resolving_view(server_args)
    if not cfg.tpu_node_normalization or cfg.device != "tpu" or cfg.nnodes == 1:
        return
    if cfg.use_ray:
        raise ValueError(
            "Coordinate-ordered multi-host TPU launches do not support Ray"
        )
    if os.environ.get("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"):
        raise ValueError("TPU topology normalization requires its own rendezvous")
    hosts = cfg.tpu_worker_hostnames or os.environ.get("TPU_WORKER_HOSTNAMES", "")
    topology = cfg.torch_tpu_topology or os.environ.get("TORCH_TPU_TOPOLOGY", "")
    if not hosts or not topology or not cfg.dist_init_addr:
        raise ValueError(
            "Multi-host TPU startup requires TPU_WORKER_HOSTNAMES, "
            "TORCH_TPU_TOPOLOGY, and --dist-init-addr (or their ServerArgs values)"
        )
    master_host, separator, port = cfg.dist_init_addr.rpartition(":")
    if not master_host or not separator:
        raise ValueError("--dist-init-addr must contain a host and port")
    master_port = int(port)
    config = TopologyConfig(
        hosts=tuple(hosts.split(",")),
        topology=dimensions(topology),
        devices_per_host=int(os.environ.get("TORCH_TPU_DEVICES_PER_HOST", "8")),
        chips_per_host=dimensions(os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")),
        bootstrap_port=int(
            os.environ.get("SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT", master_port + 1)
        ),
        master_port=master_port,
        slicebuilder_port=int(os.environ.get("TORCH_TPU_BASE_PORT", "8070")),
    )
    if (
        cfg.nnodes != len(config.hosts)
        or cfg.tp_size * cfg.pp_size != len(config.hosts) * config.devices_per_host
    ):
        raise ValueError("TPU topology normalization requires all devices in one slice")
    if not 0 <= cfg.node_rank < cfg.nnodes:
        raise ValueError("Invalid TPU node rank")
    if (
        cfg.torch_tpu_slicebuilder_addresses is not None
        and cfg.torch_tpu_slicebuilder_addresses != config.slicebuilder_addresses
    ):
        raise ValueError("SliceBuilder addresses conflict with the TPU host order")

    order = discover_host_order(config, cfg.node_rank, cfg.dist_timeout or 120)
    config = replace(config, hosts=tuple(config.hosts[index] for index in order))
    node_rank = order.index(cfg.node_rank)
    dist_init_addr = f"{config.hosts[0]}:{master_port}"
    fields = dict(
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
        tpu_worker_hostnames=",".join(config.hosts),
        torch_tpu_topology=",".join(map(str, config.topology)),
        torch_tpu_slicebuilder_addresses=config.slicebuilder_addresses,
    )
    current = dict(
        node_rank=cfg.node_rank,
        dist_init_addr=cfg.dist_init_addr,
        tpu_worker_hostnames=cfg.tpu_worker_hostnames,
        torch_tpu_topology=cfg.torch_tpu_topology,
        torch_tpu_slicebuilder_addresses=cfg.torch_tpu_slicebuilder_addresses,
    )
    if fields != current:
        declare_resolution(server_args, "tpu.normalize_host_ranks", **fields)
    os.environ.update(config.environment(node_rank))
    logger.info(
        "TPU normalized node rank=%s, rendezvous=%s; route serving traffic to node rank 0",
        node_rank,
        dist_init_addr,
    )
