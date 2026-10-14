"""Queue SPMD-safe graphs and verify their equivalence across ranks."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import torch.distributed as dist
from torch_tpu._internal.compile import tpu_torch_compile as tc

from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.graph_identity import canonical, digest

logger = get_logger(__name__)
T = TypeVar("T")


class SharedCompiler:
    """One queue across policies, traced on every rank when verifying SPMD safety."""

    def __init__(
        self, group: dist.ProcessGroup, *, verify_spmd_safe: bool = False
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.verify_spmd_safe = verify_spmd_safe
        self.mlirs: dict[str, str] = {}
        self.directory = (
            Path(os.environ["VERIFY_SPMD_SAFE_DUMP_DIR"]) if verify_spmd_safe else None
        )
        if verify_spmd_safe and self.size != dist.get_world_size():
            raise ValueError("SPMD verification requires the full TPU TP group")

    def gather(self, value: T) -> list[T]:
        result: list[Any] = [None] * self.size
        dist.all_gather_object(result, value, group=self.group)
        return result

    def enqueue(self, graph: str) -> None:
        if graph in self.mlirs:
            raise ValueError(f"Duplicate shared TPU graph name: {graph}")

    def compile(self, outputs, inputs, *, graph: str) -> SharedCompiler:
        module = tc.build_mlir(outputs, inputs)
        self.mlirs[graph] = tc.serialize_mlir_text(module, enable_debug_info=False)
        return self

    def is_ready(self) -> bool:
        return False

    def get(self) -> Any:
        raise RuntimeError("SPMD verification does not produce an executable")

    def finish(self, error: str = "") -> NoReturn:
        mlir_hashes = {graph: digest(mlir) for graph, mlir in self.mlirs.items()}
        records = self.gather({"mlir_hashes": mlir_hashes, "error": error})
        errors = [
            f"rank {rank}: {item['error']}"
            for rank, item in enumerate(records)
            if item["error"]
        ]
        keys = sorted({key for item in records for key in item["mlir_hashes"]})
        if not keys:
            errors.append("No SPMD-safe policy graphs were traced")
        differences = {}
        for graph in keys:
            peers = [item["mlir_hashes"].get(graph) for item in records]
            if any(item != peers[0] for item in peers):
                differences[graph] = peers
        if differences:
            errors.append(
                f"{len(differences)} graph(s) differ or are missing across ranks"
            )
        for graph in self.mlirs.keys() & differences.keys():
            prefix = self.directory / f"{digest(graph)}-rank-{self.rank}"
            with gzip.open(f"{prefix}.mlir.gz", "wt") as output:
                output.write(self.mlirs[graph])
        result = {
            "passed": not errors,
            "world_size": self.size,
            "rank": self.rank,
            "pid": os.getpid(),
            "keys": keys,
            "mlir_hashes": mlir_hashes,
            "differences": differences,
            "errors": errors,
        }
        (self.directory / f"worker-{self.rank}.json").write_text(
            canonical(result) + "\n"
        )
        # Every worker has retained its evidence before the parent observes exits.
        dist.barrier(group=self.group)
        if self.rank == 0:
            logger.info("SPMD_VERIFICATION_GRAPHS %s", canonical(result))
        # Verification never executes pending calls or announces server readiness.
        raise SystemExit(0 if result["passed"] else 1)

    def clear(self) -> None:
        self.mlirs.clear()


def create_shared_compile(
    group: dist.ProcessGroup, *, verify_spmd_safe: bool = False
) -> SharedCompiler | None:
    shared_compile = SharedCompiler(group, verify_spmd_safe=verify_spmd_safe)
    return shared_compile if verify_spmd_safe else None
