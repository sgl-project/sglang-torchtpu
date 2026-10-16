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


class SPMDCompiler:
    """Verify warmup graphs and physical input layouts across a TP group.

    Each rank traces every SPMD-safe specialization. ``compile`` records its
    MLIR and input layouts under the same named key on every rank. ``finish``
    compares the collected records, writes diagnostics, and exits successfully
    or raises on a mismatch. Verification does not compile or execute graphs.

    Returned in place of a native pending compilation so ``direct_compile`` can
    use its usual async path while execution stays deferred until ``finish``.
    """

    def __init__(
        self, group: dist.ProcessGroup, *, verify_spmd_safe: bool = False
    ) -> None:
        """Bind this queue to its collective group and verification output directory."""
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.verify_spmd_safe = verify_spmd_safe
        self.mlirs: dict[str, str] = {}
        self.argument_layouts: dict[str, list[list[int]]] = {}
        self.directory = (
            Path(os.environ["VERIFY_SPMD_SAFE_DUMP_DIR"]) if verify_spmd_safe else None
        )
        if verify_spmd_safe and self.size != dist.get_world_size():
            raise ValueError("SPMD verification requires the full TPU TP group")

    def gather(self, value: T) -> list[T]:
        """Exchange a value with every rank, preserving group-rank order."""
        result: list[Any] = [None] * self.size
        dist.all_gather_object(result, value, group=self.group)
        return result

    def enqueue(self, graph: str) -> None:
        """Reject graph identities already submitted by another compiled boundary."""
        if graph in self.mlirs:
            raise ValueError(f"Duplicate shared TPU graph name: {graph}")

    def compile(
        self, outputs, inputs, *, argument_layouts: list[list[int]], graph: str
    ) -> SPMDCompiler:
        """Collect lowered MLIR and ordered physical input layouts for verification."""
        module = tc.build_mlir(outputs, inputs)
        self.mlirs[graph] = tc.serialize_mlir_text(module, enable_debug_info=False)
        self.argument_layouts[graph] = argument_layouts
        return self

    def is_ready(self) -> bool:
        """Always return False: verification records graphs without an executable."""
        return False

    def get(self) -> Any:
        """Raise if an async caller tries to retrieve a verification executable.

        ``finish`` must exit or raise before the warmup queue waits on its calls.
        Reaching this method means that verification lifecycle was bypassed.
        """
        raise RuntimeError("SPMD verification does not produce an executable")

    def finish(self, error: str | None = None) -> NoReturn:
        """Check all ranks' graphs/layouts, retain diagnostics, and exit or raise."""
        mlir_hashes = {graph: digest(mlir) for graph, mlir in self.mlirs.items()}
        records = self.gather(
            {
                "mlir_hashes": mlir_hashes,
                "argument_layouts": self.argument_layouts,
                "error": error,
            }
        )
        errors = [
            f"rank {rank}: {item['error']}"
            for rank, item in enumerate(records)
            if item["error"] is not None
        ]
        keys = sorted({key for item in records for key in item["mlir_hashes"]})
        if not keys:
            errors.append("No SPMD-safe policy graphs were traced")
        differences = {}
        layout_differences = {}
        for graph in keys:
            peers = [item["mlir_hashes"].get(graph) for item in records]
            if any(item != peers[0] for item in peers):
                differences[graph] = peers
            layouts = [item["argument_layouts"].get(graph) for item in records]
            if any(item != layouts[0] for item in layouts):
                layout_differences[graph] = layouts
        if differences:
            errors.append(
                f"MLIR differs or is missing across ranks: {canonical(differences)}"
            )
        if layout_differences:
            errors.append(
                "Input layouts differ or are missing across ranks (rank order): "
                f"{canonical(layout_differences)}"
            )
        for graph in self.mlirs.keys() & (
            differences.keys() | layout_differences.keys()
        ):
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
            "argument_layouts": self.argument_layouts,
            "differences": differences,
            "layout_differences": layout_differences,
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
        if errors:
            raise RuntimeError(f"SPMD verification failed: {canonical(result)}")
        raise SystemExit(0)

    def clear(self) -> None:
        """Release this queue's retained verification inputs."""
        self.mlirs.clear()
        self.argument_layouts.clear()


def create_spmd_compiler(
    group: dist.ProcessGroup, *, verify_spmd_safe: bool = False
) -> SPMDCompiler | None:
    """Return a coordinator when graph verification is requested."""
    spmd_compiler = SPMDCompiler(group, verify_spmd_safe=verify_spmd_safe)
    return spmd_compiler if verify_spmd_safe else None
