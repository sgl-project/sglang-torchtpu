"""Queue SPMD-safe graphs for cross-rank verification or shared compilation."""

from __future__ import annotations

import gzip
import importlib.metadata
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import torch
import torch.distributed as dist
import zstandard
from torch_tpu._internal.compile import tpu_torch_compile as tc
from torch_tpu._internal.distributed import device_mesh

from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.graph_identity import canonical, digest

logger = get_logger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ExecutableInterface:
    """Tensor-free bindings needed to load another rank's executable."""

    input_digest: str
    input_specs: tuple
    argument_layouts: list
    model_indices: tuple[int, ...]
    output_treespec: Any


@dataclass
class Submission:
    """One full graph key's pending shared executable."""

    owner: int
    bind: Callable[[ExecutableInterface], None] | None = None
    interface: ExecutableInterface | None = None
    pending: Any = None
    executable: Any = None

    def is_ready(self) -> bool:
        return self.executable is not None

    def get(self) -> Any:
        if self.executable is None:
            raise RuntimeError("Shared compilation must finish before graph execution")
        return self.executable


def _prepare_executable(
    submission: Submission, serialize: bool
) -> tuple[Any, bytes | None]:
    try:
        executable = submission.pending.get()
    finally:
        submission.pending = None
    serialized = None
    if serialize:
        serialized = zstandard.ZstdCompressor(level=1).compress(
            tc.serialize_executable(executable)
        )
    return executable, serialized


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
        self.submissions: dict[str, Submission] = {}
        self.mlirs: dict[str, str] = {}
        self.argument_layouts: dict[str, list[list[int]]] = {}
        self.directory = (
            Path(os.environ["VERIFY_SPMD_SAFE_DUMP_DIR"]) if verify_spmd_safe else None
        )
        self.fallback_reason: str | None = None
        self._serializer: ThreadPoolExecutor | None = None
        if verify_spmd_safe:
            if self.size != dist.get_world_size():
                raise ValueError("SPMD verification requires the full TPU TP group")
            return
        runtime = None
        error: str | None = None
        try:
            mapping = [int(value) for value in device_mesh.get_device_assignment()]
            runtime = (
                mapping,
                [
                    importlib.metadata.version(n)
                    for n in ("torch", "torch-tpu", "jax", "libtpu")
                ],
                os.environ.get("XLA_FLAGS", ""),
                os.environ.get("LIBTPU_INIT_ARGS", ""),
            )
            if self.size != len(mapping) or self.size != dist.get_world_size():
                error = "only the full TPU TP group is supported"
        except Exception as caught:
            error = f"cannot identify the compiler runtime: {type(caught).__name__}: {caught}"
        records = self.gather((runtime, error))
        reasons = sorted({reason for _, reason in records if reason is not None})
        if any(peer != runtime for peer, _ in records):
            reasons.append("device mappings or runtimes differ across ranks")
        self.fallback_reason = "; ".join(reasons) or None
        self._serializer = ThreadPoolExecutor(thread_name_prefix="tpu-serialize")

    def gather(self, value: T) -> list[T]:
        """Exchange a value with every rank, preserving group-rank order."""
        result: list[Any] = [None] * self.size
        dist.all_gather_object(result, value, group=self.group)
        return result

    @contextmanager
    def coordinate_errors(self, stage: str) -> Iterator[None]:
        error: str | None = None
        try:
            yield
        except Exception as caught:
            logger.exception("Shared compile %s failed on rank %d", stage, self.rank)
            error = f"{type(caught).__name__}: {str(caught)[:512]}"
        errors = self.gather(error)
        if any(error is not None for error in errors):
            raise RuntimeError(f"Shared compile {stage} failed: {errors}")

    def enqueue(self, graph: str) -> None:
        if self.verify_spmd_safe:
            if graph in self.mlirs:
                raise ValueError(f"Duplicate SPMD verification graph name: {graph}")
            return
        if graph in self.submissions:
            raise ValueError(f"Duplicate shared TPU graph name: {graph}")
        self.submissions[graph] = Submission(owner=len(self.submissions) % self.size)

    def compile(
        self,
        outputs,
        inputs,
        *,
        argument_layouts: list[list[int]],
        interface: ExecutableInterface | None = None,
        graph: str,
    ) -> Submission | SPMDCompiler:
        if self.verify_spmd_safe:
            module = tc.build_mlir(outputs, inputs)
            self.mlirs[graph] = tc.serialize_mlir_text(module, enable_debug_info=False)
            self.argument_layouts[graph] = argument_layouts
            return self
        submission = self.submissions[graph]
        if interface is None:
            raise ValueError("Shared TPU compilation requires an executable interface")
        if submission.owner != self.rank:
            raise ValueError(f"Invalid TPU compile owner: {graph}")
        submission.interface = interface
        submission.pending = tc.async_traverse_and_compile(
            outputs, inputs, argument_layouts=interface.argument_layouts
        )
        return submission

    def is_ready(self) -> bool:
        """Always return False: verification records graphs without an executable."""
        return False

    def get(self) -> Any:
        """Raise if an async caller tries to retrieve a verification executable.

        ``finish`` must exit or raise before the warmup queue waits on its calls.
        Reaching this method means that verification lifecycle was bypassed.
        """
        raise RuntimeError("SPMD verification does not produce an executable")

    def finish(self, error: str | None = None) -> None:
        if self.verify_spmd_safe:
            self._finish_verification(error)
        errors = self.gather(error)
        if any(error is not None for error in errors):
            raise RuntimeError(f"Shared compile tracing failed: {errors}")
        assignments = [(graph, item.owner) for graph, item in self.submissions.items()]
        if any(peer != assignments for peer in self.gather(assignments)):
            raise ValueError("Shared TPU graph order or owners differ across ranks")
        start = time.perf_counter()
        if self.rank == 0:
            for graph, owner in assignments:
                logger.info("[shared_compile] owner=%d graph=%s", owner, graph)
        self._distribute()
        logger.info(
            "[shared_compile] compile/distribute wait=%.3fs",
            time.perf_counter() - start,
        )

    def _finish_verification(self, error: str | None) -> NoReturn:
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

    def _distribute(self) -> None:
        owned = {
            graph: item.interface
            for graph, item in self.submissions.items()
            if item.owner == self.rank
        }
        interfaces = self.gather(owned)
        with self.coordinate_errors("consumer bindings"):
            for graph, item in self.submissions.items():
                if item.owner == self.rank:
                    continue
                interface = interfaces[item.owner][graph]
                if interface is None or item.bind is None:
                    raise ValueError(f"Incomplete shared executable interface: {graph}")
                item.bind(interface)
                item.bind = None
        logger.info(
            "[shared_compile] traced=%d reused=%d cases=%d",
            len(owned),
            len(self.submissions) - len(owned),
            len(self.submissions),
        )
        # Serialize concurrently, then broadcast in the agreed queue order.
        prepared = {
            graph: self._serializer.submit(_prepare_executable, item, self.size > 1)
            for graph, item in self.submissions.items()
            if item.owner == self.rank
        }
        for graph, item in self.submissions.items():
            executable = serialized = buffer = None
            header = [None]
            if self.rank == item.owner:
                try:
                    executable, serialized = prepared.pop(graph).result()
                    header[0] = (
                        True,
                        len(serialized) if serialized is not None else None,
                    )
                except Exception as caught:
                    header[0] = (False, f"{type(caught).__name__}: {str(caught)[:512]}")
            source = dist.get_global_rank(self.group, item.owner)
            dist.broadcast_object_list(header, src=source, group=self.group)
            success, length = header[0]
            with self.coordinate_errors("preparing transfer"):
                if not success:
                    raise RuntimeError(length)
                if length is not None:
                    buffer = (
                        torch.frombuffer(bytearray(serialized), dtype=torch.uint8)
                        if self.rank == item.owner
                        else torch.empty(length, dtype=torch.uint8, device="cpu")
                    )
            if buffer is not None:
                dist.broadcast(buffer, src=source, group=self.group)
            with self.coordinate_errors("loading executable"):
                if executable is None:
                    executable = tc.load_serialized_executable(
                        zstandard.ZstdDecompressor().decompress(
                            buffer.numpy().tobytes()
                        )
                    )
                item.executable = executable

    def clear(self) -> None:
        if self._serializer is not None:
            self._serializer.shutdown(wait=True, cancel_futures=True)
        for item in self.submissions.values():
            if item.pending is not None:
                try:
                    item.pending.get()
                except Exception:
                    logger.exception("TPU compilation failed during cleanup")
            item.pending = item.bind = item.interface = None
        self.submissions.clear()
        self.mlirs.clear()
        self.argument_layouts.clear()


def create_spmd_compiler(
    group: dist.ProcessGroup, *, verify_spmd_safe: bool = False
) -> SPMDCompiler | None:
    spmd_compiler = SPMDCompiler(group, verify_spmd_safe=verify_spmd_safe)
    if spmd_compiler.fallback_reason is None:
        return spmd_compiler
    logger.warning(
        "[shared_compile] Compiling locally on every rank: %s",
        spmd_compiler.fallback_reason,
    )
    spmd_compiler.clear()
    return None
