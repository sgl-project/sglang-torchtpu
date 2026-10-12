# SPDX-License-Identifier: Apache-2.0
"""Invalidate torch.compile caches when TPU sources or device assignments change.

torch.compile cannot detect changes to Pallas kernel implementations because
they are hidden in torch.library custom ops. During TPU backend initialization,
we hash all Pallas source code in sglang-torchtpu and include the digest in the
compile cache key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def add_tpu_rank_assignment_to_compile_cache_key() -> None:
    """Scope opaque collective executables to the live WORLD device assignment.

    Call after distributed initialization, before creating compile wrappers.
    Repeated calls from target and draft runners preserve the process-wide tag.
    Standalone compilation must call this explicitly if it bypasses TpuGraphRunner.
    Querying the native assignment avoids another WORLD collective.
    """
    import torch.compiler.config as compiler_config
    import torch.distributed as dist
    from torch_tpu._internal.distributed.device_mesh import get_device_assignment

    world_size = dist.get_world_size()
    if world_size == 1:
        return
    rank_to_device = get_device_assignment()
    if (
        len(rank_to_device) != world_size
        or any(type(device) is not int or device < 0 for device in rank_to_device)
        or len(set(rank_to_device)) != world_size
    ):
        raise RuntimeError(f"Invalid TPU rank assignment: {rank_to_device}")

    # FX only sees the custom op/group name, not its compiled device ordering.
    assignment = {
        "rank": dist.get_rank(),
        "rank_to_device": rank_to_device,
        "topology": os.environ.get("TORCH_TPU_TOPOLOGY", ""),
    }
    digest = hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()
    tag = f"tpu-rank-assignment-v1:{digest}"
    existing_tag = str(compiler_config.cache_key_tag or "")
    if tag in existing_tag.split("|"):
        return
    compiler_config.cache_key_tag = f"{existing_tag}|{tag}" if existing_tag else tag
    logger.info("TPU compile cache rank assignment: %s, tag=%s", assignment, tag)


def get_pallas_kernel_sources() -> tuple[str, ...]:
    """Return Pallas source files and directories, relative to the sglang_tpu package.

    PLEASE add every new Pallas kernel to this list. Missing sources can cause
    torch.compile to reuse stale artifacts and produce confusing behavior.
    """
    return (
        "attention/pallas",
        "moe/pallas",
        "attention_pallas.py",
        "attention_pallas_mla.py",
        "attention_pallas_mla_sparse.py",
        "attention_pallas_rpa_base.py",
        "attention_pallas_rpa_batched.py",
        "attention_pallas_rpa_v3.py",
        "distributed_topk_pallas.py",
        "dsa_indexer_tpu.py",
        "dsv4/sparsecore_topk.py",
        "dsv4/streamindex_topk",
        "kernels",
        "pallas_export.py",
        "pallas_jax_op.py",
        "quantized_matmul.py",
    )


def add_pallas_kernel_sources_to_compile_cache_key() -> None:
    """Add Pallas kernel sources to torch.compile cache keys.

    Pallas custom ops are opaque in FX graphs, so source changes alone do not
    invalidate cached executables. Existing user tags are preserved.
    """
    import torch.compiler.config as compiler_config

    package_root = Path(__file__).resolve().parent
    source_digest = _hash_source_paths(package_root, get_pallas_kernel_sources())
    existing_tag = str(compiler_config.cache_key_tag or "")
    cache_tag = f"{existing_tag}|{source_digest}" if existing_tag else source_digest

    # Mutate the process-wide torch.compile configuration before compilation.
    compiler_config.cache_key_tag = cache_tag


def _hash_source_paths(source_root: Path, relative_paths: Iterable[str]) -> str:
    """Hash relative source paths and contents, independent of checkout location."""
    source_root = source_root.resolve()
    hasher = hashlib.sha256()

    files = _iter_source_files(source_root / path for path in relative_paths)
    if not files:
        raise RuntimeError("No TPU Pallas cache-key source files were found")

    for path in files:
        relative_path = path.resolve().relative_to(source_root).as_posix().encode()
        contents = path.read_bytes()
        hasher.update(len(relative_path).to_bytes(8, "big"))
        hasher.update(relative_path)
        hasher.update(len(contents).to_bytes(8, "big"))
        hasher.update(contents)
    return hasher.hexdigest()


def _iter_source_files(source_paths: Iterable[Path]) -> list[Path]:
    """Return sorted unique Python source files."""
    files: set[Path] = set()
    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"Pallas source path not found: {source_path}")
        paths = source_path.rglob("*") if source_path.is_dir() else (source_path,)
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                files.add(path)
    return sorted(files)


def initialize_jax_compile_cache() -> None:
    """Build JAX's persistent compilation cache now instead of on first compile."""
    from jax._src import compilation_cache

    compilation_cache._initialize_cache()
