# SPDX-License-Identifier: Apache-2.0
"""Invalidate torch.compile caches when TPU Pallas sources change.

torch.compile cannot detect changes to Pallas kernel implementations because
they are hidden in torch.library custom ops. During TPU backend initialization,
we hash all Pallas source code in sglang-torchtpu and include the digest in the
compile cache key.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def get_pallas_kernel_sources() -> tuple[str, ...]:
    """Return paths to Pallas source files and directories in sglang-torchtpu.

    PLEASE add every new Pallas kernel to this list. Missing sources can cause
    torch.compile to reuse stale artifacts and produce confusing behavior.
    """
    return (
        "layers/attention/pallas",
        "layers/moe/pallas",
        "hardware_backend/tpu/attention_pallas.py",
        "hardware_backend/tpu/attention_pallas_rpa_base.py",
        "hardware_backend/tpu/attention_pallas_rpa_batched.py",
        "hardware_backend/tpu/attention_pallas_rpa_v3.py",
        "hardware_backend/tpu/top_k.py",
        "hardware_backend/tpu/kernels",
        "hardware_backend/tpu/pallas_export.py",
        "hardware_backend/tpu/pallas_jax_op.py",
        "hardware_backend/tpu/quantized_matmul.py",
        "hardware_backend/tpu/topology.py",
    )


def add_pallas_kernel_sources_to_compile_cache_key() -> None:
    """Add Pallas kernel sources to torch.compile cache keys.

    Pallas custom ops are opaque in FX graphs, so source changes alone do not
    invalidate cached executables. Existing user tags are preserved.
    """
    import torch.compiler.config as compiler_config

    srt_root = Path(__file__).resolve().parents[2]
    source_digest = _hash_source_paths(srt_root, get_pallas_kernel_sources())
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
