# SPDX-License-Identifier: Apache-2.0
"""Scope torch.compile caches to TPU compile inputs that FX graphs cannot see.

The AOTAutograd cache key covers the FX graph, the torch build and
``torch.compiler.config.cache_key_tag``. Pallas kernels are opaque custom ops
in that graph and libtpu compiles the executables, so the key misses kernel
bodies, env knobs read while tracing them, libtpu/XLA flags, the
jax/jaxlib/libtpu/torch_tpu stack and the chip. With a persistent
SGLANG_CACHE_DIR, a change to any of them would reload executables built
from the old code or flags. The functions below fold those inputs, and the
collective device assignment, into ``cache_key_tag``.

Not in the key: the Python bodies of sglang custom ops that torch_tpu lowers
from outside the hashed sources, such as the collectives in
distributed/parallel_state.py. After editing one, clear the cache.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# The whole sglang_tpu package is hashed into the compile cache key. Every
# module that builds a Pallas/JAX kernel or patches JAX lowering must live in
# it (enforced by test_tpu_kernel_compile_cache_key.py). The whole package
# rather than a file list, so kernel helpers are covered too; an edit anywhere
# in it costs one recompile.
# Files in sglang/srt that kernels call while tracing: tpu_moe_utils resolves
# SGLANG_TPU_INT4_MOE_MODE, which picks the INT4 MoE kernel variant.
TPU_KERNEL_CORE_FILES = ("layers/moe/tpu_moe_utils.py",)

# Packages that trace, lower or compile the kernels. torch's own build is
# already part of the AOTAutograd key.
_TPU_STACK_DISTRIBUTIONS = ("jax", "jaxlib", "libtpu", "libtpu-nightly", "torch_tpu")

# Flags torch_tpu, libtpu and XLA read when building executables: the set
# vllm-torchtpu hashes, plus the torch_tpu lowering switches in
# csrc/common/env_vars.h. Hashed verbatim, as torch_tpu and vllm-torchtpu do:
# a reordering costs one recompile, but two orders that set different final
# values can never share a key.
_COMPILER_FLAG_ENV_VARS = (
    "LIBTPU_INIT_ARGS",
    "TORCHTPU_STORAGE_AWARE_PLACEHOLDERS",
    "TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
    "TORCH_TPU_INTERNAL_SKINNY_SC_OFFLOAD",
    "TORCH_TPU_INTERNAL_SPLIT_RNG_STATE_UPDATE",
    "TORCH_TPU_INTERNAL_XLA_OPTIONS",
    "XLA_FLAGS",
)

# envs fields read in the hashed sources that change no compiled program;
# every other field read there joins the key. Flipping one of these would otherwise
# discard every cached executable (vllm-torchtpu keeps the same kind of list,
# _TPU_COMPILE_ENV_IGNORED).
_ENV_FIELDS_NOT_KEYED = frozenset(
    {
        # Compiles the same programs, in the background.
        "SGLANG_DIRECT_COMPILE_ASYNC",
        # Serving-time recompile monitor.
        "SGLANG_TPU_CHECK_RECOMPILATION",
        "SGLANG_TPU_CHECK_RECOMPILATION_SITES",
        # PD KV transfer and its tracing, on the host.
        "SGLANG_DEBUG_TPU_KV_STATS_EVERY",
        "SGLANG_RAIDEN_DYNAMIC_HOST_STAGING",
        "SGLANG_RAIDEN_LISTENER_START_READ",
        "SGLANG_RAIDEN_PIPELINE_TRANSFER",
        "SGLANG_TPU_ENABLE_DISAGGREGATION_PERFETTO_TRACING",
        "SGLANG_TPU_KV_ASYNC",
        "SGLANG_TPU_KV_CHANNELS",
        # PP chunk scheduling; the token buckets it picks are graph shapes.
        "SGLANG_TPU_PP_CHUNK_SLACK",
        "SGLANG_TPU_PP_DYNAMIC_CHUNKS",
        # Profiler scope.
        "SGLANG_TPU_PROFILE_NUM_CHIPS",
        "SGLANG_TPU_PROFILE_NUM_SPARSE_CORES",
        "SGLANG_TPU_PROFILE_PYTHON_TRACER",
        # KV transfer timeout.
        "SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S",
        # Multi-host discovery port, derived from --dist-init-addr per launch.
        "SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT",
        # Which vision shapes are precompiled; each shape is in its graph.
        "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES",
        "SGLANG_TPU_VIT_NUM_PRECOMPILE_BUCKETS",
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES",
        # Host-side radix tree implementation.
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND",
    }
)

# No leading \b: it disables re's literal-prefix scan (10x slower here).
_ENV_FIELD_PATTERN = re.compile(rb"envs\.([A-Z][A-Z0-9_]*)")
_KERNEL_INPUTS_TAG_PREFIX = "tpu-kernel-inputs-v1:"
_PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")
_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
# Google mass storage (NVMe), network (gVNIC) and bridge devices are not TPU
# chips, and skipping them keeps hosts of one generation equal. Every other
# Google device keys the cache, so a TPU class not seen yet costs at most a miss.
_NON_TPU_PCI_CLASS_PREFIXES = ("0x01", "0x02", "0x06")


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


def add_tpu_kernel_inputs_to_compile_cache_key() -> None:
    """Fold the TPU kernel inputs FX graphs cannot see into the cache key.

    Call once per process after LIBTPU_INIT_ARGS is final and before the first
    compilation. A repeated call replaces this digest and keeps other tags.
    """
    import torch.compiler.config as compiler_config

    tag = _KERNEL_INPUTS_TAG_PREFIX + tpu_kernel_inputs_digest()
    existing_tag = str(compiler_config.cache_key_tag or "")
    kept = [
        t
        for t in (existing_tag.split("|") if existing_tag else ())
        if not t.startswith(_KERNEL_INPUTS_TAG_PREFIX)
    ]
    compiler_config.cache_key_tag = "|".join([*kept, tag])
    logger.info("TPU compile cache kernel inputs: tag=%s", tag)


def tpu_kernel_inputs_digest(
    package_root: Path | None = None, srt_root: Path | None = None
) -> str:
    """Return a digest of the kernel inputs that the AOTAutograd key misses.

    Deterministic across processes and ranks for the same inputs: source paths
    are relative to the sglang_tpu package and sglang/srt roots, and every
    collection is sorted.
    """
    source_digest, env_fields = _scan_kernel_sources(
        kernel_source_roots(package_root, srt_root)
    )
    inputs = {
        "sources": source_digest,
        # Knobs such as SGLANG_TPU_FP8_MOE_FP32_SCALES change kernel bodies
        # inside the custom op, where the FX graph cannot see them.
        "env_fields": {
            name: _env_field_value(name)
            for name in env_fields
            if name not in _ENV_FIELDS_NOT_KEYED
        },
        "compiler_flags": {
            name: os.environ.get(name) for name in _COMPILER_FLAG_ENV_VARS
        },
        "distributions": {
            name: _distribution_version(name) for name in _TPU_STACK_DISTRIBUTIONS
        },
        # The chip Pallas kernels are tiled for and libtpu compiles for.
        "tpu_pci_device_ids": _tpu_pci_device_ids(),
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def kernel_source_roots(
    package_root: Path | None = None, srt_root: Path | None = None
) -> tuple[tuple[str, Path, tuple[str, ...]], ...]:
    """Return ``(label, root, relative paths)`` for each hashed source root."""
    if package_root is None:
        package_root = Path(__file__).resolve().parent
    if srt_root is None:
        import sglang

        srt_root = Path(sglang.__file__).resolve().parent / "srt"
    return (
        ("sglang_tpu", package_root, (".",)),
        ("sglang/srt", srt_root, TPU_KERNEL_CORE_FILES),
    )


def _scan_kernel_sources(
    sources: Iterable[tuple[str, Path, Iterable[str]]],
) -> tuple[str, list[str]]:
    """Hash sources by labeled relative path and contents; collect ``envs`` reads."""
    hasher = hashlib.sha256()
    env_fields: set[str] = set()

    for label, source_root, relative_paths in sources:
        source_root = source_root.resolve()
        files = _iter_source_files(source_root / path for path in relative_paths)
        if not files:
            raise RuntimeError(f"No TPU kernel cache-key sources in {source_root}")

        for path in files:
            relative_path = (
                f"{label}/{path.relative_to(source_root).as_posix()}".encode()
            )
            contents = path.read_bytes()
            hasher.update(len(relative_path).to_bytes(8, "big"))
            hasher.update(relative_path)
            hasher.update(len(contents).to_bytes(8, "big"))
            hasher.update(contents)
            env_fields.update(
                name.decode() for name in _ENV_FIELD_PATTERN.findall(contents)
            )
    return hasher.hexdigest(), sorted(env_fields)


def _iter_source_files(source_paths: Iterable[Path]) -> list[Path]:
    """Return sorted unique Python source files."""
    files: set[Path] = set()
    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"TPU kernel source path not found: {source_path}")
        paths = source_path.rglob("*") if source_path.is_dir() else (source_path,)
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                files.add(path)
    return sorted(files)


def _env_field_value(name: str) -> str | None:
    """Effective value of ``envs.<name>``, including a default changed in code."""
    from sglang.srt.environ import EnvField, envs
    from sglang_tpu.envs import envs as tpu_envs

    for registry in (envs, tpu_envs):
        field = getattr(registry, name, None)
        if isinstance(field, EnvField):
            return repr(field.get())
    return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tpu_pci_device_ids() -> list[str]:
    """PCI device IDs of the TPU chips on this host; they name the generation.

    Read from sysfs like torch_tpu's get_tpu_device_name(), but without
    importing torch_tpu, which reserves the libtpu lockfile.
    """
    try:
        devices = list(_PCI_DEVICES_DIR.iterdir())
    except OSError:
        return []
    ids = set()
    for device in devices:
        try:
            vendor = (device / "vendor").read_text().strip()
            pci_class = (device / "class").read_text().strip()
            if vendor == _GOOGLE_PCI_VENDOR_ID and not pci_class.startswith(
                _NON_TPU_PCI_CLASS_PREFIXES
            ):
                ids.add((device / "device").read_text().strip())
        except OSError:
            continue
    return sorted(ids)


def initialize_jax_compile_cache() -> None:
    """Build JAX's persistent compilation cache now instead of on first compile."""
    from jax._src import compilation_cache

    compilation_cache._initialize_cache()
