# SPDX-License-Identifier: Apache-2.0
"""Release startup file-cache pages without removing persistent cache files."""

from __future__ import annotations

import logging
import os
import re
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.utils import LazyImport
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)
_CACHE_ROOT = Path("/dev/shm/torch_tpu_cache")
resolve_compilations = LazyImport(
    "torch_tpu._internal.compile._backend", "resolve_compilations"
)


def _filesystem_type(path: Path) -> str:
    path = path.resolve()
    mounts = []
    for index, line in enumerate(Path("/proc/self/mountinfo").read_text().splitlines()):
        fields, _, filesystem = line.partition(" - ")
        mount = Path(
            re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields.split()[4])
        )
        if path.is_relative_to(mount):
            mounts.append((len(mount.parts), index, filesystem.split()[0]))
    return max(mounts)[2]


def _handle_walk_error(error: OSError) -> None:
    if not isinstance(error, FileNotFoundError):
        raise error


def _cache_files(root: Path) -> list[Path]:
    try:
        device = root.stat().st_dev
    except FileNotFoundError:
        return []
    paths = []
    for directory, subdirectories, names in os.walk(root, onerror=_handle_walk_error):
        try:
            # Overlayfs can report different devices for files within one mount.
            if Path(directory).stat().st_dev != device:
                subdirectories.clear()
                continue
            for name in names:
                path = Path(directory) / name
                try:
                    if stat.S_ISREG(path.lstat().st_mode):
                        paths.append(path)
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            continue
    return sorted(paths)


def _advise_drop_file_cache(path: Path) -> int:
    try:
        cache_file = path.open("rb")
    except FileNotFoundError:
        return 0
    with cache_file:
        size = os.fstat(cache_file.fileno()).st_size
        os.posix_fadvise(cache_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return size


def _advise_files(paths: list[Path]) -> int:
    # Keep file advice parallel while doing directory discovery once per host.
    with ThreadPoolExecutor(
        max_workers=int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    ) as pool:
        return sum(pool.map(_advise_drop_file_cache, paths))


def release_checkpoint_page_cache(model_paths: list[str]) -> None:
    if not envs.SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE.get():
        return
    started = time.monotonic()
    paths = sorted(
        {
            path
            for model_path in model_paths
            for path in Path(model_path).glob("*.safetensors")
            if path.is_file()
        }
    )
    size = _advise_files(paths)
    logger.info(
        "Checkpoint page-cache release: files=%d advised_gb=%.3f elapsed_s=%.1f",
        len(paths),
        size / 1e9,
        time.monotonic() - started,
    )


def release_compile_page_cache() -> None:
    if not envs.SGLANG_TPU_RELEASE_COMPILE_PAGE_CACHE.get():
        return
    started = time.monotonic()
    cache_name = (
        os.environ.get(
            "TORCH_TPU_TIER2_COMPILATION_CACHE",
            os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"),
        )
        or "default"
    )
    if cache_name == "disabled":
        return
    root = (_CACHE_ROOT / cache_name).resolve()
    if not root.exists():
        return
    if _filesystem_type(root) in ("tmpfs", "ramfs"):
        return
    paths = _cache_files(root)
    size = _advise_files(paths)
    logger.info(
        "Compile-cache page-cache release: files=%d advised_gb=%.3f elapsed_s=%.1f",
        len(paths),
        size / 1e9,
        time.monotonic() - started,
    )


def release_startup_page_cache() -> None:
    """Join cache writers, then release startup files once per host."""
    resolve_compilations([])
    get_parallel().world_group.barrier()
    if get_parallel().world_group.local_rank != 0:
        return
    model_paths = [get_model().model_path]
    if draft_path := get_spec().speculative_draft_model_path:
        model_paths.append(draft_path)
    release_checkpoint_page_cache(model_paths)
    release_compile_page_cache()
