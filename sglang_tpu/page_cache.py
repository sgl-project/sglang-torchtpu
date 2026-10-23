# SPDX-License-Identifier: Apache-2.0
"""Release startup file-cache pages and disposable tmpfs compiler files."""

from __future__ import annotations

import fcntl
import logging
import os
import re
import stat
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.utils import LazyImport
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)
_CACHE_ROOT = Path("/dev/shm/torch_tpu_cache")
_checkpoint_release: Future[None] | None = None
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
    device = root.stat().st_dev
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


def _advise_file(path: Path) -> int:
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
        return sum(pool.map(_advise_file, paths))


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


def _unlink_cache_file(path: Path) -> int:
    try:
        size = path.lstat().st_size
        path.unlink()
        return size
    except FileNotFoundError:
        return 0


def _release_tmpfs_cache(root: Path, *, native: bool = False) -> None:
    started = time.monotonic()
    paths = _cache_files(root)
    removed_bytes = 0
    if native:
        # A native compile future can resolve before its tier-2 write finishes.
        # Lock entries even if their .bin has not been published yet. Keep locks
        # and directories: the runtime creates its cache directories only once.
        locks = {
            path.with_suffix(".lock")
            for path in paths
            if path.suffix in (".bin", ".lock")
        }
        for lock in sorted(locks):
            with lock.open("a+b") as cache_lock:
                fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
                removed_bytes += _unlink_cache_file(lock.with_suffix(".bin"))
    else:
        for path in paths:
            if path.suffix != ".lock":
                removed_bytes += _unlink_cache_file(path)
    logger.info(
        "Compile-cache tmpfs release: root=%s removed_gb=%.3f elapsed_s=%.1f",
        root,
        removed_bytes / 1e9,
        time.monotonic() - started,
    )


def _release_native_cache() -> None:
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
        _release_tmpfs_cache(root, native=True)
        return
    paths = _cache_files(root)
    size = _advise_files(paths)
    logger.info(
        "Compile-cache page-cache release: files=%d advised_gb=%.3f elapsed_s=%.1f",
        len(paths),
        size / 1e9,
        time.monotonic() - started,
    )


def release_compile_artifacts() -> None:
    if not envs.SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS.get():
        return
    _release_native_cache()
    # Only serialized TorchTPU/AOT and JAX caches are disposable. Inductor can
    # also contain generated Python and shared libraries used during serving.
    roots = []
    if inductor := os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        roots.append(Path(inductor) / "aotautograd")
    if jax := os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        roots.append(Path(jax))
    for root in roots:
        if root.exists() and _filesystem_type(root) in ("tmpfs", "ramfs"):
            _release_tmpfs_cache(root)


def _model_paths() -> list[str]:
    model_paths = [get_model().model_path]
    if draft_path := get_spec().speculative_draft_model_path:
        model_paths.append(draft_path)
    return model_paths


def start_checkpoint_page_cache_release() -> None:
    """Overlap checkpoint eviction with warmup after all weight readers finish."""
    global _checkpoint_release
    # Keep collective order independent of host-local cleanup settings.
    # Other local ranks can still be reading the same checkpoint files.
    get_parallel().world_group.barrier()
    if (
        not envs.SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE.get()
        or get_model().is_startup_weight_load_overlap
    ):
        return
    if int(os.environ["LOCAL_RANK"]) != 0 or _checkpoint_release is not None:
        return
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tpu-checkpoint")
    _checkpoint_release = executor.submit(release_checkpoint_page_cache, _model_paths())
    executor.shutdown(wait=False)
    logger.info("Checkpoint page-cache release queued before graph warmup")


def release_startup_page_cache() -> None:
    """Join checkpoint eviction and cache writers before releasing compiler files."""
    resolve_compilations([])
    get_parallel().world_group.barrier()
    if int(os.environ["LOCAL_RANK"]) != 0:
        return
    if _checkpoint_release is None:
        release_checkpoint_page_cache(_model_paths())
    else:
        _join_checkpoint_page_cache_release()
        # Final cleanup can follow an earlier release before HiCache allocation.
        # Keep the completed future so checkpoint files are not scanned again.
    release_compile_artifacts()


def _join_checkpoint_page_cache_release() -> None:
    started = time.monotonic()
    _checkpoint_release.result()
    logger.info(
        "Checkpoint page-cache release join: elapsed_s=%.3f",
        time.monotonic() - started,
    )
