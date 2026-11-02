# SPDX-License-Identifier: Apache-2.0
"""Load a checkpoint once per host instead of once per rank.

A feeder process reads each shard once into a staging directory on tmpfs, and the
host's ranks memory-map the staged copies. A shard is deleted once every rank has
consumed it, so staged shards stay within a fixed window of host memory.

The feeder and the ranks coordinate only through files in the staging directory.
Ranks take staged shards in any order, so a slow shard holds back only itself.
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator

import safetensors
import torch

from sglang_tpu.envs import envs
from sglang_tpu.weight_feeder import feeder as feeder_script
from sglang_tpu.weight_feeder.feeder import (
    STALL_TIMEOUT_SECONDS,
    WINDOW_SHARE_OF_STAGING,
    Shard,
    StagingDir,
    WeightFeederError,
    WeightFeederTimeout,
)

logger = logging.getLogger(__name__)

_STAGING_ROOT = envs.SGLANG_TPU_WEIGHT_FEEDER_STAGING_DIR.get()
# How often a waiting rank checks whether a shard is ready.
_READY_SHARD_POLL_SECONDS = 0.005
# How often a waiting rank checks that the feeder and its owner are still alive.
_OWNER_CHECK_INTERVAL_SECONDS = 2.0

# Loads this rank has started, per checkpoint. Ranks load in the same order, so
# each load of a checkpoint gets its own staging directory.
_checkpoint_load_counts: dict[str, int] = {}


def _remove_orphaned_staging() -> None:
    """Delete staging left behind by engines that died."""
    for entry in os.listdir(_STAGING_ROOT):
        engine_pid = entry.split("-", 1)[0]
        if engine_pid.isdigit() and not os.path.exists(f"/proc/{engine_pid}"):
            shutil.rmtree(os.path.join(_STAGING_ROOT, entry), ignore_errors=True)


def _filesystem_type(path: str) -> str:
    """Filesystem type of a path, which need not exist yet."""
    path = os.path.realpath(path)
    best_mount, best_type = "", ""
    with open("/proc/self/mountinfo") as f:
        for line in f:
            before, _, after = line.partition(" - ")
            mount = before.split()[4]
            within = path == mount or path.startswith(mount.rstrip("/") + "/")
            if within and len(mount) >= len(best_mount):
                best_mount, best_type = mount, after.split()[0]
    return best_type


def staging_is_memory_backed() -> bool:
    """Whether staging is on tmpfs, so that staged shards stay in memory."""
    on_tmpfs = _filesystem_type(_STAGING_ROOT) == "tmpfs"
    if not on_tmpfs:
        logger.warning(
            "%s is not on tmpfs, so ranks read for themselves", _STAGING_ROOT
        )
    return on_tmpfs


def can_use_shared_staging(paths: list[str]) -> bool:
    """Whether staging is usable and can hold the largest shard.

    Uses the total size, not free space, so that the engine and ranks of a host agree.
    """
    if not staging_is_memory_backed():
        return False
    try:
        os.makedirs(_STAGING_ROOT, exist_ok=True)
        staging_bytes = shutil.disk_usage(_STAGING_ROOT).total
    except OSError:
        return False
    largest = max(os.path.getsize(path) for path in paths)
    fits = largest <= staging_bytes * WINDOW_SHARE_OF_STAGING
    return fits and os.access(_STAGING_ROOT, os.W_OK)


def _list_shards(uris: list[str]) -> list[Shard]:
    """The shards of a checkpoint with their sizes."""
    return [
        Shard(i, os.path.abspath(path), os.path.getsize(path))
        for i, path in enumerate(uris)
    ]


def _num_ranks_on_host() -> int:
    """Ranks on this host, all of which must consume a shard before it is deleted."""
    from sglang.srt.runtime_context import get_server_args

    if not torch.distributed.is_initialized():
        return 1
    world_size = torch.distributed.get_world_size()
    nnodes = get_server_args().nnodes
    if world_size % nnodes:
        raise WeightFeederError(
            f"{world_size} ranks do not divide evenly across {nnodes} hosts"
        )
    return world_size // nnodes


def _replica_id() -> int:
    """The distributed world this rank belongs to within its engine."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return 0 if parallel.attn_dp_enabled else parallel.dp_rank or 0


def _engine_replica_key(replica: int) -> str:
    """A key shared by the ranks of one distributed world.

    The port separates engines under one parent process, and the replica separates
    non-attention DP replicas.
    """
    from sglang.srt.runtime_context import get_server_args

    return f"{get_server_args().port}-{replica}"


def _open_staging_dir(
    uris: list[str], engine_pid: int, replica: int, sequence: int | None = None
) -> StagingDir:
    """Open the directory of a load. `sequence` overrides this rank's load count."""
    key = hashlib.sha256("\n".join(uris).encode()).hexdigest()[:16]
    if sequence is None:
        sequence = _checkpoint_load_counts.get(key, 0)
        _checkpoint_load_counts[key] = sequence + 1
    os.makedirs(_STAGING_ROOT, exist_ok=True)
    _remove_orphaned_staging()
    name = f"{engine_pid}-{_engine_replica_key(replica)}-{key}-{sequence}"
    staging = StagingDir(os.path.join(_STAGING_ROOT, name))
    os.makedirs(staging.path, exist_ok=True)
    return staging


def _try_acquire_owner_lock(staging: StagingDir) -> int | None:
    """Take the owner lock. Returns its fd, or None if another process holds it."""
    fd = os.open(staging.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    # A non-empty lock file means an earlier owner ended. Feeding again here
    # would rewrite shards that other ranks may have mapped.
    if os.fstat(fd).st_size > 0:
        os.close(fd)
        staging.raise_if_failed()
        raise WeightFeederError("Weight feeder owner exited")
    os.write(fd, str(os.getpid()).encode())
    return fd


def _owner_is_gone(staging: StagingDir) -> bool:
    """Whether the feeder's owner has exited, which releases its lock."""
    fd = os.open(staging.lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


class _WeightFeederProcess:
    """Owner-side handle of the feeder process.

    The feeder exits when this process dies, because the kernel then closes the
    write end of a pipe the feeder waits on.
    """

    def __init__(self, staging: StagingDir, shards: list[Shard], num_ranks: int):
        self._staging = staging
        num_read_threads = envs.SGLANG_TPU_WEIGHT_FEEDER_READ_THREADS.get()
        window_bytes = envs.SGLANG_TPU_WEIGHT_FEEDER_WINDOW_GB.get() << 30
        if num_read_threads < 1 or window_bytes < 1:
            raise ValueError(
                "SGLANG_TPU_WEIGHT_FEEDER_READ_THREADS and "
                "SGLANG_TPU_WEIGHT_FEEDER_WINDOW_GB must be positive"
            )
        config = {
            "shards": [(shard.uri, shard.size) for shard in shards],
            "num_ranks": num_ranks,
            "window_bytes": window_bytes,
            "num_read_threads": num_read_threads,
        }
        with open(staging.config_path, "w") as f:
            json.dump(config, f)
        read_fd, alive_fd = os.pipe()
        self._alive_pipe = os.fdopen(alive_fd, "wb")
        command = [
            sys.executable,
            feeder_script.__file__,
            staging.path,
            str(read_fd),
        ]
        try:
            self._process = subprocess.Popen(command, pass_fds=(read_fd,))
        except OSError:
            self._alive_pipe.close()
            raise
        finally:
            os.close(read_fd)

    def has_exited(self) -> bool:
        return self._process.poll() is not None

    def wait(self) -> int:
        returncode = self._process.wait()
        self._alive_pipe.close()
        return returncode

    def kill(self) -> None:
        self._process.kill()

    def finish(self) -> None:
        """Wait for exit, then delete staging, keeping the error of a failed load."""
        returncode = self.wait()
        if returncode == 0 and not os.path.exists(self._staging.error_path):
            shutil.rmtree(self._staging.path, ignore_errors=True)
            return
        # A killed feeder reports nothing itself.
        self._staging.abort(f"Feeder exited with code {returncode}")


def _first_ready_shard(staging: StagingDir, indexes: list[int]) -> int | None:
    return next((i for i in indexes if os.path.exists(staging.ready_path(i))), None)


def _wait_for_ready_shard(
    staging: StagingDir, indexes: list[int], feeder: _WeightFeederProcess | None
) -> int:
    """Return a staged shard once there is one, failing if the feeder or owner dies."""
    start = last_check = time.monotonic()
    while (index := _first_ready_shard(staging, indexes)) is None:
        staging.raise_if_failed()
        now = time.monotonic()
        if now - start > STALL_TIMEOUT_SECONDS:
            raise WeightFeederTimeout(f"No shard staged in {now - start:.0f} s")
        if now - last_check > _OWNER_CHECK_INTERVAL_SECONDS:
            last_check = now
            if feeder is not None and feeder.has_exited():
                staging.raise_if_failed()
                if _first_ready_shard(staging, indexes) is None:
                    raise WeightFeederError("Feeder exited with shards unstaged")
            if feeder is None and _owner_is_gone(staging):
                raise WeightFeederError("Weight feeder owner exited")
        time.sleep(_READY_SHARD_POLL_SECONDS)
    return index


def _rotate_shards_for_host(uris: list[str]) -> list[str]:
    """The shard order of this host, the same for all its ranks.

    Hosts start at different shards, so hosts that start together read different data.
    """
    uris = sorted(uris)
    digest = hashlib.sha256(f"{socket.gethostname()}{uris[0]}".encode()).digest()
    first = int.from_bytes(digest[:8], "big") % len(uris)
    return uris[first:] + uris[:first]


def _iter_staged_weights(
    staging: StagingDir, remaining: list[int], feeder: _WeightFeederProcess | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield the tensors of the `remaining` shards, removing each once it is read."""
    while remaining:
        # Take any staged shard, so that a slow shard does not stall this rank.
        index = _wait_for_ready_shard(staging, remaining, feeder)
        try:
            shard = safetensors.safe_open(
                staging.shard_path(index), framework="pt", device="cpu"
            )
        except Exception:
            # A failed feeder deletes shards. Raise its error and not the missing file.
            staging.raise_if_failed()
            raise
        with shard as f:
            names = sorted(f.keys())
            for name in names:
                tensor = f.get_tensor(name)
                # The caller holds this tensor while the rank waits for the next shard,
                # so it must not keep the consumed shard mapped.
                if name == names[-1]:
                    tensor = tensor.clone()
                yield name, tensor
        remaining.remove(index)
        staging.mark_consumed(index)


def shared_weights_iterator(
    uris: list[str],
    read_shards: Callable[[list[str]], Iterator[tuple[str, torch.Tensor]]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield checkpoint tensors, with each shard fetched once per host.

    Tensors are memory-mapped from staged shards and stay valid after a shard is
    deleted. If loading stalls, each rank reads the shards it has left through
    `read_shards`, so no rank has to restart.
    """
    uris = _rotate_shards_for_host(uris)
    staging = _open_staging_dir(uris, os.getppid(), _replica_id())
    remaining = list(range(len(uris)))
    lock_fd = None
    feeder = None
    completed = False
    timed_out = False
    try:
        staging.raise_if_failed()
        lock_fd = _try_acquire_owner_lock(staging)
        if lock_fd is not None:
            feeder = _WeightFeederProcess(
                staging, _list_shards(uris), _num_ranks_on_host()
            )
            logger.info("Feeding %d shards through %s", len(uris), staging.path)
        yield from _iter_staged_weights(staging, remaining, feeder)
        if feeder is not None:
            feeder.wait()
            staging.raise_if_failed()
        completed = True
    except WeightFeederTimeout as e:
        timed_out = True
        # Tell the other ranks, which then read their remaining shards too.
        if not os.path.exists(staging.error_path):
            staging.fail(f"{type(e).__name__}: {e}")
        logger.warning(
            "Weight feeder timed out (%s). Falling back to the core iterator for "
            "%d of %d shards",
            e,
            len(remaining),
            len(uris),
        )
    except BaseException as e:
        # A failed feeder deletes shards. Raise its error and not the missing file.
        staging.raise_if_failed()
        # Tell the other ranks now, or they wait out the stall timeout. Closing this
        # generator early raises GeneratorExit here.
        staging.fail(f"A rank stopped loading: {type(e).__name__}: {e}")
        raise
    finally:
        if feeder is not None:
            if not completed:
                feeder.kill()
            feeder.finish()
        elif lock_fd is not None:
            staging.remove_all_shards()
        if lock_fd is not None:
            os.close(lock_fd)
    if timed_out:
        yield from read_shards([uris[index] for index in remaining])


def _checkpoint_shard_paths(model_path: str) -> list[str] | None:
    """The shard paths ranks will load, or None if the engine cannot tell."""
    from sglang.srt.model_loader.weight_utils import filter_duplicate_safetensors_files

    files = glob.glob(os.path.join(model_path, "*.safetensors"))
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    shards = filter_duplicate_safetensors_files(
        list(files), model_path, os.path.abspath(index_file)
    )
    # Whether the loader adds this shard depends on the model config.
    has_unindexed_draft = any(uri.endswith("/mtp.safetensors") for uri in files)
    return None if has_unindexed_draft else shards


def _can_prefetch_weights(model_path: str) -> bool:
    """Whether the feeder is enabled and the engine can tell what ranks will load."""
    from sglang.srt.configs.load_config import LoadFormat
    from sglang.srt.runtime_context import (
        get_exec,
        get_model,
        get_parallel,
    )

    on_local_path = get_model().load_format in (
        LoadFormat.AUTO,
        LoadFormat.SAFETENSORS,
    ) and os.path.isdir(model_path)
    return (
        on_local_path
        and envs.SGLANG_TPU_WEIGHT_FEEDER_ENABLED.get()
        # Skip engine prefetch for DP: ranks locate staging by their parent PID,
        # which points to the DP controller instead of the engine.
        and get_parallel().num_dp_ranks == 1
        and get_exec().moe.ep_join_mode != "scale"
    )


def start_weight_prefetch() -> None:
    """Start the feeder from the engine, before its ranks launch.

    The engine owns this feeder and its ranks only read from it. If the engine cannot
    tell what its ranks will load, it starts nothing and ranks start their own feeder.
    """
    from sglang.srt.runtime_context import get_model, get_parallel, get_server_args

    model_path = get_model().model_path
    if not _can_prefetch_weights(model_path):
        return
    shards = _checkpoint_shard_paths(model_path)
    if not shards or not can_use_shared_staging(shards):
        return
    uris = _rotate_shards_for_host(shards)
    # New ranks count their loads from zero, however many engines came before.
    # shortcut: a relaunch during a running feeder shares its directory, add a
    # launch id if engines restart mid-load.
    staging = _open_staging_dir(uris, os.getpid(), replica=0, sequence=0)
    lock_fd = _try_acquire_owner_lock(staging)
    if lock_fd is None:
        return
    parallel = get_parallel()
    num_ranks = parallel.tp_size * parallel.pp_size // get_server_args().nnodes
    try:
        feeder = _WeightFeederProcess(staging, _list_shards(uris), num_ranks)
    except Exception:
        # Ranks find no feeder here and start their own.
        logger.exception("Could not start the weight feeder early")
        shutil.rmtree(staging.path, ignore_errors=True)
        os.close(lock_fd)
        return
    logger.info("Feeding %d shards through %s", len(uris), staging.path)

    def finish() -> None:
        feeder.finish()
        os.close(lock_fd)

    threading.Thread(target=finish, name="weight-feeder-owner", daemon=True).start()
