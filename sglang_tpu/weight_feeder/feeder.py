# SPDX-License-Identifier: Apache-2.0
"""The feeder process: fetches checkpoint shards into tmpfs for the ranks of a host.

Started by path, so it imports neither sglang nor torch_tpu. It runs apart from the
ranks because read threads inside a rank would slow that rank, and all ranks wait
for the slowest. See weight_feeder/__init__.py for the protocol.
"""

import json
import mmap
import os
import queue
import shutil
import sys
import threading
import time
import traceback

import msgspec

# How often the feeder checks for consumed shards and for room in the window.
_CONSUMED_SHARD_POLL_SECONDS = 0.02
# How long a deleted shard stays open here and charged to the window before the
# feeder frees its pages. Too short, and a rank still mapping it frees them instead.
_REAP_DELAY_SECONDS = 5.0
# Ranks stop using the feeder and read their remaining shards themselves if no shard
# is consumed for this long, or if a rank waits this long for a staged shard. Must
# exceed rank startup, because an engine's feeder starts its timer before ranks launch.
STALL_TIMEOUT_SECONDS = 900.0
# Share of its filesystem that staging may fill, which caps the window and the
# largest shard. The rest stays free for other users.
WINDOW_SHARE_OF_STAGING = 0.85
# MiB per read when copying a local shard into staging, sequentially.
_READ_CHUNK_MB = 1


class WeightFeederError(RuntimeError):
    """The weight feeder failed, on this rank or on its owner."""


class WeightFeederTimeout(WeightFeederError):
    """Loading stalled. Ranks read their remaining shards themselves."""


class Shard(msgspec.Struct, frozen=True):
    """A checkpoint shard: its place in the load, its location and its bytes."""

    # Position in this host's shard order, which names its staged files, e.g. 3.
    index: int
    # Where to read it from, e.g. "/mnt/model/model-00004-of-00127.safetensors".
    uri: str
    # Size in bytes, e.g. 4983951872.
    size: int


def _unlink_if_present(path: str) -> None:
    """Unlink a file that failure cleanup may already have removed."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class StagingDir:
    """The files through which a host's feeder and ranks share one checkpoint load.

    `ready-*` marks a staged `shard-*` file, and each rank adds a `consumed-*` marker
    when done with it. `owner.lock` is held by the feeder's owner, and `error` holds
    the failure that ended the load.
    """

    def __init__(self, path: str):
        self.path = path
        self.lock_path = os.path.join(path, "owner.lock")
        self.error_path = os.path.join(path, "error")
        self.config_path = os.path.join(path, "feeder-config.json")

    def shard_path(self, index: int) -> str:
        return os.path.join(self.path, f"shard-{index:06d}.safetensors")

    def ready_path(self, index: int) -> str:
        return os.path.join(self.path, f"ready-{index:06d}")

    def consumed_marker_prefix(self, index: int) -> str:
        return f"consumed-{index:06d}-"

    def mark_consumed(self, index: int) -> None:
        marker = self.consumed_marker_prefix(index) + str(os.getpid())
        with open(os.path.join(self.path, marker), "w"):
            pass

    def count_consumed_ranks(self, entries: list[str], index: int) -> int:
        prefix = self.consumed_marker_prefix(index)
        return sum(entry.startswith(prefix) for entry in entries)

    def remove_shard(self, index: int, entries: list[str]) -> None:
        _unlink_if_present(self.shard_path(index))
        _unlink_if_present(self.ready_path(index))
        prefix = self.consumed_marker_prefix(index)
        for entry in entries:
            if entry.startswith(prefix):
                _unlink_if_present(os.path.join(self.path, entry))

    def remove_all_shards(self) -> None:
        """Free staged data, keeping the error file for ranks yet to read it."""
        for entry in os.listdir(self.path):
            if entry.startswith("shard-"):
                _unlink_if_present(os.path.join(self.path, entry))

    def fail(self, message: str) -> None:
        temporary = f"{self.error_path}.{os.getpid()}.{threading.get_ident()}"
        with open(temporary, "w") as f:
            f.write(message)
        os.replace(temporary, self.error_path)

    def abort(self, message: str) -> None:
        """Report `message` unless an error is already reported, then free the shards.

        Reporting comes first, so that ranks see this error and not a missing file.
        """
        try:
            if not os.path.exists(self.error_path):
                self.fail(message)
        except OSError:
            # Staging may be full, so free it to make room for the report.
            self.remove_all_shards()
            self.fail(message)
        self.remove_all_shards()

    def raise_if_failed(self) -> None:
        if os.path.exists(self.error_path):
            with open(self.error_path) as f:
                message = f.read()
            timed_out = message.startswith(WeightFeederTimeout.__name__)
            raise (WeightFeederTimeout if timed_out else WeightFeederError)(message)


class WeightFeeder:
    """Reads shards into staging on a pool of threads, within a memory window."""

    def __init__(
        self,
        staging: StagingDir,
        shards: list[Shard],
        num_ranks: int,
        window_bytes: int,
        num_read_threads: int,
    ):
        self._staging = staging
        self._shards = shards
        self._num_ranks = num_ranks
        self._window_bytes = window_bytes
        self._num_read_threads = num_read_threads
        self._lock = threading.Lock()
        self._next_index = 0
        self._staged: dict[int, int] = {}
        # Includes deleted shards until the reaper frees their pages.
        self._reserved_bytes = 0
        self._last_progress = time.monotonic()
        self._slowest_shard_seconds = 0.0
        self._num_fetched_shards = 0
        self._failed = threading.Event()
        self._workers = [
            threading.Thread(
                target=self._reader_loop, name=f"weight-feeder-{i}", daemon=True
            )
            for i in range(self._num_read_threads)
        ]
        self._reaper_queue: queue.SimpleQueue[tuple[int, int, float]] = (
            queue.SimpleQueue()
        )
        threading.Thread(
            target=self._reap, name="weight-feeder-reap", daemon=True
        ).start()

    def run(self) -> None:
        """Stage all shards and return once ranks consumed them or the load failed."""
        self._start_time = time.perf_counter()
        for worker in self._workers:
            worker.start()
        self._drain()
        elapsed = time.perf_counter() - self._start_time
        outcome = "failed after" if self.failed else "finished in"
        print(f"Weight feeder {outcome} {elapsed:.1f} s", file=sys.stderr, flush=True)
        # A reader stuck in a read would block the join. Exiting instead ends it and
        # frees the shard it maps, so a failed load does not wait on the mount.
        if not self.failed:
            for worker in self._workers:
                worker.join()

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def _fail(self, error: BaseException) -> None:
        # The first failure is the one to report. Report before signalling, so that
        # the feeder never exits before ranks can read the error.
        with self._lock:
            if self._failed.is_set():
                return
            traceback.print_exc()
            self._staging.abort(f"{type(error).__name__}: {error}")
            self._failed.set()

    def _reader_loop(self) -> None:
        """Claim and stage shards until none are left."""
        try:
            while (shard := self._claim_next_shard()) is not None:
                self._stage_shard(shard)
        except BaseException as e:
            self._fail(e)

    def _claim_next_shard(self) -> Shard | None:
        """Next shard for a reader thread, once the window has room for it."""
        while not self._failed.is_set():
            with self._lock:
                if self._next_index == len(self._shards):
                    return None
                shard = self._shards[self._next_index]
                has_room = (
                    self._reserved_bytes + shard.size <= self._window_bytes
                    or self._reserved_bytes == 0
                )
                if has_room:
                    self._next_index += 1
                    self._staged[shard.index] = shard.size
                    self._reserved_bytes += shard.size
                    return shard
            time.sleep(_CONSUMED_SHARD_POLL_SECONDS)
        return None

    def _drain(self) -> None:
        """Delete shards as ranks consume them, until all shards are consumed."""
        try:
            while not self._failed.is_set():
                with self._lock:
                    self._release_consumed_shards()
                    all_claimed = self._next_index == len(self._shards)
                    if all_claimed and not self._staged:
                        return
                    self._raise_if_stalled()
                time.sleep(_CONSUMED_SHARD_POLL_SECONDS)
        except BaseException as e:
            self._fail(e)

    def _log_fetch_rate(self) -> None:
        elapsed = time.perf_counter() - self._start_time
        size_gb = sum(shard.size for shard in self._shards) / 1e9
        # Readers wait for room in the window, so this rate is limited by the source
        # or the ranks, whichever is slower.
        print(
            f"Weight feeder fetched {size_gb:.1f} GB in {elapsed:.1f} s, "
            f"{size_gb / elapsed:.2f} GB/s, "
            f"slowest shard {self._slowest_shard_seconds:.1f} s",
            file=sys.stderr,
            flush=True,
        )

    def _raise_if_stalled(self) -> None:
        """Raise if a rank has failed or ranks have stopped consuming shards."""
        # Stop fetching once a rank has reported a failure.
        self._staging.raise_if_failed()
        if time.monotonic() - self._last_progress > STALL_TIMEOUT_SECONDS:
            raise WeightFeederTimeout(
                f"Ranks consumed no shard for {STALL_TIMEOUT_SECONDS:.0f} s"
            )

    def _release_consumed_shards(self) -> None:
        """Delete the shards that every rank has consumed."""
        entries = os.listdir(self._staging.path)
        for index, size in list(self._staged.items()):
            if self._staging.count_consumed_ranks(entries, index) < self._num_ranks:
                continue
            self._hold_pages_until_reaped(index, size)
            self._staging.remove_shard(index, entries)
            del self._staged[index]
            self._last_progress = time.monotonic()

    def _hold_pages_until_reaped(self, index: int, size: int) -> None:
        """Hold a shard's file open so that this process frees its pages, not a rank.

        The last process to drop a deleted tmpfs file pays to free its pages. A rank
        paying that falls behind, and every other rank waits for it through the window.
        """
        fd = os.open(self._staging.shard_path(index), os.O_RDONLY)
        self._reaper_queue.put((fd, size, time.monotonic() + _REAP_DELAY_SECONDS))

    def _reap(self) -> None:
        """Free deleted shards once their delay passes and uncharge them."""
        while True:
            fd, size, due = self._reaper_queue.get()
            time.sleep(max(0.0, due - time.monotonic()))
            os.close(fd)
            with self._lock:
                self._reserved_bytes -= size

    def _stage_shard(self, shard: Shard) -> None:
        """Reserve staging space for a shard, fill it, then mark it ready for ranks."""
        start = time.perf_counter()
        fd = os.open(
            self._staging.shard_path(shard.index), os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            # Unlike a sparse file, this reports a full tmpfs as an error and not as
            # a SIGBUS in a reader thread.
            os.posix_fallocate(fd, 0, shard.size)
            region = mmap.mmap(fd, shard.size)
        finally:
            os.close(fd)
        with region, memoryview(region) as view:
            _read_shard(shard, view)
        with open(self._staging.ready_path(shard.index), "w"):
            pass
        elapsed = time.perf_counter() - start
        with self._lock:
            self._num_fetched_shards += 1
            self._slowest_shard_seconds = max(self._slowest_shard_seconds, elapsed)
            all_fetched = self._num_fetched_shards == len(self._shards)
        if all_fetched:
            self._log_fetch_rate()


def _read_shard(shard: Shard, view: memoryview) -> None:
    """Copy a shard from a local path into its staged file."""
    fd = os.open(shard.uri, os.O_RDONLY)
    try:
        offset = 0
        while offset < shard.size:
            with view[offset : offset + (_READ_CHUNK_MB << 20)] as chunk:
                count = os.preadv(fd, [chunk], offset)
            if count == 0:
                raise WeightFeederError(f"{shard.uri} ended at {offset} bytes")
            offset += count
        # Ranks read the staged copy, so the page cache would be a second copy.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _exit_with_parent(staging: StagingDir, parent_alive_fd: int) -> None:
    """Free staging and exit when the parent dies, which closes this pipe's far end.

    Unlike PR_SET_PDEATHSIG, this has no startup race and does not depend on the parent
    thread that started this process staying alive.
    """

    def wait_for_parent() -> None:
        os.read(parent_alive_fd, 1)
        try:
            staging.abort("Weight feeder owner exited")
        finally:
            os._exit(1)

    threading.Thread(target=wait_for_parent, daemon=True).start()


def _window_for_staging(staging: StagingDir, window_bytes: int, largest: int) -> int:
    """The configured window, capped by free staging. Raises if no shard could fit."""
    usage = shutil.disk_usage(staging.path)
    staging_limit = int(usage.free * WINDOW_SHARE_OF_STAGING)
    if staging_limit < largest:
        raise WeightFeederError(
            f"{usage.free} bytes free under {staging.path} cannot hold a "
            f"{largest}-byte shard. Set SGLANG_TPU_WEIGHT_FEEDER_ENABLED=0 to read "
            "per rank."
        )
    return min(window_bytes, staging_limit)


def main() -> None:
    """Run the feeder for one load. Exits nonzero if the load fails."""
    # The staging directory of this load, e.g.
    # "/dev/shm/sglang-torchtpu-weight-feeder/4312-8000-0-9f3a1c2be07d4a16-0".
    staging = StagingDir(sys.argv[1])
    # The inherited file descriptor of the parent's liveness pipe, e.g. "7".
    _exit_with_parent(staging, int(sys.argv[2]))
    with open(staging.config_path) as f:
        config = json.load(f)
    try:
        # Drop CPU pinning inherited from a rank. The kernel limits this mask to the
        # CPUs the process may use.
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except OSError:
        pass
    try:
        shards = [Shard(i, uri, size) for i, (uri, size) in enumerate(config["shards"])]
        window_bytes = _window_for_staging(
            staging, config["window_bytes"], max(shard.size for shard in shards)
        )
        feeder = WeightFeeder(
            staging,
            shards,
            config["num_ranks"],
            window_bytes,
            config["num_read_threads"],
        )
    except Exception as e:
        staging.fail(f"{type(e).__name__}: {e}")
        raise
    feeder.run()
    if feeder.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
