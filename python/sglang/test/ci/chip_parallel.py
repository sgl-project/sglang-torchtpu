"""Run TPU test files several at a time, each pinned to one chip.

A tpu7x host has 4 chips, and a test that needs only one of them still holds
all four for its whole run today. Pinning lets four such files run at once on
the slice the job already owns, so this buys wall time without asking the
cluster for anything.

Only `chips=1` registrations are pinned; everything else keeps the whole node
to itself and runs through the usual sequential path. The two groups never
overlap, so a whole-node test never shares a host.

Pinning needs `TPU_VISIBLE_CHIPS` -- `TPU_VISIBLE_DEVICES` is not honoured by
libtpu here, and without a pin every process opens every chip and they collide
on /dev/vfio and /tmp/libtpu_lockfile. Each worker also gets private TMPDIR and
compilation-cache roots, since the shared ones would race.
"""

import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import List, Optional, Tuple

from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)

# Two TPU backend initialisations that start at the same moment collide: one
# wins and the others block ~30s inside get_c_api_client before succeeding.
# Workers running equal-length files stay in lockstep and would collide at
# every file boundary, so launches are spaced by at least this much. Measured
# at 3s: 20/20 processes clean, against 8/20 stalling unspaced.
_LAUNCH_SPACING_SECONDS = 3.0

# Env that confines a process to one chip. The bounds vars must agree with the
# pin or libtpu still tries to build a multi-chip topology.
_PIN_ENV = {
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_PROCESS_ADDRESSES": "local",
}


def default_root_dir() -> str:
    """Where the per-worker scratch lives.

    /dev/shm when it exists: torch_tpu's tier-3 cache otherwise defaults under
    $HOME, which on a CI pod is the ephemeral overlay, and filling that gets the
    pod evicted. jobset-tpu.yaml mounts /dev/shm for exactly this reason.
    SGLANG_CHIP_WORKER_ROOT overrides, for a host where neither is right.
    """
    override = os.environ.get("SGLANG_CHIP_WORKER_ROOT")
    if override:
        return override
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        return os.path.join(shm, "sglang-chip-workers")
    return os.path.join(tempfile.gettempdir(), "sglang-chip-workers")


def _worker_env(chip: int, workdir: str) -> dict:
    env = dict(os.environ)
    env.update(_PIN_ENV)
    env["TPU_VISIBLE_CHIPS"] = str(chip)
    env["TMPDIR"] = workdir
    env["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = os.path.join(workdir, "tier3")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(workdir, "inductor")
    return env


def _reset_worker_caches(workdir: str) -> None:
    """The per-file cache wipe, scoped to one worker.

    ci_utils._clear_compilation_caches() removes shared paths, which several
    workers would race on; each worker owns these instead.
    """
    for name in ("tier3", "inductor"):
        shutil.rmtree(os.path.join(workdir, name), ignore_errors=True)


def _run_one(
    filename: str, chip: int, workdir: str, timeout: float
) -> Tuple[int, float, str]:
    """One file on one chip. Returns (returncode, elapsed, captured output).

    Output is captured rather than streamed: four workers interleaved on one
    stream is unreadable, so the caller replays a file's output only if it
    failed.
    """
    log_path = os.path.join(workdir, os.path.basename(filename) + ".log")
    tic = time.perf_counter()
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            ["python3", filename, "-f"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_worker_env(chip, workdir),
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(process.pid)
            returncode = 124
            log.write(f"\n[chip {chip}] killed after {timeout:.0f}s\n")
    _reset_worker_caches(workdir)
    with open(log_path, errors="ignore") as log:
        return returncode, time.perf_counter() - tic, log.read()


def _flush_metrics(filename: str, passed: bool, elapsed: float) -> None:
    """Append this file's verdict to SGLANG_TEST_METRICS_FILE.

    Same per-file flush run_unittest_files does, for the same reason: a
    SIGKILL mid-suite still leaves the finished files recorded. Appends are
    whole lines, so concurrent workers do not interleave.
    """
    metrics_path = os.environ.get("SGLANG_TEST_METRICS_FILE")
    if not metrics_path:
        return
    try:
        with open(metrics_path, "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "file",
                        "test_file": os.path.basename(filename),
                        "status": "pass" if passed else "fail",
                        "duration": round(elapsed, 2),
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def run_files_chip_parallel(
    files: List,
    *,
    workers: int,
    root_dir: Optional[str] = None,
    timeout_per_file: Optional[float] = None,
    derive_timeout,
    continue_on_error: bool = False,
) -> Tuple[bool, List[str], List[Tuple[str, str]], dict]:
    """Run every `chips=1` file across `workers` chip-pinned processes.

    Returns (ok, passed, failed, elapsed_by_file). Files that are not
    `chips=1` are not run here -- the caller runs those sequentially.
    """
    pinned = [f for f in files if getattr(f, "chips", 0) == 1]
    if not pinned:
        return True, [], [], {}
    root_dir = root_dir or default_root_dir()

    launch_lock = threading.Lock()
    last_launch = [0.0]

    def space_launches() -> None:
        """Block until this process may start its TPU backend init."""
        with launch_lock:
            wait = _LAUNCH_SPACING_SECONDS - (time.monotonic() - last_launch[0])
            if wait > 0:
                time.sleep(wait)
            last_launch[0] = time.monotonic()

    work: queue.Queue = queue.Queue()
    for test in pinned:
        work.put(test)

    passed: List[str] = []
    failed: List[Tuple[str, str]] = []
    elapsed_by_file: dict = {}
    results_lock = threading.Lock()
    stop = threading.Event()

    def worker(chip: int) -> None:
        workdir = os.path.join(root_dir, f"chip{chip}")
        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)
        while not stop.is_set():
            try:
                test = work.get_nowait()
            except queue.Empty:
                return
            timeout = (
                timeout_per_file
                if timeout_per_file is not None
                else derive_timeout(test.est_time)
            )
            space_launches()
            returncode, elapsed, output = _run_one(
                test.filename, chip, workdir, timeout
            )
            _flush_metrics(test.filename, returncode == 0, elapsed)
            with results_lock:
                elapsed_by_file[test.filename] = elapsed
                if returncode == 0:
                    passed.append(test.filename)
                    logger.info(f"✓ chip {chip}: {test.filename} ({elapsed:.0f}s)")
                else:
                    failed.append((test.filename, f"exit code {returncode}"))
                    # Only a failure gets its output replayed, which is what
                    # keeps four concurrent workers readable.
                    logger.info(
                        f"\n✗ chip {chip}: {test.filename} "
                        f"exit {returncode} after {elapsed:.0f}s\n"
                        f"{'-' * 70}\n{output}{'-' * 70}\n"
                    )
                    if not continue_on_error:
                        stop.set()

    threads = [
        threading.Thread(target=worker, args=(chip,), daemon=True)
        for chip in range(workers)
    ]
    tic = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    logger.info(
        f"\n[chip-parallel] {len(passed)}/{len(pinned)} passed on {workers} chips "
        f"in {time.perf_counter() - tic:.0f}s\n"
    )
    return not failed, passed, failed, elapsed_by_file
