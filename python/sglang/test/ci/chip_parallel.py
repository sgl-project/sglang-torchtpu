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
from sglang.test.ci.skip_guard import command_for_test_file

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

_BATCH_RUNNER_PATH = os.path.join(os.path.dirname(__file__), "batch_runner.py")
# Files at or under this est_time share one interpreter; for them the
# torch/torch_tpu/jax import (~20s) is most of the wall time. Measured on
# the stage-a suite: 91 of 139 files finish in <=30s. Arbitrary cut.
_BATCH_MAX_EST_SECONDS = 60.0
# Arbitrary; bounds how much work one failing batch sends back for reruns.
_BATCH_MAX_FILES = 8


class _Unit:
    """What a chip worker runs in one subprocess: one file, or a batch.

    A one-file unit's verdict is final; a batch's failures are rerun alone.
    """

    def __init__(self, tests: List) -> None:
        self.tests = tests


def _plan_units(pinned: List, batch_short_files: bool) -> List[_Unit]:
    # Longest first, so a long file does not start last and run alone at the end.
    pinned = sorted(pinned, key=lambda test: -test.est_time)
    if not batch_short_files:
        return [_Unit([t]) for t in pinned]
    long_files = [t for t in pinned if t.est_time > _BATCH_MAX_EST_SECONDS]
    short_files = [t for t in pinned if t.est_time <= _BATCH_MAX_EST_SECONDS]
    units = [_Unit([t]) for t in long_files]
    for i in range(0, len(short_files), _BATCH_MAX_FILES):
        units.append(_Unit(short_files[i : i + _BATCH_MAX_FILES]))
    return units


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
    filename: str,
    chip: int,
    workdir: str,
    timeout: float,
    fail_on_all_skipped: bool = False,
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
            command_for_test_file(filename, fail_on_all_skipped),
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


def _run_batch(
    tests: List,
    chip: int,
    workdir: str,
    timeout: float,
    fail_on_all_skipped: bool = False,
) -> Tuple[List[dict], str]:
    """Several files in one interpreter on one chip.

    Returns the per-file results the runner recorded (in order, stopping at
    the first failure; a file it never reached is absent) and the output.
    """
    stem = os.path.basename(tests[0].filename)
    log_path = os.path.join(workdir, f"batch-{stem}.log")
    results_path = os.path.join(workdir, f"batch-{stem}.json")
    command = ["python3", _BATCH_RUNNER_PATH, "--results", results_path]
    if fail_on_all_skipped:
        command.append("--guard")
    command.extend(t.filename for t in tests)
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=_worker_env(chip, workdir)
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(process.pid)
            log.write(f"\n[chip {chip}] batch killed after {timeout:.0f}s\n")
    _reset_worker_caches(workdir)
    results: List[dict] = []
    if os.path.exists(results_path):
        with open(results_path) as handle:
            results = json.load(handle)
    with open(log_path, errors="ignore") as log:
        return results, log.read()


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
    fail_on_all_skipped: bool = False,
    batch_short_files: bool = False,
) -> Tuple[bool, List[str], List[Tuple[str, str]], dict]:
    """Run every `chips=1` file across `workers` chip-pinned processes.

    Returns (ok, passed, failed, elapsed_by_file). Files that are not
    `chips=1` are not run here -- the caller runs those sequentially.
    `fail_on_all_skipped` runs each file through skip_guard, as
    run_unittest_files does. `batch_short_files` runs short files several
    to an interpreter; a file that fails in a batch is rerun alone before
    it counts as failed, and a batched file's elapsed excludes the shared
    startup.
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

    def file_timeout(test) -> float:
        if timeout_per_file is not None:
            return timeout_per_file
        return derive_timeout(test.est_time)

    work: queue.Queue = queue.Queue()
    # Units still owned by some worker or waiting in the queue; a worker
    # only exits once this reaches zero, since a failed batch requeues work.
    pending = [0]
    pending_lock = threading.Lock()

    def enqueue(unit: _Unit) -> None:
        with pending_lock:
            pending[0] += 1
        work.put(unit)

    def finish_unit() -> None:
        with pending_lock:
            pending[0] -= 1

    for unit in _plan_units(pinned, batch_short_files):
        enqueue(unit)

    passed: List[str] = []
    failed: List[Tuple[str, str]] = []
    elapsed_by_file: dict = {}
    results_lock = threading.Lock()
    stop = threading.Event()

    def record(chip: int, filename: str, returncode: int, elapsed: float, output: str) -> None:
        _flush_metrics(filename, returncode == 0, elapsed)
        with results_lock:
            elapsed_by_file[filename] = elapsed
            if returncode == 0:
                passed.append(filename)
                logger.info(f"✓ chip {chip}: {filename} ({elapsed:.0f}s)")
                return
            failed.append((filename, f"exit code {returncode}"))
            # Only a failure gets its output replayed, which is what
            # keeps four concurrent workers readable.
            logger.info(
                f"\n✗ chip {chip}: {filename} "
                f"exit {returncode} after {elapsed:.0f}s\n"
                f"{'-' * 70}\n{output}{'-' * 70}\n"
            )
            if not continue_on_error:
                stop.set()

    def run_single(chip: int, workdir: str, test) -> None:
        returncode, elapsed, output = _run_one(
            test.filename, chip, workdir, file_timeout(test), fail_on_all_skipped
        )
        record(chip, test.filename, returncode, elapsed, output)

    def run_batched(chip: int, workdir: str, tests: List) -> None:
        timeout = sum(file_timeout(t) for t in tests)
        results, output = _run_batch(tests, chip, workdir, timeout, fail_on_all_skipped)
        by_file = {r["file"]: r for r in results}
        for test in tests:
            result = by_file.get(test.filename)
            if result is not None and result["returncode"] == 0:
                record(chip, test.filename, 0, result["elapsed"], output)
        # A file that failed reruns alone so the preloaded state cannot have
        # caused it; the files the batch never reached go back as one unit.
        unreached = [t for t in tests if t.filename not in by_file]
        culprits = [
            t for t in tests if by_file.get(t.filename, {}).get("returncode", 0) != 0
        ]
        if not culprits and len(unreached) == len(tests):
            # Timed out or died before recording anything: the first file
            # is the only one that ran.
            culprits, unreached = unreached[:1], unreached[1:]
        for culprit in culprits:
            # The isolated rerun's output is what gets replayed on a real
            # failure; keep the in-batch tail so a preload issue is diagnosable.
            section = output.split(f"[batch] >>> {culprit.filename}")[-1]
            tail = "\n".join(section.splitlines()[-40:])
            logger.info(
                f"~ chip {chip}: {culprit.filename} failed in a batch; rerunning alone\n"
                f"{'-' * 70}\n{tail}\n{'-' * 70}\n"
            )
            enqueue(_Unit([culprit]))
        if unreached:
            enqueue(_Unit(unreached))

    def worker(chip: int) -> None:
        workdir = os.path.join(root_dir, f"chip{chip}")
        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)
        while not stop.is_set():
            try:
                unit = work.get(timeout=1.0)
            except queue.Empty:
                with pending_lock:
                    if pending[0] == 0:
                        return
                continue
            try:
                space_launches()
                if len(unit.tests) == 1:
                    run_single(chip, workdir, unit.tests[0])
                else:
                    run_batched(chip, workdir, unit.tests)
            finally:
                finish_unit()

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
