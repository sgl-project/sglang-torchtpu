# SPDX-License-Identifier: Apache-2.0
"""Give each scheduler's main thread a physical core of its own (opt-in).

Why: TP ranks step in lockstep, so the slowest rank's host paces every chip. The
     scheduler's other threads float over the rank's CPUs; one landing on the main
     thread's core or its SMT sibling slows that rank's launches for the whole run.
How: BEFORE Scheduler.run_event_loop (init and warmup done, threads started); pin
     the main thread to one CPU of the rank's set and move the process's other
     threads off that physical core. SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING gates it;
     it needs the per-rank CPU set of SGLANG_SET_CPU_AFFINITY.
Upstream PR or issue: none.
Remove when: core pins the scheduler main thread itself.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_MAIN_CORE_MAX_BUSY_FRACTION = 0.30
_WORKER_POOL_MAX_BUSY_FRACTION = 0.50
_DEFAULT_STAT_SAMPLE_SECONDS = 0.15


def _core_of(cpu: int) -> set[int]:
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        text = open(path).read().strip()
    except OSError:
        return {cpu}
    cpus: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _read_proc_stat_cpu_times(
    stat_path: str = "/proc/stat",
) -> dict[int, tuple[int, int]]:
    cpu_times: dict[int, tuple[int, int]] = {}
    try:
        with open(stat_path) as stat_file:
            for line in stat_file:
                if (
                    not line.startswith("cpu")
                    or len(line) <= 3
                    or not line[3].isdigit()
                ):
                    continue
                parts = line.split()
                cpu_id = int(parts[0][3:])
                tick_counts = [int(value) for value in parts[1:]]
                if len(tick_counts) < 4:
                    continue
                idle_ticks = tick_counts[3] + (
                    tick_counts[4] if len(tick_counts) > 4 else 0
                )
                total_ticks = sum(tick_counts)
                cpu_times[cpu_id] = (idle_ticks, total_ticks)
    except OSError:
        return {}
    return cpu_times


def _measure_cpu_busy_fraction(
    cpus: set[int],
    sample_seconds: float = _DEFAULT_STAT_SAMPLE_SECONDS,
    stat_path: str = "/proc/stat",
) -> dict[int, float]:
    if not cpus:
        return {}
    first_snapshot = _read_proc_stat_cpu_times(stat_path)
    if not first_snapshot:
        return {}
    if sample_seconds > 0:
        time.sleep(sample_seconds)
    second_snapshot = _read_proc_stat_cpu_times(stat_path)
    busy_fraction: dict[int, float] = {}
    for cpu_id in cpus:
        if cpu_id not in first_snapshot or cpu_id not in second_snapshot:
            continue
        idle_before, total_before = first_snapshot[cpu_id]
        idle_after, total_after = second_snapshot[cpu_id]
        total_delta = total_after - total_before
        idle_delta = idle_after - idle_before
        if total_delta > 0:
            busy_fraction[cpu_id] = max(0.0, min(1.0, 1.0 - (idle_delta / total_delta)))
    return busy_fraction


def bind_rank_cpu_set(original_fn, gpu_id: int):
    """Why: TPU ranks bind one chip at index 0; all pass gpu_id 0 and share one CPU set.
    How: AROUND set_gpu_proc_affinity; pass tp_rank modulo the TP ranks per host.
    Upstream PR or issue: none.
    Remove when: core keys SGLANG_SET_CPU_AFFINITY's set on the host-local rank.
    """
    from sglang.srt.runtime_context import get_parallel

    cgroup_allowed = (
        os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    )
    parallel = get_parallel()
    tp_size_per_node = parallel.tp_size // max(parallel.nnodes // parallel.pp_size, 1)
    local_rank = parallel.tp_rank % tp_size_per_node
    try:
        return original_fn(local_rank)
    except (OSError, ValueError):
        if cgroup_allowed is None or not cgroup_allowed:
            raise
        import psutil

        total_physical_cores = psutil.cpu_count(logical=False) or len(cgroup_allowed)
        total_logical_cpus = psutil.cpu_count() or total_physical_cores
        cores_per_rank = max(1, total_physical_cores // max(tp_size_per_node, 1))
        start_core = (local_rank * cores_per_rank) % total_physical_cores
        end_core = min(total_physical_cores, start_core + cores_per_rank)
        target_cpus = set(range(start_core, end_core))
        if total_logical_cpus != total_physical_cores:
            target_cpus.update(
                core + total_physical_cores for core in range(start_core, end_core)
            )
        filtered_cpus = target_cpus & cgroup_allowed
        if not filtered_cpus:
            raise
        os.sched_setaffinity(0, filtered_cpus)
        logger.warning(
            "Fallback CPU affinity for local_rank=%d (cgroup restricted): %d CPUs",
            local_rank,
            len(filtered_cpus),
        )
        return None


def pin_scheduler_main_thread(scheduler, *args, **kwargs) -> None:
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING.get() or not hasattr(
        os, "sched_getaffinity"
    ):
        return None
    allowed = os.sched_getaffinity(0)
    if len(allowed) >= (os.cpu_count() or 0):
        return None

    sibling_map = {cpu_id: _core_of(cpu_id) for cpu_id in allowed}
    sample_cpus = set(allowed)
    for siblings in sibling_map.values():
        sample_cpus.update(siblings)
    cpu_busy_map = _measure_cpu_busy_fraction(sample_cpus)

    def _core_busy(cpu_id: int) -> float:
        siblings = sibling_map.get(cpu_id, {cpu_id})
        return max(cpu_busy_map.get(sibling, 0.0) for sibling in siblings)

    clean_pool = {
        cpu_id
        for cpu_id in allowed
        if _core_busy(cpu_id) < _WORKER_POOL_MAX_BUSY_FRACTION
    }
    if len(clean_pool) < 2:
        clean_pool = {
            cpu_id
            for cpu_id in allowed
            if cpu_busy_map.get(cpu_id, 0.0) < _WORKER_POOL_MAX_BUSY_FRACTION
        }
    if len(clean_pool) < 2:
        clean_pool = set(allowed)

    candidate_cores = [
        cpu_id
        for cpu_id in sorted(clean_pool)
        if _core_busy(cpu_id) < _MAIN_CORE_MAX_BUSY_FRACTION
    ]

    main_tid = threading.get_native_id()
    if not candidate_cores:
        logger.warning(
            "Skipping single-core scheduler pinning: all %d CPUs in partition exceed %.0f%% busy",
            len(clean_pool),
            _MAIN_CORE_MAX_BUSY_FRACTION * 100.0,
        )
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), clean_pool)
            except OSError:
                pass
        return None

    main_cpu = min(
        candidate_cores,
        key=lambda cpu_id: (round(_core_busy(cpu_id), 2), cpu_id),
    )
    rest = clean_pool - sibling_map[main_cpu]
    if not rest:
        return None

    for tid in os.listdir("/proc/self/task"):
        if int(tid) != main_tid:
            try:
                os.sched_setaffinity(int(tid), rest)
            except OSError:
                pass
    os.sched_setaffinity(0, {main_cpu})
    logger.info(
        "Scheduler main thread pinned to CPU %d (core busy %.1f%%); other threads on %d CPUs",
        main_cpu,
        _core_busy(main_cpu) * 100.0,
        len(rest),
    )
    return None
