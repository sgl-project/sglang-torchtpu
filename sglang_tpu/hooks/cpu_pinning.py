# SPDX-License-Identifier: Apache-2.0
"""Give each scheduler's main thread a physical core of its own (opt-in).

Why: TP ranks step in lockstep, so the slowest rank's host paces every chip. The
     scheduler's other threads float over the rank's CPUs; one landing on the main
     thread's core or its SMT sibling slows that rank's launches for the whole run.
How: BEFORE Scheduler.run_event_loop (init and warmup done, threads started); pin
     the main thread to one CPU of the rank's set and move the process's other
     threads off that physical core. SGLANG_TPU_PIN_SCHEDULER_MAIN_THREAD gates it;
     it needs the per-rank CPU set of SGLANG_SET_CPU_AFFINITY.
Upstream PR or issue: none.
Remove when: core pins the scheduler main thread itself.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


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


def pin_scheduler_main_thread(scheduler, *args, **kwargs) -> None:
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_PIN_SCHEDULER_MAIN_THREAD.get():
        return None
    allowed = os.sched_getaffinity(0)
    if len(allowed) >= (os.cpu_count() or 0):
        return None
    main_cpu = min(allowed)
    rest = allowed - _core_of(main_cpu)
    if not rest:
        return None
    main_tid = threading.get_native_id()
    for tid in os.listdir("/proc/self/task"):
        if int(tid) != main_tid:
            try:
                os.sched_setaffinity(int(tid), rest)
            except OSError:
                pass
    os.sched_setaffinity(0, {main_cpu})
    logger.info(
        "Scheduler main thread pinned to CPU %d; other threads on %d CPUs",
        main_cpu,
        len(rest),
    )
    return None
