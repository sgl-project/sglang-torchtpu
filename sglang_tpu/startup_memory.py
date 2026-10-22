"""Read-only startup diagnostics for the host memory used by TPU workers."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_BUDDYINFO = Path("/proc/buddyinfo")
_MIN_HEADROOM_GB = 200
_MIN_FREE_2M_GB = 60


def _free_large_blocks(text: str, page_size: int) -> dict[int, int]:
    free: dict[int, int] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"Node (\d+), zone\s+\S+\s+(.*)", line)
        if match is None:
            raise ValueError(f"Unexpected buddyinfo row: {line}")
        node = int(match[1])
        free[node] = free.get(node, 0) + sum(
            int(count) * (page_size << order)
            for order, count in enumerate(match[2].split())
            if (page_size << order) >= 2 * 1024 * 1024
        )
    return free


def log_startup_memory() -> None:
    # Missing procfs/cgroup diagnostics must not turn this log into a gate.
    try:
        limit = (_CGROUP_ROOT / "memory.max").read_text().strip()
        usage = int((_CGROUP_ROOT / "memory.current").read_text())
        if limit == "max":
            logger.info("TPU startup memory: cgroup headroom is unlimited")
        else:
            headroom = (int(limit) - usage) / 1e9
            logger.info("TPU startup memory: cgroup_headroom_gb=%.3f", headroom)
            if headroom < _MIN_HEADROOM_GB:
                logger.warning(
                    "TPU startup memory: cgroup headroom %.3f GB is below %d GB",
                    headroom,
                    _MIN_HEADROOM_GB,
                )
    except (OSError, ValueError) as exc:
        logger.warning("TPU startup memory: cgroup headroom unavailable: %s", exc)

    try:
        free = _free_large_blocks(_BUDDYINFO.read_text(), os.sysconf("SC_PAGE_SIZE"))
        if not free:
            raise ValueError("No NUMA nodes in buddyinfo")
        for node, count in sorted(free.items()):
            gb = count / 1e9
            logger.info("TPU startup memory: node=%d free_2m_gb=%.3f", node, gb)
            if gb < _MIN_FREE_2M_GB:
                logger.warning(
                    "TPU startup memory: node %d free 2 MiB+ blocks %.3f GB "
                    "are below %d GB",
                    node,
                    gb,
                    _MIN_FREE_2M_GB,
                )
    except (OSError, ValueError) as exc:
        logger.warning(
            "TPU startup memory: NUMA free 2 MiB+ blocks unavailable: %s", exc
        )
