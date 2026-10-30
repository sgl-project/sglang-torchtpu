"""DIAG (do not merge): per-rank scheduler stage timing, logged every 5 s when
SGLANG_TPU_DIAG_TIMING=1."""

import collections
import logging
import os
import time

ENABLED = os.environ.get("SGLANG_TPU_DIAG_TIMING") == "1"
logger = logging.getLogger(__name__)
_acc = collections.defaultdict(lambda: [0, 0.0, 0.0])
_last = [time.monotonic()]


class timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        if ENABLED:
            dt = time.perf_counter() - self.t
            a = _acc[self.name]
            a[0] += 1
            a[1] += dt
            a[2] = max(a[2], dt)


def maybe_log():
    if not ENABLED:
        return
    now = time.monotonic()
    if now - _last[0] < 5:
        return
    if _acc:
        logger.info(
            "DIAG timing %.1fs: %s",
            now - _last[0],
            " | ".join(
                f"{k} n={c} sum={t * 1e3:.0f} max={m * 1e3:.0f}"
                for k, (c, t, m) in sorted(_acc.items())
            ),
        )
    _acc.clear()
    _last[0] = now
