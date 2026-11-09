"""Check a completed suite's wall time without charging queue or pod startup."""

import math
import time
from typing import Callable


def run_with_execution_budget(run_suite: Callable[[], int], minutes: float) -> int:
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("execution budget must be finite and nonnegative")
    if minutes == 0:
        return run_suite()

    started = time.monotonic()
    result = run_suite()
    elapsed = time.monotonic() - started
    print(
        f"Suite execution: {elapsed:.2f}s, budget: {minutes:g} minutes",
        flush=True,
    )
    if elapsed > minutes * 60:
        print(
            f"FAIL: suite execution budget exceeded ({elapsed / 60:.2f} minutes "
            f"against a {minutes:g}-minute budget).\n"
            "PR gating runs for every PR. Keep test execution within this budget "
            "so authors get timely feedback.\n"
            "Check whether new or changed tests add expensive coverage that "
            "belongs in nightly runs. Keep focused checks in PR gating.\n"
            "Queueing and pod startup are excluded from this budget.",
            flush=True,
        )
        return result or 1
    return result
