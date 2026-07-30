"""TPU-specific test support for this fork.

Everything here is fork-local. It exists as new files so that syncing
sgl-project/sglang (bot-sync-from-upstream merges upstream main every 2 days) never
has to reconcile it -- only the small additive hooks in
``sglang/test/accuracy_test_runner.py`` and ``sglang/test/run_combined_tests.py``
live in upstream-tracked code.
"""

from sglang.test.tpu.eval_support import (
    ExternalServerLauncher,
    RecipeLauncher,
    TpuModelLaunchSettings,
    lm_eval_runner,
    wait_for_health,
)

__all__ = [
    "ExternalServerLauncher",
    "RecipeLauncher",
    "TpuModelLaunchSettings",
    "lm_eval_runner",
    "wait_for_health",
]
