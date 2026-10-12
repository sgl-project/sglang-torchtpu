# SPDX-License-Identifier: Apache-2.0
"""HookRegistry entries (``sglang.srt.plugins`` entry point). Empty by design.

Every hook is tracked debt and carries a header: Why / How (hook type and
target) / Upstream PR or issue / Remove when. A REPLACE on a method longer than
~30 lines means "write the seam instead".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (target, hook, hook_type) triples.
HOOKS: list[tuple] = []

_registered = False


def register() -> None:
    # Two entry points name this function (tpu_hooks in sglang, tpu_oot_hooks in
    # sglang-tpu); with SGLANG_PLATFORM unset both run in the same process.
    global _registered
    if _registered:
        return
    _registered = True

    if not HOOKS:
        logger.debug("sglang_tpu: no hooks registered.")
        return

    from sglang.srt.platforms import current_platform

    # Every host installs the plugin, so the entry points also run off TPU.
    if not current_platform.is_tpu():
        return

    from sglang.srt.plugins.hook_registry import HookRegistry

    for target, hook, hook_type in HOOKS:
        HookRegistry.register(target, hook, hook_type)
    logger.info("sglang_tpu: registered %d hook(s).", len(HOOKS))
