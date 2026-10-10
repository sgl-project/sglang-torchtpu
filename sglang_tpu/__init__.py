# SPDX-License-Identifier: Apache-2.0
"""SGLang's TPU backend (torch_tpu) as an out-of-tree platform plugin.

``SGLANG_PLATFORM=tpu_oot`` selects :class:`sglang_tpu.platform.TpuOOTPlatform`;
the fork's in-tree ``tpu`` entry point still resolves to
:class:`sglang_tpu.tpu_platform.TpuSRTPlatform` (see README.md).
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["activate", "ENTRY_POINT_NAME", "PLATFORM_CLASS_PATH"]

ENTRY_POINT_NAME = "tpu_oot"
PLATFORM_CLASS_PATH = "sglang_tpu.platform:TpuOOTPlatform"


def activate():
    """Return the platform class path only when selected by name; the in-tree
    ``tpu`` entry point answers auto-discovery, and two answers are an error."""
    if os.environ.get("SGLANG_PLATFORM") != ENTRY_POINT_NAME:
        return None

    # find_spec, not import: importing torch_tpu registers PJRT and takes the
    # libtpu lockfile, and activate() runs in every spawned process.
    try:
        torch_tpu_available = importlib.util.find_spec("torch_tpu") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        logger.debug("torch_tpu activation check failed.", exc_info=True)
        return None

    if not torch_tpu_available:
        logger.warning(
            "SGLANG_PLATFORM=%s was requested but torch_tpu is not importable.",
            ENTRY_POINT_NAME,
        )
        return None

    logger.info("TPU out-of-tree platform selected (%s).", PLATFORM_CLASS_PATH)
    return PLATFORM_CLASS_PATH
