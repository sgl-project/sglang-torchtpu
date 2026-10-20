"""SGLang TPU platform plugin via torch_tpu (PyTorch privateuse1 backend).

Registered as a setuptools entry_point under ``sglang.srt.platforms``.
The ``activate`` function is called during platform discovery and returns
the fully-qualified platform class name when TPU hardware is available.
"""

import importlib.util
import os

from sglang_tpu._logging import get_logger

os.environ.setdefault("TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS", "1")

logger = get_logger(__name__)


def activate():
    """Return the TPU platform class path if torch_tpu is usable, else None.

    Uses importlib.util.find_spec() instead of ``import torch_tpu`` to avoid
    triggering PjRt backend registration.  ``import torch_tpu`` registers
    the PjRt plugin and reserves the libtpu lockfile; since SGLang calls
    activate() in every spawned process (scheduler, detokenizer, tokenizer),
    importing here causes lockfile collisions between processes that don't
    even need TPU access.  The real import is deferred to init_backend().
    """
    os.environ.setdefault("TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS", "1")
    try:
        torch_tpu_available = importlib.util.find_spec("torch_tpu") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        logger.debug("torch_tpu activation check failed.", exc_info=True)
        return None

    if not torch_tpu_available:
        return None

    logger.info("TPU plugin detected (torch_tpu installable).")
    return "sglang_tpu.tpu_platform:TpuSRTPlatform"
