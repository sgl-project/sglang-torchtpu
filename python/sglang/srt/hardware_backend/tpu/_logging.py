"""TPU logger factory.

* Prefixes all logs with ``[TPU]`` for easy filtering.
* Unifies log level control under a single namespace.

Usage:
    from sglang.srt.hardware_backend.tpu._logging import get_logger
    logger = get_logger(__name__)
    logger.info("hello")     # → "[TPU] hello"
"""

from __future__ import annotations

import logging

# All tpu/ loggers live under this prefix so a single setLevel covers the
# whole backend.
_TPU_LOGGER_ROOT = __name__.rsplit(".", 1)[0]

# Grep-friendly prefix on every emitted message.
_TPU_LOG_MSG_PREFIX = "[TPU]"

_root_configured = False


class TpuPrefixPrepender(logging.Filter):
    """Prepend ``[TPU]`` to every record's message exactly once."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg.startswith(_TPU_LOG_MSG_PREFIX):
            return True
        # Bake any pending %-args into the message, then prefix and clear
        # args so downstream handlers don't try to re-format.
        record.msg = f"{_TPU_LOG_MSG_PREFIX} {msg}"
        record.args = ()
        return True


def _configure_root_once() -> None:
    """Set the TPU logger root to INFO by default (idempotent).

    Preserves existing log levels if already configured upstream.
    """
    global _root_configured
    if _root_configured:
        return
    _root_configured = True
    tpu_root_logger = logging.getLogger(_TPU_LOGGER_ROOT)
    if tpu_root_logger.level == logging.NOTSET:
        tpu_root_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a tpu-namespaced logger with the ``[TPU]`` prefix.

    Idempotent. Accepts either a fully-qualified module name (the common
    ``__name__`` case) or a short suffix. In both cases the logger is
    placed under the root TPU namespace so one parent-level setLevel
    covers the whole backend.
    """
    _configure_root_once()

    if not name.startswith(_TPU_LOGGER_ROOT):
        name = f"{_TPU_LOGGER_ROOT}.{name}"

    logger = logging.getLogger(name)
    if not any(isinstance(f, TpuPrefixPrepender) for f in logger.filters):
        logger.addFilter(TpuPrefixPrepender())
    return logger
