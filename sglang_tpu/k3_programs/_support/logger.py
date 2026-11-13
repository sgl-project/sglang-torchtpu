# SPDX-License-Identifier: Apache-2.0
"""``init_logger`` for the vendored Kimi-K3 programs, on Python logging."""

import logging


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
