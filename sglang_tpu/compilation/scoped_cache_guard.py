# SPDX-License-Identifier: Apache-2.0
"""Native cache admission with explicitly scoped TPU compilation allowances."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import get_ident
from typing import ParamSpec, TypeVar

import torch

from sglang.utils import LazyImport
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)
# Platform discovery imports this module before the TPU worker sets its rank.
sync = LazyImport("torch_tpu._internal", "sync")
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass
class _GuardState:
    owner: int | None = None
    active_section: str | None = None
    active_thread: int | None = None
    failed: bool = False


_state = _GuardState()


def arm_cache_guard() -> None:
    """Reject native cache misses after startup compilation has finished."""
    if not envs.SGLANG_TPU_ERROR_ON_RECOMPILES.get():
        return
    if _state.owner is not None:
        raise RuntimeError("TPU cache guard already installed")
    sync.synchronize(wait=False)
    torch.tpu._set_cache_only(True)
    _state.owner = get_ident()
    logger.info(
        "[TPU cache guard] armed after warmup; cache_misses=%d",
        torch.tpu._get_cache_misses(),
    )


def allow_tpu_recompilation(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Allow native compilation inside a named function, including lazy outputs.

    The native setting is process-wide: background submissions can also compile
    while this window is open. Only the scheduler owner may control the window.
    """
    section = f"{func.__module__}.{func.__qualname__}"

    @wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _allow_tpu_recompilation(section):
            result = func(*args, **kwargs)
        return result

    return wrapped


@contextmanager
def _allow_tpu_recompilation(section: str) -> Iterator[None]:
    armed = _state.owner is not None
    if not armed and not envs.SGLANG_TPU_ERROR_ON_RECOMPILES.get():
        yield
        return
    if armed:
        if get_ident() != _state.owner:
            raise RuntimeError("Cache guard must be controlled by its owner thread")
        if _state.failed:
            raise RuntimeError("Cache guard failed; restart the worker")
    if _state.active_section is not None:
        if get_ident() != _state.active_thread:
            raise RuntimeError("Cache allowance must be controlled by its owner thread")
        try:
            yield
        except BaseException:
            if armed:
                _state.failed = True
            raise
        return

    # Keep warmup and serving graph boundaries identical, without waiting for
    # device completion. Returned tensors remain live until the exit flush.
    try:
        sync.synchronize(wait=False)
        if armed:
            before = torch.tpu._get_cache_misses()
    except BaseException:
        if armed:
            _state.failed = True
        raise

    body_error = None
    try:
        if armed:
            torch.tpu._set_cache_only(False)
        _state.active_section = section
        _state.active_thread = get_ident()
        yield
        if armed and _state.failed:
            raise RuntimeError("Cache guard failed; restart the worker")
        sync.synchronize(wait=False)
    except BaseException as error:
        body_error = error
        if armed:
            _state.failed = True
        error.add_note(f"TPU compilation allowance: {section}")
        raise
    finally:
        try:
            if armed:
                torch.tpu._set_cache_only(True)
        except BaseException as error:
            _state.failed = True
            if body_error is None:
                raise
            body_error.add_note(f"Cache-only restoration also failed: {error!r}")
        finally:
            _state.active_section = None
            _state.active_thread = None
    if armed:
        try:
            delta = torch.tpu._get_cache_misses() - before
        except BaseException:
            _state.failed = True
            raise
        if delta:
            logger.warning(
                "[TPU cache guard] allowed window=%s process_cache_misses=%d",
                section,
                delta,
            )
