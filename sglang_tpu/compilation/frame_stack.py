"""Reduce warmup tracing overhead from CPython 3.12 frame-stack allocation.

Frequent calls at a stack-chunk boundary repeatedly allocate and free memory
(python/cpython#142183). Reserving a larger chunk gives tracing room without
depending on the caller's stack depth.
"""

import sys
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

# A 512 KiB frame gets a 1 MiB chunk, leaving room for tracing.
# Nested reservations exceed that remainder and get their own chunk.
_RESERVED_SLOTS = 1 << 16


def call_with_reserved_frame_stack(
    fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Call ``fn`` with extra frame-stack space on CPython 3.12."""
    return fn(*args, **kwargs)


# CPython sizes frames from co_stacksize; limit the workaround to 3.12.
if sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12):
    call_with_reserved_frame_stack.__code__ = (
        call_with_reserved_frame_stack.__code__.replace(co_stacksize=_RESERVED_SLOTS)
    )
