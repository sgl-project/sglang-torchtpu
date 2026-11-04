"""How a platform compiles free-standing tensor helpers (not model forwards).

``maybe_compile`` replaces ``@torch.compile(dynamic=..., disable=<platform>)`` on
module-level helpers: the platform's ``HelperCompilePolicy`` decides, at
decoration time, whether and how the helper is compiled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import torch

__all__ = [
    "DEFAULT_HELPER_COMPILE_POLICY",
    "HelperCompilePolicy",
    "maybe_compile",
]

F = TypeVar("F", bound=Callable[..., Any])


class HelperCompilePolicy:
    """Compiles standalone tensor helpers; the base reproduces in-tree behaviour.

    Platforms override this through ``SRTPlatform.get_helper_compile_policy()``.
    """

    def compile_helper(
        self,
        fn: F,
        *,
        dynamic: bool,
        backend: str | Callable | None,
        options: Mapping[str, Any],
    ) -> F:
        """Compile a shape-polymorphic helper, as ``@torch.compile(...)`` does."""
        kwargs = dict(options)
        if backend is not None:
            kwargs["backend"] = backend
        return torch.compile(fn, dynamic=dynamic, **kwargs)


DEFAULT_HELPER_COMPILE_POLICY = HelperCompilePolicy()


def maybe_compile(
    fn: F | None = None,
    *,
    dynamic: bool = True,
    backend: str | Callable | None = None,
    disable: bool = False,
    **options: Any,
) -> F | Callable[[F], F]:
    """``torch.compile`` a helper unless ``disable`` is set or the platform declines.

    Works as ``@maybe_compile(dynamic=True)`` or ``maybe_compile(fn, dynamic=True)``.
    """

    def decorator(target: F) -> F:
        if disable:
            return target
        from sglang.srt.platforms import current_platform

        return current_platform.get_helper_compile_policy().compile_helper(
            target, dynamic=dynamic, backend=backend, options=options
        )

    return decorator if fn is None else decorator(fn)
