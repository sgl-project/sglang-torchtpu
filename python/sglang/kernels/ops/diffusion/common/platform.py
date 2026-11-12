"""Platform predicates and the import-time fallback selector.

Several diffusion Triton kernels have no Triton on the live device (Ascend
NPU, Apple MPS, MUSA, CPU) and must resolve to a pure-``torch`` implementation.
That choice is made once at import time,
which used to mean a hand-rolled four-branch ``if`` block repeated in every
such module, each importing ``current_platform`` directly.

This module owns both halves of that:

- :func:`platform_key` — the one place the diffusion kernels ask what device
  they are on;
- :func:`select_impl` — the one place the Triton-vs-fallback choice is made.

Layering note: the authority on "what platform is this" is the
``multimodal_gen`` platform plugin registry, because it is the only one that
consults out-of-tree vendor plugins (NPU/MUSA).  ``kernels.spec.PlatformInfo``
cannot replace it until it grows MPS/MUSA members and plugin support (see the
``DeviceType`` TODO in ``kernels/spec.py``).  Until then the dependency is
deliberately confined to this single file and resolved lazily, so no other
kernel module imports upward.
"""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_CUDA_LIKE = frozenset({"cuda", "hip"})


def platform_key() -> str:
    """Return the live device family: ``cuda``/``hip``/``tpu``/``npu``/``mps``/``musa``/``xpu``/``cpu``.

    Deliberately *not* memoized: :func:`select_impl` calls it at module import
    time, and latching that first answer would freeze the choice before the
    platform plugin has resolved. Use :func:`is_cuda` / :func:`is_hip` on hot
    paths -- they delegate straight to the platform's own cached predicates.
    """
    from sglang.multimodal_gen.runtime.platforms import current_platform

    for name in ("cuda", "hip", "tpu", "npu", "mps", "musa", "xpu"):
        if getattr(current_platform, f"is_{name}")():
            return name
    return "cpu"


def is_cuda() -> bool:
    """Cheap enough for a per-call kernel guard.

    Delegates to ``current_platform.is_cuda``, which is ``lru_cache``d on the
    platform object -- the same call the pre-refactor guards made. Going
    through :func:`platform_key` instead would add an import plus a chain of
    ``getattr`` lookups to every fused-elementwise dispatch.
    """
    from sglang.multimodal_gen.runtime.platforms import current_platform

    return current_platform.is_cuda()


def is_hip() -> bool:
    """See :func:`is_cuda`; delegates to the platform's cached predicate."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    return current_platform.is_hip()


def is_xpu() -> bool:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    return current_platform.is_xpu()


def has_triton() -> bool:
    """True when the live device runs the Triton implementations."""
    return platform_key() in _CUDA_LIKE


class _LazyFallback:
    """A fallback named without importing its module until it is resolved."""

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name
        self.__name__ = name
        self.__qualname__ = f"{kind}_fallback.{name}"

    def resolve(self) -> Callable:
        from importlib import import_module

        return getattr(
            import_module(f"sglang.kernels.ops.diffusion.common.fallback_{self.kind}"),
            self.name,
        )

    def __call__(self, *args, **kwargs):
        return self.resolve()(*args, **kwargs)


def lazy_fallback(kind: str, name: str) -> Callable:
    """Name a fallback without importing its module.

    ``select_impl`` is handed every candidate at once, so a plain import here
    would pull in *all* fallback modules on every platform. Only the selected
    candidate imports ``common.fallback_<kind>``.
    """
    return _LazyFallback(kind, name)


def select_impl(triton_impl: F, **fallbacks: F) -> F:
    """Pick ``triton_impl`` on CUDA/HIP, else the fallback for this platform.

    Callers pass the fallbacks they actually have, keyed by platform
    (``tpu=``, ``npu=``, ``mps=``, ``musa=``, ``cpu=``); an unlisted platform keeps the
    Triton implementation, which is what the pre-existing per-module ``if``
    chains did.  Keeping the whole decision in one call means a module's
    exported name is bound exactly once, so the fallback wiring stays greppable
    and can later be replaced wholesale by a ``BaseFusedOp`` dispatch without
    touching call sites.
    """
    impl = fallbacks.get(platform_key(), triton_impl)
    # Resolve now: an import inside a torch.compile'd forward breaks the graph.
    if isinstance(impl, _LazyFallback):
        return impl.resolve()
    # debug_kernel_api(lazy_fallback(...)): wrap the resolved function instead,
    # so the wrapper does not import on every call.
    inner = getattr(impl, "__wrapped__", None)
    if isinstance(inner, _LazyFallback):
        from sglang.kernels.kernel_api_logging import debug_kernel_api

        return debug_kernel_api(inner.resolve())
    return impl
