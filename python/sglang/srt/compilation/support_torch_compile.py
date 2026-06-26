"""Lightweight module-level torch.compile annotations.

The annotation is intentionally metadata-only. Platform-specific runners decide
how to install, pad, and precompile annotated module boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, TypeVar

__all__ = [
    "CompilePolicy",
    "get_torch_compile_policy",
    "support_torch_compile",
]


_COMPILE_POLICY_ATTR = "_sglang_torch_compile_policy"
T = TypeVar("T")


class CompilePolicy(ABC):
    """Base class for annotation-driven torch.compile policies."""

    backend: ClassVar[str | None] = None
    fullgraph: ClassVar[bool] = True
    dynamic: ClassVar[bool] = False
    should_precompile: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model_runner: Any | None = None,
        owner: Any | None = None,
        method_name: str = "forward",
        method: Callable[..., Any] | None = None,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.owner = owner
        self.method_name = method_name
        self.method = method
        self.name = name or type(self).__name__
        self.label = label

    @abstractmethod
    def install(self) -> bool:
        """Install runtime compile wrappers for this policy."""

    @abstractmethod
    def precompile(self) -> None:
        """Warm compile artifacts for this policy."""

    @abstractmethod
    def unpad_outputs(self, out: Any, context: Any) -> Any:
        """Adapt padded compiled outputs back to the caller-visible shape."""


def support_torch_compile(
    obj: T | None = None,
    *,
    policy: type[CompilePolicy],
) -> Callable[[T], T] | T:
    """Attach a compile policy to a module boundary.

    Preferred use is annotating the module class or instance:

    ``support_torch_compile(MyModule, policy=DecoderTpuCompilePolicy)``.

    """

    if not isinstance(policy, type) or not issubclass(policy, CompilePolicy):
        raise TypeError("support_torch_compile policy must be a CompilePolicy class")

    def decorator(target_obj: T) -> T:
        setattr(target_obj, _COMPILE_POLICY_ATTR, policy)
        return target_obj

    if obj is None:
        return decorator

    return decorator(obj)


def get_torch_compile_policy(obj: Any) -> type[CompilePolicy] | None:
    """Return the compile policy class attached to ``obj``, if any."""

    return getattr(obj, _COMPILE_POLICY_ATTR, None)
