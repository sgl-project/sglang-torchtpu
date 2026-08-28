"""Lightweight module-level torch.compile annotations.

The annotation is intentionally metadata-only. Platform-specific runners decide
how to install, pad, and precompile annotated module boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpecInput

__all__ = [
    "CompilePolicy",
    "DecoderCompilePolicy",
    "NoopCompilePolicy",
    "get_torch_compile_policy",
    "support_torch_compile",
]


_COMPILE_POLICY_ATTR = "_sglang_torch_compile_policy"
T = TypeVar("T")


class CompilePolicy(ABC):
    """Base class for annotation-driven torch.compile policies."""

    # ``platform`` selects the runner that may install this policy. ``backend``
    # is only an optional torch.compile backend override; keeping these separate
    # avoids treating a backend registration name as hardware identity.
    platform: ClassVar[str | None] = None
    backend: ClassVar[str | None] = None
    fullgraph: ClassVar[bool] = True
    dynamic: ClassVar[bool] = False

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


class DecoderCompilePolicy(CompilePolicy):
    """Compile policy for decoder boundaries driven by a ForwardBatch."""

    b_bucketer: Any | None = None

    def precompile_draft(
        self, draft_attn_backend: Any, draft_extend_attn_backend: Any
    ) -> None:
        """Run warmup after speculative-draft attention backends are available."""

    @abstractmethod
    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Prepare a normal or target forward before attention planning."""

    # TODO: Move output adaptation into a platform-neutral compile runner once
    # another backend adopts annotation-driven policy installation.
    @abstractmethod
    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        """Finalize outputs and batch state after a normal or target forward."""

    def get_dp_padded_token_counts(
        self,
        forward_batch: ForwardBatch,
        model_runner: Any,
        global_num_tokens: list[int],
        global_num_tokens_for_logprob: list[int],
        *,
        attn_tp_size: int,
        cp_align_size: int,
    ) -> tuple[list[int], list[int]]:
        """Return DP token counts after platform-neutral TP/CP alignment."""

        def align(value: int, size: int) -> int:
            return (value + size - 1) // size * size if size > 1 else value

        def align_for_collectives(value: int) -> int:
            return align(align(value, attn_tp_size), cp_align_size)

        return (
            [align_for_collectives(value) for value in global_num_tokens],
            list(global_num_tokens_for_logprob),
        )

    def materialize_dp_idle_extend(
        self, forward_batch: ForwardBatch, num_tokens: int
    ) -> bool:
        """Materialize an idle DP rank for a shared extend shape, if supported."""
        return False

    def prepare_spec_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Prepare a speculative forward before attention planning."""

    def normalize_spec_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Normalize a speculative phase before phase-specific bookkeeping."""

    def prepare_spec_draft_input(
        self, forward_batch: ForwardBatch, draft_input: SpecInput
    ) -> None:
        """Prepare draft inputs consumed before the first model forward."""

    def finalize_spec_forward_batch(
        self, outputs: Any, forward_batch: ForwardBatch
    ) -> Any:
        """Finalize outputs from a speculative forward."""
        return outputs

    def should_defer_verify_output_unpadding(
        self, batch: Any, verify_input: Any
    ) -> bool:
        """Return whether this policy keeps target verify outputs padded."""
        return False

    def prepare_spec_verify(
        self, batch: Any, verify_input: Any, next_token_logits: Any
    ) -> Any:
        """Prepare inputs for speculative verification."""
        return None

    def finalize_spec_verify(
        self,
        padding: Any,
        verify_result: tuple[Any, Any, Any],
        bookkeeping_result: tuple[Any, Any],
    ) -> tuple[tuple[Any, Any, Any], tuple[Any, Any]]:
        """Restore inputs and trim verification and bookkeeping outputs."""
        return verify_result, bookkeeping_result


class NoopCompilePolicy(DecoderCompilePolicy):
    """Default noop policy carried until a decoder policy installs itself."""

    def install(self) -> bool:
        return False

    def precompile(self) -> None:
        pass

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        pass

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        return outputs


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
