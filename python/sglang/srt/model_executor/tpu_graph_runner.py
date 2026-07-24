"""TPU "graph runner": orchestrates annotated torch.compile boundaries.

Unlike CudaGraphRunner / NPUGraphRunner there is no graph capture or
replay here — PJRT (libtpu) caches HLO compilations internally, keyed
by tensor shape, so we get "compile once, dispatch by shape" behavior
without an explicit capture step. This class exists so the
torch.compile orchestration hangs off the same factory hook
(``current_platform.get_graph_runner_cls()``) as the in-tree runners,
matching the pattern used by ``CPUGraphRunner`` which is also just a
torch.compile wrapper.

Wired up by ``ModelRunner.init_device_graphs`` when both:

* ``server_args.enable_torch_compile`` is True (checked at the call
  site, mirroring the CPU path), and
* the model does not set ``_can_torch_compile = False`` (the same
  per-model opt-out the CUDA path honors in ``kernel_warmup``).

The runner discovers ``@support_torch_compile`` metadata and delegates the
boundary-specific install / warmup work to the annotated object's compile
policy or driver. It should not know decoder- or ViT-specific padding rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.compilation.support_torch_compile import (
    get_torch_compile_policy,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    SpecDecodeCompilePolicyMixin,
)
from sglang.srt.platforms import current_platform

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


def _spec_enabled(model_runner: ModelRunner) -> bool:
    """Whether speculative decoding is active for this runner."""
    return not model_runner.spec_algorithm.is_none()


def _augment_spec_policy(policy_cls: type, model_runner: ModelRunner) -> type:
    """Compose spec-decode compile behavior over ``policy_cls`` when spec decoding
    is active.
    """
    if not _spec_enabled(model_runner):
        return policy_cls
    if issubclass(policy_cls, SpecDecodeCompilePolicyMixin):
        return policy_cls
    return type(
        f"Spec{policy_cls.__name__}", (SpecDecodeCompilePolicyMixin, policy_cls), {}
    )


# This module lives outside the tpu/ tree, so __name__ does not start with
# the TPU logger root. Pass a short suffix so the logger lands cleanly under
# that root (one setLevel covers the whole backend) instead of producing a
# doubled "...tpu.sglang.srt.model_executor.tpu_graph_runner" name.
logger = get_logger("tpu_graph_runner")


class TpuGraphRunner:
    """Orchestrate ``torch.compile(backend='tpu')`` annotation policies."""

    def __init__(self, model_runner: ModelRunner) -> None:
        assert (
            model_runner.server_args.enable_torch_compile
        ), "TpuGraphRunner should only be initialized when enable_torch_compile=True."

        # Side-effectful import: registers the "tpu" dynamo backend so that
        # torch.compile(backend="tpu", ...) resolves. apply_tpu_patches does
        # this in init_backend; redo it defensively in case the runner is
        # constructed on a path that skipped that hook.
        from sglang.srt.hardware_backend.tpu.patches import (
            _register_tpu_compile_backend,
        )

        self.model_runner = model_runner
        self.model = model_runner.model

        # Per-model opt-out, matching the CUDA path in
        # ModelRunner.kernel_warmup: some Transformers-backend models
        # cannot be safely compiled (e.g. dynamic RoPE scaling).
        if not getattr(self.model, "_can_torch_compile", True):
            logger.info(
                "Model %s sets _can_torch_compile=False; skipping "
                "torch.compile install on TPU.",
                type(self.model).__name__,
            )
            # Reflect the skip to downstream code that branches on this
            # flag (kernel_warmup pre-compile loops on the CUDA side;
            # any future TPU equivalent should follow the same
            # convention).
            model_runner.server_args.override(
                "tpu_graph_runner", enable_torch_compile=False
            )
            return

        # Raise dynamo's recompile ceiling. torch.compile builds a fresh graph
        # per distinct input shape, and the TPU backbone is hit with many: one
        # per configured prefill token bucket plus the decode batch-size
        # buckets. dynamo's default cache_size_limit=8 is far too low — under
        # fullgraph=True it trips FailOnRecompileLimitHit (killing the
        # scheduler) as soon as warmup or a multi-shape benchmark walks past
        # ~8 shapes. Match the CUDA / CPU / piecewise graph runners, which all
        # set 1024 (see cuda_graph_runner.set_torch_compile_config).
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024
        # Fail loudly on compile-limit hits to expose latency cliffs early.
        # NOTE: fail_on_recompile_limit_hit and suppress_errors are mutually
        # exclusive (torch._dynamo asserts). flashinfer_backend sets
        # suppress_errors=True at import when SGLANG_ENABLE_TORCH_COMPILE is set,
        # so only enable the fail-loud flag when suppress_errors is off. The
        # recompile ceiling is already raised to 1024 above, so the limit is not
        # hit in practice.
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit") and not getattr(
            torch._dynamo.config, "suppress_errors", False
        ):
            torch._dynamo.config.fail_on_recompile_limit_hit = True

        _register_tpu_compile_backend()

        # TODO: Move policy discovery and installation into a reusable compile
        # runner when a second platform adopts support_torch_compile metadata.
        modules = (
            self.model.modules()
            if callable(getattr(self.model, "modules", None))
            else (self.model,)
        )
        installed_any = False
        installed_policies: list[Any] = []
        for module in modules:
            policy_cls = get_torch_compile_policy(module)
            if policy_cls is None:
                continue
            if policy_cls.platform not in (None, current_platform.device_type):
                continue

            method_name = "forward"
            method = getattr(module, method_name, None)
            if method is None:
                # A compile policy was explicitly attached to this module, so a
                # missing target method is a misconfiguration -- fail loudly
                # rather than silently skip and run the model uncompiled.
                raise RuntimeError(
                    f"TPU compile annotation on {type(module).__name__} targets "
                    f"method {method_name!r}, but the module has no such method."
                )

            # Compose the default spec behavior over the model's own policy when
            # spec decoding is active and the policy doesn't already implement spec.
            if _spec_enabled(model_runner) and not issubclass(
                policy_cls, SpecDecodeCompilePolicyMixin
            ):
                logger.warning(
                    "Speculative decoding is active and %s ships its own compile "
                    "policy %s; composing the default SpecDecodeCompilePolicyMixin over "
                    "it.",
                    type(module).__name__,
                    policy_cls.__name__,
                )
            policy_cls = _augment_spec_policy(policy_cls, model_runner)

            policy = policy_cls(
                model_runner=model_runner,
                owner=module,
                method_name=method_name,
                method=method,
                label=type(module).__name__,
            )
            installed = policy.install()
            if installed:
                installed_any = True
                installed_policies.append(policy)
                policy.precompile()

        if not installed_any:
            fallback_cls = _augment_spec_policy(DecoderTpuCompilePolicy, model_runner)
            logger.info(
                "enable_torch_compile=True but %s declares no TPU compile "
                "policies; falling back to %s on forward.",
                type(self.model).__name__,
                fallback_cls.__name__,
            )
            fallback_policy = fallback_cls(
                model_runner=model_runner,
                owner=self.model,
                method_name="forward",
                method=getattr(self.model, "forward", None),
                label=type(self.model).__name__,
            )
            if fallback_policy.install():
                installed_any = True
                installed_policies.append(fallback_policy)
                fallback_policy.precompile()

        if not installed_any:
            return

        # Wire serving-time recompilation detection into installed shims
        # after warmup establishes the expected compile baseline.
        if not envs.SGLANG_TPU_CHECK_RECOMPILATION.get():
            return

        from sglang.srt.hardware_backend.tpu.compilation.recompilation_checker import (
            RecompilationChecker,
        )

        checker = RecompilationChecker()
        checker.snapshot_baseline()
        model_name = type(self.model).__name__
        is_draft = getattr(model_runner, "is_draft_worker", False)
        label = f"{'draft ' if is_draft else ''}{model_name}"
        for policy in installed_policies:
            set_checker = getattr(policy, "set_recompilation_checker", None)
            if callable(set_checker):
                set_checker(checker, model_label=label)

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        """Always False — no captured graph to replay.

        PJRT/libtpu caches HLO compilations internally and dispatches
        on tensor shape, so the compiled forward is reached through the
        normal ``self.model.forward(...)`` call path, not through a
        replay step. This method exists because ModelRunner.forward
        gates the replay branch on
        ``self.<decode|prefill>_cuda_graph_runner.can_run_graph(...)`` —
        returning False keeps that branch off on TPU.
        """
        return False
