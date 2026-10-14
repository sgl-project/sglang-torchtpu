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

import time
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.compilation.support_torch_compile import (
    CompilePolicy,
    DecoderCompilePolicy,
    get_torch_compile_policy,
)
from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.core_compile_annotations import (
    annotate_core_compile_boundaries,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang_tpu.compilation.spec_decode_compile_policy import (
    BaseSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compilation.spec_decode_compile_policy_eagle3 import (
    Eagle3SpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compile_cache import (
    add_tpu_rank_assignment_to_compile_cache_key,
)
from sglang_tpu.experimental import is_experimental_spec_worker
from sglang_tpu.experimental.compilation.block_compile_policy import (
    TpuBlockSpecDecodeCompilePolicyMixin,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _spec_enabled(model_runner: ModelRunner) -> bool:
    """Whether this runner uses speculative decoding on TPU."""
    algo = model_runner.spec_algorithm
    return (
        (algo.is_eagle() and not algo.is_frozen_kv_mtp())
        or algo.is_dflash()
        or is_experimental_spec_worker(algo)
    )


def _resolve_spec_compile_policy_mixin(
    spec_algorithm: SpeculativeAlgorithm,
) -> type[BaseSpecDecodeCompilePolicyMixin]:
    """Resolve speculative compile policy mixin class based on algorithm."""
    # Checked first: the experimental worker also serves DFLASH.
    if is_experimental_spec_worker(spec_algorithm):
        return TpuBlockSpecDecodeCompilePolicyMixin
    if spec_algorithm.is_dflash():
        return DFlashSpecDecodeCompilePolicyMixin
    return Eagle3SpecDecodeCompilePolicyMixin


def _augment_spec_policy(policy_cls: type, model_runner: ModelRunner) -> type:
    """Compose speculative decoding compile behavior over a decoder policy."""
    if not _spec_enabled(model_runner) or not issubclass(
        policy_cls, DecoderCompilePolicy
    ):
        return policy_cls
    if issubclass(policy_cls, BaseSpecDecodeCompilePolicyMixin):
        return policy_cls
    if not issubclass(policy_cls, DecoderTpuCompilePolicy):
        return policy_cls
    spec_mixin_cls = _resolve_spec_compile_policy_mixin(model_runner.spec_algorithm)
    return type(f"Spec{policy_cls.__name__}", (spec_mixin_cls, policy_cls), {})


# Short suffix, not __name__: keeps the logger name the module had before it
# moved under graph_runner/.
logger = get_logger("tpu_graph_runner")


def _resolve_fused_op_dispatch(model: Any) -> None:
    """Eagerly resolve ``BaseFusedOp._forward_method`` on model submodules and
    kernel-registry singletons."""
    import importlib

    from sglang.kernels.fused_op import BaseFusedOp, get_fused_op_backend
    from sglang.kernels.registry import registry

    # TODO: drop if upstream resolves the forced-backend sentinel at import
    # or construction time instead of on first forward.
    get_fused_op_backend()

    def _resolve(op: BaseFusedOp) -> None:
        if getattr(op, "_forward_method", object()) is None:
            op._forward_method = op._resolve_forward_method()

    modules = model.modules() if callable(getattr(model, "modules", None)) else (model,)
    for module in modules:
        if isinstance(module, BaseFusedOp):
            _resolve(module)

    seen = set()
    for spec in registry.all_specs():
        mod_name, _, attr = spec.target.partition(":")
        inst_name = attr.split(".")[0]
        if not inst_name or (mod_name, inst_name) in seen:
            continue
        seen.add((mod_name, inst_name))
        try:
            obj = getattr(importlib.import_module(mod_name), inst_name, None)
        except Exception:
            continue  # spec modules are import-metadata; skip unloadable ones
        if isinstance(obj, BaseFusedOp):
            _resolve(obj)


class TpuGraphRunner:
    """Orchestrate ``torch.compile(backend='tpu')`` annotation policies."""

    def __init__(self, model_runner: ModelRunner) -> None:
        assert model_runner.server_args.enable_torch_compile, (
            "TpuGraphRunner should only be initialized when enable_torch_compile=True."
        )

        # WORLD is initialized; set the tag before any policy creates a compile
        # wrapper, since cache lookup can happen before its first warmup call.
        add_tpu_rank_assignment_to_compile_cache_key()

        # Side-effectful import: registers the "tpu" dynamo backend so that
        # torch.compile(backend="tpu", ...) resolves. apply_tpu_patches does
        # this in init_backend; redo it defensively in case the runner is
        # constructed on a path that skipped that hook.
        from sglang_tpu.patches import (
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
            # Reflect the skip to downstream readers; CaptureFlags is the
            # designated post-publish writer for this flag (as base_runner).
            from sglang.srt.runtime_context import get_flags

            get_flags().capture.enable_torch_compile = False
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

        # TODO: remove — TPU does not support CP decode-attn-TP; this only
        # keeps the singleton's first-call init out of the dynamo trace.
        from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx

        get_cp_decode_attn_tp_ctx()

        # TODO: drop if upstream resolves BaseFusedOp dispatch at construction
        # instead of on first forward (keeps it out of the dynamo trace).
        _resolve_fused_op_dispatch(self.model)

        # TODO: Move policy discovery and installation into a reusable compile
        # runner when a second platform adopts support_torch_compile metadata.
        # Core classes whose TPU policy the plugin declares (core cannot import
        # sglang_tpu) get it here, so the discovery below sees them.
        annotate_core_compile_boundaries()
        modules = (
            self.model.named_modules()
            if callable(getattr(self.model, "named_modules", None))
            else (("", self.model),)
        )
        if (
            model_runner.is_draft_worker
            and model_runner.spec_algorithm.is_eagle()
            and not model_runner.spec_algorithm.is_eagle3()
        ):
            # Native MTP wraps an annotated target backbone with its projection
            # and logits processor. The speculative policy must own the complete
            # draft, then compile its backbone and logits regions itself.
            modules = (("", self.model),)
        warmup_start = time.perf_counter()
        installed_any = False
        installed_policies: list[CompilePolicy] = []
        compile_timing_recorder = getattr(model_runner, "compile_timing", None)
        pending_warmups = []
        role = "draft" if model_runner.is_draft_worker else "target"
        for module_name, module in modules:
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

            # Compose speculative decoding behavior over the model's own decoder policy.
            if (
                _spec_enabled(model_runner)
                and issubclass(policy_cls, DecoderCompilePolicy)
                and not issubclass(policy_cls, BaseSpecDecodeCompilePolicyMixin)
            ):
                logger.warning(
                    "Speculative decoding is active and %s ships its own compile "
                    "policy %s; composing the default speculative compile policy over "
                    "it.",
                    type(module).__name__,
                    policy_cls.__name__,
                )
            policy_cls = _augment_spec_policy(policy_cls, model_runner)

            policy_kwargs = dict(
                model_runner=model_runner,
                owner=module,
                method_name=method_name,
                method=method,
                name=f"{role}/{module_name + '.' if module_name else ''}{method_name}",
                label=type(module).__name__,
                compile_timing_recorder=compile_timing_recorder,
            )
            if issubclass(policy_cls, DecoderTpuCompilePolicy):
                policy_kwargs["tracer"] = model_runner.server_args.tpu_compile_tracer
            policy = policy_cls(**policy_kwargs)
            installed = policy.install()
            if installed:
                installed_any = True
                installed_policies.append(policy)
                pending_warmups.extend(policy.precompile() or ())

        if installed_any and not any(
            isinstance(policy, DecoderCompilePolicy) for policy in installed_policies
        ):
            # A vision-only annotation skips the whole-model fallback below, so
            # the decoder would run eager with nothing in the log.
            logger.warning(
                "enable_torch_compile=True but none of the compile policies "
                "installed on %s covers its decoder (%s); decode steps run eager. "
                "Annotate the language model with support_torch_compile(...).",
                type(self.model).__name__,
                ", ".join(type(p).__name__ for p in installed_policies),
            )

        if not installed_any:
            server_args = model_runner.server_args
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
                name=f"{role}/forward",
                label=type(self.model).__name__,
                tracer=server_args.tpu_compile_tracer,
                compile_timing_recorder=compile_timing_recorder,
            )
            if fallback_policy.install():
                installed_any = True
                installed_policies.append(fallback_policy)
                pending_warmups.extend(fallback_policy.precompile() or ())

        if not installed_any:
            return

        warmup_submitted = time.perf_counter()
        from sglang_tpu.compilation.warmup_queue import (
            current_warmup_queue,
        )

        queue = current_warmup_queue()
        if queue is not None:
            queue.calls.extend(pending_warmups)
            queue.completions.append(lambda: self._finish_warmup(installed_policies))
            logger.info("TpuGraphRunner queued %d global warmups", len(pending_warmups))
            return
        if pending_warmups:
            logger.info("[direct_compile] waiting for %d warmups", len(pending_warmups))
            for warmup in pending_warmups:
                warmup.wait()
        warmup_finished = time.perf_counter()
        logger.info(
            "TpuGraphRunner warmup grid: pending=%d submit=%.3fs wait=%.3fs total=%.3fs",
            len(pending_warmups),
            warmup_submitted - warmup_start,
            warmup_finished - warmup_submitted,
            warmup_finished - warmup_start,
        )

        self._finish_warmup(installed_policies)

    def _finish_warmup(self, installed_policies: list[CompilePolicy]) -> None:
        # Baseline after model warmup. Later TPU speculative-op and KV-offload
        # warmup refresh the shared baseline in their own completion paths.
        if envs.SGLANG_TPU_CHECK_RECOMPILATION.get():
            # A draft model precompiles later, from precompile_draft, and installs
            # the checker itself once that is done.
            if not self.model_runner.is_draft_worker:
                install_recompilation_checker(
                    policies=installed_policies,
                    model_label=type(self.model).__name__,
                )
        # PD harnesses wait for this marker before issuing requests.
        logger.info("TpuGraphRunner precompile done")

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
