"""TPU "graph runner": installs torch.compile on the model.

Unlike CudaGraphRunner / NPUGraphRunner there is no graph capture or
replay here — PJRT (libtpu) caches HLO compilations internally, keyed
by tensor shape, so we get "compile once, dispatch by shape" behavior
without an explicit capture step. This class exists so the
torch.compile install hangs off the same factory hook
(``current_platform.get_graph_runner_cls()``) as the in-tree runners,
matching the pattern used by ``CPUGraphRunner`` which is also just a
torch.compile wrapper.

Wired up by ``ModelRunner.init_device_graphs`` when both:

* ``server_args.enable_torch_compile`` is True (checked at the call
  site, mirroring the CPU path), and
* the model does not set ``_can_torch_compile = False`` (the same
  per-model opt-out the CUDA path honors in ``kernel_warmup``).

Reassigning attributes on the model instance shadows the class methods
for that instance only — every existing call site in ``ModelRunner``
(``self.model.<name>(...)``) picks up the compiled version automatically.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# This module lives outside the tpu/ tree, so __name__ does not start with
# the TPU logger root. Pass a short suffix so the logger lands cleanly under
# that root (one setLevel covers the whole backend) instead of producing a
# doubled "...tpu.sglang.srt.model_executor.tpu_graph_runner" name.
logger = get_logger("tpu_graph_runner")


def _wrap_with_tpu_compile(model, attr_name: str) -> bool:
    """Replace ``model.<attr_name>`` with a torch.compile + no_grad wrapper.

    Returns True if the attribute existed and was wrapped, False if the
    model does not expose it (which is the common case for the optional
    ``forward_split_prefill`` entry point).
    """
    fn = getattr(model, attr_name, None)
    if fn is None:
        return False
    signature = inspect.signature(fn)
    compiled = torch.compile(
        torch.no_grad()(fn),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    compiled.__signature__ = signature
    setattr(model, attr_name, compiled)
    return True


def _can_split_compile(model) -> bool:
    """True if the model exposes the standard ``(model, logits_processor)``
    split, allowing us to compile the transformer body and the LP as
    separate regions instead of fusing them under one outer compile.

    The Qwen / Llama / GLM / Gemma families all match this shape via
    sglang's ``XxxForCausalLM`` wrapper, where ``self.model`` is the
    transformer backbone and ``self.logits_processor`` is the
    :class:`~sglang.srt.layers.logits_processor.LogitsProcessor` epilogue.
    """
    body = getattr(model, "model", None)
    lp = getattr(model, "logits_processor", None)
    return (
        body is not None
        and lp is not None
        and callable(getattr(body, "forward", None))
        and callable(getattr(lp, "forward", None))
    )


class TpuGraphRunner:
    """Install ``torch.compile(backend='tpu')`` on the model backbone.

    Two compile shapes are supported, picked automatically:

    * **Split-compile** (preferred). When the model exposes the standard
      ``(model, logits_processor)`` split, the transformer body and the
      :class:`LogitsProcessor` are compiled as separate regions; the
      outer ``ForCausalLM.forward`` stays uncompiled and dispatches between
      them. This isolates the LM-head matmul + TP all-gather in the LP's
      own HLO, so XLA's SC-offload pass cannot fuse the all-gather into
      the body graph — that fusion is the root cause of the
      ``tt_jit_logits_processor`` UserFatal halt on Qwen3-32B TP=8.
      Mirrors torchtpu-vllm's ``compute_selected_logits`` carve-out.

    * **Single-compile fallback**. For models that do not expose the
      split (no ``model.model`` / ``model.logits_processor``), wrap the
      outer ``forward`` and ``forward_split_prefill`` as one region each.
    """

    # Single-compile fallback targets. ``forward`` is hit by
    # forward_decode / forward_extend / forward_idle.
    # ``forward_split_prefill`` is hit by the split-prefill path on
    # models that expose it (qwen2/qwen3 family, glm4, gemma); it is
    # wrapped so first invocation populates the HLO cache, but
    # ``CompilationManager`` only precompiles ``forward`` today — see
    # the warning in ``CompilationManager.__init__``.
    _COMPILE_TARGETS: tuple[str, ...] = ("forward", "forward_split_prefill")

    def __init__(self, model_runner: ModelRunner):
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
            model_runner.server_args.enable_torch_compile = False
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
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            torch._dynamo.config.fail_on_recompile_limit_hit = True

        _register_tpu_compile_backend()

        # Each install path returns ``(output_targets, compiled_fns)``:
        #   output_targets : list of (owner, attr, fn) that need the
        #                    output-slicing shim wrapped around them
        #                    — exactly the frontier that emits
        #                    ``LogitsProcessorOutput``.
        #   compiled_fns   : dict[str, Callable] — informational, fed to
        #                    ``CompilationManager`` for the
        #                    ``forward_split_prefill`` warning.
        if _can_split_compile(self.model):
            output_targets, compiled_fns = self._install_split_compile()
        else:
            output_targets, compiled_fns = self._install_single_compile()

        if not output_targets and not compiled_fns:
            return

        from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
            TpuCompiledForwardShim,
        )

        compiled_forward_shim = TpuCompiledForwardShim(
            model_runner, output_targets=output_targets
        ).install()
        model_runner.compiled_forward_shim = compiled_forward_shim

        from sglang.srt.hardware_backend.tpu.compilation.compilation_manager import (
            CompilationManager,
        )

        CompilationManager(
            model_runner=model_runner,
            compiled_fns=compiled_fns,
            t_bucketer=compiled_forward_shim.t_bucketer,
            b_bucketer=compiled_forward_shim.b_bucketer,
        ).precompile_all()

    def _install_single_compile(
        self,
    ) -> tuple[list[tuple[object, str, Callable]], dict[str, Callable]]:
        """Fallback path: compile ``model.forward`` (and optional
        ``forward_split_prefill``) as one region each. Used when the model
        does not expose the ``(model, logits_processor)`` split.

        Both ``forward`` and ``forward_split_prefill`` can return
        ``LogitsProcessorOutput`` (the latter on its final split), so
        both go into ``output_targets`` for output slicing.
        """
        wrapped = [
            name
            for name in self._COMPILE_TARGETS
            if _wrap_with_tpu_compile(self.model, name)
        ]
        if not wrapped:
            logger.warning(
                "enable_torch_compile=True but %s exposes none of %s; "
                "nothing compiled.",
                type(self.model).__name__,
                self._COMPILE_TARGETS,
            )
            return [], {}

        logger.info(
            "Single-compile: wrapped %s.{%s} with torch.compile.",
            type(self.model).__name__,
            ", ".join(wrapped),
        )
        output_targets = [
            (self.model, name, getattr(self.model, name)) for name in wrapped
        ]
        compiled_fns = {name: getattr(self.model, name) for name in wrapped}
        return output_targets, compiled_fns

    def _install_split_compile(
        self,
    ) -> tuple[list[tuple[object, str, Callable]], dict[str, Callable]]:
        """Split-compile path: compile ``model.model.forward`` (body) and
        ``model.logits_processor.forward`` (LP) as separate regions.

        Leaves outer ``model.forward`` (the ``ForCausalLM`` wrapper)
        uncompiled — it runs in eager and dispatches body→LP, so the
        LM-head + TP all-gather end up in the LP's own HLO instead of
        being fused into the body's.
        """
        output_targets: list[tuple[object, str, Callable]] = []
        compiled_fns: dict[str, Callable] = {}

        if not _wrap_with_tpu_compile(self.model.model, "forward"):
            logger.warning(
                "Split-compile expected %s.model.forward but it is missing; "
                "falling back to single-compile.",
                type(self.model).__name__,
            )
            return self._install_single_compile()
        compiled_fns["model.forward"] = self.model.model.forward

        lp = self.model.logits_processor
        if not _wrap_with_tpu_compile(lp, "forward"):
            logger.warning(
                "Split-compile expected %s.logits_processor.forward but it "
                "is missing; body compiled, LP left in eager.",
                type(self.model).__name__,
            )
            return output_targets, compiled_fns
        output_targets.append((lp, "forward", lp.forward))
        compiled_fns["logits_processor.forward"] = lp.forward

        # ``forward_split_prefill`` (when exposed) is compiled to populate
        # the HLO cache but not added to ``output_targets`` — see
        # docstring.
        if _wrap_with_tpu_compile(self.model, "forward_split_prefill"):
            compiled_fns["forward_split_prefill"] = self.model.forward_split_prefill

        logger.info(
            "Split-compile: wrapped %s.model.forward, "
            "%s.logits_processor.forward%s with torch.compile.",
            type(self.model).__name__,
            type(self.model).__name__,
            (
                ", " + type(self.model).__name__ + ".forward_split_prefill"
                if "forward_split_prefill" in compiled_fns
                else ""
            ),
        )
        return output_targets, compiled_fns

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """Always False — no captured graph to replay.

        PJRT/libtpu caches HLO compilations internally and dispatches
        on tensor shape, so the compiled forward is reached through the
        normal ``self.model.forward(...)`` call path, not through a
        replay step. This method exists because ModelRunner.forward
        gates the replay branch on
        ``self.graph_runner and self.graph_runner.can_run(...)`` —
        returning False keeps that branch off on TPU.
        """
        return False
