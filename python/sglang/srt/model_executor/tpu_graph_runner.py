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

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger

if TYPE_CHECKING:
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
    compiled = torch.compile(
        torch.no_grad()(fn),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )
    setattr(model, attr_name, compiled)
    return True


class TpuGraphRunner:
    """Install ``torch.compile(backend='tpu')`` on the model backbone."""

    # Entry points wrapped with torch.compile. ``forward`` is hit by
    # forward_decode / forward_extend / forward_idle. ``forward_split_prefill``
    # is hit by the split-prefill path on the models that expose it
    # (e.g. qwen2, qwen3_moe, glm4, gemma).
    _COMPILE_TARGETS: tuple[str, ...] = ("forward", "forward_split_prefill")

    def __init__(self, model_runner: ModelRunner):
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
        # per prefill token bucket ({16..4096}) plus the decode batch-size
        # buckets. dynamo's default cache_size_limit=8 is far too low — under
        # fullgraph=True it trips FailOnRecompileLimitHit (killing the
        # scheduler) as soon as warmup or a multi-shape benchmark walks past
        # ~8 shapes. Match the CUDA / CPU / piecewise graph runners, which all
        # set 1024 (see cuda_graph_runner.set_torch_compile_config).
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        torch._dynamo.config.accumulated_recompile_limit = 1024
        torch._dynamo.config.recompile_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024

        _register_tpu_compile_backend()

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
            return

        logger.info(
            "Wrapped %s.{%s} with torch.compile(backend='tpu', "
            "fullgraph=True, dynamic=False) inside torch.no_grad.",
            type(self.model).__name__,
            ", ".join(wrapped),
        )

    def can_run(self, forward_batch) -> bool:
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
