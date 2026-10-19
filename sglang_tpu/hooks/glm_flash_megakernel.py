# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel (experimental, opt-in).

Why: SGLANG_TPU_GLM_FLASH_MEGAKERNEL runs single-request decode steps (the
     MTP verify) as one Pallas call per device.
How: BEFORE Glm5NextForConditionalGeneration.load_weights reserves the stacked
     expert arrays while HBM is contiguous; AFTER TpuGraphRunner.__init__ builds
     the kernel and wraps the compiled model.forward, routing the steps it
     handles to the kernel.
Upstream PR or issue: none (TPU-only kernel).
Remove when: the megakernel is a seam of the model, or dropped.
"""

from __future__ import annotations

from typing import Any


def reserve_glm_flash_megakernel_experts(model: Any, *args, **kwargs) -> None:
    """BEFORE Glm5NextForConditionalGeneration.load_weights, while HBM is contiguous."""
    from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.models.glm_flash_megakernel import ops
    from sglang_tpu.models.glm_flash_megakernel.build import expert_buffers

    # The MTP draft loads through this method too; the kernel serves the target.
    if (
        not ops.enabled()
        or hasattr(model, "_tpu_glm_flash_mk_experts")
        or isinstance(model, Glm5NextForConditionalGenerationNextN)
    ):
        return None
    model._tpu_glm_flash_mk_experts = expert_buffers(
        model.config, get_parallel().tp_size, model.model.embed_tokens.weight.device
    )
    return None


def route_glm_flash_megakernel(result: None, runner: Any, *args, **kwargs) -> None:
    """AFTER TpuGraphRunner.__init__: wrap the compiled (fullgraph) model.forward."""
    from sglang_tpu.models.glm_flash_megakernel import ops

    model = runner.model
    experts = getattr(model, "_tpu_glm_flash_mk_experts", None)
    if not ops.enabled() or experts is None:
        return result
    from sglang_tpu.envs import envs as tpu_envs

    args = runner.model_runner.server_args
    # The kernel does not write serving's pools: no mid-request fallback, no shared prefix.
    if args.max_running_requests != 1 or not args.disable_radix_cache:
        raise NotImplementedError(
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL needs --max-running-requests 1 and "
            "--disable-radix-cache"
        )
    n_query = (
        (args.speculative_num_draft_tokens or 1) if args.speculative_algorithm else 1
    )
    mk = ops.GlmFlashMegakernel(
        model,
        args.model_path,
        tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN.get(),
        experts,
        n_query,
    )
    del model._tpu_glm_flash_mk_experts
    model._tpu_glm_flash_mk = mk
    compiled = model.forward

    # Skip only the full-attention metadata; the mamba-state commit reads the linear one.
    full = getattr(runner.model_runner.attn_backend, "full_attn_backend", None)
    if full is not None:
        init_full = full.init_forward_metadata

        def init_forward_metadata(forward_batch):
            if mk.handles(forward_batch, forward_batch.input_ids):
                return None
            return init_full(forward_batch)

        full.init_forward_metadata = init_forward_metadata

    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        if mk.handles(forward_batch, input_ids):
            return mk.forward(input_ids, positions, forward_batch)
        mode = forward_batch.forward_mode
        if mk.active and (mode.is_decode() or mode.is_target_verify()):
            raise RuntimeError(
                "GLM-Flash megakernel: a decode step it cannot take arrived for a "
                "request it is serving; serving's caches for it are stale"
            )
        mk.invalidate()
        return compiled(input_ids, positions, forward_batch, *args, **kwargs)

    if hasattr(compiled, "__signature__"):
        forward.__signature__ = compiled.__signature__
    model.forward = forward
    return result
