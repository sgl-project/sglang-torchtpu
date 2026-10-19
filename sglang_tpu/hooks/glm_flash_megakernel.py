# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel (experimental, opt-in).

  Why: SGLANG_TPU_GLM_FLASH_MEGAKERNEL runs single-request decode
       steps (the MTP verify with speculative decoding) as one Pallas call per device (sglang_tpu/glm_flash_megakernel).
  How: BEFORE Glm5NextForConditionalGeneration.load_weights loads its packed
       weights (HBM still contiguous; the KV pool is sized after them); AFTER
       TpuGraphRunner.__init__ builds it and wraps the compiled model.forward,
       so the steps it handles go to it and everything else to the compiled
       decoder.
  Upstream PR or issue: none (TPU-only kernel).
  Remove when: the megakernel is a seam of the model, or dropped.
"""

from __future__ import annotations

from typing import Any


def load_glm_flash_megakernel_weights(model: Any, *args, **kwargs) -> None:
    """BEFORE Glm5NextForConditionalGeneration.load_weights: the megakernel's
    packed weights go on the device first, while HBM is still contiguous."""
    from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
    from sglang_tpu.glm_flash_megakernel import ops

    # The MTP draft loads through this method too; the kernel serves the target.
    if (not ops.enabled() or hasattr(model, "_tpu_glm_flash_mk_weights")
            or isinstance(model, Glm5NextForConditionalGenerationNextN)):
        return None
    from sglang_tpu.envs import envs as tpu_envs

    packed = tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_PACKED.get()
    if not packed:
        raise ValueError(
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL needs "
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_PACKED (packed weights dir)"
        )
    model._tpu_glm_flash_mk_weights = ops.load_packed_weights(
        packed, model.model.embed_tokens.weight.device
    )
    return None


def route_glm_flash_megakernel(result: None, runner: Any, *args, **kwargs) -> None:
    """AFTER TpuGraphRunner.__init__: the decode compile policy has replaced
    ``model.forward`` with a compiled (fullgraph) wrapper, which cannot run the
    megakernel's per-request bookkeeping. Wrap it from outside: steps the
    megakernel handles go to it, the rest to the compiled forward."""
    from sglang_tpu.glm_flash_megakernel import ops

    model = runner.model
    weights = getattr(model, "_tpu_glm_flash_mk_weights", None)
    if not ops.enabled() or weights is None:
        return result
    from sglang_tpu.envs import envs as tpu_envs

    args = runner.model_runner.server_args
    # The kernel keeps a request's caches itself and does not write serving's
    # pools, so a request must never fall back to the native decoder mid-way
    # (a second running request) or share a prefix (the radix cache).
    if args.max_running_requests != 1 or not args.disable_radix_cache:
        raise NotImplementedError(
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL needs --max-running-requests 1 and "
            "--disable-radix-cache")
    # With MTP every target decode step is a verify of all the draft tokens.
    n_query = (args.speculative_num_draft_tokens or 1) if args.speculative_algorithm else 1
    mk = ops.GlmFlashMegakernel(
        model,
        args.model_path,
        tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_PACKED.get(),
        tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN.get(),
        weights,
        n_query,
    )
    del model._tpu_glm_flash_mk_weights
    model._tpu_glm_flash_mk = mk
    compiled = model.forward

    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        if mk.handles(forward_batch, input_ids):
            return mk.forward(input_ids, positions, forward_batch)
        mode = forward_batch.forward_mode
        if mk.active and (mode.is_decode() or mode.is_target_verify()):
            raise RuntimeError(
                "GLM-Flash megakernel: a decode step it cannot take arrived for a "
                "request it is serving; serving's caches for it are stale")
        mk.invalidate()
        return compiled(input_ids, positions, forward_batch, *args, **kwargs)

    if hasattr(compiled, "__signature__"):
        forward.__signature__ = compiled.__signature__
    model.forward = forward
    return result
