# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode megakernel (experimental, opt-in).

Why: SGLANG_TPU_GLM_FLASH_MEGAKERNEL runs target decode steps (the MTP verify)
     as one Pallas call per device.
How: AFTER TpuGraphRunner.__init__ builds the kernel on the target model and
     wraps the compiled model.forward, routing the steps it handles to the
     kernel. The kernel reads and writes serving's pools in place, so the
     attention metadata, verify commit and radix snapshots stay serving's.
Upstream PR or issue: none (TPU-only kernel).
Remove when: the megakernel is a seam of the model, or dropped.
"""

from __future__ import annotations

from typing import Any

_KERNELS: list = []


def route_glm_flash_megakernel(result: None, runner: Any, *args, **kwargs) -> None:
    """AFTER TpuGraphRunner.__init__: wrap the compiled (fullgraph) model.forward."""
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
    from sglang_tpu.glm_flash_megakernel import ops

    model = runner.model
    # The MTP draft has its own graph runner; the kernel serves the target.
    if not ops.enabled() or type(model) is not Glm5NextForConditionalGeneration:
        return result
    from sglang_tpu.envs import envs as tpu_envs

    args = runner.model_runner.server_args
    n_query = (
        (args.speculative_num_draft_tokens or 1) if args.speculative_algorithm else 1
    )
    mk = ops.GlmFlashMegakernel(
        model,
        args.model_path,
        tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN.get()
        or runner.model_runner.model_config.context_len,
        n_query,
        tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS.get(),
        runner.model_runner.attn_backend,
        call_rows=tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL_CALL_ROWS.get(),
        max_requests=args.max_running_requests,
    )
    mk.radix_snapshots = not args.disable_radix_cache
    model._tpu_glm_flash_mk = mk
    _KERNELS.append(mk)
    compiled = model.forward

    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        if mk.handles(forward_batch, input_ids):
            return mk.forward(input_ids, positions, forward_batch)
        return compiled(input_ids, positions, forward_batch, *args, **kwargs)

    if hasattr(compiled, "__signature__"):
        forward.__signature__ = compiled.__signature__
    model.forward = forward
    return result
