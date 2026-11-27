# SPDX-License-Identifier: Apache-2.0
"""HookRegistry entries of the diffusion TPU platform (the
``sglang.multimodal_gen.plugins`` entry point).

Every hook is tracked debt; its Why / How / Upstream PR or issue / Remove when
header sits with its implementation. ``apply_plugin_hooks`` resolves each
target explicitly; a target of this package that does not resolve fails
startup (``_require_hooks_applied``).
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

# (target, function of h3_megakernel/hook.py, hook type): the MiniMax H3 DiT
# megakernel, with SGLANG_TPU_ENABLE_H3_MEGAKERNEL.
H3_MEGAKERNEL_HOOKS = [
    (
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.post_load_weights",
        "pack_after_load",
        HookType.AFTER,
    ),
    (
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.validate_weight_update_source",
        "refuse_weight_updates",
        HookType.BEFORE,
    ),
    (
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.validate_lora_layers",
        "refuse_lora_layers",
        HookType.BEFORE,
    ),
    (
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.LoRAPipeline._reject_lora_on_packed_weights",
        "refuse_lora_on_packed_dit",
        HookType.BEFORE,
    ),
    (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising.MiniMaxH3DenoisingStage._run_full_loop",
        "run_request",
        HookType.AROUND,
    ),
    (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence.minimax_h3_padded_seq_len",
        "pad_to_kernel_rows",
        HookType.AROUND,
    ),
]


def _h3_megakernel(name: str):
    """``h3_megakernel.hook.<name>``, imported at its first call: only a worker
    that runs the hooks loads the megakernel package."""

    def call(*args, **kwargs):
        from sglang_tpu.multimodal_gen.h3_megakernel import hook

        return getattr(hook, name)(*args, **kwargs)

    return call


def register() -> None:
    """Register the hooks of the options this process's environment sets."""
    from sglang_tpu.envs import envs

    if envs.SGLANG_TPU_ENABLE_H3_MEGAKERNEL.get():
        for target, name, hook_type in H3_MEGAKERNEL_HOOKS:
            HookRegistry.register(target, _h3_megakernel(name), hook_type)


def require_h3_megakernel_hooks() -> None:
    """Raise unless every H3 megakernel hook is applied: without them, as in an
    install that predates this entry point, the blocks would run in torch."""
    missing = [
        target
        for target, _, _ in H3_MEGAKERNEL_HOOKS
        if target not in HookRegistry._patched
    ]
    if missing:
        raise RuntimeError(
            "SGLANG_TPU_ENABLE_H3_MEGAKERNEL is set, but the MiniMax H3 "
            f"megakernel hooks are not applied on {', '.join(missing)}. The "
            "tpu_oot_hooks entry point of sglang.multimodal_gen.plugins registers "
            "them: reinstall sglang-tpu (pip install --no-deps -e .) and keep "
            "tpu_oot_hooks in SGLANG_PLUGINS when it is set."
        )
