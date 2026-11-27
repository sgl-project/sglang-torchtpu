# SPDX-License-Identifier: Apache-2.0
"""Online RL weight reload hooks for TPU FP8 linear and MoE quantization methods.

fp8_reload_hooks
  Why: TPU FP8 linear and MoE post-load preparation transposes weights to K-major
       layouts ([K, N] and [E, K, N]) and replaces checkpoint scale tensors with
       runtime channelwise/4D scale buffers. Online RL weight updates call
       QuantizeMethodBase.restore_weights_before_loading -> load_weights ->
       process_weights_after_loading; without restoring canonical [N, K] and
       [E, N, K] shapes first, SGLang weight loaders fail on shape/scale mismatch.
       In addition, online --quantization fp8 MoE in core falls through to a host
       per-tensor loop that strips Parameter.weight_loader instead of running on-TPU
       channelwise quantization.
  How: BEFORE Fp8LinearMethod.process_weights_after_loading and AROUND
       Fp8MoEMethod.process_weights_after_loading record the final post-__init__
       weight/scale layout (including packed weight_loader wrappers) before any
       post-load mutation runs; AFTER restore_weights_before_loading restores
       canonical shapes while keeping Parameter identities intact for direct_compile;
       AROUND Fp8MoEMethod.process_weights_after_loading routes online
       (--quantization fp8) channelwise TPU MoE through
       prepare_tpu_online_fp8_moe_weights + finalize_tpu_fp8_moe_after_loading.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1767 (TPU online FP8 RL support).
  Remove when: QuantizeMethodBase / Fp8LinearMethod / Fp8MoEMethod expose a
       platform weight-reload seam in shared core.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType


def record_fp8_linear_before_process(self, layer):
    from sglang_tpu.fp8_linear import record_tpu_fp8_linear_reload_state

    record_tpu_fp8_linear_reload_state(layer)
    return None


_UNSUPPORTED_RELOAD = (
    "TPU FP8 online weight reload supports block-FP8 checkpoints and online "
    "--quantization fp8 only; core post-processing for {} rebinds parameters that "
    "direct_compile has already captured, so a reload would serve stale weights."
)


def restore_fp8_linear_before_load(result, self, layer):
    from sglang_tpu.fp8_linear import restore_tpu_fp8_linear_weights_before_loading

    if self.use_mxfp8 or (self.is_checkpoint_fp8_serialized and not self.block_quant):
        raise NotImplementedError(
            _UNSUPPORTED_RELOAD.format("per-tensor or MXFP8 linear layers")
        )
    restore_tpu_fp8_linear_weights_before_loading(layer)
    return result


def _is_online_channelwise_moe(method) -> bool:
    from sglang_tpu.fp8_moe import _get_tpu_fp8_runtime

    return (
        not method.block_quant
        and not method.quant_config.is_checkpoint_fp8_serialized
        and not method.use_mxfp8
        and not method.is_fp4_expert
        and _get_tpu_fp8_runtime() == "channelwise"
    )


def restore_fp8_moe_before_load(result, self, layer):
    from sglang_tpu.fp8_moe import restore_tpu_fp8_moe_weights_before_loading

    is_fp8_expert = not self.is_fp4_expert and not self.use_mxfp8
    if is_fp8_expert and not self.block_quant and not _is_online_channelwise_moe(self):
        raise NotImplementedError(
            _UNSUPPORTED_RELOAD.format("per-tensor or non-channelwise MoE layers")
        )
    restore_tpu_fp8_moe_weights_before_loading(layer)
    return result


def process_fp8_moe_after_load(original_fn, self, layer):
    from sglang_tpu.fp8_moe import (
        finalize_tpu_fp8_moe_after_loading,
        prepare_tpu_online_fp8_moe_weights,
        record_tpu_fp8_moe_reload_state,
    )

    if not self.is_fp4_expert and not self.use_mxfp8:
        record_tpu_fp8_moe_reload_state(layer)
    if _is_online_channelwise_moe(self):
        prepare_tpu_online_fp8_moe_weights(layer)
        finalize_tpu_fp8_moe_after_loading(self, layer)
        return None
    return original_fn(self, layer)


def hook_specs() -> list[tuple]:
    return [
        (
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading",
            record_fp8_linear_before_process,
            HookType.BEFORE,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.restore_weights_before_loading",
            restore_fp8_linear_before_load,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.restore_weights_before_loading",
            restore_fp8_moe_before_load,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading",
            process_fp8_moe_after_load,
            HookType.AROUND,
        ),
    ]
