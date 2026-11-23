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
  How: AFTER Fp8LinearMethod.create_weights and Fp8MoEMethod.create_weights record
       the pre-postprocess weight/scale layout; AFTER restore_weights_before_loading
       restores canonical shapes while keeping Parameter identities intact for
       direct_compile; AROUND Fp8MoEMethod.process_weights_after_loading routes
       online (--quantization fp8) channelwise TPU MoE through
       prepare_tpu_online_fp8_moe_weights.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1767 (TPU online FP8 RL support).
  Remove when: QuantizeMethodBase / Fp8LinearMethod / Fp8MoEMethod expose a
       platform weight-reload seam in shared core.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType


def record_fp8_linear_create(result, self, layer, *args, **kwargs):
    from sglang_tpu.fp8_linear import record_tpu_fp8_linear_reload_state

    record_tpu_fp8_linear_reload_state(layer)
    return result


def restore_fp8_linear_before_load(result, self, layer):
    from sglang_tpu.fp8_linear import restore_tpu_fp8_linear_weights_before_loading

    restore_tpu_fp8_linear_weights_before_loading(layer)
    return result


def record_fp8_moe_create(result, self, layer, *args, **kwargs):
    from sglang_tpu.fp8_moe import record_tpu_fp8_moe_reload_state

    record_tpu_fp8_moe_reload_state(layer)
    return result


def restore_fp8_moe_before_load(result, self, layer):
    from sglang_tpu.fp8_moe import restore_tpu_fp8_moe_weights_before_loading

    restore_tpu_fp8_moe_weights_before_loading(layer)
    return result


def process_fp8_moe_after_load(original_fn, self, layer):
    from sglang_tpu.fp8_moe import (
        _get_tpu_fp8_runtime,
        prepare_tpu_online_fp8_moe_weights,
    )

    if (
        not self.block_quant
        and not self.quant_config.is_checkpoint_fp8_serialized
        and not self.use_mxfp8
        and not self.is_fp4_expert
        and _get_tpu_fp8_runtime() == "channelwise"
    ):
        prepare_tpu_online_fp8_moe_weights(layer)
        return None
    return original_fn(self, layer)


def hook_specs() -> list[tuple]:
    return [
        (
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.create_weights",
            record_fp8_linear_create,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.restore_weights_before_loading",
            restore_fp8_linear_before_load,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.create_weights",
            record_fp8_moe_create,
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
