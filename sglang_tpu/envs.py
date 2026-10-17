# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import EnvBool, EnvField, EnvFloat, EnvInt

__all__ = ["TpuEnvs", "envs"]

# Core closes EnvField.__set_name__ after its own Envs, reopen it for this class.
EnvField._allow_set_name = True


class TpuEnvs:
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Seconds between periodic trims. 0 turns the periodic trim off.
    SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC = EnvFloat(0.0)
    # MoE: one fused kernel for single-token packed-MXFP4 decode
    # (moe/pallas/single_token_moe.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL = EnvBool(True)
    # FP8 linear: one W8A8 kernel for decode-sized batches (kernels/fp8_linear_small_m.py);
    # False keeps the XLA path.
    SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL = EnvBool(False)
    # Launch-time xprof capture (see profile_auto.py): each stage once, without
    # /start_profile. Keep the step count small: each chip's trace is hundreds
    # of MB.
    SGLANG_TPU_ENABLE_PROFILE_AUTO = EnvBool(False)
    SGLANG_TPU_PROFILE_AUTO_STEPS = EnvInt(5)
    # Kimi-K3: run attention-residual aggregation for decode-sized batches as
    # one Pallas kernel (pending add + mix + out norm) instead of the torch
    # reference (see kimi_attn_residual.py and hooks/attn_residual.py). Not
    # bit-exact with it: the kernel scores in f32, the torch path in bf16.
    SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL = EnvBool(False)
    # GLM-5.3-Flash: one fused Pallas kernel per mHC site instead of the torch
    # hc_pre / hc_post (see mhc.py and hooks/mhc.py).
    SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC = EnvBool(False)
    # Drop Dynamo guards on nn.Module state (attributes, submodules, parameters)
    # for the decoder graph; input and global guards stay (see
    # decoder_compile_policy._guard_filter_options). Assumes module state is
    # fixed after warmup.
    SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER = EnvBool(False)

    # Speculative decoding
    # Route DFLASH/DSPARK to sglang_tpu.experimental.TpuBlockSpecWorker; SGLANG_TPU_HOST_PADDING_NO_D2H also does.
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER = EnvBool(False)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
