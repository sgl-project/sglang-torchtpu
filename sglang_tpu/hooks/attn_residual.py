# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 attention-residual decode kernel (opt-in).

register_attn_residual_ops
  Why: the TPU Pallas decode kernel (sglang_tpu.kimi_attn_residual) must stay
       out of core, and its custom ops must exist before torch.compile traces
       the model.
  How: AFTER KimiK3LinearModel.__init__ registers the kernel's custom ops and
       registers aggregate with attn_residual.register_aggregate_impl. Passes
       through unless SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL=1 when the model is
       built. The platform's process_model_after_loading then calls
       prepare_attn_residual_kernel.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1428.
  Remove when: core registers attention-residual ops ahead of the trace.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_KERNEL_MODULE = "sglang_tpu.kimi_attn_residual"

# The kernel module once a model with attention residuals was built with the
# flag set, else None. Aggregation checks this module global, a constant under
# torch.compile, instead of reading os.environ per call; with the flag off
# nothing imports the kernel module (or JAX).
_kernel: ModuleType | None = None


def register_attn_residual_ops(result: None, model: Any, *args, **kwargs) -> None:
    """AFTER KimiK3LinearModel.__init__: building an op inside the dynamo trace
    fails, so register both decode ops before it."""
    global _kernel
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL.get():
        return
    if model.config.attn_res_block_size is None:
        return
    from sglang.srt.layers.attn_residual import register_aggregate_impl
    from sglang.srt.platforms import current_platform

    kernel = importlib.import_module(_KERNEL_MODULE)
    kernel.register_kimi_attn_res_decode_ops()
    _kernel = kernel
    register_aggregate_impl(current_platform.get_dispatch_key_name(), aggregate)


def prepare_attn_residual_kernel(causal_lm: Any) -> None:
    """From the platform's process_model_after_loading, after post_load_weights
    warms get_cw: also build each aggregation point's fp32 out-norm weight
    before the trace."""
    if _kernel is None:
        return
    from sglang.srt.layers.utils import PPMissingLayer

    model = causal_lm.model
    for layer in model.layers:
        if isinstance(layer, PPMissingLayer) or not layer.use_attn_residuals:
            continue
        _kernel.get_norm_weight_f32(layer.input_layernorm)
        _kernel.get_norm_weight_f32(layer.post_attention_layernorm)
    if hasattr(model, "output_attn_res_proj"):
        _kernel.get_norm_weight_f32(model.norm)
    # The kernel reads these copies and the cached fp32 score weights (get_cw),
    # which writes that skip post-load would leave stale. The weight updater's
    # derived-cache guard rejects every update path.
    model._derived_weight_cache_error = (
        "Online weight updates are not supported with "
        "SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL=1: the attention-residual "
        "kernel reads weight copies built at load time. Restart the "
        "server to change weights."
    )


def aggregate(prefix_a, prefix_b, bank, nvb, score_proj, score_norm, out_norm):
    """attn_residual's registered aggregation: decode-sized calls in one kernel
    call for the pending add, mix and output norm. Returns (normed, prefix)
    with prefix = prefix_a + prefix_b (prefix_a if None), or None to leave the
    call to core."""
    if _kernel is None or not _kernel.supports_attn_res_decode(
        prefix_a, prefix_b, bank, score_norm, out_norm
    ):
        return None
    from sglang.srt.layers.attn_residual import get_cw

    return _kernel.attention_residual_decode_tpu(
        prefix_a,
        prefix_b,
        bank,
        nvb,
        get_cw(score_proj, score_norm),
        score_norm.variance_epsilon,
        _kernel.get_norm_weight_f32(out_norm),
        out_norm.variance_epsilon,
    )
