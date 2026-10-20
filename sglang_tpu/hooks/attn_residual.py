# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 attention-residual decode kernel (opt-in).

register_attn_residual_ops, prepare_attn_residual_kernel, aggregate,
aggregate_fused_add
  Why: attn_residual picks its aggregation implementation with in-module
       CUDA/HIP gates and has no backend seam, and the TPU Pallas decode kernel
       (sglang_tpu.kimi_attn_residual) must stay out of core.
  How: AFTER KimiK3LinearModel.__init__ registers the kernel's custom ops
       before torch.compile traces the model. AFTER
       KimiK3LinearForCausalLM.post_load_weights builds the kernel's fp32
       out-norm weights and refuses online weight updates. AROUND
       attn_residual._aggregate and _aggregate_fused_add send decode-sized
       calls to the kernel and the rest to the original. All four pass through
       unless SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL=1 when the model is built.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1428.
  Remove when: attn_residual dispatches aggregation through a backend registry
       that sglang_tpu fills from init_backend() (#1428).
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Callable

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
    kernel = importlib.import_module(_KERNEL_MODULE)
    kernel.register_kimi_attn_res_decode_ops()
    _kernel = kernel


def prepare_attn_residual_kernel(result: None, causal_lm: Any, *args, **kwargs) -> None:
    """AFTER KimiK3LinearForCausalLM.post_load_weights, which warms get_cw: also
    build each aggregation point's fp32 out-norm weight before the trace."""
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


def _decode(prefix_a, prefix_b, bank, nvb, score_proj, score_norm, out_norm):
    """One kernel call for the pending add, mix and output norm. Returns
    (normed, prefix) with prefix = prefix_a + prefix_b (prefix_a if None)."""
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


def aggregate(
    original: Callable,
    prefix_sum,
    bank,
    nvb,
    score_proj,
    score_norm,
    out_norm,
    write_bank_row=False,
):
    """AROUND attn_residual._aggregate. A fused bank write runs the original:
    the kernel has none, and the caller asks for one only on the CUDA/HIP fast
    paths."""
    if (
        _kernel is None
        or write_bank_row
        or not _kernel.supports_attn_res_decode(
            prefix_sum, None, bank, score_norm, out_norm
        )
    ):
        return original(
            prefix_sum, bank, nvb, score_proj, score_norm, out_norm, write_bank_row
        )
    return _decode(prefix_sum, None, bank, nvb, score_proj, score_norm, out_norm)[0]


def aggregate_fused_add(
    original: Callable,
    prefix_a,
    prefix_b,
    bank,
    nvb,
    score_proj,
    score_norm,
    out_norm,
    write_bank_row=False,
):
    """AROUND attn_residual._aggregate_fused_add, as aggregate with the pending
    add in the kernel. Returns (normed, prefix) like the original."""
    if (
        _kernel is None
        or write_bank_row
        or not _kernel.supports_attn_res_decode(
            prefix_a, prefix_b, bank, score_norm, out_norm
        )
    ):
        return original(
            prefix_a,
            prefix_b,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row,
        )
    return _decode(prefix_a, prefix_b, bank, nvb, score_proj, score_norm, out_norm)
