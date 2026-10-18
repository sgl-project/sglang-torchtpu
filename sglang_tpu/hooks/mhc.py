# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash fused mHC site kernel (opt-in).

register_mhc_ops, wire_hc_ffn_post_pre, hc_pre, hc_ffn_post_pre, hc_post
  Why: hc_pre / hc_post pick their implementation with in-module aiter and
       TileLang gates and have no backend seam, and the TPU Pallas kernel
       (sglang_tpu.mhc) must stay out of core.
  How: AFTER Glm5NextDecoderLayer.__init__ registers the kernel's custom ops
       before torch.compile traces the model. BEFORE MHCState.__init__ hands a
       Glm5NextDecoderLayer's state its hc_ffn_post_pre, which the layer
       otherwise passes only with the GPU cross-layer fusion on. AROUND
       Glm5NextDecoderLayer._hc_pre, hc_ffn_post_pre and hc_post send the sites
       the kernel supports to it and the rest to the torch path. All pass through
       unless SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC=1 when the model is built.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1470.
  Remove when: the mHC functions dispatch through a backend registry that
       sglang_tpu fills from init_backend().
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

# sglang_tpu.mhc once a GLM-5.3-Flash mHC layer was built with the flag set,
# else None: a module global, constant under torch.compile.
_kernel: ModuleType | None = None


def _kernel_for_batch() -> ModuleType | None:
    """The kernel, except on input-scattered (sequence-parallel) prefill
    batches: there the residual is this rank's token slice, which the kernel's
    site wiring does not handle, and the torch path runs on 1/tp of the tokens."""
    from sglang.srt.layers.layer_boundary.adapters.attention import (
        get_attn_tp_context,
    )

    if _kernel is None or get_attn_tp_context().input_scattered:
        return None
    return _kernel


def _enabled() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC.get()


def register_mhc_ops(result: None, layer: Any, *args, **kwargs) -> None:
    """AFTER Glm5NextDecoderLayer.__init__: building an op inside the dynamo
    trace fails, so register the kernel's ops before it."""
    global _kernel
    if _kernel is not None or not layer.config.mhc or not _enabled():
        return
    import sglang_tpu.mhc as kernel

    kernel.register_ops()
    _kernel = kernel


def wire_hc_ffn_post_pre(state: Any, *args, **kwargs):
    """BEFORE MHCState.__init__: give a GLM-5.3-Flash layer's state the fused
    post + pre, which declines what the kernel does not take."""
    from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

    layer = getattr(kwargs.get("hc_ffn_pre"), "__self__", None)
    if (
        kwargs.get("hc_ffn_post_pre") is None
        and isinstance(layer, Glm5NextDecoderLayer)
        and _enabled()
    ):
        kwargs["hc_ffn_post_pre"] = layer.hc_ffn_post_pre
    return (state, *args), kwargs


def _site(layer: Any) -> dict:
    from sglang.srt.models.glm5_next import _MHC_POST_MULT_VALUE

    config = layer.config
    return dict(
        hc_mult=config.hc_mult,
        rms_eps=config.rms_norm_eps,
        hc_eps=config.hc_eps,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        post_mult_value=_MHC_POST_MULT_VALUE,
    )


def hc_pre(
    original: Callable,
    self,
    hc_fn,
    hc_scale,
    hc_base,
    hidden_states,
    out_norm_weight,
    out_norm_eps,
):
    """AROUND Glm5NextDecoderLayer._hc_pre."""
    kernel = _kernel_for_batch()
    if kernel is not None:
        out = kernel.hc_pre(
            x=hidden_states,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            hc_norm_weight=None,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            **_site(self),
        )
        if out is not None:
            return out
    return original(
        self, hc_fn, hc_scale, hc_base, hidden_states, out_norm_weight, out_norm_eps
    )


def hc_ffn_post_pre(
    original: Callable,
    self,
    hidden_states,
    residual,
    h_res,
    h_post,
    out_norm_weight,
    out_norm_eps,
):
    """AROUND Glm5NextDecoderLayer.hc_ffn_post_pre. With the kernel on, a site
    it declines returns None and MHCState runs hc_post then hc_ffn_pre, as with
    the fusion off; the original's fused paths are GPU-only."""
    kernel = _kernel_for_batch()
    if kernel is not None:
        return kernel.hc_post_pre(
            x=hidden_states,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_fn=self.hc_ffn_fn,
            hc_scale=self.hc_ffn_scale,
            hc_base=self.hc_ffn_base,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            **_site(self),
        )
    return original(
        self, hidden_states, residual, h_res, h_post, out_norm_weight, out_norm_eps
    )


def hc_post(original: Callable, self, hidden_states, residual, h_res, h_post):
    """AROUND Glm5NextDecoderLayer.hc_post."""
    kernel = _kernel_for_batch()
    if kernel is not None:
        out = kernel.hc_post(
            x=hidden_states,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_mult=self.config.hc_mult,
        )
        if out is not None:
            return out
    return original(self, hidden_states, residual, h_res, h_post)
