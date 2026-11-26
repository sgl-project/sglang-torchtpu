# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 fused mHC post-pre (SGLANG_TPU_DSV41_FUSED_MHC=1; the AgentX
DSV4.1 recipe sets it, an explicit 0 turns it off).

Between two sublayers the torch path runs the post (stream remix), the next
sublayer's combine (collapse with the lagged pre) and input RMSNorm, and, at the
start of the next sublayer, its coefficients (``_mix_stats_impl``: RMS-scaled
projection, gates, Sinkhorn) from the new streams. One Pallas call
(sglang_tpu/kernels/mhc_v41.py) does all four. The result is handed over the way
the core's fused boundaries do it: ``HcState(new streams, pre, HcNormed(next
input), next coefficients)``; the next combine returns the normed rows and the
next post takes the coefficients as ``precomputed``.

Covered: the sublayer boundaries the decoder layer builds (attention -> MoE in a
layer, MoE -> the next layer's attention). Torch path: the first sublayer of
layer 0, the last layer's MoE boundary and the final head collapse, boundaries into an
Engram layer, the DSpark draft stages (no boundary), batches below
SGLANG_TPU_DSV41_FUSED_MHC_MIN_TOKENS rows, and anything outside the kernel's
shape and dtype contract (``post_pre_fn_parts``).

Hook bodies; their headers and registration are in hooks.py
(``fused_mhc_hook_specs``).
"""

from threading import Lock
from typing import Callable, Optional

import torch

# Kernel contract: hc_mult = 4, d a multiple of the 128 lanes, bf16 streams.
_HC_MULT = 4
_LANES = 128
_DEVICE_TYPE = "tpu"

# The layer buffers holding each hyper-connection's split fn, read through
# LiveHcSubLayer.fused_fn_parts (hooks/dsv4.py).
FN_PARTS_ATTRS = {True: "hc_attn_fn_parts", False: "hc_ffn_fn_parts"}

# Set by register_ops(): module globals, constant under compilation.
_post_pre_op: Optional[Callable] = None
_min_tokens = 1
_OPS_LOCK = Lock()


def register_ops() -> None:
    """Build the kernel's custom op; it must exist before the model is traced."""
    global _post_pre_op, _min_tokens
    with _OPS_LOCK:
        if _post_pre_op is not None:
            return

        import jax

        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.kernels.mhc_v41 import param_row, v41_post_pre
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _post_pre(
            y: jax.Array,
            res: jax.Array,
            pre: jax.Array,
            post: jax.Array,
            comb: jax.Array,
            fn_parts: jax.Array,
            base: jax.Array,
            scale: jax.Array,
            norm_weight: jax.Array,
            hc: int,
            iters: int,
            rms_eps: float,
            hc_eps: float,
            norm_eps: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return tuple(
                v41_post_pre(
                    y,
                    res,
                    pre,
                    post,
                    comb,
                    fn_parts,
                    param_row(base, scale),
                    norm_weight,
                    hc=hc,
                    iters=iters,
                    rms_eps=rms_eps,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                )
            )

        def _fake(y, res, pre, post, comb, fn_parts, base, scale, norm, hc, *_):
            tokens = res.shape[0]
            return (
                torch.empty_like(res),
                torch.empty_like(y),
                res.new_empty((tokens, hc), dtype=torch.float32),
                res.new_empty((tokens, hc), dtype=torch.float32),
                res.new_empty((tokens, hc * hc), dtype=torch.float32),
            )

        op = cpu_export_jax_op("sglang_tpu_pallas::mhc_v41_post_pre", _post_pre)
        op.register_fake(_fake)
        _min_tokens = max(1, tpu_envs.SGLANG_TPU_DSV41_FUSED_MHC_MIN_TOKENS.get())
        _post_pre_op = op


def split_fn(fn: torch.Tensor) -> torch.Tensor:
    """f32 [mix, hc * d] -> bf16 [3, mix, hc * d] (hi, mid, lo) whose f32 sum
    is exactly fn."""
    fn = fn.float()
    hi = fn.bfloat16()
    rest = fn - hi.float()
    mid = rest.bfloat16()
    lo = (rest - mid.float()).bfloat16()
    return torch.stack((hi, mid, lo))


def pack_fn(parts: torch.Tensor) -> torch.Tensor:
    """bf16 [3, mix, hc * d] -> contiguous [hc * d, 128], the kernel's projection
    operand (sglang_tpu/kernels/mhc_v41.pack_fn)."""
    three, mix, width = parts.shape
    stacked = torch.nn.functional.pad(
        parts.reshape(three * mix, width), (0, 0, 0, _LANES - three * mix)
    )
    return stacked.t().contiguous()


def cache_fn_parts(result, layer, *args, **kwargs):
    """AFTER DeepseekV4DecoderLayerBase.refresh_mhc_norm_weight_cache (after load
    and on weight updates): split both hyper-connections' fn once, as
    non-persistent buffers so a trace reads them as inputs, and build the op."""
    if (
        not layer.hc_pre_from_prev_sublayer
        or not getattr(layer, "boundary_mhc_fusions", True)
        or layer.hc_attn_fn.device.type != _DEVICE_TYPE
    ):
        return result
    register_ops()
    with torch.no_grad():
        for attn, fn in ((True, layer.hc_attn_fn), (False, layer.hc_ffn_fn)):
            layer.register_buffer(
                FN_PARTS_ATTRS[attn], pack_fn(split_fn(fn.data)), persistent=False
            )
    return result


def post_pre_fn_parts(hc, y, residual, coefficients, next) -> Optional[torch.Tensor]:
    """The next hyper-connection's fn parts when the kernel takes this boundary, else
    None (torch path)."""
    if _post_pre_op is None or next is None or next.hc is None:
        return None
    fn_parts = getattr(next.hc, "fused_fn_parts", None)
    if fn_parts is None or not isinstance(y, torch.Tensor) or y.dim() != 2:
        return None
    cfg, norm = hc.cfg, next.norm
    pre, post, comb = coefficients
    tokens, d = y.shape
    ok = (
        cfg.pre_from_prev
        and cfg.mult == next.hc.cfg.mult == _HC_MULT
        and tokens >= _min_tokens
        and y.device.type == _DEVICE_TYPE
        and y.dtype == residual.dtype == torch.bfloat16
        and d % _LANES == 0
        and y.is_contiguous()
        and residual.shape == (tokens, _HC_MULT, d)
        and residual.is_contiguous()
        and pre.shape == post.shape == (tokens, _HC_MULT)
        and comb.shape == (tokens, _HC_MULT, _HC_MULT)
        and fn_parts.shape == (_HC_MULT * d, _LANES)
        and norm.weight.shape == (d,)
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
    )
    return fn_parts if ok else None


def post_fusion(original, hc, y, residual, coefficients, next):
    """AROUND deepseek_v4_mhc._post_fusion: one kernel call for post, collapse,
    the next input norm and the next coefficients where it applies."""
    fn_parts = post_pre_fn_parts(hc, y, residual, coefficients, next)
    if fn_parts is None:
        return original(hc, y, residual, coefficients, next)
    from sglang.srt.models import deepseek_v4_mhc as mhc

    pre, post, comb = coefficients
    tokens, d = y.shape
    next_hc, norm = next.hc, next.norm
    new_res, normed, next_pre, next_post, next_comb = _post_pre_op(
        y,
        residual.reshape(tokens, _HC_MULT * d),
        pre,
        post,
        comb.reshape(tokens, _HC_MULT * _HC_MULT),
        fn_parts,
        next_hc.base,
        next_hc.scale,
        norm.weight,
        _HC_MULT,
        next_hc.cfg.sinkhorn_iters,
        float(next_hc.cfg.rms_eps),
        float(next_hc.cfg.eps),
        float(norm.variance_epsilon),
    )
    return mhc.HcState(
        new_res.view(tokens, _HC_MULT, d),
        pre,
        mhc.HcNormed(normed),
        (next_pre, next_post, next_comb.view(tokens, _HC_MULT, _HC_MULT)),
    )
