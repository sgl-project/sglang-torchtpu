# SPDX-License-Identifier: Apache-2.0
"""The fused mHC site kernel: hc_pre, hc_post and hc_post fused with the next
hc_pre, reached through hooks/mhc.py.

``register_ops()`` runs before torch.compile traces the model: building a custom
op inside the dynamo trace fails in inspect.signature. Each entry point declines
(returns None) a call the kernel was not validated for, which then takes the
torch path.
"""

from threading import Lock
from typing import Callable, Optional

import torch

# Kernel limits: hc_mult = 4, d % 128 == 0 and d <= 4096, post_mult = 2.0, and
# an unfolded hc_fn.
_HC_MULT = 4
_MAX_HIDDEN = 4096
_LANES = 128
_POST_MULT = 2.0

_OPS: Optional[tuple[Callable, Callable, Callable]] = None
_OPS_LOCK = Lock()


def supports_mhc_site(
    residual: torch.Tensor,
    hc_mult: int,
    *,
    x: Optional[torch.Tensor] = None,
    hc_fn: Optional[torch.Tensor] = None,
    post_mult_value: float = _POST_MULT,
    hc_norm_weight: Optional[torch.Tensor] = None,
    out_norm_weight: Optional[torch.Tensor] = None,
) -> bool:
    """Whether the kernel takes a site over residual [T, hc_mult * d]."""
    if residual.dim() != 2 or hc_mult != _HC_MULT:
        return False
    tokens, width = residual.shape
    d = width // hc_mult
    if (
        tokens == 0
        or residual.device.type != "tpu"
        or residual.dtype != torch.bfloat16
        or d * hc_mult != width
        or d % _LANES
        or d > _MAX_HIDDEN
        or post_mult_value != _POST_MULT
        or hc_norm_weight is not None
    ):
        return False
    if x is not None and (x.shape != (tokens, d) or x.dtype != residual.dtype):
        return False
    mix = (2 + hc_mult) * hc_mult
    if hc_fn is not None and hc_fn.shape != (mix, width):
        return False
    return out_norm_weight is None or out_norm_weight.shape == (d,)


def register_ops() -> None:
    global _OPS
    with _OPS_LOCK:
        if _OPS is not None:
            return

        import jax

        from sglang_tpu.layers.pallas.mhc import mhc_site, param_row
        from sglang_tpu.utils.pallas_jax_op import cpu_export_jax_op

        def _pre(
            res: jax.Array,
            fn: jax.Array,
            base: jax.Array,
            scale: jax.Array,
            norm_weight: Optional[jax.Array],
            hc: int,
            iters: int,
            rms_eps: float,
            hc_eps: float,
            norm_eps: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return mhc_site(
                None,
                res,
                None,
                None,
                fn,
                param_row(base, scale),
                hc=hc,
                iters=iters,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )

        def _seam(
            x: jax.Array,
            res: jax.Array,
            post: jax.Array,
            comb: jax.Array,
            fn: jax.Array,
            base: jax.Array,
            scale: jax.Array,
            norm_weight: Optional[jax.Array],
            hc: int,
            iters: int,
            rms_eps: float,
            hc_eps: float,
            norm_eps: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return mhc_site(
                x,
                res,
                post,
                comb,
                fn,
                param_row(base, scale),
                hc=hc,
                iters=iters,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )

        def _post(
            x: jax.Array,
            res: jax.Array,
            post: jax.Array,
            comb: jax.Array,
            hc: int,
        ) -> jax.Array:
            return mhc_site(
                x,
                res,
                post,
                comb,
                None,
                None,
                hc=hc,
                iters=1,
                rms_eps=0.0,
                hc_eps=0.0,
            )[0]

        def _gates_like(res, hc):
            tokens = res.shape[0]
            return (
                res.new_empty((tokens, res.shape[1] // hc)),
                res.new_empty((tokens, hc), dtype=torch.float32),
                res.new_empty((tokens, hc * hc), dtype=torch.float32),
            )

        pre = cpu_export_jax_op("sglang_tpu_pallas::mhc_pre", _pre)
        pre.register_fake(
            lambda res, fn, base, scale, norm, hc, *_: _gates_like(res, hc)
        )
        seam = cpu_export_jax_op("sglang_tpu_pallas::mhc_post_pre", _seam)
        seam.register_fake(
            lambda x, res, post, comb, fn, base, scale, norm, hc, *_: (
                torch.empty_like(res),
                *_gates_like(res, hc),
            )
        )
        post = cpu_export_jax_op("sglang_tpu_pallas::mhc_post", _post)
        post.register_fake(lambda x, res, *_: torch.empty_like(res))
        _OPS = (pre, seam, post)


def _ops() -> tuple[Callable, Callable, Callable]:
    assert _OPS is not None, "register_ops() must run before the model is traced"
    return _OPS


def _norm_eps(out_norm_weight, out_norm_eps, rms_eps) -> float:
    if out_norm_weight is None:
        return 0.0
    return float(rms_eps if out_norm_eps is None else out_norm_eps)


def hc_pre(
    *,
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult_value,
    hc_norm_weight,
    out_norm_weight,
    out_norm_eps,
):
    if not supports_mhc_site(
        x,
        hc_mult,
        hc_fn=hc_fn,
        post_mult_value=post_mult_value,
        hc_norm_weight=hc_norm_weight,
        out_norm_weight=out_norm_weight,
    ):
        return None
    pre, _, _ = _ops()
    layer_input, post, comb = pre(
        x,
        hc_fn,
        hc_base,
        hc_scale,
        out_norm_weight,
        hc_mult,
        sinkhorn_iters,
        float(rms_eps),
        float(hc_eps),
        _norm_eps(out_norm_weight, out_norm_eps, rms_eps),
    )
    return layer_input, comb, post, out_norm_weight is not None


def hc_post(*, x, residual, h_post, h_res, hc_mult):
    if not supports_mhc_site(residual, hc_mult, x=x):
        return None
    _, _, post = _ops()
    tokens = x.shape[0]
    return post(
        x,
        residual,
        h_post.reshape(tokens, hc_mult),
        h_res.reshape(tokens, hc_mult * hc_mult),
        hc_mult,
    )


def hc_post_pre(
    *,
    x,
    residual,
    h_post,
    h_res,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult_value,
    out_norm_weight,
    out_norm_eps,
):
    if not supports_mhc_site(
        residual,
        hc_mult,
        x=x,
        hc_fn=hc_fn,
        post_mult_value=post_mult_value,
        out_norm_weight=out_norm_weight,
    ):
        return None
    _, seam, _ = _ops()
    tokens = x.shape[0]
    new_residual, layer_input, post, comb = seam(
        x,
        residual,
        h_post.reshape(tokens, hc_mult),
        h_res.reshape(tokens, hc_mult * hc_mult),
        hc_fn,
        hc_base,
        hc_scale,
        out_norm_weight,
        hc_mult,
        sinkhorn_iters,
        float(rms_eps),
        float(hc_eps),
        _norm_eps(out_norm_weight, out_norm_eps, rms_eps),
    )
    return layer_input, new_residual, comb, post, out_norm_weight is not None
