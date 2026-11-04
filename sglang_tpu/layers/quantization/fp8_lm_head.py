"""Weight-only fp8 LM head for TPU.

An e4m3 copy of the vocab-parallel head with one scale per vocab row, stored K-major and streamed by a
Pallas kernel: bf16 activations, f32 logits. Installed as ``lm_head.quant_method``, so the logits
processor and the DFLASH candidate projection both use it; the bf16 weight stays for other readers.
"""

import threading
from typing import Optional

import torch

# vocab columns per streamed tile: [hidden, 512] fp8 is 4 MiB at hidden 8192; tuned on the
# Muse per-device head, re-tune when the kernel changes
_TN = 512
# vocab rows quantized per step at load, bounding the f32 temporary
_QUANT_ROWS = 4096
_OP = None
_OP_LOCK = threading.Lock()


def _build_op():
    """Register the Pallas op; jax and torch_tpu load here, so importing this module stays cheap."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas

    def logits_kernel(x_ref, w_ref, s_ref, o_ref):
        o_ref[...] = (
            jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
            * s_ref[...]
        )

    def fp8_head_jax(x: jax.Array, w8k: jax.Array, scale: jax.Array) -> jax.Array:
        """``x [M, H]``, ``w8k [H, Vp]`` fp8, ``scale [1, Vp]`` f32 -> f32 ``[M, Vp]``."""
        m, h = x.shape
        vp = w8k.shape[1]
        return pl.pallas_call(
            logits_kernel,
            grid=(vp // _TN,),
            in_specs=[
                pl.BlockSpec((m, h), lambda j: (0, 0)),
                pl.BlockSpec((h, _TN), lambda j: (0, j)),
                pl.BlockSpec((1, _TN), lambda j: (0, j)),
            ],
            out_specs=pl.BlockSpec((m, _TN), lambda j: (0, j)),
            out_shape=jax.ShapeDtypeStruct((m, vp), jnp.float32),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=32 * 2**20
            ),
        )(x.astype(jnp.bfloat16), w8k, scale)

    op = pallas.jax_op("sglang_tpu_pallas::fp8_lm_head", fp8_head_jax)

    def _fake(x: torch.Tensor, w8k: torch.Tensor, scale: torch.Tensor):
        del scale
        return torch.empty(
            x.shape[0], w8k.shape[1], dtype=torch.float32, device=x.device
        )

    op.register_fake(_fake)
    return op


def _get_op():
    global _OP
    with _OP_LOCK:
        if _OP is None:
            _OP = _build_op()
    return _OP


def _quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[V, H]`` -> fp8 K-major ``[H, Vp]`` and f32 ``[1, Vp]``; padded columns have scale 0.

    Built from fresh tensors: a view, a clone of a view or an in-place-built tensor bound into the
    compiled graph is re-materialized by an eager program every step.
    """
    v, h = weight.shape
    vp = -(-v // _TN) * _TN
    wk = weight.detach().t().contiguous().clone()  # [H, V] bf16
    cols, scales = [], []
    for v0 in range(0, v, _QUANT_ROWS):
        w = wk[:, v0 : v0 + _QUANT_ROWS].float()
        s = (w.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
        cols.append((w / s).to(torch.float8_e4m3fn))
        scales.append(s)
    if vp > v:
        cols.append(
            torch.zeros(h, vp - v, dtype=torch.float8_e4m3fn, device=weight.device)
        )
        scales.append(torch.zeros(vp - v, dtype=torch.float32, device=weight.device))
    w8k = torch.cat(cols, dim=1).clone()
    scale = torch.cat([s.reshape(1, -1) for s in scales], dim=1)
    return w8k, scale


class TpuFp8LmHeadMethod:
    """``lm_head.quant_method``: fp8 e4m3 weights with per-row scales, f32 logits."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.vocab_rows = int(layer.weight.shape[0])
        w8k, scale = _quantize_rows(layer.weight)
        # Buffers on the layer so --tpu-compile-tracer direct lifts them as graph inputs.
        layer.register_buffer("fp8_w8k", w8k, persistent=False)
        layer.register_buffer("fp8_scale", scale, persistent=False)
        self._op = _get_op()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        logits = self._op(x2, layer.fp8_w8k, layer.fp8_scale)[:, : self.vocab_rows]
        if bias is not None:
            logits = logits + bias
        return logits.reshape(*x.shape[:-1], self.vocab_rows)
