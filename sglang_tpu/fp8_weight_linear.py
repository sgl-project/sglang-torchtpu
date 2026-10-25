# SPDX-License-Identifier: Apache-2.0
"""Weight-only fp8 linear layers for TPU decode.

``install_fp8_linears(module)`` swaps the quant method of every unquantized sglang
linear under ``module`` for :class:`TpuFp8LinearMethod`: an e4m3 copy of the weight
with one scale per output column, stored K-major and streamed by a Pallas kernel
(bf16 activations, f32 accumulation). Calls with more than ``_MAX_ROWS`` rows
(prefill) keep the bf16 weight, which stays in place.
"""

import threading
from typing import Optional

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

# Decode-sized calls only: the kernel holds all rows of x in VMEM.
_MAX_ROWS = 64
# Quantize this many output columns at a time to bound the f32 temporaries.
_QUANT_COLS = 4096
_OP = None
_OP_LOCK = threading.Lock()


def _tile_n(n: int) -> int:
    """Output columns per streamed tile: the largest of 512 / 256 / 128 dividing ``n``."""
    for tn in (512, 256, 128):
        if n % tn == 0:
            return tn
    return 512


def _matmul_kernel(x_ref, w_ref, s_ref, o_ref):
    o_ref[...] = (
        jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32) * s_ref[...]
    )


def _fp8_matmul_jax(x: jax.Array, w8k: jax.Array, scale: jax.Array) -> jax.Array:
    """``x [M, K]``, ``w8k [K, Np]`` fp8, ``scale [1, Np]`` f32 -> f32 ``[M, Np]``."""
    m, k = x.shape
    np_ = w8k.shape[1]
    tn = _tile_n(np_)
    return pl.pallas_call(
        _matmul_kernel,
        grid=(np_ // tn,),
        in_specs=[
            pl.BlockSpec((m, k), lambda j: (0, 0)),
            pl.BlockSpec((k, tn), lambda j: (0, j)),
            pl.BlockSpec((1, tn), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((m, tn), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((m, np_), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=32 * 2**20
        ),
    )(x.astype(jnp.bfloat16), w8k, scale)


def get_fp8_matmul_op():
    """The registered torch op; build it at load time, never inside a traced forward."""
    global _OP
    with _OP_LOCK:
        if _OP is None:
            op = pallas.jax_op("sglang_tpu_pallas::fp8_weight_matmul", _fp8_matmul_jax)

            def _fake(x: torch.Tensor, w8k: torch.Tensor, scale: torch.Tensor):
                del scale
                return torch.empty(
                    x.shape[0], w8k.shape[1], dtype=torch.float32, device=x.device
                )

            op.register_fake(_fake)
            _OP = op
    return _OP


def quantize_k_major(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch ``[N, K]`` -> fp8 K-major ``[K, Np]`` and f32 ``[1, Np]``; padded columns scale 0.

    Built from fresh tensors: a view, a clone of a view or an in-place-built
    tensor bound into the compiled graph is re-materialized every step.
    """
    n, k = weight.shape
    tn = _tile_n(n)
    np_ = -(-n // tn) * tn
    wk = weight.detach().t().contiguous()  # [K, N]
    cols, scales = [], []
    for n0 in range(0, n, _QUANT_COLS):
        w = wk[:, n0 : n0 + _QUANT_COLS].float()
        s = (w.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
        cols.append((w / s).to(torch.float8_e4m3fn))
        scales.append(s)
    if np_ > n:
        cols.append(
            torch.zeros(k, np_ - n, dtype=torch.float8_e4m3fn, device=weight.device)
        )
        scales.append(torch.zeros(np_ - n, dtype=torch.float32, device=weight.device))
    w8k = torch.cat(cols, dim=1)
    scale = torch.cat([s.reshape(1, -1) for s in scales], dim=1)
    return w8k, scale


class TpuFp8LinearMethod:
    """Weight-only fp8 ``quant_method`` for one sglang linear layer."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.out_features = int(layer.weight.shape[0])
        w8k, scale = quantize_k_major(layer.weight)
        # Buffers on the layer, so the direct tracer lifts them as graph inputs.
        layer.register_buffer("fp8_w8k", w8k, persistent=False)
        layer.register_buffer("fp8_scale", scale, persistent=False)
        self._op = get_fp8_matmul_op()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] > _MAX_ROWS:
            return F.linear(x, layer.weight, bias)
        out = self._op(x2, layer.fp8_w8k, layer.fp8_scale)[:, : self.out_features]
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], self.out_features)


class TpuSharedKMajorLinearMethod:
    """``quant_method`` reading one layer's columns of a stacked K-major weight another module
    owns (the Muse megakernel's dense copies), so the linear needs no weight of its own.

    ``stack`` is ``[L, K, N]``; an fp8 stack is dequantized by per-output-column scales at
    ``scales[index, scale_row, :N]``. Every call runs an XLA dot, at any row count.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        stack: torch.Tensor,
        index: int,
        col_start: int,
        col_stop: int,
        scales: Optional[torch.Tensor] = None,
        scale_row: int = 0,
    ) -> None:
        if (stack.dtype == torch.float8_e4m3fn) != (scales is not None):
            raise ValueError("an fp8 stack needs its scales, a bf16 stack none")
        # The whole stacks, not per-layer views: a view bound into the compiled graph is
        # re-materialized every step; aliases of one tensor share one graph input.
        layer.register_buffer("shared_kmajor", stack, persistent=False)
        if scales is not None:
            layer.register_buffer("shared_kmajor_scales", scales, persistent=False)
        self._index = index
        self._cols = (col_start, col_stop)
        self._scale_row = scale_row
        self._scaled = scales is not None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c0, c1 = self._cols
        w = layer.shared_kmajor[self._index, :, c0:c1]
        out = torch.matmul(x, w.to(x.dtype))
        if self._scaled:
            scale = layer.shared_kmajor_scales[self._index, self._scale_row, c0:c1]
            out = (out.float() * scale).to(x.dtype)
        if bias is not None:
            out = out + bias
        return out


def install_fp8_linears(module: torch.nn.Module) -> int:
    """Swap every unquantized linear under ``module`` to fp8 weight-only; returns the count."""
    count = 0
    for sub in module.modules():
        if isinstance(sub, LinearBase) and isinstance(
            sub.quant_method, UnquantizedLinearMethod
        ):
            sub.quant_method = TpuFp8LinearMethod(sub)
            count += 1
    return count


class _Fp8PlainLinear(torch.nn.Module):
    """A plain ``nn.Linear`` whose decode-sized calls run through the fp8 weight-only op."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self._method = TpuFp8LinearMethod(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._method.apply(self, x, self.bias)


def install_fp8_plain_linears(module: torch.nn.Module, names: tuple[str, ...]) -> int:
    """Swap every ``nn.Linear`` child named in ``names`` under ``module``; returns the count."""
    swaps = [
        (parent, name, child)
        for parent in module.modules()
        for name, child in parent.named_children()
        if name in names and type(child) is torch.nn.Linear
    ]
    for parent, name, child in swaps:
        setattr(parent, name, _Fp8PlainLinear(child))
    return len(swaps)
