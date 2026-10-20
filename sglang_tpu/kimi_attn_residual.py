# SPDX-License-Identifier: Apache-2.0
"""Pallas decode kernel for Kimi-K3 Attention Residual aggregation on TPU.

vllm-torchtpu computes only the mixture: a Pallas prefill kernel over 16-token
tiles (kernels/kimi_k3/attention_residual.py) and a folded-fp32 torch path for
decode (models/vllm/kimi_k3/layers.py). This decode kernel also takes the
pending residual add and the output RMSNorm into the one call (the XLA form is
~8 small ops per call), and reads the whole bank with a static count of live
rows.
"""

# PEP 563 must remain disabled so `cpu_export_jax_op` sees `jax.Array` type
# objects rather than stringified annotations at registration time.

import functools
from threading import Lock
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.pallas_jax_op import cpu_export_jax_op

if TYPE_CHECKING:
    from sglang.srt.layers.layernorm import RMSNorm

# Batches up to this many rows take the kernel: it beat the torch path at every
# T measured, up to 24, on a compiled K3-like stack (v7x); the bank bound below
# keeps T <= 16.
_DECODE_MAX_TOKENS = 16
# The whole [T, NB, H] bank is one VMEM block. T = 16 at NB = 9, H = 7168 fits
# the default scoped VMEM; T = 24 at NB = 9 and T = 32 at NB = 8 overflow it.
_DECODE_MAX_BANK_ELEMS = 16 * 9 * 7168


def _decode_kernel(
    *refs, nvb: int, has_addend: bool, eps: float, out_eps: float
) -> None:
    if has_addend:
        a_ref, b_ref, bank_ref, cw_ref, w_ref, out_ref, prefix_ref = refs
        # bf16 add, rounded like the torch `prefix_a + prefix_b` it replaces.
        prefix = (
            a_ref[...].astype(jnp.float32) + b_ref[...].astype(jnp.float32)
        ).astype(a_ref.dtype)
        prefix_ref[...] = prefix
    else:
        a_ref, bank_ref, cw_ref, w_ref, out_ref = refs
        prefix = a_ref[...]
    cw = cw_ref[...]
    values = [bank_ref[:, j, :].astype(jnp.float32) for j in range(nvb)]
    values.append(prefix.astype(jnp.float32))
    # score_j = <v_j, norm_w * proj_w> / rms(v_j): the score RMSNorm and the
    # [H] -> 1 projection folded into one f32 dot.
    scores = [
        jnp.sum(v * cw, axis=-1, keepdims=True)
        * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + eps)
        for v in values
    ]
    m = functools.reduce(jnp.maximum, scores)
    exps = [jnp.exp(s - m) for s in scores]
    total = functools.reduce(jnp.add, exps)
    mixed = functools.reduce(jnp.add, [e * v for e, v in zip(exps, values)]) / total
    # Output RMSNorm on the bf16 mixture (RMSNorm.forward_native semantics).
    x = mixed.astype(out_ref.dtype).astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + out_eps)
    out_ref[...] = (x * w_ref[...].astype(jnp.float32)).astype(out_ref.dtype)


def attention_residual_decode_jax(
    prefix_a: jax.Array,
    prefix_b: jax.Array | None,
    bank: jax.Array,
    folded_weight: jax.Array,
    norm_weight: jax.Array,
    nvb: int,
    eps: float,
    out_eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Fused decode aggregation: optional pending add, score, softmax, mix and
    output RMSNorm in one kernel.

    prefix_a and the optional pending addend prefix_b are bf16 [tokens, hidden];
    bank is [tokens, num_blocks, hidden] with rows [:nvb] live; folded_weight is
    the [hidden] score weight norm.weight * proj.weight. Returns (normed, prefix)
    with prefix = prefix_a + prefix_b (prefix_a without an addend).
    """
    tokens, hidden = prefix_a.shape
    has_addend = prefix_b is not None
    row_spec = pl.BlockSpec((tokens, hidden), lambda i: (0, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    in_specs = [row_spec] + [row_spec] * has_addend
    in_specs += [pl.BlockSpec(bank.shape, lambda i: (0, 0, 0)), vec_spec, vec_spec]
    row_shape = jax.ShapeDtypeStruct((tokens, hidden), prefix_a.dtype)
    outs = pl.pallas_call(
        functools.partial(
            _decode_kernel,
            nvb=nvb,
            has_addend=has_addend,
            eps=eps,
            out_eps=out_eps,
        ),
        out_shape=[row_shape] * (1 + has_addend),
        grid=(1,),
        in_specs=in_specs,
        out_specs=[row_spec] * (1 + has_addend),
        # Default scoped VMEM: it fits _DECODE_MAX_TOKENS rows at K3's
        # NB = 8, H = 7168, and a raised limit (63 MiB) made the compiled call
        # slower than the torch path from T = 4.
        # Let XLA fold the producers of the activations (all-reduce relayouts,
        # the residual add, the bank view) into the kernel's input copies.
        compiler_params=pltpu.CompilerParams(
            allow_input_fusion=[True] * (2 + has_addend) + [False, False],
        ),
        name=f"kimi_attn_res_decode_nvb{nvb}",
    )(
        *([prefix_a] + [prefix_b] * has_addend),
        bank,
        folded_weight.reshape(1, hidden).astype(jnp.float32),
        norm_weight.reshape(1, hidden),
    )
    return outs[0], outs[1] if has_addend else prefix_a


_DECODE_OPS: tuple[Callable, Callable] | None = None
_DECODE_OPS_LOCK = Lock()


def register_kimi_attn_res_decode_ops() -> None:
    """Register the decode ops (without / with the pending addend) before
    torch.compile traces the model: building one inside the dynamo trace fails
    in inspect.signature. `nvb` and the epsilons are static op args."""
    global _DECODE_OPS
    with _DECODE_OPS_LOCK:
        if _DECODE_OPS is not None:
            return

        def _decode(
            prefix_a: jax.Array,
            bank: jax.Array,
            folded_weight: jax.Array,
            norm_weight: jax.Array,
            nvb: int,
            eps: float,
            out_eps: float,
        ) -> jax.Array:
            return attention_residual_decode_jax(
                prefix_a, None, bank, folded_weight, norm_weight, nvb, eps, out_eps
            )[0]

        def _decode_add(
            prefix_a: jax.Array,
            prefix_b: jax.Array,
            bank: jax.Array,
            folded_weight: jax.Array,
            norm_weight: jax.Array,
            nvb: int,
            eps: float,
            out_eps: float,
        ) -> tuple[jax.Array, jax.Array]:
            return attention_residual_decode_jax(
                prefix_a, prefix_b, bank, folded_weight, norm_weight, nvb, eps, out_eps
            )

        decode = cpu_export_jax_op("sglang_tpu_pallas::kimi_attn_res_decode", _decode)
        decode.register_fake(lambda prefix_a, *_: torch.empty_like(prefix_a))
        decode_add = cpu_export_jax_op(
            "sglang_tpu_pallas::kimi_attn_res_decode_add", _decode_add
        )
        decode_add.register_fake(
            lambda prefix_a, *_: (
                torch.empty_like(prefix_a),
                torch.empty_like(prefix_a),
            )
        )
        _DECODE_OPS = (decode, decode_add)


def attention_residual_decode_tpu(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor | None,
    bank: torch.Tensor,
    nvb: int,
    cw: torch.Tensor,
    eps: float,
    norm_weight: torch.Tensor,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TPU entry for the fused decode kernel. Returns (normed, prefix)."""
    assert _DECODE_OPS is not None, (
        "register_kimi_attn_res_decode_ops() must run at model init"
    )
    decode, decode_add = _DECODE_OPS
    if prefix_b is None:
        return decode(prefix_a, bank, cw, norm_weight, nvb, eps, out_eps), prefix_a
    return decode_add(prefix_a, prefix_b, bank, cw, norm_weight, nvb, eps, out_eps)


def _default_rms_norm(norm: "RMSNorm") -> bool:
    # The kernel implements RMSNorm.forward_native for the default config.
    return (
        not norm.cast_x_before_out_mul
        and norm.override_orig_dtype is None
        and norm.variance_size_override is None
    )


def supports_attn_res_decode(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor | None,
    bank: torch.Tensor,
    score_norm: "RMSNorm",
    out_norm: "RMSNorm",
) -> bool:
    """Shape gate: decode-sized bf16 batches of 128-lane rows whose bank fits
    one VMEM block, a pending addend that matches the prefix, and
    default-config RMSNorms."""
    return (
        0 < prefix_a.shape[0] <= _DECODE_MAX_TOKENS
        and bank.numel() <= _DECODE_MAX_BANK_ELEMS
        and prefix_a.shape[1] % 128 == 0
        and prefix_a.dtype == torch.bfloat16
        and (
            prefix_b is None
            or (prefix_b.dtype == prefix_a.dtype and prefix_b.shape == prefix_a.shape)
        )
        and _default_rms_norm(score_norm)
        and _default_rms_norm(out_norm)
    )


def get_norm_weight_f32(norm: "RMSNorm") -> torch.Tensor:
    """Cached fp32 copy of `norm.weight`, which the kernel upcasts anyway: an
    fp32 [H] vector reaches it as a free bitcast, a bf16 one costs a [1, H]
    relayout per call. Built at post-load, like attn_residual.get_cw."""
    w = getattr(norm, "_attn_res_weight_f32", None)
    if w is None:
        if torch.compiler.is_compiling():
            # Built inside the trace, the copy would be a graph constant guarded
            # on its absence: the next call would recompile.
            raise RuntimeError(
                "The fp32 out-norm weight must be built before tracing "
                "(post_load_weights)."
            )
        w = norm._attn_res_weight_f32 = norm.weight.detach().float().contiguous()
    return w
