# SPDX-License-Identifier: Apache-2.0
"""Pallas kernels for Kimi-K3 Attention Residual aggregation on TPU.

vllm-torchtpu computes only the mixture: a Pallas prefill kernel over 16-token
tiles (kernels/kimi_k3/attention_residual.py) and a folded-fp32 torch path for
decode (models/vllm/kimi_k3/layers.py). This decode kernel also takes the
pending residual add and the output RMSNorm into the one call (the XLA form is
~8 small ops per call), and reads the whole bank with a static count of live
rows. The prefill kernel runs the same per-token math over a grid of token
tiles, in place of the torch reference that materializes [T, nvb + 1, H] and
makes several passes over it at every aggregation point. It needs the bank
block-major ([NB, T, H] storage behind the [T, NB, H] view, set up by the bank
hook): token-major, a block's rows for a token tile are a sublane gather that
costs several times the math (compile-only tpu7x schedules: ~3.3M vs ~0.58M
cycles per 4096-token aggregation at nvb = 8).
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


def decode_math(
    prefix, bank_ref, cw, w, *, nvb: int, eps: float, out_eps: float, out_dtype
):
    """The kernel body on loaded values: ``prefix`` bf16 [T, H] (pending add
    already applied), ``bank_ref`` the [T, NB, H] bank ref, ``cw`` / ``w`` f32
    [1, H]. Returns the normed mixture in ``out_dtype``. Shared with the
    two-level all-reduce kernel that fuses this aggregation (same ops, same
    order)."""
    values, scores = bank_scores(bank_ref, cw, nvb=nvb, eps=eps)
    return decode_finish(
        prefix, values, scores, cw, w, eps=eps, out_eps=out_eps, out_dtype=out_dtype
    )


def _score(v, cw, eps):
    # score_j = <v_j, norm_w * proj_w> / rms(v_j): the score RMSNorm and the
    # [H] -> 1 projection folded into one f32 dot.
    return jnp.sum(v * cw, axis=-1, keepdims=True) * jax.lax.rsqrt(
        jnp.mean(v * v, axis=-1, keepdims=True) + eps
    )


def bank_scores(bank_ref, cw, *, nvb: int, eps: float):
    """The part of decode_math that does not depend on the prefix: the f32
    bank rows and their scores (a kernel may run it before the prefix is
    ready)."""
    values = [bank_ref[:, j, :].astype(jnp.float32) for j in range(nvb)]
    return values, [_score(v, cw, eps) for v in values]


def decode_finish(prefix, values, scores, cw, w, *, eps, out_eps, out_dtype):
    """The rest of decode_math once ``prefix`` is known."""
    values = values + [prefix.astype(jnp.float32)]
    scores = scores + [_score(values[-1], cw, eps)]
    m = functools.reduce(jnp.maximum, scores)
    exps = [jnp.exp(s - m) for s in scores]
    total = functools.reduce(jnp.add, exps)
    mixed = functools.reduce(jnp.add, [e * v for e, v in zip(exps, values)]) / total
    return _out_norm(mixed, w, out_eps, out_dtype)


def _out_norm(mixed, w, out_eps, out_dtype):
    # Output RMSNorm on the bf16 mixture (RMSNorm.forward_native semantics).
    x = mixed.astype(out_dtype).astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + out_eps)
    return (x * w.astype(jnp.float32)).astype(out_dtype)


def add_prefix(a, b):
    """bf16 add, rounded like the torch ``prefix_a + prefix_b`` it replaces."""
    return (a.astype(jnp.float32) + b.astype(jnp.float32)).astype(a.dtype)


def _decode_kernel(
    *refs, nvb: int, has_addend: bool, eps: float, out_eps: float
) -> None:
    if has_addend:
        a_ref, b_ref, bank_ref, cw_ref, w_ref, out_ref, prefix_ref = refs
        prefix = add_prefix(a_ref[...], b_ref[...])
        prefix_ref[...] = prefix
    else:
        a_ref, bank_ref, cw_ref, w_ref, out_ref = refs
        prefix = a_ref[...]
    out_ref[...] = decode_math(
        prefix,
        bank_ref,
        cw_ref[...],
        w_ref[...],
        nvb=nvb,
        eps=eps,
        out_eps=out_eps,
        out_dtype=out_ref.dtype,
    )


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


# Prefill: tokens per grid step. A step double-buffers the tile's live bank rows
# ([nvb, tile, H], 3.7 MiB at 8 x 32 x 7168), its prefix rows and its outputs.
_PREFILL_TILE = 32
# Above every compiled decode / verify batch (up to 64 rows): decode keeps its
# own kernel or the torch path, while the 1024 / 4096 prefill buckets qualify.
_PREFILL_MIN_TOKENS = 256
_PREFILL_VMEM_LIMIT = 48 * 1024 * 1024


def prefill_tokens_supported(tokens: int) -> bool:
    """Batches the prefill kernel takes: at least _PREFILL_MIN_TOKENS tokens,
    in whole _PREFILL_TILE-token tiles."""
    return tokens >= _PREFILL_MIN_TOKENS and tokens % _PREFILL_TILE == 0


def _prefill_vmem_limit(nvb: int, hidden: int, has_addend: bool) -> int:
    from sglang_tpu.kernels.vmem_limit import padded_bytes, scoped_limit

    tile = _PREFILL_TILE
    rows = padded_bytes((tile, hidden), 2)
    need = (
        2 * padded_bytes((nvb, tile, hidden), 2)  # bank block, double-buffered
        + 2 * 2 * (1 + has_addend) * rows  # prefix rows in, outputs
        + 2 * 2 * padded_bytes((1, hidden), 4)  # cw, out-norm weight
        + 4 * padded_bytes((tile, hidden), 4)  # f32 prefix, row, mix, norm
    )
    return scoped_limit(_PREFILL_VMEM_LIMIT, need)


def _prefill_kernel(
    *refs, nvb: int, has_addend: bool, eps: float, out_eps: float
) -> None:
    if has_addend:
        a_ref, b_ref, bank_ref, cw_ref, w_ref, out_ref, prefix_ref = refs
        prefix = add_prefix(a_ref[...], b_ref[...])
        prefix_ref[...] = prefix
    else:
        a_ref, bank_ref, cw_ref, w_ref, out_ref = refs
        prefix = a_ref[...]
    cw = cw_ref[...]
    # decode_math's ops in its order, streamed: a bank row is converted to f32
    # when it is scored and again when it is mixed, so the tile's rows are not
    # all held in f32 at once. bank_ref[j] is block j's dense [tile, H] slab.
    p = prefix.astype(jnp.float32)
    scores = [_score(bank_ref[j].astype(jnp.float32), cw, eps) for j in range(nvb)]
    scores.append(_score(p, cw, eps))
    m = functools.reduce(jnp.maximum, scores)
    exps = [jnp.exp(s - m) for s in scores]
    total = functools.reduce(jnp.add, exps)
    acc = exps[0] * bank_ref[0].astype(jnp.float32)
    for j in range(1, nvb):
        acc = acc + exps[j] * bank_ref[j].astype(jnp.float32)
    acc = acc + exps[nvb] * p
    out_ref[...] = _out_norm(acc / total, w_ref[...], out_eps, out_ref.dtype)


def attention_residual_prefill_jax(
    prefix_a: jax.Array,
    prefix_b: jax.Array | None,
    bank: jax.Array,
    folded_weight: jax.Array,
    norm_weight: jax.Array,
    nvb: int,
    eps: float,
    out_eps: float,
) -> tuple[jax.Array, jax.Array]:
    """attention_residual_decode_jax for prefill-sized batches, with the bank
    block-major: [num_blocks, tokens, hidden], rows [:nvb] live (1 <= nvb).
    The same per-token math over a grid of _PREFILL_TILE-token tiles
    (prefill_tokens_supported); each step reads only the live rows."""
    tokens, hidden = prefix_a.shape
    tile = _PREFILL_TILE
    has_addend = prefix_b is not None
    row_spec = pl.BlockSpec((tile, hidden), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    bank_spec = pl.BlockSpec((nvb, tile, hidden), lambda i: (0, i, 0))
    in_specs = [row_spec] * (1 + has_addend) + [bank_spec, vec_spec, vec_spec]
    row_shape = jax.ShapeDtypeStruct((tokens, hidden), prefix_a.dtype)
    outs = pl.pallas_call(
        functools.partial(
            _prefill_kernel,
            nvb=nvb,
            has_addend=has_addend,
            eps=eps,
            out_eps=out_eps,
        ),
        out_shape=[row_shape] * (1 + has_addend),
        grid=(tokens // tile,),
        in_specs=in_specs,
        out_specs=[row_spec] * (1 + has_addend),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            # Not the bank: a layout mismatch then shows up as a separate copy
            # instead of a transpose folded into the kernel's input.
            allow_input_fusion=[True] * (1 + has_addend) + [False] * 3,
            vmem_limit_bytes=_prefill_vmem_limit(nvb, hidden, has_addend),
        ),
        name=f"kimi_attn_res_prefill_nvb{nvb}",
    )(
        *([prefix_a] + [prefix_b] * has_addend),
        bank,
        folded_weight.reshape(1, hidden).astype(jnp.float32),
        norm_weight.reshape(1, hidden),
    )
    return outs[0], outs[1] if has_addend else prefix_a


_DECODE_OPS: tuple[Callable, Callable] | None = None
_PREFILL_OPS: tuple[Callable, Callable] | None = None
_DECODE_OPS_LOCK = Lock()


def _export_ops(name: str, jax_fn: Callable) -> tuple[Callable, Callable]:
    """One aggregation's torch ops, without / with the pending addend."""

    def _plain(
        prefix_a: jax.Array,
        bank: jax.Array,
        folded_weight: jax.Array,
        norm_weight: jax.Array,
        nvb: int,
        eps: float,
        out_eps: float,
    ) -> jax.Array:
        return jax_fn(
            prefix_a, None, bank, folded_weight, norm_weight, nvb, eps, out_eps
        )[0]

    def _add(
        prefix_a: jax.Array,
        prefix_b: jax.Array,
        bank: jax.Array,
        folded_weight: jax.Array,
        norm_weight: jax.Array,
        nvb: int,
        eps: float,
        out_eps: float,
    ) -> tuple[jax.Array, jax.Array]:
        return jax_fn(
            prefix_a, prefix_b, bank, folded_weight, norm_weight, nvb, eps, out_eps
        )

    plain = cpu_export_jax_op(f"sglang_tpu_pallas::{name}", _plain)
    plain.register_fake(lambda prefix_a, *_: torch.empty_like(prefix_a))
    add = cpu_export_jax_op(f"sglang_tpu_pallas::{name}_add", _add)
    add.register_fake(
        lambda prefix_a, *_: (torch.empty_like(prefix_a), torch.empty_like(prefix_a))
    )
    return plain, add


def register_kimi_attn_res_decode_ops(prefill: bool = False) -> None:
    """Register the decode ops (and with ``prefill`` the prefill ops) before
    torch.compile traces the model: building one inside the dynamo trace fails
    in inspect.signature. `nvb` and the epsilons are static op args."""
    global _DECODE_OPS, _PREFILL_OPS
    with _DECODE_OPS_LOCK:
        if _DECODE_OPS is None:
            _DECODE_OPS = _export_ops(
                "kimi_attn_res_decode", attention_residual_decode_jax
            )
        if prefill and _PREFILL_OPS is None:
            _PREFILL_OPS = _export_ops(
                "kimi_attn_res_prefill", attention_residual_prefill_jax
            )


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
    return _run(
        _DECODE_OPS, prefix_a, prefix_b, bank, nvb, cw, eps, norm_weight, out_eps
    )


def attention_residual_prefill_tpu(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor | None,
    bank: torch.Tensor,
    nvb: int,
    cw: torch.Tensor,
    eps: float,
    norm_weight: torch.Tensor,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TPU entry for the prefill kernel. ``bank`` is the [tokens, NB, hidden]
    view of block-major storage (supports_attn_res_prefill); the kernel reads
    the storage. Returns (normed, prefix)."""
    assert _PREFILL_OPS is not None, (
        "register_kimi_attn_res_decode_ops(prefill=True) must run at model init"
    )
    storage = bank.permute(1, 0, 2)
    return _run(
        _PREFILL_OPS, prefix_a, prefix_b, storage, nvb, cw, eps, norm_weight, out_eps
    )


def _run(ops, prefix_a, prefix_b, bank, nvb, cw, eps, norm_weight, out_eps):
    plain, add = ops
    if prefix_b is None:
        return plain(prefix_a, bank, cw, norm_weight, nvb, eps, out_eps), prefix_a
    return add(prefix_a, prefix_b, bank, cw, norm_weight, nvb, eps, out_eps)


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
        and _common_gate(prefix_a, prefix_b, score_norm, out_norm)
    )


def supports_attn_res_prefill(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor | None,
    bank: torch.Tensor,
    score_norm: "RMSNorm",
    out_norm: "RMSNorm",
) -> bool:
    """Shape gate for the prefill kernel: prefill_tokens_supported batches
    whose [tokens, NB, hidden] bank is stored block-major (the bank hook's
    layout), and the decode gate's row, addend and RMSNorm conditions."""
    return (
        prefill_tokens_supported(prefix_a.shape[0])
        and bank.dim() == 3
        and bank.shape[0] == prefix_a.shape[0]
        and bank.shape[2] == prefix_a.shape[1]
        and bank.dtype == torch.bfloat16
        and bank.permute(1, 0, 2).is_contiguous()
        and _common_gate(prefix_a, prefix_b, score_norm, out_norm)
    )


def _common_gate(prefix_a, prefix_b, score_norm, out_norm) -> bool:
    return (
        prefix_a.shape[1] % 128 == 0
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
