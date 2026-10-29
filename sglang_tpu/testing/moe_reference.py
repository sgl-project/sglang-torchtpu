"""Dense pure-JAX references for the TPU MoE kernels (``fused_moe_op``, ``gmm_v2``).

Everything here runs in float32 at HIGHEST matmul precision, with no tiling,
permutation, or padding, so a kernel test's diff against it can only be the
kernel's own arithmetic. Weights are dequantized from the stored formats
before the matmul; activation quantization inside the kernels is not modeled.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

_HIGHEST = jax.lax.Precision.HIGHEST

# float4_e2m1fn magnitudes indexed by the low 3 code bits; bit 3 is the sign.
_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)

Activation = Union[str, Callable[[jax.Array, jax.Array], jax.Array]]


def e8m0_to_f32(scale_u8) -> jax.Array:
    """MX e8m0 scale byte -> 2 ** (byte - 127)."""
    return jnp.exp2(jnp.asarray(scale_u8).astype(jnp.float32) - 127.0)


def unpack_fp4(packed_u8, axis: int) -> jax.Array:
    """uint8 holding two e2m1 codes -> float32 values, doubling ``axis``.

    The low nibble is the even element along ``axis`` (HF MXFP4 order,
    mxfp4_tensor.py fuse_uint4_to_uint8; gmm_v2 unpacks the same way).
    """
    packed = jnp.asarray(packed_u8).astype(jnp.uint8)
    axis = axis % packed.ndim
    codes = jnp.stack([packed & 0xF, packed >> 4], axis=axis + 1)
    shape = list(packed.shape)
    shape[axis] *= 2
    codes = codes.reshape(shape)
    magnitude = jnp.asarray(_E2M1_VALUES)[(codes & 0x7).astype(jnp.int32)]
    return jnp.where((codes & 0x8) != 0, -magnitude, magnitude)


def unpack_int4(words, axis: int, *, offset_binary: bool = False) -> jax.Array:
    """int32 words of eight int4 values -> float32 values, 8x along ``axis``.

    Nibbles are LSB-first along ``axis`` (w4a16_moe.py module docstring).
    ``offset_binary`` reads the checkpoint's ``q + 8`` encoding; otherwise the
    nibbles are two's complement, which is what the kernel consumes after
    prepare_tpu_w4a16_moe_weights flips each nibble's top bit.
    """
    words = jnp.asarray(words)
    assert words.dtype in (jnp.int32, jnp.uint32), words.dtype
    words = jax.lax.bitcast_convert_type(words, jnp.uint32)
    axis = axis % words.ndim
    shifts = jnp.arange(8, dtype=jnp.uint32) * 4
    expand = [1] * (words.ndim + 1)
    expand[axis + 1] = 8
    nibbles = (jnp.expand_dims(words, axis + 1) >> shifts.reshape(expand)) & 0xF
    values = nibbles.astype(jnp.int32)
    values = (
        values - 8 if offset_binary else jnp.where(values >= 8, values - 16, values)
    )
    shape = list(words.shape)
    shape[axis] *= 8
    return values.reshape(shape).astype(jnp.float32)


def apply_gmm_scale(values, scale) -> jax.Array:
    """Dequantize a gmm_v2 rhs: values [G, K, N] times scale [G, nb, 1, ns].

    Each of the ``nb`` scale rows covers K // nb contiguous K rows and each
    scale column covers N // ns contiguous output columns (gmm_v2.py
    WeightsRef.get_scale repeats a scale over scale_block_n columns).
    """
    values = jnp.asarray(values).astype(jnp.float32)
    scale = jnp.asarray(scale).astype(jnp.float32)
    _, k, n = values.shape
    _, num_blocks, one, scale_n = scale.shape
    assert one == 1 and k % num_blocks == 0 and n % scale_n == 0
    expanded = jnp.repeat(scale[:, :, 0, :], k // num_blocks, axis=1)
    expanded = jnp.repeat(expanded, n // scale_n, axis=2)
    return values * expanded


def dequant_mxfp4_rows(packed_u8, scale_e8m0) -> jax.Array:
    """Checkpoint MXFP4 [..., K // 2] uint8 + [..., K // 32] e8m0 -> [..., K]."""
    values = unpack_fp4(packed_u8, axis=-1)
    block = values.shape[-1] // scale_e8m0.shape[-1]
    return values * jnp.repeat(e8m0_to_f32(scale_e8m0), block, axis=-1)


def swiglu_oai(gate, up, *, alpha: float = 1.702, limit: float = 7.0) -> jax.Array:
    """Clamped SwiGLU with ``up + 1``; the defaults are the released alpha and limit."""
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
    return gate * jax.nn.sigmoid(alpha * gate) * (up + 1.0)


def gated_activation(gate, up, activation: Activation) -> jax.Array:
    """``act(gate) * up`` for every activation name the TPU MoE kernels accept."""
    if callable(activation):
        return activation(gate, up)
    if activation == "silu":
        return jax.nn.silu(gate) * up
    if activation == "gelu":
        # gmm_v2.apply_act_fn calls jax.nn.gelu with its default tanh approximation.
        return jax.nn.gelu(gate, approximate=True) * up
    if activation in ("swiglu_oai", "swigluoai"):
        return swiglu_oai(gate, up)
    if activation.startswith("situ:"):
        # "situ:{beta}:{linear_beta|none}", built by fused_moe_call.
        _, beta, linear_beta = activation.split(":")
        beta = float(beta)
        if linear_beta != "none":
            up = float(linear_beta) * jnp.tanh(up / float(linear_beta))
        return beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate) * up
    if activation.startswith("silu_and_mul_clamp:"):
        # Gate is clamped from above only, up on both sides (fused_moe_op.py).
        limit = float(activation.split(":", 1)[1])
        return jax.nn.silu(jnp.minimum(gate, limit)) * jnp.clip(up, -limit, limit)
    raise ValueError(f"unsupported activation: {activation}")


def _bias_rows(bias, ids) -> jax.Array:
    # bias [G, n] or [G, 1, n] -> rows picked by ids.
    bias = jnp.asarray(bias).astype(jnp.float32)
    return bias.reshape(bias.shape[0], -1)[ids]


def moe_reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    activation: Activation,
    w1_bias=None,
    w2_bias=None,
) -> jax.Array:
    """Dense MoE in float32: sum_j w_j * (act(x @ w1[e_j] + b1) @ w2[e_j] + b2).

    ``w1`` is [E, K, 2N] with gate in the first N columns and up in the last N;
    ``w2`` is [E, N, H_out]; both already dequantized. Duplicate expert ids in a
    row are summed like distinct slots.
    """
    x = jnp.asarray(hidden_states).astype(jnp.float32)
    w1 = jnp.asarray(w1).astype(jnp.float32)
    w2 = jnp.asarray(w2).astype(jnp.float32)
    weights = jnp.asarray(topk_weights).astype(jnp.float32)
    ids = jnp.asarray(topk_ids).astype(jnp.int32)
    num_experts, k, two_n = w1.shape
    assert x.shape[1] == k and w2.shape[:2] == (num_experts, two_n // 2)

    # Every expert on every token, then pick the routed ones: [m, topk, 2N].
    all_gmm1 = jnp.einsum("mk,ekn->men", x, w1, precision=_HIGHEST)
    gmm1 = jnp.take_along_axis(all_gmm1, ids[:, :, None], axis=1)
    if w1_bias is not None:
        gmm1 = gmm1 + _bias_rows(w1_bias, ids)
    n = two_n // 2
    act = gated_activation(gmm1[..., :n], gmm1[..., n:], activation)

    # Fold the routing weight in before GMM2: [m, E, N] per-expert inputs.
    onehot = jax.nn.one_hot(ids, num_experts, dtype=jnp.float32)
    per_expert = jnp.einsum("mje,mj,mjn->men", onehot, weights, act, precision=_HIGHEST)
    out = jnp.einsum("men,enh->mh", per_expert, w2, precision=_HIGHEST)
    if w2_bias is not None:
        out = out + jnp.einsum(
            "mj,mjh->mh", weights, _bias_rows(w2_bias, ids), precision=_HIGHEST
        )
    return out


def gmm_reference(
    lhs,
    rhs,
    group_sizes,
    *,
    rhs_bias=None,
    group_offset: int = 0,
    fuse_act: Optional[Activation] = None,
) -> jax.Array:
    """Grouped matmul in float32 matching gmm_v2's row/group contract.

    Rows of global group g use local weight ``g - group_offset``; rows of groups
    outside the local window and rows past ``sum(group_sizes)`` are zero.
    With ``fuse_act`` the [.., 2N] product is reduced to act(gate) * up.
    """
    lhs = jnp.asarray(lhs).astype(jnp.float32)
    rhs = jnp.asarray(rhs).astype(jnp.float32)
    sizes = np.asarray(group_sizes)
    num_local = rhs.shape[0]
    out_n = rhs.shape[2] // 2 if fuse_act is not None else rhs.shape[2]
    out = jnp.zeros((lhs.shape[0], out_n), jnp.float32)
    start = 0
    for g, size in enumerate(sizes.tolist()):
        local = g - group_offset
        if size and 0 <= local < num_local:
            acc = jnp.matmul(lhs[start : start + size], rhs[local], precision=_HIGHEST)
            if rhs_bias is not None:
                acc = acc + _bias_rows(rhs_bias, local)
            if fuse_act is not None:
                half = acc.shape[-1] // 2
                acc = gated_activation(acc[:, :half], acc[:, half:], fuse_act)
            out = out.at[start : start + size].set(acc)
        start += size
    return out


def max_rel_error(actual, expected) -> float:
    """max |actual - expected| normalized by max |expected| (the MoE tests' metric)."""
    actual = np.asarray(jnp.asarray(actual).astype(jnp.float32))
    expected = np.asarray(jnp.asarray(expected).astype(jnp.float32))
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.all(np.isfinite(actual)), "non-finite kernel output"
    return float(
        np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-12)
    )
