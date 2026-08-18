# SPDX-License-Identifier: Apache-2.0
# Binary-search top-k/top-p/min-p filtering, adapted from
# https://github.com/vllm-project/tpu-inference
# tpu_inference/layers/common/binary_search.py (itself from T5X).
# Copyright 2024 The T5X Authors. Apache-2.0.
"""Sort-free distribution filtering for TPU sampling.

The torch fallback (`top_k_top_p_min_p_sampling_from_probs_torch`) needs a
full-vocabulary sort plus boolean-mask scatters, which run eagerly on TPU at
~415 ms per decode step. This module finds the same cutoffs by a 31-step
binary search over float32 bit space — only dense element-wise ops and
reductions, exported as ONE compiled XLA program via `cpu_export_jax_op`.

Semantics match the torch reference up to ties: the cutoff keeps every value
tied with the boundary, where a sort breaks ties arbitrarily. min-p uses the
global max probability as its reference, same as the torch path.
"""

# NOTE: no `from __future__ import annotations` here — torch_tpu's
# _infer_static_argnums identifies tensor args by `annotation is jax.Array`,
# which requires evaluated (non-string) annotations on `_wrapped`.
from collections.abc import Callable
from typing import Optional

import torch

_TPU_FILTER_PROBS_OP: Optional[Callable] = None


def build_jax_filter_probs_fn() -> Callable:
    """Build the pure-JAX filtering function (imports JAX lazily).
    Exposed separately from the custom op so it is unit-testable on CPU."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    def _int32_bsearch(batch_shape, predicate):
        # Largest int32 for which the predicate is False; predicate must be
        # monotonic (False then True) per batch element.
        current_bits = jnp.zeros(batch_shape, dtype=jnp.int32)
        # Bit 31 (sign) compares in the opposite order of all other bits.
        predicate_satisfied = predicate(current_bits)
        current_bits = current_bits | jnp.where(
            predicate_satisfied, jnp.uint32(1 << 31), jnp.uint32(0)
        ).astype(jnp.int32)

        def loop_body(i, bits):
            bit = jnp.int32(1) << (30 - i)
            return bits | jnp.where(predicate(bits | bit), jnp.int32(0), bit)

        return lax.fori_loop(0, 31, loop_body, current_bits)

    non_sign_bits = (1 << 31) - 1
    exponent_bits = (1 << 31) - (1 << (31 - 8))

    def _monotonic_bits(x):
        # int32 bit pattern -> float32 bit pattern with total-order-consistent
        # int32 ordering (the IEEE-754 total-order xor trick).
        return x ^ jnp.where(x < 0, jnp.int32(non_sign_bits), jnp.int32(0))

    def _float32_bsearch(batch_shape, predicate):
        # Largest finite float32 for which the predicate is False.
        def int32_predicate(x):
            x = _monotonic_bits(x)
            is_finite = (x & exponent_bits) != exponent_bits
            x_f32 = lax.bitcast_convert_type(x, jnp.float32)
            return jnp.where(is_finite, predicate(x_f32), x >= 0)

        result = _int32_bsearch(batch_shape, int32_predicate)
        return lax.bitcast_convert_type(_monotonic_bits(result), jnp.float32)

    def _wrapped(
        probs: jax.Array,  # [bs, vocab] float32, rows sum to ~1
        top_ks: jax.Array,  # [bs] int32 (<=0 means disabled)
        top_ps: jax.Array,  # [bs] float32 (>=1 means disabled)
        min_ps: jax.Array,  # [bs] float32 (<=0 means disabled)
        need_top_k: int,
        need_top_p: int,
        need_min_p: int,
    ) -> jax.Array:
        probs = probs.astype(jnp.float32)
        batch_shape = probs.shape[:-1]
        vocab_size = probs.shape[-1]
        # 31 reductions follow; keep the vocab axis off the minor (lane)
        # dimension so each reduction crosses sublanes, not vector lanes.
        probs_t = jnp.swapaxes(probs, -1, -2) if probs.ndim > 1 else probs
        reduce_axis = max(probs.ndim - 2, 0)

        keep = jnp.ones(probs.shape, dtype=jnp.bool_)

        if need_top_k:
            k = jnp.where(top_ks <= 0, jnp.int32(vocab_size), top_ks)
            k = jnp.minimum(k, jnp.int32(vocab_size))

            def topk_predicate(threshold):
                threshold = lax.expand_dims(-threshold, (reduce_axis,))
                count_gt = jnp.sum(probs_t > threshold, axis=reduce_axis)
                return count_gt >= k

            cutoff = -_float32_bsearch(batch_shape, topk_predicate)
            keep &= probs >= cutoff[..., None]

        if need_top_p:

            def topp_predicate(threshold):
                threshold = lax.expand_dims(threshold, (reduce_axis,))
                mass = jnp.sum(
                    jnp.where(probs_t >= threshold, probs_t, 0.0),
                    axis=reduce_axis,
                )
                return mass < top_ps

            cutoff = _float32_bsearch(batch_shape, topp_predicate)
            # Disabled rows (p >= 1) keep everything: float rounding can make
            # the row mass land below 1.0 and produce a spurious cutoff.
            cutoff = jnp.where(top_ps >= 1.0, -jnp.inf, cutoff)
            keep &= probs >= cutoff[..., None]

        if need_min_p:
            max_probs = jnp.max(probs, axis=-1)
            keep &= probs >= (max_probs * min_ps)[..., None]

        return jnp.where(keep, probs, 0.0)

    return _wrapped


def _get_tpu_filter_probs_op() -> Callable:
    """Create the fused top-k/top-p/min-p probability-mask op once per process.
    Registration stays lazy so non-TPU sampling does not import JAX."""

    global _TPU_FILTER_PROBS_OP
    if _TPU_FILTER_PROBS_OP is not None:
        return _TPU_FILTER_PROBS_OP

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    op = cpu_export_jax_op(
        "sglang_tpu_pallas::filter_probs", build_jax_filter_probs_fn()
    )

    def _fake(
        probs: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        need_top_k: int,
        need_top_p: int,
        need_min_p: int,
    ):
        del top_ks, top_ps, min_ps, need_top_k, need_top_p, need_min_p
        return probs.new_empty(probs.shape, dtype=torch.float32)

    op.register_fake(_fake)
    _TPU_FILTER_PROBS_OP = op
    return op


def tpu_filter_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_k: bool,
    need_top_p: bool,
    need_min_p: bool,
) -> torch.Tensor:
    """Zero out probabilities excluded by top-k/top-p/min-p, without sorting.
    The result is unnormalized; `torch.multinomial` renormalizes internally."""

    if probs.device.type != "tpu":
        raise ValueError(f"tpu_filter_probs requires a TPU tensor, got {probs.device}")
    if probs.ndim != 2:
        raise ValueError(f"tpu_filter_probs expects a 2D tensor, got {probs.shape}")
    return _get_tpu_filter_probs_op()(
        probs.float(),
        top_ks.to(torch.int32),
        top_ps.float(),
        min_ps.float(),
        int(need_top_k),
        int(need_top_p),
        int(need_min_p),
    )
