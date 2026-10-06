# SPDX-License-Identifier: Apache-2.0
# Binary-search top-k/top-p/min-p filtering. The bit-descent search is
# adapted from https://github.com/vllm-project/tpu-inference
# tpu_inference/layers/common/binary_search.py (itself from T5X).
# Copyright 2024 The T5X Authors. Apache-2.0.
"""Sort-free distribution filtering for TPU sampling.

The torch fallback (`top_k_top_p_min_p_sampling_from_probs_torch`) needs a
full-vocabulary sort plus boolean-mask scatters, which run eagerly on TPU at
~415 ms per decode step. This module finds the same cutoffs by a 30-step
binary search over float32 bit patterns — only dense element-wise ops and
reductions, exported as ONE compiled XLA program via `cpu_export_jax_op`.

All three filters are lower bounds on one per-row cutoff. Top-k and top-p
predicates each hold on a prefix of the threshold axis, so the largest
threshold satisfying either is `max(top_k_cutoff, top_p_cutoff)` — exactly
what ANDing the two masks needs — and one search with the OR'd predicate
finds it. min-p contributes `max_prob * min_p` through a plain `maximum`.

Probabilities are non-negative and below 2.0, so their int32 bit patterns
are already ordered like their values, bit 31 (sign) is never set, and
neither is bit 30 (the exponent bit marking |x| >= 2): descending bits 29..0
reaches every representable cutoff without T5X's sign-flip or finiteness
handling.

Semantics match the torch reference up to ties: the cutoff keeps every value
tied with the boundary, where a sort breaks ties arbitrarily. min-p uses the
global max probability as its reference, same as the torch path. A row whose
filters are all unreachable (`top_p == 1.0` with row mass rounding under 1.0,
`top_k >= vocab`) finds no passing threshold and keeps a cutoff of 0.0, which
keeps the whole row — no special-casing needed.
"""

# NOTE: no `from __future__ import annotations` here — torch_tpu's
# _infer_static_argnums identifies tensor args by `annotation is jax.Array`,
# which requires evaluated (non-string) annotations on `_wrapped`.
from collections.abc import Callable
from typing import Optional

import torch

from sglang.srt.hardware_backend.tpu.row_pad import pad_rows, row_bucket

_TPU_FILTER_PROBS_OP: Optional[Callable] = None


def _get_tpu_filter_probs_op() -> Callable:
    """Create the fused top-k/top-p/min-p probability-mask op once per process.
    Registration stays lazy so non-TPU sampling does not import JAX."""

    global _TPU_FILTER_PROBS_OP
    if _TPU_FILTER_PROBS_OP is not None:
        return _TPU_FILTER_PROBS_OP

    import jax
    import jax.numpy as jnp
    from jax import lax

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    # Cutoffs live in [0, 2), where only bits 29..0 of the pattern vary.
    search_bits = 30

    def _largest_passing_threshold(predicate, batch_size):
        # Greedy bit descent, high to low: keep a bit iff the predicate still
        # passes with it set. Bit-pattern order is value order here, so this
        # returns the largest float32 the predicate accepts — or 0.0 when it
        # accepts none, which downstream keeps the whole row.
        def keep_bit_if_still_passing(index, bits):
            candidate = bits | jnp.left_shift(1, search_bits - 1 - index)
            threshold = lax.bitcast_convert_type(candidate, jnp.float32)
            return jnp.where(predicate(threshold), candidate, bits)

        bits = lax.fori_loop(
            0,
            search_bits,
            keep_bit_if_still_passing,
            jnp.zeros(batch_size, dtype=jnp.int32),
        )
        return lax.bitcast_convert_type(bits, jnp.float32)

    def _wrapped(
        probs: jax.Array,  # [bs, vocab] float32, rows sum to ~1
        top_ks: jax.Array,  # [bs] int32 (TOP_K_ALL when disabled)
        top_ps: jax.Array,  # [bs] float32 (>= 1 means disabled)
        min_ps: jax.Array,  # [bs] float32 (0 means disabled)
        need_top_k: int,
        need_top_p: int,
        need_min_p: int,
    ) -> jax.Array:
        probs = probs.astype(jnp.float32)
        batch_size, vocab_size = probs.shape

        if need_top_k:
            # SamplingParams.verify rejects top_k < 1 and normalizes
            # "disabled" to TOP_K_ALL, so clipping covers the disabled case
            # and any zero-padded ghost row in one op.
            k = jnp.clip(top_ks, 1, vocab_size)

        def keeps_enough(threshold):
            kept = probs >= threshold[:, None]
            enough = None
            if need_top_k:
                enough = jnp.sum(kept, axis=-1, dtype=jnp.int32) >= k
            if need_top_p:
                enough_mass = jnp.sum(jnp.where(kept, probs, 0.0), axis=-1) >= top_ps
                enough = enough_mass if enough is None else enough | enough_mass
            return enough

        cutoff = jnp.zeros(batch_size, dtype=jnp.float32)
        if need_top_k or need_top_p:
            cutoff = _largest_passing_threshold(keeps_enough, batch_size)
        if need_min_p:
            cutoff = jnp.maximum(cutoff, jnp.max(probs, axis=-1) * min_ps)
        return jnp.where(probs >= cutoff[:, None], probs, 0.0)

    op = cpu_export_jax_op("sglang_tpu_pallas::filter_probs", _wrapped)

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
    # The exported op compiles once per argument shape: bucket rows to a power
    # of two. Ghost rows: zero probs, k=0 (clipped to 1), top_p=1, min_p=0.
    rows = probs.shape[0]
    bucket = row_bucket(rows)
    filtered = _get_tpu_filter_probs_op()(
        pad_rows(probs.float(), bucket),
        pad_rows(top_ks.to(torch.int32), bucket),
        pad_rows(top_ps.float(), bucket, fill=1.0),
        pad_rows(min_ps.float(), bucket),
        int(need_top_k),
        int(need_top_p),
        int(need_min_p),
    )
    return filtered if bucket == rows else filtered[:rows]
