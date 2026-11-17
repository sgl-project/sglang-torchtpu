"""Top-k selection adapters for the DSA Indexer score matrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from glm52_jax.kernels.radix_topk.tuned_configs import (
    DEFAULT_RADIX_TOPK_CONFIG,
    get_tuned_radix_topk_config,
)
from glm52_jax.kernels.radix_topk.v2 import tuned_configs as radix_topk_v2_tuning

_NEG_INF = float("-inf")


def _compact_topk_indices(valid: jax.Array, indices: jax.Array) -> jax.Array:
    """Compact valid unordered candidates before ``-1`` padding."""

    masked = jnp.where(valid, indices, -1)

    def compact_invalid_to_end() -> jax.Array:
        num_rows, k = indices.shape
        destination = jnp.where(
            valid,
            jnp.cumsum(valid, axis=-1, dtype=jnp.int32) - 1,
            k,
        )
        row = jnp.arange(num_rows, dtype=jnp.int32)[:, None]
        compacted = jnp.full((num_rows, k + 1), -1, dtype=indices.dtype)
        compacted = compacted.at[row, destination].set(masked)
        return compacted[:, :k]

    # Long-context rows have K finite candidates, so the serving hot path
    # bypasses compaction. Short-context/padded rows take the O(K) fallback to
    # preserve the valid-prefix ABI required by exact sparse attention.
    return jax.lax.cond(jnp.all(valid), lambda: masked, compact_invalid_to_end)


def _compact_valid_topk_indices(indices: jax.Array, valid_lengths: jax.Array) -> jax.Array:
    """Normalize logical candidates before returning them or mapping slots."""
    valid = (indices >= 0) & (indices < valid_lengths[:, None])
    return _compact_topk_indices(valid, indices)


def _validate_scores(scores: jax.Array, k: int) -> None:
    if scores.ndim != 2:
        raise ValueError(f"DSA top-k scores must be rank 2, got shape={scores.shape}")
    if scores.dtype != jnp.float32:
        raise TypeError(f"DSA top-k scores must be float32, got dtype={scores.dtype}")
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError(f"index_topk must be in [1, {scores.shape[-1]}], got {k}")


def _supports_radix_topk_v2(num_rows: int, k: int, score_width: int = 0) -> bool:
    return (
        # V1's cooperative input banks grow with row width. Long contexts
        # require V2's bounded candidate buffers even with fewer query rows.
        (num_rows >= 24 or score_width >= 524288)
        and k in radix_topk_v2_tuning.SUPPORTED_K
        and jax.devices()[0].device_kind in ("TPU7x", "TPU v7", "TPU v7x")
    )


def select_indexer_radix_topk_indices(
    scores: jax.Array,
    *,
    k: int,
    valid_lengths: jax.Array | None = None,
    page_table: jax.Array | None = None,
    page_size: int = 64,
    v2_config: radix_topk_v2_tuning.RadixTopKV2 | None = None,
) -> jax.Array:
    """Return exact unordered radix top-k positions, or physical cache slots.

    On v7, use V2 for supported K when the local score matrix has at least
    24 rows or at least 512K columns. V2 pads partial
    input blocks; unsupported hardware and K retain V1 tuning.

    A shared page_table requests physical slots. V2 fuses the mapping into
    selection; V1 masks and compacts logical candidates before a separate map.
    Both paths return -1 for invalid entries. An enclosing pipeline can pass
    v2_config to bypass per-tile dispatch while preserving per-row bounds.
    """

    _validate_scores(scores, k)
    if page_table is not None:
        if page_table.ndim != 1 or page_table.dtype != jnp.int32 or not page_table.shape[0]:
            raise ValueError("page_table must be a nonempty int32 vector shared by these rows")
        if page_size < 1:
            raise ValueError("page_size must be positive")
    if valid_lengths is not None and (
        valid_lengths.shape != scores.shape[:1] or valid_lengths.dtype != jnp.int32
    ):
        raise ValueError("valid_lengths must be an int32 vector with one entry per row")
    if _supports_radix_topk_v2(scores.shape[0], k, scores.shape[-1]):
        from glm52_jax.kernels.radix_topk import radix_topk_v2

        mapping_kwargs = (
            {} if page_table is None else dict(page_table=page_table, page_size=page_size)
        )

        def run(config, x, lengths):
            if page_table is not None and scores.shape[-1] >= 524288:
                # Reserve SparseCore VMEM for the long physical page table.
                # The 512K/1M tuning table measures selection without mapping.
                config = radix_topk_v2_tuning.RadixTopKV2(
                    min(config.block_size, 16384), min(config.reconstruct_threshold, 4096)
                )
            return radix_topk_v2(
                x,
                k=k,
                valid_lengths=lengths,
                block_size=config.block_size,
                reconstruct_threshold=config.reconstruct_threshold,
                reverse_scan=True,
                indices_only=True,
                **mapping_kwargs,
            )

        lengths = None if valid_lengths is None else jnp.clip(valid_lengths, 0, scores.shape[-1])
        if v2_config is not None:
            return run(v2_config, scores, lengths)
        config = radix_topk_v2_tuning.get_score_width_config(scores.shape[-1], k)
        return run(config, scores, lengths)

    from glm52_jax.kernels.radix_topk import radix_topk_v1

    config = get_tuned_radix_topk_config(scores.shape[-1], k)
    if config is None:
        config = DEFAULT_RADIX_TOPK_CONFIG
    padding = (-scores.shape[-1]) % config.input_alignment
    padded_scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=_NEG_INF)
    indices = radix_topk_v1(
        padded_scores,
        k=k,
        use_approx_top_k=False,
        num_seq_windows=config.num_seq_windows,
        digit_width=config.digit_width,
        num_digits=config.num_digits,
        use_tc_tiling_on_sc=config.use_tc_tiling_on_sc,
        indices_only=True,
    )[..., :k]
    if page_table is None:
        return indices

    from glm52_jax.kernels.dsa.physical_mapping import map_logical_topk

    lengths = (
        jnp.full(scores.shape[:1], scores.shape[-1], jnp.int32)
        if valid_lengths is None
        else jnp.clip(valid_lengths, 0, scores.shape[-1])
    )
    indices = _compact_valid_topk_indices(indices, lengths)
    return map_logical_topk(
        indices,
        page_table[None, :],
        jnp.zeros(scores.shape[:1], jnp.int32),
        lengths,
        page_size=page_size,
    )


def select_indexer_topk(
    scores: jax.Array,
    *,
    k: int,
    implementation: str,
) -> tuple[jax.Array, jax.Array]:
    """Return XLA top-k score values and sequence-local token indices.

    Score construction and masking belong to the DSA scorer. This adapter only
    selects candidates for the XLA implementations and normalizes them to a
    ``(values, indices)`` ABI. Radix selection uses the indices-only
    ``select_indexer_radix_topk_indices`` adapter instead, avoiding a redundant
    gather of selected score values.
    """

    _validate_scores(scores, k)

    if implementation == "exact_lax":
        return jax.lax.top_k(scores, k)

    if implementation == "approx":
        candidate_values, candidate_indices = jax.lax.approx_max_k(
            scores,
            k,
            recall_target=0.70,
            aggregate_to_topk=False,
        )
        return _order_candidates(candidate_values, candidate_indices, k=k)

    raise ValueError(f"unknown DSA top-k implementation: {implementation}")


def _order_candidates(
    candidate_values: jax.Array,
    candidate_indices: jax.Array,
    *,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    """Order a K-sized candidate set and keep invalid ``-inf`` entries last."""

    values, selection = jax.lax.top_k(candidate_values, k)
    indices = jnp.take_along_axis(candidate_indices, selection, axis=-1)
    return values, indices
