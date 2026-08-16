# SPDX-License-Identifier: Apache-2.0
"""Exact TPU TopK primitives for vocab-parallel token sampling.
Sharded reduction communicates only local TopK logits and global token IDs."""

from collections.abc import Callable
from typing import NamedTuple

import torch

_TPU_TOP_K_OP: Callable | None = None
_TPU_SHARDED_TOP_K_OP: Callable | None = None


def supports_distributed_tpu_top_k(lm_head, top_k: int) -> bool:
    """Return whether an LM head has a usable vocab-sharded TPU layout.
    Replicated or incomplete heads must retain the full-logit sampling path."""

    required_layout = (
        "num_embeddings",
        "num_embeddings_per_partition",
        "shard_indices",
    )
    return (
        getattr(lm_head, "enable_tp", False)
        and all(hasattr(lm_head, attr) for attr in required_layout)
        and 0 < top_k <= lm_head.num_embeddings
        and top_k <= lm_head.num_embeddings_per_partition
    )


class TpuVocabShard(NamedTuple):
    """Describe valid base and added tokens inside one padded LM-head shard.
    Device kernels use this metadata to mask padding and reconstruct global IDs.

    A NamedTuple rather than the usual msgspec.Struct: it is constructed
    inside the torch.compile'd forward (fullgraph, so no graph break is
    possible there), and Dynamo cannot instantiate msgspec's C-level
    ``__new__`` in-graph while NamedTuples are traced natively."""

    width: int
    global_vocab_size: int
    org_vocab_start: int
    org_vocab_count: int
    org_vocab_padded_count: int
    added_vocab_start: int
    added_vocab_count: int

    @classmethod
    def from_lm_head(cls, lm_head) -> "TpuVocabShard":
        """Build validated host metadata from an LM head's partition indices.
        Invalid padded layouts fail before they can produce device token IDs."""

        if not getattr(lm_head, "enable_tp", False):
            raise ValueError("distributed TPU TopK requires a vocab-sharded LM head")
        indices = lm_head.shard_indices
        shard = cls(
            width=lm_head.num_embeddings_per_partition,
            global_vocab_size=lm_head.num_embeddings,
            org_vocab_start=indices.org_vocab_start_index,
            org_vocab_count=indices.num_org_elements,
            org_vocab_padded_count=indices.num_org_elements_padded,
            added_vocab_start=indices.added_vocab_start_index,
            added_vocab_count=indices.num_added_elements,
        )
        if shard.org_vocab_padded_count > shard.width:
            raise ValueError(
                "base-vocabulary padded width exceeds LM-head shard width: "
                f"{shard.org_vocab_padded_count} > {shard.width}"
            )
        if shard.org_vocab_padded_count + shard.added_vocab_count > shard.width:
            raise ValueError("added vocabulary exceeds LM-head shard width")
        return shard


def _get_tpu_top_k_op() -> Callable:
    """Create the exported native-XLA TopK custom op once per process.
    Registration remains lazy so non-TPU sampling does not import JAX."""

    global _TPU_TOP_K_OP
    if _TPU_TOP_K_OP is not None:
        return _TPU_TOP_K_OP

    import jax

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(values: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
        # torch_tpu's aten::topk lowering is currently a full StableHLO
        # sort plus slice. lax.top_k exports XLA's native exact TopK op.
        return jax.lax.top_k(values, k)

    op = cpu_export_jax_op("sglang_tpu_pallas::exact_top_k", _wrapped)

    def _fake(values: torch.Tensor, k: int):
        shape = (*values.shape[:-1], k)
        return (
            values.new_empty(shape),
            values.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    _TPU_TOP_K_OP = op
    return op


def tpu_top_k(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exact descending TopK values from a two-dimensional TPU tensor.
    Indices are int32 positions in the input's final dimension."""

    if values.device.type != "tpu":
        raise ValueError(f"tpu_top_k requires a TPU tensor, got {values.device}")
    if values.ndim != 2:
        raise ValueError(f"tpu_top_k expects a 2D tensor, got shape {values.shape}")
    if not 0 < k <= values.shape[-1]:
        raise ValueError(f"k must be in [1, {values.shape[-1]}], got {k}")
    return _get_tpu_top_k_op()(values, k)


def _get_tpu_sharded_top_k_op() -> Callable:
    """Create the fused padding-mask, local TopK, and ID-mapping custom op.
    Fusion avoids materializing a vocabulary-sized global-ID tensor."""

    global _TPU_SHARDED_TOP_K_OP
    if _TPU_SHARDED_TOP_K_OP is not None:
        return _TPU_SHARDED_TOP_K_OP

    import jax
    import jax.numpy as jnp

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        values: jax.Array,
        k: int,
        org_vocab_start: int,
        org_vocab_count: int,
        org_vocab_padded_count: int,
        added_vocab_start: int,
        added_vocab_count: int,
    ) -> tuple[jax.Array, jax.Array]:
        positions = jnp.arange(values.shape[-1], dtype=jnp.int32)
        added_positions = positions - org_vocab_padded_count
        valid = (positions < org_vocab_count) | (
            (added_positions >= 0) & (added_positions < added_vocab_count)
        )
        masked_values = jnp.where(valid, values, -jnp.inf)
        top_values, top_positions = jax.lax.top_k(masked_values, k)
        token_ids = jnp.where(
            top_positions < org_vocab_count,
            org_vocab_start + top_positions,
            added_vocab_start + top_positions - org_vocab_padded_count,
        )
        return top_values, token_ids.astype(jnp.int32)

    op = cpu_export_jax_op("sglang_tpu_pallas::exact_sharded_top_k", _wrapped)

    def _fake(
        values: torch.Tensor,
        k: int,
        org_vocab_start: int,
        org_vocab_count: int,
        org_vocab_padded_count: int,
        added_vocab_start: int,
        added_vocab_count: int,
    ):
        del (
            org_vocab_start,
            org_vocab_count,
            org_vocab_padded_count,
            added_vocab_start,
            added_vocab_count,
        )
        shape = (*values.shape[:-1], k)
        return (
            values.new_empty(shape),
            values.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    _TPU_SHARDED_TOP_K_OP = op
    return op


def tpu_sharded_top_k(
    values: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one shard's exact TopK while excluding padded columns.
    Returned int32 IDs already refer to the unpadded global vocabulary."""

    if values.device.type != "tpu":
        raise ValueError(
            f"tpu_sharded_top_k requires a TPU tensor, got {values.device}"
        )
    if values.ndim != 2:
        raise ValueError(
            f"tpu_sharded_top_k expects a 2D tensor, got shape {values.shape}"
        )
    if values.shape[-1] != shard.width:
        raise ValueError(
            "vocab shard width does not match local logits: "
            f"{shard.width} != {values.shape[-1]}"
        )
    if not 0 < k <= shard.width or k > shard.global_vocab_size:
        raise ValueError(
            "k must fit both the local padded shard and global vocabulary, "
            f"got k={k}, shard_width={shard.width}, "
            f"global_vocab_size={shard.global_vocab_size}"
        )
    return _get_tpu_sharded_top_k_op()(
        values,
        k,
        shard.org_vocab_start,
        shard.org_vocab_count,
        shard.org_vocab_padded_count,
        shard.added_vocab_start,
        shard.added_vocab_count,
    )


def register_tpu_top_k_ops() -> None:
    """Register local and global TopK ops before full-graph tracing begins.
    Eager registration keeps custom-op schemas visible to torch.compile."""

    _get_tpu_top_k_op()
    _get_tpu_sharded_top_k_op()


def distributed_tpu_top_k(
    local_logits: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
    *,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge exact sharded TopK logits into a replicated global TopK.
    Only k logits and token IDs per shard cross the tensor-parallel group."""

    sharded_top_k_logits, sharded_top_k_token_ids = tpu_sharded_top_k(
        local_logits, shard, k
    )

    if group.world_size > 1:
        if shard.global_vocab_size <= 2**24:
            # Token IDs are exact in fp32 for every practical serving
            # vocabulary. Packing values and IDs halves collective latency at
            # the cost of a tiny payload increase over bf16 + int32.
            packed_sharded_top_k = torch.stack(
                (sharded_top_k_logits.float(), sharded_top_k_token_ids.float()),
                dim=-1,
            )
            gathered_sharded_top_k = group.all_gather(packed_sharded_top_k, dim=-2)
            gathered_sharded_top_k_logits = gathered_sharded_top_k[..., 0]
            gathered_sharded_top_k_token_ids = gathered_sharded_top_k[..., 1].to(
                torch.int32
            )
        else:
            # Preserve exact IDs for unusually large vocabularies.
            gathered_sharded_top_k_logits = group.all_gather(
                sharded_top_k_logits, dim=-1
            )
            gathered_sharded_top_k_token_ids = group.all_gather(
                sharded_top_k_token_ids, dim=-1
            )
    else:
        gathered_sharded_top_k_logits = sharded_top_k_logits
        gathered_sharded_top_k_token_ids = sharded_top_k_token_ids

    global_top_k_logits, global_top_k_positions = tpu_top_k(
        gathered_sharded_top_k_logits, k
    )
    global_top_k_token_ids = torch.gather(
        gathered_sharded_top_k_token_ids,
        dim=1,
        index=global_top_k_positions.to(torch.int64),
    )
    return global_top_k_logits.float(), global_top_k_token_ids.to(torch.int32)
