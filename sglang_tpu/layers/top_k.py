# SPDX-License-Identifier: Apache-2.0
"""Exact TPU TopK primitives and Pallas distributed-candidate dispatch."""

from collections.abc import Callable
from typing import NamedTuple

import torch

from sglang_tpu.utils.row_pad import pad_rows, row_bucket

_TPU_TOP_K_OP: Callable | None = None


def supports_distributed_tpu_top_k(lm_head, top_k: int) -> bool:
    """Return whether an LM head has a usable vocab-sharded TPU layout.
    Replicated or incomplete heads must retain the full-logit sampling path."""

    required_layout = (
        "num_embeddings",
        "num_embeddings_per_partition",
        "org_vocab_size",
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
    org_vocab_size: int
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
        if not hasattr(lm_head, "org_vocab_size"):
            raise ValueError(
                "distributed TPU TopK requires explicit original-vocabulary metadata"
            )

        indices = lm_head.shard_indices
        shard = cls(
            width=int(lm_head.num_embeddings_per_partition),
            global_vocab_size=int(lm_head.num_embeddings),
            org_vocab_size=int(lm_head.org_vocab_size),
            org_vocab_start=int(indices.org_vocab_start_index),
            org_vocab_count=int(indices.num_org_elements),
            org_vocab_padded_count=int(indices.num_org_elements_padded),
            added_vocab_start=int(indices.added_vocab_start_index),
            added_vocab_count=int(indices.num_added_elements),
        )
        if not 0 < shard.org_vocab_size <= shard.global_vocab_size:
            raise ValueError("original vocabulary must fit the global vocabulary")
        if shard.org_vocab_padded_count > shard.width:
            raise ValueError(
                "base-vocabulary padded width exceeds LM-head shard width: "
                f"{shard.org_vocab_padded_count} > {shard.width}"
            )
        if shard.org_vocab_padded_count + shard.added_vocab_count > shard.width:
            raise ValueError("added vocabulary exceeds LM-head shard width")
        return shard


class PallasDistributedTopKConfig(NamedTuple):
    """Rank-consistent, compile-static contract for the Pallas TopK ring."""

    shard: TpuVocabShard
    world_size: int
    model_rank: int
    candidate_width: int
    output_dtype: torch.dtype


class PallasDistributedTopKNegotiation(NamedTuple):
    """Result of one rank-consistent Pallas capability negotiation."""

    config: PallasDistributedTopKConfig | None
    fallback_reason: str | None


class _TopKCapabilityDescriptor(NamedTuple):
    rank: int | None
    shard: TpuVocabShard | None
    output_dtype: torch.dtype | None
    candidate_width: int
    head_tp_size: int
    fallback_reason: str | None


def _is_canonical_pallas_shard(
    shard: TpuVocabShard,
    rank: int,
    world_size: int,
) -> bool:
    """Whether rank-based ID reconstruction exactly matches this local shard."""

    if not 0 <= rank < world_size:
        return False
    org_start = min(rank * shard.org_vocab_padded_count, shard.org_vocab_size)
    org_end = min(
        (rank + 1) * shard.org_vocab_padded_count,
        shard.org_vocab_size,
    )
    added_width = shard.width - shard.org_vocab_padded_count
    added_start = min(
        shard.org_vocab_size + rank * added_width,
        shard.global_vocab_size,
    )
    added_end = min(
        shard.org_vocab_size + (rank + 1) * added_width,
        shard.global_vocab_size,
    )
    return (
        added_width >= 0
        and shard.org_vocab_start == org_start
        and shard.org_vocab_count == org_end - org_start
        and shard.added_vocab_start == added_start
        and shard.added_vocab_count == added_end - added_start
    )


def _pallas_shards_cover_vocab(shard: TpuVocabShard, world_size: int) -> bool:
    """Whether the canonical shard widths cover both vocabulary segments."""

    added_width = shard.width - shard.org_vocab_padded_count
    return (
        shard.org_vocab_padded_count * world_size >= shard.org_vocab_size
        and added_width * world_size >= shard.global_vocab_size - shard.org_vocab_size
    )


def _group_spans_world(group) -> bool:
    """Return whether the selected TP group is the complete distributed world."""

    try:
        from sglang.srt.runtime_context import get_parallel

        world_group = get_parallel().world_group
    except (AssertionError, RuntimeError, AttributeError):
        return False

    group_ranks = tuple(getattr(group, "ranks", ()))
    world_ranks = tuple(getattr(world_group, "ranks", ()))
    return (
        int(getattr(group, "world_size", 0))
        == int(getattr(world_group, "world_size", -1))
        and group_ranks
        and group_ranks == world_ranks
    )


def negotiate_pallas_distributed_top_k_config(
    lm_head,
    *,
    group,
    expected_dtype: torch.dtype | None,
    local_fallback_reason: str | None = None,
) -> PallasDistributedTopKNegotiation:
    """Negotiate one Pallas capability contract across every participating rank.

    Unsupported or inconsistent properties select full-vocabulary logits on
    every rank. The consensus collective itself is deliberately not caught: a
    transport failure cannot safely become a rank-local fallback decision.
    """

    from sglang_tpu.layers.pallas.distributed_topk import (
        MAX_DISTRIBUTED_TOP_K,
        SUPPORTED_WORLD_SIZES,
    )

    world_size = int(getattr(group, "world_size", 0))
    model_rank = getattr(group, "rank_in_group", None)
    if world_size not in SUPPORTED_WORLD_SIZES:
        return PallasDistributedTopKNegotiation(
            None, f"TP{world_size} is not supported by the Pallas kernel"
        )
    if not _group_spans_world(group):
        return PallasDistributedTopKNegotiation(
            None, "the LM-head tensor-parallel group is not the full world"
        )

    try:
        candidate_width = min(
            MAX_DISTRIBUTED_TOP_K,
            int(getattr(lm_head, "num_embeddings", 0)),
            int(getattr(lm_head, "num_embeddings_per_partition", 0)),
        )
    except (TypeError, ValueError):
        candidate_width = 0
    try:
        shard = TpuVocabShard.from_lm_head(lm_head)
    except (AttributeError, TypeError, ValueError):
        shard = None
    try:
        layout_supported = supports_distributed_tpu_top_k(lm_head, candidate_width)
    except (TypeError, ValueError):
        layout_supported = False
    try:
        head_tp_size = int(getattr(lm_head, "tp_size", 0))
    except (TypeError, ValueError):
        head_tp_size = 0

    fallback_reason = local_fallback_reason
    if fallback_reason is None:
        if expected_dtype not in (torch.bfloat16, torch.float32):
            fallback_reason = f"logits dtype {expected_dtype} is unsupported"
        elif candidate_width != MAX_DISTRIBUTED_TOP_K:
            fallback_reason = (
                f"the shard cannot provide {MAX_DISTRIBUTED_TOP_K} candidates"
            )
        elif head_tp_size != world_size:
            fallback_reason = (
                f"LM-head TP size {head_tp_size} does not match group size {world_size}"
            )
        elif not layout_supported or shard is None:
            fallback_reason = "the LM head lacks an explicit sharded-vocabulary layout"
        elif not isinstance(model_rank, int) or not _is_canonical_pallas_shard(
            shard, model_rank, world_size
        ):
            fallback_reason = "the local vocabulary shard is not canonical"

    descriptor = _TopKCapabilityDescriptor(
        rank=model_rank,
        shard=shard,
        output_dtype=expected_dtype,
        candidate_width=candidate_width,
        head_tp_size=head_tp_size,
        fallback_reason=fallback_reason,
    )
    descriptors = group.all_gather_object(descriptor)
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != world_size:
        return PallasDistributedTopKNegotiation(
            None, "capability consensus returned an invalid descriptor list"
        )

    by_rank: dict[int, _TopKCapabilityDescriptor] = {}
    for item in descriptors:
        if not isinstance(item, _TopKCapabilityDescriptor):
            return PallasDistributedTopKNegotiation(
                None, "capability consensus returned an invalid descriptor"
            )
        if (
            not isinstance(item.rank, int)
            or not 0 <= item.rank < world_size
            or item.rank in by_rank
        ):
            return PallasDistributedTopKNegotiation(
                None, "capability consensus returned invalid rank metadata"
            )
        by_rank[item.rank] = item
    if len(by_rank) != world_size:
        return PallasDistributedTopKNegotiation(
            None, "capability consensus did not cover every rank"
        )

    if shard is None:
        return PallasDistributedTopKNegotiation(
            None, "the local LM head has no valid vocabulary shard"
        )
    shared = (
        shard.width,
        shard.global_vocab_size,
        shard.org_vocab_size,
        shard.org_vocab_padded_count,
        expected_dtype,
    )
    for rank in range(world_size):
        peer = by_rank[rank]
        if peer.fallback_reason is not None:
            return PallasDistributedTopKNegotiation(
                None, f"rank {rank}: {peer.fallback_reason}"
            )
        if (
            peer.shard is None
            or peer.candidate_width != MAX_DISTRIBUTED_TOP_K
            or peer.head_tp_size != world_size
        ):
            return PallasDistributedTopKNegotiation(
                None, f"rank {rank}: incomplete Pallas capability metadata"
            )
        peer_shared = (
            peer.shard.width,
            peer.shard.global_vocab_size,
            peer.shard.org_vocab_size,
            peer.shard.org_vocab_padded_count,
            peer.output_dtype,
        )
        if peer_shared != shared or not _is_canonical_pallas_shard(
            peer.shard, rank, world_size
        ):
            return PallasDistributedTopKNegotiation(
                None, f"rank {rank}: vocabulary metadata disagrees across ranks"
            )

    if not _pallas_shards_cover_vocab(shard, world_size):
        return PallasDistributedTopKNegotiation(
            None, "the canonical shards do not cover the complete vocabulary"
        )

    return PallasDistributedTopKNegotiation(
        PallasDistributedTopKConfig(
            shard=shard,
            world_size=world_size,
            model_rank=model_rank,
            candidate_width=candidate_width,
            output_dtype=expected_dtype,
        ),
        None,
    )


def _get_tpu_top_k_op() -> Callable:
    """Create the exported native-XLA TopK custom op once per process.
    Registration remains lazy so non-TPU sampling does not import JAX."""

    global _TPU_TOP_K_OP
    if _TPU_TOP_K_OP is not None:
        return _TPU_TOP_K_OP

    import jax

    from sglang_tpu.utils.pallas_jax_op import cpu_export_jax_op

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
    # The exported op compiles once per argument shape: bucket rows to a power of two.
    rows = values.shape[0]
    bucket = row_bucket(rows)
    top_values, top_ids = _get_tpu_top_k_op()(pad_rows(values, bucket), k)
    if bucket != rows:
        return top_values[:rows], top_ids[:rows]
    return top_values, top_ids


def register_tpu_top_k_ops() -> None:
    """Register the local TopK schema before full-graph tracing begins."""
    _get_tpu_top_k_op()


def distributed_tpu_top_k(
    local_logits: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
    *,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact distributed TopK through the direct-HBM Pallas ring."""

    from sglang_tpu.layers.pallas.distributed_topk import (
        MAX_DISTRIBUTED_TOP_K,
        SUPPORTED_WORLD_SIZES,
    )

    if local_logits.device.type != "tpu":
        raise ValueError(
            f"distributed_tpu_top_k requires a TPU tensor, got {local_logits.device}"
        )
    if local_logits.ndim != 2:
        raise ValueError(
            f"distributed_tpu_top_k expects a 2D tensor, got shape {local_logits.shape}"
        )
    if local_logits.shape[-1] != shard.width:
        raise RuntimeError(
            "distributed TPU TopK local width changed after capability negotiation"
        )
    if local_logits.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"distributed TPU TopK does not support {local_logits.dtype}"
        )
    world_size = int(group.world_size)
    model_rank = int(group.rank_in_group)
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise RuntimeError(f"Pallas distributed TopK does not support TP{world_size}")
    if not _is_canonical_pallas_shard(shard, model_rank, world_size):
        raise RuntimeError(
            "distributed TPU TopK vocabulary shard changed after negotiation"
        )
    if min(shard.width, shard.global_vocab_size) < MAX_DISTRIBUTED_TOP_K:
        raise RuntimeError(
            "distributed TPU TopK shard cannot supply the fixed candidate width"
        )
    if not _pallas_shards_cover_vocab(shard, world_size):
        raise RuntimeError(
            "distributed TPU TopK shards do not cover the complete vocabulary"
        )
    if not 0 < k <= MAX_DISTRIBUTED_TOP_K:
        raise ValueError(f"k must be in [1, {MAX_DISTRIBUTED_TOP_K}], got {k}")
    if local_logits.shape[0] == 0:
        shape = (0, k)
        return (
            local_logits.new_empty(shape, dtype=torch.float32),
            local_logits.new_empty(shape, dtype=torch.int32),
        )

    from sglang_tpu.layers.distributed_topk_pallas import (
        pallas_distributed_top_k,
    )

    return pallas_distributed_top_k(
        local_logits,
        org_vocab_size=shard.org_vocab_size,
        global_vocab_size=shard.global_vocab_size,
        org_vocab_padded_count=shard.org_vocab_padded_count,
        k=k,
        world_size=world_size,
    )
