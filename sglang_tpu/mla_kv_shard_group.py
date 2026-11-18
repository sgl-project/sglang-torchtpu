# SPDX-License-Identifier: Apache-2.0
"""Process groups of the attention-TP ranks that share one sparse-MLA latent cache."""

from __future__ import annotations

_GROUPS: dict = {}


def mla_kv_shard_group(shards: int):
    """GroupCoordinator over this rank's shard group, consecutive attention-TP
    ranks ``[base, base + shards)`` (the two TensorCores of a chip at shards=2).

    Every rank creates every group, in the same order, the first time any
    sparse backend asks; the backends are built in the same order everywhere.
    """
    if shards in _GROUPS:
        return _GROUPS[shards]
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp = parallel.attn_tp_size
    if shards == tp:
        # The whole attention-TP group: reuse it rather than build a twin.
        _GROUPS[shards] = parallel.attn_tp_group
        return _GROUPS[shards]
    if tp % shards:
        raise ValueError(
            f"attention-TP size {tp} is not a multiple of the MLA shard group {shards}"
        )
    world = get_world_group()
    # The validator admits attention TP alone (no attention DP or CP, no PP), so the
    # attention-TP groups are the world's contiguous blocks of tp ranks.
    group_ranks = [
        list(range(start, start + shards))
        for start in range(0, dist.get_world_size(), shards)
    ]
    group = init_model_parallel_group(
        group_ranks,
        world.local_rank,
        dist.get_backend(world.device_group),
        group_name=f"mla_kv_shard{shards}",
    )
    if group.rank_in_group != parallel.attn_tp_rank % shards:
        raise RuntimeError(
            "MLA shard group order disagrees with the attention-TP rank: "
            f"{group.rank_in_group} vs {parallel.attn_tp_rank % shards}"
        )
    _GROUPS[shards] = group
    return group
