# SPDX-License-Identifier: Apache-2.0
"""Torch custom ops for one-shot (single-hop) TPU collectives.

XLA's all-gather has a high fixed latency even for one token row;
these Pallas kernels write every shard straight to every peer instead.
"""

import functools
from collections.abc import Callable
from threading import Lock

import torch

_OPS: dict[int, Callable] = {}
_OPS_LOCK = Lock()


@functools.cache
def _get_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def _build_all_gather_op(world_size: int) -> Callable:
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.distributed.pallas.oneshot_all_gather import (
        oneshot_all_gather,
    )
    from sglang_tpu.utils.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)
    sharded = jax.shard_map(
        functools.partial(oneshot_all_gather, world_size=world_size),
        mesh=mesh,
        in_specs=(P("model"), P("model")),
        out_specs=P(),
        check_vma=False,
    )

    def _wrapped(x: jax.Array, rank_input: jax.Array) -> jax.Array:
        return sharded(x, rank_input)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::oneshot_all_gather_tp{world_size}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(P("model"), P("model")),
        output_partition_specs=(P(),),
    )


def get_oneshot_all_gather_op(world_size: int) -> Callable:
    op = _OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(world_size)
        if op is None:
            op = _build_all_gather_op(world_size)
            _OPS[world_size] = op
    return op


def oneshot_all_gather(
    x: torch.Tensor, *, rank_input: torch.Tensor, world_size: int
) -> torch.Tensor:
    """All-gather ``x`` [m, n] along dim 0 in TP-rank order -> [world_size * m, n].

    ``rank_input`` is this rank's TP rank as an int32[1] tensor that is already a
    graph input (see pallas_distributed_top_k for why it cannot be built here).
    """
    out = get_oneshot_all_gather_op(world_size)(x, rank_input)
    return out.reshape(world_size * x.shape[0], *x.shape[1:])
