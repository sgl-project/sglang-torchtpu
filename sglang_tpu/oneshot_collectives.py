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

    from sglang_tpu.kernels.oneshot_all_gather import (
        oneshot_all_gather,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

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


def _cached_op(
    cache: dict[int, Callable], world_size: int, build: Callable[[int], Callable]
) -> Callable:
    op = cache.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = cache.get(world_size)
        if op is None:
            op = build(world_size)
            cache[world_size] = op
    return op


def get_oneshot_all_gather_op(world_size: int) -> Callable:
    return _cached_op(_OPS, world_size, _build_all_gather_op)


def oneshot_all_gather(
    x: torch.Tensor, *, rank_input: torch.Tensor, world_size: int
) -> torch.Tensor:
    """All-gather ``x`` [m, n] along dim 0 in TP-rank order -> [world_size * m, n].

    ``rank_input`` is this rank's TP rank as an int32[1] tensor that is already a
    graph input (see pallas_distributed_top_k for why it cannot be built here).
    """
    out = get_oneshot_all_gather_op(world_size)(x, rank_input)
    return out.reshape(world_size * x.shape[0], *x.shape[1:])


_A2A_OPS: dict[int, Callable] = {}


def _build_all_to_all_op(world_size: int) -> Callable:
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.kernels.oneshot_all_to_all import oneshot_all_to_all as kernel
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)
    # Each device's [world_size, rows, 128] input and output: its "model" slice.
    sharded = jax.shard_map(
        functools.partial(kernel, world_size=world_size),
        mesh=mesh,
        in_specs=(P("model"),),
        out_specs=P("model"),
        check_vma=False,
    )

    def _wrapped(x: jax.Array) -> jax.Array:
        return sharded(x)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::oneshot_all_to_all_tp{world_size}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(P("model"),),
        output_partition_specs=(P("model"),),
    )


def get_oneshot_all_to_all_op(world_size: int) -> Callable:
    return _cached_op(_A2A_OPS, world_size, _build_all_to_all_op)


# Rows of a [rows, 128] block: whole tiles for 8- to 32-bit dtypes.
_A2A_ROW_ALIGN = 32


def oneshot_all_to_all(x: torch.Tensor, *, world_size: int) -> torch.Tensor:
    """Row ``j`` of ``x`` [world_size, n] goes to TP rank ``j``; row ``i`` of the
    result came from TP rank ``i``. TP must span the world (TP rank = mesh
    position, see kernels/oneshot_all_to_all.py); rows pad to whole tiles."""
    n = x.shape[1]
    block = _A2A_ROW_ALIGN * 128
    padded = -(-n // block) * block
    if padded != n:
        x = torch.nn.functional.pad(x, (0, padded - n))
    out = get_oneshot_all_to_all_op(world_size)(
        x.contiguous().view(world_size, padded // 128, 128)
    ).view(world_size, padded)
    return out if padded == n else out[:, :n].contiguous()
