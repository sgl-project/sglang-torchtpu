# SPDX-License-Identifier: Apache-2.0
"""Two-shot Pallas all-reduce for decode-sized TP reductions of row-parallel linears.

``install_twoshot_all_reduce(module, names)`` wraps reducing, bias-free
RowParallelLinear children so that small all-reduces run through
``kernels/twoshot_all_reduce.py``; every other call keeps the linear's own reduction.
"""

import functools
from collections.abc import Callable
from threading import Lock

import torch

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
from sglang.srt.runtime_context import get_forward

_OPS: dict[int, Callable] = {}
_OPS_LOCK = Lock()
# Above this many bytes per rank XLA's all-reduce is as fast (measured on TP8 v7x,
# where bf16 [32, 8192] already matches it).
TWOSHOT_ALL_REDUCE_MAX_BYTES = 256 * 1024


@functools.cache
def _get_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def _build_op(world_size: int) -> Callable:
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.kernels.twoshot_all_reduce import twoshot_all_reduce
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)
    sharded = jax.shard_map(
        functools.partial(twoshot_all_reduce, world_size=world_size),
        mesh=mesh,
        in_specs=(P("model"),),
        out_specs=P(),
        check_vma=False,
    )

    def _wrapped(x: jax.Array) -> jax.Array:
        return sharded(x)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::twoshot_all_reduce_tp{world_size}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(P("model"),),
        output_partition_specs=(P(),),
    )


def get_twoshot_all_reduce_op(world_size: int) -> Callable:
    """The registered op for one TP size; build it before tracing, never inside it."""
    op = _OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(world_size)
        if op is None:
            op = _build_op(world_size)
            _OPS[world_size] = op
    return op


def _tp_size(linear: RowParallelLinear) -> int:
    """Ranks the linear's partial output is reduced over (its retained TP group)."""
    return get_group_rank_size(linear.tp_group)[1]


class TwoShotAllReduceLinear(torch.nn.Module):
    """A reducing, bias-free RowParallelLinear whose small all-reduces run two-shot."""

    def __init__(self, linear: RowParallelLinear) -> None:
        super().__init__()
        self.linear = linear
        self.world_size = _tp_size(linear)

    def _fits(self, x: torch.Tensor) -> bool:
        """A non-empty 2-D payload under the size cap, outside forwards that
        reduce-scatter or skip the reduction."""
        output_bytes = x.shape[0] * self.linear.output_size * x.element_size()
        return (
            x.dim() == 2
            and 0 < output_bytes <= TWOSHOT_ALL_REDUCE_MAX_BYTES
            and not get_forward().sp_active
            and not should_skip_mlp_all_reduce()
        )

    def forward(self, x: torch.Tensor, skip_all_reduce: bool = False, **kwargs):
        if skip_all_reduce or not self._fits(x):
            return self.linear(x, skip_all_reduce=skip_all_reduce, **kwargs)
        output, output_bias = self.linear(x, skip_all_reduce=True, **kwargs)
        return get_twoshot_all_reduce_op(self.world_size)(output), output_bias


def install_twoshot_all_reduce(
    module: torch.nn.Module, names: tuple[str, ...], *, world_size: int
) -> int:
    """Wrap every reducing, bias-free TP RowParallelLinear child named in ``names``.

    The kernel's mesh is the whole world, so layers that reduce over a smaller
    group (DP attention, a draft TP context) keep their own all-reduce, as do
    outputs whose columns don't split into 128-lane chunks per rank. Returns the count.
    """
    swaps = [
        (parent, name, child)
        for parent in module.modules()
        for name, child in parent.named_children()
        if name in names
        and isinstance(child, RowParallelLinear)
        and child.reduce_results
        and _tp_size(child) == world_size > 1
        and child.output_size % (128 * world_size) == 0
        and child.bias is None
        and not child.use_dp_attention_reduce
        and not child.use_decode_attn_tp
    ]
    for parent, name, child in swaps:
        get_twoshot_all_reduce_op(world_size)
        setattr(parent, name, TwoShotAllReduceLinear(child))
    return len(swaps)
