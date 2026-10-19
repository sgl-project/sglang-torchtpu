"""Source-level rank-local operations with an explicit direct-compile input.

Outside a rank-parameterized trace these are ordinary slices of the caller's
rank. Dynamo specializes the caller's rank; only direct compilation lifts it.
Rank lifting is independent of how compilation is scheduled.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

import jax
import jax.numpy as jnp
import torch
import torch.distributed as dist
from torch_tpu._internal.distributed.device_mesh import get_device_assignment

from sglang_tpu.utils.pallas_jax_op import cpu_export_jax_op

_RANK: ContextVar[torch.Tensor | None] = ContextVar("tpu_trace_rank", default=None)


@contextmanager
def trace_rank(rank: torch.Tensor):
    token = _RANK.set(rank)
    try:
        yield
    finally:
        _RANK.reset(token)


@cache
def _slice_op(rank_to_device: tuple[int, ...] | None):
    def sliced(
        x: jax.Array, rank: jax.Array, dim: int, parts: int, rank_stride: int
    ) -> jax.Array:
        length = x.shape[dim] // parts
        index = rank[0]
        if rank_to_device is not None:
            index = jnp.asarray(rank_to_device, dtype=jnp.int32)[index]
        return jax.lax.dynamic_slice_in_dim(
            x, (index // rank_stride % parts) * length, length, axis=dim
        )

    suffix = "" if rank_to_device is None else "_" + "_".join(map(str, rank_to_device))
    return cpu_export_jax_op(f"sglang::rank_local_slice{suffix}", sliced)


def traced_rank(
    expected_rank: int, *, group_size: int | None = None
) -> torch.Tensor | None:
    """Return the rank input, optionally within a contiguous parallel group."""
    rank = None if torch.compiler.is_dynamo_compiling() else _RANK.get()
    if rank is None:
        return None
    logical_rank = dist.get_rank()
    if group_size is not None:
        logical_rank %= group_size
        rank = rank % group_size
    if expected_rank != logical_rank:
        raise ValueError("Rank inputs require contiguous logical rank groups")
    return rank


def rank_slice(
    tensor,
    dim: int,
    parts: int,
    rank: int,
    *,
    rank_stride: int = 1,
    rank_to_device: tuple[int, ...] | None = None,
):
    """Select a chunk; rank-parameterized traces map the logical rank to its chunk index.

    ``rank_stride`` consecutive ranks share a chunk (e.g. four ranks per
    TP16 fallback block of a striped TP4 projection weight).
    ``rank_to_device`` maps the logical rank input when physical cores own chunks.
    """
    if parts == 1:
        # One chunk is rank-agnostic (e.g. decode QKVOG RoPE rows at tp_size=1).
        return tensor
    traced = None if torch.compiler.is_dynamo_compiling() else _RANK.get()
    if traced is None:
        return tensor.tensor_split(parts, dim)[rank]
    mesh_size = len(get_device_assignment())
    supported = (rank_stride == 1 and parts == 2) or parts * rank_stride == mesh_size
    if rank_stride < 1 or not supported or tensor.shape[dim] % parts:
        raise ValueError("rank_slice requires an evenly divided full-mesh dimension")
    selected_rank = dist.get_rank()
    if rank_to_device is not None:
        if sorted(rank_to_device) != list(range(mesh_size)):
            raise ValueError("rank_to_device must be a full-mesh permutation")
        selected_rank = rank_to_device[selected_rank]
    if rank != selected_rank // rank_stride % parts:
        raise ValueError("Rank inputs require full-mesh logical rank order")
    # RoPE positions are int64; JAX's default export would narrow them to int32.
    with jax.enable_x64():
        return _slice_op(rank_to_device)(tensor, traced, dim, parts, rank_stride)
