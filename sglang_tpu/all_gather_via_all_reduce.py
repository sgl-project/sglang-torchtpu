# SPDX-License-Identifier: Apache-2.0
"""TP all-gather built from the in-place all-reduce, for Dynamo-traced TPU graphs.

Under the Dynamo tracer, ``GroupCoordinator.all_gather_into_tensor`` lowers to
the ``reg_all_gather_into_tensor`` custom op. On a group that spans more than
one chip (tpu7x TP >= 4), the compiled program returns wrong blocks for the
cross-chip peers: every rank keeps its own block (and its same-chip neighbour's),
while the other blocks hold unrelated data. The LM head's vocabulary gather then
mixes in garbage logits for most of the vocabulary
(sgl-project/sglang-torchtpu#1700). The in-place all-reduce custom op that every
decoder layer already uses compiles correctly, so the gather is rebuilt from it:
each rank writes its block into a zero ``[world, *block]`` buffer at its own
index and the group sums the buffers. Adding zeros leaves every value unchanged,
except that a ``-0.0`` comes back as ``+0.0``.

Only Dynamo-traced calls on a group that spans more than one chip take this
path. Eager mode, the direct tracer (it sets ``torch.compiler.is_compiling()``
but not ``is_dynamo_compiling()``) and single-chip groups keep the native
collective.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

# Group ranks -> whether they sit on more than one chip.
_SPANS_CHIPS: dict[tuple[int, ...], bool] = {}


def gather_blocks_via_all_reduce(group: Any, block: torch.Tensor) -> torch.Tensor:
    """Concatenates every rank's ``block`` along dim 0, using one all-reduce.

    Args:
      group: A ``GroupCoordinator``-like object with ``world_size``,
        ``rank_in_group`` and an ``all_reduce(tensor) -> tensor`` that sums over
        the group.
      block: This rank's block. Every rank passes the same shape and dtype.

    Returns:
      A ``[world_size * block.shape[0], *block.shape[1:]]`` tensor with the
      values ``all_gather_into_tensor`` writes into its output (``-0.0``
      becomes ``+0.0``).
    """
    world_size = group.world_size
    blocks = block.new_zeros((world_size, *block.shape))
    blocks[group.rank_in_group] = block
    blocks = group.all_reduce(blocks)
    return blocks.reshape(world_size * block.shape[0], *block.shape[1:])


def _count_chips(ranks: tuple[int, ...]) -> int:
    """The number of distinct chips that hold the world ranks ``ranks``."""
    from sglang_tpu.pp_stage_chips import _stage_core_locations

    return len({(loc.slice_index, loc.coords) for loc in _stage_core_locations(ranks)})


@torch.compiler.assume_constant_result
def spans_multiple_chips(ranks: tuple[int, ...]) -> bool:
    """Whether the world ranks ``ranks`` sit on more than one chip.

    Dynamo evaluates this once per group while tracing and bakes the answer
    in. When the runtime reports no chip locations, the group is assumed to
    span chips: the all-reduce gather is correct everywhere, only slower.
    """
    spans = _SPANS_CHIPS.get(ranks)
    if spans is None:
        try:
            spans = _count_chips(ranks) > 1
        except Exception:  # noqa: BLE001 - any lookup failure means "unknown"
            spans = True
        _SPANS_CHIPS[ranks] = spans
    return spans


def all_gather_into_tensor_around(
    original: Callable[..., Any],
    group: Any,
    output: torch.Tensor,
    input: torch.Tensor,  # noqa: A002 - mirrors GroupCoordinator's signature
) -> None:
    """AROUND hook for ``GroupCoordinator.all_gather_into_tensor``."""
    if (
        not torch.compiler.is_dynamo_compiling()
        or group.world_size == 1
        or not spans_multiple_chips(tuple(group.ranks))
    ):
        return original(group, output, input)
    output.copy_(gather_blocks_via_all_reduce(group, input).view(output.shape))
    return None
