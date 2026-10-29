# SPDX-License-Identifier: Apache-2.0
"""TP all-gather built from the in-place all-reduce, for compiled TPU graphs.

Under the Dynamo tracer, ``GroupCoordinator.all_gather_into_tensor`` lowers to
the ``reg_all_gather_into_tensor`` custom op. On a TP group that spans more than
one chip (tpu7x TP >= 4), the compiled program returns wrong blocks for the
cross-chip peers: every rank keeps its own block (and its same-chip neighbour's),
while the other blocks hold unrelated data. The LM head's vocabulary gather then
mixes in garbage logits for most of the vocabulary
(sgl-project/sglang-torchtpu#1700). The in-place all-reduce custom op that every
decoder layer already uses compiles correctly, so the gather is rebuilt from it:
each rank writes its block into a zero ``[world, *block]`` buffer at its own
index and the group sums the buffers. Adding zero leaves every value unchanged,
so the result equals a correct all-gather; the only bit-level difference is that
a -0.0 element comes back as +0.0.

The hook keys on ``torch.compiler.is_compiling()``, which both TPU tracers set
while tracing (the direct tracer sets ``torch.compiler._is_compiling_flag`` in
``sglang_tpu/compilation/direct_compile.py``). So compiled graphs from either
tracer use this path, and eager calls keep the native all-gather. The direct
tracer did not show the defect, and the rewrite is value-preserving there too.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def gather_blocks_via_all_reduce(group: Any, block: torch.Tensor) -> torch.Tensor:
    """Concatenates every rank's ``block`` along dim 0, using one all-reduce.

    Args:
      group: A ``GroupCoordinator``-like object with ``world_size``,
        ``rank_in_group`` and an ``all_reduce(tensor) -> tensor`` that sums over
        the group.
      block: This rank's block. Every rank passes the same shape and dtype.

    Returns:
      A ``[world_size * block.shape[0], *block.shape[1:]]`` tensor, equal to what
      ``all_gather_into_tensor`` writes into its output.
    """
    world_size = group.world_size
    blocks = block.new_zeros((world_size, *block.shape))
    blocks[group.rank_in_group] = block
    blocks = group.all_reduce(blocks)
    return blocks.reshape(world_size * block.shape[0], *block.shape[1:])


def all_gather_into_tensor_around(
    original: Callable[..., Any],
    group: Any,
    output: torch.Tensor,
    input: torch.Tensor,  # noqa: A002 - mirrors GroupCoordinator's signature
) -> None:
    """AROUND hook for ``GroupCoordinator.all_gather_into_tensor``.

    Only traced calls (``torch.compiler.is_compiling()``, set by both TPU
    tracers) on a multi-rank group take the all-reduce path; eager calls and
    single-rank groups call the original collective.
    """
    if not torch.compiler.is_compiling() or group.world_size == 1:
        return original(group, output, input)
    output.copy_(gather_blocks_via_all_reduce(group, input).view(output.shape))
    return None
