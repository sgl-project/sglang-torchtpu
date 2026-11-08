# SPDX-License-Identifier: Apache-2.0
"""TP all-gather built from the in-place all-reduce, for Dynamo-traced TPU graphs.

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

The workaround is scoped to the two vocabulary logits gathers in
``LogitsProcessor``: the TP gather (``_logits_gatherer``, which falls back to
``tensor_model_parallel_all_gather`` on TPU) and the attention-TP gather used
with ``--enable-dp-lm-head`` (``_gather_attn_tp_logits``). ``GroupCoordinator``
collectives stay native. Hooking them changed every per-layer gather (DP-attention
token gathers, fused-MoE token gathers), moved about 2x the bytes, and slowed
GLM-5.3-Flash DP8 by 8 to 14% (sgl-project/sglang-torchtpu#2013). A similar
compiled-gather fault on the DCP query-head gather is handled at its own call
site (``sglang_tpu/attention_pallas_rpa_dcp.py``).

The hook keys on ``torch.compiler.is_dynamo_compiling()``, so only graphs the
Dynamo tracer builds take this path. The direct tracer and eager calls keep the
native all-gather. The direct tracer never showed the defect, and it must not
take this path: ``blocks[group.rank_in_group]`` is a Python int that the direct
tracer bakes into the graph, and ``--use-spmd-compile`` ships one owner rank's
executable to every rank. Dynamo specializes each rank's graph on its own rank
(see ``sglang_tpu/rank_local.py``), so the constant is correct there.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from sglang.srt.runtime_context import get_parallel


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


def gather_last_dim_via_all_reduce(group: Any, x: torch.Tensor) -> torch.Tensor:
    """Concatenates every rank's ``x`` along its last dim, using one all-reduce."""
    world_size = group.world_size
    gathered = gather_blocks_via_all_reduce(group, x)
    gathered = gathered.reshape((world_size, *x.shape)).movedim(0, -2)
    return gathered.reshape((*x.shape[:-1], world_size * x.shape[-1]))


def _use_all_reduce_gather(group: Any) -> bool:
    return torch.compiler.is_dynamo_compiling() and group.world_size > 1


class TpuLogitsGatherer:
    """Wraps ``LogitsProcessor._logits_gatherer`` for the TP logits gather.

    Under Dynamo on a multi-rank TP group, gathers through the all-reduce;
    otherwise calls the wrapped gatherer.
    """

    def __init__(self, fallback: Callable[[torch.Tensor], torch.Tensor]):
        self._fallback = fallback

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        group = get_parallel().tp_group
        if not _use_all_reduce_gather(group):
            return self._fallback(logits)
        return gather_last_dim_via_all_reduce(group, logits)


def wrap_logits_gatherer(result: Any, processor: Any, *args: Any, **kwargs: Any):
    """AFTER hook for ``LogitsProcessor.__init__``."""
    gatherer = getattr(processor, "_logits_gatherer", None)
    if gatherer is not None and not isinstance(gatherer, TpuLogitsGatherer):
        processor._logits_gatherer = TpuLogitsGatherer(gatherer)
    return result


def gather_attn_tp_logits_around(
    original: Callable[..., torch.Tensor], processor: Any, logits: torch.Tensor
) -> torch.Tensor:
    """AROUND hook for ``LogitsProcessor._gather_attn_tp_logits``.

    Covers the ``--enable-dp-lm-head`` logits gather over the attention-TP
    group. Rows of zero and a vocabulary that does not divide evenly use the
    original, as do eager and direct-tracer calls.
    """
    group = get_parallel().attn_tp_group
    if (
        not _use_all_reduce_gather(group)
        or logits.shape[0] == 0
        or processor.vocab_size % group.world_size
    ):
        return original(processor, logits)
    return gather_last_dim_via_all_reduce(group, logits)
