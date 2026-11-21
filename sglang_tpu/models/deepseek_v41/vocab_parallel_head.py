# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 LM head split over the vocab across the attention-DP ranks.

Under attention DP at attention TP 1 every rank holds the whole head [V, D]
(1.32 GB in BF16 for V4.1-Flash) and reads all of it at every target verify and
every DSpark draft call, only to score its own few rows. Split, rank i holds the
vocab rows [i*V/W, (i+1)*V/W):

1. every rank's rows are gathered on every rank ([W*T, D]);
2. each rank scores them against its vocab slice ([W*T, V/W]);
3. an all-to-all returns each rank its own T rows over every slice, laid side
   by side into [T, V].

Each logit is the same dot product over D, accumulated in FP32 and rounded once
to BF16: bit-equal on CPU, but the smaller TPU matmul may reorder the FP32
reduction, so a value near a BF16 rounding boundary can move by one ulp
(vllm-torchtpu measured 493 of 5,171,200 logits). The collectives pair across
ranks: every rank must call the head with the same row count, which the DP
lockstep buckets (and the idle-rank mirror of a decode step) guarantee for the
compiled target and draft forwards. An eager head call on one rank would hang:
the prompt-logprob gather after a compiled prefill (compiled_forward_shim) is
one, so serving prompt logprobs needs the whole head (the TPU spec workers
reject logprobs outright).

Ported from vllm-torchtpu models/vllm/deepseek_v41/common/vocab_parallel_head.py.
"""

import functools
import logging
from collections.abc import Callable

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

logger = logging.getLogger(__name__)

_AXIS = "model"


class VocabParallelLMHead(ParallelLMHead):
    """A ParallelLMHead whose ``weight`` holds only this rank's vocab slice.

    The class of the target's head is swapped in place (shard_lm_head), so every
    other attribute (org_vocab_size, tp_size 1, quant_method) still describes
    the whole vocab the logits cover.
    """

    head_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    full_vocab: int


def make_head(mesh, world: int, vocab: int):
    """The SPMD head over ``mesh``: (rows [W*T, D] bf16, weight [V, D] bf16),
    both sharded on axis 0, -> logits [W*T, V] bf16 sharded on axis 0."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    assert vocab % world == 0, f"vocab {vocab} is not divisible by {world} ranks"
    shard = vocab // world

    def local(x, w):
        rows, width = x.shape
        # The gather of every rank's rows as an all-to-all of W copies: XLA's
        # all-gather returned wrong logits in a batched-prefill graph
        # (SGLANG_TPU_LOGITS_GATHER_ALLREDUCE), and its SparseCore offload was
        # unsafe on multi-rank decode in vllm-torchtpu; the all-to-all is neither.
        copies = jnp.broadcast_to(x[None], (world, rows, width))
        every = jax.lax.all_to_all(
            copies.reshape(world * rows, width), _AXIS, 0, 0, tiled=True
        )
        part = jax.lax.dot_general(
            every,
            w,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        # [owner rank, T, V/W] -> [vocab slice, T, V/W] -> [T, V].
        mine = jax.lax.all_to_all(
            part.reshape(world, rows, shard), _AXIS, 0, 0, tiled=True
        )
        return jnp.transpose(mine, (1, 0, 2)).reshape(rows, vocab)

    def head(x, w):
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(_AXIS, None), P(_AXIS, None)),
            out_specs=P(_AXIS, None),
            check_vma=False,
        )(x, w)

    return head


@functools.cache
def _build_head_op(world: int, vocab: int, width: int) -> Callable:
    """The torch op of ``make_head``; builds the mesh, so call it at load time on
    every rank, never first inside a trace."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    # get_pallas_mesh is rank-aligned: mesh position i is distributed rank i.
    mesh = get_pallas_mesh((_AXIS,), (world,))
    head = make_head(mesh, world, vocab)

    def _wrapped(x: jax.Array, w: jax.Array) -> jax.Array:
        return head(x, w)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::dsv41_vocab_parallel_head_w{world}_v{vocab}_d{width}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(P(_AXIS, None), P(_AXIS, None)),
        output_partition_specs=(P(_AXIS, None),),
    )


def whole_head_reason(*, model, lm_head, parallel, world_size: int) -> str | None:
    """Why the target's ``lm_head`` must stay whole, or None to split it."""
    if parallel.attn_dp_size <= 1 or parallel.attn_tp_size != 1:
        return (
            "it needs attention DP at attention TP 1, got attn_dp_size "
            f"{parallel.attn_dp_size} attn_tp_size {parallel.attn_tp_size}"
        )
    if parallel.tp_size != world_size or parallel.pp_size != 1:
        return (
            f"its mesh spans the world of {world_size} ranks in rank order; got "
            f"tp_size {parallel.tp_size} pp_size {parallel.pp_size}"
        )
    if type(lm_head) is not ParallelLMHead:
        return f"the head is a {type(lm_head).__name__}, not a plain ParallelLMHead"
    if lm_head is model.model.embed_tokens:
        return "the head is tied to the input embedding"
    if lm_head.tp_size != 1:
        return f"the head is already sharded over {lm_head.tp_size} ranks"
    if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
        return "the head is quantized"
    if lm_head.bias is not None:
        return "the head has a bias"
    if model.logits_processor.use_fp32_lm_head:
        return "the logits processor projects in FP32"
    weight = lm_head.weight
    vocab = weight.shape[0]
    if (
        weight.dtype != torch.bfloat16
        or vocab % world_size
        or vocab != lm_head.org_vocab_size
    ):
        return (
            f"it needs an unpadded BF16 vocab divisible by {world_size}; got "
            f"{tuple(weight.shape)} {weight.dtype}, org vocab {lm_head.org_vocab_size}"
        )
    return None


def shard_lm_head(model) -> bool:
    """Keep only this rank's vocab slice of ``model.lm_head`` where the head can
    be split; returns whether it was split. Run before the KV pool is sized and
    before a draft shares the head."""
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    lm_head = model.lm_head
    world = dist.get_world_size()
    reason = whole_head_reason(
        model=model, lm_head=lm_head, parallel=get_parallel(), world_size=world
    )
    if reason is not None:
        logger.info("DeepSeek-V4.1 LM head kept whole: %s", reason)
        return False
    rank = dist.get_rank()
    vocab, width = lm_head.weight.shape
    shard = vocab // world
    op = _build_head_op(world, vocab, width)
    piece = lm_head.weight.data[rank * shard : (rank + 1) * shard].clone()
    lm_head.__class__ = VocabParallelLMHead
    lm_head.weight = torch.nn.Parameter(piece, requires_grad=False)
    lm_head.head_op = op
    lm_head.full_vocab = vocab
    logger.info(
        "DeepSeek-V4.1 LM head vocab-parallel: rank %d holds vocab rows [%d, %d) of %d",
        rank,
        rank * shard,
        (rank + 1) * shard,
        vocab,
    )
    return True


def vocab_parallel_logits(
    lm_head: VocabParallelLMHead, hidden: torch.Tensor
) -> torch.Tensor:
    """[T, D] rows of this rank -> [T, V] BF16 logits over the whole vocab."""
    return lm_head.head_op(hidden.to(torch.bfloat16), lm_head.weight)
