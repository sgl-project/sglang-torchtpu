# SPDX-License-Identifier: Apache-2.0
"""DSpark greedy Markov drafts over the vocab split across the attention-DP
ranks (SGLANG_TPU_DSV41_DSPARK_MARKOV_SHARD).

``dspark.greedy_markov_drafts`` reads the whole Markov tables at every draft
step: the w1 row lookup is a one-hot matmul over the full vocab and the bias is
``latent @ w2.T`` (2 x ~66 MB per step, ~0.3 ms a cycle at c4/c32), after the
vocab-parallel head has already all-to-all'd full-vocab logits back to every
rank. Here, as the vocab-parallel head does for the logits, rank i holds vocab
rows [i*V/W, (i+1)*V/W) of the head, w1 and w2, and every rank works on every
rank's rows:

1. gather every rank's draft rows and anchor ids;
2. score them against the local head slice, once for all gamma steps;
3. per step: the owner of each previous token contributes its w1 row (local
   one-hot, psum -> the exact latent), every rank adds its slice of the bias
   and takes its local argmax, and the (value, index) pairs are gathered; the
   global pick is the largest value, ties to the lowest rank, i.e. the lowest
   vocab index, as torch.argmax's first maximum;
4. return each rank's own rows.

Each logit and bias is the same dot product as the unsharded path, rounded the
same way, so the drafts are equal on CPU; on TPU the smaller matmuls may reorder
the FP32 reductions (the vocab-parallel head's caveat), so a near-tie can flip.
The collectives pair across ranks: the DP lockstep draft forward calls it on
every rank with the same row count, as for the vocab-parallel head.

Note the Markov tables are sliced at load; the rank offset enters as a sharded
input rather than an axis index.
"""

import functools
import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

_AXIS = "model"
_W1 = "_tpu_markov_w1_slice"
_W2 = "_tpu_markov_w2_slice"
_RANK = "_tpu_markov_rank"
_LOGGED: set = set()


def make_markov(mesh, world: int, vocab: int, gamma: int):
    """The SPMD drafts over ``mesh``: (normed rows [W*bs*gamma, D] bf16, anchor
    ids [W*bs] int32, rank [W] int32, head [V, D], w1 [V, R], w2 [V, R]), all
    sharded on axis 0, -> drafts [W*bs, gamma] int32 sharded on axis 0."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    assert vocab % world == 0, f"vocab {vocab} is not divisible by {world} ranks"
    shard = vocab // world

    def gather(v):
        """Every rank's ``v`` stacked on a new axis 0: the vocab-parallel head's
        all-to-all of W copies (XLA's all-gather was unsafe there)."""
        copies = jnp.broadcast_to(v[None], (world, *v.shape))
        flat = copies.reshape(world * v.shape[0], *v.shape[1:])
        every = jax.lax.all_to_all(flat, _AXIS, 0, 0, tiled=True)
        return every.reshape(world, *v.shape)

    def local(x, prev, rank, w, w1, w2):
        rows, width = x.shape
        bs = prev.shape[0]
        every = gather(x).reshape(world * rows, width)
        prev = gather(prev).reshape(world * bs)
        # As the vocab-parallel head: FP32 dot rounded once to BF16, then FP32.
        base = (
            jax.lax.dot_general(
                every, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        base = base.reshape(world * bs, gamma, shard)
        offset = rank[0] * shard
        lanes = jnp.arange(shard, dtype=jnp.int32)
        drafts = []
        for step in range(gamma):
            one_hot = ((prev - offset)[:, None] == lanes[None, :]).astype(w1.dtype)
            # One owner contributes its row, the others zeros: the sum is exact.
            part = jax.lax.dot_general(
                one_hot,
                w1,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            latent = jax.lax.psum(part, _AXIS).astype(w1.dtype)
            bias = (
                jax.lax.dot_general(
                    latent.astype(w2.dtype),
                    w2,
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                .astype(w2.dtype)
                .astype(jnp.float32)
            )
            score = base[:, step] + bias
            value = jnp.max(score, axis=-1)
            index = jnp.argmax(score, axis=-1).astype(jnp.int32) + offset
            values, indices = gather(value), gather(index)
            best = jnp.argmax(values, axis=0)  # first maximum: the lowest rank
            prev = jnp.take_along_axis(indices, best[None], axis=0)[0]
            drafts.append(prev)
        mine = jnp.stack(drafts, axis=1)
        return jax.lax.dynamic_slice(mine, (rank[0] * bs, 0), (bs, gamma))

    def drafts(x, prev, rank, w, w1, w2):
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                P(_AXIS, None),
                P(_AXIS),
                P(_AXIS),
                P(_AXIS, None),
                P(_AXIS, None),
                P(_AXIS, None),
            ),
            out_specs=P(_AXIS, None),
            check_vma=False,
        )(x, prev, rank, w, w1, w2)

    return drafts


@functools.cache
def _build_op(
    world: int, vocab: int, width: int, rank_dim: int, gamma: int
) -> Callable:
    """The torch op of ``make_markov``; builds the mesh, so call it at load."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh((_AXIS,), (world,))
    fn = make_markov(mesh, world, vocab, gamma)

    def _wrapped(
        x: jax.Array,
        prev: jax.Array,
        rank: jax.Array,
        w: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
    ) -> jax.Array:
        return fn(x, prev, rank, w, w1, w2)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::dsv41_dspark_markov_w{world}_v{vocab}_d{width}_r{rank_dim}_g{gamma}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(
            P(_AXIS, None),
            P(_AXIS),
            P(_AXIS),
            P(_AXIS, None),
            P(_AXIS, None),
            P(_AXIS, None),
        ),
        output_partition_specs=(P(_AXIS, None),),
    )


def prepare_markov_shard(model) -> bool:
    """Slice this rank's vocab rows of the draft's Markov tables into buffers and
    build the op, at load (outside any trace). Returns whether it is ready."""
    import torch.distributed as dist

    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_DSV41_DSPARK_MARKOV_SHARD.get():
        return False
    head = model.markov_head
    vocab = int(head.vocab_size)
    world, rank = dist.get_world_size(), dist.get_rank()
    w1 = head.markov_w1.weight
    if vocab % world or w1.device.type != "tpu":
        logger.info(
            "DSpark Markov shard off: vocab %d over %d ranks on %s",
            vocab,
            world,
            w1.device,
        )
        return False
    shard = vocab // world
    rows = slice(rank * shard, (rank + 1) * shard)
    head.register_buffer(_W1, w1.data[rows].contiguous().clone(), persistent=False)
    head.register_buffer(
        _W2, head.markov_w2.weight.data[rows].contiguous().clone(), persistent=False
    )
    head.register_buffer(
        _RANK,
        torch.tensor([rank], dtype=torch.int32, device=w1.device),
        persistent=False,
    )
    lm_width = int(model.config.hidden_size)
    _build_op(world, vocab, lm_width, int(w1.shape[1]), int(model.gamma))
    logger.info(
        "DSpark Markov shard: rank %d holds vocab rows [%d, %d) of w1/w2",
        rank,
        rows.start,
        rows.stop,
    )
    return True


def applies(model, bs: int) -> bool:
    """Whether the sharded drafts serve this draft call; logged once per bs."""
    from sglang_tpu.models.deepseek_v41.vocab_parallel_head import VocabParallelLMHead

    used = isinstance(model.lm_head, VocabParallelLMHead) and (
        getattr(model.markov_head, _W1, None) is not None
    )
    key = (int(bs), used)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info("[dsv41] site=dspark_markov_shard rows=%d used=%s", key[0], used)
    return used


def sharded_drafts(model, normed: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """[bs, gamma] int64 drafts from this rank's normed [bs * gamma, D] rows and
    [bs] int32 anchor ids; call only where ``applies``."""
    import torch.distributed as dist

    head = model.markov_head
    lm_head = model.lm_head
    w1 = getattr(head, _W1)
    op = _build_op(
        dist.get_world_size(),
        int(head.vocab_size),
        int(lm_head.weight.shape[1]),
        int(w1.shape[1]),
        int(model.gamma),
    )
    out = op(
        normed.to(torch.bfloat16),
        prev.to(torch.int32),
        getattr(head, _RANK),
        lm_head.weight,
        w1,
        getattr(head, _W2),
    )
    return out.to(torch.int64)


def refresh_markov_shard(head) -> None:
    """Copy this rank's rows of the live w1/w2 into the existing slice buffers in
    place, so compiled programs that captured the buffers see the new values."""
    w1 = getattr(head, _W1)
    shard = int(w1.shape[0])
    start = int(getattr(head, _RANK).cpu()[0]) * shard
    w1.copy_(head.markov_w1.weight.data[start : start + shard])
    getattr(head, _W2).copy_(head.markov_w2.weight.data[start : start + shard])


def tpu_dsv41_dspark_refresh_markov_after_direct_update(result, model, named_tensors):
    """AFTER _model_load_weights_direct: the raw update API bypasses
    post_load_weights, so refresh the slices of any Markov table it wrote."""
    heads = {}
    for name, _ in named_tensors:
        module_path, _, param = name.rpartition(".")
        head_path, _, table = module_path.rpartition(".")
        if param != "weight" or table not in ("markov_w1", "markov_w2"):
            continue
        head = model.get_submodule(head_path) if head_path else model
        if getattr(head, _W1, None) is not None:
            heads[id(head)] = head
    for head in heads.values():
        refresh_markov_shard(head)
    return result


def tpu_dsv41_dspark_prepare_markov_after_load(result, model, *args, **kwargs):
    """AFTER DeepseekV4ForCausalLMDSpark.post_load_weights."""
    prepare_markov_shard(model)
    return result
