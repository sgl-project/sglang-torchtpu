# SPDX-License-Identifier: Apache-2.0
"""Fast TPU Pallas Markov draft loop for DeepSeek-V4.1 DSpark.

Replaces the 5x ``one_hot @ w1`` (``[bs, 129280] @ [129280, 256]``, reading 330 MB
of HBM per draft step) and FP32 ``base_logits`` materialization with an 8-row
tile-aligned Pallas DMA gather of ``w1[prev]`` (4 KB per batch row, 8-way
pipelined) plus fused BF16 ``base_logits`` addition and argmax.

When the target LM head is vocab-sharded across the attention-DP ranks
(``VocabParallelLMHead``), :func:`sharded_greedy_markov_pallas` additionally
fuses the vocab-parallel LM head projection with a vocab-sharded Markov loop:
each rank scores only its ``V / W`` slice against its sharded ``lm_head`` and
sharded ``markov_w2`` (8x less HBM read & compute) and exchanges only the
per-row ``(max_val, argmax_idx)`` (~8 KB) instead of all-to-alling the full
``[W, bs * gamma, V / W]`` BF16 logits (~50 MB at c=256).
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
import logging
from threading import Lock

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch

logger = logging.getLogger(__name__)

_OP_NAME = "sglang_tpu_pallas::dsv41_dspark_greedy_markov"
_OP = None
_OP_LOCK = Lock()

_AXIS = "model"
_SHARDED_OPS: dict[tuple[int, int, int, int, int], object] = {}
_SHARDED_LOCK = Lock()

SHARDED_W2_ATTR = "_tpu_sharded_markov_w2"


def _gather_w1_rows(w1: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather ``w1[indices]`` via 8-way pipelined, 8-row tile-aligned Pallas DMAs.

    ``w1`` is BF16 ``[V, R]`` (``V % 8 == 0``, ``R % 128 == 0``), ``indices``
    is INT32 ``[B]``. Returns BF16 ``[B, R]``. Avoids both XLA's buggy unsharded
    BF16 gather under TP1/DP8 and the 66 MB ``one_hot @ w1`` matmul.
    """
    bs = indices.shape[0]
    vocab, rank = w1.shape
    padded_bs = ((bs + 7) // 8) * 8
    padded_indices = (
        jnp.pad(indices, (0, padded_bs - bs), constant_values=0)
        if padded_bs > bs
        else indices
    )
    max_base = vocab - 8
    pipe = 8

    def kernel(idx_ref, w1_hbm, out_vmem, tmp_vmem, sem):
        sublanes = jnp.arange(8, dtype=jnp.int32)[:, None]
        g = pl.program_id(0)
        b0 = g * pipe
        offs = []
        copies = []
        for k in range(pipe):
            idx = jnp.clip(idx_ref[b0 + k], 0, vocab - 1)
            base = jnp.minimum((idx // 8) * 8, max_base)
            offs.append(idx - base)
            c = pltpu.make_async_copy(
                w1_hbm.at[pl.ds(base, 8), :],
                tmp_vmem.at[k, :, :],
                sem.at[k],
            )
            c.start()
            copies.append(c)
        for k in range(pipe):
            copies[k].wait()
            row = jnp.sum(
                jnp.where(
                    sublanes == offs[k],
                    tmp_vmem[k].astype(jnp.float32),
                    0.0,
                ),
                axis=0,
            ).astype(w1.dtype)
            out_vmem[k, :] = row

    gathered = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_bs, rank), w1.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(padded_bs // pipe,),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec((pipe, rank), lambda g, _: (g, 0)),
            scratch_shapes=(
                pltpu.VMEM((pipe, 8, rank), w1.dtype),
                pltpu.SemaphoreType.DMA((pipe,)),
            ),
        ),
        name="dspark_gather_w1_rows",
    )(padded_indices, w1)
    return gathered[:bs]


def _greedy_markov_jax(
    base_logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    anchor: jax.Array,
) -> jax.Array:
    """``[bs, gamma, V]`` BF16 logits, ``[V, R]`` BF16 w1/w2, ``[bs]`` i32 -> ``[bs, gamma]`` i32."""
    gamma = base_logits.shape[1]
    prev = anchor.astype(jnp.int32)
    drafts = []
    for step in range(gamma):
        latent = _gather_w1_rows(w1, prev)
        bias = jax.lax.dot_general(
            latent.astype(w2.dtype),
            w2,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        scores = base_logits[:, step, :].astype(jnp.float32) + bias
        prev = jnp.argmax(scores, axis=-1).astype(jnp.int32)
        drafts.append(prev)
    return jnp.stack(drafts, axis=1)


def _fake_greedy_markov(base_logits, w1, w2, anchor):
    return anchor.new_empty(
        (base_logits.shape[0], base_logits.shape[1]), dtype=torch.int32
    )


def _get_op():
    global _OP
    if _OP is not None:
        return _OP
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OP_LOCK:
        if _OP is None:
            op = cpu_export_jax_op(_OP_NAME, _greedy_markov_jax)
            op.register_fake(_fake_greedy_markov)
            _OP = op
    return _OP


def register_dspark_markov_op() -> None:
    """Register the Pallas Markov draft op before torch.compile traces."""
    _get_op()


def greedy_markov_pallas(
    base_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """Run the greedy Markov draft chain via Pallas DMA gather on TPU."""
    return _get_op()(
        base_logits.contiguous(),
        w1.contiguous(),
        w2.contiguous(),
        anchor.to(torch.int32).contiguous(),
    ).to(torch.int64)


# --------------------------------------------------------------------------- #
# Sharded-vocab LM head + greedy Markov draft (zero logit all-to-all).
# --------------------------------------------------------------------------- #
def _make_sharded_markov_fn(mesh, *, world: int, vocab: int, gamma: int):
    from jax.sharding import PartitionSpec as P

    assert vocab % world == 0
    shard = vocab // world

    def local(x, lm_w, w1, w2_shard, anchor):
        # x: [bs * gamma, D], lm_w: [shard, D], w1: [vocab, R],
        # w2_shard: [shard, R], anchor: [bs].
        bs = anchor.shape[0]
        width = x.shape[1]

        # Gather every DP rank's draft hidden states via all_to_all (safe under
        # DP8 multi-rank decode, same as vocab_parallel_head.py).
        copies = jnp.broadcast_to(x[None], (world, bs * gamma, width))
        every_x = jax.lax.all_to_all(
            copies.reshape(world * bs * gamma, width), _AXIS, 0, 0, tiled=True
        )
        # Local vocab slice logits: [W * bs, gamma, shard] BF16 — stays local!
        base_part = (
            jax.lax.dot_general(
                every_x,
                lm_w,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            .astype(jnp.bfloat16)
            .reshape(world * bs, gamma, shard)
        )

        # Gather all ranks' anchor tokens: [W * bs] INT32 (~1 KB).
        prev_all = jax.lax.all_gather(
            anchor.astype(jnp.int32), _AXIS, axis=0, tiled=True
        )

        drafts_all = []
        for step in range(gamma):
            latent = _gather_w1_rows(w1, prev_all)  # [W * bs, R] BF16
            bias_part = jax.lax.dot_general(
                latent.astype(w2_shard.dtype),
                w2_shard,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [W * bs, shard] FP32
            scores_part = base_part[:, step, :].astype(jnp.float32) + bias_part
            local_idx = jnp.argmax(scores_part, axis=-1).astype(jnp.int32)
            local_val = jnp.max(scores_part, axis=-1)

            # Exchange only per-row (max_val, local_argmax): ~8 KB instead of ~50 MB!
            # Avoid axis_index/partitionId (unsupported by XLA SPMD); recover
            # global vocab id from winning rank index * shard.
            all_vals = jax.lax.all_gather(local_val, _AXIS, axis=0, tiled=False)
            all_idxs = jax.lax.all_gather(local_idx, _AXIS, axis=0, tiled=False)
            best_rank = jnp.argmax(all_vals, axis=0)
            prev_all = (
                jnp.take_along_axis(all_idxs, best_rank[None, :], axis=0)[0]
                + best_rank.astype(jnp.int32) * shard
            )
            drafts_all.append(prev_all)

        # stacked is replicated [world, bs, gamma] (~5 KB). Use all_to_all to
        # route rank r's [bs, gamma] slice to rank r without axis_index /
        # partitionId (which XLA SPMD rejects).
        stacked = jnp.stack(drafts_all, axis=1).reshape(world, bs, gamma)
        mine = jax.lax.all_to_all(stacked, _AXIS, 0, 0, tiled=False)
        return mine[0]

    def sharded_markov(
        x: jax.Array,
        lm_w: jax.Array,
        w1: jax.Array,
        w2_shard: jax.Array,
        anchor: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                P(_AXIS, None),
                P(_AXIS, None),
                P(),
                P(_AXIS, None),
                P(_AXIS),
            ),
            out_specs=P(_AXIS, None),
            check_vma=False,
        )(x, lm_w, w1, w2_shard, anchor)

    in_specs = (P(_AXIS, None), P(_AXIS, None), P(), P(_AXIS, None), P(_AXIS))
    return sharded_markov, in_specs, (P(_AXIS, None),)


def _build_sharded_op(world: int, vocab: int, width: int, rank: int, gamma: int):
    from torch_tpu._internal.pallas import get_pallas_mesh
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh((_AXIS,), (world,))
    fn, in_specs, out_specs = _make_sharded_markov_fn(
        mesh, world=world, vocab=vocab, gamma=gamma
    )
    name = (
        f"sglang_tpu_pallas::dsv41_sharded_dspark_markov"
        f"_w{world}_v{vocab}_d{width}_r{rank}_g{gamma}"
    )
    op = cpu_export_jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
    )
    op.register_fake(
        lambda x, lm_w, w1, w2_s, anchor: anchor.new_empty(
            (anchor.shape[0], gamma), dtype=torch.int32
        )
    )
    return op


def get_sharded_markov_op(
    world: int, vocab: int, width: int, rank: int, gamma: int
):
    key = (int(world), int(vocab), int(width), int(rank), int(gamma))
    op = _SHARDED_OPS.get(key)
    if op is not None:
        return op
    with _SHARDED_LOCK:
        op = _SHARDED_OPS.get(key)
        if op is None:
            op = _build_sharded_op(*key)
            _SHARDED_OPS[key] = op
    return op


def sharded_greedy_markov_pallas(
    x: torch.Tensor,
    lm_w: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    anchor: torch.Tensor,
    *,
    world: int,
    vocab: int,
    gamma: int,
) -> torch.Tensor:
    """Fused vocab-parallel LM head + sharded-vocab greedy Markov draft on TPU."""
    op = get_sharded_markov_op(
        world=world,
        vocab=vocab,
        width=int(lm_w.shape[1]),
        rank=int(w1.shape[1]),
        gamma=gamma,
    )
    return op(
        x.contiguous(),
        lm_w,
        w1,
        w2_shard,
        anchor.to(torch.int32).contiguous(),
    ).to(torch.int64)


def prepare_model_for_dsv41_sharded_markov(model) -> bool:
    """Shard ``markov_head.markov_w2`` across DP ranks to match ``VocabParallelLMHead``
    and pre-register the sharded Markov op before torch.compile traces."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41.vocab_parallel_head import VocabParallelLMHead

    if not tpu_envs.SGLANG_TPU_DSV41_SHARDED_DSPARK_MARKOV.get():
        return False
    head = getattr(model, "markov_head", None)
    lm_head = getattr(model, "lm_head", None)
    if head is None or lm_head is None:
        return False
    if getattr(head, SHARDED_W2_ATTR, False):
        return True
    if not isinstance(lm_head, VocabParallelLMHead):
        return False
    if getattr(model, "_use_fp32_lm_head", False):
        return False

    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world = int(parallel.world_group.world_size)
    rank = int(parallel.world_group.rank_in_group)
    vocab = int(head.vocab_size)
    gamma = int(model.gamma)
    w1 = head.markov_w1.weight
    w2 = head.markov_w2.weight
    if (
        world <= 1
        or w2.device.type != "tpu"
        or w1.shape[0] != vocab
        or w2.shape[0] != vocab
        or vocab != int(lm_head.full_vocab)
        or vocab % world != 0
        or (vocab // world) != int(lm_head.weight.shape[0])
        or vocab % 8 != 0
        or w1.shape[1] % 128 != 0
    ):
        return False

    shard = vocab // world
    start = rank * shard
    with torch.no_grad():
        old = w2.data
        # clone() ensures storage_offset == 0 on every rank (required by SPMD shared compile).
        sliced = old[start : start + shard].clone().contiguous()
        head.markov_w2.weight = torch.nn.Parameter(sliced, requires_grad=False)
        storage = old.untyped_storage()
        if storage.data_ptr() != sliced.untyped_storage().data_ptr():
            try:
                storage.resize_(0)
            except RuntimeError:
                pass
        del old

    get_sharded_markov_op(
        world=world,
        vocab=vocab,
        width=int(lm_head.weight.shape[1]),
        rank=int(w1.shape[1]),
        gamma=gamma,
    )
    setattr(head, SHARDED_W2_ATTR, True)
    setattr(head, "_tpu_sharded_markov_world", world)
    logger.info(
        "SGLANG_TPU_DSV41_SHARDED_DSPARK_MARKOV: sharded markov_w2 to [%d, %d] "
        "(world=%d, rank=%d, gamma=%d).",
        shard,
        int(w1.shape[1]),
        world,
        rank,
        gamma,
    )
    return True
