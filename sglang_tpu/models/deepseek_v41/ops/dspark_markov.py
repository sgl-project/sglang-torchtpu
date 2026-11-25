# SPDX-License-Identifier: Apache-2.0
"""Fast TPU Pallas Markov draft loop for DeepSeek-V4.1 DSpark.

Replaces the 5x ``one_hot @ w1`` (``[bs, 129280] @ [129280, 256]``, reading 330 MB
of HBM per draft step) and FP32 ``base_logits`` materialization with an 8-row
tile-aligned Pallas DMA gather of ``w1[prev]`` (4 KB per batch row) plus fused
BF16 ``base_logits`` addition and argmax.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

from threading import Lock

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch

_OP_NAME = "sglang_tpu_pallas::dsv41_dspark_greedy_markov"
_OP = None
_OP_LOCK = Lock()


def _gather_w1_rows(w1: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather ``w1[indices]`` via 8-row tile-aligned Pallas DMA copies.

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

    def kernel(idx_ref, w1_hbm, out_vmem, tmp_vmem, sem):
        sublanes = jnp.arange(8, dtype=jnp.int32)[:, None]
        for b in range(padded_bs):
            idx = jnp.clip(idx_ref[b], 0, vocab - 1)
            base = jnp.minimum((idx // 8) * 8, max_base)
            off = idx - base
            copy = pltpu.make_async_copy(
                w1_hbm.at[pl.ds(base, 8), :], tmp_vmem, sem.at[0]
            )
            copy.start()
            copy.wait()
            row = jnp.sum(
                jnp.where(sublanes == off, tmp_vmem[...].astype(jnp.float32), 0.0),
                axis=0,
            ).astype(w1.dtype)
            out_vmem[b, :] = row

    gathered = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_bs, rank), w1.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec((padded_bs, rank), lambda _, __: (0, 0)),
            scratch_shapes=(
                pltpu.VMEM((8, rank), w1.dtype),
                pltpu.SemaphoreType.DMA((1,)),
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
    return anchor.new_empty((base_logits.shape[0], base_logits.shape[1]), dtype=torch.int32)


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
