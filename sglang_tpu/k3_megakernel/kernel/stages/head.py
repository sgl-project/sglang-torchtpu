# SPDX-License-Identifier: Apache-2.0
"""Greedy LM head inside the stack kernel (one decode row).

Each rank holds ``lm_head`` rows ``[V / world, H]`` (its vocab shard). The rows stream
through a two-slot VMEM ring in ``chunk``-row pieces; every piece is one NT dot of the
final hidden row against it and a running first-max argmax. The ranks then exchange a
packed ``(max, global id)`` row in one one-shot all-gather and every rank picks the
largest logit, the lowest id among equal ones (the XLA path's ``argmax`` order).
"""

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import Topology
from sglang_tpu.k3_megakernel.kernel.collectives import AllGather, AllGatherSpec

_LANE = 128
_BIG = 3.0e38


def head_chunk(vocab_shard: int, hidden: int, budget: int) -> int:
    """Largest multiple of 128 rows dividing ``vocab_shard`` whose two slots fit ``budget`` bytes."""
    fits = [
        c
        for c in range(_LANE, vocab_shard + 1, _LANE)
        if vocab_shard % c == 0 and 2 * c * hidden * 2 <= budget
    ]
    if not fits:
        raise ValueError(f"no lm_head chunk of {vocab_shard} rows fits {budget} bytes")
    return max(fits)


class HeadSpec(msgspec.Struct, frozen=True):
    topo: Topology
    vocab_shard: int
    hidden: int
    chunk: int

    @property
    def exchange(self) -> AllGatherSpec:
        return AllGatherSpec(self.topo, 1, jnp.float32, flat=True, cols=_LANE)

    @property
    def n_chunks(self) -> int:
        return self.vocab_shard // self.chunk

    def scratch_shapes(self) -> list:
        return [pltpu.SemaphoreType.DMA((2,))] + self.exchange.scratch_shapes()


class GreedyHead:
    """Binds a ``HeadSpec`` to the weight ``head_hbm`` ``[V / world, H]``, a ``[2, chunk, H]``
    bf16 ring view and the scratch of ``HeadSpec.scratch_shapes``."""

    def __init__(self, spec: HeadSpec, me, head_hbm, ring, sems, *exchange_scratch):
        self.spec, self.me, self.w, self.ring, self.sems = (
            spec,
            me,
            head_hbm,
            ring,
            sems,
        )
        self.exchange = AllGather(spec.exchange, me, *exchange_scratch)

    def init(self) -> None:
        self.exchange.init()

    def _copy(self, i):
        c = self.spec.chunk
        return pltpu.make_async_copy(
            self.w.at[pl.ds(i * c, c)], self.ring.at[i % 2], self.sems.at[i % 2]
        )

    def prefetch(self) -> None:
        """Starts the first two chunks (the ring must be free)."""
        for i in range(min(2, self.spec.n_chunks)):
            self._copy(i).start()

    def _local_best(self, x):
        """``x`` ``[1, H]`` bf16 -> (max logit, its global id) ``[1, 1]`` f32 over this shard."""
        spec = self.spec
        col = jax.lax.broadcasted_iota(jnp.int32, (1, spec.chunk), 1).astype(
            jnp.float32
        )
        base = (self.me * spec.vocab_shard).astype(jnp.float32)
        best = jnp.full((1, 1), -_BIG, jnp.float32)
        best_id = jnp.zeros((1, 1), jnp.float32)
        for i in range(spec.n_chunks):
            self._copy(i).wait()
            logits = jax.lax.dot_general(
                x,
                self.ring[i % 2],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [1, chunk]
            if i + 2 < spec.n_chunks:
                self._copy(i + 2).start()
            m = jnp.max(logits, axis=1, keepdims=True)
            idx = jnp.min(jnp.where(logits == m, col, _BIG), axis=1, keepdims=True) + (
                base + i * spec.chunk
            )
            take = m > best  # strict: an earlier chunk keeps a tie
            best, best_id = jnp.where(take, m, best), jnp.where(take, idx, best_id)
        return best, best_id

    def token(self, x) -> jax.Array:
        """Greedy token id (0-d int32) of the final hidden row ``x`` ``[1, H]`` bf16."""
        best, best_id = self._local_best(x)
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, _LANE), 1)
        packed = jnp.where(lane == 0, best, jnp.where(lane == 1, best_id, 0.0))
        row = self.exchange(packed)  # [1, world * 128], rank order
        world = self.spec.topo.world
        rows = jnp.concatenate(
            [row[:, r * _LANE : (r + 1) * _LANE] for r in range(world)], axis=0
        )  # [world, 128]
        lanes = jax.lax.broadcasted_iota(jnp.int32, rows.shape, 1)
        vals = jnp.sum(jnp.where(lanes == 0, rows, 0.0), axis=1, keepdims=True)
        ids = jnp.sum(jnp.where(lanes == 1, rows, 0.0), axis=1, keepdims=True)
        top = jnp.max(vals)
        return jnp.min(jnp.where(vals == top, ids, _BIG)).astype(jnp.int32)
