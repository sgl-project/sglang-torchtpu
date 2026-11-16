# SPDX-License-Identifier: Apache-2.0
"""Sibling expert steal: the push protocol between the two cores of a chip.

Under the steal layout the two TensorCores of a chip own different EP8 expert
groups with the same TP4 quarter: kernel rank ``r`` holds group
``2 * (r // 8) + r % 2`` and quarter ``(r % 8) // 2`` (:func:`expert_group`,
pre-shard directory :func:`expert_directory`). After routing both cores hold
every row's experts, so each computes the same plan (:func:`steal_plan`): the
core whose group routes more distinct experts gives its
``(D_mine - D_sibling) // 2`` highest-id experts (at most ``slots``) to its
sibling. It pushes their weight slabs from its HBM into the sibling's HBM
staging slots; the sibling runs them after its own experts and adds them to
its partial sum, which the layer's reduction sums with every other rank's.

Protocol per layer, the same code on both cores (:class:`ExpertSteal`):

* ``begin(plan, layer)``: send the plan packet (always; it holds the give
  count, the layer and the given experts in slot order), then push the given
  slabs, one receive semaphore per staging slot.
* ``receive(expected)``: wait for the sibling's packet. Its give count and
  expert ids, not this core's own plan, decide which slabs to wait for and
  run, so a plan mismatch cannot hang (every core sends and waits for exactly
  one packet a layer, and the slab waits follow the sender's count) and the
  numerics stay right (the sender skips exactly what it declared). A packet
  that differs from the one this core expected counts as a mismatch.
* ``wait_slab(slot)``: wait for one staging slot's bytes.
* ``finish()``: wait for this core's own sends.

Staging slots are reused next layer only after the layer's reduction, which
orders the two cores (every rank waits for its sibling's partial).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

PACKET_SHAPE = (8, 128)
PACKET_MAGIC = 0x5715
# Packet rows: row 0 lane 0 the give count, lane 1 the layer, lane 2 the
# magic; row 1 lane j the local id of the expert in staging slot j.
(HEADER_ROW, IDS_ROW) = range(2)
(GIVE_LANE, LAYER_LANE, MAGIC_LANE) = range(3)


def expert_group(rank, steal_layout: bool = True):
    """The EP8 expert group of kernel ``rank`` (the stock layout: rank // 4)."""
    return 2 * (rank // 8) + rank % 2 if steal_layout else rank // 4


def expert_quarter(rank, steal_layout: bool = True):
    """The TP4 intermediate quarter of kernel ``rank``."""
    return (rank % 8) // 2 if steal_layout else rank % 4


def expert_directory(rank: int) -> int:
    """The pre-shard rank directory (group * 4 + quarter) that holds kernel
    ``rank``'s experts under the steal layout; a permutation within each host."""
    return 4 * expert_group(rank) + expert_quarter(rank)


def used_experts(winners, live_rows, group, local_expert_count):
    """[1, 128] bool: the local experts of ``group`` that a live row routes.

    ``winners`` [batch, top_k] global expert ids, ``live_rows`` a [batch]
    mask or None. Per route slot the ownership test and the live mask stay
    one-dimensional, as in the router: Mosaic rejects reshaping a [batch]
    mask loaded as a VMEM row into a [batch, 1] column."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    used = jnp.zeros((1, 128), jnp.bool_)
    for route in range(winners.shape[1]):
        column = winners[:, route]
        owned = column // local_expert_count == group
        if live_rows is not None:
            owned &= live_rows
        local = jnp.where(owned, column % local_expert_count, -1)
        used |= jnp.any(local[:, None] == lanes, axis=0, keepdims=True)
    return used


def _count(mask):
    return jnp.sum(mask.astype(jnp.int32))


def _higher_used(used):
    """[1, 128] int32: per lane, how many used lanes have a higher id (exact:
    a 0/1 product of at most 128 terms)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    columns = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    product = jax.lax.dot_general(
        jnp.broadcast_to(used.astype(jnp.bfloat16), (8, 128)),
        (rows > columns).astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return product[:1].astype(jnp.int32)


def _side(used, count):
    """The ``count`` highest-id used experts: their mask and staging slots
    (slot 0 the highest id)."""
    higher = _higher_used(used)
    moved = used & (higher < count)
    return moved, higher


def steal_plan(winners, live_rows, group, local_expert_count, slots):
    """Both cores' identical plan from every row's routed experts.

    ``winners`` [batch, top_k] global expert ids, ``group`` this core's expert
    group (the sibling holds ``group ^ 1``). ``give`` experts of this core's
    group move to the sibling, ``take`` of the sibling's move here."""
    mine = used_experts(winners, live_rows, group, local_expert_count)
    theirs = used_experts(winners, live_rows, group ^ 1, local_expert_count)
    d_mine, d_theirs = _count(mine), _count(theirs)
    give = jnp.minimum(jnp.maximum(d_mine - d_theirs, 0) // 2, slots)
    take = jnp.minimum(jnp.maximum(d_theirs - d_mine, 0) // 2, slots)
    given, given_slots = _side(mine, give)
    taken, taken_slots = _side(theirs, take)
    return SimpleNamespace(
        used=mine,
        given=given,
        given_slots=given_slots,
        give=give,
        taken=taken,
        taken_slots=taken_slots,
        take=take,
        d_mine=d_mine,
        d_theirs=d_theirs,
    )


def plan_packet(moved, moved_slots, count, layer, slots):
    """The [8, 128] int32 packet announcing ``count`` moved experts."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    ids = jnp.zeros((1, 128), jnp.int32)
    for slot in range(slots):
        ids += jnp.where(
            lanes == slot,
            jnp.sum(jnp.where(moved & (moved_slots == slot), lanes, 0)),
            0,
        )
    header = jnp.where(
        lanes == GIVE_LANE,
        count,
        jnp.where(
            lanes == LAYER_LANE, layer, jnp.where(lanes == MAGIC_LANE, PACKET_MAGIC, 0)
        ),
    )
    rows = jax.lax.broadcasted_iota(jnp.int32, PACKET_SHAPE, 0)
    return jnp.where(
        rows == HEADER_ROW, header, jnp.where(rows == IDS_ROW, ids, 0)
    ).astype(jnp.int32)


class ExpertSteal:
    """One core's side of the per-layer steal protocol.

    ``sources``: this core's expert weight families for the layer (HBM refs
    indexed by local expert id); ``staging``: the staging families
    ([slots, ...] HBM refs, written by the sibling); ``packets``: VMEM
    [2, 8, 128] int32 (0: sent, 1: received); ``packet_tables``: SMEM
    [2, 8, 128] int32 copies of them; ``semaphores``: DMA semaphores
    (packet send, packet receive, slab send, the packet table copies [2],
    slab receives [slots]); ``errors``: SMEM int32 [1] mismatch counter.
    """

    def __init__(
        self,
        sibling,
        sources,
        staging,
        packets,
        packet_tables,
        semaphores,
        errors,
    ):
        self.sibling = sibling
        self.sources = tuple(sources)
        self.staging = tuple(staging)
        self.packets = packets
        self.packet_tables = packet_tables
        (
            self.packet_send,
            self.packet_receive,
            self.slab_send,
            self.table_copies,
            self.slab_receives,
        ) = semaphores
        self.errors = errors
        self.slots = self.staging[0].shape[0]

    def _packet_copy(self):
        return tpu.make_async_remote_copy(
            self.packets.at[0],
            self.packets.at[1],
            self.packet_send,
            self.packet_receive,
            device_id=(self.sibling,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _table_copy(self, which):
        return tpu.make_async_copy(
            self.packets.at[which],
            self.packet_tables.at[which],
            self.table_copies.at[which],
        )

    def _slab_copies(self, expert, slot):
        return [
            tpu.make_async_remote_copy(
                source.at[expert],
                staged.at[slot],
                self.slab_send,
                self.slab_receives.at[slot],
                device_id=(self.sibling,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for source, staged in zip(self.sources, self.staging, strict=True)
        ]

    def begin(self, plan, layer):
        """Send the plan packet, then push the given experts' slabs."""
        self.packets[0] = plan_packet(
            plan.given, plan.given_slots, plan.give, layer, self.slots
        )
        self._packet_copy().start()
        table = self._table_copy(0)
        table.start()
        table.wait()
        give = self.packet_tables[0, HEADER_ROW, GIVE_LANE]

        @pl.loop(0, give)
        def _(slot):
            expert = self.packet_tables[0, IDS_ROW, slot]
            for copy in self._slab_copies(expert, slot):
                copy.start()

    def receive(self, expected_packet):
        """Wait for the sibling's packet; returns its (count, table) where
        ``table[IDS_ROW, j]`` is staging slot j's expert (sibling-local id).
        Counts a mismatch when the packet is not ``expected_packet``."""
        self._packet_copy().wait_recv()
        table = self._table_copy(1)
        table.start()
        received = self.packets[1]
        mismatch = jnp.any(received != expected_packet).astype(jnp.int32)
        table.wait()
        self.errors[0] += mismatch
        return self.packet_tables[1, HEADER_ROW, GIVE_LANE], self.packet_tables.at[1]

    def wait_slab(self, slot):
        for copy in self._slab_copies(0, slot):
            copy.wait_recv()

    def finish(self):
        """Wait for this core's sends (the packet and its pushed slabs)."""
        self._packet_copy().wait_send()
        give = self.packet_tables[0, HEADER_ROW, GIVE_LANE]

        @pl.loop(0, give)
        def _(slot):
            for copy in self._slab_copies(0, slot):
                copy.wait_send()


def scratch_shapes(slots):
    """The protocol's scratch, in :class:`ExpertSteal`'s order after the HBM
    refs: packets, packet tables, semaphores (as one tuple) and errors."""
    return [
        tpu.VMEM((2, *PACKET_SHAPE), jnp.int32),
        tpu.SMEM((2, *PACKET_SHAPE), jnp.int32),
        tpu.SemaphoreType.DMA(()),
        tpu.SemaphoreType.DMA(()),
        tpu.SemaphoreType.DMA(()),
        tpu.SemaphoreType.DMA((2,)),
        tpu.SemaphoreType.DMA((slots,)),
        tpu.SMEM((1,), jnp.int32),
    ]
