# SPDX-License-Identifier: Apache-2.0
"""Sibling expert steal: the push protocol between the two cores of a chip.

Under the steal layout the two TensorCores of a chip own different EP8 expert
groups with the same TP4 quarter: kernel rank ``r`` holds group
``2 * (r // 8) + r % 2`` and quarter ``(r % 8) // 2`` (:func:`expert_group`,
pre-shard directory :func:`expert_directory`; ``KernelConfig.expert_steal``
puts the decode router, the rank loader and the EP8 prefill in this layout,
:func:`owned_expert_group`). After routing both cores hold
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

In the decode megakernel (variant B, ``KernelConfig.expert_steal`` at the
grouped B8/B16 steps) the receiver runs the stolen experts as local experts
``local_expert_count + slot`` after its own (:func:`steal_routes`), each into
its own FP32 accumulator, and returns those weighted outputs
(:meth:`ExpertSteal.send_returns`); the giver adds them after its own experts
in ascending id order. The given experts are its highest-id ones, so every
row adds the same terms in the same order as without the steal: the outputs
are bit-identical to the steal layout alone.
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


def owned_expert_group(rank, expert_tensor_parallel: int, steal_layout: bool):
    """The expert group (of ``32 // expert_tensor_parallel``) whose experts
    kernel ``rank`` holds: ``rank // expert_tensor_parallel``, or at EP8/TP4
    in the steal layout (``KernelConfig.expert_steal``) :func:`expert_group`.
    The decode router's ownership test and the EP8 prefill's expert offset."""
    if steal_layout and expert_tensor_parallel == 4:
        return expert_group(rank)
    return rank // expert_tensor_parallel


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


def steal_slots(batch_size: int) -> int:
    """Staging slots (the most experts one core gives its sibling a layer):
    2 at 8 rows, 4 at 16."""
    return 2 if batch_size <= 8 else 4


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


def steal_routes(
    winners,
    probabilities,
    live_rows,
    group,
    local_expert_count,
    given,
    taken_ids,
    taken_count,
):
    """The local routes under the steal, in the router's layout.

    ``winners`` / ``probabilities`` [batch, top_k] are every row's selections
    in top-k order. A route is local when its expert is one of ``group``'s
    except the ``given`` ones ([1, 128] mask of local ids; it keeps its local
    id), or the sibling group's expert in staging slot ``j < taken_count``
    (``taken_ids`` [1, 128]: lane j the expert's sibling-local id), which runs
    as local id ``local_expert_count + j``. Routes fill each row's slots in
    selection order. Returns (selected experts [batch, top_k], probabilities,
    route counts [batch]); with nothing given or taken they equal the
    router's own ownership pass.
    """
    batch_size, route_count = winners.shape
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    route_slots = jnp.arange(route_count)[None, :]
    given_lanes = given.astype(jnp.float32)
    taken_lanes = jnp.where(lanes < taken_count, taken_ids, -1)
    lane_values = lanes.astype(jnp.float32)
    selected_experts = jnp.zeros((batch_size, route_count), jnp.int32)
    selected_probabilities = jnp.zeros((batch_size, route_count), jnp.float32)
    route_counts = jnp.zeros((batch_size,), jnp.int32)
    for route in range(route_count):
        column = winners[:, route]
        expert_group = column // local_expert_count
        local = column % local_expert_count
        # Lane sums of 0/1 terms and lane ids (< 128): exact in FP32.
        given_hit = (
            jnp.sum(jnp.where(local[:, None] == lanes, given_lanes, 0.0), axis=1) > 0
        )
        slot_hit = local[:, None] == taken_lanes
        taken_hit = jnp.sum(jnp.where(slot_hit, 1.0, 0.0), axis=1) > 0
        taken_slot = jnp.sum(jnp.where(slot_hit, lane_values, 0.0), axis=1).astype(
            jnp.int32
        )
        own = expert_group == group
        owned = (own & ~given_hit) | ((expert_group == (group ^ 1)) & taken_hit)
        if live_rows is not None:
            owned &= live_rows
        expert = jnp.where(own, local, local_expert_count + taken_slot)
        destination = owned[:, None] & (route_slots == route_counts[:, None])
        selected_experts = jnp.where(destination, expert[:, None], selected_experts)
        selected_probabilities = jnp.where(
            destination, probabilities[:, route][:, None], selected_probabilities
        )
        route_counts = route_counts + owned.astype(jnp.int32)
    return selected_experts, selected_probabilities, route_counts


class ExpertSteal:
    """One core's side of the per-layer steal protocol.

    ``sources``: this core's expert weight families for the layer (HBM refs
    indexed by local expert id); ``staging``: the staging families
    ([slots, ...] HBM refs, written by the sibling); ``packets``: VMEM
    [2, 8, 128] int32 (0: sent, 1: received); ``packet_tables``: SMEM
    [2, 8, 128] int32 copies of them; ``semaphores``: DMA semaphores
    (packet send, packet receive, slab send, the packet table copies [2],
    slab receives [slots]); ``errors``: SMEM int32 [1] mismatch counter.

    Variant B (the decode megakernel) adds ``ready`` (SMEM int32 [slots]:
    staging slot j waited for this layer, :meth:`stage`), ``returns`` (VMEM
    [slots, ...] FP32, the sibling's returned outputs of this core's given
    experts) and ``return_semaphores`` (send, receive).
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
        *,
        ready=None,
        returns=None,
        return_semaphores=(None, None),
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
        self.ready = ready
        self.returns = returns
        self.return_send, self.return_receive = return_semaphores

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
        if self.ready is not None:
            for slot in range(self.slots):
                self.ready[slot] = 0
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

    def stage(self, slot):
        """Wait for staging slot ``slot``'s bytes once a layer (``ready``)."""

        @pl.when(self.ready[slot] == 0)
        def _():
            self.wait_slab(slot)
            self.ready[slot] = 1

    def _return_copy(self, outputs, slot):
        return tpu.make_async_remote_copy(
            outputs.at[slot],
            self.returns.at[slot],
            self.return_send,
            self.return_receive,
            device_id=(self.sibling,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def send_returns(self, count, outputs):
        """Return the stolen experts' outputs (``outputs[j]``, j < ``count``,
        the received packet's count) into the sibling's ``returns``. A slot
        the stream did not stage (only after a plan mismatch) is waited for
        and returned as zeros."""
        assert outputs.shape[1:] == self.returns.shape[1:], "return shapes"

        @pl.loop(0, count)
        def _(slot):
            @pl.when(self.ready[slot] == 0)
            def unstaged():
                self.wait_slab(slot)
                self.ready[slot] = 1
                outputs.at[slot][...] = jnp.zeros(outputs.shape[1:], outputs.dtype)

            self._return_copy(outputs, slot).start()

    def wait_returns(self, give):
        """Wait for the sibling's returns of this core's ``give`` experts."""

        @pl.loop(0, give)
        def _(slot):
            self._return_copy(self.returns, slot).wait_recv()

    def finish(self):
        """Wait for this core's sends (the packet, its pushed slabs and, with
        returns, the received packet's returned outputs)."""
        self._packet_copy().wait_send()
        give = self.packet_tables[0, HEADER_ROW, GIVE_LANE]

        @pl.loop(0, give)
        def _(slot):
            for copy in self._slab_copies(0, slot):
                copy.wait_send()

        if self.returns is not None:

            @pl.loop(0, self.packet_tables[1, HEADER_ROW, GIVE_LANE])
            def _(slot):
                self._return_copy(self.returns, slot).wait_send()


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


def return_scratch_shapes(slots, batch_size, width):
    """Variant B's scratch after :func:`scratch_shapes`: the ready flags, the
    returned outputs ([slots, batch, width] FP32) and the return semaphores
    (send, receive)."""
    return [
        tpu.SMEM((slots,), jnp.int32),
        tpu.VMEM((slots, batch_size, width), jnp.float32),
        tpu.SemaphoreType.DMA(()),
        tpu.SemaphoreType.DMA(()),
    ]
