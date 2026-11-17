# SPDX-License-Identifier: Apache-2.0
"""Sibling expert steal: the push protocol between the two cores of a chip.

Under the steal layout the two TensorCores of a chip own different EP8 expert
groups with the same TP4 quarter: kernel rank ``r`` holds group
``2 * (r // 8) + r % 2`` and quarter ``(r % 8) // 2`` (:func:`expert_group`,
pre-shard directory :func:`expert_directory`; ``KernelConfig.expert_steal``
puts the decode router, the rank loader and the EP8 prefill in this layout,
:func:`owned_expert_group`). After routing both cores hold
every row's experts, so each computes the same plan (:class:`StealPlanner`,
:func:`steal_plan`): the core whose group routes more distinct experts gives its
``(D_mine - D_sibling) // 2`` highest-id experts (at most ``slots``) to its
sibling. It pushes their weight slabs from its HBM into the sibling's HBM
staging slots; the sibling runs them after its own experts and adds them to
its partial sum, which the layer's reduction sums with every other rank's.

Protocol per layer, the same code on both cores (:class:`ExpertSteal`):

* ``begin(plan, layer)``: send the plan packet (always; it holds the give
  count, the layer and the given experts in slot order), then push the given
  slabs, one receive semaphore per staging slot.
* ``receive(expected)`` (or ``receive_packet``, which leaves the packet's
  SMEM table copy in flight): wait for the sibling's packet. Its give count and
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
``local_expert_count + slot`` after its own (:meth:`StealPlanner.routes`),
each into its own FP32 accumulator, and returns those weighted outputs
(:meth:`ExpertSteal.send_returns`); the giver adds them after its own experts
in ascending id order. The given experts are its highest-id ones, so every
row adds the same terms in the same order as without the steal: the outputs
are bit-identical to the steal layout alone.
"""

from __future__ import annotations

import math
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


# Route value of a route slot that runs no expert here under the steal's route
# layout (StealPlanner.routes): above every expert lane, so neither the expert
# schedule nor the initial expert copies pick it.
UNROUTED = 255


def route_code(winners, live_rows, group, local_expert_count):
    """Per route (``winners``: global expert ids, any shape): its local id when
    ``group`` holds the expert, local id + 128 when the sibling group
    (``group ^ 1``) does, else -1 (also for rows outside ``live_rows``, a mask
    of ``winners``' shape or None). Integer compares and selects only."""
    pair_local = winners - (group // 2) * (2 * local_expert_count)
    high = pair_local >= local_expert_count
    local = jnp.where(high, pair_local - local_expert_count, pair_local)
    keep = (pair_local >= 0) & (pair_local < 2 * local_expert_count)
    if live_rows is not None:
        keep &= live_rows
    return jnp.where(keep, jnp.where(high == (group % 2 == 1), local, local + 128), -1)


def steal_slots(batch_size: int) -> int:
    """Staging slots (the most experts one core gives its sibling a layer):
    2 at 8 rows, 4 at 16."""
    return 2 if batch_size <= 8 else 4


class StealPlan(SimpleNamespace):
    """Both cores' identical plan (:meth:`StealPlanner.plan`), int32 scalars:
    ``d_mine`` / ``d_theirs`` the distinct experts this core's and the
    sibling's group route, ``give`` / ``take`` the experts this core gives /
    takes and ``ids[s]`` (s < ``slots``) the moving experts' local ids in
    staging-slot order (the heavier group's highest used ids, slot 0 the
    highest; valid for s < give + take)."""

    def gives(self, local):
        """Whether this core gives local expert(s) ``local``: exactly the
        experts its packet declares, also when its plan differs from the
        sibling's."""
        given = jnp.zeros(jnp.shape(local), jnp.bool_)
        for slot, expert in enumerate(self.ids):
            given |= (local == expert) & (slot < self.give)
        return given


class StealPlanner:
    """The steal plan and routes, built one top-k selection at a time.

    ``add(selection, winners, probabilities)`` (every row's winner of one
    selection, [batch]) marks each live route's expert in this core's or the
    sibling's used-expert lanes and keeps the routes' codes
    (:func:`route_code`) and probabilities by selection slot. The decode
    router calls it inside its top-k loop, where it fills the argmax chain's
    idle slots. :meth:`plan` then counts the distinct experts of each group
    and finds the moving experts with a few lane reductions, and
    :meth:`routes` gives the local routes from the codes with elementwise ops.
    """

    def __init__(
        self,
        batch_size,
        top_k,
        live_rows,
        group,
        local_expert_count,
        slots,
        timing=None,
    ):
        self.top_k, self.live_rows, self.group = top_k, live_rows, group
        self.local_expert_count, self.slots = local_expert_count, slots
        self.timing = timing
        self.lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        self.route_slots = jax.lax.broadcasted_iota(jnp.int32, (1, top_k), 1)
        self.mine = jnp.zeros((batch_size, 128), jnp.bool_)
        self.theirs = jnp.zeros((batch_size, 128), jnp.bool_)
        self.codes = jnp.full((batch_size, top_k), -1, jnp.int32)
        self.probabilities = jnp.zeros((batch_size, top_k), jnp.float32)

    def add(self, selection, winners, probabilities=None):
        code = route_code(winners, self.live_rows, self.group, self.local_expert_count)
        code = code[:, None]
        self.mine |= code == self.lanes
        self.theirs |= code == self.lanes + 128
        here = self.route_slots == selection
        self.codes = jnp.where(here, code, self.codes)
        if probabilities is not None:
            self.probabilities = jnp.where(
                here, probabilities[:, None], self.probabilities
            )

    @classmethod
    def from_winners(
        cls, winners, live_rows, group, local_expert_count, slots, probabilities=None
    ):
        """The planner of whole [batch, top_k] selections."""
        planner = cls(*winners.shape, live_rows, group, local_expert_count, slots)
        for selection in range(winners.shape[1]):
            planner.add(
                selection,
                winners[:, selection],
                None if probabilities is None else probabilities[:, selection],
            )
        return planner

    def plan(self):
        """The core whose group routes more distinct experts gives its
        highest-id ones: ``(D_mine - D_sibling) // 2`` (at most ``slots``),
        or with ``timing`` the count that its timing model finishes first
        (:func:`modelled_moves`)."""
        lanes = self.lanes.astype(jnp.float32)
        # Lane counts and ids below 128: exact FP32 reductions.
        used_mine = jnp.any(self.mine, axis=0, keepdims=True).astype(jnp.float32)
        used_theirs = jnp.any(self.theirs, axis=0, keepdims=True).astype(jnp.float32)
        d_mine = jnp.sum(used_mine).astype(jnp.int32)
        d_theirs = jnp.sum(used_theirs).astype(jnp.int32)
        difference = d_mine - d_theirs
        moved = modelled_moves(d_mine, d_theirs, self.slots, self.timing)
        giving = (difference > 0).astype(jnp.float32)
        heavy = used_theirs + giving * (used_mine - used_theirs)
        candidates = jnp.where(heavy > 0, lanes, -1.0)
        ids = []
        for _ in range(self.slots):
            top = jnp.max(candidates)
            ids.append(top.astype(jnp.int32))
            candidates = jnp.where(lanes < top, candidates, -1.0)
        give = jnp.where(difference > 0, moved, 0)
        return StealPlan(
            used=used_mine > 0,
            d_mine=d_mine,
            d_theirs=d_theirs,
            give=give,
            take=moved - give,
            ids=ids,
        )

    def routes(self, plan, taken_ids, taken_count):
        """The local routes under the steal: [batch, top_k] experts in
        selection slots (UNROUTED where a slot runs nothing here), their
        probabilities, and the route count (top_k for a row with any local
        route, else 0; every slot below it is checked against the expert).

        A route is local when its expert is this core's group's except the
        ones ``plan`` gives, or the sibling's expert ``taken_ids[s]`` with
        s < ``taken_count`` (staging slot s), which runs as local id
        ``local_expert_count + s``. The expert schedule and the stream ignore
        route order within a row."""
        codes = self.codes
        mine = (codes >= 0) & (codes < 128)
        theirs = codes >= 128
        local = jnp.where(theirs, codes - 128, codes)
        owned = mine & ~plan.gives(local)
        slot_of = jnp.zeros(codes.shape, jnp.int32)
        for slot, expert in enumerate(taken_ids):
            hit = theirs & (local == expert) & (slot < taken_count)
            owned |= hit
            slot_of = jnp.where(hit, slot, slot_of)
        experts = jnp.where(
            owned,
            jnp.where(mine, local, self.local_expert_count + slot_of),
            UNROUTED,
        )
        counts = jnp.where(jnp.any(owned, axis=1), self.top_k, 0)
        return experts, self.probabilities, counts


def modelled_moves(d_mine, d_theirs, slots, timing=None):
    """How many experts the heavier core gives (works on scalars or numpy).

    Without ``timing``: ``|D_mine - D_theirs| // 2``, at most ``slots``. With
    ``timing`` = (slab, lead, copy, back), in tenths of one expert's stream
    time: the k <= that bound whose modelled layer end is earliest, fewer on
    ties. The giver ends after ``D_heavy - k`` experts; the receiver after
    its ``D_light`` own experts or, if later, when the k slabs have landed
    (``slab * k`` after the plan, ``lead`` of it before its stream starts)
    and been copied in (``copy * k``), then the k stolen experts, and its
    returns reach the giver ``back`` after that."""
    bound = abs(d_mine - d_theirs) // 2
    moved = min(bound, slots) if isinstance(bound, int) else jnp.minimum(bound, slots)
    if timing is None:
        return moved
    slab, lead, copy, back = timing
    maximum, minimum, where = (
        (max, min, lambda c, a, b: a if c else b)
        if isinstance(bound, int)
        else (jnp.maximum, jnp.minimum, jnp.where)
    )
    light, heavy = minimum(d_mine, d_theirs), maximum(d_mine, d_theirs)
    moved, best = 0, 10 * heavy
    for count in range(1, slots + 1):
        receiver = maximum(10 * light, (slab + copy) * count - lead)
        cost = maximum(receiver + 10 * count + back, 10 * (heavy - count))
        better = (count <= bound) & (cost < best)
        moved, best = where(better, count, moved), where(better, cost, best)
    return moved


def steal_plan(winners, live_rows, group, local_expert_count, slots):
    """Both cores' identical plan from every row's routed experts.

    ``winners`` [batch, top_k] global expert ids, ``group`` this core's expert
    group (the sibling holds ``group ^ 1``)."""
    return StealPlanner.from_winners(
        winners, live_rows, group, local_expert_count, slots
    ).plan()


def packet(count, layer, ids, slots):
    """The [8, 128] int32 plan packet announcing ``count`` moved experts
    (``ids[s]``: staging slot s's local expert id)."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    id_row = jnp.zeros((1, 128), jnp.int32)
    for slot in range(slots):
        id_row = jnp.where((lanes == slot) & (slot < count), ids[slot], id_row)
    header = jnp.where(
        lanes == GIVE_LANE,
        count,
        jnp.where(
            lanes == LAYER_LANE, layer, jnp.where(lanes == MAGIC_LANE, PACKET_MAGIC, 0)
        ),
    )
    rows = jax.lax.broadcasted_iota(jnp.int32, PACKET_SHAPE, 0)
    return jnp.where(
        rows == HEADER_ROW, header, jnp.where(rows == IDS_ROW, id_row, 0)
    ).astype(jnp.int32)


def packet_fields(value, slots):
    """A packet value's count and staging-slot expert ids (int32 scalars)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, PACKET_SHAPE, 0)
    lanes = jax.lax.broadcasted_iota(jnp.int32, PACKET_SHAPE, 1)
    values = value.astype(jnp.float32)  # small integers: exact

    def field(row, lane):
        return jnp.sum(jnp.where((rows == row) & (lanes == lane), values, 0.0)).astype(
            jnp.int32
        )

    return field(HEADER_ROW, GIVE_LANE), [field(IDS_ROW, slot) for slot in range(slots)]


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
    """The local routes under the steal, in the router's layout (compacted
    in selection order; the decode megakernel uses StealPlanner.routes).

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
    [2, 8, 128] int32 (0: the sent count and ids, 1: a copy of the received
    packet); ``semaphores``: DMA semaphores
    (packet send, packet receive, slab send, the packet table copies [2],
    slab receives [slots]); ``errors``: SMEM int32 [1] mismatch counter.

    Variant B (the decode megakernel) adds ``ready`` (SMEM int32 [slots]:
    staging slot j waited for this layer, :meth:`stage`), ``returns`` (VMEM
    [slots, ...] FP32, the sibling's returned outputs of this core's given
    experts), ``return_semaphores`` (send, receive) and optionally ``landed``
    (:meth:`landed`).
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
        landed=None,
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
        self.landed_hint = landed

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
        """Send the plan packet, then push the given experts' slabs (slot 0
        first). The packet table 0 in SMEM holds the give count and the given
        ids, stored from the plan's scalars."""
        if self.ready is not None:
            for slot in range(self.slots):
                self.ready[slot] = 0
        self.packets[0] = packet(plan.give, layer, plan.ids, self.slots)
        self._packet_copy().start()
        table = self.packet_tables.at[0]
        table[HEADER_ROW, GIVE_LANE] = plan.give
        for slot in range(self.slots):
            table[IDS_ROW, slot] = plan.ids[slot]

        @pl.loop(0, plan.give)
        def _(slot):
            for copy in self._slab_copies(table[IDS_ROW, slot], slot):
                copy.start()

    def receive_packet(self, expected_packet):
        """Wait for the sibling's packet and start its copy to packet table 1
        (:meth:`wait_table` before reading it). Returns the packet and whether
        it differs from ``expected_packet`` (counted as a mismatch)."""
        self._packet_copy().wait_recv()
        self._table_copy(1).start()
        received = self.packets[1]
        mismatch = jnp.any(received != expected_packet)
        self.errors[0] += mismatch.astype(jnp.int32)
        return received, mismatch

    def wait_table(self):
        self._table_copy(1).wait()

    def receive(self, expected_packet):
        """Wait for the sibling's packet; returns its (count, table) where
        ``table[IDS_ROW, j]`` is staging slot j's expert (sibling-local id).
        Counts a mismatch when the packet is not ``expected_packet``."""
        self.receive_packet(expected_packet)
        self.wait_table()
        return self.packet_tables[1, HEADER_ROW, GIVE_LANE], self.packet_tables.at[1]

    def wait_slab(self, slot):
        for copy in self._slab_copies(0, slot):
            copy.wait_recv()

    def landed(self, slot):
        """Whether staging slot ``slot``'s bytes have all arrived: its receive
        semaphore against the slab size in DMA counter units (32 bytes, as the
        decode MoE's shared-reduction poll counts them). Only a hint for when
        to start reading the slot; :meth:`wait_slab` still consumes it.
        ``landed`` (constructor) stands in for the poll where semaphores
        cannot be read (interpret mode)."""
        if self.landed_hint is not None:
            return self.landed_hint(slot)
        units = sum(
            math.prod(family.shape[1:]) * family.dtype.itemsize
            for family in self.staging
        )
        return pl.semaphore_read(self.slab_receives.at[slot]) >= units // 32

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
