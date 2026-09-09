# SPDX-License-Identifier: Apache-2.0

"""Static schedule selection for the fused TP16 prefill kernel.

The token count is part of a JAX array's static shape, so schedule selection can
happen in ordinary Python before Pallas tracing.  The resulting frozen plan is
captured by the Pallas call: loop counts, scratch shapes, and routing tables are
compile-time constants in the generated TPU program.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ._contract import TP_SIZE

MIN_TUNED_TOKENS = 1024
MAX_TUNED_TOKENS = 16384


def _build_ici_schedule() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Build the three-wave nearest-neighbor schedule for a 2x2x2 slice."""
    ici_masks = (2, 4, 8)
    steps = []
    for distance in range(1, len(ici_masks) + 1):
        link_load = {mask: 0 for mask in ici_masks}
        entries = []
        for combination in itertools.combinations(ici_masks, distance):
            offset = 0
            for mask in combination:
                offset |= mask
            link_mask = min(combination, key=lambda mask: (link_load[mask], mask))
            link_load[link_mask] += 1
            entries.append((link_mask, offset ^ link_mask))
        steps.append(tuple(sorted(entries)))
    return tuple(steps)


@dataclass(frozen=True)
class AllToAllPlan:
    """Project local rows for every destination and exchange completed tiles."""

    name: str
    collective_id: int
    local_rows: int
    row_block_rows: int
    destination_orders: tuple[tuple[int, ...], ...]
    # Q occupies the first 512 destination columns; K and V complete a
    # 640-column QKV tile, and OG occupies the remaining 512 columns. This
    # natural split needs two matmuls and three output DMAs per destination.
    projection_widths: tuple[int, int] = (640, 512)
    norm_block_rows: int = 64
    # Eight in-flight Q/OG payloads leave enough independent MXU work to hide
    # output latency without consuming the VMEM needed by 640-column weights.
    send_slots: int = 8
    weight_buffers: int = 2
    # Alternate raw BF16 projection buffers so destination d can project while
    # destination d-1 runs its VPU-heavy attention epilogue and output copies.
    projection_buffers: int = 2
    vmem_limit_bytes: int = 64 * 1024 * 1024


@dataclass(frozen=True)
class HybridGatherAllToAllPlan:
    """Gather source rows before projecting and exchanging destination tiles."""

    name: str  # Human-readable schedule name used in benchmark output.
    collective_id: int  # Stable barrier channel shared by equivalent invocations.
    local_rows: int  # Token rows initially owned by each rank.
    destinations_per_rank: int  # Output destinations projected by each rank.
    # Gather waves. Each entry identifies the peer link and source slot to
    # forward as ``(link_mask, slot_xor)``.
    routing_steps: tuple[tuple[tuple[int, int], ...], ...]
    # Canonical source offsets in the order they are packed in gather scratch.
    packed_source_offsets: tuple[int, ...]
    # ``(first_source, source_count)`` projected after each gather wave.
    projection_groups: tuple[tuple[int, int], ...]
    input_link_masks: tuple[int, ...]  # Peer links used by the input gather.
    projection_slots: int  # Scratch result slots rotated by output rounds.
    # Destination-link order for each source position; local destination is 0.
    output_orders: tuple[tuple[int, ...], ...]
    vmem_limit_bytes: int  # Maximum VMEM allocation accepted by the launcher.

    @property
    def source_shards_per_rank(self) -> int:
        return TP_SIZE // self.destinations_per_rank


SchedulePlan = AllToAllPlan | HybridGatherAllToAllPlan

# Mosaic embeds the barrier channel in the compiled custom-call body. Keep this
# namespace finite and schedule-owned: a layer-owned ID would turn every layer
# into a separate TorchTPU/Pallas specialization. Different synchronization
# patterns retain different IDs, as required by Pallas collective semantics.
_GATHER_8_SOURCES_COLLECTIVE_ID = 100
_GATHER_2_SOURCES_COLLECTIVE_ID = 101
_ALL_TO_ALL_COLLECTIVE_ID = 102
_ALL_TO_ALL_GRID_COLLECTIVE_ID = 103


def supports_token_count(global_tokens: int) -> bool:
    """Whether ``global_tokens`` has a tuned prefill schedule.

    The optimized implementation is limited to the 1K--16K prefill range.
    Token counts must also map to one of the static TP16 local-row schedules
    selected by :func:`make_schedule_plan`.
    """
    if not MIN_TUNED_TOKENS <= global_tokens <= MAX_TUNED_TOKENS:
        return False
    if global_tokens % TP_SIZE:
        return False
    local_rows = global_tokens // TP_SIZE
    return local_rows in (64, 128) or (local_rows >= 256 and local_rows % 64 == 0)


def _eight_source_gather_steps() -> tuple[tuple[tuple[int, int], ...], ...]:
    nearby_steps = _build_ici_schedule()[:2]
    # After the second wave, the farthest source is available through all three
    # neighboring chips.  Stripe its hidden columns across those three links.
    farthest_source_step = ((2, 12), (4, 10), (8, 6))
    return nearby_steps + (farthest_source_step,)


_TWO_SOURCE_OUTPUT_ORDERS = (
    # Start empirically slower chip routes first so their DMA tails overlap
    # later MXU work.  Mask 0 is the local chip and therefore runs last.
    (7, 3, 5, 6, 1, 2, 4, 0),
    (6, 7, 1, 5, 3, 4, 2, 0),
)

_ALL_TO_ALL_DESTINATION_ORDERS = (
    # The two core parities use different permutations to spread traffic over
    # ICI links.  Long routes start first and the local copy remains last.
    (15, 7, 11, 13, 3, 5, 9, 14, 6, 10, 12, 2, 4, 8, 1, 0),
    (13, 15, 3, 11, 7, 9, 5, 12, 14, 2, 10, 6, 8, 4, 1, 0),
)


def make_schedule_plan(local_rows: int) -> SchedulePlan:
    """Return the measured-fastest static plan for ``local_rows``."""
    if local_rows == 64:
        return HybridGatherAllToAllPlan(
            name="gather-8-sources",
            collective_id=_GATHER_8_SOURCES_COLLECTIVE_ID,
            local_rows=local_rows,
            destinations_per_rank=2,
            routing_steps=_eight_source_gather_steps(),
            packed_source_offsets=(0, 1, 2, 4, 3, 5, 6, 7),
            projection_groups=((0, 1), (1, 3), (4, 3), (7, 1)),
            input_link_masks=(2, 4, 8),
            projection_slots=1,
            output_orders=(),
            vmem_limit_bytes=64 * 1024 * 1024,
        )
    if local_rows == 128:
        return HybridGatherAllToAllPlan(
            name="gather-2-sources",
            collective_id=_GATHER_2_SOURCES_COLLECTIVE_ID,
            local_rows=local_rows,
            destinations_per_rank=8,
            routing_steps=(((1, 0),),),
            packed_source_offsets=(0, 1),
            projection_groups=((0, 1), (1, 1)),
            input_link_masks=(1,),
            projection_slots=8,
            output_orders=_TWO_SOURCE_OUTPUT_ORDERS,
            vmem_limit_bytes=62 * 1024 * 1024,
        )
    if local_rows < 256 or local_rows > 1024 or local_rows % 64:
        raise ValueError(
            "supported local rows are 64, 128, or 256--1024 in multiples of 64"
        )
    return AllToAllPlan(
        name="all-to-all-grid" if local_rows > 512 else "all-to-all",
        collective_id=(
            _ALL_TO_ALL_GRID_COLLECTIVE_ID
            if local_rows > 512
            else _ALL_TO_ALL_COLLECTIVE_ID
        ),
        local_rows=local_rows,
        row_block_rows=min(local_rows, 512),
        destination_orders=_ALL_TO_ALL_DESTINATION_ORDERS,
    )
