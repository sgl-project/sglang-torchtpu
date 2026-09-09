# SPDX-License-Identifier: Apache-2.0

"""Hybrid gather all-to-all schedules for small token counts.

These schedules trade a small activation gather for taller matmuls. A static
``HybridGatherAllToAllPlan`` describes either the 1K eight-source/two-
destination transpose or the 2K two-source/eight-destination transpose.
Python conditionals on that plan are resolved while Pallas traces the program;
they are not runtime TPU branches.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._contract import (
    COLUMNS_PER_DESTINATION,
    HEAD_DIM,
    HIDDEN_SIZE,
    KEY_COLUMNS,
    QUERY_COLUMNS,
    TP_SIZE,
    VALUE_COLUMNS,
)
from ._epilogue import gathered_interleaved_rope
from ._schedule import HybridGatherAllToAllPlan


def _kernel_hybrid_gather_a2a(
    rank_metadata_ref,
    x_ref,
    weight_hbm_ref,
    gain_ref,
    rope_ref,
    out_hbm_ref,
    *,
    axis_name: str,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    plan: HybridGatherAllToAllPlan,
):
    """Gather normalized source rows, project them, then exchange outputs."""
    local_rows, hidden_size = x_ref.shape
    destination_count = plan.destinations_per_rank
    source_count = plan.source_shards_per_rank
    output_link_masks = tuple(range(1, destination_count))
    assert rope_ref.shape == (local_rows, 2 * HEAD_DIM)
    assert (local_rows, destination_count, source_count) in (
        (64, 2, 8),
        (128, 8, 2),
    )

    # The eight-source plan loads both destination weights as one 2304-column
    # tile. The two-source plan streams one 1152-column destination at a time.
    strided_destination_groups = destination_count == 8
    weight_partition_count = destination_count if strided_destination_groups else 1
    weight_width = destination_count * COLUMNS_PER_DESTINATION // weight_partition_count
    projection_width = (
        plan.projection_slots * weight_width
        if strided_destination_groups
        else destination_count * COLUMNS_PER_DESTINATION
    )
    second_weight_shape = (
        (hidden_size, weight_width) if destination_count == 8 else (8, 128)
    )
    farthest_source_shape = (
        (2 * local_rows, COLUMNS_PER_DESTINATION)
        if destination_count == 2
        else (8, 128)
    )
    gather_copy_count = sum(len(step) for step in plan.routing_steps)

    @pl.with_scoped(
        # Preserve FP32 residual RMSNorm semantics, but gather only the BF16
        # normalized rows consumed by the MXU.
        gathered_x=pltpu.VMEM((source_count * local_rows, hidden_size), jnp.bfloat16),
        gathered_rope=pltpu.VMEM(
            (source_count * local_rows, 2 * HEAD_DIM), rope_ref.dtype
        ),
        weight0=pltpu.VMEM((hidden_size, weight_width), weight_hbm_ref.dtype),
        weight1=pltpu.VMEM(second_weight_shape, weight_hbm_ref.dtype),
        projected=pltpu.VMEM(
            (source_count * local_rows, projection_width), out_hbm_ref.dtype
        ),
        farthest_source_projected=pltpu.VMEM(farthest_source_shape, out_hbm_ref.dtype),
        weight_sem0=pltpu.SemaphoreType.DMA,
        weight_sem1=pltpu.SemaphoreType.DMA,
        gather_send_sems=pltpu.SemaphoreType.DMA((gather_copy_count,)),
        gather_recv_sems=pltpu.SemaphoreType.DMA((gather_copy_count,)),
        rope_gather_send_sems=pltpu.SemaphoreType.DMA((gather_copy_count,)),
        rope_gather_recv_sems=pltpu.SemaphoreType.DMA((gather_copy_count,)),
        local_output_sems=pltpu.SemaphoreType.DMA((source_count,)),
        output_send_sems=pltpu.SemaphoreType.DMA((source_count, destination_count - 1)),
        output_recv_sems=pltpu.SemaphoreType.DMA((source_count, destination_count - 1)),
        farthest_source_send_sem=pltpu.SemaphoreType.DMA,
        farthest_source_recv_sem=pltpu.SemaphoreType.DMA,
        farthest_rope_send_sem=pltpu.SemaphoreType.DMA,
        farthest_rope_recv_sem=pltpu.SemaphoreType.DMA,
        farthest_source_output_sem=pltpu.SemaphoreType.DMA,
    )
    def _run(
        gathered_x,
        gathered_rope,
        weight0,
        weight1,
        projected,
        farthest_source_projected,
        weight_sem0,
        weight_sem1,
        gather_send_sems,
        gather_recv_sems,
        rope_gather_send_sems,
        rope_gather_recv_sems,
        local_output_sems,
        output_send_sems,
        output_recv_sems,
        farthest_source_send_sem,
        farthest_source_recv_sem,
        farthest_rope_send_sem,
        farthest_rope_recv_sem,
        farthest_source_output_sem,
    ):
        # Run routing in physical TPU coordinates. The metadata also maps
        # physical source IDs back to logical TP ranks for canonical token rows.
        my_id = rank_metadata_ref[0]

        def physical_to_logical(physical_id):
            return rank_metadata_ref[2 + physical_id]

        if strided_destination_groups:
            # A rank's low source-group bits select a strided set of output
            # destinations; the high bits select its position within that set.
            source_position = lax.rem(my_id, jnp.int32(source_count))
            position_in_group = my_id // source_count
        else:
            # Adjacent ranks form a destination pair.  Each rank gathers eight
            # same-parity source shards from across the ICI cube.
            source_position = my_id // destination_count
            position_in_group = lax.rem(my_id, jnp.int32(destination_count))

        def peer(peer_id):
            return {axis_name: peer_id}

        def output_mask(round_index):
            mask = jnp.int32(plan.output_orders[0][round_index])
            for source_index in range(1, len(plan.output_orders)):
                mask = lax.select(
                    source_position == source_index,
                    jnp.int32(plan.output_orders[source_index][round_index]),
                    mask,
                )
            return mask

        def output_round(link_mask):
            round_index = jnp.int32(plan.output_orders[0].index(link_mask))
            for source_index in range(1, len(plan.output_orders)):
                round_index = lax.select(
                    source_position == source_index,
                    jnp.int32(plan.output_orders[source_index].index(link_mask)),
                    round_index,
                )
            return round_index

        def packed_position(source_offset):
            if strided_destination_groups:
                return source_offset
            # Group shards by gather wave.  Only offsets 3 and 4 change places.
            return lax.select(
                source_offset == 3,
                jnp.int32(4),
                lax.select(source_offset == 4, jnp.int32(3), source_offset),
            )

        local_source_slot = source_position if destination_count == 8 else 0
        local_source_rows = pl.ds(local_source_slot * local_rows, local_rows)
        destination_group = (
            source_position
            if strided_destination_groups
            else my_id // destination_count
        )
        group_column_start = pl.multiple_of(
            destination_group * destination_count * COLUMNS_PER_DESTINATION,
            128,
        )

        def weight_copy(partition, slot=0):
            if strided_destination_groups:
                destination = source_count * partition + destination_group
                column_start = pl.multiple_of(
                    destination * COLUMNS_PER_DESTINATION,
                    128,
                )
            else:
                column_start = group_column_start
            return pltpu.make_async_copy(
                weight_hbm_ref.at[:, pl.ds(column_start, weight_width)],
                weight0 if slot == 0 else weight1,
                weight_sem0 if slot == 0 else weight_sem1,
            )

        first_mask = output_mask(0) if strided_destination_groups else 0
        first_partition = (
            lax.bitwise_xor(position_in_group, first_mask)
            if strided_destination_groups
            else 0
        )
        first_weight_copy = weight_copy(first_partition)
        first_weight_copy.start()
        if destination_count == 8:
            second_weight_copy = weight_copy(
                lax.bitwise_xor(position_in_group, output_mask(1)),
                slot=1,
            )
            second_weight_copy.start()

        def output_device_mask(link_mask):
            # The two-source groups hold one core parity across all chips, so a
            # chip-distance mask becomes an even logical-rank mask.
            return source_count * link_mask if strided_destination_groups else link_mask

        # Make remote scratch/output availability explicit.  Local RMSNorm and
        # weight prefetch hide most of this peer-barrier round trip.
        neighbor_masks = plan.input_link_masks + tuple(
            output_device_mask(mask) for mask in output_link_masks
        )
        barrier = pltpu.get_barrier_semaphore()
        for mask in neighbor_masks:
            pl.semaphore_signal(
                barrier,
                inc=1,
                device_id=peer(lax.bitwise_xor(my_id, jnp.int32(mask))),
                device_id_type=pl.DeviceIdType.MESH,
            )

        if use_rope:
            # The launcher prefetches this small table into VMEM. A direct
            # assignment avoids a separate DMA startup and semaphore wait.
            gathered_rope.at[local_source_rows][...] = rope_ref[...]

        with jax.named_scope("rmsnorm"):
            x_f32 = x_ref[...].astype(jnp.float32)
            inverse_rms = lax.rsqrt(
                jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps
            )
            gathered_x.at[local_source_rows][...] = (
                x_f32 * inverse_rms * gain_ref[...].astype(jnp.float32)[jnp.newaxis, :]
            ).astype(gathered_x.dtype)
        pl.semaphore_wait(barrier, len(neighbor_masks))

        # Do not let a fast neighbor's next invocation satisfy a slow rank's
        # current global wait. A fresh scoped semaphore keeps this second phase
        # independent while ensuring every peer consumed the global signals.
        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for mask in neighbor_masks:
                pl.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=peer(lax.bitwise_xor(my_id, jnp.int32(mask))),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(second_barrier, len(neighbor_masks))

        def gather_copy(step_index, entry_index, semaphore_index, *, rope=False):
            link_mask, slot_xor = plan.routing_steps[step_index][entry_index]
            if destination_count == 8:
                # Copy the local parity slot to the sibling core's same slot.
                source_offset = source_position
                destination_offset = source_position
            else:
                source_offset = jnp.int32(slot_xor // destination_count)
                destination_offset = lax.bitwise_xor(
                    source_offset,
                    jnp.int32(link_mask // destination_count),
                )

            source_rows = pl.ds(
                packed_position(source_offset) * local_rows,
                local_rows,
            )
            destination_rows = pl.ds(
                packed_position(destination_offset) * local_rows,
                local_rows,
            )
            if rope:
                source_ref = gathered_rope.at[source_rows]
                rope_destination_rows = (
                    pl.ds(position_in_group * local_rows, local_rows)
                    if destination_count == 2
                    and step_index == len(plan.routing_steps) - 1
                    else destination_rows
                )
                destination_ref = gathered_rope.at[rope_destination_rows]
                send_sem = rope_gather_send_sems.at[semaphore_index]
                recv_sem = rope_gather_recv_sems.at[semaphore_index]
            elif destination_count == 2 and step_index == len(plan.routing_steps) - 1:
                # Three neighbors each deliver one hidden-column stripe of the
                # farthest source, balancing the final gather over all ICI axes.
                chunk_widths = (2688, 2688, 2816)
                column_start = sum(chunk_widths[:entry_index])
                columns = pl.ds(column_start, chunk_widths[entry_index])
                source_ref = gathered_x.at[source_rows, columns]
                destination_ref = gathered_x.at[
                    pl.ds(position_in_group * local_rows, local_rows),
                    columns,
                ]
                send_sem = gather_send_sems.at[semaphore_index]
                recv_sem = gather_recv_sems.at[semaphore_index]
            else:
                source_ref = gathered_x.at[source_rows]
                destination_ref = gathered_x.at[destination_rows]
                send_sem = gather_send_sems.at[semaphore_index]
                recv_sem = gather_recv_sems.at[semaphore_index]

            return pltpu.make_async_remote_copy(
                src_ref=source_ref,
                dst_ref=destination_ref,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=peer(lax.bitwise_xor(my_id, jnp.int32(link_mask))),
                device_id_type=pl.DeviceIdType.MESH,
            )

        def apply_epilogue(accumulator, source_rows, packed_width):
            """Apply destination-local Q/K math while the projection is live."""
            row_count = accumulator.shape[0]
            destination_groups = packed_width // COLUMNS_PER_DESTINATION
            query_heads = QUERY_COLUMNS // HEAD_DIM
            key_heads = KEY_COLUMNS // HEAD_DIM
            value_heads = VALUE_COLUMNS // HEAD_DIM
            qk_heads = query_heads + key_heads
            qkv_heads = qk_heads + value_heads
            qkv_columns = QUERY_COLUMNS + KEY_COLUMNS + VALUE_COLUMNS
            packed = accumulator.astype(jnp.bfloat16).reshape(
                row_count,
                destination_groups,
                COLUMNS_PER_DESTINATION,
            )
            qkv = packed[..., :qkv_columns].reshape(
                row_count,
                destination_groups,
                qkv_heads,
                HEAD_DIM,
            )
            qk = qkv[..., :qk_heads, :].reshape(
                row_count,
                destination_groups * qk_heads,
                HEAD_DIM,
            )
            qk = qk.astype(jnp.float32)
            if use_qk_norm:
                inverse_rms = lax.rsqrt(
                    jnp.mean(qk * qk, axis=-1, keepdims=True) + eps_qk
                )
                per_destination_scale = jnp.concatenate(
                    (
                        jnp.full(
                            (query_heads,),
                            scale_query_by,
                            dtype=jnp.float32,
                        ),
                        jnp.ones((key_heads,), dtype=jnp.float32),
                    )
                )
                head_scale = jnp.tile(per_destination_scale, destination_groups)
                normalization_scale = inverse_rms * head_scale[None, :, None]
            if use_rope:
                rope = gathered_rope.at[source_rows][...]
                qk = gathered_interleaved_rope(
                    qk,
                    rope[:, :HEAD_DIM],
                    rope[:, HEAD_DIM:],
                )
            if use_qk_norm:
                # A per-head scalar commutes with RoPE. Apply the combined
                # RMS/query scale once after rotation to avoid a second
                # full-width query multiply.
                qk *= normalization_scale

            qk = qk.astype(jnp.bfloat16).reshape(
                row_count,
                destination_groups,
                qk_heads,
                HEAD_DIM,
            )
            q = qk[..., :query_heads, :].reshape(
                row_count, destination_groups, QUERY_COLUMNS
            )
            k = qk[..., query_heads, :]
            v = qkv[..., qk_heads, :]
            og = packed[..., qkv_columns:]

            return jnp.concatenate(
                (
                    q,
                    k,
                    v,
                    og,
                ),
                axis=-1,
            ).reshape(row_count, packed_width)

        def project_sources(
            source_start,
            source_count_for_matmul,
            *,
            output_slot=0,
            weight_ref=weight0,
        ):
            source_rows = pl.ds(
                source_start * local_rows,
                source_count_for_matmul * local_rows,
            )
            accumulator = lax.dot_general(
                gathered_x.at[source_rows][...],
                weight_ref[...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            output_columns = pl.ds(output_slot * weight_width, weight_width)
            projected.at[source_rows, output_columns][...] = apply_epilogue(
                accumulator,
                source_rows,
                weight_width,
            )

        local_output_slot = (
            plan.output_orders[0].index(0) % plan.projection_slots
            if strided_destination_groups
            else position_in_group
        )
        local_output_columns = pl.ds(
            local_output_slot * COLUMNS_PER_DESTINATION,
            COLUMNS_PER_DESTINATION,
        )

        def output_shard_index(source_position_index):
            if destination_count == 8:
                physical_source = 2 * position_in_group + source_position_index
            else:
                physical_source = (
                    destination_count * source_position_index + position_in_group
                )
            return physical_to_logical(physical_source)

        def start_two_destination_outputs(group_start, group_size):
            """Send completed 1K results to the two ranks in this pair."""
            for packed_index in range(group_start, group_start + group_size):
                source_offset = plan.packed_source_offsets[packed_index]
                source_rows = pl.ds(packed_index * local_rows, local_rows)
                source_index = lax.bitwise_xor(
                    source_position,
                    jnp.int32(source_offset),
                )
                output_rows = pl.ds(
                    output_shard_index(source_index) * local_rows,
                    local_rows,
                )
                pltpu.make_async_copy(
                    projected.at[source_rows, local_output_columns],
                    out_hbm_ref.at[output_rows, :],
                    local_output_sems.at[source_index],
                ).start()
                remote_columns = pl.ds(
                    lax.bitwise_xor(position_in_group, jnp.int32(1))
                    * COLUMNS_PER_DESTINATION,
                    COLUMNS_PER_DESTINATION,
                )
                pltpu.make_async_remote_copy(
                    src_ref=projected.at[source_rows, remote_columns],
                    dst_ref=out_hbm_ref.at[output_rows, :],
                    send_sem=output_send_sems.at[source_index, 0],
                    recv_sem=output_recv_sems.at[source_index, 0],
                    device_id=peer(lax.bitwise_xor(my_id, jnp.int32(1))),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        def start_strided_destination_output(link_mask, output_slot, source_slot):
            """Send one source shard's result to its strided destination."""
            output_columns = pl.ds(output_slot * weight_width, weight_width)
            source_rows = pl.ds(source_slot * local_rows, local_rows)
            output_rows = pl.ds(
                output_shard_index(source_slot) * local_rows,
                local_rows,
            )

            @pl.when(link_mask == 0)
            def _copy_local():
                pltpu.make_async_copy(
                    projected.at[source_rows, output_columns],
                    out_hbm_ref.at[output_rows, :],
                    local_output_sems.at[source_slot],
                ).start()

            @pl.when(link_mask != 0)
            def _send_remote():
                output_index = link_mask - 1
                pltpu.make_async_remote_copy(
                    src_ref=projected.at[source_rows, output_columns],
                    dst_ref=out_hbm_ref.at[output_rows, :],
                    send_sem=output_send_sems.at[source_slot, output_index],
                    recv_sem=output_recv_sems.at[source_slot, output_index],
                    device_id=peer(
                        lax.bitwise_xor(
                            my_id,
                            jnp.int32(output_device_mask(link_mask)),
                        )
                    ),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        # Pipeline input gather waves with projection of the rows delivered by
        # the preceding wave.
        semaphore_index = 0
        for step_index, entries in enumerate(plan.routing_steps):
            for entry_index in range(len(entries)):
                gather_copy(
                    step_index,
                    entry_index,
                    semaphore_index + entry_index,
                ).start()
            rope_entry_count = (
                1
                if destination_count == 2 and step_index == len(plan.routing_steps) - 1
                else len(entries)
            )
            if use_rope:
                for entry_index in range(rope_entry_count):
                    gather_copy(
                        step_index,
                        entry_index,
                        semaphore_index + entry_index,
                        rope=True,
                    ).start()

            group_start, group_size = plan.projection_groups[step_index]
            if step_index == 0:
                first_weight_copy.wait()
                project_sources(local_source_slot, group_size)
                if destination_count == 8:
                    start_strided_destination_output(
                        first_mask,
                        0,
                        local_source_slot,
                    )
            else:
                project_sources(group_start, group_size)

            if destination_count == 2 and step_index < 2:
                start_two_destination_outputs(group_start, group_size)

            for entry_index in range(len(entries)):
                copy = gather_copy(
                    step_index,
                    entry_index,
                    semaphore_index + entry_index,
                )
                copy.wait_send()
                copy.wait_recv()
            if use_rope:
                for entry_index in range(rope_entry_count):
                    rope_copy = gather_copy(
                        step_index,
                        entry_index,
                        semaphore_index + entry_index,
                        rope=True,
                    )
                    rope_copy.wait_send()
                    rope_copy.wait_recv()
            semaphore_index += len(entries)

        if destination_count == 2:
            # The paired core now holds the other half of the farthest source.
            # Exchange the assembled rows on-chip so each destination rank can
            # compute its own 1152 output columns in one 128-row matmul.
            farthest_source_position = lax.bitwise_xor(
                source_position,
                jnp.int32(7),
            )
            assembled_rows = pl.ds(position_in_group * local_rows, local_rows)
            farthest_source_copy = pltpu.make_async_remote_copy(
                src_ref=gathered_x.at[assembled_rows],
                dst_ref=gathered_x.at[assembled_rows],
                send_sem=farthest_source_send_sem,
                recv_sem=farthest_source_recv_sem,
                device_id=peer(lax.bitwise_xor(my_id, jnp.int32(1))),
                device_id_type=pl.DeviceIdType.MESH,
            )
            farthest_source_copy.start()
            farthest_source_copy.wait_recv()
            if use_rope:
                farthest_rope_copy = pltpu.make_async_remote_copy(
                    src_ref=gathered_rope.at[assembled_rows],
                    dst_ref=gathered_rope.at[assembled_rows],
                    send_sem=farthest_rope_send_sem,
                    recv_sem=farthest_rope_recv_sem,
                    device_id=peer(lax.bitwise_xor(my_id, jnp.int32(1))),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                farthest_rope_copy.start()
                farthest_rope_copy.wait_recv()

            start_two_destination_outputs(*plan.projection_groups[2])
            farthest_accumulator = lax.dot_general(
                gathered_x.at[pl.ds(0, 2 * local_rows)][...],
                weight0.at[:, local_output_columns][...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            farthest_rows = pl.ds(0, 2 * local_rows)
            farthest_source_projected[...] = apply_epilogue(
                farthest_accumulator,
                farthest_rows,
                COLUMNS_PER_DESTINATION,
            )
            farthest_physical_source = 2 * farthest_source_position
            farthest_output_rows = pl.ds(
                physical_to_logical(farthest_physical_source) * local_rows,
                2 * local_rows,
            )
            pltpu.make_async_copy(
                farthest_source_projected,
                out_hbm_ref.at[farthest_output_rows, :],
                farthest_source_output_sem,
            ).start()
            farthest_source_copy.wait_send()
            if use_rope:
                farthest_rope_copy.wait_send()
        else:
            # Complete the first destination after the sibling source arrives.
            project_sources(1 - local_source_slot, 1)
            start_strided_destination_output(
                first_mask,
                0,
                1 - local_source_slot,
            )

            # Stream the remaining seven destination weights through two
            # buffers.  Output DMAs from earlier rounds overlap later matmuls.
            pending_weight_copy = second_weight_copy
            for round_index in range(1, weight_partition_count):
                link_mask = output_mask(round_index)
                pending_weight_copy.wait()

                next_weight_copy = None
                if round_index + 1 < weight_partition_count:
                    next_weight_copy = weight_copy(
                        lax.bitwise_xor(
                            position_in_group,
                            output_mask(round_index + 1),
                        ),
                        slot=(round_index + 1) % 2,
                    )
                    next_weight_copy.start()

                output_slot = round_index % plan.projection_slots
                current_weight = weight0 if round_index % 2 == 0 else weight1
                # Once the sibling rows have arrived, project both source
                # shards together. The taller matmul halves steady-state
                # launch count while preserving each source's output rows.
                project_sources(
                    0,
                    source_count,
                    output_slot=output_slot,
                    weight_ref=current_weight,
                )
                for source_slot in range(source_count):
                    start_strided_destination_output(
                        link_mask,
                        output_slot,
                        source_slot,
                    )
                pending_weight_copy = next_weight_copy

        # Drain output transfers and receive counts before scratch is released.
        if strided_destination_groups:
            for source_slot in range(source_count):
                source_rows = pl.ds(source_slot * local_rows, local_rows)
                output_rows = pl.ds(
                    output_shard_index(source_slot) * local_rows,
                    local_rows,
                )
                pltpu.make_async_copy(
                    projected.at[source_rows, local_output_columns],
                    out_hbm_ref.at[output_rows, :],
                    local_output_sems.at[source_slot],
                ).wait()

            for output_index, link_mask in enumerate(output_link_masks):
                round_index = output_round(link_mask)
                output_slot = round_index % plan.projection_slots
                output_columns = pl.ds(output_slot * weight_width, weight_width)
                for source_slot in range(source_count):
                    source_rows = pl.ds(source_slot * local_rows, local_rows)
                    output_rows = pl.ds(
                        output_shard_index(source_slot) * local_rows,
                        local_rows,
                    )
                    copy = pltpu.make_async_remote_copy(
                        src_ref=projected.at[source_rows, output_columns],
                        dst_ref=out_hbm_ref.at[output_rows, :],
                        send_sem=output_send_sems.at[source_slot, output_index],
                        recv_sem=output_recv_sems.at[source_slot, output_index],
                        device_id=peer(
                            lax.bitwise_xor(
                                my_id,
                                jnp.int32(output_device_mask(link_mask)),
                            )
                        ),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    copy.wait_send()
                    copy.wait_recv()
        else:
            for packed_index in range(source_count - 1):
                source_offset = plan.packed_source_offsets[packed_index]
                source_rows = pl.ds(packed_index * local_rows, local_rows)
                source_index = lax.bitwise_xor(
                    source_position,
                    jnp.int32(source_offset),
                )
                output_rows = pl.ds(
                    output_shard_index(source_index) * local_rows,
                    local_rows,
                )
                pltpu.make_async_copy(
                    projected.at[source_rows, local_output_columns],
                    out_hbm_ref.at[output_rows, :],
                    local_output_sems.at[source_index],
                ).wait()

                remote_columns = pl.ds(
                    lax.bitwise_xor(position_in_group, jnp.int32(1))
                    * COLUMNS_PER_DESTINATION,
                    COLUMNS_PER_DESTINATION,
                )
                remote_copy = pltpu.make_async_remote_copy(
                    src_ref=projected.at[source_rows, remote_columns],
                    dst_ref=out_hbm_ref.at[output_rows, :],
                    send_sem=output_send_sems.at[source_index, 0],
                    recv_sem=output_recv_sems.at[source_index, 0],
                    device_id=peer(lax.bitwise_xor(my_id, jnp.int32(1))),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                remote_copy.wait_send()
                remote_copy.wait_recv()

            farthest_physical_source = 2 * farthest_source_position
            farthest_output_rows = pl.ds(
                physical_to_logical(farthest_physical_source) * local_rows,
                2 * local_rows,
            )
            pltpu.make_async_copy(
                farthest_source_projected,
                out_hbm_ref.at[farthest_output_rows, :],
                farthest_source_output_sem,
            ).wait()

    _run()


def run_hybrid_gather_a2a(
    local_x: jax.Array,
    replicated_weight: jax.Array,
    replicated_gain: jax.Array,
    local_cos: jax.Array,
    local_sin: jax.Array,
    local_rank_metadata: jax.Array,
    *,
    axis_name: str,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    plan: HybridGatherAllToAllPlan,
    collective_id: int,
) -> jax.Array:
    """Create the plan-specialized Pallas call for one rank."""
    if local_x.shape != (plan.local_rows, HIDDEN_SIZE):
        raise ValueError(
            f"plan expects x shape ({plan.local_rows}, {HIDDEN_SIZE}), "
            f"got {local_x.shape}"
        )
    if local_cos.shape != (plan.local_rows, HEAD_DIM) or local_sin.shape != (
        plan.local_rows,
        HEAD_DIM,
    ):
        raise ValueError("local cos/sin rows must match the plan's local token rows")
    if local_rank_metadata.shape != (TP_SIZE + 2,):
        raise ValueError(
            f"rank metadata must have local shape ({TP_SIZE + 2},), "
            f"got {local_rank_metadata.shape}"
        )
    local_rope = jnp.concatenate((local_cos, local_sin), axis=-1)
    return pl.pallas_call(
        functools.partial(
            _kernel_hybrid_gather_a2a,
            axis_name=axis_name,
            eps=eps,
            eps_qk=eps_qk,
            scale_query_by=scale_query_by,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
            plan=plan,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (TP_SIZE * plan.local_rows, COLUMNS_PER_DESTINATION),
            jnp.bfloat16,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(),
                pl.BlockSpec(),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=plan.vmem_limit_bytes,
            collective_id=collective_id,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        name=f"fused_rmsnorm_qkvog_mm_a2a_epilogue_{plan.name}",
    )(
        local_rank_metadata,
        local_x,
        replicated_weight,
        replicated_gain,
        local_rope,
    )
