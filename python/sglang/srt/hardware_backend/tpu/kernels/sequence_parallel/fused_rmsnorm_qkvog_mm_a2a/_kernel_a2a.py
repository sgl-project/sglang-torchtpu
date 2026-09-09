# SPDX-License-Identifier: Apache-2.0

"""Direct all-to-all schedule for the fused projection."""

from __future__ import annotations

import functools
from typing import Any

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
    OUTPUT_GATE_COLUMNS,
    QUERY_COLUMNS,
    TP_SIZE,
    VALUE_COLUMNS,
)
from ._epilogue import gathered_interleaved_rope
from ._schedule import AllToAllPlan

# Reading guide for the Pallas primitives used below:
#
# * ``*_hbm_ref`` names refer to large, off-chip HBM allocations. Buffers
#   created by ``pl.with_scoped`` live in the smaller, faster on-chip VMEM.
# * ``ref.at[...]`` describes a view into a memory reference; ``pl.ds(start,
#   size)`` is a dynamic slice beginning at ``start`` with length ``size``.
# * ``make_async_copy`` returns a DMA descriptor. ``start()`` launches the
#   transfer and ``wait()`` prevents its buffer or result from being used too
#   early.
# * The MXU performs the large matrix multiplications, while the vector unit
#   handles RMSNorm, Q/K normalization, scaling, and RoPE.


def _signal_all_peers(axis_name: str) -> None:
    # Every rank, including this one, contributes one signal to every rank's
    # barrier. The matching wait for TP_SIZE signals is delayed until the first
    # output send, so setup and projection work can overlap the rendezvous.
    barrier = pltpu.get_barrier_semaphore()
    for peer in range(TP_SIZE):
        pltpu.semaphore_signal(
            barrier,
            device_id={axis_name: jnp.int32(peer)},
            device_id_type=pltpu.DeviceIdType.MESH,
        )


def _wait_for_all_peers_reuse_safe(axis_name: str) -> None:
    """Finish the global barrier without leaking signals across invocations."""
    pltpu.semaphore_wait(pltpu.get_barrier_semaphore(), TP_SIZE)

    # A fast rank must not re-enter the next invocation and signal the global
    # barrier before every peer consumed this invocation's signals. Rendezvous
    # once more on a fresh scoped semaphore before any rank can leave setup.
    @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
    def _(second_barrier):
        for peer in range(TP_SIZE):
            pltpu.semaphore_signal(
                second_barrier,
                device_id={axis_name: jnp.int32(peer)},
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(second_barrier, TP_SIZE)


def _kernel_a2a(
    rank_metadata_ref,
    x_hbm_ref,
    weight_hbm_ref,
    gain_hbm_ref,
    cos_hbm_ref,
    sin_hbm_ref,
    q_hbm_ref,
    kv_hbm_ref,
    og_hbm_ref,
    *,
    axis_name: str,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    plan: AllToAllPlan,
):
    """Project local rows and send final Q/K/V/OG tiles to each destination."""
    # Each rank begins with only its token rows, but with projection weights for
    # every destination rank. For each destination this kernel computes that
    # rank's Q/K/V/OG columns, applies Q/K normalization, query scaling, and
    # RoPE, then sends the finished rows. Each rank therefore ends with its own
    # heads for every source rank's tokens; no standalone activation all-gather
    # is needed.
    total_local_rows, hidden_size = x_hbm_ref.shape
    block_rows = plan.row_block_rows
    projection_widths = plan.projection_widths
    tiles_per_destination = len(projection_widths)
    max_projection_width = max(projection_widths)
    send_width = max(QUERY_COLUMNS, OUTPUT_GATE_COLUMNS)
    assert sum(projection_widths) == COLUMNS_PER_DESTINATION
    # A task is one projection tile. The plan uses two adjacent tasks per
    # destination: one for packed [Q | K | V], followed by one for OG.
    task_count = TP_SIZE * tiles_per_destination

    assert weight_hbm_ref.shape == (
        hidden_size,
        TP_SIZE * COLUMNS_PER_DESTINATION,
    )
    assert total_local_rows == plan.local_rows
    output_rows = TP_SIZE * total_local_rows
    assert cos_hbm_ref.shape == sin_hbm_ref.shape == (total_local_rows, HEAD_DIM)
    assert q_hbm_ref.shape == (output_rows, QUERY_COLUMNS)
    assert kv_hbm_ref.shape == (output_rows, KEY_COLUMNS + VALUE_COLUMNS)
    assert og_hbm_ref.shape == (output_rows, OUTPUT_GATE_COLUMNS)

    # ``with_scoped`` allocates this working set once per Pallas grid program
    # and passes the resulting VMEM buffers and semaphores to ``_run``.
    @pl.with_scoped(
        # The serving residual stream is FP32. Normalize from that source
        # precision, then store BF16 for the MXU just like the unfused path's
        # branch-dtype boundary.
        normalized=pltpu.VMEM((block_rows, hidden_size), jnp.bfloat16),
        # Stream one small row tile from HBM at a time for input RMSNorm.
        x_tile=pltpu.VMEM((plan.norm_block_rows, hidden_size), x_hbm_ref.dtype),
        gain=pltpu.VMEM((hidden_size,), gain_hbm_ref.dtype),
        cos=pltpu.VMEM((block_rows, HEAD_DIM), cos_hbm_ref.dtype),
        sin=pltpu.VMEM((block_rows, HEAD_DIM), sin_hbm_ref.dtype),
        # Dedicated QKV and OG weight slots let HBM DMA for the next destination
        # overlap MXU work for the current destination.
        weight_tiles=pltpu.VMEM(
            (plan.weight_buffers, hidden_size, max_projection_width),
            weight_hbm_ref.dtype,
        ),
        # Raw BF16 projections are double-buffered: one buffer feeds the vector
        # epilogue and output DMA while the MXU fills the other.
        projection_buffers=pltpu.VMEM(
            (plan.projection_buffers, block_rows, COLUMNS_PER_DESTINATION),
            jnp.bfloat16,
        ),
        # Q and OG share a ring of same-width payload buffers. KV keeps one
        # destination-specific buffer because all destination sends may overlap.
        send_buffers=pltpu.VMEM(
            (plan.send_slots, block_rows, send_width), q_hbm_ref.dtype
        ),
        kv_send_buffers=pltpu.VMEM(
            (TP_SIZE, block_rows, KEY_COLUMNS + VALUE_COLUMNS), kv_hbm_ref.dtype
        ),
        # A separate semaphore tracks each independently reusable DMA buffer.
        # Remote receives share ``recv_sem`` and are drained by byte count.
        x_sem=pltpu.SemaphoreType.DMA,
        gain_sem=pltpu.SemaphoreType.DMA,
        rope_sems=pltpu.SemaphoreType.DMA((2,)),
        weight_sems=pltpu.SemaphoreType.DMA((plan.weight_buffers,)),
        send_sems=pltpu.SemaphoreType.DMA((plan.send_slots,)),
        kv_send_sems=pltpu.SemaphoreType.DMA((TP_SIZE,)),
        recv_sem=pltpu.SemaphoreType.DMA,
    )
    def _run(
        normalized,
        x_tile,
        gain,
        cos,
        sin,
        weight_tiles,
        projection_buffers,
        send_buffers,
        kv_send_buffers,
        x_sem,
        gain_sem,
        rope_sems,
        weight_sems,
        send_sems,
        kv_send_sems,
        recv_sem,
    ):
        # TorchTPU's logical TP order is not necessarily physical TPU order.
        # Scheduling and remote-copy destinations use physical IDs; output row
        # placement uses the source's logical rank so attention sees canonical
        # token order.
        my_id = rank_metadata_ref[0]
        logical_id = rank_metadata_ref[1]
        # Large inputs use several grid programs, each responsible for one
        # contiguous row block. ``multiple_of`` is an alignment promise to the
        # compiler; it does not round or otherwise change the row offset.
        row_start = (
            pl.multiple_of(
                pl.program_id(0) * block_rows,
                plan.norm_block_rows,
            )
            if plan.local_rows > block_rows
            else 0
        )

        def task_coordinates(task):
            # Flatten [destination round, projection tile] into one task index.
            # ``destination_orders`` contains XOR masks rather than absolute
            # IDs. XORing every mask with ``my_id`` visits all physical ranks,
            # while the parity-specific orders spread traffic across links.
            destination_round = task // tiles_per_destination
            column_index = task % tiles_per_destination
            destination_mask = lax.select(
                lax.rem(my_id, jnp.int32(2)) == 0,
                jnp.int32(plan.destination_orders[0][destination_round]),
                jnp.int32(plan.destination_orders[1][destination_round]),
            )
            destination = lax.bitwise_xor(my_id, destination_mask)
            column_start = sum(projection_widths[:column_index])
            column_width = projection_widths[column_index]
            return destination, column_start, column_width

        def weight_copy(task, slot):
            destination, column_start, column_width = task_coordinates(task)
            # The replicated matrix is destination-major:
            # [destination 0's QKVOG | ... | destination 15's QKVOG].
            # This expression selects the current destination and its QKV or
            # OG sub-tile. ``multiple_of`` only records its known alignment.
            flat_column_start = pl.multiple_of(
                destination * COLUMNS_PER_DESTINATION + column_start,
                128,
            )
            return pltpu.make_async_copy(
                weight_hbm_ref.at[:, pl.ds(flat_column_start, column_width)],
                weight_tiles.at[slot, :, pl.ds(0, column_width)],
                weight_sems.at[slot],
            )

        def wait_send_slot(slot):
            # Reconstructing an async copy with the same refs and semaphore is
            # Pallas's API for waiting on a previously started transfer.
            payload = send_buffers.at[slot]
            pltpu.make_async_copy(
                payload,
                payload,
                send_sems.at[slot],
            ).wait()

        def wait_kv_send_slot(slot):
            # As above, this reconstructs the descriptor of an earlier KV send
            # and waits for it; it does not launch a VMEM-to-VMEM copy.
            pltpu.make_async_copy(
                kv_send_buffers.at[slot],
                kv_send_buffers.at[slot],
                kv_send_sems.at[slot],
            ).wait()

        def start_output_copy(source, target, send_sem, destination):
            # ``pl.when`` emits device-side conditionals. A remote destination
            # uses the TPU interconnect; this rank's own destination is a local
            # VMEM-to-HBM DMA with the same output layout.
            @pl.when(destination != my_id)
            def _send_remote():
                pltpu.make_async_remote_copy(
                    src_ref=source,
                    dst_ref=target,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id={axis_name: destination},
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            @pl.when(destination == my_id)
            def _copy_local():
                pltpu.make_async_copy(source, target, send_sem).start()

        def qk_epilogue(value):
            # ``value`` is [rows, query heads + one key head, HEAD_DIM]. The
            # projection is stored in BF16, but normalization and RoPE use FP32
            # intermediates before the final BF16 conversion.
            value = value.astype(jnp.bfloat16).astype(jnp.float32)
            if use_qk_norm:
                inverse_rms = lax.rsqrt(
                    jnp.mean(value * value, axis=-1, keepdims=True) + eps_qk
                )
                head_scale = jnp.concatenate(
                    (
                        # Query heads receive both RMS normalization and the
                        # configured query scale; the key head receives only
                        # RMS normalization.
                        jnp.full(
                            (QUERY_COLUMNS // HEAD_DIM,),
                            scale_query_by,
                            dtype=jnp.float32,
                        ),
                        jnp.ones((1,), dtype=jnp.float32),
                    )
                )
                normalization_scale = inverse_rms * head_scale[None, :, None]
            if use_rope:
                value = gathered_interleaved_rope(
                    value,
                    cos[...][:, None, :],
                    sin[...][:, None, :],
                )
            if use_qk_norm:
                # A per-head scalar commutes with RoPE. Applying the combined
                # RMS/query scale once after rotation removes one full-width
                # FP32 multiply without changing the mathematical result.
                value *= normalization_scale
            return value.astype(jnp.bfloat16)

        # Protect remote output buffers, but overlap the barrier round trip with
        # local normalization, RoPE-table loads, and the first weight loads.
        # These independent ``start`` calls put several DMAs in flight; each is
        # waited only at the point where its data or buffer is first needed.
        _signal_all_peers(axis_name)
        gain_copy = pltpu.make_async_copy(gain_hbm_ref, gain, gain_sem)
        gain_copy.start()
        weight_copy(0, 0).start()
        weight_copy(1, 1).start()
        if use_rope:
            rope_rows = pl.ds(row_start, block_rows)
            cos_copy = pltpu.make_async_copy(
                cos_hbm_ref.at[rope_rows],
                cos,
                rope_sems.at[0],
            )
            sin_copy = pltpu.make_async_copy(
                sin_hbm_ref.at[rope_rows],
                sin,
                rope_sems.at[1],
            )
            cos_copy.start()
            sin_copy.start()
        gain_copy.wait()

        # Normalize a small row tile at a time to bound FP32 VMEM use. Here
        # ``input_rows`` addresses the original HBM input, while ``local_rows``
        # addresses the corresponding rows of the on-chip normalized buffer.
        for row_block in range(block_rows // plan.norm_block_rows):
            local_rows = pl.ds(
                row_block * plan.norm_block_rows,
                plan.norm_block_rows,
            )
            input_rows = pl.ds(
                row_start + row_block * plan.norm_block_rows,
                plan.norm_block_rows,
            )
            x_copy = pltpu.make_async_copy(x_hbm_ref.at[input_rows], x_tile, x_sem)
            x_copy.start()
            x_copy.wait()
            x_f32 = x_tile[...].astype(jnp.float32)
            inverse_rms = lax.rsqrt(
                jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps
            )
            normalized.at[local_rows][...] = (
                x_f32 * inverse_rms * gain[...].astype(jnp.float32)[None, :]
            ).astype(normalized.dtype)

        if use_rope:
            cos_copy.wait()
            sin_copy.wait()

        # On every destination rank, rows are stored in canonical source-rank
        # order. This rank therefore always writes the same row interval; only
        # the remote destination changes as the pipeline advances.
        source_rows = pl.ds(
            logical_id * total_local_rows + row_start,
            block_rows,
        )

        def project_destination_into_buffer(destination_round, buffer_index):
            """Project one destination's QKV and OG columns into BF16 VMEM."""
            qkv_task = destination_round * tiles_per_destination
            og_task = qkv_task + 1
            _, _, qkv_width = task_coordinates(qkv_task)
            _, _, og_width = task_coordinates(og_task)

            # Refill each weight slot as soon as its current matmul has
            # consumed it, overlapping the following destination's HBM load
            # with independent MXU work.
            weight_copy(qkv_task, 0).wait()
            with jax.named_scope("qkv_matmul"):
                # [block_rows, hidden] @ [hidden, Q+K+V] -> [block_rows, Q+K+V].
                # Contract dimension 1 of the activations with dimension 0 of
                # the weights, and accumulate the MXU result in FP32.
                projection_buffers.at[buffer_index, :, pl.ds(0, qkv_width)][...] = (
                    lax.dot_general(
                        normalized[...],
                        weight_tiles.at[0, :, pl.ds(0, qkv_width)][...],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)
                )
            if og_task + 1 < task_count:
                weight_copy(og_task + 1, 0).start()

            weight_copy(og_task, 1).wait()
            with jax.named_scope("og_matmul"):
                # The second matmul projects the same normalized rows to OG.
                projection_buffers.at[
                    buffer_index,
                    :,
                    pl.ds(qkv_width, og_width),
                ][...] = lax.dot_general(
                    normalized[...],
                    weight_tiles.at[1, :, pl.ds(0, og_width)][...],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(
                    jnp.bfloat16
                )
            if og_task + 2 < task_count:
                weight_copy(og_task + 2, 1).start()

        def finish_destination_from_buffer(destination_round, buffer_index):
            """Apply the attention epilogue and send one buffered destination."""
            qkv_task = destination_round * tiles_per_destination
            og_task = qkv_task + 1
            destination, _, qkv_width = task_coordinates(qkv_task)
            _, _, og_width = task_coordinates(og_task)
            q_send_slot = qkv_task % plan.send_slots
            og_send_slot = og_task % plan.send_slots

            # Keep completed output DMAs in flight until their payload slots
            # are needed again. The intervening destination work hides their
            # send-side latency.
            if qkv_task >= plan.send_slots:
                wait_send_slot(q_send_slot)
                wait_send_slot(og_send_slot)

            projection = projection_buffers.at[buffer_index]
            # One destination's raw projection is packed as [Q | K | V | OG].
            # Q and K share normalization/RoPE; V bypasses that epilogue, and
            # OG came from the second matmul.
            qkv_projected = projection.at[:, pl.ds(0, qkv_width)][...]
            og_projected = projection.at[:, pl.ds(qkv_width, og_width)][...]
            with jax.named_scope("qkv_epilogue"):
                # Reinterpret the flat Q+K columns as eight query heads followed
                # by one key head. The slices below flatten Q again and pack
                # [K | V] into the KV payload expected by the destination.
                qk = qk_epilogue(
                    qkv_projected[:, : QUERY_COLUMNS + KEY_COLUMNS].reshape(
                        block_rows,
                        (QUERY_COLUMNS + KEY_COLUMNS) // HEAD_DIM,
                        HEAD_DIM,
                    )
                )
                send_buffers.at[q_send_slot][...] = qk[
                    :, : QUERY_COLUMNS // HEAD_DIM, :
                ].reshape(block_rows, QUERY_COLUMNS)
                kv_send_buffers.at[destination_round, :, pl.ds(0, KEY_COLUMNS)][...] = (
                    qk[:, QUERY_COLUMNS // HEAD_DIM :, :]
                    .reshape(block_rows, KEY_COLUMNS)
                    .astype(kv_hbm_ref.dtype)
                )
                kv_send_buffers.at[
                    destination_round,
                    :,
                    pl.ds(KEY_COLUMNS, VALUE_COLUMNS),
                ][...] = qkv_projected[
                    :,
                    QUERY_COLUMNS
                    + KEY_COLUMNS : QUERY_COLUMNS
                    + KEY_COLUMNS
                    + VALUE_COLUMNS,
                ].astype(
                    kv_hbm_ref.dtype
                )
            send_buffers.at[og_send_slot][...] = og_projected.astype(jnp.bfloat16)

            # Projection is independent of remote output buffers, so wait for
            # the setup barrier only immediately before the first send.
            if destination_round == 0:
                _wait_for_all_peers_reuse_safe(axis_name)

            # This is the all-to-all itself: send this source rank's fixed row
            # interval to the physical rank that owns these Q/K/V/OG columns.
            # Q, packed KV, and OG use separate asynchronous transfers.
            start_output_copy(
                send_buffers.at[q_send_slot],
                q_hbm_ref.at[source_rows, :],
                send_sems.at[q_send_slot],
                destination,
            )
            start_output_copy(
                kv_send_buffers.at[destination_round],
                kv_hbm_ref.at[source_rows, :],
                kv_send_sems.at[destination_round],
                destination,
            )
            start_output_copy(
                send_buffers.at[og_send_slot],
                og_hbm_ref.at[source_rows, :],
                send_sems.at[og_send_slot],
                destination,
            )

        # Prime a two-stage software pipeline. Storing the raw BF16 result in
        # VMEM releases MXU accumulators before the epilogue. In steady state,
        # destination d projects into one buffer while destination d-1 applies
        # Q/K norm, RoPE, casts, and output copies from the other buffer.
        assert plan.projection_buffers == 2
        project_destination_into_buffer(0, 0)
        for destination_round in range(1, TP_SIZE):
            current_buffer = destination_round % plan.projection_buffers
            previous_buffer = (destination_round - 1) % plan.projection_buffers
            project_destination_into_buffer(destination_round, current_buffer)
            finish_destination_from_buffer(destination_round - 1, previous_buffer)
        finish_destination_from_buffer(
            TP_SIZE - 1,
            (TP_SIZE - 1) % plan.projection_buffers,
        )
        # Drain every outgoing transfer before its VMEM payload disappears.
        # Q/OG use the reusable slot ring; KV has one slot per destination.
        for slot in range(plan.send_slots):
            wait_send_slot(slot)
        for slot in range(TP_SIZE):
            wait_kv_send_slot(slot)

        # Each remote source contributes every output component. Reconstructing
        # three waits with the shared receive semaphore consumes their aggregate
        # byte counts while avoiding any additional data movement.
        received_rows = pl.ds(0, (TP_SIZE - 1) * block_rows)
        for output_ref in (q_hbm_ref, kv_hbm_ref, og_hbm_ref):
            payload = output_ref.at[received_rows, :]
            pltpu.make_async_copy(payload, payload, recv_sem).wait()

    _run()


def run_a2a(
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
    kv_dtype: Any,
    plan: AllToAllPlan,
    collective_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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

    # A scalar grid runs one program for up to ``row_block_rows`` local rows.
    # Larger inputs launch one independent program per contiguous row block.
    grid = (
        (plan.local_rows // plan.row_block_rows,)
        if plan.local_rows > plan.row_block_rows
        else ()
    )
    # Rank metadata is scalar-prefetched; the five tensor inputs and all three
    # outputs remain HBM references consumed by the kernel above.
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 3,
        grid=grid,
    )
    # Shapes, feature flags, loop counts, routing tables, and collective ID are
    # compile-time properties of this specialized Pallas program.
    q, kv, og = pl.pallas_call(
        functools.partial(
            _kernel_a2a,
            axis_name=axis_name,
            eps=eps,
            eps_qk=eps_qk,
            scale_query_by=scale_query_by,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
            plan=plan,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (TP_SIZE * plan.local_rows, QUERY_COLUMNS),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (TP_SIZE * plan.local_rows, KEY_COLUMNS + VALUE_COLUMNS),
                kv_dtype,
            ),
            jax.ShapeDtypeStruct(
                (TP_SIZE * plan.local_rows, OUTPUT_GATE_COLUMNS),
                jnp.bfloat16,
            ),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=plan.vmem_limit_bytes,
            collective_id=collective_id,
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        name="fused_rmsnorm_qkvog_mm_a2a_epilogue",
    )(
        local_rank_metadata,
        local_x,
        replicated_weight,
        replicated_gain,
        local_cos,
        local_sin,
    )
    # The kernel communicates K and V together, then this wrapper restores the
    # public four-output interface by splitting the packed last dimension.
    return q, kv[:, :KEY_COLUMNS], kv[:, KEY_COLUMNS:], og
