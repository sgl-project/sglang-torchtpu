# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Context Parallel (CP) schedulers for RPA."""

import dataclasses
import functools
from collections.abc import Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.batched_rpa_pcp import (
    configs,
    schedule,
    scheduler,
    utils,
)


@dataclasses.dataclass(frozen=True)
class CPScheduler(scheduler.BaseScheduler):
    """Base class for page-striped Context Parallel (CP) schedulers.

    Expects `extra_scalars=(cp_rank, ...)` where `cp_rank` is an `int32[1]` array
    giving the rank of the current device in the CP group.

    Attributes:
      cp_group_size: Number of devices in the CP group.
    """

    cp_group_size: int

    def get_cp_group_size(self) -> int:
        return self.cp_group_size

    def _get_cp_rank(self, extra: Sequence[jax.Array] | Sequence[jax.Ref]) -> jax.Array:
        if not extra:
            raise ValueError(
                f"{type(self).__name__} requires `extra_scalars=(cp_rank, ...)`."
            )
        return extra[0][0]


@dataclasses.dataclass(frozen=True)
class StripedCacheScheduler(CPScheduler):
    """This scheduler only attends to this rank's local KV cache."""

    scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.CACHE_ONLY

    def get_kv_lens(
        self,
        kv_lens: jax.Array,
        q_lens: jax.Array,
        page_size: int = 1,
        extra_scalars: Sequence[jax.Array] = (),
        kv_lens_global: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del kv_lens_global
        cp_rank = self._get_cp_rank(extra_scalars)
        global_cache_lens = kv_lens - q_lens
        local_cache_lens = utils.cp_local_cache_len(
            global_cache_lens, self.cp_group_size, cp_rank, page_size
        )
        return local_cache_lens, jnp.zeros_like(q_lens), local_cache_lens


# ---------------------------------------------------------------------------
# PCP ring attention (google3 cl/985434096).
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingSems:
    """The ring's semaphores."""

    dma_sems: Any  # [NUM_INPUT_SLOTS, 2] send/recv per buffer slot.
    sync_sem: Any  # credit-based flow control semaphore.


class RingAttention:
    """Rotates the KV window around the PCP ring during the cache phase.

    Each rank owns one page-stripe of the KV cache. Step `t` is hop `t % P` of
    one KV block: at hop 0 the rank fetches its own block from HBM, and at each
    later hop the block it holds arrived from the previous rank, so after P
    hops its Q has attended the whole cache under a single online softmax.

    Every rank runs the same step sequence, with the same slot rotation as the
    pipeline (`slot == step % 3`): the block computed at step `t` sits in slot
    `t % 3` and, unless `t` is the last hop, is sent to the next rank's slot
    `(t + 1) % 3`, where that rank computes on it at its step `t + 1`.

    Synchronization, in three parts.

    Entry barrier (`initial_handshake`): a rank does not touch a neighbour
    (remote DMA or remote semaphore signal) before both neighbours have
    entered this kernel call. The barrier semaphore belongs to the kernel's
    `collective_id`, so unlike the scoped semaphores it is safe to signal on a
    peer that is still outside the kernel.

    Slot credits (`sync_sem`): the block a rank sends at step `t` lands in the
    receiver's slot `(t + 1) % 3`, last used by the receiver's step `t - 2`.
    The receiver signals one credit per step, once that step no longer needs
    its slot (compute done and its own onward send complete); the sender
    takes one credit per step before sending. With the two credits granted at
    entry the sender is then never more than two steps ahead of the receiver
    releasing slots, which is exactly the condition above. Credits are counted
    on every step, including the hop-0 steps whose block comes from HBM:
    those occupy a slot too.

    Send completion: the onward send of step `t` is waited at the top of step
    `t + 1`, before that step issues the HBM fetch that reuses the slot (the
    pipeline fetches two steps ahead into slot `(t + 3) % 3`), and before the
    slot's credit is released.

    google3 differs in the last two: it counts credits per received block
    only and waits a step's send at the end of the next step, i.e. after the
    fetch that reuses the slot has been issued. Both are fine as long as all
    ranks stay within a step of each other; the scheme here does not depend on
    that.
    """

    n_buffer: int = 3
    # Slots that are free when the kernel starts and have no earlier step to
    # release them: those of steps 0 and 1 (step `t` waits for step `t - 2`).
    initial_credits: int = 2

    def __init__(
        self, cfgs: configs.RpaConfigs, sems: RingSems, *, kv_window_ref: jax.Ref
    ):
        if configs.NUM_INPUT_SLOTS != self.n_buffer:
            raise ValueError(
                f"RingAttention requires NUM_INPUT_SLOTS == {self.n_buffer} for"
                f" triple buffering, got {configs.NUM_INPUT_SLOTS}"
            )
        if cfgs.serve.cp_group_size is None or cfgs.serve.cp_group_size < 2:
            raise ValueError(
                f"RingAttention requires cp_group_size >= 2, got"
                f" {cfgs.serve.cp_group_size=}"
            )
        self.cfgs = cfgs
        self.size: int = cfgs.serve.cp_group_size
        self.kv_window = kv_window_ref
        self.sems = sems
        my_id = lax.axis_index(cfgs.serve.ring_axis_name)
        self.next_id = self._device_id(lax.rem(my_id + 1, self.size))
        self.prev_id = self._device_id(lax.rem(my_id + self.size - 1, self.size))
        self._step_flags = None

    def _device_id(self, rank):
        names = self.cfgs.serve.ring_mesh_axis_names
        if names is None:
            return (rank,)
        return tuple(
            rank if name == self.cfgs.serve.ring_axis_name else lax.axis_index(name)
            for name in names
        )

    def _signal(self, sem, value, device_id):
        pl.semaphore_signal(
            sem, value, device_id=device_id, device_id_type=pl.DeviceIdType.MESH
        )

    def initial_handshake(self):
        """Entry barrier, then the initial slot credits. Call before the pipeline."""
        barrier = pltpu.get_barrier_semaphore()
        for device_id in (self.prev_id, self.next_id):
            self._signal(barrier, 1, device_id)
        pl.semaphore_wait(barrier, 2)
        self._signal(self.sems.sync_sem, self.initial_credits, self.prev_id)

    def drain_credits(self):
        """Consume the successor's initial grants. Call after the pipeline."""
        pl.semaphore_wait(self.sems.sync_sem, self.initial_credits)

    def _remote_copy(self, src_slot, dst_slot):
        return pltpu.make_async_remote_copy(
            src_ref=self.kv_window.at[src_slot],
            dst_ref=self.kv_window.at[dst_slot],
            send_sem=self.sems.dma_sems.at[src_slot, 0],
            recv_sem=self.sems.dma_sems.at[dst_slot, 1],
            device_id=self.next_id,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _sends(self, schedule_ref, step):
        is_valid = schedule_ref.s_idx[step, 0] != -1
        hop = lax.rem(schedule_ref.k_idx[step, 0], self.size)
        return jnp.logical_and(is_valid, hop < self.size - 1)

    def release_previous_step(self, step, schedule_ref):
        """Top of step `step`: the previous step is done with its slot.

        Must run before the step issues its lookahead fetch, which may target
        that slot. `rpa_body` calls it inside the fence region.
        """
        prev_step = jnp.maximum(step - 1, 0)
        has_prev = step >= 1
        prev_slot = lax.rem(prev_step, self.n_buffer)

        @pl.when(jnp.logical_and(has_prev, self._sends(schedule_ref, prev_step)))
        def _wait_prev_send():
            self._remote_copy(prev_slot, prev_slot).wait_send()

        @pl.when(has_prev)
        def _release_prev_slot():
            self._signal(self.sems.sync_sem, 1, self.prev_id)

    def receive_and_forward(self, step, schedule_ref, total_steps):
        """Take this step's rotated block, then pass it on."""
        del total_steps
        slot = lax.rem(step, self.n_buffer)
        next_slot = lax.rem(step + 1, self.n_buffer)

        is_valid = schedule_ref.s_idx[step, 0] != -1
        hop = lax.rem(schedule_ref.k_idx[step, 0], self.size)
        receives = jnp.logical_and(is_valid, hop > 0)
        sends = self._sends(schedule_ref, step)

        @pl.when(receives)
        def _wait_recv():
            self._remote_copy(slot, slot).wait_recv()

        # One credit per step: the receiver has released the slot of its step
        # `step - 2`, which is the slot this step's block goes to.
        pl.semaphore_wait(self.sems.sync_sem, 1)

        send_op = self._remote_copy(slot, next_slot)

        @pl.when(sends)
        def _start_send():
            send_op.start()

        self._step_flags = (sends, send_op)

    def finish(self, step: int | jax.Array, schedule_ref, total_steps):
        """End of the step. Only the last step of a pipeline run has work here:
        it has no next step to release its slot."""
        del schedule_ref
        assert self._step_flags is not None
        sends, send_op = self._step_flags
        self._step_flags = None
        is_last_step = step == total_steps - 1

        @pl.when(jnp.logical_and(is_last_step, sends))
        def _wait_last_send():
            send_op.wait_send()

        @pl.when(is_last_step)
        def _release_last_slot():
            self._signal(self.sems.sync_sem, 1, self.prev_id)


@dataclasses.dataclass(frozen=True)
class RingCacheScheduler(StripedCacheScheduler):
    """PCP ring scheduler rotating the page-striped KV cache across ranks.

    `kv_lens - q_lens` is the *global* cached length of each sequence, which
    must be the same on every rank, as must `cu_q_lens`: all ranks have to run
    the same step sequence. A sequence with no cached token gets no step.

    The stripe length used to mask a block received at hop `h` is that of rank
    `cp_rank - h`, so `cp_rank` must increase by one (mod the group size) along
    the ring axis.

    Attributes:
      ring_axis_name: Mesh axis the ring runs on.
      ring_mesh_axis_names: Names of all axes of the enclosing `shard_map`
        mesh, in mesh order; None when the ring axis is the only one.
      collective_id: Pallas collective id of the kernel (barrier semaphore).
    """

    ring_axis_name: str = "pcp"
    ring_mesh_axis_names: tuple[str, ...] | None = None
    collective_id: int = 3

    def get_collective_id(self) -> int | None:
        return self.collective_id

    def get_ring_sems_spec(self) -> RingSems:
        return RingSems(
            dma_sems=pltpu.SemaphoreType.DMA((configs.NUM_INPUT_SLOTS, 2)),
            sync_sem=pltpu.SemaphoreType.REGULAR,
        )

    def create_ring(
        self,
        cfgs: configs.RpaConfigs,
        ring_sems: RingSems,
        *,
        kv_window_ref: jax.Ref,
    ) -> RingAttention:
        return RingAttention(cfgs, ring_sems, kv_window_ref=kv_window_ref)

    def get_kv_lens(
        self,
        kv_lens: jax.Array,
        q_lens: jax.Array,
        page_size: int = 1,
        extra_scalars: Sequence[jax.Array] = (),
        kv_lens_global: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del page_size, extra_scalars, kv_lens_global
        global_cache_lens = kv_lens - q_lens
        return global_cache_lens, jnp.zeros_like(q_lens), global_cache_lens

    def get_kv_start(
        self,
        k_idx: jax.Array,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> jax.Array:
        del extra_refs
        return (k_idx // self.cp_group_size) * cfgs.bkv_sz

    def get_local_k_end(
        self,
        s_idx: jax.Array,
        k_idx: jax.Array,
        kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> jax.Array:
        del kv_new_lens_ref
        cp_rank = self._get_cp_rank(extra_refs)
        cp_size = self.cp_group_size
        hop = lax.rem(k_idx, cp_size)
        # The block held at hop `h` was fetched by the rank `h` places up the
        # ring; mask it with that rank's stripe length.
        src_rank = lax.rem(cp_rank + cp_size - hop, cp_size)
        global_cache_len = kv_cache_lens_ref[s_idx]
        return utils.cp_local_cache_len(
            global_cache_len, cp_size, src_rank, cfgs.serve.page_size
        )

    @jax.named_scope("q_loop_cp_ring")
    def q_loop(
        self,
        q_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_start,
        q_end,
        q_offset,
        kv_cache_len,
        kv_new_len,
        num_k,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> schedule.LoopCarry:
        del q_offset, kv_new_len, num_k
        cp_size = self.cp_group_size

        # Every rank walks rank 0's stripe length (the longest) so all ranks
        # execute identical step counts and rotate in lockstep.
        local_kv_cache_len = utils.cp_local_cache_len(
            kv_cache_len, cp_size, 0, cfgs.serve.page_size
        )
        num_k = pl.cdiv(local_kv_cache_len, cfgs.bkv_sz)
        num_k_groups = pl.cdiv(num_k, cfgs.batch_size)
        end_k_idx = num_k_groups * cp_size * cfgs.batch_size

        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        k_loop_fn = functools.partial(
            self.k_loop,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            kv_cache_len=local_kv_cache_len,
            kv_new_len=0,
            end_k_idx=end_k_idx,
            p_offset=p_offset,
            page_indices_ref=page_indices_ref,
            schedule_ref=schedule_ref,
            schedule_hbm_ref=schedule_hbm_ref,
            dma_sem=dma_sem,
            cfgs=cfgs,
            extra_refs=extra_refs,
        )
        return jax.lax.fori_loop(0, end_k_idx, k_loop_fn, carry)

    @jax.named_scope("k_loop_cp_ring")
    def k_loop(
        self,
        k_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> schedule.LoopCarry:
        del q_end, kv_new_len, end_k_idx, extra_refs
        count = carry.count
        step, target_lane = divmod(count, cfgs.batch_size)
        cp_group_size = self.cp_group_size

        step_within_q = k_idx // cfgs.batch_size
        k_group = step_within_q // cp_group_size
        hop = lax.rem(step_within_q, cp_group_size)
        is_round_0 = hop == 0
        num_k = pl.cdiv(kv_cache_len, cfgs.bkv_sz)
        local_k_idx = k_group * cfgs.batch_size + target_lane
        is_valid_task = local_k_idx < num_k
        loop_k_idx = local_k_idx * cp_group_size + hop
        is_last_k = jnp.where(
            (hop == cp_group_size - 1) & (local_k_idx == num_k - 1), 1, 0
        )

        schedule_ref.s_idx[step, target_lane] = jnp.where(is_valid_task, s_idx, -1)
        schedule_ref.q_idx[step, target_lane] = jnp.where(is_valid_task, q_idx, 0)
        schedule_ref.k_idx[step, target_lane] = jnp.where(is_valid_task, loop_k_idx, 0)
        schedule_ref.is_last_k[step, target_lane] = is_last_k
        schedule_ref.do_writeback[step, target_lane] = 0

        schedule_ref.dma_q[step, target_lane, 0] = jnp.where(is_valid_task, q_src, 0)
        schedule_ref.dma_q[step, target_lane, 1] = jnp.where(
            is_valid_task, q_sz_task, 0
        )

        kv_len_start = local_k_idx * cfgs.bkv_sz
        kv_p_start = local_k_idx * cfgs.bkv_p
        kv_left_frm_cache = jnp.where(
            is_valid_task, jnp.maximum(kv_cache_len - kv_len_start, 0), 0
        )

        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = jnp.clip(kv_left_frm_cache - dst_vmem, 0, cfgs.serve.page_size)
            p_idx = jnp.minimum(kv_p_start + i, cfgs.serve.pages_per_seq - 1)
            src_hbm = page_indices_ref[p_offset + p_idx]
            # Only round 0 (hop == 0) fetches from HBM; subsequent rounds receive KV
            # from peer devices over the ICI ring.
            fetch_sz = jnp.where(is_round_0, dma_sz, 0)
            schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm
            schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
            schedule_ref.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
                fetch_sz, cfgs.kv_token_align
            )

        for i in range(cfgs.bkv_p_new):
            dma_entry = schedule_ref.dma_kv_new[step, target_lane, i]
            dma_entry.fetch_hbm[...] = 0
            dma_entry.fetch_vmem[...] = 0
            dma_entry.wb_hbm[...] = 0
            dma_entry.wb_vmem[...] = 0
            dma_entry.set_flags(0, 0)

        return schedule.advance_carry(
            carry,
            schedule_ref,
            schedule_hbm_ref,
            dma_sem,
            cfgs=cfgs,
        )


@dataclasses.dataclass(frozen=True)
class StripedNewTokensScheduler(CPScheduler):
    """This scheduler writes back only owned pages.

    Two uses, told apart by `cfgs.serve.paged_new_kv`:

    Contiguous new K/V (decode context parallelism): a sequence's new K/V are
    its `q_len` rows of keys/values, and each new K block is written back by
    the first Q block that attends it.

    Paged new K/V (prefill context parallelism, google3 cl/985434096):
    `extra_scalars = (cp_rank, new_kv_page_indices[, new_kv_write_rows])`. A
    sequence's `kv_new_len` new tokens are read through its row of
    `new_kv_page_indices` (page `j` of the new tokens is page
    `new_kv_page_indices[s, j]` of keys/values), so they need not be the
    sequence's query rows: under PCP a rank's queries are two zig-zag blocks of
    the chunk while its keys are the whole chunk. `q_offsets` is the global
    position of the first query. A sequence writes its new tokens back on its
    last Q block, when its `new_kv_write_rows` entry is nonzero (every sequence
    when the operand is absent); that Q block then walks all the new K blocks,
    not only those up to its causal bound, so that every page this rank owns
    is fetched and written. (google3 stops at the causal bound, which leaves
    the owned pages behind a rank's last query unwritten.) A sequence with no
    query writes nothing.
    """

    scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.NEW_TOKENS_ONLY

    def _writes_new_kv(self, s_idx, cfgs: configs.RpaConfigs, extra_refs):
        """Paged new K/V: whether sequence `s_idx` writes its new tokens back."""
        if cfgs.serve.skip_kv_write:
            return False
        if len(extra_refs) > 2:
            return extra_refs[2][s_idx] != 0
        return True

    @jax.named_scope("q_loop_cp_new")
    def q_loop(
        self,
        q_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_start,
        q_end,
        q_offset,
        kv_cache_len,
        kv_new_len,
        num_k,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> schedule.LoopCarry:
        kwargs = dict(
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            num_k=num_k,
            p_offset=p_offset,
            page_indices_ref=page_indices_ref,
            schedule_ref=schedule_ref,
            schedule_hbm_ref=schedule_hbm_ref,
            dma_sem=dma_sem,
            cfgs=cfgs,
            extra_refs=extra_refs,
        )
        if not cfgs.serve.paged_new_kv:
            return super().q_loop(q_idx, carry, **kwargs)

        q_len = q_end - q_start
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)
        start_k_idx, end_k_idx = self.get_k_range(
            q_idx=q_idx,
            q_len=q_len,
            q_sz_task=q_sz_task,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            num_k=num_k,
            cfgs=cfgs,
            extra_refs=extra_refs,
        )
        # The writing Q block fetches every new K block, so that the pages this
        # rank owns past the block's causal bound are written too. The keys
        # past the bound are masked for attention as usual.
        is_wb_q = (q_idx == pl.cdiv(q_len, cfgs.bq_sz) - 1) & self._writes_new_kv(
            s_idx, cfgs, extra_refs
        )
        end_k_idx = jnp.where(is_wb_q, num_k, end_k_idx)

        def k_loop_fn(k_idx, c: schedule.LoopCarry) -> schedule.LoopCarry:
            return self.k_loop(
                k_idx,
                c,
                s_idx=s_idx,
                q_idx=q_idx,
                q_end=q_end,
                q_src=q_src,
                q_sz_task=q_sz_task,
                kv_cache_len=kv_cache_len,
                kv_new_len=kv_new_len,
                end_k_idx=end_k_idx,
                p_offset=p_offset,
                page_indices_ref=page_indices_ref,
                schedule_ref=schedule_ref,
                schedule_hbm_ref=schedule_hbm_ref,
                dma_sem=dma_sem,
                cfgs=cfgs,
                extra_refs=extra_refs,
            )

        return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

    def get_rpa_schedule(
        self, cfgs: configs.RpaConfigs, multiplier: int = 1
    ) -> schedule.RpaSchedule:
        """Returns the RpaSchedule struct with CP-specific DMA struct."""
        dma_kv_new_struct_cls = (
            schedule.HeadAlongSublaneDmaNewCP
            if cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
            else schedule.SeqAlongLaneDmaNew
        )
        return schedule.RpaSchedule.create_shape_dtype(
            cfgs,
            dma_kv_new_struct_cls=dma_kv_new_struct_cls,
            multiplier=multiplier,
        )

    @jax.named_scope("k_loop_cp")
    def k_loop(
        self,
        k_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ) -> schedule.LoopCarry:
        cp_rank = self._get_cp_rank(extra_refs)
        cp_group_size = self.cp_group_size
        count = carry.count
        step, target_lane = divmod(count, cfgs.batch_size)

        schedule_ref.s_idx[step, target_lane] = s_idx
        schedule_ref.q_idx[step, target_lane] = q_idx
        schedule_ref.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        schedule_ref.is_last_k[step, target_lane] = is_last_k

        schedule_ref.dma_q[step, target_lane, 0] = q_src
        schedule_ref.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        k_len = kv_cache_len + kv_new_len
        kv_left = k_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)

        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = kv_left_frm_cache - dst_vmem
            dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                p_idx = kv_p_start + i
                local_page_i = jnp.minimum(
                    p_idx // cp_group_size,
                    cfgs.serve.pages_per_seq - 1,
                )
                src_hbm_cp = page_indices_ref[p_offset + local_page_i]
                dma_sz = jnp.where(p_idx % cp_group_size == cp_rank, dma_sz, 0)
                schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm_cp
                schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
                schedule_ref.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
                    dma_sz, cfgs.kv_token_align
                )
            else:
                p_idx = jnp.minimum(kv_p_start + i, cfgs.serve.pages_per_seq - 1)
                src_hbm = page_indices_ref[p_offset + p_idx]
                schedule_ref.dma_kv_cache[step, target_lane, i, 0] = src_hbm
                schedule_ref.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
                schedule_ref.dma_kv_cache[step, target_lane, i, 2] = 0

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        if cfgs.serve.paged_new_kv:
            return self._paged_new_kv_entries(
                carry,
                step=step,
                target_lane=target_lane,
                s_idx=s_idx,
                q_idx=q_idx,
                q_len=q_end - (q_src - q_idx * cfgs.bq_sz),
                kv_cache_len=kv_cache_len,
                kv_len_start=kv_len_start,
                bkv_sz_cache=bkv_sz_cache,
                new_sz=new_sz,
                new_kv_pos=kv_new_len - kv_left_frm_new,
                p_offset=p_offset,
                page_indices_ref=page_indices_ref,
                schedule_ref=schedule_ref,
                schedule_hbm_ref=schedule_hbm_ref,
                dma_sem=dma_sem,
                cfgs=cfgs,
                extra_refs=extra_refs,
            )

        # Writeback logic: each new k block is written back by the first q block
        # that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - kv_cache_len)) // cfgs.bq_sz

        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        if cfgs.serve.skip_kv_write:
            # The new keys are still fetched and attended; only the write is off.
            do_writeback = 0
        schedule_ref.do_writeback[step, target_lane] = do_writeback
        src_hbm = q_end - kv_left_frm_new

        def fill_dma_kv_new(i, dst_vmem, dma_sz):
            dma_entry = schedule_ref.dma_kv_new[step, target_lane, i]
            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                align = cfgs.kv_token_align
                # Fetch: the new tokens are contiguous, so one DMA moves all of them
                # even though this loop runs once per page. Entry 0 carries that DMA
                # and the other entries stay zero.
                if i == 0:
                    src, _, fetch_val = utils.align_span(src_hbm, new_sz, align)
                    dma_entry.fetch_hbm[...] = src
                    dma_entry.fetch_vmem[...] = utils.align_to(bkv_sz_cache, align)
                    fetch_val = jnp.where(new_sz > 0, fetch_val, 0)
                else:
                    fetch_val = 0

                # Writeback: one DMA per page, since page_indices scatters them, and
                # only for the pages this rank owns.
                tok_idx = kv_len_start + dst_vmem
                p_idx = jnp.minimum(
                    tok_idx >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq * cp_group_size - 1,
                )
                p_off = tok_idx & cfgs.serve.page_size_mask
                dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
                dma_entry.wb_hbm[...] = (
                    page_indices_ref[p_offset + p_idx // cp_group_size]
                    << cfgs.serve.page_size_log2
                ) | dst
                dma_entry.wb_vmem[...] = dst_vmem - lead
                dma_entry.set_flags(
                    fetch_val,
                    jnp.where(
                        (dma_sz > 0) & (p_idx % cp_group_size == cp_rank), wb_val, 0
                    ),
                )
            else:
                tok_idx = kv_len_start + dst_vmem
                p_idx = jnp.minimum(
                    tok_idx >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq * cp_group_size - 1,
                )
                p_off = tok_idx & cfgs.serve.page_size_mask
                local_slot = p_idx // cp_group_size
                dst_hbm = (
                    page_indices_ref[p_offset + local_slot] << cfgs.serve.page_size_log2
                ) | p_off
                wb_val = jnp.where(
                    p_idx % cp_group_size == cp_rank, dma_sz, jnp.int32(0)
                )

                dma_entry.fetch_hbm[...] = src_hbm
                dma_entry.fetch_vmem[...] = dst_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.set_flags(dma_sz, wb_val)

        if cfgs.bkv_p_new < cfgs.bkv_p:
            # General decode path for any bkv_p_new
            curr_vmem = bkv_sz_cache
            curr_rem = new_sz

            for i in range(cfgs.bkv_p_new):
                slot_start = (curr_vmem // cfgs.serve.page_size) * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size

                dma_sz = jnp.minimum(curr_rem, slot_end - curr_vmem)
                dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)

                fill_dma_kv_new(i, curr_vmem, dma_sz)

                curr_vmem += dma_sz
                curr_rem = jnp.maximum(0, curr_rem - dma_sz)
        else:
            iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
            for i in range(iters):
                slot_start = i * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size

                dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

                fill_dma_kv_new(i, dst_vmem, dma_sz)

        return schedule.advance_carry(
            carry,
            schedule_ref,
            schedule_hbm_ref,
            dma_sem,
            cfgs=cfgs,
        )

    def _paged_new_kv_entries(
        self,
        carry: schedule.LoopCarry,
        *,
        step,
        target_lane,
        s_idx,
        q_idx,
        q_len,
        kv_cache_len,
        kv_len_start,
        bkv_sz_cache,
        new_sz,
        new_kv_pos,
        p_offset,
        page_indices_ref: jax.Ref,
        schedule_ref: schedule.RpaSchedule,
        schedule_hbm_ref: schedule.RpaSchedule,
        dma_sem: jax.Ref,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref],
    ) -> schedule.LoopCarry:
        """New-K/V DMA entries of one step with paged new K/V.

        The block's new tokens sit at VMEM offsets
        `[bkv_sz_cache, bkv_sz_cache + new_sz)`; `new_kv_pos` is the index of
        the first of them among the sequence's new tokens. A fetch must stay
        inside one page of keys/values (pages of the new tokens, counted from
        the first new token) and a writeback inside one cache page (pages of
        the sequence, counted from position 0). The two grids are offset by
        `kv_cache_len % page_size`, so each cache page of the block is cut in
        two at that offset: entries `2 * i` and `2 * i + 1`. The first is empty
        when the cached length is a whole number of pages.
        """
        if len(extra_refs) < 2:
            raise ValueError(
                "Paged new K/V requires `extra_scalars=(cp_rank,"
                " new_kv_page_indices, ...)`."
            )
        if cfgs.serve.kv_layout != configs.KVLayout.HEAD_ALONG_SUBLANE:
            raise NotImplementedError("Paged new K/V supports HEAD_ALONG_SUBLANE only.")
        assert cfgs.bkv_p_new == 2 * cfgs.bkv_p
        cp_rank = self._get_cp_rank(extra_refs)
        cp_group_size = self.cp_group_size
        page_size = cfgs.serve.page_size
        log2 = cfgs.serve.page_size_log2
        mask = cfgs.serve.page_size_mask
        new_kv_page_indices_ref = extra_refs[1]
        new_pages_per_seq = new_kv_page_indices_ref.shape[0] // cfgs.serve.num_seqs
        new_kv_row = s_idx * new_pages_per_seq

        # Writeback logic: the sequence's last q block writes every new k block
        # (see `q_loop`).
        q_wb = pl.cdiv(q_len, cfgs.bq_sz) - 1
        schedule_ref.do_writeback[step, target_lane] = jnp.where(
            (new_sz > 0)
            & (q_idx == q_wb)
            & self._writes_new_kv(s_idx, cfgs, extra_refs),
            1,
            0,
        )

        cut = kv_cache_len & mask
        new_end = bkv_sz_cache + new_sz
        for i in range(cfgs.bkv_p):
            page_start = i * page_size
            pieces = (
                (page_start, page_start + cut),
                (page_start + cut, page_start + page_size),
            )
            for half, (lo, hi) in enumerate(pieces):
                start = jnp.maximum(lo, bkv_sz_cache)
                dma_sz = jnp.maximum(0, jnp.minimum(hi, new_end) - start)
                # An empty entry points at offset 0: its fields are still
                # turned into (zero-size) DMAs.
                dst_vmem = jnp.where(dma_sz > 0, start, 0)

                rel_off = new_kv_pos + (dst_vmem - bkv_sz_cache)
                rel_off = jnp.where(dma_sz > 0, rel_off, 0)
                new_page = jnp.minimum(rel_off >> log2, new_pages_per_seq - 1)
                fetch_hbm = (new_kv_page_indices_ref[new_kv_row + new_page] << log2) | (
                    rel_off & mask
                )

                tok_idx = kv_len_start + dst_vmem
                p_idx = jnp.minimum(
                    tok_idx >> log2, cfgs.serve.pages_per_seq * cp_group_size - 1
                )
                dst_hbm = (
                    page_indices_ref[p_offset + p_idx // cp_group_size] << log2
                ) | (tok_idx & mask)
                wb_val = jnp.where(
                    p_idx % cp_group_size == cp_rank, dma_sz, jnp.int32(0)
                )

                dma_entry = schedule_ref.dma_kv_new[step, target_lane, 2 * i + half]
                dma_entry.fetch_hbm[...] = fetch_hbm
                dma_entry.fetch_vmem[...] = dst_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.set_flags(dma_sz, wb_val)

        return schedule.advance_carry(
            carry,
            schedule_ref,
            schedule_hbm_ref,
            dma_sem,
            cfgs=cfgs,
        )


# ---------------------------------------------------------------------------
# SGLang schedulers for decode context parallelism (DCP).
#
# google3's `StripedCacheScheduler` derives a rank's stripe length from the
# global length and always runs next to a `StripedNewTokensScheduler` launch.
# SGLang's backend hands the kernel the stripe lengths it already keeps, runs
# decode as one launch (`StripedDecodeScheduler`), and applies the sliding
# window in global positions.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StripedCacheOnlyScheduler(CPScheduler):
    """Attends this rank's page stripe of the cache; nothing is written.

    `kv_lens` are the lengths of this rank's stripes (local page `j` is global
    page `j * cp_group_size + cp_rank`) and `kv_lens_global` the global lengths,
    whose last `q_len` positions are the queries. Every stripe key precedes
    every query, so there is no causal bound; the kernels get the first query's
    global position as `q_offsets` and mask the sliding window in global
    positions. One KV block always runs, so a query block with no key in range
    (an empty stripe, or a window that starts past it) still writes output 0
    and LSE -inf.
    """

    scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.CACHE_ONLY

    def _new_token_lens(
        self,
        kv_lens: jax.Array,
        q_lens: jax.Array,
        kv_lens_global: jax.Array,
        page_size: int,
        cp_rank: jax.Array,
    ) -> jax.Array:
        """Per sequence, how many of the `kv_lens` keys come from keys/values."""
        del kv_lens, kv_lens_global, page_size, cp_rank
        return jnp.zeros_like(q_lens)

    def get_kv_lens(
        self,
        kv_lens: jax.Array,
        q_lens: jax.Array,
        page_size: int = 1,
        extra_scalars: Sequence[jax.Array] = (),
        kv_lens_global: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if kv_lens_global is None:
            raise ValueError(f"{type(self).__name__} requires `kv_lens_global`.")
        cp_rank = self._get_cp_rank(extra_scalars)
        kv_new_lens = self._new_token_lens(
            kv_lens, q_lens, kv_lens_global, page_size, cp_rank
        )
        return kv_lens - kv_new_lens, kv_new_lens, kv_lens_global - q_lens

    def get_k_range(
        self,
        *,
        q_idx,
        q_len,
        q_sz_task,
        q_offset,
        kv_cache_len,
        num_k,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ):
        del q_len, q_sz_task, kv_cache_len
        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            # First local key inside the window of the block's first query,
            # whose global position is `q_offset + q_idx * bq_sz`. The first
            # local page at or after the window's first global page p0 is
            # ceil((p0 - cp_rank) / cp_group_size); keys of that page before
            # the window are masked by the kernel.
            cp_rank = self._get_cp_rank(extra_refs)
            first_global = jnp.maximum(
                q_offset + q_idx * cfgs.bq_sz - sliding_window + 1, 0
            )
            first_page = first_global >> cfgs.serve.page_size_log2
            first_local_page = (
                jnp.maximum(first_page - cp_rank + self.cp_group_size - 1, 0)
                // self.cp_group_size
            )
            start_k_idx = (first_local_page << cfgs.serve.page_size_log2) // cfgs.bkv_sz
        return start_k_idx, jnp.maximum(num_k, start_k_idx + 1)


@dataclasses.dataclass(frozen=True)
class StripedDecodeScheduler(StripedCacheOnlyScheduler):
    """DCP decode in one launch (`AttentionScope.STRIPED`).

    Attends this rank's stripe and, on the rank that owns the page of a
    single-token row's new token (global position `kv_lens_global - 1`), also
    that token: `kv_lens` then counts it, it is fetched from keys/values,
    stitched at the end of the stripe -- exactly its slot, since the owner's
    stripe ends on that page -- and written back from there. Multi-token rows
    attend their stripe only; their tokens are the chunk pass's
    (`StripedChunkScheduler`).
    """

    scope: ClassVar[configs.AttentionScope] = configs.AttentionScope.STRIPED

    def _new_token_lens(
        self,
        kv_lens: jax.Array,
        q_lens: jax.Array,
        kv_lens_global: jax.Array,
        page_size: int,
        cp_rank: jax.Array,
    ) -> jax.Array:
        owns_token = (
            (kv_lens_global > 0)
            & (q_lens == 1)
            & (((kv_lens_global - 1) // page_size) % self.cp_group_size == cp_rank)
        )
        return (owns_token & (kv_lens > 0)).astype(q_lens.dtype)


@dataclasses.dataclass(frozen=True)
class StripedChunkScheduler(StripedNewTokensScheduler):
    """DCP chunk pass: `StripedNewTokensScheduler` minus the single-token rows.

    A single-token row is the STRIPED launch's (attended and written there),
    so it is left unprocessed here: output untouched, LSE -inf.
    """

    def get_k_range(
        self,
        *,
        q_idx,
        q_len,
        q_sz_task,
        q_offset,
        kv_cache_len,
        num_k,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
    ):
        start_k_idx, end_k_idx = super().get_k_range(
            q_idx=q_idx,
            q_len=q_len,
            q_sz_task=q_sz_task,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            num_k=num_k,
            cfgs=cfgs,
            extra_refs=extra_refs,
        )
        return start_k_idx, jnp.where(q_len == 1, start_k_idx, end_k_idx)
