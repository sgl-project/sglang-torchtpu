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
import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.batched_rpa import (
    bref_override,
    configs,
    flash_attention,
    pipeline,
    schedule,
)
from sglang_tpu.attention.pallas.batched_rpa import scheduler as scheduler_lib
from sglang_tpu.attention.pallas.batched_rpa import (
    stitch_utils,
    utils,
)


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
    assert start % cfgs.serve.packing_kv == 0
    start //= cfgs.serve.packing_kv
    kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
    kv_ref = kv_u32_ref.reshape(-1, cfgs.aligned_kv_head_dim)

    # kv_ref must be annotated with no_hazard_no_deps so XLA knows not to add a
    # dependency between the HBM->VMEM DMA copy-in from stage n - 1 and the vld
    # instructions for matmuls at stage n.
    kv_ref = pltpu.annotate(kv_ref, no_hazard_no_deps=True)

    if cfgs.serve.packing_kv == 1:
        k = utils.strided_load(
            kv_ref,
            start,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        v = utils.strided_load(
            kv_ref,
            start + 1,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        return [(k, v)]

    kv = utils.strided_load(
        kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride, cfgs.bkv_stride
    )
    bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)

    return utils.convert_to_target_bitwidth(
        kv, target_bitwidth=bitwidth, kv_dtype=cfgs.serve.dtype_kv
    )


def add_attention_sink(l: jax.Array, m: jax.Array, sink_ref: jax.Ref) -> jax.Array:
    """Adds exp(sink - m) to one lane's softmax denominator l: [KV, rows, 128].

    The sink is one extra key per row with logit sink[head] and a zero value.
    """
    num_rows = l.shape[1]
    num_kv_heads, period, num_lanes = sink_ref.shape
    if num_rows <= period:
        sink = sink_ref[:, :num_rows]
    else:
        assert num_rows % period == 0, (num_rows, period)
        sink = jnp.broadcast_to(
            sink_ref[...][:, None],
            (num_kv_heads, num_rows // period, period, num_lanes),
        ).reshape(l.shape)
    l_sink = l.astype(jnp.float32) + jnp.exp(sink - m.astype(jnp.float32))
    return l_sink.astype(l.dtype)


def store_lse(
    lse_vref: jax.Ref,
    b_idx: jax.Array,
    m: jax.Array,
    l: jax.Array,
    stage_ref: jax.Ref | None,
    sink_ref: jax.Ref | None,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stores one lane's lse = log(sum exp(s)) in f32, natural log.

    m, l: [KV, bq * G, lanes], rows (token, aligned q head) with the value
    repeated over the lanes. Token t of KV head h lands in the row
    lse_vref[b_idx, h, t, 0] with q head g in lane g. Without a sink, l == 0
    gives -inf; a sink adds exp(sink) to the sum.
    """
    num_heads = cfgs.aligned_num_q_heads_per_kv_head
    num_lanes = m.shape[-1]
    lane = lax.broadcasted_iota(jnp.int32, (cfgs.bq_sz, num_lanes), 1)

    def pack(rows: jax.Array) -> jax.Array:
        # Lane g of token t takes row t * num_heads + g.
        if rows.shape[0] == num_heads:
            row = lax.broadcasted_iota(jnp.int32, rows.shape, 0)
            own = lax.broadcasted_iota(jnp.int32, rows.shape, 1) == row
            masked = jnp.where(own, rows.astype(jnp.float32), 0.0)
            return jnp.sum(masked, axis=0, keepdims=True)
        stage_ref[...] = rows.astype(jnp.float32)
        packed = jnp.zeros((cfgs.bq_sz, num_lanes), jnp.float32)
        for head in range(num_heads):
            head_rows = stage_ref[pl.ds(head, cfgs.bq_sz, num_heads)]
            packed += jnp.where(lane == head, head_rows, 0.0)
        return packed

    for kv_head in range(m.shape[0]):
        m_packed = pack(m[kv_head])
        l_packed = pack(l[kv_head])
        if sink_ref is None:
            lse = m_packed + jnp.log(l_packed)
        else:
            # log(l * e^m + e^sink), finite also for a row that attended no key.
            sink = pack(sink_ref[kv_head, :num_heads])
            top = jnp.maximum(m_packed, sink)
            lse = top + jnp.log(
                l_packed * jnp.exp(m_packed - top) + jnp.exp(sink - top)
            )
        lse_vref.at[b_idx, kv_head].reshape(cfgs.bq_sz, num_lanes)[...] = lse


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_list: list[jax.Array],
    l_list: list[jax.Array],
    m_list: list[jax.Array] | None,
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    *,
    attention_sink_ref: jax.Ref | None,
    lse_stage_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
):
    """Normalizes and stores the output (and LSE) of lanes at their last KV step.

    Unlike google3, the attention sink joins the softmax denominator here, once
    per query block and in float32 (`add_attention_sink`), instead of seeding
    the running max and sum; and the LSE is stored as one float32 row of lanes
    per (KV head, token) (`store_lse`).
    """

    def _accum(
        b_idx: jax.Array,
        batch_acc: jax.Array,
        batch_l: jax.Array,
        batch_m: jax.Array | None,
    ):
        if cfgs.serve.return_lse:
            store_lse(
                lse_o_vref,
                b_idx,
                batch_m,
                batch_l,
                lse_stage_ref,
                attention_sink_ref,
                cfgs=cfgs,
            )
        if cfgs.model.has_attention_sink:
            batch_l = add_attention_sink(batch_l, batch_m, attention_sink_ref)
        if cfgs.serve.attention_scope.walks_cache_stripe:
            # A row that attended no key (empty stripe) keeps l == 0 and
            # outputs 0.
            batch_l = jnp.where(batch_l == 0, jnp.ones_like(batch_l), batch_l)
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (
            cfgs.serve.dtype_out == jnp.float32
            or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16
        ):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc * pltpu.reciprocal(batch_l, approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        pad_width = [[0, 0] for _ in range(out.ndim)]
        pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.compute_kv_head_dim
        out = jnp.pad(out, pad_width, constant_values=0)
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

    if cfgs.fuse_accum:
        for b in range(cfgs.batch_size):
            _accum(b, acc_list[b], l_list[b], None if m_list is None else m_list[b])
    else:
        # Adding a conditional causes a scheduling barrier. In prefill, we often
        # use small block sizes, so it's not worth executing the accumulation
        # on every block. In decode, because of the large block sizes / and or
        # batch sizes, we almost always use accumulation on every block. Please
        # tune `fuse_accum` for your use case.
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            acc_val = acc_list[b]
            l_val = l_list[b]
            m_val = None if m_list is None else m_list[b]
            accum_named_call = jax.named_call(_accum, name=f"accum_{b}")
            jax.lax.cond(
                is_last_k, accum_named_call, lambda *_: None, b, acc_val, l_val, m_val
            )


def get_scale_factors(
    k: jax.Array,
    v: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    if not cfgs.serve.per_token_scale:
        return k, v, cfgs.serve.scale_k, cfgs.serve.scale_v

    b = k.shape[0]
    num_heads = k.shape[1]
    # Number of scale channels in VMEM is determined by the scale factor bitwidth
    # and the VMEM kv bitwidth (k.dtype). However, we cannot use the
    # scale_channels configs.py member variable because of the case of unpacking
    # uint8_t to fp4_e2m1fn.
    if cfgs.serve.per_token_scale_dtype is None:
        scale_channels = 0
    else:
        scale_bits = jax.dtypes.itemsize_bits(cfgs.serve.per_token_scale_dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        scale_channels = max(1, scale_bits // kv_bits)

    # Extract multi-channel scale factor slices from the trailing sublanes
    k_scale_slice = k[
        :, :, cfgs.model.head_dim : cfgs.model.head_dim + scale_channels, :
    ]
    v_scale_slice = v[
        :, :, cfgs.model.head_dim : cfgs.model.head_dim + scale_channels, :
    ]

    # If there are multiple scale channels, pltpu.bitcast operates on the
    # second to last dimension, which is the head dimension, which is the same
    # as the scale_channels dimension, so this works out of the box.
    if scale_channels > 1:
        k_scale = pltpu.bitcast(k_scale_slice, cfgs.serve.per_token_scale_dtype)
        v_scale = pltpu.bitcast(v_scale_slice, cfgs.serve.per_token_scale_dtype)

    # If there is only one scale channel, just change dtype to the scale dtype.
    else:
        k_scale = k_scale_slice.astype(cfgs.serve.per_token_scale_dtype)
        v_scale = v_scale_slice.astype(cfgs.serve.per_token_scale_dtype)

    # In QK multiplication, we always scale after the qk matmul, so we can
    # unify per-token and per-tensor scaling by broadcasting the scale factor
    # to the head dimension outside of the flash attention kernel.
    k_scale = k_scale.reshape(b, num_heads, cfgs.bkv_sz)
    # To match the accumulation dtype of qk.
    k_scale = k_scale.astype(jnp.float32)
    # Add a new dimension for the head dimension.
    k_scale = k_scale[:, :, jnp.newaxis, :]

    # In PV multiplication, we scale the p rather than the pv for per-token
    # scaling (as it is mathematically necessary to scale before the matmul for
    # per-token scaling) so we can NOT unify per-token and per-tensor scaling,
    # so we broadcast the scale factor along the head dimension inside the
    # flash attention kernel itself and do not do so here.
    v_scale = v_scale.reshape(b, num_heads, cfgs.bkv_sz)

    k = k[:, :, : cfgs.compute_kv_head_dim, :]
    v = v[:, :, : cfgs.compute_kv_head_dim, :]

    return k, v, k_scale, v_scale


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StitchMetadata:
    """Scalars needed to merge newly arrived KV into the cached KV window."""

    bkv_sz_frm_cache: list[jax.Array]
    bkv_sz_frm_new: list[jax.Array]
    new_kv_len_start: list[jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ComputeMetadata:
    """Scalars needed to build the QK mask, but not needed by the stitch."""

    causal_offset: list[jax.Array]
    local_k_start: list[jax.Array] | None = None
    local_k_end: list[jax.Array] | None = None
    # Sliding window in global positions over a page-striped cache.
    k_start: list[jax.Array] | None = None
    global_q_k_gap: list[jax.Array] | None = None


# The `pl.when` region in `rpa_body` is a basic block boundary, so splitting
# these keeps the mask arithmetic isolated to the bodies that need it.
def fetch_stitch_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    cu_q_lens_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> StitchMetadata:
    """Fetches the scalars consumed by the KV stitch.

    Args:
      step: Current pipeline step.
      schedule_ref: Per-step sequence, query, and KV block indices.
      cu_q_lens_ref: Cumulative query lengths, indexed by sequence.
      kv_cache_lens_ref: Cached KV length per sequence.
      kv_new_lens_ref: Newly arrived KV length per sequence.
      cfgs: Kernel configs.

    Returns:
      The stitch boundaries for every batch element in this step.
    """
    bkv_sz_frm_cache_list = []
    bkv_sz_frm_new_list = []
    new_kv_len_start_list = []

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        # Clamp the sentinel (-1, written by mask_out_steps for padded steps)
        # so the eager arm of jnp.where below reads an in-bounds slot. The
        # kernel runs with disable_bounds_checks=True, which assumes indices
        # are already in range.
        safe_s_idx = jnp.maximum(0, s_idx)
        k_idx = schedule_ref.k_idx[step, b_idx]
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[safe_s_idx], 0)
        kv_new_len_val = jnp.where(is_valid, kv_new_lens_ref[safe_s_idx], 0)
        q_end = jnp.where(is_valid, cu_q_lens_ref[safe_s_idx + 1], 0)

        # KV stitch boundaries: where the cached tokens end and the newly arrived
        # tokens begin, which is what `stitch_utils.stitch_new_kv_lane` merges on.
        total_kv_len = kv_cache_len_val + kv_new_len_val
        kv_left = jnp.maximum(total_kv_len - k_id, 0)
        if cfgs.serve.skip_kv_update:
            kv_left_frm_cache = kv_left
        else:
            kv_left_frm_cache = jnp.maximum(kv_cache_len_val - k_id, 0)
        kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)
        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        bkv_sz_frm_cache_list.append(bkv_sz_frm_cache)
        bkv_sz_frm_new_list.append(jnp.minimum(kv_left, cfgs.bkv_sz) - bkv_sz_frm_cache)
        new_kv_len_start_list.append(q_end - kv_left_frm_new)

    return StitchMetadata(
        bkv_sz_frm_cache=bkv_sz_frm_cache_list,
        bkv_sz_frm_new=bkv_sz_frm_new_list,
        new_kv_len_start=new_kv_len_start_list,
    )


def fetch_compute_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    extra_refs: Sequence[jax.Ref] = (),
    *,
    cfgs: configs.RpaConfigs,
    scheduler: scheduler_lib.BaseScheduler,
) -> ComputeMetadata:
    """Fetches the scalars consumed by the QK mask.

    Args:
      step: Current pipeline step.
      schedule_ref: Per-step sequence, query, and KV block indices.
      q_offsets_ref: Query offset per sequence.
      kv_cache_lens_ref: Cached KV length per sequence. Only read when the
        attention scope needs `local_k_start` / `local_k_end`.
      kv_new_lens_ref: New KV length per sequence. Only read when the attention
        scope needs `local_k_end`.
      extra_refs: Additional scalar prefetch refs for custom schedulers.
      cfgs: Kernel configs.
      scheduler: Scheduler instance defining KV coordinate mapping.

    Returns:
      The mask intervals for every batch element in this step.
    """
    scope = cfgs.serve.attention_scope
    causal_offset_list = []
    local_k_start_list = [] if scope.skips_cached_keys else None
    local_k_end_list = [] if scope.walks_cache_stripe else None
    # Page-striped cache with a sliding window: the window runs in global
    # positions, `q_offsets` being the global position of the first query.
    global_window = (
        cfgs.serve.cp_group_size is not None
        and cfgs.model.sliding_window is not None
        and scope.walks_cache_stripe
    )
    k_start_list = [] if global_window else None
    global_q_k_gap_list = [] if global_window else None

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        # Clamp the sentinel (-1, written by mask_out_steps for padded steps)
        # so the eager arm of jnp.where below reads an in-bounds slot. The
        # kernel runs with disable_bounds_checks=True, which assumes indices
        # are already in range.
        safe_s_idx = jnp.maximum(0, s_idx)
        q_idx = schedule_ref.q_idx[step, b_idx]
        k_idx = schedule_ref.k_idx[step, b_idx]
        q_offset = jnp.where(is_valid, q_offsets_ref[safe_s_idx], 0)

        # Causal base offset: K_base - Q_base
        q_base = scheduler.get_q_start(q_idx, q_offset, cfgs)
        k_base = scheduler.get_kv_start(k_idx, cfgs, extra_refs)
        k_id = jnp.where(is_valid, k_base, 0)
        causal_offset = k_id - q_base
        causal_offset_list.append(causal_offset)

        if local_k_start_list is not None:
            # NEW_TOKENS_ONLY: keys before the first new token are masked.
            kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[safe_s_idx], 0)
            local_k_start_list.append(kv_cache_len_val - k_id)
        if local_k_end_list is not None:
            # CACHE_ONLY / STRIPED: every key of the call precedes every query;
            # keys past its length are stale.
            kv_len_val = jnp.where(
                is_valid,
                kv_cache_lens_ref[safe_s_idx] + kv_new_lens_ref[safe_s_idx],
                0,
            )
            local_k_end_list.append(kv_len_val - k_id)
        if global_q_k_gap_list is not None:
            # Local key j holds global position
            # j + (j // page_size) * (cp - 1) * page_size + rank * page_size. The
            # mask adds the j-dependent stripe term per key; the scalar part of
            # the gap is q_global - (k_id + rank * page_size).
            cp_rank = extra_refs[0][0]
            k_start_list.append(k_id)
            global_q_k_gap_list.append(q_base - (k_id + cp_rank * cfgs.serve.page_size))

    return ComputeMetadata(
        causal_offset=causal_offset_list,
        local_k_start=local_k_start_list,
        local_k_end=local_k_end_list,
        k_start=k_start_list,
        global_q_k_gap=global_q_k_gap_list,
    )


def generate_mask(
    shape: tuple[int, int, int, int],
    *,
    bq_start: int,
    step_meta: ComputeMetadata,
    cfgs: configs.RpaConfigs,
) -> list[jax.Array]:
    """Generates causal, sliding window, and attention scope mask for QK computation."""
    b, k_heads, tq, s = shape
    scope = cfgs.serve.attention_scope
    sliding_window = cfgs.model.sliding_window

    if scope.walks_cache_stripe and sliding_window is None:
        # Every query attends every key of the call, so the causal compare
        # (q_kv_diff >= offset) is always true and only keys beyond the (local)
        # length need masking. A single [1, 1, s] key-validity vector broadcast
        # over the tile replaces the full [k_heads, tq, s] compare/select chain.
        assert step_meta.local_k_end is not None
        kv_iota = lax.broadcasted_iota(jnp.int32, (1, 1, s), 2)
        return [kv_iota < step_meta.local_k_end[b_idx] for b_idx in range(b)]

    kv_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 2)
    q_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
    q_iota //= cfgs.aligned_num_q_heads_per_kv_head
    q_kv_diff = q_iota - kv_iota
    if step_meta.global_q_k_gap is not None:
        # Page-striped cache: the rank term of a key's global position is scalar
        # and lives in global_q_k_gap; the local-index part is computed per
        # batch element below from the block's first key.
        lp = cfgs.serve.page_size_log2
        stripe_gap = (cfgs.serve.cp_group_size - 1) << lp

    masks = []
    for b_idx in range(b):
        # NOTE: Goal is to compute q_len >= kv_len. But we want to utilize scalar
        # compute as much as possible before involving vector compute. Therefore, we
        # break down a computational steps into following equations to separate out
        # scalar and vector compute.
        # q_len = q_iota + (bq_start + processed_q_len)
        # kv_len = kv_iota + processed_kv_len
        # Step 1: We already preprocessed causal_offset
        #   causal_offset = kv_len - q_len
        # Step 2
        #   offset = causal_offset - bq_start
        offset = step_meta.causal_offset[b_idx] - bq_start
        if scope.walks_cache_stripe:
            mask_b = kv_iota < step_meta.local_k_end[b_idx]
        else:
            mask_b = q_kv_diff >= offset
        if sliding_window is not None:
            if step_meta.global_q_k_gap is None:
                mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)
            else:
                # Same in global positions: gap = q_global - (k_id + rank * page)
                # at q_iota = kv_iota = 0 of the block; the stripe term adds
                # (cp - 1) pages per local page of the key's local index.
                local_key = kv_iota + step_meta.k_start[b_idx]
                q_kv_diff_global = q_kv_diff - (local_key >> lp) * stripe_gap
                window_offset = sliding_window - (
                    bq_start + step_meta.global_q_k_gap[b_idx]
                )
                mask_b = jnp.logical_and(mask_b, q_kv_diff_global < window_offset)

        if step_meta.local_k_start is not None:
            mask_b = jnp.logical_and(mask_b, kv_iota >= step_meta.local_k_start[b_idx])

        masks.append(mask_b)

    return masks


def rpa_body(
    step: int | jax.Array,
    slot: int | jax.Array,
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    extra_refs: Sequence[jax.Ref] = (),
    attention_sink_ref: jax.Ref | None,  # [KV, period, 128] f32
    lse_stage_ref: jax.Ref | None = None,
    # Configs.
    cfgs: configs.RpaConfigs,
    scheduler: scheduler_lib.BaseScheduler,
    # DMA-issue hooks. Both only start copies; the matching waits live in
    # `pipeline.py`
    start_next_fetch: Callable[[], None],
    start_kv_writeback: Callable[[], None],
):
    del slot

    kv_in_vref = pltpu.annotate(kv_in_vref, no_hazard_no_deps=True)

    # Step 1: Fetch the metadata the stitch needs. Everything the mask needs is
    # deferred until after the fence below, so it does not gate the writeback.
    stitch_meta = fetch_stitch_metadata(
        step,
        schedule_ref,
        cu_q_lens_ref,
        kv_cache_lens_ref,
        kv_new_lens_ref,
        cfgs=cfgs,
    )

    always_true = kv_cache_lens_ref[0] >= 0

    # This conditional always executes. We create a basic block boundary here so:
    # 1. We force our KV writeback to be scheduled at the beginning of the
    #    compute region (as soon as stitching completes).
    # 2. We guarantee our stitching is correct: with `no_hazard_no_deps=True`, the
    #    boundary ensures VMEM writes from stitching finish before subsequent
    #    loads read the updated KV cache.

    @pl.when(always_true)
    @jax.named_scope("stitch_and_issue_writeback")
    def _kv_store_fence():
        if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            stitch_results = []
            for b_idx in range(cfgs.batch_size):
                res = stitch_utils.stitch_new_kv_lane(
                    kv_in_vref,
                    b_idx,
                    stitch_meta.bkv_sz_frm_cache[b_idx],
                    stitch_meta.new_kv_len_start[b_idx],
                    stitch_meta.bkv_sz_frm_new[b_idx],
                    cfgs=cfgs,
                )
                stitch_results.append(res)
            for b_idx in range(cfgs.batch_size):
                stitch_utils.store_new_kv_lane(
                    kv_in_vref,
                    b_idx,
                    stitch_results[b_idx],
                    cfgs=cfgs,
                )

        start_kv_writeback()

    start_next_fetch()

    # Step 2a: Fetch mask metadata after the fence to overlap with compute and
    # in-flight writeback.
    step_meta = fetch_compute_metadata(
        step,
        schedule_ref,
        q_offsets_ref,
        kv_cache_lens_ref,
        kv_new_lens_ref,
        extra_refs,
        cfgs=cfgs,
        scheduler=scheduler,
    )

    # Step 2b: Fetch inputs.
    q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
    q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.aligned_q_head_dim)
    q_loaded = utils.strided_load(
        q_ref,
        0,
        cfgs.batch_size * cfgs.model.num_kv_heads * cfgs.bq_sz * q_p,
        1,
        dtype=cfgs.serve.dtype_q,
    )
    q = q_loaded.reshape(
        cfgs.batch_size,
        cfgs.model.num_kv_heads,
        cfgs.bq_sz * cfgs.aligned_num_q_heads_per_kv_head,
        cfgs.aligned_q_head_dim,
    )
    q = q[..., : cfgs.compute_kv_head_dim]

    # We want to load k, v from (batch, bkv_sz, bkv_stride, kv_packing, d)
    # where bkv_stride ~= num_kv_heads * 2 // kv_packing
    # to 2x (batch, num_kv_heads, bkv_sz, d)
    # We use strided_load to avoid the expensive transpose.
    k_b = []
    v_b = []

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        for b_idx in range(cfgs.batch_size):
            ks = []
            vs = []
            for kv_head in range(cfgs.model.num_kv_heads):
                k_slice = kv_in_vref.at[b_idx, kv_head * 2].bitcast(jnp.uint32)
                v_slice = kv_in_vref.at[b_idx, kv_head * 2 + 1].bitcast(jnp.uint32)

                target_shape = (-1, cfgs.kv_vmem_lanes)
                k_head_ref = k_slice.reshape(target_shape)
                v_head_ref = v_slice.reshape(target_shape)
                pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv

                # Load as uint32 to avoid dtype conversion during strided load
                k_head_u32 = utils.strided_load(k_head_ref, 0, pack_dim, 1)
                v_head_u32 = utils.strided_load(v_head_ref, 0, pack_dim, 1)

                if cfgs.serve.dtype_kv == jnp.uint8:
                    fp4 = jnp.float4_e2m1fn
                    k_head = pltpu.bitcast(k_head_u32, fp4)[:, : cfgs.bkv_sz]
                    v_head = pltpu.bitcast(v_head_u32, fp4)[:, : cfgs.bkv_sz]
                    ks.append(k_head)
                    vs.append(v_head)
                else:
                    k_head_loaded = pltpu.bitcast(k_head_u32, cfgs.serve.dtype_kv)
                    v_head_loaded = pltpu.bitcast(v_head_u32, cfgs.serve.dtype_kv)
                    k_head = k_head_loaded[:, : cfgs.bkv_sz]
                    v_head = v_head_loaded[:, : cfgs.bkv_sz]
                    ks.append(k_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))
                    vs.append(v_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))

            k_b.append(jnp.stack(ks, axis=0))
            v_b.append(jnp.stack(vs, axis=0))
    else:
        for b_idx in range(cfgs.batch_size):
            heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
            num_kv_heads = cfgs.model.num_kv_heads
            num_kv_chunks = num_kv_heads * cfgs.num_lane_folds_in_head_dim
            ks = []
            vs = []
            for kv_chunk_start in range(0, num_kv_chunks, heads_per_load):
                bkv_lst = strided_load_bkv(
                    kv_in_vref,
                    b_idx,
                    kv_chunk_start * 2,
                    cfgs=cfgs,
                )
                ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
                vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
            k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
            k = k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)
            v = v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)

            k = k[:num_kv_chunks]
            v = v[:num_kv_chunks]
            if cfgs.num_lane_folds_in_head_dim > 1:
                chunk_starts = range(0, num_kv_chunks, num_kv_heads)
                k = jnp.concat([k[i : i + num_kv_heads] for i in chunk_starts], axis=-1)
                v = jnp.concat([v[i : i + num_kv_heads] for i in chunk_starts], axis=-1)
            k_b.append(k)
            v_b.append(v)

    # Stack to (batch, num_heads, bkv_sz, num_lanes)
    k = jnp.stack(k_b, axis=0)
    v = jnp.stack(v_b, axis=0)

    k, v, k_scale, v_scale = get_scale_factors(k, v, cfgs=cfgs)

    # Step 3: Perform compute.
    m_val = m_scratch_ref[...]
    l_val = l_scratch_ref[...]
    acc_val = acc_scratch_ref[...]
    # The sink is folded in by `calculate_and_store_out`, not seeded into the
    # running max / sum as in google3, so the flash attention steps get none.
    sink_val = None
    want_m = cfgs.serve.return_lse or cfgs.model.has_attention_sink
    # Scopes whose rows can attend no key zero p on the masked keys: such a
    # row then keeps l == 0 instead of weighting masked keys by
    # exp(mask_value - mask_value).
    zero_masked = cfgs.serve.attention_scope.walks_cache_stripe

    l_new_list = []
    m_new_list = []
    acc_new_list = []

    prev_p = prev_alpha_list = prev_q_slice = None
    for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
        bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
        q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
        q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
        q_slice = slice(q_start, q_end)

        custom_mask = generate_mask(
            shape=(
                cfgs.batch_size,
                cfgs.model.num_kv_heads,
                q_end - q_start,
                cfgs.bkv_sz,
            ),
            bq_start=bq_start,
            step_meta=step_meta,
            cfgs=cfgs,
        )

        p, alpha_list, m_next, l_next, m_carry, l_carry = (
            flash_attention.flash_attention_qk_softmax(
                step,
                q[:, :, q_slice],
                k,
                m_val[:, q_slice],
                l_val[:, q_slice],
                schedule_ref.is_last_k,
                custom_mask=custom_mask,
                cfgs=cfgs,
                bq_start=bq_start,
                k_scale=k_scale,
                attention_sink=sink_val,
                zero_masked=zero_masked,
            )
        )
        m_scratch_ref[:, q_slice] = m_carry
        l_scratch_ref[:, q_slice] = l_carry
        if want_m:
            m_new_list.append(m_next)
        l_new_list.append(l_next)

        if prev_p is not None:
            o_next, o_carry = flash_attention.flash_attention_pv(
                prev_p,
                v,
                prev_alpha_list,
                acc_val[:, prev_q_slice],
                step,
                schedule_ref.is_last_k,
                cfgs=cfgs,
                v_scale=v_scale,
                attention_sink=sink_val,
            )
            acc_scratch_ref[:, prev_q_slice] = o_carry
            acc_new_list.append(o_next)

        prev_p = p
        prev_alpha_list = alpha_list
        prev_q_slice = q_slice

    assert prev_p is not None
    o_next, o_carry = flash_attention.flash_attention_pv(
        prev_p,
        v,
        prev_alpha_list,
        acc_val[:, prev_q_slice],
        step,
        schedule_ref.is_last_k,
        cfgs=cfgs,
        v_scale=v_scale,
        attention_sink=sink_val,
    )
    acc_scratch_ref[:, prev_q_slice] = o_carry
    acc_new_list.append(o_next)
    m_next = None
    if want_m:
        m_next = jnp.concatenate(m_new_list, axis=2)
    l_next = jnp.concatenate(l_new_list, axis=2)
    acc_next = jnp.concatenate(acc_new_list, axis=2)

    # Step 4: Write back outputs.
    calculate_and_store_out(
        step,
        schedule_ref,
        acc_next,
        l_next,
        m_next,
        o_vref,
        lse_o_vref,
        attention_sink_ref=attention_sink_ref,
        lse_stage_ref=lse_stage_ref,
        cfgs=cfgs,
    )


# Define main kernel.


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    lse_hbm_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
) -> tuple[
    bref_override.BatchingQRef,
    bref_override.KVBufferedRefSeqAlongLane
    | bref_override.KVBufferedRefHeadAlongSublane,
    bref_override.BatchingORef,
    bref_override.BatchingLSERef | None,
]:
    # `use_lookahead` is disabled everywhere: slot rotation and prefetch depth are
    # driven explicitly by `pipeline.run_manual_triple_buffer`.
    in_slots = configs.NUM_INPUT_SLOTS
    out_slots = configs.NUM_OUTPUT_SLOTS
    kv_cache_spec = pl.BlockSpec(
        block_shape=cfgs.kv_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=in_slots, use_lookahead=False),
    )
    q_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=in_slots, use_lookahead=False),
    )
    o_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
    )

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        kv_cache_alloc_cls = bref_override.KVBufferedRefSeqAlongLane
    else:
        kv_cache_alloc_cls = bref_override.KVBufferedRefHeadAlongSublane

    kv_cache_alloc = kv_cache_alloc_cls.input_output(
        spec=kv_cache_spec,
        dtype_or_type=kv_cache_hbm_ref,
        buffer_count=in_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )
    q_alloc = bref_override.BatchingQRef.input(
        spec=q_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=in_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )
    o_alloc = bref_override.BatchingORef.output(
        spec=o_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=out_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )

    lse_alloc = None
    if cfgs.serve.return_lse:
        lse_spec = pl.BlockSpec(
            block_shape=cfgs.lse_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i,),
            pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
        )
        lse_alloc = bref_override.BatchingLSERef.output(
            spec=lse_spec,
            dtype_or_type=lse_hbm_ref,
            buffer_count=out_slots,
            use_lookahead=False,
            cfgs=cfgs,
        )

    return q_alloc, kv_cache_alloc, o_alloc, lse_alloc


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
    serve = cfgs.serve
    name = f"RPA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
    name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
    if cfgs.model.sliding_window:
        name += f"-sw{cfgs.model.sliding_window}"
    if serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
        name += "-new_only"
    elif serve.attention_scope == configs.AttentionScope.CACHE_ONLY:
        name += "-cache_only"
    elif serve.attention_scope == configs.AttentionScope.STRIPED:
        name += "-striped"
    elif serve.attention_scope == configs.AttentionScope.FULL:
        name += "-full"
    if cfgs.model.has_attention_sink:
        name += "-sink"
    if serve.skip_kv_write:
        name += "-nowb"
    if serve.return_lse:
        name += "-lse"
    if serve.cp_group_size is not None:
        name += f"-cp{serve.cp_group_size}"
    return name


def get_kernel_metadata(
    cfgs: configs.RpaConfigs,
) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


def rpa_kernel(
    cu_q_lens: jax.Array,
    q_offsets: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    attention_sink_hbm: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
    scheduler: scheduler_lib.BaseScheduler,
    extra_scalars: Sequence[jax.Array] = (),
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Perform batched ragged paged attention with scheduler data.

    Args:
      cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
        length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
        b=cu_q_lens[i+1] represents q/k/v of sequence i.
      q_offsets: [max_num_seqs]. Token offset for queries in each sequence.
      kv_cache_lens: [max_num_seqs]. Existing kv cache length of each sequence.
      kv_new_lens: [max_num_seqs]. New kv length of each sequence.
      schedule_hbm: Output of scheduler kernel. It informs which: 1. seqs 2. q
        block 3. kv block that should be processed at a given step.
      q_hbm: [max_num_tokens, num_q_heads_per_kv_heads, cdiv(num_kv_heads,
        q_packing), q_packing, head_dim]. Output of q projection that has been
        pre-processed to align with existing kv cache data layout.
      new_kv_hbm: [max_num_tokens, cdiv(num_kv_heads * 2, kv_packing), kv_packing,
        head_dim]. Output of k & v projection that has been pre-processed to align
        with existing kv cache data layout.
      kv_cache_hbm: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
        kv_packing, head_dim]. Stores existing kv cache data where k & vs are
        concatenated along num kv heads dim.
      lse_hbm: [num_kv_heads, max_num_tokens, 1, num_lanes] f32 with return_lse,
        else None. Lane g of a (KV head, token) row holds aligned q head g.
      attention_sink_hbm: [num_kv_heads, period, num_lanes] f32 or None. Per-row
        sink logits from `wrapper.prepare_attention_sink`.
      cfgs: Configuration of the kernel.
      scheduler: Scheduler instance defining schedule shape and KV coordinates.
      extra_scalars: Dynamic scalar arrays prefetched into SMEM for `scheduler`.

    Returns:
      out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
      new_kv_cache: [num_pages, page_size, num_kv_heads // kv_packing, kv_packing,
        head_dim]. Result of new kv cache.
      lse_out: the updated lse_hbm, or None.
    """
    return_lse = cfgs.serve.return_lse

    def ragged_paged_attention_pipeline(
        # Scalar prefetch.
        cu_q_lens_ref: jax.Ref,
        q_offsets_ref: jax.Ref,
        kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        extra_refs: Sequence[jax.Ref],
        # Inputs.
        schedule_hbm_ref: schedule.RpaSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        lse_hbm_ref: jax.Ref | None,
        attention_sink_ref: jax.Ref | None,
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
        o_lse_hbm_ref: jax.Ref | None = None,
    ):

        del o_kv_cache_hbm_ref
        if o_lse_hbm_ref is not None:
            del o_lse_hbm_ref

        q_alloc, kv_cache_alloc, o_alloc, lse_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs
        )

        actual_steps = schedule_hbm_ref.actual_steps[0]
        num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
            schedule_ref=scheduler.get_rpa_schedule(cfgs).scratch_shapes(),
            dma_sem=pltpu.SemaphoreType.DMA((1,)),
            scratches=(
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # m
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # l
                pltpu.VMEM(
                    cfgs.acc_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # acc
            ),
            lse_stage=(
                pltpu.VMEM(cfgs.lse_stage_shape, dtype=jnp.float32)
                if return_lse and cfgs.bq_sz > 1
                else None
            ),
        )
        def _run(final_allocs, schedule_ref, dma_sem, scratches, lse_stage):
            # Initialize Q to zeros to prevent NaN pollution.
            #
            # When a query block is partially filled, tail slots in uninitialized VMEM
            # may contain residual NaNs. In our implementation, invalid Q rows bypass
            # invalid row masking and produce NaNs in p_rowsum and later pollute
            # l_scratch and acc_scratch. Once l_scratch / acc_scratch becomes NaN,
            # subsequent iterations cannot recover (0 * NaN = NaN).

            # Initialize KV cache to zeros.
            # When performing p * v, we perform causal masking on lhs (p) by zeroing
            # out columns that should not be processed for a given row. Even if we
            # don't perform masking on rows of rhs (v), the output is still correct
            # since result of multiplication will be zero thanks zero on lhs. However,
            # this assumption does not hold if a row of rhs has NaNs. To avoid this,
            # we initialize scratch memory with non-zero values. Even if the scratch
            # memory is storing kv cache from previous step, as long as the data is
            # not NaNs, there will be no numeric concerns.

            # The sink does not seed m / l (see `calculate_and_store_out`).
            scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
            scratches[1][...] = jnp.zeros_like(scratches[1])

            q_alloc, kv_cache_alloc, o_alloc, lse_alloc = final_allocs

            num_lanes = pltpu.get_tpu_info().num_lanes
            q_ref_flat = q_alloc.window_ref.bitcast(jnp.uint32).reshape(-1, num_lanes)
            kv_ref_flat = kv_cache_alloc.window_ref.bitcast(jnp.uint32).reshape(
                -1, num_lanes
            )

            # Explicitly zero out scratches and Q/KV buffers.
            for ref in (scratches[2], q_ref_flat, kv_ref_flat):
                ref[...] = jnp.zeros_like(ref)

            def execute_schedule_chunk(start_step, num_steps):
                # All reads are aligned to 128 and some extra steps are copied in the
                # process.
                aligned_start_step = (start_step // 128) * 128
                prefix_steps = start_step - aligned_start_step

                flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
                flat_smem = jax.tree_util.tree_leaves(schedule_ref)
                dma_list = []
                for h, s in zip(flat_hbm, flat_smem):
                    if jax.typeof(h).memory_space == pltpu.HBM:
                        element_size = s.shape[0] // cfgs.max_steps_ub
                        read_size = element_size * (num_steps + prefix_steps)
                        read_size = utils.align_to(read_size, 1024)
                        read_size = jnp.minimum(read_size, s.shape[0])

                        src_off = element_size * aligned_start_step
                        src_off = pl.multiple_of(src_off, 128)

                        copy = pltpu.make_async_copy(
                            h.at[pl.ds(src_off, read_size)],
                            s.at[pl.ds(0, read_size)],
                            dma_sem.at[0],
                        )
                        copy.start()
                        dma_list.append(copy)
                jax.tree.map(lambda x: x.wait(), dma_list)

                total_steps = num_steps + prefix_steps
                kv_bound = kv_cache_alloc.bind(
                    src_ref=(
                        kv_cache_hbm_ref,
                        new_kv_hbm_ref,
                        schedule_ref,
                    ),
                    dst_ref=(
                        kv_cache_hbm_ref,
                        new_kv_hbm_ref,
                        schedule_ref,
                    ),
                )
                q_bound = q_alloc.bind(src_ref=(q_hbm_ref, schedule_ref))
                o_bound = o_alloc.bind(dst_ref=(o_hbm_ref, schedule_ref))
                lse_bound = (
                    lse_alloc.bind(dst_ref=(lse_hbm_ref, schedule_ref))
                    if return_lse
                    else None
                )

                def compute_fn(
                    step,
                    slot,
                    total_steps,
                    out_slot,
                    start_next_fetch,
                    start_kv_writeback,
                ):
                    del total_steps
                    rpa_body(
                        step,
                        slot,
                        q_alloc.window_ref.at[slot],
                        kv_cache_alloc.window_ref.at[slot],
                        o_alloc.window_ref.at[out_slot],
                        lse_alloc.window_ref.at[out_slot] if return_lse else None,
                        schedule_ref,
                        scratches[0],
                        scratches[1],
                        scratches[2],
                        cu_q_lens_ref=cu_q_lens_ref,
                        q_offsets_ref=q_offsets_ref,
                        kv_cache_lens_ref=kv_cache_lens_ref,
                        kv_new_lens_ref=kv_new_lens_ref,
                        extra_refs=extra_refs,
                        attention_sink_ref=attention_sink_ref,
                        lse_stage_ref=lse_stage,
                        cfgs=cfgs,
                        scheduler=scheduler,
                        start_next_fetch=start_next_fetch,
                        start_kv_writeback=start_kv_writeback,
                    )

                pipeline.run_manual_triple_buffer(
                    total_steps,
                    compute_fn,
                    q_bound,
                    o_bound,
                    kv_bound,
                    lse_bref=lse_bound,
                )

            @pl.loop(0, num_safe_step_iterations)
            def loop_body(step_idx):
                start = step_idx * cfgs.max_steps_ub
                rem = actual_steps % cfgs.max_steps_ub
                last_step_size = jnp.where(rem == 0, cfgs.max_steps_ub, rem)
                is_last_step = step_idx == num_safe_step_iterations - 1
                size = jnp.where(is_last_step, last_step_size, cfgs.max_steps_ub)

                execute_schedule_chunk(start, size)

        _run()

    scalar_prefetches = (
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        extra_scalars,
    )
    num_scalar_prefetch = len(scalar_prefetches)
    num_active_scalers = len(jax.tree_util.tree_leaves(scalar_prefetches))

    out_shape = [q_hbm, kv_cache_hbm, lse_hbm if return_lse else None]

    schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    input_output_aliases = {
        num_active_scalers + schedule_leaves: 0,
        num_active_scalers + schedule_leaves + 2: 1,
    }
    if return_lse:
        input_output_aliases[num_active_scalers + schedule_leaves + 3] = (
            2  # lse_hbm -> out[2]
        )

    return pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
                pl.BlockSpec(memory_space=pltpu.VMEM)
                if attention_sink_hbm is not None
                else None,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            flags={
                "XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True,
            },
        ),
        input_output_aliases=input_output_aliases,
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )(
        *scalar_prefetches,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        lse_hbm if return_lse else None,
        attention_sink_hbm,
    )
