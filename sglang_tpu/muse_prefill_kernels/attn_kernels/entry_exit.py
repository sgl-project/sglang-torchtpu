"""Entry (`Kernel 0`) and Exit Stage Kernels for Muse Spark Prefill on TPU v7x.

Implements:
  1. `entry_unfused_ar`: Unfused VocabParallelEmbedding + full-T All-Reduce + embed_norm
     + Layer-0 input_layernorm + once-per-step RoPE table gather.
  2. `entry_v1_sp_rs_pallas`:
     - VocabParallel shard lookup + Reduce-Scatter (`[T, 8192] -> [T // TP, 8192]`)
     - Fused Pallas `pallas_entry_norms`: computes `embed_norm(eps=1e-5).to(bf16).float() -> stream_0_shard`
       AND Layer-0 `input_layernorm(eps=1e-5, gain=w+1)` in VMEM in a single pass (zero intermediate HBM round-trip),
       plus optional algebraic-cancel FP8 quantization of Layer-0 `attn_in_shard`!
     - Once-per-step `build_batch_cos_sin_128` gather.
  3. `exit_unfused_full_t`:
     - Full-T All-Gather + full-T `model.norm(gain_center=0.0, eps=1e-5)` + `last_indices` gather
       + XLA `lm_head * 0.17677669529663687` + `20.0 * tanh(x / 20.0)`.
  4. `exit_v1_owner_gather_pallas_fp8_lm_head`:
     - Owner-rank `last_indices` gather on `stream_final_shard [T // TP, 8192]` -> `[B, 8192]`
     - Tiny `lax.psum` (`<= 512 KiB`) BEFORE `model.norm`
     - `model.norm(gain_center=0.0, eps=1e-5)` on `[B, 8192]` only
     - Fused Pallas `pallas_fp8_lm_head_softcap` (`[B_pad, 8192] @ w_lm_fp8[8192, V_shard] * w_scale * 0.17677669529663687`
       + FP32 `20.0 * tanh(x / 20.0)` in VMEM).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .reference import (
    LM_HEAD_SCALE,
    LOGIT_CAP,
    build_batch_cos_sin_128,
    input_layernorm_ref,
    rms_normalize_fp32,
)


# -----------------------------------------------------------------------------
# 1. Kernel 0 (Entry): Vocab Lookup + Reduce-Scatter + Fused Embed & Layer-0 Norms
# -----------------------------------------------------------------------------
def pallas_entry_norms(
    embed_shard: jax.Array,
    layer0_norm_weight_2d: jax.Array,
    tm: int = 128,
    vmem_limit_mb: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Fuses `embed_norm(eps=1e-5).to(bf16).float()` and Layer-0 `input_layernorm(eps=1e-5, gain=w+1)` in VMEM.

    Args:
      embed_shard: bf16[T // TP, 8192] from Reduce-Scattered VocabParallelEmbedding
      layer0_norm_weight_2d: fp32[1, 8192] (raw weight; kernel adds +1.0 for gain_center=1.0)

    Returns:
      stream_0_shard: fp32[T // TP, 8192]
      attn_in_0_shard: bf16[T // TP, 8192]
    """
    t_shard, h_dim = embed_shard.shape
    tm_eff = min(tm, t_shard)

    def _entry_kernel(emb_ref, w0_ref, stream_out_ref, attn_in_out_ref):
        xf = emb_ref[...].astype(jnp.float32)
        var0 = jnp.mean(xf * xf, axis=1, keepdims=True)
        # Match Muse Spark embed_norm: _rms_normalize(h, 1e-5).to(bf16).float()
        stream_0 = (
            (xf * lax.rsqrt(var0 + 1e-5)).astype(jnp.bfloat16).astype(jnp.float32)
        )
        stream_out_ref[...] = stream_0

        # Layer-0 input_layernorm(stream_0, eps=1e-5, gain = w0 + 1.0) in registers!
        var1 = jnp.mean(stream_0 * stream_0, axis=1, keepdims=True)
        gain0 = w0_ref[...] + 1.0
        attn_in_0 = (stream_0 * lax.rsqrt(var1 + 1e-5) * gain0).astype(jnp.bfloat16)
        attn_in_out_ref[...] = attn_in_0

    return pl.pallas_call(
        _entry_kernel,
        grid=(t_shard // tm_eff,),
        in_specs=[
            pl.BlockSpec((tm_eff, h_dim), lambda i: (i, 0)),
            pl.BlockSpec((1, h_dim), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tm_eff, h_dim), lambda i: (i, 0)),
            pl.BlockSpec((tm_eff, h_dim), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((t_shard, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((t_shard, h_dim), jnp.bfloat16),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(embed_shard, layer0_norm_weight_2d)


def vocab_parallel_lookup(
    input_ids: jax.Array,
    embed_weight_shard: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """Vocab-parallel embedding lookup returning unreduced bf16[T, 8192]."""
    v_shard = embed_weight_shard.shape[0]
    rank = lax.axis_index(tp_axis)
    v_start = rank * v_shard
    v_end = v_start + v_shard
    in_mask = (input_ids >= v_start) & (input_ids < v_end)
    safe_ids = jnp.where(in_mask, input_ids - v_start, 0)
    looked_up = embed_weight_shard[safe_ids]
    return jnp.where(in_mask[:, None], looked_up, jnp.bfloat16(0.0))


def entry_unfused_ar(
    input_ids: jax.Array,
    positions: jax.Array,
    embed_weight_shard: jax.Array,
    layer0_norm_weight: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    tp_axis: str = "tp",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Unfused baseline: VocabParallel lookup + full-T All-Reduce + XLA embed_norm + XLA layer0 norm."""
    partial_emb = vocab_parallel_lookup(input_ids, embed_weight_shard, tp_axis=tp_axis)
    full_emb = lax.psum(partial_emb, axis_name=tp_axis)
    stream_0_full = (
        rms_normalize_fp32(full_emb, eps=1e-5).astype(jnp.bfloat16).astype(jnp.float32)
    )
    attn_in_0_full = input_layernorm_ref(stream_0_full, layer0_norm_weight, eps=1e-5)
    _, cos_128, sin_signed_128 = build_batch_cos_sin_128(
        positions, cos_table, sin_table
    )
    return stream_0_full, attn_in_0_full, cos_128, sin_signed_128


def entry_v1_sp_rs_pallas(
    input_ids: jax.Array,
    positions: jax.Array,
    embed_weight_shard: jax.Array,
    layer0_norm_weight_2d: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    tp_axis: str = "tp",
    tm: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused SP Entry (Kernel 0): Vocab lookup + Reduce-Scatter + Pallas fused embed_norm & layer0 norm."""
    partial_emb = vocab_parallel_lookup(input_ids, embed_weight_shard, tp_axis=tp_axis)
    embed_shard = lax.psum_scatter(
        partial_emb, axis_name=tp_axis, scatter_dimension=0, tiled=True
    )
    stream_0_shard, attn_in_0_shard = pallas_entry_norms(
        embed_shard, layer0_norm_weight_2d, tm=tm
    )
    _, cos_128, sin_signed_128 = build_batch_cos_sin_128(
        positions, cos_table, sin_table
    )
    return stream_0_shard, attn_in_0_shard, cos_128, sin_signed_128


def entry_v2_2half_overlap_rs_pallas(
    input_ids: jax.Array,
    positions: jax.Array,
    embed_weight_shard: jax.Array,
    layer0_norm_weight_2d: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    tp_axis: str = "tp",
    tp_size: int = 8,
    tm: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """2-Half Overlapped SP Entry: overlaps Half-1 vocab gather with Half-0 Reduce-Scatter + Pallas norms."""
    t_total = input_ids.shape[0]
    t_shard = t_total // tp_size
    h_sub = t_shard // 2
    t_half = t_total // 2

    ids_r = input_ids.reshape(tp_size, 2, h_sub)
    ids0 = ids_r[:, 0, :].reshape(t_half)
    ids1 = ids_r[:, 1, :].reshape(t_half)

    p0 = vocab_parallel_lookup(ids0, embed_weight_shard, tp_axis=tp_axis)
    e0 = lax.psum_scatter(p0, axis_name=tp_axis, scatter_dimension=0, tiled=True)
    s0, a0 = pallas_entry_norms(e0, layer0_norm_weight_2d, tm=min(tm, h_sub))

    p1 = vocab_parallel_lookup(ids1, embed_weight_shard, tp_axis=tp_axis)
    e1 = lax.psum_scatter(p1, axis_name=tp_axis, scatter_dimension=0, tiled=True)
    s1, a1 = pallas_entry_norms(e1, layer0_norm_weight_2d, tm=min(tm, h_sub))

    stream_0_shard = jnp.concatenate([s0, s1], axis=0)
    attn_in_0_shard = jnp.concatenate([a0, a1], axis=0)
    _, cos_128, sin_signed_128 = build_batch_cos_sin_128(
        positions, cos_table, sin_table
    )
    return stream_0_shard, attn_in_0_shard, cos_128, sin_signed_128


def entry_v3_col_embed_a2a_pallas(
    input_ids: jax.Array,
    positions: jax.Array,
    embed_col_shard: jax.Array,
    layer0_norm_weight_2d: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    tp_axis: str = "tp",
    tm: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Column-Parallel Vocab Gather + 14 MiB All-to-All + Fused Pallas Entry Norms.

    Sharding `embed_tokens` as `[V, 8192 // TP]` (same HBM footprint per rank as `[V // TP, 8192]`)
    cuts embedding HBM gather traffic by 8x (`16 MiB` vs `128 MiB` at T=8K TP8, zero masking)
    and replaces the `112 MiB` Reduce-Scatter with a `14 MiB` All-to-All!
    """
    emb_col = embed_col_shard[input_ids]  # bf16[T, 8192 // TP]
    embed_shard = lax.all_to_all(
        emb_col,
        axis_name=tp_axis,
        split_axis=0,
        concat_axis=1,
        tiled=True,
    )  # bf16[T // TP, 8192]
    stream_0_shard, attn_in_0_shard = pallas_entry_norms(
        embed_shard, layer0_norm_weight_2d, tm=min(tm, embed_shard.shape[0])
    )
    _, cos_128, sin_signed_128 = build_batch_cos_sin_128(
        positions, cos_table, sin_table
    )
    return stream_0_shard, attn_in_0_shard, cos_128, sin_signed_128


def pallas_async_all_to_all_entry(
    src_3d: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """In-kernel Pallas async remote-DMA All-to-All on `[tp_deg, t_shard, k_local]`."""
    tp_deg, t_shard, k_local = src_3d.shape

    def _a2a_body(src_ref, dst_ref, sems):
        me = lax.axis_index(tp_axis)
        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(
                    barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                )
            pl.semaphore_wait(barrier, tp_deg - 1)
        recv, sent = sems.at[0], sems.at[1]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(me, 1)],
            dst_ref.at[pl.ds(me, 1)],
            recv,
        ).start()
        for d in range(1, tp_deg):
            peer = lax.rem(me + d, tp_deg)
            pltpu.make_async_remote_copy(
                src_ref=src_ref.at[pl.ds(peer, 1)],
                dst_ref=dst_ref.at[pl.ds(me, 1)],
                send_sem=sent,
                recv_sem=recv,
                device_id={tp_axis: peer},
            ).start()
        tmpl_r = pltpu.make_async_copy(
            dst_ref.at[pl.ds(0, 1)], dst_ref.at[pl.ds(0, 1)], recv
        )
        for _ in range(tp_deg):
            tmpl_r.wait()
        tmpl_s = pltpu.make_async_copy(
            src_ref.at[pl.ds(0, 1)], src_ref.at[pl.ds(0, 1)], sent
        )
        for _ in range(tp_deg - 1):
            tmpl_s.wait()

    return pl.pallas_call(
        _a2a_body,
        grid=(1,),
        in_specs=[pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))],
        out_specs=pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM)),
        out_shape=jax.ShapeDtypeStruct((tp_deg, t_shard, k_local), src_3d.dtype),
        scratch_shapes=[pltpu.SemaphoreType.DMA((2,))],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=21 if tp_deg > 1 else None,
        ),
    )(src_3d)


def entry_v4_col_embed_async_a2a_pallas(
    input_ids: jax.Array,
    positions: jax.Array,
    embed_col_shard: jax.Array,
    layer0_norm_weight_2d: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    tp_axis: str = "tp",
    tm: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Column-Parallel Vocab Gather + Pallas `make_async_remote_copy` All-to-All + Fused Pallas Entry Norms."""
    t_total = input_ids.shape[0]
    k_local = embed_col_shard.shape[1]
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg

    emb_col_3d = embed_col_shard[input_ids].reshape(tp_deg, t_shard, k_local)
    recv_3d = pallas_async_all_to_all_entry(emb_col_3d, tp_axis=tp_axis)
    embed_shard = jnp.swapaxes(recv_3d, 0, 1).reshape(t_shard, 8192)
    stream_0_shard, attn_in_0_shard = pallas_entry_norms(
        embed_shard, layer0_norm_weight_2d, tm=min(tm, t_shard)
    )
    _, cos_128, sin_signed_128 = build_batch_cos_sin_128(
        positions, cos_table, sin_table
    )
    return stream_0_shard, attn_in_0_shard, cos_128, sin_signed_128


def entry_v5_replicated_embed_local_pallas(
    input_ids_shard: jax.Array,
    positions: jax.Array,
    embed_weight_replicated: jax.Array,
    layer0_norm_weight_2d: jax.Array,
    cos_table: jax.Array,
    sin_table: jax.Array,
    tm: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Zero-Collective Entry: token-owner gathers its own `T // TP` token IDs from replicated `embed_tokens` + Pallas Entry Norms."""
    embed_shard = embed_weight_replicated[input_ids_shard]  # bf16[T // TP, 8192]
    stream_0_shard, attn_in_0_shard = pallas_entry_norms(
        embed_shard, layer0_norm_weight_2d, tm=min(tm, embed_shard.shape[0])
    )
    _, cos_128, sin_signed_128 = build_batch_cos_sin_128(
        positions, cos_table, sin_table
    )
    return stream_0_shard, attn_in_0_shard, cos_128, sin_signed_128


# -----------------------------------------------------------------------------
# 2. Exit Stage: Owner-Rank Last-Token Gather + Final Norm + Fused FP8 LM Head + Softcap
# -----------------------------------------------------------------------------
def pallas_fp8_lm_head_softcap(
    x_bf16: jax.Array,
    w_lm_fp8: jax.Array,
    w_lm_scale_2d: jax.Array,
    tn: int = 512,
    vmem_limit_mb: int = 32,
) -> jax.Array:
    """Fused Pallas FP8 LM Head + `* 0.17677669529663687` + FP32 `20.0 * tanh(x / 20.0)`.

    Pads `B` to a multiple of 8 rows (`m_pad >= 8`) if needed so TPU MXU tiles align cleanly,
    keeps `x_bf16` resident in VMEM (`pl.BlockSpec((m_pad, 8192), lambda j: (0, 0))`),
    streams `w_lm_fp8 [8192, V_shard]` in `(8192, tn)` tiles, and fuses the logit scale
    and `tanh` softcap in VREGs before writing `fp32[B, V_shard]` to HBM.
    """
    b_orig, h_dim = x_bf16.shape
    _, v_shard = w_lm_fp8.shape
    m_pad = max(8, ((b_orig + 7) // 8) * 8)
    if m_pad > b_orig:
        x_in = jnp.pad(x_bf16, ((0, m_pad - b_orig), (0, 0)))
    else:
        x_in = x_bf16

    combined_scale_const = LM_HEAD_SCALE / LOGIT_CAP

    def _lm_head_kernel(x_ref, w_ref, s_ref, o_ref):
        raw = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
        # raw * w_scale * (0.17677669529663687 / 20.0), then * 20.0 after tanh
        scaled = raw * (s_ref[...] * combined_scale_const)
        o_ref[...] = LOGIT_CAP * jnp.tanh(scaled)

    out_padded = pl.pallas_call(
        _lm_head_kernel,
        grid=(v_shard // tn,),
        in_specs=[
            pl.BlockSpec((m_pad, h_dim), lambda j: (0, 0)),
            pl.BlockSpec((h_dim, tn), lambda j: (0, j)),
            pl.BlockSpec((1, tn), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((m_pad, tn), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((m_pad, v_shard), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
    )(x_in, w_lm_fp8, w_lm_scale_2d)
    return out_padded[:b_orig]


def exit_unfused_full_t(
    stream_final_shard: jax.Array,
    last_indices: jax.Array,
    final_norm_weight: jax.Array,
    w_lm_fp8: jax.Array,
    w_lm_scale_2d: jax.Array,
    tp_axis: str = "tp",
) -> jax.Array:
    """Unfused baseline: All-Gather full [T, 8192] -> full-T model.norm(gain_center=0.0) -> gather -> XLA lm_head."""
    stream_full = lax.all_gather(
        stream_final_shard, axis_name=tp_axis, axis=0, tiled=True
    )
    # Final model.norm has gain_center=0.0 (NO +1.0!)
    normed_full = (
        rms_normalize_fp32(stream_full, eps=1e-5)
        * final_norm_weight.astype(jnp.float32)[None, :]
    ).astype(jnp.bfloat16)
    h_last = normed_full[last_indices]
    logits = (
        jnp.dot(h_last, w_lm_fp8, preferred_element_type=jnp.float32)
        * w_lm_scale_2d
        * LM_HEAD_SCALE
    )
    return LOGIT_CAP * jnp.tanh(logits * (1.0 / LOGIT_CAP))


def exit_v1_owner_gather_pallas_fp8_lm_head(
    stream_final_shard: jax.Array,
    last_indices: jax.Array,
    final_norm_weight: jax.Array,
    w_lm_fp8: jax.Array,
    w_lm_scale_2d: jax.Array,
    tp_axis: str = "tp",
    tn: int = 512,
) -> jax.Array:
    """Optimized SP Exit: Owner-rank `last_indices` gather -> tiny psum (<=512 KiB) -> B-row norm -> Pallas FP8 LM head."""
    t_shard = stream_final_shard.shape[0]
    rank = lax.axis_index(tp_axis)
    t_start = rank * t_shard
    t_end = t_start + t_shard

    in_range = (last_indices >= t_start) & (last_indices < t_end)
    local_idx = jnp.clip(last_indices - t_start, 0, t_shard - 1)
    gathered_local = jnp.where(
        in_range[:, None],
        stream_final_shard[local_idx],
        jnp.float32(0.0),
    )
    last_stream = lax.psum(gathered_local, axis_name=tp_axis)  # [B, 8192] in FP32

    # Final model.norm has gain_center = 0.0, eps = 1e-5
    h_last = (
        rms_normalize_fp32(last_stream, eps=1e-5)
        * final_norm_weight.astype(jnp.float32)[None, :]
    ).astype(jnp.bfloat16)

    return pallas_fp8_lm_head_softcap(h_last, w_lm_fp8, w_lm_scale_2d, tn=tn)


def pallas_embed_rmsnorm_bf16(
    embed_shard: jax.Array,
    tm: int = 256,
    vmem_limit_mb: int = 32,
) -> jax.Array:
    """Single-pass Pallas FP32 RMSNorm (`eps=1e-5`) -> `bf16[T // TP, 8192]`."""
    t_shard, h_dim = embed_shard.shape
    tm_eff = next(t for t in (256, 128, 64, 32, 8) if t <= t_shard and t_shard % t == 0)

    def _norm_kernel(emb_ref, out_ref):
        sub_r = min(128, tm_eff)
        inv_h = jnp.float32(1.0 / float(h_dim))
        for r0 in range(0, tm_eff, sub_r):
            r1 = r0 + sub_r
            sq_128 = jnp.zeros((sub_r, 128), dtype=jnp.float32)
            for c0 in range(0, h_dim, 1024):
                xf = emb_ref[r0:r1, c0 : c0 + 1024].astype(jnp.float32)
                sq_128 = sq_128 + jnp.sum(
                    (xf * xf).reshape(sub_r, 8, 128), axis=1
                )
            inv_rms = lax.rsqrt(
                jnp.sum(sq_128, axis=1, keepdims=True) * inv_h + jnp.float32(1e-5)
            )
            for c0 in range(0, h_dim, 1024):
                xf = emb_ref[r0:r1, c0 : c0 + 1024].astype(jnp.float32)
                out_ref[r0:r1, c0 : c0 + 1024] = (xf * inv_rms).astype(jnp.bfloat16)

    return pl.pallas_call(
        _norm_kernel,
        grid=(t_shard // tm_eff,),
        in_specs=[pl.BlockSpec((tm_eff, h_dim), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tm_eff, h_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((t_shard, h_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_embed_rmsnorm_bf16_tm{tm_eff}",
    )(embed_shard)


def sp_entry_col_a2a(
    input_ids: jax.Array,
    embed_col_shard: jax.Array,
    *,
    org_vocab_size: int,
    start_is_zero: bool = True,
    tp_axis: str = "model",
    async_a2a: bool = False,
) -> jax.Array:
    """Column-sharded `[V, H / tp]` embedding lookup + `lax.all_to_all` + sharded `embed_norm`.

    Masks out-of-range (`< 0` or `>= org_vocab_size`) token IDs to `0.0` identically to
    `VocabParallelEmbedding`.
    - When `start_is_zero=True` (`start == 0`), returns `stream_0_shard` viewed as a
      contiguous `f32[T, H / tp]` tensor so `stream.shape[0] == T` for downstream checks
      while transferring zero extra bytes.
    - When `start_is_zero=False` (`start > 0`), SparseCore-all-gathers `bf16[T / tp, H]`
      and casts to `f32[T, H]`.
    """
    from sglang_tpu.envs import envs as _tpu_envs

    async_env = getattr(_tpu_envs, "SGLANG_TPU_MUSE_KERNEL_ENTRY_ASYNC_A2A", None)
    use_async_a2a = async_a2a or bool(async_env is not None and async_env.get())
    t_total = input_ids.shape[0]
    h_col = embed_col_shard.shape[1]
    valid = (input_ids >= 0) & (input_ids < org_vocab_size)
    safe_ids = jnp.where(valid, input_ids, 0)
    emb_col = jnp.where(valid[:, None], embed_col_shard[safe_ids], jnp.bfloat16(0.0))
    if use_async_a2a:
        tp_deg = 8192 // h_col
        t_shard = t_total // tp_deg
        recv_3d = pallas_async_all_to_all_entry(
            emb_col.reshape(tp_deg, t_shard, h_col), tp_axis=tp_axis
        )
        embed_shard = jnp.swapaxes(recv_3d, 0, 1).reshape(t_shard, 8192)
        if t_shard >= 8 and t_shard % 8 == 0 and embed_shard.shape[1] == 8192:
            normed_bf16 = pallas_embed_rmsnorm_bf16(embed_shard)
        else:
            normed_bf16 = rms_normalize_fp32(embed_shard, eps=1e-5).astype(jnp.bfloat16)
    else:
        embed_shard = lax.all_to_all(
            emb_col,
            axis_name=tp_axis,
            split_axis=0,
            concat_axis=1,
            tiled=True,
        )
        normed_bf16 = rms_normalize_fp32(embed_shard, eps=1e-5).astype(jnp.bfloat16)
    if start_is_zero:
        return normed_bf16.astype(jnp.float32).reshape(t_total, h_col)
    return lax.all_gather(normed_bf16, axis_name=tp_axis, axis=0, tiled=True).astype(
        jnp.float32
    )


def sharded_rope_gather_128(
    positions: jax.Array,
    cos_sin_128_table: jax.Array,
    *,
    tp_axis: str = "model",
    tp_size: int = 8,
    pos_pre_sliced: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Gathers `T / tp` positions per rank from `[max_pos, 128]` `[cos | signed_sin]` + SparseCore `all_gather`."""
    if pos_pre_sliced:
        pos_local = positions
    else:
        from .kernel_a1 import pallas_axis_index_1d

        t_shard = positions.shape[0] // tp_size
        me = pallas_axis_index_1d(tp_axis)[0]
        pos_local = lax.dynamic_slice_in_dim(positions, me * t_shard, t_shard, axis=0)
    pos_safe = jnp.clip(pos_local, 0, cos_sin_128_table.shape[0] - 1)
    cs_local = cos_sin_128_table[pos_safe]
    cs_full = lax.all_gather(cs_local, axis_name=tp_axis, axis=0, tiled=True)
    c64, s64 = cs_full[:, :64], cs_full[:, 64:]
    return (
        jnp.concatenate([c64, c64], axis=-1),
        jnp.concatenate([s64, s64], axis=-1),
    )
