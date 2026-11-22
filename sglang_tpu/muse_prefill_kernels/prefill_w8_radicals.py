# SPDX-License-Identifier: Apache-2.0
"""Wave-8 Radical Prefill Kernel Prototypes for Muse Spark on TPU v7x.

Implements strictly opt-in prefill experiments behind `SGLANG_TPU_MUSE_KERNEL_*` env flags:
  1. `A3` Local K-Deep `Wo` (`SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO = 1 / 2`):
     - Mode 1: In-kernel XOR push-A2A into VMEM-stationary `recv_vmem [tp_deg, t_shard, 1024]`
       (`16 MiB` at `T=16K`) + single-pass `Wo` (`64 MiB` read 1x instead of `128 MiB` 2x,
       0 `acc_vmem` scratch, eliminating `896 MiB/layer` VMEM RMW traffic).
     - Mode 2 (`CCS=1`): Overlaps per-chunk prologue + XOR push-A2A (`pallas_async_a2a_xor_3d`)
       for causal chunks `0..K-2` underneath `A1(k+1) + RPAm(k+1)`, and fuses the final chunk's
       remote push-A2A directly inside `pallas_a3_wo_kdeep_ccs_last_a2a` while `j=0` computes
       `rows 0 : t_shard - c_last`.
  2. `D` Run-Coalesced Gather + Permutation-Absorbing `Wsel` MXU Combine
     (`SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER = 1`):
     - Coalesces `inv_perm_c` per 64-token tile into `<= 257` contiguous HBM runs (`~160` 3D
       DMAs vs `512` 1-row DMAs) with capacity `512` (zero overflow risk even on padded tiles)
       and absorbs the run-sort permutation into `Wsel [64, 512] bf16` on MXU.
  3. `D` Single-Call In-Kernel Push All-Gather (`SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG = 1`):
     - Replaces the 2-call `_d_with_next_in_gather` split with 1 single `4slab` Kernel D launch
       (`wunpool` loaded 1x instead of 2x, `-1.95 GB/step` HBM) that pushes `next_attn_in[:rows0]`
       directly to all peers' `gathered_hbm[me]` via `make_async_remote_copy` at `step == steps_ag`
       while steps `steps_ag .. num_steps - 1` run on TensorCore (`-61` XLA `all-gather` ops).
  4. Native TPU7x `FP8 x NVFP4` (`float4_e2m1fn`, `qbs=256`) Dense `Wo` & `wunpool`
     (`SGLANG_TPU_MUSE_KERNEL_W4_WO = 256`, `SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL = 256`):
     - Halves `Wo` (`64 -> 32 MiB/L`, `-1.91 GiB/TC`) and `wunpool` (`32 -> 16 MiB/L` HBM & VMEM,
       `-0.95 GiB/TC`), feeding `jnp.float4_e2m1fn` directly into TPU7x MXU `fp8 x fp4 -> f32`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs
from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a3
from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a1 import pallas_axis_index_1d
from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a3 import (
    STATIC_A3_ACT_MULT,
    STATIC_A3_ACT_SCALE,
    _dual_head_rmsnorm_128,
    pallas_gate_static_fp8_quant,
)
from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_b1 import (
    LANE,
    vpu_tree_max_128,
    vpu_tree_sum_128,
)
from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_d import (
    prepare_d_static_inputs,
)

_FP8_MAX = 448.0


# -----------------------------------------------------------------------------
# 0. Load-Time PyTorch FP8 -> NVFP4 (qbs=256) Dense Weight Quantizer
# -----------------------------------------------------------------------------
def quantize_dense_fp8_to_nvfp4_torch(
    w_fp8,
    w_scale_1d,
    *,
    w_shape: tuple[int, ...],
    block: int = 256,
):
    """Convert K-major FP8 `[K, N]` (or `[tp, K_local, N]`) + `f32[N]` scale to NVFP4.

    Returns `(w_fp4_x2, w_scale_bf16, fp4_logical_shape)` where:
      - `w_fp4_x2` has dtype `torch.float4_e2m1fn_x2` and shape `(*w_shape[:-1], N // 2)`,
        bridging through TorchTPU/DLPack to JAX `jnp.float4_e2m1fn` of shape `w_shape`.
      - `w_scale_bf16` has dtype `torch.bfloat16` and shape
        `(tp, K_local // block, N)` when `len(w_shape) == 3` else `(K // block, N)`.
    """
    import torch

    from sglang_tpu.fp4_moe import _materialize_packed_fp4
    from sglang_tpu.moe.nvfp4_requant import quantize_fp8_to_nvfp4_blocks

    n_dim = int(w_shape[-1])
    k_full = int(w_fp8.numel() // n_dim)
    w_2d = w_fp8.reshape(1, k_full, n_dim)
    s_1d = w_scale_1d.reshape(1, n_dim)
    codes, sb, _, _ = quantize_fp8_to_nvfp4_blocks(w_2d, s_1d, block)
    packed_2d = _materialize_packed_fp4(codes[0].contiguous())
    if len(w_shape) == 3:
        tp_d, k_loc, _ = w_shape
        assert k_loc % block == 0
        packed_out = packed_2d.view(tp_d, k_loc, n_dim // 2)
        sb_out = sb[0].reshape(tp_d, k_loc // block, n_dim).to(torch.bfloat16).contiguous()
    else:
        packed_out = packed_2d.view(k_full, n_dim // 2)
        sb_out = sb[0].reshape(k_full // block, n_dim).to(torch.bfloat16).contiguous()
    return packed_out, sb_out, tuple(w_shape)


# -----------------------------------------------------------------------------
# 1. A3 Local K-Deep Wo + Per-Chunk XOR Async All-to-All (+ Optional W4_WO)
# -----------------------------------------------------------------------------
def _resolve_parity_stagger(parity_stagger: bool | None, tp_deg: int) -> bool:
    """Match `_prepare_wo`'s K_8 parity-staggered peer permutation when `A3_STAGE_SENDS in (1, 2)`."""
    if parity_stagger is not None:
        return bool(parity_stagger and tp_deg == 8)
    if tp_deg != 8:
        return False
    ss_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS", None)
    return bool(ss_env is not None and int(ss_env.get() or 0) in (1, 2))


def _kdeep_step_peer_jax(
    me: jax.Array, d: int, tp_deg: int, *, parity_stagger: bool
) -> jax.Array:
    """Return symmetric involution peer for step `d` matching `kernel_a3._a3_v16_step_peer_jax`."""
    ka3_fn = getattr(kernel_a3, "_a3_v16_step_peer_jax", None)
    if ka3_fn is not None:
        return ka3_fn(
            me,
            d,
            tp_deg,
            step_xors=tuple(range(tp_deg)),
            parity_stagger=parity_stagger,
        )
    if not parity_stagger or tp_deg != 8:
        return lax.bitwise_xor(me, jnp.int32(d))
    if d == 0:
        return me
    if d == 1:
        return lax.bitwise_xor(me, jnp.int32(1))
    if d in (2, 3):
        is_even = lax.bitwise_and(me, jnp.int32(1)) == jnp.int32(0)
        xor = jnp.where(
            is_even,
            jnp.int32(2 if d == 2 else 4),
            jnp.int32(4 if d == 2 else 2),
        )
        return lax.bitwise_xor(me, xor)
    if d in (4, 5):
        pop1 = lax.bitwise_xor(me, lax.shift_right_logical(me, jnp.int32(1)))
        pop2 = lax.bitwise_xor(pop1, lax.shift_right_logical(me, jnp.int32(2)))
        pop_odd = lax.bitwise_and(pop2, jnp.int32(1)) != jnp.int32(0)
        xor = jnp.where(
            pop_odd,
            jnp.int32(5 if d == 4 else 3),
            jnp.int32(3 if d == 4 else 5),
        )
        return lax.bitwise_xor(me, xor)
    return lax.bitwise_xor(me, jnp.int32(d))


def pallas_async_a2a_xor_3d(
    src_3d: jax.Array,
    *,
    tp_axis: str = "model",
    collective_id: int = 31,
    parity_stagger: bool | None = None,
) -> jax.Array:
    """In-kernel async remote-copy XOR All-to-All on `[tp_deg, c_rows, k_local]`.

    Delivers `peer`'s slice `me` directly into `dst_3d[d, :, :]` so `dst_3d`
    matches `wo_xor_3d` (respecting `parity_stagger` when `A3_STAGE_SENDS in (1, 2)`).
    """
    tp_deg, c_rows, k_local = src_3d.shape
    use_ps = _resolve_parity_stagger(parity_stagger, tp_deg)

    def _xor_a2a_kernel(src_ref, dst_ref, rsems, ssems):
        me = lax.axis_index(tp_axis)
        pltpu.make_async_copy(
            src_ref.at[pl.ds(me, 1)],
            dst_ref.at[pl.ds(0, 1)],
            rsems.at[0],
        ).start()
        if tp_deg > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, tp_deg):
                pl.semaphore_signal(
                    barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                )
            pl.semaphore_wait(barrier, tp_deg - 1)
            for d in range(1, tp_deg):
                peer = _kdeep_step_peer_jax(me, d, tp_deg, parity_stagger=use_ps)
                pltpu.make_async_remote_copy(
                    src_ref=src_ref.at[pl.ds(peer, 1)],
                    dst_ref=dst_ref.at[pl.ds(d, 1)],
                    send_sem=ssems.at[d],
                    recv_sem=rsems.at[d],
                    device_id={tp_axis: peer},
                ).start()
        for d in range(tp_deg):
            pltpu.make_async_copy(
                dst_ref.at[pl.ds(d, 1)],
                dst_ref.at[pl.ds(d, 1)],
                rsems.at[d],
            ).wait()
        if tp_deg > 1:
            for d in range(1, tp_deg):
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(0, 1)],
                    src_ref.at[pl.ds(0, 1)],
                    ssems.at[d],
                ).wait()

    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    return pl.pallas_call(
        _xor_a2a_kernel,
        grid=(1,),
        in_specs=[any_spec],
        out_specs=any_spec,
        out_shape=jax.ShapeDtypeStruct((tp_deg, c_rows, k_local), src_3d.dtype),
        scratch_shapes=[
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((tp_deg,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=collective_id if tp_deg > 1 else None,
        ),
        name=f"pallas_a3_xor_a2a_r{c_rows}{'_ps' if use_ps else ''}",
    )(src_3d)


def pallas_gate_static_fp8_quant_xor_a2a(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    *,
    tp_axis: str = "model",
    tp_deg: int = 8,
    tm: int = 256,
    vmem_limit_mb: int = 32,
    head_norm_eps: float | None = None,
    rpa_packed_out: bool = False,
    collective_id: int = 35,
    parity_stagger: bool | None = None,
) -> jax.Array:
    """Fused prologue + direct VMEM->peer-HBM XOR push All-to-All (`A3_KDEEP_WO=3`).

    Computes `clip(sigmoid(gate) * rmsnorm(attn) * 32.0, -448, 448) -> fp8_e4m3fn` directly
    into a double-buffered `(2, tm_c, k_local)` VMEM tile in peer-push order (`d = 1..tp-1, 0`)
    and pushes each tile straight to peer `dst_hbm[d]` via `make_async_remote_copy`, removing
    both the intermediate `q_loc` HBM write+read round-trip and the standalone A2A launch.
    """
    t_len, k_local = gate.shape
    assert t_len % tp_deg == 0, (t_len, tp_deg)
    c_rows = t_len // tp_deg
    use_ps = _resolve_parity_stagger(parity_stagger, tp_deg)
    zs_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE", None)
    use_zero_spill = bool(zs_env is not None and zs_env.get())
    max_tm = min(128 if use_zero_spill else tm, c_rows)
    tm_c = next(
        t for t in (512, 256, 128, 64, 32, 16, 8) if t <= max_tm and c_rows % t == 0
    )
    n_tiles_c = c_rows // tm_c
    total_steps = tp_deg * n_tiles_c
    is_rpa_packed = bool(rpa_packed_out and normed_attn_out.size != t_len * k_local)

    me_out = lax.axis_index(tp_axis)
    if tp_deg > 1:
        peers = jnp.stack(
            [
                _kdeep_step_peer_jax(me_out, d, tp_deg, parity_stagger=use_ps)
                for d in range(1, tp_deg)
            ]
            + [me_out]
        ).astype(jnp.int32)
    else:
        peers = jnp.zeros((1,), dtype=jnp.int32)

    if is_rpa_packed:
        num_q_slices = k_local // 128
        attn_arg = normed_attn_out.reshape(1, t_len, num_q_slices, 2, 128)
        attn_spec = pl.BlockSpec(
            (1, tm_c, num_q_slices, 2, 128),
            lambda d_i, t_i, p_ref: (0, p_ref[d_i] * n_tiles_c + t_i, 0, 0, 0),
        )
    else:
        num_q_slices = 0
        attn_arg = normed_attn_out.reshape(t_len, k_local)
        attn_spec = pl.BlockSpec(
            (tm_c, k_local),
            lambda d_i, t_i, p_ref: (p_ref[d_i] * n_tiles_c + t_i, 0),
        )
    gate_spec = pl.BlockSpec(
        (tm_c, k_local),
        lambda d_i, t_i, p_ref: (p_ref[d_i] * n_tiles_c + t_i, 0),
    )

    def _quant_push_kernel(peers_ref, attn_ref, gate_ref, dst_ref, q_vmem, tile_sems, rsems):
        me = lax.axis_index(tp_axis)
        d_idx = pl.program_id(0)
        t_idx = pl.program_id(1)
        step = d_idx * n_tiles_c + t_idx
        slot = lax.rem(step, 2)

        if tp_deg > 1:

            @pl.when(step == 0)
            def _barrier():
                bar = pltpu.get_barrier_semaphore()
                for kk in range(1, tp_deg):
                    pl.semaphore_signal(bar, device_id={tp_axis: lax.rem(me + kk, tp_deg)})
                pl.semaphore_wait(bar, tp_deg - 1)

        @pl.when(step >= 2)
        def _wait_prev_slot():
            pltpu.make_async_copy(
                q_vmem.at[pl.ds(slot, 1)],
                q_vmem.at[pl.ds(slot, 1)],
                tile_sems.at[slot],
            ).wait()

        if is_rpa_packed:
            lane = lax.broadcasted_iota(jnp.int32, (tm_c, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            attn_words = attn_ref.bitcast(jnp.uint32).at[0].reshape(
                tm_c * num_q_slices, 128
            )
            for s in range(num_q_slices):
                c0, c1 = s * 128, (s + 1) * 128
                w = attn_words[pl.ds(s, tm_c, stride=num_q_slices)]
                even_raw = pltpu.bitcast(lax.shift_left(w, jnp.uint32(16)), jnp.float32)
                odd_raw = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
                if s < num_q_slices // 2:
                    pair_f32 = jnp.where(
                        is_lo, even_raw, pltpu.roll(odd_raw, shift=64, axis=1)
                    )
                else:
                    pair_f32 = jnp.where(
                        is_lo, pltpu.roll(even_raw, shift=64, axis=1), odd_raw
                    )
                pair_f32 = pair_f32 + 0.0
                if head_norm_eps is not None:
                    pair_bf16 = _dual_head_rmsnorm_128(
                        pair_f32, is_lo, mask_lo, mask_hi, 1.0, eps=head_norm_eps
                    ).astype(jnp.bfloat16)
                else:
                    pair_bf16 = pair_f32.astype(jnp.bfloat16)
                g_sig = jax.nn.sigmoid(gate_ref[:, c0:c1].astype(jnp.float32)).astype(
                    jnp.bfloat16
                )
                gf = (g_sig * pair_bf16).astype(jnp.float32) * STATIC_A3_ACT_MULT
                q_vmem[slot, :, c0:c1] = jnp.clip(gf, -_FP8_MAX, _FP8_MAX).astype(
                    jnp.float8_e4m3fn
                )
        else:
            sub_r = min(128, tm_c)
            lane = lax.broadcasted_iota(jnp.int32, (sub_r, 128), 1)
            is_lo = lane < 64
            mask_lo = is_lo.astype(jnp.float32)
            mask_hi = 1.0 - mask_lo
            for r0 in range(0, tm_c, sub_r):
                r1 = r0 + sub_r
                for c0 in range(0, k_local, 128):
                    c1 = c0 + 128
                    a_f32 = attn_ref[r0:r1, c0:c1].astype(jnp.float32)
                    if head_norm_eps is not None:
                        a_bf16 = _dual_head_rmsnorm_128(
                            a_f32, is_lo, mask_lo, mask_hi, 1.0, eps=head_norm_eps
                        ).astype(jnp.bfloat16)
                    else:
                        a_bf16 = a_f32.astype(jnp.bfloat16)
                    g_sig = jax.nn.sigmoid(
                        gate_ref[r0:r1, c0:c1].astype(jnp.float32)
                    ).astype(jnp.bfloat16)
                    gf = (g_sig * a_bf16).astype(jnp.float32) * STATIC_A3_ACT_MULT
                    q_vmem[slot, r0:r1, c0:c1] = jnp.clip(
                        gf, -_FP8_MAX, _FP8_MAX
                    ).astype(jnp.float8_e4m3fn)

        if tp_deg > 1:

            @pl.when(d_idx < tp_deg - 1)
            def _push_remote():
                d_val = d_idx + 1
                peer = peers_ref[d_idx]
                pltpu.make_async_remote_copy(
                    src_ref=q_vmem.at[pl.ds(slot, 1)],
                    dst_ref=dst_ref.at[pl.ds(d_val, 1), pl.ds(t_idx * tm_c, tm_c), :],
                    send_sem=tile_sems.at[slot],
                    recv_sem=rsems.at[d_val],
                    device_id={tp_axis: peer},
                ).start()

        @pl.when(d_idx == tp_deg - 1)
        def _copy_local():
            pltpu.make_async_copy(
                src_ref=q_vmem.at[pl.ds(slot, 1)],
                dst_ref=dst_ref.at[pl.ds(0, 1), pl.ds(t_idx * tm_c, tm_c), :],
                sem=tile_sems.at[slot],
            ).start()

        @pl.when(step == total_steps - 1)
        def _drain():
            for ks in range(min(2, total_steps)):
                pltpu.make_async_copy(
                    q_vmem.at[pl.ds(ks, 1)],
                    q_vmem.at[pl.ds(ks, 1)],
                    tile_sems.at[ks],
                ).wait()
            if tp_deg > 1:
                for dd in range(1, tp_deg):
                    pltpu.make_async_copy(
                        dst_ref.at[pl.ds(dd, 1), pl.ds(0, c_rows), :],
                        dst_ref.at[pl.ds(dd, 1), pl.ds(0, c_rows), :],
                        rsems.at[dd],
                    ).wait()

    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    tag = (
        f"pallas_a3_pro_xor_a2a_r{c_rows}_tm{tm_c}"
        + ("_rpao" if is_rpa_packed else "")
        + ("_ps" if use_ps else "")
    )
    return pl.pallas_call(
        _quant_push_kernel,
        out_shape=jax.ShapeDtypeStruct((tp_deg, c_rows, k_local), jnp.float8_e4m3fn),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(tp_deg, n_tiles_c),
            in_specs=[attn_spec, gate_spec],
            out_specs=any_spec,
            scratch_shapes=[
                pltpu.VMEM((2, tm_c, k_local), jnp.float8_e4m3fn),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((tp_deg,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            collective_id=collective_id if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=tag,
    )(peers, attn_arg, gate)


def _kdeep_dot_tile(
    q_src_3d_ref,
    w_ref,
    w_sc_ref,
    *,
    tp_deg: int,
    m0: int,
    m1: int,
    c0: int,
    c1: int,
    w4_qbs: int,
    k_local: int,
) -> jax.Array:
    """Compute `[m1 - m0, c1 - c0] bf16` tile over all `tp_deg` K-blocks in VREGs (0 `acc_vmem`)."""
    if w4_qbs > 0:
        n_kb = k_local // w4_qbs
        acc = jnp.zeros((m1 - m0, c1 - c0), dtype=jnp.float32)
        for d in range(tp_deg):
            for kb in range(n_kb):
                k0, k1 = kb * w4_qbs, (kb + 1) * w4_qbs
                pop = jnp.matmul(
                    q_src_3d_ref[d, m0:m1, k0:k1],
                    w_ref[d, k0:k1, c0:c1],
                    preferred_element_type=jnp.float32,
                )
                acc = acc + pop * w_sc_ref[d, kb : kb + 1, c0:c1].astype(jnp.float32)
        return (acc * jnp.float32(STATIC_A3_ACT_SCALE)).astype(jnp.bfloat16)

    acc = jnp.dot(
        q_src_3d_ref[0, m0:m1, :],
        w_ref[0, :, c0:c1],
        preferred_element_type=jnp.float32,
    )
    for d in range(1, tp_deg):
        acc = acc + jnp.dot(
            q_src_3d_ref[d, m0:m1, :],
            w_ref[d, :, c0:c1],
            preferred_element_type=jnp.float32,
        )
    sc_sub = w_sc_ref[:, c0:c1] * jnp.float32(STATIC_A3_ACT_SCALE)
    return (acc * sc_sub).astype(jnp.bfloat16)


def pallas_a3_wo_kdeep_static_fp8(
    q_3d: jax.Array,
    wo_xor_3d: jax.Array,
    wo_scale: jax.Array,
    *,
    tp_axis: str = "model",
    bn: int = 1024,
    w4_qbs: int = 0,
    pre_exchanged: bool = False,
    parity_stagger: bool | None = None,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """VMEM-stationary `[tp_deg, t_shard, 1024]` K-deep single-pass `Wo` GEMM (`0` `acc_vmem`).

    Streams `wo_xor_3d` (`64 MiB` FP8 or `32 MiB` NVFP4) strictly ONCE across `grid=(n_dim // bn,)`
    while keeping `q` (`16 MiB` at `T=16K`) resident in VMEM and accumulating all `K=8192`
    contractions in `[bm=256, sub_n=256]` VREG tiles (`64` VREGs, `0` `896 MiB` VMEM RMW).
    """
    tp_deg, t_shard, k_local = q_3d.shape
    _, _, n_dim = wo_xor_3d.shape
    bn = 512 if (bn > 512 and t_shard > 3584 and w4_qbs == 0) else bn
    num_j = n_dim // bn
    bm = next(t for t in (256, 128, 64, 32, 8) if t <= t_shard and t_shard % t == 0)
    sub_n = min(256, bn)
    use_ps = _resolve_parity_stagger(parity_stagger, tp_deg)
    w_sc_spec = (
        pl.BlockSpec((tp_deg, k_local // w4_qbs, bn), lambda j: (0, 0, j))
        if w4_qbs > 0
        else pl.BlockSpec((1, bn), lambda j: (0, j))
    )
    w_tag = f"_w4q{w4_qbs}" if w4_qbs > 0 else ""

    if pre_exchanged:

        def _pre_exchanged_kernel(q_ref, w_ref, w_sc_ref, out_ref):
            for m0 in range(0, t_shard, bm):
                m1 = m0 + bm
                for c0 in range(0, bn, sub_n):
                    c1 = c0 + sub_n
                    out_ref[m0:m1, c0:c1] = _kdeep_dot_tile(
                        q_ref,
                        w_ref,
                        w_sc_ref,
                        tp_deg=tp_deg,
                        m0=m0,
                        m1=m1,
                        c0=c0,
                        c1=c1,
                        w4_qbs=w4_qbs,
                        k_local=k_local,
                    )

        return pl.pallas_call(
            _pre_exchanged_kernel,
            grid=(num_j,),
            in_specs=[
                pl.BlockSpec((tp_deg, t_shard, k_local), lambda j: (0, 0, 0)),
                pl.BlockSpec((tp_deg, k_local, bn), lambda j: (0, 0, j)),
                w_sc_spec,
            ],
            out_specs=pl.BlockSpec((t_shard, bn), lambda j: (0, j)),
            out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_mb * (1 << 20),
            ),
            name=f"pallas_a3_wo_kdeep_pre_bn{bn}{w_tag}",
        )(q_3d, wo_xor_3d, wo_scale)

    def _fused_a2a_kdeep_kernel(
        src_hbm_ref,
        w_ref,
        w_sc_ref,
        out_hbm_ref,
        recv_vmem,
        out_vmem,
        rsems,
        ssems,
        out_sems,
    ):
        j = pl.program_id(0)
        me = lax.axis_index(tp_axis)
        out_slot = lax.rem(j, 2)

        @pl.when(j == 0)
        def _init_a2a():
            pltpu.make_async_copy(
                src_hbm_ref.at[pl.ds(me, 1)],
                recv_vmem.at[pl.ds(0, 1)],
                rsems.at[0],
            ).start()
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(
                        barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                    )
                pl.semaphore_wait(barrier, tp_deg - 1)
                for d in range(1, tp_deg):
                    peer = _kdeep_step_peer_jax(me, d, tp_deg, parity_stagger=use_ps)
                    pltpu.make_async_remote_copy(
                        src_ref=src_hbm_ref.at[pl.ds(peer, 1)],
                        dst_ref=recv_vmem.at[pl.ds(d, 1)],
                        send_sem=ssems.at[d],
                        recv_sem=rsems.at[d],
                        device_id={tp_axis: peer},
                    ).start()
            for d in range(tp_deg):
                pltpu.make_async_copy(
                    recv_vmem.at[pl.ds(d, 1)],
                    recv_vmem.at[pl.ds(d, 1)],
                    rsems.at[d],
                ).wait()

        @pl.when(j >= 2)
        def _wait_prev_out():
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(t_shard, bn),
                out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                out_sems.at[out_slot],
            ).wait()

        for m0 in range(0, t_shard, bm):
            m1 = m0 + bm
            for c0 in range(0, bn, sub_n):
                c1 = c0 + sub_n
                out_vmem[out_slot, m0:m1, c0:c1] = _kdeep_dot_tile(
                    recv_vmem,
                    w_ref,
                    w_sc_ref,
                    tp_deg=tp_deg,
                    m0=m0,
                    m1=m1,
                    c0=c0,
                    c1=c1,
                    w4_qbs=w4_qbs,
                    k_local=k_local,
                )

        pltpu.make_async_copy(
            out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(t_shard, bn),
            out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(j * bn, bn)],
            out_sems.at[out_slot],
        ).start()

        @pl.when(j == num_j - 1)
        def _drain():
            for s in range(min(2, num_j)):
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(s, 1), :, :].reshape(t_shard, bn),
                    out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                    out_sems.at[s],
                ).wait()
            if tp_deg > 1:
                for d in range(1, tp_deg):
                    pltpu.make_async_copy(
                        src_hbm_ref.at[pl.ds(0, 1)],
                        src_hbm_ref.at[pl.ds(0, 1)],
                        ssems.at[d],
                    ).wait()

    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    return pl.pallas_call(
        _fused_a2a_kdeep_kernel,
        grid=(num_j,),
        in_specs=[
            any_spec,
            pl.BlockSpec((tp_deg, k_local, bn), lambda j: (0, 0, j)),
            w_sc_spec,
        ],
        out_specs=any_spec,
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tp_deg, t_shard, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((2, t_shard, bn), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=32 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_a3_wo_kdeep_bn{bn}{w_tag}{'_ps' if use_ps else ''}",
    )(q_3d, wo_xor_3d, wo_scale)


def pallas_a3_wo_kdeep_ccs_last_a2a(
    q_prev_xor_3d: jax.Array,
    q_last_local_3d: jax.Array,
    wo_xor_3d: jax.Array,
    wo_scale: jax.Array,
    *,
    tp_axis: str = "model",
    bn: int = 1024,
    w4_qbs: int = 0,
    parity_stagger: bool | None = None,
    vmem_limit_mb: int = 64,
) -> jax.Array:
    """CCS Mode-2 K-Deep `Wo`: overlaps final chunk's `5.25 MiB` A2A under `j=0` prefix MXU dots."""
    tp_deg, prev_rows, k_local = q_prev_xor_3d.shape
    _, last_rows, _ = q_last_local_3d.shape
    t_shard = prev_rows + last_rows
    _, _, n_dim = wo_xor_3d.shape
    bn = 512 if (bn > 512 and t_shard > 3584 and w4_qbs == 0) else bn
    num_j = n_dim // bn
    bm_prev = next(
        t for t in (256, 128, 64, 32, 8) if t <= prev_rows and prev_rows % t == 0
    )
    bm_last = next(
        t for t in (256, 128, 64, 32, 8) if t <= last_rows and last_rows % t == 0
    )
    sub_n = min(256, bn)
    use_ps = _resolve_parity_stagger(parity_stagger, tp_deg)
    w_sc_spec = (
        pl.BlockSpec((tp_deg, k_local // w4_qbs, bn), lambda j: (0, 0, j))
        if w4_qbs > 0
        else pl.BlockSpec((1, bn), lambda j: (0, j))
    )
    w_tag = f"_w4q{w4_qbs}" if w4_qbs > 0 else ""

    def _ccs_kdeep_kernel(
        q_prev_ref,
        q_last_hbm_ref,
        w_ref,
        w_sc_ref,
        out_hbm_ref,
        q_last_recv_vmem,
        out_vmem,
        rsems,
        ssems,
        out_sems,
    ):
        j = pl.program_id(0)
        me = lax.axis_index(tp_axis)
        out_slot = lax.rem(j, 2)

        @pl.when(j == 0)
        def _start_last_a2a():
            pltpu.make_async_copy(
                q_last_hbm_ref.at[pl.ds(me, 1)],
                q_last_recv_vmem.at[pl.ds(0, 1)],
                rsems.at[0],
            ).start()
            if tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, tp_deg):
                    pl.semaphore_signal(
                        barrier, device_id={tp_axis: lax.rem(me + d, tp_deg)}
                    )
                pl.semaphore_wait(barrier, tp_deg - 1)
                for d in range(1, tp_deg):
                    peer = _kdeep_step_peer_jax(me, d, tp_deg, parity_stagger=use_ps)
                    pltpu.make_async_remote_copy(
                        src_ref=q_last_hbm_ref.at[pl.ds(peer, 1)],
                        dst_ref=q_last_recv_vmem.at[pl.ds(d, 1)],
                        send_sem=ssems.at[d],
                        recv_sem=rsems.at[d],
                        device_id={tp_axis: peer},
                    ).start()

        @pl.when(j >= 2)
        def _wait_prev_out():
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(t_shard, bn),
                out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                out_sems.at[out_slot],
            ).wait()

        # Compute Chunks 0..K-2 (`0 : prev_rows`) immediately while Chunk K-1 A2A flies on ICI!
        for m0 in range(0, prev_rows, bm_prev):
            m1 = m0 + bm_prev
            for c0 in range(0, bn, sub_n):
                c1 = c0 + sub_n
                out_vmem[out_slot, m0:m1, c0:c1] = _kdeep_dot_tile(
                    q_prev_ref,
                    w_ref,
                    w_sc_ref,
                    tp_deg=tp_deg,
                    m0=m0,
                    m1=m1,
                    c0=c0,
                    c1=c1,
                    w4_qbs=w4_qbs,
                    k_local=k_local,
                )

        @pl.when(j == 0)
        def _wait_last_a2a():
            for d in range(tp_deg):
                pltpu.make_async_copy(
                    q_last_recv_vmem.at[pl.ds(d, 1)],
                    q_last_recv_vmem.at[pl.ds(d, 1)],
                    rsems.at[d],
                ).wait()

        for m0 in range(0, last_rows, bm_last):
            m1 = m0 + bm_last
            for c0 in range(0, bn, sub_n):
                c1 = c0 + sub_n
                out_vmem[out_slot, prev_rows + m0 : prev_rows + m1, c0:c1] = (
                    _kdeep_dot_tile(
                        q_last_recv_vmem,
                        w_ref,
                        w_sc_ref,
                        tp_deg=tp_deg,
                        m0=m0,
                        m1=m1,
                        c0=c0,
                        c1=c1,
                        w4_qbs=w4_qbs,
                        k_local=k_local,
                    )
                )

        pltpu.make_async_copy(
            out_vmem.at[pl.ds(out_slot, 1), :, :].reshape(t_shard, bn),
            out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(j * bn, bn)],
            out_sems.at[out_slot],
        ).start()

        @pl.when(j == num_j - 1)
        def _drain():
            for s in range(min(2, num_j)):
                pltpu.make_async_copy(
                    out_vmem.at[pl.ds(s, 1), :, :].reshape(t_shard, bn),
                    out_hbm_ref.at[pl.ds(0, t_shard), pl.ds(0, bn)],
                    out_sems.at[s],
                ).wait()
            if tp_deg > 1:
                for d in range(1, tp_deg):
                    pltpu.make_async_copy(
                        q_last_hbm_ref.at[pl.ds(0, 1)],
                        q_last_hbm_ref.at[pl.ds(0, 1)],
                        ssems.at[d],
                    ).wait()

    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    return pl.pallas_call(
        _ccs_kdeep_kernel,
        grid=(num_j,),
        in_specs=[
            pl.BlockSpec((tp_deg, prev_rows, k_local), lambda j: (0, 0, 0)),
            any_spec,
            pl.BlockSpec((tp_deg, k_local, bn), lambda j: (0, 0, j)),
            w_sc_spec,
        ],
        out_specs=any_spec,
        out_shape=jax.ShapeDtypeStruct((t_shard, n_dim), jnp.bfloat16),
        scratch_shapes=[
            pltpu.VMEM((tp_deg, last_rows, k_local), jnp.float8_e4m3fn),
            pltpu.VMEM((2, t_shard, bn), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((tp_deg,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=34 if tp_deg > 1 else None,
            vmem_limit_bytes=vmem_limit_mb * (1 << 20),
        ),
        name=f"pallas_a3_wo_kdeep_ccs_p{prev_rows}_l{last_rows}_bn{bn}{w_tag}{'_ps' if use_ps else ''}",
    )(q_prev_xor_3d, q_last_local_3d, wo_xor_3d, wo_scale)


def a3_v16_kdeep_sp(
    normed_attn_out: jax.Array,
    gate: jax.Array,
    wo_xor_3d: jax.Array,
    wo_scale: jax.Array,
    *,
    tp_axis: str = "model",
    bn: int = 1024,
    head_norm_eps: float | None = None,
    rpa_packed_out: bool = False,
    prologue_tm: int = 1024,
    valid_tokens_1d: jax.Array | None = None,
    w4_qbs: int = 0,
    parity_stagger: bool | None = None,
) -> jax.Array:
    """Full A3 prologue + fused A2A K-deep `Wo` (`A3_KDEEP_WO=1` or `W4_WO=256`)."""
    t_total, k_local = gate.shape
    tp_deg = 8192 // k_local
    t_shard = t_total // tp_deg
    eff_ptm = 512 if valid_tokens_1d is not None else min(prologue_tm, t_total)
    q_local_3d = pallas_gate_static_fp8_quant(
        normed_attn_out,
        gate,
        tm=eff_ptm,
        head_norm_eps=head_norm_eps,
        rpa_packed_out=rpa_packed_out,
        valid_tokens_1d=valid_tokens_1d,
    ).reshape(tp_deg, t_shard, k_local)
    return pallas_a3_wo_kdeep_static_fp8(
        q_local_3d,
        wo_xor_3d,
        wo_scale,
        tp_axis=tp_axis,
        bn=bn,
        w4_qbs=w4_qbs,
        pre_exchanged=False,
        parity_stagger=parity_stagger,
    )


# -----------------------------------------------------------------------------
# 2. Kernel D Run-Coalesced Gather + Wsel MXU Combine + Single-Call Push-AG + W4
# -----------------------------------------------------------------------------
def precompute_d_coal_runs_and_wsel(
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    *,
    tok_tile: int,
    row_start: int = 0,
    num_rows: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Precompute exact variable-length contiguous HBM run descriptors and `Wsel [tok_tile, P]`.

    Returns:
      - `run_meta`: `int32[num_steps, n_pairs, 4]` where `run_meta[s, 0, 3] = n_runs`
        (`1 <= n_runs <= n_pairs`) and `run_meta[s, r, :3] = (src_start, dst_start, run_len)`
        with `sum_{r < n_runs} run_len == n_pairs` identically for ALL integer inputs.
      - `wsel`: `bfloat16[num_steps, tok_tile, n_pairs]` placing `local_topk_weights[t, k]` at
        the sorted destination column `j` so `jnp.dot(wsel[s], vals_sorted)` both unpermutes
        and combines in a single MXU pass.
    """
    chunk_size, topk = inv_perm_c.shape
    num_rows = (chunk_size - row_start) if num_rows is None else num_rows
    num_steps = num_rows // tok_tile
    n_pairs = tok_tile * topk

    inv_tiles = lax.dynamic_slice_in_dim(
        inv_perm_c, row_start, num_rows, axis=0
    ).reshape(num_steps, n_pairs)
    w_tiles = (
        lax.dynamic_slice_in_dim(local_topk_weights, row_start, num_rows, axis=0)
        .astype(jnp.bfloat16)
        .reshape(num_steps, n_pairs)
    )
    pair_iota = jnp.arange(n_pairs, dtype=jnp.int32)
    tok_iota = jnp.arange(tok_tile, dtype=jnp.int32)[:, None]

    def _per_step(inv_1d: jax.Array, w_1d: jax.Array):
        idx_sorted, perm = lax.sort_key_val(inv_1d, pair_iota)
        is_start = jnp.concatenate(
            [
                jnp.array([True]),
                idx_sorted[1:] != (idx_sorted[:-1] + jnp.int32(1)),
            ],
            axis=0,
        )
        n_runs = jnp.sum(is_start.astype(jnp.int32))
        # Compact run start positions via stable sort on (~is_start) (puts all True first).
        _, start_order = lax.sort_key_val(
            jnp.where(is_start, jnp.int32(0), jnp.int32(1)), pair_iota
        )
        dst_starts = pair_iota[start_order]
        src_starts = idx_sorted[start_order]
        next_dst = jnp.concatenate(
            [dst_starts[1:], jnp.array([n_pairs], dtype=jnp.int32)], axis=0
        )
        valid_r = pair_iota < n_runs
        end_dst = jnp.where(
            pair_iota + jnp.int32(1) < n_runs, next_dst, jnp.int32(n_pairs)
        )
        run_lens = jnp.where(valid_r, end_dst - dst_starts, jnp.int32(0))
        src_starts = jnp.where(valid_r, src_starts, jnp.int32(0))
        dst_starts = jnp.where(valid_r, dst_starts, jnp.int32(0))
        n_runs_col = jnp.broadcast_to(n_runs, (n_pairs,))
        meta_s = jnp.stack([src_starts, dst_starts, run_lens, n_runs_col], axis=-1)

        # Wsel[t, j] = w_1d[perm[j]] if (perm[j] >> 3) == t else 0.0 (for power-of-2 topk=8).
        tok_of_j = (perm // topk)[None, :]
        w_sorted = w_1d[perm][None, :]
        wsel_s = jnp.where(tok_of_j == tok_iota, w_sorted, jnp.bfloat16(0.0))
        return meta_s, wsel_s

    return jax.vmap(_per_step)(inv_tiles, w_tiles)


def pallas_d_radical_4slab(
    out_buf_3d: jax.Array,
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    stream_shard_mid: jax.Array,
    wunpool: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    tok_tile: int = 64,
    fp8_unpool: bool = False,
    fp8_unpool_qbs: int = 0,
    emit_fp8_attn_in: bool = False,
    emit_static_fp8_attn_in: bool = False,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
    row_start: int = 0,
    num_rows: int | None = None,
    stream_out: jax.Array | None = None,
    coal_gather: bool = False,
    in_kernel_ag_rows: int = 0,
    ag_axis_name: str | None = None,
    ag_tp_deg: int = 8,
    ag_part0_static_fp8: bool = False,
    w4_unpool_qbs: int = 0,
    wunpool_w4_scale: jax.Array | None = None,
):
    """Unified Wave-8 Kernel D supporting `D_COAL_GATHER`, `D_IN_KERNEL_AG`, and `W4_WUNPOOL`."""
    chunk_size, topk = inv_perm_c.shape
    n2_cols, lanes = out_buf_3d.shape[1], out_buf_3d.shape[2]
    he_dim = n2_cols * lanes
    w_k, h_dim = wunpool.shape
    assert he_dim == w_k == 4096 and h_dim == 8192
    if jnp.dtype(out_buf_3d.dtype) == jnp.float8_e4m3fn:
        _sfp8_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS", None)
        if _sfp8_env is not None:
            _raw_sfp8 = _sfp8_env.get()
            try:
                from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
                    resolve_static_fp8_rs_scale,
                )

                _abs_sfp8, _ = resolve_static_fp8_rs_scale(_raw_sfp8)
            except ImportError:
                _abs_sfp8 = abs(float(_raw_sfp8 or 0.0))
            if _abs_sfp8 > 0.0:
                local_topk_weights = (
                    local_topk_weights.astype(jnp.float32)
                    * jnp.float32(1.0 / _abs_sfp8)
                ).astype(local_topk_weights.dtype)

    env_tok_tile = envs.SGLANG_TPU_MUSE_KERNEL_D_TOK_TILE.get()
    if env_tok_tile not in (0, 64) and tok_tile == 64:
        tok_tile = env_tok_tile
    tok_tile_eff = min(tok_tile, chunk_size)
    assert chunk_size % tok_tile_eff == 0

    use_w4 = w4_unpool_qbs > 0 and wunpool_w4_scale is not None
    env_fp8_qbs = envs.SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL.get()
    eff_fp8_qbs = (
        4096
        if use_w4
        else (
            fp8_unpool_qbs
            if fp8_unpool_qbs > 0
            else (env_fp8_qbs if env_fp8_qbs > 0 else (4096 if fp8_unpool else 0))
        )
    )
    fp8_unpool = eff_fp8_qbs > 0

    use_push_ag = (
        in_kernel_ag_rows > 0
        and in_kernel_ag_rows < chunk_size
        and in_kernel_ag_rows % tok_tile_eff == 0
        and ag_axis_name is not None
        and row_start == 0
        and num_rows is None
        and stream_out is None
    )
    steps_ag = (in_kernel_ag_rows // tok_tile_eff) if use_push_ag else 0
    rest_rows = (chunk_size - in_kernel_ag_rows) if use_push_ag else 0

    num_rows = (chunk_size - row_start) if num_rows is None else num_rows
    assert row_start % tok_tile_eff == 0 and num_rows % tok_tile_eff == 0
    tile_off = row_start // tok_tile_eff
    tok_map = (
        (lambda i, *_: (i, 0)) if tile_off == 0 else (lambda i, *_: (i + tile_off, 0))
    )
    num_steps = num_rows // tok_tile_eff
    n_pairs = tok_tile_eff * topk

    ones_1d = jnp.ones((h_dim,), dtype=jnp.float32)
    vecs_8x8192 = prepare_d_static_inputs(
        ones_1d if use_w4 else wunpool_scale,
        post_ffn_norm_w,
        alpha_ffn,
        beta_ffn,
        next_input_norm_w,
    )
    if coal_gather:
        run_meta, wsel_3d = precompute_d_coal_runs_and_wsel(
            inv_perm_c,
            local_topk_weights,
            tok_tile=tok_tile_eff,
            row_start=row_start,
            num_rows=num_rows,
        )
        w_in_arg = wsel_3d
        w_in_spec = pl.BlockSpec((1, tok_tile_eff, n_pairs), lambda i, *_: (i, 0, 0))
        # Pack into flat 1D int32 so TPU Mosaic SMEM never pads minor dim=4 to 128B (>1 MiB OOM).
        n_runs_col = run_meta[:, 0:1, 3]
        packed_pairs = jnp.stack(
            [
                run_meta[:, :, 0],
                (run_meta[:, :, 1] << jnp.int32(12)) | run_meta[:, :, 2],
            ],
            axis=-1,
        ).reshape(num_steps, 2 * n_pairs)
        scalar_pref = jnp.concatenate([n_runs_col, packed_pairs], axis=1).reshape(-1)
    else:
        w_pad = jnp.pad(
            local_topk_weights.astype(jnp.bfloat16), ((0, 0), (0, LANE - topk))
        )
        w_in_arg = w_pad
        w_in_spec = pl.BlockSpec((tok_tile_eff, LANE), tok_map)
        scalar_pref = inv_perm_c.reshape(-1)

    slab_w = 2048
    num_slabs = h_dim // slab_w
    sub_n = 512
    inv_h = 1.0 / float(h_dim)
    next_in_cols = (h_dim + LANE) if emit_fp8_attn_in else h_dim
    next_in_dtype = (
        jnp.float8_e4m3fn
        if (emit_fp8_attn_in or emit_static_fp8_attn_in)
        else jnp.bfloat16
    )
    ag_part0_is_s32_fp8 = bool(
        use_push_ag
        and ag_part0_static_fp8
        and not (emit_fp8_attn_in or emit_static_fp8_attn_in)
    )
    gathered_dtype = (
        jnp.float8_e4m3fn if ag_part0_is_s32_fp8 else next_in_dtype
    )
    d_opt_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_D_OPT", None)
    eff_d_opt = int(d_opt_env.get() or 0) if d_opt_env is not None else 0
    d_fast_dma = (eff_d_opt & 1) != 0
    d_fuse_p34 = ((eff_d_opt & 2) != 0) and not emit_fp8_attn_in
    use_salu_overlap = envs.SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP.get()
    eff_salu_ovlp = use_salu_overlap and (tok_tile_eff % num_slabs == 0) and not coal_gather

    w4_sc_3d = (
        wunpool_w4_scale.reshape(he_dim // w4_unpool_qbs, num_slabs, slab_w).swapaxes(0, 1)
        if use_w4
        else None
    )

    def _d_radical_kernel(*args):
        idx = 0
        pref_ref = args[idx]
        idx += 1
        src_hbm_ref = args[idx]
        wunpool_hbm_ref = args[idx + 1]
        w_in_ref = args[idx + 2]
        stream_mid_ref = args[idx + 3]
        vecs_ref = args[idx + 4]
        idx += 5
        w4_sc_ref = args[idx] if use_w4 else None
        idx += 1 if use_w4 else 0
        if stream_out is not None:
            idx += 1  # skip aliased stream_out input ref

        new_stream_out_ref = args[idx]
        next_in_out_ref = args[idx + 1]
        idx += 2
        gathered_hbm_ref = args[idx] if use_push_ag else None
        idx += 1 if use_push_ag else 0

        buf_2x_ref = args[idx]
        sem_2x_ref = args[idx + 1]
        u_scratch_ref = args[idx + 2]
        u_fp8_scratch_ref = args[idx + 3]
        w_cache_ref = args[idx + 4]
        w_sem_ref = args[idx + 5]
        idx += 6
        if use_push_ag:
            ag_tile_vmem = args[idx]
            ag_local_sems = args[idx + 1]
            ag_rsems = args[idx + 2]
            ag_ssems = args[idx + 3]
            idx += 4
            ag_p0_fp8_vmem = args[idx] if ag_part0_is_s32_fp8 else ag_tile_vmem
        else:
            ag_tile_vmem = ag_p0_fp8_vmem = ag_local_sems = ag_rsems = ag_ssems = None

        step = pl.program_id(0)
        slot = step % 2
        me = lax.axis_index(ag_axis_name) if use_push_ag else jnp.int32(0)

        def _start_step_range(s, sl, t0, t1):
            if coal_gather:
                base_s = s * (1 + 2 * n_pairs)
                n_runs = pref_ref[base_s]

                def _issue_run(r, _):
                    off = base_s + 1 + 2 * r
                    s0 = pref_ref[off]
                    packed = pref_ref[off + 1]
                    d0 = lax.shift_right_logical(packed, jnp.int32(12))
                    ln = packed & jnp.int32(0xFFF)
                    pltpu.make_async_copy(
                        src_ref=src_hbm_ref.at[pl.ds(s0, ln), :, :],
                        dst_ref=buf_2x_ref.at[sl, pl.ds(d0, ln), :, :],
                        sem=sem_2x_ref.at[sl],
                    ).start()
                    return None

                lax.fori_loop(0, n_runs, _issue_run, None)
            else:
                base_tok = (
                    (s + tile_off) * tok_tile_eff if tile_off else s * tok_tile_eff
                )
                for t in range(t0, t1):
                    tok = base_tok + t
                    dst_base = t * topk
                    idx_base = tok * topk
                    idx8 = [pref_ref[idx_base + k] for k in range(topk)]
                    for k in range(topk):
                        pltpu.make_async_copy(
                            src_ref=src_hbm_ref.at[pl.ds(idx8[k], 1), :, :],
                            dst_ref=buf_2x_ref.at[sl, pl.ds(dst_base + k, 1), :, :],
                            sem=sem_2x_ref.at[sl],
                        ).start()

        def _start_step(s, sl):
            _start_step_range(s, sl, 0, tok_tile_eff)

        def _wait_step(sl):
            pltpu.make_async_copy(
                src_ref=buf_2x_ref.at[sl],
                dst_ref=buf_2x_ref.at[sl],
                sem=sem_2x_ref.at[sl],
            ).wait()

        @pl.when(step == 0)
        def _init_step0():
            # Slab-first wunpool DMA start before step-0 gathers (#1409 D #1)!
            for s in range(num_slabs):
                pltpu.make_async_copy(
                    src_ref=wunpool_hbm_ref.at[:, pl.ds(s * slab_w, slab_w)],
                    dst_ref=w_cache_ref.at[s],
                    sem=w_sem_ref.at[s],
                ).start()
            _start_step(0, 0)
            if use_push_ag and ag_tp_deg > 1:
                barrier = pltpu.get_barrier_semaphore()
                for d in range(1, ag_tp_deg):
                    pl.semaphore_signal(
                        barrier,
                        device_id={ag_axis_name: lax.rem(me + d, ag_tp_deg)},
                    )
                pl.semaphore_wait(barrier, ag_tp_deg - 1)

        if not eff_salu_ovlp:

            @pl.when(step + 1 < num_steps)
            def _prefetch_next():
                _start_step(step + 1, 1 - slot)

        _wait_step(slot)

        # Phase 1: Combine into u_scratch_ref / u_fp8_scratch_ref
        amax_u_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
        if coal_gather:
            wsel_tile = w_in_ref[0, :, :]  # bf16[tok_tile_eff, n_pairs]
            for c_blk in range(n2_cols // 8):
                c0, c1 = c_blk * 8, (c_blk + 1) * 8
                vals_2d = (
                    buf_2x_ref[slot, :, c0:c1, :]
                    .astype(jnp.bfloat16)
                    .reshape(n_pairs, 8 * lanes)
                )
                u_f32 = jnp.dot(
                    wsel_tile, vals_2d, preferred_element_type=jnp.float32
                )
                u_bf16 = u_f32.astype(jnp.bfloat16)
                u_scratch_ref[:, c0 * lanes : c1 * lanes] = u_bf16
                if fp8_unpool:
                    amax_u_128 = jnp.maximum(
                        amax_u_128, vpu_tree_max_128(jnp.abs(u_bf16.astype(jnp.float32)))
                    )
        else:
            w_tok = w_in_ref[:, :topk].reshape(tok_tile_eff, topk, 1, 1)
            use_c16_half = (
                envs.SGLANG_TPU_MUSE_KERNEL_D_T64_C16.get() and tok_tile_eff % 32 == 0
            )
            if use_c16_half:
                for h_t in range(tok_tile_eff // 32):
                    r0, r1 = h_t * 32, (h_t + 1) * 32
                    m0, m1 = r0 * topk, r1 * topk
                    w_h = w_tok[r0:r1]
                    for c_blk in range(2):
                        c0, c1 = c_blk * 16, (c_blk + 1) * 16
                        vals_sub = (
                            buf_2x_ref[slot, m0:m1, c0:c1, :]
                            .astype(jnp.bfloat16)
                            .reshape(32, topk, 16, lanes)
                        )
                        u_sub = jnp.sum(vals_sub * w_h, axis=1).reshape(32, 16 * lanes)
                        u_scratch_ref[r0:r1, c0 * lanes : c1 * lanes] = u_sub
                if fp8_unpool:
                    for c_blk in range(4):
                        u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                        amax_u_128 = jnp.maximum(
                            amax_u_128,
                            vpu_tree_max_128(
                                jnp.abs(u_scratch_ref[:, u0:u1].astype(jnp.float32))
                            ),
                        )
            else:
                for c_blk in range(n2_cols // 8):
                    c0, c1 = c_blk * 8, (c_blk + 1) * 8
                    vals_sub = (
                        buf_2x_ref[slot, :, c0:c1, :]
                        .astype(jnp.bfloat16)
                        .reshape(tok_tile_eff, topk, 8, lanes)
                    )
                    u_sub = jnp.sum(vals_sub * w_tok, axis=1).reshape(
                        tok_tile_eff, 8 * lanes
                    )
                    u_scratch_ref[:, c0 * lanes : c1 * lanes] = u_sub
                    if fp8_unpool:
                        amax_u_128 = jnp.maximum(
                            amax_u_128,
                            vpu_tree_max_128(jnp.abs(u_sub.astype(jnp.float32))),
                        )

        if fp8_unpool:
            row_amax_u = jnp.max(amax_u_128, axis=1, keepdims=True)
            u_sc = jnp.where(
                row_amax_u == 0,
                jnp.zeros_like(row_amax_u),
                row_amax_u * jnp.float32(1.0 / _FP8_MAX),
            )
            inv_u_sc = jnp.where(
                row_amax_u == 0,
                jnp.zeros_like(row_amax_u),
                jnp.float32(_FP8_MAX) / jnp.maximum(row_amax_u, jnp.float32(1e-30)),
            )
            for c_blk in range(4):
                u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                u_fp8_scratch_ref[:, u0:u1] = jnp.clip(
                    u_scratch_ref[:, u0:u1].astype(jnp.float32) * inv_u_sc,
                    -_FP8_MAX,
                    _FP8_MAX,
                ).astype(jnp.float8_e4m3fn)
        else:
            u_sc = None

        # Phase 2: Per-slab waited wunpool dot + post_feedforward_layernorm variance
        if eff_salu_ovlp and not d_fast_dma:

            @pl.when(step == 0)
            def _wait_w_all():
                for sw_i in range(num_slabs):
                    pltpu.make_async_copy(
                        src_ref=w_cache_ref.at[sw_i],
                        dst_ref=w_cache_ref.at[sw_i],
                        sem=w_sem_ref.at[sw_i],
                    ).wait()

        next_s_k = jnp.minimum(step + 1, num_steps - 1)
        t_per_slab_k = tok_tile_eff // num_slabs
        sq_ffn_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
        if d_fuse_p34:
            s_aa_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            s_ab_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            s_bb_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
        for s in range(num_slabs):
            if not eff_salu_ovlp or d_fast_dma:

                @pl.when(step == 0)
                def _wait_w_slab(s=s):
                    pltpu.make_async_copy(
                        src_ref=w_cache_ref.at[s],
                        dst_ref=w_cache_ref.at[s],
                        sem=w_sem_ref.at[s],
                    ).wait()

            s_col0 = s * slab_w
            for sub in range(slab_w // sub_n):
                cn0, cn1 = sub * sub_n, (sub + 1) * sub_n
                gn0, gn1 = s_col0 + cn0, s_col0 + cn1
                if use_w4:
                    n_w4_kb = he_dim // w4_unpool_qbs
                    acc = jnp.zeros((tok_tile_eff, sub_n), dtype=jnp.float32)
                    for kb in range(n_w4_kb):
                        k0, k1 = kb * w4_unpool_qbs, (kb + 1) * w4_unpool_qbs
                        pop = jnp.matmul(
                            u_fp8_scratch_ref[:, k0:k1],
                            w_cache_ref[s, k0:k1, cn0:cn1],
                            preferred_element_type=jnp.float32,
                        )
                        acc = acc + pop * w4_sc_ref[s, kb : kb + 1, cn0:cn1].astype(
                            jnp.float32
                        )
                    y_sub = (acc * u_sc).astype(jnp.bfloat16).astype(jnp.float32)
                else:
                    w_sc_sub = vecs_ref[0:1, gn0:gn1]
                    acc = jnp.dot(
                        u_fp8_scratch_ref[:, :] if fp8_unpool else u_scratch_ref[:, :],
                        w_cache_ref[s, :, cn0:cn1],
                        preferred_element_type=jnp.float32,
                    )
                    sc_prod = (u_sc * w_sc_sub) if fp8_unpool else w_sc_sub
                    y_sub = (acc * sc_prod).astype(jnp.bfloat16).astype(jnp.float32)
                fw_sub = y_sub * vecs_ref[1:2, gn0:gn1]
                new_stream_out_ref[:, gn0:gn1] = fw_sub
                sq_ffn_128 = sq_ffn_128 + vpu_tree_sum_128(fw_sub * fw_sub)
                if d_fuse_p34:
                    a_smid = vecs_ref[2:3, gn0:gn1] * stream_mid_ref[:, gn0:gn1]
                    b_fw = vecs_ref[3:4, gn0:gn1] * fw_sub
                    s_aa_128 = s_aa_128 + vpu_tree_sum_128(a_smid * a_smid)
                    s_ab_128 = s_ab_128 + vpu_tree_sum_128(a_smid * b_fw)
                    s_bb_128 = s_bb_128 + vpu_tree_sum_128(b_fw * b_fw)
            if eff_salu_ovlp:
                _start_step_range(
                    next_s_k,
                    1 - slot,
                    s * t_per_slab_k,
                    (s + 1) * t_per_slab_k,
                )

        inv_rms_ffn = lax.rsqrt(
            jnp.sum(sq_ffn_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(post_norm_eps)
        )

        # Phase 3: Algebraic residual-norm variance (`D_OPT & 2`) or full Phase 3 pass.
        if d_fuse_p34:
            sum_aa = jnp.sum(s_aa_128, axis=1, keepdims=True)
            sum_ab = jnp.sum(s_ab_128, axis=1, keepdims=True)
            sum_bb = jnp.sum(s_bb_128, axis=1, keepdims=True)
            sq_next_row = jnp.maximum(
                jnp.float32(0.0),
                sum_aa
                + jnp.float32(2.0) * inv_rms_ffn * sum_ab
                + (inv_rms_ffn * inv_rms_ffn) * sum_bb,
            )
            inv_rms_next = lax.rsqrt(
                sq_next_row * jnp.float32(inv_h) + jnp.float32(rms_eps)
            )
        else:
            p_step_3 = 256 if d_fast_dma else slab_w
            sq_next_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            amax_next_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            for c0 in range(0, h_dim, p_step_3):
                c1 = c0 + p_step_3
                fw_s = new_stream_out_ref[:, c0:c1]
                branch_s = (fw_s * inv_rms_ffn).astype(jnp.bfloat16).astype(jnp.float32)
                new_s = (
                    vecs_ref[2:3, c0:c1] * stream_mid_ref[:, c0:c1]
                    + vecs_ref[3:4, c0:c1] * branch_s
                )
                new_stream_out_ref[:, c0:c1] = new_s
                sq_next_128 = sq_next_128 + vpu_tree_sum_128(new_s * new_s)
                if emit_fp8_attn_in:
                    unscaled_next = new_s * vecs_ref[4:5, c0:c1]
                    amax_next_128 = jnp.maximum(
                        amax_next_128, vpu_tree_max_128(jnp.abs(unscaled_next))
                    )

            inv_rms_next = lax.rsqrt(
                jnp.sum(sq_next_128, axis=1, keepdims=True) * jnp.float32(inv_h)
                + jnp.float32(rms_eps)
            )

        # If in-kernel push-AG is active, drain previous slot or launch Part-0 remote push at step == steps_ag!
        if use_push_ag:

            @pl.when((step >= 2) & (step < steps_ag))
            def _wait_ag_part0_slot():
                pltpu.make_async_copy(
                    ag_p0_fp8_vmem.at[pl.ds(slot, 1), :, :],
                    gathered_hbm_ref.at[
                        pl.ds(0, 1), pl.ds(0, tok_tile_eff), :
                    ],
                    ag_local_sems.at[slot],
                ).wait()

            @pl.when(step == steps_ag)
            def _launch_push_ag():
                for s_i in range(min(2, steps_ag)):
                    pltpu.make_async_copy(
                        ag_p0_fp8_vmem.at[pl.ds(s_i, 1), :, :],
                        gathered_hbm_ref.at[
                            pl.ds(0, 1), pl.ds(0, tok_tile_eff), :
                        ],
                        ag_local_sems.at[s_i],
                    ).wait()
                if ag_tp_deg > 1:
                    for d in range(1, ag_tp_deg):
                        peer = (
                            lax.bitwise_xor(me, d)
                            if (ag_tp_deg & (ag_tp_deg - 1)) == 0
                            else lax.rem(me + d, ag_tp_deg)
                        )
                        pltpu.make_async_remote_copy(
                            src_ref=gathered_hbm_ref.at[
                                pl.ds(me, 1), pl.ds(0, in_kernel_ag_rows), :
                            ],
                            dst_ref=gathered_hbm_ref.at[
                                pl.ds(me, 1), pl.ds(0, in_kernel_ag_rows), :
                            ],
                            send_sem=ag_ssems.at[d],
                            recv_sem=ag_rsems.at[d],
                            device_id={ag_axis_name: peer},
                        ).start()

            @pl.when(step >= steps_ag + 2)
            def _wait_rest_slot():
                pltpu.make_async_copy(
                    ag_tile_vmem.at[pl.ds(slot, 1), :, :].reshape(
                        tok_tile_eff, next_in_cols
                    ),
                    next_in_out_ref.at[pl.ds(0, tok_tile_eff), :],
                    ag_local_sems.at[slot],
                ).wait()

            dst_next_ref = ag_tile_vmem
        else:
            dst_next_ref = next_in_out_ref

        def _write_next_slab(c0: int, c1: int, val: jax.Array):
            if use_push_ag:
                if ag_part0_is_s32_fp8:

                    @pl.when(step < steps_ag)
                    def _w_p0(c0=c0, c1=c1, val=val):
                        ag_p0_fp8_vmem[slot, :, c0:c1] = jnp.clip(
                            val.astype(jnp.float32) * jnp.float32(32.0),
                            -_FP8_MAX,
                            _FP8_MAX,
                        ).astype(jnp.float8_e4m3fn)

                    @pl.when(step >= steps_ag)
                    def _w_rest(c0=c0, c1=c1, val=val):
                        dst_next_ref[slot, :, c0:c1] = val

                else:
                    dst_next_ref[slot, :, c0:c1] = val
            else:
                dst_next_ref[:, c0:c1] = val

        if d_fuse_p34:
            inv_rms_next_eff = (
                inv_rms_next * jnp.float32(32.0)
                if emit_static_fp8_attn_in
                else inv_rms_next
            )
            out_dt = jnp.float8_e4m3fn if emit_static_fp8_attn_in else jnp.bfloat16
            for c0 in range(0, h_dim, 256):
                c1 = c0 + 256
                fw_s = new_stream_out_ref[:, c0:c1]
                branch_s = (fw_s * inv_rms_ffn).astype(jnp.bfloat16).astype(jnp.float32)
                new_s = (
                    vecs_ref[2:3, c0:c1] * stream_mid_ref[:, c0:c1]
                    + vecs_ref[3:4, c0:c1] * branch_s
                )
                new_stream_out_ref[:, c0:c1] = new_s
                scaled_next = new_s * inv_rms_next_eff * vecs_ref[4:5, c0:c1]
                if emit_static_fp8_attn_in:
                    scaled_next = jnp.clip(scaled_next, -_FP8_MAX, _FP8_MAX)
                _write_next_slab(c0, c1, scaled_next.astype(out_dt))
        elif emit_fp8_attn_in:
            row_amax_unscaled = jnp.max(amax_next_128, axis=1, keepdims=True)
            inv_q_sc = jnp.where(
                row_amax_unscaled == 0,
                jnp.zeros_like(row_amax_unscaled),
                jnp.float32(_FP8_MAX) / jnp.maximum(row_amax_unscaled, jnp.float32(1e-30)),
            )
            eff_row_sc = (row_amax_unscaled * inv_rms_next) * jnp.float32(1.0 / _FP8_MAX)
            s_scaled = jnp.clip(eff_row_sc * jnp.float32(8.0), 0.0, _FP8_MAX)
            s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
            s_hi_f32 = s_hi_f8.astype(jnp.float32)
            s_lo_f8 = jnp.clip(
                (s_scaled - s_hi_f32) * jnp.float32(16.0), -_FP8_MAX, _FP8_MAX
            ).astype(jnp.float8_e4m3fn)
            lane = lax.broadcasted_iota(jnp.int32, (tok_tile_eff, LANE), 1)
            _write_next_slab(
                h_dim,
                h_dim + LANE,
                jnp.where(
                    lane < 64,
                    jnp.broadcast_to(s_hi_f8, (tok_tile_eff, LANE)),
                    jnp.broadcast_to(s_lo_f8, (tok_tile_eff, LANE)),
                ),
            )
            for s in range(num_slabs):
                c0, c1 = s * slab_w, (s + 1) * slab_w
                new_s = new_stream_out_ref[:, c0:c1]
                _write_next_slab(
                    c0,
                    c1,
                    jnp.clip(
                        new_s * vecs_ref[4:5, c0:c1] * inv_q_sc,
                        -_FP8_MAX,
                        _FP8_MAX,
                    ).astype(jnp.float8_e4m3fn),
                )
        elif emit_static_fp8_attn_in:
            inv_rms_next_s32 = inv_rms_next * jnp.float32(32.0)
            for s in range(num_slabs):
                c0, c1 = s * slab_w, (s + 1) * slab_w
                new_s = new_stream_out_ref[:, c0:c1]
                _write_next_slab(
                    c0,
                    c1,
                    jnp.clip(
                        new_s * inv_rms_next_s32 * vecs_ref[4:5, c0:c1],
                        -_FP8_MAX,
                        _FP8_MAX,
                    ).astype(jnp.float8_e4m3fn),
                )
        else:
            for s in range(num_slabs):
                c0, c1 = s * slab_w, (s + 1) * slab_w
                new_s = new_stream_out_ref[:, c0:c1]
                _write_next_slab(
                    c0,
                    c1,
                    (new_s * inv_rms_next * vecs_ref[4:5, c0:c1]).astype(jnp.bfloat16),
                )

        if use_push_ag:

            @pl.when(step < steps_ag)
            def _dma_part0_local():
                pltpu.make_async_copy(
                    ag_p0_fp8_vmem.at[pl.ds(slot, 1), :, :],
                    gathered_hbm_ref.at[
                        pl.ds(me, 1), pl.ds(step * tok_tile_eff, tok_tile_eff), :
                    ],
                    ag_local_sems.at[slot],
                ).start()

            @pl.when(step >= steps_ag)
            def _dma_rest_local():
                r_st = step - steps_ag
                pltpu.make_async_copy(
                    ag_tile_vmem.at[pl.ds(slot, 1), :, :].reshape(
                        tok_tile_eff, next_in_cols
                    ),
                    next_in_out_ref.at[pl.ds(r_st * tok_tile_eff, tok_tile_eff), :],
                    ag_local_sems.at[slot],
                ).start()

            @pl.when(step == num_steps - 1)
            def _drain_push_ag():
                n_rest_steps = num_steps - steps_ag
                for k_i in range(min(2, n_rest_steps)):
                    s_i = (steps_ag + k_i) % 2
                    pltpu.make_async_copy(
                        ag_tile_vmem.at[pl.ds(s_i, 1), :, :].reshape(
                            tok_tile_eff, next_in_cols
                        ),
                        next_in_out_ref.at[pl.ds(0, tok_tile_eff), :],
                        ag_local_sems.at[s_i],
                    ).wait()
                if ag_tp_deg > 1:
                    for d in range(1, ag_tp_deg):
                        pltpu.make_async_copy(
                            gathered_hbm_ref.at[
                                pl.ds(0, 1), pl.ds(0, in_kernel_ag_rows), :
                            ],
                            gathered_hbm_ref.at[
                                pl.ds(0, 1), pl.ds(0, in_kernel_ag_rows), :
                            ],
                            ag_rsems.at[d],
                        ).wait()
                        pltpu.make_async_copy(
                            gathered_hbm_ref.at[
                                pl.ds(0, 1), pl.ds(0, in_kernel_ag_rows), :
                            ],
                            gathered_hbm_ref.at[
                                pl.ds(0, 1), pl.ds(0, in_kernel_ag_rows), :
                            ],
                            ag_ssems.at[d],
                        ).wait()

        if eff_salu_ovlp:

            @pl.when(step == num_steps - 1)
            def _drain_extra_pf():
                _wait_step(1 - slot)

    u_fp8_rows = tok_tile_eff if fp8_unpool else 8
    any_spec = pl.BlockSpec(memory_space=getattr(pltpu, "ANY", pltpu.HBM))
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    in_specs = [
        hbm_spec,
        hbm_spec,
        w_in_spec,
        pl.BlockSpec((tok_tile_eff, h_dim), tok_map),
        pl.BlockSpec((8, h_dim), lambda i, *_: (0, 0)),
    ]
    call_args = [
        scalar_pref,
        out_buf_3d,
        wunpool,
        w_in_arg,
        stream_shard_mid,
        vecs_8x8192,
    ]
    if use_w4:
        in_specs.append(
            pl.BlockSpec(
                (num_slabs, he_dim // w4_unpool_qbs, slab_w), lambda i, *_: (0, 0, 0)
            )
        )
        call_args.append(w4_sc_3d)

    aliases: dict[int, int] = {}
    if stream_out is not None:
        aliases = {len(call_args): 0}
        in_specs.append(hbm_spec)
        call_args.append(stream_out)

    if use_push_ag:
        out_shape = (
            jax.ShapeDtypeStruct((chunk_size, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((rest_rows, next_in_cols), next_in_dtype),
            jax.ShapeDtypeStruct(
                (ag_tp_deg, in_kernel_ag_rows, next_in_cols), gathered_dtype
            ),
        )
        out_specs = [
            pl.BlockSpec((tok_tile_eff, h_dim), tok_map),
            hbm_spec,
            any_spec,
        ]
    else:
        out_shape = (
            jax.ShapeDtypeStruct((chunk_size, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((num_rows, next_in_cols), next_in_dtype),
        )
        out_specs = [
            pl.BlockSpec((tok_tile_eff, h_dim), tok_map),
            pl.BlockSpec((tok_tile_eff, next_in_cols), lambda i, *_: (i, 0)),
        ]

    w_cache_dtype = jnp.float4_e2m1fn if use_w4 else jnp.float8_e4m3fn
    in_buf_dtype = out_buf_3d.dtype
    scratches = [
        pltpu.VMEM((2, n_pairs, n2_cols, lanes), in_buf_dtype),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((tok_tile_eff, he_dim), jnp.bfloat16),
        pltpu.VMEM((u_fp8_rows, he_dim), jnp.float8_e4m3fn),
        pltpu.VMEM((num_slabs, he_dim, slab_w), w_cache_dtype),
        pltpu.SemaphoreType.DMA((num_slabs,)),
    ]
    if use_push_ag:
        scratches.extend(
            [
                pltpu.VMEM((2, tok_tile_eff, next_in_cols), next_in_dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((ag_tp_deg,)),
                pltpu.SemaphoreType.DMA((ag_tp_deg,)),
            ]
        )
        if ag_part0_is_s32_fp8:
            scratches.append(
                pltpu.VMEM((2, tok_tile_eff, next_in_cols), jnp.float8_e4m3fn)
            )

    name = (
        f"pallas_d_rad_4slab_t{tok_tile_eff}"
        + ("_coal" if coal_gather else "")
        + (f"_pag{in_kernel_ag_rows}" if use_push_ag else "")
        + (f"_w4q{w4_unpool_qbs}" if use_w4 else "")
        + (f"_fp8q{eff_fp8_qbs}" if eff_fp8_qbs > 0 and not use_w4 else "")
        + (f"_do{eff_d_opt}" if eff_d_opt else "")
    )
    cp_kw = dict(
        disable_bounds_checks=True,
        vmem_limit_bytes=62 * 1024 * 1024,
    )
    if use_push_ag:
        cp_kw["dimension_semantics"] = ("arbitrary",)
        cp_kw["collective_id"] = 33 if ag_tp_deg > 1 else None

    return pl.pallas_call(
        _d_radical_kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_steps,),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratches,
        ),
        compiler_params=pltpu.CompilerParams(**cp_kw),
        input_output_aliases=aliases,
        name=name,
    )(*call_args)
