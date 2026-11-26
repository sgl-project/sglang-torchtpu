# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill kernels in the serving model (SGLANG_TPU_MUSE_KERNEL_ENABLED).

On a pure EXTEND batch the decoder layers with NVFP4 experts run sequence parallel:
each TP rank keeps the fp32 residual stream of its ``T / tp`` tokens, and per layer
runs Kernel A1 (all-gather + qkvg + QK norm + RoPE), the model's own RadixAttention
(it writes the KV pools and runs RPA), Kernel A3 (gate + o_proj + reduce-scatter) and
the MoE bridge pipeline (B1 + B2 + all-gather + routing + fused-RS GMM + D). The first
and last layers (FP8 experts) and every other batch keep the native path; the stream
is sliced before the run and all-gathered after it.

Kernels: A1 = kernel_a1.pallas_qkvg_qk_rope_kv after a BF16 all-gather (a1_v1), A3 =
kernel_a3.a3_v3_2half_overlap_rs (BF16 activations), MoE = pipeline's BF16 track.
SGLANG_TPU_MUSE_KERNEL_A1_VARIANT / _A3_VARIANT select the two-half A1 (v5 bf16,
v6 static fp8) and the Option 3D A3 (v4_3d bf16, v16 static fp8, replicated Wo).
Dense weights become per-output-channel fp8 copies; pre/post_expert_proj are gathered
to replicated copies (Option 3A). They are prepared before the KV pool is sized.

SGLANG_TPU_MUSE_KERNEL_DEBUG_COMPARE also runs every kernel layer natively on the same
input and logs per-layer cosine / max-abs / rel-L2 of the output stream (TP rank 0).
"""

import functools
import inspect
import logging
import math
import threading
import time
from typing import NamedTuple, Optional

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)


def _env_bool(name: str) -> bool:
    field = getattr(tpu_envs, name, None)
    return bool(field is not None and field.get())

_AXIS = "model"
# B2's two halves and A3's two halves tile T / tp in multiples of 256 rows.
_ROWS_PER_RANK_ALIGN = 256
_HEAD_DIM = 64
_FP8_MAX = 448.0


class _Ops(NamedTuple):
    gather: object
    a1: object
    a3: object
    moe: object
    moe_fp8: object = None
    sum_stream: object = None
    # SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D: A1 fed by the previous layer's D-side gather of
    # its first part, and the MoE ops that emit that gather.
    a1_agd: object = None
    moe_agd: object = None
    moe_fp8_agd: object = None
    a1_owner: object = None
    # SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM: per-chunk AG + A1 (+ optional A3 prologue) ops.
    ag_chunk: object = None
    a1_chunk: object = None
    a1_chunk_agd: object = None
    a3_prologue_chunk: object = None
    a3_wo: object = None
    a3_wo_2chunk: object = None
    a3_wo_3chunk: object = None
    gather_fp8_stream: object = None
    # Wave-8 CCS Mode-2 K-deep A3 ops (per-chunk prologue+A2A + fused last-chunk K-deep Wo).
    a3_pro_a2a_chunk: object = None
    a3_pro_last_chunk: object = None
    a3_kdeep_ccs: object = None
    # Wave-9 Static-FP8 Ulysses sequence-parallel ops (`A1_VARIANT="v10_sp"`, TP=8 / TP=16).
    a1_ulysses_qkv_chunk: object = None
    a1_ulysses_a2a_pack_chunk: object = None
    a1_ulysses_gate_chunk: object = None
    a3_ulysses_rev_a2a_chunk: object = None
    a3_ulysses_gate_fuse_chunk: object = None
    a3_ulysses_wo: object = None
    a3_ulysses_wo_2chunk: object = None


_HYBRID_A1_VARIANTS = (
    "v6_bf16kv",
    "v6_s8row_bf16kv",
    "v6_s8grp128_bf16kv",
    "v6_s8str128_bf16kv",
)
_A1_VARIANTS = ("v1", "v5", "v6", "v5x4", "v5x8", "v10_sp", *_HYBRID_A1_VARIANTS)
_A3_VARIANTS = ("v3", "v4_3d", "v16")


def _resolve_a1_chunks(
    t_shard: int, n_parts: int, a1_chunks: str
) -> tuple[bool, tuple[int, ...]]:
    """Returns `(local_first, chunk_sizes)` summing to `t_shard` (32-row aligned)."""
    if not a1_chunks:
        h_sub = t_shard // n_parts
        return False, (h_sub,) * n_parts
    spec = a1_chunks.strip()
    local_first = spec.startswith("local+")
    if local_first:
        spec = spec[len("local+") :]
    raw = tuple(int(x) for x in spec.split(",") if x.strip())
    raw_sum = sum(raw)
    if t_shard == raw_sum:
        return local_first, raw
    if t_shard > raw_sum and t_shard % raw_sum == 0:
        scale = t_shard // raw_sum
        return local_first, tuple(c * scale for c in raw)
    # Fallback for shorter buckets (e.g. T=256 -> t_shard=32, T=2048 -> t_shard=256, T=6144 -> t_shard=768):
    # split into valid >=64-row (or 32-row when t_shard % 64 != 0) multiples summing to t_shard.
    unit = 64 if (t_shard >= 64 and t_shard % 64 == 0) else 32
    k = min(len(raw), max(1, t_shard // unit))
    base = (t_shard // k // unit) * unit
    rem = t_shard - base * k
    chunks = [base + (unit if i < rem // unit else 0) for i in range(k)]
    return local_first if k > 1 else False, tuple(chunks)


def _a1_body(
    h_q_local: int,
    h_kv_local: int,
    variant: str,
    rpa_layout: bool = False,
    a1_chunks: str = "",
    a1_1way: bool = False,
    a1_zc: bool = False,
    skip_pad_tokens: bool = False,
    parts: tuple[int, ...] = (),
    ag_under_d: bool = False,
):
    """Per-rank A1: bf16 attn_in ``[T / tp, H]`` -> q, gate ``[T, q_size]``, fp8 kv ``[T, 2 * kv_size]``.

    ``rpa_layout`` (two-half variants): q is RPA v3's packed-KV-head ``[1, T, Hq / 2, 2, 128]``
    and kv its merged fp8 ``[T, 1, 4, 128]`` (kernel_a1 ``rpa_layout``).
    ``parts`` (bf16 two-half body): uneven part row weights (``_a1_part_rows``).
    ``ag_under_d``: the body takes ``(ag0, attn_in_rest, ...)``, part 0 already all-gathered
    (``[tp, rows0, H]``, by the previous layer's MoE op) and this rank's remaining rows.
    """
    import jax.numpy as jnp
    from jax import lax

    from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather
    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

    eff_a1_zc = a1_zc or skip_pad_tokens or (variant == "v6_s8str128_bf16kv")
    bridge_ag_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG.get()
    chunk_ag_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_CHUNK.get()
    a1_ag_mode = (
        chunk_ag_mode
        if chunk_ag_mode > 0
        else (bridge_ag_mode if bridge_ag_mode in (1, 2, 3) else 0)
    )

    def _ag(x_in, *, tiled: bool):
        if a1_ag_mode >= 1:
            return xor1_bridge_all_gather(
                x_in, axis_name=_AXIS, tiled=tiled, xy_torus=(a1_ag_mode >= 3)
            )
        return lax.all_gather(x_in, _AXIS, axis=0, tiled=tiled)

    def a1_body(attn_in_shard, wqkvg, wqkvg_scale, cos, sin):
        x_full = _ag(attn_in_shard, tiled=True)
        return kernel_a1.pallas_qkvg_qk_rope_kv(
            x_full,
            wqkvg,
            wqkvg_scale,
            cos,
            sin,
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
        )

    def body_1way_zc(attn_in_shard, wqkvg, wqkvg_scale, cos, sin, *opt_valid):
        t_shard = attn_in_shard.shape[0]
        n_parts = int(variant.split("x")[1]) if "x" in variant else 2
        tp = cos.shape[0] // t_shard
        eff_chunks_spec = a1_chunks or (
            "384,256,192,128,64" if variant == "v5x4" else ""
        )
        _, chunks = _resolve_a1_chunks(t_shard, n_parts, eff_chunks_spec)
        vtok = opt_valid[0] if skip_pad_tokens else None
        ag = _ag(attn_in_shard[: chunks[0]], tiled=True)
        outs = None
        row_off = 0
        for i, c_sz in enumerate(chunks):
            if i + 1 < len(chunks):
                next_off = row_off + c_sz
                next_sz = chunks[i + 1]
                x_next, ag = lax.optimization_barrier(
                    (attn_in_shard[next_off : next_off + next_sz], ag)
                )
                ag_next = _ag(x_next, tiled=True)
            outs = kernel_a1.pallas_a1_chunk_1way_zc(
                ag,
                wqkvg,
                wqkvg_scale,
                cos,
                sin,
                outs,
                valid_tokens_1d=vtok,
                tp_deg=tp,
                t_shard=t_shard,
                row_off=row_off,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
            )
            row_off += c_sz
            if i + 1 < len(chunks):
                ag, *outs = lax.optimization_barrier((ag_next, *outs))
                outs = tuple(outs)
        q, gate, kv = outs
        return (q[None], gate, kv) if rpa_layout else (q, gate, kv)

    def body_1way_chunks(attn_in_shard, wqkvg, wqkvg_scale, cos, sin, ag0=None):
        lead = 0 if ag0 is None else ag0.shape[1]
        t_shard = lead + attn_in_shard.shape[0]
        n_parts = int(variant.split("x")[1]) if "x" in variant else 2
        tp = cos.shape[0] // t_shard
        use_a1_dyn = (
            variant == "v6"
            and not ag_under_d
            and ag0 is None
            and _env_bool("SGLANG_TPU_MUSE_PREFILL_A1_W8A8_DYN")
        )
        x, w_scale = attn_in_shard, wqkvg_scale
        if use_a1_dyn:
            x = kernel_a1.quant_a1_dyn_fp8_packed(attn_in_shard)
            w_scale = wqkvg_scale
        elif variant == "v6":
            if x.dtype != jnp.float8_e4m3fn:
                x = (
                    jnp.clip(
                        x.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                        -_FP8_MAX,
                        _FP8_MAX,
                    ).astype(jnp.float8_e4m3fn)
                    if x.shape[0] > 0
                    else x.astype(jnp.float8_e4m3fn)
                )
            w_scale = wqkvg_scale * kernel_a1.STATIC_A1_ACT_SCALE
        eff_chunks = a1_chunks
        if not eff_chunks and variant == "v6" and ag0 is None and not ag_under_d:
            fast_ch_env = getattr(
                tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_CHUNKS", None
            )
            fast_ch_raw = (
                fast_ch_env.get().strip().lower() if fast_ch_env is not None else ""
            )
            if fast_ch_raw in ("1", "true", "512", "local+512,512"):
                eff_chunks = "local+512,512"
            elif fast_ch_raw in ("1024", "1chunk", "local+1024"):
                eff_chunks = "local+1024"
            elif fast_ch_raw in ("all512", "512,512"):
                eff_chunks = "512,512"
            elif fast_ch_raw not in ("", "0", "false", "off"):
                eff_chunks = fast_ch_env.get().strip()
            elif use_a1_dyn:
                eff_chunks = "local+512,512"
        local_first, chunks = _resolve_a1_chunks(t_shard, n_parts, eff_chunks)
        cos_3d = cos.reshape(tp, t_shard, cos.shape[1])
        sin_3d = sin.reshape(tp, t_shard, sin.shape[1])
        if ag0 is not None:
            assert not local_first and chunks[0] == lead, (local_first, chunks, lead)
            ag = ag0
        else:
            ag = _ag(x[: chunks[0]], tiled=False)
        if local_first:
            outs = kernel_a1.pallas_a1_chunk_1way(
                x[None],
                wqkvg,
                w_scale,
                cos_3d,
                sin_3d,
                None,
                t_shard=t_shard,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
                mode="local",
                tp_axis=_AXIS,
            )
            ag, *outs = lax.optimization_barrier((ag, *outs))
            outs = tuple(outs)
            chunk_mode = "remote"
        else:
            outs = None
            chunk_mode = "all"

        row_off = 0
        for i, c_sz in enumerate(chunks):
            if i + 1 < len(chunks):
                next_off = row_off + c_sz
                next_sz = chunks[i + 1]
                x_off = next_off - lead
                x_next, ag = lax.optimization_barrier(
                    (x[x_off : x_off + next_sz], ag)
                )
                ag_next = _ag(x_next, tiled=False)
            outs = kernel_a1.pallas_a1_chunk_1way(
                ag,
                wqkvg,
                w_scale,
                cos_3d[:, row_off : row_off + c_sz, :],
                sin_3d[:, row_off : row_off + c_sz, :],
                outs,
                t_shard=t_shard,
                row_off=row_off,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
                mode=chunk_mode,
                tp_axis=_AXIS,
            )
            row_off += c_sz
            if i + 1 < len(chunks):
                ag, *outs = lax.optimization_barrier((ag_next, *outs))
                outs = tuple(outs)

        q, gate, kv = (o.reshape(tp * t_shard, *o.shape[2:]) for o in outs)
        if rpa_layout:
            return q[None], gate, kv[:, None]
        return q, gate, kv

    def body_hybrid_bf16kv_zc(
        attn_in_shard,
        wqg,
        wqg_scale,
        wkv_2d,
        wkv_scale_2d,
        a1_gain,
        a1_inv_gain,
        cos,
        sin,
        *opt_valid,
    ):
        """Zero-Copy Hybrid `Q+Gate` (`s8_strided128` / `s8row` / outlier-clamped `fp8` 2D tiled SC-AG) + Fused `kv_local` `s8` Pre-Quant + Optional SMEM Pad-Token Skip."""
        t_shard, hidden = attn_in_shard.shape
        tp = cos.shape[0] // t_shard
        wkv_3d = wkv_2d.reshape(tp, hidden, -1)
        wkv_sc_3d = wkv_scale_2d[:, None, :]
        is_s8 = variant in (
            "v6_s8row_bf16kv",
            "v6_s8grp128_bf16kv",
            "v6_s8str128_bf16kv",
        )
        strided_128 = variant != "v6_s8row_bf16kv"
        fast_ch_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_CHUNKS", None)
        fast_ch_raw = (
            fast_ch_env.get().strip().lower() if fast_ch_env is not None else ""
        )
        fast_default_chunks = ""
        if is_s8 and not a1_chunks and fast_ch_raw not in ("", "0", "false", "off"):
            if fast_ch_raw in ("1", "true", "512", "local+512,512"):
                fast_default_chunks = "local+512,512"
            elif fast_ch_raw in ("1024", "1chunk", "local+1024"):
                fast_default_chunks = "local+1024"
            elif fast_ch_raw in ("all512", "512,512"):
                fast_default_chunks = "512,512"
            else:
                fast_default_chunks = fast_ch_env.get().strip()
        default_chunks = (
            fast_default_chunks
            if fast_default_chunks
            else ("local+384,256,256,128" if is_s8 else "512,256,256")
        )
        local_first, chunks = _resolve_a1_chunks(
            t_shard, 4 if is_s8 else 3, a1_chunks or default_chunks
        )
        vtok = opt_valid[0] if skip_pad_tokens else None
        wqg_chunk_sc = wqg_scale if is_s8 else wqg_scale * kernel_a1.STATIC_A1_ACT_SCALE
        need_x_s8 = is_s8 and (len(chunks) > 1)

        if is_s8:
            q0 = kernel_a1.pallas_quant_chunk0_s8_unsliced(
                attn_in_shard,
                a1_inv_gain,
                c0_rows=chunks[0],
                strided_128=strided_128,
            )
            ag = _ag(q0, tiled=True)
            kv_tm = (
                (64 if any(c % 128 != 0 for c in chunks) else 128)
                if need_x_s8
                else (256 if t_shard % 256 == 0 else 128)
            )
            kv_local, x_s8_all = kernel_a1.pallas_sp_kv_local_bf16_fkvq(
                attn_in_shard,
                wkv_3d,
                wkv_sc_3d,
                cos,
                sin,
                a1_inv_gain if need_x_s8 else None,
                quant_s8=need_x_s8,
                strided_128=strided_128,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
                tp_axis=_AXIS,
                tm=kv_tm,
            )
        else:
            q0 = jnp.clip(
                attn_in_shard[: chunks[0]].astype(jnp.float32) * a1_inv_gain,
                -_FP8_MAX,
                _FP8_MAX,
            ).astype(jnp.float8_e4m3fn)
            ag = _ag(q0, tiled=True)
            kv_local, x_s8_all = kernel_a1.pallas_sp_kv_local_bf16_fkvq(
                attn_in_shard,
                wkv_3d,
                wkv_sc_3d,
                cos,
                sin,
                None,
                quant_s8=False,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
                tp_axis=_AXIS,
                tm=256,
            )

        if local_first:
            x_local = attn_in_shard if is_s8 else (attn_in_shard * a1_gain)
            qg_outs, _ = kernel_a1.pallas_a1_qg_1way_fkv_zc(
                x_local,
                None,
                wqg,
                wqg_scale,
                cos,
                sin,
                None,
                s8_packed=False,
                valid_tokens_1d=vtok,
                tp_deg=tp,
                t_shard=t_shard,
                row_off=0,
                h_q_local=h_q_local,
                rpa_layout=rpa_layout,
                mode="local",
                tp_axis=_AXIS,
            )
            if need_x_s8:
                ag, kv_local, x_s8_all, *qg_outs = lax.optimization_barrier(
                    (ag, kv_local, x_s8_all, *qg_outs)
                )
            else:
                ag, kv_local, *qg_outs = lax.optimization_barrier(
                    (ag, kv_local, *qg_outs)
                )
            qg_outs = tuple(qg_outs)
            chunk_mode = "remote"
        else:
            if need_x_s8:
                ag, kv_local, x_s8_all = lax.optimization_barrier(
                    (ag, kv_local, x_s8_all)
                )
            else:
                ag, kv_local = lax.optimization_barrier((ag, kv_local))
            qg_outs = None
            chunk_mode = "all"

        kv_recv = None
        row_off = 0
        for i, c_sz in enumerate(chunks):
            if i + 1 < len(chunks):
                next_off = row_off + c_sz
                next_sz = chunks[i + 1]
                if is_s8:
                    s8_slice, ag = lax.optimization_barrier(
                        (x_s8_all[next_off : next_off + next_sz], ag)
                    )
                    ag_next = _ag(s8_slice, tiled=True)
                else:
                    x_slice, ag = lax.optimization_barrier(
                        (attn_in_shard[next_off : next_off + next_sz], ag)
                    )
                    ag_next = _ag(
                        jnp.clip(
                            x_slice.astype(jnp.float32) * a1_inv_gain,
                            -_FP8_MAX,
                            _FP8_MAX,
                        ).astype(jnp.float8_e4m3fn),
                        tiled=True,
                    )
            qg_outs, kv_out = kernel_a1.pallas_a1_qg_1way_fkv_zc(
                ag,
                a1_gain if is_s8 else None,
                wqg,
                wqg_chunk_sc,
                cos,
                sin,
                qg_outs,
                kv_local=kv_local if i == 0 else None,
                s8_packed=is_s8,
                strided_128=strided_128,
                valid_tokens_1d=vtok,
                tp_deg=tp,
                t_shard=t_shard,
                row_off=row_off,
                h_q_local=h_q_local,
                rpa_layout=rpa_layout,
                mode=chunk_mode,
                tp_axis=_AXIS,
            )
            if i == 0:
                kv_recv = kv_out
            row_off += c_sz
            if i + 1 < len(chunks):
                ag, *qg_outs = lax.optimization_barrier((ag_next, *qg_outs))
                qg_outs = tuple(qg_outs)

        q, gate = qg_outs
        kv = kv_recv.reshape(tp * t_shard, *kv_recv.shape[2:])
        return (q[None], gate, kv) if rpa_layout else (q, gate, kv)

    def body_hybrid_bf16kv(
        attn_in_shard,
        wqg,
        wqg_scale,
        wkv_2d,
        wkv_scale_2d,
        a1_gain,
        a1_inv_gain,
        cos,
        sin,
    ):
        """Hybrid `Q+Gate` (`s8grp128` / `s8row` / outlier-clamped `fp8` SC-AG) + Exact Local BF16 `KV` + Fused Chunk-0 `KV` P2P A2A."""
        t_shard, hidden = attn_in_shard.shape
        tp = cos.shape[0] // t_shard
        wkv_3d = wkv_2d.reshape(tp, hidden, -1)
        wkv_sc_3d = wkv_scale_2d[:, None, :]
        is_s8 = variant in ("v6_s8row_bf16kv", "v6_s8grp128_bf16kv")
        per_group_128 = variant == "v6_s8grp128_bf16kv"
        default_chunks = "local+256,256,256,256" if is_s8 else "512,256,256"
        local_first, chunks = _resolve_a1_chunks(
            t_shard, 4 if is_s8 else 3, a1_chunks or default_chunks
        )
        cos_3d = cos.reshape(tp, t_shard, cos.shape[1])
        sin_3d = sin.reshape(tp, t_shard, sin.shape[1])
        me = kernel_a1.pallas_axis_index_1d(_AXIS)[0]
        cos_own = lax.dynamic_index_in_dim(cos_3d, me, axis=0, keepdims=False)
        sin_own = lax.dynamic_index_in_dim(sin_3d, me, axis=0, keepdims=False)

        def _quant_chunk(xc):
            if is_s8:
                return kernel_a1.quant_chunk_s8_packed(
                    xc, a1_inv_gain, per_group_128=per_group_128
                )
            return jnp.clip(
                xc.astype(jnp.float32) * a1_inv_gain, -_FP8_MAX, _FP8_MAX
            ).astype(jnp.float8_e4m3fn)

        wqg_chunk_sc = wqg_scale if is_s8 else wqg_scale * kernel_a1.STATIC_A1_ACT_SCALE
        ag = _ag(_quant_chunk(attn_in_shard[: chunks[0]]), tiled=False)
        kv_local = kernel_a1.pallas_sp_kv_local_bf16_rpal(
            attn_in_shard,
            wkv_3d,
            wkv_sc_3d,
            cos_own,
            sin_own,
            h_kv_local=h_kv_local,
            rpa_layout=rpa_layout,
        )
        if local_first:
            x_local = attn_in_shard if is_s8 else (attn_in_shard * a1_gain)
            qg_outs = kernel_a1.pallas_a1_chunk_1way(
                x_local[None],
                wqg,
                wqg_scale,
                cos_3d,
                sin_3d,
                None,
                t_shard=t_shard,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
                mode="local",
                tp_axis=_AXIS,
                q_gate_only=True,
            )
            ag, kv_local, *qg_outs = lax.optimization_barrier((ag, kv_local, *qg_outs))
            qg_outs = tuple(qg_outs)
            chunk_mode = "remote"
        else:
            ag, kv_local = lax.optimization_barrier((ag, kv_local))
            qg_outs = None
            chunk_mode = "all"

        kv_recv = None
        row_off = 0
        for i, c_sz in enumerate(chunks):
            if i + 1 < len(chunks):
                next_off = row_off + c_sz
                next_sz = chunks[i + 1]
                x_slice, ag = lax.optimization_barrier(
                    (attn_in_shard[next_off : next_off + next_sz], ag)
                )
                ag_next = _ag(_quant_chunk(x_slice), tiled=False)
            qg_outs, kv_out = kernel_a1.pallas_a1_qg_1way_fkv(
                ag,
                a1_gain if is_s8 else None,
                wqg,
                wqg_chunk_sc,
                cos_3d[:, row_off : row_off + c_sz, :],
                sin_3d[:, row_off : row_off + c_sz, :],
                qg_outs,
                kv_local=kv_local if i == 0 else None,
                s8_packed=is_s8,
                per_group_128=per_group_128,
                t_shard=t_shard,
                row_off=row_off,
                h_q_local=h_q_local,
                rpa_layout=rpa_layout,
                mode=chunk_mode,
                tp_axis=_AXIS,
            )
            if i == 0:
                kv_recv = kv_out
            row_off += c_sz
            if i + 1 < len(chunks):
                ag, *qg_outs = lax.optimization_barrier((ag_next, *qg_outs))
                qg_outs = tuple(qg_outs)

        q, gate = (o.reshape(tp * t_shard, *o.shape[2:]) for o in qg_outs)
        kv = kv_recv.reshape(tp * t_shard, *kv_recv.shape[2:])
        if rpa_layout:
            return q[None], gate, kv[:, None]
        return q, gate, kv

    def body_two_half(attn_in_shard, wqkvg, wqkvg_scale, cos, sin, ag0=None):
        # ag0 (ag_under_d): attn_in_shard holds this rank's rows after part 0.
        lead = 0 if ag0 is None else ag0.shape[1]
        t_shard = lead + attn_in_shard.shape[0]
        n_parts = int(variant.split("x")[1]) if "x" in variant else 2
        h_sub = t_shard // n_parts
        tp = cos.shape[0] // t_shard
        rows = _a1_part_rows(t_shard, parts, n_default_parts=n_parts)
        if (
            rows
        ):  # uneven parts: one (tp, 1, t_shard) buffer, each part at its row offset
            n_parts, h_sub = 1, t_shard
        else:
            rows = [h_sub] * n_parts
        x, w_scale = attn_in_shard, wqkvg_scale
        into_half = kernel_a1.pallas_qkvg_qk_rope_kv_into_half
        if variant == "v6":
            if (
                not ag_under_d
                and ag0 is None
                and _env_bool("SGLANG_TPU_MUSE_PREFILL_A1_W8A8_DYN")
            ):
                x = kernel_a1.quant_a1_dyn_fp8_packed(attn_in_shard)
                w_scale = wqkvg_scale
            else:
                # Static x32 fp8 of the normed input clips |x| > 14; 1/32 folds into w_scale.
                if x.dtype != jnp.float8_e4m3fn:
                    x = jnp.clip(
                        x.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                        -_FP8_MAX,
                        _FP8_MAX,
                    ).astype(jnp.float8_e4m3fn)
                w_scale = wqkvg_scale * kernel_a1.STATIC_A1_ACT_SCALE
            into_half = kernel_a1.pallas_qkvg_qk_rope_kv_into_half_1way
        # Row (r, half, i) of the 4D tables and outputs is token r * t_shard + half * h_sub + i.
        cos_4d = cos.reshape(tp, n_parts, h_sub, cos.shape[1])
        sin_4d = sin.reshape(tp, n_parts, h_sub, sin.shape[1])
        q_size, kv_size = h_q_local * 64, h_kv_local * 64
        q_cols = (h_q_local // 2, 2, 128) if rpa_layout else (q_size,)
        kv_cols = (4, 128) if rpa_layout else (2 * kv_size,)
        outs = (
            jnp.empty((tp, n_parts, h_sub, *q_cols), jnp.bfloat16),
            jnp.empty((tp, n_parts, h_sub, q_size), jnp.bfloat16),
            jnp.empty((tp, n_parts, h_sub, *kv_cols), jnp.float8_e4m3fn),
        )
        kernel_kw = dict(
            h_q_local=h_q_local,
            h_kv_local=h_kv_local,
            # The kernels' grid is h_sub // tm: tm must divide h_sub (a multiple of 128).
            tm=256 if h_sub % 256 == 0 else 128,
        )
        if variant != "v6":
            kernel_kw["rpa_layout"] = rpa_layout
        offs = [sum(rows[:i]) for i in range(len(rows))]
        if ag0 is not None:
            assert rows[0] == lead, (rows, lead)
            xs = [None] + [
                x[o - lead : o - lead + r] for o, r in zip(offs[1:], rows[1:])
            ]
        else:
            xs = [x[o : o + r] for o, r in zip(offs, rows)]

        def kernel(i, ag, outs):
            kw = dict(kernel_kw, half_idx=i)
            if n_parts == 1:  # uneven: tm must divide the part and its offset
                tm = 256 if rows[i] % 256 == 0 and offs[i] % 256 == 0 else 128
                kw.update(half_idx=0, tm=tm, row_block_off=offs[i] // tm)
            return into_half(ag, wqkvg, w_scale, cos_4d, sin_4d, *outs, **kw)

        # Part i + 1's all-gather is issued before part i's kernel so they can overlap.
        ag = _ag(xs[0], tiled=False) if ag0 is None else ag0
        for i in range(len(rows)):
            if i + 1 < len(rows):
                x_next, ag = lax.optimization_barrier((xs[i + 1], ag))
                ag_next = _ag(x_next, tiled=False)
            outs = kernel(i, ag, outs)
            if i + 1 < len(rows):
                ag, *outs = lax.optimization_barrier((ag_next, *outs))
        q, gate, kv = (o.reshape(tp * t_shard, *o.shape[3:]) for o in outs)
        if rpa_layout:
            return q[None], gate, kv[:, None]
        return q, gate, kv

    if rpa_layout and variant == "v1":
        raise ValueError(f"A1 RPA layout needs a multi-part A1 variant, not {variant}")
    bf16_two_half = variant in ("v5", "v5x4", "v5x8") and not (
        eff_a1_zc or a1_1way or a1_chunks
    )
    v6_agd_ok = (
        variant == "v6"
        and not eff_a1_zc
        and not skip_pad_tokens
        and not a1_chunks.strip().startswith("local+")
    )
    if parts and not (bf16_two_half or (variant == "v6" and not (eff_a1_zc or a1_1way or a1_chunks))):
        raise ValueError(
            "A1 PARTS need two-half A1 v5 / v5x4 / v5x8 / v6 without A1_ZC, A1_1WAY, A1_CHUNKS "
            f"or SKIP_PAD_ATTN_TOKENS (variant {variant})"
        )
    if ag_under_d and not (bf16_two_half or v6_agd_ok):
        raise ValueError(
            "A1 AG_UNDER_D needs A1 v5 / v5x4 / v5x8 (two-half) or A1 v6 (non-local+) without "
            f"A1_ZC or SKIP_PAD_ATTN_TOKENS (variant {variant})"
        )
    if variant == "v1":
        return a1_body
    if variant in _HYBRID_A1_VARIANTS:
        return body_hybrid_bf16kv_zc if eff_a1_zc else body_hybrid_bf16kv
    if eff_a1_zc and variant != "v6":
        return body_1way_zc
    if a1_1way or a1_chunks or (rpa_layout and variant == "v6"):
        if ag_under_d:
            return lambda ag0, rest, *a: body_1way_chunks(rest, *a, ag0=ag0)
        return body_1way_chunks
    if ag_under_d:
        return lambda ag0, rest, *a: body_two_half(rest, *a, ag0=ag0)
    return body_two_half


def _a1_part_rows(
    t_shard: int, parts: tuple[int, ...], *, n_default_parts: int = 2
) -> list[int]:
    """Rows per rank of each A1 part for weights ``parts`` (e.g. 2,3,2,1), or [] when the
    shard does not split into 128-row multiples (the variant's equal parts then run)."""
    if parts and t_shard % (sum(parts) * 128) == 0:
        return [t_shard // sum(parts) * w for w in parts]
    finer_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS", None)
    if finer_env is not None and finer_env.get() and t_shard >= 128 and t_shard % 128 == 0:
        n_blocks = t_shard // 128
        if parts:
            eff_w = parts[:n_blocks]
        elif t_shard % (n_default_parts * 128) != 0:
            eff_w = (1,) * min(n_default_parts, n_blocks)
        else:
            return []
        k = len(eff_w)
        if k <= 1:
            return []
        w_sum = sum(eff_w)
        rem_blocks = n_blocks - k
        raw_extra = [rem_blocks * w / w_sum for w in eff_w]
        blocks = [1 + int(math.floor(x)) for x in raw_extra]
        leftover = n_blocks - sum(blocks)
        order = sorted(
            range(k), key=lambda i: (raw_extra[i] - math.floor(raw_extra[i]), eff_w[i], -i), reverse=True
        )
        for idx in order[:leftover]:
            blocks[idx] += 1
        return [b * 128 for b in blocks]
    return []


def _a1_first_rows(
    t_shard: int,
    variant: str,
    parts: tuple[int, ...],
    *,
    a1_chunks: str = "",
    a1_1way: bool = False,
    rpa_layout: bool = False,
) -> int:
    """Rows per rank of A1 part 0 (`body_two_half` or `body_1way_chunks`)."""
    n_parts = int(variant.split("x")[1]) if "x" in variant else 2
    if a1_1way or a1_chunks or (rpa_layout and variant == "v6"):
        local_first, chunks = _resolve_a1_chunks(t_shard, n_parts, a1_chunks)
        if local_first:
            raise ValueError("A1_AG_UNDER_D does not support local+ A1_CHUNKS")
        return chunks[0]
    rows = _a1_part_rows(t_shard, parts, n_default_parts=n_parts)
    return rows[0] if rows else t_shard // n_parts


def _a3_body(
    variant: str,
    tp_size: int,
    head_norm_eps: Optional[float] = None,
    rpa_packed_out: bool = False,
    prologue_tm: int = 512,
    opt_wo: bool = False,
    skip_pad_tokens: bool = False,
):
    """Per-rank A3: attn, gate ``[T, q_size]`` (this rank's heads) -> ``[T / tp, H]`` (its tokens).

    ``head_norm_eps`` set: ``attn`` is the raw attention output and A3 applies the per-head
    RMS normalization itself (in the Pallas prologue for v3 / v16).
    ``rpa_packed_out`` (v16): ``attn`` is RPA v3's KV-packed output and A3's prologue diagonally
    selects each head's own 64 lanes directly in VREGs.
    """
    import jax.numpy as jnp
    from jax import lax

    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a3

    def body(attn, gate, wo, wo_scale, *opt_valid):
        rows = gate.shape[0] // tp_size
        vtok = opt_valid[0] if (skip_pad_tokens and opt_valid) else None
        if variant == "v3":
            if head_norm_eps is None:
                return kernel_a3.a3_v3_2half_overlap_rs(
                    attn, gate, wo, wo_scale, tp_axis=_AXIS
                )
            # Each half reads rows // 2 rows per rank in place: tm must divide them.
            return kernel_a3.a3_v3_2half_overlap_rs(
                attn,
                gate,
                wo,
                wo_scale,
                tp_axis=_AXIS,
                tm=256 if rows // 2 % 256 == 0 else 128,
                head_norm_eps=head_norm_eps,
            )
        if variant == "v4_3d":
            if head_norm_eps is not None:
                heads = attn.reshape(attn.shape[0], -1, _HEAD_DIM).astype(jnp.float32)
                ms = jnp.mean(heads * heads, axis=-1, keepdims=True)
                heads = heads * lax.rsqrt(ms + head_norm_eps)
                attn = heads.astype(jnp.bfloat16).reshape(attn.shape)
            return kernel_a3.a3_v4_option3d_a2a_sp(
                attn,
                gate,
                wo,
                wo_scale,
                tp_axis=_AXIS,
                tm=512 if rows % 512 == 0 else 256,
            )
        w4_wo_qbs = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get() or 0)
        a3_kdeep = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get() or 0)
        if (a3_kdeep >= 1 and vtok is None) or (
            w4_wo_qbs > 0 and jnp.dtype(wo.dtype) == jnp.float4_e2m1fn
        ):
            from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
                a3_v16_kdeep_sp,
            )

            eff_w4_wo = (
                w4_wo_qbs
                if (w4_wo_qbs > 0 and jnp.dtype(wo.dtype) == jnp.float4_e2m1fn)
                else 0
            )
            eff_wo_sc = (
                wo_scale.squeeze(0)
                if (eff_w4_wo > 0 and wo_scale.ndim == 4)
                else wo_scale
            )
            return a3_v16_kdeep_sp(
                attn,
                gate,
                wo,
                eff_wo_sc,
                tp_axis=_AXIS,
                head_norm_eps=head_norm_eps,
                rpa_packed_out=rpa_packed_out,
                prologue_tm=1024 if opt_wo else prologue_tm,
                valid_tokens_1d=vtok,
                w4_qbs=eff_w4_wo,
            )
        return kernel_a3.a3_v16_option3d_zero_copy9_persrc_sem_async_a2a_wo_3d_sp(
            attn,
            gate,
            wo,
            wo_scale,
            tp_axis=_AXIS,
            head_norm_eps=head_norm_eps,
            rpa_packed_out=rpa_packed_out,
            prologue_tm=prologue_tm,
            opt_wo=opt_wo,
            valid_tokens_1d=vtok,
        )

    return body


@functools.cache
def _build_entry_op(
    org_vocab_size: int, start_is_zero: bool, async_a2a: bool = False
):
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang_tpu.muse_prefill_kernels.attn_kernels import entry_exit
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh()

    def _entry_body(input_ids: jax.Array, embed_col: jax.Array) -> jax.Array:
        return entry_exit.sp_entry_col_a2a(
            input_ids,
            embed_col,
            org_vocab_size=org_vocab_size,
            start_is_zero=start_is_zero,
            tp_axis=_AXIS,
            async_a2a=async_a2a,
        )

    fn = jax.shard_map(
        _entry_body,
        mesh=mesh,
        in_specs=(P(_AXIS), P(_AXIS)),
        out_specs=P(_AXIS),
        check_vma=False,
    )
    mode_tag = ("s0" if start_is_zero else "rep") + ("_aa2a" if async_a2a else "")

    def entry(input_ids: jax.Array, embed_col: jax.Array) -> jax.Array:
        return fn(input_ids, embed_col)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::muse_prefill_sp_entry_{mode_tag}_v{org_vocab_size}",
        entry,
        mesh=mesh,
        input_partition_specs=(P(_AXIS), P(_AXIS)),
        output_partition_specs=(P(_AXIS),),
    )


@functools.cache
def _build_rope_op(tp_size: int):
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang_tpu.muse_prefill_kernels.attn_kernels import entry_exit
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh()

    def _rope_body(
        positions: jax.Array, cos_sin_128: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return entry_exit.sharded_rope_gather_128(
            positions,
            cos_sin_128,
            tp_axis=_AXIS,
            tp_size=tp_size,
            pos_pre_sliced=True,
        )

    fn = jax.shard_map(
        _rope_body,
        mesh=mesh,
        in_specs=(P(_AXIS), P(_AXIS)),
        out_specs=(P(_AXIS), P(_AXIS)),
        check_vma=False,
    )

    def rope(
        positions: jax.Array, cos_sin_128: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return fn(positions, cos_sin_128)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::muse_prefill_sharded_rope_tp{tp_size}",
        rope,
        mesh=mesh,
        input_partition_specs=(P(_AXIS), P(_AXIS)),
        output_partition_specs=(P(_AXIS), P(_AXIS)),
    )


def _w8_carb_cache_tag() -> str:
    return (
        f"sn{tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_SUB_N.get()}"
        f"_fn{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM.get())}"
        f"_bn2k{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_BN2048.get())}"
        f"_kd{tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get()}"
        f"_dc{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER.get())}"
        f"_diag{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG.get())}"
        f"_w4o{tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get()}"
        f"_w4u{tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get()}"
        f"_rnp{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK.get())}"
        f"_cls{tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get()}"
        f"_a32c{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_CCS_2CHUNK.get())}"
        f"_pcmr{int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE.get())}"
    )


def _ccs_wo_uses_kdeep_nvfp4() -> bool:
    if int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get() or 0) > 0:
        return True
    ccs_pro_env = getattr(
        tpu_envs, "SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE", None
    )
    return not (
        tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM.get()
        and ccs_pro_env is not None
        and bool(ccs_pro_env.get())
        and int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get() or 0) == 0
    )


def _a3_supports(variant: str, rows_per_rank: int) -> bool:
    if variant != "v16":
        return True
    if rows_per_rank <= 1024:
        return True
    return rows_per_rank <= 4096 and rows_per_rank % 512 == 0


@functools.cache
def _ops(
    h_q_local: int,
    h_kv_local: int,
    a1_variant: str,
    a3_variant: str,
    fused_glue: bool,
    head_norm_eps: float,
    a1_rpa_layout: bool = False,
    a3_rpa_out: bool = False,
    a1_chunks: str = "",
    a1_1way: bool = False,
    packed_sorts: bool = False,
    router_hilo_pre: bool = False,
    gmm_layout_deq: bool = False,
    fp8_layers: bool = False,
    d_tok_tile: int = 64,
    d_zero_spill: bool = False,
    d_prepack_vecs: bool = False,
    sortfree_prefill: bool = False,
    sortfree_mode: int = 0,
    dequant_1pass: bool = False,
    b2_pack_bf16_sc: bool = False,
    exit_owner_gather: bool = False,
    a3_opt: bool = False,
    b2_fp8_lhs: bool = False,
    d_fp8_unpool: int = 0,
    a1_zc: bool = False,
    skip_pad_tokens: bool = False,
    a1_parts: tuple[int, ...] = (),
    a1_ag_under_d: bool = False,
    tp16_owner_a1: bool = False,
    ep_tp2_mode: bool = False,
    causal_chunk_stream: bool = False,
    causal_chunk_spec: str = "",
    l61_prune: bool = False,
    a3_stage_sends: int = 0,
    ccs_chunk_a3_prologue: bool = False,
    d_opt: int = 0,
    d_mxu_combine: bool = False,
    b1_b2_prepack: bool = False,
    b1_b2_fast: bool = False,
    zero_sort_routing: bool = False,
    fp8_stream_gather: bool = False,
    ccs_fuse_ag_a1: int = 0,
    overlap_deq0_routing_ag: bool = False,
    carb_tag: str = "",
    ulysses_rev_bf16: bool = False,
) -> _Ops:
    """Per-rank torch ops: every tensor is this rank's block (P("model") on dim 0)."""
    import jax
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh()
    if mesh.shape[_AXIS] == 16 and ccs_fuse_ag_a1 not in (0, 3):
        raise ValueError(
            "SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1 not in (0, 3) is disabled on TP=16 until #2833 is root-caused"
        )
    tag = f"q{h_q_local}_kv{h_kv_local}"
    glue = "_fg" if fused_glue else ""
    a1_skip = skip_pad_tokens and a1_variant not in ("v1", "v6")
    a3_skip = skip_pad_tokens and a3_variant == "v16"

    def shard(body, n_in: int, n_out: int):
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(_AXIS),) * n_in,
            out_specs=(P(_AXIS),) * n_out if n_out > 1 else P(_AXIS),
            check_vma=False,
        )

    def export(name: str, fn, n_in: int, n_out: int):
        return cpu_export_jax_op(
            f"sglang_tpu_pallas::muse_prefill_{name}_{tag}",
            fn,
            mesh=mesh,
            input_partition_specs=(P(_AXIS),) * n_in,
            output_partition_specs=(P(_AXIS),) * n_out,
        )

    from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather

    xor1_ag_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG.get()
    xor1_moe_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE.get()
    xor1_chunk_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_CHUNK.get()
    xor1_b2_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_B2.get()
    xor1_under_d_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_UNDER_D.get()
    exit_ag_mode = (
        xor1_b2_mode
        if xor1_b2_mode > 0
        else (xor1_ag_mode if xor1_ag_mode in (1, 2, 3) else 0)
    )
    gather_sm = shard(
        (
            lambda x: xor1_bridge_all_gather(
                x, axis_name=_AXIS, tiled=True, xy_torus=(exit_ag_mode >= 3)
            )
            if exit_ag_mode >= 1
            else lax.all_gather(x, _AXIS, axis=0, tiled=True)
        ),
        1,
        1,
    )

    def gather(x: jax.Array) -> jax.Array:
        return gather_sm(x)

    ccs_tag = (
        "_ccs"
        + (
            "_"
            + causal_chunk_spec.replace(";", "__").replace(":", "_").replace(",", "_")
            if causal_chunk_spec
            else ""
        )
        if causal_chunk_stream
        else ""
    )
    gather_fp8_op = None
    if fp8_stream_gather:
        import jax.numpy as jnp
        from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
            resolve_causal_chunks,
            unpermute_from_chunk_major_jax,
        )

        tp_sz_ag = mesh.shape[_AXIS]

        def gather_fp8_body(x: jax.Array) -> jax.Array:
            out_dt = x.dtype
            t_sh, h_sz = x.shape
            xf = x.astype(jnp.float32)
            row_amax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
            row_sc = jnp.where(
                row_amax == 0,
                jnp.bfloat16(0.0),
                (row_amax * jnp.float32(1.0 / _FP8_MAX)).astype(jnp.bfloat16),
            )
            sc_f32 = row_sc.astype(jnp.float32)
            inv_sc = jnp.where(
                sc_f32 == 0,
                jnp.float32(0.0),
                jnp.float32(1.0) / jnp.maximum(sc_f32, jnp.float32(1e-30)),
            )
            q_u8 = lax.bitcast_convert_type(
                jnp.clip(xf * inv_sc, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn),
                jnp.uint8,
            )
            sc_u16 = lax.bitcast_convert_type(row_sc, jnp.uint16).astype(jnp.uint32)
            lo_u8 = (sc_u16 & jnp.uint32(0xFF)).astype(jnp.uint8)
            hi_u8 = lax.shift_right_logical(sc_u16, jnp.uint32(8)).astype(jnp.uint8)
            lane_ag = jnp.arange(128, dtype=jnp.int32)[None, :]
            sc_u8 = jnp.where(
                lane_ag < 64,
                jnp.broadcast_to(lo_u8, (t_sh, 128)),
                jnp.broadcast_to(hi_u8, (t_sh, 128)),
            )
            packed_u8 = jnp.concatenate([q_u8, sc_u8], axis=1)
            ag_u8 = (
                xor1_bridge_all_gather(
                    packed_u8,
                    axis_name=_AXIS,
                    tiled=True,
                    xy_torus=(exit_ag_mode >= 3),
                )
                if exit_ag_mode >= 1
                else lax.all_gather(packed_u8, _AXIS, axis=0, tiled=True)
            )
            if causal_chunk_stream:
                ag_chunks = resolve_causal_chunks(t_sh, causal_chunk_spec)
                if len(ag_chunks) > 1:
                    ag_u8 = unpermute_from_chunk_major_jax(ag_u8, tp_sz_ag, ag_chunks)
            q_full = lax.bitcast_convert_type(ag_u8[:, :h_sz], jnp.float8_e4m3fn)
            ps = ag_u8[:, h_sz : h_sz + 128]
            lo16 = ps[:, 0:1].astype(jnp.uint32)
            hi16 = lax.shift_left(ps[:, 64:65].astype(jnp.uint32), jnp.uint32(8))
            sc_full = lax.bitcast_convert_type(
                (lo16 | hi16).astype(jnp.uint16), jnp.bfloat16
            )
            return (q_full.astype(jnp.bfloat16) * sc_full).astype(out_dt)

        gather_fp8_sm = shard(gather_fp8_body, 1, 1)

        def gather_fp8(x: jax.Array) -> jax.Array:
            return gather_fp8_sm(x)

        gather_fp8_op = export(
            "gather_fp8"
            + ccs_tag
            + (f"_xag{exit_ag_mode}" if exit_ag_mode > 0 else ""),
            gather_fp8,
            1,
            1,
        )

    sum_sm = shard(lambda x: lax.psum(x, _AXIS), 1, 1)

    def sum_stream(x: jax.Array) -> jax.Array:
        return sum_sm(x)

    a1_body = _a1_body(
        h_q_local,
        h_kv_local,
        a1_variant,
        rpa_layout=a1_rpa_layout,
        a1_chunks=a1_chunks,
        a1_1way=a1_1way,
        a1_zc=a1_zc,
        skip_pad_tokens=a1_skip,
        parts=a1_parts,
    )
    a1_n_in = (9 if a1_variant in _HYBRID_A1_VARIANTS else 5) + (1 if a1_skip else 0)
    a1_sm = shard(a1_body, a1_n_in, 3)

    def fixed_arity(fn, n_in: int, ret):
        # torch.library.custom_op rejects *args: give the wrapper one named Array param per input.
        params = [
            inspect.Parameter(
                f"x{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
            )
            for i in range(n_in)
        ]
        fn.__signature__ = inspect.Signature(params, return_annotation=ret)
        return fn

    def a1(*args: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return a1_sm(*args)

    fixed_arity(a1, a1_n_in, tuple[jax.Array, jax.Array, jax.Array])

    a3_body = _a3_body(
        a3_variant,
        mesh.shape[_AXIS],
        head_norm_eps if (fused_glue or a3_rpa_out) else None,
        rpa_packed_out=a3_rpa_out,
        opt_wo=a3_opt,
        skip_pad_tokens=a3_skip,
    )
    a3_n_in = 4 + (1 if a3_skip else 0)
    a3_sm = shard(a3_body, a3_n_in, 1)

    def a3(*args: jax.Array) -> jax.Array:
        return a3_sm(*args)

    fixed_arity(a3, a3_n_in, jax.Array)

    def moe_body(*xs, next_in_gather_rows=0):
        # The pipeline's BF16 track (handoff: cosine 1.0 vs its reference).
        outs = fused_moe_full_bridge_and_ffn(
            *xs,
            mesh=mesh,
            router_mode="bf16_hilo_pre" if router_hilo_pre else "bf16_hilo",
            b2_layout="nstream",
            b2_two_half=True,
            b2_deq0_first=True,
            hoist_routing=True,
            d_layout="4slab",
            d_tok_tile=d_tok_tile,
            in_shard_map=True,
            b2_deq0_new_buffer=fused_glue,
            next_in_gather_rows=next_in_gather_rows,
            next_in_gather_static_fp8=(a1_variant in ("v6", "v10_sp")),
            ep_tp2_mode=ep_tp2_mode,
            causal_chunk_stream=causal_chunk_stream,
            causal_chunk_spec=causal_chunk_spec,
        )
        # AG_UNDER_D: (stream, the next layer's gathered A1 part 0, its other rows).
        return (outs[0], outs[4], outs[1]) if next_in_gather_rows else outs[:2]

    moe_sm = shard(moe_body, 22, 2)

    def moe(
        attn_out_shard: jax.Array,
        stream_shard: jax.Array,
        alpha_attn: jax.Array,
        beta_attn: jax.Array,
        pre_ffn_norm_w: jax.Array,
        wrouter: jax.Array,
        e_score_bias: jax.Array,
        wpool: jax.Array,
        wpool_scale: jax.Array,
        pre_expert_norm_w: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        post_expert_norm_w: jax.Array,
        group_offset: jax.Array,
        wunpool: jax.Array,
        wunpool_scale: jax.Array,
        post_ffn_norm_w: jax.Array,
        alpha_ffn: jax.Array,
        beta_ffn: jax.Array,
        next_input_norm_w: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return moe_sm(
            attn_out_shard,
            stream_shard,
            alpha_attn,
            beta_attn,
            pre_ffn_norm_w,
            wrouter,
            e_score_bias,
            wpool,
            wpool_scale,
            pre_expert_norm_w,
            w1,
            w2,
            w1_scale,
            w2_scale,
            post_expert_norm_w,
            group_offset,
            wunpool,
            wunpool_scale,
            post_ffn_norm_w,
            alpha_ffn,
            beta_ffn,
            next_input_norm_w,
        )

    a1_chunks_tag = (
        f"_ch_{a1_chunks.replace('+', 'p').replace(',', '_')}" if a1_chunks else ""
    )
    a1_parts_tag = "_p" + "".join(map(str, a1_parts)) if a1_parts else ""
    d_t64_c16 = tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_T64_C16.get()
    short_pt = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.get()
    sf_scatter = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER.get()
    fp4_qbs = tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS.get()
    strip_n = tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N.get()
    kp_fp8 = tpu_envs.SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_FP8.get()
    kp_nc = tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_NC.get()
    b2_1w = tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE.get()
    finer_bkt = _env_bool("SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS")
    a3_zsp = _env_bool("SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE")
    d_t128_fast = _env_bool("SGLANG_TPU_MUSE_KERNEL_D_T128_FAST")
    a1_fast_ch_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A1_FAST_CHUNKS", None)
    a1_fast_ch = (
        a1_fast_ch_env.get().strip().replace("+", "p").replace(",", "_")
        if a1_fast_ch_env is not None
        else ""
    )
    moe_suffix = (
        glue
        + ("_tp2" if ep_tp2_mode else "")
        + ("_fbk" if finer_bkt else "")
        + ("_ps" if packed_sorts else "")
        + (
            f"_sf{sortfree_mode}"
            if sortfree_prefill and sortfree_mode
            else ("_sf" if sortfree_prefill else "")
        )
        + (f"_bs{sf_scatter}" if sortfree_prefill and sf_scatter else "")
        + ("_rhp" if router_hilo_pre else "")
        + ("_gl" if gmm_layout_deq and fused_glue else "")
        + ("_d1p" if dequant_1pass and gmm_layout_deq and fused_glue else "")
        + ("_bpsc" if b2_pack_bf16_sc and gmm_layout_deq and fused_glue else "")
        + (f"_dt{d_tok_tile}" if d_tok_tile != 64 else "")
        + ("_dt128f" if d_t128_fast else "")
        + ("_dzs" if d_zero_spill else "")
        + ("_dpv" if d_prepack_vecs else "")
        + ("_b2f8" if b2_fp8_lhs else "")
        + ("_b21w" if b2_1w else "")
        + (f"_df8q{d_fp8_unpool}" if d_fp8_unpool > 0 else "")
        + ("_dc16" if d_t64_c16 else "")
        + ("_spt" if short_pt else "")
        + (f"_fq4_{fp4_qbs}" if fp4_qbs != 0 else "")
        + (f"_sn{strip_n}" if strip_n > 0 else "")
        + (f"_kf{kp_fp8}" if kp_fp8 > 0 else "")
        + (f"_knc{kp_nc}" if kp_nc > 0 else "")
        + (
            f"_xagm{xor1_moe_mode}"
            if xor1_moe_mode > 0
            else (f"_xag{xor1_ag_mode}" if xor1_ag_mode >= 2 else "")
        )
        + ("_xagd" if xor1_under_d_mode else "")
        + (
            f"_fb12_{tpu_envs.SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2.get()
            else ""
        )
        + (
            f"_facc{tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get()
            else ""
        )
        + (
            f"_rb{tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get()
            else ""
        )
        + (
            f"_psf{tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get()
            else ""
        )
        + (
            f"_flq{tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get()
            else ""
        )
        + (f"_do{d_opt}" if d_opt else "")
        + ("_dmxu" if d_mxu_combine else "")
        + ("_b12f" if b1_b2_fast else "")
        + ("_b12p" if b1_b2_prepack and not b1_b2_fast else "")
        + ("_zsr" if zero_sort_routing else "")
        + ("_odeq0" if overlap_deq0_routing_ag else "")
        + (
            "_dcoal"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER.get()
            else ""
        )
        + (
            "_dikag"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG.get()
            else ""
        )
        + (
            f"_w4u{tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get()
            else ""
        )
        + (
            "_rnp"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK.get()
            else ""
        )
        + (
            f"_cls{tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get()}"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get()
            else ""
        )
        + (
            "_pcmr"
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE.get()
            else ""
        )
        + ("_f8ni" if a1_variant == "v10_sp" and d_opt > 0 else "")
    )
    fp8_tiles_tag = tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES.get().replace(
        ",", "_"
    )
    fp8_qbs = tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP8_LHS_QBS.get()
    fp8_extra = (f"_ft{fp8_tiles_tag}" if fp8_tiles_tag else "") + (
        f"_fq{fp8_qbs}" if fp8_qbs != 0 else ""
    )
    a1_dyn = (
        a1_variant == "v6"
        and not a1_ag_under_d
        and _env_bool("SGLANG_TPU_MUSE_PREFILL_A1_W8A8_DYN")
    )
    a1_sub_n = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_SUB_N.get() or 0)
    a1_fnorm = _env_bool("SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM")
    a1_w8_tag = (f"_sn{a1_sub_n}" if a1_sub_n > 0 else "") + (
        "_fn" if a1_fnorm else ""
    )
    a1_name = (
        ("a1" if a1_variant == "v1" else f"a1_{a1_variant}")
        + ("_dyn" if a1_dyn else "")
        + ("_rpal" if a1_rpa_layout else "")
        + ("_1w" if a1_1way else "")
        + a1_chunks_tag
        + (f"_fch_{a1_fast_ch}" if a1_fast_ch and not a1_chunks else "")
        + ("_zc" if a1_zc else "")
        + ("_sk" if a1_skip else "")
        + ("_fbk" if finer_bkt else "")
        + a1_parts_tag
        + (
            f"_xagc{xor1_chunk_mode}"
            if xor1_chunk_mode > 0
            else (
                f"_xag{xor1_ag_mode}" if xor1_ag_mode in (1, 2, 3) else ""
            )
        )
        + a1_w8_tag
    )
    agd_ops = {}
    if (a1_ag_under_d or causal_chunk_stream) and a1_variant != "v10_sp":
        if causal_chunk_stream:
            from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                resolve_causal_chunks,
            )

            _first_rows_fn = lambda ts: resolve_causal_chunks(ts, causal_chunk_spec)[0]
        else:
            _first_rows_fn = functools.partial(
                _a1_first_rows,
                variant=a1_variant,
                parts=a1_parts,
                a1_chunks=a1_chunks,
                a1_1way=a1_1way,
                rpa_layout=a1_rpa_layout,
            )
        agd_ops = _agd_ops(
            a1_body=_a1_body(
                h_q_local,
                h_kv_local,
                a1_variant,
                rpa_layout=a1_rpa_layout,
                a1_chunks=a1_chunks,
                a1_1way=a1_1way,
                parts=a1_parts,
                ag_under_d=True,
            ),
            moe_body=moe_body,
            first_rows=_first_rows_fn,
            shard=shard,
            export=export,
            fixed_arity=fixed_arity,
            a1_name=a1_name,
            moe_name=(
                "moe"
                + moe_suffix
                + f"_agd_{a1_variant}"
                + a1_parts_tag
                + a1_chunks_tag
                + ccs_tag
            ),
            fp8_moe_name=(
                "moe_fp8"
                + moe_suffix
                + fp8_extra
                + f"_agd_{a1_variant}"
                + a1_parts_tag
                + a1_chunks_tag
                + ccs_tag
                if fp8_layers
                else None
            ),
        )
    ag_chunk_op = None
    a1_chunk_op = None
    a1_chunk_agd_op = None
    a1_ulysses_qkv_chunk_op = None
    a1_ulysses_a2a_pack_chunk_op = None
    a1_ulysses_gate_chunk_op = None
    a3_ulysses_rev_a2a_chunk_op = None
    a3_ulysses_gate_fuse_chunk_op = None
    a3_ulysses_wo_op = None
    a3_ulysses_wo_2chunk_op = None
    if causal_chunk_stream and a1_variant == "v10_sp":
        import jax.numpy as jnp
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1, kernel_a3

        tp_d_ul = mesh.shape[_AXIS]

        def _prep_ul_fp8_slice(x_slice):
            if x_slice.dtype != jnp.float8_e4m3fn:
                return (
                    jnp.clip(
                        x_slice.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                        -_FP8_MAX,
                        _FP8_MAX,
                    ).astype(jnp.float8_e4m3fn)
                    if x_slice.shape[0] > 0
                    else x_slice.astype(jnp.float8_e4m3fn)
                )
            return x_slice

        def a1_ul_qkv_body(attn_in_k, wqkvg, wqkvg_sc3d, cos_k, sin_k):
            c_k = attn_in_k.shape[0]
            # Avoid outer stablehlo.partition_id; TorchTPU GSPMD rejects it in cpu_export_jax_op.
            me = kernel_a1.pallas_axis_index_1d(_AXIS)[0]
            x_fp8_k = _prep_ul_fp8_slice(attn_in_k)
            cos_own = lax.dynamic_index_in_dim(
                cos_k.reshape(tp_d_ul, c_k, 128), me, axis=0, keepdims=False
            )
            sin_own = lax.dynamic_index_in_dim(
                sin_k.reshape(tp_d_ul, c_k, 128), me, axis=0, keepdims=False
            )
            q_3d, kv_3d = kernel_a1.pallas_a1_ulysses_qkv_local(
                x_fp8_k,
                wqkvg,
                wqkvg_sc3d * kernel_a1.STATIC_A1_ACT_SCALE,
                cos_own,
                sin_own,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
            )
            return q_3d, kv_3d, x_fp8_k

        def a1_ul_a2a_pack_body(q_3d, kv_3d):
            tp_d_p, c_k, q_cols = q_3d.shape
            c_tok = tp_d_p * c_k
            q_2d = lax.all_to_all(
                q_3d, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(c_tok, q_cols)
            kv_2d = lax.all_to_all(
                kv_3d, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(c_tok, kv_3d.shape[-1])
            return kernel_a1.pallas_pack_ulysses_qkv_for_rpa(
                q_2d, kv_2d, h_q_local=h_q_local, h_kv_local=h_kv_local, tm=512
            )

        def a1_ul_gate_body(x_k, w_gate, w_gate_sc):
            x_fp8_k = _prep_ul_fp8_slice(x_k)
            return kernel_a1.pallas_ulysses_local_gate(
                x_fp8_k,
                w_gate,
                w_gate_sc * kernel_a1.STATIC_A1_ACT_SCALE,
            )

        def a3_ul_rev_a2a_body(raw_attn_k):
            c_tok, q_loc = raw_attn_k.shape
            c_k = c_tok // tp_d_ul
            q_3d = kernel_a3.pallas_ulysses_norm_fp8_for_rev_a2a(
                raw_attn_k,
                head_norm_eps=head_norm_eps,
                quantize_fp8=not ulysses_rev_bf16,
            ).reshape(tp_d_ul, c_k, q_loc)
            return lax.all_to_all(
                q_3d, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(tp_d_ul, c_k, q_loc)

        def a3_ul_gate_fuse_body(attn_3d, gate_k):
            return kernel_a3.pallas_ulysses_gate_fuse(
                attn_3d, gate_k, input_is_prescaled=not ulysses_rev_bf16
            )

        def a3_ul_wo_body(q_fp8, wo_in, wo_sc):
            return kernel_a3.pallas_ulysses_wo_local(q_fp8, wo_in, wo_sc)

        def a3_ul_wo_2chunk_body(q0_fp8, q1_fp8, wo_in, wo_sc):
            return kernel_a3.pallas_ulysses_wo_local((q0_fp8, q1_fp8), wo_in, wo_sc)

        a1_ul_qkv_sm = shard(a1_ul_qkv_body, 5, 3)
        a1_ul_a2a_pack_sm = shard(a1_ul_a2a_pack_body, 2, 2)
        a1_ul_gate_sm = shard(a1_ul_gate_body, 3, 1)
        a3_ul_rev_a2a_sm = shard(a3_ul_rev_a2a_body, 1, 1)
        a3_ul_gate_fuse_sm = shard(a3_ul_gate_fuse_body, 2, 1)
        a3_ul_wo_sm = shard(a3_ul_wo_body, 3, 1)
        a3_ul_wo_2c_sm = shard(a3_ul_wo_2chunk_body, 4, 1)

        def a1_ul_qkv(
            attn_in_k: jax.Array,
            wqkvg: jax.Array,
            wqkvg_sc3d: jax.Array,
            cos_k: jax.Array,
            sin_k: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return a1_ul_qkv_sm(attn_in_k, wqkvg, wqkvg_sc3d, cos_k, sin_k)

        def a1_ul_a2a_pack(
            q_3d: jax.Array, kv_3d: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return a1_ul_a2a_pack_sm(q_3d, kv_3d)

        def a1_ul_gate(
            x_k: jax.Array, w_gate: jax.Array, w_gate_sc: jax.Array
        ) -> jax.Array:
            return a1_ul_gate_sm(x_k, w_gate, w_gate_sc)

        def a3_ul_rev_a2a(raw_attn_k: jax.Array) -> jax.Array:
            return a3_ul_rev_a2a_sm(raw_attn_k)

        def a3_ul_gate_fuse(attn_fp8_3d: jax.Array, gate_k: jax.Array) -> jax.Array:
            return a3_ul_gate_fuse_sm(attn_fp8_3d, gate_k)

        def a3_ul_wo(
            q_fp8: jax.Array, wo_in: jax.Array, wo_sc: jax.Array
        ) -> jax.Array:
            return a3_ul_wo_sm(q_fp8, wo_in, wo_sc)

        def a3_ul_wo_2chunk(
            q0_fp8: jax.Array,
            q1_fp8: jax.Array,
            wo_in: jax.Array,
            wo_sc: jax.Array,
        ) -> jax.Array:
            return a3_ul_wo_2c_sm(q0_fp8, q1_fp8, wo_in, wo_sc)

        a1_ulysses_qkv_chunk_op = export(
            "a1_ulysses_qkv_2op" + a1_w8_tag, a1_ul_qkv, 5, 3
        )
        a1_ulysses_a2a_pack_chunk_op = export(
            "a1_ulysses_a2a_pack_2op", a1_ul_a2a_pack, 2, 2
        )
        a1_ulysses_gate_chunk_op = export("a1_ulysses_gate", a1_ul_gate, 3, 1)
        rev_bf16_sfx = "_bf16" if ulysses_rev_bf16 else ""
        a3_ulysses_rev_a2a_chunk_op = export(
            "a3_ulysses_rev_a2a_e"
            + f"{head_norm_eps:g}".replace("-", "m").replace(".", "p")
            + rev_bf16_sfx,
            a3_ul_rev_a2a,
            1,
            1,
        )
        a3_ulysses_gate_fuse_chunk_op = export(
            "a3_ulysses_gate_fuse" + rev_bf16_sfx, a3_ul_gate_fuse, 2, 1
        )
        a3_ulysses_wo_op = export("a3_ulysses_wo", a3_ul_wo, 3, 1)
        a3_ulysses_wo_2chunk_op = export(
            "a3_ulysses_wo_2chunk", a3_ul_wo_2chunk, 4, 1
        )
    if causal_chunk_stream and a1_variant != "v10_sp":
        import jax.numpy as jnp
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

        ccs_ag_mode = (
            xor1_chunk_mode
            if xor1_chunk_mode > 0
            else (xor1_ag_mode if xor1_ag_mode in (1, 2, 3) else 0)
        )
        use_xor1_ccs_ag = (ccs_ag_mode >= 1) or (ccs_fuse_ag_a1 == 2)

        def _prep_ccs_fp8_slice(x_slice):
            if x_slice.dtype != jnp.float8_e4m3fn:
                return (
                    jnp.clip(
                        x_slice.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                        -_FP8_MAX,
                        _FP8_MAX,
                    ).astype(jnp.float8_e4m3fn)
                    if x_slice.shape[0] > 0
                    else x_slice.astype(jnp.float8_e4m3fn)
                )
            return x_slice

        def _ccs_all_gather(x_fp8):
            if use_xor1_ccs_ag:
                return xor1_bridge_all_gather(
                    x_fp8,
                    axis_name=_AXIS,
                    tiled=False,
                    xy_torus=(ccs_ag_mode >= 3),
                )
            return lax.all_gather(x_fp8, _AXIS, axis=0, tiled=False)

        def ag_chunk_body(x_slice):
            return _ccs_all_gather(_prep_ccs_fp8_slice(x_slice))

        ag_chunk_sm = shard(ag_chunk_body, 1, 1)

        def ag_chunk(x_slice: jax.Array) -> jax.Array:
            return ag_chunk_sm(x_slice)

        def a1_chunk_body(ag_k, wqkvg, wqkvg_scale, cos_k, sin_k, next_fp8=None):
            tp_d, c_k, _ = ag_k.shape
            w_sc = wqkvg_scale * kernel_a1.STATIC_A1_ACT_SCALE
            cos_3d = cos_k.reshape(tp_d, c_k, 128)
            sin_3d = sin_k.reshape(tp_d, c_k, 128)
            a1_res = kernel_a1.pallas_a1_chunk_1way(
                ag_k,
                wqkvg,
                w_sc,
                cos_3d,
                sin_3d,
                None,
                t_shard=c_k,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=a1_rpa_layout,
                mode="all",
                tp_axis=_AXIS,
                tm=256 if c_k % 256 == 0 else 128,
                next_ag_in=next_fp8,
            )
            q_k, gate_k, kv_k = a1_res[:3]
            c_tok = tp_d * c_k
            if a1_rpa_layout:
                formatted = (
                    q_k.reshape(1, c_tok, h_q_local // 2, 2, 128),
                    gate_k.reshape(c_tok, h_q_local * 64),
                    kv_k.reshape(c_tok, 1, 4, 128),
                )
            else:
                formatted = (
                    q_k.reshape(c_tok, h_q_local * 64),
                    gate_k.reshape(c_tok, h_q_local * 64),
                    kv_k.reshape(c_tok, h_kv_local * 128),
                )
            return (*formatted, a1_res[3]) if next_fp8 is not None else formatted

        a1_chunk_sm = shard(a1_chunk_body, 5, 3)

        def a1_chunk(
            ag_k: jax.Array,
            wqkvg: jax.Array,
            wqkvg_scale: jax.Array,
            cos_k: jax.Array,
            sin_k: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return a1_chunk_sm(ag_k, wqkvg, wqkvg_scale, cos_k, sin_k)

        ccs_ag_sfx = (
            f"_xag{ccs_ag_mode}"
            if ccs_ag_mode > 0
            else ("_xagccs" if use_xor1_ccs_ag else "")
        )
        rpal_sfx = "_rpal" if a1_rpa_layout else "_unpk"
        ag_chunk_op = export(
            "ag_chunk" + ccs_ag_sfx,
            ag_chunk,
            1,
            1,
        )
        a1_chunk_op = export("a1_chunk_v6" + rpal_sfx + a1_w8_tag, a1_chunk, 5, 3)
        if ccs_fuse_ag_a1 > 0:

            def a1_chunk_agd_body(ag_k, next_slice, wqkvg, wqkvg_scale, cos_k, sin_k):
                next_fp8 = _prep_ccs_fp8_slice(next_slice)
                if ccs_fuse_ag_a1 == 3:
                    return a1_chunk_body(
                        ag_k, wqkvg, wqkvg_scale, cos_k, sin_k, next_fp8=next_fp8
                    )
                next_fp8_b, ag_k_b = lax.optimization_barrier((next_fp8, ag_k))
                ag_next = _ccs_all_gather(next_fp8_b)
                q_out, gate_out, kv_out = a1_chunk_body(
                    ag_k_b, wqkvg, wqkvg_scale, cos_k, sin_k
                )
                return q_out, gate_out, kv_out, ag_next

            a1_chunk_agd_sm = shard(a1_chunk_agd_body, 6, 4)

            def a1_chunk_agd(
                ag_k: jax.Array,
                next_slice: jax.Array,
                wqkvg: jax.Array,
                wqkvg_scale: jax.Array,
                cos_k: jax.Array,
                sin_k: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                return a1_chunk_agd_sm(
                    ag_k, next_slice, wqkvg, wqkvg_scale, cos_k, sin_k
                )

            agd_sfx = "_inag" if ccs_fuse_ag_a1 == 3 else ccs_ag_sfx
            a1_chunk_agd_op = export(
                "a1_chunk_agd_v6" + rpal_sfx + a1_w8_tag + agd_sfx,
                a1_chunk_agd,
                6,
                4,
            )
    a3_pro_chunk_op = None
    a3_wo_op = None
    a3_wo_2chunk_op = None
    a3_wo_3chunk_op = None
    if (
        causal_chunk_stream
        and a3_variant == "v16"
        and ccs_chunk_a3_prologue
        and a1_variant != "v10_sp"
    ):
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a3

        eff_h_eps = head_norm_eps if (fused_glue or a3_rpa_out) else None
        tp_sz_pro = mesh.shape[_AXIS]

        def a3_pro_chunk_body(attn_k, gate_k):
            c_tok_k, k_loc = gate_k.shape
            c_k_loc = c_tok_k // tp_sz_pro
            ptm = 1024 if a3_opt else 512
            q_2d = kernel_a3.pallas_gate_static_fp8_quant(
                attn_k,
                gate_k,
                tm=min(ptm, c_tok_k),
                head_norm_eps=eff_h_eps,
                rpa_packed_out=a3_rpa_out,
            )
            return q_2d.reshape(tp_sz_pro, c_k_loc, k_loc)

        a3_pro_sm = shard(a3_pro_chunk_body, 2, 1)

        def a3_pro_chunk(attn_k: jax.Array, gate_k: jax.Array) -> jax.Array:
            return a3_pro_sm(attn_k, gate_k)

        a3_kdeep_ccs_wo1 = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get() or 0)
        w4_wo_ccs_mode = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get() or 0)
        a3_bn2k_ccs = _env_bool("SGLANG_TPU_MUSE_KERNEL_A3_BN2048")

        def a3_wo_only_body(q_3d, wo, wo_scale, *opt_valid):
            import jax.numpy as jnp

            vtok = opt_valid[0] if (a3_skip and opt_valid) else None
            is_nvfp4_wo = (
                w4_wo_ccs_mode > 0 and jnp.dtype(wo.dtype) == jnp.float4_e2m1fn
            )
            if (a3_kdeep_ccs_wo1 == 1 and vtok is None) or is_nvfp4_wo:
                if is_nvfp4_wo and vtok is not None:
                    raise ValueError(
                        "SGLANG_TPU_MUSE_KERNEL_W4_WO under CCS requires "
                        "SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_ATTN_TOKENS=0."
                    )
                from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
                    pallas_a3_wo_kdeep_static_fp8,
                )

                eff_w4 = w4_wo_ccs_mode if is_nvfp4_wo else 0
                eff_sc = (
                    wo_scale.squeeze(0)
                    if (eff_w4 > 0 and wo_scale.ndim == 4)
                    else wo_scale
                )
                return pallas_a3_wo_kdeep_static_fp8(
                    q_3d,
                    wo,
                    eff_sc,
                    tp_axis=_AXIS,
                    bn=2048 if a3_bn2k_ccs else 1024,
                    w4_qbs=eff_w4,
                )
            return kernel_a3.pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8(
                q_3d,
                wo,
                wo_scale,
                tp_axis=_AXIS,
                opt_wo=a3_opt,
                valid_tokens_1d=vtok,
            )

        a3_wo_n_in = 3 + (1 if a3_skip else 0)
        a3_wo_sm = shard(a3_wo_only_body, a3_wo_n_in, 1)

        def a3_wo_only(*args: jax.Array) -> jax.Array:
            return a3_wo_sm(*args)

        fixed_arity(a3_wo_only, a3_wo_n_in, jax.Array)
        a3_pro_chunk_op = export(
            "a3_pro_chunk"
            + glue
            + ("_rpao" if a3_rpa_out else "")
            + ("_opt" if a3_opt else "")
            + ("_zsp" if a3_zsp else ""),
            a3_pro_chunk,
            2,
            1,
        )
        a3_wo_op = export(
            "a3_v16_wo"
            + ("_opt" if a3_opt else "")
            + ("_sk" if a3_skip else "")
            + (f"_ss{a3_stage_sends}" if a3_stage_sends else "")
            + ("_bn2k" if a3_bn2k_ccs else "")
            + ("_kd1" if a3_kdeep_ccs_wo1 == 1 else "")
            + (f"_w4q{w4_wo_ccs_mode}" if w4_wo_ccs_mode > 0 else ""),
            a3_wo_only,
            a3_wo_n_in,
            1,
        )
        if (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_CCS_2CHUNK.get()
            and a3_opt
            and not a3_skip
            and a3_kdeep_ccs_wo1 == 0
            and w4_wo_ccs_mode == 0
        ):

            def a3_wo_2chunk_body(q0_3d, q1_3d, wo_in, wo_sc_in):
                return kernel_a3.pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8(
                    q0_3d,
                    q1_3d,
                    wo_in,
                    wo_sc_in,
                    tp_axis=_AXIS,
                )

            a3_wo_2c_sm = shard(a3_wo_2chunk_body, 4, 1)

            def a3_wo_2chunk(
                q0_3d: jax.Array,
                q1_3d: jax.Array,
                wo_in: jax.Array,
                wo_sc_in: jax.Array,
            ) -> jax.Array:
                return a3_wo_2c_sm(q0_3d, q1_3d, wo_in, wo_sc_in)

            a3_wo_2chunk_op = export(
                "a3_v16_wo_2chunk"
                + (f"_ss{a3_stage_sends}" if a3_stage_sends else "")
                + ("_bn2k" if a3_bn2k_ccs else ""),
                a3_wo_2chunk,
                4,
                1,
            )

            def a3_wo_3chunk_body(q0_3d, q1_3d, q2_3d, wo_in, wo_sc_in):
                return kernel_a3.pallas_option3d_3chunk_async_a2a_wo_3d_static_fp8(
                    q0_3d,
                    q1_3d,
                    q2_3d,
                    wo_in,
                    wo_sc_in,
                    tp_axis=_AXIS,
                )

            a3_wo_3c_sm = shard(a3_wo_3chunk_body, 5, 1)

            def a3_wo_3chunk(
                q0_3d: jax.Array,
                q1_3d: jax.Array,
                q2_3d: jax.Array,
                wo_in: jax.Array,
                wo_sc_in: jax.Array,
            ) -> jax.Array:
                return a3_wo_3c_sm(q0_3d, q1_3d, q2_3d, wo_in, wo_sc_in)

            a3_wo_3chunk_op = export(
                "a3_v16_wo_3chunk"
                + (f"_ss{a3_stage_sends}" if a3_stage_sends else "")
                + ("_bn2k" if a3_bn2k_ccs else ""),
                a3_wo_3chunk,
                5,
                1,
            )
    a3_pro_a2a_chunk_op = None
    a3_pro_last_chunk_op = None
    a3_kdeep_ccs_op = None
    a3_kdeep_mode = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get() or 0)
    w4_wo_mode = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get() or 0)
    a3_bn2k = _env_bool("SGLANG_TPU_MUSE_KERNEL_A3_BN2048")
    if (
        causal_chunk_stream
        and a3_variant == "v16"
        and a3_kdeep_mode >= 2
        and a1_variant != "v10_sp"
    ):
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a3
        from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
            pallas_a3_wo_kdeep_ccs_last_a2a,
            pallas_async_a2a_xor_3d,
            pallas_gate_static_fp8_quant_xor_a2a,
        )

        pro_eps = head_norm_eps if (fused_glue or a3_rpa_out) else None
        tp_d_mesh = mesh.shape[_AXIS]

        def _pro_only_body(attn_c: jax.Array, gate_c: jax.Array) -> jax.Array:
            t_tot_c, k_loc_c = gate_c.shape
            c_rows = t_tot_c // tp_d_mesh
            ptm = next(
                t for t in (1024, 512, 256, 128, 64) if t <= t_tot_c and t_tot_c % t == 0
            )
            return kernel_a3.pallas_gate_static_fp8_quant(
                attn_c,
                gate_c,
                tm=ptm,
                head_norm_eps=pro_eps,
                rpa_packed_out=a3_rpa_out,
            ).reshape(tp_d_mesh, c_rows, k_loc_c)

        def _pro_and_a2a_body(attn_c: jax.Array, gate_c: jax.Array) -> jax.Array:
            if a3_kdeep_mode >= 3:
                return pallas_gate_static_fp8_quant_xor_a2a(
                    attn_c,
                    gate_c,
                    tp_axis=_AXIS,
                    tp_deg=tp_d_mesh,
                    head_norm_eps=pro_eps,
                    rpa_packed_out=a3_rpa_out,
                )
            q_loc = _pro_only_body(attn_c, gate_c)
            return pallas_async_a2a_xor_3d(q_loc, tp_axis=_AXIS)

        pro_a2a_sm = shard(_pro_and_a2a_body, 2, 1)
        pro_last_sm = shard(_pro_only_body, 2, 1)

        def a3_pro_a2a_chunk(attn_c: jax.Array, gate_c: jax.Array) -> jax.Array:
            return pro_a2a_sm(attn_c, gate_c)

        def a3_pro_last_chunk(attn_c: jax.Array, gate_c: jax.Array) -> jax.Array:
            return pro_last_sm(attn_c, gate_c)

        def _kdeep_ccs_body(
            q_prev_xor: jax.Array,
            q_last_loc: jax.Array,
            wo_in: jax.Array,
            wo_sc_in: jax.Array,
        ) -> jax.Array:
            import jax.numpy as jnp

            eff_w4 = (
                w4_wo_mode
                if (w4_wo_mode > 0 and jnp.dtype(wo_in.dtype) == jnp.float4_e2m1fn)
                else 0
            )
            eff_sc = (
                wo_sc_in.squeeze(0)
                if (eff_w4 > 0 and wo_sc_in.ndim == 4)
                else wo_sc_in
            )
            return pallas_a3_wo_kdeep_ccs_last_a2a(
                q_prev_xor,
                q_last_loc,
                wo_in,
                eff_sc,
                tp_axis=_AXIS,
                w4_qbs=eff_w4,
            )

        kdeep_ccs_sm = shard(_kdeep_ccs_body, 4, 1)

        def a3_kdeep_ccs(
            q_prev_xor: jax.Array,
            q_last_loc: jax.Array,
            wo_in: jax.Array,
            wo_sc_in: jax.Array,
        ) -> jax.Array:
            return kdeep_ccs_sm(q_prev_xor, q_last_loc, wo_in, wo_sc_in)

        ss_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS", None)
        ss_val = int(ss_env.get() or 0) if ss_env is not None else 0
        ps_tag = f"_ss{ss_val}" if ss_val in (1, 2) else ""
        pro_tag = glue + ("_rpao" if a3_rpa_out else "") + ("_zsp" if a3_zsp else "")
        a3_pro_a2a_chunk_op = export(
            "a3_pro_a2a_chunk"
            + pro_tag
            + ("_fpro" if a3_kdeep_mode >= 3 else "")
            + ps_tag,
            a3_pro_a2a_chunk,
            2,
            1,
        )
        a3_pro_last_chunk_op = export(
            "a3_pro_last_chunk" + pro_tag, a3_pro_last_chunk, 2, 1
        )
        a3_kdeep_ccs_op = export(
            "a3_kdeep_ccs"
            + ps_tag
            + (f"_w4o{w4_wo_mode}" if w4_wo_mode > 0 else ""),
            a3_kdeep_ccs,
            4,
            1,
        )
    owner_op = None
    if tp16_owner_a1:
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

        def owner_body(x, weights, scales, cos, sin):
            def project(w, scale):
                return kernel_a1.pallas_qkvg_qk_rope_kv(
                    x,
                    w,
                    scale,
                    cos,
                    sin,
                    h_q_local=h_q_local,
                    h_kv_local=h_kv_local,
                    kernel_name="tp16_owner_a1_qkvg",
                )

            q, gate, kv = jax.vmap(project)(weights, scales)

            # The input axis enumerates destination head shards. After exchange
            # it enumerates source token shards in the original sequence order.
            def exchange(y):
                return lax.all_to_all(
                    y, _AXIS, split_axis=0, concat_axis=0, tiled=True
                ).reshape(-1, y.shape[-1])

            return exchange(q), exchange(gate), exchange(kv)

        owner_sm = shard(owner_body, 5, 3)

        def owner_a1(
            x: jax.Array,
            weights: jax.Array,
            scales: jax.Array,
            cos: jax.Array,
            sin: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return owner_sm(x, weights, scales, cos, sin)

        owner_op = export("a1_owner", owner_a1, 5, 3)

    return _Ops(
        a1_owner=owner_op,
        gather=export("gather", gather, 1, 1),
        # The default variants keep their original op names (compile-cache keys).
        a1=export(a1_name, a1, a1_n_in, 3),
        a3=export(
            ("a3" if a3_variant == "v3" else f"a3_{a3_variant}")
            + glue
            + ("_rpao" if a3_rpa_out else "")
            + ("_opt" if a3_opt else "")
            + ("_sk" if a3_skip else "")
            + ("_zsp" if a3_zsp else "")
            + (f"_ss{a3_stage_sends}" if a3_stage_sends else "")
            + ("_bn2k" if a3_bn2k else "")
            + (f"_kd{a3_kdeep_mode}" if a3_kdeep_mode > 0 else "")
            + (f"_w4o{w4_wo_mode}" if w4_wo_mode > 0 else ""),
            a3,
            a3_n_in,
            1,
        ),
        # pipeline reads env flags while tracing; the name keys the cache.
        moe=export("moe" + moe_suffix, moe, 22, 2),
        moe_fp8=export("moe_fp8" + moe_suffix + fp8_extra, moe, 22, 2)
        if fp8_layers
        else None,
        sum_stream=export("sum_stream", sum_stream, 1, 1)
        if (exit_owner_gather or l61_prune)
        else None,
        ag_chunk=ag_chunk_op,
        a1_chunk=a1_chunk_op,
        a1_chunk_agd=a1_chunk_agd_op,
        a3_prologue_chunk=a3_pro_chunk_op,
        a3_wo=a3_wo_op,
        a3_wo_2chunk=a3_wo_2chunk_op,
        a3_wo_3chunk=a3_wo_3chunk_op,
        gather_fp8_stream=gather_fp8_op,
        a3_pro_a2a_chunk=a3_pro_a2a_chunk_op,
        a3_pro_last_chunk=a3_pro_last_chunk_op,
        a3_kdeep_ccs=a3_kdeep_ccs_op,
        a1_ulysses_qkv_chunk=a1_ulysses_qkv_chunk_op,
        a1_ulysses_a2a_pack_chunk=a1_ulysses_a2a_pack_chunk_op,
        a1_ulysses_gate_chunk=a1_ulysses_gate_chunk_op,
        a3_ulysses_rev_a2a_chunk=a3_ulysses_rev_a2a_chunk_op,
        a3_ulysses_gate_fuse_chunk=a3_ulysses_gate_fuse_chunk_op,
        a3_ulysses_wo=a3_ulysses_wo_op,
        a3_ulysses_wo_2chunk=a3_ulysses_wo_2chunk_op,
        **agd_ops,
    )


def _agd_ops(
    *,
    a1_body,
    moe_body,
    first_rows,
    shard,
    export,
    fixed_arity,
    a1_name,
    moe_name,
    fp8_moe_name,
) -> dict:
    """AG_UNDER_D op pair: A1 taking (gathered part 0, rest rows, ...) and the MoE op emitting it."""
    import jax

    a1_sm = shard(a1_body, 6, 3)

    def a1_agd(*args: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return a1_sm(*args)

    fixed_arity(a1_agd, 6, tuple[jax.Array, jax.Array, jax.Array])

    def moe_agd_body(*xs):
        return moe_body(*xs, next_in_gather_rows=first_rows(xs[0].shape[0]))

    moe_sm = shard(moe_agd_body, 22, 3)

    def moe_agd(*args: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return moe_sm(*args)

    fixed_arity(moe_agd, 22, tuple[jax.Array, jax.Array, jax.Array])
    return dict(
        a1_agd=export(a1_name + "_agd", a1_agd, 6, 3),
        moe_agd=export(moe_name, moe_agd, 22, 3),
        moe_fp8_agd=export(fp8_moe_name, moe_agd, 22, 3) if fp8_moe_name else None,
    )


def _router_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[E, H]`` gate -> f32 ``[H, E]``, or with ROUTER_HILO_PRE B1's bf16 hi/lo split ``[2 * H, E]``."""
    w = weight.detach().t().float().contiguous()
    if not tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTER_HILO_PRE.get():
        return w
    hi = w.to(
        torch.bfloat16
    )  # round-to-nearest-even, as kernel_b1.prepare_b1_static_inputs
    # lo = 0: on TPU, XLA folds B1's per-forward f32(bf16(w)) back to w, so serving's lo is all zero.
    return torch.cat([hi, torch.zeros_like(hi)]).contiguous()


def _fp8_k_major(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, K]`` linear weight -> fp8 e4m3 ``[K, N]`` and f32 ``[N]`` column scales."""
    w = weight.detach().t().float()
    scale = (w.abs().amax(dim=0) / _FP8_MAX).clamp_min(1e-12)
    return (w / scale).to(torch.float8_e4m3fn).contiguous(), scale


def _prepare_wo(
    o_proj_weight: torch.Tensor,
    *,
    a3_variant: str,
    gather,
    tp_rank: int,
    skip_v16_perm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """A3's fp8 Wo, ``[N]`` scales and the shape A3 reads Wo in.

    Wo holds this rank's K rows (v3) or all K rows (Option 3D).
    """
    if a3_variant == "v3":
        wo, wo_scale = _fp8_k_major(o_proj_weight)
        return wo, wo_scale, tuple(wo.shape)
    w_kn = gather(o_proj_weight.t().contiguous())
    if a3_variant == "v16" and not skip_v16_perm:
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a3

        # [tp, K / tp, N]: v16 step d multiplies the block from peer(tp_rank, d) by slice d.
        k_local = o_proj_weight.shape[1]
        tp_deg = w_kn.shape[0] // k_local
        ss_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS.get()
        parity_stagger = tp_deg == 8 and ss_mode in (1, 2)
        order = [
            kernel_a3.a3_v16_step_peer(
                tp_rank, d, tp_deg, parity_stagger=parity_stagger
            )
            for d in range(tp_deg)
        ]
        w_kn = w_kn.view(len(order), k_local, -1).index_select(
            0, torch.tensor(order, device=w_kn.device)
        )
    # One column scale over the full K: Option 3D scales the sum of all sources' partials.
    wo, wo_scale = _fp8_k_major(w_kn.reshape(-1, w_kn.shape[-1]).t())
    return wo, wo_scale, tuple(w_kn.shape)


def _diff_stats(ref: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """[cosine, min row cosine, max |diff|, rel L2] of two ``[T, H]`` tensors."""
    ref, out = ref.float(), out.float()
    tiny = 1e-30
    cos = (ref * out).sum() / (ref.norm() * out.norm() + tiny)
    row_cos = (ref * out).sum(-1) / (ref.norm(dim=-1) * out.norm(dim=-1) + tiny)
    diff = ref - out
    return torch.stack(
        [cos, row_cos.min(), diff.abs().max(), diff.norm() / (ref.norm() + tiny)]
    )


class _LayerWeights(torch.nn.Module):
    """One kernel layer's prepared copies (buffers, so the compiled graph takes them as inputs)."""

    def __init__(
        self,
        layer,
        gather,
        a3_variant: str,
        tp_rank: int,
        a1_variant: str = "v1",
        next_norm_weight: Optional[torch.Tensor] = None,
        tp16_owner_a1: bool = False,
    ) -> None:
        super().__init__()
        from sglang_tpu.fp8_moe import _own_storage

        attn, moe = layer.self_attn, layer.mlp
        q, k, v = attn.qkv_proj.weight.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=0
        )
        hybrid_buffers: dict[str, torch.Tensor] = {}
        if a1_variant in _HYBRID_A1_VARIANTS:
            from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

            kv_all = gather(torch.cat([k, v], dim=0))
            two_kv = 2 * attn.kv_size
            tp_sz = kv_all.shape[0] // two_kv
            wk_list, wks_list = [], []
            for r in range(tp_sz):
                wk_r, wks_r = _fp8_k_major(kv_all[r * two_kv : (r + 1) * two_kv])
                wk_list.append(wk_r)
                wks_list.append(wks_r)
            norm = layer.input_layernorm
            w_norm = (
                norm.gain().detach().float()
                if hasattr(norm, "gain")
                else (norm.weight.detach().float() + getattr(norm, "gain_center", 1.0))
            )
            qg_raw = torch.cat([q, attn.gate_proj.weight], dim=0)
            if a1_variant == "v6_bf16kv":
                g_clip = (w_norm.abs() * 0.5).clamp_min(1.0)
                wqkvg, wqkvg_scale = _fp8_k_major(qg_raw.float() * g_clip[None, :])
                a1_gain = (1.0 / g_clip).to(torch.bfloat16)[None, :].contiguous()
                a1_inv_gain = (
                    (float(kernel_a1.STATIC_A1_ACT_MULT) / g_clip)[None, :]
                    .float()
                    .contiguous()
                )
            else:
                wqkvg, wqkvg_scale = _fp8_k_major(qg_raw)
                g_safe = w_norm.abs().clamp_min(1.0).to(torch.bfloat16)
                a1_gain = g_safe[None, :].contiguous()
                a1_inv_gain = (
                    (1.0 / g_safe.float()).to(torch.bfloat16)[None, :].contiguous()
                )
            hybrid_buffers = {
                "wkv": torch.cat(wk_list, dim=0).contiguous(),
                "wkv_scale": torch.stack(wks_list, dim=0).contiguous(),
                "a1_gain": a1_gain,
                "a1_inv_gain": a1_inv_gain,
            }
        elif a1_variant == "v10_sp":
            # Static-FP8 Ulysses: replicated QKV ([tp, H, Q+2*KV]) + replicated Gate ([H, H]),
            # gathered from pre-quantized per-rank FP8 columns for 0-ULP weight identity vs v6.
            wqkv_loc, wqkv_sc_loc = _fp8_k_major(torch.cat([q, k, v], dim=0))
            hidden_w, qkv_cols = wqkv_loc.shape
            wqkvg = _own_storage(
                gather(wqkv_loc.view(torch.uint8))
                .view(torch.float8_e4m3fn)
                .reshape(-1, hidden_w, qkv_cols)
            )
            wqkvg_scale = _own_storage(gather(wqkv_sc_loc[None]))
            wg_loc, wg_sc_loc = _fp8_k_major(attn.gate_proj.weight)
            w_gate = _own_storage(
                gather(wg_loc.t().contiguous().view(torch.uint8))
                .view(torch.float8_e4m3fn)
                .t()
                .contiguous()
            )
            w_gate_scale = _own_storage(gather(wg_sc_loc))
            hybrid_buffers = {
                "wqkvg_scale_3d": _own_storage(wqkvg_scale[:, None, :]),
                "w_gate": w_gate,
                "w_gate_scale": w_gate_scale,
                "w_gate_scale_2d": _own_storage(w_gate_scale[None]),
            }
        else:
            # A1 reads its columns as [q | gate | k | v].
            wqkvg, wqkvg_scale = _fp8_k_major(
                torch.cat([q, attn.gate_proj.weight, k, v], dim=0)
            )
        if tp16_owner_a1:
            hidden, columns = wqkvg.shape
            # Gather already-quantized columns to preserve the original FP8 values.
            wqkvg = _own_storage(
                gather(wqkvg.view(torch.uint8))
                .view(torch.float8_e4m3fn)
                .reshape(-1, hidden, columns)
            )
            wqkvg_scale = _own_storage(gather(wqkvg_scale[None]))
        wo, wo_scale, self.wo_shape = _prepare_wo(
            attn.o_proj.weight,
            a3_variant=a3_variant,
            gather=gather,
            tp_rank=tp_rank,
            skip_v16_perm=(a1_variant == "v10_sp"),
        )
        w4_wo_qbs = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get() or 0)
        if (
            a3_variant == "v16"
            and a1_variant != "v10_sp"
            and w4_wo_qbs == 256
            and _ccs_wo_uses_kdeep_nvfp4()
        ):
            from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
                quantize_dense_fp8_to_nvfp4_torch,
            )

            wo, wo_scale, _ = quantize_dense_fp8_to_nvfp4_torch(
                wo, wo_scale, w_shape=self.wo_shape, block=256
            )
            self.wo_shape = tuple(wo.shape)
        else:
            wo = _own_storage(wo.reshape(self.wo_shape))
        if wqkvg_scale.ndim == 1:
            hybrid_buffers["wqkvg_scale_2d"] = _own_storage(wqkvg_scale[None])
        if wo_scale.ndim == 1:
            hybrid_buffers["wo_scale_2d"] = _own_storage(wo_scale[None])
        # Column-parallel shards -> full [out, in] weights, replicated (Option 3A).
        wpool, wpool_scale = _fp8_k_major(gather(moe.pre_expert_proj.weight))
        wunpool, wunpool_scale = _fp8_k_major(gather(moe.post_expert_proj.weight))
        w4_wu_qbs = int(tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get() or 0)
        if w4_wu_qbs == 256:
            from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
                quantize_dense_fp8_to_nvfp4_torch,
            )

            wunpool, wunpool_scale, _ = quantize_dense_fp8_to_nvfp4_torch(
                wunpool, wunpool_scale, w_shape=tuple(wunpool.shape), block=256
            )
        elif (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_PREPACK_VECS.get()
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_ZERO_SPILL.get()
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_TOK_TILE.get() > 64
        ) and next_norm_weight is not None:
            ws_1d = wunpool_scale.reshape(1, -1).float()
            pfg_1d = (
                layer.post_feedforward_layernorm.weight.reshape(1, -1).float() + 1.0
            )
            a_1d = layer.post_feedforward_residual_gate.alpha.reshape(1, -1).float()
            b_1d = layer.post_feedforward_residual_gate.beta.reshape(1, -1).float()
            nig_1d = next_norm_weight.reshape(1, -1).float() + 1.0
            z_1d = torch.zeros_like(ws_1d)
            wunpool_scale = torch.cat(
                [ws_1d, pfg_1d, a_1d, b_1d, nig_1d, z_1d, z_1d, z_1d], dim=0
            ).contiguous()
        b12_prepack_buffers: dict[str, torch.Tensor] = {}
        if (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK.get()
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST.get()
        ):
            aa_1d = layer.post_attention_residual_gate.alpha.reshape(1, -1).float()
            ba_1d = layer.post_attention_residual_gate.beta.reshape(1, -1).float()
            pff_1d = (
                layer.pre_feedforward_layernorm.weight.reshape(1, -1).float() + 1.0
            )
            zb1_1d = torch.zeros_like(aa_1d)
            b1_vecs = torch.cat(
                [aa_1d, ba_1d, pff_1d, zb1_1d, zb1_1d, zb1_1d, zb1_1d, zb1_1d],
                dim=0,
            ).contiguous()
            b1_bias_t = (
                moe.e_score_correction_bias.detach()
                .float()
                .unsqueeze(1)
                .expand(-1, 128)
                .contiguous()
            )
            wps_1d = wpool_scale.reshape(1, -1).float()
            pen_1d = moe.pre_expert_norm.weight.reshape(1, -1).float() + 1.0
            zb2_1d = torch.zeros_like(wps_1d)
            b2_vecs = torch.cat(
                [wps_1d, pen_1d, zb2_1d, zb2_1d, zb2_1d, zb2_1d, zb2_1d, zb2_1d],
                dim=0,
            ).contiguous()
            b12_prepack_buffers = {
                "b1_vecs": b1_vecs,
                "b1_bias_t": b1_bias_t,
                "b2_vecs": b2_vecs,
            }
        buffers = {
            "wqkvg": wqkvg,
            "wqkvg_scale": wqkvg_scale,
            **hybrid_buffers,
            "wo": wo,
            "wo_scale": wo_scale,
            "wpool": wpool,
            "wpool_scale": wpool_scale,
            "wunpool": wunpool,
            "wunpool_scale": wunpool_scale,
            "wrouter": _router_weight(moe.gate.weight),
            **b12_prepack_buffers,
        }
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())


def _v10_sp_config_error(
    tp_size: int,
    *,
    h_q_local: int,
    h_kv_local: int,
    causal_chunk_stream: bool,
    a3_variant: str,
    a1_rpa_layout: bool,
    a3_rpa_out: bool,
    kv_pack_heads: bool,
    skip_pad_attn_tokens: bool,
    ep_tp2_mode: bool = False,
) -> Optional[str]:
    """Validate Static-FP8 Ulysses (`A1_VARIANT='v10_sp'`) requirements across TP=8 and TP=16."""
    expected_kv_pack = tp_size == 8
    expected_heads = (16, 2) if tp_size == 8 else (8, 1)
    if (
        tp_size not in (8, 16)
        or (h_q_local, h_kv_local) != expected_heads
        or not causal_chunk_stream
        or not a1_rpa_layout
        or a3_variant != "v16"
        or ep_tp2_mode
        or a3_rpa_out
        or skip_pad_attn_tokens
        or kv_pack_heads != expected_kv_pack
    ):
        kv_pack_str = "1" if expected_kv_pack else "0"
        return (
            f"native TP{tp_size} A1_VARIANT='v10_sp' requires (h_q_local, h_kv_local)={expected_heads}, "
            f"CAUSAL_CHUNK_STREAM=1, A1_RPA_LAYOUT=1, A3_VARIANT='v16', pure EP, "
            f"KV_PACK_HEADS={kv_pack_str}, A3_RPA_OUT=0, and SKIP_PAD_ATTN_TOKENS=0"
        )
    return None


class MusePrefillKernels(torch.nn.Module):
    """Sequence-parallel kernel run of decoder layers ``[start, end)`` on eligible batches."""

    def __init__(
        self,
        decoder,
        start: int,
        end: int,
        tp_size: int,
        tp_rank: int,
        *,
        ep_tp2_mode: bool = False,
    ):
        super().__init__()
        attn = decoder.layers[start].self_attn
        self.start, self.end = start, end
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.ep_tp2_mode = ep_tp2_mode
        self.finer_buckets = _env_bool("SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS")
        self.kv_size = attn.kv_size
        self.a1_variant = tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT.get()
        self.a3_variant = tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_VARIANT.get()
        self.fused_glue = tpu_envs.SGLANG_TPU_MUSE_KERNEL_FUSED_GLUE.get()
        self.exit_gather_bf16 = tpu_envs.SGLANG_TPU_MUSE_KERNEL_EXIT_GATHER_BF16.get()
        self.exit_owner_gather = tpu_envs.SGLANG_TPU_MUSE_KERNEL_EXIT_OWNER_GATHER.get()
        self.l61_prune = tpu_envs.SGLANG_TPU_MUSE_L61_PRUNE.get()
        if self.l61_prune:
            from sglang_tpu.attention_pallas_rpa_v3 import _get_write_kv_only_op

            _get_write_kv_only_op()
        self.capture_dflash = tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAPTURE_DFLASH.get()
        # q / kv go straight to the backend's prepared RPA op; TP=8 needs KV_PACK_HEADS (pack=2),
        # while TP=16 already has h_kv_local=1 (pack=1).
        self.a1_rpa_layout = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT.get()
            and not ep_tp2_mode
            and (
                tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get()
                or (tp_size == 16 and self.a1_variant in ("v6", "v10_sp"))
            )
        )
        self.a3_rpa_out = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_RPA_OUT.get()
            and tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get()
            and self.a3_variant == "v16"
        )
        self.a1_chunks = tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_CHUNKS.get()
        self.a1_1way = tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_1WAY.get()
        self.fp8_layers = tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP8_LAYERS.get()
        self.a1_zc = tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_ZC.get()
        self.a3_opt = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_OPT.get() and self.a3_variant == "v16"
        )
        self.skip_pad_attn_tokens = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_ATTN_TOKENS.get()
        )
        self._a1_skip = self.skip_pad_attn_tokens and self.a1_variant not in (
            "v1",
            "v6",
            "v10_sp",
        )
        self._a3_skip = self.skip_pad_attn_tokens and self.a3_variant == "v16"
        self.sharded_rope = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SHARDED_ROPE.get()
        self.a1_parts = tuple(
            int(w)
            for w in tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_PARTS.get().split(",")
            if w
        )
        self.a1_ag_under_d = (
            False
            if self.a1_variant == "v10_sp"
            else tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D.get()
        )
        self.causal_chunk_stream = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM.get()
        )
        self.causal_chunk_spec = tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_SPEC.get() or (
            "256,512" if self.a1_variant == "v10_sp" else ""
        )
        emb_mod = getattr(decoder, "embed_tokens", None)
        self.sp_entry = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_SP_ENTRY.get()
            and emb_mod is not None
            and hasattr(emb_mod, "weight")
            and getattr(emb_mod, "num_added_embeddings", 0) == 0
        )
        if self.a1_variant not in _A1_VARIANTS or self.a3_variant not in _A3_VARIANTS:
            raise ValueError(
                f"Muse prefill kernels: A1 variant {self.a1_variant!r} not in "
                f"{_A1_VARIANTS} or A3 variant {self.a3_variant!r} not in {_A3_VARIANTS}"
            )
        self.tp16_owner_a1 = (
            tp_size == 16
            and self.a1_variant == "v1"
            and tpu_envs.SGLANG_TPU_MUSE_KERNEL_TP16_OWNER_A1.get()
        )
        if self.a1_variant == "v10_sp":
            v10_err = _v10_sp_config_error(
                self.tp_size,
                h_q_local=attn.num_heads,
                h_kv_local=attn.num_kv_heads,
                causal_chunk_stream=self.causal_chunk_stream,
                a3_variant=self.a3_variant,
                a1_rpa_layout=self.a1_rpa_layout,
                a3_rpa_out=self.a3_rpa_out,
                kv_pack_heads=tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get(),
                skip_pad_attn_tokens=self.skip_pad_attn_tokens,
                ep_tp2_mode=self.ep_tp2_mode,
            )
            if v10_err is not None:
                raise ValueError(v10_err)
        if self.causal_chunk_stream and (
            self.a1_variant not in ("v6", "v10_sp")
            or (self.tp_size != 16 and not self.a1_rpa_layout)
            or self.ep_tp2_mode
            or self.tp16_owner_a1
            or self.skip_pad_attn_tokens
        ):
            raise ValueError(
                "CAUSAL_CHUNK_STREAM requires A1_VARIANT in ('v6', 'v10_sp'), "
                "(A1_RPA_LAYOUT=1 or TP=16), pure EP, and SKIP_PAD_ATTN_TOKENS=0"
            )
        self.ops = _ops(
            attn.num_heads,
            attn.num_kv_heads,
            self.a1_variant,
            self.a3_variant,
            self.fused_glue,
            attn.norm_eps,
            self.a1_rpa_layout,
            self.a3_rpa_out,
            self.a1_chunks,
            self.a1_1way,
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PACKED_SORTS.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTER_HILO_PRE.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT.get(),
            self.fp8_layers,
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_TOK_TILE.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_ZERO_SPILL.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_PREPACK_VECS.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_MODE.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DEQUANT_1PASS.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE.get(),
            self.exit_owner_gather,
            self.a3_opt,
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_FP8_LHS.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL.get(),
            self.a1_zc,
            self.skip_pad_attn_tokens,
            self.a1_parts,
            self.a1_ag_under_d,
            self.tp16_owner_a1,
            self.ep_tp2_mode,
            self.causal_chunk_stream,
            self.causal_chunk_spec,
            self.l61_prune,
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_OPT.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP8_STREAM_GATHER.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG.get(),
            _w8_carb_cache_tag(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_ULYSSES_REV_BF16.get(),
        )
        self.a3_kdeep_wo = int(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get() or 0
        )
        self._cm_perm_cache: dict[tuple[tuple[int, ...], bool], torch.Tensor] = {}
        self.debug_compare = tpu_envs.SGLANG_TPU_MUSE_KERNEL_DEBUG_COMPARE.get()
        self.nan_trap = tpu_envs.SGLANG_TPU_MUSE_NAN_TRAP.get()
        hidden = decoder.config.hidden_size
        device = decoder.norm.weight.device
        if self.ep_tp2_mode:
            e_per_owner = 256 // (tp_size // 2)
            self.register_buffer(
                "ep_tp2_group_offset",
                torch.tensor([tp_rank * e_per_owner], dtype=torch.int32, device=device),
                persistent=False,
            )
        # Interleaved pairs rotate as (x0 * cos - x1 * sin, x1 * cos + x0 * sin).
        self.register_buffer(
            "sin_sign",
            torch.tensor([-1.0, 1.0], device=device).repeat(attn.head_dim // 2),
            persistent=False,
        )
        self.entry_op = None
        if self.sp_entry:
            h_col = hidden // tp_size
            emb_w = emb_mod.weight.detach()
            local_col = emb_w[:, tp_rank * h_col : (tp_rank + 1) * h_col].contiguous()
            org_vocab = int(
                getattr(emb_mod, "org_vocab_size", decoder.config.vocab_size)
            )
            if getattr(emb_mod, "enable_tp", True):
                shard_idx = getattr(emb_mod, "shard_indices", None)
                src_w = emb_w
                if shard_idx is not None:
                    valid_rows = (
                        shard_idx.org_vocab_end_index - shard_idx.org_vocab_start_index
                    )
                    if valid_rows < src_w.shape[0]:
                        row_mask = (
                            torch.arange(src_w.shape[0], device=device) < valid_rows
                        )
                        src_w = torch.where(
                            row_mask[:, None], src_w, torch.zeros_like(src_w)
                        )
                embed_col = _gather_sp_entry_embed_col(
                    self._gather,
                    src_w,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    h_col=h_col,
                )
            else:
                if org_vocab < local_col.shape[0]:
                    row_mask = (
                        torch.arange(local_col.shape[0], device=device) < org_vocab
                    )
                    embed_col = torch.where(
                        row_mask[:, None], local_col, torch.zeros_like(local_col)
                    )
                else:
                    embed_col = local_col + 0
            if not bool(torch.isfinite(embed_col).all().cpu()):
                raise RuntimeError(
                    f"Muse sp_entry embed_col contains non-finite values on rank={tp_rank}"
                )
            self.register_buffer("embed_col", embed_col, persistent=False)
            self.entry_op = _build_entry_op(
                org_vocab,
                start_is_zero=(start == 0),
                async_a2a=_env_bool("SGLANG_TPU_MUSE_KERNEL_ENTRY_ASYNC_A2A"),
            )
        self.rope_op = None
        if self.sharded_rope:
            c64 = decoder.rotary_emb.cos.detach().float().repeat_interleave(2, dim=-1)
            s64 = (
                decoder.rotary_emb.sin.detach().float().repeat_interleave(2, dim=-1)
                * self.sin_sign[None, :]
            )
            self.register_buffer(
                "rope_cos_sin_128",
                torch.cat([c64, s64], dim=-1).contiguous(),
                persistent=False,
            )
            self.rope_op = _build_rope_op(tp_size)
        # The last kernel layer's Kernel D also emits a next-layer input nobody reads.
        self.register_buffer(
            "no_next_norm", torch.zeros(hidden, device=device), persistent=False
        )
        self._has_nope = any(
            getattr(decoder.layers[i].self_attn, "rotary_emb", None) is None
            for i in range(start, end)
        )
        self.weights = torch.nn.ModuleList(
            _LayerWeights(
                decoder.layers[i],
                self._gather,
                self.a3_variant,
                tp_rank,
                self.a1_variant,
                decoder.layers[i + 1].input_layernorm.weight
                if i + 1 < end
                else self.no_next_norm,
                tp16_owner_a1=self.tp16_owner_a1,
            )
            for i in range(start, end)
        )
        self._init_eplb(decoder, attn, device)
        if self.debug_compare:
            self.register_buffer(
                "debug_stats",
                torch.zeros(end - start, 4, device=device),
                persistent=False,
            )
            self.register_buffer(
                "debug_count",
                torch.zeros(1, dtype=torch.int32, device=device),
                persistent=False,
            )
            if tp_rank == 0:
                threading.Thread(target=self._log_debug_stats, daemon=True).start()
        if self.nan_trap:
            self.register_buffer(
                "nan_trap_first",
                torch.full((4,), -1, dtype=torch.int32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "nan_trap_last",
                torch.full((4,), -1, dtype=torch.int32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "nan_trap_count",
                torch.zeros(1, dtype=torch.int32, device=device),
                persistent=False,
            )
            self._nan_trap_seen = 0
            self._nan_trap_armed_logged = False
        torch.accelerator.synchronize()
        logger.info(
            "Muse prefill kernels: layers [%d, %d), A1 %s%s%s, A3 %s%s, fused glue %s, "
            "%.2f GiB of weight copies per device%s",
            start,
            end,
            self.a1_variant,
            " (RPA layout)" if self.a1_rpa_layout else "",
            f" (chunks={self.a1_chunks or '1way'})"
            if (self.a1_chunks or self.a1_1way)
            else "",
            self.a3_variant,
            " (RPA out)" if self.a3_rpa_out else "",
            self.fused_glue,
            sum(w.nbytes() for w in self.weights) / 2**30,
            ", debug compare on" if self.debug_compare else "",
        )

    @classmethod
    def from_model(cls, model) -> Optional["MusePrefillKernels"]:
        """The kernel run for ``model``, or None (logged) when its layout is unsupported."""
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled
        from sglang.srt.runtime_context import get_parallel
        from sglang_tpu.fp4_moe import qblock_layout
        from sglang_tpu.muse_prefill_kernels.attn_kernels.reference import (
            QUERY_SCALE,
        )

        parallel = get_parallel()
        decoder = model.model
        cfg = model.text_config
        reason = None
        tp = parallel.tp_size
        if tp == 16 and tpu_envs.SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1.get() not in (0, 3):
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1 not in (0, 3) is disabled on TP=16 until #2833 is root-caused"
            )
        native_tp16 = tp == 16 and tpu_envs.SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL.get()
        ep_tp2_prefill = (
            _env_bool("SGLANG_TPU_MUSE_KERNEL_EP_TP2_PREFILL")
            and tp in (8, 16)
            and tp == parallel.attn_tp_size == 2 * parallel.moe_ep_size
            and getattr(parallel, "moe_tp_size", 1) == 2
        )
        pure_ep_prefill = (
            tp == parallel.attn_tp_size == parallel.moe_ep_size
            and (tp == 8 or native_tp16)
        )
        if not (pure_ep_prefill or ep_tp2_prefill):
            reason = (
                "needs matching TP/attention/EP sizes of 8 or opted-in 16, "
                "or opted-in hybrid EP(P/2)xTP2"
            )
        elif (native_tp16 or (ep_tp2_prefill and tp == 16)) and (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        ) != (
            128,
            16,
        ):
            reason = "native TP16 needs eight query heads and one KV head per rank"
        elif (ep_tp2_prefill and tp == 16) and (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT.get() != "v1"
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_VARIANT.get() != "v3"
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D.get()
            or (
                tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get()
                and tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT.get()
            )
        ):
            reason = "hybrid EP8xTP2 TP16 needs unpacked A1 v1 and A3 v3"
        elif ((pure_ep_prefill or ep_tp2_prefill) and tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT.get() == "v10_sp") and (
            v10_reason := _v10_sp_config_error(
                tp,
                h_q_local=cfg.num_attention_heads // tp,
                h_kv_local=cfg.num_key_value_heads // tp,
                causal_chunk_stream=tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM.get(),
                a3_variant=tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_VARIANT.get(),
                a1_rpa_layout=tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT.get(),
                a3_rpa_out=tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_RPA_OUT.get(),
                kv_pack_heads=tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get(),
                skip_pad_attn_tokens=tpu_envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_ATTN_TOKENS.get(),
                ep_tp2_mode=ep_tp2_prefill,
            )
        ) is not None:
            reason = v10_reason
        elif native_tp16 and (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT.get() not in ("v1", "v6", "v10_sp")
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_VARIANT.get() not in ("v3", "v16")
            or (
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT.get()
                and tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT.get() not in ("v6", "v10_sp")
            )
        ):
            reason = (
                "native TP16 needs A1 v1/v6/v10_sp (v6/v10_sp for A1_RPA_LAYOUT) "
                "and A3 v3/v16"
            )
        elif parallel.attn_dcp_size != 1 or is_dp_attention_enabled():
            reason = "does not support DCP or DP attention"
        elif decoder.pp_group.world_size != 1:
            reason = "does not support pipeline parallelism"
        elif not (
            math.isclose(cfg.qk_scale_factor, QUERY_SCALE, rel_tol=1e-6)
            and math.isclose(cfg.output_multiplier, 0.17677669529663687, rel_tol=1e-6)
            and cfg.routed_scaling_factor == 1.0
            and cfg.rms_norm_eps == 1e-5
            and cfg.post_norm_eps == 1e-8
        ) or (
            cfg.hidden_size != 8192
            or cfg.moe_hidden_size != 4096
            or cfg.num_local_experts != 256
            or cfg.num_experts_per_tok != 8
            or cfg.head_dim != 64
        ):
            reason = "the kernels hard-code another Muse Spark geometry"
        if reason is not None:
            logger.warning("Muse prefill kernels off: %s", reason)
            return None

        fp8_layers = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP8_LAYERS.get() and not ep_tp2_prefill
        )
        ep_tp2_e_owner = 256 // max(parallel.moe_ep_size, 1)

        def runs_on_kernels(layer) -> bool:
            experts = layer.mlp.experts
            is_nvfp4 = (
                experts.w13_weight.dtype == torch.float4_e2m1fn_x2
                and qblock_layout(experts) == 0
            )
            if ep_tp2_prefill:
                return (
                    is_nvfp4
                    and getattr(layer.mlp, "_is_hybrid_ep_tp2", False)
                    and experts.__dict__.get("_tpu_ep_tp2_packed_fp4", False)
                    and tuple(experts.w13_weight.shape)
                    == (ep_tp2_e_owner, 4096, 2048)
                    and tuple(experts.w2_weight.shape) == (ep_tp2_e_owner, 2048, 2048)
                )
            is_fp8 = (
                fp8_layers
                and experts.w13_weight.dtype == torch.float8_e4m3fn
                and getattr(experts, "weight_block_size", None) is None
            )
            return (
                (is_nvfp4 or is_fp8)
                and experts.handles_moe_reduction
                and not experts.__dict__.get("_tpu_fused_rs_expert_tp", False)
            )

        eligible = [
            i for i, layer in enumerate(decoder.layers) if runs_on_kernels(layer)
        ]
        if not eligible or eligible != list(range(eligible[0], eligible[-1] + 1)):
            logger.warning(
                "Muse prefill kernels off: NVFP4 fused-RS layers %s are not one run",
                eligible,
            )
            return None
        return cls(
            decoder,
            start=eligible[0],
            end=eligible[-1] + 1,
            tp_size=parallel.tp_size,
            tp_rank=parallel.tp_rank,
            ep_tp2_mode=ep_tp2_prefill,
        )

    def handles(self, forward_batch, num_tokens: int, capturing: bool) -> bool:
        row_align = (
            128
            if (self.finer_buckets and self.a3_variant == "v16")
            else _ROWS_PER_RANK_ALIGN
        )
        return (
            forward_batch.forward_mode == ForwardMode.EXTEND
            and (not capturing or self.capture_dflash)
            and num_tokens % (row_align * self.tp_size) == 0
            and _a3_supports(self.a3_variant, num_tokens // self.tp_size)
        )

    def _gather(self, shard: torch.Tensor) -> torch.Tensor:
        return self.ops.gather(shard)

    def _cm_perm_tensor(
        self, chunks: tuple[int, ...], device, *, inverse: bool = False
    ) -> torch.Tensor:
        cache = getattr(self, "_cm_perm_cache", None)
        key = (chunks, inverse)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None and cached.device == device:
                return cached
        from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
            from_chunk_major_perm,
            to_chunk_major_perm,
        )

        fn = from_chunk_major_perm if inverse else to_chunk_major_perm
        t = torch.tensor(fn(self.tp_size, chunks), dtype=torch.long, device=device)
        if cache is not None and not torch.compiler.is_compiling():
            cache[key] = t
        return t

    def embed_stream(self, decoder, input_ids: torch.Tensor) -> torch.Tensor:
        """Column-sharded embedding + `all_to_all` + `embed_norm` (sharded as `[T, H/tp]` if `start == 0`)."""
        if self.causal_chunk_stream:
            from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                resolve_causal_chunks,
            )

            chunks = resolve_causal_chunks(
                input_ids.shape[0] // self.tp_size, self.causal_chunk_spec
            )
            if len(chunks) > 1:
                perm = self._cm_perm_tensor(chunks, input_ids.device, inverse=False)
                input_ids = input_ids.index_select(0, perm)
        return self.entry_op(input_ids.to(torch.int32), self.embed_col)

    def _rope_tables(self, rotary_emb, positions):
        """GPT-J interleaved fp32 ``[T, 128]`` cos and signed-sin tables (two heads per row)."""
        if self.rope_op is not None:
            from sglang_tpu.rank_local import rank_slice

            pos_shard = rank_slice(positions, 0, self.tp_size, self.tp_rank).to(
                torch.int32
            )
            return self.rope_op(pos_shard, self.rope_cos_sin_128)
        cos = rotary_emb.cos.index_select(0, positions).repeat_interleave(2, dim=-1)
        sin = rotary_emb.sin.index_select(0, positions).repeat_interleave(2, dim=-1)
        sin = sin * self.sin_sign
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)

    def _prepare_valid_tokens(self, forward_batch, positions: torch.Tensor):
        if not (self._a1_skip or self._a3_skip):
            return None
        ext_lens = getattr(forward_batch, "extend_seq_lens", None)
        if isinstance(ext_lens, torch.Tensor):
            return ext_lens.sum(dim=0, keepdim=True).to(torch.int32)
        return positions.new_full((1,), positions.shape[0], dtype=torch.int32)

    def forward(
        self,
        decoder,
        positions,
        stream,
        forward_batch,
        exit_last_index: Optional[torch.Tensor] = None,
        captured_by_slot: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Replicated fp32 stream ``[T, H]`` into layer ``start`` -> out of layer ``end - 1``."""
        from sglang_tpu.rank_local import rank_slice

        hidden = self.no_next_norm.shape[0]
        if stream.shape[1] != hidden:
            stream_shard = stream.view(-1, hidden)
        else:
            stream_shard = None
        t_shard = (
            stream_shard.shape[0]
            if stream_shard is not None
            else stream.shape[0] // self.tp_size
        )
        t_total = t_shard * self.tp_size
        chunks: tuple[int, ...] = ()
        rope_chunks = None
        nope_chunks = None
        chunk_metas = None
        inv_cm_perm = None
        if self.causal_chunk_stream:
            from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                build_chunk_rpa_metas_torch,
                resolve_causal_chunks,
            )

            chunks = resolve_causal_chunks(t_shard, self.causal_chunk_spec)
            if len(chunks) > 1:
                inv_cm_perm = self._cm_perm_tensor(chunks, stream.device, inverse=True)
                if stream_shard is None:
                    perm_cm = self._cm_perm_tensor(
                        chunks, stream.device, inverse=False
                    )
                    stream_cm = stream.index_select(0, perm_cm)
                    stream_shard = rank_slice(stream_cm, 0, self.tp_size, self.tp_rank)
                pos_for_rope = positions.index_select(
                    0, self._cm_perm_tensor(chunks, positions.device, inverse=False)
                ) if self.rope_op is not None else positions
                rope = self._rope_tables(decoder.rotary_emb, pos_for_rope)
                if self.rope_op is not None:
                    # Sharded rope returns [T] in chunk-major order; unpermute to contiguous
                    # [T] so slicing [t0:t1] gives exact causal chunk [tp, c_k, 128].
                    rope = (
                        rope[0].index_select(0, inv_cm_perm),
                        rope[1].index_select(0, inv_cm_perm),
                    )
                nope = (
                    (torch.ones_like(rope[0]), torch.zeros_like(rope[1]))
                    if getattr(self, "_has_nope", True)
                    else None
                )
                rope_chunks = []
                nope_chunks = [] if nope is not None else None
                t_off = 0
                for c_k in chunks:
                    c_tok = self.tp_size * c_k
                    rope_chunks.append(
                        (
                            rope[0][t_off : t_off + c_tok].contiguous(),
                            rope[1][t_off : t_off + c_tok].contiguous(),
                        )
                    )
                    if nope_chunks is not None:
                        nope_chunks.append(
                            (
                                nope[0][t_off : t_off + c_tok].contiguous(),
                                nope[1][t_off : t_off + c_tok].contiguous(),
                            )
                        )
                    t_off += c_tok
                chunk_metas = build_chunk_rpa_metas_torch(
                    get_attn_backend().forward_metadata,
                    tp_size=self.tp_size,
                    chunks=chunks,
                )
        if stream_shard is None:
            stream_shard = rank_slice(stream, 0, self.tp_size, self.tp_rank)
        if chunk_metas is None:
            rope = self._rope_tables(decoder.rotary_emb, positions)
            nope = (
                (torch.ones_like(rope[0]), torch.zeros_like(rope[1]))
                if getattr(self, "_has_nope", True)
                else None
            )
        valid_tokens = self._prepare_valid_tokens(forward_batch, positions)
        attn_in = decoder.layers[self.start].input_layernorm(stream_shard)
        attn_in = attn_in.to(torch.bfloat16)
        stats = []
        calib_hists = []
        self._cur_real_tokens_t = self._prepare_eplb_calib_tokens(forward_batch, stream)
        trap_ctx = self._init_nan_trap(
            forward_batch=forward_batch,
            stream=stream,
            stream_shard=stream_shard,
            chunks=chunks,
            t_shard=t_shard,
            t_total=t_total,
        )
        use_agd_out = (
            self.a1_ag_under_d or (chunk_metas is not None)
        ) and self.a1_variant != "v10_sp"
        use_l61_prune = (
            getattr(self, "l61_prune", False)
            and exit_last_index is not None
            and self.ops.sum_stream is not None
            and self.a1_rpa_layout
            and not self.debug_compare
            and not self.eplb_calib_active
            and t_total > 128
            and exit_last_index.shape[0] <= 128
            and hasattr(get_attn_backend(), "write_kv_only")
        )
        for offset, weights in enumerate(self.weights):
            idx = self.start + offset
            layer = decoder.layers[idx]
            if use_l61_prune and idx == self.end - 1:
                out_l61 = self._layer_l61_pruned(
                    layer=layer,
                    weights=weights,
                    rope=rope if layer.self_attn.rotary_emb is not None else nope,
                    stream_shard=stream_shard,
                    attn_in=attn_in,
                    forward_batch=forward_batch,
                    exit_last_index=exit_last_index,
                    t_shard=t_shard,
                    t_total=t_total,
                    valid_tokens=valid_tokens,
                    inv_cm_perm=inv_cm_perm,
                    chunks=chunks if chunk_metas is not None else None,
                    rope_chunks=(
                        rope_chunks
                        if layer.self_attn.rotary_emb is not None
                        else nope_chunks
                    )
                    if chunk_metas is not None
                    else None,
                    chunk_metas=chunk_metas,
                    trap_ctx=trap_ctx,
                    layer_idx=idx,
                )
                self._emit_nan_trap(trap_ctx)
                return out_l61
            next_norm = (
                decoder.layers[idx + 1].input_layernorm.weight
                if idx + 1 < self.end
                else self.no_next_norm
            )
            if self.debug_compare:
                dbg_stream = self._gather(stream_shard)
                if inv_cm_perm is not None:
                    dbg_stream = dbg_stream.index_select(0, inv_cm_perm)
                ref = layer(positions, dbg_stream, forward_batch)
            layer_out = self._layer(
                layer=layer,
                weights=weights,
                rope=rope if layer.self_attn.rotary_emb is not None else nope,
                stream_shard=stream_shard,
                attn_in=attn_in,
                next_norm=next_norm,
                forward_batch=forward_batch,
                valid_tokens=valid_tokens,
                gather_next=use_agd_out and idx + 1 < self.end,
                chunks=chunks if chunk_metas is not None else None,
                rope_chunks=(
                    rope_chunks
                    if layer.self_attn.rotary_emb is not None
                    else nope_chunks
                )
                if chunk_metas is not None
                else None,
                chunk_metas=chunk_metas,
                trap_ctx=trap_ctx,
                layer_idx=idx,
            )
            if self.eplb_calib_active:
                stream_shard, attn_in, layer_hist = layer_out
                calib_hists.append(layer_hist)
            else:
                stream_shard, attn_in = layer_out
            if trap_ctx is not None:
                _record_nan_trap(
                    trap_ctx,
                    idx,
                    4,
                    ((stream_shard, 0),),
                    trap_ctx["shard_real"],
                    trap_ctx["shard_orig"],
                )
            if captured_by_slot is not None:
                slots = decoder._capture_slots_by_layer.get(idx)
                if slots:
                    if self.ops.gather_fp8_stream is not None:
                        snap = self.ops.gather_fp8_stream(
                            stream_shard.to(torch.bfloat16)
                        )
                    else:
                        snap = self._gather(stream_shard.to(torch.bfloat16))
                        if inv_cm_perm is not None:
                            snap = snap.index_select(0, inv_cm_perm)
                    for slot in slots:
                        captured_by_slot[slot] = snap
            if self.debug_compare:
                got_dbg = self._gather(stream_shard)
                if inv_cm_perm is not None:
                    got_dbg = got_dbg.index_select(0, inv_cm_perm)
                stats.append(_diff_stats(ref, got_dbg))
        self._cur_real_tokens_t = None
        if self.eplb_calib_active:
            self._finish_eplb_calib_step(calib_hists)
        if self.debug_compare:
            self.debug_stats.copy_(torch.stack(stats))
            self.debug_count.add_(1)
        if exit_last_index is not None and self.ops.sum_stream is not None:
            eff_orig = torch.where(
                exit_last_index < 0,
                exit_last_index + t_total,
                exit_last_index,
            )
            eff_idx = (
                inv_cm_perm.index_select(0, eff_orig.long()).to(eff_orig.dtype)
                if inv_cm_perm is not None
                else eff_orig
            )
            t_start = self.tp_rank * t_shard
            in_range = (eff_idx >= t_start) & (eff_idx < t_start + t_shard)
            local_idx = (eff_idx - t_start).clamp(0, t_shard - 1)
            local_rows = torch.where(
                in_range.unsqueeze(1),
                stream_shard.index_select(0, local_idx),
                0.0,
            )
            out_exit = self.ops.sum_stream(local_rows)
            if trap_ctx is not None:
                ext_lens = getattr(forward_batch, "extend_seq_lens", None)
                exit_mask = (
                    (ext_lens > 0)
                    if (
                        isinstance(ext_lens, torch.Tensor)
                        and ext_lens.shape == exit_last_index.shape
                    )
                    else torch.ones_like(exit_last_index, dtype=torch.bool)
                )
                _record_nan_trap(
                    trap_ctx,
                    self.end - 1,
                    5,
                    ((out_exit, 0),),
                    exit_mask,
                    eff_orig.to(torch.int32),
                )
                self._emit_nan_trap(trap_ctx)
            return out_exit
        if self.ops.gather_fp8_stream is not None:
            if self.exit_gather_bf16:
                out_full = self.ops.gather_fp8_stream(
                    stream_shard.to(torch.bfloat16)
                ).float()
            else:
                out_full = self.ops.gather_fp8_stream(stream_shard)
        elif self.exit_gather_bf16:
            out_full = self._gather(stream_shard.to(torch.bfloat16)).float()
            if inv_cm_perm is not None:
                out_full = out_full.index_select(0, inv_cm_perm)
        else:
            out_full = self._gather(stream_shard)
            if inv_cm_perm is not None:
                out_full = out_full.index_select(0, inv_cm_perm)
        if trap_ctx is not None:
            _record_nan_trap(
                trap_ctx,
                self.end - 1,
                5,
                ((out_full, 0),),
                trap_ctx["full_real"],
                trap_ctx["orig_full"],
            )
            self._emit_nan_trap(trap_ctx)
        return out_full

    def _layer_l61_pruned(
        self,
        *,
        layer,
        weights,
        rope,
        stream_shard,
        attn_in,
        forward_batch,
        exit_last_index,
        t_shard: int,
        t_total: int,
        valid_tokens=None,
        inv_cm_perm=None,
        chunks=None,
        rope_chunks=None,
        chunk_metas=None,
        trap_ctx=None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Exact L61 pruning: run A1 + KV write for all T rows, then slice Q and residual
        to ``last_token_indices`` (padded to 128) before RPAm -> o_proj -> MoE."""
        from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
            ChunkRpaMetaTorch,
        )
        from sglang_tpu.rank_local import rank_slice

        attn = layer.self_attn
        be = get_attn_backend()
        wqkvg_scale = getattr(weights, "wqkvg_scale_2d", None)
        if wqkvg_scale is None:
            wqkvg_scale = weights.wqkvg_scale[None]
        a1_extra = (valid_tokens,) if self._a1_skip else ()
        q_size = attn.num_heads * 64
        bs = exit_last_index.shape[0]
        ext_lens = forward_batch.extend_seq_lens
        active = ext_lens > 0
        eff_idx = torch.where(active, exit_last_index, 0).clamp(0, t_total - 1)
        idx_128 = torch.nn.functional.pad(eff_idx.long(), (0, 128 - bs))
        q_tok_dim = 1 if self.a1_rpa_layout else 0
        if self.a1_variant == "v10_sp":
            q_pruned, gate_pruned, kv_pruned = self._l61_pruned_ulysses_v10_attn_inputs(
                weights=weights,
                attn_in=attn_in,
                rope=rope,
                chunks=chunks,
                rope_chunks=rope_chunks,
                chunk_metas=chunk_metas,
                forward_batch=forward_batch,
                attn=attn,
                be=be,
                idx_128=idx_128,
                eff_idx=eff_idx,
                active=active,
                bs=bs,
                q_size=q_size,
                t_shard=t_shard,
                inv_cm_perm=inv_cm_perm,
                trap_ctx=trap_ctx,
                layer_idx=layer_idx,
            )
        elif chunk_metas is not None and chunks is not None and rope_chunks is not None:
            if isinstance(attn_in, tuple):
                ag_cur, rest_in = attn_in
                rest_off = 0
            else:
                ag_cur = self.ops.ag_chunk(attn_in[: chunks[0]])
                rest_in = attn_in
                rest_off = chunks[0]
            q_pruned = None
            gate_pruned = None
            kv_pruned_u8 = None
            kv_dtype = None
            c_off = 0
            for k, c_k in enumerate(chunks):
                has_next_c = k + 1 < len(chunks)
                if has_next_c and getattr(self.ops, "a1_chunk_agd", None) is not None:
                    c_next = chunks[k + 1]
                    q_k, gate_k, kv_k, ag_next = self.ops.a1_chunk_agd(
                        ag_cur,
                        rest_in[rest_off : rest_off + c_next],
                        weights.wqkvg,
                        wqkvg_scale,
                        rope_chunks[k][0],
                        rope_chunks[k][1],
                    )
                    rest_off += c_next
                else:
                    if has_next_c:
                        c_next = chunks[k + 1]
                        ag_next = self.ops.ag_chunk(
                            rest_in[rest_off : rest_off + c_next]
                        )
                        rest_off += c_next
                    q_k, gate_k, kv_k = self.ops.a1_chunk(
                        ag_cur,
                        weights.wqkvg,
                        wqkvg_scale,
                        rope_chunks[k][0],
                        rope_chunks[k][1],
                    )
                be.write_kv_only(kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k])
                c_tok = self.tp_size * c_k
                if trap_ctx is not None:
                    _record_nan_trap(
                        trap_ctx,
                        layer_idx,
                        1,
                        ((q_k, q_tok_dim), (kv_k, 0)),
                        trap_ctx["full_real"][c_off : c_off + c_tok],
                        trap_ctx["orig_full"][c_off : c_off + c_tok],
                    )
                loc_idx = (idx_128 - c_off).clamp(0, c_tok - 1)
                q_sel = q_k.index_select(1, loc_idx)
                gate_sel = gate_k.reshape(-1, q_size).index_select(0, loc_idx)
                kv_dtype = kv_k.dtype
                kv_sel_u8 = kv_k.view(torch.uint8).index_select(0, loc_idx)
                if k == 0:
                    q_pruned = q_sel
                    gate_pruned = gate_sel
                    kv_pruned_u8 = kv_sel_u8
                else:
                    in_c = (idx_128 >= c_off) & (idx_128 < c_off + c_tok)
                    in_q = in_c.view(1, -1, *([1] * (q_sel.ndim - 2)))
                    in_kv = in_c.view(-1, *([1] * (kv_sel_u8.ndim - 1)))
                    q_pruned = torch.where(in_q, q_sel, q_pruned)
                    gate_pruned = torch.where(in_c[:, None], gate_sel, gate_pruned)
                    kv_pruned_u8 = torch.where(in_kv, kv_sel_u8, kv_pruned_u8)
                c_off += c_tok
                if k + 1 < len(chunks):
                    ag_cur = ag_next
            assert q_pruned is not None and gate_pruned is not None and kv_pruned_u8 is not None
            kv_pruned = kv_pruned_u8.view(kv_dtype)
        else:
            if self.a1_variant in _HYBRID_A1_VARIANTS:
                q_all, gate_all, kv_all = self.ops.a1(
                    attn_in,
                    weights.wqkvg,
                    wqkvg_scale,
                    weights.wkv,
                    weights.wkv_scale,
                    weights.a1_gain,
                    weights.a1_inv_gain,
                    rope[0],
                    rope[1],
                    *a1_extra,
                )
            elif isinstance(attn_in, tuple):
                q_all, gate_all, kv_all = self.ops.a1_agd(
                    *attn_in, weights.wqkvg, wqkvg_scale, rope[0], rope[1]
                )
            elif self.tp16_owner_a1 and attn_in.shape[0] == 512:
                q_all, gate_all, kv_all = self.ops.a1_owner(
                    attn_in,
                    weights.wqkvg,
                    weights.wqkvg_scale[:, None, :],
                    rank_slice(rope[0], 0, self.tp_size, self.tp_rank),
                    rank_slice(rope[1], 0, self.tp_size, self.tp_rank),
                )
            else:
                wqkvg = weights.wqkvg
                if self.tp16_owner_a1:
                    wqkvg = rank_slice(wqkvg, 0, self.tp_size, self.tp_rank).squeeze(0)
                    wqkvg_scale = rank_slice(
                        weights.wqkvg_scale, 0, self.tp_size, self.tp_rank
                    )
                q_all, gate_all, kv_all = self.ops.a1(
                    attn_in, wqkvg, wqkvg_scale, rope[0], rope[1], *a1_extra
                )
            be.write_kv_only(kv_all, attn.attn, forward_batch)
            if trap_ctx is not None:
                a1_rows = q_all.shape[q_tok_dim]
                a1_mask, a1_orig = (
                    (trap_ctx["shard_real"], trap_ctx["shard_orig"])
                    if a1_rows == t_shard
                    else (trap_ctx["full_real"], trap_ctx["orig_full"])
                )
                _record_nan_trap(
                    trap_ctx, layer_idx, 1, ((q_all, q_tok_dim), (kv_all, 0)), a1_mask, a1_orig
                )
            q_pruned = q_all.index_select(1, idx_128)
            gate_pruned = gate_all.index_select(0, idx_128)
            kv_pruned = kv_all.index_select(0, idx_128)

        cm_idx = (
            inv_cm_perm.index_select(0, eff_idx.long())
            if inv_cm_perm is not None
            else eff_idx.long()
        )
        t_start = self.tp_rank * t_shard
        in_range = (cm_idx >= t_start) & (cm_idx < t_start + t_shard) & active
        local_idx = (cm_idx - t_start).clamp(0, t_shard - 1)
        local_rows = torch.where(
            in_range.unsqueeze(1),
            stream_shard.index_select(0, local_idx),
            0.0,
        )
        residual_bs = self.ops.sum_stream(local_rows)
        residual_128 = torch.nn.functional.pad(residual_bs, (0, 0, 0, 128 - bs))

        md = be.forward_metadata
        zero_i32 = md.kv_lens.new_zeros((1,), dtype=torch.int32)
        q_lens_pruned = active.to(torch.int32)
        cu_q_lens_pruned = torch.cat(
            [zero_i32, torch.cumsum(q_lens_pruned, dim=0, dtype=torch.int32)], dim=0
        )
        n_act = q_lens_pruned.sum()
        dist_pruned = torch.stack([zero_i32[0], zero_i32[0], n_act]).to(torch.int32)
        pruned_meta = ChunkRpaMetaTorch(
            kv_lens=md.kv_lens,
            cu_q_lens=cu_q_lens_pruned,
            cu_kv_lens=md.cu_kv_lens,
            page_indices=md.page_indices,
            swa_page_indices=md.swa_page_indices,
            swa_group_page_indices=md.swa_group_page_indices,
            distribution=dist_pruned,
        )
        out_128 = be.forward_prepared(
            q_pruned, kv_pruned, attn.attn, forward_batch, chunk_meta=pruned_meta
        )
        attn_out = attn._per_head_normalize(out_128).to(torch.bfloat16)
        gated = torch.sigmoid(gate_pruned) * attn_out
        attn_proj = attn._forward_o_proj(gated, forward_batch)
        if trap_ctx is not None:
            l61_mask = (
                torch.arange(128, device=out_128.device) < bs
            ) & torch.nn.functional.pad(active, (0, 128 - bs))
            l61_orig = idx_128.to(torch.int32)
            _record_nan_trap(trap_ctx, layer_idx, 2, ((out_128, 0),), l61_mask, l61_orig)
            _record_nan_trap(trap_ctx, layer_idx, 3, ((attn_proj, 0),), l61_mask, l61_orig)
        if getattr(self.ops, "moe", None) is not None or getattr(
            self, "eplb_moe_op", None
        ) is not None:
            from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                resolve_causal_chunks,
            )

            moe = layer.mlp
            experts = moe.experts
            is_fp8_expert = experts.w13_weight.dtype == torch.float8_e4m3fn
            t_shard_moe = 128
            t_total_moe = t_shard_moe * self.tp_size
            valid_mask_128 = (
                (torch.arange(128, device=attn_proj.device) < bs)
                & torch.nn.functional.pad(active, (0, 128 - bs))
            ).unsqueeze(1)
            attn_128 = torch.where(valid_mask_128, attn_proj, 0)
            attn_pad = torch.nn.functional.pad(
                attn_128, (0, 0, 0, t_total_moe - 128)
            )
            res_pad = torch.nn.functional.pad(
                residual_128, (0, 0, 0, t_total_moe - 128)
            )
            inv_128 = None
            if self.causal_chunk_stream:
                chunks_128 = resolve_causal_chunks(t_shard_moe, self.causal_chunk_spec)
                if len(chunks_128) > 1:
                    perm_128 = self._cm_perm_tensor(
                        chunks_128, attn_pad.device, inverse=False
                    )
                    inv_128 = self._cm_perm_tensor(
                        chunks_128, attn_pad.device, inverse=True
                    )
                    attn_pad = attn_pad.index_select(0, perm_128)
                    res_pad = res_pad.index_select(0, perm_128)
            attn_shard_61 = rank_slice(attn_pad, 0, self.tp_size, self.tp_rank)
            res_shard_61 = rank_slice(res_pad, 0, self.tp_size, self.tp_rank)
            lw_bufs = getattr(weights, "_buffers", {})
            b1_alpha_arg = lw_bufs.get("b1_vecs")
            if b1_alpha_arg is None:
                b1_alpha_arg = layer.post_attention_residual_gate.alpha
            b1_bias_arg = lw_bufs.get("b1_bias_t")
            if b1_bias_arg is None:
                b1_bias_arg = moe.e_score_correction_bias
            b2_wpsc_arg = lw_bufs.get("b2_vecs")
            if b2_wpsc_arg is None:
                b2_wpsc_arg = weights.wpool_scale
            moe_args_61 = (
                attn_shard_61,
                res_shard_61,
                b1_alpha_arg,
                layer.post_attention_residual_gate.beta,
                layer.pre_feedforward_layernorm.weight,
                weights.wrouter,
                b1_bias_arg,
                weights.wpool,
                b2_wpsc_arg,
                moe.pre_expert_norm.weight,
                experts.w13_weight,
                experts.w2_weight,
                experts.w13_weight_scale,
                experts.w2_weight_scale,
                moe.post_expert_norm.weight,
                self.ep_tp2_group_offset
                if self.ep_tp2_mode
                else experts._tpu_fused_rs_group_offset,
                weights.wunpool,
                weights.wunpool_scale,
                layer.post_feedforward_layernorm.weight,
                layer.post_feedforward_residual_gate.alpha,
                layer.post_feedforward_residual_gate.beta,
                self.no_next_norm,
            )
            vt_61 = (
                (residual_128.new_full((1,), bs, dtype=torch.int32),)
                if getattr(self, "eplb_moe_op", None) is not None
                and self._cur_real_tokens_t is not None
                else ()
            )
            moe_out_shard, _ = self._moe_op(
                is_fp8_expert=is_fp8_expert, gather_next=False
            )(*moe_args_61, *vt_61)
            out_idx = torch.arange(bs, device=moe_out_shard.device)
            if inv_128 is not None:
                out_idx = inv_128.index_select(0, out_idx)
            t_lo = self.tp_rank * t_shard_moe
            in_r = (out_idx >= t_lo) & (out_idx < t_lo + t_shard_moe)
            loc_i = (out_idx - t_lo).clamp(0, t_shard_moe - 1)
            loc_rows = torch.where(
                in_r.unsqueeze(1),
                moe_out_shard.index_select(0, loc_i),
                0.0,
            )
            out_l61 = self.ops.sum_stream(loc_rows)
            if trap_ctx is not None:
                _record_nan_trap(
                    trap_ctx, layer_idx, 4, ((out_l61, 0),), active, eff_idx.to(torch.int32)
                )
            return out_l61

        residual_128 = layer.post_attention_residual_gate(
            residual_128, layer.post_attention_layernorm(attn_proj)
        )
        router_in = layer.pre_feedforward_layernorm(residual_128)
        ffn_out = layer.mlp(router_in.to(torch.bfloat16), router_in)
        residual_128 = layer.post_feedforward_residual_gate(
            residual_128, layer.post_feedforward_layernorm(ffn_out)
        )
        out_l61 = residual_128[:bs]
        if trap_ctx is not None:
            _record_nan_trap(
                trap_ctx, layer_idx, 4, ((out_l61, 0),), active, eff_idx.to(torch.int32)
            )
        return out_l61

    def _forward_rpa_raw_packed(self, q, kv, attn_layer, forward_batch, chunk_meta=None):
        """Runs packed-KV RPA v3 and returns raw ``bf16[T, Hq, 128]`` before ``_unpack_out_heads``."""
        from sglang_tpu.attention_pallas_rpa_base import (
            _fused_kv_operands,
            _pack_qkv_heads,
        )

        be = get_attn_backend()
        md = chunk_meta if chunk_meta is not None else be.forward_metadata
        if (
            be.kv_cache_quant_dtype is None
            or (chunk_meta is None and md.cp_write_segments is not None)
            or be._dcp != 1
            or attn_layer.k_scale_float not in (None, 1.0)
            or attn_layer.v_scale_float not in (None, 1.0)
        ):
            raise NotImplementedError(
                "A3 RPA packed input needs a scale-1 packed fp8 pool"
            )
        kv_cache, kv_layer_idx = _fused_kv_operands(
            be.token_to_kv_pool, attn_layer.layer_id
        )
        sliding_window = attn_layer.sliding_window_size
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        elif sliding_window is not None:
            sliding_window += 1  # RPA counts the current token, SGLang does not.
        is_swa = sliding_window is not None
        if chunk_meta is not None:
            if is_swa and md.swa_page_indices is not None:
                group = be._swa_layer_groups.get(attn_layer.layer_id, 0)
                swa_pages = getattr(md, "swa_group_pages", None)
                page_indices = (
                    md.swa_page_indices
                    if group == 0 or md.swa_group_page_indices is None
                    else (
                        swa_pages[group - 1]
                        if swa_pages is not None
                        else md.swa_group_page_indices[group - 1]
                    )
                )
            else:
                page_indices = md.page_indices
            kv_lens = md.kv_lens
            decode_query_size = 1
        else:
            page_indices = be._layer_page_indices(
                attn_layer.layer_id, is_swa_layer=is_swa
            )
            kv_lens = be._kv_lens(forward_batch)
            decode_query_size = md.decode_query_size
        if self.a1_rpa_layout:
            op = getattr(be, "_prepared_op", None)
            if op is None:
                raise NotImplementedError("A1_RPA_LAYOUT requires backend._prepared_op")
            q_in, k_in, v_in = q, kv, kv
        else:
            op = be._op
            num_tokens = q.shape[0]
            q_3d = q.view(num_tokens, attn_layer.tp_q_head_num, attn_layer.qk_head_dim)
            k_3d = kv[:, : self.kv_size].view(
                num_tokens, attn_layer.tp_k_head_num, attn_layer.qk_head_dim
            )
            v_3d = kv[:, self.kv_size :].view(
                num_tokens, attn_layer.tp_v_head_num, attn_layer.v_head_dim
            )
            q_in, k_in, v_in = _pack_qkv_heads(
                q_3d, k_3d, v_3d, attn_layer.tp_k_head_num
            )
        return op(
            q_in,
            k_in,
            v_in,
            kv_cache,
            kv_lens,
            page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
            None,
            float(attn_layer.scaling),
            sliding_window,
            attn_layer.logit_cap if attn_layer.logit_cap else None,
            1.0,
            1.0,
            decode_query_size,
            use_causal_mask=True,
            layer_idx=kv_layer_idx,
        )

    def _forward_rpa_unpacked_fp8(
        self, q, kv, attn_layer, forward_batch, chunk_meta=None
    ):
        """Runs unpacked scale-1 FP8-KV RPA v3 on ``q [T, Hq*d]`` and ``kv [T, 2*Hkv*d]``."""
        from sglang_tpu.attention_pallas_rpa_base import _fused_kv_operands

        be = get_attn_backend()
        md = chunk_meta if chunk_meta is not None else be.forward_metadata
        if (
            be.kv_cache_quant_dtype is None
            or (chunk_meta is None and md.cp_write_segments is not None)
            or be._dcp != 1
            or attn_layer.k_scale_float not in (None, 1.0)
            or attn_layer.v_scale_float not in (None, 1.0)
        ):
            raise NotImplementedError(
                "Unpacked chunked FP8 RPA needs a scale-1 fp8 KV pool with dcp=1"
            )
        kv_cache, kv_layer_idx = _fused_kv_operands(
            be.token_to_kv_pool, attn_layer.layer_id
        )
        sliding_window = attn_layer.sliding_window_size
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        elif sliding_window is not None:
            sliding_window += 1
        is_swa = sliding_window is not None
        if chunk_meta is not None:
            if is_swa and md.swa_page_indices is not None:
                group = be._swa_layer_groups.get(attn_layer.layer_id, 0)
                swa_pages = getattr(md, "swa_group_pages", None)
                page_indices = (
                    md.swa_page_indices
                    if group == 0 or md.swa_group_page_indices is None
                    else (
                        swa_pages[group - 1]
                        if swa_pages is not None
                        else md.swa_group_page_indices[group - 1]
                    )
                )
            else:
                page_indices = md.page_indices
            kv_lens = md.kv_lens
            decode_query_size = 1
        else:
            page_indices = be._layer_page_indices(
                attn_layer.layer_id, is_swa_layer=is_swa
            )
            kv_lens = be._kv_lens(forward_batch)
            decode_query_size = md.decode_query_size
        num_tokens = q.shape[0]
        q_3d = q.view(num_tokens, attn_layer.tp_q_head_num, attn_layer.qk_head_dim)
        k_3d = (
            kv[:, : self.kv_size]
            .contiguous()
            .view(num_tokens, attn_layer.tp_k_head_num, attn_layer.qk_head_dim)
        )
        v_3d = (
            kv[:, self.kv_size :]
            .contiguous()
            .view(num_tokens, attn_layer.tp_v_head_num, attn_layer.v_head_dim)
        )
        op = be._kernel_op(forward_batch, True)
        out = op(
            q_3d,
            k_3d,
            v_3d,
            kv_cache,
            kv_lens,
            page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
            None,
            float(attn_layer.scaling),
            sliding_window,
            attn_layer.logit_cap if attn_layer.logit_cap else None,
            1.0,
            1.0,
            decode_query_size,
            use_causal_mask=True,
            layer_idx=kv_layer_idx,
        )
        out = out.reshape(num_tokens, attn_layer.tp_q_head_num, -1)
        out = out[..., : attn_layer.v_head_dim]
        return out.reshape(num_tokens, attn_layer.tp_q_head_num * attn_layer.v_head_dim)

    def _layer(
        self,
        *,
        layer,
        weights,
        rope,
        stream_shard,
        attn_in,
        next_norm,
        forward_batch,
        valid_tokens=None,
        gather_next=False,
        chunks=None,
        rope_chunks=None,
        chunk_metas=None,
        trap_ctx=None,
        layer_idx: int = -1,
    ):
        """``attn_in`` is a tensor, or (AG_UNDER_D) the previous layer's (gathered A1 part 0,
        remaining rows); ``gather_next`` returns this layer's next input in that form."""
        from sglang_tpu.rank_local import rank_slice

        attn, moe = layer.self_attn, layer.mlp
        # q is already QK-normed, scaled and rotated; k/v are fp8 (scale 1), as the
        # fp8 KV pool stores them.
        wqkvg_scale = getattr(weights, "wqkvg_scale_2d", None)
        if wqkvg_scale is None:
            wqkvg_scale = weights.wqkvg_scale[None]
        wo_scale = getattr(weights, "wo_scale_2d", None)
        if wo_scale is None:
            wo_scale = weights.wo_scale[None]
        wo = weights.wo if weights.wo.shape == weights.wo_shape else weights.wo.view(weights.wo_shape)
        a1_extra = (valid_tokens,) if self._a1_skip else ()
        a3_extra = (valid_tokens,) if self._a3_skip else ()
        q_tok_dim = 1 if self.a1_rpa_layout else 0
        attn_out_shard = None
        if self.a1_variant == "v10_sp":
            attn_out_shard = self._forward_ulysses_v10(
                attn=attn,
                weights=weights,
                wo=wo,
                wo_scale=wo_scale,
                attn_in=attn_in,
                rope=rope,
                chunks=chunks,
                rope_chunks=rope_chunks,
                chunk_metas=chunk_metas,
                forward_batch=forward_batch,
                trap_ctx=trap_ctx,
                layer_idx=layer_idx,
            )
        elif (
            chunk_metas is not None
            and chunks is not None
            and rope_chunks is not None
            and self.a3_kdeep_wo >= 2
            and len(chunks) >= 2
            and not self._a3_skip
            and self.ops.a3_kdeep_ccs is not None
        ):
            attn_out_shard = self._forward_ccs_kdeep_a3(
                attn=attn,
                weights=weights,
                wqkvg_scale=wqkvg_scale,
                wo=wo,
                wo_scale=wo_scale,
                attn_in=attn_in,
                chunks=chunks,
                rope_chunks=rope_chunks,
                chunk_metas=chunk_metas,
                forward_batch=forward_batch,
                trap_ctx=trap_ctx,
                layer_idx=layer_idx,
            )
        elif chunk_metas is not None and chunks is not None and rope_chunks is not None:
            if isinstance(attn_in, tuple):
                ag_cur, rest_in = attn_in
                rest_off = 0
            else:
                ag_cur = self.ops.ag_chunk(attn_in[: chunks[0]])
                rest_in = attn_in
                rest_off = chunks[0]
            be = None if self.a3_rpa_out else get_attn_backend()
            use_chunk_pro = (
                self.ops.a3_prologue_chunk is not None
                and self.ops.a3_wo is not None
            )
            q_parts = []
            out_parts = []
            gate_parts = []
            out_tail: tuple[int, ...] = ()
            q_size = attn.num_heads * 64
            c_off = 0
            for k, c_k in enumerate(chunks):
                has_next_c = k + 1 < len(chunks)
                if has_next_c and getattr(self.ops, "a1_chunk_agd", None) is not None:
                    c_next = chunks[k + 1]
                    q_k, gate_k, kv_k, ag_next = self.ops.a1_chunk_agd(
                        ag_cur,
                        rest_in[rest_off : rest_off + c_next],
                        weights.wqkvg,
                        wqkvg_scale,
                        rope_chunks[k][0],
                        rope_chunks[k][1],
                    )
                    rest_off += c_next
                else:
                    if has_next_c:
                        c_next = chunks[k + 1]
                        ag_next = self.ops.ag_chunk(
                            rest_in[rest_off : rest_off + c_next]
                        )
                        rest_off += c_next
                    q_k, gate_k, kv_k = self.ops.a1_chunk(
                        ag_cur,
                        weights.wqkvg,
                        wqkvg_scale,
                        rope_chunks[k][0],
                        rope_chunks[k][1],
                    )
                if self.a3_rpa_out:
                    out_k = self._forward_rpa_raw_packed(
                        q_k, kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k]
                    )
                elif self.a1_rpa_layout:
                    out_k = be.forward_prepared(
                        q_k, kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k]
                    )
                else:
                    out_k = self._forward_rpa_unpacked_fp8(
                        q_k, kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k]
                    )
                out_tok = (
                    out_k.squeeze(0)
                    if (out_k.ndim > 2 and out_k.shape[0] == 1)
                    else out_k
                )
                c_tok = self.tp_size * c_k
                if trap_ctx is not None:
                    c_real = trap_ctx["full_real"][c_off : c_off + c_tok]
                    c_orig = trap_ctx["orig_full"][c_off : c_off + c_tok]
                    _record_nan_trap(
                        trap_ctx, layer_idx, 1, ((q_k, q_tok_dim), (kv_k, 0)), c_real, c_orig
                    )
                    _record_nan_trap(
                        trap_ctx, layer_idx, 2, ((out_tok, 0),), c_real, c_orig
                    )
                c_off += c_tok
                if use_chunk_pro:
                    if not self.fused_glue and not self.a3_rpa_out:
                        out_tok = attn._per_head_normalize(out_tok).to(torch.bfloat16)
                    q_parts.append(self.ops.a3_prologue_chunk(out_tok, gate_k))
                else:
                    out_tail = out_tok.shape[1:]
                    out_parts.append(out_tok.view(self.tp_size, c_k, *out_tail))
                    gate_parts.append(gate_k.view(self.tp_size, c_k, q_size))
                if k + 1 < len(chunks):
                    ag_cur = ag_next
            if use_chunk_pro:
                if (
                    self.ops.a3_wo_2chunk is not None
                    and len(q_parts) == 2
                    and not a3_extra
                ):
                    attn_out_shard = self.ops.a3_wo_2chunk(
                        q_parts[0], q_parts[1], wo, wo_scale
                    )
                elif (
                    self.ops.a3_wo_3chunk is not None
                    and len(q_parts) == 3
                    and not a3_extra
                ):
                    attn_out_shard = self.ops.a3_wo_3chunk(
                        q_parts[0], q_parts[1], q_parts[2], wo, wo_scale
                    )
                else:
                    q_3d = torch.cat(q_parts, dim=1)
                    attn_out_shard = self.ops.a3_wo(q_3d, wo, wo_scale, *a3_extra)
            else:
                out = torch.cat(out_parts, dim=1).reshape(-1, *out_tail)
                gate = torch.cat(gate_parts, dim=1).reshape(-1, q_size)
        else:
            if self.a1_variant in _HYBRID_A1_VARIANTS:
                q, gate, kv = self.ops.a1(
                    attn_in,
                    weights.wqkvg,
                    wqkvg_scale,
                    weights.wkv,
                    weights.wkv_scale,
                    weights.a1_gain,
                    weights.a1_inv_gain,
                    rope[0],
                    rope[1],
                    *a1_extra,
                )
            elif isinstance(attn_in, tuple):  # AG_UNDER_D
                q, gate, kv = self.ops.a1_agd(
                    *attn_in, weights.wqkvg, wqkvg_scale, rope[0], rope[1]
                )
            elif self.tp16_owner_a1 and attn_in.shape[0] == 512:
                q, gate, kv = self.ops.a1_owner(
                    attn_in,
                    weights.wqkvg,
                    weights.wqkvg_scale[:, None, :],
                    rank_slice(rope[0], 0, self.tp_size, self.tp_rank),
                    rank_slice(rope[1], 0, self.tp_size, self.tp_rank),
                )
            else:
                wqkvg = weights.wqkvg
                if self.tp16_owner_a1:
                    wqkvg = rank_slice(wqkvg, 0, self.tp_size, self.tp_rank).squeeze(0)
                    wqkvg_scale = rank_slice(
                        weights.wqkvg_scale, 0, self.tp_size, self.tp_rank
                    )
                q, gate, kv = self.ops.a1(
                    attn_in, wqkvg, wqkvg_scale, rope[0], rope[1], *a1_extra
                )
            if trap_ctx is not None:
                a1_rows = q.shape[q_tok_dim]
                a1_mask, a1_orig = (
                    (trap_ctx["shard_real"], trap_ctx["shard_orig"])
                    if a1_rows == trap_ctx["t_shard"]
                    else (trap_ctx["full_real"], trap_ctx["orig_full"])
                )
                _record_nan_trap(
                    trap_ctx, layer_idx, 1, ((q, q_tok_dim), (kv, 0)), a1_mask, a1_orig
                )
            if self.a3_rpa_out:
                out = self._forward_rpa_raw_packed(q, kv, attn.attn, forward_batch)
            elif self.a1_rpa_layout:
                out = get_attn_backend().forward_prepared(q, kv, attn.attn, forward_batch)
            else:
                k = kv[:, : self.kv_size].to(torch.bfloat16)
                v = kv[:, self.kv_size :].to(torch.bfloat16)
                out = attn.attn(q, k, v, forward_batch)
            if trap_ctx is not None:
                out_chk = (
                    out.squeeze(0) if (out.ndim > 2 and out.shape[0] == 1) else out
                )
                _record_nan_trap(
                    trap_ctx,
                    layer_idx,
                    2,
                    ((out_chk, 0),),
                    trap_ctx["full_real"],
                    trap_ctx["orig_full"],
                )
        if attn_out_shard is None:
            if not self.fused_glue and not self.a3_rpa_out:
                out = attn._per_head_normalize(out).to(torch.bfloat16)
            attn_out_shard = self.ops.a3(out, gate, wo, wo_scale, *a3_extra)
        if trap_ctx is not None:
            _record_nan_trap(
                trap_ctx,
                layer_idx,
                3,
                ((attn_out_shard, 0),),
                trap_ctx["shard_real"],
                trap_ctx["shard_orig"],
            )
        experts = moe.experts
        is_fp8_expert = experts.w13_weight.dtype == torch.float8_e4m3fn
        lw_bufs = getattr(weights, "_buffers", {})
        b1_alpha_arg = lw_bufs.get("b1_vecs")
        if b1_alpha_arg is None:
            b1_alpha_arg = layer.post_attention_residual_gate.alpha
        b1_bias_arg = lw_bufs.get("b1_bias_t")
        if b1_bias_arg is None:
            b1_bias_arg = moe.e_score_correction_bias
        b2_wpsc_arg = lw_bufs.get("b2_vecs")
        if b2_wpsc_arg is None:
            b2_wpsc_arg = weights.wpool_scale
        moe_args = (
            attn_out_shard,
            stream_shard,
            b1_alpha_arg,
            layer.post_attention_residual_gate.beta,
            layer.pre_feedforward_layernorm.weight,
            weights.wrouter,
            b1_bias_arg,
            weights.wpool,
            b2_wpsc_arg,
            moe.pre_expert_norm.weight,
            experts.w13_weight,
            experts.w2_weight,
            experts.w13_weight_scale,
            experts.w2_weight_scale,
            moe.post_expert_norm.weight,
            self.ep_tp2_group_offset
            if self.ep_tp2_mode
            else experts._tpu_fused_rs_group_offset,
            weights.wunpool,
            weights.wunpool_scale,
            layer.post_feedforward_layernorm.weight,
            layer.post_feedforward_residual_gate.alpha,
            layer.post_feedforward_residual_gate.beta,
            next_norm,
        )
        if gather_next:
            stream_shard, gathered, rest = self._moe_op(
                is_fp8_expert=is_fp8_expert, gather_next=True
            )(*moe_args, *self._moe_real_tokens_arg())
            return stream_shard, (gathered, rest)
        return self._moe_op(is_fp8_expert=is_fp8_expert, gather_next=False)(
            *moe_args, *self._moe_real_tokens_arg()
        )

    def _prepare_ulysses_chunks(
        self, weights, attn_in, rope, chunks, rope_chunks, chunk_metas
    ):
        """Returns `(eff_chunks, eff_rope_chunks, eff_metas, in_slices, wqkvg_sc3d, w_gate, w_gate_sc)`."""
        if chunk_metas is not None and chunks is not None and rope_chunks is not None:
            eff_chunks = chunks
            eff_rope_chunks = rope_chunks
            eff_metas = chunk_metas
        else:
            eff_chunks = (attn_in.shape[0],)
            eff_rope_chunks = (rope,)
            eff_metas = (None,)
        in_slices = []
        r_off = 0
        for c_k in eff_chunks:
            in_slices.append(attn_in[r_off : r_off + c_k])
            r_off += c_k
        wqkvg_sc3d = getattr(weights, "wqkvg_scale_3d", None)
        if wqkvg_sc3d is None:
            wqkvg_sc3d = weights.wqkvg_scale[:, None, :]
        w_gate_sc = getattr(weights, "w_gate_scale_2d", None)
        if w_gate_sc is None:
            w_gate_sc = weights.w_gate_scale[None]
        return (
            eff_chunks,
            eff_rope_chunks,
            eff_metas,
            in_slices,
            wqkvg_sc3d,
            weights.w_gate,
            w_gate_sc,
        )

    def _forward_ulysses_v10(
        self,
        *,
        attn,
        weights,
        wo,
        wo_scale,
        attn_in,
        rope,
        chunks,
        rope_chunks,
        chunk_metas,
        forward_batch,
        trap_ctx=None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Static-FP8 Ulysses sequence-parallel CCS (and single-chunk fallback) path."""
        be = get_attn_backend()
        (
            eff_chunks,
            eff_rope_chunks,
            eff_metas,
            in_slices,
            wqkvg_sc3d,
            w_gate,
            w_gate_sc,
        ) = self._prepare_ulysses_chunks(
            weights, attn_in, rope, chunks, rope_chunks, chunk_metas
        )

        q3_cur, kv3_cur, x_fp8_cur = self.ops.a1_ulysses_qkv_chunk(
            in_slices[0],
            weights.wqkvg,
            wqkvg_sc3d,
            eff_rope_chunks[0][0],
            eff_rope_chunks[0][1],
        )
        q_parts = []
        attn_fp8_prev = None
        gate_prev = None
        c_off = 0
        for k, c_k in enumerate(eff_chunks):
            has_next = k + 1 < len(eff_chunks)
            q_k, kv_k = self.ops.a1_ulysses_a2a_pack_chunk(q3_cur, kv3_cur)
            gate_k = self.ops.a1_ulysses_gate_chunk(x_fp8_cur, w_gate, w_gate_sc)
            if attn_fp8_prev is not None:
                q_parts.append(
                    self.ops.a3_ulysses_gate_fuse_chunk(attn_fp8_prev, gate_prev)
                )
                attn_fp8_prev = None
            if has_next:
                q3_next, kv3_next, x_fp8_next = self.ops.a1_ulysses_qkv_chunk(
                    in_slices[k + 1],
                    weights.wqkvg,
                    wqkvg_sc3d,
                    eff_rope_chunks[k + 1][0],
                    eff_rope_chunks[k + 1][1],
                )
            out_k = be.forward_prepared(
                q_k, kv_k, attn.attn, forward_batch, chunk_meta=eff_metas[k]
            )
            out_tok = (
                out_k.squeeze(0)
                if (out_k.ndim > 2 and out_k.shape[0] == 1)
                else out_k
            )
            if trap_ctx is not None:
                c_tok = self.tp_size * c_k
                c_real = trap_ctx["full_real"][c_off : c_off + c_tok]
                c_orig = trap_ctx["orig_full"][c_off : c_off + c_tok]
                _record_nan_trap(
                    trap_ctx, layer_idx, 1, ((q_k, 1), (kv_k, 0)), c_real, c_orig
                )
                _record_nan_trap(
                    trap_ctx, layer_idx, 2, ((out_tok, 0),), c_real, c_orig
                )
                c_off += c_tok
            attn_fp8_k = self.ops.a3_ulysses_rev_a2a_chunk(out_tok)
            if has_next:
                attn_fp8_prev = attn_fp8_k
                gate_prev = gate_k
                q3_cur, kv3_cur, x_fp8_cur = q3_next, kv3_next, x_fp8_next
            else:
                q_parts.append(self.ops.a3_ulysses_gate_fuse_chunk(attn_fp8_k, gate_k))

        if len(q_parts) == 2 and self.ops.a3_ulysses_wo_2chunk is not None:
            return self.ops.a3_ulysses_wo_2chunk(q_parts[0], q_parts[1], wo, wo_scale)
        q_all_fp8 = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=0)
        return self.ops.a3_ulysses_wo(q_all_fp8, wo, wo_scale)

    def _l61_pruned_ulysses_v10_attn_inputs(
        self,
        *,
        weights,
        attn_in,
        rope,
        chunks,
        rope_chunks,
        chunk_metas,
        forward_batch,
        attn,
        be,
        idx_128,
        eff_idx,
        active,
        bs: int,
        q_size: int,
        t_shard: int,
        inv_cm_perm,
        trap_ctx=None,
        layer_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Writes KV & gathers pruned L61 `(q_pruned, gate_pruned, kv_pruned)` for `v10_sp`."""
        (
            eff_chunks,
            eff_rope_chunks,
            eff_metas,
            in_slices,
            wqkvg_sc3d,
            w_gate,
            w_gate_sc,
        ) = self._prepare_ulysses_chunks(
            weights, attn_in, rope, chunks, rope_chunks, chunk_metas
        )

        q3_cur, kv3_cur, x_fp8_cur = self.ops.a1_ulysses_qkv_chunk(
            in_slices[0],
            weights.wqkvg,
            wqkvg_sc3d,
            eff_rope_chunks[0][0],
            eff_rope_chunks[0][1],
        )
        q_pruned = None
        kv_pruned_u8 = None
        kv_dtype = None
        gate_parts = []
        c_off = 0
        for k, c_k in enumerate(eff_chunks):
            has_next = k + 1 < len(eff_chunks)
            q_k, kv_k = self.ops.a1_ulysses_a2a_pack_chunk(q3_cur, kv3_cur)
            gate_parts.append(self.ops.a1_ulysses_gate_chunk(x_fp8_cur, w_gate, w_gate_sc))
            if has_next:
                q3_cur, kv3_cur, x_fp8_cur = self.ops.a1_ulysses_qkv_chunk(
                    in_slices[k + 1],
                    weights.wqkvg,
                    wqkvg_sc3d,
                    eff_rope_chunks[k + 1][0],
                    eff_rope_chunks[k + 1][1],
                )
            be.write_kv_only(kv_k, attn.attn, forward_batch, chunk_meta=eff_metas[k])
            c_tok = self.tp_size * c_k
            if trap_ctx is not None:
                _record_nan_trap(
                    trap_ctx,
                    layer_idx,
                    1,
                    ((q_k, 1), (kv_k, 0)),
                    trap_ctx["full_real"][c_off : c_off + c_tok],
                    trap_ctx["orig_full"][c_off : c_off + c_tok],
                )
            loc_idx = (idx_128 - c_off).clamp(0, c_tok - 1)
            q_sel = q_k.index_select(1, loc_idx)
            kv_dtype = kv_k.dtype
            kv_sel_u8 = kv_k.view(torch.uint8).index_select(0, loc_idx)
            if k == 0:
                q_pruned = q_sel
                kv_pruned_u8 = kv_sel_u8
            else:
                in_c = (idx_128 >= c_off) & (idx_128 < c_off + c_tok)
                in_q = in_c.view(1, -1, *([1] * (q_sel.ndim - 2)))
                in_kv = in_c.view(-1, *([1] * (kv_sel_u8.ndim - 1)))
                q_pruned = torch.where(in_q, q_sel, q_pruned)
                kv_pruned_u8 = torch.where(in_kv, kv_sel_u8, kv_pruned_u8)
            c_off += c_tok

        assert q_pruned is not None and kv_pruned_u8 is not None
        kv_pruned = kv_pruned_u8.view(kv_dtype)
        gate_shard = gate_parts[0] if len(gate_parts) == 1 else torch.cat(gate_parts, dim=0)
        cm_idx = (
            inv_cm_perm.index_select(0, eff_idx.long())
            if inv_cm_perm is not None
            else eff_idx.long()
        )
        t_start = self.tp_rank * t_shard
        in_range = (cm_idx >= t_start) & (cm_idx < t_start + t_shard) & active
        local_idx = (cm_idx - t_start).clamp(0, t_shard - 1)
        local_gate_rows = torch.where(
            in_range.unsqueeze(1),
            gate_shard.index_select(0, local_idx),
            0.0,
        )
        gate_bs_full = self.ops.sum_stream(local_gate_rows)
        gate_pruned = torch.nn.functional.pad(
            gate_bs_full[:, self.tp_rank * q_size : (self.tp_rank + 1) * q_size],
            (0, 0, 0, 128 - bs),
        )
        return q_pruned, gate_pruned, kv_pruned

    def _forward_ccs_kdeep_a3(
        self,
        *,
        attn,
        weights,
        wqkvg_scale,
        wo,
        wo_scale,
        attn_in,
        chunks,
        rope_chunks,
        chunk_metas,
        forward_batch,
        trap_ctx=None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """CCS Mode-2 (`A3_KDEEP_WO=2`): overlap Chunks `0..K-2` prologue+A2A + fused last-chunk K-deep `Wo`."""
        if isinstance(attn_in, tuple):
            ag_cur, rest_in = attn_in
            rest_off = 0
        else:
            ag_cur = self.ops.ag_chunk(attn_in[: chunks[0]])
            rest_in = attn_in
            rest_off = chunks[0]
        be = None if self.a3_rpa_out else get_attn_backend()
        q_prev_parts = []
        q_last_loc = None
        q_tok_dim = 1 if self.a1_rpa_layout else 0
        c_off = 0
        for k, _c_k in enumerate(chunks):
            has_next = k + 1 < len(chunks)
            if has_next and getattr(self.ops, "a1_chunk_agd", None) is not None:
                c_next = chunks[k + 1]
                q_k, gate_k, kv_k, ag_next = self.ops.a1_chunk_agd(
                    ag_cur,
                    rest_in[rest_off : rest_off + c_next],
                    weights.wqkvg,
                    wqkvg_scale,
                    rope_chunks[k][0],
                    rope_chunks[k][1],
                )
                rest_off += c_next
            else:
                if has_next:
                    c_next = chunks[k + 1]
                    ag_next = self.ops.ag_chunk(rest_in[rest_off : rest_off + c_next])
                    rest_off += c_next
                q_k, gate_k, kv_k = self.ops.a1_chunk(
                    ag_cur,
                    weights.wqkvg,
                    wqkvg_scale,
                    rope_chunks[k][0],
                    rope_chunks[k][1],
                )
            if self.a3_rpa_out:
                out_k = self._forward_rpa_raw_packed(
                    q_k, kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k]
                )
            elif self.a1_rpa_layout:
                out_k = be.forward_prepared(
                    q_k, kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k]
                )
            else:
                out_k = self._forward_rpa_unpacked_fp8(
                    q_k, kv_k, attn.attn, forward_batch, chunk_meta=chunk_metas[k]
                )
            out_tok = (
                out_k.squeeze(0)
                if (out_k.ndim > 2 and out_k.shape[0] == 1)
                else out_k
            )
            c_tok = self.tp_size * _c_k
            if trap_ctx is not None:
                c_real = trap_ctx["full_real"][c_off : c_off + c_tok]
                c_orig = trap_ctx["orig_full"][c_off : c_off + c_tok]
                _record_nan_trap(
                    trap_ctx, layer_idx, 1, ((q_k, q_tok_dim), (kv_k, 0)), c_real, c_orig
                )
                _record_nan_trap(trap_ctx, layer_idx, 2, ((out_tok, 0),), c_real, c_orig)
            c_off += c_tok
            if not self.fused_glue and not self.a3_rpa_out:
                out_tok = attn._per_head_normalize(out_tok).to(torch.bfloat16)
            if has_next:
                q_prev_parts.append(self.ops.a3_pro_a2a_chunk(out_tok, gate_k))
                ag_cur = ag_next
            else:
                q_last_loc = self.ops.a3_pro_last_chunk(out_tok, gate_k)
        q_prev_xor = (
            q_prev_parts[0]
            if len(q_prev_parts) == 1
            else torch.cat(q_prev_parts, dim=1)
        )
        assert q_last_loc is not None
        return self.ops.a3_kdeep_ccs(q_prev_xor, q_last_loc, wo, wo_scale)

    def _moe_op(self, *, is_fp8_expert: bool, gather_next: bool):
        if self.eplb_moe_op is not None:
            ops = (
                (self.eplb_moe_agd_op, self.eplb_moe_fp8_agd_op)
                if gather_next
                else (self.eplb_moe_op, self.eplb_moe_fp8_op)
            )
        else:
            ops = (
                (self.ops.moe_agd, self.ops.moe_fp8_agd)
                if gather_next
                else (self.ops.moe, self.ops.moe_fp8)
            )
        return ops[1] if (is_fp8_expert and ops[1] is not None) else ops[0]

    def _moe_real_tokens_arg(self) -> tuple:
        if self.eplb_moe_op is not None and self._cur_real_tokens_t is not None:
            return (self._cur_real_tokens_t,)
        return ()

    def _init_eplb(self, decoder, attn, device) -> None:
        from sglang_tpu.muse_prefill_kernels.eplb import (
            OnlineEplbController,
            PrefillEplbManager,
            build_eplb_moe_op,
            load_eplb_stats_file,
            resolve_prefill_eplb_config,
        )

        self.eplb_mode, self.eplb_replicas = resolve_prefill_eplb_config(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE.get(),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REPLICAS.get(),
        )
        self.skip_pad_moe_tokens = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_MOE_TOKENS.get()
        )
        self.eplb_moe_op = None
        self.eplb_moe_fp8_op = None
        self.eplb_moe_agd_op = None
        self.eplb_moe_fp8_agd_op = None
        self.eplb_manager = None
        self.eplb_controller = None
        self.eplb_calib_active = False
        self._eplb_rebalanced = False
        self._cur_real_tokens_t = None
        if self.ep_tp2_mode or (not self.eplb_mode and not self.skip_pad_moe_tokens):
            return
        self.eplb_stats_path = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH.get()
            if self.eplb_mode
            else ""
        )
        self.eplb_calib_forwards = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS.get()
            if self.eplb_mode
            else 0
        )
        self.eplb_skip_forwards = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_SKIP_FORWARDS.get()
            if self.eplb_mode
            else 0
        )
        self.eplb_min_calib_tokens = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MIN_CALIB_TOKENS.get()
            if self.eplb_mode
            else 256
        )
        self.eplb_rebalance_interval = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REBALANCE_INTERVAL.get()
            if self.eplb_mode
            else 0
        )
        self.eplb_skew_threshold = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_SKEW_THRESHOLD.get()
            if self.eplb_mode
            else 1.22
        )
        eff_stats_path = self.eplb_stats_path
        if (
            self.eplb_mode
            and not eff_stats_path
            and self.eplb_calib_forwards <= 0
            and _env_bool("SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_AUTO_STATS")
        ):
            eff_stats_path = str(
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_STATS_PATH.get() or ""
            ).strip()
        loaded_counts = (
            load_eplb_stats_file(
                eff_stats_path,
                num_layers=self.end - self.start,
                start_layer=self.start,
                missing_ok=(self.eplb_calib_forwards > 0),
            )
            if self.eplb_mode
            else None
        )
        self.eplb_calib_active = bool(
            self.eplb_mode
            and self.eplb_calib_forwards > 0
            and (loaded_counts is None or self.eplb_rebalance_interval > 0)
        )
        packed_sorts = tpu_envs.SGLANG_TPU_MUSE_KERNEL_PACKED_SORTS.get()
        router_hilo_pre = tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTER_HILO_PRE.get()
        gmm_layout_deq = tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT.get()
        d_tok_tile = tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_TOK_TILE.get()
        d_zero_spill = tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_ZERO_SPILL.get()
        d_prepack_vecs = tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_PREPACK_VECS.get()
        sortfree_prefill = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get()
        sortfree_mode = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_MODE.get()
        sf_scatter = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER.get()
        dequant_1pass = tpu_envs.SGLANG_TPU_MUSE_KERNEL_DEQUANT_1PASS.get()
        b2_pack_bf16_sc = tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE.get()
        b2_fp8_lhs = tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_FP8_LHS.get()
        d_fp8_unpool = tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL.get()
        d_t64_c16 = tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_T64_C16.get()
        short_pt = tpu_envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.get()
        eplb_extra = (
            ("_ps" if packed_sorts else "")
            + (
                f"_sf{sortfree_mode}"
                if sortfree_prefill and sortfree_mode
                else ("_sf" if sortfree_prefill else "")
            )
            + (f"_bs{sf_scatter}" if sortfree_prefill and sf_scatter else "")
            + ("_rhp" if router_hilo_pre else "")
            + ("_gl" if gmm_layout_deq and self.fused_glue else "")
            + ("_d1p" if dequant_1pass and gmm_layout_deq and self.fused_glue else "")
            + (
                "_bpsc"
                if b2_pack_bf16_sc and gmm_layout_deq and self.fused_glue
                else ""
            )
            + (f"_dt{d_tok_tile}" if d_tok_tile != 64 else "")
            + (
                "_dt128f"
                if _env_bool("SGLANG_TPU_MUSE_KERNEL_D_T128_FAST")
                else ""
            )
            + ("_dzs" if d_zero_spill else "")
            + ("_dpv" if d_prepack_vecs else "")
            + ("_b2f8" if b2_fp8_lhs else "")
            + (f"_df8q{d_fp8_unpool}" if d_fp8_unpool > 0 else "")
            + ("_dc16" if d_t64_c16 else "")
            + ("_spt" if short_pt else "")
            + (
                f"_fb12_{tpu_envs.SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2.get()
                else ""
            )
            + (
                f"_facc{tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get()
                else ""
            )
            + (
                f"_rb{tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get()
                else ""
            )
            + (
                f"_psf{tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get()
                else ""
            )
            + (
                f"_flq{tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get()
                else ""
            )
            + (
                f"_do{tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_OPT.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_OPT.get()
                else ""
            )
            + (
                "_dmxu"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE.get()
                else ""
            )
            + (
                "_b12f"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST.get()
                else ""
            )
            + (
                "_b12p"
                if (
                    tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK.get()
                    and not tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST.get()
                )
                else ""
            )
            + (
                "_zsr"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING.get()
                else ""
            )
            + (
                "_odeq0"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG.get()
                else ""
            )
            + (
                "_dcoal"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER.get()
                else ""
            )
            + (
                "_dikag"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG.get()
                else ""
            )
            + (
                f"_w4u{tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get()
                else ""
            )
            + (
                "_rnp"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK.get()
                else ""
            )
            + (
                f"_cls{tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get()}"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get()
                else ""
            )
            + (
                "_pcmr"
                if tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE.get()
                else ""
            )
            + (
                "_f8ni"
                if self.a1_variant in ("v6", "v10_sp")
                and tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_OPT.get() > 0
                else ""
            )
        )
        fp8_tiles_tag = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES.get().replace(
                ",", "_"
            )
        )
        fp8_qbs = tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP8_LHS_QBS.get()
        fp8_extra = (f"_ft{fp8_tiles_tag}" if fp8_tiles_tag else "") + (
            f"_fq{fp8_qbs}" if fp8_qbs != 0 else ""
        )
        build_moe_op = functools.partial(
            build_eplb_moe_op,
            h_q_local=attn.num_heads,
            h_kv_local=attn.num_kv_heads,
            fused_glue=self.fused_glue,
            eplb_mode=self.eplb_mode,
            eplb_calib=self.eplb_calib_active,
            skip_pad_tokens=self.skip_pad_moe_tokens,
            router_hilo_pre=router_hilo_pre,
            d_tok_tile=d_tok_tile,
            axis_name=_AXIS,
            next_in_gather_static_fp8=(self.a1_variant in ("v6", "v10_sp")),
            causal_chunk_stream=self.causal_chunk_stream,
            causal_chunk_spec=self.causal_chunk_spec,
        )
        self.eplb_moe_op = build_moe_op(op_extra_tag=eplb_extra)
        if self.fp8_layers:
            self.eplb_moe_fp8_op = build_moe_op(
                op_extra_tag=eplb_extra + "_fp8" + fp8_extra
            )
        if (
            self.a1_ag_under_d or self.causal_chunk_stream
        ) and self.a1_variant != "v10_sp":
            a1_ch_tag = (
                f"_ch_{self.a1_chunks.replace('+', 'p').replace(',', '_')}"
                if self.a1_chunks
                else ""
            )
            if self.causal_chunk_stream:
                from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                    resolve_causal_chunks,
                )

                _eplb_first_rows = lambda ts: resolve_causal_chunks(
                    ts, self.causal_chunk_spec
                )[0]
                ccs_eplb_tag = "_ccs" + (
                    "_"
                    + self.causal_chunk_spec.replace(";", "__")
                    .replace(":", "_")
                    .replace(",", "_")
                    if self.causal_chunk_spec
                    else ""
                )
            else:
                _eplb_first_rows = functools.partial(
                    _a1_first_rows,
                    variant=self.a1_variant,
                    parts=self.a1_parts,
                    a1_chunks=self.a1_chunks,
                    a1_1way=self.a1_1way,
                    rpa_layout=self.a1_rpa_layout,
                )
                ccs_eplb_tag = ""
            build_agd_op = functools.partial(
                build_moe_op,
                next_in_gather_rows_fn=_eplb_first_rows,
                next_in_gather_static_fp8=(self.a1_variant == "v6"),
            )
            agd_tag = (
                f"_agd_{self.a1_variant}"
                + ("_p" + "".join(map(str, self.a1_parts)) if self.a1_parts else "")
                + a1_ch_tag
                + ccs_eplb_tag
            )
            self.eplb_moe_agd_op = build_agd_op(op_extra_tag=eplb_extra + agd_tag)
            if self.fp8_layers:
                self.eplb_moe_fp8_agd_op = build_agd_op(
                    op_extra_tag=eplb_extra + "_fp8" + fp8_extra + agd_tag
                )
        if not self.eplb_mode:
            logger.info("Muse prefill MoE skip-pad tokens enabled (EPLB off)")
            return
        init_perms: list[np.ndarray] | None = None
        dual_decode = _env_bool("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB")
        dec_counts_map: dict[int, np.ndarray] | None = None
        num_e = int(getattr(decoder.config, "num_local_experts", 256))
        try:
            from sglang_tpu.models.muse_spark import _placed_expert_order

            layer_perms = []
            any_placed = False
            for l_idx in range(self.start, self.end):
                ord_t = _placed_expert_order(
                    f"model.layers.{l_idx}.mlp.experts.w13_weight", num_e
                )
                if ord_t is None:
                    layer_perms.append(np.arange(num_e, dtype=np.int32))
                else:
                    any_placed = True
                    layer_perms.append(ord_t.detach().cpu().numpy().astype(np.int32))
            if any_placed:
                init_perms = layer_perms
        except Exception:
            init_perms = None
        dec_stats_path = str(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_STATS_PATH.get() or ""
        ).strip()
        if dual_decode and dec_stats_path:
            from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
                load_calibration_counts,
            )

            dec_counts_map = load_calibration_counts(
                dec_stats_path,
                num_layers=len(decoder.layers),
                num_experts=num_e,
            )
        self.eplb_manager = PrefillEplbManager(
            decoder=decoder,
            weights_list=self.weights,
            gather_fn=self._gather,
            start=self.start,
            end=self.end,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            mode=self.eplb_mode,
            num_replicas=self.eplb_replicas,
            initial_col_perms=init_perms,
            dual_decode=dual_decode,
            decode_counts_by_layer=dec_counts_map,
            num_base_local=num_e // self.tp_size,
        )
        if loaded_counts is not None:
            self.eplb_manager.rebalance_from_counts(loaded_counts)
            self._eplb_rebalanced = True
        if self.eplb_calib_active:
            self.eplb_controller = OnlineEplbController(
                num_layers=self.end - self.start,
                num_experts=self.eplb_manager.num_experts,
                ep_size=self.tp_size,
                num_local_base=self.eplb_manager.num_base_local,
                num_replicas=self.eplb_replicas,
                mode=self.eplb_mode,
                calib_forwards=self.eplb_calib_forwards,
                skip_forwards=self.eplb_skip_forwards,
                min_calib_tokens=self.eplb_min_calib_tokens,
                rebalance_interval=self.eplb_rebalance_interval,
                skew_threshold=self.eplb_skew_threshold,
                initial_rebalanced=loaded_counts is not None,
            )
            self.eplb_manager.warm_rebalance_kernels()
            self.register_buffer(
                "eplb_counts",
                torch.zeros(
                    self.end - self.start, num_e, dtype=torch.int64, device=device
                ),
                persistent=False,
            )
            self.register_buffer(
                "eplb_calib_step",
                torch.zeros(1, dtype=torch.int32, device=device),
                persistent=False,
            )
        logger.info(
            "Muse prefill EPLB enabled: mode=%s, replicas=%d, skip_pad=%s, "
            "stats_loaded=%s, calib_forwards=%d, rebalance_interval=%d, skew_threshold=%.2f",
            self.eplb_mode,
            self.eplb_replicas,
            self.skip_pad_moe_tokens,
            loaded_counts is not None,
            self.eplb_calib_forwards,
            self.eplb_rebalance_interval,
            self.eplb_skew_threshold,
        )

    def _prepare_eplb_calib_tokens(self, forward_batch, stream):
        if not (self.eplb_calib_active or self.skip_pad_moe_tokens):
            return None
        ext_lens = getattr(forward_batch, "extend_seq_lens", None)
        if isinstance(ext_lens, torch.Tensor):
            return ext_lens.sum(dim=0, keepdim=True).to(torch.int32)
        return stream.new_full((1,), stream.shape[0], dtype=torch.int32)

    def _finish_eplb_calib_step(self, calib_hists: list[torch.Tensor]) -> None:
        batch_hists = torch.stack(calib_hists)
        if torch.compiler.is_compiling():
            self.eplb_counts.add_(batch_hists.to(torch.int64))
            self.eplb_calib_step.add_(1)
            return
        from sglang_tpu.compilation.direct_compile import insert_callback

        if insert_callback(
            batch_hists,
            self._consume_eplb_calib_hists,
            callback_factory=_make_eplb_hist_callback,
        ):
            return
        self._consume_eplb_calib_hists(batch_hists)

    def _consume_eplb_calib_hists(self, batch_hists: torch.Tensor) -> None:
        from sglang_tpu.compilation.warmup_queue import current_warmup_queue
        from sglang_tpu.muse_prefill_kernels.eplb import save_eplb_stats_file

        if self.eplb_controller is None or self.eplb_manager is None:
            return
        hists_np = batch_hists.detach().cpu().numpy().astype("int64", copy=False)
        counts_to_apply, eff_tile_m = self.eplb_controller.observe_batch(
            hists_np,
            self.eplb_manager.current_col_perms,
            in_warmup_queue=current_warmup_queue() is not None,
        )
        if counts_to_apply is None:
            return
        self.eplb_manager.rebalance_from_counts(counts_to_apply, tile_m=eff_tile_m)
        self._eplb_rebalanced = True
        if self.tp_rank == 0 and self.eplb_stats_path:
            try:
                save_eplb_stats_file(
                    self.eplb_stats_path, counts_to_apply, start_layer=self.start
                )
            except OSError as exc:
                logger.warning(
                    "Muse prefill EPLB: failed saving stats to %s: %s",
                    self.eplb_stats_path,
                    exc,
                )
        if self.tp_rank == 0:
            logger.info(
                "Muse prefill EPLB rebalance #%d applied (eligible_steps=%d, "
                "skipped_dummy=%d, observed_skew=%.3f)",
                self.eplb_controller.rebalance_count,
                self.eplb_controller.eligible_steps,
                self.eplb_controller.skipped_dummy_steps,
                self.eplb_controller.last_observed_skew,
            )

    def _init_nan_trap(
        self,
        *,
        forward_batch,
        stream: torch.Tensor,
        stream_shard: torch.Tensor,
        chunks: tuple[int, ...],
        t_shard: int,
        t_total: int,
    ) -> Optional[dict[str, object]]:
        if not getattr(self, "nan_trap", False):
            return None
        from sglang_tpu.rank_local import rank_slice

        ext_lens = getattr(forward_batch, "extend_seq_lens", None)
        real_tok = (
            ext_lens.sum(dim=0, keepdim=True).to(torch.int32)
            if isinstance(ext_lens, torch.Tensor)
            else stream.new_full((1,), t_total, dtype=torch.int32)
        )
        orig_full = torch.arange(t_total, device=stream.device, dtype=torch.int32)
        full_real = orig_full < real_tok
        if len(chunks) > 1:
            perm_cm = self._cm_perm_tensor(chunks, stream.device, inverse=False).to(
                torch.int32
            )
            shard_orig = rank_slice(perm_cm, 0, self.tp_size, self.tp_rank)
        else:
            shard_orig = self.tp_rank * t_shard + torch.arange(
                t_shard, device=stream.device, dtype=torch.int32
            )
        shard_real = shard_orig < real_tok
        trap_ctx: dict[str, object] = {
            "records": [],
            "t_total": t_total,
            "t_shard": t_shard,
            "orig_full": orig_full,
            "full_real": full_real,
            "shard_orig": shard_orig,
            "shard_real": shard_real,
        }
        _record_nan_trap(
            trap_ctx, self.start, 0, ((stream_shard, 0),), shard_real, shard_orig
        )
        return trap_ctx

    def _emit_nan_trap(self, trap_ctx: Optional[dict[str, object]]) -> None:
        if trap_ctx is None or not trap_ctx["records"]:
            return
        packed = torch.stack(trap_ctx["records"])
        from sglang_tpu.compilation.direct_compile import insert_callback

        if insert_callback(
            packed,
            self._consume_nan_trap,
            callback_factory=_make_nan_trap_callback,
        ):
            return
        if torch.compiler.is_compiling():
            if hasattr(self, "nan_trap_count"):
                bad_mask = packed[:, 2] > 0
                n_bad_recs = bad_mask.sum(dtype=torch.int32)
                has_bad = n_bad_recs > 0
                first_idx = bad_mask.to(torch.int32).argmax().reshape(1)
                rev_idx = (
                    packed.shape[0]
                    - 1
                    - torch.flip(bad_mask, (0,)).to(torch.int32).argmax()
                ).reshape(1)
                first_row = packed.index_select(0, first_idx).squeeze(0)
                last_row = packed.index_select(0, rev_idx).squeeze(0)
                self.nan_trap_first.copy_(
                    torch.where(has_bad, first_row, self.nan_trap_first)
                )
                self.nan_trap_last.copy_(
                    torch.where(has_bad, last_row, self.nan_trap_last)
                )
                self.nan_trap_count.add_(n_bad_recs.reshape(1))
            return
        self._consume_nan_trap(packed)

    def flush_nan_trap_host(self) -> None:
        if not getattr(self, "nan_trap", False) or not hasattr(self, "nan_trap_count"):
            return
        if not getattr(self, "_nan_trap_armed_logged", False):
            self._nan_trap_armed_logged = True
            if self.tp_rank == 0:
                logger.info("MUSE_NAN_TRAP armed (rank=0, compiled buffer mode)")
        count = int(self.nan_trap_count.cpu()[0])
        seen = getattr(self, "_nan_trap_seen", 0)
        if count == seen:
            return
        self._nan_trap_seen = count
        first = [int(x) for x in self.nan_trap_first.cpu().tolist()]
        last = [int(x) for x in self.nan_trap_last.cpu().tolist()]
        self.last_nan_trap = [first] if first == last else [first, last]
        stage_name = (
            _NAN_TRAP_STAGE_NAMES[first[1]]
            if 0 <= first[1] < len(_NAN_TRAP_STAGE_NAMES)
            else str(first[1])
        )
        last_stage = (
            _NAN_TRAP_STAGE_NAMES[last[1]]
            if 0 <= last[1] < len(_NAN_TRAP_STAGE_NAMES)
            else str(last[1])
        )
        logger.error(
            "MUSE_NAN_TRAP rank=%d first=(layer=%d stage=%s n_bad=%d first_tok=%d) "
            "last=(layer=%d stage=%s n_bad=%d first_tok=%d) total_bad_stages=%d",
            self.tp_rank,
            first[0],
            stage_name,
            first[2],
            first[3],
            last[0],
            last_stage,
            last[2],
            last[3],
            count - seen,
        )

    def _consume_nan_trap(self, report: torch.Tensor) -> None:
        arr = report.detach().cpu()
        if not getattr(self, "_nan_trap_armed_logged", False):
            self._nan_trap_armed_logged = True
            if self.tp_rank == 0:
                logger.info(
                    "MUSE_NAN_TRAP armed (rank=0, %d checkpoints)", int(arr.shape[0])
                )
        bad = arr[arr[:, 2] > 0]
        self.last_nan_trap = bad.tolist()
        if bad.numel() == 0:
            return
        first = [int(x) for x in bad[0].tolist()]
        stage_name = (
            _NAN_TRAP_STAGE_NAMES[first[1]]
            if 0 <= first[1] < len(_NAN_TRAP_STAGE_NAMES)
            else str(first[1])
        )
        logger.error(
            "MUSE_NAN_TRAP rank=%d first=(layer=%d stage=%s n_bad=%d first_tok=%d) count=%d",
            self.tp_rank,
            first[0],
            stage_name,
            first[2],
            first[3],
            int(bad.shape[0]),
        )

    def _log_debug_stats(self) -> None:
        seen = 0
        while True:
            time.sleep(5)
            count = int(self.debug_count.cpu()[0])
            if count == seen:
                continue
            seen = count
            rows = self.debug_stats.cpu().tolist()
            worst = min(range(len(rows)), key=lambda i: rows[i][1])
            logger.info(
                "Muse prefill kernel compare #%d (layer: cos, min row cos, max abs, "
                "rel L2): worst layer %d %s; all %s",
                count,
                self.start + worst,
                [round(x, 6) for x in rows[worst]],
                {
                    self.start + i: [round(x, 6) for x in row]
                    for i, row in enumerate(rows)
                },
            )


_NAN_TRAP_STAGE_NAMES = ("entry", "a1", "rpa", "a3", "moe", "exit")


def _gather_sp_entry_embed_col(
    gather_fn,
    src_w: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    h_col: int,
) -> torch.Tensor:
    """Gather `[src_rows, h_col]` slices across TP ranks with 128-row alignment."""
    src_rows = int(src_w.shape[0])
    pad_rows = ((src_rows + 127) // 128) * 128
    src_w_pad = (
        torch.nn.functional.pad(src_w, (0, 0, 0, pad_rows - src_rows))
        if pad_rows > src_rows
        else src_w
    )
    embed_col = None
    for c in range(tp_size):
        g_c = gather_fn(src_w_pad[:, c * h_col : (c + 1) * h_col].contiguous())
        if c == tp_rank:
            if pad_rows > src_rows:
                embed_col = (
                    g_c.reshape(tp_size, pad_rows, h_col)[:, :src_rows, :]
                    .reshape(tp_size * src_rows, h_col)
                    .contiguous()
                )
            else:
                embed_col = g_c
    assert embed_col is not None
    own_slice = src_w[:, tp_rank * h_col : (tp_rank + 1) * h_col]
    got_own = embed_col[tp_rank * src_rows : (tp_rank + 1) * src_rows]
    if not bool(torch.equal(got_own.cpu(), own_slice.cpu())):
        raise RuntimeError(
            f"Muse sp_entry embed_col self-slice mismatch on rank={tp_rank}"
        )
    own_amax = float(src_w.abs().amax().cpu()) if src_w.numel() > 0 else 0.0
    col_amax = float(embed_col.abs().amax().cpu()) if embed_col.numel() > 0 else 0.0
    if not (col_amax <= max(1e4, own_amax * 10.0)):
        raise RuntimeError(
            f"Muse sp_entry embed_col corrupt magnitude ({col_amax} > max(1e4, 10*{own_amax})) on rank={tp_rank}"
        )
    return embed_col


def _record_nan_trap(
    trap_ctx: dict[str, object],
    layer_idx: int,
    stage_id: int,
    items: tuple[tuple[torch.Tensor, int], ...],
    real_mask: torch.Tensor,
    orig_idx: torch.Tensor,
) -> None:
    bad_row = None
    for tensor, tok_dim in items:
        x = tensor if tok_dim == 0 else tensor.movedim(tok_dim, 0)
        cur = (~torch.isfinite(x.float())).reshape(x.shape[0], -1).any(dim=1)
        bad_row = cur if bad_row is None else (bad_row | cur)
    assert bad_row is not None
    bad_real = bad_row & real_mask
    n_bad = bad_real.sum(dtype=torch.int32)
    first_bad = (
        torch.where(bad_real, orig_idx, int(trap_ctx["t_total"])).min().to(torch.int32)
    )
    trap_ctx["records"].append(
        torch.stack(
            [
                n_bad.new_full((), layer_idx, dtype=torch.int32),
                n_bad.new_full((), stage_id, dtype=torch.int32),
                n_bad,
                first_bad,
            ]
        )
    )


def _make_nan_trap_callback(model):
    """Picklable direct_compile callback factory binding to each rank's MusePrefillKernels."""
    if isinstance(model, MusePrefillKernels):
        return model._consume_nan_trap
    pk = getattr(model, "prefill_kernels", None)
    if pk is None and hasattr(model, "model"):
        pk = getattr(model.model, "prefill_kernels", None)
    if pk is None:
        raise AttributeError(
            f"Could not resolve prefill_kernels on {type(model).__name__} for NaN trap callback"
        )
    return pk._consume_nan_trap


def _make_eplb_hist_callback(model):
    """Picklable direct_compile callback factory binding to each rank's MusePrefillKernels."""
    if isinstance(model, MusePrefillKernels):
        return model._consume_eplb_calib_hists
    pk = getattr(model, "prefill_kernels", None)
    if pk is None and hasattr(model, "model"):
        pk = getattr(model.model, "prefill_kernels", None)
    if pk is None:
        raise AttributeError(
            f"Could not resolve prefill_kernels on {type(model).__name__} for EPLB callback"
        )
    return pk._consume_eplb_calib_hists
