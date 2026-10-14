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
"""GPT-OSS TP8 decode megakernel for TPU v7x: one pallas_call per decode step, all layers.

Runs per rank under `shard_map` over a 1-D "tp" mesh. Each rank owns NQ/R query heads, one
KV head and either an I/R slice of every expert ("tp" MoE) or E/R whole experts ("ep" MoE).
See DESIGN.md for the full design; per layer l (fori over (sliding, full) layer pairs):

  A  RMSNorm -> QKV (+bias) -> neox RoPE -> q [BR, 512] f32, new k/v [BR, 128] (KV dtype)
  B  per sequence: attention over the paged KV pool *in place* (one DMA per page, K/V rows
     interleaved as in sglang's fused TPU pool) + the new token (from VMEM), sinks, window
  C  o_proj -> in-kernel all-reduce (remote DMA, one-shot all-to-all) -> + bo + residual
  D  RMSNorm -> router (+bias) -> top-k softmax -> active-expert list (VMEM -> SMEM)
  E  for every active local expert chunk: gate|up -> swigluoai -> down (+biases), weighted
     -> in-kernel all-reduce -> + residual

Weights stream HBM -> VMEM: the attention/router weights of layer l+1 are prefetched into a
two-slot buffer while layer l runs; expert chunks stream through a `w_slots`-deep ring.

The new token's K/V leave the kernel packed as `kvnew [L, BR, 1, packing, 128]` and a second, tiny,
non-communicating pallas_call (`kv_scatter`) writes them into the pools in place. (XLA TPU
does not place outputs of a communicating pallas_call - `collective_id` set - into live-out
buffers, so aliasing the pool inside the megakernel would copy the whole pool each step; see
vllm-torchtpu `kernels/megakernel/_frozen/qwen35/pool_io.py`.)

`mode="attn"` / `mode="moe"` build single-layer, single-phase kernels with the identical
bodies; with XLA `psum` between them they form the XLA-collective fallback used to check the
in-kernel all-reduce and to compare against.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from .config import DecodeConfig
from .reference import LayerWeights, ar_jdtype, kv_jdtype, yarn_tables

BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
NT_DIMS = (((1,), (1,)), ((), ()))
NEG = -1e30
COLLECTIVE_ID = 0

AW_NAMES = ("wqkv", "wo", "wr", "vec", "sinks")
EW_NAMES = ("w13", "b13", "w2", "b2")
W_NAMES = AW_NAMES + EW_NAMES
EW_NAMES_FP4 = ("w13q", "w13s", "b13", "w2q", "w2s", "b2")
EW_NAMES_FP8 = ("w13", "c13", "b13", "w2", "c2", "b2")
F4 = jnp.float4_e2m1fn


def w_names(c: DecodeConfig) -> tuple[str, ...]:
    return AW_NAMES + {"mxfp4": EW_NAMES_FP4, "fp8": EW_NAMES_FP8}.get(
        c.w_dtype, EW_NAMES
    )


# rows of the per-layer `vec` [8, VW] f32 operand
V_LN1, V_LN2, V_BO, V_BQKV, V_BR = range(5)


# ----------------------------------------------------------------------------- packing


def packed_specs(c: DecodeConfig) -> dict[str, jax.ShapeDtypeStruct]:
    """Global shapes of the packed per-rank weights (leading axis R, sharded over "tp")."""
    R, L, Hp, EL, NJ, T = c.R, c.L, c.Hp, c.EL, c.NJ, c.tne
    s = jax.ShapeDtypeStruct
    out = {
        "wqkv": s((R, L, Hp, c.QKVWL), BF16),
        "wo": s((R, L, c.QWL, Hp), BF16),
        "wr": s((R, L, Hp, 128), BF16),
        "vec": s((R, L, 8, c.VW), F32),
        "sinks": s((R, L, c.NQL, 128), F32),
    }
    if c.w_dtype == "mxfp4":
        # 8 e2m1 codes per uint32 along K (row k of a group of 8 at bits 4k, the
        # `pltpu.bitcast` packing); bf16 power-of-two scale per 32 K rows and column
        out.update(
            {
                "w13q": s((R, L, EL, NJ, Hp // 8, 2 * T), jnp.uint32),
                # scale rows padded to the bf16 sublane tile: unpadded rows made XLA relayout-
                # copy the scale arrays every step (seen in the AOT HLO)
                "w13s": s((R, L, EL, NJ, c.SR13, 2 * T), BF16),
                "b13": s((R, L, EL, NJ, 1, 2 * T), F32),
                "w2q": s((R, L, EL, NJ, T // 8, Hp), jnp.uint32),
                "w2s": s((R, L, EL, NJ, c.SR2, Hp), BF16),
                "b2": s((R, L, EL, 1, Hp), F32),
            }
        )
    elif c.w_dtype == "fp8":
        # e4m3 weights; column c of a chunk is w8[:, c] * c13[c] (power of two)
        out.update(
            {
                "w13": s((R, L, EL, NJ, Hp, 2 * T), jnp.float8_e4m3fn),
                "c13": s((R, L, EL, NJ, 1, 2 * T), F32),
                "b13": s((R, L, EL, NJ, 1, 2 * T), F32),
                "w2": s((R, L, EL, NJ, T, Hp), jnp.float8_e4m3fn),
                "c2": s((R, L, EL, NJ, 1, Hp), F32),
                "b2": s((R, L, EL, 1, Hp), F32),
            }
        )
    else:
        out.update(
            {
                "w13": s((R, L, EL, NJ, Hp, 2 * T), BF16),
                "b13": s((R, L, EL, NJ, 1, 2 * T), F32),
                "w2": s((R, L, EL, NJ, T, Hp), BF16),
                "b2": s((R, L, EL, 1, Hp), F32),
            }
        )
    return {k: out[k] for k in w_names(c)}


def mxfp4_pack(w, scale_rows=None):
    """[..., K, N] MXFP4-representable weights -> (uint32 codes [..., K/8, N], bf16 scales
    [..., scale_rows, N] (rows past K/32 are padding)). K must be a multiple of 32."""
    from .reference import mxfp4_quantize

    code, scale = mxfp4_quantize(w)
    code = code.astype(np.uint32)
    words = np.zeros(code.shape[:-2] + (code.shape[-2] // 8, code.shape[-1]), np.uint32)
    for k in range(8):
        words |= code[..., k::8, :] << np.uint32(4 * k)
    if scale_rows is not None:
        pad = [(0, 0)] * (scale.ndim - 2) + [(0, scale_rows - scale.shape[-2]), (0, 0)]
        scale = np.pad(scale, pad, constant_values=1.0)
    return jnp.asarray(words), jnp.asarray(scale).astype(BF16)


def fp8_pack(w):
    """[..., K, N] MXFP4-representable weights -> (e4m3 [..., K, N], f32 [..., 1, N]) with
    w == w8 * colscale exactly when a column's 32-blocks span <= 13 octaves (practically
    always: e4m3 spans 2^-9..448, e2m1 values 0.5..6). Inexact elements are counted."""
    w = np.asarray(jnp.asarray(w).astype(F32))
    colmax = np.max(np.abs(w), axis=-2, keepdims=True)
    g = np.where(colmax > 0, np.floor(np.log2(np.maximum(colmax, 1e-38))) - 7, 0)
    f = np.exp2(g).astype(np.float32)
    w8 = jnp.asarray(w / f).astype(jnp.float8_e4m3fn)
    bad = int(np.sum(np.asarray(w8.astype(F32)) * f != w))
    if bad:
        import warnings

        warnings.warn(f"fp8_pack: {bad} expert weights not exactly representable")
    return w8, jnp.asarray(f)


def mxfp4_unpack_xla(words, scale):
    """XLA-side inverse of `mxfp4_pack` (same semantics as `pltpu.bitcast` u32 -> fp4)."""
    f = lax.bitcast_convert_type(words, F4)  # [..., K/8, N, 8], element k = bits 4k
    f = jnp.swapaxes(f, -1, -2)
    f = f.reshape(f.shape[:-3] + (f.shape[-3] * 8, f.shape[-1])).astype(F32)
    scale = scale[..., : f.shape[-2] // 32, :]
    return f * jnp.repeat(scale.astype(F32), 32, axis=-2)


def pool_specs(c: DecodeConfig) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
    """(swa, full) pools: [R, L/2, num_pages, page, 1, packing, 128] in the KV dtype.

    Per rank and layer this is exactly sglang's fused TPU pool `[num_blocks, page,
    2*kv_heads/packing, packing, head_dim_padded]` with one KV head per rank: K in packing
    slot 0, V in slot 1 (fp8: slots 2-3 are padding), head_dim 64 zero-padded to 128. On TPU
    the `packing` slots of one token share a 32-bit word per lane; this kernel reads them the
    way the stock RPA v3 kernel does (bitcast the VMEM page to uint32, shift, bitcast back).
    """
    dt = kv_jdtype(c)
    sh = lambda nb: jax.ShapeDtypeStruct(
        (c.R, c.L // 2, nb, c.page, 1, c.kv_packing, 128), dt
    )
    return sh(c.nb_swa), sh(c.nb_full)


def pack_weights(c: DecodeConfig, layers: list[LayerWeights]) -> dict[str, jax.Array]:
    """Global model-layout weights -> per-rank kernel layout, stacked [R, L, ...]."""
    H, Hp, D, R = c.H, c.Hp, c.D, c.R
    QW, KW, QWL, NQL = c.NQ * D, c.NKV * D, c.QWL, c.NQL
    E, I, EL, ILR, IL, NJ, T = c.E, c.I, c.EL, c.IL_RAW, c.IL, c.NJ, c.tne

    def pad_to(x, shape, value=0.0):
        return jnp.pad(
            x, [(0, s - d) for s, d in zip(shape, x.shape)], constant_values=value
        )

    out = {k: [] for k in W_NAMES}
    for r in range(R):
        per = {k: [] for k in W_NAMES}
        for w in layers:
            q = w.wqkv[:, r * QWL : (r + 1) * QWL]
            k = pad_to(w.wqkv[:, QW + r * D : QW + (r + 1) * D], (H, 128))
            v = pad_to(w.wqkv[:, QW + KW + r * D : QW + KW + (r + 1) * D], (H, 128))
            per["wqkv"].append(
                pad_to(jnp.concatenate([q, k, v], -1).astype(BF16), (Hp, c.QKVWL))
            )
            bq = w.bqkv[r * QWL : (r + 1) * QWL]
            bk = pad_to(w.bqkv[QW + r * D : QW + (r + 1) * D], (128,))
            bv = pad_to(w.bqkv[QW + KW + r * D : QW + KW + (r + 1) * D], (128,))
            vec = jnp.zeros((8, c.VW), F32)
            vec = vec.at[V_LN1, :H].set(w.ln1.astype(F32))
            vec = vec.at[V_LN2, :H].set(w.ln2.astype(F32))
            vec = vec.at[V_BO, :H].set(w.bo.astype(F32))
            vec = vec.at[V_BQKV, : c.QKVWL].set(
                jnp.concatenate([bq, bk, bv]).astype(F32)
            )
            vec = vec.at[V_BR, :128].set(pad_to(w.br.astype(F32), (128,), NEG))
            per["vec"].append(vec)
            per["wo"].append(
                pad_to(w.wo[r * QWL : (r + 1) * QWL].astype(BF16), (QWL, Hp))
            )
            per["wr"].append(pad_to(w.wr.astype(BF16), (Hp, 128)))
            sk = w.sinks[r * NQL : (r + 1) * NQL].astype(F32)
            per["sinks"].append(jnp.broadcast_to(sk[:, None], (NQL, 128)))
            if c.moe_mode == "tp":
                ex, cols = np.arange(E), np.arange(*c.moe_cols(r))
            else:
                ex, cols = np.arange(r * EL, (r + 1) * EL), np.arange(I)
            g = pad_to(w.w13[ex][:, :, cols], (EL, Hp, IL)).reshape(EL, Hp, NJ, T)
            u = pad_to(w.w13[ex][:, :, I + cols], (EL, Hp, IL)).reshape(EL, Hp, NJ, T)
            per["w13"].append(
                jnp.concatenate([g, u], -1).transpose(0, 2, 1, 3).astype(BF16)
            )
            b_g = pad_to(w.b13[ex][:, cols], (EL, IL)).reshape(EL, NJ, 1, T)
            b_u = pad_to(w.b13[ex][:, I + cols], (EL, IL)).reshape(EL, NJ, 1, T)
            per["b13"].append(jnp.concatenate([b_g, b_u], -1).astype(F32))
            w2 = pad_to(w.w2[ex][:, cols], (EL, IL, Hp)).reshape(EL, NJ, T, Hp)
            per["w2"].append(w2.astype(BF16))
            per["b2"].append(pad_to(w.b2[ex].astype(F32), (EL, Hp))[:, None])
        for k_ in W_NAMES:
            out[k_].append(jnp.stack(per[k_]))
    out = {k_: jnp.stack(v_) for k_, v_ in out.items()}
    if c.w_dtype == "mxfp4":
        out["w13q"], out["w13s"] = mxfp4_pack(out.pop("w13"), c.SR13)
        out["w2q"], out["w2s"] = mxfp4_pack(out.pop("w2"), c.SR2)
    elif c.w_dtype == "fp8":
        out["w13"], out["c13"] = fp8_pack(out["w13"])
        out["w2"], out["c2"] = fp8_pack(out["w2"])
    return {k_: out[k_] for k_ in w_names(c)}


def build_pools(c: DecodeConfig, k_hist, v_hist, pos, pt_swa, pt_full, base=None):
    """Scatter dense history [L, B, T, NKV, D] (KV dtype) into (swa, full) pools (numpy).

    `base`: optional (swa, full) float arrays to write into (e.g. garbage-filled for tests).
    """
    dt = kv_jdtype(c)
    kh = np.asarray(k_hist.astype(F32))
    vh = np.asarray(v_hist.astype(F32))
    if base is None:
        pools = [np.zeros(s.shape, np.float32) for s in pool_specs(c)]
    else:
        pools = [np.array(b, np.float32) for b in base]
    pos = np.asarray(pos)
    for l in range(c.L):
        pool, pt = (pools[0], pt_swa) if c.sliding[l] else (pools[1], pt_full)
        for b in range(c.B):
            for t in range(int(pos[b])):
                pg = int(pt[b, t // c.page])
                pool[:, l // 2, pg, t % c.page, 0, :2, :] = 0.0
                pool[:, l // 2, pg, t % c.page, 0, 0, : c.D] = kh[l, b, t]
                pool[:, l // 2, pg, t % c.page, 0, 1, : c.D] = vh[l, b, t]
    return tuple(jnp.asarray(p).astype(dt) for p in pools)


def rope_tables(c: DecodeConfig, pos):
    """pos [B] -> cos, sin [BR, 128] f32 (two heads per lane block; sin signed per half)."""
    cos, sin = yarn_tables(c, pos)
    h = c.D // 2
    sin_s = jnp.concatenate([-sin[:, :h], sin[:, h:]], -1)
    cos, sin_s = jnp.tile(cos, (1, 128 // c.D)), jnp.tile(sin_s, (1, 128 // c.D))
    pad = ((0, c.BR - c.B), (0, 0))
    return jnp.pad(cos, pad).astype(F32), jnp.pad(sin_s, pad).astype(F32)


# ----------------------------------------------------------------------------- kernel


def _scratch(c: DecodeConfig, mode: str, comm: bool):
    BR, Hp, QWL, NS, T, R = c.BR, c.Hp, c.QWL, c.w_slots, c.tne, c.R
    kvdt, ardt = kv_jdtype(c), ar_jdtype(c)
    V = pltpu.VMEM
    s = {
        "res": V((BR, Hp), F32),
        "xn": V((BR, Hp), BF16),
        "awvec": V((2, 8, c.VW), F32),
        "sem_aw": pltpu.SemaphoreType.DMA((2, len(AW_NAMES))),
        "sem_m": pltpu.SemaphoreType.DMA((4,)),
    }
    if mode in ("mega", "attn"):
        s.update(
            {
                # per-sequence [NQL, 128] tiles (row b*NQL + h) so phase B only does aligned
                # dynamic loads/stores; phase A/C move between layouts with strided accesses
                "qh": V(
                    (BR * c.NQL, 128), F32
                ),  # head h's q in lanes 0..63 (pre-scaled)
                "kr8": V(
                    (BR * c.NQL, 128), F32
                ),  # new K of seq b replicated over NQL rows
                "vr8": V((BR * c.NQL, 128), F32),
                "oh": V(
                    (BR * c.NQL, 128), F32
                ),  # head h's output in its pair-block half
                "kvn": V((BR, 1, c.kv_packing, 128), kvdt),
                "awqkv": V((2, Hp, c.QKVWL), BF16),
                "awo": V((2, QWL, Hp), BF16),
                "awsink": V((2, c.NQL, 128), F32),
                "kvb": V((c.kv_bufs, c.TKMAX, 1, c.kv_packing, 128), kvdt),
                "sem_kv": pltpu.SemaphoreType.DMA((c.kv_bufs,)),
                "it_b": pltpu.SMEM((c.NIMAX,), I32),
                "it_j": pltpu.SMEM((c.NIMAX,), I32),
                "nit": pltpu.SMEM(
                    (1,), I32
                ),  # items listed by the last attention prepare
            }
        )
    if mode in ("mega", "moe"):
        s.update(
            {
                "awr": V((2, Hp, 128), BF16),
                "wf": V((BR, 128), F32),
                "actv": V((8, 128), I32),
                "acts": pltpu.SMEM((8, 128), I32),
                "ite": pltpu.SMEM((c.EL + 1,), I32),
                "macc": V((BR, Hp), F32),
                "b13r": V((NS, 1, 2 * T), F32),
                "b2r": V((NS, 1, Hp), F32),
                "sem_e": pltpu.SemaphoreType.DMA((NS, 6)),
            }
        )
        if c.w_dtype == "mxfp4":
            s.update(
                {
                    "w13qr": V((NS, Hp // 8, 2 * T), jnp.uint32),
                    "w13sr": V((NS, c.SR13, 2 * T), BF16),
                    "w2qr": V((NS, T // 8, Hp), jnp.uint32),
                    "w2sr": V((NS, c.SR2, Hp), BF16),
                    "w13r": V((1, Hp, 2 * T), BF16),  # dequantized chunk (one, reused)
                    "w2r": V((1, T, Hp), BF16),
                }
            )
        elif c.w_dtype == "fp8":
            s.update(
                {
                    "w13r": V((NS, Hp, 2 * T), jnp.float8_e4m3fn),
                    "w2r": V((NS, T, Hp), jnp.float8_e4m3fn),
                    "c13r": V((NS, 1, 2 * T), F32),
                    "c2r": V((NS, 1, Hp), F32),
                }
            )
        else:
            s.update(
                {
                    "w13r": V((NS, Hp, 2 * T), BF16),
                    "w2r": V((NS, T, Hp), BF16),
                }
            )
    if comm and c.AR_RSAG:  # reduce-scatter chunks (AR dtype) + all-gather of f32 sums
        CW = c.CWA
        s.update(
            {
                "send": V((2, R, BR, CW), ardt),
                "recv": V((2, R, BR, CW), ardt),
                "ag_send": V((2, BR, CW), F32),
                "ag_recv": V((2, R, BR, CW), F32),
                "arout": V((BR, Hp), F32),
                "ssem": pltpu.SemaphoreType.DMA((2, R - 1)),
                "rsem": pltpu.SemaphoreType.DMA((2, R - 1)),
                "agssem": pltpu.SemaphoreType.DMA((2, R - 1)),
                "agrsem": pltpu.SemaphoreType.DMA((2, R - 1)),
            }
        )
    elif comm:
        if c.AR_FOLD:  # live rows only, 128-lane chunks stacked as rows, f32
            s.update(
                {
                    "send": V((2, c.FR, 128), F32),
                    "recv": V((2, R, c.FR, 128), F32),
                    "arout": V((BR, Hp), F32),
                }
            )
        else:
            s.update({"send": V((2, BR, Hp), ardt), "recv": V((2, R, BR, Hp), ardt)})
        s.update(
            {
                "ssem": pltpu.SemaphoreType.DMA((2, R - 1)),
                "rsem": pltpu.SemaphoreType.DMA((2, R - 1)),
            }
        )
    return s


def _tile_bytes(shape, dtype) -> int:
    """VMEM bytes of an array after (sublane, lane) tile padding (estimate)."""
    it = jnp.dtype(dtype).itemsize
    shape = (1,) * max(0, 2 - len(shape)) + tuple(shape)
    if it < 4 and shape[-2] * it == 4 and len(shape) > 2:
        # sglang KV packing: the `packing` slots of a token form one 32-bit sublane
        rows = math.prod(shape[:-2])
        return (-(-rows // 8) * 8) * (-(-shape[-1] // 128) * 128) * 4
    sub = 8 * (4 // it)
    lead = math.prod(shape[:-2])
    return lead * (-(-shape[-2] // sub) * sub) * (-(-shape[-1] // 128) * 128) * it


def vmem_bytes(c: DecodeConfig, mode: str = "mega") -> dict[str, int]:
    comm = mode == "mega" and c.R > 1
    out = {}
    for k, v in _scratch(c, mode, comm).items():
        if "VMEM" in str(getattr(v, "memory_space", "")).upper():
            out[k] = _tile_bytes(v.shape, v.dtype)
    # VMEM-resident operands / outputs
    out["x+cos+sin+out"] = 3 * _tile_bytes((c.BR, c.Hp), F32) + 2 * _tile_bytes(
        (c.BR, 128), F32
    )
    out["total"] = sum(out.values())
    return out


def build(
    c: DecodeConfig, mode: str = "mega", layer: int | None = None, *, interpret=False
):
    """Return a per-rank callable (to be used under shard_map over axis "tp").

    mode="mega": f(rank, pos, pt_win [B*NWP], pt_full [B*MP], newpg [2B], x, cos, sin, *W,
                   kv_swa, kv_full)
                 -> (out [BR, Hp] bf16, kvnew [L, BR, 1, packing, 128] (K slot 0, V slot 1)
                     [, kv_swa, kv_full if kv_write == "kernel" (aliased, written in place)])
    mode="attn": same operands with x := res [BR, Hp] f32 -> (partial f32, kvnew[1])
    mode="moe":  same operands with x := res [BR, Hp] f32 -> partial f32
    """
    c.check()
    assert mode in ("mega", "attn", "moe")
    assert (layer is None) == (mode == "mega")
    comm = mode == "mega" and c.R > 1  # all-reduce buffers present
    exchange = comm and "ar" not in c.ablate  # remote DMAs actually issued
    AB = set(c.ablate) if mode == "mega" else set()
    B, BR, L, Hp, H, R = c.B, c.BR, c.L, c.Hp, c.H, c.R
    QWL, NQL, EL, NJ, T, NS = c.QWL, c.NQL, c.EL, c.NJ, c.tne, c.w_slots
    PAGE, MP, NWP, NBUF = c.page, c.max_pages, c.NWP, c.kv_bufs
    W = c.window
    kvdt, ardt = kv_jdtype(c), ar_jdtype(c)
    scale = float(c.sm_scale)
    scratch = _scratch(c, mode, comm)
    s_names = list(scratch)
    aw_need = {
        "mega": AW_NAMES,
        "attn": ("wqkv", "wo", "vec", "sinks"),
        "moe": ("wr", "vec"),
    }[mode]
    aw_dst = {
        "wqkv": "awqkv",
        "wo": "awo",
        "wr": "awr",
        "vec": "awvec",
        "sinks": "awsink",
    }
    WN = w_names(c)
    KW = (
        mode == "mega" and c.kv_write == "kernel"
    )  # in-kernel new-K/V write (aliased pools)
    assert not (KW and c.split_pools), "split_pools needs kv_write='scatter'"
    n_pool = L if c.split_pools else 2
    n_in = 8 + len(WN) + n_pool

    def kernel(*refs):
        rank_s, pos_s, ptw_s, ptf_s, newpg_s, x_ref, cos_ref, sin_ref = refs[:8]
        Wr = dict(zip(WN, refs[8 : 8 + len(WN)]))
        pool_refs = refs[8 + len(WN) : n_in]
        if c.split_pools:  # per-layer refs, indexed by layer pair
            kv_swa, kv_full = pool_refs[: L // 2], pool_refs[L // 2 :]
        else:
            kv_swa, kv_full = pool_refs
        n_out = 1 if mode == "moe" else (4 if KW else 2)
        outs = refs[n_in : n_in + n_out]
        S = dict(zip(s_names, refs[n_in + n_out :]))
        rank = rank_s[0]

        def cp(src, dst, sem):
            return pltpu.make_async_copy(src, dst, sem)

        def rnd(v):
            return v.astype(BF16).astype(F32)

        # ---------------------------------------------------------- KV word packing
        # One uint32 per (token, lane): K in the low `bw` bits, V in the next `bw` (what the
        # stock RPA v3 kernel's `load_bkv` unpacks). bf16 goes through f32 bit patterns so
        # only 32-bit integer ops are needed.
        def kv_encode(kq, vq):
            if c.kv_dtype == "bf16":
                kb = pltpu.bitcast(kq.astype(F32), U32) >> 16
                vb = pltpu.bitcast(vq.astype(F32), U32) & jnp.uint32(0xFFFF0000)
                return kb | vb
            kb = pltpu.bitcast(kq, jnp.uint8).astype(U32)
            vb = pltpu.bitcast(vq, jnp.uint8).astype(U32)
            return kb | (vb << 8)

        def kv_decode(w):
            if c.kv_dtype == "bf16":
                k = pltpu.bitcast(w << 16, F32)
                v = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), F32)
                return k.astype(BF16), v.astype(BF16)
            k = pltpu.bitcast((w & jnp.uint32(0xFF)).astype(jnp.uint8), kvdt)
            v = pltpu.bitcast(((w >> 8) & jnp.uint32(0xFF)).astype(jnp.uint8), kvdt)
            # e4m3 K/V go straight into the MXU against bf16 q / p (native on v7x; ~25%
            # fewer ops per KV block than converting to bf16 first, static LLO count)
            return k, v

        # On TPU the packed words are read/written through a uint32 view of the VMEM ref, as
        # RPA v3 does (validated by AOT compile only). The TPU interpreter cannot bitcast refs,
        # so interpret mode goes through a value reshape + `pltpu.bitcast`, which has the same
        # (documented) packing semantics: `packing` consecutive second-minor rows -> one word.
        def load_words(slot, rows):  # first `rows` tokens of KV item buffer `slot`
            if interpret:
                v = S["kvb"][slot, pl.ds(0, rows)]
                return pltpu.bitcast(v.reshape(rows * c.kv_packing, 128), U32)
            return (
                S["kvb"].bitcast(U32).at[slot, pl.ds(0, rows)].reshape(rows, 128)[...]
            )

        def zero_kvb():  # stale-but-finite is all masking needs; VMEM starts arbitrary
            if interpret:
                S["kvb"][...] = jnp.zeros(S["kvb"].shape, kvdt)
            else:
                S["kvb"].bitcast(U32).reshape(NBUF * c.TKMAX, 128)[...] = jnp.zeros(
                    (NBUF * c.TKMAX, 128), U32
                )

        def store_words(ref, w):
            rows = w.shape[0]
            if interpret:
                ref[...] = pltpu.bitcast(w, kvdt).reshape(rows, 1, c.kv_packing, 128)
            else:
                ref.bitcast(U32).reshape(rows, 128)[...] = w

        # ---------------------------------------------------------- attention-weight buffer
        def aw_copies(l, slot):
            return [
                cp(Wr[n].at[l], S[aw_dst[n]].at[slot], S["sem_aw"].at[slot, i])
                for i, n in enumerate(AW_NAMES)
                if n in aw_need
            ]

        def vrow(slot, row, width):
            return S["awvec"][slot, row : row + 1, 0:width]

        # ---------------------------------------------------------- all-reduce
        # One-shot all-to-all all-reduce, ported from vllm-torchtpu's generic megakernel
        # (`kernels/megakernel/comm.py`, non-RSAG path): every rank DMAs its partial into
        # slot `me` of every peer's receive buffer, then sums all R slots in rank order.
        def oneshot_copies(slot, me):
            return [
                pltpu.make_async_remote_copy(
                    src_ref=S["send"].at[slot],
                    dst_ref=S["recv"].at[slot, me],
                    send_sem=S["ssem"].at[slot, off - 1],
                    recv_sem=S["rsem"].at[slot, off - 1],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for off in range(1, R)
            ]

        # RSAG (vllm-torchtpu comm.py RSAG path): rank r owns column chunk r. Round 1 sends
        # chunk (me^off) of the partial to peer me^off; round 2 sends the f32 sum of my
        # chunk to every peer. f32 in round 2 keeps the result identical to the one-shot sum.
        def rs_copies(slot, me):
            return [
                pltpu.make_async_remote_copy(
                    src_ref=S["send"].at[slot, me ^ off],
                    dst_ref=S["recv"].at[slot, me],
                    send_sem=S["ssem"].at[slot, off - 1],
                    recv_sem=S["rsem"].at[slot, off - 1],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for off in range(1, R)
            ]

        def ag_copies(slot, me):
            return [
                pltpu.make_async_remote_copy(
                    src_ref=S["ag_send"].at[slot],
                    dst_ref=S["ag_recv"].at[slot, me],
                    send_sem=S["agssem"].at[slot, off - 1],
                    recv_sem=S["agrsem"].at[slot, off - 1],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for off in range(1, R)
            ]

        def ar_copies(slot, me):  # every descriptor of one AR (for send waits)
            if c.AR_RSAG:
                return rs_copies(slot, me) + ag_copies(slot, me)
            return oneshot_copies(slot, me)

        NHB = Hp // 128

        def ar_start(slot, val, l, used_before):
            """Start the all-reduce of `val` [BR, Hp] f32; returns `finish() -> [BR, Hp] f32`.

            The sends leave immediately; the caller can do independent work (e.g. the next
            layer's KV prefetch) before `finish` waits for the 7 peer slots. No per-AR
            barrier: slot reuse is safe because a rank can only reach the next use of a slot
            after receiving every peer's contribution to the other slot in between.
            With `AR_FOLD` only the B live rows travel, as [B*Hp/128, 128] f32 holding
            AR-dtype-rounded values (same sum as an AR-dtype payload).
            """
            if not comm:
                return lambda: val.astype(ardt).astype(F32)
            me = lax.axis_index("tp").astype(I32)
            if exchange and used_before:

                @pl.when(l > 0)  # this slot's sends from the previous layer have left
                def _():
                    for d in ar_copies(slot, me):
                        d.wait_send()

            if c.AR_RSAG:
                CW = c.CWA
                v = val.astype(ardt)
                for j in range(R):
                    w = max(0, min(Hp - j * CW, CW))
                    if w == CW:
                        S["send"][slot, j] = v[:, j * CW : (j + 1) * CW]
                    else:
                        S["send"][slot, j] = jnp.zeros((BR, CW), ardt)
                        if w:
                            S["send"][slot, j, :, 0:w] = v[:, j * CW : j * CW + w]
                S["recv"][slot, me] = S["send"][slot, me]
                descs = rs_copies(slot, me) if exchange else []
                for d in descs:
                    d.start()

                def finish_rsag():
                    for d in descs:
                        d.wait_recv()
                    red = S["recv"][slot, 0].astype(F32)
                    for r in range(1, R):
                        red = red + S["recv"][slot, r].astype(F32)
                    S["ag_send"][slot] = red
                    S["ag_recv"][slot, me] = red
                    ag_descs = ag_copies(slot, me) if exchange else []
                    for d in ag_descs:
                        d.start()
                    for d in ag_descs:
                        d.wait_recv()
                    for j in range(R):
                        w = max(0, min(Hp - j * CW, CW))
                        if w:
                            S["arout"][:, j * CW : j * CW + w] = S["ag_recv"][
                                slot, j, :, 0:w
                            ]
                    return S["arout"][...]

                return finish_rsag

            if c.AR_FOLD:
                v = val.astype(ardt).astype(F32)
                for j in range(NHB):
                    S["send"][slot, j * B : (j + 1) * B, :] = v[
                        0:B, j * 128 : (j + 1) * 128
                    ]
            else:
                S["send"][slot] = val.astype(ardt)
            S["recv"][slot, me] = S["send"][slot]
            descs = oneshot_copies(slot, me) if exchange else []
            for d in descs:
                d.start()

            def finish():
                for d in descs:
                    d.wait_recv()
                tot = S["recv"][slot, 0].astype(F32)
                for r in range(1, R):
                    tot = tot + S["recv"][slot, r].astype(F32)
                if not c.AR_FOLD:
                    return tot
                for j in range(NHB):
                    S["arout"][0:B, j * 128 : (j + 1) * 128] = tot[j * B : (j + 1) * B]
                return S["arout"][...]

            return finish

        # ---------------------------------------------------------- A: QKV + RoPE
        def phase_a(l, s, kn_idx, pi=None):
            res = S["res"][...]
            var = jnp.sum(res * res, axis=-1, keepdims=True) * (1.0 / H)
            xn = (res * lax.rsqrt(var + c.eps) * vrow(s, V_LN1, Hp)).astype(BF16)
            qkv = jnp.dot(xn, S["awqkv"][s], preferred_element_type=F32)
            qkv = qkv + vrow(s, V_BQKV, c.QKVWL)
            lane = lax.broadcasted_iota(I32, (BR, 128), 1)
            first = (lane % 64) < 32
            cs, sn = cos_ref[...], sin_ref[...]

            def rp(blk):
                rot = jnp.where(first, pltpu.roll(blk, 96, 1), pltpu.roll(blk, 32, 1))
                return blk * cs + rot * sn

            low = lane < 64
            for j in range(QWL // 128):
                blk = rp(qkv[:, j * 128 : (j + 1) * 128]) * scale  # heads 2j | 2j+1
                for h in (2 * j, 2 * j + 1):
                    qv = pltpu.roll(blk, 64, 1) if h % 2 else blk
                    S["qh"][pl.ds(h, BR, stride=NQL)] = jnp.where(low, qv, 0.0)
            k = rp(qkv[:, QWL : QWL + 128])
            v = qkv[:, QWL + 128 : QWL + 256]
            if (
                c.kv_dtype == "fp8"
            ):  # saturate like the stock path's fp8 KV quantization
                k, v = jnp.clip(k, -448.0, 448.0), jnp.clip(v, -448.0, 448.0)
            kq, vq = k.astype(kvdt), v.astype(kvdt)
            kf, vf = kq.astype(F32), vq.astype(F32)
            for r in range(NQL):
                S["kr8"][pl.ds(r, BR, stride=NQL)] = kf
                S["vr8"][pl.ds(r, BR, stride=NQL)] = vf
            store_words(S["kvn"], kv_encode(kq, vq))
            d = cp(S["kvn"], outs[1].at[kn_idx], S["sem_m"].at[0])
            d.start()
            if not KW:
                return (d,)
            # write each live sequence's packed new-token slot into its page, in place
            pool_out = outs[2] if s == 0 else outs[3]

            def tok_cp(b):
                return cp(
                    S["kvn"].at[pl.ds(b, 1)],
                    pool_out.at[pi, newpg_s[s * B + b], pl.ds(pos_s[b] % PAGE, 1)],
                    S["sem_m"].at[1],
                )

            def start(b, _):
                tok_cp(b).start()
                return 0

            lax.fori_loop(0, B, start, 0)

            class _WaitAll:
                @staticmethod
                def wait():
                    def w(b, _):
                        tok_cp(0).wait()
                        return 0

                    lax.fori_loop(0, B, w, 0)

            return (d, _WaitAll)

        # ---------------------------------------------------------- B: paged attention
        def attn_helpers(pi, sliding):
            pool = kv_swa if sliding else kv_full
            nb = c.nb_swa if sliding else c.nb_full
            NPBL = c.npb_swa if sliding else c.npb
            TKL = NPBL * PAGE

            def rng(b):
                pos = pos_s[b]
                lo = jnp.maximum(pos - (W - 1), 0) if sliding else jnp.int32(0)
                plo = lo // PAGE  # first page holding an attended history token
                npg = jnp.where(pos > 0, (jnp.maximum(pos - 1, 0) // PAGE) - plo + 1, 0)
                return pos, lo, plo, npg, jnp.maximum((npg + NPBL - 1) // NPBL, 1)

            def page_cp(slot, i, page, lp=None):
                lp = pi if lp is None else lp
                src = pool[lp].at[page] if c.split_pools else pool.at[lp, page]
                return cp(
                    src, S["kvb"].at[slot, pl.ds(i * PAGE, PAGE)], S["sem_kv"].at[slot]
                )

            def page_start(slot, i, page):
                if c.split_pools and not isinstance(pi, int):
                    # per-layer refs cannot be indexed by the traced pair index
                    for lp in range(L // 2):

                        @pl.when(pi == lp)
                        def _():
                            page_cp(slot, i, page, lp).start()
                else:
                    page_cp(slot, i, page).start()

            def kv_start(k, slot):
                b, j = S["it_b"][k], S["it_j"][k]
                _, _, plo, npg, _ = rng(b)
                for i in range(NPBL):
                    q = j * NPBL + i  # page index within the attended range

                    @pl.when(q < npg)
                    def _():
                        if sliding:
                            page = ptw_s[b * NWP + q]
                        else:
                            page = ptf_s[b * MP + jnp.minimum(plo + q, MP - 1)]
                        page_start(slot, i, jnp.clip(page, 0, nb - 1))

            def kv_wait(k, slot):
                b, j = S["it_b"][k], S["it_j"][k]
                npg = rng(b)[3]
                for i in range(NPBL):

                    @pl.when(j * NPBL + i < npg)
                    def _():
                        # a wait only uses the destination/semaphore: any layer's ref works
                        page_cp(slot, i, 0, 0 if c.split_pools else None).wait()

            return rng, kv_start, kv_wait, NPBL, TKL

        def phase_b_prepare(pi, sliding):
            """List the layer's (sequence, page-block) items on the scalar core and start the
            first `kv_bufs - 1` items' page DMAs. Independent of the residual stream, so the
            megakernel runs it for layer l+1 while layer l's MoE all-reduce is in flight."""
            rng, kv_start, _, _, _ = attn_helpers(pi, sliding)

            def collect(b, k):
                n = rng(b)[4]

                def one(j, k):
                    S["it_b"][k] = b
                    S["it_j"][k] = j
                    return k + 1

                return lax.fori_loop(0, n, one, k)

            n_items = lax.fori_loop(0, B, collect, jnp.int32(0))
            S["nit"][0] = n_items
            for k in range(NBUF - 1):

                @pl.when(k < n_items)
                def _():
                    kv_start(k, k)

        def phase_b_run(s, pi, sliding):
            """Attention for all sequences as one flat stream of the prepared items: KV page
            DMAs run `kv_bufs - 1` items ahead of compute across sequence boundaries; each
            item's pages share one DMA semaphore (one page-sized wait per started page)."""
            rng, kv_start, kv_wait, NPBL, TKL = attn_helpers(pi, sliding)
            n_items = S["nit"][0]

            tokoff = lax.broadcasted_iota(I32, (NQL, TKL), 1)
            row8 = lax.broadcasted_iota(I32, (NQL, 128), 0)
            lane8 = lax.broadcasted_iota(I32, (NQL, 128), 1)
            odd = (row8 % 2) == 1
            low = lane8 < 64
            sink = S["awsink"][s]
            zeros = jnp.zeros((NQL, 128), F32)

            def item(k, carry):
                m, l_, acc = carry
                slot = k % NBUF
                kv_wait(k, slot)

                @pl.when(k + NBUF - 1 < n_items)
                def _():
                    kv_start(k + NBUF - 1, (k + NBUF - 1) % NBUF)

                b, j = S["it_b"][k], S["it_j"][k]
                pos, lo, plo, npg, n = rng(b)
                rows = pl.ds(pl.multiple_of(b * NQL, NQL), NQL)
                qbf = S["qh"][rows].astype(BF16)
                # first item of a sequence: restart the online softmax from sink + new token
                snew = jnp.sum(qbf.astype(F32) * S["kr8"][rows], axis=1, keepdims=True)
                m0 = jnp.maximum(sink, snew)
                pnew = jnp.exp(snew - m0)
                first = j == 0
                m = jnp.where(first, m0, m)
                l_ = jnp.where(first, jnp.exp(sink - m0) + pnew, l_)
                acc = jnp.where(
                    first, pnew.astype(BF16).astype(F32) * S["vr8"][rows], acc
                )

                kt, vt = kv_decode(load_words(slot, TKL))
                sc = lax.dot_general(qbf, kt, NT_DIMS, preferred_element_type=F32)
                kpos = (plo + j * NPBL) * PAGE + tokoff
                valid = (kpos >= lo) & (kpos < pos)
                sc = jnp.where(valid, sc, NEG)
                m_new = jnp.maximum(m, jnp.max(sc, axis=1, keepdims=True))
                alpha = jnp.exp(m - m_new)
                p = jnp.exp(sc - m_new[:, :1])
                l_ = alpha * l_ + jnp.sum(p, axis=1, keepdims=True)
                acc = alpha * acc + jnp.dot(
                    p.astype(BF16), vt, preferred_element_type=F32
                )

                @pl.when(j == n - 1)
                def _():
                    o = jnp.where(low, acc / l_, 0.0)
                    S["oh"][rows] = jnp.where(odd, pltpu.roll(o, 64, 1), o)

                return m_new, l_, acc

            lax.fori_loop(0, n_items, item, (zeros, zeros, zeros))

        def phase_c(s):
            # head pair (2j, 2j+1) occupy the two halves of o_proj input block j
            blocks = [
                S["oh"][pl.ds(2 * j, BR, stride=NQL)]
                + S["oh"][pl.ds(2 * j + 1, BR, stride=NQL)]
                for j in range(NQL // 2)
            ]
            o = jnp.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
            return jnp.dot(o.astype(BF16), S["awo"][s], preferred_element_type=F32)

        # ---------------------------------------------------------- D: router
        def phase_d(s):
            res = S["res"][...]
            var = jnp.sum(res * res, axis=-1, keepdims=True) * (1.0 / H)
            xn = (res * lax.rsqrt(var + c.eps) * vrow(s, V_LN2, Hp)).astype(BF16)
            S["xn"][...] = xn
            lg = jnp.dot(xn, S["awr"][s], preferred_element_type=F32) + vrow(
                s, V_BR, 128
            )
            lane = lax.broadcasted_iota(I32, (BR, 128), 1)
            live = lax.broadcasted_iota(I32, (BR, 128), 0) < B
            ids, vals = [], []
            for _ in range(c.TOPK):
                mx = jnp.max(lg, axis=1, keepdims=True)
                ix = jnp.min(jnp.where(lg == mx, lane, 128), axis=1, keepdims=True)
                lg = jnp.where(lane == ix, -jnp.inf, lg)
                ids.append(ix)
                vals.append(mx)
            ex = [jnp.exp(v - vals[0]) for v in vals]
            den = sum(ex)
            wf = jnp.zeros((BR, 128), F32)
            hit = jnp.zeros((BR, 128), I32)
            for ix, e_ in zip(ids, ex):
                wf = wf + jnp.where(lane == ix, e_ / den, 0.0)
                hit = hit | (lane == ix).astype(I32)
            S["wf"][...] = jnp.where(live, wf, 0.0)
            act = jnp.max(jnp.where(live, hit, 0), axis=0, keepdims=True)
            S["actv"][...] = jnp.broadcast_to(act, (8, 128))
            d = cp(S["actv"], S["acts"], S["sem_m"].at[2])
            d.start()
            d.wait()
            base = rank * EL if c.moe_mode == "ep" else jnp.int32(0)

            def collect(e, n):
                S["ite"][n] = e
                return n + (S["acts"][0, base + e] != 0).astype(I32)

            return lax.fori_loop(0, EL, collect, jnp.int32(0)), base

        # ---------------------------------------------------------- E: experts
        def phase_e(l, n_act, base):
            NI = n_act * NJ
            if c.moe_mode == "ep":
                b2flag = jnp.float32(1.0)
            else:
                b2flag = jnp.where(rank == 0, 1.0, 0.0).astype(F32)

            def e_cps(i, slot):
                e = S["ite"][i // NJ]
                j = i % NJ
                sem = S["sem_e"]
                ds = [
                    cp(Wr["b13"].at[l, e, j], S["b13r"].at[slot], sem.at[slot, 0]),
                    cp(Wr["b2"].at[l, e], S["b2r"].at[slot], sem.at[slot, 1]),
                ]
                names = {
                    "mxfp4": ("w13q", "w13s", "w2q", "w2s"),
                    "fp8": ("w13", "c13", "w2", "c2"),
                }.get(c.w_dtype, ("w13", "w2"))
                for i, n in enumerate(names):
                    ds.append(
                        cp(Wr[n].at[l, e, j], S[n + "r"].at[slot], sem.at[slot, 2 + i])
                    )
                return ds

            for k in range(NS - 1):

                @pl.when(k < NI)
                def _():
                    for d in e_cps(k, k):
                        d.start()

            S["macc"][...] = jnp.zeros((BR, Hp), F32)
            lane = lax.broadcasted_iota(I32, (BR, 128), 1)

            def item(i, _):
                slot = i % NS
                for d in e_cps(i, slot):
                    d.wait()

                @pl.when(i + NS - 1 < NI)
                def _():
                    for d in e_cps(i + NS - 1, (i + NS - 1) % NS):
                        d.start()

                e = S["ite"][i // NJ]
                j = i % NJ
                if c.w_dtype == "mxfp4":
                    dequant(S["w13qr"], S["w13sr"], slot, S["w13r"], Hp)
                    dequant(S["w2qr"], S["w2sr"], slot, S["w2r"], T)
                    wslot = 0
                else:
                    wslot = slot
                h = jnp.dot(S["xn"][...], S["w13r"][wslot], preferred_element_type=F32)
                if c.w_dtype == "fp8":  # bf16 x e4m3 runs natively on the v7x MXU
                    h = h * S["c13r"][slot]
                h = h + S["b13r"][slot]
                g = jnp.minimum(h[:, :T], c.limit)
                u = jnp.clip(h[:, T:], -c.limit, c.limit)
                a = ((u + 1.0) * (g * jax.nn.sigmoid(c.alpha * g))).astype(BF16)
                y = jnp.dot(a, S["w2r"][wslot], preferred_element_type=F32)
                if c.w_dtype == "fp8":
                    y = y * S["c2r"][slot]
                y = y + jnp.where(j == 0, b2flag, 0.0) * S["b2r"][slot]
                wcol = jnp.sum(
                    jnp.where(lane == base + e, S["wf"][...], 0.0),
                    axis=1,
                    keepdims=True,
                )
                S["macc"][...] += wcol * y
                return 0

            lax.fori_loop(0, NI, item, 0)
            return S["macc"][...]

        # ---------------------------------------------------------- MXFP4 dequant
        def dequant(q_ref, s_ref, slot, out_ref, K):
            """uint32 codes [K/8, N] + bf16 scales [K/32, N] of ring slot -> bf16 [K, N].

            Static 128-row chunks (16 code rows, 4 scale rows) keep live values small; the
            e2m1 -> bf16 unpack is `pltpu.bitcast` + convert, as in gmm_v2's fp4 path.
            """
            for ch in range(K // 128):
                w = pltpu.bitcast(q_ref[slot, ch * 16 : (ch + 1) * 16, :], F4).astype(
                    BF16
                )
                for g in range(4):
                    r0 = ch * 128 + g * 32
                    sc = s_ref[slot, ch * 4 + g : ch * 4 + g + 1, :]
                    out_ref[0, r0 : r0 + 32, :] = w[g * 32 : (g + 1) * 32] * sc

        # ---------------------------------------------------------- composition
        if mode == "mega":
            if (
                comm and not exchange
            ):  # "ar" ablation: peers' slots must hold finite data
                for k in ("recv", "send", "ag_recv"):
                    if k in S:
                        S[k][...] = jnp.zeros(S[k].shape, S[k].dtype)
            if comm and c.AR_FOLD:
                S["arout"][...] = jnp.zeros((BR, Hp), F32)
            if exchange:
                me = lax.axis_index("tp").astype(I32)
                barrier = pltpu.get_barrier_semaphore()
                for off in range(1, R):
                    pl.semaphore_signal(
                        barrier,
                        1,
                        device_id=(me ^ off,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                pl.semaphore_wait(barrier, R - 1)
            for d in aw_copies(0, 0):
                d.start()
            S["res"][...] = x_ref[...].astype(F32)
            S["oh"][...] = jnp.zeros((BR * NQL, 128), F32)
            zero_kvb()

            if "attn" not in AB and "attn_kv" not in AB:
                phase_b_prepare(0, sliding=True)

            def pair(p, _):
                for s in (0, 1):
                    l = 2 * p + s
                    for d in aw_copies(0, s):
                        d.wait()

                    @pl.when(l + 1 < L)
                    def _():
                        for d in aw_copies(l + 1, 1 - s):
                            d.start()

                    if "attn" not in AB:
                        kn_descs = phase_a(l, s, l, p)
                        if "attn_kv" not in AB:
                            phase_b_run(s, p, sliding=(s == 0))
                        for d in kn_descs:
                            d.wait()
                        finish = ar_start(0, phase_c(s), l, True)
                        S["res"][...] = rnd(
                            S["res"][...] + finish() + vrow(s, V_BO, Hp)
                        )

                    def prepare_next():
                        if "attn" in AB or "attn_kv" in AB:
                            return
                        if s == 0:
                            phase_b_prepare(p, sliding=False)
                        else:

                            @pl.when(p + 1 < L // 2)
                            def _():
                                phase_b_prepare(p + 1, sliding=True)

                    if "moe" not in AB:
                        n_act, base = phase_d(s)
                        if "experts" in AB:
                            part = jnp.zeros((BR, Hp), F32)
                        else:
                            part = phase_e(l, n_act, base)
                        finish = ar_start(1, part, l, True)
                        prepare_next()  # overlaps the MoE all-reduce
                        S["res"][...] = rnd(S["res"][...] + finish())
                    else:
                        prepare_next()
                return 0

            lax.fori_loop(0, L // 2, pair, 0)
            outs[0][...] = S["res"][...].astype(BF16)
            if exchange:
                me = lax.axis_index("tp").astype(I32)
                used = [sl for sl, blk in ((0, "attn"), (1, "moe")) if blk not in AB]
                for slot in used:
                    for d in ar_copies(slot, me):
                        d.wait_send()
        else:
            l = layer
            for d in aw_copies(l, 0):
                d.start()
            for d in aw_copies(l, 0):
                d.wait()
            S["res"][...] = x_ref[...]
            if mode == "attn":
                S["oh"][...] = jnp.zeros((BR * NQL, 128), F32)
                zero_kvb()
                kn_descs = phase_a(l, 0, 0)
                phase_b_prepare(l // 2, sliding=c.sliding[l])
                phase_b_run(0, l // 2, sliding=c.sliding[l])
                for d in kn_descs:
                    d.wait()
                outs[0][...] = phase_c(0)
            else:
                n_act, base = phase_d(0)
                outs[0][...] = phase_e(l, n_act, base)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [smem] * 5 + [vmem] * 3 + [hbm] * (len(WN) + n_pool)
    nkn = L if mode == "mega" else 1
    kn_shape = jax.ShapeDtypeStruct((nkn, BR, 1, c.kv_packing, 128), kvdt)
    aliases = {}
    if mode == "mega":
        out_shape = (jax.ShapeDtypeStruct((BR, Hp), BF16), kn_shape)
        out_specs = (vmem, hbm)
        if KW:
            sw, fu = pool_specs(c)
            out_shape += (
                jax.ShapeDtypeStruct(sw.shape[1:], sw.dtype),
                jax.ShapeDtypeStruct(fu.shape[1:], fu.dtype),
            )
            out_specs += (hbm, hbm)
            aliases = {n_in - 2: 2, n_in - 1: 3}
    elif mode == "attn":
        out_shape = (jax.ShapeDtypeStruct((BR, Hp), F32), kn_shape)
        out_specs = (vmem, hbm)
    else:
        out_shape = jax.ShapeDtypeStruct((BR, Hp), F32)
        out_specs = vmem
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if exchange else None,
            vmem_limit_bytes=c.vmem_mib * 1024 * 1024,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"gptoss_decode_{mode}" + ("" if layer is None else f"_l{layer}"),
    )


def build_kv_scatter(c: DecodeConfig, *, interpret=False):
    """f(newpg, pos, kvnew, kv_swa, kv_full) -> (kv_swa, kv_full), in place.

    `newpg [2*B]`: page of each sequence's new token in the SWA pool (first B) and in the
    full pool. Writes each live sequence's packed new-token slot `[1, packing, 128]` (K, V)
    with one HBM->HBM DMA along the (untiled) token dim, as RPA v3's cache update does.
    Not communicating, so XLA honours the pool aliasing (no pool copy).
    """
    B, L, PAGE = c.B, c.L, c.page

    def kernel(newpg_s, pos_s, kvn, kvs_in, kvf_in, kvs, kvf, sem):
        del kvs_in, kvf_in

        def pair(p, _):
            for s in (0, 1):
                l = 2 * p + s
                pool = kvs if s == 0 else kvf

                def cp(b):
                    pos = pos_s[b]
                    return pltpu.make_async_copy(
                        kvn.at[l, pl.ds(b, 1)],
                        pool.at[p, newpg_s[s * B + b], pl.ds(pos % PAGE, 1)],
                        sem.at[0],
                    )

                def start(b, _):
                    cp(b).start()
                    return 0

                def wait(b, _):
                    cp(0).wait()
                    return 0

                lax.fori_loop(0, B, start, 0)
                lax.fori_loop(0, B, wait, 0)
            return 0

        lax.fori_loop(0, L // 2, pair, 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    sw, fu = pool_specs(c)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(sw.shape[1:], sw.dtype),
            jax.ShapeDtypeStruct(fu.shape[1:], fu.dtype),
        ),
        in_specs=[smem] * 2 + [hbm] * 3,
        out_specs=(hbm, hbm),
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        input_output_aliases={3: 0, 4: 1},
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="gptoss_decode_kv_scatter",
    )


def page_args(c: DecodeConfig, pos, pt_swa, pt_full):
    """XLA-side page-table glue (tiny): window pages for sliding layers [B*NWP], the full
    table [B*MP], and the new token's page in each pool [2*B]."""
    B, MP, NWP = c.B, c.max_pages, c.NWP
    plo = jnp.maximum(pos - (c.window - 1), 0) // c.page
    widx = jnp.minimum(plo[:, None] + jnp.arange(NWP)[None, :], MP - 1)
    ptw = jnp.take_along_axis(pt_swa, widx, axis=1)
    cur = (pos // c.page)[:, None]
    newpg = jnp.concatenate(
        [
            jnp.take_along_axis(pt_swa, cur, 1)[:, 0],
            jnp.take_along_axis(pt_full, cur, 1)[:, 0],
        ]
    )
    return (
        ptw.reshape(-1).astype(I32),
        pt_full.reshape(-1).astype(I32),
        newpg.astype(I32),
    )


# ----------------------------------------------------------------------------- step


def make_step(
    c: DecodeConfig,
    mesh,
    *,
    xla_ar: bool = False,
    interpret: bool = False,
    scatter: bool = True,
):
    """Jittable decode step over `mesh` (1-D axis "tp" of size R).

    step(x [B, H] bf16, pos [B] i32, pt_swa [B, MP] i32, pt_full [B, MP] i32, W, kv_swa, kv_full)
      -> (hidden [B, H] bf16, kv_swa, kv_full)
    W: dict from `pack_weights` (leading axis R). Pools: `pool_specs` (leading axis R).
    xla_ar=True runs per-layer attn/moe kernels with XLA psum in between (fallback).
    """
    R = c.R
    mega = None if xla_ar else build(c, "mega", interpret=interpret)
    attn = (
        [build(c, "attn", l, interpret=interpret) for l in range(c.L)]
        if xla_ar
        else None
    )
    moe = (
        [build(c, "moe", l, interpret=interpret) for l in range(c.L)]
        if xla_ar
        else None
    )
    in_kernel_kv = (not xla_ar) and c.kv_write == "kernel"
    sc = (
        build_kv_scatter(c, interpret=interpret)
        if scatter and not in_kernel_kv
        else None
    )
    ardt = ar_jdtype(c)

    def per_rank(rank, x, pos, pt_s, pt_f, W, kv_s, kv_f):
        W = {k: v[0] for k, v in W.items()}
        kv_s, kv_f = kv_s[0], kv_f[0]
        rank = rank.astype(I32)
        pos = pos.astype(I32)
        pt_s, pt_f, newpg = page_args(c, pos, pt_s.astype(I32), pt_f.astype(I32))
        cos, sin = rope_tables(c, pos)
        xp = jnp.pad(x, ((0, c.BR - c.B), (0, c.Hp - c.H))).astype(BF16)
        wl = [W[k] for k in w_names(c)]
        if not xla_ar:
            res_ = mega(rank, pos, pt_s, pt_f, newpg, xp, cos, sin, *wl, kv_s, kv_f)
            out, kvn = res_[0], res_[1]
            if in_kernel_kv:
                kv_s, kv_f = res_[2], res_[3]
        else:
            res = xp.astype(F32)
            kvns = []
            for l in range(c.L):
                part, kvn_l = attn[l](
                    rank, pos, pt_s, pt_f, newpg, res, cos, sin, *wl, kv_s, kv_f
                )
                tot = lax.psum(part.astype(ardt).astype(F32), "tp")
                bo = W["vec"][l, V_BO : V_BO + 1, : c.Hp]
                res = (res + tot + bo).astype(BF16).astype(F32)
                part = moe[l](
                    rank, pos, pt_s, pt_f, newpg, res, cos, sin, *wl, kv_s, kv_f
                )
                tot = lax.psum(part.astype(ardt).astype(F32), "tp")
                res = (res + tot).astype(BF16).astype(F32)
                kvns.append(kvn_l)
            out, kvn = res.astype(BF16), jnp.concatenate(kvns)
        if sc is not None:
            kv_s, kv_f = sc(newpg, pos, kvn, kv_s, kv_f)
        return out[: c.B, : c.H], kv_s[None], kv_f[None]

    w_spec = {k: P("tp") for k in w_names(c)}
    smap = jax.shard_map(
        per_rank,
        mesh=mesh,
        in_specs=(P("tp"), P(), P(), P(), P(), w_spec, P("tp"), P("tp")),
        out_specs=(P(), P("tp"), P("tp")),
        check_vma=False,
    )

    def step(x, pos, pt_swa, pt_full, W, kv_swa, kv_full):
        rank = jnp.arange(R, dtype=I32)
        return smap(rank, x, pos, pt_swa, pt_full, W, kv_swa, kv_full)

    return step
