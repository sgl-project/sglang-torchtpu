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
"""Pure-JAX reference of one GPT-OSS TP-R decode step (all layers, T=1 per sequence).

Model math as in sglang's `gpt_oss.py`: RMSNorm, QKV with bias, YaRN neox RoPE, attention
with per-head sinks and a 128-token sliding window on even layers, o_proj with bias, router
with bias, softmax over the top-k logits, swigluoai experts (clamped, alpha 1.702) with biases.

The weights are *global, model-layout, unpadded* (`LayerWeights`). The reference reproduces
the tensor-parallel decomposition explicitly so it can be compared bit-for-bit-ish with the
sharded kernel: each rank's o_proj / MoE partial is rounded to the all-reduce dtype and the
partials are summed in f32 in rank order (exactly what both the in-kernel all-reduce and the
XLA-collective fallback compute). Matmul inputs are bf16 with f32 accumulation; the residual
stream is rounded to bf16 after every add; new K/V are rounded to the KV-cache dtype before
they are attended (as the stock ragged-paged-attention kernel does).

KV history is a dense per-sequence tensor here; `megakernel.build_pools` scatters it into the
paged sglang TPU pool layout for the kernel.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from .config import DecodeConfig

BF16 = jnp.bfloat16
F32 = jnp.float32


def kv_jdtype(c: DecodeConfig):
    return jnp.float8_e4m3fn if c.kv_dtype == "fp8" else BF16


def ar_jdtype(c: DecodeConfig):
    return BF16 if c.ar_dtype == "bf16" else F32


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "ln1",
        "wqkv",
        "bqkv",
        "sinks",
        "wo",
        "bo",
        "ln2",
        "wr",
        "br",
        "w13",
        "b13",
        "w2",
        "b2",
    ],
    meta_fields=[],
)
@dataclasses.dataclass
class LayerWeights:
    ln1: jax.Array  # [H]
    wqkv: jax.Array  # [H, NQ*D + 2*NKV*D]  (q | k | v)
    bqkv: jax.Array  # [NQ*D + 2*NKV*D]
    sinks: jax.Array  # [NQ]
    wo: jax.Array  # [NQ*D, H]
    bo: jax.Array  # [H]
    ln2: jax.Array  # [H]
    wr: jax.Array  # [H, E]
    br: jax.Array  # [E]
    w13: jax.Array  # [E, H, 2*I]  (gate | up)
    b13: jax.Array  # [E, 2*I]
    w2: jax.Array  # [E, I, H]
    b2: jax.Array  # [E, H]


def random_layer(c: DecodeConfig, key, scale: float = 1.0) -> LayerWeights:
    ks = jax.random.split(key, 13)
    H, D, NQ, NKV, E, I = c.H, c.D, c.NQ, c.NKV, c.E, c.I
    qkvw = NQ * D + 2 * NKV * D

    def n(k, shape, s):
        return (jax.random.normal(k, shape, F32) * s).astype(BF16)

    return LayerWeights(
        ln1=(1.0 + 0.1 * jax.random.normal(ks[0], (H,), F32)).astype(BF16),
        wqkv=n(ks[1], (H, qkvw), scale / np.sqrt(H)),
        bqkv=n(ks[2], (qkvw,), 0.1),
        sinks=jax.random.normal(ks[3], (NQ,), F32).astype(BF16),
        wo=n(ks[4], (NQ * D, H), scale / np.sqrt(NQ * D)),
        bo=n(ks[5], (H,), 0.1),
        ln2=(1.0 + 0.1 * jax.random.normal(ks[6], (H,), F32)).astype(BF16),
        wr=n(ks[7], (H, E), 1.0 / np.sqrt(H)),
        br=n(ks[8], (E,), 0.1),
        w13=n(ks[9], (E, H, 2 * I), scale / np.sqrt(H)),
        b13=n(ks[10], (E, 2 * I), 0.1),
        w2=n(ks[11], (E, I, H), scale / np.sqrt(I)),
        b2=n(ks[12], (E, H), 0.1),
    )


# Same YaRN table construction as the T20 prefill megakernel's reference.
def yarn_tables(c: DecodeConfig, positions, beta_fast=32.0, beta_slow=1.0):
    """[N, D] cos and sin (neox halves duplicated), YaRN-scaled as GPT-OSS configures it."""
    D, theta, factor, orig = c.D, c.rope_theta, c.rope_factor, c.rope_orig
    pos_freqs = theta ** (jnp.arange(0, D, 2, dtype=F32) / D)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)

    def corr(rot):
        return (D * np.log(orig / (rot * 2 * np.pi))) / (2 * np.log(theta))

    low = max(np.floor(corr(beta_fast)), 0)
    high = min(np.ceil(corr(beta_slow)), D - 1)
    ramp = jnp.clip((jnp.arange(D // 2, dtype=F32) - low) / max(high - low, 1e-3), 0, 1)
    extra_mask = 1.0 - ramp
    inv_freq = inv_inter * (1 - extra_mask) + inv_extra * extra_mask
    mscale = 0.1 * np.log(factor) + 1.0
    f = positions.astype(F32)[:, None] * inv_freq[None, :]
    cos = jnp.cos(f) * mscale
    sin = jnp.sin(f) * mscale
    return jnp.concatenate([cos, cos], -1), jnp.concatenate([sin, sin], -1)


def rmsnorm(x, w, eps):
    xf = x.astype(F32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(var + eps) * w.astype(F32)).astype(BF16)


def rope(x, cos, sin):  # x [B, nh, D] f32
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    rot = jnp.concatenate([-x2, x1], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def mm(a, b):
    return jnp.dot(a.astype(BF16), b.astype(BF16), preferred_element_type=F32)


def swigluoai(c: DecodeConfig, g, u):
    g = jnp.minimum(g, c.limit)
    u = jnp.clip(u, -c.limit, c.limit)
    return (u + 1.0) * (g * jax.nn.sigmoid(c.alpha * g))


def tp_sum(c: DecodeConfig, partials):
    """All-reduce semantics: round each rank's f32 partial to the AR dtype, sum in f32."""
    tot = None
    for p in partials:
        p = p.astype(ar_jdtype(c)).astype(F32)
        tot = p if tot is None else tot + p
    return tot


def attention(c: DecodeConfig, q, kn, vn, kh, vh, pos, sinks, sliding):
    """q [B, NQ, D] f32 (unscaled); kn/vn [B, NKV, D] dequantized new K/V; kh/vh [B, T, NKV, D]
    dequantized history (slot t holds position t); attends history [lo, pos) plus the new token."""
    B, T = kh.shape[0], kh.shape[1]
    G = c.NQ // c.NKV
    qb = (q * c.sm_scale).astype(BF16)
    k_all = jnp.concatenate([kh, kn[:, None]], 1)  # [B, T+1, NKV, D]
    v_all = jnp.concatenate([vh, vn[:, None]], 1)
    t = jnp.arange(T + 1)[None, :]
    p_ = pos[:, None]
    lo = jnp.maximum(p_ - (c.window - 1), 0) if sliding else jnp.zeros_like(p_)
    valid = ((t >= lo) & (t < p_)) | (t == T)  # [B, T+1]
    kr = jnp.repeat(k_all, G, axis=2).astype(BF16)  # [B, T+1, NQ, D]
    vr = jnp.repeat(v_all, G, axis=2).astype(BF16)
    s = jnp.einsum("bhd,bthd->bht", qb, kr, preferred_element_type=F32)
    s = jnp.where(valid[:, None, :], s, -jnp.inf)
    sk = sinks.astype(F32)[None, :, None]
    m = jnp.maximum(jnp.max(s, -1, keepdims=True), sk)
    p = jnp.exp(s - m)
    l = jnp.sum(p, -1, keepdims=True) + jnp.exp(sk - m)
    o = jnp.einsum("bht,bthd->bhd", p.astype(BF16), vr, preferred_element_type=F32)
    return o / l  # [B, NQ, D] f32


def moe_partials(c: DecodeConfig, x2, ids, wts, w: LayerWeights):
    """Per-rank f32 MoE partials (dense over experts; exact for routed slots)."""
    R, I, E = c.R, c.I, c.E
    wfull = jnp.zeros((x2.shape[0], E), F32)
    for k in range(c.TOPK):
        wfull = wfull + jax.nn.one_hot(ids[:, k], E, dtype=F32) * wts[:, k : k + 1]
    parts = []
    for r in range(R):
        if c.moe_mode == "tp":
            cols = np.arange(*c.moe_cols(r))
            experts, rows, b2flag = range(E), cols, 1.0 if r == 0 else 0.0
        else:
            el = E // R
            cols = np.arange(I)
            experts, rows, b2flag = range(r * el, (r + 1) * el), cols, 1.0
        acc = jnp.zeros((x2.shape[0], c.H), F32)
        for e in experts:
            g_w = w.w13[e][:, cols]
            u_w = w.w13[e][:, I + cols]
            h_g = mm(x2, g_w) + w.b13[e][cols].astype(F32)
            h_u = mm(x2, u_w) + w.b13[e][I + cols].astype(F32)
            a = swigluoai(c, h_g, h_u).astype(BF16)
            y = mm(a, w.w2[e][rows]) + b2flag * w.b2[e].astype(F32)
            acc = acc + wfull[:, e : e + 1] * y
        parts.append(acc)
    return parts


# ----------------------------------------------------------------------------- MXFP4
FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32
)  # e2m1 codes 0..7


def mxfp4_quantize(w):
    """w [..., K, N] -> (codes uint8 [..., Kp, N] (e2m1, sign bit 8), scale f32 [..., Kp/32, N]).

    OCP MX: one power-of-two scale per 32 consecutive K elements, 2^(floor(log2 amax) - 2);
    K is zero-padded to a multiple of 32. Idempotent on already-representable weights.
    """
    w = np.asarray(jnp.asarray(w).astype(F32))
    K = w.shape[-2]
    Kp = -(-K // 32) * 32
    pad = [(0, 0)] * (w.ndim - 2) + [(0, Kp - K), (0, 0)]
    w = np.pad(w, pad)
    blk = w.reshape(w.shape[:-2] + (Kp // 32, 32, w.shape[-1]))
    amax = np.max(np.abs(blk), axis=-2)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))) - 2, -126)
    scale = np.exp2(np.clip(e, -126, 127)).astype(np.float32)
    x = np.clip(blk / scale[..., None, :], -6.0, 6.0)
    mag = np.abs(x)
    code = np.argmin(np.abs(mag[..., None] - FP4_VALUES), axis=-1).astype(np.uint8)
    code = code | np.where((x < 0) & (code > 0), 8, 0).astype(np.uint8)
    return code.reshape(w.shape), scale


def mxfp4_dequantize(code, scale, K=None):
    v = FP4_VALUES[code & 7] * np.where(code & 8, -1.0, 1.0).astype(np.float32)
    v = v * np.repeat(scale, 32, axis=-2)
    return v if K is None else v[..., :K, :]


def mxfp4_roundtrip(c: DecodeConfig, w: LayerWeights) -> LayerWeights:
    """Expert weights snapped to MXFP4-representable values (blocks along each K dim)."""
    w13 = mxfp4_dequantize(*mxfp4_quantize(w.w13), K=c.H)
    w2 = mxfp4_dequantize(*mxfp4_quantize(w.w2), K=c.I)
    return dataclasses.replace(
        w, w13=jnp.asarray(w13).astype(BF16), w2=jnp.asarray(w2).astype(BF16)
    )


def route(c: DecodeConfig, x2, w: LayerWeights, margins=None):
    logits = mm(x2, w.wr) + w.br.astype(F32)
    vals, ids = jax.lax.top_k(logits, c.TOPK)
    if margins is not None:  # gap between the k-th and (k+1)-th logit: tiny = near-tie
        nxt = jax.lax.top_k(logits, c.TOPK + 1)[0][:, -1]
        margins.append(vals[:, -1] - nxt)
    ex = jnp.exp(vals - vals[:, :1])
    return ids.astype(jnp.int32), ex / jnp.sum(ex, -1, keepdims=True)


def layer(
    c: DecodeConfig,
    res,
    cos,
    sin,
    pos,
    kh,
    vh,
    w: LayerWeights,
    sliding: bool,
    margins=None,
):
    """res [B, H] bf16; kh/vh [B, T, NKV, D] in the KV dtype. Returns res, (k_new, v_new)."""
    B = res.shape[0]
    NQ, NKV, D, R = c.NQ, c.NKV, c.D, c.R
    qw, kw = NQ * D, NKV * D
    a = rmsnorm(res, w.ln1, c.eps)
    qkv = mm(a, w.wqkv) + w.bqkv.astype(F32)
    q = rope(qkv[:, :qw].reshape(B, NQ, D), cos, sin)
    k = rope(qkv[:, qw : qw + kw].reshape(B, NKV, D), cos, sin)
    v = qkv[:, qw + kw :].reshape(B, NKV, D)
    kq, vq = k.astype(kv_jdtype(c)), v.astype(kv_jdtype(c))
    o = attention(
        c,
        q,
        kq.astype(F32),
        vq.astype(F32),
        kh.astype(F32),
        vh.astype(F32),
        pos,
        w.sinks,
        sliding,
    )
    o = o.reshape(B, qw).astype(BF16)
    ql = qw // R
    parts = [
        mm(o[:, r * ql : (r + 1) * ql], w.wo[r * ql : (r + 1) * ql]) for r in range(R)
    ]
    res = (res.astype(F32) + tp_sum(c, parts) + w.bo.astype(F32)).astype(BF16)
    x2 = rmsnorm(res, w.ln2, c.eps)
    ids, wts = route(c, x2, w, margins)
    res = (res.astype(F32) + tp_sum(c, moe_partials(c, x2, ids, wts, w))).astype(BF16)
    return res, (kq, vq)


def step(
    c: DecodeConfig, x, pos, k_hist, v_hist, weights: list[LayerWeights], margins=None
):
    """x [B, H] bf16 (embedded token); pos [B]; k_hist/v_hist [L, B, T, NKV, D] (KV dtype).

    Returns hidden [B, H] bf16 before the final norm, and per-layer new K/V [L, B, NKV, D].
    `margins`: optional list that receives each layer's top-k routing margin [B].
    """
    cos, sin = yarn_tables(c, pos)
    res = x
    kn, vn = [], []
    for li, w in enumerate(weights):
        res, (k, v) = layer(
            c, res, cos, sin, pos, k_hist[li], v_hist[li], w, c.sliding[li], margins
        )
        kn.append(k)
        vn.append(v)
    return res, jnp.stack(kn), jnp.stack(vn)
