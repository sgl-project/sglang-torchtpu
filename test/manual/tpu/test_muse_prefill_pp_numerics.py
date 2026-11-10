"""Numerics of the PP prefill stage (sglang_tpu.muse_megakernel.prefill_pp: its fused kernels
around the batched RPA prefill attention, with the serving env) on one TPU device, against a
plain-jnp reference of each layer (f32, HIGHEST precision dots, bf16 queries, the kernels' bf16 /
fp8 rounding points).

One stage of three layers at the real per-layer geometry (all 128 q / 16 kv heads, 256
experts): an fp8-expert sliding-window layer, an fp4 global layer and an fp4 sliding-window
layer, a 512-token chunk after 2560 cached tokens (the window cuts the cached keys, the global
layer reads them all); layer 0 sits on slot 1 so the global layer is on cfg.global_slots. The
stage runs with 1, 2 and 3 valid layers; layer l's reference is
fed the kernel's stream after l - 1 layers (teacher forcing), and its chunk K/V is compared
with what the kernel wrote into the pool.

MUSE_PP_QBLOCK=256 requantizes the NVFP4 experts to block-256 scales (the reference uses the
requantized weights) and also reports the expert outputs against the exact block-16 reference;
MUSE_PP_QBLOCK_A8=1 adds the fp8 rows of gmm_v2's W4A8 product to the reference.

    python3 test/manual/tpu/test_muse_prefill_pp_numerics.py
"""

import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.muse_megakernel import prefill_pp as PP
from sglang_tpu.muse_megakernel.prefill_pp import moe as PP_MOE
from sglang_tpu.muse_megakernel.prefill_pp.config import fp4_qblock, fp4_qblock_a8

# the batched RPA prefill attention as the P/D prefill serves it (read when the stage is traced);
# MUSE_PP_TEST_RPA_TRANSPOSED=0 runs the standard tile (bf16 queries) as a control
TRANSPOSED = os.environ.get("MUSE_PP_TEST_RPA_TRANSPOSED", "1")
os.environ.update({
    "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1", "SGLANG_TPU_RPA_SKIP_PAD_Q": "1",
    "SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP": "256", "SGLANG_TPU_RPA_NO_MAX": "1",
    "SGLANG_TPU_RPA_OUT_HEAD_NORM_EPS": "1e-05", "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK": "1",
    "SGLANG_TPU_RPA_TRANSPOSED": TRANSPOSED,
})  # fmt: skip
if (
    TRANSPOSED == "1"
):  # the standard tile overflows VMEM at these blocks; its own search picks
    os.environ.update({
        "SGLANG_TPU_FORCE_RPA_PREFILL_BLOCKS": "256:32:768:2",
        "SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS": "128:512",
    })  # fmt: skip

HIGH = jax.lax.Precision.HIGHEST
T, CTX = 512, 2560
KINDS = ("fp8", "fp4", "fp4")
GLOBAL = (False, True, False)
ROPE_BASE = 10000.0
LOG2E = 1.4426950408889634
# per-row relative error of a layer's output stream: ~12 bf16 roundings per layer
TOL_ROW, TOL_MEAN = 3e-2, 1e-2
TOL_KV = 1e-6  # the chunk's fp8 K/V in the pool: exactly the kernel's k, v rounded
TOL_QKV = 1e-2  # attention inputs: bf16 roundings of nearly equal f32 values
# The served kernels against this f32 reference: RPA's standard bf16 tile alone is 1.6% off
# (bf16 P without a running max, outputs that nearly cancel before the head norm), the fp8-q
# transposed tile 2.2%; gmm_v2's W8A8 (fp8 layers) 3.6-4.3%, NVFP4 W4A16 0.45%. The port itself
# is checked exactly: the stage's attention and slot outputs equal the served calls' bits.
TOL_ATTN = 3e-2
TOL_EXPERTS = {"fp4": 1e-2, "fp8": 5e-2}
# a token whose top-k set differs from the reference's must be a near tie of biased scores
NEAR_TIE = 2e-3


def config() -> PP.PPConfig:
    return PP.PPConfig(
        hidden=8192, head_dim=64, q_heads=128, kv_heads=16, n_experts=256, top_k=8,
        expert_hidden=4096, inter=4096, rms_eps=1e-5, post_eps=1e-8, scale_query_by=5.473,
        attn_scale=0.125, router_mult=0.1768, route_scale=1.0, router_eps=1e-15, window=2048,
        rows=T, slots=len(KINDS) + 1, page=256, pages_per_seq=64, global_pages=65,
        swa_pages=27, max_position=16384, global_slots=(2,), fp8_slots=(1,),
    )  # fmt: skip


def _hash(shape, salt):
    i = jnp.zeros(shape, jnp.uint32)
    for d in range(len(shape)):
        i = i * jnp.uint32(shape[d]) + jax.lax.broadcasted_iota(jnp.uint32, shape, d)
    x = i * jnp.uint32(2654435761) + jnp.uint32(salt * 40503 + 1)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(2246822519)
    return x ^ (x >> 13)


def _unit(shape, salt):  # uniform [-1, 1)
    return (_hash(shape, salt) >> 8).astype(jnp.float32) / 2.0**23 - 1.0


_FP8_RMS = float(
    np.sqrt(np.mean(np.asarray(jnp.asarray(
        (np.arange(256) & 0xBF).astype(np.uint8)).view(jnp.float8_e4m3fn).astype(np.float32)) ** 2))
)  # fmt: skip
_E2M1 = jnp.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], jnp.float32
)
_E2M1_RMS = float(jnp.sqrt(jnp.mean(_E2M1**2)))


def _fp8_matrix(shape, salt):
    """fp8 values of rms _FP8_RMS: random bytes, top exponent bit clear."""
    b = (_hash(shape, salt) & 0xBF).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(b, jnp.float8_e4m3fn)


def _tiles(w, tn):
    """[K, N] -> column-tile major [N / tn, K, tn]."""
    k, n = w.shape
    return w.reshape(k, n // tn, tn).transpose(1, 0, 2)


def build_weights(c: PP.PPConfig, seed: int):
    S, H, He, E = c.slots, c.hidden, c.expert_hidden, c.n_experts
    qd, kvd, nq = c.q_dim, c.kv_dim, c.qkvg_dim
    dense = {
        "wqkvg": [],
        "wo": [],
        "wpool": [],
        "wunpool": [],
        "wrouter": [],
        "vecs": [],
    }
    for l in range(S):
        s0 = seed + 100 * l
        for i, (name, (k, n)) in enumerate(
            {
                "wqkvg": (H, nq),
                "wo": (qd, H),
                "wpool": (H, He),
                "wunpool": (He, H),
            }.items()
        ):
            dense[name].append(_tiles(_fp8_matrix((k, n), s0 + 50 + i), c.tn))
        dense["wrouter"].append(
            (_unit((H, E), s0 + 7) * (4 / H**0.5) * 3**0.5).astype(jnp.bfloat16)
        )
        vec = jnp.zeros((PP.V_ROWS, c.vec_width), jnp.float32)
        u = lambda n, salt: _unit((n,), s0 + salt)  # noqa: E731
        for name in ("pre_attn_w", "pre_ffn_w", "post_ffn_w"):
            vec = vec.at[PP.V[name], :H].set(1.0 + 0.2 * u(H, PP.V[name]))
        for pre in ("attn", "ffn"):
            z = 1.5 * u(H, 20 + len(pre))
            beta = jax.nn.sigmoid(z)
            vec = vec.at[PP.V[pre + "_beta"], :H].set(beta)
            vec = vec.at[PP.V[pre + "_alpha"], :H].set(
                jnp.sqrt(jnp.clip(jax.nn.sigmoid(-z) * (1 + beta), 1e-3))
            )
        for name in ("pre_expert_w", "post_expert_w"):
            vec = vec.at[PP.V[name], :He].set(1.0 + 0.2 * u(He, PP.V[name]))
        vec = vec.at[PP.V["expert_bias"], :E].set(0.02 * u(E, 30))
        for name, (k, n, gain) in {
            "wqkvg_s": (H, nq, 1.0), "wo_s": (qd, H, 1.0), "wpool_s": (H, He, 1.0),
            "wunpool_s": (He, H, 3.0),
        }.items():  # fmt: skip
            vec = vec.at[PP.V[name], :n].set(
                (1.0 + 0.2 * u(n, 40 + PP.V[name])) * gain / (_FP8_RMS * k**0.5)
            )
        dense["vecs"].append(vec)
    w = {n: jnp.stack(v) for n, v in dense.items()}
    # NVFP4 experts (random e2m1 codes, bf16 block scales paired into int32 words), one set per
    # NVFP4 layer, shared by the slots without one; the fp8 set (e4m3 bytes, top exponent bit
    # clear; f32 channel scales). Hashes are drawn as u32 and viewed as bytes.
    sets = {}
    first = c.global_slots[0] - GLOBAL.index(True)
    for l, kind in enumerate(KINDS):
        if kind != "fp4":
            continue
        s = first + l
        arrs = []
        for j, (shape, dt) in enumerate(PP.expert_shapes(c, "fp4")):
            salt = seed + 100 * s + j
            if dt == jnp.uint8:
                h = _hash((*shape[:-1], shape[-1] // 4), salt)
                arrs.append(jax.lax.bitcast_convert_type(h, jnp.uint8).reshape(shape))
            else:
                k = He if j == 1 else c.inter
                sc = (1.0 + 0.5 * _unit((*shape, 2), salt)) / (_E2M1_RMS * k**0.5)
                arrs.append(
                    jax.lax.bitcast_convert_type(sc.astype(jnp.bfloat16), jnp.int32)
                )
        sets[s] = tuple(arrs)
    w["fp4"] = [sets.get(s, next(iter(sets.values()))) for s in range(c.slots)]
    arrs = []
    for j, (shape, dt) in enumerate(PP.expert_shapes(c, "fp8")):
        salt = seed + 900 + j
        if dt == PP.FP8:
            h = _hash((*shape[:-1], shape[-1] // 4), salt) & jnp.uint32(0xBFBFBFBF)
            arrs.append(
                jax.lax.bitcast_convert_type(
                    jax.lax.bitcast_convert_type(h, jnp.uint8).reshape(shape), PP.FP8
                )
            )
        else:
            k = He if j == 1 else c.inter
            arrs.append((1.0 + 0.2 * _unit(shape, salt)) / (_FP8_RMS * k**0.5))
    w["fp8"] = tuple(arrs)
    return w


def rope_table(n):
    inv = 1.0 / (ROPE_BASE ** (np.arange(0, 64, 2, dtype=np.float64) / 64))
    ang = np.arange(n, dtype=np.float64)[:, None] * inv
    return jnp.asarray(np.concatenate(
        [np.repeat(np.cos(ang), 2, 1), np.repeat(np.sin(ang), 2, 1)], 1), jnp.float32)  # fmt: skip


def _codes_scales(packed, scales):
    """[K / 2, N] u8 e2m1 pairs + [K / (2 g), N] i32 bf16-scale pairs -> codes' values [K, N]
    and the block scales [K / g, N], f32."""
    codes = jnp.stack([packed & 0xF, packed >> 4], axis=1).reshape(-1, packed.shape[1])
    u = jax.lax.bitcast_convert_type(scales, jnp.uint32)
    halves = jnp.stack([u & 0xFFFF, u >> 16], axis=1).astype(jnp.uint16)
    sc = jax.lax.bitcast_convert_type(halves, jnp.bfloat16).astype(jnp.float32)
    return _E2M1[codes.astype(jnp.int32)], sc.reshape(-1, packed.shape[1])


def _dq4(packed, scales):
    """[K / 2, N] u8 e2m1 pairs (low nibble row 2b) + [K / (2 g), N] i32 bf16-scale pairs (low
    half block 2j) of g-row blocks -> [K, N] f32."""
    codes = jnp.stack([packed & 0xF, packed >> 4], axis=1).reshape(-1, packed.shape[1])
    v = _E2M1[codes.astype(jnp.int32)]
    u = jax.lax.bitcast_convert_type(scales, jnp.uint32)
    halves = jnp.stack([u & 0xFFFF, u >> 16], axis=1).astype(jnp.uint16)
    sc = jax.lax.bitcast_convert_type(halves, jnp.bfloat16).astype(jnp.float32)
    sc = sc.reshape(-1, packed.shape[1])  # [K / g, N]
    return v * jnp.repeat(sc, codes.shape[0] // sc.shape[0], axis=0)


def _untile(t):
    """[N / tn, K, tn] fp8 -> [K, N] f32."""
    nt, k, tn = t.shape
    return t.astype(jnp.float32).transpose(1, 0, 2).reshape(k, nt * tn)


def _bf(x):
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _rms(x, eps):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def _heads_rms(x, nh, eps):
    return _rms(x.reshape(x.shape[0], nh, -1), eps).reshape(x.shape)


def _rope(x, nh, tab):
    xs = x.reshape(x.shape[0], nh, 64)
    cos, sin = tab[:, None, :64], tab[:, None, 64:]
    partner = jnp.stack([-xs[..., 1::2], xs[..., 0::2]], axis=-1).reshape(xs.shape)
    return (xs * cos + partner * sin).reshape(x.shape)


def _dot(a, b):
    return jnp.dot(a, b, precision=HIGH, preferred_element_type=jnp.float32)


def pool_tokens(pool, table, n):
    """Pool pages -> K, V [n, kv_heads, hd] f32 of positions [0, n) through the page table."""
    pos = jnp.arange(n)
    blk = pool[table[pos // 256]]  # [n, 2 kvh, hd / 4, 4, page]
    tok = jnp.take_along_axis(blk, (pos % 256)[:, None, None, None, None], axis=4)[
        ..., 0
    ]
    tok = tok.astype(jnp.float32).reshape(n, -1, 64)
    return tok[:, 0::2], tok[:, 1::2]


def ref_attn(c, w, l, x, pool, table, rope, ids_in=None, qkv_in=None, o_in=None):
    """Attention and routing of layer l from stream x [T, H] f32: (stream, expert input, ids,
    routing weights, chunk K, chunk V as fp8 values, biased scores, (q, k, v, attention)).
    Teacher forcing: qkv_in, the kernel's (q, k, v) into attention; o_in, its attention output
    into the gate and o projection; ids_in, its expert picks."""
    H, He, E, K = c.hidden, c.expert_hidden, c.n_experts, c.top_k
    qd, kvd, nqh, nkv = c.q_dim, c.kv_dim, c.q_heads, c.kv_heads
    vec = w["vecs"][l]
    row = lambda name, n: vec[PP.V[name], :n]  # noqa: E731
    is_g = GLOBAL[l]
    a_in = _bf(_rms(x, c.rms_eps) * row("pre_attn_w", H))
    qkvg = _dot(a_in, _untile(w["wqkvg"][l])) * row("wqkvg_s", c.qkvg_dim)
    q = _heads_rms(qkvg[:, :qd], nqh, c.rms_eps) * c.scale_query_by
    k = _heads_rms(qkvg[:, qd : qd + kvd], nkv, c.rms_eps)
    v = qkvg[:, qd + kvd : qd + 2 * kvd]
    og = qkvg[:, qd + 2 * kvd :]
    tab = rope[CTX : CTX + T]
    if not is_g:
        q, k = _rope(q, nqh, tab), _rope(k, nkv, tab)
    q_out, k_out, v_out = _bf(q), _bf(k), _bf(v)
    if qkv_in is not None:
        q, k, v = (a.astype(jnp.float32) for a in qkv_in)
    # the served transposed RPA tile: fp8 Q^T of the bf16 q times sm_scale * log2(e), base-2
    # logits; K/V reach the pool as fp8 of the bf16 values the stage hands it
    f8max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    q8 = _bf(q) * (c.attn_scale * LOG2E)
    if TRANSPOSED == "1":
        q8 = jnp.clip(q8, -f8max, f8max).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    q8 = q8.reshape(T, nkv, nqh // nkv, 64)
    k8 = _bf(k).astype(jnp.float8_e4m3fn).astype(jnp.float32).reshape(T, nkv, 64)
    v8 = _bf(v).astype(jnp.float8_e4m3fn).astype(jnp.float32).reshape(T, nkv, 64)
    ck, cv = pool_tokens(pool, table, CTX)
    keys, vals = jnp.concatenate([ck, k8]), jnp.concatenate([cv, v8])
    qp = CTX + jnp.arange(T)[:, None]
    kp = jnp.arange(CTX + T)[None, :]
    live = kp <= qp
    if not is_g:
        live &= qp - kp < c.window

    def heads_out(qg):
        sc = jnp.einsum("thgd,nhd->thgn", qg, keys, precision=HIGH)
        sc = jnp.where(live[:, None, None, :], sc, -jnp.inf)
        if TRANSPOSED == "1":
            # the served tile: no running max (NO_MAX), l from the bf16 P (ones rows in V^T)
            pr = _bf(jnp.exp2(sc))
            return jnp.einsum("thgn,nhd->thgd", pr, vals, precision=HIGH) / pr.sum(
                -1, keepdims=True
            )
        pr = jnp.exp2(sc - sc.max(-1, keepdims=True))
        return jnp.einsum("thgn,nhd->thgd", _bf(pr), vals, precision=HIGH) / pr.sum(
            -1, keepdims=True
        )

    out = heads_out(q8)
    attn = _bf(_heads_rms(out.reshape(T, qd), nqh, c.rms_eps))
    if o_in is not None:
        attn = o_in.astype(jnp.float32)
    gated = _bf(jax.nn.sigmoid(_bf(og)) * attn)
    a = _bf(_dot(gated, _untile(w["wo"][l])) * row("wo_s", H))
    x = row("attn_alpha", H) * x + row("attn_beta", H) * _bf(_rms(a, c.post_eps))
    ffn = _bf(_rms(x, c.rms_eps) * row("pre_ffn_w", H))
    sc = jax.nn.sigmoid(_dot(ffn, w["wrouter"][l].astype(jnp.float32)) * c.router_mult)
    biased = sc + row("expert_bias", E)
    _, ids = jax.lax.top_k(biased, K)  # ties to the lower id
    if ids_in is not None:  # the kernel's picks: rows that flipped at a near tie
        ids = ids_in
    tw = jnp.take_along_axis(sc, ids, 1)
    tw = tw / (tw.sum(1, keepdims=True) + c.router_eps) * c.route_scale
    pooled = _bf(_dot(ffn, _untile(w["wpool"][l])) * row("wpool_s", He))
    xe = _bf(_rms(pooled, c.rms_eps) * row("pre_expert_w", He))
    return x, xe, ids, tw, k8, v8, biased, (q_out, k_out, v_out, attn)


def served_attention(c, is_global, q, k, v, pool, table):
    """The native layer's RPA call (muse_spark_prefill_native_pp.py) on these q, k, v."""
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import ragged_paged_attention

    o, _ = ragged_paged_attention(
        q.reshape(T, c.q_heads, c.head_dim),
        k.astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, c.head_dim),
        v.astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, c.head_dim),
        pool, jnp.array([CTX + T], jnp.int32), table, jnp.array([0, T], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32), sm_scale=c.head_dim**-0.5,
        sliding_window=None if is_global else c.window, k_scale=1.0, v_scale=1.0,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE, out_head_norm_eps=-1.0,
    )  # fmt: skip
    return o.reshape(T, c.q_dim)


def ref_post(c, w, l, x, yn, tw):
    """The weighted slot sum, unpooler and residual of layer l; yn [T, K, He]."""
    H = c.hidden
    vec = w["vecs"][l]
    row = lambda name, n: vec[PP.V[name], :n]  # noqa: E731
    u = _bf(jnp.sum(yn * tw[:, :, None], axis=1))
    f = _bf(_dot(u, _untile(w["wunpool"][l])) * row("wunpool_s", H))
    return row("ffn_alpha", H) * x + row("ffn_beta", H) * _bf(
        _rms(f * row("post_ffn_w", H), c.post_eps)
    )


def ref_layer(c, w, l, x, pool, table, rope, slot, kernel):
    """One layer, each piece fed the kernel's input to it. kernel: its (ids, routing weights,
    q, k, v, attention output, expert input, slot outputs). Returns the pieces' references:
    (q, k, v) from x; attention from the kernel's q, k, v; chunk K, V as fp8 of its k, v; the
    expert input from its attention output; normalized slot outputs from its expert input
    and picks; the stream out from its attention output and slot outputs; rows whose top-k
    flipped and the worst flipped row's biased-score gap."""
    ids_k, tw_k, q_k, k_k, v_k, o_k, xe_k, y_k = kernel
    ids_k = ids_k[:, : c.top_k]
    attn = jax.jit(ref_attn, static_argnums=(0, 2))
    _, _, _, _, _, _, _, (q, k, v, _) = attn(c, w, l, x, pool, table, rope)
    _, _, _, _, k8, v8, _, (_, _, _, o) = attn(
        c, w, l, x, pool, table, rope, None, (q_k, k_k, v_k)
    )
    _, _, ids, _, _, _, biased, _ = attn(c, w, l, x, pool, table, rope, None, None, o_k)
    same = jnp.all(jnp.sort(ids, 1) == jnp.sort(ids_k, 1), axis=1)
    picked = jnp.take_along_axis(biased, ids_k, 1).min(1)
    unpicked = jnp.where(
        (jnp.arange(c.n_experts)[None, None, :] == ids_k[:, :, None]).any(1), -jnp.inf,
        biased).max(1)  # fmt: skip
    gap = float(jnp.where(same, -jnp.inf, unpicked - picked).max())
    x, xe, _, tw, _, _, _, _ = attn(c, w, l, x, pool, table, rope, ids_k, None, o_k)
    yn = expert_outputs(
        c, w, l, xe_k.astype(jnp.float32), ids_k, slot, a8=fp4_qblock_a8()
    )
    pew = w["vecs"][l][PP.V["post_expert_w"], : c.expert_hidden]
    yn_k = _bf(_rms(y_k.astype(jnp.float32) * pew, c.post_eps))
    x = jax.jit(ref_post, static_argnums=(0, 2))(c, w, l, x, yn_k, tw)
    return (q, k, v), o, k8, v8, xe, (yn, yn_k), x, ~same, gap


def expert_outputs(c, w, l, xe, ids, slot, cap=96, a8=False):
    """Normalized outputs [T, K, He] of each (token, slot)'s expert, at most cap rows each;
    a8: NVFP4 experts through the block-scaled W4A8 product."""
    He, E = c.expert_hidden, c.n_experts
    kind = KINDS[l]
    pew = w["vecs"][l][PP.V["post_expert_w"], :He]
    hit = ids[:, :, None] == jnp.arange(E)[None, None, :]  # [T, K, E]
    rows_hit = hit.any(1)  # [T, E]
    assert int(rows_hit.sum(0).max()) <= cap, int(rows_hit.sum(0).max())
    ex = w["fp8"] if kind == "fp8" else w["fp4"][slot]

    def w8a8(x, w, ws):
        """gmm_v2's fp8 path: bf16 rows quantized to fp8 per 512-wide K block (scale amax / 448,
        its inverse and the scaled rows in bf16), each block's product rounded to its bf16
        accumulator, scaled and summed in bf16."""
        f8max = jnp.bfloat16(float(jnp.finfo(jnp.float8_e4m3fn).max))
        acc = jnp.zeros((x.shape[0], w.shape[1]), jnp.bfloat16)
        for b0 in range(0, x.shape[1], 512):
            xb = x[:, b0 : b0 + 512].astype(jnp.bfloat16)
            sc = jnp.max(jnp.abs(xb), axis=1, keepdims=True) / f8max
            inv = jnp.where(sc == 0, jnp.bfloat16(0), jnp.bfloat16(1) / sc)
            q = (xb * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32)
            part = _dot(q, w[b0 : b0 + 512]).astype(jnp.bfloat16)
            part = part * sc * ws.astype(jnp.bfloat16)[None, :]
            acc = acc + part
        return acc.astype(jnp.float32)

    def w4a8(x, packed, scales):
        """gmm_v2's block-scaled W4A8 (MUSE_PP_QBLOCK_A8): bf16 rows quantized to fp8 per
        weight block (amax / 448), each block's codes product rounded to bf16, then scaled by
        the row and block scales in bf16 and summed in bf16."""
        v, sb = _codes_scales(packed, scales)
        g = v.shape[0] // sb.shape[0]
        f8max = jnp.bfloat16(float(jnp.finfo(jnp.float8_e4m3fn).max))
        acc = jnp.zeros((x.shape[0], v.shape[1]), jnp.bfloat16)
        for b in range(sb.shape[0]):
            xb = x[:, b * g : (b + 1) * g].astype(jnp.bfloat16)
            sc = jnp.max(jnp.abs(xb), axis=1, keepdims=True) / f8max
            inv = jnp.where(sc == 0, jnp.bfloat16(0), jnp.bfloat16(1) / sc)
            q = (xb * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32)
            part = _dot(q, v[b * g : (b + 1) * g]).astype(jnp.bfloat16)
            acc = acc + part * sc * sb[b].astype(jnp.bfloat16)[None, :]
        return acc.astype(jnp.float32)

    def one(e, w13, s13, w2, s2, xe, hit_e):
        idx = jnp.nonzero(hit_e, size=cap, fill_value=T)[0]
        xr = jnp.concatenate([xe, jnp.zeros((1, He))])[idx]
        if kind == "fp8":
            gu = w8a8(xr, w13[e].astype(jnp.float32), s13[e, 0, 0])
            act = _bf(
                gu[:, : c.inter] * jax.nn.sigmoid(gu[:, : c.inter]) * gu[:, c.inter :]
            )
            y = _bf(w8a8(act, w2[e].astype(jnp.float32), s2[e, 0, 0]))
        elif a8:
            gu = w4a8(xr, w13[e], s13[e])
            act = _bf(
                gu[:, : c.inter] * jax.nn.sigmoid(gu[:, : c.inter]) * gu[:, c.inter :]
            )
            y = _bf(w4a8(act, w2[e], s2[e]))
        else:
            gu = _dot(xr, _dq4(w13[e], s13[e]))
            act = _bf(
                gu[:, : c.inter] * jax.nn.sigmoid(gu[:, : c.inter]) * gu[:, c.inter :]
            )
            y = _bf(_dot(act, _dq4(w2[e], s2[e])))
        return idx, _bf(_rms(y * pew, c.post_eps))

    out = jnp.zeros((T + 1, E, He), jnp.bfloat16)  # bf16 values: exact
    run = jax.jit(one)
    for e in range(E):
        idx, ye = run(jnp.int32(e), *ex, xe, rows_hit[:, e])
        out = out.at[idx, e].set(ye.astype(jnp.bfloat16))
    return jnp.take_along_axis(out[:T], ids[:, :, None], axis=1).astype(jnp.float32)


def main() -> int:
    c = config()
    dev = jax.devices()[0]
    qblock = fp4_qblock()
    with jax.default_device(dev):
        w = build_weights(c, seed=11)
        if qblock:
            requant = jax.jit(lambda a, b: PP_MOE.requant_fp4_blocks(a, b, qblock))
            done = {}
            for ex in w["fp4"]:
                if id(ex) not in done:
                    w13, s13 = requant(ex[0], ex[1])
                    w2, s2 = requant(ex[2], ex[3])
                    done[id(ex)] = (w13, s13, w2, s2)
            w["fp4_exact"] = w["fp4"]
            w["fp4"] = [done[id(ex)] for ex in w["fp4"]]
        rope = rope_table(c.max_position)
        tables = {}
        perm = (
            np.random.default_rng(0).permutation(c.global_pages - 1)[: c.pages_per_seq]
            + 1
        )
        tables[True] = jnp.asarray(perm, jnp.int32)
        tables[False] = jnp.asarray(
            1 + np.arange(c.pages_per_seq) % (c.swa_pages - 1), jnp.int32
        )
        pools = [
            (
                _unit(
                    PP.pool_shape(
                        c, c.global_pages if s in c.global_slots else c.swa_pages
                    ),
                    5 + s,
                )
                * 0.5
            ).astype(PP.FP8)
            for s in range(c.slots)
        ]
        x0 = _bf(_unit((T, c.hidden), 3))
        layers = tuple(range(len(KINDS)))
        full = PP.slot_table(c, 0, layers, KINDS, tuple(l for l in layers if GLOBAL[l]))
        first = c.global_slots[0] - GLOBAL.index(True)  # the slot of layer 0
        w_ref = {k: (v[first:] if k in PP.WEIGHT_ORDER else v) for k, v in w.items()}

        def run(tab, x0, rope, weights, ex4, ex8, pools, tables):
            return PP.stage_forward(
                c,
                jnp.array([CTX, -1], jnp.int32),
                tab,
                x0,
                rope,
                weights,
                ex4,
                ex8,
                pools,
                jnp.array([CTX + T], jnp.int32),
                tables,
                jnp.array([0, T], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32),
                scratch=True,
            )

        jitted = jax.jit(
            run
        )  # weights as arguments: closed over, they would be constants
        weights = [w[k] for k in PP.WEIGHT_ORDER]

        def call(tab):
            return jitted(
                tab,
                x0,
                rope,
                weights,
                w["fp4"],
                w["fp8"],
                pools,
                (tables[True], tables[False]),
            )

        ok = True
        x_prev = x0
        for n in range(1, len(KINDS) + 1):
            tab = full.copy()
            for s in range(first + n, c.slots):
                tab[PP.HDR + s * PP.NF + PP.F_VALID] = 0
            x_k, pools_k, routes = call(jnp.asarray(tab))
            l = n - 1
            pool, kpool = pools[first + l], pools_k[first + l]
            table = tables[GLOBAL[l]]
            kernel = routes[first + l]
            qkv_r, o_r, k8, v8, xe_r, (yn_r, yn_k), x_r, flipped, gap = ref_layer(
                c, w_ref, l, x_prev, pool, table, rope, first + l, kernel
            )
            _, _, q_k, k_k, v_k, o_k, xe_k, _ = kernel
            rel = jnp.linalg.norm(x_k - x_r, axis=1) / jnp.linalg.norm(x_r, axis=1)

            def rel_of(a, b):
                a, b = a.astype(jnp.float32), b.astype(jnp.float32)
                return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

            qkv_rel = max(rel_of(a, b) for a, b in zip((q_k, k_k, v_k), qkv_r))
            o_rel = rel_of(o_k, o_r)
            xe_rel = rel_of(xe_k, xe_r)
            live = (
                jnp.abs(yn_r).sum(-1, keepdims=True) > 0
            )  # the picked (token, slot) rows
            y_rel = rel_of(jnp.where(live, yn_k, 0), yn_r)
            y_exact = ""
            if qblock and KINDS[l] == "fp4":
                w_exact = {**w_ref, "fp4": w["fp4_exact"]}
                yn_x = expert_outputs(
                    c,
                    w_exact,
                    l,
                    xe_k.astype(jnp.float32),
                    kernel[0][:, : c.top_k],
                    first + l,
                )
                y_exact = f" (vs exact block-16 ref {rel_of(jnp.where(live, yn_k, 0), yn_x):.2e})"
            ids_k, tw_k = kernel[0], kernel[1]
            ex = w["fp8"] if KINDS[l] == "fp8" else w["fp4"][first + l]
            y_served = jax.jit(lambda *a: PP_MOE.experts(c, *a))(xe_k, ids_k, tw_k, ex)
            y_port = rel_of(kernel[7], y_served)
            o_served = jax.jit(served_attention, static_argnums=(0, 1))(
                c, GLOBAL[l], q_k, k_k, v_k, pool, table
            )
            o_port = rel_of(o_k, o_served)
            kk, vv = pool_tokens(kpool, table, CTX + T)
            kv_rel = float(
                jnp.linalg.norm(jnp.concatenate([kk[CTX:] - k8, vv[CTX:] - v8]))
                / jnp.linalg.norm(jnp.concatenate([k8, v8]))
            )
            cached_same = bool(jnp.all(kk[:CTX] == pool_tokens(pool, table, CTX)[0]))
            worst, mean = float(rel.max()), float(rel.mean())
            n_bad = int((rel > 0.1).sum())
            n_flip = int(flipped.sum())
            good = (
                worst < TOL_ROW
                and mean < TOL_MEAN
                and qkv_rel < TOL_QKV
                and o_rel < TOL_ATTN
                and xe_rel < TOL_QKV
                and y_rel < TOL_EXPERTS[KINDS[l]]
                and o_port == 0.0
                and y_port == 0.0
                and kv_rel < TOL_KV
                and cached_same
                and gap < NEAR_TIE
            )
            ok &= good
            print(
                f"layer {l} ({KINDS[l]}, {'global' if GLOBAL[l] else 'window'}): q/k/v rel "
                f"{qkv_rel:.2e} | attention rel {o_rel:.2e} | expert input rel {xe_rel:.2e} | "
                f"expert out rel {y_rel:.2e}{y_exact} | vs served calls: attention {o_port:.1e} experts "
                f"{y_port:.1e} | chunk kv rel {kv_rel:.2e} cached "
                f"kept {cached_same} | row rel max {worst:.2e} mean {mean:.2e} rows>10% {n_bad} "
                f"| top-k flips {n_flip} (worst gap {gap:.1e}) | {'OK' if good else 'FAIL'}"
            )
            if not math.isfinite(worst):
                print("  non-finite kernel output")
            x_prev = x_k
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
