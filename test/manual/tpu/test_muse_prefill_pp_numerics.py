"""Numerics of the PP prefill stage megakernel (sglang_tpu.muse_megakernel.prefill_pp) on one
TPU device, against a plain-jnp reference of each layer (f32, HIGHEST precision dots, the
kernel's bf16 / fp8 rounding points).

One stage of three layers at the real per-layer geometry (all 128 q / 16 kv heads, 256
experts): an fp8-expert sliding-window layer, an fp4 global layer and an fp4 sliding-window
layer, a 512-token chunk after 2560 cached tokens (the window cuts the cached keys, the global
layer reads them all). The kernel runs with 1, 2 and 3 valid slots; layer l's reference is
fed the kernel's stream after l - 1 layers (teacher forcing), and its chunk K/V is compared
with what the kernel wrote into the pool.

    python3 test/manual/tpu/test_muse_prefill_pp_numerics.py
"""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.muse_megakernel import prefill_pp as PP

HIGH = jax.lax.Precision.HIGHEST
T, CTX = 512, 2560
KINDS = ("fp8", "fp4", "fp4")
GLOBAL = (False, True, False)
ROPE_BASE = 10000.0
# per-row relative error of a layer's output stream: ~12 bf16 roundings per layer
TOL_ROW, TOL_MEAN = 3e-2, 1e-2
TOL_KV = 6e-2  # fp8 K/V of the chunk: a moved input flips a rounding
# a token whose top-k set differs from the reference's must be a near tie of biased scores
NEAR_TIE = 2e-3


def config() -> PP.PPConfig:
    return PP.PPConfig(
        hidden=8192, head_dim=64, q_heads=128, kv_heads=16, n_experts=256, top_k=8,
        expert_hidden=4096, inter=4096, rms_eps=1e-5, post_eps=1e-8, scale_query_by=5.473,
        attn_scale=0.125, router_mult=0.1768, route_scale=1.0, router_eps=1e-15, window=2048,
        rows=T, slots=len(KINDS), page=256, pages_per_seq=64, global_pools=1, swa_pools=2,
        global_pages=65, swa_pages=27, slab_rows=2048, max_position=16384,
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
    S, H, He, E, ti = c.slots, c.hidden, c.expert_hidden, c.n_experts, c.ti
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
    shapes = PP.weight_shapes(c)
    w["w13"] = jax.lax.bitcast_convert_type(
        _hash(shapes["w13"][0], seed + 1), jnp.int32
    )
    w["w2"] = jax.lax.bitcast_convert_type(_hash(shapes["w2"][0], seed + 2), jnp.int32)
    # fp8 slots hold e4m3 bytes with the top exponent bit clear
    s13 = jnp.zeros(shapes["s13"][0], jnp.bfloat16)
    s2 = jnp.zeros(shapes["s2"][0], jnp.bfloat16)
    woff = soff = 0
    for l, kind in enumerate(KINDS):
        wr, sr = PP.expert_rows(kind, He)
        if kind == "fp8":
            for name in ("w13", "w2"):
                blk = w[name][:, :, woff : woff + wr]
                w[name] = (
                    w[name].at[:, :, woff : woff + wr].set(blk & jnp.int32(-1077952577))
                )  # 0xBFBFBFBF
            b13, b2 = 1.0 / (_FP8_RMS * He**0.5), 1.0 / (_FP8_RMS * c.inter**0.5)
        else:
            b13, b2 = 1.0 / (_E2M1_RMS * He**0.5), 1.0 / (_E2M1_RMS * c.inter**0.5)
        r13 = (1.0 + 0.5 * _unit((E, s13.shape[1], sr, ti), seed + 10 + l)) * b13
        r2 = (1.0 + 0.5 * _unit((E, s2.shape[1], sr, ti), seed + 20 + l)) * b2
        if kind == "fp8":  # one channelwise row, repeated over the tile
            r13 = jnp.broadcast_to(r13[:, :, :1], r13.shape)
            r2 = jnp.broadcast_to(r2[:, :, :1], r2.shape)
        s13 = s13.at[:, :, soff : soff + sr].set(r13.astype(jnp.bfloat16))
        s2 = s2.at[:, :, soff : soff + sr].set(r2.astype(jnp.bfloat16))
        woff, soff = woff + wr, soff + sr
    w["s13"], w["s2"] = s13, s2
    return w


def rope_table(n):
    inv = 1.0 / (ROPE_BASE ** (np.arange(0, 64, 2, dtype=np.float64) / 64))
    ang = np.arange(n, dtype=np.float64)[:, None] * inv
    return jnp.asarray(np.concatenate(
        [np.repeat(np.cos(ang), 2, 1), np.repeat(np.sin(ang), 2, 1)], 1), jnp.float32)  # fmt: skip


def _dq4(words, scales):
    """[r, n] i32 words + [r / 2, n] bf16 -> [8 r, n] f32: nibble j of word w is row 8 w + j."""
    sh = jnp.arange(0, 32, 4, dtype=jnp.uint32)[None, :, None]
    codes = (jax.lax.bitcast_convert_type(words, jnp.uint32)[:, None, :] >> sh) & 0xF
    v = _E2M1[codes.astype(jnp.int32)].reshape(words.shape[0] * 8, words.shape[1])
    return v * jnp.repeat(scales.astype(jnp.float32), 16, axis=0)


def _dq8(words):
    """[r, n] i32 -> [4 r, n] f32: byte i of word w is row 4 w + i."""
    sh = jnp.arange(0, 32, 8, dtype=jnp.uint32)[None, :, None]
    b = (
        (jax.lax.bitcast_convert_type(words, jnp.uint32)[:, None, :] >> sh) & 0xFF
    ).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(b, jnp.float8_e4m3fn).astype(jnp.float32).reshape(
        words.shape[0] * 4, words.shape[1])  # fmt: skip


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


def ref_attn(c, w, l, x, pool, table, rope, ids_in=None):
    """Attention and routing of layer l from stream x [T, H] f32: (stream, expert input,
    ids, routing weights, chunk K, chunk V as fp8 values)."""
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
    qb = _bf(q).reshape(T, nkv, nqh // nkv, 64)
    k8 = k.astype(jnp.float8_e4m3fn).astype(jnp.float32).reshape(T, nkv, 64)
    v8 = v.astype(jnp.float8_e4m3fn).astype(jnp.float32).reshape(T, nkv, 64)
    ck, cv = pool_tokens(pool, table, CTX)
    keys, vals = jnp.concatenate([ck, k8]), jnp.concatenate([cv, v8])
    qp = CTX + jnp.arange(T)[:, None]
    kp = jnp.arange(CTX + T)[None, :]
    live = kp <= qp
    if not is_g:
        live &= qp - kp < c.window
    s = jnp.einsum("thgd,nhd->thgn", qb, keys, precision=HIGH) * c.attn_scale
    s = jnp.where(live[:, None, None, :], s, -jnp.inf)
    p = jnp.exp(s - s.max(-1, keepdims=True))
    out = jnp.einsum("thgn,nhd->thgd", _bf(p), vals, precision=HIGH) / p.sum(
        -1, keepdims=True
    )
    attn = _bf(_heads_rms(out.reshape(T, qd), nqh, c.rms_eps))
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
    return x, xe, ids, tw, k8, v8, biased


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


def ref_layer(c, w, l, x, pool, table, rope, kernel_ids):
    """One layer: (stream out, chunk K, chunk V, rows whose top-k flipped, the worst flipped
    row's biased-score gap between its picks' minimum and the best unpicked score)."""
    attn = jax.jit(ref_attn, static_argnums=(0, 2))
    _, _, ids, _, _, _, biased = attn(c, w, l, x, pool, table, rope)
    same = jnp.all(jnp.sort(ids, 1) == jnp.sort(kernel_ids, 1), axis=1)
    picked = jnp.take_along_axis(biased, kernel_ids, 1).min(1)
    unpicked = jnp.where(
        (jnp.arange(c.n_experts)[None, None, :] == kernel_ids[:, :, None]).any(1), -jnp.inf,
        biased).max(1)  # fmt: skip
    gap = float(jnp.where(same, -jnp.inf, unpicked - picked).max())
    x, xe, ids, tw, k8, v8, _ = attn(c, w, l, x, pool, table, rope, kernel_ids)
    yn = expert_outputs(c, w, l, xe, ids)
    return (
        jax.jit(ref_post, static_argnums=(0, 2))(c, w, l, x, yn, tw),
        k8,
        v8,
        ~same,
        gap,
    )


def expert_outputs(c, w, l, xe, ids, cap=96):
    """Normalized outputs [T, K, He] of each (token, slot)'s expert, at most cap rows each."""
    He, E, ti = c.expert_hidden, c.n_experts, c.ti
    nti = c.inter // ti
    kind = KINDS[l]
    woff = sum(PP.expert_rows(k, He)[0] for k in KINDS[:l])
    soff = sum(PP.expert_rows(k, He)[1] for k in KINDS[:l])
    wr, sr = PP.expert_rows(kind, He)
    pew = w["vecs"][l][PP.V["post_expert_w"], :He]
    hit = ids[:, :, None] == jnp.arange(E)[None, None, :]  # [T, K, E]
    rows_hit = hit.any(1)  # [T, E]
    assert int(rows_hit.sum(0).max()) <= cap, int(rows_hit.sum(0).max())

    def one(e, w13, s13, w2, s2, xe, hit_e):
        ww = {"w13": w13, "s13": s13, "w2": w2, "s2": s2}

        def mat(name_w, name_s, tiles):
            ws = ww[name_w][e][tiles, woff : woff + wr]
            ss = ww[name_s][e][tiles, soff : soff + sr]
            if kind == "fp8":
                m = jax.vmap(_dq8)(ws) * ss[:, :1].astype(jnp.float32)
            else:
                m = jax.vmap(_dq4)(ws, ss)
            return m  # [n_tiles, K, ti]

        g = jnp.concatenate(list(mat("w13", "s13", jnp.arange(nti))), axis=1)
        up = jnp.concatenate(list(mat("w13", "s13", nti + jnp.arange(nti))), axis=1)
        dn = jnp.concatenate(list(mat("w2", "s2", jnp.arange(He // ti))), axis=1)
        idx = jnp.nonzero(hit_e, size=cap, fill_value=T)[0]
        xr = jnp.concatenate([xe, jnp.zeros((1, He))])[idx]
        gu, uu = _dot(xr, g), _dot(xr, up)
        h = _bf(gu * jax.nn.sigmoid(gu) * uu)
        y = _bf(_dot(h, dn))
        return idx, _bf(_rms(y * pew, c.post_eps))

    out = jnp.zeros((T + 1, E, He), jnp.bfloat16)  # bf16 values: exact
    run = jax.jit(one)
    for e in range(E):
        idx, ye = run(
            jnp.int32(e), w["w13"], w["s13"], w["w2"], w["s2"], xe, rows_hit[:, e]
        )
        out = out.at[idx, e].set(ye.astype(jnp.bfloat16))
    return jnp.take_along_axis(out[:T], ids[:, :, None], axis=1).astype(jnp.float32)


def main() -> int:
    c = config()
    dev = jax.devices()[0]
    with jax.default_device(dev):
        w = build_weights(c, seed=11)
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
        kvg = (
            _unit((c.global_pools, *PP.pool_shape(c, c.global_pages)), 5) * 0.5
        ).astype(PP.FP8)
        kvs = (_unit((c.swa_pools, *PP.pool_shape(c, c.swa_pages)), 6) * 0.5).astype(
            PP.FP8
        )
        x0 = _bf(_unit((T, c.hidden), 3))
        call = jax.jit(PP.stage_call(c, debug=True))
        layers = tuple(range(len(KINDS)))
        full = PP.slot_table(c, 0, layers, KINDS, tuple(l for l in layers if GLOBAL[l]))
        ok = True
        x_prev = x0
        for n in range(1, len(KINDS) + 1):
            tab = full.copy()
            for s in range(n, c.slots):
                tab[PP.HDR + s * PP.NF + PP.F_VALID] = 0
            ctl = jnp.array([CTX, -1], jnp.int32)
            outs = call(
                ctl,
                jnp.asarray(tab),
                tables[True],
                tables[False],
                x0,
                rope,
                *[w[k] for k in PP.WEIGHT_ORDER],
                kvg,
                kvs,
            )
            x_k, kvg_k, kvs_k = outs[:3]
            kernel_ids = outs[8].reshape(T, c.top_k)
            l = n - 1
            pidx = int(full[PP.HDR + l * PP.NF + PP.F_POOL])
            pool = (kvg if GLOBAL[l] else kvs)[pidx]
            table = tables[GLOBAL[l]]
            x_r, k8, v8, flipped, gap = ref_layer(
                c, w, l, x_prev, pool, table, rope, kernel_ids
            )
            rel = jnp.linalg.norm(x_k - x_r, axis=1) / jnp.linalg.norm(x_r, axis=1)
            kpool = (kvg_k if GLOBAL[l] else kvs_k)[pidx]
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
                and kv_rel < TOL_KV
                and cached_same
                and gap < NEAR_TIE
            )
            ok &= good
            print(
                f"layer {l} ({KINDS[l]}, {'global' if GLOBAL[l] else 'window'}): row rel max "
                f"{worst:.2e} mean {mean:.2e} rows>10% {n_bad} | top-k flips {n_flip} (worst "
                f"gap {gap:.1e}) | chunk kv rel {kv_rel:.2e} "
                f"cached kept {cached_same} | {'OK' if good else 'FAIL'}"
            )
            if not math.isfinite(worst):
                print("  non-finite kernel output")
            x_prev = x_k
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
