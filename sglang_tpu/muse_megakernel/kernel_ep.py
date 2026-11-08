"""Muse Spark decode step in the EP + sequence-parallel layout (XLA prototype).

kernel.py keeps every token on every rank and slices every expert by its intermediate dim
(expert-TP), so each layer reduce-scatters [R, K, 4096] expert partials by token. Here:

- Sequence parallel: rank r owns rows [r * R/tp, (r + 1) * R/tp): residual, norms, pooler,
  router, combine and unpooler run on those rows only, with the pooler and unpooler
  replicated in full on every rank.
- Attention stays head-parallel (the KV pools are unchanged): all-gather the normed rows
  before it, reduce-scatter the o_proj partials by row after it.
- Expert parallel: rank r owns experts [r * E/tp, (r + 1) * E/tp) whole. Each owned row's
  top-k slots travel to the expert owners in one ragged all-to-all and come back in another.

Math and bf16 rounding points follow test/manual/tpu/test_muse_megakernel_numerics.py's
reference. An expert is stored as its tp intermediate slices, summed slice by slice as the
reference does. This is the layout's correctness and communication prototype: attention
gathers whole pages and experts dequantize their weights to f32 each step, so only its
collective times are representative.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

AXIS = "model"
FP8 = jnp.float8_e4m3fn
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
REPLICATED_ROWS = (
    "pre_attn_w",
    "attn_alpha",
    "attn_beta",
    "pre_ffn_w",
    "post_ffn_w",
    "ffn_alpha",
    "ffn_beta",
)


@dataclasses.dataclass(frozen=True)
class EPConfig:
    """One compiled step's static shape and the model constants it reads."""

    tp: int
    n_layers: int
    global_layers: tuple[int, ...]
    kinds: tuple[str, ...]  # per layer: "fp4" or "fp8" experts
    q: int  # verify rows per request
    n_req: int
    rows: int  # padded rows, a multiple of tp
    page_size: int
    pages_per_seq: int  # page-table entries per request
    hidden: int = 8192
    head_dim: int = 64
    q_heads: int = 16  # per rank
    kv_heads: int = 2  # per rank
    n_experts: int = 256
    top_k: int = 8
    expert_hidden: int = 4096
    inter_local: int = 512  # one intermediate slice of an expert
    window: int = 2048
    rms_eps: float = 1e-5
    post_eps: float = 1e-8
    scale_query_by: float = 5.473
    attn_scale: float = 0.125
    router_mult: float = 0.1768
    route_scale: float = 1.0
    router_eps: float = 1e-15

    @property
    def rows_per_rank(self) -> int:
        return self.rows // self.tp

    @property
    def experts_per_rank(self) -> int:
        return self.n_experts // self.tp

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def validate(self) -> None:
        assert self.rows % self.tp == 0 and self.n_req * self.q <= self.rows
        assert self.n_experts % self.tp == 0 and len(self.kinds) == self.n_layers
        assert self.window % self.page_size == 0


def _rms(x: jax.Array, eps: float, w: jax.Array | None = None) -> jax.Array:
    y = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def _bf(x: jax.Array) -> jax.Array:
    """f32 rounded to bf16 precision; reduce_precision, as XLA may drop a cast round trip
    (xla_allow_excess_precision)."""
    return jax.lax.reduce_precision(
        x.astype(jnp.float32), exponent_bits=8, mantissa_bits=7
    )


def _heads_rms(x: jax.Array, n_heads: int, eps: float) -> jax.Array:
    return _rms(x.reshape(x.shape[0], n_heads, -1), eps).reshape(x.shape)


def _rope(x: jax.Array, n_heads: int, table: jax.Array) -> jax.Array:
    """Interleaved pairs; table rows [cos x2 | sin x2] per row's position."""
    hd = table.shape[1] // 2
    xs = x.reshape(x.shape[0], n_heads, hd)
    c, s = table[:, None, :hd], table[:, None, hd:]
    partner = jnp.stack([-xs[..., 1::2], xs[..., 0::2]], axis=-1).reshape(xs.shape)
    return (xs * c + partner * s).reshape(x.shape)


def _mv(x: jax.Array, w: jax.Array, s: jax.Array) -> jax.Array:
    """bf16 rows times an fp8 matrix with per-output-channel scales, in f32."""
    return (
        jnp.dot(
            x.astype(jnp.bfloat16),
            w.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        * s
    )


def dequant(words: jax.Array, scale: jax.Array, kind: str) -> jax.Array:
    """f32 [..., K, N] from u32 words along K: 8 e2m1 codes (16-row block scales) or 4 e4m3
    bytes (per-column scale in row 0). f32, as code x scale needs more bits than bf16 has."""
    *lead, kw, n = words.shape
    if kind == "fp4":
        shifts = jnp.arange(0, 32, 4, dtype=jnp.int32)[:, None]
        codes = (
            words[..., :, None, :] >> shifts
        ) & 0xF  # nibble j of word w is row 8w + j
        v = jnp.asarray(E2M1, jnp.float32)[codes].reshape(*lead, kw * 8, n)
        return v * jnp.repeat(scale.astype(jnp.float32), 16, axis=-2)
    shifts = jnp.arange(0, 32, 8, dtype=jnp.int32)[:, None]
    b = ((words[..., :, None, :] >> shifts) & 0xFF).astype(jnp.uint8)
    v = (
        jax.lax.bitcast_convert_type(b, FP8)
        .astype(jnp.float32)
        .reshape(*lead, kw * 4, n)
    )
    return v * scale[..., :1, :].astype(jnp.float32)


def _excl_cumsum(x: jax.Array) -> jax.Array:
    return jnp.cumsum(x) - x


def _attention(
    c: EPConfig,
    l: int,
    qn: jax.Array,
    kb: jax.Array,
    vb: jax.Array,
    pool: jax.Array,
    pos: jax.Array,
    table: jax.Array,
) -> jax.Array:
    """This rank's heads for every request: the request's cached pages plus its own rows.

    Args:
      qn: [R, q_dim] normed (and roped) queries; kb, vb: [R, kv_dim] this step's K, V.
      pool: [pages, 2 * kv_heads, head_dim // 4, 4, page] fp8 (K_h0, V_h0, K_h1, ...).
    """
    ps, hd, kvh = c.page_size, c.head_dim, c.kv_heads
    g = c.q_heads // kvh
    is_global = l in c.global_layers
    n_pages = c.pages_per_seq if is_global else c.window // ps + 1
    outs = []
    for j in range(c.n_req):
        rows = slice(j * c.q, (j + 1) * c.q)
        p = pos[rows]
        p_lo = p[0]
        first_page = 0 if is_global else jnp.maximum(p_lo - c.window + 1, 0) // ps
        # Padded so the slice never clamps (a clamped start would shift onto other pages).
        padded = jnp.pad(table, (0, n_pages))
        ent = jax.lax.dynamic_slice(
            padded, (j * c.pages_per_seq + first_page,), (n_pages,)
        )
        blk = pool[ent]  # [n_pages, 2 kvh, hd/4, 4, ps]
        blk = blk.reshape(n_pages, kvh, 2, hd, ps).astype(jnp.float32)
        kv = jnp.moveaxis(blk, -1, 1).reshape(n_pages * ps, kvh, 2, hd)
        kpos = first_page * ps + jnp.arange(n_pages * ps)
        cached_ok = kpos < p_lo
        if not is_global:
            cached_ok &= kpos >= p_lo - c.window + 1
        keys = jnp.concatenate([kv[:, :, 0], kb[rows].reshape(c.q, kvh, hd)])
        vals = jnp.concatenate([kv[:, :, 1], vb[rows].reshape(c.q, kvh, hd)])
        allpos = jnp.concatenate([kpos, p])
        live = jnp.concatenate(
            [
                jnp.broadcast_to(cached_ok, (c.q, kpos.shape[0])),
                jnp.ones((c.q, c.q), bool),
            ],
            axis=1,
        )
        live &= allpos[None, :] <= p[:, None]
        if not is_global:
            live &= p[:, None] - allpos[None, :] < c.window
        qs = _bf(qn[rows].reshape(c.q, kvh, g, hd))
        s = (
            jnp.einsum("thgd,nhd->thgn", qs, keys, precision=jax.lax.Precision.HIGHEST)
            * c.attn_scale
        )
        s = jnp.where(live[:, None, None, :], s, -jnp.inf)
        pr = jnp.exp(s - jnp.max(s, axis=-1, keepdims=True))
        acc = jnp.einsum(
            "thgn,nhd->thgd", _bf(pr), vals, precision=jax.lax.Precision.HIGHEST
        )
        outs.append((acc / jnp.sum(pr, axis=-1, keepdims=True)).reshape(c.q, -1))
    out = jnp.concatenate(outs)
    return jnp.pad(out, ((0, c.rows - out.shape[0]), (0, 0)))


def _kv_write(
    c: EPConfig,
    l: int,
    pool: jax.Array,
    kb: jax.Array,
    vb: jax.Array,
    pos: jax.Array,
    table: jax.Array,
    n_live: jax.Array,
) -> jax.Array:
    """This step's K, V of the live rows into their pages (dead rows into page 0)."""
    ps, hd, kvh = c.page_size, c.head_dim, c.kv_heads
    r = jnp.arange(c.rows)
    req = jnp.minimum(r // c.q, c.n_req - 1)
    page = jnp.where(r < n_live, table[req * c.pages_per_seq + pos // ps], 0)
    kv = jnp.stack([kb.reshape(-1, kvh, hd), vb.reshape(-1, kvh, hd)], axis=2)
    kv = kv.reshape(c.rows, 2 * kvh, hd // 4, 4).astype(FP8)
    return pool.at[page, :, :, :, pos % ps].set(kv)


def _experts(
    c: EPConfig,
    kind: str,
    x: jax.Array,
    local_e: jax.Array,
    n_recv: jax.Array,
    w: Sequence[jax.Array],
) -> jax.Array:
    """bf16 outputs of the received slots, sorted by local expert; [rows, expert_hidden].

    Args:
      x: [M, expert_hidden] received slots; local_e: [M] their local expert (rows >= n_recv
        are padding).
      w: this rank's experts: w13 [E/tp, tp, K13, 2 Il], scale, w2 [E/tp, tp, K2, He], scale.
    """
    m = x.shape[0]
    key = jnp.where(jnp.arange(m) < n_recv, local_e, c.experts_per_rank)
    order = jnp.argsort(key)
    sizes = jnp.bincount(key, length=c.experts_per_rank + 1)[: c.experts_per_rank]
    xs = x[order].astype(jnp.float32)
    il = c.inter_local
    hi = jax.lax.Precision.HIGHEST
    y = jnp.zeros((m, c.expert_hidden), jnp.float32)
    for d in range(c.tp):  # the reference's per-slice rounding
        w13 = dequant(w[0][:, d], w[1][:, d], kind)
        w2 = dequant(w[2][:, d], w[3][:, d], kind)
        gu = jax.lax.ragged_dot(
            xs, w13, sizes, precision=hi, preferred_element_type=jnp.float32
        )
        gate, up = gu[:, :il], gu[:, il:]
        h = _bf(gate * jax.nn.sigmoid(gate) * up)
        y = y + _bf(
            jax.lax.ragged_dot(
                h, w2, sizes, precision=hi, preferred_element_type=jnp.float32
            )
        )
    return (
        jnp.zeros_like(y).at[order].set(y).astype(jnp.bfloat16)
    )  # back to arrival order


def _moe(
    c: EPConfig,
    l: int,
    xe: jax.Array,
    ids: jax.Array,
    live_rows: jax.Array,
    w: Sequence[jax.Array],
) -> jax.Array:
    """[rows_per_rank, top_k, expert_hidden] bf16 expert outputs of this rank's rows."""
    rc, k, tp = c.rows_per_rank, c.top_k, c.tp
    n = rc * k
    dest = jnp.where(live_rows[:, None], ids // c.experts_per_rank, tp).reshape(n)
    order = jnp.argsort(dest)
    send = jnp.bincount(dest, length=tp + 1)[:tp].astype(jnp.int32)
    counts = jax.lax.all_gather(send, AXIS)  # [src, dst]
    me = jax.lax.axis_index(AXIS)
    in_off = _excl_cumsum(send)
    out_off = jnp.sum(jnp.where(jnp.arange(tp)[:, None] < me, counts, 0), axis=0)
    recv = counts[:, me]
    cap = c.rows * k  # every source's every slot

    with jax.named_scope("ep_dispatch"):
        slot_x = jnp.repeat(xe.astype(jnp.bfloat16), k, axis=0)[order]
        slot_e = (ids % c.experts_per_rank).reshape(n).astype(jnp.int32)[order]
        xr = jax.lax.ragged_all_to_all(
            slot_x,
            jnp.zeros((cap, c.expert_hidden), jnp.bfloat16),
            in_off,
            send,
            out_off,
            recv,
            axis_name=AXIS,
        )
        er = jax.lax.ragged_all_to_all(
            slot_e,
            jnp.zeros((cap,), jnp.int32),
            in_off,
            send,
            out_off,
            recv,
            axis_name=AXIS,
        )
    with jax.named_scope("ep_experts"):
        yr = _experts(c, c.kinds[l], xr, er, jnp.sum(recv), w)
    with jax.named_scope("ep_combine"):
        # Back to each source, where it sent from: its input offset for this rank.
        src_in_off = jnp.sum(jnp.where(jnp.arange(tp)[None, :] < me, counts, 0), axis=1)
        ys = jax.lax.ragged_all_to_all(
            yr,
            jnp.zeros((n, c.expert_hidden), jnp.bfloat16),
            _excl_cumsum(recv),
            recv,
            src_in_off,
            send,
            axis_name=AXIS,
        )
    y = jnp.zeros_like(ys).at[order].set(ys).reshape(rc, k, c.expert_hidden)
    return jnp.where(live_rows[:, None, None], y, 0)


def _layer(
    c: EPConfig,
    l: int,
    stream: jax.Array,
    a: dict,
    wl: dict,
    experts: Sequence[jax.Array],
    pool: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One decoder layer on this rank's rows; (stream, the layer's KV pool)."""
    rep = {
        name: wl[name] for name in (*REPLICATED_ROWS, "pre_expert_w", "post_expert_w")
    }
    me = jax.lax.axis_index(AXIS)
    rc = c.rows_per_rank
    live_rows = me * rc + jnp.arange(rc) < a["n_live"]
    pos = a["pos"]
    table = a["pf"] if l in c.global_layers else a["psw"]

    with jax.named_scope("ep_attn_gather"):
        x_own = _bf(_rms(stream, c.rms_eps, rep["pre_attn_w"])).astype(jnp.bfloat16)
        x = jax.lax.all_gather(x_own, AXIS, tiled=True)  # [R, H]
    with jax.named_scope("ep_attention"):
        qkvg = _mv(x, wl["wqkvg"], wl["wqkvg_s"])
        qd, kd = c.q_dim, c.kv_dim
        q, k, v, og = (
            qkvg[:, :qd],
            qkvg[:, qd : qd + kd],
            qkvg[:, qd + kd : qd + 2 * kd],
            qkvg[:, qd + 2 * kd :],
        )
        qn = _heads_rms(q, c.q_heads, c.rms_eps) * c.scale_query_by
        kn = _heads_rms(k, c.kv_heads, c.rms_eps)
        if l not in c.global_layers:
            rope = a["rope"][pos]
            qn, kn = _rope(qn, c.q_heads, rope), _rope(kn, c.kv_heads, rope)
        kb = kn.astype(FP8).astype(jnp.float32)
        vb = v.astype(FP8).astype(jnp.float32)
        out = _attention(c, l, qn, kb, vb, pool, pos, table)
        pool = _kv_write(c, l, pool, kb, vb, pos, table, a["n_live"])
        attn = _bf(_heads_rms(out, c.q_heads, c.rms_eps))
        gated = _bf(jax.nn.sigmoid(_bf(og)) * attn)
        o = _bf(_mv(gated, wl["wo"], wl["wo_s"]))  # partials travel as bf16
    with jax.named_scope("ep_attn_scatter"):
        o_own = jax.lax.psum_scatter(o, AXIS, scatter_dimension=0, tiled=True)
    a_own = _bf(o_own)
    stream = rep["attn_alpha"] * stream + rep["attn_beta"] * _bf(
        _rms(a_own, c.post_eps)
    )

    with jax.named_scope("ep_route"):
        ffn_bf = _bf(_rms(stream, c.rms_eps, rep["pre_ffn_w"]))
        pooled = _bf(_mv(ffn_bf, wl["wpool"], wl["wpool_s"]))
        xe = _bf(_rms(pooled, c.rms_eps, rep["pre_expert_w"]))
        sc = jax.nn.sigmoid(
            jnp.dot(
                ffn_bf,
                wl["wrouter"].astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            * c.router_mult
        )
        _, ids = jax.lax.top_k(sc + wl["expert_bias"], c.top_k)
    y = _moe(c, l, xe, ids, live_rows, experts)
    with jax.named_scope("ep_combine_rows"):
        ynorm = _bf(_rms(y.astype(jnp.float32) * rep["post_expert_w"], c.post_eps))
        wt = jnp.take_along_axis(sc, ids, axis=1)
        wt = wt / (jnp.sum(wt, axis=1, keepdims=True) + c.router_eps) * c.route_scale
        u = _bf(jnp.sum(ynorm * wt[:, :, None], axis=1))
        f = _bf(_mv(u, wl["wunpool"], wl["wunpool_s"]))
        stream = rep["ffn_alpha"] * stream + rep["ffn_beta"] * _bf(
            _rms(f * rep["post_ffn_w"], c.post_eps)
        )
    return stream, pool


HEAD_PARALLEL = ("wqkvg", "wqkvg_s", "wo", "wo_s")


def step(
    c: EPConfig,
    acts: dict,
    weights: dict,
    experts: list,
    pools: list,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, list]:
    """One decode step: (hid, aux, pools), for tracing inside a caller's jit.

    acts: n_live [] i32, pos [R] i32, pf / psw [n_req * pages_per_seq] i32, emb [R, H] bf16,
      rope [max_position, 2 head_dim] f32, final_gain [H] f32; replicated.
    weights: per layer (lists of length L) replicated REPLICATED_ROWS [H] f32, pre_expert_w /
      post_expert_w [He] f32, expert_bias [E] f32, wrouter [H, E] bf16, wpool [H, He] /
      wunpool [He, H] fp8 with per-column scales wpool_s / wunpool_s; head-parallel wqkvg
      [tp, H, qkvg] / wo [tp, q_dim, H] fp8 with wqkvg_s [tp, qkvg] / wo_s [tp, H], sharded
      on dim 0.
    experts: per layer (w13, w13_scale, w2, w2_scale) [E, tp, ...], sharded on experts.
    pools: per layer [tp, pages, 2 kv_heads, head_dim // 4, 4, page] fp8, sharded on dim 0.
    mesh: the 1-D AXIS mesh; None: the caller's (jax.sharding.use_mesh).
    Returns hid [R, H] bf16 (final norm; dead rows zero) and aux [R, L * H] bf16 (each
    layer's output stream), row-sharded, and the updated pools.
    """
    c.validate()

    def body(acts, weights, experts, pools):
        me = jax.lax.axis_index(AXIS)
        rc = c.rows_per_rank
        emb = jax.lax.dynamic_slice_in_dim(acts["emb"], me * rc, rc).astype(jnp.float32)
        stream = _bf(_rms(emb, c.rms_eps))
        aux, new_pools = [], []
        for l in range(c.n_layers):
            wl = {
                name: (ws[l][0] if name in HEAD_PARALLEL else ws[l])
                for name, ws in weights.items()
            }
            with jax.named_scope(f"layer{l}"):
                stream, pool = _layer(c, l, stream, acts, wl, experts[l], pools[l][0])
            aux.append(stream.astype(jnp.bfloat16))
            new_pools.append(pool[None])
        live = (me * rc + jnp.arange(rc) < acts["n_live"])[:, None]
        hid = jnp.where(live, _bf(_rms(stream, c.rms_eps, acts["final_gain"])), 0)
        return hid.astype(jnp.bfloat16), jnp.concatenate(aux, axis=1), new_pools

    specs = (
        jax.tree.map(lambda _: P(), acts),
        {
            name: [P(AXIS) if name in HEAD_PARALLEL else P() for _ in ws]
            for name, ws in weights.items()
        },
        jax.tree.map(lambda _: P(AXIS), experts),
        jax.tree.map(lambda _: P(AXIS), pools),
    )
    kw = {} if mesh is None else {"mesh": mesh}
    f = jax.shard_map(
        body,
        in_specs=specs,
        out_specs=(P(AXIS), P(AXIS), [P(AXIS)] * c.n_layers),
        check_vma=False,
        **kw,
    )
    return f(acts, weights, experts, pools)


def step_fn(c: EPConfig, mesh: Mesh):
    """step jitted on mesh, the pools donated: fn(acts, weights, experts, pools)."""
    return jax.jit(
        lambda acts, weights, experts, pools: step(
            c, acts, weights, experts, pools, mesh
        ),
        donate_argnums=(3,),
    )
