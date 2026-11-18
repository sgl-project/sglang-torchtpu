# SPDX-License-Identifier: Apache-2.0
"""The per-device body of the sequence-parallel prefill program (EP MoE).

Between layers rank m owns rows [m * Ts, (m + 1) * Ts) of the chunk (Ts = rows / tp). Per layer:

    own rows --AG (e4m3 + row scale)--> attention on every row with this rank's heads
      -> gated output of every row --a2a by row (e4m3 + row scale)--> own rows, every head
      -> o proj on own rows (replicated wo) and the attention residual
      -> router and replicated pool on own rows
      -> each (row, slot) --remote-DMA a2a (common/a2a.py: e4m3 + row scale, expert-grouped)--> its
         expert's rank (E / tp whole experts) -> grouped whole-expert FFN, streamed per expert
         chunk --remote-DMA a2a (bf16)--> the row owner
      -> post-expert norm, weighted sum, replicated unpool and FFN residual on own rows

Dtypes: e4m3 activations (a row scale each) into every dense dot and the experts, bf16 q into
attention; f32 at the norms and the residual; bf16 only for sums that feed a norm (the
expert outputs, the unpool output).

Weights: wqkvg is this rank's head slice; wo / wpool / wunpool are replicated ([qd * tp, H] /
[H, He] / [He, H], scales over every column); a layer's experts are this rank's whole experts
as (w13, s13, w2, s2): u32 words [E/tp, K / pack, N] of e2m1 (8) or e4m3 (4) codes along K,
scales [E/tp, K / block, 1, N]. K order / w13 columns: natural / [all gate | all up], or with
sp_ep_layout decode/spep's (fp4 kperm / per-chunk [gate | up]).
"""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.common import a2a
from sglang_tpu.muse_megakernel.common.config import PFConfig
from sglang_tpu.muse_megakernel.common.consts import LANE, small_row
from sglang_tpu.muse_megakernel.common.numerics import _rms, _topk_iter, heads_rms_xla
from sglang_tpu.muse_megakernel.common.qkv import qkv_from_proj

# dtype instances: RPA / gmm_v2 configs read .itemsize
_F32, _BF16, _FP8 = (
    jnp.dtype(t) for t in (jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn)
)
_FP8_MAX = 448.0
# expert gmm tiles, swept on v7x (32 whole experts, ~5k rows; tile_m 128 re-read each expert's
# weights per 128 rows): 256 / 4096 / 2048 ran 0.81 ms fp8, 0.96 ms fp4 vs 1.9 ms at 128 / 1024 /
# 1024; 2048 / 4096 overflows VMEM on fp8
EXPERT_TILES = (256, 4096, 2048)
# local-expert chunks of the streamed MoE all-to-alls; 8 measured slower on v7x 4096 rows
# (smaller gmm calls, more unpack block overlap outweigh the shorter exposed wait)
N_CHUNKS = 4


def _bf(x):
    """x rounded to bf16, back in f32."""
    return x.astype(_BF16).astype(_F32)


def n_expert_refs(c: PFConfig) -> int:
    """Expert operands of the SP program: (w13, s13, w2, s2) per layer."""
    return 4 * c.n_layers


def _counts(dest, n: int):
    """[n] int32: how many of dest's entries equal each of 0..n-1."""
    return jax.nn.one_hot(dest, n, dtype=jnp.int32).sum(axis=0)


def _fp8_rows(x):
    """f32 rows -> (e4m3 rows, f32 [n, 1] scales): each row scaled to e4m3's range."""
    s = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True), 1e-30) / _FP8_MAX
    return (x / s).astype(_FP8), s


def build_forward(c: PFConfig, axis: str = "model"):
    """prefill/tp's (meta, pos, kv_lens, cu_q, dist, pf, psw, emb, *weights, *experts, *kv) ->
    (hidden, aux, *kv), computed in the sequence-parallel layout with an EP MoE."""
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention,
    )
    from sglang_tpu.moe.pallas.gmm_v2 import TileSizes, gmm_v2

    c.validate()
    if c.fp8_gather or c.expert_groups:
        raise NotImplementedError(
            "the sequence-parallel prefill program has no e4m3 gathers "
            "(SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER) or stacked expert groups"
        )
    if len(c.expert_kinds) != c.n_layers:
        raise ValueError(f"expert_kinds {c.expert_kinds}: one per layer needed")
    L, T, tp, hd, K = c.n_layers, c.rows, c.tp, c.head_dim, c.top_k
    Ts, El = T // tp, c.n_experts // tp
    H, He, qd, kvd = c.hidden, c.expert_hidden, c.q_dim, c.kv_dim
    P = Ts * K  # (row, slot) pairs this rank routes
    G = tp * El  # global experts
    # every (source, expert) segment padded to a2a.SEG rows on both sides
    Pp = P + G * (a2a.SEG - 1)  # padded send rows
    # receive rows: every rank's every slot, padded (whole unpack blocks)
    cap = T * K + G * (a2a.SEG - 1) + El * (a2a.GROUP_ALIGN - 1)
    cap = -(-cap // a2a.UNPACK_ROWS) * a2a.UNPACK_ROWS
    xchg = a2a.exchange(tp, cap, Pp, He)
    # decode-sized steps (swept on v7x, 6 layers, ctx 4096): few rows per expert want a small
    # tile_m (R=16 / 64: 64 beat 256 by 12% / 4%), and at 16 rows the chunk pipeline has no
    # transfer to hide (R=16: 1 chunk 0.48 vs 4 chunks 0.72 ms/layer; R=32 / 64: 4 chunks win)
    expert_tiles = (64,) + EXPERT_TILES[1:] if T <= 1024 else EXPERT_TILES
    n_chunks = 1 if T <= 16 else N_CHUNKS

    def ag(x):
        return jax.lax.all_gather(x, axis, axis=0, tiled=True)

    def to_owner(x):
        """[T, d] (this rank's heads of every row) -> [Ts, tp, d]: own rows, every rank's
        heads in rank order. e4m3 travels as bytes."""
        b = jax.lax.bitcast_convert_type(x, jnp.uint8) if x.dtype == _FP8 else x
        b = jax.lax.all_to_all(b, axis, 0, 0, tiled=True).reshape(tp, Ts, -1)
        b = b.transpose(1, 0, 2)
        return jax.lax.bitcast_convert_type(b, _FP8) if x.dtype == _FP8 else b

    def dense(x8, xs, w, sm, name: str, n: int):
        """e4m3 rows (row scales xs) @ an e4m3 K-major slab, times its per-output-channel
        scales -> f32."""
        y = jnp.dot(x8, w, preferred_element_type=_F32)
        return y * xs * small_row(sm, name + "_s")[:, :n]

    def attention(l, q, kvs, kv_pool, kv_lens, pf, psw, cu_q, dist):
        is_global = l in c.global_layers
        return ragged_paged_attention(
            # bf16 q: the prefill RPA's transposed / ones-V tiles take no e4m3 q
            q.astype(_BF16).reshape(T, c.q_heads, hd),
            kvs[:, :kvd].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
            kvs[:, kvd:].astype(jnp.float8_e4m3fn).reshape(T, c.kv_heads, hd),
            kv_pool,
            kv_lens,
            pf if is_global else psw,
            cu_q,
            dist,
            sm_scale=c.attn_scale,
            sliding_window=None if is_global else c.window,
            k_scale=1.0,
            v_scale=1.0,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    def route(ffn, wrouter, sm):
        """(ids, weights) [Ts, top_k] of own rows: sigmoid scores, top-k of the biased ones."""
        ep = wrouter.shape[1]
        sc = jax.nn.sigmoid(
            jnp.dot(ffn, wrouter, preferred_element_type=_F32) * c.router_mult
        )
        ids, _ = _topk_iter(sc + small_row(sm, "expert_bias")[:, :ep], K)
        ids = jnp.concatenate(ids, axis=1)
        w = jnp.take_along_axis(sc, ids, axis=1)
        return ids, w / (
            jnp.sum(w, axis=1, keepdims=True) + c.router_eps
        ) * c.route_scale

    def expert_ffn(kind: str, x, group_sizes, w13, s13, w2, s2, group_offset, lo, hi):
        """silu(x @ gate) * (x @ up) @ w2 over rows grouped by local expert -> bf16 [cap, He].
        gmm_v2 quantizes x and h to e4m3 (W4A8 / W8A8) and accumulates in f32."""
        q_dtype = jnp.float4_e2m1fn if kind == "fp4" else jnp.float8_e4m3fn

        def gmm(lhs, rhs, scale, fuse_act=None):
            return gmm_v2(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                group_offset=group_offset,
                rhs_scale=scale,
                # rows outside the groups (receive padding) are never read back
                zero_initialize=False,
                preferred_element_type=_BF16,
                rhs_quant_dtype=q_dtype,
                maybe_quantize_lhs=True,
                lhs_scale_per_row=True,
                fuse_act=fuse_act,
                tile_info=TileSizes(*expert_tiles),
            )

        if s13.ndim == 3:  # [E, K / block, N] -> gmm_v2's [E, K / block, 1, N]
            s13, s2 = s13[:, :, None, :], s2[:, :, None, :]
        if not c.sp_ep_layout:
            # silu(gate) * up in the w13 kernel's epilogue
            return gmm(gmm(x, w13, s13, fuse_act="silu"), w2, s2)
        # decode EP layout: w13 columns are per-chunk [gate | up] (decode/spep chunk_cols, chunk
        # = I / tp), which the fused silu cannot read; split them after an unfused gmm. Gate /
        # up come out in w2's row order (kperm for fp4, natural for fp8).
        h = gmm(x, w13, s13)
        act = a2a.silu_mul_chunks(h, lo, hi, w13.shape[-1] // (2 * tp))
        return gmm(act, w2, s2)

    def moe(l, me, xe8, xs, ids, experts):
        """Own rows' expert outputs [Ts, top_k, He] bf16: each (row, slot)'s e4m3 input (row
        scale xs) to its expert's rank and back by streamed remote-DMA all-to-alls (common/a2a.py):
        chunk j's expert FFN runs while chunk j + 1 arrives and chunk j - 1 returns."""
        flat = ids.reshape(P)
        # by global expert: rank, then local expert
        order = jnp.argsort(flat, stable=True)
        sorted_ids = flat[order]
        pln = a2a.plan(ag(_counts(sorted_ids, G)[None]), me, tp, El, n_chunks)
        rows = a2a.padded_rows(pln, sorted_ids)
        me1 = me.reshape(1)
        # padded send row -> own row; padding rows read row 0 and are never read back
        src = (order // K)[jnp.zeros((Pp,), jnp.int32).at[rows].set(jnp.arange(P))]
        send = a2a.pack_rows(xe8, xs)[src]
        recv, *dsems = xchg.dispatch_send(pln.dispatch, me1, send)
        back, csems, ccnts, ys = None, [], [], []
        for j in range(n_chunks):
            # chunk j's arrivals; the last wait also drains this rank's sends
            sems, cnts, keep = [dsems[2 * j]], [pln.dispatch_counts[j, 0]], ()
            if j == n_chunks - 1:
                sems += dsems[1::2]
                cnts += list(pln.dispatch_counts[:, 1])
                keep = (send,)
            recv = xchg.wait(recv, jnp.stack(cnts), sems, *keep)
            lo, hi = pln.chunk_rows[j], pln.chunk_rows[j + 1]
            y = expert_ffn(
                c.expert_kinds[l],
                a2a.unpack_rows(recv, lo, hi, He),
                a2a.chunk_group_sizes(pln, j, n_chunks),
                *experts[4 * l : 4 * l + 4],
                jnp.ones((1,), jnp.int32),
                lo,
                hi,
            )
            tab = pln.combine[j : j + 1]
            if back is None:
                back, *cs = xchg.combine_first(tab, me1, y)
            else:
                back, *cs = xchg.combine_next(tab, me1, y, back)
            # pin the order: chunk j + 1's wait after chunk j's FFN and combine start
            recv, back, dsems, cs = jax.lax.optimization_barrier(
                (recv, back, dsems, cs)
            )
            csems += cs
            ccnts += list(pln.combine_counts[j])
            ys.append(y)
        back = xchg.wait(back, jnp.stack(ccnts), csems, *ys)
        y_own = back[rows[jnp.argsort(order)]]
        return y_own.reshape(Ts, K, He)

    def combine(y, tw, sm):
        """u [Ts, He] f32: sum over slots of weight * norm(post_expert_w * y)."""
        post = small_row(sm, "post_expert_w")[0, :He]
        ynorm = _rms(y.astype(_F32) * post, c.post_eps)
        return jnp.sum(ynorm * tw[:, :, None], axis=1)

    def residual(stream, sm, name: str, delta):
        return small_row(sm, name + "_alpha") * stream + small_row(
            sm, name + "_beta"
        ) * _bf(_rms(delta, c.post_eps))

    def forward(
        meta,
        pos,
        kv_lens,
        cu_q,
        dist,
        pf,
        psw,
        emb,
        rope,
        final_norm,
        smalls,
        wqkvg,
        wo,
        wpool,
        wrouter,
        wunpool,
        *rest,
    ):
        experts, kv = rest[: n_expert_refs(c)], list(rest[n_expert_refs(c) :])
        # the TP rank from meta: axis_index lowers to partition-id, which this export rejects
        me = meta[1]
        rope_rows = jnp.take(rope, pos, axis=0)
        own = jax.lax.dynamic_slice_in_dim(emb, me * Ts, Ts, axis=0)
        # the native decoder streams the bf16-rounded normalized embedding
        stream = _bf(_rms(own, c.rms_eps))
        aux = []
        for l in range(L):
            sm = smalls[l]
            a8, a_s = _fp8_rows(_rms(stream, c.rms_eps) * small_row(sm, "pre_attn_w"))
            qkvg = dense(ag(a8), ag(a_s), wqkvg[l], sm, "wqkvg", c.qkvg_dim)
            q, kvs, gate = qkv_from_proj(
                c, qkvg, rope_rows if c.use_rope(l) else None, xla=True
            )
            o, kv[l] = attention(l, q, kvs, kv[l], kv_lens, pf, psw, cu_q, dist)
            attn = heads_rms_xla(o.reshape(T, qd), c.q_heads, hd, c.rms_eps)
            g8, g_s = _fp8_rows(jax.nn.sigmoid(gate.astype(_F32)) * attn)
            # own rows' every head, then the o-proj on the whole wo: no partial sums
            g = (to_owner(g8).astype(_F32) * to_owner(g_s)).reshape(Ts, tp * qd)
            g8, g_s = _fp8_rows(g)
            stream = residual(stream, sm, "attn", dense(g8, g_s, wo[l], sm, "wo", H))

            ffn = _rms(stream, c.rms_eps) * small_row(sm, "pre_ffn_w")
            ids, tw = route(ffn.astype(_BF16), wrouter[l], sm)
            f8, f_s = _fp8_rows(ffn)
            pooled = dense(f8, f_s, wpool[l], sm, "wpool", He)
            xe8, xs = _fp8_rows(
                _rms(pooled, c.rms_eps) * small_row(sm, "pre_expert_w")[:, :He]
            )
            u8, u_s = _fp8_rows(combine(moe(l, me, xe8, xs, ids, experts), tw, sm))
            # bf16: the unpool output feeds the post-FFN norm
            f = dense(u8, u_s, wunpool[l], sm, "wunpool", H).astype(_BF16)
            stream = residual(
                stream, sm, "ffn", f.astype(_F32) * small_row(sm, "post_ffn_w")
            )
            if l in c.capture_layers:
                aux.append(stream.astype(_BF16))
        hidden = (_rms(stream.astype(_BF16), c.rms_eps) * final_norm[0]).astype(_BF16)
        aux_out = jnp.concatenate(aux, axis=1) if aux else jnp.zeros((Ts, LANE), _BF16)
        return (ag(hidden), ag(aux_out), *kv)

    return forward
