# SPDX-License-Identifier: Apache-2.0
"""Whole-decoder Pallas megakernel for Muse Spark decode and DFlash verify on TPU.

One grid-less ``pallas_call`` per device per step takes the embedded rows and runs all
decoder layers (paged attention, router, experts with the post-expert norm,
pooler/unpooler) and the final norm. Every tensor-parallel reduction is an in-kernel
one-hop remote DMA. The LM head, logits and sampling stay in sglang.

It reads the weights sglang already loaded, in their serving layouts:

* experts: the fused-RS expert-TP layout (``w13 [E, H_e, 2*I_local]`` fp4 or fp8,
  ``w2 [E, I_local, H_e]``); per-layer refs, addressed by a branch tree on the layer
  index so the layer loop can stay rolled;
* KV: the paged fused pools ``[pages, page_size, kv_heads_x2/2, 2, 128]`` bf16 of the
  hybrid SWA allocator, one ref per layer, read in place;
* attention / pooler / unpooler / router: stacked K-major copies made at load time
  (``ops.MuseMegakernel.from_model``), because the matvec streams ``[K, N]`` tiles faster
  than the checkpoint's ``[N, K]``; fp8 with per-output-channel scales, except the router.

Rows are ``num_requests * rows_per_request`` tokens padded to a multiple of the TP size:
one per request at decode, a DFlash verify block per request at target-verify. Each row
carries its own position. The step's own K/V rows join attention from VMEM, causal within
a verify block, and come back as ``newkv`` for ``ops`` to scatter into the pools.

The expert phase follows the fused-RS expert-TP contract: each rank evaluates its
intermediate slice of every selected expert for all rows, the partial outputs are
reduce-scattered by token, summed in f32, rounded to bf16, normalised by the post-expert
norm, weighted and summed. Collective rounds per layer: o_proj reduce-scatter and
all-gather, pooler all-gather, expert reduce-scatter, combined-output all-gather, unpooler
all-gather.
"""

import contextlib
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128
NEG = -1e30
LOG2E = 1.4426950408889634
# largest score bound MKConfig.attn_unshifted accepts
UNSHIFTED_MAX = 60.0
# Barrier semaphore id; distinct from the fused-RS MoE (0) and two-shot all-reduce (7) kernels.
_COLLECTIVE_ID = 6
FP4_PACK = 8
FP4_GROUP = 16

# Profiling only: set True to name in-kernel regions, visible in an xprof trace
# with libtpu's --xla_enable_custom_call_region_trace=true.
_TRACE = False


def _scope(name):
    return jax.named_scope(name) if _TRACE else contextlib.nullcontext()


# small-slab rows (replicated per layer, f32, width = hidden)
SMALL = dict(
    pre_attn_w=0,
    attn_alpha=1,
    attn_beta=2,
    pre_ffn_w=3,
    post_ffn_w=4,
    ffn_alpha=5,
    ffn_beta=6,
    pre_expert_w=7,
    expert_bias=8,
    post_expert_w=9,
    # per-output-channel scales of the fp8 dense weights
    wqkvg_s=10,
    wo_s=11,
    wpool_s=12,
    wunpool_s=13,
    # NVFP4 per-expert global scales (scale_fp8): gate in lanes [0, Ep), up in [Ep, 2 Ep)
    expert_gs=14,
    # and down in lanes [0, Ep)
    expert_gs_down=15,
)
SMALL_ROWS = 16


class MKConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static description of one compiled megakernel (model geometry + step shape)."""

    hidden: int
    head_dim: int
    q_heads: int  # per device
    kv_heads: int  # per device
    n_layers: int
    n_experts: int
    top_k: int
    expert_hidden: int
    inter_local: int
    tp: int
    rms_eps: float
    post_eps: float
    scale_query_by: float
    attn_scale: float
    router_mult: float
    route_scale: float
    router_eps: float
    window: int
    global_layers: tuple  # NoPE + full attention
    expert_kinds: tuple  # per layer: "fp4" | "fp8"
    page_size: int
    pages_per_seq: int
    n_pages_entries: int  # length of the flattened page tables
    max_position: int  # rope table rows
    rows: int  # padded rows R (multiple of tp)
    rows_per_req: int  # 1 at decode, the verify block at target-verify
    n_req: int  # requests in the bucket
    capture_layers: tuple  # aux hidden-state layers for the draft model
    pool_padded_head_dim: int = 128
    # streaming
    tile_k: int = 1024
    tile_n: int = 1152
    # dense weight stream: ring slots (must divide the per-layer tile count) and issue depth
    ring_slots: int = 11
    stream_depth: int = 10
    # expert slots = expert_depth + 1, sized for fp4 slabs; fp8 layers use adjacent slot pairs
    expert_depth: int = 3
    # fp4 experts used by at most this many live (row, slot) pairs (0, 1, 2, 4 or 8) run the
    # exact fp8 block-diagonal dot on just those rows, padded up to the next power of two;
    # busier experts dequantize to bf16 for all rows
    compact_rows: int = 8
    # fp4 experts carry e4m3 block scales plus per-expert globals (smalls rows expert_gs*)
    # instead of bf16 scales with the globals folded in
    scale_fp8: bool = False
    # K chunk of the block-diagonal dot; 128-256 beat 512 at 2-8 rows
    blockdiag_kc: int = 256
    # in a step of at most few_rows_max rows (n_req * rows_per_req), the o-proj all-reduce is one
    # gather of those rows' partials (one latency phase instead of two) and the ffn-output
    # gather ships only those rows
    single_row_coll: bool = True
    # one gather ships every live row to every peer: it trades bytes for a latency phase
    few_rows_max: int = 2
    fp4_chunk_words: int = 128
    chunk_pages: int = 2
    # KV chunks are fetched in groups of kv_group per layer branch-tree walk (each walk has a
    # fixed cost); the ring holds two groups: one being consumed, the next in flight
    kv_group: int = 4
    # independent online-softmax streams over consecutive chunks, merged at the end
    attn_streams: int = 2
    # Attention without a running max: q and k are RMS-normalized per head, so |score| <=
    # scale_query_by * head_dim * attn_scale and p = exp2(score) stays far inside f32 range. The
    # queries carry attn_scale * log2(e) (rounded to bf16 once from f32); row sums stay lane partials.
    attn_unshifted: bool = False
    # With attn_unshifted, global-layer clean chunks run software-pipelined: one loop iteration
    # scores attn_streams chunks while it multiplies the previous ones' probabilities (VMEM) by V.
    attn_pipe: bool = False
    # a stream with no chunk left in its loop iteration skips the chunk (lax.cond) instead of
    # scoring a re-read chunk fully masked
    attn_skip_idle_streams: bool = False
    # SEQ_ALONG_LANE only. > 0: SWA layers fetch chunks of this many pages (a divisor of
    # chunk_pages) into sub-slots of the same ring, and the attention body is specialized per
    # layer type. A window spans a few short chunks; only full-attention layers gain from long ones.
    swa_chunk_pages: int = 0
    # With swa_chunk_pages: a full-attention layer's last chunk masks by key column only instead
    # of by [rows, keys] (same values; a row mask over a long chunk is costly).
    attn_col_mask: bool = False
    # Pool rows hold every local KV head side by side in one pool_padded_head_dim-lane row
    # (kv_heads * head_dim == pool_padded_head_dim) instead of one zero-padded row per head:
    # half the KV bytes and attention work at head_dim 64. The serving pool must match.
    kv_packed: bool = False
    # SEQ_ALONG_LANE pools (TpuSeqAlongLaneKVPool, batched RPA, page >= 256): a page block is
    # [2 * kv_heads, head_dim // pk, pk, page_size] with slots [K_h0, V_h0, K_h1, ...], tokens on
    # lanes and pk consecutive head dims per 32-bit word. Attention runs the packed path on K/V
    # transposed to [kv_dim, tokens]; kv_write_call merges new token columns into their pages.
    kv_sal: bool = False
    kv_sal_fp8: bool = False
    # fp8 SEQ_ALONG_LANE chunks reach the MXU as fp8 (exact bf16 x fp8 products): 1 = K only,
    # 2 = K and V; 0 widens the whole page block to bf16 on the VPU first
    sal_fp8_dot: int = 0
    # each row's expert outputs leave for their reduce-scatter owner as soon as the row's last
    # expert is done, overlapping y_rs with the rest of the expert loop
    y_early: bool = False
    # The expert loop runs owner block by owner block (peers from me+1 on, own rows last), so
    # each block of reduce-scatter rows is final part-way through the loop and leaves for its
    # owner as one DMA while the remaining experts run; y_rs then only waits. Same arithmetic
    # (every (row, slot) is written by exactly one expert).
    y_block_send: bool = False
    # xfull holds the bf16 expert inputs as packed bf16 pairs instead of f32 (exact: the
    # values are bf16): half its VMEM, for the larger steps.
    xfull_packed: bool = False
    # The busy-expert output merge runs one top-k slot at a time and the post-expert combine
    # one owned row at a time, so their temporaries stay [R, He] / [K, He] instead of growing
    # with every row of the step (Mosaic places those values in VMEM). Same arithmetic.
    row_bounded_temps: bool = False
    # page tables arrive as int32 words holding two 16-bit page ids (even entry low): half the
    # SMEM. Past ~600 KB of tables every page lookup stops being an SMEM read (24 requests at
    # 1M context: the 96-row step took 670 ms, the KV write 12 ms).
    packed_pages: bool = False
    # attention multiplies the bf16 queries by attn_scale once instead of every score; exact
    # because ops only sets it for a power-of-two scale
    fold_attn_scale: bool = False
    # router top-k by k rounds of row max (_topk_iter) instead of per-row ranks
    topk_iter: bool = False
    # refills of dense tiles consumed after the expert phase (the unpooler's and the next
    # layer's), due while pool / router stream, are held until the expert loop ends
    lazy_dense: bool = True
    # the next layer's smalls are fetched right after this layer's attention instead of after
    # its unpooler (where they queue behind the lazy refills)
    smalls_early: bool = True
    # distinct experts, their (row, slot) groups and the y-block ends from vector ops and one
    # DMA to SMEM instead of scalar SMEM loops (y_early keeps the scalar loops)
    vec_route: bool = True
    # Not bit-exact. The unpooler all-gather folds into the next layer's qkvg matvec: the
    # attention-input RMS is a per-row scalar, so qkvg = r * ((alpha x + s_f beta f wp) g) @ W,
    # the x and f parts one stacked dot per K block, each f block consumed as it arrives (own
    # first). The residual stream stays exact; only the qkvg LHS rounding moves.
    fold_f: bool = False
    # Not bit-exact. The o-proj all-gather folds into the pooler and router matvecs the same way.
    fold_o: bool = False
    # ship each finished wo column tile to its reduce-scatter owner during the o-proj matvec
    o_rs_streamed: bool = True
    # peer order of a round's sends: "ring" (me+1, me+2, ...) or "xor" (me^1, me^2, ...; ring
    # unless tp is a power of two)
    coll_order: str = "xor"
    # SEQ_ALONG_LANE KV write: one load of every layer's new rows, every live request's page
    # read-modify-write in flight per layer, and a layer's write-backs overlapping the next
    # layer's reads (two slot banks). Same bytes and order of updates per page.
    kv_write_pipelined: bool = False
    vmem_limit_bytes: int = 60 * 2**20

    @property
    def q_dim(self):
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    @property
    def qkvg_dim(self):
        return 2 * self.q_dim + 2 * self.kv_dim

    @property
    def g(self):
        return self.q_heads // self.kv_heads

    @property
    def hidden_local(self):
        return self.hidden // self.tp

    @property
    def expert_hidden_local(self):
        return self.expert_hidden // self.tp

    @property
    def n_experts_padded(self):
        return -(-self.n_experts // LANE) * LANE

    @property
    def pool_kv_heads(self):
        return 1 if (self.kv_packed or self.kv_sal) else self.kv_heads

    @property
    def packed_attn(self):
        # attention sees one row per token, every local head side by side
        return self.kv_packed or self.kv_sal

    @property
    def kv_dtype(self):
        return jnp.float8_e4m3fn if self.kv_sal_fp8 else jnp.bfloat16

    @property
    def sal_pack(self):
        # head dims per 32-bit word of a SEQ_ALONG_LANE page block
        return 4 if self.kv_sal_fp8 else 2

    @property
    def sal_words(self):
        # 32-bit word rows of one SEQ_ALONG_LANE page block: (slot, word) pairs
        return 2 * self.kv_heads * self.head_dim // self.sal_pack

    @property
    def kv_block(self):
        # one page block (SEQ_ALONG_LANE) or one token's row (fused) of a layer's pool
        if self.kv_sal:
            return (
                2 * self.kv_heads,
                self.head_dim // self.sal_pack,
                self.sal_pack,
                self.page_size,
            )
        return (self.pool_kv_heads, 2, self.pool_padded_head_dim)

    @property
    def kv_ring(self):
        if (
            self.kv_sal
        ):  # every chunk the expert w13 slots hold (the ring is a view of them)
            return (
                (self.expert_depth + 1)
                * (self.expert_hidden // FP4_PACK)
                // self.sal_words
            )
        return 2 * self.kv_group

    @property
    def swa_kv_group(self):
        """SWA chunks per prefetch group: half the ring counted in SWA sub-slots."""
        return (
            self.kv_ring
            // 2
            * (self.chunk_pages // (self.swa_chunk_pages or self.chunk_pages))
        )

    @property
    def rows_per_dev(self):
        return self.rows // self.tp

    def use_rope(self, l):
        return l not in self.global_layers

    def validate(self):
        assert self.rows % self.tp == 0 and self.rows >= self.tp
        assert self.n_req * self.rows_per_req <= self.rows
        assert 1 <= self.stream_depth <= self.ring_slots - 1
        assert "fp8" not in self.expert_kinds or (self.expert_depth + 1) % 2 == 0, (
            "fp8 layers pair the expert slots"
        )
        assert self.compact_rows in (0, 1, 2, 4, 8)
        # bf16-packed collective buffers are half the per-device width and must stay lane tiles
        assert (
            self.hidden_local % (2 * LANE) == 0
            and self.expert_hidden_local % (2 * LANE) == 0
        )
        assert (
            not self.kv_packed
            or self.kv_heads * self.head_dim == self.pool_padded_head_dim
        )
        assert not self.kv_sal or (
            not self.kv_packed
            and self.page_size % LANE == 0
            and self.head_dim % (8 * self.sal_pack) == 0
        )
        assert self.attn_streams >= 1 and (self.kv_ring // 2) % self.attn_streams == 0
        # the SAL ring is a view of the expert w13 slots: [chunk, sal_words, chunk tokens]
        assert not self.kv_sal or (
            self.chunk_pages * self.page_size == 2 * self.inter_local
            and (self.expert_depth + 1)
            * (self.expert_hidden // FP4_PACK)
            % self.sal_words
            == 0
        )
        assert not self.swa_chunk_pages or (
            self.kv_sal
            and self.chunk_pages % self.swa_chunk_pages == 0
            and self.swa_kv_group % self.attn_streams == 0
        )
        assert not self.attn_col_mask or self.swa_chunk_pages
        # 1.1: bf16 q and fp8 k rounding; exp2(60 * log2 e) * 1M keys * |v| stays under f32 max
        assert not self.attn_unshifted or (
            1.1 * self.scale_query_by * self.head_dim * self.attn_scale <= UNSHIFTED_MAX
        )
        assert self.coll_order in ("ring", "xor")
        assert not (self.fold_f or self.fold_o) or fold_fits(self)
        # with attn_pipe only the pipelined loop refills the ring: a window's chunks must fit it
        cpt = self.chunk_pages * self.page_size
        assert not self.attn_pipe or (
            self.attn_unshifted
            and self.kv_sal
            and self.kv_ring >= 2 * self.attn_streams
            and (self.window + cpt - 2) // cpt + 1 <= self.kv_ring
        )
        return self


# ----------------------------------------------------------------------------- small helpers
def _rms(x, eps, w=None):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def _heads_rms(x, n_heads, d, eps, gain=1.0):
    parts = [_rms(x[:, h * d : (h + 1) * d], eps) * gain for h in range(n_heads)]
    return parts[0] if n_heads == 1 else jnp.concatenate(parts, axis=1)


def _rope_roll(x, n_heads, d, cos_x, sin_x):
    """Interleaved rotary with lane-expanded tables: the pair partner is one lane roll away."""
    w = n_heads * d
    xv = x.astype(jnp.float32)
    c = jnp.concatenate([cos_x] * n_heads, axis=1) if n_heads > 1 else cos_x
    s = jnp.concatenate([sin_x] * n_heads, axis=1) if n_heads > 1 else sin_x
    lane = jax.lax.broadcasted_iota(jnp.int32, xv.shape, 1)
    even = (lane % 2) == 0
    partner = jnp.where(
        even, pltpu.roll(xv, shift=w - 1, axis=1), pltpu.roll(xv, shift=1, axis=1)
    )
    return xv * c + jnp.where(even, -1.0, 1.0) * partner * s


def _topk_by_rank(scores, k, n_live, rank_ref, score_ref=None):
    """Top-k by rank: ``rank[r, e]`` = number of experts beating ``e`` (higher score, ties to
    the lower id), so pick ``j`` is the expert of rank ``j`` -- the same picks as ``k`` argmax
    rounds, from one ``[E, E]`` comparison per live row instead of a dependent chain. Rows past
    ``n_live`` keep stale ranks."""
    R, E = scores.shape
    lower = jax.lax.broadcasted_iota(jnp.int32, (E, E), 0) < jax.lax.broadcasted_iota(
        jnp.int32, (E, E), 1
    )
    if score_ref is None:
        s_t = scores.T  # [E, R]: e' on sublanes
        for r in range(R):

            @pl.when(r < n_live)
            def _(r=r):
                col, row = s_t[:, r : r + 1], scores[r : r + 1, :]
                beats = (col > row) | ((col == row) & lower)
                rank_ref[r : r + 1, :] = jnp.sum(
                    beats.astype(jnp.int32), axis=0, keepdims=True
                )
    else:
        # The same comparisons over 8-row blocks in a loop: one block's [E, E] temporaries live
        # at a time instead of one set per unrolled row (they grew VMEM ~0.3 MiB per row).
        score_ref[...] = scores

        def block(b, carry):
            r0 = pl.multiple_of(b * 8, 8)
            rows8 = score_ref[pl.ds(r0, 8), :]  # [8, E]
            cols8 = rows8.T  # [E, 8]
            for i in range(8):

                @pl.when(r0 + i < n_live)
                def _(i=i):
                    col, row = cols8[:, i : i + 1], rows8[i : i + 1, :]
                    beats = (col > row) | ((col == row) & lower)
                    rank_ref[pl.ds(r0 + i, 1), :] = jnp.sum(
                        beats.astype(jnp.int32), axis=0, keepdims=True
                    )

            return carry

        jax.lax.fori_loop(0, (n_live + 7) // 8, block, 0)

    rank = rank_ref[...]
    idx = jax.lax.broadcasted_iota(jnp.int32, (R, E), 1)
    picks = [rank == j for j in range(k)]
    return [jnp.sum(jnp.where(p, idx, 0), axis=1, keepdims=True) for p in picks], picks


def _topk_iter(scores, k):
    """Top-k as k rounds of a row max and its lowest index: the picks of _topk_by_rank (higher
    score first, ties to the lower id), a cost flat in the rows where the ranks pay one [E, E]
    comparison per row."""
    R, E = scores.shape
    idx = jax.lax.broadcasted_iota(jnp.int32, (R, E), 1)
    ids, picks = [], []
    for _ in range(k):
        m = jnp.max(scores, axis=1, keepdims=True)
        sel = jnp.min(jnp.where(scores == m, idx, E), axis=1, keepdims=True)
        pick = idx == sel
        ids.append(sel)
        picks.append(pick)
        scores = jnp.where(pick, -jnp.inf, scores)
    return ids, picks


def _bf16_bits(x):
    """f32 -> its round-to-nearest-even bf16 bit pattern in the low 16 bits of a u32."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint32)
    return (b + jnp.uint32(0x7FFF) + ((b >> 16) & jnp.uint32(1))) >> 16


def _pack_bf16x2(x):
    """f32 ``[..., 2n]`` -> u32 ``[..., n]``: bf16 of the first half low, of the second half high.
    Halves a collective payload without a 16-row bf16 tile on 8-row buffers."""
    n = x.shape[-1] // 2
    return _bf16_bits(x[..., :n]) | (_bf16_bits(x[..., n:]) << 16)


def _unpack_bf16x2(w):
    lo = jax.lax.bitcast_convert_type(w << 16, jnp.float32)
    hi = jax.lax.bitcast_convert_type(w & jnp.uint32(0xFFFF0000), jnp.float32)
    return jnp.concatenate([lo, hi], axis=-1)


def _branch(idx, fns, lo=0):
    """Run ``fns[idx - lo]`` for a traced ``idx`` through a balanced tree of two-way branches.

    ``lax.switch`` over ~60 layers costs time linear in the branch count; a tree costs log2(n)
    compares.
    """
    if len(fns) == 1:
        fns[0]()
        return
    mid = len(fns) // 2
    jax.lax.cond(
        idx < lo + mid,
        lambda: _branch(idx, fns[:mid], lo),
        lambda: _branch(idx, fns[mid:], lo + mid),
    )


def _dispatch(idx, refs, fn, only=None):
    """Run ``fn(ref)`` on ``refs[idx]``; a traced index goes through ``_branch``. ``only``: the
    sorted indices ``idx`` can take, so the tree has one leaf per index rather than per ref."""
    if isinstance(idx, int):
        fn(refs[idx])
        return
    if only is None:
        _branch(idx, [functools.partial(fn, r) for r in refs])
        return

    def tree(keys):
        if len(keys) == 1:
            fn(refs[keys[0]])
            return
        mid = len(keys) // 2
        jax.lax.cond(
            idx < keys[mid], lambda: tree(keys[:mid]), lambda: tree(keys[mid:])
        )

    tree(tuple(only))


class _AttnGeom(NamedTuple):
    """Chunk geometry of one attention body: ``cp`` pages per chunk, ``kr`` ring slots fetched in
    groups of ``kg``. ``is_global`` is a Python bool when the body is specialized per layer
    type, else the layer's traced flag."""

    cp: int
    kg: int
    kr: int
    is_global: Any


def _sel(flag, a, b):
    """``where(flag, a, b)``, resolved at trace time for a Python-bool flag."""
    if isinstance(flag, bool):
        return a if flag else b
    return jnp.where(flag, a, b)


def _in_window(is_global, cond):
    """``is_global or cond``; None when every key is in range."""
    if isinstance(is_global, bool):
        return None if is_global else cond
    return jnp.logical_or(is_global, cond)


def _dequant_fp4(words, scale):
    """``[cw, N]`` u32 + ``[cw*8/16, N]`` scales -> ``[cw*8, N]`` bf16."""
    cw, n = words.shape
    v = pltpu.bitcast(words, jnp.float4_e2m1fn).astype(jnp.bfloat16)
    gq = cw * FP4_PACK // FP4_GROUP
    s = scale.astype(jnp.bfloat16)
    s = jnp.broadcast_to(s[:, None, :], (gq, FP4_GROUP, n)).reshape(cw * FP4_PACK, n)
    return v * s


def _fp4_dot(x, wslot, sslot, K, N, chunk_words):
    kw = K // FP4_PACK
    gq = chunk_words * FP4_PACK // FP4_GROUP
    # e4m3 scales tile 32 rows per sublane group: read whole tiles, slice them as f32
    srows = 32 if sslot.dtype == jnp.float8_e4m3fn and gq < 32 else 0
    acc = sblk = None
    for c in range(0, kw, chunk_words):
        cw = min(chunk_words, kw - c)
        g0, gn = c * FP4_PACK // FP4_GROUP, cw * FP4_PACK // FP4_GROUP
        if srows:
            if g0 % srows == 0:
                sblk = sslot[pl.ds(g0, srows), pl.ds(0, N)].astype(jnp.float32)
            sc = sblk[g0 % srows : g0 % srows + gn]
        else:
            sc = sslot[pl.ds(g0, gn), pl.ds(0, N)]
        w = _dequant_fp4(wslot[pl.ds(c, cw), pl.ds(0, N)], sc)
        part = jnp.dot(
            x[:, c * FP4_PACK : (c + cw) * FP4_PACK].astype(jnp.bfloat16),
            w,
            preferred_element_type=jnp.float32,
        )
        acc = part if acc is None else acc + part
    return acc


def _fp4_blockdiag_dot(xc, wslot, sslot, K, N, kc):
    """``xc [A, K]`` bf16 @ an NVFP4 slab (``wslot`` u32 words ``[K/8, N]``, block-16 scales
    ``sslot [K/16, N]``) -> f32 ``[A, N]``, exact in f32.

    The e2m1 codes go to the MXU as fp8 (exact) at the fp8 weight-push rate. The input is
    expanded block-diagonally per ``kc``-row chunk, row ``a*G + g`` = row ``a`` masked to scale
    group ``g``, so each group's partial product lands in its own result row and its scale
    multiplies the ``[A*G, N]`` result instead of the ``[K, N]`` weights.
    """
    A = xc.shape[0]
    kc = min(K, kc)
    gpc = kc // FP4_GROUP
    rows = A * gpc
    diag = (
        jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % gpc
        == jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 1) // FP4_GROUP
    )
    # e4m3 scales tile 32 rows per sublane group: read whole tiles, slice them as f32
    srows = 32 if sslot.dtype == jnp.float8_e4m3fn and gpc < 32 else gpc
    acc = sblk = None
    for c0 in range(0, K, kc):
        xs = xc[:, c0 : c0 + kc]
        lhs = jnp.where(
            diag, jnp.broadcast_to(xs[:, None, :], (A, gpc, kc)).reshape(rows, kc), 0
        ).astype(jnp.bfloat16)
        w = pltpu.bitcast(
            wslot[pl.ds(c0 // FP4_PACK, kc // FP4_PACK), pl.ds(0, N)], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
        g0 = c0 // FP4_GROUP
        if g0 % srows == 0:
            sblk = sslot[pl.ds(g0, srows), pl.ds(0, N)].astype(jnp.float32)
        s = sblk[g0 % srows : g0 % srows + gpc]
        part = res * (jnp.concatenate([s] * A, axis=0) if A > 1 else s)
        acc = part if acc is None else acc + part
    return jnp.concatenate(
        [
            jnp.sum(acc[a * gpc : (a + 1) * gpc], axis=0, keepdims=True)
            for a in range(A)
        ],
        axis=0,
    )


def _fp8_dot(x, wslot, scale_row, K, N, chunk_words):
    """fp8 weights held as ``[K/4, N]`` u32 words; channelwise ``scale_row [1, N]``."""
    kw = K // 4
    acc = None
    for c in range(0, kw, chunk_words):
        cw = min(chunk_words, kw - c)
        w = pltpu.bitcast(wslot[pl.ds(c, cw), pl.ds(0, N)], jnp.float8_e4m3fn)
        part = jnp.dot(
            x[:, c * 4 : (c + cw) * 4].astype(jnp.bfloat16),
            w,
            preferred_element_type=jnp.float32,
        )
        acc = part if acc is None else acc + part
    return acc * scale_row


@functools.lru_cache(maxsize=None)
def _choose_tile(K, N, tk_max, tn_max):
    tns = [d for d in range(LANE, min(N, tn_max) + 1, LANE) if N % d == 0] or [N]
    tks = [d for d in range(16, min(K, tk_max) + 1, 16) if K % d == 0] or [K]
    return max(tks), max(tns)


# ----------------------------------------------------------------------------- weight stream
_PACK = {jnp.dtype(jnp.bfloat16): 2, jnp.dtype(jnp.float8_e4m3fn): 4}


def dense_plan(c):
    """(name, K, N, dtype) of every dense per-layer weight, in consumption order."""
    shapes = (
        ("wqkvg", c.hidden, c.qkvg_dim),
        ("wo", c.q_dim, c.hidden),
        ("wpool", c.hidden, c.expert_hidden_local),
        ("wrouter", c.hidden, c.n_experts_padded),
        ("wunpool", c.expert_hidden, c.hidden_local),
    )
    # qkvg / o / pool / unpool stream as fp8 e4m3 with per-output-channel scales
    fp8 = ("wqkvg", "wo", "wpool", "wunpool")
    return tuple(
        (n, K, N, jnp.float8_e4m3fn if n in fp8 else jnp.bfloat16) for n, K, N in shapes
    )


# The ring sizes worth streaming through: below the minimum the issue window runs too few
# tiles ahead to hide DMA latency; the maximum is the largest ring ops.py asks for.
MIN_RING_SLOTS = 8
MAX_RING_SLOTS = 11


def ring_slot_shape(c):
    """u32 words per ring slot: one [tile_k, tile_n] bf16 tile."""
    return (c.tile_k // 2, c.tile_n)


def fold_fits(c):
    """Whether c can fold its all-gathers: xor peers (arrival-ordered K groups), block-gathered
    rows, K tiles of a power-of-two count of hidden_local blocks, a group's tiles within the issue
    depth (it waits them all before its first dot)."""
    few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max
    if c.tp < 2 or c.tp & (c.tp - 1) or c.coll_order != "xor" or few:
        return False
    Hl = c.hidden_local
    for grp in (("wqkvg",), ("wpool", "wrouter")):
        tiles = [t for t in dense_tiles(c, folded=False) if t[0] in grp]
        if not all(
            t[3] % Hl == 0 and (t[3] // Hl) & (t[3] // Hl - 1) == 0 for t in tiles
        ):
            return False
        g = _fold_g(c, grp)
        if sum(1 for t in tiles if t[1] < g // (t[3] // Hl)) > c.stream_depth:
            return False
    return True


def fold_groups(c):
    """Weights whose matvecs consume an all-gather's column blocks as they arrive."""
    return (("wqkvg",),) * c.fold_f + (("wpool", "wrouter"),) * c.fold_o


def _fold_g(c, grp):
    """Column blocks per fold group: the widest K tile of the group, in blocks."""
    tiles = {(t[0], t[3]) for t in dense_tiles(c, folded=False) if t[0] in grp}
    return max(tk for _, tk in tiles) // c.hidden_local


def dense_tiles(c, folded=True):
    """(name, ki, ni, tk, tn, dtype) of every ring tile of one layer, in consumption order.

    The ring size must divide the tile count, so a count with no divisor in
    ``[MIN_RING_SLOTS, MAX_RING_SLOTS]`` gets its largest tiles split along K until it has
    one (tp16's 26 tiles would otherwise leave a 2-slot ring that runs 1 tile ahead).
    A fold group's tiles run K group by K group (all its weights per group); ki is then logical:
    group j holds the physical K blocks a fold consumes j-th (_WeightStream.phys_k).
    """
    rk, rn = ring_slot_shape(c)
    plan = dense_plan(c)
    shapes = [_choose_tile(K, N, rk * _PACK[jnp.dtype(dt)], rn) for _, K, N, dt in plan]
    for _ in plan:
        T = sum((K // tk) * (N // tn) for (_, K, N, _), (tk, tn) in zip(plan, shapes))
        if any(T % d == 0 for d in range(MIN_RING_SLOTS, MAX_RING_SLOTS + 1)):
            break
        splittable = [
            i
            for i, ((_, K, _, _), (tk, _)) in enumerate(zip(plan, shapes))
            if tk % 32 == 0 and K % (tk // 2) == 0
        ]
        if not splittable:
            break
        i = max(
            splittable,
            key=lambda i: shapes[i][0] * shapes[i][1] * jnp.dtype(plan[i][3]).itemsize,
        )
        shapes[i] = (shapes[i][0] // 2, shapes[i][1])
    geo = {name: (K, N, dt, tk, tn) for (name, K, N, dt), (tk, tn) in zip(plan, shapes)}
    groups = fold_groups(c) if folded else ()
    in_grp = {n: grp for grp in groups for n in grp}
    tiles, done = [], set()
    for name, K, N, dt in plan:
        if name in done:
            continue
        if name not in in_grp:
            _, _, _, tk, tn = geo[name]
            tiles += [
                (name, ki, ni, tk, tn, dt)
                for ni in range(N // tn)
                for ki in range(K // tk)
            ]
            continue
        grp = in_grp[name]
        G = _fold_g(c, grp)
        for j in range(K // (G * c.hidden_local)):
            for w in grp:
                _, Nw, dtw, tk, tn = geo[w]
                s_per = G * c.hidden_local // tk
                tiles += [
                    (w, j * s_per + s_, ni, tk, tn, dtw)
                    for s_ in range(s_per)
                    for ni in range(Nw // tn)
                ]
        done |= set(grp)
    return tuple(tiles)


def fit_ring_slots(c, cap):
    """Largest ring size <= cap that divides the dense tiles per layer (the slot schedule repeats per layer)."""
    T = len(dense_tiles(c))
    return max(d for d in range(2, cap + 1) if T % d == 0)


class _WeightStream:
    """Every dense weight of a layer as one static tile sequence through one VMEM ring.

    The sequence is identical for every layer, so tile t of any layer lives in slot
    ``t % slots`` and the issue window runs ``depth`` tiles ahead across matvec and layer
    boundaries: attention, collectives and the expert phase do not leave the DMA engine idle
    while there is dense weight left to fetch. ``depth <= slots - 1``: the slot a new tile lands
    in held the tile consumed just before it. Slots are u32 words, viewed as the tile's dtype.
    """

    def __init__(self, W, buf, sems, tiles, depth, k_rot=None):
        self.W, self.buf, self.sems = W, buf, sems
        # name -> (traced own block, blocks per group G, tiles per group s_per): logical K tile
        # j * s_per + s streams the physical tile of group (own // G) ^ j (fold arrival order)
        self.k_rot = k_rot or {}
        # lazy_dense: while hold is set, refills of tiles >= hold go to self.held (flush issues them)
        self.hold, self.held = None, []
        self.slots = buf.shape[0]
        self.tiles, self.ranges = tiles, {}
        for t, (name, _, ni, tk, tn, dt) in enumerate(tiles):
            t0 = self.ranges[name][0] if name in self.ranges else t
            self.ranges[name] = (t0, t + 1, tk, tn, ni + 1, dt)
        self.T = len(self.tiles)
        if self.T % self.slots:
            raise ValueError(
                f"{self.T} dense tiles per layer do not divide into {self.slots} ring slots"
            )
        self.depth = min(depth, self.slots - 1, self.T)

    def _copy(self, l, t):
        name, ki, ni, tk, tn, dt = self.tiles[t]
        pack = _PACK[jnp.dtype(dt)]
        s_ = t % self.slots
        pk = self.phys_k(name, ki)
        r0 = (
            pk * (tk // pack)
            if isinstance(pk, int)
            else pl.multiple_of(pk * (tk // pack), 8)
        )
        src = (
            self.W[name]
            .bitcast(jnp.uint32)
            .at[l, pl.ds(r0, tk // pack), pl.ds(ni * tn, tn)]
        )
        return pltpu.make_async_copy(
            src, self.buf.at[s_, pl.ds(0, tk // pack), pl.ds(0, tn)], self.sems.at[s_]
        )

    def phys_k(self, name, ki):
        if name not in self.k_rot:
            return ki
        own, G, s_per = self.k_rot[name]
        return jnp.bitwise_xor(own // G, ki // s_per) * s_per + ki % s_per

    def mv_fold(self, l, names, next_l, G, bw, wait_block, lhs_block):
        """Matvecs of the fold group names over arriving K blocks of width bw.

        Group j holds the G physical blocks k = ((own // G) ^ j) * G + i; per block,
        wait_block(k) waits for it and lhs_block(k) gives the bf16 LHS of its K rows (any row
        count). Returns name -> unscaled f32 [rows, N]."""
        own = self.k_rot[names[0]][0]
        geo = {n: self.ranges[n] for n in names}
        acc = {n: [None] * geo[n][4] for n in names}
        t, t_end = min(geo[n][0] for n in names), max(geo[n][1] for n in names)

        def group_of(u):
            return self.tiles[u][1] // (G * bw // self.tiles[u][3])

        while t < t_end:
            # one group: the consecutive tiles of its K blocks (dense_tiles order)
            j, gt = group_of(t), []
            while t < t_end and group_of(t) == j:
                gt.append(t)
                t += 1
            with _scope("ring_wait"):
                for u in gt:
                    self._copy(l, u).wait()
            pg = jnp.bitwise_xor(own // G, j)
            for i in range(G):
                k = pg * G + i
                wait_block(k)
                lhs = lhs_block(k)
                for u in gt:
                    name, ki, ni, tk, tn, dt = self.tiles[u]
                    bpt = tk // bw
                    if i // bpt != ki % (G // bpt):
                        continue
                    r0 = (i % bpt) * bw
                    w = self.buf.at[u % self.slots].bitcast(dt)[
                        pl.ds(r0, bw), pl.ds(0, tn)
                    ]
                    part = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
                    acc[name][ni] = (
                        part if acc[name][ni] is None else acc[name][ni] + part
                    )
            for u in gt:
                self.refill(l, u, next_l)
        return {
            n: (a[0] if len(a) == 1 else jnp.concatenate(a, axis=1))
            for n, a in acc.items()
        }

    def refill(self, l, t, next_l):
        """Issue the tile depth past tile t, which was just consumed."""
        u = t + self.depth
        if self.hold is not None and u >= self.hold:
            self.held.append((l, t, next_l))
        elif u < self.T:
            self._copy(l, u).start()
        elif next_l is not None:
            self._copy(next_l, u - self.T).start()

    def flush(self):
        held, self.held, self.hold = self.held, [], None
        for l, t, next_l in held:
            self.refill(l, t, next_l)

    def prologue(self, l):
        for t in range(self.depth):
            self._copy(l, t).start()

    def mv(self, x, l, name, next_l, sm=None, on_col=None):
        """``x [R, K]`` @ ``W_l[name]`` -> f32 ``[R, N]``; fp8 weights take their scale row from ``sm``.
        ``on_col(ni, col)`` gets each finished (scaled) column tile as soon as its last K tile is in."""
        assert name not in self.k_rot, "folded weights stream through mv_fold"
        t0, t1, tk, tn, nn, dt = self.ranges[name]
        nk = (t1 - t0) // nn
        is_fp8 = jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn)
        xb = x.astype(jnp.bfloat16)
        cols = [None] * nn
        for t in range(t0, t1):
            with _scope("ring_wait"):
                self._copy(l, t).wait()
            _, ki, ni, _, _, _ = self.tiles[t]
            w = self.buf.at[t % self.slots].bitcast(dt)[pl.ds(0, tk), pl.ds(0, tn)]
            part = jnp.dot(
                xb[:, ki * tk : (ki + 1) * tk], w, preferred_element_type=jnp.float32
            )
            cols[ni] = part if cols[ni] is None else cols[ni] + part
            self.refill(l, t, next_l)
            if on_col is not None and ki == nk - 1:
                if is_fp8:
                    r = SMALL[name + "_s"]
                    cols[ni] = cols[ni] * sm[r : r + 1, ni * tn : (ni + 1) * tn]
                on_col(ni, cols[ni])
        out = cols[0] if nn == 1 else jnp.concatenate(cols, axis=1)
        if is_fp8 and on_col is None:
            r = SMALL[name + "_s"]
            out = out * sm[r : r + 1, : out.shape[1]]
        return out


# ----------------------------------------------------------------------------- collectives
class _Coll:
    """One-hop collectives over the mesh axis as remote DMAs into peers' VMEM pads.

    Every collective kind has its own pad and its own send/receive semaphores: a peer
    one round ahead then cannot satisfy this round's receive wait with its next round's
    bytes. A kind's pad is reused only after at least one other rendezvous round, so a
    single buffer per kind suffices. ``slot`` is the sender's position in the gathered
    result (its TP rank for column-parallel outputs, its mesh index otherwise).
    """

    def __init__(self, tp, me, axis, pads, ssems, rsems, order="ring"):
        self.tp, self.me, self.axis = tp, me, axis
        self.pads, self.ssems, self.rsems = pads, ssems, rsems
        self.kinds = {k: i for i, k in enumerate(pads)}
        self.order = order if tp & (tp - 1) == 0 else "ring"

    def _peer(self, d):
        if self.order == "xor":
            return jnp.bitwise_xor(self.me, d)
        return jax.lax.rem(self.me + d, self.tp)

    def _copy(self, kind, src, dst, peer):
        i = self.kinds[kind]
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=self.ssems.at[i],
            recv_sem=self.rsems.at[i],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def all_gather(self, kind, src_ref, slot, live=None):
        """``src_ref`` from every device into ``pad[slot_of_sender]``; returns the pad ref.

        ``live(m)``: whether device m's part is needed at all; parts of dead devices are
        neither sent nor waited for, and their pad slots hold stale data.
        """
        pad = self.pads[kind]
        if live is None:
            return self.all_gather_start(kind, src_ref, slot)()
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]

        @pl.when(live(self.me))
        def _():
            for c in cps:
                c.start()
            for c in cps:
                c.wait_send()

        for d, c in zip(range(1, self.tp), cps):

            @pl.when(live(self._peer(d)))
            def _(c=c):
                c.wait_recv()

        return pad

    def gather_few(self, kind, src_ref, slot):
        """``src_ref [F, W]`` from every device into ``pad[slot_of_sender]`` of the pad ``[tp, F, W]``."""
        pad = self.pads[kind]
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        return pad

    def all_gather_start(self, kind, src_ref, slot):
        """Issue ``all_gather``; returns the function that waits for it and returns the pad."""
        pad = self.pads[kind]
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()

        def wait():
            for c in cps:
                c.wait()
            return pad

        return wait

    def _block_copy_to(self, kind, src_ref, dst_slot, rsems, peer):
        return pltpu.make_async_remote_copy(
            src_ref=src_ref,
            dst_ref=self.pads[kind].at[dst_slot],
            send_sem=self.ssems.at[self.kinds[kind]],
            recv_sem=rsems.at[dst_slot],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def all_gather_blocks_start(self, kind, src_ref, slot, rsems):
        """all_gather whose sends signal the receiver's rsems[slot], so a receiver can wait
        for one sender's block (all_gather_blocks_wait)."""
        self.pads[kind][slot] = src_ref[...]
        for d in range(1, self.tp):
            self._block_copy_to(kind, src_ref, slot, rsems, self._peer(d)).start()

    def all_gather_blocks_wait(self, kind, src_ref, slot, rsems):
        """(wait_block(k), finish()) of an all_gather_blocks_start: the first waits
        for pad slot k (own slot: no-op), the second for this device's sends."""

        def wait_block(k):
            pl.when(k != slot)(
                self._block_copy_to(kind, src_ref, k, rsems, self.me).wait_recv
            )

        def finish():
            for d in range(1, self.tp):
                self._block_copy_to(
                    kind, src_ref, slot, rsems, self._peer(d)
                ).wait_send()

        return wait_block, finish

    def block_copy(self, kind, send_ref, owner, rows_per):
        """The ``reduce_scatter_rows`` DMA of rows ``[owner*rows_per, ...)`` to ``owner``."""
        return self._copy(
            kind,
            send_ref.at[pl.ds(pl.multiple_of(owner * rows_per, rows_per), rows_per)],
            self.pads[kind].at[self.me],
            owner,
        )

    def reduce_scatter_rows(self, kind, send_ref, rows_per, n_live, started=False):
        """bf16-packed ``send_ref [tp*rows_per, ...]``: device m receives everyone's rows
        ``[m*rows_per, ...)``; returns the pad ``[tp, rows_per, ...]`` (own slot filled locally).
        Devices whose rows are all past ``n_live`` neither receive nor get sent to.
        ``started``: every live owner's ``block_copy`` was already started; only wait."""
        pad = self.pads[kind]

        def rows(ref, m):
            return ref.at[pl.ds(pl.multiple_of(m * rows_per, rows_per), rows_per)]

        my_live = self.me * rows_per < n_live

        @pl.when(my_live)
        def _():
            pad[self.me] = rows(send_ref, self.me)[...]

        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cp = self.block_copy(kind, send_ref, peer, rows_per)
            cps.append((peer, cp))
            if started:
                continue

            @pl.when(peer * rows_per < n_live)
            def _(cp=cp):
                cp.start()

        for peer, cp in cps:

            @pl.when(peer * rows_per < n_live)
            def _(cp=cp):
                cp.wait_send()

        @pl.when(my_live)
        def _():
            for _, cp in cps:
                cp.wait_recv()

        return pad

    def row_copy(self, kind, send_ref, r, rows_per):
        """Row ``r`` of ``send_ref`` to its reduce-scatter owner ``r // rows_per``."""
        owner = r // rows_per
        return owner, self._copy(
            kind,
            send_ref.at[pl.ds(r, 1)],
            self.pads[kind].at[self.me, pl.ds(r - owner * rows_per, 1)],
            owner,
        )

    def reduce_scatter_rows_finish(self, kind, send_ref, rows_per, n_live):
        """``reduce_scatter_rows`` whose remote rows were already started one by one
        (``row_copy``, every row of every live owner exactly once): wait for them."""
        pad = self.pads[kind]
        my_live = self.me * rows_per < n_live

        @pl.when(my_live)
        def _():
            pad[self.me] = send_ref[
                pl.ds(pl.multiple_of(self.me * rows_per, rows_per), rows_per)
            ]

        for r in range(self.tp * rows_per):
            owner, cp = self.row_copy(kind, send_ref, r, rows_per)

            @pl.when((owner != self.me) & (owner * rows_per < n_live))
            def _(cp=cp):
                cp.wait_send()

        @pl.when(my_live)
        def _():
            _, cp = self.row_copy(kind, send_ref, 0, rows_per)
            for _ in range((self.tp - 1) * rows_per):
                cp.wait_recv()

        return pad

    def reduce_scatter_cols(self, kind, src_ref, width):
        """``src_ref [R, tp*width]`` f32: device m receives everyone's columns ``[m*width, ...)``."""
        pad = self.pads[kind]
        pad[self.me] = src_ref[:, pl.ds(pl.multiple_of(self.me * width, LANE), width)]
        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cps.append(
                self._copy(
                    kind,
                    src_ref.at[:, pl.ds(pl.multiple_of(peer * width, LANE), width)],
                    pad.at[self.me],
                    peer,
                )
            )
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        return pad

    def _cols_copy(self, kind, src_ref, width, m):
        c0 = m * width if isinstance(m, int) else pl.multiple_of(m * width, LANE)
        return self._copy(
            kind, src_ref.at[:, pl.ds(c0, width)], self.pads[kind].at[self.me], m
        )

    def reduce_scatter_cols_streamed(self, kind, src_ref, width):
        """``reduce_scatter_cols`` issued one column block at a time: ``send(m)`` once
        ``src_ref[:, m*width:...]`` is written; ``finish()`` waits and returns the pad."""
        pad = self.pads[kind]

        def send(m):
            pl.when(self.me != m)(self._cols_copy(kind, src_ref, width, m).start)

            @pl.when(self.me == m)
            def _():
                c0 = (
                    m * width if isinstance(m, int) else pl.multiple_of(m * width, LANE)
                )
                pad[self.me] = src_ref[:, pl.ds(c0, width)]

        def finish():
            # every block is the same size: these descriptors count the same bytes
            cps = [
                self._cols_copy(kind, src_ref, width, self._peer(d))
                for d in range(1, self.tp)
            ]
            for cp in cps:
                cp.wait_send()
            for cp in cps:
                cp.wait_recv()
            return pad

        return send, finish


def _page(tab_ref, ent, packed):
    """Page id ``ent`` of an SMEM page table, unpacked or two ids per int32 word."""
    if not packed:
        return tab_ref[ent]
    word = tab_ref[ent // 2]
    return jax.lax.shift_right_logical(word, (ent % 2) * 16) & 0xFFFF


def pack_page_table(tab):
    """int32 ``[2n]`` page ids (< 65536) -> int32 ``[n]`` words, even entry in the low half."""
    assert tab.shape[0] % 2 == 0
    return (tab[0::2] & 0xFFFF) | (tab[1::2] << 16)


def route_blocks(c):
    """Lane blocks per vec_route field: (row, slot) pairs and distinct experts both fit."""
    return -(-c.rows * c.top_k // LANE)


def route_rows(c):
    """vec_route table rows: uniq, count, offset, perm fields, then n_u, then blk_end."""
    return -(-(4 * route_blocks(c) + 2) // 8) * 8


def _route_table(ids_list, n_live, K, Ep, NB, pair_key, n_keys):
    """vec_route table ``[route_rows, LANE]`` int32 from per-slot expert ids ``K x [R, 1]``.

    Distinct experts are ordered by (the smallest ``pair_key`` [P, 1] of their pairs, id), pair
    q = k*R + r. Rows ``[f*NB, (f+1)*NB)`` hold field f: 0 experts, 1 their live (row, slot)
    counts, 2 offsets into field 3, 3 pairs as ``row * K + slot`` grouped by expert; row 4*NB
    the distinct count; row 4*NB+1 lane b the experts whose key is <= b. Prefix sums run on
    the MXU (0/1 and counts <= R*K are exact in bf16, summed in f32)."""
    R = ids_list[0].shape[0]
    P = R * K
    f32 = jnp.float32
    col_e = jnp.concatenate(ids_list, axis=0)  # [P, 1]
    q_iota = jax.lax.broadcasted_iota(jnp.int32, (P, 1), 0)
    r_of, k_of = q_iota % R, q_iota // R
    enc = r_of * K + k_of
    live = r_of < n_live
    e_lane = jax.lax.broadcasted_iota(jnp.int32, (P, Ep), 1)
    hit = (col_e == e_lane) & live  # [P, Ep]
    onehot = jnp.where(hit, 1.0, 0.0)
    count = jnp.sum(onehot, axis=0, keepdims=True)  # [1, Ep]
    present = count > 0
    big = float(n_keys + 1)
    key = jnp.min(
        jnp.where(hit, pair_key.astype(f32), big), axis=0, keepdims=True
    )  # [1, Ep]
    e_row = jax.lax.broadcasted_iota(jnp.int32, (1, Ep), 1).astype(f32)
    # per-expert values as columns, padded to a lane tile so the transpose is tile-aligned
    cols = jnp.transpose(
        jnp.concatenate(
            [
                key,
                e_row,
                count,
                jnp.where(present, 1.0, 0.0),
                jnp.zeros((LANE - 4, Ep), f32),
            ],
            axis=0,
        )
    )  # [Ep, LANE]
    key_c, id_c, cnt_c, pres_c = cols[:, 0:1], cols[:, 1:2], cols[:, 2:3], cols[:, 3:4]
    # earlier[e, e']: present e' sorts before e
    earlier = present & ((key < key_c) | ((key == key_c) & (e_row < id_c)))  # [Ep, Ep]
    pos_c = jnp.sum(jnp.where(earlier, 1.0, 0.0), axis=1, keepdims=True).astype(
        jnp.int32
    )
    off_c = jnp.sum(jnp.where(earlier, count, 0.0), axis=1, keepdims=True)  # [Ep, 1]
    off_e = jnp.transpose(
        jnp.concatenate([off_c, jnp.zeros((Ep, LANE - 1), f32)], axis=1)
    )[0:1]  # [1, Ep]
    # each pair's rank among the earlier live pairs of its expert
    same = jax.lax.dot_general(
        onehot.astype(jnp.bfloat16),
        onehot.astype(jnp.bfloat16),
        (((1,), (1,)), ((), ())),
        preferred_element_type=f32,
    )  # [P, P]
    q_row = jax.lax.broadcasted_iota(jnp.int32, (P, P), 0)
    q_col = jax.lax.broadcasted_iota(jnp.int32, (P, P), 1)
    live_col = (q_col % R) < n_live
    rank = jnp.sum(
        jnp.where((q_col < q_row) & live_col, same, 0.0), axis=1, keepdims=True
    )
    dest = (jnp.sum(onehot * off_e, axis=1, keepdims=True) + rank).astype(jnp.int32)
    rows = [None] * (4 * NB)
    for b in range(NB):
        u_lane = jax.lax.broadcasted_iota(jnp.int32, (Ep, LANE), 1) + b * LANE
        m = (pres_c > 0) & (pos_c == u_lane)  # [Ep, LANE]
        for f, v in enumerate((id_c, cnt_c, off_c)):
            rows[f * NB + b] = jnp.sum(
                jnp.where(m, v, 0.0), axis=0, keepdims=True
            ).astype(jnp.int32)
        t_lane = jax.lax.broadcasted_iota(jnp.int32, (P, LANE), 1) + b * LANE
        rows[3 * NB + b] = jnp.sum(
            jnp.where((dest == t_lane) & live, enc, 0), axis=0, keepdims=True
        )
    n_u = jnp.sum(jnp.where(present, 1, 0), axis=1, keepdims=True)
    b_lane = jax.lax.broadcasted_iota(jnp.int32, (Ep, LANE), 1).astype(f32)
    blk_end = jnp.sum(
        jnp.where((pres_c > 0) & (key_c <= b_lane), 1, 0), axis=0, keepdims=True
    )
    n_rows = -(-(4 * NB + 2) // 8) * 8
    return jnp.concatenate(
        rows
        + [jnp.broadcast_to(n_u, (1, LANE)), blk_end]
        + [jnp.zeros((n_rows - 4 * NB - 2, LANE), jnp.int32)],
        axis=0,
    )


def few_rows(c):
    """Rows a step can have, if few enough for the gather-only collectives (else 1, unused)."""
    n = c.n_req * c.rows_per_req
    return n if n <= c.few_rows_max else 1


# ----------------------------------------------------------------------------- scratch
def scratch_vmem_bytes(cfg: MKConfig) -> int:
    """VMEM bytes the scratch plan declares (Mosaic's own temporaries come on top)."""
    total = 0
    for _, ref in scratch_plan(cfg):
        if getattr(ref, "memory_space", None) == pltpu.VMEM:
            total += math.prod(ref.shape) * jnp.dtype(ref.dtype).itemsize
    return total


def scratch_plan(cfg: MKConfig):
    c = cfg
    R, K, H = c.rows, c.top_k, c.hidden
    He, Il = c.expert_hidden, c.inter_local
    nslot = c.expert_depth + 1
    # expert rings in u32 words sized for fp4 (8 K-rows per word); an fp8 slab (4 per word)
    # takes two adjacent slots
    w13_words = He // FP4_PACK
    w2_words = Il // FP4_PACK
    cpt = c.chunk_pages * c.page_size
    sdt = jnp.float8_e4m3fn if c.scale_fp8 else jnp.bfloat16
    items = [
        (
            "ring",
            pltpu.VMEM((c.ring_slots, *ring_slot_shape(c)), jnp.uint32),
        ),  # dense weight stream
        ("ring_sems", pltpu.SemaphoreType.DMA((c.ring_slots,))),
        ("w13", pltpu.VMEM((nslot, w13_words, 2 * Il), jnp.uint32)),
        ("w13s", pltpu.VMEM((nslot, He // FP4_GROUP, 2 * Il), sdt)),
        ("w2", pltpu.VMEM((nslot, w2_words, He), jnp.uint32)),
        ("w2s", pltpu.VMEM((nslot, max(Il // FP4_GROUP, 16), He), sdt)),
        ("w13_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w13s_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w2_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w2s_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("smalls", pltpu.VMEM((2, SMALL_ROWS, H), jnp.float32)),
        ("small_sems", pltpu.SemaphoreType.DMA((2,))),
        ("final_norm", pltpu.VMEM((8, H), jnp.float32)),
        ("rope_blk", pltpu.VMEM((R, 8, 2 * c.head_dim), jnp.float32)),
        ("misc_sems", pltpu.SemaphoreType.DMA((R + 4,))),
        ("xpack", pltpu.VMEM((R, H // 2), jnp.uint32)),
        ("o_send", pltpu.VMEM((R, c.hidden_local // 2), jnp.uint32)),
        ("ysend", pltpu.VMEM((R, K, He // 2), jnp.uint32)),
        ("attn_buf", pltpu.VMEM((R, c.q_dim), jnp.float32)),
        ("qn_s", pltpu.VMEM((R, c.q_dim), jnp.float32)),
        ("kb_s", pltpu.VMEM((R, c.kv_dim), jnp.float32)),
        ("vb_s", pltpu.VMEM((R, c.kv_dim), jnp.float32)),
        ("wk_s", pltpu.VMEM((R, K, LANE), jnp.float32)),
        ("u_send", pltpu.VMEM((c.rows_per_dev, He // 2), jnp.uint32)),
        ("pool_send", pltpu.VMEM((R, c.expert_hidden_local // 2), jnp.uint32)),
        ("f_send", pltpu.VMEM((R, c.hidden_local // 2), jnp.uint32)),
        (
            "kv_chunk",
            # SEQ_ALONG_LANE chunks live in the expert w13 slots
            pltpu.VMEM((1, 8, LANE), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM((c.kv_ring, cpt, *c.kv_block), jnp.bfloat16),
        ),
        # one per (sub-)slot: a chunk's page copies all signal it, one wait takes the chunk
        (
            "kv_chunk_sems",
            pltpu.SemaphoreType.DMA((max(c.kv_ring, 2 * c.swa_kv_group),)),
        ),
        # attn_pipe: probabilities of the chunks in flight, double-buffered by loop iteration
        (
            "attn_p",
            pltpu.VMEM(
                (2, c.attn_streams, c.kv_heads * c.rows_per_req * c.g, cpt)
                if c.attn_pipe
                else (1, 1, 8, LANE),
                jnp.bfloat16,
            ),
        ),
        (
            "kv_row",
            pltpu.VMEM((c.sal_words, R), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM((R, *c.kv_block), jnp.bfloat16),
        ),
        ("kv_row_sem", pltpu.SemaphoreType.DMA((1,))),
        ("uniq", pltpu.SMEM((R * K + 8,), jnp.int32)),
        # per distinct expert u: live (row, slot) count; its pairs, encoded row * K + slot, are
        # perm[ex_off[u] : ex_off[u] + ex_cnt[u]] (ex_pos is the fill cursor)
        ("ex_cnt", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_off", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_pos", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("perm", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_of", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        (
            "xfull",
            pltpu.VMEM((R, He // 2), jnp.uint32)
            if c.xfull_packed
            else pltpu.VMEM((R, He), jnp.float32),
        ),
        ("rank", pltpu.VMEM((R, c.n_experts_padded), jnp.int32)),
        # row_bounded_temps: router scores, read 8 rows at a time by the top-k
        (
            "topk_s",
            pltpu.VMEM(
                (R if c.row_bounded_temps else 8, c.n_experts_padded), jnp.float32
            ),
        ),
        ("ids_v", pltpu.VMEM((R, LANE), jnp.int32)),
        ("ids_s", pltpu.SMEM((R, LANE), jnp.int32)),
        ("ids_sem", pltpu.SemaphoreType.DMA((1,))),
        ("seen", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        # y_early: each row's last expert (index into uniq), and the rows grouped by it:
        # rows done at u are dn_rows[dn_end[u] - dn_cnt[u] : dn_end[u]]
        ("row_done", pltpu.SMEM((R,), jnp.int32)),
        ("dn_cnt", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("dn_end", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("dn_rows", pltpu.SMEM((R,), jnp.int32)),
        # y_block_send: uniq index past the last expert first used by owner block b (b counted
        # from the next rank on), so block b's rows are final once expert blk_end[b] - 1 is done
        ("blk_end", pltpu.SMEM((16,), jnp.int32)),
        ("route_v", pltpu.VMEM((route_rows(c), LANE), jnp.int32)),
        ("route_s", pltpu.SMEM((route_rows(c), LANE), jnp.int32)),
        ("route_sem", pltpu.SemaphoreType.DMA((1,))),
        # collective pads: bf16 pairs packed into u32 (_pack_bf16x2)
        ("pad_o_rs", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("pad_o_ag", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("pad_pool", pltpu.VMEM((c.tp, R, c.expert_hidden_local // 2), jnp.uint32)),
        ("pad_y", pltpu.VMEM((c.tp, c.rows_per_dev, K, He // 2), jnp.uint32)),
        ("pad_u", pltpu.VMEM((c.tp, c.rows_per_dev, He // 2), jnp.uint32)),
        ("pad_f", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("row1_send", pltpu.VMEM((few_rows(c), H // 2), jnp.uint32)),
        ("pad_o1", pltpu.VMEM((c.tp, few_rows(c), H // 2), jnp.uint32)),
        ("pad_f1", pltpu.VMEM((c.tp, few_rows(c), c.hidden_local // 2), jnp.uint32)),
        ("coll_ssems", pltpu.SemaphoreType.DMA((8,))),
        ("coll_rsems", pltpu.SemaphoreType.DMA((8,))),
    ]
    if c.fold_f or c.fold_o:
        items += [
            # fold LHS: the stream part [R, H] and per-column gains (row 0)
            ("fold_x", pltpu.VMEM((R, H), jnp.float32)),
            ("fold_v", pltpu.VMEM((8, H), jnp.float32)),
            # one receive semaphore per sender of the folded all-gathers
            ("o_rsems", pltpu.SemaphoreType.DMA((c.tp,))),
            ("f_rsems", pltpu.SemaphoreType.DMA((c.tp,))),
        ]
    if c.scale_fp8:
        items += [
            # the fp8 layers' bf16 channel scales, which the e4m3 slabs cannot hold
            ("w13c", pltpu.VMEM((nslot, 2, 2 * Il), jnp.bfloat16)),
            ("w2c", pltpu.VMEM((nslot, 2, He), jnp.bfloat16)),
            ("gs_s", pltpu.SMEM((2, 2 * c.n_experts_padded), jnp.float32)),
            ("gs_sem", pltpu.SemaphoreType.DMA((1,))),
        ]
    return items


# ----------------------------------------------------------------------------- the program
WEIGHT_NAMES = (
    "rope",
    "final_norm",
    "smalls",
    "wqkvg",
    "wo",
    "wpool",
    "wrouter",
    "wunpool",
)


def build_kernel(cfg: MKConfig, axis: str = "model"):
    """Per-device body: ``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)``.

    ``meta`` int32 SMEM: [n_live_rows, tp_rank]. ``emb`` [R, H] bf16 is the token
    embedding (gathered and reduced by sglang). ``experts`` is 4 refs per layer
    (w13, w13_scale, w2, w2_scale) in layer order. Outputs: hidden ``[R, H]`` bf16, aux
    ``[R, ncap*H]`` bf16 (or ``[R, 128]`` dummy), then the aliased KV refs.
    """
    c = cfg.validate()
    R, K, H, E = c.rows, c.top_k, c.hidden, c.n_experts
    hd, hq, hkv, g = c.head_dim, c.q_heads, c.kv_heads, c.g
    ph = c.pool_kv_heads
    He, Il, Hel, Hl = (
        c.expert_hidden,
        c.inter_local,
        c.expert_hidden_local,
        c.hidden_local,
    )
    L = c.n_layers
    q = c.rows_per_req
    ps = c.page_size
    cpt = c.chunk_pages * ps
    pd = c.pool_padded_head_dim
    scratch = scratch_plan(c)
    snames = [n for n, _ in scratch]
    nw = len(WEIGHT_NAMES)
    ncap = len(c.capture_layers)
    fp4_layers = [l for l in range(L) if c.expert_kinds[l] == "fp4"]
    # rolled range: the contiguous block of fp4 layers
    roll_lo = fp4_layers[0] if fp4_layers else None
    roll_hi = fp4_layers[-1] + 1 if fp4_layers else None
    if fp4_layers and fp4_layers != list(range(roll_lo, roll_hi)):
        raise NotImplementedError("fp4 expert layers must be one contiguous range")
    gl = tuple(sorted(c.global_layers))

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, emb_ref, *rest):
        W = dict(zip(WEIGHT_NAMES, rest[:nw]))
        rest = rest[nw:]
        EX = rest[: 4 * L]
        rest = rest[4 * L :]
        KV = rest[:L]  # read-only here; see kv_write_call
        hid_out, aux_out, newkv_out = rest[L], rest[L + 1], rest[L + 2]
        S = dict(zip(snames, rest[L + 3 :]))
        n_live = meta_ref[0]
        tp_rank = meta_ref[1]
        me = jax.lax.axis_index(axis) if c.tp > 1 else 0
        coll = _Coll(
            c.tp,
            me,
            axis,
            {
                "o_rs": S["pad_o_rs"],
                "o_ag": S["pad_o_ag"],
                "pool": S["pad_pool"],
                "y": S["pad_y"],
                "u": S["pad_u"],
                "f": S["pad_f"],
                "o1": S["pad_o1"],
                "f1": S["pad_f1"],
            },
            S["coll_ssems"],
            S["coll_rsems"],
            order=c.coll_order,
        )
        F = few_rows(c)
        few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max
        # o-proj column tile m is reduce-scatter block m: ship each as it finishes
        o_rs_streamed = (
            c.o_rs_streamed
            and c.tp > 1
            and not few
            and all(tn == Hl for name, _, _, _, tn, _ in dense_tiles(c) if name == "wo")
        )
        k_rot = {}
        for grp in fold_groups(c):
            # f blocks sit in pad slots by TP rank, o blocks by mesh index
            own = tp_rank if grp == ("wqkvg",) else me
            for name, _, _, tk, _, _ in dense_tiles(c):
                if name in grp:
                    k_rot[name] = (own, _fold_g(c, grp), _fold_g(c, grp) * Hl // tk)
        ws = _WeightStream(
            W,
            S["ring"],
            S["ring_sems"],
            dense_tiles(c),
            c.stream_depth,
            k_rot=k_rot,
        )
        msem = S["misc_sems"]

        def clear_seen(e, carry):
            S["seen"][e] = 0
            return carry

        vec_route = c.vec_route and not c.y_early
        if not vec_route:
            jax.lax.fori_loop(0, c.n_experts_padded, clear_seen, 0)
        # Entry barrier: no peer's remote write lands in this device's VMEM before it has
        # entered the kernel. The call aliases nothing, so its collective_id costs no copy.
        if c.tp > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, c.tp):
                pl.semaphore_signal(
                    barrier,
                    device_id=(jnp.mod(me + d, c.tp),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, c.tp - 1)

        def single_row(x, multi, one):
            """``multi(x)`` -> [R, H], or in a step of at most few_rows_max rows ``one(x)``'s [F, H]
            rows (the rest zero)."""
            if not few:
                return multi(x)
            y = one(x)
            return jnp.concatenate([y, jnp.zeros((R - F, y.shape[1]), y.dtype)], axis=0)

        def allreduce_row(x):
            S["row1_send"][...] = _pack_bf16x2(x[0:F, :])
            pad = coll.gather_few("o1", S["row1_send"], me)
            tot = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                tot = tot + _unpack_bf16x2(pad[d])
            return _unpack_bf16x2(_pack_bf16x2(tot))

        def allreduce(x):
            return single_row(x, allreduce_rows, allreduce_row)

        def allreduce_rows(x):
            """f32 ``[R, H]`` partials -> bf16-exact totals: column reduce-scatter + all-gather.
            Partials travel as bf16 and are summed in f32, as the native bf16 all-reduce."""
            for d in range(c.tp):
                S["xpack"][:, d * Hl // 2 : (d + 1) * Hl // 2] = _pack_bf16x2(
                    x[:, d * Hl : (d + 1) * Hl]
                )
            return allgather_sum(coll.reduce_scatter_cols("o_rs", S["xpack"], Hl // 2))

        def allgather_sum(pad):
            """Reduce-scattered column blocks in pad -> summed, all-gathered [R, H] (fold_o:
            the all-gather is only started)."""
            mine = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                mine = mine + _unpack_bf16x2(pad[d])
            S["o_send"][...] = _pack_bf16x2(mine)
            if c.fold_o:
                coll.all_gather_blocks_start("o_ag", S["o_send"], me, S["o_rsems"])
                return None
            pad2 = coll.all_gather("o_ag", S["o_send"], me)
            return jnp.concatenate(
                [_unpack_bf16x2(pad2[d]) for d in range(c.tp)], axis=1
            )

        def f_blocks():
            return coll.all_gather_blocks_wait("f", S["f_send"], tp_rank, S["f_rsems"])

        def o_blocks():
            return coll.all_gather_blocks_wait("o_ag", S["o_send"], me, S["o_rsems"])

        def fold_lhs(pad):
            """Per-block fold LHS: [stream part; arriving block * gains], bf16 [2R, Hl]."""

            def lhs(k):
                cols = pl.ds(pl.multiple_of(k * Hl, LANE), Hl)
                part = S["fold_x"][:, cols]
                if pad is None:
                    return part.astype(jnp.bfloat16)
                blk = _unpack_bf16x2(S[pad][k]) * S["fold_v"][0:1, cols]
                return jnp.concatenate([part, blk], axis=0).astype(jnp.bfloat16)

            return lhs

        def gathered(pad):
            return jnp.concatenate(
                [_unpack_bf16x2(S[pad][d]) for d in range(c.tp)], axis=1
            )

        def fold_qkvg(l, x, pslot, sm, nxt):
            """qkvg of layer l whose input stream is x plus layer l-1's ffn branch, its f still
            arriving; returns (qkvg, layer l's input stream, exact)."""
            g = sm[SMALL["pre_attn_w"] : SMALL["pre_attn_w"] + 1, :]

            def prow(name):
                return S["smalls"][pslot, SMALL[name] : SMALL[name] + 1, :]

            S["fold_x"][...] = prow("ffn_alpha") * x * g
            S["fold_v"][0:1, :] = prow("ffn_beta") * prow("post_ffn_w") * g
            wait_block, finish = f_blocks()
            with _scope("qkvg_fold"):
                acc = ws.mv_fold(
                    l,
                    ("wqkvg",),
                    nxt,
                    _fold_g(c, ("wqkvg",)),
                    Hl,
                    wait_block,
                    fold_lhs("pad_f"),
                )["wqkvg"]
            finish()
            y = gathered("pad_f") * prow("post_ffn_w")
            s_f = jax.lax.rsqrt(
                jnp.mean(jnp.square(y), axis=-1, keepdims=True) + c.post_eps
            )
            stream = prow("ffn_alpha") * x + prow("ffn_beta") * (y * s_f).astype(
                jnp.bfloat16
            ).astype(jnp.float32)
            r = jax.lax.rsqrt(
                jnp.mean(jnp.square(stream), axis=-1, keepdims=True) + c.rms_eps
            )
            sc = sm[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]
            return r * (acc[:R] + s_f * acc[R:]) * sc, stream

        def plain_qkvg(l, attn_in, sm, nxt):
            """qkvg through the fold tile order with nothing arriving (the first layer)."""
            S["fold_x"][...] = attn_in.astype(jnp.float32)
            acc = ws.mv_fold(
                l,
                ("wqkvg",),
                nxt,
                _fold_g(c, ("wqkvg",)),
                Hl,
                lambda k: None,
                fold_lhs(None),
            )["wqkvg"]
            return acc * sm[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]

        def fold_pool_router(l, x, sm, nxt):
            """(pooled, router logits, the stream after the attention residual) of layer l, the
            o-proj all-gather still arriving; x is the layer's input stream."""
            g = sm[SMALL["pre_ffn_w"] : SMALL["pre_ffn_w"] + 1, :]
            alpha = sm[SMALL["attn_alpha"] : SMALL["attn_alpha"] + 1, :]
            beta = sm[SMALL["attn_beta"] : SMALL["attn_beta"] + 1, :]
            S["fold_x"][...] = alpha * x * g
            S["fold_v"][0:1, :] = beta * g
            wait_block, finish = o_blocks()
            names = ("wpool", "wrouter")
            with _scope("pool_router_fold"):
                acc = ws.mv_fold(
                    l,
                    names,
                    nxt,
                    _fold_g(c, names),
                    Hl,
                    wait_block,
                    fold_lhs("pad_o_ag"),
                )
            finish()
            a = gathered("pad_o_ag")
            s_a = jax.lax.rsqrt(
                jnp.mean(jnp.square(a), axis=-1, keepdims=True) + c.post_eps
            )
            stream = alpha * x + beta * (a * s_a).astype(jnp.bfloat16).astype(
                jnp.float32
            )
            r = jax.lax.rsqrt(
                jnp.mean(jnp.square(stream), axis=-1, keepdims=True) + c.rms_eps
            )

            def comb(v):
                return r * (v[:R] + s_a * v[R:])

            pooled = (
                comb(acc["wpool"]) * sm[SMALL["wpool_s"] : SMALL["wpool_s"] + 1, :Hel]
            )
            logits = comb(acc["wrouter"]) * c.router_mult
            return pooled.astype(jnp.bfloat16).astype(jnp.float32), logits, stream

        # ---------------- embedding: weightless RMS norm of the bf16 rows, kept in bf16 precision
        stream = (
            _rms(emb_ref[...].astype(jnp.float32), c.rms_eps)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )

        # ---------------- rotary rows for every row's position
        cps = []
        for b in range(R):
            p = jnp.clip(pos_ref[b], 0, c.max_position - 1)
            cp = pltpu.make_async_copy(
                W["rope"].at[pl.ds(pl.multiple_of((p // 8) * 8, 8), 8)],
                S["rope_blk"].at[b],
                msem.at[b],
            )
            cp.start()
            cps.append(cp)
        rr = []
        for b in range(R):
            cps[b].wait()
            p = jnp.clip(pos_ref[b], 0, c.max_position - 1)
            sel = jax.lax.broadcasted_iota(jnp.int32, (8, 2 * hd), 0) == (
                p - (p // 8) * 8
            )
            rr.append(
                jnp.sum(jnp.where(sel, S["rope_blk"][b], 0.0), axis=0, keepdims=True)
            )
        rtab = jnp.concatenate(rr, axis=0)
        cos_x, sin_x = rtab[:, :hd], rtab[:, hd:]
        row_live = jax.lax.broadcasted_iota(jnp.int32, (R, 1), 0) < n_live

        def flags(l):
            if isinstance(l, int):
                return (l not in gl), (l in gl)
            is_g = False
            for x in gl:
                is_g = jnp.logical_or(is_g, l == x)
            return jnp.logical_not(is_g), is_g

        def smalls_copy(l, slot):
            return pltpu.make_async_copy(
                W["smalls"].at[l], S["smalls"].at[slot], S["small_sems"].at[slot]
            )

        # ------------------------------------------------------------ attention
        def attention(l, qkvg_fn, use_rope, is_global, nxt, sm):
            """Returns the all-reduced o-proj output, or with fold_o None: its all-gather is
            in flight (fold_pool_router consumes it)."""
            per_geom(is_global, lambda geo: kv_prologue(geo, l, 0))
            with _scope("qkvg_mv"):
                qkvg = qkvg_fn()
            qd, kd = c.q_dim, c.kv_dim
            qv, kv_, vv, og = (
                qkvg[:, :qd],
                qkvg[:, qd : qd + kd],
                qkvg[:, qd + kd : qd + 2 * kd],
                qkvg[:, qd + 2 * kd :],
            )
            qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
            kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
            qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_x, sin_x), qn)
            kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_x, sin_x), kn)
            # the pool's bf16 rounding of K/V, as in the native path (K after norm+rope)
            kb = kn.astype(c.kv_dtype).astype(jnp.bfloat16)
            vb = vv.astype(c.kv_dtype).astype(jnp.bfloat16)
            # ---- this step's K/V rows go out to newkv (scattered into the pools by kv_write_call);
            # attention reads them from VMEM, the cache only for positions before them.
            if (
                c.kv_sal
            ):  # [slot * words + word, row]: an identity matmul transposes exactly
                eye = (
                    jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 0)
                    == jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 1)
                ).astype(jnp.bfloat16)
                S["kv_row"][...] = jnp.concatenate(
                    [
                        pltpu.bitcast(
                            jax.lax.dot_general(
                                eye,
                                x[:, h * hd : (h + 1) * hd],
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            ).astype(c.kv_dtype),
                            jnp.uint32,
                        )
                        for h in range(hkv)
                        for x in (kb, vb)
                    ],
                    axis=0,
                )
            for r in range(0 if c.kv_sal else R):
                if (
                    c.kv_packed
                ):  # the projection row already holds the heads side by side
                    S["kv_row"][r] = jnp.concatenate(
                        [kb[r : r + 1], vb[r : r + 1]], axis=0
                    ).reshape(1, 2, pd)
                    continue
                zpad = jnp.zeros((R, pd - hd), jnp.bfloat16)
                parts = []
                for h in range(hkv):
                    parts.append(
                        jnp.concatenate(
                            [kb[r : r + 1, h * hd : (h + 1) * hd], zpad[r : r + 1]],
                            axis=1,
                        )
                    )
                    parts.append(
                        jnp.concatenate(
                            [vb[r : r + 1, h * hd : (h + 1) * hd], zpad[r : r + 1]],
                            axis=1,
                        )
                    )
                S["kv_row"][r] = jnp.concatenate(parts, axis=0).reshape(hkv, 2, pd)
            kv_out_cp = pltpu.make_async_copy(
                S["kv_row"], newkv_out.at[l], S["kv_row_sem"].at[0]
            )
            kv_out_cp.start()

            # ---- attention per live request (rows of one request share their page fetches), in a
            # runtime loop: the body is emitted once, not once per bucket slot
            S["qn_s"][...] = qn
            S["kb_s"][...] = kb.astype(jnp.float32)
            S["vb_s"][...] = vb.astype(jnp.float32)
            S["attn_buf"][...] = jnp.zeros((R, c.q_dim), jnp.float32)
            with _scope("attn_core"):
                n_req_live = jnp.minimum((n_live + q - 1) // q, c.n_req)

                def requests(geo):
                    def req_loop(j, carry):
                        request_attention_body(geo, l, j, n_req_live)
                        return carry

                    jax.lax.fori_loop(0, n_req_live, req_loop, 0)

                per_geom(is_global, requests)
                kv_out_cp.wait()
            attn = S["attn_buf"][...]
            attn = (
                _heads_rms(attn, hq, hd, c.rms_eps)
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            attn = (
                jax.nn.sigmoid(og.astype(jnp.bfloat16).astype(jnp.float32)) * attn
            ).astype(jnp.bfloat16)
            if o_rs_streamed:
                send, finish = coll.reduce_scatter_cols_streamed(
                    "o_rs", S["xpack"], Hl // 2
                )

                def ship(m, col):
                    S["xpack"][:, m * Hl // 2 : (m + 1) * Hl // 2] = _pack_bf16x2(col)
                    send(m)

                with _scope("o_mv"):
                    ws.mv(attn, l, "wo", nxt, sm=sm, on_col=ship)
                with _scope("o_allreduce"):
                    return allgather_sum(finish())
            with _scope("o_mv"):
                o = ws.mv(attn, l, "wo", nxt, sm=sm)
            with _scope("o_allreduce"):
                return allreduce(o)

        def req_geom(geo, j):
            """First cache chunk and chunk count of request j (cached keys: [first_tok, p_lo))."""
            cpt_g = geo.cp * ps
            r0 = j * q
            p_lo = pos_ref[r0]
            first_tok = _sel(geo.is_global, 0, jnp.maximum(p_lo - c.window + 1, 0))
            first_ch = first_tok // cpt_g
            n_ch = jnp.where(
                jnp.logical_and(r0 < n_live, p_lo > 0),
                (p_lo - 1) // cpt_g - first_ch + 1,
                0,
            )
            return p_lo, first_ch, n_ch

        KR, NS = c.kv_ring, c.attn_streams
        KG = KR // 2
        geo_global = _AttnGeom(c.chunk_pages, KG, KR, True)
        if c.swa_chunk_pages:
            geo_swa = _AttnGeom(
                c.swa_chunk_pages, c.swa_kv_group, 2 * c.swa_kv_group, False
            )

        def per_geom(is_global, fn):
            """``fn(geometry)`` for a layer: one shared body, or with swa_chunk_pages a body per
            layer type."""
            if not c.swa_chunk_pages:
                fn(_AttnGeom(c.chunk_pages, KG, KR, is_global))
            elif isinstance(is_global, bool):
                fn(geo_global if is_global else geo_swa)
            else:
                pl.when(is_global)(lambda: fn(geo_global))
                pl.when(jnp.logical_not(is_global))(lambda: fn(geo_swa))

        def ring_at(geo, slot, pi):
            """Ring entry and page of page ``pi`` of chunk slot ``slot``: a shorter chunk is a
            sub-slot of consecutive pages of an entry."""
            f = c.chunk_pages // geo.cp
            if f == 1:
                return slot, pi
            return slot // f, (slot % f) * geo.cp + pi

        def chunk_pages_of(geo, j, ci):
            pages = []
            for pi in range(geo.cp):
                pg_no = ci * geo.cp + pi
                ent = jnp.minimum(
                    j * c.pages_per_seq + jnp.minimum(pg_no, c.pages_per_seq - 1),
                    c.n_pages_entries - 1,
                )
                pages.append(
                    _sel(
                        geo.is_global,
                        _page(pf_ref, ent, c.packed_pages),
                        _page(psw_ref, ent, c.packed_pages),
                    )
                )
            return pages

        def kv_ring_ref():
            """SEQ_ALONG_LANE: the expert w13 slots, idle during attention, as [chunk, sal_words,
            chunk tokens] u32; the page DMA relayouts the pool's tiles, so chunk loads are whole vregs."""
            if c.kv_sal:
                return S["w13"].reshape(KR, c.sal_words, cpt)
            return S["kv_chunk"]

        def page_src(ref, page):
            if c.kv_sal:  # [2 * kv_heads, hd // pk, pk, page] -> its 32-bit words, tokens on lanes
                return ref.at[page].bitcast(jnp.uint32).reshape(c.sal_words, ps)
            return ref.at[page]

        def chunk_dst(geo, slot, pi):
            if c.kv_sal:
                e, p = ring_at(geo, slot, pi)
                return kv_ring_ref().at[e, :, pl.ds(p * ps, ps)]
            return S["kv_chunk"].at[slot, pl.ds(pi * ps, ps)]

        def chunk_sem(geo, slot):
            return S["kv_chunk_sems"].at[slot]

        def issue_range(geo, l, j, ci0, slot0, n):
            """Chunks ``ci0 .. ci0+n`` of request j into ring slots ``slot0 ..``: ONE layer
            branch-tree walk whose leaf is a small runtime loop, so the 62
            leaves stay small (unrolled per-chunk code in every leaf bloats the kernel)."""

            def leaf(ref):
                def one(u, carry):
                    slot = (slot0 + u) % geo.kr
                    for pi, page in enumerate(chunk_pages_of(geo, j, ci0 + u)):
                        pltpu.make_async_copy(
                            page_src(ref, page),
                            chunk_dst(geo, slot, pi),
                            chunk_sem(geo, slot),
                        ).start()
                    return carry

                jax.lax.fori_loop(0, n, one, 0)

            only = None
            if isinstance(geo.is_global, bool) and c.swa_chunk_pages:
                only = [x for x in range(c.n_layers) if (x in gl) == geo.is_global]
            _dispatch(l, KV, leaf, only)

        def chunk_ref(geo, slot):
            """Ring (sub-)slot ``slot`` of geometry geo: SAL [sal_words, cp * ps] words."""
            if not c.kv_sal:
                return S["kv_chunk"].at[slot]
            e, p = ring_at(geo, slot, 0)
            if geo.cp == c.chunk_pages:
                return kv_ring_ref().at[e]
            return kv_ring_ref().at[
                e, :, pl.ds(pl.multiple_of(p * ps, LANE), geo.cp * ps)
            ]

        def wait_chunk(geo, slot):
            ref = chunk_ref(geo, slot)
            pltpu.make_async_copy(ref, ref, chunk_sem(geo, slot)).wait()

        def kv_prologue(geo, l, j):
            # the request's first two groups of chunks
            _, first_ch, n_ch = req_geom(geo, j)
            issue_range(geo, l, j, first_ch, 0, jnp.minimum(n_ch, geo.kr))

        def rows_at(ref, r0, n):
            """``n`` rows from a dynamic start: one-row accesses, as a multi-row dynamic sublane
            window must start at a multiple of 8."""
            return (
                jnp.concatenate([ref[pl.ds(r0 + t, 1), :] for t in range(n)], axis=0)
                if n > 1
                else ref[pl.ds(r0, 1), :]
            )

        def request_attention_body(geo, l, j, n_req_live):
            """Both local KV heads in one MXU op per step: the query rows of all heads are
            stacked ``[hkv*q*g, .]`` (rows ``(h, t, gi)``). Padded pools: scored against the
            chunk's interleaved key rows ``(token, h)``, scores of the other head masked. Packed
            pools (kv_packed): a token's one row holds every head side by side and a query row
            carries its q in its head's lanes, zeros elsewhere, so the op scores only same-head
            pairs: one key row per token and no head mask."""
            is_global, KG, KR, cpt = geo.is_global, geo.kg, geo.kr, geo.cp * ps
            r0 = pl.multiple_of(j * q, q)
            qg = q * g
            p_lo, first_ch, n_ch = req_geom(geo, j)
            qrows = rows_at(S["qn_s"], r0, q)  # [q, q_dim]

            def qrow(h, t, gi):
                x = qrows[t : t + 1, (h * g + gi) * hd : (h * g + gi + 1) * hd]
                if not c.packed_attn:
                    return x
                z = jnp.zeros((1, hd), x.dtype)
                return jnp.concatenate(
                    [x if hh == h else z for hh in range(hkv)], axis=1
                )

            q2 = jnp.concatenate(
                [
                    qrow(h, t, gi)
                    for h in range(hkv)
                    for t in range(q)
                    for gi in range(g)
                ],
                axis=0,
            )  # [hkv*qg, hd or pd] f32
            if c.attn_unshifted:  # scores in log2 units: p = exp2(score)
                q2 = (q2 * (c.attn_scale * LOG2E)).astype(jnp.bfloat16)
            else:
                q2 = q2.astype(jnp.bfloat16)
                if c.fold_attn_scale:
                    q2 = q2 * jnp.asarray(c.attn_scale, jnp.bfloat16)
            sc = 1.0 if (c.attn_unshifted or c.fold_attn_scale) else c.attn_scale
            rowpos = jnp.concatenate(
                [jnp.broadcast_to(pos_ref[r0 + t], (g, 1)) for t in range(q)] * hkv,
                axis=0,
            )  # [hkv*qg, 1]
            rowhead = jax.lax.broadcasted_iota(jnp.int32, (hkv * qg, 1), 0) // qg

            # unshifted: m is unused and lsum holds [rows, LANE] lane partials of the row sums
            init = (
                jnp.full((hkv * qg, 1), NEG, jnp.float32),
                jnp.zeros((hkv * qg, LANE if c.attn_unshifted else 1), jnp.float32),
                jnp.zeros((hkv * qg, c.kv_dim if c.packed_attn else pd), jnp.float32),
            )
            col = jax.lax.broadcasted_iota(jnp.int32, (1, cpt * ph), 1)
            colhead = col % ph
            coltok = col // ph
            kdim = c.kv_dim if c.packed_attn else hd

            def online(carry, s_, v, v_t=False):
                """``v_t``: v is [kv_dim, tokens] (SEQ_ALONG_LANE chunks), else [tokens, dim]."""
                m, lsum, acc = carry
                with _scope("softmax"):
                    if c.attn_unshifted:
                        m_new, corr = m, None
                        pexp = jnp.exp2(s_)
                        if lsum.shape[1] == LANE:
                            for lt in range(pexp.shape[1] // LANE):
                                lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                        else:
                            lsum = lsum + jnp.sum(pexp, axis=1, keepdims=True)
                    else:
                        m_new = jnp.maximum(m, jnp.max(s_, axis=1, keepdims=True))
                        corr = jnp.exp(m - m_new)
                        pexp = jnp.exp(s_ - m_new)
                        lsum = lsum * corr + jnp.sum(pexp, axis=1, keepdims=True)
                with _scope("pv"):
                    acc = acc[:, : v.shape[0 if v_t else 1]]
                    acc = (acc if corr is None else acc * corr) + jax.lax.dot_general(
                        pexp.astype(jnp.bfloat16),
                        v,
                        (((1,), (1 if v_t else 0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                return m_new, lsum, acc

            def chunk_kv(slot):
                """K and V of a ring chunk: SAL [kv_dim, cpt] (rows h * hd + dim, tokens on lanes)."""
                with _scope("kv_unpack"):
                    if c.kv_sal:
                        fp8_k = c.kv_sal_fp8 and c.sal_fp8_dot >= 1
                        fp8_v = c.kv_sal_fp8 and c.sal_fp8_dot >= 2
                        x8 = pltpu.bitcast(chunk_ref(geo, slot)[...], c.kv_dtype)
                        xb = None if (fp8_k and fp8_v) else x8.astype(jnp.bfloat16)
                        xk, xv = (x8 if fp8_k else xb), (x8 if fp8_v else xb)
                        kk = jnp.concatenate(
                            [xk[2 * h * hd : (2 * h + 1) * hd] for h in range(hkv)],
                            axis=0,
                        )
                        vk = jnp.concatenate(
                            [
                                xv[(2 * h + 1) * hd : (2 * h + 2) * hd]
                                for h in range(hkv)
                            ],
                            axis=0,
                        )
                        return kk, vk
                    u = (
                        S["kv_chunk"]
                        .at[slot]
                        .bitcast(jnp.uint32)
                        .reshape(cpt * ph, pd)[...]
                    )  # rows (token, h)
                    kk = pltpu.bitcast((u & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
                    vk = pltpu.bitcast((u >> 16).astype(jnp.uint16), jnp.bfloat16)
                    return kk, vk

            def qk(kk):
                with _scope("qk"):
                    s_ = jax.lax.dot_general(
                        q2,
                        kk if c.kv_sal else kk[:, :kdim],
                        (((1,), (0 if c.kv_sal else 1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                return s_ if sc == 1.0 else s_ * sc

            def chunk_scores(i, valid, clean=False):
                """Masked scores and V of chunk i; ``clean`` chunks lie wholly in the past of
                every row and skip the position and window masks."""
                ci = first_ch + i
                kk, vk = chunk_kv(i % KR)
                live = None if c.packed_attn else colhead == rowhead
                if not clean:
                    keypos = coltok + ci * cpt
                    if c.attn_col_mask and is_global is True:
                        # a cached key (before p_lo) precedes every verify row: [1, cpt] mask
                        pos_live = (keypos < p_lo) & valid
                    else:
                        pos_live = (keypos < p_lo) & (keypos <= rowpos) & valid
                        in_win = _in_window(is_global, rowpos - keypos < c.window)
                        if in_win is not None:
                            pos_live = pos_live & in_win
                    live = pos_live if live is None else live & pos_live
                s_ = qk(kk)
                return (s_ if live is None else jnp.where(live, s_, NEG)), vk

            def qk_exp(i, lsum):
                """bf16 probabilities of clean chunk i and the lane partials of their row sums."""
                s_ = qk(chunk_kv(i % KR)[0])
                with _scope("softmax"):
                    pexp = jnp.exp2(s_)
                    for lt in range(cpt // LANE):
                        lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                return pexp.astype(jnp.bfloat16), lsum

            def pv_chunk(i, pb):
                with _scope("pv"):
                    return jax.lax.dot_general(
                        pb,
                        chunk_kv(i % KR)[1],
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )

            def refill_global(v0):
                """Chunks [v0, v0 + NS) that exist into their ring slots: one walk over the global
                layers' pools, its leaf a small loop."""
                n = jnp.clip(n_ch - v0, 0, NS)
                # pipelined chunks are a global layer's: its page table alone
                geo_g = geo._replace(is_global=True)

                def leaf(ref):
                    def one(u, carry):
                        v = v0 + u
                        for pi, page in enumerate(
                            chunk_pages_of(geo_g, j, first_ch + v)
                        ):
                            pltpu.make_async_copy(
                                page_src(ref, page),
                                chunk_dst(geo, v % KR, pi),
                                chunk_sem(geo, v % KR),
                            ).start()
                        return carry

                    jax.lax.fori_loop(0, n, one, 0)

                pl.when(n > 0)(lambda: _dispatch(l, KV, leaf, list(gl)))

            def pipelined(n_it):
                """Clean chunks [0, NS * n_it), n_it >= 1: iteration k multiplies chunks NS*k.. by V
                while it scores chunks NS*(k+1).. (independent MXU chains), then refills the slots
                its V just left with the chunks one ring past them."""
                zl = jnp.zeros((hkv * qg, LANE), jnp.float32)
                for t in range(NS):
                    wait_chunk(geo, t)
                lsums = []
                for t in range(NS):
                    pb, ls_ = qk_exp(t, zl)
                    S["attn_p"][0, t] = pb
                    lsums.append(ls_)

                def it(k, carry):
                    lsums, accs = carry
                    i0 = NS * k
                    with _scope("kv_wait"):
                        for t in range(NS):
                            wait_chunk(geo, (i0 + NS + t) % KR)
                    pbs = [S["attn_p"][k % 2, t] for t in range(NS)]
                    accs = [
                        a + pv_chunk(i0 + t, pb)
                        for t, (a, pb) in enumerate(zip(accs, pbs))
                    ]
                    nxt_ = [qk_exp(i0 + NS + t, lsums[t]) for t in range(NS)]
                    for t in range(NS):
                        S["attn_p"][(k + 1) % 2, t] = nxt_[t][0]
                    refill_global(i0 + KR)
                    return [x[1] for x in nxt_], accs

                za = jnp.zeros((hkv * qg, c.kv_dim), jnp.float32)
                lsums, accs = jax.lax.fori_loop(0, n_it - 1, it, (lsums, [za] * NS))
                i0 = NS * (n_it - 1)
                accs = [
                    a + pv_chunk(i0 + t, S["attn_p"][(n_it - 1) % 2, t])
                    for t, a in enumerate(accs)
                ]
                refill_global(i0 + KR)
                return init[0], sum(lsums[1:], lsums[0]), sum(accs[1:], accs[0])

            def body_two_stream(k, carry2, clean=False):
                i0 = 2 * k
                i1 = i0 + 1
                valid_b = i1 < n_ch

                # entering group m >= 1: group m - 1's slots are free; fetch group m + 1. With
                # attn_pipe these loops never pass the ring (code size: no layer walk here).
                if not c.attn_pipe:

                    @pl.when(
                        jnp.logical_and(
                            jnp.logical_and(i0 % KG == 0, i0 >= KG), i0 + KG < n_ch
                        )
                    )
                    def _():
                        issue_range(
                            geo,
                            l,
                            j,
                            first_ch + i0 + KG,
                            (i0 + KG) % KR,
                            jnp.minimum(KG, n_ch - i0 - KG),
                        )

                with _scope("kv_wait"):
                    wait_chunk(geo, i0 % KR)

                    @pl.when(
                        valid_b
                    )  # an odd tail re-reads chunk i0, already waited for
                    def _():
                        wait_chunk(geo, i1 % KR)

                s_a, v_a = chunk_scores(i0, True, clean)
                if c.attn_skip_idle_streams and not clean:
                    # a past-the-end stream skips its chunk instead of scoring it fully masked
                    carry_b = jax.lax.cond(
                        valid_b,
                        lambda cr: online(
                            cr, *chunk_scores(i1, True, False), v_t=c.kv_sal
                        ),
                        lambda cr: cr,
                        carry2[1],
                    )
                    return online(carry2[0], s_a, v_a, v_t=c.kv_sal), carry_b
                s_b, v_b = chunk_scores(
                    i1 if clean else jnp.minimum(i1, n_ch - 1), valid_b, clean
                )
                return (
                    online(carry2[0], s_a, v_a, v_t=c.kv_sal),
                    online(carry2[1], s_b, v_b, v_t=c.kv_sal),
                )

            def body_streams(k, carries, clean=False):
                """NS streams over chunks NS*k .. NS*k+NS-1: every wait before any compute."""
                i0 = NS * k

                @pl.when(
                    jnp.logical_and(
                        jnp.logical_and(i0 % KG == 0, i0 >= KG), i0 + KG < n_ch
                    )
                )
                def _():
                    issue_range(
                        geo,
                        l,
                        j,
                        first_ch + i0 + KG,
                        (i0 + KG) % KR,
                        jnp.minimum(KG, n_ch - i0 - KG),
                    )

                valid = [i0 + s < n_ch for s in range(NS)]
                with _scope("kv_wait"):
                    wait_chunk(geo, i0 % KR)
                    for s in range(1, NS):

                        @pl.when(
                            valid[s]
                        )  # a past-the-end stream re-reads the last chunk
                        def _(s=s):
                            wait_chunk(geo, (i0 + s) % KR)

                out = []
                for s in range(NS):
                    if c.attn_skip_idle_streams and not clean and s > 0:
                        out.append(
                            jax.lax.cond(
                                valid[s],
                                lambda cr, i=i0 + s: online(
                                    cr, *chunk_scores(i, True, False), v_t=c.kv_sal
                                ),
                                lambda cr: cr,
                                carries[s],
                            )
                        )
                        continue
                    i = i0 + s if (clean or s == 0) else jnp.minimum(i0 + s, n_ch - 1)
                    s_, v = chunk_scores(i, True if s == 0 else valid[s], clean)
                    out.append(online(carries[s], s_, v, v_t=c.kv_sal))
                return tuple(out)

            # Global-layer chunks before a request's last one lie wholly in the past of every
            # verify row: chunks [0, n_clean) need only the cross-head mask.
            n_clean = _sel(is_global, jnp.maximum(n_ch - 1, 0), 0)
            if NS != 2 and not c.attn_pipe:
                with _scope("chunk_loop"):
                    n_it, carries = (n_ch + NS - 1) // NS, (init,) * NS
                    if not isinstance(
                        n_clean, int
                    ):  # SWA-only bodies have no clean chunks
                        carries = jax.lax.fori_loop(
                            0,
                            n_clean // NS,
                            functools.partial(body_streams, clean=True),
                            carries,
                        )
                    carries = jax.lax.fori_loop(
                        n_clean // NS, n_it, body_streams, carries
                    )
                    m_all = functools.reduce(jnp.maximum, [cr[0] for cr in carries])
                    ws_ = [jnp.exp(cr[0] - m_all) for cr in carries]
                    carry = (
                        m_all,
                        sum(cr[1] * w for cr, w in zip(carries, ws_)),
                        sum(cr[2] * w for cr, w in zip(carries, ws_)),
                    )
            else:
                with _scope("chunk_loop"):
                    # Two independent online-softmax streams over even and odd chunks, merged at
                    # the end: two dependency chains per iteration for the scheduler to interleave.
                    # A stream that saw no live key keeps m = NEG and gets weight exp(NEG - m) = 0.
                    n_pairs, carry2, k0 = (n_ch + 1) // 2, (init, init), 0
                    if c.attn_pipe and not isinstance(n_clean, int):
                        # attn_streams-chunk pipeline first; its refills cover every later chunk
                        n_it = n_clean // NS
                        carry2 = jax.lax.cond(
                            n_it > 0,
                            lambda: (pipelined(n_it), init),
                            lambda: (init, init),
                        )
                        k0 = n_it * NS // 2
                    if not isinstance(
                        n_clean, int
                    ):  # SWA-only bodies have no clean chunks
                        carry2 = jax.lax.fori_loop(
                            k0,
                            n_clean // 2,
                            functools.partial(body_two_stream, clean=True),
                            carry2,
                        )
                    (m_a, l_a, acc_a), (m_b, l_b, acc_b) = jax.lax.fori_loop(
                        n_clean // 2, n_pairs, body_two_stream, carry2
                    )
                    m_ab = jnp.maximum(m_a, m_b)
                    w_a, w_b = jnp.exp(m_a - m_ab), jnp.exp(m_b - m_ab)
                    carry = (m_ab, l_a * w_a + l_b * w_b, acc_a * w_a + acc_b * w_b)
            if c.attn_unshifted:
                carry = (carry[0], jnp.sum(carry[1], axis=1, keepdims=True), carry[2])

            # every ring slot is read and waited for: the next request's first chunks fly
            # during this request's own-row attention
            @pl.when(j + 1 < n_req_live)
            def _():
                kv_prologue(geo, l, j + 1)

            # the request's own new rows (causal within a verify block), padded to a sublane
            # tile (a 1-row operand does not lower), filler keys masked. Padded pools: key rows
            # (h, t); packed: the projection rows as they are, heads side by side.
            q8 = -(-q // 8) * 8
            krows = rows_at(S["kb_s"], r0, q).astype(jnp.bfloat16)
            vrows = rows_at(S["vb_s"], r0, q).astype(jnp.bfloat16)
            if c.packed_attn:
                zrows = (
                    jnp.zeros((q8 - q, c.kv_dim), jnp.bfloat16) if q8 > q else None
                )  # Mosaic has no 0-row vectors
                kn = jnp.concatenate([krows, zrows], axis=0) if q8 > q else krows
                vn = jnp.concatenate([vrows, zrows], axis=0) if q8 > q else vrows
                nt = jax.lax.broadcasted_iota(jnp.int32, (1, q8), 1)
                head_ok = True
            else:
                zrows = jnp.zeros((q8 - q, hd), jnp.bfloat16) if q8 > q else None
                kn = jnp.concatenate(
                    [
                        x
                        for h in range(hkv)
                        for x in (
                            [krows[:, h * hd : (h + 1) * hd]]
                            + ([zrows] if q8 > q else [])
                        )
                    ],
                    axis=0,
                )
                vn = jnp.concatenate(
                    [
                        x
                        for h in range(hkv)
                        for x in (
                            [vrows[:, h * hd : (h + 1) * hd]]
                            + ([zrows] if q8 > q else [])
                        )
                    ],
                    axis=0,
                )
                ncol = jax.lax.broadcasted_iota(jnp.int32, (1, hkv * q8), 1)
                nt = ncol % q8
                head_ok = ncol // q8 == rowhead
            newpos = jnp.zeros(nt.shape, jnp.int32) + 2**30
            for t in range(q):
                newpos = jnp.where(nt == t, pos_ref[r0 + t], newpos)
            live_new = head_ok & (newpos <= rowpos)
            in_win = _in_window(is_global, rowpos - newpos < c.window)
            if in_win is not None:
                live_new = live_new & in_win
            s_ = jax.lax.dot_general(
                q2, kn, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if sc != 1.0:
                s_ = s_ * sc
            _, lsum, acc = online(carry, jnp.where(live_new, s_, NEG), vn)
            res = acc / jnp.maximum(lsum, 1e-30)  # [hkv*qg, hd or pd]
            lanes = (
                (lambda h: slice(h * hd, (h + 1) * hd))
                if c.packed_attn
                else (lambda h: slice(0, hd))
            )
            for t in range(q):
                S["attn_buf"][pl.ds(r0 + t, 1), :] = jnp.concatenate(
                    [
                        res[h * qg + t * g + gi : h * qg + t * g + gi + 1, lanes(h)]
                        for h in range(hkv)
                        for gi in range(g)
                    ],
                    axis=1,
                )

        # ------------------------------------------------------------ MoE
        NBR = route_blocks(c)
        y_block = c.y_block_send and c.tp > 1

        def rget(name, i):
            """Entry i of a routing table: the vec_route table, else its SMEM array."""
            if not vec_route:
                return S[name][i]
            if name == "blk_end":
                return S["route_s"][4 * NBR + 1, i]
            f = {"uniq": 0, "ex_cnt": 1, "ex_off": 2, "perm": 3}[name]
            return S["route_s"][f * NBR + i // LANE, i % LANE]

        def unique_experts_vec(ids_list):
            # a pair's key: its row owner's y block in send order (peers from me+1 on, own last)
            o = jax.lax.broadcasted_iota(jnp.int32, (R * K, 1), 0) % R // c.rows_per_dev
            key = jax.lax.rem(o - me - 1 + c.tp, c.tp) if y_block else jnp.zeros_like(o)
            S["route_v"][...] = _route_table(
                ids_list, n_live, K, c.n_experts_padded, NBR, key, c.tp
            )
            cp = pltpu.make_async_copy(S["route_v"], S["route_s"], S["route_sem"].at[0])
            cp.start()
            cp.wait()
            return S["route_s"][4 * NBR, 0]

        def unique_experts(ids_list, l):
            """Distinct experts of the live rows' slots, in first-use order, into ``uniq`` (SMEM),
            with each expert's (row, slot) pairs grouped in ``perm``."""
            if vec_route:
                return unique_experts_vec(ids_list)
            S["ids_v"][:, 0:K] = jnp.concatenate(ids_list, axis=1)
            cp = pltpu.make_async_copy(S["ids_v"], S["ids_s"], S["ids_sem"].at[0])
            cp.start()
            cp.wait()
            stamp = l + 1  # "seen" is cleared once per call; one stamp per layer

            def count(slot, n):
                e = S["ids_s"][slot // K, slot % K]
                new = S["seen"][e] != stamp

                @pl.when(new)
                def _():
                    S["seen"][e] = stamp
                    S["uniq"][n] = e
                    S["ex_of"][e] = n
                    S["ex_cnt"][n] = 0

                u = S["ex_of"][e]
                S["ex_cnt"][u] = S["ex_cnt"][u] + 1
                return n + new.astype(jnp.int32)

            if c.y_block_send and c.tp > 1:
                # Scan the rows owner block by owner block, peers from me+1 on and own rows
                # last: uniq comes out grouped by the block that first uses each expert, and
                # blk_end[b] closes block b's group.
                Rc = c.rows_per_dev

                def count_in_block(row0):
                    def body(t, n):
                        r = row0 + t // K
                        e = S["ids_s"][r, t % K]
                        live = r < n_live
                        new = (S["seen"][e] != stamp) & live

                        @pl.when(new)
                        def _():
                            S["seen"][e] = stamp
                            S["uniq"][n] = e
                            S["ex_of"][e] = n
                            S["ex_cnt"][n] = 0

                        @pl.when(live)
                        def _():
                            u = S["ex_of"][e]
                            S["ex_cnt"][u] = S["ex_cnt"][u] + 1

                        return n + new.astype(jnp.int32)

                    return body

                n_u = 0
                for b in range(c.tp):
                    owner = jax.lax.rem(me + 1 + b, c.tp)
                    n_u = jax.lax.fori_loop(0, Rc * K, count_in_block(owner * Rc), n_u)
                    S["blk_end"][b] = n_u
            else:
                n_u = jax.lax.fori_loop(0, n_live * K, count, 0)

            def offsets(u, off):
                S["ex_off"][u] = off
                S["ex_pos"][u] = off
                return off + S["ex_cnt"][u]

            jax.lax.fori_loop(0, n_u, offsets, 0)

            def place(slot, carry):
                u = S["ex_of"][S["ids_s"][slot // K, slot % K]]
                p = S["ex_pos"][u]
                S["perm"][p] = slot
                S["ex_pos"][u] = p + 1
                return carry

            jax.lax.fori_loop(0, n_live * K, place, 0)
            if c.y_early:
                # rows past n_live (sent only for a partly live owner) go with the last expert
                for r in range(R):
                    S["row_done"][r] = jnp.where(r < n_live, 0, jnp.maximum(n_u - 1, 0))

                def last_use(slot, carry):
                    r = slot // K
                    u = S["ex_of"][S["ids_s"][r, slot % K]]
                    S["row_done"][r] = jnp.maximum(S["row_done"][r], u)
                    return carry

                jax.lax.fori_loop(0, n_live * K, last_use, 0)

                def zero(u, carry):
                    S["dn_cnt"][u] = 0
                    return carry

                jax.lax.fori_loop(0, jnp.maximum(n_u, 1), zero, 0)
                for r in range(R):
                    d = S["row_done"][r]
                    S["dn_cnt"][d] = S["dn_cnt"][d] + 1

                def ends(u, off):
                    S["dn_end"][u] = off
                    return off + S["dn_cnt"][u]

                jax.lax.fori_loop(0, jnp.maximum(n_u, 1), ends, 0)
                for r in range(R):
                    d = S["row_done"][r]
                    S["dn_rows"][S["dn_end"][d]] = r
                    S["dn_end"][d] = S["dn_end"][d] + 1
            return n_u

        def moe(l, ffn_in32, sm, kind, nxt, pool_router=None):
            """pool_router(): (pooled, logits) from a fold, instead of from ffn_in32.
            Returns the ffn branch, or with fold_f and a next layer None: its all-gather is
            in flight (the next layer's fold_qkvg consumes it)."""
            Rc = c.rows_per_dev
            Ep = c.n_experts_padded
            if c.lazy_dense:
                ws.hold = ws.ranges["wunpool"][0]
            if pool_router is not None:
                pooled, logits = pool_router()
                S["pool_send"][...] = _pack_bf16x2(pooled)
                pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
            else:
                ffn_in = ffn_in32.astype(jnp.bfloat16)
                # pooler (column-parallel by TP rank) first: its gather flies during routing
                with _scope("pool_mv"):
                    pooled = (
                        ws.mv(ffn_in, l, "wpool", nxt, sm=sm)
                        .astype(jnp.bfloat16)
                        .astype(jnp.float32)
                    )
                S["pool_send"][...] = _pack_bf16x2(pooled)
                pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
                # router: one bf16 pass (as torch_tpu's DEFAULT-precision fp32 F.linear)
                with _scope("router_mv"):
                    logits = ws.mv(ffn_in32, l, "wrouter", nxt) * c.router_mult
            with _scope("topk"):
                sc = jax.nn.sigmoid(logits)
                # padded lanes carry a -1e30 bias, so they are never selected
                bias = sm[SMALL["expert_bias"] : SMALL["expert_bias"] + 1, :Ep]
                if c.topk_iter:
                    ids_list, picks = _topk_iter(sc + bias, K)
                else:
                    ids_list, picks = _topk_by_rank(
                        sc + bias,
                        K,
                        n_live,
                        S["rank"],
                        S["topk_s"] if c.row_bounded_temps else None,
                    )
                wsel = jnp.concatenate(
                    [
                        jnp.sum(jnp.where(p, sc, 0.0), axis=1, keepdims=True)
                        for p in picks
                    ],
                    axis=1,
                )
                wsel = (
                    wsel
                    / (jnp.sum(wsel, axis=1, keepdims=True) + c.router_eps)
                    * c.route_scale
                )  # [R, K]
            with _scope("unique_experts"):
                n_u = unique_experts(ids_list, l)
            if kind == "fp4":
                nslot, W13, W2 = c.expert_depth + 1, S["w13"], S["w2"]
            else:
                nslot = (c.expert_depth + 1) // 2
                W13 = S["w13"].reshape(nslot, He // 4, 2 * Il)
                W2 = S["w2"].reshape(nslot, Il // 4, He)
            depth = nslot - 1
            base = 4 * l if isinstance(l, int) else None
            # scale slabs: e4m3 block scales (fp4, scale_fp8) or bf16 (fp4 folded, fp8 channel)
            chan = c.scale_fp8 and kind != "fp4"
            S13 = S["w13c"] if chan else S["w13s"]
            S2 = S["w2c"] if chan else S["w2s"]
            gscale = c.scale_fp8 and kind == "fp4"
            if gscale:
                gs_copy = pltpu.make_async_copy(
                    W["smalls"].at[l, pl.ds(SMALL["expert_gs"], 2), pl.ds(0, 2 * Ep)],
                    S["gs_s"],
                    S["gs_sem"].at[0],
                )
                gs_copy.start()

            def ex_refs(li):
                return [EX[4 * li + j] for j in range(4)]

            def copies(i, slot, e=None):
                e = rget("uniq", i) if e is None else e
                out = []
                pack = FP4_PACK if kind == "fp4" else 4
                kw13, kw2 = He // pack, Il // pack
                for j, (dst, sem, rows_, cols, is_scale) in enumerate(
                    (
                        (W13, S["w13_sems"], kw13, 2 * Il, False),
                        (
                            S13,
                            S["w13s_sems"],
                            He // FP4_GROUP if kind == "fp4" else 2,
                            2 * Il,
                            True,
                        ),
                        (W2, S["w2_sems"], kw2, He, False),
                        (
                            S2,
                            S["w2s_sems"],
                            Il // FP4_GROUP if kind == "fp4" else 2,
                            He,
                            True,
                        ),
                    )
                ):
                    out.append((j, e, dst, sem, rows_, cols, is_scale))
                return out

            def expert_dmas(refs, i, slot, e=None):
                """The four slab DMAs of distinct expert ``i`` into ``slot``, from one layer's refs."""
                for j, e, dst, sem, rows_, cols, is_scale in copies(i, slot, e):
                    src = refs[j] if is_scale else refs[j].bitcast(jnp.uint32)
                    pltpu.make_async_copy(
                        src.at[e, pl.ds(0, rows_), pl.ds(0, cols)],
                        dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                        sem.at[slot],
                    ).start()

            def for_layer(fn):
                """``fn(refs)`` on this layer's expert refs: ONE layer branch-tree walk."""
                if isinstance(l, int):
                    fn([EX[4 * l + jj] for jj in range(4)])
                else:
                    _branch(
                        l,
                        [
                            functools.partial(fn, [EX[4 * li + jj] for jj in range(4)])
                            for li in range(roll_lo, roll_hi)
                        ],
                        lo=roll_lo,
                    )

            def start_expert(i, slot):
                # expert id read before the walk: one SMEM load instead of one per layer leaf
                e = rget("uniq", i)
                for_layer(lambda refs: expert_dmas(refs, i, slot, e))

            def wait_expert(slot):
                pack = FP4_PACK if kind == "fp4" else 4
                kw13, kw2 = He // pack, Il // pack
                tmpl = [
                    EX[
                        4
                        * (
                            roll_lo
                            if roll_lo is not None and not isinstance(l, int)
                            else l
                        )
                        + jj
                    ]
                    for jj in range(4)
                ]
                for dst, sem, rows_, cols, j in (
                    (W13, S["w13_sems"], kw13, 2 * Il, 0),
                    (
                        S13,
                        S["w13s_sems"],
                        He // FP4_GROUP if kind == "fp4" else 2,
                        2 * Il,
                        1,
                    ),
                    (W2, S["w2_sems"], kw2, He, 2),
                    (
                        S2,
                        S["w2s_sems"],
                        Il // FP4_GROUP if kind == "fp4" else 2,
                        He,
                        3,
                    ),
                ):
                    src = tmpl[j] if j in (1, 3) else tmpl[j].bitcast(jnp.uint32)
                    pltpu.make_async_copy(
                        src.at[0, pl.ds(0, rows_), pl.ds(0, cols)],
                        dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                        sem.at[slot],
                    ).wait()

            # the first `depth` experts in one walk; the leaf is a small loop (code size)
            def prologue(refs):
                def one(t, carry):
                    expert_dmas(refs, t, t % nslot)
                    return carry

                jax.lax.fori_loop(0, jnp.minimum(depth, n_u), one, 0)

            for_layer(prologue)

            with _scope("pool_ag"):
                pad = pool_wait()
            with _scope("ex_setup"):
                x_full = jnp.concatenate(
                    [_unpack_bf16x2(pad[d]) for d in range(c.tp)], axis=1
                )  # [R, He]
                x_full = _rms(
                    x_full,
                    c.rms_eps,
                    sm[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He],
                ).astype(jnp.bfloat16)
                ids3 = jnp.concatenate(
                    [i_[:, :, None] for i_ in ids_list], axis=1
                )  # [R, K, 1]
                # every live (row, slot) gets exactly one expert output, written bf16-packed;
                # rows past n_live stay zero
                S["ysend"][...] = jnp.zeros((R, K, He // 2), jnp.uint32)
                if c.xfull_packed:
                    S["xfull"][...] = _pack_bf16x2(x_full.astype(jnp.float32))
                else:
                    S["xfull"][...] = x_full.astype(jnp.float32)

            def ebody(i, carry):
                slot = i % nslot

                @pl.when(i + depth < n_u)
                def _():
                    start_expert(i + depth, (i + depth) % nslot)

                with _scope("ex_wait"):
                    wait_expert(slot)
                e = rget("uniq", i)

                def full_expert():
                    if kind == "fp4":
                        gu = _fp4_dot(
                            x_full,
                            W13.at[slot],
                            S["w13s"].at[slot],
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                        )
                    else:
                        gu = _fp8_dot(
                            x_full,
                            W13.at[slot],
                            S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(jnp.float32),
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                        )
                    gate, up = gu[:, :Il], gu[:, Il:]
                    if gscale:
                        gate, up = gate * S["gs_s"][0, e], up * S["gs_s"][0, Ep + e]
                    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    if kind == "fp4":
                        y = _fp4_dot(
                            h,
                            W2.at[slot],
                            S["w2s"].at[slot],
                            Il,
                            He,
                            min(Il // FP4_PACK, 16),
                        )
                    else:
                        y = _fp8_dot(
                            h,
                            W2.at[slot],
                            S2[slot, 0:1, :].astype(jnp.float32),
                            Il,
                            He,
                            min(Il // 4, 32),
                        )
                    if gscale:
                        y = y * S["gs_s"][1, e]
                    # slot space: row r, slot k gets y[r] where ids[r, k] == e
                    with _scope("ex_accum"):
                        if c.row_bounded_temps:
                            yp = _pack_bf16x2(y)
                            for k in range(K):
                                S["ysend"][:, k, :] = jnp.where(
                                    ids_list[k] == e, yp, S["ysend"][:, k, :]
                                )
                        else:
                            S["ysend"][...] = jnp.where(
                                ids3 == e, _pack_bf16x2(y)[:, None, :], S["ysend"][...]
                            )

                def compact_expert(nrows):
                    # one variant per row bucket: the expanded input and the scale work scale
                    # with the rows; bucket rows past the expert's count repeat its last pair
                    off, cnt = rget("ex_off", i), rget("ex_cnt", i)
                    pairs = [
                        rget("perm", off + jnp.minimum(j, cnt - 1))
                        for j in range(nrows)
                    ]
                    xrows = [S["xfull"][pl.ds(p // K, 1), :] for p in pairs]
                    if c.xfull_packed:
                        xrows = [_unpack_bf16x2(x) for x in xrows]
                    xc = jnp.concatenate(xrows, axis=0).astype(jnp.bfloat16)
                    gu = _fp4_blockdiag_dot(
                        xc, W13.at[slot], S["w13s"].at[slot], He, 2 * Il, c.blockdiag_kc
                    )
                    gate, up = gu[:, :Il], gu[:, Il:]
                    if gscale:
                        gate, up = gate * S["gs_s"][0, e], up * S["gs_s"][0, Ep + e]
                    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    yc = _fp4_blockdiag_dot(
                        h, W2.at[slot], S["w2s"].at[slot], Il, He, c.blockdiag_kc
                    )
                    yp = _pack_bf16x2(yc * S["gs_s"][1, e] if gscale else yc)
                    # a padded row repeats the last pair, so its write repeats that pair's
                    # identical write: no branch per row (loop code size)
                    for j, p in enumerate(pairs):
                        S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]

                def compact_scoped(nrows):
                    with _scope(f"ex_c{nrows}"):
                        compact_expert(nrows)

                def full_scoped():
                    with _scope("ex_full"):
                        full_expert()

                if kind == "fp4" and c.compact_rows:
                    cnt = rget("ex_cnt", i)
                    # (bucket, variant) picked by a compare tree: one branch per level
                    arms = [
                        (b, functools.partial(compact_scoped, b))
                        for b in (1, 2, 4, 8)
                        if b <= c.compact_rows
                    ]
                    arms.append((None, full_scoped))

                    def pick(arms):
                        if len(arms) == 1:
                            arms[0][1]()
                            return
                        mid = len(arms) // 2
                        jax.lax.cond(
                            cnt <= arms[mid - 1][0],
                            lambda: pick(arms[:mid]),
                            lambda: pick(arms[mid:]),
                        )

                    pick(arms)
                else:
                    full_scoped()
                if c.y_early:

                    def send_row(t, carry):
                        r = S["dn_rows"][S["dn_end"][i] - S["dn_cnt"][i] + t]
                        owner, cp = coll.row_copy("y", S["ysend"], r, Rc)

                        @pl.when((owner != me) & (owner * Rc < n_live))
                        def _():
                            cp.start()

                        return carry

                    jax.lax.fori_loop(0, S["dn_cnt"][i], send_row, 0)
                elif c.y_block_send and c.tp > 1:
                    # carry: the next owner block to send. Block b is final once its group's
                    # last expert is done (an empty group fires with the group before it); one
                    # send site keeps the loop body small.
                    def ready(b):
                        end = rget("blk_end", jnp.minimum(b, c.tp - 2))
                        return (b < c.tp - 1) & (end <= i + 1)

                    def send_block(b):
                        owner = jax.lax.rem(me + 1 + b, c.tp)
                        pl.when(owner * Rc < n_live)(
                            coll.block_copy("y", S["ysend"], owner, Rc).start
                        )
                        return b + 1

                    return jax.lax.while_loop(ready, send_block, carry)
                return carry

            if gscale:
                gs_copy.wait()
            with _scope("experts"):
                jax.lax.fori_loop(0, n_u, ebody, jnp.int32(0))
            ws.flush()
            # ---- token-sharded reduce of the per-rank partials, then norm + combine
            with _scope("y_rs"):
                if c.y_early:
                    pad = coll.reduce_scatter_rows_finish("y", S["ysend"], Rc, n_live)
                else:
                    pad = coll.reduce_scatter_rows(
                        "y",
                        S["ysend"],
                        Rc,
                        n_live,
                        started=c.y_block_send and c.tp > 1,
                    )
            with _scope("combine"):
                if c.row_bounded_temps:
                    # one owned row at a time: the same per-(row, slot) norm and weighted sum
                    pw = sm[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
                    r0 = pl.multiple_of(me * Rc, Rc) if c.tp > 1 else 0
                    wk = jnp.concatenate(
                        [wsel[:, k : k + 1][:, :, None] for k in range(K)], axis=1
                    )  # [R, K, 1]
                    S["wk_s"][...] = jnp.broadcast_to(wk, (R, K, LANE))
                    mine_live = me * Rc < n_live

                    def combine_row(r, carry):
                        ysum = _unpack_bf16x2(pad[0, pl.ds(r, 1)])
                        for d in range(1, c.tp):
                            ysum = ysum + _unpack_bf16x2(
                                pad[d, pl.ds(r, 1)]
                            )  # [1, K, He]
                        ysum = (
                            jnp.where(mine_live, ysum, 0.0)
                            .astype(jnp.bfloat16)
                            .astype(jnp.float32)
                        )
                        ysc = ysum * pw[None]
                        nrm = jax.lax.rsqrt(
                            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True)
                            + c.post_eps
                        )
                        ynorm = (ysc * nrm).astype(jnp.bfloat16).astype(jnp.float32)
                        wrow = S["wk_s"][pl.ds(r0 + r, 1), :, 0:1]  # [1, K, 1]
                        u = jnp.sum(ynorm * wrow, axis=1)  # [1, He]
                        S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
                        return carry

                    jax.lax.fori_loop(0, Rc, combine_row, 0)
                else:
                    ysum = _unpack_bf16x2(pad[0])
                    for d in range(1, c.tp):
                        ysum = ysum + _unpack_bf16x2(pad[d])  # [Rc, K, He]
                    # rows past n_live received nothing: keep them finite
                    ysum = (
                        jnp.where(me * Rc < n_live, ysum, 0.0)
                        .astype(jnp.bfloat16)
                        .astype(jnp.float32)
                    )
                    pw = sm[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
                    ysc = ysum * pw[None]
                    nrm = jax.lax.rsqrt(
                        jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
                    )
                    ynorm = (ysc * nrm).astype(jnp.bfloat16).astype(jnp.float32)
                    # this device's rows' routing weights
                    r0 = pl.multiple_of(me * Rc, Rc) if c.tp > 1 else 0
                    wk = jnp.concatenate(
                        [wsel[:, k : k + 1][:, :, None] for k in range(K)], axis=1
                    )  # [R, K, 1]
                    S["wk_s"][...] = jnp.broadcast_to(wk, (R, K, LANE))
                    wmine = S["wk_s"][pl.ds(r0, Rc), :, 0:1]  # [Rc, K, 1]
                    u = jnp.sum(ynorm * wmine, axis=1)  # [Rc, He]
                    S["u_send"][...] = _pack_bf16x2(u)
            with _scope("u_ag"):
                padu = coll.all_gather(
                    "u", S["u_send"], me, live=lambda m: m * Rc < n_live
                )
            u_full = jnp.concatenate(
                [_unpack_bf16x2(padu[d]) for d in range(c.tp)], axis=0
            )  # [R, He]
            u_full = jnp.where(row_live, u_full, 0.0)
            with _scope("unpool_mv"):
                f = (
                    ws.mv(u_full, l, "wunpool", nxt, sm=sm)
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
            if nxt is not None and not c.smalls_early:
                smalls_copy(l + 1, (l + 1) % 2).start()

            def gather_rows(f):
                S["f_send"][...] = _pack_bf16x2(f)
                padf = coll.all_gather("f", S["f_send"], tp_rank)
                return jnp.concatenate(
                    [_unpack_bf16x2(padf[d]) for d in range(c.tp)], axis=1
                )  # [R, H]

            def gather_row(f):
                S["row1_send"][:, 0 : Hl // 2] = _pack_bf16x2(f[0:F, :])
                padf = coll.gather_few(
                    "f1", S["row1_send"].at[:, pl.ds(0, Hl // 2)], tp_rank
                )
                return jnp.concatenate(
                    [_unpack_bf16x2(padf[d]) for d in range(c.tp)], axis=1
                )  # [F, H]

            if c.fold_f and nxt is not None:
                S["f_send"][...] = _pack_bf16x2(f)
                coll.all_gather_blocks_start("f", S["f_send"], tp_rank, S["f_rsems"])
                return None
            with _scope("f_ag"):
                return single_row(f, gather_rows, gather_row)

        def capture(l, stream):
            for ci, cl in enumerate(c.capture_layers):
                if isinstance(l, int):
                    if l == cl:
                        aux_out[:, ci * H : (ci + 1) * H] = stream.astype(jnp.bfloat16)
                    continue

                @pl.when(l == cl)
                def _(ci=ci, stream=stream):
                    aux_out[:, ci * H : (ci + 1) * H] = stream.astype(jnp.bfloat16)

        def layer(l, carry, kind, next_layer, prefetched, pend=False):
            """carry: the layer's input stream, or with pend (fold_f) the previous layer's
            stream before its ffn residual, whose f is still arriving. Returns the output
            stream, or with fold_f and a next layer the stream before the ffn residual."""
            nxt = l + 1 if next_layer else None
            slot = l % 2
            if not prefetched:
                smalls_copy(l, slot).start()
                ws.prologue(l)
            with _scope("smalls_wait"):
                smalls_copy(l, slot).wait()
            sm = S["smalls"][slot]
            use_rope, is_global = flags(l)

            def row(name, n=H):
                return sm[SMALL[name] : SMALL[name] + 1, :n]

            hold = {}
            if pend:

                def qkvg_fn():
                    qkvg, hold["stream"] = fold_qkvg(l, carry, (l + 1) % 2, sm, nxt)
                    return qkvg

            else:
                stream = carry
                attn_in = _rms(stream, c.rms_eps, row("pre_attn_w")).astype(
                    jnp.bfloat16
                )
                if c.fold_f:
                    qkvg_fn = functools.partial(plain_qkvg, l, attn_in, sm, nxt)
                else:
                    qkvg_fn = functools.partial(ws.mv, attn_in, l, "wqkvg", nxt, sm=sm)
            with _scope("attention"):
                a = attention(l, qkvg_fn, use_rope, is_global, nxt, sm)
            if (
                nxt is not None and c.smalls_early
            ):  # layer l - 1's smalls (fold_qkvg) are read
                smalls_copy(l + 1, (l + 1) % 2).start()
            if pend:
                stream = hold["stream"]
                capture(l - 1, stream)
            if c.fold_o:

                def pool_router():
                    pooled, logits, hold["mid"] = fold_pool_router(l, stream, sm, nxt)
                    return pooled, logits

                with _scope("moe"):
                    f = moe(l, None, sm, kind, nxt, pool_router=pool_router)
                stream = hold["mid"]
            else:
                a = a.astype(jnp.bfloat16).astype(jnp.float32)
                stream = row("attn_alpha") * stream + row("attn_beta") * _rms(
                    a, c.post_eps
                ).astype(jnp.bfloat16).astype(jnp.float32)
                ffn_in32 = _rms(stream, c.rms_eps, row("pre_ffn_w"))
                with _scope("moe"):
                    f = moe(l, ffn_in32, sm, kind, nxt)
            if f is None:  # fold_f: the next layer finishes the ffn residual
                return stream
            stream = row("ffn_alpha") * stream + row("ffn_beta") * _rms(
                f * row("post_ffn_w"), c.post_eps
            ).astype(jnp.bfloat16).astype(jnp.float32)
            capture(l, stream)
            return stream

        # every layer but the last prefetches the next one's smalls and tiles
        state = {"pref": False, "pend": False}

        def emit(l, stream, kind, next_layer):
            stream = layer(l, stream, kind, next_layer, state["pref"], state["pend"])
            state["pref"] = next_layer
            state["pend"] = c.fold_f and next_layer
            return stream

        def rolled(i, v):
            # every rolled layer follows a layer that left its f in flight (with fold_f)
            assert state["pend"] == c.fold_f
            return layer(i, v, "fp4", True, True, c.fold_f)

        for l in range(L):
            if roll_lo is not None and l == roll_lo and l < roll_hi:
                lo, hi = l, roll_hi
                if not state["pref"]:
                    stream = emit(lo, stream, "fp4", next_layer=lo + 1 < L)
                    lo += 1
                if hi > lo:
                    if hi < L:
                        stream = jax.lax.fori_loop(lo, hi, rolled, stream)
                        state["pref"] = True
                    else:
                        if hi - 1 > lo:
                            stream = jax.lax.fori_loop(lo, hi - 1, rolled, stream)
                        state["pref"] = True
                        stream = emit(hi - 1, stream, "fp4", next_layer=False)
            elif roll_lo is None or not (roll_lo <= l < roll_hi):
                stream = emit(l, stream, c.expert_kinds[l], next_layer=l + 1 < L)
        # ---------------- final norm (on the bf16-cast stream, as the native path)
        PH_final = pltpu.make_async_copy(
            W["final_norm"], S["final_norm"], msem.at[R + 2]
        )
        PH_final.start()
        PH_final.wait()
        hidden = _rms(
            stream.astype(jnp.bfloat16).astype(jnp.float32),
            c.rms_eps,
            S["final_norm"][0:1, :],
        )
        hid_out[...] = jnp.where(row_live, hidden, 0.0).astype(jnp.bfloat16)
        if ncap == 0:
            aux_out[...] = jnp.zeros(aux_out.shape, aux_out.dtype)

    return kernel, scratch


def megakernel_call(cfg: MKConfig, axis: str = "model"):
    """``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv) -> (hidden, aux, newkv)``
    for one device (call inside ``shard_map``)."""
    kernel, scratch = build_kernel(cfg, axis)
    L = cfg.n_layers
    nw = len(WEIGHT_NAMES)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    ncap = len(cfg.capture_layers)
    aux_w = ncap * cfg.hidden if ncap else LANE

    def call(meta, pos, pages_full, pages_swa, emb, *rest):
        weights = rest[:nw]
        experts = rest[nw : nw + 4 * L]
        kv = rest[nw + 4 * L :]
        assert len(kv) == L
        out_shape = [
            jax.ShapeDtypeStruct((cfg.rows, cfg.hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((cfg.rows, aux_w), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                newkv_shape(cfg), jnp.uint32 if cfg.kv_sal else jnp.bfloat16
            ),
        ]
        # The pools are read-only here. With any remote DMA in the kernel XLA treats the call
        # as a collective and copies every aliased operand: the in-place write is kv_write_call.
        f = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=[smem] * 4
            + [vmem]
            + [hbm] * (nw + 4 * L + L),  # emb: bf16 rows, or the f32 stream
            out_specs=[vmem, vmem, hbm],
            scratch_shapes=[s for _, s in scratch],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=cfg.vmem_limit_bytes,
                collective_id=_COLLECTIVE_ID if cfg.tp > 1 else None,
            ),
            name=f"muse_megakernel_r{cfg.rows}_q{cfg.rows_per_req}",
        )
        return f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)

    return call


def newkv_shape(cfg: MKConfig):
    if cfg.kv_sal:  # new token columns of a page block, [slot * words + word, row]
        return (cfg.n_layers, cfg.sal_words, cfg.rows)
    return (cfg.n_layers, cfg.rows, *cfg.kv_block)


def _kv_write_sal_pipelined_call(cfg: MKConfig):
    """``_kv_write_sal_call`` with the DMAs overlapped: requests' page RMWs per layer in flight
    together, layer l's writes behind layer l + 1's reads."""
    L, R, ps, q, J = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req, cfg.n_req
    gl = set(cfg.global_layers)
    nw = cfg.sal_words
    S = 2 * J  # page slots per layer: a request's first and last page

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        blk, new_v, nsem, rsem, wsem = rest[2 * L : 2 * L + 5]
        n_live = meta_ref[0]
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, ps), 1)
        load = pltpu.make_async_copy(newkv, new_v, nsem.at[0])
        load.start()

        def slots(l):
            """(slot, request, side, active, page) of layer l's page RMWs."""
            tab = pf_ref if l in gl else psw_ref
            out = []
            for j in range(J):
                first, last = pos_ref[j * q] // ps, pos_ref[j * q + q - 1] // ps
                for side in range(2):
                    pg_no = first if side == 0 else last
                    active = (j * q < n_live) & ((side == 0) | (last != first))
                    ent = jnp.minimum(
                        j * cfg.pages_per_seq + pg_no, cfg.n_pages_entries - 1
                    )
                    out.append(
                        (
                            2 * j + side,
                            j,
                            pg_no,
                            active,
                            _page(tab, ent, cfg.packed_pages),
                        )
                    )
            return out

        def write(l, s, page):
            return pltpu.make_async_copy(
                blk.at[l % 2, s], kv_out[l].at[page], wsem.at[l % 2, s]
            )

        load.wait()
        for l in range(L):
            cur = slots(l)
            if l >= 2:  # the bank's slots are free once layer l - 2's writes land
                for s, _, _, active, page in slots(l - 2):
                    pl.when(active)(write(l - 2, s, page).wait)
            reads = [
                pltpu.make_async_copy(
                    kv_out[l].at[page], blk.at[l % 2, s], rsem.at[l % 2, s]
                )
                for s, _, _, _, page in cur
            ]
            for (s, _, _, active, _), rd in zip(cur, reads):
                pl.when(active)(rd.start)
            nv = new_v[l]
            for (s, j, pg_no, active, page), rd in zip(cur, reads):

                @pl.when(active)
                def _(s=s, j=j, pg_no=pg_no, page=page, rd=rd):
                    rd.wait()
                    words = blk.at[l % 2, s].bitcast(jnp.uint32).reshape(nw, ps)
                    w = words[...]
                    for t in range(q):
                        r = j * q + t
                        p = pos_ref[r]
                        hit = (lane == p % ps) & (p // ps == pg_no) & (r < n_live)
                        w = jnp.where(hit, nv[:, r : r + 1], w)
                    words[...] = w
                    write(l, s, page).start()

        for l in range(max(0, L - 2), L):
            for s, _, _, active, page in slots(l):
                pl.when(active)(write(l, s, page).wait)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[
                pltpu.VMEM((2, S, *cfg.kv_block), cfg.kv_dtype),
                pltpu.VMEM((L, nw, R), jnp.uint32),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((2, S)),
                pltpu.SemaphoreType.DMA((2, S)),
            ],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(),
            name=f"muse_kv_write_salp_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call


def _kv_write_sal_call(cfg: MKConfig):
    """SEQ_ALONG_LANE pools: a token is one lane column of its page block, so each live request
    read-modify-writes the (at most two) pages its consecutive rows land in, per layer."""
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)
    nw = cfg.sal_words

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        blk, new_v, sems = rest[2 * L], rest[2 * L + 1], rest[2 * L + 2]
        n_live = meta_ref[0]
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, ps), 1)
        for j in range(cfg.n_req):

            @pl.when(j * q < n_live)
            def _(j=j):
                first, last = pos_ref[j * q] // ps, pos_ref[j * q + q - 1] // ps
                for l in range(L):
                    tab = pf_ref if l in gl else psw_ref
                    cp = pltpu.make_async_copy(newkv.at[l], new_v, sems.at[2])
                    cp.start()
                    cp.wait()
                    for side in range(2):
                        pg_no = first if side == 0 else last

                        @pl.when((side == 0) | (last != first))
                        def _(pg_no=pg_no):
                            ent = jnp.minimum(
                                j * cfg.pages_per_seq + pg_no, cfg.n_pages_entries - 1
                            )
                            page = _page(tab, ent, cfg.packed_pages)
                            rd = pltpu.make_async_copy(
                                kv_out[l].at[page], blk, sems.at[0]
                            )
                            rd.start()
                            rd.wait()
                            words = blk.bitcast(jnp.uint32).reshape(nw, ps)
                            w = words[...]
                            nv = new_v[...]
                            for t in range(q):
                                r = j * q + t
                                p = pos_ref[r]
                                hit = (
                                    (lane == p % ps) & (p // ps == pg_no) & (r < n_live)
                                )
                                w = jnp.where(hit, nv[:, r : r + 1], w)
                            words[...] = w
                            wr = pltpu.make_async_copy(
                                blk, kv_out[l].at[page], sems.at[1]
                            )
                            wr.start()
                            wr.wait()

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[
                pltpu.VMEM(cfg.kv_block, cfg.kv_dtype),
                pltpu.VMEM((nw, R), jnp.uint32),
                pltpu.SemaphoreType.DMA((3,)),
            ],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(),
            name=f"muse_kv_write_sal_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call


def kv_write_call(cfg: MKConfig):
    """``f(meta, pos, pages_full, pages_swa, newkv, *kv) -> kv``: scatter this step's K/V rows
    into every layer's pool in place (no remote DMA, so XLA aliases the pools without a copy)."""
    if cfg.kv_sal and cfg.kv_write_pipelined:
        return _kv_write_sal_pipelined_call(cfg)
    if cfg.kv_sal:
        return _kv_write_sal_call(cfg)
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        sems = rest[2 * L]
        n_live = meta_ref[0]
        cps = []
        for l in range(L):
            tab = pf_ref if l in gl else psw_ref
            flat = kv_out[l].reshape(kv_out[l].shape[0] * ps, *kv_out[l].shape[2:])
            for r in range(R):
                p = pos_ref[r]
                ent = jnp.minimum(
                    (r // q) * cfg.pages_per_seq + p // ps, cfg.n_pages_entries - 1
                )
                tok = _page(tab, ent, cfg.packed_pages) * ps + p % ps
                cp = pltpu.make_async_copy(
                    newkv.at[l, pl.ds(r, 1)], flat.at[pl.ds(tok, 1)], sems.at[l * R + r]
                )

                @pl.when(r < n_live)
                def _(cp=cp):
                    cp.start()

                cps.append((r, cp))
        for r, cp in cps:

            @pl.when(r < n_live)
            def _(cp=cp):
                cp.wait()

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[pltpu.SemaphoreType.DMA((L * R,))],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(),
            name=f"muse_kv_write_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call
