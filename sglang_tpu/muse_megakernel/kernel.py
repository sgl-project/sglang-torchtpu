# SPDX-License-Identifier: Apache-2.0
"""Whole-decoder Pallas megakernel for Muse Spark decode and DFlash verify on TPU.

One grid-less ``pallas_call`` per device per step takes the embedded rows and
runs all
decoder layers (paged attention, router, experts with the post-expert norm,
pooler/unpooler) and the final norm. Every tensor-parallel reduction is an
in-kernel
one-hop remote DMA. The LM head, logits and sampling stay in sglang.

It reads the weights sglang already loaded, in their serving layouts:

* experts: the fused-RS expert-TP layout (``w13 [E, H_e, 2*I_local]`` fp4 or
fp8,
    ``w2 [E, I_local, H_e]``); per-layer refs, addressed by a branch tree on the
    layer
    index so the layer loop can stay rolled;
* KV: the paged pools of the hybrid SWA allocator, one ref per layer, read in
place: fused
    ``[pages, page_size, kv_heads_x2/2, 2, 128]`` bf16, or SEQ_ALONG_LANE
    ``[pages, 2 * kv_heads, head_dim // pk, pk, page_size]`` bf16 / fp8 (tokens on
    lanes);
* attention / pooler / unpooler / router: stacked K-major copies made at load
time
    (``ops.MuseMegakernel.from_model``), because the matvec streams ``[K, N]``
    tiles faster
    than the checkpoint's ``[N, K]``; fp8 with per-output-channel scales, except
    the router.

Rows are ``num_requests * rows_per_request`` tokens padded to a multiple of the
TP size:
one per request at decode, a DFlash verify block per request at target-verify.
Each row
carries its own position. The step's own K/V rows join attention from VMEM,
causal within
a verify block, and come back as ``newkv`` for ``ops`` to scatter into the
pools.

The expert phase follows the fused-RS expert-TP contract: each rank evaluates
its
intermediate slice of every selected expert for all rows, the partial outputs
are
reduce-scattered by token, summed in f32, rounded to bf16, normalised by the
post-expert
norm, weighted and summed. Collective rounds per layer: o_proj reduce-scatter
and
all-gather, pooler all-gather, expert reduce-scatter, combined-output
all-gather, unpooler
all-gather.
"""

import contextlib
import functools
import math
import os
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
# MUSE_MK_TRACE=1: named regions as on-device trace spans (2: inner regions too).
_TRACE = int(os.environ.get("MUSE_MK_TRACE", "0")) >= 1


# per-chunk attention regions: thousands per step, they distort the timing they show
_TRACE_INNER = int(os.environ.get("MUSE_MK_TRACE", "0")) >= 2
_INNER = frozenset(
    (
        "kv_unpack",
        "qk",
        "softmax",
        "pv",
        "kv_wait",
        "ring_wait",
        "ex_wait",
        "full_expert",
        "ex_gather",
        "bd_w13",
        "bd_w2",
        "ex_scatter",
        "ex_accum",
    )
)


def _scope(name):
    on = _TRACE and (_TRACE_INNER or name not in _INNER)
    return jax.named_scope(name) if on else contextlib.nullcontext()


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
# attn_preones: u32 rows per ring chunk (64 of K/V words + 8 of fp8 ones) and slot cap
PREONES_ROWS = 72
PREONES_MAX_SLOTS = 24
# four e4m3 1.0 bytes in one u32 word
_FP8_ONES_WORD = 0x38383838


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
    # K chunk of the block-diagonal dot; 128-256 beat 512 at 2-8 rows
    blockdiag_kc: int = 256
    # explicit compact row buckets (ascending, e.g. (1, 2, 3, 4)); empty: powers of two up to
    # compact_rows. The MXU streams 16 rows per bucket row, so a tighter bucket is cheaper.
    compact_arms: tuple = ()
    # router top-k by k rounds of row max (_topk_iter) instead of per-row ranks
    topk_iter: bool = False
    # an expert waits for its w13 slabs only, then for w2 just before the down projection
    ex_split_wait: bool = True
    # fp4 experts: every other iteration issues the next two experts in one layer-branch walk
    # (~0.03 us per expert less). Its looped leaves grow the expert loop's code: with per-layer
    # expert operands (62 leaves) the loop no longer fits the instruction memory.
    ex_issue_pairs: bool = False
    # fp4 experts carry e4m3 block scales plus per-expert globals (smalls rows expert_gs*)
    # instead of bf16 scales with the globals folded in (10% fewer expert bytes)
    scale_fp8: bool = False
    # in a step of at most few_rows_max rows (n_req * rows_per_req), the o-proj all-reduce is one
    # gather of those rows' partials (one latency phase instead of two) and the ffn-output
    # gather ships only those rows
    single_row_coll: bool = True
    # one gather ships every live row to every peer: it trades bytes for a latency phase
    few_rows_max: int = 2
    fp4_chunk_words: int = 128
    chunk_pages: int = 2
    # the KV ring holds 2 * kv_group chunks; each consumed pair of slots is refilled kv_ring
    # chunks ahead
    kv_group: int = 4
    # Pool rows hold every local KV head side by side in one pool_padded_head_dim-lane row
    # (kv_heads * head_dim == pool_padded_head_dim) instead of one zero-padded row per head:
    # half the KV bytes and attention work at head_dim 64. The serving pool must match.
    kv_packed: bool = False
    # every layer in one rolled loop (expert kind picked per layer by a branch) instead of
    # emitting the non-fp4 layers statically: one copy of the layer body
    unify_layers: bool = False
    # [lo, hi) fp4 layer ranges whose expert slabs arrive stacked ([hi - lo, E, ...] per slab),
    # so the rolled loop picks a range by a small compare tree and indexes the layer in it
    expert_groups: tuple = ()
    # distinct experts, their (row, slot) groups and the y-block ends from vector ops and one
    # DMA to SMEM instead of scalar SMEM loops
    vec_route: bool = True
    # Attention without a running max: q and k are RMS-normalized per head, so |score| <=
    # scale_query_by * head_dim * attn_scale (~44 for Muse) and p = exp(score) with its row sums
    # stays far inside f32 range: no max, shift or rescale. The queries carry attn_scale, exact
    # when it is a power of two (validate()).
    attn_unshifted: bool = False
    # With attn_unshifted: the queries carry attn_scale * log2(e), rounded to bf16 once from f32
    # (a different rounding of q than bf16(q) * attn_scale, same size), so p = exp2(score)
    # without a multiply per score.
    attn_exp2: bool = False
    # SEQ_ALONG_LANE pools (TpuSeqAlongLaneKVPool): a page block is [2 * kv_heads, head_dim // pk,
    # pk, page_size], slots [K_h0, V_h0, K_h1, ...], tokens on lanes, pk head dims per 32-bit word.
    # Attention reads K and V as [kv_dim, tokens]; kv_write_call merges new token columns into pages.
    kv_sal: bool = False
    kv_sal_fp8: bool = False
    # fp8 SAL: 1 feeds the cached K to the MXU as fp8 (a mixed bf16 x fp8 dot, exact), 2 also V
    sal_fp8_dot: int = 0
    # Global-layer clean chunks run software-pipelined: one loop iteration scores attn_streams
    # chunks (QK, exp) while it multiplies the previous attn_streams chunks' probabilities by V,
    # which go through VMEM; the ring (kv_group >= attn_streams) is refilled one iteration ahead.
    attn_pipe: bool = False
    attn_streams: int = 4
    # debug: attention reads this step's own K/V rows at bf16 precision whatever the pool dtype
    debug_own_kv_bf16: bool = False
    # ship each finished wo column tile to its reduce-scatter owner during the o-proj matvec
    o_rs_streamed: bool = True
    # with o_rs_streamed: each device streams the wo column tiles from its next peer's block on,
    # its own block last, so the last remote block leaves one tile before the matvec ends
    o_rs_rotate: bool = True
    # experts run owner block by owner block (own rows last); a block's expert partials leave
    # for their reduce-scatter owner once its last expert is done (each (row, slot) is still
    # written by one expert: same values)
    y_block_send: bool = True
    # peer order of a round's sends: "ring" (me+1, me+2, ...) or "xor" (me^1, me^2, ...; ring
    # unless tp is a power of two)
    coll_order: str = "xor"
    # a round starts its remote copies before copying its own part into its pad
    coll_local_last: bool = True
    # Not bit-exact. The unpooler all-gather folds into the next layer's qkvg matvec: the
    # attention-input RMS is a per-row scalar, so qkvg = r * ((alpha x + s_f beta f wp) g) @ W,
    # the x and f parts one stacked dot per K block, each f block consumed as it arrives (own
    # first). The residual stream stays exact; only the qkvg LHS rounding moves.
    fold_f: bool = False
    # Not bit-exact. The o-proj all-gather folds into the pooler and router matvecs the same way.
    fold_o: bool = False
    # LAZY dense refill: refills of tiles consumed after the expert phase (the unpooler's and
    # the next layer's), due while pool / router stream, are held until the expert loop ends
    # so they do not compete with the expert DMAs.
    lazy_dense: bool = True
    # the next layer's smalls are fetched right after this layer's attention instead of after
    # its unpooler (where they queue behind the LAZY refills; fold_f leaves no f wait to hide them)
    smalls_early: bool = True
    # > 0: row-tiled program for steps past the base program's VMEM (ported from
    # sglang-torchtpu-mkrows): the residual stream, collective pads and expert outputs live in
    # HBM and every per-row pass runs row_tile rows at a time, so VMEM barely grows with rows.
    row_tile: int = 0
    # row-tiled: rows per dense matmul pass (every pass re-pushes the weight tile to the MXU)
    mv_rows: int = 256
    # row-tiled: rows per pass of a busy expert (each pass dequantizes its slabs again)
    expert_chunk: int = 32
    # optimization flags (all default off on MKConfig; enabled by decode_config)
    ex_opt: bool = False
    ex_kperm: int = 0
    tiled_vmem_y: int = 0
    tiled_o_rot: bool = False
    tiled_vmem_pads: bool = False
    tiled_fold: bool = False
    attn_opt: int = 0
    # fp8 SAL global attention on steps of more than 64 attention rows per request: each ring
    # chunk owns 72 u32 rows of the expert w13 slots ([K0; V0; K1; V1] then 8 rows of fp8 ones,
    # written once per layer), so QK is one 256-deep dot on the raw chunk and PV one dot
    # against [V0; K1; V1; ones] whose ones lanes give the softmax row sums on the MXU (one
    # f32 accumulator, no VPU row-sum adds). The HBM pool layout is unchanged.
    attn_preones: bool = False
    # with attn_preones: queries go to the MXU as per-row-scaled fp8
    attn_q_fp8: bool = False
    # pipelined_opt loop carry for > 64 attention rows: 0 one lane-partial row sum per stream
    # (NS x [M, 128]) and 2 accumulators; 1 one row sum (tree-summed per chunk) and one
    # accumulator; 2 one row sum and 2 accumulators
    attn_carry: int = 0
    alias_oy: bool = True
    alias_p: bool = True
    trim_sems: bool = True
    tiled_pair_y: int = 0
    pair_ag: int = 0
    pair_o_rs: bool = False
    swa_chunk_pages: int = 0
    attn_skip_idle_streams: bool = False
    attn_col_mask: bool = False
    fold_attn_scale: bool = False
    expert_split: int = 0
    xfull_packed: bool = False
    ex_via_fp8: bool = False
    ex_bd_all: bool = False
    alias_tiled_temps: bool = False
    y_quant: int = 0
    combine_loop: bool = False
    untiled_max_rows: int = 128
    bd_fp8_lhs: bool = False
    bd_n_tile: int = 0
    vec_dequant_kperm: bool = False
    ex_fp8_lhs: bool = False
    ex_bd_max_rows: int = 0
    dense_fp8_lhs: bool = False
    attn_p_fp8: bool = False
    topk_xpose: bool = False
    ex_w2_stage: bool = False
    ex_gs_smem: bool = False
    ex_lhs_fp8: int = 0
    pair_y_stream: bool = False
    sal_kv_split: bool = False
    attn_head_fp8: int = 0
    alias_oag_f: bool = False
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
        return 1 if self.packed_attn else self.kv_heads

    @property
    def packed_attn(self):
        # attention sees one key row per token with every local head side by side
        return self.kv_packed or self.kv_sal

    @property
    def attn_width(self):
        return self.kv_dim if self.packed_attn else self.pool_padded_head_dim

    @property
    def kv_dtype(self):
        return jnp.float8_e4m3fn if self.kv_sal_fp8 else jnp.bfloat16

    @property
    def sal_pack(self):
        return 4 if self.kv_sal_fp8 else 2

    @property
    def sal_words(self):
        # 32-bit word rows of one SAL page block: (slot, word) pairs
        return 2 * self.kv_heads * self.head_dim // self.sal_pack

    @property
    def sal_block(self):
        return (
            2 * self.kv_heads,
            self.head_dim // self.sal_pack,
            self.sal_pack,
            self.page_size,
        )

    @property
    def preones_slots(self):
        """attn_preones ring chunks: 72-row slots of the expert w13 slots, at most 24."""
        rows = (self.expert_depth + 1) * (self.expert_hidden // FP4_PACK)
        return min(PREONES_MAX_SLOTS, rows // PREONES_ROWS)

    @property
    def use_sal_preones(self):
        return bool(
            self.attn_preones
            and self.kv_sal
            and self.kv_sal_fp8
            and self.sal_fp8_dot >= 2
            and self.attn_exp2
            and self.attn_pipe
            and self.kv_heads == 2
            and self.head_dim == 64
            and self.sal_words == 64
            and self.chunk_pages * self.page_size == 2 * self.inter_local
            and self.preones_slots >= 2 * self.attn_streams
        )

    @property
    def kv_ring(self):
        if self.use_sal_preones:
            return self.preones_slots
        if self.kv_sal:  # every chunk the expert w13 slots hold (sal_kv_view)
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
    def split_rows(self):
        """expert_split's bound on the few-row loop's experts: no larger than its arms."""
        return min(self.expert_split, self.compact_rows)

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
        assert self.expert_split in (0, 1, 2, 4)
        assert not self.expert_split or self.vec_route
        arms = self.compact_arms
        assert all(1 <= a < b for a, b in zip(arms, arms[1:] + (1 << 30,)))
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
        assert not self.kv_sal_fp8 or self.kv_sal
        # the SAL ring is a view of the expert w13 slots: [chunk, sal_words, chunk tokens]
        assert not self.kv_sal or (
            self.chunk_pages * self.page_size == 2 * self.inter_local
            and (self.expert_depth + 1)
            * (self.expert_hidden // FP4_PACK)
            % self.sal_words
            == 0
        )
        assert self.sal_fp8_dot == 0 or self.kv_sal_fp8
        assert not self.swa_chunk_pages or (
            self.kv_sal
            and not self.use_sal_preones
            and 0 < self.swa_chunk_pages < self.chunk_pages
            and self.chunk_pages % self.swa_chunk_pages == 0
            and (self.swa_chunk_pages * self.page_size) % LANE == 0
        )
        assert not self.attn_col_mask or self.swa_chunk_pages
        assert not self.fold_attn_scale or math.frexp(self.attn_scale)[0] == 0.5
        # 1.02: bf16 rounding of q and k (1.07: fp8 k); exp(60) * 1M keys * |v| stays under f32 max
        assert not self.attn_unshifted or (
            (1.07 if self.kv_sal_fp8 else 1.02)
            * self.scale_query_by
            * self.head_dim
            * self.attn_scale
            <= UNSHIFTED_MAX
            and (self.attn_exp2 or math.frexp(self.attn_scale)[0] == 0.5)
        )
        assert not self.attn_exp2 or self.attn_unshifted
        assert not self.attn_pipe or (
            self.attn_unshifted
            and self.packed_attn
            and self.attn_streams % 2 == 0
            and self.kv_ring >= 2 * self.attn_streams
        )
        assert self.coll_order in ("ring", "xor")
        if self.fold_f or self.fold_o:
            # arrival-ordered K groups need xor peers; few-row steps gather rows, not blocks
            assert (
                self.tp > 1
                and self.tp & (self.tp - 1) == 0
                and self.coll_order == "xor"
            )
            assert not self.unify_layers, "layer 0 has no pending all-gather to fold"
            few = (
                self.single_row_coll
                and self.n_req * self.rows_per_req <= self.few_rows_max
            )
            assert not few
            Hl = self.hidden_local
            for grp in fold_groups(self):
                tiles = [t for t in dense_tiles(self) if t[0] in grp]
                assert all(
                    t[3] % Hl == 0 and (t[3] // Hl) & (t[3] // Hl - 1) == 0
                    for t in tiles
                )
            assert fold_fits(self)
        # sliding-window layers never refill the KV ring: a window's chunks must fit it
        swa_cp = self.swa_chunk_pages or self.chunk_pages
        swa_cpt = swa_cp * self.page_size
        swa_kr = 2 * self.swa_kv_group if self.swa_chunk_pages else self.kv_ring
        assert (self.window + swa_cpt - 2) // swa_cpt + 1 <= swa_kr
        assert not self.row_tile or (
            self.rows % self.row_tile == 0
            and self.row_tile % 16 == 0
            and self.rows % self.tp == 0
            and owner_window_rows(self) <= self.rows
            and not (
                ((self.fold_f or self.fold_o) and not self.tiled_vmem_pads)
                or self.scale_fp8
                or self.unify_layers
            )
            and not (
                self.single_row_coll
                and self.n_req * self.rows_per_req <= self.few_rows_max
            )
            and self.expert_chunk % 8 == 0
            and self.expert_chunk >= self.compact_rows
            and self.vec_route
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


def _topk_by_rank(scores, k, n_live, rank_ref):
    """Top-k by rank: ``rank[r, e]`` = number of experts beating ``e`` (higher score, ties to

    the lower id), so pick ``j`` is the expert of rank ``j`` -- the same picks as
    ``k`` argmax
    rounds, from one ``[E, E]`` comparison per live row instead of a dependent
    chain. Rows past
    ``n_live`` keep stale ranks.
    """
    R, E = scores.shape
    s_t = scores.T  # [E, R]: e' on sublanes
    lower = jax.lax.broadcasted_iota(jnp.int32, (E, E), 0) < jax.lax.broadcasted_iota(
        jnp.int32, (E, E), 1
    )
    for r in range(R):

        @pl.when(r < n_live)
        def _(r=r):
            col, row = s_t[:, r : r + 1], scores[r : r + 1, :]
            beats = (col > row) | ((col == row) & lower)
            rank_ref[r : r + 1, :] = jnp.sum(
                beats.astype(jnp.int32), axis=0, keepdims=True
            )

    rank = rank_ref[...]
    idx = jax.lax.broadcasted_iota(jnp.int32, (R, E), 1)
    picks = [rank == j for j in range(k)]
    return [jnp.sum(jnp.where(p, idx, 0), axis=1, keepdims=True) for p in picks], picks


def _topk_iter(scores, k):
    """Top-k as k rounds of a row max and its lowest index: the picks of _topk_by_rank (higher

    score first, ties to the lower id), a cost flat in the rows where the ranks
    pay one [E, E]
    comparison per row (faster from 16 rows, slower at 8).
    """
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


def _topk_xpose(sc, bias, k, router_eps, route_scale):
    """Transposed batched top-k for up to LANE rows (zero cross-lane XLU reductions).

    ``sc`` is ``[br, E]`` f32 (``1 <= br <= LANE``, ``E % LANE == 0``), ``bias`` is
    ``[1, E]`` f32. Transposes to ``[E, LANE]`` once so all ``k`` max / tie-break /
    gather rounds reduce vertically along sublanes, then transposes ``[2*k, LANE]``
    back in a single 128x128 XLU op.
    """
    br, E = sc.shape
    assert 1 <= br <= LANE and E % LANE == 0
    sc_pad = (
        sc
        if br == LANE
        else jnp.concatenate([sc, jnp.zeros((LANE - br, E), jnp.float32)], axis=0)
    )
    s_t = jnp.transpose(sc_pad + bias)  # [E, LANE]
    idx_t = jax.lax.broadcasted_iota(jnp.int32, (E, LANE), 0)
    ids_rows = []
    for _ in range(k):
        m = jnp.max(s_t, axis=0, keepdims=True)
        sel = jnp.min(jnp.where(s_t == m, idx_t, E), axis=0, keepdims=True)
        ids_rows.append(sel)
        s_t = jnp.where(idx_t == sel, -jnp.inf, s_t)
    sc_t = jnp.transpose(sc_pad)  # [E, LANE]
    wsel_rows = [
        jnp.sum(jnp.where(idx_t == sel, sc_t, 0.0), axis=0, keepdims=True)
        for sel in ids_rows
    ]
    wsel_t = jnp.concatenate(wsel_rows, axis=0)  # [k, LANE]
    wsel_t = (
        wsel_t / (jnp.sum(wsel_t, axis=0, keepdims=True) + router_eps) * route_scale
    )
    ids_full = jnp.transpose(
        jnp.concatenate(
            [
                jnp.concatenate(ids_rows, axis=0),
                jnp.zeros((LANE - k, LANE), jnp.int32),
            ],
            axis=0,
        )
    )  # [LANE, LANE] int32
    wsel_full = jnp.transpose(
        jnp.concatenate([wsel_t, jnp.zeros((LANE - k, LANE), jnp.float32)], axis=0)
    )  # [LANE, LANE] f32
    return ids_full[:br], wsel_full[:br]


def _pack_xfull_256(x):
    """Pack f32 ``[M, He]`` into u32 ``[M, He // 2]`` in 256-col (128-word) blocks."""
    nb = x.shape[-1] // (2 * LANE)
    lo = jnp.concatenate(
        [x[..., b * 2 * LANE : (2 * b + 1) * LANE] for b in range(nb)], axis=-1
    )
    hi = jnp.concatenate(
        [x[..., (2 * b + 1) * LANE : (2 * b + 2) * LANE] for b in range(nb)],
        axis=-1,
    )
    return _bf16_bits(lo) | (_bf16_bits(hi) << 16)


def _unpack_xfull_256_chunk(w, out_fp8: bool = False):
    """Unpack ``n_w`` (multiple of LANE) u32 cols packed by ``_pack_xfull_256``."""
    nb = w.shape[-1] // LANE
    parts = [_unpack_bf16x2(w[..., b * LANE : (b + 1) * LANE]) for b in range(nb)]
    res = parts[0] if nb == 1 else jnp.concatenate(parts, axis=-1)
    return res.astype(jnp.float8_e4m3fn) if out_fp8 else res.astype(jnp.bfloat16)


def _bf16_bits(x):
    """f32 -> its round-to-nearest-even bf16 bit pattern in the low 16 bits of a u32."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint32)
    return (b + jnp.uint32(0x7FFF) + ((b >> 16) & jnp.uint32(1))) >> 16


def _pack_bf16x2(x):
    """f32 ``[..., 2n]`` -> u32 ``[..., n]``: bf16 of the first half low, of the second half high.

    Halves a collective payload without a 16-row bf16 tile on 8-row buffers.
    """
    n = x.shape[-1] // 2
    return _bf16_bits(x[..., :n]) | (_bf16_bits(x[..., n:]) << 16)


def _unpack_bf16x2(w):
    lo = jax.lax.bitcast_convert_type(w << 16, jnp.float32)
    hi = jax.lax.bitcast_convert_type(w & jnp.uint32(0xFFFF0000), jnp.float32)
    return jnp.concatenate([lo, hi], axis=-1)


def y_words(c: MKConfig) -> int:
    """u32 words per (row, slot) expert partial in ysend / pad_y."""
    return (c.expert_hidden // 4 + LANE) if c.y_quant else (c.expert_hidden // 2)


def _row_amax(ax):
    """Row-wise max of non-negative 2D tensor [M, K] using 128-lane VPU tree before 1 XLU max."""
    if ax.ndim == 2 and ax.shape[1] > LANE and ax.shape[1] % LANE == 0:
        n_tiles = ax.shape[1] // LANE
        tiles = [ax[:, k * LANE : (k + 1) * LANE] for k in range(n_tiles)]
        while len(tiles) > 1:
            tiles = [
                jnp.maximum(tiles[2 * i], tiles[2 * i + 1])
                if 2 * i + 1 < len(tiles)
                else tiles[2 * i]
                for i in range((len(tiles) + 1) // 2)
            ]
        return jnp.max(tiles[0], axis=1, keepdims=True)
    return jnp.max(ax, axis=1, keepdims=True)


def _scale_fp8_range(x, nonneg: bool = False):
    """Per-row scale of x [M, K] into [-448, 448] f32 without fp8 cast -> (x_scaled, x_sc [M, 1])."""
    x32 = x if x.dtype == jnp.float32 else x.astype(jnp.float32)
    amax = _row_amax(x32 if nonneg else jnp.abs(x32))
    sc = amax * jnp.float32(1.0 / 448.0)
    # Route sc through select so libtpu does not rematerialize amax / 448 at every consumer.
    sc = jnp.where(amax == 0, jnp.zeros_like(sc), sc)
    inv = jnp.where(
        amax == 0,
        jnp.zeros_like(amax),
        jnp.float32(448.0) / jnp.maximum(amax, jnp.float32(1e-30)),
    )
    return x32 * inv, sc


def _quant_fp8_rows(x, nonneg: bool = False):
    """Per-row dynamic float8_e4m3fn quantization of x [M, K] -> (x_fp8, x_sc [M, 1] f32)."""
    x_scaled, sc = _scale_fp8_range(x, nonneg=nonneg)
    return x_scaled.astype(jnp.float8_e4m3fn), sc


def _pack_int8_scaled(x):
    """f32/bf16 [..., 4n] -> u32 [..., n + LANE]: 4 signed int8s per u32 + per-lane (block-32) f32 scales."""
    n = x.shape[-1] // 4
    n_lanes = (4 * n) // LANE
    amax = jnp.abs(x[..., :LANE]).astype(jnp.float32)
    for k in range(1, n_lanes):
        amax = jnp.maximum(
            amax, jnp.abs(x[..., k * LANE : (k + 1) * LANE]).astype(jnp.float32)
        )
    sc = amax * jnp.float32(1.0 / 127.0)
    sc = jnp.where(amax == 0, jnp.zeros_like(sc), sc)
    inv = jnp.where(
        amax == 0,
        jnp.zeros_like(amax),
        jnp.float32(127.0) / jnp.maximum(amax, jnp.float32(1e-30)),
    )
    inv_n = inv if n == LANE else jnp.concatenate([inv] * (n // LANE), axis=-1)
    words = None
    for b in range(4):
        xb = x[..., b * n : (b + 1) * n]
        xb32 = xb if xb.dtype == jnp.float32 else xb.astype(jnp.float32)
        qb = jnp.clip(jnp.rint(xb32 * inv_n), -127.0, 127.0).astype(jnp.int32)
        ub = jax.lax.bitcast_convert_type(qb, jnp.uint32) & jnp.uint32(0xFF)
        words = ub if b == 0 else (words | (ub << (8 * b)))
    sc_u32 = jax.lax.bitcast_convert_type(sc, jnp.uint32)
    return jnp.concatenate([words, sc_u32], axis=-1)


def _unpack_int8_scaled(w):
    """u32 [..., n + LANE] -> f32 [..., 4n] using lane-replicated scale (zero XLU ops)."""
    n = w.shape[-1] - LANE
    words = jax.lax.bitcast_convert_type(w[..., :n], jnp.int32)
    sc_tile = jax.lax.bitcast_convert_type(w[..., n : n + LANE], jnp.float32)
    sc_n = sc_tile if n == LANE else jnp.concatenate([sc_tile] * (n // LANE), axis=-1)
    q0 = ((words << 24) >> 24).astype(jnp.float32) * sc_n
    q1 = ((words << 16) >> 24).astype(jnp.float32) * sc_n
    q2 = ((words << 8) >> 24).astype(jnp.float32) * sc_n
    q3 = (words >> 24).astype(jnp.float32) * sc_n
    return jnp.concatenate([q0, q1, q2, q3], axis=-1)


def _pack_y(x, c: MKConfig):
    if c.y_quant:
        return _pack_int8_scaled(x)
    return _pack_bf16x2(x if x.dtype == jnp.float32 else x.astype(jnp.float32))


def _unpack_y(w, c: MKConfig):
    return _unpack_int8_scaled(w) if c.y_quant else _unpack_bf16x2(w)


def _branch(idx, fns, lo=0):
    """Run ``fns[idx - lo]`` for a traced ``idx`` through a balanced tree of two-way branches.

    ``lax.switch`` over ~60 layers costs time linear in the branch count; a tree
    costs log2(n)
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


def _select(idx, keys, fn):
    """Run ``fn(k)`` for the ``k`` in sorted ``keys`` equal to ``idx`` (a traced index must be

    one of them) through a balanced compare tree.
    """
    if isinstance(idx, int):
        fn(idx)
        return
    if len(keys) == 1:
        fn(keys[0])
        return
    mid = len(keys) // 2
    jax.lax.cond(
        idx < keys[mid],
        lambda: _select(idx, keys[:mid], fn),
        lambda: _select(idx, keys[mid:], fn),
    )


def _dispatch(idx, refs, fn, only=None):
    """Run ``fn(ref)`` on ``refs[idx]``; a traced index goes through ``_branch``.

    ``only``: the sorted indices ``idx`` can take, so the tree has a leaf per
    index rather than per ref.
    """
    if isinstance(idx, int):
        fn(refs[idx])
        return
    if only is None:
        _branch(idx, [functools.partial(fn, r) for r in refs])
        return
    _select(idx, list(only), lambda k: fn(refs[k]))


class _AttnGeom(NamedTuple):
    """Chunk geometry of one attention body: ``cp`` pages per chunk, ``kr`` ring slots fetched in

    groups of ``kg``. ``is_global`` is a Python bool when the body is specialized
    per layer
    type, else the layer's traced flag.
    """

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


def perm_256(t, axis: int):
    """Permute each 256-slice along ``axis`` from (16, 16) to transposed (16, 16)."""
    axis = axis % t.ndim
    d = t.shape[axis]
    assert d % 256 == 0, f"perm_256 requires multiple of 256, got {d}"
    dt = t.dtype
    work = t.astype(jnp.float8_e4m3fn) if dt == jnp.float4_e2m1fn else t
    sh = work.shape
    work = work.reshape(*sh[:axis], d // 256, 16, 16, *sh[axis + 1 :])
    work = jnp.swapaxes(work, axis + 1, axis + 2).reshape(sh)
    return work.astype(dt) if dt == jnp.float4_e2m1fn else work


def permute_weights_kperm(weights: tuple, kinds: tuple, tp: int, hel: int, he: int):
    """Permute wpool, wpool_s, and pre_expert_w on fp4 layers so expert x arrives in kperm order."""
    L = len(kinds)
    if not any(k == "fp4" for k in kinds):
        return weights
    fp4_mask = jnp.array([k == "fp4" for k in kinds], dtype=jnp.bool_)
    sm, wqkvg, wo, wpool, wrouter, wunpool = weights

    def _perm_sm(s):
        s4 = s.reshape(tp, L, *s.shape[1:])
        wps = s4[:, :, SMALL["wpool_s"], :hel]
        s4 = s4.at[:, :, SMALL["wpool_s"], :hel].set(
            jnp.where(fp4_mask[None, :, None], perm_256(wps, -1), wps)
        )
        pew = s4[:, :, SMALL["pre_expert_w"], :he]
        s4 = s4.at[:, :, SMALL["pre_expert_w"], :he].set(
            jnp.where(fp4_mask[None, :, None], perm_256(pew, -1), pew)
        )
        return s4.reshape(s.shape)

    def _perm_wp(wp):
        wp4 = wp.reshape(tp, L, *wp.shape[1:])
        wp4 = jnp.where(fp4_mask[None, :, None, None], perm_256(wp4, -1), wp4)
        return wp4.reshape(wp.shape)

    sm_p = jax.block_until_ready(jax.jit(_perm_sm, out_shardings=sm.sharding)(sm))
    wp_p = jax.block_until_ready(jax.jit(_perm_wp, out_shardings=wpool.sharding)(wpool))
    return (sm_p, wqkvg, wo, wp_p, wrouter, wunpool)


def permute_expert_slab_kperm(slab: tuple):
    """Permute an fp4 expert slab (w13, w13s, w2, w2s) for zero-shuffle kperm dequant."""
    w13, w13s, w2, w2s = slab
    w13_p = jax.block_until_ready(
        jax.jit(lambda x: perm_256(perm_256(x, -2), -1), out_shardings=w13.sharding)(
            w13
        )
    )
    w13s_p = jax.block_until_ready(
        jax.jit(lambda x: perm_256(x, -1), out_shardings=w13s.sharding)(w13s)
    )
    w2_p = jax.block_until_ready(
        jax.jit(lambda x: perm_256(x, -2), out_shardings=w2.sharding)(w2)
    )
    return (w13_p, w13s_p, w2_p, w2s)


# ex_kperm >= 3: largest dequantized fp4 weight after the block scales are normalized (e4m3
# keeps its 3 mantissa bits down to 2^-6, so the weights stay in its normal range)
WFP8_TARGET_MAX = 32.0
_FP4_MAX = 6.0


def expert_wfp8_globals(slab: tuple, target_max: float = WFP8_TARGET_MAX):
    """Per-(rank, expert) (gate, up, down) globals of an fp4 slab: [rows] f32 each, such that

    block scale / global keeps every dequantized weight at most target_max.
    """
    _, w13s, _, w2s = slab
    il = w13s.shape[-1] // 2

    def f(s13, s2):
        a13 = jnp.abs(s13.astype(jnp.float32)) * _FP4_MAX
        a2 = jnp.abs(s2.astype(jnp.float32)) * _FP4_MAX
        # Snap per-expert globals to exact powers of 2 so s / global preserves
        # every e4m3 block scale's 3-bit mantissa without rounding error.
        half_target = target_max * 0.5
        return tuple(
            jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(x, 1e-30) / half_target)))
            for x in (
                jnp.max(a13[..., :il], axis=(1, 2)),
                jnp.max(a13[..., il:], axis=(1, 2)),
                jnp.max(a2, axis=(1, 2)),
            )
        )

    shd = jax.sharding.NamedSharding(
        w13s.sharding.mesh, jax.sharding.PartitionSpec(w13s.sharding.spec[0])
    )
    return jax.block_until_ready(jax.jit(f, out_shardings=(shd, shd, shd))(w13s, w2s))


def normalize_expert_slab_wfp8(slab: tuple, gs: tuple):
    """Divides an fp4 slab's block scales by its (gate, up, down) globals (ex_kperm >= 3)."""
    w13, w13s, w2, w2s = slab
    il = w13s.shape[-1] // 2

    def f(s13, s2, gg, gu, gd):
        inv13 = jnp.concatenate(
            [
                jnp.broadcast_to((1.0 / gg)[:, None, None], s13[..., :il].shape),
                jnp.broadcast_to((1.0 / gu)[:, None, None], s13[..., il:].shape),
            ],
            axis=-1,
        )
        s13n = (s13.astype(jnp.float32) * inv13).astype(s13.dtype)
        s2n = (s2.astype(jnp.float32) / gd[:, None, None]).astype(s2.dtype)
        return s13n, s2n

    s13n, s2n = jax.block_until_ready(
        jax.jit(f, out_shardings=(w13s.sharding, w2s.sharding))(w13s, w2s, *gs)
    )
    return (w13, s13n, w2, s2n)


def apply_expert_gs_to_smalls(smalls, gs: tuple, kinds: tuple, tp: int, n_experts: int):
    """Writes the (gate, up, down) globals into the fp4 layers' smalls rows expert_gs*."""
    L = len(kinds)
    E = n_experts
    Ep = -(-E // LANE) * LANE
    fp4 = jnp.array([k == "fp4" for k in kinds], dtype=jnp.bool_)

    def f(sm, gg, gu, gd):
        s4 = sm.reshape(tp, L, *sm.shape[1:])
        r_up, r_dn = SMALL["expert_gs"], SMALL["expert_gs_down"]
        bc = lambda g: jnp.broadcast_to(g.reshape(tp, 1, E), (tp, L, E))
        up = s4[:, :, r_up, :]
        new_up = up.at[:, :, 0:E].set(bc(gg)).at[:, :, Ep : Ep + E].set(bc(gu))
        dn = s4[:, :, r_dn, :]
        new_dn = dn.at[:, :, 0:E].set(bc(gd))
        s4 = s4.at[:, :, r_up, :].set(jnp.where(fp4[None, :, None], new_up, up))
        s4 = s4.at[:, :, r_dn, :].set(jnp.where(fp4[None, :, None], new_dn, dn))
        return s4.reshape(sm.shape)

    return jax.block_until_ready(jax.jit(f, out_shardings=smalls.sharding)(smalls, *gs))


def _dequant_fp4(words, scale, via_fp8=False):
    """``[cw, N]`` u32 + ``[cw*8/16, N]`` scales -> ``[cw*8, N]`` bf16."""
    cw, n = words.shape
    f4 = pltpu.bitcast(words, jnp.float4_e2m1fn)
    v = (
        f4.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
        if via_fp8
        else f4.astype(jnp.bfloat16)
    )
    gq = cw * FP4_PACK // FP4_GROUP
    s = scale if scale.dtype == jnp.bfloat16 else scale.astype(jnp.bfloat16)
    s = jnp.broadcast_to(s[:, None, :], (gq, FP4_GROUP, n)).reshape(cw * FP4_PACK, n)
    return v * s


def _dequant_fp4_kperm(words, scale, via_fp8=False, out_fp8=False, vec=False):
    """Zero-shuffle NVFP4 dequant for K permuted as (B, r=16, g=16).

    ``out_fp8``: the dequantized weights go to the MXU as e4m3 (ex_kperm >= 3: the
    block scales are normalized so every weight is at most WFP8_TARGET_MAX).
    """
    cw, n = words.shape
    if vec and out_fp8 and cw == 64:
        # Dequantize two 32-word (256-row) halves sequentially into fp8 so peak bf16
        # live registers stay [256, n] (32 vregs at n=256) instead of [512, n] (64 vregs).
        w0 = _dequant_fp4_kperm(
            words[:32], scale[:16], via_fp8=via_fp8, out_fp8=True, vec=True
        )
        w1 = _dequant_fp4_kperm(
            words[32:], scale[16:], via_fp8=via_fp8, out_fp8=True, vec=True
        )
        return jnp.concatenate([w0, w1], axis=0)
    nb = cw // 32
    f4 = pltpu.bitcast(words, jnp.float4_e2m1fn)
    v = (
        f4.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
        if via_fp8
        else f4.astype(jnp.bfloat16)
    )
    s = scale if scale.dtype == jnp.bfloat16 else scale.astype(jnp.bfloat16)
    if vec:
        res = (v.reshape(nb, 16, 16, n) * s.reshape(nb, 1, 16, n)).reshape(
            cw * FP4_PACK, n
        )
    else:
        parts = []
        for b in range(nb):
            sb = s[b * 16 : (b + 1) * 16]
            base = b * 256
            for r in range(16):
                parts.append(v[base + r * 16 : base + (r + 1) * 16] * sb)
        res = jnp.concatenate(parts, axis=0)
    return res.astype(jnp.float8_e4m3fn) if out_fp8 else res


def _scale_rows(sslot, g0, gn, N, blk, n0=0):
    """Scale rows [g0, g0 + gn) of a slab as f32.

    e4m3 slabs tile 32 rows: a narrower slice reads its whole tile once (cached in
    blk) and slices it.
    """
    if sslot.dtype != jnp.float8_e4m3fn or gn % 32 == 0:
        return sslot[pl.ds(g0, gn), pl.ds(n0, N)].astype(jnp.float32), blk
    if g0 % 32 == 0 or blk is None:
        blk = sslot[pl.ds(g0 - g0 % 32, 32), pl.ds(n0, N)].astype(jnp.float32)
    return blk[g0 % 32 : g0 % 32 + gn], blk


def _fp4_dot(
    x,
    wslot,
    sslot,
    K,
    N,
    chunk_words,
    fast_bf16=False,
    kperm=0,
    via_fp8=False,
    vec_dequant=False,
    fp8_lhs=False,
    n_tile=0,
    lhs_chunk_fn=None,
    prequant_sc=None,
    out_stage_ref=None,
    post_scale=None,
    force_bf16_w=False,
    nrows=None,
):
    kw = K // FP4_PACK
    m_rows = (
        nrows
        if nrows is not None
        else (
            x.shape[0]
            if x is not None
            else (out_stage_ref.shape[0] if out_stage_ref is not None else 0)
        )
    )
    can_fp8_m = m_rows >= 32 and m_rows % 32 == 0
    use_fp8_lhs = (fp8_lhs or prequant_sc is not None) and kperm >= 3 and can_fp8_m
    if kperm > 0:
        # Native fp8 x fp8 vmatmul.e4m3 contracts K = 512 (64 words) per MXU pass.
        chunk_words = min(kw, 64 if use_fp8_lhs else (32 if vec_dequant else 64))
    elif fast_bf16:
        chunk_words = min(kw, 32)
    use_bf16 = (fast_bf16 or kperm > 0) and not use_fp8_lhs
    xf = x_sc = xb = None
    if lhs_chunk_fn is None:
        if use_fp8_lhs and prequant_sc is None:
            xf, x_sc = _quant_fp8_rows(x)
        elif use_fp8_lhs:
            xf, x_sc = x, prequant_sc
        else:
            xb = (
                (x if x.dtype == jnp.bfloat16 else x.astype(jnp.bfloat16))
                if use_bf16
                else None
            )
    else:
        x_sc = prequant_sc
    out_fp8_w = (kperm >= 3) and not (force_bf16_w and not use_fp8_lhs)
    direct_bf16 = (use_bf16 or use_fp8_lhs) and sslot.dtype == jnp.bfloat16
    tn_step = max(256, n_tile // 4) if (n_tile > 0 and N > 512) else N
    cols = [] if out_stage_ref is None else None
    for n0 in range(0, N, tn_step):
        tn = min(tn_step, N - n0)
        acc = blk = None
        for c in range(0, kw, chunk_words):
            cw = min(chunk_words, kw - c)
            g0 = c * FP4_PACK // FP4_GROUP
            gq = cw * FP4_PACK // FP4_GROUP
            if direct_bf16:
                sc = sslot[pl.ds(g0, gq), pl.ds(n0, tn)]
            else:
                sc, blk = _scale_rows(sslot, g0, gq, tn, blk, n0=n0)
            if kperm > 0:
                w = _dequant_fp4_kperm(
                    wslot[pl.ds(c, cw), pl.ds(n0, tn)],
                    sc,
                    via_fp8=(via_fp8 or kperm >= 2),
                    out_fp8=out_fp8_w,
                    vec=vec_dequant,
                )
            else:
                w = _dequant_fp4(
                    wslot[pl.ds(c, cw), pl.ds(n0, tn)], sc, via_fp8=via_fp8
                )
            if lhs_chunk_fn is not None:
                lhs = lhs_chunk_fn(c, cw, use_fp8_lhs)
            elif use_fp8_lhs:
                lhs = xf[:, c * FP4_PACK : (c + cw) * FP4_PACK]
            elif xb is not None:
                lhs = xb[:, c * FP4_PACK : (c + cw) * FP4_PACK]
            else:
                lhs = x[:, c * FP4_PACK : (c + cw) * FP4_PACK].astype(jnp.bfloat16)
            part = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
            acc = part if acc is None else acc + part
        if use_fp8_lhs and x_sc is not None:
            acc = acc * (x_sc * post_scale if post_scale is not None else x_sc)
        elif post_scale is not None:
            acc = acc * post_scale
        if out_stage_ref is not None:
            out_stage_ref[:, pl.ds(n0, tn)] = (
                acc if out_stage_ref.dtype == jnp.float32 else acc.astype(jnp.bfloat16)
            )
        else:
            cols.append(acc)
    if out_stage_ref is not None:
        return None
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


def _fp4_blockdiag_dot(
    xc,
    wslot,
    sslot,
    K,
    N,
    kc,
    kperm=False,
    fp8_lhs=False,
    n_tile=0,
    lhs_chunk_fn=None,
    prequant_sc=None,
    nrows=None,
    post_scale=None,
):
    """``xc [A, K]`` bf16 @ an NVFP4 slab (``wslot`` u32 words ``[K/8, N]``, block-16 scales

    ``sslot [K/16, N]``) -> f32 ``[A, N]``, exact in f32 (or native fp8xfp8 when
    ``fp8_lhs``).
    """
    A = (
        nrows
        if nrows is not None
        else (xc.shape[0] if xc is not None else prequant_sc.shape[0])
    )
    if kperm:
        assert kc == 256, f"kperm blockdiag requires kc=256, got {kc}"
    kc = min(K, kc)
    gpc = kc // FP4_GROUP
    rows = A * gpc
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 1)
    diag = jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % gpc == (
        col_iota % FP4_GROUP if kperm else col_iota // FP4_GROUP
    )
    can_fp8_m = A >= 32 and A % 32 == 0
    use_fp8_lhs = (fp8_lhs or prequant_sc is not None) and can_fp8_m
    if lhs_chunk_fn is None:
        if use_fp8_lhs and prequant_sc is None:
            xf, x_sc = _quant_fp8_rows(xc)
        elif use_fp8_lhs:
            xf, x_sc = xc, prequant_sc
        else:
            xf = x_sc = None
    else:
        xf = None
        x_sc = prequant_sc
    z_lhs = jnp.zeros((rows, kc), jnp.float8_e4m3fn if use_fp8_lhs else jnp.bfloat16)

    def make_lhs(c0):
        if lhs_chunk_fn is not None:
            xs = lhs_chunk_fn(c0 // FP4_PACK, kc // FP4_PACK, use_fp8_lhs)
        elif use_fp8_lhs:
            xs = xf[:, c0 : c0 + kc]
        else:
            xs = xc[:, c0 : c0 + kc].astype(jnp.bfloat16)
        return jnp.where(
            diag,
            jnp.broadcast_to(xs[:, None, :], (A, gpc, kc)).reshape(rows, kc),
            z_lhs,
        )

    # At A >= 4 (rows >= 64), cap column tile at 512 ([64, 512] f32 = 32 vregs) and
    # compute make_lhs(c0) on the fly so %v never spills on the 64-vreg v7x file.
    bd_tn = min(n_tile, 512) if (n_tile > 0 and A >= 4) else n_tile
    if bd_tn > 0 and N > bd_tn and A >= 4:
        cols = []
        for n0 in range(0, N, bd_tn):
            tn = min(bd_tn, N - n0)
            acc = blk = None
            for c0 in range(0, K, kc):
                lhs = make_lhs(c0)
                w = pltpu.bitcast(
                    wslot[pl.ds(c0 // FP4_PACK, kc // FP4_PACK), pl.ds(n0, tn)],
                    jnp.float4_e2m1fn,
                ).astype(jnp.float8_e4m3fn)
                res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
                s, blk = _scale_rows(sslot, c0 // FP4_GROUP, gpc, tn, blk, n0=n0)
                part = res * (jnp.concatenate([s] * A, axis=0) if A > 1 else s)
                acc = part if acc is None else acc + part
            cols.append(
                jnp.concatenate(
                    [
                        jnp.sum(acc[a * gpc : (a + 1) * gpc], axis=0, keepdims=True)
                        for a in range(A)
                    ],
                    axis=0,
                )
            )
        out = jnp.concatenate(cols, axis=1)
        sc_mul = x_sc if (use_fp8_lhs and x_sc is not None) else None
        if post_scale is not None:
            sc_mul = post_scale if sc_mul is None else sc_mul * post_scale
        return out * sc_mul if sc_mul is not None else out

    acc = blk = None
    for c0 in range(0, K, kc):
        lhs = make_lhs(c0)
        w = pltpu.bitcast(
            wslot[pl.ds(c0 // FP4_PACK, kc // FP4_PACK), pl.ds(0, N)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
        s, blk = _scale_rows(sslot, c0 // FP4_GROUP, gpc, N, blk)
        part = res * (jnp.concatenate([s] * A, axis=0) if A > 1 else s)
        acc = part if acc is None else acc + part
    out = jnp.concatenate(
        [
            jnp.sum(acc[a * gpc : (a + 1) * gpc], axis=0, keepdims=True)
            for a in range(A)
        ],
        axis=0,
    )
    sc_mul = x_sc if (use_fp8_lhs and x_sc is not None) else None
    if post_scale is not None:
        sc_mul = post_scale if sc_mul is None else sc_mul * post_scale
    return out * sc_mul if sc_mul is not None else out


def _fp8_dot(
    x,
    wslot,
    scale_row,
    K,
    N,
    chunk_words,
    fp8_lhs=False,
    lhs_chunk_fn=None,
    prequant_sc=None,
    out_stage_ref=None,
    post_scale=None,
    nrows=None,
):
    """fp8 weights held as ``[K/4, N]`` u32 words; channelwise ``scale_row [1, N]``."""
    kw = K // 4
    m_rows = (
        nrows
        if nrows is not None
        else (
            x.shape[0]
            if x is not None
            else (out_stage_ref.shape[0] if out_stage_ref is not None else 0)
        )
    )
    can_fp8_m = m_rows >= 32 and m_rows % 32 == 0
    use_fp8_lhs = (fp8_lhs or prequant_sc is not None) and can_fp8_m
    if use_fp8_lhs:
        chunk_words = min(kw, 128)  # 128 u32 words = 512 fp8 contraction depth
    xf = x_sc = None
    if lhs_chunk_fn is None:
        if use_fp8_lhs and prequant_sc is None:
            xf, x_sc = _quant_fp8_rows(x)
        elif use_fp8_lhs:
            xf, x_sc = x, prequant_sc
    else:
        x_sc = prequant_sc
    tn_step = 256 if (out_stage_ref is not None and N > 256) else N
    cols = [] if out_stage_ref is None else None
    for n0 in range(0, N, tn_step):
        tn = min(tn_step, N - n0)
        acc = None
        for c in range(0, kw, chunk_words):
            cw = min(chunk_words, kw - c)
            w = pltpu.bitcast(wslot[pl.ds(c, cw), pl.ds(n0, tn)], jnp.float8_e4m3fn)
            if lhs_chunk_fn is not None:
                lhs = lhs_chunk_fn(c // 2, cw // 2, use_fp8_lhs)
            elif use_fp8_lhs:
                lhs = xf[:, c * 4 : (c + cw) * 4]
            else:
                lhs = x[:, c * 4 : (c + cw) * 4].astype(jnp.bfloat16)
            part = jnp.dot(
                lhs,
                w,
                preferred_element_type=jnp.float32,
            )
            acc = part if acc is None else acc + part
        sc_sub = scale_row[:, n0 : n0 + tn]
        row_mul = x_sc if (use_fp8_lhs and x_sc is not None) else None
        if post_scale is not None:
            row_mul = post_scale if row_mul is None else row_mul * post_scale
        acc = acc * (row_mul * sc_sub) if row_mul is not None else acc * sc_sub
        if out_stage_ref is not None:
            out_stage_ref[:, pl.ds(n0, tn)] = (
                acc if out_stage_ref.dtype == jnp.float32 else acc.astype(jnp.bfloat16)
            )
        else:
            cols.append(acc)
    if out_stage_ref is not None:
        return None
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


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


def ring_slot_shape(c):
    """u32 words per ring slot: one [tile_k, tile_n] bf16 tile."""
    return (c.tile_k // 2, c.tile_n)


def fold_groups(c):
    """Weights whose matvecs consume an all-gather's column blocks as they arrive."""
    return (("wqkvg",),) * c.fold_f + (("wpool", "wrouter"),) * c.fold_o


def _fold_g(c, grp):
    """Column blocks per fold group: the widest K tile of the group, in blocks."""
    rk, rn = ring_slot_shape(c)
    return (
        max(
            _choose_tile(K, N, rk * _PACK[jnp.dtype(dt)], rn)[0]
            for name, K, N, dt in dense_plan(c)
            if name in grp
        )
        // c.hidden_local
    )


def owner_window_rows(c):
    """Returns the rows of the 8-row-aligned window around an owner block.

    Owner blocks of rows_per_dev rows start at 8-row boundaries only when
    rows_per_dev is a multiple of 8; otherwise the block is read from the aligned
    window around it.

    Args:
        c: the MKConfig.
    """
    rc = c.rows_per_dev
    return rc if rc % 8 == 0 else -(-(rc + 7) // 8) * 8


def fold_fits(c):
    """Returns whether every fold group's tiles fit in the weight stream.

    A fold group waits for all its tiles before its first dot. A 3-slot ring
    (stream depth 2) is too short for the folds.

    Args:
        c: the MKConfig, with its ring and stream depth chosen.
    """
    hl = c.hidden_local
    for grp in fold_groups(c):
        tiles = [t for t in dense_tiles(c) if t[0] in grp]
        g = _fold_g(c, grp)
        if sum(1 for t in tiles if t[1] < g // (t[3] // hl)) > c.stream_depth:
            return False
    return True


def dense_tiles(c):
    """(name, ki, ni, tk, tn, dtype) of every ring tile of one layer, in consumption order.

    A fold group's tiles run K group by K group (all its weights per group); ki is
    then logical:
    group j holds the physical K blocks a fold consumes j-th
    (_WeightStream.phys_k).
    """
    rk, rn = ring_slot_shape(c)
    geo = {}
    for name, K, N, dt in dense_plan(c):
        geo[name] = (K, N, dt) + _choose_tile(K, N, rk * _PACK[jnp.dtype(dt)], rn)
    folded = {n: grp for grp in fold_groups(c) for n in grp}
    tiles, done = [], set()
    for name, K, N, dt in dense_plan(c):
        if name in done:
            continue
        if name not in folded:
            _, _, _, tk, tn = geo[name]
            tiles += [
                (name, ki, ni, tk, tn, dt)
                for ni in range(N // tn)
                for ki in range(K // tk)
            ]
            continue
        grp = folded[name]
        G = _fold_g(c, grp)
        for j in range(K // (G * c.hidden_local)):
            for w in grp:
                _, Nw, dtw, tk, tn = geo[w]
                s_per = G * c.hidden_local // tk
                tiles += [
                    (w, j * s_per + s, ni, tk, tn, dtw)
                    for s in range(s_per)
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

    The sequence is identical for every layer, so tile t of any layer lives in
    slot
    ``t % slots`` and the issue window runs ``depth`` tiles ahead across matvec
    and layer
    boundaries: attention, collectives and the expert phase do not leave the DMA
    engine idle
    while there is dense weight left to fetch. ``depth <= slots - 1``: the slot a
    new tile lands
    in held the tile consumed just before it. Slots are u32 words, viewed as the
    tile's dtype.
    """

    def __init__(
        self, W, buf, sems, tiles, depth, col_rot=None, k_rot=None, fp8_lhs=False
    ):
        self.W, self.buf, self.sems = W, buf, sems
        self.fp8_lhs = fp8_lhs
        # name -> traced shift: column tile ni of that weight streams physical column
        # (ni + shift) % n_cols (a per-device order)
        self.col_rot = col_rot or {}
        # name -> (traced own block, blocks per group G, tiles per group s_per): logical K tile
        # j * s_per + s streams the physical tile of group (own // G) ^ j (fold arrival order)
        self.k_rot = k_rot or {}
        # LAZY: while hold is set, refills of tiles >= hold go to self.held (issued by flush)
        self.hold, self.held = None, []
        self.slots = buf.shape[0]
        self.tiles, self.ranges = tiles, {}
        for t, (name, _, ni, tk, tn, dt) in enumerate(tiles):
            t0 = self.ranges[name][0] if name in self.ranges else t
            self.ranges[name] = (t0, t + 1, tk, tn, ni + 1, dt)
        self.T = len(self.tiles)
        if self.T % self.slots:
            raise ValueError(
                f"{self.T} dense tiles per layer do not divide into {self.slots} ring"
                " slots"
            )
        self.depth = min(depth, self.slots - 1, self.T)

    def _copy(self, l, t):
        name, ki, ni, tk, tn, dt = self.tiles[t]
        pack = _PACK[jnp.dtype(dt)]
        s_ = t % self.slots
        col = self.phys_col(name, ni)
        c0 = ni * tn if isinstance(col, int) else pl.multiple_of(col * tn, LANE)
        pk = self.phys_k(name, ki)
        r0 = (
            pk * (tk // pack)
            if isinstance(pk, int)
            else pl.multiple_of(pk * (tk // pack), 8)
        )
        src = (
            self.W[name].bitcast(jnp.uint32).at[l, pl.ds(r0, tk // pack), pl.ds(c0, tn)]
        )
        return pltpu.make_async_copy(
            src,
            self.buf.at[s_, pl.ds(0, tk // pack), pl.ds(0, tn)],
            self.sems.at[s_],
        )

    def phys_k(self, name, ki):
        if name not in self.k_rot:
            return ki
        own, G, s_per = self.k_rot[name]
        return jnp.bitwise_xor(own // G, ki // s_per) * s_per + ki % s_per

    def refill(self, l, t, next_l):
        """Issue the tile depth past tile t, which was just consumed."""
        u = t + self.depth
        if self.hold is not None and u >= self.hold:
            self.held.append((l, t, next_l))
            return
        if u < self.T:
            self._copy(l, u).start()
        elif next_l is not None:
            cp = self._copy(next_l, u - self.T)
            if isinstance(next_l, int):
                cp.start()
            else:
                # a rolled loop's last layer has no next layer
                pl.when(next_l < self.W["smalls"].shape[0])(cp.start)

    def flush(self):
        """Issue the held refills."""
        held, self.held, self.hold = self.held, [], None
        for l, t, next_l in held:
            self.refill(l, t, next_l)

    def mv_fold(self, l, names, next_l, G, bw, wait_block, lhs_block):
        """Matvecs of the fold group names over arriving K blocks of width bw.

        Group j holds the G physical blocks k = ((own // G) ^ j) * G + i; per block,
        wait_block(k) waits for it and lhs_block(k) gives the bf16 LHS of its K rows
        (any row
        count). Returns name -> unscaled f32 [rows, N].
        """
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
                    s = ki % (G // bpt)
                    if i // bpt != s:
                        continue
                    r0 = (i % bpt) * bw
                    w = self.buf.at[u % self.slots].bitcast(dt)[
                        pl.ds(r0, bw), pl.ds(0, tn)
                    ]
                    if (
                        self.fp8_lhs
                        and jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn)
                        and lhs.shape[0] >= 32
                    ):
                        xf_d, xsc_d = _quant_fp8_rows(lhs)
                        part = (
                            jnp.dot(xf_d, w, preferred_element_type=jnp.float32) * xsc_d
                        )
                    else:
                        part = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
                    acc[name][ni] = (
                        part if acc[name][ni] is None else acc[name][ni] + part
                    )
            for u in gt:
                self.refill(l, u, next_l)
        return {
            n: a[0] if len(a) == 1 else jnp.concatenate(a, axis=1)
            for n, a in acc.items()
        }

    def phys_col(self, name, ni):
        if name not in self.col_rot:
            return ni
        return jax.lax.rem(ni + self.col_rot[name], self.ranges[name][4])

    def prologue(self, l):
        for t in range(self.depth):
            self._copy(l, t).start()

    def mv(self, x, l, name, next_l, sm=None, on_col=None, sm_ref=None):
        """``x [R, K]`` @ ``W_l[name]`` -> f32 ``[R, N]``; fp8 weights take their scale row from ``sm``.

        ``on_col(ni, col)`` gets each finished (scaled) column tile as soon as its
        last K tile is in.
        """
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
            lhs_s = xb[:, ki * tk : (ki + 1) * tk]
            if self.fp8_lhs and is_fp8 and xb.shape[0] >= 32:
                xf_d, xsc_d = _quant_fp8_rows(lhs_s)
                part = jnp.dot(xf_d, w, preferred_element_type=jnp.float32) * xsc_d
            else:
                part = jnp.dot(lhs_s, w, preferred_element_type=jnp.float32)
            cols[ni] = part if cols[ni] is None else cols[ni] + part
            self.refill(l, t, next_l)
            if on_col is not None and ki == nk - 1:
                col = self.phys_col(name, ni)
                if is_fp8 and isinstance(col, int):
                    r = SMALL[name + "_s"]
                    cols[ni] = cols[ni] * sm[r : r + 1, ni * tn : (ni + 1) * tn]
                elif (
                    is_fp8
                ):  # a rotated column's scales: a dynamic lane window of the ref
                    r = SMALL[name + "_s"]
                    cols[ni] = (
                        cols[ni]
                        * sm_ref[r : r + 1, pl.ds(pl.multiple_of(col * tn, LANE), tn)]
                    )
                on_col(col, cols[ni])
        out = cols[0] if nn == 1 else jnp.concatenate(cols, axis=1)
        if is_fp8 and on_col is None:
            r = SMALL[name + "_s"]
            out = out * sm[r : r + 1, : out.shape[1]]
        return out


# ----------------------------------------------------------------------------- collectives
class _Coll:
    """One-hop collectives over the mesh axis as remote DMAs into peers' VMEM pads.

    Every collective kind has its own pad and its own send/receive semaphores: a
    peer
    one round ahead then cannot satisfy this round's receive wait with its next
    round's
    bytes. A kind's pad is reused only after at least one other rendezvous round,
    so a
    single buffer per kind suffices. ``slot`` is the sender's position in the
    gathered
    result (its TP rank for column-parallel outputs, its mesh index otherwise).
    """

    def __init__(
        self,
        tp,
        me,
        axis,
        pads,
        ssems,
        rsems,
        order="ring",
        local_last=False,
        pair_ag=0,
        pair_ssems=None,
        pair_rsems=None,
        can_pair_fold=False,
    ):
        self.tp, self.me, self.axis = tp, me, axis
        self.pads, self.ssems, self.rsems = pads, ssems, rsems
        self.kinds = {k: i for i, k in enumerate(pads)}
        self.order = order if tp & (tp - 1) == 0 else "ring"
        self.local_last = local_last
        self.pair_ag = pair_ag if (tp == 8 and pair_ssems is not None) else 0
        self.pair_ssems = pair_ssems
        self.pair_rsems = pair_rsems
        self.can_pair_fold = bool(self.pair_ag and can_pair_fold)

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
        )

    def _local(self, fn, issued):
        """The round's own-part copy: before its remote copies, or after them with

        local_last (``issued`` False / True).
        """
        if self.local_last == issued:
            fn()

    def all_gather(self, kind, src_ref, slot, live=None):
        """``src_ref`` from every device into ``pad[slot_of_sender]``; returns the pad ref.

        ``live(m)``: whether device m's part is needed at all; parts of dead devices
        are
        neither sent nor waited for, and their pad slots hold stale data.
        """
        pad = self.pads[kind]
        if live is None or self.pair_ag >= 1:
            return self.all_gather_start(kind, src_ref, slot)()

        def local():
            pad[slot] = src_ref[...]

        self._local(local, False)
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]

        @pl.when(live(self.me))
        def _():
            for c in cps:
                c.start()
            self._local(local, True)
            for c in cps:
                c.wait_send()

        if self.local_last:
            pl.when(jnp.logical_not(live(self.me)))(local)
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
        if self.pair_ag >= 1:
            ik = self.kinds[kind]
            core = self.me % 2
            chip = self.me // 2
            partner = jnp.bitwise_xor(self.me, 1)
            pad[self.me] = src_ref[...]
            cross_cps = [
                self._copy(
                    kind,
                    src_ref,
                    pad.at[self.me],
                    2 * jnp.bitwise_xor(chip, d) + core,
                )
                for d in (1, 2, 3)
            ]
            for cp in cross_cps:
                cp.start()

            def wait_pair():
                for cp in cross_cps:
                    cp.wait()
                pair_cps = [
                    pltpu.make_async_remote_copy(
                        src_ref=pad.at[2 * c_k + core],
                        dst_ref=pad.at[2 * c_k + core],
                        send_sem=self.pair_ssems.at[ik],
                        recv_sem=self.pair_rsems.at[ik],
                        device_id=(partner,),
                    )
                    for c_k in range(4)
                ]
                for cp in pair_cps:
                    cp.start()
                for cp in pair_cps:
                    cp.wait()
                return pad

            return wait_pair

        def local():
            pad[slot] = src_ref[...]

        self._local(local, False)
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        self._local(local, True)

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
        )

    def all_gather_blocks_start(self, kind, src_ref, slot, rsems):
        """all_gather whose sends signal the receiver's rsems[slot], so a receiver can wait

        for one sender's block (all_gather_blocks_wait).
        """
        pad = self.pads[kind]
        if self.can_pair_fold:
            ik = self.kinds[kind]
            core = self.me % 2
            chip = self.me // 2
            partner = jnp.bitwise_xor(self.me, 1)
            pad[self.me] = src_ref[...]
            pltpu.make_async_remote_copy(
                src_ref=src_ref,
                dst_ref=pad.at[self.me],
                send_sem=self.pair_ssems.at[ik],
                recv_sem=rsems.at[self.me],
                device_id=(partner,),
            ).start()
            for d in (1, 2, 3):
                peer = 2 * jnp.bitwise_xor(chip, d) + core
                self._block_copy_to(kind, src_ref, self.me, rsems, peer).start()
            return

        def local():
            pad[slot] = src_ref[...]

        self._local(local, False)
        for d in range(1, self.tp):
            self._block_copy_to(kind, src_ref, slot, rsems, self._peer(d)).start()
        self._local(local, True)

    def all_gather_blocks_wait(self, kind, src_ref, slot, rsems):
        """(wait_block(k), finish()) of an all_gather_blocks_start: the first waits

        for pad slot k (own slot: no-op), the second for this device's sends.
        """
        if self.can_pair_fold:
            ik = self.kinds[kind]
            core = self.me % 2
            chip = self.me // 2
            partner = jnp.bitwise_xor(self.me, 1)
            pad = self.pads[kind]

            def wait_block_pair(k):
                @pl.when(k % 2 == 0)
                def _():
                    pg = k // 2

                    @pl.when(pg == chip)
                    def _():
                        k_other = 2 * chip + (1 - core)
                        pltpu.make_async_remote_copy(
                            src_ref=src_ref,
                            dst_ref=pad.at[k_other],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[k_other],
                            device_id=(self.me,),
                        ).wait_recv()
                        pltpu.make_async_remote_copy(
                            src_ref=src_ref,
                            dst_ref=pad.at[self.me],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[self.me],
                            device_id=(partner,),
                        ).wait_send()

                    @pl.when(pg != chip)
                    def _():
                        k_self = 2 * pg + core
                        k_other = 2 * pg + (1 - core)
                        self._block_copy_to(
                            kind, src_ref, k_self, rsems, self.me
                        ).wait_recv()
                        relay = pltpu.make_async_remote_copy(
                            src_ref=pad.at[k_self],
                            dst_ref=pad.at[k_self],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[k_self],
                            device_id=(partner,),
                        )
                        relay.start()
                        pltpu.make_async_remote_copy(
                            src_ref=pad.at[k_other],
                            dst_ref=pad.at[k_other],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[k_other],
                            device_id=(self.me,),
                        ).wait_recv()
                        relay.wait_send()

            def finish_pair():
                for d in (1, 2, 3):
                    peer = 2 * jnp.bitwise_xor(chip, d) + core
                    self._block_copy_to(kind, src_ref, self.me, rsems, peer).wait_send()

            return wait_block_pair, finish_pair

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
        """Rows ``[owner*rows_per, ...)`` of ``send_ref`` into ``owner``'s pad slot ``me``."""
        return self._copy(
            kind,
            send_ref.at[pl.ds(pl.multiple_of(owner * rows_per, rows_per), rows_per)],
            self.pads[kind].at[self.me],
            owner,
        )

    def reduce_scatter_rows(self, kind, send_ref, rows_per, n_live, started=False):
        """bf16-packed ``send_ref [tp*rows_per, ...]``: device m receives everyone's rows

        ``[m*rows_per, ...)``; returns the pad ``[tp, rows_per, ...]`` (own slot
        filled locally).
        Devices whose rows are all past ``n_live`` neither receive nor get sent to.
        ``started``: every live owner's ``block_copy`` was already started; only
        wait.
        """
        pad = self.pads[kind]
        my_live = self.me * rows_per < n_live

        def local():
            @pl.when(my_live)
            def _():
                pad[self.me] = send_ref[
                    pl.ds(pl.multiple_of(self.me * rows_per, rows_per), rows_per)
                ]

        self._local(local, False)
        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cps.append((peer, self.block_copy(kind, send_ref, peer, rows_per)))
        if not started:
            for peer, cp in cps:
                pl.when(peer * rows_per < n_live)(cp.start)
        self._local(local, True)
        for peer, cp in cps:
            pl.when(peer * rows_per < n_live)(cp.wait_send)

        @pl.when(my_live)
        def _():
            for _, cp in cps:
                cp.wait_recv()

        return pad

    def _cols_copy(self, kind, src_ref, width, m):
        c0 = m * width if isinstance(m, int) else pl.multiple_of(m * width, LANE)
        return self._copy(
            kind, src_ref.at[:, pl.ds(c0, width)], self.pads[kind].at[self.me], m
        )

    def reduce_scatter_cols(self, kind, src_ref, width):
        """``src_ref [R, tp*width]`` f32: device m receives everyone's columns ``[m*width, ...)``."""
        pad = self.pads[kind]

        def local():
            pad[self.me] = src_ref[
                :, pl.ds(pl.multiple_of(self.me * width, LANE), width)
            ]

        self._local(local, False)
        cps = [
            self._cols_copy(kind, src_ref, width, self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        self._local(local, True)
        for c in cps:
            c.wait()
        return pad

    def reduce_scatter_cols_streamed(self, kind, src_ref, width):
        """``reduce_scatter_cols`` issued one column block at a time: ``send(m)`` once

        ``src_ref[:, m*width:...]`` is written; ``finish()`` waits and returns the
        pad.
        """
        pad = self.pads[kind]

        def send(m):
            cp = self._cols_copy(kind, src_ref, width, m)
            pl.when(self.me != m)(cp.start)

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


def expert_units(c):
    """Expert operand units in order: ("layer", l) with 4 refs, or ("group", lo, hi) with 4

    stacked refs.
    """
    units, l = [], 0
    starts = {lo: hi for lo, hi in c.expert_groups}
    while l < c.n_layers:
        if l in starts:
            units.append(("group", l, starts[l]))
            l = starts[l]
        else:
            units.append(("layer", l))
            l += 1
    return units


def n_expert_refs(c):
    return 4 * len(expert_units(c))


def route_blocks(c):
    """Lane blocks per vec_route field: (row, slot) pairs and distinct experts both fit."""
    return -(-c.rows * c.top_k // LANE)


def route_rows(c):
    """vec_route table rows: uniq, count, offset, perm fields, then n_u, then blk_end (and blk_mid)."""
    return -(-(4 * route_blocks(c) + 2 + bool(c.split_rows)) // 8) * 8


def _route_table(ids_list, n_live, K, Ep, NB, pair_key, n_keys, split=0):
    """vec_route table ``[route_rows, LANE]`` int32 from per-slot expert ids ``K x [R, 1]``.

    Distinct experts are ordered by (the smallest ``pair_key`` [P, 1] of their
    pairs, id), pair
    q = k*R + r. Rows ``[f*NB, (f+1)*NB)`` hold field f: 0 experts, 1 their live
    (row, slot)
    counts, 2 offsets into field 3, 3 pairs as ``row * K + slot`` grouped by
    expert; row 4*NB
    the distinct count; row 4*NB+1 lane b the experts whose key is <= b. With
    ``split``, experts of more than ``split`` pairs sort after the others of their
    key, and row 4*NB+2 lane b counts the experts before key b's large ones.
    Prefix sums run on the MXU (0/1 and counts <= R*K are exact in bf16, summed in
    f32).
    """
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
    mult = 2 if split else 1
    if split:
        key = 2 * key + jnp.where(count > split, 1.0, 0.0)
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
    key_c, id_c, cnt_c, pres_c = (
        cols[:, 0:1],
        cols[:, 1:2],
        cols[:, 2:3],
        cols[:, 3:4],
    )
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
        jnp.where((pres_c > 0) & (key_c <= mult * b_lane + mult - 1), 1, 0),
        axis=0,
        keepdims=True,
    )
    tail = [jnp.broadcast_to(n_u, (1, LANE)), blk_end]
    if split:
        tail.append(
            jnp.sum(
                jnp.where((pres_c > 0) & (key_c <= 2 * b_lane), 1, 0),
                axis=0,
                keepdims=True,
            )
        )
    n_rows = -(-(4 * NB + len(tail)) // 8) * 8
    return jnp.concatenate(
        rows + tail + [jnp.zeros((n_rows - 4 * NB - len(tail), LANE), jnp.int32)],
        axis=0,
    )


def compact_buckets(c):
    """Row buckets of the compact expert variants, ascending: an expert with n live pairs runs

    the smallest bucket >= n (busier experts take the full path).
    """
    arms = c.compact_arms
    if (
        not arms
        and c.ex_opt
        and c.compact_rows == 8
        and c.rows <= 64
        and len(expert_units(c)) <= 8
    ):
        arms = (1, 2, 4, 6, 8)
    if arms:
        # buckets past the first that covers every row are unreachable (one pair per row)
        cover = [b for b in arms if b >= c.rows][:1]
        return tuple(b for b in arms if b < c.rows) + tuple(cover)
    return tuple(b for b in (1, 2, 4, 8) if b <= c.compact_rows)


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


# row-tiled program: semaphores of tile_sems
_TS_AUX, _TS_OUT, _TS_LOCAL, _TS_ROPE = 0, 1, 2, 3
_TS_N = 3 + 8
# row-tiled collectives: (send, receive) semaphore index of each kind; o_rs sends alternate
# between two (or four, with tiled_o_rot) buffers, each with its own send semaphore
_TC = {
    "o_rs": 0,
    "o_rs1": 1,
    "o_ag": 2,
    "pool": 3,
    "y": 4,
    "u": 5,
    "f": 6,
    "o_rs2": 7,
    "o_rs3": 8,
}
_TC_VPADS_S = {"o_rs": 0, "o_rs1": 1, "o_rs2": 2, "o_rs3": 3, "y": 4}
_TC_VPADS_R = {"o_rs": 0, "y": 1}
# the weight rings, small slabs and attention buffers, the same in both programs
_SHARED_SCRATCH = (
    "ring",
    "ring_sems",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "w13_sems",
    "w13s_sems",
    "w2_sems",
    "w2s_sems",
    "smalls",
    "small_sems",
    "final_norm",
    "kv_chunk",
    "kv_chunk_sems",
    "attn_p",
    "attn_psc",
    "kv_row",
    "kv_row_sem",
)


def _can_w2_attn_scratch(c: MKConfig) -> bool:
    return bool(
        c.attn_head_fp8 >= 1
        and c.attn_pipe
        and c.attn_opt in (3, 4, 5)
        and c.kv_sal
        and c.kv_sal_fp8
        and c.sal_fp8_dot >= 2
        and c.attn_exp2
        and not c.use_sal_preones
        and c.kv_heads == 2
        and c.head_dim == 64
        and 2 * c.kv_heads * c.rows_per_req * c.g > LANE
        and c.rows_per_req * c.g <= 64
        and c.chunk_pages * c.page_size == 1024
        and (c.expert_depth + 1) * (c.inter_local // FP4_PACK) >= 192
        and c.expert_hidden >= 4 * c.attn_streams * LANE
    )


def _use_attn_p_fp8(c: MKConfig) -> bool:
    return bool(
        (c.attn_p_fp8 or (c.attn_head_fp8 >= 2 and _can_w2_attn_scratch(c)))
        and c.attn_pipe
        and c.attn_opt in (3, 4, 5)
        and c.kv_sal
        and c.kv_sal_fp8
        and c.sal_fp8_dot >= 2
        and not (c.use_sal_preones and c.kv_heads * c.rows_per_req * c.g > 64)
    )


def _attn_p_rows(c: MKConfig) -> int:
    m_raw = c.kv_heads * c.rows_per_req * c.g
    if _can_w2_attn_scratch(c) and m_raw > 64:
        return 128
    if _use_attn_p_fp8(c):
        return -(-m_raw // 32) * 32
    return m_raw


def _attn_p_ysend_streams(c: MKConfig) -> int:
    """How many piped-attention streams can borrow ysend (disjoint lifetimes)."""
    cpt = c.chunk_pages * c.page_size
    yw = y_words(c)
    if not (
        c.alias_p
        and (c.row_tile == 0 or c.tiled_vmem_y >= 1)
        and c.attn_pipe
        and c.kv_sal
        and cpt <= yw
        and cpt % LANE == 0
        and c.kv_heads * c.g == 2 * c.top_k
    ):
        return 0
    m_pad = _attn_p_rows(c)
    if _use_attn_p_fp8(c):
        cap = (4 * c.top_k * c.rows) // m_pad
        return min(c.attn_streams, cap if 2 * cpt <= yw else cap // 2)
    cap = (2 * c.top_k * c.rows) // m_pad
    return min(c.attn_streams, cap if 2 * cpt <= yw else cap // 2)


def _alias_attn_p_ysend(c: MKConfig) -> bool:
    """Whether all piped-attention probabilities borrow ysend."""
    return c.attn_pipe and _attn_p_ysend_streams(c) == c.attn_streams


def _alias_pads_into_y(c: MKConfig) -> bool:
    """Whether pad_o_rs and pad_o_ag can borrow pad_y (pool_ag + f_ag separate lifetimes)."""
    few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max
    return (
        c.alias_oy
        and (
            (c.row_tile == 0 and c.ex_opt and not few)
            or (c.row_tile > 0 and c.tiled_vmem_pads and c.tiled_vmem_y >= 2)
        )
        and c.rows_per_dev * c.top_k == c.rows
        and c.top_k == 8
        and y_words(c) >= 2 * (c.hidden_local // 2)
    )


def _alias_oag_into_f(c: MKConfig) -> bool:
    """Whether pad_o_ag can borrow pad_f when pad_y is in HBM (vy < 2)."""
    return bool(
        c.alias_oag_f
        and c.alias_oy
        and c.row_tile > 0
        and c.tiled_vmem_pads
        and not _alias_pads_into_y(c)
    )


def _alias_tiled_xfull_q(c: MKConfig) -> bool:
    return bool(c.row_tile and c.alias_tiled_temps and c.q_dim <= c.expert_hidden // 2)


def _alias_tiled_padu_attn(c: MKConfig) -> bool:
    return bool(
        c.row_tile
        and c.alias_tiled_temps
        and c.tiled_vmem_pads
        and c.rows_per_dev % 8 == 0
        and c.tp * c.rows_per_dev == c.rows
        and c.q_dim <= c.expert_hidden // 2
    )


def _alias_tiled_ids_kv(c: MKConfig) -> bool:
    return bool(c.row_tile and c.alias_tiled_temps and c.kv_dim == LANE)


def _alias_tiled_obuf_ysend(c: MKConfig) -> bool:
    ob_n = 4 if (c.tiled_o_rot or c.pair_o_rs) else 2
    return bool(
        c.row_tile
        and c.alias_tiled_temps
        and c.tiled_vmem_y >= 1
        and (c.rows * c.top_k) % (ob_n * 8) == 0
        and (c.rows * c.top_k) // ob_n >= c.rows
        and y_words(c) >= c.hidden_local // 2
    )


def scratch_plan(cfg: MKConfig):
    return _scratch_plan_tiled(cfg) if cfg.row_tile else _scratch_plan_base(cfg)


def _scratch_plan_tiled(c: MKConfig) -> list:
    """Row-tiled scratch: per-row VMEM is the matmul LHS, accumulators and small per-row

    vectors; everything [R, H]-sized or [R, top_k, He]-sized lives in HBM unless
    tiled_vmem_pads / tiled_vmem_y keep them in VMEM.
    """
    R, K, H, RT = c.rows, c.top_k, c.hidden, c.row_tile
    He, Hl, Hel, tp, Rc = (
        c.expert_hidden,
        c.hidden_local,
        c.expert_hidden_local,
        c.tp,
        c.rows_per_dev,
    )
    yw = y_words(c)
    hbm, vmem, smem = pltpu.HBM, pltpu.VMEM, pltpu.SMEM
    u32, f32 = jnp.uint32, jnp.float32
    nsc = 8 if c.vec_route else R * K + 8
    base = dict(_scratch_plan_base(c))
    use_p_fp8 = _use_attn_p_fp8(c)
    if _alias_attn_p_ysend(c):
        base["attn_p"] = vmem(
            (1, 1, 32 if use_p_fp8 else 8, LANE),
            jnp.float8_e4m3fn if use_p_fp8 else jnp.bfloat16,
        )
    vpads = c.tiled_vmem_pads
    vy = c.tiled_vmem_y
    alias_oy = _alias_pads_into_y(c)
    pmem = vmem if vpads else hbm
    ob_n = 4 if (c.tiled_o_rot or c.pair_o_rs) else 2
    items = [(n, base[n]) for n in _SHARED_SCRATCH] + [
        ("rtab", vmem((R, 2 * c.head_dim), f32)),
        ("rope_blk", vmem((8, 8, 2 * c.head_dim), f32)),
        ("lhs", vmem((R, H), jnp.bfloat16)),
        ("acc", vmem((R, max(c.qkvg_dim, Hl, Hel, c.n_experts_padded)), f32)),
    ]
    if not _alias_tiled_xfull_q(c):
        items += [("qn_s", vmem((R, c.q_dim), f32))]
    if not _alias_tiled_padu_attn(c):
        items += [("attn_buf", vmem((R, c.q_dim), f32))]
    if not _alias_tiled_ids_kv(c):
        items += [
            ("kb_s", vmem((R, c.kv_dim), f32)),
            ("vb_s", vmem((R, c.kv_dim), f32)),
        ]
    items += [
        ("ids_v", vmem((R, LANE), jnp.int32)),
        ("wsel_v", vmem((R, LANE), f32)),
        ("wk_s", vmem((owner_window_rows(c), K, LANE), f32)),
        ("xfull", vmem((R, He // 2), u32)),
        ("pool_send", vmem((R, Hel // 2), u32)),
    ]
    if not _alias_tiled_obuf_ysend(c):
        items += [("o_buf", vmem((ob_n, R, Hl // 2), u32))]
    items += [
        ("o_send", vmem((R, Hl // 2), u32)),
        ("f_send", vmem((R, Hl // 2), u32)),
        ("u_send", vmem((Rc, He // 2), u32)),
    ]
    if not vpads:
        items += [("bin", vmem((2, RT, H), jnp.bfloat16))]
    items += [
        ("btile", vmem((RT, H), jnp.bfloat16)),
        ("tile_sems", pltpu.SemaphoreType.DMA((_TS_N,))),
        # outstanding expert-output row DMAs per staging buffer, then the emit count
        ("ypend", smem((8,), jnp.int32)),
        # the scalar routing tables (vec_route builds route_s instead)
        ("uniq", smem((nsc,), jnp.int32)),
        ("ex_cnt", smem((nsc,), jnp.int32)),
        ("ex_off", smem((nsc,), jnp.int32)),
        ("ex_pos", smem((nsc,), jnp.int32)),
        ("perm", smem((nsc,), jnp.int32)),
        ("ex_of", smem((c.n_experts_padded,), jnp.int32)),
        ("ids_s", smem((8 if c.vec_route else R, LANE), jnp.int32)),
        ("seen", smem((c.n_experts_padded,), jnp.int32)),
        ("blk_end", smem((max(c.tp, 8),), jnp.int32)),
        ("route_v", vmem((route_rows(c), LANE), jnp.int32)),
        ("route_s", smem((route_rows(c), LANE), jnp.int32)),
        ("route_sem", pltpu.SemaphoreType.DMA((1,))),
        ("h_stream", pmem((R, H), f32)),
    ]
    if not (c.vec_route and c.trim_sems):
        items += [("ids_sem", pltpu.SemaphoreType.DMA((1,)))]
    if c.ex_gs_smem and (c.scale_fp8 or c.ex_kperm >= 3):
        items += [("gs_s", smem((8, c.n_experts_padded), f32))]
    if not alias_oy:
        items += [("pad_o_rs", pmem((tp, R, Hl // 2), u32))]
        if not _alias_oag_into_f(c):
            items += [("pad_o_ag", pmem((tp, R, Hl // 2), u32))]
    n_tc_s = len(_TC_VPADS_S) if (vpads and c.trim_sems) else len(_TC)
    n_tc_r = len(_TC_VPADS_R) if (vpads and c.trim_sems) else len(_TC)
    items += [
        ("pad_pool", pmem((tp, R, Hel // 2), u32)),
        (
            "pad_y",
            (vmem if vy >= 2 else hbm)((tp, Rc, K, yw), u32),
        ),
        (
            "pad_u",
            vmem((tp, Rc, He // 2), u32) if vpads else hbm((R, He // 2), u32),
        ),
        ("pad_f", pmem((tp, R, Hl // 2), u32)),
        ("tc_ssems", pltpu.SemaphoreType.DMA((n_tc_s,))),
        ("tc_rsems", pltpu.SemaphoreType.DMA((n_tc_r,))),
    ]
    if not (vpads and c.trim_sems):
        items += [
            ("pass_sems", pltpu.SemaphoreType.DMA((2, 4))),
            # pipelined o / f all-gathers: per row tile, each sender's copy signals it
            ("oag_tsems", pltpu.SemaphoreType.DMA((R // RT,))),
            ("fag_tsems", pltpu.SemaphoreType.DMA((R // RT,))),
            # pipelined u all-gather: per owner
            ("uag_osems", pltpu.SemaphoreType.DMA((tp,))),
        ]
    if not vpads:
        items += [
            ("sin", vmem((2, RT, H), f32)),
            ("sout", vmem((2, RT, H), f32)),
            ("ptile", vmem((2, tp, RT, Hl // 2), u32)),
            ("utile", vmem((2, RT, He // 2), u32)),
            ("pltile", vmem((2, tp, RT, Hel // 2), u32)),
        ]
    else:
        n_coll_sems = 4 if c.trim_sems else 8
        items += [
            ("coll_ssems", pltpu.SemaphoreType.DMA((n_coll_sems,))),
            ("coll_rsems", pltpu.SemaphoreType.DMA((n_coll_sems,))),
        ]
        if c.tiled_pair_y or c.pair_ag or c.pair_o_rs:
            items += [
                ("pair_ssems", pltpu.SemaphoreType.DMA((7,))),
                ("pair_rsems", pltpu.SemaphoreType.DMA((7,))),
            ]
    if c.fold_f or c.fold_o:
        items += [
            ("fold_v", vmem((8, H), f32)),
            ("o_rsems", pltpu.SemaphoreType.DMA((tp,))),
            ("f_rsems", pltpu.SemaphoreType.DMA((tp,))),
        ]
    if vy >= 1:
        items += [("ysend", vmem((R, K, yw), u32))]
    else:
        items += [
            ("ystage", vmem((2, c.expert_chunk, yw), u32)),
            ("ystage_sems", pltpu.SemaphoreType.DMA((2,))),
        ]
    if vy < 2:
        items += [
            ("yrow", vmem((2, tp, 1, K, yw), u32)),
            ("yrow_sems", pltpu.SemaphoreType.DMA((2,))),
        ]
    return items


def _scratch_plan_base(cfg: MKConfig):
    c = cfg
    R, K, H = c.rows, c.top_k, c.hidden
    He, Il = c.expert_hidden, c.inter_local
    yw = y_words(c)
    use_p_fp8 = _use_attn_p_fp8(c)
    p_dt = jnp.float8_e4m3fn if use_p_fp8 else jnp.bfloat16
    p_rows = _attn_p_rows(c)
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
        (
            "w2s",
            pltpu.VMEM(
                (nslot, max(Il // FP4_GROUP, 32 if c.scale_fp8 else 16), He), sdt
            ),
        ),
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
        ("ysend", pltpu.VMEM((R, K, yw), jnp.uint32)),
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
            # SAL chunks live in the expert w13 slots (sal_kv_view)
            pltpu.VMEM((1, 8, LANE), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM(
                (
                    c.kv_ring,
                    c.chunk_pages,
                    c.page_size,
                    c.pool_kv_heads,
                    2,
                    c.pool_padded_head_dim,
                ),
                jnp.bfloat16,
            ),
        ),
        # one per (sub-)slot: a chunk's page copies all signal it, one wait takes the chunk
        (
            "kv_chunk_sems",
            pltpu.SemaphoreType.DMA(
                (
                    max(c.kv_ring, 2 * c.swa_kv_group)
                    if c.swa_chunk_pages
                    else c.kv_ring,
                )
            ),
        ),
        # attn_pipe: probabilities of the chunks in flight, double-buffered by loop iteration
        (
            "attn_p",
            pltpu.VMEM((1, 1, 32 if use_p_fp8 else 8, LANE), p_dt)
            if _alias_attn_p_ysend(c)
            else pltpu.VMEM(
                (
                    2,
                    (c.attn_streams - _attn_p_ysend_streams(c)) if c.attn_pipe else 1,
                    p_rows,
                    cpt if c.attn_pipe else LANE,
                ),
                p_dt,
            ),
        ),
        (
            "attn_psc",
            pltpu.VMEM(
                (
                    2,
                    c.attn_streams,
                    -(-(c.kv_heads * c.rows_per_req * c.g) // 8) * 8,
                    LANE,
                ),
                jnp.float32,
            )
            if (use_p_fp8 and not (c.attn_head_fp8 >= 2 and _can_w2_attn_scratch(c)))
            else pltpu.VMEM((1, 1, 8, LANE), jnp.float32),
        ),
        (
            "kv_row",
            pltpu.VMEM((c.sal_words, R), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM(
                (R, c.pool_kv_heads, 2, c.pool_padded_head_dim), jnp.bfloat16
            ),
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
        ("ids_v", pltpu.VMEM((R, LANE), jnp.int32)),
        ("ids_s", pltpu.SMEM((R, LANE), jnp.int32)),
        ("ids_sem", pltpu.SemaphoreType.DMA((1,))),
        ("seen", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        # y_block_send: uniq index past the last expert first used by owner block b
        ("blk_end", pltpu.SMEM((max(c.tp, 8),), jnp.int32)),
        ("route_v", pltpu.VMEM((route_rows(c), LANE), jnp.int32)),
        ("route_s", pltpu.SMEM((route_rows(c), LANE), jnp.int32)),
        ("route_sem", pltpu.SemaphoreType.DMA((1,))),
        # collective pads: bf16 pairs packed into u32 (_pack_bf16x2)
        (
            "pad_pool",
            pltpu.VMEM((c.tp, R, c.expert_hidden_local // 2), jnp.uint32),
        ),
        ("pad_y", pltpu.VMEM((c.tp, c.rows_per_dev, K, yw), jnp.uint32)),
        ("pad_u", pltpu.VMEM((c.tp, c.rows_per_dev, He // 2), jnp.uint32)),
        ("pad_f", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("coll_ssems", pltpu.SemaphoreType.DMA((8,))),
        ("coll_rsems", pltpu.SemaphoreType.DMA((8,))),
    ]
    if not (c.ex_opt and c.topk_iter):
        items += [("rank", pltpu.VMEM((R, c.n_experts_padded), jnp.int32))]
    if not _alias_pads_into_y(c):
        items += [
            ("pad_o_rs", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
            ("pad_o_ag", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ]
    if not (
        c.ex_opt
        and not (c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max)
    ):
        items += [
            ("row1_send", pltpu.VMEM((few_rows(c), H // 2), jnp.uint32)),
            ("pad_o1", pltpu.VMEM((c.tp, few_rows(c), H // 2), jnp.uint32)),
            (
                "pad_f1",
                pltpu.VMEM((c.tp, few_rows(c), c.hidden_local // 2), jnp.uint32),
            ),
        ]
    if c.pair_ag >= 1:
        items += [
            ("pair_ssems", pltpu.SemaphoreType.DMA((8,))),
            ("pair_rsems", pltpu.SemaphoreType.DMA((8,))),
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
        ]
    if c.scale_fp8 or c.ex_kperm >= 3:
        items += [
            # this layer's (gate, up, down) globals, one expert per lane
            ("gs_v", pltpu.VMEM((8, c.n_experts_padded), jnp.float32)),
        ]
        if c.ex_gs_smem:
            items += [("gs_s", pltpu.SMEM((8, c.n_experts_padded), jnp.float32))]
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
    (w13, w13_scale, w2, w2_scale) in layer order. Outputs: hidden ``[R, H]``
    bf16, aux
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
    # HBM scratch (row-tiled steps) is passed in as extra pallas_call outputs: Mosaic only
    # allocates VMEM, SMEM and semaphore scratch
    plan = scratch_plan(c)
    hbm_scratch = [(n, s) for n, s in plan if _is_hbm(s)]
    scratch = [(n, s) for n, s in plan if not _is_hbm(s)]
    hnames = [n for n, _ in hbm_scratch]
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
    kind_layers = {}
    for _l, _k in enumerate(c.expert_kinds):
        kind_layers.setdefault(_k, []).append(_l)

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, emb_ref, *rest):
        W = dict(zip(WEIGHT_NAMES, rest[:nw]))
        rest = rest[nw:]
        NE = n_expert_refs(c)
        EX = rest[:NE]
        rest = rest[NE:]
        # layer -> (its 4 expert refs, its index in stacked refs or None)
        layer_ex = {}
        for ui, unit in enumerate(expert_units(c)):
            refs = [EX[4 * ui + j] for j in range(4)]
            if unit[0] == "layer":
                layer_ex[unit[1]] = (refs, None)
            else:
                for li in range(unit[1], unit[2]):
                    layer_ex[li] = (refs, li - unit[1])
        group_lo = {lo: hi for lo, hi in c.expert_groups}
        KV = rest[:L]  # read-only here; see kv_write_call
        hid_out, aux_out, newkv_out = rest[L], rest[L + 1], rest[L + 2]
        nh = len(hnames)
        S = dict(zip(hnames, rest[L + 3 : L + 3 + nh]))
        S.update(zip(snames, rest[L + 3 + nh :]))
        yw = y_words(c)
        if _alias_pads_into_y(c):
            py3 = S["pad_y"].reshape(c.tp, R, yw)
            S["pad_o_rs"] = py3.at[:, :, pl.ds(0, Hl // 2)]
            S["pad_o_ag"] = py3.at[:, :, pl.ds(Hl // 2, Hl // 2)]
        elif _alias_oag_into_f(c):
            S["pad_o_ag"] = S["pad_f"]
        if _alias_tiled_xfull_q(c):
            S["qn_s"] = S["xfull"].bitcast(jnp.float32).at[:, pl.ds(0, c.q_dim)]
        if _alias_tiled_padu_attn(c):
            S["attn_buf"] = (
                S["pad_u"]
                .reshape(R, He // 2)
                .bitcast(jnp.float32)
                .at[:, pl.ds(0, c.q_dim)]
            )
        if _alias_tiled_ids_kv(c):
            S["kb_s"] = S["ids_v"].bitcast(jnp.float32)
            S["vb_s"] = S["wsel_v"]
        if _alias_tiled_obuf_ysend(c):
            ob_n = 4 if (c.tiled_o_rot or c.pair_o_rs) else 2
            S["o_buf"] = (
                S["ysend"]
                .reshape(ob_n, (R * K) // ob_n, yw)
                .at[:, pl.ds(0, R), pl.ds(0, Hl // 2)]
            )
        n_live = meta_ref[0]
        tp_rank = meta_ref[1]
        me = jax.lax.axis_index(axis) if c.tp > 1 else 0
        can_pair_fold = bool(
            c.pair_ag >= 1
            and c.tp == 8
            and all(_fold_g(c, g) == 2 for g in fold_groups(c))
        )
        if c.row_tile and c.tiled_vmem_pads and c.trim_sems:
            coll_pads = {
                "o_ag": S["pad_o_ag"],
                "pool": S["pad_pool"],
                "u": S["pad_u"],
                "f": S["pad_f"],
            }
        else:
            coll_pads = {
                "o_rs": S["pad_o_rs"],
                "o_ag": S["pad_o_ag"],
                "pool": S["pad_pool"],
                "y": S["pad_y"],
                "u": S["pad_u"],
                "f": S["pad_f"],
                # absent in row-tiled steps, which run their own collectives
                "o1": S.get("pad_o1"),
                "f1": S.get("pad_f1"),
            }
        coll = _Coll(
            c.tp,
            me,
            axis,
            coll_pads,
            S.get("coll_ssems"),
            S.get("coll_rsems"),
            order=c.coll_order,
            local_last=c.coll_local_last,
            pair_ag=c.pair_ag,
            pair_ssems=S.get("pair_ssems"),
            pair_rsems=S.get("pair_rsems"),
            can_pair_fold=can_pair_fold,
        )
        # rotated o-proj columns: this device's own reduce-scatter block streams last
        F = few_rows(c)
        few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max
        # o-proj column tile m is reduce-scatter block m: ship each as it finishes
        o_rs_streamed = (
            c.o_rs_streamed
            and c.tp > 1
            and not few
            and all(tn == Hl for name, _, _, _, tn, _ in dense_tiles(c) if name == "wo")
        )
        use_pair_o_rs = bool(
            c.row_tile
            and c.pair_o_rs
            and c.tiled_vmem_pads
            and c.tp == 8
            and c.expert_hidden == 4 * c.hidden_local
        )
        # (row-tiled steps ship o-proj column tile m to device m themselves)
        o_rot = (
            o_rs_streamed
            and c.o_rs_rotate
            and (not c.row_tile or (c.tiled_o_rot and not use_pair_o_rs))
        )
        k_rot = {}
        for grp in fold_groups(c):
            # f blocks sit in pad slots by TP rank, o blocks by mesh index
            own = me if can_pair_fold else (tp_rank if grp == ("wqkvg",) else me)
            for name, _, _, tk, _, _ in dense_tiles(c):
                if name in grp:
                    k_rot[name] = (own, _fold_g(c, grp), _fold_g(c, grp) * Hl // tk)
        ws = _WeightStream(
            W,
            S["ring"],
            S["ring_sems"],
            dense_tiles(c),
            c.stream_depth,
            col_rot={"wo": me + 1} if o_rot else None,
            k_rot=k_rot,
            fp8_lhs=c.dense_fp8_lhs,
        )
        msem = S.get("misc_sems")

        def clear_seen(e, carry):
            S["seen"][e] = 0
            return carry

        if not c.vec_route:
            jax.lax.fori_loop(0, c.n_experts_padded, clear_seen, 0)
        # Entry barrier: no peer's remote write lands in this device's VMEM before it has
        # entered the kernel. The call aliases nothing, so its collective_id costs no copy.
        if c.tp > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, c.tp):
                pl.semaphore_signal(
                    barrier,
                    device_id=(jnp.mod(me + d, c.tp),),
                )
            pl.semaphore_wait(barrier, c.tp - 1)

        def single_row(x, multi, one):
            """``multi(x)`` -> [R, H], or in a step of at most few_rows_max rows ``one(x)``'s [F, H]

            rows (the rest zero).
            """
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

            Partials travel as bf16 and are summed in f32, as the native bf16
            all-reduce.
            """
            for d in range(c.tp):
                S["xpack"][:, d * Hl // 2 : (d + 1) * Hl // 2] = _pack_bf16x2(
                    x[:, d * Hl : (d + 1) * Hl]
                )
            return allgather_sum(coll.reduce_scatter_cols("o_rs", S["xpack"], Hl // 2))

        def allgather_sum(pad):
            """Reduce-scattered column blocks in pad -> summed, all-gathered [R, H] (fold_o:

            the all-gather is only started).
            """
            mine = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                mine = mine + _unpack_bf16x2(pad[d])
            S["o_send"][...] = _pack_bf16x2(mine)
            if c.fold_o:
                coll.all_gather_blocks_start("o_ag", S["o_send"], me, S["o_rsems"])
                return None
            with _scope("o_ag"):
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

            arriving; returns (qkvg, layer l's input stream, exact).
            """
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

            o-proj all-gather still arriving; x is the layer's input stream.
            """
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

        if not c.row_tile:  # the row-tiled program streams these from HBM
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
                    jnp.sum(
                        jnp.where(sel, S["rope_blk"][b], 0.0), axis=0, keepdims=True
                    )
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

            in flight (fold_pool_router consumes it).
            """
            per_geom(is_global, lambda geo: kv_prologue(geo, l, 0))
            fill_preones()
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
            own_dt = jnp.bfloat16 if c.debug_own_kv_bf16 else c.kv_dtype
            kb = kn.astype(own_dt).astype(jnp.bfloat16)
            vb = vv.astype(own_dt).astype(jnp.bfloat16)
            zpad = jnp.zeros((R, pd - hd), jnp.bfloat16)
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

                def run_geom(geo):
                    def req_loop(j, carry):
                        request_attention_body(geo, l, j, n_req_live)
                        return carry

                    jax.lax.fori_loop(0, n_req_live, req_loop, 0)

                per_geom(is_global, run_geom)
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
                with _scope("o_mv"):
                    send, finish = coll.reduce_scatter_cols_streamed(
                        "o_rs", S["xpack"], Hl // 2
                    )

                    def ship(m, col):
                        w2 = Hl // 2
                        c0 = (
                            m * w2
                            if isinstance(m, int)
                            else pl.multiple_of(m * w2, LANE)
                        )
                        S["xpack"][:, pl.ds(c0, w2)] = _pack_bf16x2(col)
                        send(m)

                    ws.mv(
                        attn,
                        l,
                        "wo",
                        nxt,
                        sm=sm,
                        on_col=ship,
                        sm_ref=S["smalls"].at[l % 2],
                    )
                with _scope("o_allreduce"):
                    with _scope("o_rs_wait"):
                        pad = finish()
                    return allgather_sum(pad)
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

        KR = c.kv_ring
        KG = KR // 2
        geo_global = _AttnGeom(c.chunk_pages, KG, KR, True)
        if c.swa_chunk_pages:
            geo_swa = _AttnGeom(
                c.swa_chunk_pages, c.swa_kv_group, 2 * c.swa_kv_group, False
            )

        def per_geom(is_global, fn):
            """``fn(geometry)`` for a layer: one shared body, or with swa_chunk_pages a body per layer type."""
            if not c.swa_chunk_pages:
                fn(_AttnGeom(c.chunk_pages, KG, KR, is_global))
            elif isinstance(is_global, bool):
                fn(geo_global if is_global else geo_swa)
            else:
                pl.when(is_global)(lambda: fn(geo_global))
                pl.when(jnp.logical_not(is_global))(lambda: fn(geo_swa))

        def ring_at(geo, slot, pi):
            """Ring entry and page of page ``pi`` of chunk slot ``slot``."""
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
                pages.append(_sel(geo.is_global, pf_ref[ent], psw_ref[ent]))
            return pages

        def issue_range(geo, l, j, ci0, slot0, n):
            """Chunks ``ci0 ..

            ci0+n`` of request j into ring slots ``slot0 ..``: ONE layer branch-tree
            walk whose leaf is a small runtime loop, so the 62 leaves stay small
            (unrolled per-chunk code in every leaf bloats the kernel).
            """

            def leaf(ref):
                def one(u, carry):
                    slot = (slot0 + u) % geo.kr
                    for pi, page in enumerate(chunk_pages_of(geo, j, ci0 + u)):
                        pltpu.make_async_copy(
                            page_src(ref, page),
                            chunk_dst(geo, slot, pi),
                            S["kv_chunk_sems"].at[slot],
                        ).start()
                    return carry

                jax.lax.fori_loop(0, n, one, 0)

            only = None
            if isinstance(geo.is_global, bool) and c.swa_chunk_pages:
                only = [x for x in range(c.n_layers) if (x in gl) == geo.is_global]
            _dispatch(l, KV, leaf, only)

        def kv_ring_ref():
            """The KV chunk ring.

            SAL: the expert w13 slots, idle during attention (every expert DMA of the
            previous layer was waited, this layer's start after attention), as [chunk,
            sal_words, chunk tokens] u32 in the native tiling: the page DMA relayouts
            the pool's (pk, 128) tiles, so chunk loads are whole vregs.
            """
            if c.kv_sal:
                return S["w13"].reshape(c.kv_ring, c.sal_words, c.chunk_pages * ps)
            return S["kv_chunk"]

        use_po = c.use_sal_preones

        def w13_flat():
            return S["w13"].reshape(
                (c.expert_depth + 1) * (c.expert_hidden // FP4_PACK), cpt
            )

        def po_row(slot, off=0):
            if isinstance(slot, int):
                return slot * PREONES_ROWS + off
            return pl.multiple_of(slot * PREONES_ROWS + off, 8)

        def ring_words(geo, slot, off=0, n=None):
            """u32 rows [off, off + n) of a ring chunk (n: sal_words) as a ref."""
            n = c.sal_words if n is None else n
            e, p = ring_at(geo, slot, 0)
            if use_po:
                if geo.cp == c.chunk_pages:
                    return w13_flat().at[pl.ds(po_row(e, off), n), :]
                return w13_flat().at[
                    pl.ds(po_row(e, off), n),
                    pl.ds(pl.multiple_of(p * ps, LANE), geo.cp * ps),
                ]
            if geo.cp == c.chunk_pages:
                if off == 0 and n == c.sal_words:
                    return kv_ring_ref().at[e]
                return kv_ring_ref().at[e, pl.ds(off, n), :]
            if off == 0 and n == c.sal_words:
                return kv_ring_ref().at[
                    e, :, pl.ds(pl.multiple_of(p * ps, LANE), geo.cp * ps)
                ]
            return kv_ring_ref().at[
                e, pl.ds(off, n), pl.ds(pl.multiple_of(p * ps, LANE), geo.cp * ps)
            ]

        def fill_preones():
            """The fp8 ones rows of every attn_preones ring chunk (page DMAs never touch them)."""
            if not use_po:
                return
            ones = jnp.full((8, cpt), _FP8_ONES_WORD, jnp.uint32)
            wf = w13_flat()
            for sl in range(KR):
                wf[pl.ds(sl * PREONES_ROWS + c.sal_words, 8), :] = ones

        def chunk_dst(geo, slot, pi):
            if use_po:
                e, p = ring_at(geo, slot, pi)
                return w13_flat().at[pl.ds(po_row(e), c.sal_words), pl.ds(p * ps, ps)]
            if c.kv_sal:
                e, p = ring_at(geo, slot, pi)
                return kv_ring_ref().at[e, :, pl.ds(p * ps, ps)]
            return S["kv_chunk"].at[slot, pi]

        def page_src(ref, page):
            if c.kv_sal:  # [2 * kv_heads, hd // pk, pk, page] -> its 32-bit words, tokens on lanes
                return ref.at[page].bitcast(jnp.uint32).reshape(c.sal_words, ps)
            return ref.at[page]

        def wait_chunk(geo, slot):
            if use_po or c.kv_sal:
                r = ring_words(geo, slot)
                pltpu.make_async_copy(r, r, S["kv_chunk_sems"].at[slot]).wait()
                return
            ring = kv_ring_ref()
            pltpu.make_async_copy(
                ring.at[slot], ring.at[slot], S["kv_chunk_sems"].at[slot]
            ).wait()

        def kv_prologue(geo, l, j):
            # the request's first geo.kr chunks
            _, first_ch, n_ch = req_geom(geo, j)
            issue_range(geo, l, j, first_ch, 0, jnp.minimum(n_ch, geo.kr))

        def rows_at(ref, r0, n):
            """``n`` rows from a dynamic start: one-row accesses, as a multi-row dynamic sublane

            window must start at a multiple of 8.
            """
            return (
                jnp.concatenate([ref[pl.ds(r0 + t, 1), :] for t in range(n)], axis=0)
                if n > 1
                else ref[pl.ds(r0, 1), :]
            )

        def request_attention_body(geo, l, j, n_req_live):
            """Both local KV heads in one MXU op per step: the query rows of all heads are

            stacked ``[hkv*q*g, .]`` (rows ``(h, t, gi)``). Padded pools: scored
            against the
            chunk's interleaved key rows ``(token, h)``, scores of the other head
            masked. Packed
            pools (kv_packed): a token's one row holds every head side by side and a
            query row
            carries its q in its head's lanes, zeros elsewhere, so the op scores only
            same-head
            pairs: one key row per token and no head mask.
            """
            is_global, KR_g, cpt_g = geo.is_global, geo.kr, geo.cp * ps
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

            q2_raw = jnp.concatenate(
                [
                    qrow(h, t, gi)
                    for h in range(hkv)
                    for t in range(q)
                    for gi in range(g)
                ],
                axis=0,
            )  # [hkv*qg, hd or pd] f32
            if c.attn_exp2:  # scores in log2 units: p = exp2(score)
                q2, sc = (q2_raw * (c.attn_scale * LOG2E)).astype(jnp.bfloat16), 1.0
            elif c.attn_unshifted or c.fold_attn_scale:  # power-of-two scale: exact
                q2 = q2_raw.astype(jnp.bfloat16) * jnp.asarray(
                    c.attn_scale, jnp.bfloat16
                )
                sc = 1.0
            else:
                q2, sc = q2_raw.astype(jnp.bfloat16), c.attn_scale
            rowpos = jnp.concatenate(
                [jnp.broadcast_to(pos_ref[r0 + t], (g, 1)) for t in range(q)] * hkv,
                axis=0,
            )  # [hkv*qg, 1]
            rowhead = jax.lax.broadcasted_iota(jnp.int32, (hkv * qg, 1), 0) // qg

            # unshifted: m is unused and the chunk loops keep [rows, LANE] lane partials of the
            # row sums (reduced across lanes once, before the request's own rows)
            init = (
                jnp.full((hkv * qg, 1), NEG, jnp.float32),
                jnp.zeros((hkv * qg, LANE if c.attn_unshifted else 1), jnp.float32),
                jnp.zeros((hkv * qg, c.attn_width), jnp.float32),
            )
            col = jax.lax.broadcasted_iota(jnp.int32, (1, cpt_g * ph), 1)
            colhead = col % ph
            coltok = col // ph
            kdim = pd if c.packed_attn else hd

            def pv(pexp, v, v_t):
                """P @ V; ``v_t``: v is [kv_dim, tokens] (SAL chunks), else [tokens, dim]."""
                lhs = (
                    pexp
                    if pexp.dtype in (jnp.bfloat16, jnp.float8_e4m3fn)
                    else pexp.astype(jnp.bfloat16)
                )
                return jax.lax.dot_general(
                    lhs,
                    v,
                    (((1,), (1 if v_t else 0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )

            def online(carry, s_, v, v_t=False):
                m, lsum, acc = carry
                if c.attn_unshifted:
                    with _scope("softmax"):
                        pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                        if lsum.shape[1] == LANE:
                            for lt in range(pexp.shape[1] // LANE):
                                lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                        else:
                            lsum = lsum + jnp.sum(pexp, axis=1, keepdims=True)
                    with _scope("pv"):
                        acc = acc[:, : v.shape[0 if v_t else 1]] + pv(pexp, v, v_t)
                    return m, lsum, acc
                with _scope("softmax"):
                    m_new = jnp.maximum(m, jnp.max(s_, axis=1, keepdims=True))
                    corr = jnp.exp(m - m_new)
                    pexp = jnp.exp(s_ - m_new)
                    lsum = lsum * corr + jnp.sum(pexp, axis=1, keepdims=True)
                with _scope("pv"):
                    acc = acc[:, : v.shape[0 if v_t else 1]] * corr + pv(pexp, v, v_t)
                return m_new, lsum, acc

            def sal_unpack(slot):
                """A ring chunk -> K, V as [kv_dim, cpt_g], rows h * hd + dim, tokens on lanes."""
                fp8_k = c.kv_sal_fp8 and c.sal_fp8_dot >= 1
                fp8_v = c.kv_sal_fp8 and c.sal_fp8_dot >= 2
                x8 = pltpu.bitcast(
                    ring_words(geo, slot)[...], c.kv_dtype
                )  # [2 * kv_heads * hd, cpt_g]
                xb = None if (fp8_k and fp8_v) else x8.astype(jnp.bfloat16)
                xk, xv = (x8 if fp8_k else xb), (x8 if fp8_v else xb)
                kk = jnp.concatenate(
                    [xk[2 * h * hd : (2 * h + 1) * hd] for h in range(hkv)], 0
                )
                vk = jnp.concatenate(
                    [xv[(2 * h + 1) * hd : (2 * h + 2) * hd] for h in range(hkv)], 0
                )
                return kk, vk

            def sal_unpack_k(slot):
                w_hd = hd // c.sal_pack
                fp8_k = c.kv_sal_fp8 and c.sal_fp8_dot >= 1
                parts = []
                for h in range(hkv):
                    xh = pltpu.bitcast(
                        ring_words(geo, slot, 2 * h * w_hd, w_hd)[...], c.kv_dtype
                    )
                    parts.append(xh if fp8_k else xh.astype(jnp.bfloat16))
                return jnp.concatenate(parts, axis=0)

            def sal_unpack_v(slot):
                w_hd = hd // c.sal_pack
                fp8_v = c.kv_sal_fp8 and c.sal_fp8_dot >= 2
                parts = []
                for h in range(hkv):
                    xh = pltpu.bitcast(
                        ring_words(geo, slot, (2 * h + 1) * w_hd, w_hd)[...], c.kv_dtype
                    )
                    parts.append(xh if fp8_v else xh.astype(jnp.bfloat16))
                return jnp.concatenate(parts, axis=0)

            def chunk_kv(slot):
                with _scope("kv_unpack"):
                    if c.kv_sal:
                        return sal_unpack(slot)
                    u = (
                        S["kv_chunk"]
                        .at[slot]
                        .bitcast(jnp.uint32)
                        .reshape(cpt_g * ph, pd)[...]
                    )  # rows (token, h)
                    # u32 -> u16 truncation keeps the low half
                    kk = pltpu.bitcast(u.astype(jnp.uint16), jnp.bfloat16)
                    vk = pltpu.bitcast((u >> 16).astype(jnp.uint16), jnp.bfloat16)
                    return kk, vk

            def chunk_k(slot):
                if c.kv_sal and c.sal_kv_split:
                    with _scope("kv_unpack"):
                        return sal_unpack_k(slot)
                kk, _ = chunk_kv(slot)
                return kk

            def chunk_v(slot):
                if c.kv_sal and c.sal_kv_split:
                    with _scope("kv_unpack"):
                        return sal_unpack_v(slot)
                _, vk = chunk_kv(slot)
                return vk

            def qk(kk):
                s_ = jax.lax.dot_general(
                    q2,
                    kk if c.kv_sal else kk[:, :kdim],
                    (((1,), (0 if c.kv_sal else 1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                return s_ * sc if sc != 1.0 else s_

            def chunk_scores(i, valid, clean=False):
                """Masked scores and V of chunk i; ``clean`` chunks lie wholly in the past of

                every row and skip the position and window masks.
                """
                slot = i % KR_g
                ci = first_ch + i
                kk, vk = chunk_kv(slot)
                live = None if c.packed_attn else colhead == rowhead
                if not clean:
                    keypos = coltok + ci * cpt_g
                    if c.attn_col_mask and is_global is True:
                        pos_live = (keypos < p_lo) & valid
                    else:
                        pos_live = (keypos < p_lo) & (keypos <= rowpos) & valid
                        in_win = _in_window(is_global, rowpos - keypos < c.window)
                        if in_win is not None:
                            pos_live = pos_live & in_win
                    live = pos_live if live is None else live & pos_live
                with _scope("qk"):
                    s_ = qk(kk)
                return (s_ if live is None else jnp.where(live, s_, NEG)), vk

            def refill(ref, v):
                """Chunk v (clamped to the last one) into ring slot v % KR_g."""
                slot = v % KR_g
                ci = first_ch + jnp.minimum(v, n_ch - 1)
                for pi in range(geo.cp):
                    pg_no = ci * geo.cp + pi
                    ent = jnp.minimum(
                        j * c.pages_per_seq + jnp.minimum(pg_no, c.pages_per_seq - 1),
                        c.n_pages_entries - 1,
                    )
                    page = pf_ref[ent]  # a global layer's table
                    pltpu.make_async_copy(
                        page_src(ref, page),
                        chunk_dst(geo, slot, pi),
                        S["kv_chunk_sems"].at[slot],
                    ).start()

            def body(k, carry2, ref=None, clean=False):
                """Chunks 2k, 2k+1 as two independent online-softmax streams (two dependency

                chains for the scheduler). With ref, the two slots just read take chunks
                2k + KR_g, 2k + 1 + KR_g, issued straight-line so the copies' scalar
                work
                overlaps the vector work.
                """
                i0, i1 = 2 * k, 2 * k + 1
                valid_b = i1 < n_ch
                with _scope("kv_wait"):
                    wait_chunk(geo, i0 % KR_g)

                    @pl.when(
                        valid_b
                    )  # an odd tail re-reads chunk i0, already waited for
                    def _():
                        wait_chunk(geo, i1 % KR_g)

                s_a, v_a = chunk_scores(i0, True, clean)
                if c.attn_skip_idle_streams and not clean and ref is None:
                    carry_b = jax.lax.cond(
                        valid_b,
                        lambda cr: online(cr, *chunk_scores(i1, True, False), c.kv_sal),
                        lambda cr: cr,
                        carry2[1],
                    )
                    return online(carry2[0], s_a, v_a, c.kv_sal), carry_b
                s_b, v_b = chunk_scores(
                    i1 if clean else jnp.minimum(i1, n_ch - 1), valid_b, clean
                )
                out = (
                    online(carry2[0], s_a, v_a, c.kv_sal),
                    online(carry2[1], s_b, v_b, c.kv_sal),
                )
                if ref is not None:
                    refill(ref, i0 + KR_g)
                    refill(ref, i1 + KR_g)
                return out

            # Global-layer chunks before a request's last one lie wholly in the past of every
            # verify row: chunks [0, n_clean) need only the cross-head mask.
            n_clean = _sel(is_global, jnp.maximum(n_ch - 1, 0), 0)
            NS = c.attn_streams
            gls = sorted(c.global_layers)
            n_alias_p = _attn_p_ysend_streams(c)

            if not isinstance(n_clean, int):
                # The first KR_g chunks come from kv_prologue. Global pairs [0, k_ref) refill the
                # rest: all real chunks but the very last refill when n_ch - KR_g is odd (a copy of
                # the last chunk, drained after the loops). Sliding-window layers never refill.
                k_ref = jnp.clip((n_ch - KR_g + 1) // 2, 0, n_clean // 2)
                extra = (k_ref > 0) & (2 * k_ref - 1 + KR_g >= n_ch)

                def refill_loop(ref):
                    return jax.lax.fori_loop(
                        0,
                        k_ref,
                        functools.partial(body, ref=ref, clean=True),
                        (init, init),
                    )

                def branch_layers(ids, fns):
                    """fns[i]() for l == ids[i], through a balanced tree of two-way branches."""
                    if len(fns) == 1:
                        return fns[0]()
                    mid = len(fns) // 2
                    return jax.lax.cond(
                        l < ids[mid],
                        lambda: branch_layers(ids[:mid], fns[:mid]),
                        lambda: branch_layers(ids[mid:], fns[mid:]),
                    )

                use_p_fp8 = _use_attn_p_fp8(c)
                M_p = hkv * qg
                M_p_fp8 = -(-M_p // 32) * 32

                def psc_get(par, t):
                    v = S["attn_psc"][par, t]
                    return v[:M_p] if v.shape[0] > M_p else v

                def psc_put(par, t, psc):
                    m_sc = -(-M_p // 8) * 8
                    v = jnp.broadcast_to(psc, (M_p, LANE))
                    if m_sc > M_p:
                        v = jnp.concatenate(
                            [v, jnp.zeros((m_sc - M_p, LANE), jnp.float32)], axis=0
                        )
                    S["attn_psc"][par, t] = v

                def p_get(par, t):
                    if use_p_fp8:
                        if t >= n_alias_p:
                            raw_p = S["attn_p"][par, t - n_alias_p]
                        else:
                            ys_fp8 = (
                                S["ysend"]
                                .bitcast(jnp.float8_e4m3fn)
                                .reshape(R * 4 * c.top_k, yw)
                            )
                            if 2 * cpt_g <= yw:
                                c0 = (
                                    par * cpt_g
                                    if isinstance(par, int)
                                    else pl.multiple_of(par * cpt_g, LANE)
                                )
                                raw_p = ys_fp8[
                                    pl.ds(t * M_p_fp8, M_p_fp8), pl.ds(c0, cpt_g)
                                ]
                            else:
                                r_p = (
                                    (par * n_alias_p + t) * M_p_fp8
                                    if isinstance(par, int)
                                    else pl.multiple_of(
                                        (par * n_alias_p + t) * M_p_fp8, 32
                                    )
                                )
                                raw_p = ys_fp8[pl.ds(r_p, M_p_fp8), pl.ds(0, cpt_g)]
                        return raw_p[:M_p] if M_p_fp8 > M_p else raw_p
                    if t >= n_alias_p:
                        return S["attn_p"][par, t - n_alias_p]
                    if 2 * cpt_g <= yw:
                        c0 = (
                            par * cpt_g
                            if isinstance(par, int)
                            else pl.multiple_of(par * cpt_g, LANE)
                        )
                        return (
                            S["ysend"]
                            .bitcast(jnp.bfloat16)[pl.ds(t * q, q), :, pl.ds(c0, cpt_g)]
                            .reshape(hkv * qg, cpt_g)
                        )
                    ys_bf16 = (
                        S["ysend"].bitcast(jnp.bfloat16).reshape(R * 2 * c.top_k, yw)
                    )
                    r_p = (
                        (par * n_alias_p + t) * M_p
                        if isinstance(par, int)
                        else pl.multiple_of((par * n_alias_p + t) * M_p, 8)
                    )
                    return ys_bf16[pl.ds(r_p, M_p), pl.ds(0, cpt_g)]

                def p_put(par, t, pb):
                    if use_p_fp8:
                        pb_pad = (
                            jnp.concatenate(
                                [pb, jnp.zeros((M_p_fp8 - M_p, cpt_g), pb.dtype)],
                                axis=0,
                            )
                            if M_p_fp8 > M_p
                            else pb
                        )
                        if t >= n_alias_p:
                            S["attn_p"][par, t - n_alias_p] = pb_pad
                            return
                        ys_fp8 = (
                            S["ysend"]
                            .bitcast(jnp.float8_e4m3fn)
                            .reshape(R * 4 * c.top_k, yw)
                        )
                        if 2 * cpt_g <= yw:
                            c0 = (
                                par * cpt_g
                                if isinstance(par, int)
                                else pl.multiple_of(par * cpt_g, LANE)
                            )
                            ys_fp8[pl.ds(t * M_p_fp8, M_p_fp8), pl.ds(c0, cpt_g)] = (
                                pb_pad
                            )
                        else:
                            r_p = (
                                (par * n_alias_p + t) * M_p_fp8
                                if isinstance(par, int)
                                else pl.multiple_of((par * n_alias_p + t) * M_p_fp8, 32)
                            )
                            ys_fp8[pl.ds(r_p, M_p_fp8), pl.ds(0, cpt_g)] = pb_pad
                        return
                    if t >= n_alias_p:
                        S["attn_p"][par, t - n_alias_p] = pb
                        return
                    if 2 * cpt_g <= yw:
                        c0 = (
                            par * cpt_g
                            if isinstance(par, int)
                            else pl.multiple_of(par * cpt_g, LANE)
                        )
                        S["ysend"].bitcast(jnp.bfloat16)[
                            pl.ds(t * q, q), :, pl.ds(c0, cpt_g)
                        ] = pb.reshape(q, 2 * c.top_k, cpt_g)
                        return
                    ys_bf16 = (
                        S["ysend"].bitcast(jnp.bfloat16).reshape(R * 2 * c.top_k, yw)
                    )
                    r_p = (
                        (par * n_alias_p + t) * M_p
                        if isinstance(par, int)
                        else pl.multiple_of((par * n_alias_p + t) * M_p, 8)
                    )
                    ys_bf16[pl.ds(r_p, M_p), pl.ds(0, cpt_g)] = pb

                def qk_exp(i, lsum):
                    """Probabilities (bf16) of clean chunk i and the lane partials of their row sums."""
                    kk = chunk_k(i % KR_g)
                    with _scope("qk"):
                        s_ = qk(kk)
                    with _scope("softmax"):
                        pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                        for lt in range(cpt_g // LANE):
                            lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                    return pexp.astype(jnp.bfloat16), lsum

                def pv_chunk(i, pb):
                    vk = chunk_v(i % KR_g)
                    with _scope("pv"):
                        return pv(pb, vk, c.kv_sal)

                def refill_global(v0):
                    """Chunks [v0, v0 + NS) that exist into their ring slots: one walk over the global

                    layers' pools, its leaf a small loop.
                    """
                    n = jnp.clip(n_ch - v0, 0, NS)

                    def leaf(ref):
                        def one(u, carry):
                            v = v0 + u
                            for pi in range(geo.cp):
                                ent = jnp.minimum(
                                    j * c.pages_per_seq
                                    + jnp.minimum(
                                        (first_ch + v) * geo.cp + pi,
                                        c.pages_per_seq - 1,
                                    ),
                                    c.n_pages_entries - 1,
                                )
                                pltpu.make_async_copy(
                                    page_src(ref, pf_ref[ent]),
                                    chunk_dst(geo, v % KR_g, pi),
                                    S["kv_chunk_sems"].at[v % KR_g],
                                ).start()
                            return carry

                        jax.lax.fori_loop(0, n, one, 0)

                    @pl.when(n > 0)
                    def _():
                        _dispatch(l, KV, leaf, only=gls)

                def refill_slot(ref, v):
                    """Chunk v (known < n_ch) into ring slot v % KR_g, straight-line."""
                    slot = v % KR_g
                    ci = first_ch + v
                    for pi in range(geo.cp):
                        pg_no = ci * geo.cp + pi
                        ent = jnp.minimum(
                            j * c.pages_per_seq
                            + jnp.minimum(pg_no, c.pages_per_seq - 1),
                            c.n_pages_entries - 1,
                        )
                        pltpu.make_async_copy(
                            page_src(ref, pf_ref[ent]),
                            chunk_dst(geo, slot, pi),
                            S["kv_chunk_sems"].at[slot],
                        ).start()

                def pipelined_po(n_it):
                    """attn_preones clean-chunk pipeline (M > 64): QK on the raw [K0; V0; K1; V1] chunk

                    with zero-interleaved queries, PV on [V0; K1; V1; ones] carrying one
                    [M, 224]
                    accumulator whose ones lanes hold the row sums.
                    """
                    M = hkv * qg
                    qs = q2_raw * (c.attn_scale * LOG2E)
                    if c.attn_q_fp8:
                        q_max = jnp.maximum(
                            jnp.max(jnp.abs(qs), axis=1, keepdims=True), 1e-12
                        )
                        q_sc = q_max * (1.0 / 256.0)
                        q_l = (qs / q_sc).astype(jnp.float8_e4m3fn)
                        q_scb = q_sc.astype(jnp.bfloat16)
                    else:
                        q_l = qs.astype(jnp.bfloat16)
                        q_scb = None
                    zq = jnp.zeros((M, hd), q_l.dtype)
                    q_int = jnp.concatenate(
                        [q_l[:, 0:hd], zq, q_l[:, hd : 2 * hd], zq], axis=1
                    )  # [M, 4 * hd]
                    AW = 3 * hd + 32  # [V0, K1, V1] lanes, then 32 row-sum lanes

                    def qk_po(i):
                        x8 = pltpu.bitcast(
                            ring_words(geo, i % KR_g)[...], jnp.float8_e4m3fn
                        )
                        with _scope("qk"):
                            s_ = jax.lax.dot_general(
                                q_int,
                                x8,
                                (((1,), (0,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            )
                        with _scope("softmax"):
                            if q_scb is not None:
                                return jnp.exp2(s_.astype(jnp.bfloat16) * q_scb)
                            return jnp.exp2(s_).astype(jnp.bfloat16)

                    def pv_po(i, pb):
                        v_aug = pltpu.bitcast(
                            ring_words(geo, i % KR_g, hd // 4, AW // 4)[...],
                            jnp.float8_e4m3fn,
                        )  # [3 * hd + 32, cpt_g]
                        with _scope("pv"):
                            return jax.lax.dot_general(
                                pb,
                                v_aug,
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            )

                    for t in range(NS):
                        wait_chunk(geo, t % KR_g)
                    for t in range(NS):
                        p_put(0, t, qk_po(t))

                    def step_po(k, ac, ref=None):
                        i0 = NS * k
                        par, npar = k % 2, (k + 1) % 2
                        with _scope("kv_wait"):
                            for t in range(NS):
                                wait_chunk(geo, (i0 + NS + t) % KR_g)
                        for t in range(NS):
                            ac = ac + pv_po(i0 + t, p_get(par, t))
                        for t in range(NS):
                            p_put(npar, t, qk_po(i0 + NS + t))
                        if ref is not None:
                            for t in range(NS):
                                refill_slot(ref, i0 + KR_g + t)
                        else:
                            refill_global(i0 + KR_g)
                        return ac

                    za = jnp.zeros((M, AW), jnp.float32)
                    k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                    if isinstance(l, int):
                        ac = jax.lax.fori_loop(
                            0, k_full, functools.partial(step_po, ref=KV[l]), za
                        )
                    elif R <= 64 or c.attn_opt == 5:
                        ac = branch_layers(
                            gls,
                            [
                                lambda r=KV[x]: jax.lax.fori_loop(
                                    0, k_full, functools.partial(step_po, ref=r), za
                                )
                                for x in gls
                            ],
                        )
                    else:
                        ac, k_full = za, 0
                    ac = jax.lax.fori_loop(k_full, n_it - 1, step_po, ac)
                    i0 = NS * (n_it - 1)
                    for t in range(NS):
                        ac = ac + pv_po(i0 + t, p_get((n_it - 1) % 2, t))
                    refill_global(i0 + KR_g)
                    acc_out = jnp.concatenate(
                        [ac[:, 0:hd], ac[:, 2 * hd : 3 * hd]], axis=1
                    )
                    l_out = jnp.concatenate(
                        [
                            ac[:, 3 * hd : 3 * hd + 32] * (1.0 / 32.0),
                            jnp.zeros((M, LANE - 32), jnp.float32),
                        ],
                        axis=1,
                    )
                    return (init[0], l_out, acc_out), init

                def pipelined_opt(n_it):
                    """Hoisted straight-line KV refill + 2-acc PV/QK pipeline (stack2 for q<=4)."""
                    M = hkv * qg
                    if use_po and M > 64:
                        return pipelined_po(n_it)
                    use_fp8_q = (
                        (c.attn_opt == 4 or c.attn_q_fp8)
                        and c.attn_exp2
                        and c.kv_sal_fp8
                        and c.sal_fp8_dot >= 1
                    )
                    if use_fp8_q:
                        q_lhs, q_sc = _quant_fp8_rows(q2_raw * (c.attn_scale * LOG2E))
                    else:
                        q_lhs = q2
                        q_sc = sc
                    use_fp8_p = use_p_fp8

                    def _scale_pv(dpv, psc_v):
                        return (
                            dpv * psc_v
                            if psc_v.shape[1] == dpv.shape[1]
                            else dpv * psc_v[:, 0:1]
                        )

                    use_stack2 = M <= 64 and NS % 2 == 0
                    if use_stack2:
                        zq = jnp.zeros((M, q_lhs.shape[1]), q_lhs.dtype)
                        q2_blk = jnp.concatenate(
                            [
                                jnp.concatenate([q_lhs, zq], axis=1),
                                jnp.concatenate([zq, q_lhs], axis=1),
                            ],
                            axis=0,
                        )

                    def qk_exp_opt(i, lsum):
                        kk = chunk_k(i % KR_g)
                        with _scope("qk"):
                            s_ = jax.lax.dot_general(
                                q_lhs,
                                kk,
                                (((1,), (0,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            )
                            if use_fp8_q or q_sc != 1.0:
                                s_ = s_ * q_sc
                        with _scope("softmax"):
                            pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                            for lt in range(cpt_g // LANE):
                                lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                            if use_fp8_p:
                                pq, psc = _quant_fp8_rows(pexp, nonneg=True)
                                return (pq, psc), lsum
                        return pexp.astype(jnp.bfloat16), lsum

                    def qk_pair(i_a, i_b, ls0, ls1):
                        ka = chunk_k(i_a % KR_g)
                        kb_ = chunk_k(i_b % KR_g)
                        k_cat = jnp.concatenate([ka, kb_], axis=0)
                        with _scope("qk"):
                            sb = jax.lax.dot_general(
                                q2_blk,
                                k_cat,
                                (((1,), (0,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            )
                            sa, sb_ = sb[:M], sb[M:]
                            if use_fp8_q or q_sc != 1.0:
                                sa, sb_ = sa * q_sc, sb_ * q_sc
                        with _scope("softmax"):
                            pa = jnp.exp2(sa) if c.attn_exp2 else jnp.exp(sa)
                            pb = jnp.exp2(sb_) if c.attn_exp2 else jnp.exp(sb_)
                            for lt in range(cpt_g // LANE):
                                ls0 = ls0 + pa[:, lt * LANE : (lt + 1) * LANE]
                                ls1 = ls1 + pb[:, lt * LANE : (lt + 1) * LANE]
                            if use_fp8_p:
                                pab_q, psc_ab = _quant_fp8_rows(
                                    jnp.concatenate([pa, pb], axis=1), nonneg=True
                                )
                                return (
                                    pab_q[:, :cpt_g],
                                    pab_q[:, cpt_g:],
                                    psc_ab,
                                    ls0,
                                    ls1,
                                )
                        return (
                            pa.astype(jnp.bfloat16),
                            pb.astype(jnp.bfloat16),
                            None,
                            ls0,
                            ls1,
                        )

                    def pv_pair(i_a, i_b, pa_b, pb_b):
                        va = chunk_v(i_a % KR_g)
                        vb_ = chunk_v(i_b % KR_g)
                        p_cat = jnp.concatenate([pa_b, pb_b], axis=1)
                        v_cat = jnp.concatenate([va, vb_], axis=1)
                        with _scope("pv"):
                            return jax.lax.dot_general(
                                p_cat,
                                v_cat,
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            )

                    zl = jnp.zeros((M, LANE), jnp.float32)
                    za = jnp.zeros((M, c.attn_width), jnp.float32)
                    for t in range(NS):
                        wait_chunk(geo, t % KR_g)
                    if not use_stack2 and _can_w2_attn_scratch(c):
                        w_hd = hd // c.sal_pack
                        w2_rows = (c.expert_depth + 1) * (c.inter_local // FP4_PACK)
                        w2_f32 = S["w2"].bitcast(jnp.float32).reshape(w2_rows, He)
                        qh_list = []
                        q_sch = []
                        for h in range(hkv):
                            qr_h = q2_raw[h * qg : (h + 1) * qg]
                            if qg < 64:
                                qr_h = jnp.concatenate(
                                    [
                                        qr_h,
                                        jnp.zeros(
                                            (64 - qg, qr_h.shape[1]), jnp.float32
                                        ),
                                    ],
                                    axis=0,
                                )
                            if use_fp8_q:
                                qh_q, qh_s = _quant_fp8_rows(
                                    qr_h * (c.attn_scale * LOG2E)
                                )
                                qh_list.append(qh_q)
                                q_sch.append(qh_s)
                            else:
                                qh_list.append(
                                    (qr_h * (c.attn_scale * LOG2E)).astype(jnp.bfloat16)
                                )
                                q_sch.append(1.0)
                        p_sub_dt = (
                            jnp.float8_e4m3fn if c.attn_head_fp8 >= 2 else jnp.bfloat16
                        )
                        ys_sub_rows = R * (4 if c.attn_head_fp8 >= 2 else 2) * c.top_k

                        def p_get_sub(par, t, h, s0):
                            if t >= n_alias_p:
                                return S["attn_p"][
                                    par,
                                    t - n_alias_p,
                                    pl.ds(h * 64, 64),
                                    pl.ds(s0, 512),
                                ]
                            ys_2d = (
                                S["ysend"].bitcast(p_sub_dt).reshape(ys_sub_rows, yw)
                            )
                            if 2 * cpt_g <= yw:
                                c0 = (
                                    par * cpt_g + s0
                                    if isinstance(par, int)
                                    else pl.multiple_of(par * cpt_g + s0, LANE)
                                )
                                return ys_2d[
                                    pl.ds(t * 128 + h * 64, 64), pl.ds(c0, 512)
                                ]
                            r_sub = (par * n_alias_p + t) * 128 + h * 64
                            r_sub = (
                                r_sub
                                if isinstance(par, int)
                                else pl.multiple_of(r_sub, 64)
                            )
                            return ys_2d[pl.ds(r_sub, 64), pl.ds(s0, 512)]

                        def p_put_sub(par, t, h, s0, pb_sub):
                            if t >= n_alias_p:
                                S["attn_p"][
                                    par,
                                    t - n_alias_p,
                                    pl.ds(h * 64, 64),
                                    pl.ds(s0, 512),
                                ] = pb_sub
                                return
                            ys_2d = (
                                S["ysend"].bitcast(p_sub_dt).reshape(ys_sub_rows, yw)
                            )
                            if 2 * cpt_g <= yw:
                                c0 = (
                                    par * cpt_g + s0
                                    if isinstance(par, int)
                                    else pl.multiple_of(par * cpt_g + s0, LANE)
                                )
                                ys_2d[pl.ds(t * 128 + h * 64, 64), pl.ds(c0, 512)] = (
                                    pb_sub
                                )
                                return
                            r_sub = (par * n_alias_p + t) * 128 + h * 64
                            r_sub = (
                                r_sub
                                if isinstance(par, int)
                                else pl.multiple_of(r_sub, 64)
                            )
                            ys_2d[pl.ds(r_sub, 64), pl.ds(s0, 512)] = pb_sub

                        def psc_get_w2(par, t, h, s_idx):
                            r_sc = (
                                64 + par * 64
                                if isinstance(par, int)
                                else pl.multiple_of(64 + par * 64, 64)
                            )
                            c_sc = (4 * t + 2 * h + s_idx) * LANE
                            return w2_f32[pl.ds(r_sc, 64), pl.ds(c_sc, LANE)]

                        def psc_put_w2(par, t, h, s_idx, v_sc):
                            r_sc = (
                                64 + par * 64
                                if isinstance(par, int)
                                else pl.multiple_of(64 + par * 64, 64)
                            )
                            c_sc = (4 * t + 2 * h + s_idx) * LANE
                            w2_f32[pl.ds(r_sc, 64), pl.ds(c_sc, LANE)] = v_sc

                        z_kv = jnp.zeros((64, 512), jnp.float8_e4m3fn)

                        def kh_half(slot, h, s0):
                            with _scope("kv_unpack"):
                                kr = pltpu.bitcast(
                                    ring_words(geo, slot, 2 * h * w_hd, w_hd)[
                                        :, pl.ds(s0, 512)
                                    ],
                                    jnp.float8_e4m3fn,
                                )
                                return jnp.concatenate(
                                    [kr, z_kv] if h == 0 else [z_kv, kr], axis=0
                                )

                        def vh_half(slot, h, s0):
                            with _scope("kv_unpack"):
                                vr = pltpu.bitcast(
                                    ring_words(geo, slot, (2 * h + 1) * w_hd, w_hd)[
                                        :, pl.ds(s0, 512)
                                    ],
                                    jnp.float8_e4m3fn,
                                )
                                return jnp.concatenate(
                                    [vr, z_kv] if h == 0 else [z_kv, vr], axis=0
                                )

                        def qk_all_h(i_start, par):
                            for h in range(hkv):
                                ls_h = w2_f32[pl.ds(0, 64), pl.ds((2 + h) * LANE, LANE)]
                                qh = qh_list[h]
                                qsc_h = q_sch[h]
                                for t in range(NS):
                                    slot = (i_start + t) % KR_g
                                    for s_idx, s0 in enumerate((0, 512)):
                                        kp = kh_half(slot, h, s0)
                                        with _scope("qk"):
                                            s_h = jax.lax.dot_general(
                                                qh,
                                                kp,
                                                (((1,), (0,)), ((), ())),
                                                preferred_element_type=jnp.float32,
                                            )
                                            if use_fp8_q:
                                                s_h = s_h * qsc_h
                                        with _scope("softmax"):
                                            if c.attn_head_fp8 >= 2:
                                                pe = jnp.exp2(s_h)
                                                sm0 = jnp.maximum(
                                                    pe[:, :128], pe[:, 128:256]
                                                )
                                                sm1 = jnp.maximum(
                                                    pe[:, 256:384], pe[:, 384:512]
                                                )
                                                amax = jnp.max(
                                                    jnp.maximum(sm0, sm1),
                                                    axis=1,
                                                    keepdims=True,
                                                )
                                                psc = jnp.where(
                                                    amax == 0,
                                                    jnp.zeros_like(amax),
                                                    amax * jnp.float32(1.0 / 448.0),
                                                )
                                                inv = jnp.where(
                                                    amax == 0,
                                                    jnp.zeros_like(amax),
                                                    jnp.float32(448.0)
                                                    / jnp.maximum(
                                                        amax, jnp.float32(1e-30)
                                                    ),
                                                )
                                                pf = (pe * inv).astype(
                                                    jnp.float8_e4m3fn
                                                )
                                                pf_f = pf.astype(jnp.float32)
                                                dd = (
                                                    (pf_f[:, :128] + pf_f[:, 128:256])
                                                    + (
                                                        pf_f[:, 256:384]
                                                        + pf_f[:, 384:512]
                                                    )
                                                ) * psc
                                                ls_h = ls_h + dd
                                                p_put_sub(par, t, h, s0, pf)
                                                psc_put_w2(
                                                    par,
                                                    t,
                                                    h,
                                                    s_idx,
                                                    jnp.broadcast_to(psc, (64, LANE)),
                                                )
                                            else:
                                                pe = jnp.exp2(s_h)
                                                dd = (pe[:, :128] + pe[:, 128:256]) + (
                                                    pe[:, 256:384] + pe[:, 384:512]
                                                )
                                                ls_h = ls_h + dd
                                                p_put_sub(
                                                    par,
                                                    t,
                                                    h,
                                                    s0,
                                                    pe.astype(jnp.bfloat16),
                                                )
                                w2_f32[pl.ds(0, 64), pl.ds((2 + h) * LANE, LANE)] = ls_h

                        def pv_all_h(i0, par):
                            for a_idx in (0, 1):
                                acc_cur = w2_f32[
                                    pl.ds(0, 64), pl.ds(a_idx * LANE, LANE)
                                ]
                                for t in range(a_idx, NS, 2):
                                    slot = (i0 + t) % KR_g
                                    for h in range(hkv):
                                        for s_idx, s0 in enumerate((0, 512)):
                                            pb_s = p_get_sub(par, t, h, s0)
                                            vp_s = vh_half(slot, h, s0)
                                            with _scope("pv"):
                                                dpv = jax.lax.dot_general(
                                                    pb_s,
                                                    vp_s,
                                                    (((1,), (1,)), ((), ())),
                                                    preferred_element_type=jnp.float32,
                                                )
                                            if c.attn_head_fp8 >= 2:
                                                dpv = dpv * psc_get_w2(par, t, h, s_idx)
                                            acc_cur = acc_cur + dpv
                                w2_f32[pl.ds(0, 64), pl.ds(a_idx * LANE, LANE)] = (
                                    acc_cur
                                )

                        z64 = jnp.zeros((64, LANE), jnp.float32)
                        for b_i in range(4):
                            w2_f32[pl.ds(0, 64), pl.ds(b_i * LANE, LANE)] = z64
                        qk_all_h(0, 0)

                        def step_h(k, dummy, ref=None):
                            i0 = NS * k
                            par, npar = k % 2, (k + 1) % 2
                            with _scope("kv_wait"):
                                for t in range(NS):
                                    wait_chunk(geo, (i0 + NS + t) % KR_g)
                            pv_all_h(i0, par)
                            qk_all_h(i0 + NS, npar)
                            if ref is not None:
                                for t in range(NS):
                                    refill_slot(ref, i0 + KR_g + t)
                            else:
                                refill_global(i0 + KR_g)
                            return dummy

                        k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                        if isinstance(l, int):
                            jax.lax.fori_loop(
                                0,
                                k_full,
                                functools.partial(step_h, ref=KV[l]),
                                jnp.int32(0),
                            )
                        elif R <= 64 or c.attn_opt == 5:
                            branch_layers(
                                gls,
                                [
                                    lambda r=KV[x]: jax.lax.fori_loop(
                                        0,
                                        k_full,
                                        functools.partial(step_h, ref=r),
                                        jnp.int32(0),
                                    )
                                    for x in gls
                                ],
                            )
                        else:
                            k_full = 0
                        jax.lax.fori_loop(k_full, n_it - 1, step_h, jnp.int32(0))
                        pv_all_h(NS * (n_it - 1), (n_it - 1) % 2)
                        refill_global(NS * (n_it - 1) + KR_g)
                        acc_pk = (
                            w2_f32[pl.ds(0, 64), pl.ds(0, LANE)]
                            + w2_f32[pl.ds(0, 64), pl.ds(LANE, LANE)]
                        )[:qg]
                        col_i = jax.lax.broadcasted_iota(jnp.int32, (qg, LANE), 1)
                        acc_out = jnp.concatenate(
                            [
                                jnp.where(col_i < hd, acc_pk, 0.0),
                                jnp.where(col_i >= hd, acc_pk, 0.0),
                            ],
                            axis=0,
                        )
                        ls_out = jnp.concatenate(
                            [
                                w2_f32[pl.ds(0, 64), pl.ds(2 * LANE, LANE)][:qg],
                                w2_f32[pl.ds(0, 64), pl.ds(3 * LANE, LANE)][:qg],
                            ],
                            axis=0,
                        )
                        return (init[0], ls_out, acc_out), init
                    if not use_stack2 and c.attn_carry:
                        n_acc = 1 if c.attn_carry == 1 else 2

                        def qk_tree(i):
                            kk = chunk_k(i % KR_g)
                            with _scope("qk"):
                                s_ = jax.lax.dot_general(
                                    q_lhs,
                                    kk,
                                    (((1,), (0,)), ((), ())),
                                    preferred_element_type=jnp.float32,
                                )
                                if use_fp8_q or q_sc != 1.0:
                                    s_ = s_ * q_sc
                            with _scope("softmax"):
                                if c.attn_carry == 3:
                                    pb = jnp.exp2(s_.astype(jnp.bfloat16))
                                    parts = [
                                        pb[:, lt * LANE : (lt + 1) * LANE]
                                        for lt in range(cpt_g // LANE)
                                    ]
                                    while len(parts) > 1:
                                        parts = [
                                            parts[j] + parts[j + 1]
                                            for j in range(0, len(parts), 2)
                                        ]
                                    return pb, parts[0].astype(jnp.float32)
                                pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                                parts = [
                                    pexp[:, lt * LANE : (lt + 1) * LANE]
                                    for lt in range(cpt_g // LANE)
                                ]
                                while len(parts) > 1:
                                    parts = [
                                        parts[j] + parts[j + 1]
                                        for j in range(0, len(parts), 2)
                                    ]
                                if use_fp8_p:
                                    pf, psc = _quant_fp8_rows(pexp, nonneg=True)
                                    return (pf, psc), parts[0]
                            return pexp.astype(jnp.bfloat16), parts[0]

                        pair = c.attn_carry == 4
                        if pair:
                            zq_p = jnp.zeros((M, q_lhs.shape[1]), q_lhs.dtype)
                            q_blk_p = jnp.concatenate(
                                [
                                    jnp.concatenate([q_lhs, zq_p], axis=1),
                                    jnp.concatenate([zq_p, q_lhs], axis=1),
                                ],
                                axis=0,
                            )

                        def lane_tree(x):
                            parts = [
                                x[:, lt * LANE : (lt + 1) * LANE]
                                for lt in range(cpt_g // LANE)
                            ]
                            while len(parts) > 1:
                                parts = [
                                    parts[j] + parts[j + 1]
                                    for j in range(0, len(parts), 2)
                                ]
                            return parts[0]

                        def qk_all(i_start, par):
                            d = zl
                            if not pair:
                                for t in range(NS):
                                    pb, dd = qk_tree(i_start + t)
                                    if use_fp8_p:
                                        p_put(par, t, pb[0])
                                        psc_put(par, t, pb[1])
                                    else:
                                        p_put(par, t, pb)
                                    d = d + dd
                                return d
                            for u in range(NS // 2):
                                ka = chunk_k((i_start + 2 * u) % KR_g)
                                kb2 = chunk_k((i_start + 2 * u + 1) % KR_g)
                                with _scope("qk"):
                                    s2 = jax.lax.dot_general(
                                        q_blk_p,
                                        jnp.concatenate([ka, kb2], axis=0),
                                        (((1,), (0,)), ((), ())),
                                        preferred_element_type=jnp.float32,
                                    )
                                for h, s_ in enumerate((s2[:M], s2[M:])):
                                    if use_fp8_q or q_sc != 1.0:
                                        s_ = s_ * q_sc
                                    with _scope("softmax"):
                                        pe = (
                                            jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                                        )
                                        d = d + lane_tree(pe)
                                        if use_fp8_p:
                                            pf, psc = _quant_fp8_rows(pe, nonneg=True)
                                            p_put(par, 2 * u + h, pf)
                                            psc_put(par, 2 * u + h, psc)
                                        else:
                                            p_put(
                                                par, 2 * u + h, pe.astype(jnp.bfloat16)
                                            )
                            return d

                        ls = qk_all(0, 0)

                        def pv_all(i0, par, accs):
                            accs = list(accs)
                            if pair:
                                for u in range(NS // 2):
                                    va = chunk_v((i0 + 2 * u) % KR_g)
                                    vb2 = chunk_v((i0 + 2 * u + 1) % KR_g)
                                    pst = jnp.concatenate(
                                        [p_get(par, 2 * u), p_get(par, 2 * u + 1)],
                                        axis=0,
                                    )
                                    with _scope("pv"):
                                        o2 = jax.lax.dot_general(
                                            pst,
                                            jnp.concatenate([va, vb2], axis=0),
                                            (((1,), (1,)), ((), ())),
                                            preferred_element_type=jnp.float32,
                                        )
                                    W = c.attn_width
                                    if use_fp8_p:
                                        part_pv = _scale_pv(
                                            o2[:M, :W], psc_get(par, 2 * u)
                                        ) + _scale_pv(
                                            o2[M:, W : 2 * W], psc_get(par, 2 * u + 1)
                                        )
                                    else:
                                        part_pv = o2[:M, :W] + o2[M:, W : 2 * W]
                                    accs[u % n_acc] = accs[u % n_acc] + part_pv
                                return accs
                            for t in range(NS):
                                pv_t = pv_chunk(i0 + t, p_get(par, t))
                                if use_fp8_p:
                                    pv_t = _scale_pv(pv_t, psc_get(par, t))
                                accs[t % n_acc] = accs[t % n_acc] + pv_t
                            return accs

                        def step_c(k, carry, ref=None):
                            ls, accs = carry
                            i0 = NS * k
                            par, npar = k % 2, (k + 1) % 2
                            with _scope("kv_wait"):
                                for t in range(NS):
                                    wait_chunk(geo, (i0 + NS + t) % KR_g)
                            accs = pv_all(i0, par, accs)
                            ls = ls + qk_all(i0 + NS, npar)
                            if ref is not None:
                                for t in range(NS):
                                    refill_slot(ref, i0 + KR_g + t)
                            else:
                                refill_global(i0 + KR_g)
                            return ls, accs

                        k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                        init_c = (ls, [za] * n_acc)
                        if isinstance(l, int):
                            carry_c = jax.lax.fori_loop(
                                0, k_full, functools.partial(step_c, ref=KV[l]), init_c
                            )
                        elif R <= 64 or c.attn_opt == 5:
                            carry_c = branch_layers(
                                gls,
                                [
                                    lambda r=KV[x]: jax.lax.fori_loop(
                                        0,
                                        k_full,
                                        functools.partial(step_c, ref=r),
                                        init_c,
                                    )
                                    for x in gls
                                ],
                            )
                        else:
                            carry_c, k_full = init_c, 0
                        ls, accs = jax.lax.fori_loop(k_full, n_it - 1, step_c, carry_c)
                        accs = pv_all(NS * (n_it - 1), (n_it - 1) % 2, accs)
                        refill_global(NS * (n_it - 1) + KR_g)
                        return (init[0], ls, sum(accs[1:], accs[0])), init
                    if not use_stack2:
                        lsums = []
                        for t in range(NS):
                            pb, ls_ = qk_exp_opt(t, zl)
                            if use_fp8_p:
                                p_put(0, t, pb[0])
                                psc_put(0, t, pb[1])
                            else:
                                p_put(0, t, pb)
                            lsums.append(ls_)

                        use_acc2 = R > 64

                        def step_ns(k, carry, ref=None):
                            lsums, accs = carry
                            i0 = NS * k
                            par, npar = k % 2, (k + 1) % 2
                            with _scope("kv_wait"):
                                for t in range(NS):
                                    wait_chunk(geo, (i0 + NS + t) % KR_g)
                            pbs = [p_get(par, t) for t in range(NS)]
                            if use_acc2:
                                ac0, ac1 = accs
                                for t, pb in enumerate(pbs):
                                    dpv = pv_chunk(i0 + t, pb)
                                    if use_fp8_p:
                                        dpv = _scale_pv(dpv, psc_get(par, t))
                                    if t % 2 == 0:
                                        ac0 = ac0 + dpv
                                    else:
                                        ac1 = ac1 + dpv
                                accs = [ac0, ac1]
                            else:
                                new_accs = []
                                for t, (a, pb) in enumerate(zip(accs, pbs)):
                                    dpv = pv_chunk(i0 + t, pb)
                                    if use_fp8_p:
                                        dpv = _scale_pv(dpv, psc_get(par, t))
                                    new_accs.append(a + dpv)
                                accs = new_accs
                            nxt = [qk_exp_opt(i0 + NS + t, lsums[t]) for t in range(NS)]
                            for t in range(NS):
                                if use_fp8_p:
                                    p_put(npar, t, nxt[t][0][0])
                                    psc_put(npar, t, nxt[t][0][1])
                                else:
                                    p_put(npar, t, nxt[t][0])
                            if ref is not None:
                                for t in range(NS):
                                    refill_slot(ref, i0 + KR_g + t)
                            else:
                                refill_global(i0 + KR_g)
                            return [x[1] for x in nxt], accs

                        k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                        init_ns = (lsums, [za, za] if use_acc2 else [za] * NS)
                        hoist_ns = R <= 64 or c.attn_opt == 5
                        if isinstance(l, int):
                            carry_ns = jax.lax.fori_loop(
                                0,
                                k_full,
                                functools.partial(step_ns, ref=KV[l]),
                                init_ns,
                            )
                        elif hoist_ns:
                            carry_ns = branch_layers(
                                gls,
                                [
                                    lambda r=KV[x]: jax.lax.fori_loop(
                                        0,
                                        k_full,
                                        functools.partial(step_ns, ref=r),
                                        init_ns,
                                    )
                                    for x in gls
                                ],
                            )
                        else:
                            carry_ns = init_ns
                            k_full = 0
                        lsums, accs = jax.lax.fori_loop(
                            k_full, n_it - 1, step_ns, carry_ns
                        )
                        i0 = NS * (n_it - 1)
                        last_par = (n_it - 1) % 2
                        if use_acc2:
                            ac0, ac1 = accs
                            for t in range(NS):
                                dpv = pv_chunk(i0 + t, p_get(last_par, t))
                                if use_fp8_p:
                                    dpv = _scale_pv(dpv, psc_get(last_par, t))
                                if t % 2 == 0:
                                    ac0 = ac0 + dpv
                                else:
                                    ac1 = ac1 + dpv
                            accs = [ac0, ac1]
                        else:
                            new_accs = []
                            for t, a in enumerate(accs):
                                dpv = pv_chunk(i0 + t, p_get(last_par, t))
                                if use_fp8_p:
                                    dpv = _scale_pv(dpv, psc_get(last_par, t))
                                new_accs.append(a + dpv)
                            accs = new_accs
                        refill_global(i0 + KR_g)
                        return (
                            init[0],
                            sum(lsums[1:], lsums[0]),
                            sum(accs[1:], accs[0]),
                        ), init

                    ls0, ls1 = zl, zl
                    for u in range(NS // 2):
                        pa, pb, psc_ab, ls0, ls1 = qk_pair(2 * u, 2 * u + 1, ls0, ls1)
                        p_put(0, 2 * u, pa)
                        p_put(0, 2 * u + 1, pb)
                        if use_fp8_p:
                            psc_put(0, 2 * u, psc_ab)

                    def step_body(k, carry, ref=None):
                        ls0, ls1, ac0, ac1 = carry
                        i0 = NS * k
                        par, npar = k % 2, (k + 1) % 2
                        with _scope("kv_wait"):
                            for t in range(NS):
                                wait_chunk(geo, (i0 + NS + t) % KR_g)
                        for u in range(NS // 2):
                            dpv = pv_pair(
                                i0 + 2 * u,
                                i0 + 2 * u + 1,
                                p_get(par, 2 * u),
                                p_get(par, 2 * u + 1),
                            )
                            if use_fp8_p:
                                dpv = _scale_pv(dpv, psc_get(par, 2 * u))
                            if u % 2 == 0:
                                ac0 = ac0 + dpv
                            else:
                                ac1 = ac1 + dpv
                        for u in range(NS // 2):
                            pa, pb, psc_ab, ls0, ls1 = qk_pair(
                                i0 + NS + 2 * u, i0 + NS + 2 * u + 1, ls0, ls1
                            )
                            p_put(npar, 2 * u, pa)
                            p_put(npar, 2 * u + 1, pb)
                            if use_fp8_p:
                                psc_put(npar, 2 * u, psc_ab)
                        if ref is not None:
                            for t in range(NS):
                                refill_slot(ref, i0 + KR_g + t)
                        else:
                            refill_global(i0 + KR_g)
                        return ls0, ls1, ac0, ac1

                    k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                    init4 = (ls0, ls1, za, za)
                    if isinstance(l, int):
                        carry4 = jax.lax.fori_loop(
                            0, k_full, functools.partial(step_body, ref=KV[l]), init4
                        )
                    else:
                        carry4 = branch_layers(
                            gls,
                            [
                                lambda r=KV[x]: jax.lax.fori_loop(
                                    0,
                                    k_full,
                                    functools.partial(step_body, ref=r),
                                    init4,
                                )
                                for x in gls
                            ],
                        )
                    ls0, ls1, ac0, ac1 = jax.lax.fori_loop(
                        k_full, n_it - 1, step_body, carry4
                    )
                    i0 = NS * (n_it - 1)
                    par = (n_it - 1) % 2
                    for u in range(NS // 2):
                        dpv = pv_pair(
                            i0 + 2 * u,
                            i0 + 2 * u + 1,
                            p_get(par, 2 * u),
                            p_get(par, 2 * u + 1),
                        )
                        if use_fp8_p:
                            dpv = _scale_pv(dpv, psc_get(par, 2 * u))
                        if u % 2 == 0:
                            ac0 = ac0 + dpv
                        else:
                            ac1 = ac1 + dpv
                    refill_global(i0 + KR_g)
                    return (init[0], ls0 + ls1, ac0 + ac1), init

                def pipelined(n_it):
                    """Clean chunks [0, NS * n_it), n_it >= 1."""
                    zl = jnp.zeros((hkv * qg, LANE), jnp.float32)
                    for t in range(NS):
                        wait_chunk(geo, t % KR_g)
                    lsums = []
                    for t in range(NS):
                        pb, ls_ = qk_exp(t, zl)
                        p_put(0, t, pb)
                        lsums.append(ls_)

                    def it(k, carry):
                        lsums, accs = carry
                        i0 = NS * k
                        with _scope("kv_wait"):
                            for t in range(NS):
                                wait_chunk(geo, (i0 + NS + t) % KR_g)
                        pbs = [p_get(k % 2, t) for t in range(NS)]
                        accs = [
                            a + pv_chunk(i0 + t, pb)
                            for t, (a, pb) in enumerate(zip(accs, pbs))
                        ]
                        nxt = [qk_exp(i0 + NS + t, lsums[t]) for t in range(NS)]
                        for t in range(NS):
                            p_put((k + 1) % 2, t, nxt[t][0])
                        refill_global(i0 + KR_g)
                        return [x[1] for x in nxt], accs

                    za = jnp.zeros((hkv * qg, c.attn_width), jnp.float32)
                    lsums, accs = jax.lax.fori_loop(0, n_it - 1, it, (lsums, [za] * NS))
                    i0 = NS * (n_it - 1)
                    accs = [
                        a + pv_chunk(i0 + t, p_get((n_it - 1) % 2, t))
                        for t, a in enumerate(accs)
                    ]
                    refill_global(i0 + KR_g)
                    return (
                        init[0],
                        sum(lsums[1:], lsums[0]),
                        sum(accs[1:], accs[0]),
                    ), init

            with _scope("chunk_loop"):
                if isinstance(n_clean, int):
                    (m_a, l_a, acc_a), (m_b, l_b, acc_b) = jax.lax.fori_loop(
                        0, (n_ch + 1) // 2, body, (init, init)
                    )
                else:
                    if (
                        c.attn_pipe
                        and c.attn_opt in (3, 4, 5)
                        and c.kv_sal
                        and (
                            c.row_tile > 0
                            or hkv * qg <= 64
                            or c.attn_carry > 0
                            or c.attn_q_fp8
                            or c.attn_p_fp8
                            or c.attn_head_fp8 > 0
                        )
                    ):
                        if not gls or (isinstance(l, int) and l not in gls):
                            carry2, k0 = (init, init), 0
                        else:
                            n_it = n_clean // NS
                            carry2 = jax.lax.cond(
                                n_it > 0,
                                lambda: pipelined_opt(n_it),
                                lambda: (init, init),
                            )
                            k0 = n_it * NS // 2
                    elif c.attn_pipe:
                        # global chunks before the last one are clean; SWA layers have none
                        n_it = n_clean // NS
                        carry2 = jax.lax.cond(
                            n_it > 0, lambda: pipelined(n_it), lambda: (init, init)
                        )
                        k0 = n_it * NS // 2
                    elif isinstance(l, int):
                        carry2 = refill_loop(KV[l]) if l in gls else (init, init)
                        k0 = k_ref if l in gls else 0
                    else:
                        carry2 = jax.lax.cond(
                            is_global,
                            lambda: branch_layers(
                                gls,
                                [functools.partial(refill_loop, KV[x]) for x in gls],
                            ),
                            lambda: (init, init),
                        )
                        k0 = _sel(is_global, k_ref, 0)
                    carry2 = jax.lax.fori_loop(
                        k0, n_clean // 2, functools.partial(body, clean=True), carry2
                    )
                    (m_a, l_a, acc_a), (m_b, l_b, acc_b) = jax.lax.fori_loop(
                        n_clean // 2, (n_ch + 1) // 2, body, carry2
                    )

                    if not c.attn_pipe:

                        @pl.when(extra & is_global)
                        def _():
                            wait_chunk(geo, (2 * k_ref - 1 + KR_g) % KR_g)

                if c.attn_unshifted:  # the streams just add
                    carry = (
                        m_a,
                        jnp.sum(l_a + l_b, axis=1, keepdims=True),
                        acc_a + acc_b,
                    )
                else:
                    # a stream that saw no live key keeps m = NEG: weight exp(NEG - m) = 0
                    m_ab = jnp.maximum(m_a, m_b)
                    w_a, w_b = jnp.exp(m_a - m_ab), jnp.exp(m_b - m_ab)
                    carry = (m_ab, l_a * w_a + l_b * w_b, acc_a * w_a + acc_b * w_b)

            # every ring slot is read and every copy waited for: the next request's first
            # chunks fly during this request's own-row attention (request 0's were requested
            # before the qkv matvec)
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
                        for x in [krows[:, h * hd : (h + 1) * hd]]
                        + ([zrows] if q8 > q else [])
                    ],
                    axis=0,
                )
                vn = jnp.concatenate(
                    [
                        x
                        for h in range(hkv)
                        for x in [vrows[:, h * hd : (h + 1) * hd]]
                        + ([zrows] if q8 > q else [])
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
        Rc = c.rows_per_dev
        y_block = c.y_block_send and c.tp > 1

        def blk_owner(b):
            """Owner of expert block b: peers in send order, own rows last."""
            if isinstance(b, int) and b == c.tp - 1:
                return me
            if coll.order == "xor":
                return jnp.bitwise_xor(me, c.tp - 1 - b)
            return jax.lax.rem(me + 1 + b, c.tp)

        NBR = route_blocks(c)

        def route_get(field, i):
            """vec_route table entry (0 expert, 1 count, 2 offset, 3 pair, 5 y-block end, 6 blk_mid)."""
            if field in (5, 6):
                return S["route_s"][4 * NBR + 1 + (field == 6), i]
            return S["route_s"][field * NBR + i // LANE, i % LANE]

        def rget(name, i):
            if not c.vec_route:
                return S[name][i]
            return route_get(
                {
                    "uniq": 0,
                    "ex_cnt": 1,
                    "ex_off": 2,
                    "perm": 3,
                    "blk_end": 5,
                    "blk_mid": 6,
                }[name],
                i,
            )

        def unique_experts_vec(ids_list):
            P = R * K
            r_of = jax.lax.broadcasted_iota(jnp.int32, (P, 1), 0) % R
            if y_block:
                # a pair's key: the y block of its row's owner (own rows last)
                o = r_of // Rc
                key = (
                    (c.tp - 1) - jnp.bitwise_xor(o, me)
                    if coll.order == "xor"
                    else jax.lax.rem(o - me - 1 + c.tp, c.tp)
                )
            else:
                key = jnp.zeros((P, 1), jnp.int32)
            S["route_v"][...] = _route_table(
                ids_list, n_live, K, c.n_experts_padded, NBR, key, c.tp, c.split_rows
            )
            cp = pltpu.make_async_copy(S["route_v"], S["route_s"], S["route_sem"].at[0])
            cp.start()
            cp.wait()
            return S["route_s"][4 * NBR, 0]

        def unique_experts(ids_list, l):
            """Distinct experts of the live rows' slots, in first-use order, into ``uniq`` (SMEM),

            with each expert's (row, slot) pairs grouped in ``perm``.
            """
            if c.vec_route:
                # row-tiled steps route from ids_v, which their top-k fills tile by tile
                return route_tiled() if c.row_tile else unique_experts_vec(ids_list)
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

            if y_block:
                # rows owner block by owner block (own rows last): uniq comes out grouped by
                # the block that first uses each expert, and blk_end[b] closes block b's group
                n_u = 0
                for b in range(c.tp):
                    row0 = blk_owner(b) * Rc
                    n_u = jax.lax.fori_loop(
                        row0 * K,
                        (row0 + jnp.clip(n_live - row0, 0, Rc)) * K,
                        count,
                        n_u,
                    )
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
            return n_u

        def moe(l, ffn_in32, sm, kind, nxt, pool_router=None):
            """pool_router(): (pooled, logits) from a fold, instead of from ffn_in32.

            Returns the ffn branch, or with fold_f and a next layer None: its
            all-gather is in flight (the next layer's fold_qkvg consumes it).
            """
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
                if c.topk_xpose and c.topk_iter and 64 <= R <= LANE and R % 8 == 0:
                    ids_full, wsel_full = _topk_xpose(
                        sc, bias, K, c.router_eps, c.route_scale
                    )
                    ids_list = [ids_full[:, j : j + 1] for j in range(K)]
                    wsel = wsel_full[:, :K]
                else:
                    if c.topk_iter:
                        ids_list, picks = _topk_iter(sc + bias, K)
                    else:
                        ids_list, picks = _topk_by_rank(sc + bias, K, n_live, S["rank"])
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

            def kind_fns(kind):
                """(issue the first expert DMAs, run the expert loop) for one expert kind."""
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
                gscale = (c.scale_fp8 or c.ex_kperm >= 3) and kind == "fp4"
                use_gs_smem = bool(gscale and c.ex_gs_smem and "gs_s" in S)
                gs_cp = (
                    pltpu.make_async_copy(S["gs_v"], S["gs_s"], S["route_sem"].at[0])
                    if use_gs_smem
                    else None
                )

                def globals_of(e):
                    """(gate, up, down) global scales of expert e."""
                    if use_gs_smem:
                        return S["gs_s"][0, e], S["gs_s"][1, e], S["gs_s"][2, e]
                    lane = jax.lax.broadcasted_iota(jnp.int32, (3, Ep), 1)
                    g = jnp.sum(
                        jnp.where(lane == e, S["gs_v"][0:3, :], 0.0),
                        axis=1,
                        keepdims=True,
                    )
                    return g[0:1], g[1:2], g[2:3]

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

                def expert_dmas(refs, i, slot, e=None, off=None):
                    """The four slab DMAs of distinct expert ``i`` into ``slot``, from one layer's refs

                    (``off``: the layer's index in stacked refs).
                    """
                    for j, e, dst, sem, rows_, cols, is_scale in copies(i, slot, e):
                        src = refs[j] if off is None else refs[j].at[off]
                        src = src if is_scale else src.bitcast(jnp.uint32)
                        pltpu.make_async_copy(
                            src.at[e, pl.ds(0, rows_), pl.ds(0, cols)],
                            dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                            sem.at[slot],
                        ).start()

                def for_layer(fn):
                    """``fn(refs, off)`` on this layer's expert refs: ONE branch-tree walk, over

                    stacked groups (then indexed by the layer) or single layers.
                    """
                    if isinstance(l, int):
                        fn(*layer_ex[l])
                        return
                    keys = sorted(
                        {
                            li if layer_ex[li][1] is None else li - layer_ex[li][1]
                            for li in kind_layers[kind]
                        }
                    )

                    def leaf(k):
                        refs, _ = layer_ex[k]
                        fn(refs, l - k if k in group_lo else None)

                    _select(l, keys, leaf)

                def start_expert(i, slot):
                    # expert id read before the walk: one SMEM load instead of one per leaf
                    e = rget("uniq", i)
                    for_layer(lambda refs, off: expert_dmas(refs, i, slot, e, off))

                def wait_expert(slot, parts=(0, 1, 2, 3)):
                    pack = FP4_PACK if kind == "fp4" else 4
                    kw13, kw2 = He // pack, Il // pack
                    tl = l if isinstance(l, int) else kind_layers[kind][0]
                    trefs, toff = layer_ex[tl]
                    tmpl = [r if toff is None else r.at[toff] for r in trefs]
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
                        if j not in parts:
                            continue
                        src = tmpl[j] if j in (1, 3) else tmpl[j].bitcast(jnp.uint32)
                        pltpu.make_async_copy(
                            src.at[0, pl.ds(0, rows_), pl.ds(0, cols)],
                            dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                            sem.at[slot],
                        ).wait()

                def issue_experts(lo, hi):
                    """Distinct experts [lo, hi) in one walk; the leaf is a small loop (code size:

                    one set of descriptors per leaf).
                    """

                    def leaf(refs, off):
                        def one(t, carry):
                            expert_dmas(refs, t, t % nslot, off=off)
                            return carry

                        jax.lax.fori_loop(lo, hi, one, 0)

                    for_layer(leaf)

                # pairs: even iterations issue two experts per layer walk (prefetch 1-2 deep, 2-3
                # right after the issue): half the walks
                pairs = c.ex_issue_pairs and nslot >= 4
                pdepth = nslot - 2 if pairs else depth

                def do_prologue():
                    if gscale:
                        S["gs_v"][0:3, :] = jnp.concatenate(
                            [
                                sm[SMALL["expert_gs"] : SMALL["expert_gs"] + 1, 0:Ep],
                                sm[
                                    SMALL["expert_gs"] : SMALL["expert_gs"] + 1,
                                    Ep : 2 * Ep,
                                ],
                                sm[
                                    SMALL["expert_gs_down"] : SMALL["expert_gs_down"]
                                    + 1,
                                    0:Ep,
                                ],
                            ],
                            axis=0,
                        )
                        if gs_cp is not None:
                            gs_cp.start()
                    issue_experts(0, jnp.minimum(pdepth, n_u))

                def ebody(i, carry, part=None):
                    slot = i % nslot

                    if pairs:
                        pl.when((i % 2 == 0) & (i + pdepth < n_u))(
                            lambda: issue_experts(
                                i + pdepth, jnp.minimum(i + pdepth + 2, n_u)
                            )
                        )
                    else:

                        @pl.when(i + depth < n_u)
                        def _():
                            start_expert(i + depth, (i + depth) % nslot)

                    # split: the w2 slabs land behind w13 (FIFO), so bd_w13 starts before they do
                    split = c.ex_split_wait
                    with _scope("ex_wait"):
                        wait_expert(slot, (0, 1) if split else (0, 1, 2, 3))

                    def w2_wait():
                        if split:
                            with _scope("ex_wait"):
                                wait_expert(slot, (2, 3))

                    e = rget("uniq", i)

                    def gather_x_base(ps_):
                        if c.xfull_packed:
                            packed = jnp.concatenate(
                                [S["xfull"][pl.ds(p // K, 1), :] for p in ps_], axis=0
                            )
                            return _unpack_bf16x2(packed).astype(jnp.bfloat16)
                        return jnp.concatenate(
                            [S["xfull"][pl.ds(p // K, 1), :] for p in ps_], axis=0
                        ).astype(jnp.bfloat16)

                    def full_expert():
                        with _scope("full_expert"):
                            full_expert_()

                    def busy_bucket(nrows):
                        off, cnt = rget("ex_off", i), rget("ex_cnt", i)
                        ps_ = [
                            rget("perm", off + jnp.minimum(j, cnt - 1))
                            for j in range(nrows)
                        ]
                        with _scope("ex_gather"):
                            xc = gather_x_base(ps_)
                        cw13 = (
                            64
                            if c.ex_opt and c.fp4_chunk_words == 128
                            else c.fp4_chunk_words
                        )
                        if kind == "fp4":
                            gu = _fp4_dot(
                                xc,
                                W13.at[slot],
                                S["w13s"].at[slot],
                                He,
                                2 * Il,
                                cw13,
                                fast_bf16=c.ex_opt,
                                kperm=c.ex_kperm,
                                via_fp8=c.ex_via_fp8,
                                vec_dequant=c.vec_dequant_kperm,
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 1),
                                n_tile=c.bd_n_tile,
                            )
                        else:
                            gu = _fp8_dot(
                                xc,
                                W13.at[slot],
                                S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(jnp.float32),
                                He,
                                2 * Il,
                                c.fp4_chunk_words,
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 1),
                            )
                        gate, up = gu[:, :Il], gu[:, Il:]
                        if gscale:
                            g_gate, g_up, g_down = globals_of(e)
                            gate, up = gate * g_gate, up * g_up
                        h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                        w2_wait()
                        if kind == "fp4":
                            y = _fp4_dot(
                                h,
                                W2.at[slot],
                                S["w2s"].at[slot],
                                Il,
                                He,
                                min(Il // FP4_PACK, 64 if c.ex_opt else 16),
                                fast_bf16=c.ex_opt,
                                kperm=c.ex_kperm,
                                via_fp8=c.ex_via_fp8,
                                vec_dequant=c.vec_dequant_kperm,
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 2),
                                n_tile=c.bd_n_tile,
                            )
                        else:
                            y = _fp8_dot(
                                h,
                                W2.at[slot],
                                S2[slot, 0:1, :].astype(jnp.float32),
                                Il,
                                He,
                                min(Il // 4, 128 if c.ex_opt else 32),
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 2),
                            )
                        yp = _pack_y(y * g_down if gscale else y, c)
                        with _scope("ex_scatter"):
                            for j, p in enumerate(ps_):
                                S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[
                                    j : j + 1
                                ]

                    def full_expert_():
                        cw13 = (
                            64
                            if c.ex_opt and c.fp4_chunk_words == 128
                            else c.fp4_chunk_words
                        )
                        if kind == "fp4":
                            gu = _fp4_dot(
                                x_full,
                                W13.at[slot],
                                S["w13s"].at[slot],
                                He,
                                2 * Il,
                                cw13,
                                fast_bf16=c.ex_opt,
                                kperm=c.ex_kperm,
                                via_fp8=c.ex_via_fp8,
                                vec_dequant=c.vec_dequant_kperm,
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 1),
                                n_tile=c.bd_n_tile,
                            )
                        else:
                            gu = _fp8_dot(
                                x_full,
                                W13.at[slot],
                                S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(jnp.float32),
                                He,
                                2 * Il,
                                c.fp4_chunk_words,
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 1),
                            )
                        gate, up = gu[:, :Il], gu[:, Il:]
                        if gscale:
                            g_gate, g_up, g_down = globals_of(e)
                            gate, up = gate * g_gate, up * g_up
                        h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                        w2_wait()
                        if kind == "fp4":
                            y = _fp4_dot(
                                h,
                                W2.at[slot],
                                S["w2s"].at[slot],
                                Il,
                                He,
                                min(Il // FP4_PACK, 64 if c.ex_opt else 16),
                                fast_bf16=c.ex_opt,
                                kperm=c.ex_kperm,
                                via_fp8=c.ex_via_fp8,
                                vec_dequant=c.vec_dequant_kperm,
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 2),
                                n_tile=c.bd_n_tile,
                            )
                        else:
                            y = _fp8_dot(
                                h,
                                W2.at[slot],
                                S2[slot, 0:1, :].astype(jnp.float32),
                                Il,
                                He,
                                min(Il // 4, 128 if c.ex_opt else 32),
                                fp8_lhs=(c.ex_fp8_lhs or c.ex_lhs_fp8 >= 2),
                            )
                        if gscale:
                            y = y * g_down
                        # slot space: row r, slot k gets y[r] where ids[r, k] == e
                        with _scope("ex_accum"):
                            S["ysend"][...] = jnp.where(
                                ids3 == e, _pack_y(y, c)[:, None, :], S["ysend"][...]
                            )

                    def compact_expert(nrows):
                        # one variant per row bucket: the expanded input and the scale work scale
                        # with the rows; bucket rows past the expert's count repeat its last pair
                        bd_kc = c.blockdiag_kc
                        off, cnt = rget("ex_off", i), rget("ex_cnt", i)
                        ps_ = [
                            rget("perm", off + jnp.minimum(j, cnt - 1))
                            for j in range(nrows)
                        ]
                        with _scope("ex_gather"):
                            xc = gather_x_base(ps_)
                        with _scope("bd_w13"):
                            gu = _fp4_blockdiag_dot(
                                xc,
                                W13.at[slot],
                                S["w13s"].at[slot],
                                He,
                                2 * Il,
                                bd_kc,
                                kperm=bool(c.ex_kperm),
                                fp8_lhs=c.bd_fp8_lhs,
                                n_tile=c.bd_n_tile,
                            )
                            gate, up = gu[:, :Il], gu[:, Il:]
                            if gscale:
                                g_gate, g_up, g_down = globals_of(e)
                                gate, up = gate * g_gate, up * g_up
                            h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                        w2_wait()
                        with _scope("bd_w2"):
                            yc = _fp4_blockdiag_dot(
                                h,
                                W2.at[slot],
                                S["w2s"].at[slot],
                                Il,
                                He,
                                bd_kc,
                                kperm=bool(c.ex_kperm),
                                fp8_lhs=c.bd_fp8_lhs,
                                n_tile=c.bd_n_tile,
                            )
                            yp = _pack_y(yc * g_down if gscale else yc, c)
                        with _scope("ex_scatter"):
                            # a padded row repeats the last pair's row, so its write repeats that
                            # pair's identical write: no branch per row
                            for j, p in enumerate(ps_):
                                S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[
                                    j : j + 1
                                ]

                    def bd_chunks():
                        bd_kc = c.blockdiag_kc
                        off, cnt = rget("ex_off", i), rget("ex_cnt", i)

                        def bd_step(ch, carry_):
                            ps_ = [
                                rget("perm", off + jnp.minimum(ch * 8 + j, cnt - 1))
                                for j in range(8)
                            ]
                            with _scope("ex_gather"):
                                xc = gather_x_base(ps_)
                            with _scope("bd_w13"):
                                gu = _fp4_blockdiag_dot(
                                    xc,
                                    W13.at[slot],
                                    S["w13s"].at[slot],
                                    He,
                                    2 * Il,
                                    bd_kc,
                                    kperm=bool(c.ex_kperm),
                                    fp8_lhs=c.bd_fp8_lhs,
                                    n_tile=c.bd_n_tile,
                                )
                                gate, up = gu[:, :Il], gu[:, Il:]
                                if gscale:
                                    g_gate, g_up, g_down = globals_of(e)
                                    gate, up = gate * g_gate, up * g_up
                                h = (gate * jax.nn.sigmoid(gate) * up).astype(
                                    jnp.bfloat16
                                )
                            pl.when(ch == 0)(w2_wait)
                            with _scope("bd_w2"):
                                yc = _fp4_blockdiag_dot(
                                    h,
                                    W2.at[slot],
                                    S["w2s"].at[slot],
                                    Il,
                                    He,
                                    bd_kc,
                                    kperm=bool(c.ex_kperm),
                                    fp8_lhs=c.bd_fp8_lhs,
                                    n_tile=c.bd_n_tile,
                                )
                                yp = _pack_y(yc * g_down if gscale else yc, c)
                            with _scope("ex_scatter"):
                                for j, p in enumerate(ps_):
                                    S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[
                                        j : j + 1
                                    ]
                            return carry_

                        jax.lax.fori_loop(0, (cnt + 7) // 8, bd_step, 0)

                    if kind == "fp4" and c.compact_rows:
                        cnt = rget("ex_cnt", i)
                        # (upper count bound, variant) per bucket, picked by a compare tree
                        # an expert has at most one pair per row: a bucket of R rows covers all
                        buckets = compact_buckets(c)
                        arms = [
                            (b, functools.partial(compact_expert, b)) for b in buckets
                        ]
                        if buckets[-1] < R:
                            can_bd_base = c.ex_kperm == 0 or c.blockdiag_kc == 256
                            if c.ex_bd_all and can_bd_base:
                                arms.append((None, bd_chunks))
                            else:
                                lo_arm = buckets[-1]
                                if c.ex_bd_max_rows > lo_arm and can_bd_base:
                                    bd_cap = min(c.ex_bd_max_rows, R)
                                    if bd_cap >= R:
                                        arms.append((None, bd_chunks))
                                    else:
                                        arms.append((bd_cap, bd_chunks))
                                    lo_arm = bd_cap
                                if lo_arm < R:
                                    if c.ex_opt and R <= 48:
                                        bb_list = (
                                            (8, 16, 24, 32)
                                            if c.ex_w2_stage
                                            else (16, 24, 32)
                                        )
                                        for bb in bb_list:
                                            if lo_arm < bb < R:
                                                arms.append(
                                                    (
                                                        bb,
                                                        functools.partial(
                                                            busy_bucket, bb
                                                        ),
                                                    )
                                                )
                                        arms.append(
                                            (None, functools.partial(busy_bucket, R))
                                        )
                                    else:
                                        arms.append((None, full_expert))
                        else:
                            arms[-1] = (None, arms[-1][1])
                        if part == "few":
                            arms = [
                                a
                                for a in arms
                                if a[0] is not None and a[0] <= c.split_rows
                            ]
                            arms[-1] = (None, arms[-1][1])
                        elif part == "many":
                            arms = [
                                a for a in arms if a[0] is None or a[0] > c.split_rows
                            ]

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
                        full_expert()
                    if y_block:
                        # carry: the next owner block to send. Block nb is final once its
                        # group's last expert is done; an empty group fires with the one before
                        # (at expert 0 if no group before it has an expert).
                        def ready(nb):
                            end = rget("blk_end", jnp.minimum(nb, c.tp - 2))
                            return (nb < c.tp - 1) & (end <= i + 1)

                        def send_block(nb):
                            owner = blk_owner(nb)
                            pl.when(owner * Rc < n_live)(
                                coll.block_copy("y", S["ysend"], owner, Rc).start
                            )
                            return nb + 1

                        return jax.lax.while_loop(ready, send_block, carry)
                    return carry

                def do_loop():
                    if gs_cp is not None:
                        gs_cp.wait()
                    with _scope("experts"):
                        if c.split_rows and kind == "fp4":

                            def group(g_, carry):
                                lo = jnp.where(
                                    g_ == 0, 0, rget("blk_end", jnp.maximum(g_ - 1, 0))
                                )
                                mid, hi = rget("blk_mid", g_), rget("blk_end", g_)
                                carry = jax.lax.fori_loop(
                                    lo, mid, functools.partial(ebody, part="few"), carry
                                )
                                return jax.lax.fori_loop(
                                    mid,
                                    hi,
                                    functools.partial(ebody, part="many"),
                                    carry,
                                )

                            jax.lax.fori_loop(
                                0, c.tp if y_block else 1, group, jnp.int32(0)
                            )
                        else:
                            jax.lax.fori_loop(0, n_u, ebody, jnp.int32(0))

                return do_prologue, do_loop

            kinds_here = (
                ([c.expert_kinds[l]] if isinstance(l, int) else sorted(kind_layers))
                if kind is None
                else [kind]
            )
            fns = {k: kind_fns(k) for k in kinds_here}

            def by_kind(j):
                if len(kinds_here) == 1:
                    fns[kinds_here[0]][j]()
                    return
                is_fp8 = False
                for x in kind_layers["fp8"]:
                    is_fp8 = jnp.logical_or(is_fp8, l == x)
                jax.lax.cond(is_fp8, fns["fp8"][j], fns["fp4"][j])

            by_kind(0)

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
                S["ysend"][...] = jnp.zeros((R, K, yw), jnp.uint32)
                if c.xfull_packed:
                    S["xfull"][...] = _pack_bf16x2(x_full.astype(jnp.float32))
                else:
                    S["xfull"][...] = x_full.astype(jnp.float32)

            by_kind(1)
            # flushing inside the expert loop (once its prefetch ends) measured slower
            ws.flush()
            # ---- token-sharded reduce of the per-rank partials, then norm + combine
            with _scope("y_rs"):
                pad = coll.reduce_scatter_rows(
                    "y", S["ysend"], Rc, n_live, started=y_block
                )
            with _scope("combine"):
                ysum = _unpack_y(pad[0], c)
                for d in range(1, c.tp):
                    ysum = ysum + _unpack_y(pad[d], c)  # [Rc, K, He]
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
            if not c.smalls_early:
                prefetch_smalls(l, nxt)

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

        def prefetch_smalls(l, nxt):
            """The next layer's smalls into the slot layer l - 1 used."""
            if nxt is None:
                return
            if isinstance(l, int):
                smalls_copy(l + 1, (l + 1) % 2).start()
            else:
                pl.when(l + 1 < L)(smalls_copy(l + 1, (l + 1) % 2).start)

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
            """carry: the layer's input stream, or with pend (fold_f) the previous

            layer's stream before its ffn residual, whose f is still arriving. Returns
            the
            output stream, or with fold_f and a next layer the stream before the ffn
            residual.
            """
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
            if c.smalls_early:  # layer l - 1's smalls (fold_qkvg) are read
                prefetch_smalls(l, nxt)
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

        # ============================================================ row-tiled program
        # Phase by phase over the whole step: each dense weight tile and expert slab is read
        # from HBM once and applied to every row; per-row passes run RT rows at a time.
        RT = c.row_tile
        NT = R // RT if RT else 0
        f32, bf16 = jnp.float32, jnp.bfloat16

        def tsem(i):
            return S["tile_sems"].at[i]

        def rows_of(t):
            return pl.ds(pl.multiple_of(t * RT, RT), RT)

        def for_tiles(fn):
            """fn(rows, t) for every row tile in one runtime loop (one emitted body)."""

            def body(t, carry):
                fn(rows_of(t), t)
                return carry

            jax.lax.fori_loop(0, NT, body, 0)

        def dma(src, dst, i):
            cp = pltpu.make_async_copy(src, dst, tsem(i))
            cp.start()
            cp.wait()

        def tile_pass(loads, compute, stores=(), ready=None):
            """for_tiles with the HBM traffic double-buffered.

            Tile t + 1's loads and tile t - 1's stores fly while tile t computes.
            loads / stores: (hbm_view_of(rows), buf [2, ...]); compute(rows, t, loaded
            views) returns the values to store. ready[i](t), if given, waits until
            load i's rows of tile t have landed (a pipelined all-gather) before that
            load is issued.
            """
            nl = len(loads)
            ready = ready or [None] * nl

            def ld_start(i, t, s_):
                if ready[i] is not None:
                    ready[i](t)
                ld(i, t, s_).start()

            def ld(i, t, s_):
                src, buf = loads[i]
                return pltpu.make_async_copy(
                    src(rows_of(t)), buf.at[s_], S["pass_sems"].at[s_, i]
                )

            def st(j, t, s_):
                dst, buf = stores[j]
                return pltpu.make_async_copy(
                    buf.at[s_], dst(rows_of(t)), S["pass_sems"].at[s_, nl + j]
                )

            for i in range(nl):
                ld_start(i, 0, 0)

            def body(t, carry):
                s_ = t % 2
                for i in range(nl):
                    ld(i, t, s_).wait()

                @pl.when(t + 1 < NT)
                def _():
                    for i in range(nl):
                        ld_start(i, t + 1, 1 - s_)

                vals = compute(rows_of(t), t, [buf.at[s_] for _, buf in loads])
                for j, v in enumerate(vals):
                    pl.when(t >= 2)(lambda j=j: st(j, t - 2, s_).wait())
                    stores[j][1].at[s_][...] = v
                    st(j, t, s_).start()
                return carry

            jax.lax.fori_loop(0, NT, body, 0)
            for j in range(len(stores)):
                for t in range(max(0, NT - 2), NT):
                    st(j, t, t % 2).wait()

        def stream_view(rows):
            return S["h_stream"].at[rows]

        def tile_live(t):
            return jax.lax.broadcasted_iota(jnp.int32, (RT, 1), 0) + t * RT < n_live

        def pad_view(pad):
            return lambda rows: pad.at[:, rows]

        def gathered_tile(v):
            """A row tile of an all-gathered [tp, R, w/2] pad -> f32 [RT, tp * w]."""
            return jnp.concatenate([_unpack_bf16x2(v[d]) for d in range(c.tp)], axis=1)

        tc_smap = _TC_VPADS_S if (c.tiled_vmem_pads and c.trim_sems) else _TC
        tc_rmap = _TC_VPADS_R if (c.tiled_vmem_pads and c.trim_sems) else _TC
        use_pair_y = bool(
            c.row_tile and c.tiled_pair_y >= 1 and c.tp == 8 and c.tiled_vmem_y >= 2
        )
        stream_pair_y = bool(use_pair_y and y_block and c.pair_y_stream)

        def rcopy(src, dst, peer, kind, send_kind=None):
            return pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=S["tc_ssems"].at[tc_smap[send_kind or kind]],
                recv_sem=S["tc_rsems"].at[tc_rmap[kind]],
                device_id=(peer,),
            )

        def ag_tile_start(kind, src, pad, slot, t, sems):
            """Sends row tile t of src to pad[slot] on every device once it is written.

            src is VMEM [R, w] and pad HBM [tp, R, w]; the receivers' sems[t] count
            the copies.
            """
            rows = rows_of(t)
            pltpu.make_async_copy(src.at[rows], pad.at[slot, rows], sems.at[t]).start()
            for d in range(1, c.tp):
                pltpu.make_async_remote_copy(
                    src_ref=src.at[rows],
                    dst_ref=pad.at[slot, rows],
                    send_sem=S["tc_ssems"].at[_TC[kind]],
                    recv_sem=sems.at[t],
                    device_id=(coll._peer(d),),
                ).start()

        def ag_tile_ready(src, pad, sems):
            """ready(t): every device's row tile t of a pipelined all-gather has landed."""

            def ready(t):
                tmpl = pltpu.make_async_copy(
                    src.at[pl.ds(0, RT)], pad.at[0, pl.ds(0, RT)], sems.at[t]
                )
                for _ in range(c.tp):
                    tmpl.wait()

            return ready

        def ag_tile_sent(kind, src, pad):
            """Waits for this device's sends of a pipelined all-gather (before src is reused)."""
            for d in range(1, c.tp):
                rcopy(src, pad.at[0], coll._peer(d), kind).wait_send()

        def ag_start(kind, src, dst, live=None):
            """src (VMEM) into dst, the same HBM region on every device; returns the wait.

            live(m): whether device m's part is needed (dead parts are neither sent
            nor waited).
            """
            local = pltpu.make_async_copy(src, dst, tsem(_TS_LOCAL))
            local.start()
            cps = [
                (coll._peer(d), rcopy(src, dst, coll._peer(d), kind))
                for d in range(1, c.tp)
            ]

            def start():
                for _, cp in cps:
                    cp.start()

            if live is None:
                start()
            else:
                pl.when(live(me))(start)

            def wait():
                local.wait()
                if live is None:
                    for _, cp in cps:
                        cp.wait()
                    return

                def wait_sends():
                    for _, cp in cps:
                        cp.wait_send()

                pl.when(live(me))(wait_sends)
                for peer, cp in cps:
                    pl.when(live(peer))(cp.wait_recv)

            return wait

        def route_tiled():
            """_route_table from ids_v row tile by row tile: per-expert counts and keys, then

            each pair's slot in perm from a running per-expert count; temporaries stay
            [RT * top_k, n_experts] instead of [R * top_k, R * top_k].
            """
            Ep, NB, Rc = c.n_experts_padded, NBR, c.rows_per_dev
            big = float(c.tp + 1)
            i32 = jnp.int32

            def tile_ids(t):
                ids = S["ids_v"][rows_of(t), pl.ds(0, K)]  # [RT, K]
                r = jax.lax.broadcasted_iota(i32, (RT, 1), 0) + t * RT
                return ids, r

            def key_of(r):
                if stream_pair_y:
                    return (3 - jnp.bitwise_xor(r // (2 * Rc), me // 2)).astype(f32)
                if not y_block or use_pair_y:
                    return jnp.zeros(r.shape, f32)
                o = r // Rc
                if coll.order == "xor":
                    return ((c.tp - 1) - jnp.bitwise_xor(o, me)).astype(f32)
                return jax.lax.rem(o - me - 1 + c.tp, c.tp).astype(f32)

            R_ct = (
                64
                if (c.topk_xpose and R % 64 == 0)
                else (32 if (c.topk_xpose and R % 32 == 0) else RT)
            )

            def count_tile(t, carry):
                count, key = carry
                ids = S["ids_v"][
                    pl.ds(pl.multiple_of(t * R_ct, R_ct), R_ct), pl.ds(0, K)
                ]
                r = jax.lax.broadcasted_iota(i32, (R_ct, 1), 0) + t * R_ct
                live, kr = r < n_live, key_of(r)
                e_lane = jax.lax.broadcasted_iota(i32, (R_ct, Ep), 1)
                for k in range(K):
                    hit = (ids[:, k : k + 1] == e_lane) & live
                    count = count + jnp.sum(
                        jnp.where(hit, 1.0, 0.0), axis=0, keepdims=True
                    )
                    key = jnp.minimum(
                        key, jnp.min(jnp.where(hit, kr, big), axis=0, keepdims=True)
                    )
                return count, key

            count, key = jax.lax.fori_loop(
                0,
                R // R_ct,
                count_tile,
                (jnp.zeros((1, Ep), f32), jnp.full((1, Ep), big, f32)),
            )
            present = count > 0
            split = c.split_rows
            mult = 2 if split else 1
            if split:
                key = 2 * key + jnp.where(count > split, 1.0, 0.0)
            e_row = jax.lax.broadcasted_iota(i32, (1, Ep), 1).astype(f32)
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
            key_c, id_c, cnt_c, pres_c = (
                cols[:, 0:1],
                cols[:, 1:2],
                cols[:, 2:3],
                cols[:, 3:4],
            )
            earlier = present & ((key < key_c) | ((key == key_c) & (e_row < id_c)))
            pos_c = jnp.sum(jnp.where(earlier, 1.0, 0.0), axis=1, keepdims=True).astype(
                i32
            )
            off_c = jnp.sum(jnp.where(earlier, count, 0.0), axis=1, keepdims=True)
            off_e = jnp.transpose(
                jnp.concatenate([off_c, jnp.zeros((Ep, LANE - 1), f32)], axis=1)
            )[0:1]
            rv = S["route_v"]
            rv[...] = jnp.zeros(rv.shape, i32)
            for b in range(NB):
                u_lane = jax.lax.broadcasted_iota(i32, (Ep, LANE), 1) + b * LANE
                m = (pres_c > 0) & (pos_c == u_lane)
                for f, v in enumerate((id_c, cnt_c, off_c)):
                    rv[f * NB + b : f * NB + b + 1, :] = jnp.sum(
                        jnp.where(m, v, 0.0), axis=0, keepdims=True
                    ).astype(i32)
            n_u = jnp.sum(jnp.where(present, 1, 0), axis=1, keepdims=True)
            rv[4 * NB : 4 * NB + 1, :] = jnp.broadcast_to(n_u, (1, LANE))
            b_lane = jax.lax.broadcasted_iota(i32, (Ep, LANE), 1).astype(f32)
            rv[4 * NB + 1 : 4 * NB + 2, :] = jnp.sum(
                jnp.where((pres_c > 0) & (key_c <= mult * b_lane + mult - 1), 1, 0),
                axis=0,
                keepdims=True,
            )
            if split:
                rv[4 * NB + 2 : 4 * NB + 3, :] = jnp.sum(
                    jnp.where((pres_c > 0) & (key_c <= 2 * b_lane), 1, 0),
                    axis=0,
                    keepdims=True,
                )
            PT = RT * K

            def perm_tile(t, running):
                ids, _ = tile_ids(t)
                col_e = jnp.concatenate([ids[:, k : k + 1] for k in range(K)], axis=0)
                q_ = jax.lax.broadcasted_iota(i32, (PT, 1), 0)
                r_of = q_ % RT + t * RT
                enc = r_of * K + q_ // RT
                live = r_of < n_live
                hit = (col_e == jax.lax.broadcasted_iota(i32, (PT, Ep), 1)) & live
                onehot = jnp.where(hit, 1.0, 0.0)
                same = jax.lax.dot_general(
                    onehot.astype(bf16),
                    onehot.astype(bf16),
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=f32,
                )  # [PT, PT]
                q_row = jax.lax.broadcasted_iota(i32, (PT, PT), 0)
                q_col = jax.lax.broadcasted_iota(i32, (PT, PT), 1)
                live_col = (q_col % RT + t * RT) < n_live
                rank = jnp.sum(
                    jnp.where((q_col < q_row) & live_col, same, 0.0),
                    axis=1,
                    keepdims=True,
                )
                dest = (
                    jnp.sum(onehot * (off_e + running), axis=1, keepdims=True) + rank
                ).astype(i32)
                for b in range(NB):
                    t_lane = jax.lax.broadcasted_iota(i32, (PT, LANE), 1) + b * LANE
                    rv[3 * NB + b : 3 * NB + b + 1, :] = rv[
                        3 * NB + b : 3 * NB + b + 1, :
                    ] + jnp.sum(
                        jnp.where((dest == t_lane) & live, enc, 0),
                        axis=0,
                        keepdims=True,
                    )
                return running + jnp.sum(onehot, axis=0, keepdims=True)

            jax.lax.fori_loop(0, NT, perm_tile, jnp.zeros((1, Ep), f32))
            cp = pltpu.make_async_copy(rv, S["route_s"], S["route_sem"].at[0])
            cp.start()
            cp.wait()
            return S["route_s"][4 * NB, 0]

        def mv_tiled(l, name, nxt, smr, col_off=None, on_col=None):
            """S["lhs"] [R, K] @ W_l[name] -> S["acc"] columns (scaled fp8); on_col(ni, col) once

            column tile ni is final. One MXU pass per mv_rows rows per weight tile.
            """
            assert name not in ws.k_rot
            t0, t1, tk, tn, nn, dt = ws.ranges[name]
            nk = (t1 - t0) // nn
            is_fp8 = jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn)
            MR = min(R, c.mv_rows)
            assert R % MR == 0
            for t in range(t0, t1):
                with _scope("ring_wait"):
                    ws._copy(l, t).wait()
                _, ki, ni, _, _, _ = ws.tiles[t]
                col = ws.phys_col(name, ni)
                c0 = ni * tn if col_off is None else col_off(ni)
                w = ws.buf.at[t % ws.slots].bitcast(dt)[pl.ds(0, tk), pl.ds(0, tn)]
                for m0 in range(0, R, MR):
                    lhs_sub = S["lhs"][pl.ds(m0, MR), pl.ds(ki * tk, tk)]
                    if c.dense_fp8_lhs and is_fp8 and MR >= 32:
                        xf_d, xsc_d = _quant_fp8_rows(lhs_sub)
                        part = jnp.dot(xf_d, w, preferred_element_type=f32) * xsc_d
                    else:
                        part = jnp.dot(
                            lhs_sub,
                            w,
                            preferred_element_type=f32,
                        )
                    if ki > 0:
                        part = S["acc"][pl.ds(m0, MR), pl.ds(c0, tn)] + part
                    if ki == nk - 1 and is_fp8:
                        r = SMALL[name + "_s"]
                        sc_col = (
                            smr[r : r + 1, pl.ds(ni * tn, tn)]
                            if isinstance(col, int)
                            else smr[
                                r : r + 1, pl.ds(pl.multiple_of(col * tn, LANE), tn)
                            ]
                        )
                        part = part * sc_col
                    S["acc"][pl.ds(m0, MR), pl.ds(c0, tn)] = part
                ws.refill(l, t, nxt)
                if on_col is not None and ki == nk - 1:
                    on_col(ni, col)

        def tiled_fold_lhs(pad):
            def lhs(k):
                cols = pl.ds(pl.multiple_of(k * Hl, LANE), Hl)
                part = S["lhs"][:, cols]
                if pad is None:
                    return part
                blk = (_unpack_bf16x2(S[pad][k]) * S["fold_v"][0:1, cols]).astype(bf16)
                return jnp.concatenate([part, blk], axis=0)

            return lhs

        def attention_tiled(l, use_rope, is_global, nxt, smr, pend=False):
            """Leaves the all-reduced o-proj output all-gathered in pad_o_ag (or in flight)."""
            per_geom(is_global, lambda geo: kv_prologue(geo, l, 0))
            fill_preones()
            qd, kd = c.q_dim, c.kv_dim
            if pend:
                psmr = S["smalls"].at[(l + 1) % 2]
                g_pa = smr[SMALL["pre_attn_w"] : SMALL["pre_attn_w"] + 1, :H]
                p_alpha = psmr[SMALL["ffn_alpha"] : SMALL["ffn_alpha"] + 1, :H]
                p_beta = psmr[SMALL["ffn_beta"] : SMALL["ffn_beta"] + 1, :H]
                p_pfw = psmr[SMALL["post_ffn_w"] : SMALL["post_ffn_w"] + 1, :H]
                for t in range(R // RT):
                    rs = pl.ds(t * RT, RT)
                    S["lhs"][rs] = (p_alpha * S["h_stream"][rs] * g_pa).astype(bf16)
                S["fold_v"][0:1, :] = p_beta * p_pfw * g_pa
                wait_block, finish = f_blocks()
                with _scope("qkvg_fold"):
                    acc_q = ws.mv_fold(
                        l,
                        ("wqkvg",),
                        nxt,
                        _fold_g(c, ("wqkvg",)),
                        Hl,
                        wait_block,
                        tiled_fold_lhs("pad_f"),
                    )["wqkvg"]
                finish()
                wq_sc = smr[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]
                for t in range(R // RT):
                    r0, r1 = t * RT, (t + 1) * RT
                    rs = pl.ds(r0, RT)
                    y = gathered_tile(S["pad_f"].at[:, rs]) * p_pfw
                    s_f = jax.lax.rsqrt(
                        jnp.mean(jnp.square(y), axis=-1, keepdims=True) + c.post_eps
                    )
                    s = p_alpha * S["h_stream"][rs] + p_beta * (y * s_f).astype(
                        bf16
                    ).astype(f32)
                    capture_tile(l - 1, rs, s)
                    S["h_stream"][rs] = s
                    r_inv = jax.lax.rsqrt(
                        jnp.mean(jnp.square(s), axis=-1, keepdims=True) + c.rms_eps
                    )
                    S["acc"][rs, pl.ds(0, c.qkvg_dim)] = (
                        r_inv * (acc_q[r0:r1] + s_f * acc_q[R + r0 : R + r1]) * wq_sc
                    )
            elif c.fold_f:
                with _scope("qkvg_mv"):
                    acc_q = ws.mv_fold(
                        l,
                        ("wqkvg",),
                        nxt,
                        _fold_g(c, ("wqkvg",)),
                        Hl,
                        lambda k: None,
                        tiled_fold_lhs(None),
                    )["wqkvg"]
                wq_sc = smr[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]
                S["acc"][:, pl.ds(0, c.qkvg_dim)] = acc_q * wq_sc
            else:
                with _scope("qkvg_mv"):
                    mv_tiled(l, "wqkvg", nxt, smr)

            def qkv_tile(rows, t):
                qkvg = S["acc"][rows, pl.ds(0, c.qkvg_dim)]
                qv, kv_, vv = (
                    qkvg[:, :qd],
                    qkvg[:, qd : qd + kd],
                    qkvg[:, qd + kd : qd + 2 * kd],
                )
                rt_ = S["rtab"][rows, :]
                cos_t, sin_t = rt_[:, :hd], rt_[:, hd:]
                qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
                kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
                qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_t, sin_t), qn)
                kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_t, sin_t), kn)
                kb = kn.astype(c.kv_dtype).astype(bf16)
                vb = vv.astype(c.kv_dtype).astype(bf16)
                S["qn_s"][rows] = qn
                S["kb_s"][rows] = kb.astype(f32)
                S["vb_s"][rows] = vb.astype(f32)
                S["attn_buf"][rows] = jnp.zeros((RT, qd), f32)
                if c.kv_sal:
                    return
                r0 = t * RT
                for r in range(RT):
                    if c.kv_packed:
                        S["kv_row"][r0 + r] = jnp.concatenate(
                            [kb[r : r + 1], vb[r : r + 1]], axis=0
                        ).reshape(1, 2, pd)
                        continue
                    zp = jnp.zeros((1, pd - hd), bf16)
                    parts = []
                    for h in range(hkv):
                        for x in (kb, vb):
                            parts.append(
                                jnp.concatenate(
                                    [x[r : r + 1, h * hd : (h + 1) * hd], zp], axis=1
                                )
                            )
                    S["kv_row"][r0 + r] = jnp.concatenate(parts, axis=0).reshape(
                        hkv, 2, pd
                    )

            if c.ex_opt and c.kv_sal and R <= c.untiled_max_rows:
                qkvg = S["acc"][:, pl.ds(0, c.qkvg_dim)]
                qv, kv_, vv = (
                    qkvg[:, :qd],
                    qkvg[:, qd : qd + kd],
                    qkvg[:, qd + kd : qd + 2 * kd],
                )
                rt_ = S["rtab"][...]
                cos_t, sin_t = rt_[:, :hd], rt_[:, hd:]
                qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
                kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
                qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_t, sin_t), qn)
                kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_t, sin_t), kn)
                kb = kn.astype(c.kv_dtype).astype(bf16)
                vb = vv.astype(c.kv_dtype).astype(bf16)
                S["qn_s"][...] = qn
                S["kb_s"][...] = kb.astype(f32)
                S["vb_s"][...] = vb.astype(f32)
                S["attn_buf"][...] = jnp.zeros((R, qd), f32)
            else:
                for_tiles(qkv_tile)
            if (
                c.kv_sal
            ):  # [slot * words + word, row]: an identity matmul transposes exactly
                if not (c.ex_opt and c.kv_sal and R <= c.untiled_max_rows):
                    kb = S["kb_s"][...].astype(bf16)
                    vb = S["vb_s"][...].astype(bf16)
                eye = (
                    jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 0)
                    == jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 1)
                ).astype(bf16)
                S["kv_row"][...] = jnp.concatenate(
                    [
                        pltpu.bitcast(
                            jax.lax.dot_general(
                                eye,
                                x[:, h * hd : (h + 1) * hd],
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=f32,
                            ).astype(c.kv_dtype),
                            jnp.uint32,
                        )
                        for h in range(hkv)
                        for x in (kb, vb)
                    ],
                    axis=0,
                )
            kv_out_cp = pltpu.make_async_copy(
                S["kv_row"], newkv_out.at[l], S["kv_row_sem"].at[0]
            )
            kv_out_cp.start()
            with _scope("attn_core"):
                n_req_live = jnp.minimum((n_live + q - 1) // q, c.n_req)

                def run_geom(geo):
                    def req_loop(j, carry):
                        request_attention_body(geo, l, j, n_req_live)
                        return carry

                    jax.lax.fori_loop(0, n_req_live, req_loop, 0)

                per_geom(is_global, run_geom)
                kv_out_cp.wait()

            if (c.ex_opt or c.tiled_vmem_pads) and R <= c.untiled_max_rows:
                attn_all = (
                    _heads_rms(S["attn_buf"][...], hq, hd, c.rms_eps)
                    .astype(bf16)
                    .astype(f32)
                )
                og_all = S["acc"][:, pl.ds(qd + 2 * kd, qd)]
                S["lhs"][:, pl.ds(0, qd)] = (
                    jax.nn.sigmoid(og_all.astype(bf16).astype(f32)) * attn_all
                ).astype(bf16)
            else:

                def o_in_tile(rows, t):
                    attn = (
                        _heads_rms(S["attn_buf"][rows], hq, hd, c.rms_eps)
                        .astype(bf16)
                        .astype(f32)
                    )
                    og = S["acc"][rows, pl.ds(qd + 2 * kd, qd)]
                    S["lhs"][rows, pl.ds(0, qd)] = (
                        jax.nn.sigmoid(og.astype(bf16).astype(f32)) * attn
                    ).astype(bf16)

                for_tiles(o_in_tile)
            # o-proj column tile m is reduce-scatter block m; two (or four) send buffers alternate
            assert all(t[4] == Hl for t in ws.tiles if t[0] == "wo")
            obuf = S["o_buf"]
            ors_kinds = ("o_rs", "o_rs1", "o_rs2", "o_rs3")
            nb_o = 4 if (c.tiled_o_rot or use_pair_o_rs) else 2

            def o_rc(b, peer):
                return rcopy(
                    obuf.at[b],
                    S["pad_o_rs"].at[me],
                    peer,
                    "o_rs",
                    send_kind=ors_kinds[b],
                )

            def o_lc(b):
                return pltpu.make_async_copy(
                    obuf.at[b], S["pad_o_rs"].at[me], tsem(_TS_LOCAL)
                )

            def o_wait(m):
                pl.when(me != m)(lambda: o_rc(m % 2, m).wait_send())
                if not c.tiled_vmem_pads:
                    pl.when(me == m)(lambda: o_lc(m % 2).wait())

            def o_pair_cp(b):
                return pltpu.make_async_remote_copy(
                    src_ref=obuf.at[b],
                    dst_ref=S["pad_o_rs"].at[b],
                    send_sem=S["pair_ssems"].at[6],
                    recv_sem=S["pair_rsems"].at[6],
                    device_id=(jnp.bitwise_xor(me, 1),),
                )

            def o_ship(ni, m):
                if use_pair_o_rs:
                    b_p = ni // 2
                    packed = _pack_bf16x2(S["acc"][:, pl.ds(0, Hl)])

                    @pl.when((ni % 2) == (me % 2))
                    def _():
                        S["xfull"][:, pl.ds(b_p * (Hl // 2), Hl // 2)] = packed

                    @pl.when((ni % 2) != (me % 2))
                    def _():
                        obuf[b_p] = packed
                        o_pair_cp(b_p).start()

                    return

                b = ni % nb_o
                if c.tiled_o_rot:
                    if ni >= nb_o and ni - nb_o < c.tp - 1:
                        o_rc(
                            (ni - nb_o) % nb_o, ws.phys_col("wo", ni - nb_o)
                        ).wait_send()
                elif ni >= 2:
                    o_wait(ni - 2)

                if c.tiled_o_rot or (
                    (c.ex_opt or c.tiled_vmem_pads) and R <= c.untiled_max_rows
                ):
                    obuf.at[b][...] = _pack_bf16x2(S["acc"][:, pl.ds(0, Hl)])
                else:

                    def pack_tile(rows, t):
                        obuf.at[b][rows] = _pack_bf16x2(S["acc"][rows, pl.ds(0, Hl)])

                    for_tiles(pack_tile)
                if c.tiled_o_rot:
                    if ni < c.tp - 1:
                        o_rc(b, m).start()
                    elif c.tiled_vmem_pads:
                        S["pad_o_rs"][me] = obuf[b]
                    else:
                        o_lc(b).start()
                else:
                    pl.when(me != m)(lambda: o_rc(b, m).start())
                    if c.tiled_vmem_pads:
                        pl.when(me == m)(lambda: S["pad_o_rs"].__setitem__(me, obuf[b]))
                    else:
                        pl.when(me == m)(lambda: o_lc(b).start())

            with _scope("o_mv"):
                mv_tiled(l, "wo", nxt, smr, col_off=lambda ni: 0, on_col=o_ship)
            with _scope("o_allreduce"):
                if use_pair_o_rs:
                    core_o = me % 2
                    chip_o = me // 2
                    for b_p in range(4):
                        o_pair_cp(b_p).wait()
                    if c.combine_loop and R > 64:
                        for b_p in range(4):
                            for t in range(R // RT):
                                rs = pl.ds(t * RT, RT)
                                obuf.at[b_p][rs] = _pack_bf16x2(
                                    _unpack_bf16x2(
                                        S["xfull"][rs, pl.ds(b_p * (Hl // 2), Hl // 2)]
                                    )
                                    + _unpack_bf16x2(S["pad_o_rs"][b_p, rs])
                                )
                    else:
                        for b_p in range(4):
                            obuf[b_p] = _pack_bf16x2(
                                _unpack_bf16x2(
                                    S["xfull"][:, pl.ds(b_p * (Hl // 2), Hl // 2)]
                                )
                                + _unpack_bf16x2(S["pad_o_rs"][b_p])
                            )
                    S["pad_o_rs"][4 + chip_o] = obuf.at[chip_o][...]
                    ors_cps = [
                        rcopy(
                            obuf.at[jnp.bitwise_xor(chip_o, d)],
                            S["pad_o_rs"].at[4 + chip_o],
                            2 * jnp.bitwise_xor(chip_o, d) + core_o,
                            "o_rs",
                        )
                        for d in (1, 2, 3)
                    ]
                    for cp in ors_cps:
                        cp.start()
                    for cp in ors_cps:
                        cp.wait_send()
                    for cp in ors_cps:
                        cp.wait_recv()
                    for t in range(R // RT):
                        rs = pl.ds(t * RT, RT)
                        mine = _unpack_bf16x2(S["pad_o_rs"][4, rs])
                        for d in range(5, 8):
                            mine = mine + _unpack_bf16x2(S["pad_o_rs"][d, rs])
                        S["o_send"][rs] = _pack_bf16x2(mine)
                    if c.fold_o:
                        coll.all_gather_blocks_start(
                            "o_ag", S["o_send"], me, S["o_rsems"]
                        )
                    else:
                        with _scope("o_ag"):
                            coll.all_gather("o_ag", S["o_send"], me)
                else:
                    if c.tiled_o_rot:
                        for ni in range(max(0, c.tp - nb_o), c.tp - 1):
                            o_rc(ni % nb_o, ws.phys_col("wo", ni)).wait_send()
                        if not c.tiled_vmem_pads:
                            o_lc((c.tp - 1) % nb_o).wait()
                    else:
                        for m in range(max(0, c.tp - 2), c.tp):
                            o_wait(m)
                    for d in range(1, c.tp):
                        o_rc(0, coll._peer(d)).wait_recv()

                    if c.tiled_vmem_pads:
                        for t in range(R // RT):
                            rs = pl.ds(t * RT, RT)
                            mine = _unpack_bf16x2(S["pad_o_rs"][0, rs])
                            for d in range(1, c.tp):
                                mine = mine + _unpack_bf16x2(S["pad_o_rs"][d, rs])
                            S["o_send"][rs] = _pack_bf16x2(mine)
                        if c.fold_o:
                            coll.all_gather_blocks_start(
                                "o_ag", S["o_send"], me, S["o_rsems"]
                            )
                        else:
                            with _scope("o_ag"):
                                coll.all_gather("o_ag", S["o_send"], me)
                    else:

                        def o_sum_tile(rows, t, v):
                            mine = _unpack_bf16x2(v[0][0])
                            for d in range(1, c.tp):
                                mine = mine + _unpack_bf16x2(v[0][d])
                            S["o_send"][rows] = _pack_bf16x2(mine)
                            # pipelined all-gather: the post-attention pass waits per row tile
                            ag_tile_start(
                                "o_ag",
                                S["o_send"],
                                S["pad_o_ag"],
                                me,
                                t,
                                S["oag_tsems"],
                            )
                            return []

                        tile_pass([(pad_view(S["pad_o_rs"]), S["ptile"])], o_sum_tile)

        def expert_io_tiled(l, kind, n_u):
            """Expert slab ring of layer l for the row-tiled expert loop (the base program's

            kind_fns DMAs, stacked groups included): starts the first depth experts'
            DMAs.
            """
            if kind == "fp4":
                nslot, W13, W2 = c.expert_depth + 1, S["w13"], S["w2"]
            else:
                nslot = (c.expert_depth + 1) // 2
                W13 = S["w13"].reshape(nslot, He // 4, 2 * Il)
                W2 = S["w2"].reshape(nslot, Il // 4, He)
            depth = nslot - 1
            S13, S2 = (
                S["w13s"],
                S["w2s"],
            )  # scale_fp8 is off in row-tiled steps (validate)
            pack = FP4_PACK if kind == "fp4" else 4
            parts = (
                (W13, S["w13_sems"], He // pack, 2 * Il, False),
                (
                    S13,
                    S["w13s_sems"],
                    He // FP4_GROUP if kind == "fp4" else 2,
                    2 * Il,
                    True,
                ),
                (W2, S["w2_sems"], Il // pack, He, False),
                (
                    S2,
                    S["w2s_sems"],
                    Il // FP4_GROUP if kind == "fp4" else 2,
                    He,
                    True,
                ),
            )

            def slab_copy(refs, off, e, slot, j):
                dst, sem, rows_, cols, is_scale = parts[j]
                src = refs[j] if off is None else refs[j].at[off]
                src = src if is_scale else src.bitcast(jnp.uint32)
                return pltpu.make_async_copy(
                    src.at[e, pl.ds(0, rows_), pl.ds(0, cols)],
                    dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                    sem.at[slot],
                )

            def for_layer(fn):
                """fn(refs, off) on this layer's expert refs: ONE branch-tree walk."""
                if isinstance(l, int):
                    fn(*layer_ex[l])
                    return
                keys = sorted(
                    {
                        li if layer_ex[li][1] is None else li - layer_ex[li][1]
                        for li in kind_layers[kind]
                    }
                )

                def leaf(k):
                    refs, _ = layer_ex[k]
                    fn(refs, l - k if k in group_lo else None)

                _select(l, keys, leaf)

            def issue_experts(lo, hi):
                def leaf(refs, off):
                    def one(t, carry):
                        e = rget("uniq", t)
                        for j in range(4):
                            slab_copy(refs, off, e, t % nslot, j).start()
                        return carry

                    jax.lax.fori_loop(lo, hi, one, 0)

                for_layer(leaf)

            pairs = c.ex_opt and nslot >= 4 and len(expert_units(c)) <= 8
            pdepth = nslot - 2 if pairs else depth

            def start_expert(i):
                if pairs:
                    pl.when((i % 2 == 0) & (i + pdepth < n_u))(
                        lambda: issue_experts(
                            i + pdepth, jnp.minimum(i + pdepth + 2, n_u)
                        )
                    )
                else:

                    @pl.when(i + depth < n_u)
                    def _():
                        e = rget("uniq", i + depth)
                        slot = (i + depth) % nslot
                        for_layer(
                            lambda refs, off: [
                                slab_copy(refs, off, e, slot, j).start()
                                for j in range(4)
                            ]
                        )

            def wait_expert(slot, j_parts=(0, 1, 2, 3)):
                tl = l if isinstance(l, int) else kind_layers[kind][0]
                trefs, toff = layer_ex[tl]
                for j in j_parts:
                    slab_copy(trefs, toff, 0, slot, j).wait()

            issue_experts(0, jnp.minimum(pdepth, n_u))
            return nslot, W13, W2, S13, S2, start_expert, wait_expert

        def moe_tiled(l, smr, kind, nxt):
            """Leaves the ffn branch all-gathered in pad_f (or in flight)."""
            Rc = c.rows_per_dev
            Ep = c.n_experts_padded
            vy = c.tiled_vmem_y
            vpads = c.tiled_vmem_pads
            if c.lazy_dense:
                ws.hold = ws.ranges["wunpool"][0]
            if c.fold_o:
                g_pf = smr[SMALL["pre_ffn_w"] : SMALL["pre_ffn_w"] + 1, :H]
                a_alpha = smr[SMALL["attn_alpha"] : SMALL["attn_alpha"] + 1, :H]
                a_beta = smr[SMALL["attn_beta"] : SMALL["attn_beta"] + 1, :H]
                for t in range(R // RT):
                    rs = pl.ds(t * RT, RT)
                    S["lhs"][rs] = (a_alpha * S["h_stream"][rs] * g_pf).astype(bf16)
                S["fold_v"][0:1, :] = a_beta * g_pf
                wait_block, finish = o_blocks()
                names = ("wpool", "wrouter")
                with _scope("pool_router_fold"):
                    acc_pr = ws.mv_fold(
                        l,
                        names,
                        nxt,
                        _fold_g(c, names),
                        Hl,
                        wait_block,
                        tiled_fold_lhs("pad_o_ag"),
                    )
                finish()
                wp_sc = smr[SMALL["wpool_s"] : SMALL["wpool_s"] + 1, :Hel]
                for t in range(R // RT):
                    r0, r1 = t * RT, (t + 1) * RT
                    rs = pl.ds(r0, RT)
                    a_val = gathered_tile(S["pad_o_ag"].at[:, rs])
                    s_a = jax.lax.rsqrt(
                        jnp.mean(jnp.square(a_val), axis=-1, keepdims=True) + c.post_eps
                    )
                    s = a_alpha * S["h_stream"][rs] + a_beta * (a_val * s_a).astype(
                        bf16
                    ).astype(f32)
                    S["h_stream"][rs] = s
                    r_inv = jax.lax.rsqrt(
                        jnp.mean(jnp.square(s), axis=-1, keepdims=True) + c.rms_eps
                    )
                    pooled_t = (
                        (
                            r_inv
                            * (
                                acc_pr["wpool"][r0:r1]
                                + s_a * acc_pr["wpool"][R + r0 : R + r1]
                            )
                            * wp_sc
                        )
                        .astype(bf16)
                        .astype(f32)
                    )
                    S["pool_send"][rs] = _pack_bf16x2(pooled_t)
                    S["acc"][rs, pl.ds(0, Ep)] = r_inv * (
                        acc_pr["wrouter"][r0:r1]
                        + s_a * acc_pr["wrouter"][R + r0 : R + r1]
                    )
                pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
            else:
                with _scope("pool_mv"):
                    mv_tiled(l, "wpool", nxt, smr)
                if (c.ex_opt or vpads) and R <= c.untiled_max_rows:
                    S["pool_send"][...] = _pack_bf16x2(
                        S["acc"][:, pl.ds(0, Hel)].astype(bf16).astype(f32)
                    )
                else:

                    def pool_tile(rows, t):
                        pooled = S["acc"][rows, pl.ds(0, Hel)].astype(bf16).astype(f32)
                        S["pool_send"][rows] = _pack_bf16x2(pooled)

                    for_tiles(pool_tile)
                if vpads:
                    pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
                else:
                    pool_wait = ag_start(
                        "pool", S["pool_send"], S["pad_pool"].at[tp_rank]
                    )
                with _scope("router_mv"):
                    mv_tiled(l, "wrouter", nxt, smr)
            bias = smr[SMALL["expert_bias"] : SMALL["expert_bias"] + 1, :Ep]

            def topk_rows(rows):
                sc = jax.nn.sigmoid(S["acc"][rows, pl.ds(0, Ep)] * c.router_mult)
                ids_list, picks = _topk_iter(sc + bias, K)
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
                )
                S["ids_v"][rows, pl.ds(0, K)] = jnp.concatenate(ids_list, axis=1)
                S["wsel_v"][rows, pl.ds(0, K)] = wsel

            with _scope("topk"):
                if c.topk_xpose and R >= 64:
                    if R <= LANE:
                        ids_f, wsel_f = _topk_xpose(
                            jax.nn.sigmoid(S["acc"][:, pl.ds(0, Ep)] * c.router_mult),
                            bias,
                            K,
                            c.router_eps,
                            c.route_scale,
                        )
                        S["ids_v"][...] = ids_f
                        S["wsel_v"][...] = wsel_f
                    else:
                        br_tk = (
                            LANE
                            if R % LANE == 0
                            else (64 if R % 64 == 0 else (32 if R % 32 == 0 else RT))
                        )

                        def topk_xpose_tile(t, carry_):
                            rs_tk = pl.ds(pl.multiple_of(t * br_tk, br_tk), br_tk)
                            ids_f, wsel_f = _topk_xpose(
                                jax.nn.sigmoid(
                                    S["acc"][rs_tk, pl.ds(0, Ep)] * c.router_mult
                                ),
                                bias,
                                K,
                                c.router_eps,
                                c.route_scale,
                            )
                            S["ids_v"][rs_tk] = ids_f
                            S["wsel_v"][rs_tk] = wsel_f
                            return carry_

                        jax.lax.fori_loop(0, R // br_tk, topk_xpose_tile, 0)
                elif c.ex_opt and R <= c.untiled_max_rows:
                    topk_rows(pl.ds(0, R))
                else:
                    for_tiles(lambda rows, t: topk_rows(rows))
            with _scope("unique_experts"):
                n_u = unique_experts(None, l)
            nslot, W13, W2, S13, S2, start_expert, wait_expert = expert_io_tiled(
                l, kind, n_u
            )
            # ex_kperm >= 3: normalized fp4 block scales, per-expert (gate, up, down) globals
            gscale_t = c.ex_kperm >= 3 and kind == "fp4"
            use_gs_smem_t = bool(gscale_t and c.ex_gs_smem and "gs_s" in S)
            gs_cp = None
            if gscale_t:
                gs_mat = jnp.concatenate(
                    [
                        smr[SMALL["expert_gs"] : SMALL["expert_gs"] + 1, 0:Ep],
                        smr[SMALL["expert_gs"] : SMALL["expert_gs"] + 1, Ep : 2 * Ep],
                        smr[
                            SMALL["expert_gs_down"] : SMALL["expert_gs_down"] + 1, 0:Ep
                        ],
                    ],
                    axis=0,
                )  # [3, Ep]
                if use_gs_smem_t:
                    S["acc"][pl.ds(0, 8), pl.ds(0, Ep)] = jnp.concatenate(
                        [gs_mat, jnp.zeros((5, Ep), f32)], axis=0
                    )
                    gs_cp = pltpu.make_async_copy(
                        S["acc"].at[pl.ds(0, 8), pl.ds(0, Ep)],
                        S["gs_s"],
                        S["route_sem"].at[0],
                    )
                    gs_cp.start()
            with _scope("pool_ag"):
                pool_wait()
            if gs_cp is not None:
                gs_cp.wait()
            pre_w = smr[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He]
            prequant_x = bool(c.ex_lhs_fp8 >= 1 and (c.ex_kperm >= 3 or kind != "fp4"))
            xsc_ref = S["ids_v"].bitcast(f32)

            with _scope("ex_setup"):
                if vpads:
                    for t in range(R // RT):
                        rs = pl.ds(t * RT, RT)
                        x = _rms(
                            gathered_tile(S["pad_pool"].at[:, rs]), c.rms_eps, pre_w
                        ).astype(bf16)
                        if prequant_x:
                            xs_f32, xsc = _scale_fp8_range(x)
                            S["xfull"][rs] = _pack_bf16x2(xs_f32)
                            xsc_ref[rs] = jnp.broadcast_to(xsc, (RT, LANE))
                        else:
                            S["xfull"][rs] = _pack_bf16x2(x.astype(f32))
                else:

                    def xfull_tile(rows, t, v):
                        x = _rms(gathered_tile(v[0]), c.rms_eps, pre_w).astype(bf16)
                        if prequant_x:
                            xs_f32, xsc = _scale_fp8_range(x)
                            S["xfull"][rows] = _pack_bf16x2(xs_f32)
                            xsc_ref[rows] = jnp.broadcast_to(xsc, (RT, LANE))
                        else:
                            S["xfull"][rows] = _pack_bf16x2(x.astype(f32))
                        return []

                    tile_pass([(pad_view(S["pad_pool"]), S["pltile"])], xfull_tile)
                if vy >= 1:
                    if c.combine_loop and R > 64:
                        z_rt = jnp.zeros((RT, K, yw), jnp.uint32)
                        for t in range(R // RT):
                            S["ysend"][pl.ds(t * RT, RT)] = z_rt
                    else:
                        S["ysend"][...] = jnp.zeros((R, K, yw), jnp.uint32)

            pad_y = S["pad_y"]
            if vy == 0:
                y_rsem = S["tc_rsems"].at[tc_rmap["y"]]
                for b in range(3):
                    S["ypend"][b] = 0
                ystage = S["ystage"]

                def ydrain(b):
                    """Waits for the row DMAs out of ystage[b] (local or remote: same bytes)."""
                    tmpl = pltpu.make_async_copy(
                        ystage.at[0, pl.ds(0, 1)],
                        pad_y.at[0, 0, pl.ds(0, 1)],
                        S["ystage_sems"].at[b],
                    )

                    def w(i, carry):
                        tmpl.wait()
                        return carry

                    jax.lax.fori_loop(0, S["ypend"][b], w, 0)
                    S["ypend"][b] = 0

                def yemit(yp, first, n_valid):
                    n = yp.shape[0]
                    b = S["ypend"][2] % 2
                    ydrain(b)
                    ystage.at[b][pl.ds(0, n), :] = yp

                    def issue(j, carry):
                        p = rget("perm", first + j)
                        r, k = p // K, p % K
                        owner, rl = r // Rc, r % Rc
                        src = ystage.at[b, pl.ds(j, 1)]
                        dst = pad_y.at[me, rl, pl.ds(k, 1)]

                        @pl.when(owner == me)
                        def _():
                            pltpu.make_async_copy(
                                src, dst, S["ystage_sems"].at[b]
                            ).start()

                        @pl.when(owner != me)
                        def _():
                            pltpu.make_async_remote_copy(
                                src_ref=src,
                                dst_ref=dst,
                                send_sem=S["ystage_sems"].at[b],
                                recv_sem=y_rsem,
                                device_id=(owner,),
                            ).start()

                        return carry

                    jax.lax.fori_loop(0, n_valid, issue, 0)
                    S["ypend"][b] = n_valid
                    S["ypend"][2] = S["ypend"][2] + 1

            def y_block_cp(owner):
                return rcopy(
                    S["ysend"].at[pl.ds(pl.multiple_of(owner * Rc, Rc), Rc)],
                    pad_y.at[me],
                    owner,
                    "y",
                )

            if use_pair_y:
                core_y = me % 2
                chip_y = me // 2
                partner_y = jnp.bitwise_xor(me, 1)
                other_y = 1 - core_y

                def ysend_blk(rk):
                    return S["ysend"].at[pl.ds(pl.multiple_of(rk * Rc, Rc), Rc)]

                yp_cps = [
                    pltpu.make_async_remote_copy(
                        src_ref=ysend_blk(2 * c_o + other_y),
                        dst_ref=pad_y.at[c_o],
                        send_sem=S["pair_ssems"].at[4],
                        recv_sem=S["pair_rsems"].at[4],
                        device_id=(partner_y,),
                    )
                    for c_o in range(4)
                ]

            def ystore(yp, pairs, first, n_valid):
                if vy >= 1:
                    for j, p in enumerate(pairs):
                        S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]
                else:
                    yemit(yp, first, n_valid)

            split_w = c.ex_split_wait and c.ex_opt
            bd_kc = c.blockdiag_kc
            cw13_fp4 = (
                64 if c.ex_opt and c.fp4_chunk_words == 128 else c.fp4_chunk_words
            )
            cw2_fp4 = min(Il // FP4_PACK, 64 if c.ex_opt else 16)
            cw2_fp8 = min(Il // 4, 128 if c.ex_opt else 32)

            def globals_t(e):
                if use_gs_smem_t:
                    return S["gs_s"][0, e], S["gs_s"][1, e], S["gs_s"][2, e]
                lane = jax.lax.broadcasted_iota(jnp.int32, (3, Ep), 1)
                g = jnp.sum(jnp.where(lane == e, gs_mat, 0.0), axis=1, keepdims=True)
                return g[0:1], g[1:2], g[2:3]

            def wait_w2_if_needed(slot, wait_w2):
                if split_w:
                    with _scope("ex_wait"):
                        if isinstance(wait_w2, bool):
                            if wait_w2:
                                wait_expert(slot, (2, 3))
                        else:
                            pl.when(wait_w2)(lambda: wait_expert(slot, (2, 3)))

            def gather_x(pairs):
                if c.ex_opt:
                    packed = jnp.concatenate(
                        [S["xfull"][pl.ds(p // K, 1), :] for p in pairs], axis=0
                    )
                    return _unpack_bf16x2(packed).astype(bf16)
                return jnp.concatenate(
                    [_unpack_bf16x2(S["xfull"][pl.ds(p // K, 1), :]) for p in pairs],
                    axis=0,
                ).astype(bf16)

            def pq_sc_of(pairs):
                return jnp.concatenate(
                    [xsc_ref[pl.ds(p // K, 1), :] for p in pairs],
                    axis=0,
                )[:, 0:1]

            def run_ffn_and_store(
                nrows, pair_fn, slot, e, blockdiag, first, n_valid, wait_w2=True
            ):
                can_fp8_x = bool(
                    prequant_x and not blockdiag and nrows >= 32 and nrows % 32 == 0
                )
                can_stage_ex = bool(
                    c.ex_w2_stage
                    and not blockdiag
                    and nrows >= 16
                    and nrows % 16 == 0
                    and nrows <= R
                    and He + Il <= H
                    and 2 * Il <= S["acc"].shape[1]
                )
                if can_stage_ex:
                    pq_chunks = []
                    for s0 in range(0, nrows, 16):
                        ps16 = [pair_fn(s0 + j) for j in range(16)]
                        S["lhs"][pl.ds(s0, 16), pl.ds(0, He)] = gather_x(ps16)
                        if prequant_x:
                            pq_chunks.append(pq_sc_of(ps16))
                    pq_sc = jnp.concatenate(pq_chunks, axis=0) if prequant_x else None
                    w13_prequant_sc = pq_sc if can_fp8_x else None
                    w13_post_sc = pq_sc if (prequant_x and not can_fp8_x) else None

                    def x_stage_chunk_fn(c_w, cw_w, out_fp8):
                        w_x = S["lhs"][
                            pl.ds(0, nrows),
                            pl.ds(c_w * FP4_PACK, cw_w * FP4_PACK),
                        ]
                        return w_x.astype(jnp.float8_e4m3fn) if out_fp8 else w_x

                    gu_stage = S["acc"].at[pl.ds(0, nrows), pl.ds(0, 2 * Il)]
                    if kind == "fp4":
                        _fp4_dot(
                            None,
                            W13.at[slot],
                            S["w13s"].at[slot],
                            He,
                            2 * Il,
                            cw13_fp4,
                            fast_bf16=c.ex_opt,
                            kperm=c.ex_kperm,
                            via_fp8=c.ex_via_fp8,
                            vec_dequant=c.vec_dequant_kperm,
                            fp8_lhs=c.ex_fp8_lhs,
                            n_tile=c.bd_n_tile,
                            lhs_chunk_fn=x_stage_chunk_fn,
                            prequant_sc=w13_prequant_sc,
                            out_stage_ref=gu_stage,
                            post_scale=w13_post_sc,
                            nrows=nrows,
                        )
                    else:
                        _fp8_dot(
                            None,
                            W13.at[slot],
                            S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(f32),
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                            fp8_lhs=c.ex_fp8_lhs,
                            lhs_chunk_fn=x_stage_chunk_fn,
                            prequant_sc=w13_prequant_sc,
                            out_stage_ref=gu_stage,
                            post_scale=w13_post_sc,
                            nrows=nrows,
                        )
                    if gscale_t:
                        g_gate, g_up, g_down = globals_t(e)
                    else:
                        g_gate = g_up = g_down = None
                    quant_h = bool(
                        c.ex_lhs_fp8 >= 2
                        and (c.ex_kperm >= 3 or kind != "fp4")
                        and Il == 512
                        and nrows >= 32
                        and nrows % 32 == 0
                    )
                    if quant_h:
                        gt0 = S["acc"][pl.ds(0, nrows), pl.ds(0, 256)]
                        up0 = S["acc"][pl.ds(0, nrows), pl.ds(Il, 256)]
                        if gscale_t:
                            gt0, up0 = gt0 * g_gate, up0 * g_up
                        h0 = gt0 * jax.nn.sigmoid(gt0) * up0
                        m0 = jnp.maximum(jnp.abs(h0[:, :128]), jnp.abs(h0[:, 128:]))
                        S["acc"][pl.ds(0, nrows), pl.ds(0, 256)] = h0
                        gt1 = S["acc"][pl.ds(0, nrows), pl.ds(256, 256)]
                        up1 = S["acc"][pl.ds(0, nrows), pl.ds(Il + 256, 256)]
                        if gscale_t:
                            gt1, up1 = gt1 * g_gate, up1 * g_up
                        h1 = gt1 * jax.nn.sigmoid(gt1) * up1
                        m1 = jnp.maximum(jnp.abs(h1[:, :128]), jnp.abs(h1[:, 128:]))
                        amax = jnp.max(jnp.maximum(m0, m1), axis=1, keepdims=True)
                        h_pq_sc = jnp.where(
                            amax == 0,
                            jnp.zeros_like(amax),
                            amax * jnp.float32(1.0 / 448.0),
                        )
                        inv_h = jnp.where(
                            amax == 0,
                            jnp.zeros_like(amax),
                            jnp.float32(448.0) / jnp.maximum(amax, jnp.float32(1e-30)),
                        )
                        S["lhs"][pl.ds(0, nrows), pl.ds(He + 256, 256)] = (
                            h1 * inv_h
                        ).astype(bf16)
                        h0_r = S["acc"][pl.ds(0, nrows), pl.ds(0, 256)]
                        S["lhs"][pl.ds(0, nrows), pl.ds(He, 256)] = (
                            h0_r * inv_h
                        ).astype(bf16)
                    else:
                        for c_h in range(0, Il, 256):
                            th = min(256, Il - c_h)
                            gt = S["acc"][pl.ds(0, nrows), pl.ds(c_h, th)]
                            up_ = S["acc"][pl.ds(0, nrows), pl.ds(Il + c_h, th)]
                            if gscale_t:
                                gt, up_ = gt * g_gate, up_ * g_up
                            S["lhs"][pl.ds(0, nrows), pl.ds(He + c_h, th)] = (
                                gt * jax.nn.sigmoid(gt) * up_
                            ).astype(bf16)
                        h_pq_sc = None
                    wait_w2_if_needed(slot, wait_w2)

                    def h_chunk_fn(c_w, cw_w, out_fp8):
                        w_h = S["lhs"][
                            pl.ds(0, nrows),
                            pl.ds(He + c_w * FP4_PACK, cw_w * FP4_PACK),
                        ]
                        return w_h.astype(jnp.float8_e4m3fn) if out_fp8 else w_h

                    y_stage = S["lhs"].at[pl.ds(0, nrows), pl.ds(0, He)]
                    if kind == "fp4":
                        _fp4_dot(
                            None,
                            W2.at[slot],
                            S["w2s"].at[slot],
                            Il,
                            He,
                            cw2_fp4,
                            fast_bf16=c.ex_opt,
                            kperm=c.ex_kperm,
                            via_fp8=c.ex_via_fp8,
                            vec_dequant=c.vec_dequant_kperm,
                            fp8_lhs=c.ex_fp8_lhs,
                            n_tile=c.bd_n_tile,
                            lhs_chunk_fn=h_chunk_fn,
                            prequant_sc=h_pq_sc,
                            out_stage_ref=y_stage,
                            post_scale=(g_down if gscale_t else None),
                            nrows=nrows,
                        )
                    else:
                        _fp8_dot(
                            None,
                            W2.at[slot],
                            S2[slot, 0:1, :].astype(f32),
                            Il,
                            He,
                            cw2_fp8,
                            fp8_lhs=c.ex_fp8_lhs,
                            lhs_chunk_fn=h_chunk_fn,
                            prequant_sc=h_pq_sc,
                            out_stage_ref=y_stage,
                            nrows=nrows,
                        )
                    for s0 in range(0, nrows, 16):
                        yp = _pack_y(S["lhs"][pl.ds(s0, 16), pl.ds(0, He)], c)
                        if vy >= 1:
                            for j in range(16):
                                p = pair_fn(s0 + j)
                                S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[
                                    j : j + 1
                                ]
                        else:
                            pl.when(s0 < n_valid)(
                                lambda s0=s0, yp=yp: yemit(
                                    yp, first + s0, jnp.clip(n_valid - s0, 0, 16)
                                )
                            )
                    return

                pairs = [pair_fn(j) for j in range(nrows)]
                pq_sc = pq_sc_of(pairs) if prequant_x else None
                w13_prequant_sc = pq_sc if can_fp8_x else None
                w13_post_sc = pq_sc if (prequant_x and not can_fp8_x) else None
                xc_raw = gather_x(pairs)
                xc = xc_raw.astype(jnp.float8_e4m3fn) if can_fp8_x else xc_raw
                if blockdiag:
                    gu = _fp4_blockdiag_dot(
                        xc,
                        W13.at[slot],
                        S["w13s"].at[slot],
                        He,
                        2 * Il,
                        bd_kc,
                        kperm=bool(c.ex_kperm),
                        fp8_lhs=c.bd_fp8_lhs,
                        n_tile=c.bd_n_tile,
                        prequant_sc=w13_prequant_sc,
                        nrows=nrows,
                        post_scale=w13_post_sc,
                    )
                elif kind == "fp4":
                    gu = _fp4_dot(
                        xc,
                        W13.at[slot],
                        S["w13s"].at[slot],
                        He,
                        2 * Il,
                        cw13_fp4,
                        fast_bf16=c.ex_opt,
                        kperm=c.ex_kperm,
                        via_fp8=c.ex_via_fp8,
                        vec_dequant=c.vec_dequant_kperm,
                        fp8_lhs=c.ex_fp8_lhs,
                        n_tile=c.bd_n_tile,
                        prequant_sc=w13_prequant_sc,
                        post_scale=w13_post_sc,
                        nrows=nrows,
                    )
                else:
                    gu = _fp8_dot(
                        xc,
                        W13.at[slot],
                        S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(f32),
                        He,
                        2 * Il,
                        c.fp4_chunk_words,
                        fp8_lhs=c.ex_fp8_lhs,
                        prequant_sc=w13_prequant_sc,
                        post_scale=w13_post_sc,
                        nrows=nrows,
                    )
                gate, up = gu[:, :Il], gu[:, Il:]
                if gscale_t:
                    g_gate, g_up, g_down = globals_t(e)
                    gate, up = gate * g_gate, up * g_up
                h = (gate * jax.nn.sigmoid(gate) * up).astype(bf16)
                wait_w2_if_needed(slot, wait_w2)
                h_pq = bool(
                    c.ex_lhs_fp8 >= 2
                    and (c.ex_kperm >= 3 or kind != "fp4")
                    and not blockdiag
                    and nrows >= 32
                    and nrows % 32 == 0
                )
                if blockdiag:
                    y = _fp4_blockdiag_dot(
                        h,
                        W2.at[slot],
                        S["w2s"].at[slot],
                        Il,
                        He,
                        bd_kc,
                        kperm=bool(c.ex_kperm),
                        fp8_lhs=(c.bd_fp8_lhs or h_pq),
                        n_tile=c.bd_n_tile,
                    )
                elif kind == "fp4":
                    y = _fp4_dot(
                        h,
                        W2.at[slot],
                        S["w2s"].at[slot],
                        Il,
                        He,
                        cw2_fp4,
                        fast_bf16=c.ex_opt,
                        kperm=c.ex_kperm,
                        via_fp8=c.ex_via_fp8,
                        vec_dequant=c.vec_dequant_kperm,
                        fp8_lhs=(c.ex_fp8_lhs or h_pq),
                        n_tile=c.bd_n_tile,
                    )
                else:
                    y = _fp8_dot(
                        h,
                        W2.at[slot],
                        S2[slot, 0:1, :].astype(f32),
                        Il,
                        He,
                        cw2_fp8,
                        fp8_lhs=(c.ex_fp8_lhs or h_pq),
                    )
                if gscale_t:
                    y = y * g_down
                ystore(_pack_y(y, c), pairs, first, n_valid)

            def pick_tree(arms, cnt):
                if len(arms) == 1:
                    arms[0][1]()
                    return
                mid = len(arms) // 2
                jax.lax.cond(
                    cnt <= arms[mid - 1][0],
                    lambda: pick_tree(arms[:mid], cnt),
                    lambda: pick_tree(arms[mid:], cnt),
                )

            def ebody(i, carry, part=None):
                slot = i % nslot
                start_expert(i)
                with _scope("ex_wait"):
                    wait_expert(slot, (0, 1) if split_w else (0, 1, 2, 3))
                e = rget("uniq", i)
                off, cnt = rget("ex_off", i), rget("ex_cnt", i)

                def compact(nrows):
                    # bucket rows past the expert's count repeat its last pair (same write)
                    with _scope(f"ex_c{nrows}"):
                        run_ffn_and_store(
                            nrows,
                            lambda j: rget("perm", off + jnp.minimum(j, cnt - 1)),
                            slot,
                            e,
                            True,
                            off,
                            cnt,
                        )

                def full():
                    can_bd = kind == "fp4" and (c.ex_kperm == 0 or bd_kc == 256)
                    use_bd_chunk = c.ex_bd_all and can_bd
                    CH = 8 if use_bd_chunk else min(c.expert_chunk, R)

                    def full_one(nrows):
                        run_ffn_and_store(
                            nrows,
                            lambda j: rget("perm", off + jnp.minimum(j, cnt - 1)),
                            slot,
                            e,
                            False,
                            off,
                            cnt,
                        )

                    def chunk(ch, carry_):
                        run_ffn_and_store(
                            CH,
                            lambda j: rget(
                                "perm", off + jnp.minimum(ch * CH + j, cnt - 1)
                            ),
                            slot,
                            e,
                            use_bd_chunk,
                            off + ch * CH,
                            jnp.minimum(CH, cnt - ch * CH),
                            wait_w2=(ch == 0),
                        )
                        return carry_

                    def run_chunks():
                        jax.lax.fori_loop(0, (cnt + CH - 1) // CH, chunk, 0)

                    def bd_chunk_step(ch, carry_):
                        run_ffn_and_store(
                            8,
                            lambda j: rget(
                                "perm", off + jnp.minimum(ch * 8 + j, cnt - 1)
                            ),
                            slot,
                            e,
                            True,
                            off + ch * 8,
                            jnp.minimum(8, cnt - ch * 8),
                            wait_w2=(ch == 0),
                        )
                        return carry_

                    def run_bd_chunks():
                        jax.lax.fori_loop(0, (cnt + 7) // 8, bd_chunk_step, 0)

                    with _scope("ex_full"):
                        if c.ex_opt and vy >= 1 and not use_bd_chunk:
                            lo_b = c.compact_rows if kind == "fp4" else 0
                            far = []
                            if c.ex_bd_max_rows > lo_b and can_bd:
                                bd_cap = min(c.ex_bd_max_rows, R)
                                far.append((bd_cap, run_bd_chunks))
                                lo_b = bd_cap
                            if lo_b < R:
                                raw_fb = (8, 16, 32) if c.ex_w2_stage else (16, 24)
                                fb = tuple(b for b in raw_fb if lo_b < b <= R)
                                for b in fb:
                                    far.append((b, functools.partial(full_one, b)))
                                if fb and fb[-1] >= R:
                                    far[-1] = (None, far[-1][1])
                                else:
                                    far.append((None, run_chunks))
                            else:
                                far[-1] = (None, far[-1][1])
                            pick_tree(far, cnt)
                        else:
                            run_chunks()

                if kind == "fp4" and c.compact_rows:
                    buckets = compact_buckets(c)
                    arms = [(b, functools.partial(compact, b)) for b in buckets]
                    if buckets[-1] < R:
                        arms.append((None, full))
                    else:
                        arms[-1] = (None, arms[-1][1])
                    if part == "few":
                        arms = [
                            a for a in arms if a[0] is not None and a[0] <= c.split_rows
                        ]
                        arms[-1] = (None, arms[-1][1])
                    elif part == "many":
                        arms = [a for a in arms if a[0] is None or a[0] > c.split_rows]
                    pick_tree(arms, cnt)
                else:
                    full()
                if stream_pair_y:

                    def ready_py(nb):
                        end = rget("blk_end", jnp.minimum(nb, 2))
                        return (nb < 3) & (end <= i + 1)

                    def send_py(nb):
                        c_o = jnp.bitwise_xor(chip_y, 3 - nb)
                        _select(c_o, [0, 1, 2, 3], lambda co: yp_cps[co].start())
                        return nb + 1

                    return jax.lax.while_loop(ready_py, send_py, carry)
                if vy >= 1 and y_block and not use_pair_y:

                    def ready(nb):
                        end = rget("blk_end", jnp.minimum(nb, c.tp - 2))
                        return (nb < c.tp - 1) & (end <= i + 1)

                    def send_block(nb):
                        owner = blk_owner(nb)
                        pl.when(owner * Rc < n_live)(y_block_cp(owner).start)
                        return nb + 1

                    return jax.lax.while_loop(ready, send_block, carry)
                return carry

            with _scope("experts"):
                if c.split_rows and kind == "fp4":
                    n_groups = (
                        4
                        if stream_pair_y
                        else (c.tp if (y_block and not use_pair_y) else 1)
                    )

                    def group(g_, carry):
                        lo = jnp.where(
                            g_ == 0, 0, rget("blk_end", jnp.maximum(g_ - 1, 0))
                        )
                        mid, hi = rget("blk_mid", g_), rget("blk_end", g_)
                        carry = jax.lax.fori_loop(
                            lo, mid, functools.partial(ebody, part="few"), carry
                        )
                        return jax.lax.fori_loop(
                            mid, hi, functools.partial(ebody, part="many"), carry
                        )

                    nb_sent = jax.lax.fori_loop(0, n_groups, group, jnp.int32(0))
                else:
                    nb_sent = jax.lax.fori_loop(0, n_u, ebody, jnp.int32(0))
            if vy == 0:
                ydrain(0)
                ydrain(1)
            ws.flush()
            my_live = me * Rc < n_live
            with _scope("y_rs"):
                if use_pair_y:
                    if stream_pair_y:

                        def send_rem_py(nb, carry_):
                            c_o = jnp.bitwise_xor(chip_y, 3 - nb)
                            _select(c_o, [0, 1, 2, 3], lambda co: yp_cps[co].start())
                            return carry_

                        jax.lax.fori_loop(nb_sent, 4, send_rem_py, 0)
                        for cp in yp_cps:
                            cp.wait()
                    else:
                        for cp in yp_cps:
                            cp.start()
                        for cp in yp_cps:
                            cp.wait()
                    if c.combine_loop:

                        def pair_reduce_row(r, carry_):
                            for c_o in range(4):
                                r_same = 2 * c_o + core_y
                                S["ysend"].at[r_same * Rc + r][...] = _pack_y(
                                    _unpack_y(S["ysend"].at[r_same * Rc + r][...], c)
                                    + _unpack_y(pad_y.at[c_o, r][...], c),
                                    c,
                                )
                            return carry_

                        jax.lax.fori_loop(0, Rc, pair_reduce_row, 0)
                    else:
                        for c_o in range(4):
                            r_same = 2 * c_o + core_y
                            ysend_blk(r_same)[...] = _pack_y(
                                _unpack_y(ysend_blk(r_same)[...], c)
                                + _unpack_y(pad_y[c_o], c),
                                c,
                            )
                    if c.tiled_pair_y >= 2:
                        m1_y = jnp.where(core_y == 0, 2, 1).astype(jnp.int32)
                        m2_y = 3 - m1_y
                        nb1_y = 2 * jnp.bitwise_xor(chip_y, m1_y) + core_y
                        nb2_y = 2 * jnp.bitwise_xor(chip_y, m2_y) + core_y
                        c2a = [
                            rcopy(ysend_blk(nb1_y), pad_y.at[4], nb1_y, "y"),
                            rcopy(
                                ysend_blk(2 * jnp.bitwise_xor(chip_y, 3) + core_y),
                                pad_y.at[5],
                                nb1_y,
                                "y",
                            ),
                        ]
                        for cp in c2a:
                            cp.start()
                        for cp in c2a:
                            cp.wait_send()
                        for cp in c2a:
                            cp.wait_recv()
                        if c.combine_loop:

                            def pair_c2_row(r, carry_):
                                pad_y.at[7, r][...] = _pack_y(
                                    _unpack_y(S["ysend"].at[nb2_y * Rc + r][...], c)
                                    + _unpack_y(pad_y.at[5, r][...], c),
                                    c,
                                )
                                return carry_

                            jax.lax.fori_loop(0, Rc, pair_c2_row, 0)
                        else:
                            pad_y[7] = _pack_y(
                                _unpack_y(ysend_blk(nb2_y)[...], c)
                                + _unpack_y(pad_y[5], c),
                                c,
                            )
                        pad_y[5] = ysend_blk(me)[...]
                        c2b = pltpu.make_async_remote_copy(
                            src_ref=pad_y.at[7],
                            dst_ref=pad_y.at[6],
                            send_sem=S["pair_ssems"].at[5],
                            recv_sem=S["pair_rsems"].at[5],
                            device_id=(nb2_y,),
                        )
                        c2b.start()
                        c2b.wait()
                    else:
                        pad_y[4 + chip_y] = ysend_blk(me)[...]
                        yc_cps = [
                            rcopy(
                                ysend_blk(2 * jnp.bitwise_xor(chip_y, d) + core_y),
                                pad_y.at[4 + chip_y],
                                2 * jnp.bitwise_xor(chip_y, d) + core_y,
                                "y",
                            )
                            for d in (1, 2, 3)
                        ]
                        for cp in yc_cps:
                            cp.start()
                        for cp in yc_cps:
                            cp.wait_send()
                        for cp in yc_cps:
                            cp.wait_recv()
                elif vy >= 1:
                    own_slice = pl.ds(pl.multiple_of(me * Rc, Rc), Rc)
                    if vy >= 2:

                        @pl.when(my_live)
                        def _():
                            pad_y[me] = S["ysend"][own_slice]

                    else:
                        ly_cp = pltpu.make_async_copy(
                            S["ysend"].at[own_slice], pad_y.at[me], tsem(_TS_LOCAL)
                        )
                        pl.when(my_live)(ly_cp.start)
                    if not y_block:
                        for d in range(1, c.tp):
                            peer = coll._peer(d)
                            pl.when(peer * Rc < n_live)(y_block_cp(peer).start)
                    for d in range(1, c.tp):
                        peer = coll._peer(d)
                        pl.when(peer * Rc < n_live)(y_block_cp(peer).wait_send)
                    if vy == 1:
                        pl.when(my_live)(ly_cp.wait)

                    @pl.when(my_live)
                    def _():
                        for d in range(1, c.tp):
                            y_block_cp(coll._peer(d)).wait_recv()

                else:
                    # every peer's rows for this device's live rows: K slots of each, landed
                    row_msg = pltpu.make_async_remote_copy(
                        src_ref=pad_y.at[0, pl.ds(0, 1)],
                        dst_ref=pad_y.at[me, pl.ds(0, 1)],
                        send_sem=S["tc_ssems"].at[tc_smap["y"]],
                        recv_sem=S["tc_rsems"].at[tc_rmap["y"]],
                        device_id=(me,),
                    )

                    def recv_row(_, carry):
                        row_msg.wait_recv()
                        return carry

                    jax.lax.fori_loop(
                        0, (c.tp - 1) * jnp.clip(n_live - me * Rc, 0, Rc), recv_row, 0
                    )

            with _scope("combine"):
                pw = smr[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
                # this device's routing weights: the 8-row-aligned window around its
                # rows
                win = owner_window_rows(c)
                if win == Rc:
                    w0, w_off = pl.multiple_of(me * Rc, 8), 0
                else:
                    w0 = pl.multiple_of(jnp.minimum((me * Rc) // 8 * 8, R - win), 8)
                    w_off = me * Rc - w0
                wmine = S["wsel_v"][pl.ds(w0, win), pl.ds(0, K)]  # [win, K]
                wk = jnp.concatenate(
                    [wmine[:, k : k + 1][:, :, None] for k in range(K)], axis=1
                )
                S["wk_s"][...] = jnp.broadcast_to(wk, (win, K, LANE))
                if vy >= 2:
                    y_slabs = (
                        (4, 5, 6)
                        if (use_pair_y and c.tiled_pair_y >= 2)
                        else ((4, 5, 6, 7) if use_pair_y else tuple(range(c.tp)))
                    )
                    if c.combine_loop:

                        def combine_vmem_row(r, carry_):
                            ysum = _unpack_y(pad_y.at[y_slabs[0], r][...], c)
                            for d in y_slabs[1:]:
                                ysum = ysum + _unpack_y(pad_y.at[d, r][...], c)
                            ysum = (
                                jnp.where(me * Rc + r < n_live, ysum, 0.0)
                                .astype(bf16)
                                .astype(f32)
                            )
                            ysc = ysum * pw
                            nrm = jax.lax.rsqrt(
                                jnp.mean(jnp.square(ysc), axis=-1, keepdims=True)
                                + c.post_eps
                            )
                            ynorm = (ysc * nrm).astype(bf16).astype(f32)
                            u = jnp.sum(
                                ynorm * S["wk_s"][w_off + r, :, 0:1],
                                axis=0,
                                keepdims=True,
                            )
                            S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
                            return carry_

                        jax.lax.fori_loop(0, Rc, combine_vmem_row, 0)
                    else:
                        ysum = _unpack_y(pad_y[y_slabs[0]], c)  # [Rc, K, He]
                        for d in y_slabs[1:]:
                            ysum = ysum + _unpack_y(pad_y[d], c)
                        r_live = (
                            me * Rc + jax.lax.broadcasted_iota(jnp.int32, (Rc, 1, 1), 0)
                        ) < n_live
                        ysum = jnp.where(r_live, ysum, 0.0).astype(bf16).astype(f32)
                        ysc = ysum * pw[None]
                        nrm = jax.lax.rsqrt(
                            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True)
                            + c.post_eps
                        )
                        ynorm = (ysc * nrm).astype(bf16).astype(f32)
                        wk_rc = (
                            S["wk_s"][:, :, 0:1]
                            if win == Rc
                            else jnp.concatenate(
                                [S["wk_s"][w_off + r, :, 0:1][None] for r in range(Rc)],
                                axis=0,
                            )
                        )
                        u = jnp.sum(ynorm * wk_rc, axis=1)  # [Rc, He]
                        S["u_send"][...] = _pack_bf16x2(u)
                else:

                    def ycopy(r, b):
                        return pltpu.make_async_copy(
                            S["pad_y"].at[:, pl.ds(r, 1)],
                            S["yrow"].at[b],
                            S["yrow_sems"].at[b],
                        )

                    pl.when(my_live)(ycopy(0, 0).start)

                    def combine_row(r, carry):
                        b = r % 2

                        @pl.when(my_live & (r + 1 < Rc))
                        def _():
                            ycopy(r + 1, 1 - b).start()

                        pl.when(my_live)(ycopy(r, b).wait)
                        yr = S["yrow"].at[b]
                        ysum = _unpack_y(yr[0, 0], c)  # [K, He]
                        for d in range(1, c.tp):
                            ysum = ysum + _unpack_y(yr[d, 0], c)
                        # rows past n_live received nothing (stale HBM): keep them zero
                        ysum = (
                            jnp.where(my_live & (me * Rc + r < n_live), ysum, 0.0)
                            .astype(bf16)
                            .astype(f32)
                        )
                        ysc = ysum * pw
                        nrm = jax.lax.rsqrt(
                            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True)
                            + c.post_eps
                        )
                        ynorm = (ysc * nrm).astype(bf16).astype(f32)
                        u = jnp.sum(
                            ynorm * S["wk_s"][w_off + r, :, 0:1], axis=0, keepdims=True
                        )
                        S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
                        return carry

                    jax.lax.fori_loop(0, Rc, combine_row, 0)

            if vpads:
                with _scope("u_ag"):
                    padu = coll.all_gather(
                        "u", S["u_send"], me, live=lambda m: m * Rc < n_live
                    )
                    if c.combine_loop and R % 16 == 0:

                        def _store_u16(r0, raw16):
                            r_ok16 = (
                                r0 + jax.lax.broadcasted_iota(jnp.int32, (16, 1), 0)
                            ) < n_live
                            S["lhs"][pl.ds(r0, 16), pl.ds(0, He)] = jnp.where(
                                r_ok16, _unpack_bf16x2(raw16), 0.0
                            ).astype(bf16)

                        if Rc % 16 == 0:

                            def u_ag_chunk16(ch, carry_):
                                r0 = pl.multiple_of(ch * 16, 16)
                                d = (ch * 16) // Rc
                                sub = pl.multiple_of((ch * 16) % Rc, 16)
                                _store_u16(r0, padu.at[d][pl.ds(sub, 16)])
                                return carry_

                            jax.lax.fori_loop(0, R // 16, u_ag_chunk16, 0)
                        elif Rc == 8:

                            def u_ag_pair8(ch, carry_):
                                r0 = pl.multiple_of(ch * 16, 16)
                                raw16 = jnp.concatenate(
                                    [padu.at[2 * ch][...], padu.at[2 * ch + 1][...]],
                                    axis=0,
                                )
                                _store_u16(r0, raw16)
                                return carry_

                            jax.lax.fori_loop(0, R // 16, u_ag_pair8, 0)
                        else:
                            S["xfull"][...] = jnp.concatenate(
                                [padu[d] for d in range(c.tp)], axis=0
                            )

                            def u_ag_xfull16(ch, carry_):
                                r0 = pl.multiple_of(ch * 16, 16)
                                _store_u16(r0, S["xfull"][pl.ds(r0, 16)])
                                return carry_

                            jax.lax.fori_loop(0, R // 16, u_ag_xfull16, 0)
                    else:
                        upk = jnp.concatenate([padu[d] for d in range(c.tp)], axis=0)
                        r_ok = jax.lax.broadcasted_iota(jnp.int32, (R, 1), 0) < n_live
                        S["lhs"][:, pl.ds(0, He)] = jnp.where(
                            r_ok, _unpack_bf16x2(upk), 0.0
                        ).astype(bf16)
            else:
                # u all-gather, pipelined by owner: the unpool pass loads a row tile once
                # its owners' rows have landed (each owner signals its own semaphore)
                u_osems = S["uag_osems"]
                # an owner block off an 8-row boundary cannot be one HBM DMA window: send
                # it row by row (the waits below take one DMA each)
                if Rc % 8 == 0:
                    u_parts = [
                        (
                            S["u_send"],
                            S["pad_u"].at[pl.ds(pl.multiple_of(me * Rc, 8), Rc)],
                        )
                    ]
                else:
                    u_parts = [
                        (
                            S["u_send"].at[pl.ds(r, 1)],
                            S["pad_u"].at[pl.ds(me * Rc + r, 1)],
                        )
                        for r in range(Rc)
                    ]

                @pl.when(my_live)
                def _():
                    for src, dst in u_parts:
                        pltpu.make_async_copy(src, dst, u_osems.at[me]).start()
                        for d in range(1, c.tp):
                            pltpu.make_async_remote_copy(
                                src_ref=src,
                                dst_ref=dst,
                                send_sem=S["tc_ssems"].at[_TC["u"]],
                                recv_sem=u_osems.at[me],
                                device_id=(coll._peer(d),),
                            ).start()

                def u_ready(t):

                    def wait_owner(o):
                        def wait():
                            src, dst = u_parts[0]
                            for _ in u_parts:
                                pltpu.make_async_copy(src, dst, u_osems.at[o]).wait()

                        pl.when(o * Rc < n_live)(wait)

                    if RT % Rc == 0:
                        for j in range(RT // Rc):
                            wait_owner(t * (RT // Rc) + j)
                    elif Rc % RT == 0:
                        pl.when((t * RT) % Rc == 0)(lambda: wait_owner(t * RT // Rc))
                    else:
                        # owners whose rows tile t is the first to touch
                        first = jnp.where(t == 0, 0, (t * RT - 1) // Rc + 1)
                        last = ((t + 1) * RT - 1) // Rc
                        for j in range(RT // Rc + 2):
                            pl.when(first + j <= last)(
                                lambda j=j: wait_owner(first + j)
                            )

                def unpool_tile(rows, t, v):
                    u = jnp.where(tile_live(t), _unpack_bf16x2(v[0][...]), 0.0)
                    S["lhs"][rows, pl.ds(0, He)] = u.astype(bf16)
                    return []

                with _scope("u_ag"):
                    tile_pass(
                        [(lambda rows: S["pad_u"].at[rows], S["utile"])],
                        unpool_tile,
                        ready=[u_ready],
                    )

                @pl.when(my_live)
                def _():
                    for d in range(1, c.tp):
                        for src, dst in u_parts:
                            rcopy(src, dst, coll._peer(d), "u").wait_send()

            with _scope("unpool_mv"):
                mv_tiled(l, "wunpool", nxt, smr)
            if nxt is not None and not c.smalls_early:
                prefetch_smalls(l, nxt)

            if vpads:
                if c.combine_loop and R > 64:
                    for t in range(R // RT):
                        rs = pl.ds(t * RT, RT)
                        S["f_send"][rs] = _pack_bf16x2(
                            S["acc"][rs, pl.ds(0, Hl)].astype(bf16).astype(f32)
                        )
                else:
                    S["f_send"][...] = _pack_bf16x2(
                        S["acc"][:, pl.ds(0, Hl)].astype(bf16).astype(f32)
                    )
                if c.fold_f and nxt is not None:
                    coll.all_gather_blocks_start(
                        "f", S["f_send"], tp_rank, S["f_rsems"]
                    )
                else:
                    with _scope("f_ag"):
                        coll.all_gather("f", S["f_send"], tp_rank)
            else:

                def f_tile(rows, t):
                    f = S["acc"][rows, pl.ds(0, Hl)].astype(bf16).astype(f32)
                    S["f_send"][rows] = _pack_bf16x2(f)
                    # pipelined all-gather: the ffn residual pass waits per row tile
                    ag_tile_start(
                        "f", S["f_send"], S["pad_f"], tp_rank, t, S["fag_tsems"]
                    )

                for_tiles(f_tile)

        def capture_tile(l, rows, s):
            for ci, cl in enumerate(c.capture_layers):

                def write(ci=ci):
                    S["btile"][...] = s.astype(bf16)
                    dma(S["btile"], aux_out.at[rows, pl.ds(ci * H, H)], _TS_AUX)

                if isinstance(l, int):
                    if l == cl:
                        write()
                    continue
                pl.when(l == cl)(write)

        def layer_tiled(l, kind, next_layer, prefetched, pend=False):
            nxt = l + 1 if next_layer else None
            slot = l % 2
            if not prefetched:
                smalls_copy(l, slot).start()
                ws.prologue(l)
            with _scope("smalls_wait"):
                smalls_copy(l, slot).wait()
            smr = S["smalls"].at[slot]
            use_rope, is_global = flags(l)

            def row(name, n=H):
                return smr[SMALL[name] : SMALL[name] + 1, :n]

            if not pend:
                if c.tiled_vmem_pads:
                    for t in range(R // RT):
                        rs = pl.ds(t * RT, RT)
                        S["lhs"][rs] = _rms(
                            S["h_stream"][rs], c.rms_eps, row("pre_attn_w")
                        ).astype(bf16)
                else:

                    def pre_attn_tile(rows, t, v):
                        S["lhs"][rows] = _rms(
                            v[0][...], c.rms_eps, row("pre_attn_w")
                        ).astype(bf16)
                        return []

                    tile_pass([(stream_view, S["sin"])], pre_attn_tile)
            with _scope("attention"):
                attention_tiled(l, use_rope, is_global, nxt, smr, pend=pend)
            if c.smalls_early:
                prefetch_smalls(l, nxt)

            if not c.fold_o:
                if c.tiled_vmem_pads:
                    for t in range(R // RT):
                        rs = pl.ds(t * RT, RT)
                        s = row("attn_alpha") * S["h_stream"][rs] + row(
                            "attn_beta"
                        ) * _rms(
                            gathered_tile(S["pad_o_ag"].at[:, rs]), c.post_eps
                        ).astype(bf16).astype(f32)
                        S["h_stream"][rs] = s
                        S["lhs"][rs] = _rms(s, c.rms_eps, row("pre_ffn_w")).astype(bf16)
                else:
                    oag_rdy = ag_tile_ready(S["o_send"], S["pad_o_ag"], S["oag_tsems"])

                    def post_attn_tile(rows, t, v):
                        s = row("attn_alpha") * v[1][...] + row("attn_beta") * _rms(
                            gathered_tile(v[0]), c.post_eps
                        ).astype(bf16).astype(f32)
                        S["lhs"][rows] = _rms(s, c.rms_eps, row("pre_ffn_w")).astype(
                            bf16
                        )
                        return [s]

                    tile_pass(
                        [
                            (pad_view(S["pad_o_ag"]), S["ptile"]),
                            (stream_view, S["sin"]),
                        ],
                        post_attn_tile,
                        [(stream_view, S["sout"])],
                        ready=[oag_rdy, None],
                    )
                    ag_tile_sent("o_ag", S["o_send"], S["pad_o_ag"])
            with _scope("moe"):
                moe_tiled(l, smr, kind, nxt)

            if not (c.fold_f and nxt is not None):
                if c.tiled_vmem_pads:
                    for t in range(R // RT):
                        rs = pl.ds(t * RT, RT)
                        s = row("ffn_alpha") * S["h_stream"][rs] + row(
                            "ffn_beta"
                        ) * _rms(
                            gathered_tile(S["pad_f"].at[:, rs]) * row("post_ffn_w"),
                            c.post_eps,
                        ).astype(bf16).astype(f32)
                        capture_tile(l, rs, s)
                        S["h_stream"][rs] = s
                else:
                    fag_rdy = ag_tile_ready(S["f_send"], S["pad_f"], S["fag_tsems"])

                    def ffn_tile(rows, t, v):
                        s = row("ffn_alpha") * v[1][...] + row("ffn_beta") * _rms(
                            gathered_tile(v[0]) * row("post_ffn_w"), c.post_eps
                        ).astype(bf16).astype(f32)
                        capture_tile(l, rows, s)
                        return [s]

                    tile_pass(
                        [(pad_view(S["pad_f"]), S["ptile"]), (stream_view, S["sin"])],
                        ffn_tile,
                        [(stream_view, S["sout"])],
                        ready=[fag_rdy, None],
                    )
                    ag_tile_sent("f", S["f_send"], S["pad_f"])

        def tiled_main():
            # rotary rows for every row's position, 8 rows per round of DMAs
            def rope_round(b, carry):
                r0 = pl.multiple_of(b * 8, 8)
                cps, ps_ = [], []
                for i in range(8):
                    p = jnp.clip(pos_ref[r0 + i], 0, c.max_position - 1)
                    cp = pltpu.make_async_copy(
                        W["rope"].at[pl.ds(pl.multiple_of((p // 8) * 8, 8), 8)],
                        S["rope_blk"].at[i],
                        tsem(_TS_ROPE + i),
                    )
                    cp.start()
                    cps.append(cp)
                    ps_.append(p)
                rr = []
                for i in range(8):
                    cps[i].wait()
                    sel = jax.lax.broadcasted_iota(jnp.int32, (8, 2 * hd), 0) == (
                        ps_[i] - (ps_[i] // 8) * 8
                    )
                    rr.append(
                        jnp.sum(
                            jnp.where(sel, S["rope_blk"][i], 0.0), axis=0, keepdims=True
                        )
                    )
                S["rtab"][pl.ds(r0, 8), :] = jnp.concatenate(rr, axis=0)
                return carry

            jax.lax.fori_loop(0, R // 8, rope_round, 0)

            # embedding: weightless RMS norm of the bf16 rows, kept in bf16 precision
            if c.tiled_vmem_pads:
                dma(emb_ref, S["lhs"], _TS_AUX)
                for t in range(R // RT):
                    rs = pl.ds(t * RT, RT)
                    S["h_stream"][rs] = (
                        _rms(S["lhs"][rs].astype(f32), c.rms_eps)
                        .astype(bf16)
                        .astype(f32)
                    )
            else:

                def emb_tile(rows, t, v):
                    return [
                        _rms(v[0][...].astype(f32), c.rms_eps).astype(bf16).astype(f32)
                    ]

                tile_pass(
                    [(lambda rows: emb_ref.at[rows], S["bin"])],
                    emb_tile,
                    [(stream_view, S["sout"])],
                )
            pref = False
            pend = False
            for l in range(L):
                if roll_lo is not None and l == roll_lo and l < roll_hi:
                    lo, hi = l, roll_hi
                    if not pref:
                        layer_tiled(lo, "fp4", lo + 1 < L, False, pend=pend)
                        pref = lo + 1 < L
                        pend = c.fold_f and (lo + 1 < L)
                        lo += 1
                    last = hi if hi < L else hi - 1

                    def rolled(i, v):
                        assert pend == c.fold_f
                        layer_tiled(i, "fp4", True, True, pend=c.fold_f)
                        return v

                    if last > lo:
                        jax.lax.fori_loop(lo, last, rolled, 0)
                        pref = True
                        pend = c.fold_f
                    if hi == L and hi - 1 >= lo:
                        layer_tiled(hi - 1, "fp4", False, pref, pend=pend)
                        pref = False
                        pend = False
                elif roll_lo is None or not (roll_lo <= l < roll_hi):
                    layer_tiled(l, c.expert_kinds[l], l + 1 < L, pref, pend=pend)
                    pref = l + 1 < L
                    pend = c.fold_f and (l + 1 < L)
            dma(W["final_norm"], S["final_norm"], _TS_OUT)

            # final norm on the bf16-cast stream, as the native path
            if c.tiled_vmem_pads:
                if ncap == 0:
                    S["btile"][...] = jnp.zeros((RT, H), bf16)
                for t in range(R // RT):
                    rs = pl.ds(t * RT, RT)
                    if ncap == 0:
                        dma(S["btile"].at[:, pl.ds(0, LANE)], aux_out.at[rs], _TS_AUX)
                    hidden = _rms(
                        S["h_stream"][rs].astype(bf16).astype(f32),
                        c.rms_eps,
                        S["final_norm"][0:1, :],
                    )
                    S["lhs"][rs] = jnp.where(tile_live(t), hidden, 0.0).astype(bf16)
                dma(S["lhs"], hid_out, _TS_OUT)
            else:

                def final_tile(rows, t, v):
                    if ncap == 0:
                        S["btile"][...] = jnp.zeros((RT, H), bf16)
                        dma(S["btile"].at[:, pl.ds(0, LANE)], aux_out.at[rows], _TS_AUX)
                    hidden = _rms(
                        v[0][...].astype(bf16).astype(f32),
                        c.rms_eps,
                        S["final_norm"][0:1, :],
                    )
                    return [jnp.where(tile_live(t), hidden, 0.0).astype(bf16)]

                tile_pass(
                    [(stream_view, S["sin"])],
                    final_tile,
                    [(lambda rows: hid_out.at[rows], S["bin"])],
                )

        if c.row_tile:
            tiled_main()
            return

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

        if c.unify_layers:
            smalls_copy(0, 0).start()
            ws.prologue(0)
            stream = jax.lax.fori_loop(
                0, L, lambda i, v: layer(i, v, None, True, True), stream
            )
        for l in range(L) if not c.unify_layers else ():
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

    return kernel, scratch, hbm_scratch


def _is_hbm(ref) -> bool:
    return getattr(ref, "memory_space", None) == pltpu.HBM


def megakernel_call(cfg: MKConfig, axis: str = "model"):
    """``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv) -> (hidden, aux, newkv)``

    for one device (call inside ``shard_map``).
    """
    kernel, scratch, hbm_scratch = build_kernel(cfg, axis)
    L = cfg.n_layers
    nw = len(WEIGHT_NAMES)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    ncap = len(cfg.capture_layers)
    aux_w = ncap * cfg.hidden if ncap else LANE

    def call(meta, pos, pages_full, pages_swa, emb, *rest):
        weights = rest[:nw]
        ne = n_expert_refs(cfg)
        experts = rest[nw : nw + ne]
        kv = rest[nw + ne :]
        assert len(kv) == L
        out_shape = [
            jax.ShapeDtypeStruct((cfg.rows, cfg.hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((cfg.rows, aux_w), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                newkv_shape(cfg), jnp.uint32 if cfg.kv_sal else jnp.bfloat16
            ),
        ] + [jax.ShapeDtypeStruct(r.shape, r.dtype) for _, r in hbm_scratch]
        # The pools are read-only here. With any remote DMA in the kernel XLA treats the call
        # as a collective and copies every aliased operand: the in-place write is kv_write_call.
        f = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=[smem] * 4
            + [hbm if cfg.row_tile else vmem]
            + [hbm]
            * (nw + n_expert_refs(cfg) + L),  # emb: bf16 rows, or the f32 stream
            out_specs=([hbm, hbm, hbm] if cfg.row_tile else [vmem, vmem, hbm])
            + [hbm] * len(hbm_scratch),
            scratch_shapes=[s for _, s in scratch],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=cfg.vmem_limit_bytes,
                collective_id=_COLLECTIVE_ID if cfg.tp > 1 else None,
            ),
            name=f"muse_megakernel_r{cfg.rows}_q{cfg.rows_per_req}",
        )
        out = f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)
        return tuple(out[:3])

    return call


def newkv_shape(cfg: MKConfig):
    if cfg.kv_sal:  # new token columns of a page block, [slot * words + word, row]
        return (cfg.n_layers, cfg.sal_words, cfg.rows)
    return (
        cfg.n_layers,
        cfg.rows,
        cfg.pool_kv_heads,
        2,
        cfg.pool_padded_head_dim,
    )


def _kv_write_sal_call(cfg: MKConfig):
    """SAL pools: a token is one lane column of its page block, so each live request

    read-modify-writes the (at most two) pages its consecutive rows land in. Every
    layer's page
    is in flight at once; requests run one after another.
    """
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)
    nw = cfg.sal_words

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        blk, new_v, rsem, wsem, nsem = rest[2 * L : 2 * L + 5]
        n_live = meta_ref[0]
        cp = pltpu.make_async_copy(newkv, new_v, nsem.at[0])
        cp.start()
        cp.wait()
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, ps), 1)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, R), 1)
        n_req_live = jnp.minimum((n_live + q - 1) // q, cfg.n_req)

        def page_of(l, j, pg_no):
            tab = pf_ref if l in gl else psw_ref
            return tab[
                jnp.minimum(j * cfg.pages_per_seq + pg_no, cfg.n_pages_entries - 1)
            ]

        def rmw(j, pg_no):
            for l in range(L):
                pltpu.make_async_copy(
                    kv_out[l].at[page_of(l, j, pg_no)], blk.at[l], rsem.at[l]
                ).start()

            def merge(l, carry):
                pltpu.make_async_copy(kv_out[0].at[0], blk.at[l], rsem.at[l]).wait()
                words = blk.at[l].bitcast(jnp.uint32).reshape(nw, ps)
                w = words[...]
                nv = jax.lax.bitcast_convert_type(new_v[l], jnp.int32)  # [nw, R]
                for t in range(q):
                    r = j * q + t
                    p = pos_ref[r]
                    newcol = jax.lax.bitcast_convert_type(
                        jnp.sum(jnp.where(col == r, nv, 0), axis=1, keepdims=True),
                        jnp.uint32,
                    )
                    hit = (lane == p % ps) & (p // ps == pg_no) & (r < n_live)
                    w = jnp.where(hit, newcol, w)
                words[...] = w
                return carry

            jax.lax.fori_loop(0, L, merge, 0)
            for l in range(L):
                pltpu.make_async_copy(
                    blk.at[l], kv_out[l].at[page_of(l, j, pg_no)], wsem.at[l]
                ).start()

            def drain(l, carry):
                pltpu.make_async_copy(blk.at[l], kv_out[0].at[0], wsem.at[l]).wait()
                return carry

            jax.lax.fori_loop(0, L, drain, 0)

        def req(j, carry):
            first = pos_ref[j * q] // ps
            last = pos_ref[jnp.minimum(j * q + q - 1, n_live - 1)] // ps
            rmw(j, first)
            pl.when(last != first)(lambda: rmw(j, last))
            return carry

        jax.lax.fori_loop(0, n_req_live, req, 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    blk_bytes = L * math.prod(cfg.sal_block) * jnp.dtype(cfg.kv_dtype).itemsize

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[
                pltpu.VMEM((L, *cfg.sal_block), cfg.kv_dtype),
                pltpu.VMEM((L, nw, R), jnp.uint32),
                pltpu.SemaphoreType.DMA((L,)),
                pltpu.SemaphoreType.DMA((L,)),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=blk_bytes + 16 * 2**20
            ),
            name=f"muse_kv_write_sal_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call


def kv_write_call(cfg: MKConfig):
    """``f(meta, pos, pages_full, pages_swa, newkv, *kv) -> kv``: scatter this step's K/V rows

    into every layer's pool in place (no remote DMA, so XLA aliases the pools
    without a copy).
    """
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
                tok = tab[ent] * ps + p % ps
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
