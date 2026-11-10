# SPDX-License-Identifier: Apache-2.0
"""MKConfig: the static description of one compiled megakernel step."""

import math

import jax.numpy as jnp
import msgspec

from sglang_tpu.muse_megakernel.mk.consts import (
    FP4_PACK,
    LANE,
    PREONES_MAX_SLOTS,
    PREONES_ROWS,
    QBLOCK_SIZES,
    UNSHIFTED_MAX,
)
from sglang_tpu.muse_megakernel.mk.plan import (
    dense_tiles,
    fold_fits,
    fold_groups,
    owner_window_rows,
)
from sglang_tpu.muse_megakernel.mk.tables import (
    swa_window_entries,
    table_entries_padded,
)


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
    # row-tiled VMEM-pads passes run as one loop over row tiles, not unrolled per tile
    tile_loops: bool = False
    # VMEM-Y at 240-256 rows: half-row yrow buffers; several buffers borrow others
    vmem_trim: bool = False
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
    # pages_swa stays in HBM and each step DMAs every request's window of it into SMEM: at
    # 1M context and cc >= 28 the two page tables would crowd the 1 MiB SMEM out
    swa_window_table: bool = False
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
    # SGLANG_TPU_MUSE_MK_QBLOCK: fp4 experts requantized to one bf16 scale per qblock K rows
    # (scale rows [0, K / qblock) of the slabs); the dots push raw codes to the MXU
    qblock: int = 0
    # dense weights left out of the ring (an FFN-section hook streams its own copies)
    dense_skip: tuple = ()

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

    def _validate_qblock(self):
        if self.qblock not in QBLOCK_SIZES:
            raise ValueError(
                f"SGLANG_TPU_MUSE_MK_QBLOCK={self.qblock}: 0 or one of {QBLOCK_SIZES}"
            )
        for name, k in (
            ("w13 K (expert_hidden)", self.expert_hidden),
            ("w2 K per rank (inter_local)", self.inter_local),
        ):
            if k % self.qblock:
                raise ValueError(
                    f"SGLANG_TPU_MUSE_MK_QBLOCK={self.qblock}: {name}={k} is not a multiple"
                )
        if self.scale_fp8 or self.ex_kperm >= 3 or self.bd_fp8_lhs:
            raise ValueError(
                "SGLANG_TPU_MUSE_MK_QBLOCK needs bf16 block scales and a bf16 expert LHS "
                "(scale_fp8, ex_kperm >= 3 and bd_fp8_lhs are off)"
            )

    def validate(self):
        assert self.rows % self.tp == 0 and self.rows >= self.tp
        if self.qblock:
            self._validate_qblock()
        # windows are DMA'd in TABLE_ALIGN-entry units; only the SAL kv write reads them
        assert not self.swa_window_table or (
            self.kv_sal and table_entries_padded(self) >= swa_window_entries(self)
        )
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
            # pair o_rs with streamed Y: one row off by ~25% at layer 3 of the 192- and
            # 256-row numerics cases (deterministic; VMEM Y and plain o_rs pass)
            and not (self.pair_o_rs and self.tiled_vmem_pads and self.tiled_vmem_y == 0)
            and self.expert_chunk % 8 == 0
            and self.expert_chunk >= self.compact_rows
            and self.vec_route
        )
        return self
