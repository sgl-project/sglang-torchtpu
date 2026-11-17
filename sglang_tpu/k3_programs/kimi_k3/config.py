# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel settings for the Kimi-K3 megakernels.

Kernel modules read their settings from :func:`active` when they are
imported (most settings are module constants), so a non-default config must
be installed with :func:`install` before the first kernel import. Serving and
the standalone tools install overrides before building the program.
"""

import dataclasses
import hashlib
import json
from typing import Any

from sglang_tpu.k3_programs.config import set_active_kernel_config


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    # Decode context parallel attention.
    dcp_group: int = 32
    # Attention data parallelism of the decoder (TP32 / dp heads per DP group
    # of 32 / dp ranks, each with its own rows); needs dcp_group = 32 / dp.
    attention_dp: int = 1
    # Block tables: "local" (vLLM's rank-local DCP tables) or "global" (global
    # page ids).
    dcp_tables: str = "local"
    # MLA decode key tile (tokens per page fold). 2048 is opt-in and applies
    # only to FP8 plain decode at attention_dp=2, dcp_group=16 with 32 rows per
    # DP group (B64); everything else uses 1024.
    mla_decode_ktile: int = 1024
    # Routed experts and weight layout (the pre-shard format).
    expert_gate_up: str = "mxfp4"  # "mxfp4" or "fp8" (reblocked)
    mxfp4_block: int = 2048
    mxfp4_pair_tile: int = 512
    mxfp4_static_tile: int = 0
    # None: 1024 at B96, 512 at power-of-two B1-B128, else unsplit (0).
    mxfp4_dot_tile: int | None = None
    pooled_weights: bool = True
    shared_shards: bool | None = None  # None: at B128
    compact_only: bool = False
    singleton_block: bool | None = None  # None: at B16-B128
    stage_blocks: bool | None = None  # None: at B16-B128
    # 8-row steps: stage a singleton route's active row with one select over
    # the 8-row latent tile instead of a branch per row (the same bits).
    b8_singleton_select: bool = False
    # 8-row steps, grouped MXFP4 routes: an expert pair that is not two
    # singletons and whose experts have at most 4 rows each runs on compact
    # tiles of each expert's rows (both staged by one one-hot [16, 8] x [8, K]
    # product; the source rows without local routes are zeroed first): the
    # paired gate/up's register operands narrowed to 4 rows a tile (as B16's
    # capacity-4 pairs), then each expert's finish on its tile (block-major
    # down, a one-hot selector placing the BF16 rows) instead of the 8-row
    # pair. Busier pairs and the tail keep the 8-row path. Each row's
    # products, scale reductions and sums are unchanged (bit-identical when
    # the routed rows' latents are finite).
    b8_compact_experts: bool = False
    b16_compact_experts: bool = True
    # B16 compact experts: stage each tile's rows with a one-hot [8, 16] x
    # [16, K] product (as B64, B96 and B128 do) instead of a scalar scan of
    # the row bitmask. The source rows without local routes are zeroed first.
    b16_one_hot_staging: bool = False
    # B16-B128 grouped routed experts (compact tiles): one copy of the
    # eight-row passes serves an iteration's pair or tail of experts with more
    # rows than a compact tile (a loop over its one or two experts) instead
    # of three unrolled copies. The same products in the same order; the
    # B16 expert-loop overlay then fits beside the KDA layer's path.
    shared_expert_passes: bool = False
    shared_expert_finishers: bool = True
    compact_overflow: bool = True
    # Grouped routed experts at B8/B16: after routing, one SMEM table holds
    # the ascending expert schedule (ids, row counts, single rows, row masks);
    # the expert loop reads it with scalar loads instead of reducing the
    # pending routes every iteration.
    expert_schedule: bool = False
    b4_attention_gather: bool = True
    # Token-scattered decode attention output projection (attention DP1, 8+
    # rows): the cross-host round carries each host's four-pair FP32 sum
    # rounded to BF16 (half its bytes). Every rank rounds its own host's sum
    # too and adds the four hosts in the same order, so the host copies of a
    # row stay bitwise equal. Changes numerics: one more BF16 rounding.
    attention_host_bf16: bool = False
    # Steps with attention_rank_rows (below): the output projection starts its
    # cross-host round in the layer's KDA or MLA branch, and the decoder waits
    # for it, adds the four hosts' sums and stores the rank's rows after the
    # branches join. The instruction overlay of the post-join code (loaded on
    # demand in the first KDA layer after each MLA layer) then loads while the
    # round is in flight. The same sums in the same order (bit-identical).
    attention_host_round_after_join: bool = False
    # Token-scattered decode attention at 8 and 16 rows (attention DP1, four
    # row owners): each rank of a pair reduces its own half of the pair's rows
    # at full width (the local round gives each rank both column halves' pair
    # partials of its own rows), so the output projection ends without the
    # pair's half-swap. The post-attention mixture runs on those rows, the MoE
    # input gathers from all eight local ranks, and each rank's updated prefix
    # rows reach its pair peer before the MoE output is added. The same sums
    # in the same order (bit-identical).
    attention_rank_rows: bool = False
    # Steps with attention_rank_rows (8 and 16 rows, four row owners): both
    # ranks of a pair hold the same rows before attention, so the gathers of
    # those rows (each layer's attention input, the aux hidden capture and the
    # final norm's rows) take each other pair's rows from the core of the
    # other parity of its chip (rank ^ 2k ^ 1) instead of the same parity
    # (rank ^ 2k): every core-0 rank hears from core-1 ranks and vice versa
    # (in traces same-parity rounds add ~0.8 us to the core-0 ranks' lag
    # behind the core-1 ranks). The same slots and semaphores; the same bits.
    crossed_pair_gathers: bool = False
    # Token-scattered decode MoE with the routed and shared reduce-scatters
    # (B6-B28 rows, pooled weights): before the experts only the shared
    # expert's gate/up projection runs; its down projection and the start of
    # the shared reduce-scatter run after the routed send, in the routed
    # reduction's waits (the shared shards leave from spare rows of the
    # shared receive buffer), and the first two expert slots' gate/up copies
    # are issued before their down copies. The same products, sums and
    # rounding points (bit-identical).
    decode_late_shared_down: bool = False
    # Token-scattered decode MoE at 8 rows (rank-pair row owners, both
    # reduce-scatters, expert schedule, pooled weights): the routed
    # reduce-scatter gives each lane its rank pair's tokens, so the pair's two
    # cores on every host hold them after the host round, and rank (host h,
    # core c) projects them through latent-up lane block 2h + c and sends the
    # result straight to the tokens' other owners: no routed all-gather, and
    # the output round replaces the lane-to-owner transpose. The shared expert
    # reduce-scatters to the same (tokens, block) owners, and the latent-up
    # copy starts with the last expert pair. Programs that take it read
    # ``latent_up_tp`` (the latent_up blocks indexed by host and core). The
    # same products, sums and rounding points (bit-identical).
    moe_token_parallel_up: bool = False
    # Serving batches beyond the powers of two, by request count: plain
    # decode and three-draft verifies (e.g. "48" / "12"; multiples of 16 rows
    # take the staged MoE paths).
    extra_plain_batches: tuple[str, ...] = ()
    extra_spec_batches: tuple[str, ...] = ()
    # MLA verify page plan overrides (pages a step, steps in flight; 0: the
    # kernel's choice), for verifies of more than four rows.
    verify_page_span: int = 0
    verify_page_buffers: int = 0
    # More DCP32 verify shapes (rows x rows per sequence, e.g. "16x4,48x4")
    # on the grouped FP8 QK/PV path (mla_dcp.GROUPED_SPEC_SHAPES).
    grouped_spec_extra: tuple[str, ...] = ()
    fused_cache_commit: bool = True
    decode_no_hlo_calls: bool = True  # no deduplicated HLO calls at B8..B128, DP B2/B4
    shard_routing: bool = False
    kda_staged: bool = False
    weight_streams: tuple[tuple[str, tuple], ...] = ()  # (weight, (axis, streams))
    # VMEM and compiler settings.
    vmem_limit_mb: int = 64
    vmem_spill_reserve_mb: float = 5.0
    internal_scratch_kb: int = 0
    bounds_checks: bool = False
    # Prefill attention.
    # Prefill MLA flash tiles (queries x keys per step).
    mla_flash_qtile: int = 1024
    mla_flash_ktile: int = 2048
    # Kimi-K3.1: the prefill's MLA attends each token's DSA top-k (token-split
    # indexer, sparse MLA kernels) instead of every earlier token.
    prefill_sparse: bool = False
    # Kimi-K3.1 decode indexer: slots of its weight stream (wq_b / wk /
    # weights_proj, nine 128 KB chunks a layer; batches up to 8 rows); ten
    # issues every chunk when the MLA layer starts.
    dsa_weight_ring: int = 3
    # Kimi-K3.1 decode selection: once every row's threshold bin holds at most
    # 16 keys over the DCP group, gather them and finish the rounds locally
    # (batches up to 8 rows).
    dsa_gather_finish: bool = False
    # Kimi-K3.1 decode selection: 4-bit key rounds (8: the exact top-k). Fewer
    # select every key at or above the threshold bin's lower bound -- a
    # superset of the top-k, ~2048 + the bin's other keys.
    dsa_select_rounds: int = 8
    # Kimi-K3.1 decode selection, 5-round bin selection of up to 16 rows: the
    # first count round takes 63 coarse thresholds (2^26 apart) and 64 fine
    # ones (2^19 apart) centred on each row's threshold of the previous step
    # (per MLA layer, row and request, carried between steps by the decode
    # ops). When every row's top-k threshold lands in a fine bin one more
    # round finishes (two count exchanges instead of three); otherwise two
    # more rounds, as before. The same thresholds whatever the centres.
    dsa_seeded_window: bool = False
    # Kimi-K3.1 decode attention (grouped FP8 verifies): a step whose rows
    # select at most this many of its tokens (their union) attends only them,
    # gathered into that many slots by a one-hot matmul; busier steps run
    # dense with the mask. 0: always dense.
    dsa_compact_tokens: int = 0
    # Compact verify steps append their gathered tokens to a window of this
    # many slots, folded once when full and at each row's last step (instead
    # of one fold a step). 0: fold every compact step.
    dsa_compact_window: int = 0
    # Grouped FP8 verify folds (compact steps, window flushes, dense steps):
    # each query row's 96-row block of the four-row score tile is masked by
    # that row's own token mask, instead of a [4 x 96, tokens] mask built by
    # a select chain over the row index. The same masked scores.
    dsa_row_block_masks: bool = False
    # Kimi-K3.1 decode attention whose query gather leaves during the indexer
    # (unchunked DSA calls, up to 8 rows): the gather wait, the FP8 query
    # unpack and the softmax state reset run inside the selection's second
    # exchange round (between its sends and its wait; seeded window: the
    # round after the window round) instead of at the start of the
    # attention, and the first steps' key copies start once the keys scratch
    # is complete (dsa_exchange_fill: in the first exchange's wait). The same
    # values in the same buffers, earlier.
    dsa_prologue_in_select: bool = False
    # Kimi-K3.1 decode epilogue: commit the step's latent rows and DSA
    # index keys in one kernel (paged._commit_dsa_rows_fused) instead of the
    # latent commit kernel and 48 XLA scatters.
    dsa_fused_commit: bool = False
    # Kimi-K3.1 decode selection scoring loop: slots of the key stage ring
    # (each index page's keys reach the HBM keys scratch through it; a slot's
    # copy is waited this many pages after it starts, so two leave one page
    # step for a copy's latency).
    dsa_stage_slots: int = 2
    # Kimi-K3.1 decode selection scoring loop, selections of up to 8 rows:
    # index pages in flight (0: dsa_dcp.PAGE_SLOTS). The ring's first copies
    # start from a loop (one copy of the issue code whatever the depth).
    dsa_page_slots: int = 0
    # Kimi-K3.1 decode selection, up to 8 rows whose keys fit the count
    # passes' two resident chunks: the scoring loop writes each page's keys
    # straight into that buffer (no stage ring and no copy a page); the count
    # passes read them in place (the first pass copies nothing back from
    # HBM) and the HBM keys scratch is filled by one copy a row, in flight
    # under the exchange rounds. The same keys and thresholds.
    dsa_resident_keys: bool = False
    # Kimi-K3.1 decode selection scoring loop: score the in-flight tokens on
    # a page in one loop over the sequence's tokens (a row's update a step)
    # instead of a branch per (row, token): the same keys in less code.
    dsa_patch_loop: bool = False
    # Kimi-K3.1 decode indexer: plan the selection's steps and start the
    # scoring ring's first index-page copies while the index query/key gather
    # is in flight (neither needs the gathered queries) instead of after it.
    dsa_early_plan: bool = False
    # Kimi-K3.1 decode selection: the step activity writes only the entries
    # the attention reads -- each sequence's first row's steps, up to the
    # longest row's live steps and one more -- instead of every row's steps
    # of the whole block table (the scalar loop ran batch x the table's
    # steps a layer: 8 x 32 at a 1M-context table, for ~11 live steps at
    # 344k). The attention reads the same values (bit-identical).
    dsa_live_activity: bool = False
    # Kimi-K3.1 decode selection, grouped count passes (up to 8 rows a
    # group): the count buffer holds its rows' key chunks an odd number of
    # 8-row tiles apart (a chunk of an even number of tiles padded by 8
    # rows), so a key row's strided load of the 8 rows is two VMEM loads
    # instead of eight (a 1M-context table's 128-row chunk: 64 loads and 56
    # ORs a count block instead of 16 and 8). The same keys and counts.
    dsa_odd_stride: bool = False
    # Kimi-K3.1 decode selection: the first count exchange's wait (its remote
    # copies in flight) runs the work that needs no threshold instead of the
    # end of the rounds: the dense step activity's fill (every attention step
    # marked), each row's own in-flight key and the bin selection's p_tie
    # (lane extractions into SMEM) and the resident keys' copy wait (with
    # dsa_prologue_in_select, then the attention's first key copies). The
    # same SMEM values and copies (bit-identical).
    dsa_exchange_fill: bool = False
    # Kimi-K3.1 decode selection scoring loop: each in-flight token's keys go
    # into its page for all of the sequence's rows with one lane rotation and
    # a masked select (instead of a lane sum per row and token). The same
    # keys (bit-identical).
    dsa_vector_patch: bool = False
    # Kimi-K3.1 decode selection scoring loop, sequences of 4+ rows: on a page
    # holding the sequence's in-flight tokens only the products of its parts
    # (a subpage at 8 rows a sequence, two at 4) below the sequence's last
    # position run (one loop over them); the subpages past it take INT_MIN
    # keys, the value the masked products give them. Other pages keep one
    # straight-line body. The same keys (bit-identical).
    dsa_skip_empty_subpages: bool = False
    # MLA DCP combine of unchunked one-group calls with three heads a rank
    # (B1, B4, B8 and the 8-row verifies) in two stages: the chip's two
    # cores (members 2k, 2k + 1) first merge their partials on-chip, each
    # core for the owners of its parity, then send each owner one merged
    # partial per chip -- half the cross-chip combine bytes. Merged partials
    # round to BF16 once more.
    dcp_pair_combine: bool = False
    # DCP attention step loop: power-of-two divisions and remainders of
    # positions, pages and steps as arithmetic shifts and masks (floor
    # division for every int32, without the sign fixups), and the step that
    # writes a verify sequence's cached rows found once before the loop
    # instead of from its positions every step. Bit-identical.
    dcp_shift_scalars: bool = False
    # FP8 query packets (DCP query gather): unpack only the last 128-lane
    # tile through 32-bit lanes (the scale's bytes, the zeroed pad lanes);
    # lanes 0:512 pass through as bytes. The same query bytes and scale bits.
    dcp_query_tail_unpack: bool = False
    # Decode layer loop: test KDA first in the layer-type branch, so the
    # instruction overlays place the post-join code for the 68 KDA layers.
    decode_kda_branch_first: bool = True
    # Decode MLA layers with the DSA indexer (attention DP1): run the q/kv
    # projections, the new row's hand-out and the selection inside a loop of
    # one trip (the trip count is read at run time). The instruction overlays
    # then hold the MLA code in two overlays split at that loop, and the one
    # the KDA path prefetches fits in IMEM beside the KDA path's code. The
    # same operations run in the same order (bit-identical).
    decode_mla_prefix_region: bool = False
    # Decode MLA layers (attention DP1): the new rows' hand-out copies for the
    # cache commit start as soon as q6 has written the rows and are waited in
    # q8, after the attention, instead of starting and waiting back to back
    # there. Nothing writes the rows in between (bit-identical).
    decode_mla_early_handout: bool = False
    # 8- and 16-row decode steps: keep the small embedding all-reduce on the
    # TensorCore instead of the SparseCore collective offload.
    decode_tc_all_reduce: bool = False
    # Decode steps of up to 32 rows (the DSpark verifies included): the rank
    # holding a token's embedding row reads it and sends it to the other
    # ranks in one kernel (kernels.token_kernels.owner_embedding) instead of
    # the local gather and the FP32 psum of the rows. The psum adds the
    # owner's BF16 value to zeros, which only turns -0.0 into +0.0; the
    # kernel's one-hot row sum does the same (bit-identical).
    decode_embedding_broadcast: bool = False
    # DSpark verify ops: the greedy decisions (each row's vocabulary argmax,
    # the accepted count and the bonus token) in one kernel with a single
    # exchange of per-shard maxima (kernels.token_kernels.greedy_decisions)
    # instead of jnp.argmax, an all_gather and ~20 small XLA ops. The same
    # first-index maxima and lowest-rank ties, NaN counting as the maximum
    # (bit-identical decisions).
    verify_greedy_kernel: bool = False
    # Kimi-K3.1 DSpark verify ops (attention DP1, with dsa_fused_commit): the
    # accepted KDA states, latent rows and DSA index keys commit in one
    # kernel (paged.commit_verify_tail) instead of the KDA state commit and
    # the fused DSA commit. It builds the FP8 latent rows from the decoder's
    # BF16 handoff in VMEM (no XLA convert, slices and concatenation) and
    # runs the DSA commit under the first KDA post-state's reads. The same
    # bytes (bit-identical).
    verify_fused_tail: bool = False
    # 16-row decode steps: records in the instruction-overlay allocation table
    # (XLA's xla_tpu_overlay_allocation_table_size; 0: the compiler's 10). At
    # 16 rows the KDA layers' overlays are larger than IMEM. With 2 records
    # the overlay handler skips the back-edge prefetch of the first MLA
    # overlay, so on KDA layers it prefetches the experts overlay while the
    # post-join code runs instead of loading it on demand. Only the prelude's
    # overlay handler changes; the program's code is the same.
    decode_b16_overlay_table: int = 0
    # MLA DSpark drafts with more than two heads a rank (Kimi-K3.1's): hold
    # o_proj whole in VMEM at up to 8 rows and down_proj at up to 16 (attention
    # DP1) instead of streaming them one 896-column chunk at a time.
    draft_whole_weights: bool = False
    # MLA DSpark single-request draft op (attention DP1): the fused draft
    # kernel reads the verify's packed ids and the decisions directly (scalar
    # prefetch) and the layer-norm stacks as [layers, width] (no relayout
    # copies); one commit kernel reads the rows' positions and block table
    # from the ids, writes each latent tile its rows touch once
    # (paged.commit_draft_rows) and computes the step's emitted tokens, next
    # ids and control state. The op runs two kernels and one convert instead
    # of ~35 small XLA ops. The same integers and cache bytes (bit-identical).
    draft_fused_epilogue: bool = False
    # draft_fused_epilogue's scalar copies (the output vector, the next ids'
    # block tables and the control state; all a request's table entries
    # whatever its context): this many words a loop iteration (0 or 1: one).
    # A word an iteration pays the loop's exit test and branch: 7-8 bundles a
    # word, ~7.2K a step at a 1M-context table (8 x 64 entries in the ids).
    # The same words.
    draft_epilogue_unroll: int = 0
    # DSpark verify operand packing (DecodeOps._build_verify_metadata: the
    # ids, the plan rows and the verify modes in one operand) folded into the
    # op before the verify: the MLA single-request draft op also returns the
    # next verify's packet with verify mode 1 (a continuing request whose
    # plan is unchanged and whose scheduler took all eight rows verifies
    # from it), and the batched enter op returns its packet beside the ids.
    # One small program fewer a step; the same packet (bit-identical).
    draft_verify_packet: bool = False
    # MLA DSpark drafts' two-stage DCP combine (host merge, then cross-host;
    # mla_dcp.DcpAttention.host_combine) with a core-pair stage first on a
    # call's last chunk (an earlier chunk's stage one is in flight under the
    # next chunk's pages), as dcp_pair_combine does for the decoder: each
    # core sends its chip mate the partials of the owners on the mate's
    # parity's lanes (on-chip) and merges the two partials of each owner on
    # its own parity's, so stage one sends one merged partial a chip instead
    # of one a core -- half its cross-chip bytes within the host. The
    # pair-merged partials round to BF16 once more (numerics change).
    draft_pair_combine: bool = False
    # MLA DSpark drafts: the LM head streams through six chunk slots instead
    # of two -- the gate / up buffers' halves join the two scoped ones once
    # the last layer's gate and up products have read them, and the first
    # four chunks start right then (under the last down_proj all-reduce).
    # The same products in the same order (bit-identical).
    draft_lm_prefetch: bool = False
    # MLA DSpark drafts with FP8 pools: the DCP attention stages its pages as
    # [span * 512, 640] in the MXU's native tile (each page DMA retiles a pool
    # block, as attention DP2's plain decode does) instead of the pool's
    # [span * 128, 4, 640] blocks, whose loads relayout every tile before QK.
    # The same FP8 bytes in the same token order reach the same products
    # (bit-identical).
    draft_native_pages: bool = False
    # MLA DSpark drafts: the DCP attention runs each sequence's page steps
    # but its last without the visibility mask and the stale-token zeroing
    # (every token of an earlier step is visible; the last step, peeled from
    # the loop, keeps both), and its probability-value products accumulate
    # without the 1/256 of their scaled FP8 probabilities, which is applied
    # once before the normalization (a power of two: the accumulator holds
    # exactly 256 times its value at every step). Bit-identical.
    draft_step_trims: bool = False
    # Decode MoE router (weight arena, unsharded top-k): every rank copies and
    # projects both router halves, its own and its pair peer's (14 MiB of
    # arena instead of 7), instead of exchanging score halves with the peer.
    decode_router_both_halves: bool = False
    # Decode MoE at 8 and 16 rows with both router halves (attention DP1):
    # the top-16 sorts each lane's seven candidates once and then takes one
    # cross-lane max per selection instead of an argmax and a lane gather
    # (any selection with an equal score, a NaN or a subnormal reruns
    # today's argmax loop, so the experts, probabilities and sum order stay
    # the same); the latent all-gather starts right after the router
    # products; with the expert steal the plan stays in vectors, its scalars
    # and the first experts move to the scalar unit in one batch and the
    # first pair's gate/up copies start right after the top-k. Bit-identical.
    decode_presorted_routing: bool = False
    # Decode KDA input (q|k|v|g|b) and output projections as FP8 e4m3 with one
    # FP32 scale per output column (amax / 448), converted from the pre-shard
    # at load (attention DP1). The activations stay BF16; the products run on
    # the MXU's FP8 weight path. The decay-gate projections stay BF16. The
    # prefill keeps its own BF16 input projection (colocated programs no
    # longer share it). Changes numerics: weight quantization.
    kda_fp8_projections: bool = False
    # Decode weights in shapes XLA passes to the megakernel without a relayout
    # copy (attention DP1), converted at load: the BF16 norm stacks
    # (attn_norm, ffn_norm, m_qnorm, m_knorm [layers, 1, width] and
    # final_norm) as FP32, and the KDA short-conv filter, decay rate and
    # time-step bias as FP32 [layers, 1, width] rows (filter: 4608 lanes in
    # (tap, q/k/v, head, channel) order; decay rate and bias: head h at lanes
    # 128 h.., the decay rate repeated over its head's lanes). XLA lays out a
    # BF16 [layers, 1, width] parameter with the layers as the tiled rows and
    # the other KDA shapes with their own tiled dims, so each of these 8
    # families was relaid out on every step (~23 us of an 8-row verify). The
    # norms multiply in FP32 either way and the KDA values are the same:
    # bit-identical. The prefill keeps its own BF16 copies.
    decode_copy_free_weights: bool = False
    # Prefill MoE expert parallelism: 8 (EP8/TP4, the pre-shard layout), 16
    # (EP16/TP2: 56 experts, TP2 within the chip) or 32 (28 whole experts);
    # 16 and 32 regroup the EP8 group's shards at load.
    prefill_expert_parallel: int = 16
    # Decode routed experts: 8 (EP8/TP4, the pre-shard layout) or 16 (the
    # prefill's EP16/TP2 regrouping, run as TP4-shaped half experts), so
    # colocated programs can share EP16 experts with the fused prefill MoE.
    decode_expert_parallel: int = 8
    # EP8/TP4 routed experts in the sibling-steal layout: the two cores of a
    # chip hold different expert groups with the same TP4 quarter (kernel rank
    # r: group 2 * (r // 8) + r % 2, quarter (r % 8) // 2, read from pre-shard
    # directory 4 * group + quarter; kernels.expert_steal). The decode router,
    # the rank loader and the EP8 prefill follow it. Numerics: each host's
    # eight-lane routed sum adds the same lane partials in another order.
    expert_steal: bool = False
    # Packed dispatch/shared overlap and fused down/combine/reduce-scatter.
    # EP16: FP32 at all serving windows, BF16 at 4K/8K only.
    # False: the unfused SparseCore gather/combine.
    prefill_fused_moe: bool = True
    # Both prefill MoE rings: BF16 halves transfer bytes, rounding every hop.
    moe_reduce_dtype: str = "float32"
    # Weight loading.
    read_streams: int = 4
    load_inflight_gb: float = 16.0
    # Diagnostics (off in serving): dcp_skip and debug_skip skip named work, so
    # results are invalid (timing or hang bisection only); the rest add trace
    # scopes, VMEM reports or VMEM padding.
    dcp_skip: tuple[str, ...] = ()
    debug_skip: tuple[str, ...] = ()
    phase_scopes: bool = False
    pool_report: bool = False
    vmem_report: bool = False
    vmem_pad_kb: int = 0

    def __post_init__(self) -> None:
        if self.prefill_expert_parallel not in (8, 16, 32):
            raise ValueError("prefill_expert_parallel is 8, 16 or 32")
        if self.decode_expert_parallel not in (8, 16):
            raise ValueError("decode_expert_parallel is 8 or 16")
        if self.expert_steal and self.decode_expert_parallel != 8:
            raise ValueError("expert_steal needs decode_expert_parallel=8")
        if self.expert_steal and self.attention_dp != 1:
            raise ValueError("expert_steal needs attention_dp=1")
        if self.kda_fp8_projections and self.attention_dp != 1:
            raise ValueError("kda_fp8_projections needs attention_dp=1")
        if self.decode_copy_free_weights and self.attention_dp != 1:
            raise ValueError("decode_copy_free_weights needs attention_dp=1")
        if self.moe_reduce_dtype not in ("float32", "bfloat16"):
            raise ValueError("moe_reduce_dtype is float32 or bfloat16")
        if self.mla_decode_ktile not in (1024, 2048):
            raise ValueError("mla_decode_ktile supports 1024 or experimental 2048")
        if self.mla_decode_ktile == 2048 and (
            self.attention_dp != 2 or self.dcp_group != 16
        ):
            raise ValueError(
                "mla_decode_ktile=2048 requires attention_dp=2 and dcp_group=16"
            )

    def key(self) -> str:
        """A stable digest of every field, for caches of compiled kernels."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def replace(self, **fields: Any) -> "KernelConfig":
        return dataclasses.replace(
            self, **{name: _coerce(self, name, value) for name, value in fields.items()}
        )


def _coerce(config: KernelConfig, name: str, value: Any) -> Any:
    """Parse a command-line string into the field's type."""
    current = getattr(config, name)
    if not isinstance(value, str):
        return value
    if current is None:  # an optional override: parse by the declared type
        if value.lower() in ("", "none"):
            return None
        kind = str(KernelConfig.__dataclass_fields__[name].type)
        if kind.startswith("bool"):
            return value.lower() in ("1", "true", "yes", "on")
        if kind.startswith("int"):
            return int(value)
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if name == "weight_streams":
        return tuple(sorted((k, tuple(v)) for k, v in json.loads(value).items()))
    if isinstance(current, tuple):
        return tuple(filter(None, value.split(",")))
    return value


_ACTIVE: KernelConfig | None = None


def install(config: KernelConfig) -> KernelConfig:
    """Fix the process's kernel config. Kernel modules read it at import, so
    installing a different config after the first import is an error."""
    global _ACTIVE
    if _ACTIVE is not None and config != _ACTIVE:
        raise RuntimeError(
            f"K3 kernel config is already fixed ({_ACTIVE}); install {config} before "
            "the first kernel import"
        )
    _ACTIVE = config
    set_active_kernel_config(config)
    return config


def active() -> KernelConfig:
    """The installed config, or the defaults (installed on first use)."""
    return _ACTIVE if _ACTIVE is not None else install(KernelConfig())
