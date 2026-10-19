"""The GLM-5.3-Flash megakernel: one Pallas program per device per target step.

Inside the program: the embedding lookup, every decoder layer (mHC, KDA or DSA
attention with its indexer, MoE or dense MLP), the tensor-parallel reductions as
in-kernel remote DMA, the final norm and the LM head.  A step can carry several
query positions (an MTP verify); the KDA states then keep one checkpoint per query.

The layer schedule comes from the spec: three dense-MLP KDA layers, then
repetitions of [DSA, KDA, KDA, KDA] with MoE as a fori_loop over traced ordinals
(kda_ord / dsa_ord / moe_ord are affine in the layer index), then the remainder.

DSA attention is absorbed: the key half of kv_b folds into the query and the value
half into the output, so attention runs on the 512-wide latent cache.

Weights stream from HBM through VMEM rings.  The DSA caches are only read here; a
step's new rows come out as small outputs that `row_write_call` stores, because
XLA copies every operand that a kernel with remote DMA aliases.
"""

from __future__ import annotations

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import phases as K

F32 = jnp.float32
BF16 = jnp.bfloat16
AXIS = "tp"


# Names in-kernel regions for a device trace (with libtpu's
# --xla_enable_custom_call_region_trace=true); off in normal builds.
_TRACE = False


def _scope(name):
    return jax.named_scope(name) if _TRACE else contextlib.nullcontext()


def _wait(copy, name):
    """copy.wait(), named for the trace: how long a phase waits on its DMA."""
    with _scope(name):
        copy.wait()


def kda_ord(L):
    return L - (L + 1) // 4


def dsa_ord(L):
    return (L - 3) // 4


def moe_ord(L):
    return L - 3


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    max_len: int                  # cache capacity; must be a multiple of index_kpool
    n_periods: int = 10           # body repetitions; fewer builds a partial model
    tile_rows: int = 256
    vmem_mib: int = 110
    slots: int = 4                # weight-ring depth
    prefetch: int = 2             # must satisfy prefetch <= slots - 2
    ring: bool = True             # False restores the synchronous baseline
    n_query: int = 1              # query positions per call; 2 verifies an MTP
                                  # draft alongside the accepted token
    n_layers: int | None = None   # hard cap, for bisecting a numerical disagreement
    attn_tile: int = 512          # cache tokens per DSA stream tile (128 pools)
    # KDA qkv / o projections as fp8 e4m3 with per-output-channel scales (the
    # runner quantizes them once), streamed through the bf16 ring.
    fp8_kda: bool = True
    # Experts as per-output-channel fp8, K-major (the pack step's *_ch arrays): one
    # dot and a scale row per expert matvec instead of folded 128x128 block scales.
    ch_experts: bool = True
    # Collective payloads in bf16 (each rank's partial rounded before the f32 sum),
    # as the native TP all-reduce of bf16 activations does.
    bf16_collectives: bool = True
    # K-major dense stream: [K, km_cols] blocks, km_slots ring slots, km_issue in
    # flight.  Four queries overflow VMEM with 512-wide blocks.
    km_slots: int = 4
    km_issue: int = 2
    # The embedding table as [V, D] instead of [V, 1, D]: a caller holding it 2-D
    # (serving's torch tensor, 8-row tiles) would otherwise re-tile all of it per step.
    embed_2d: bool = False
    init_streams: bool = False    # take the residual stream as an input instead of
                                  # embedding a token; lets a single layer be driven
                                  # from the reference's own input, which is the only
                                  # comparison that isolates one layer

    @property
    def km_cols(self) -> int:
        return 512 if self.n_query == 1 else 256

    def __post_init__(self):
        if self.max_len % self.attn_tile:
            raise ValueError(f"max_len={self.max_len} must be a multiple of "
                             f"attn_tile={self.attn_tile}")
        if self.ring and self.prefetch > self.slots - 2:
            raise ValueError(
                f"prefetch={self.prefetch} needs slots >= {self.prefetch + 2}, got "
                f"{self.slots}: issuing into a slot that still holds an unwaited "
                "transfer puts two DMAs on one semaphore, which corrupts results "
                "and halts the core -- and only once the tile count exceeds the "
                "ring, so a small shape will not reproduce it")

    def layers(self, spec) -> int:
        """How many decoder layers this configuration actually runs."""
        n = min(spec.num_hidden_layers,
                spec.first_k_dense_replace + 4 * self.n_periods + 2)
        return n if self.n_layers is None else min(n, self.n_layers)


class SlabIndex:
    """Static offsets into the per-layer small-tensor slab.

    Inside the rolled layer loop the layer index is a traced value, so the float32
    offsets cannot be looked up per layer.  They do not need to be: the packer
    appends in a fixed order, so every layer of a given kind has identical offsets.
    The maps are therefore keyed by kind, and the constructor asserts that all
    layers of a kind really do agree rather than trusting it.
    """

    def __init__(self, small_map, small_f32_map, spec):
        self.m = {k: (v[0], v[1], tuple(v[2])) for k, v in small_map.items()}
        by_kind: dict[str, dict] = {}
        for L, m in small_f32_map.items():
            L = int(L)
            kind = self.kind_of(spec, L)
            if kind in by_kind and by_kind[kind] != m:
                raise ValueError(
                    f"layer {L} has different float32 slab offsets from another "
                    f"{kind} layer; the rolled loop assumes they match")
            by_kind[kind] = m
        self.f = by_kind

    @staticmethod
    def kind_of(spec, L: int) -> str:
        if L == spec.mtp_layer_idx:
            return "mtp"
        a = "kda" if spec.layer_types[L] == "linear_attention" else "dsa"
        return f"{a}_{spec.mlp_layer_types[L]}"

    def get(self, ref, tag):
        off, nrows, shape = self.m[tag]
        n = 1
        for d in shape:
            n *= d
        return ref[off : off + nrows].reshape(-1)[:n].reshape(shape)

    def getf(self, ref, kind, tag):
        """Rank-2 on purpose: rank-1 slices of these feed shape casts that do not
        lower, and the failure surfaces far from here."""
        off, n = self.f[kind][tag]
        return ref[0:1, off : off + n]


class _Half:
    """One half of a two-slot VMEM buffer, chosen by `i`, indexed like a ref."""

    def __init__(self, buf):
        self.buf, self.i = buf, 0

    def __getitem__(self, idx):
        return self.buf.at[self.i][idx]


def pack_kmajor(w, rows, tile_n):
    """[L, N, K] (output-major, as packed) -> K-major ring slots [L, n, rows, tile_n].

    Slot s holds column blocks s*k .. s*k+k-1 (k = rows // K) of W^T [K, N], each
    [K, tile_n], stacked along rows, so a slot is one contiguous copy and each
    block feeds a plain x[Q, K] @ W[K, tile_n] dot."""
    L, N, K = w.shape
    k = rows // K
    if k < 1 or N % (tile_n * k):
        raise ValueError(f"cannot pack [{N}, {K}] into [{rows}, {tile_n}] slots")
    n = N // (tile_n * k)
    out = (w.transpose(0, 2, 1).reshape(L, K, n, k, tile_n)
           .transpose(0, 2, 3, 1, 4).reshape(L, n, k * K, tile_n))
    if k * K < rows:   # K does not divide the slot (e.g. 1536 of 4096): pad, unread
        out = jnp.pad(out, ((0, 0), (0, 0), (0, rows - k * K), (0, 0)))
    return out


def pack_slots(w, tile_rows, hidden):
    """[L, out, in] -> [L, out // (tile_rows * k), tile_rows, hidden], k = hidden // in:
    ring slot s holds row tiles s*k .. s*k+k-1 side by side, so one copy fills it."""
    L, out, inn = w.shape
    k = hidden // inn
    n = out // (tile_rows * k)
    return (w.reshape(L, n, k, tile_rows, inn).transpose(0, 1, 3, 2, 4)
            .reshape(L, n, tile_rows, k * inn))


def _dma(src, dst, sem):
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()


def pick_tile(out_dim: int, want: int) -> int:
    """Largest divisor of `out_dim` that is at most `want` and sublane-aligned.

    A fixed tile only works when it divides every projection's output width.  It
    does for the real geometry and does not for a small test geometry, and a test
    shape that cannot be tiled is exactly the shape a bug would hide in.
    """
    for d in range(min(want, out_dim), 0, -8):
        if out_dim % d == 0:
            return d
    return out_dim


def _flags(cfg: KernelConfig):
    return (cfg.fp8_kda, cfg.ch_experts, cfg.bf16_collectives,
            cfg.km_cols, cfg.km_slots, cfg.km_issue)


def scratch_shapes(spec, layout, cfg: KernelConfig):
    """VMEM scratch, sized from the spec; the comment on each is why it is that big."""
    FP8_KDA, CH_EXPERTS, BF16_COLL, KM_TILE, KM_SLOTS, KM_PF = _flags(cfg)
    tp = layout.tp
    H = spec.hidden_size
    qsh = spec.kda_qkv_dim // tp
    hkda = spec.linear_num_heads // tp
    nh = spec.num_attention_heads // tp
    dh = spec.linear_head_dim
    qk, vd, kvl = spec.qk_head_dim, spec.v_head_dim, spec.kv_lora_rank
    idh = spec.index_head_dim
    moesh = spec.moe_intermediate_size // tp
    ffnsh = spec.intermediate_size // tp
    TR = cfg.tile_rows
    L = cfg.max_len
    slab_rows = layout.small_rows()

    f8 = jnp.float8_e4m3fn
    S = cfg.slots if cfg.ring else 1
    ffnsh = spec.intermediate_size // tp
    # A one-query step keeps its state in st / cv; the checkpoints are placeholders.
    # Two slots: each checkpoint is written out while the next query runs.
    ck_st = (2, hkda, dh, dh) if cfg.n_query > 1 else (1, 8, 128)
    ck_cv = ((2, spec.linear_conv_kernel_dim - 1, 3 * qsh)
             if cfg.n_query > 1 else (1, 8, 128))
    return [
        # Two of each: the next layer's slab is fetched during this layer.
        pltpu.VMEM((2, slab_rows, H), BF16),           # sm   : the layer's slab
        pltpu.VMEM((2, 1, H), F32),                    # smf  : its float32 part
        # One ring per weight shape.  Each ring has its own semaphore array: two
        # in-flight transfers must never share a semaphore index, and sharing one
        # array across rings would do exactly that.
        pltpu.VMEM((KM_SLOTS, H, KM_TILE), BF16),      # rbf  : K-major [K, TN] tiles
        (pltpu.VMEM((S, H, 2 * moesh), f8) if CH_EXPERTS else      # r8a : expert gate/up
         pltpu.VMEM((S, max(TR, 2 * moesh), H), f8)),   #        (and fp8 in == H tiles)
        pltpu.VMEM((S, TR, max(nh * vd, ffnsh, 128)), f8),   # r8b : fp8, out == H
        (pltpu.VMEM((S, moesh, H), f8) if CH_EXPERTS else          # r8d : expert down
         pltpu.VMEM((S, H, max(moesh, 128)), f8)),
        # With CH_EXPERTS the fp8 in == H projections get their own ring and each
        # expert slot carries its scale rows; otherwise these are placeholders.
        pltpu.VMEM((S, TR, H) if CH_EXPERTS else (1, 32, 128), f8),       # r8i
        pltpu.VMEM((S, 1, 2 * moesh) if CH_EXPERTS else (1, 1, 128), F32),  # egsr
        pltpu.VMEM((S, 1, H) if CH_EXPERTS else (1, 1, 128), F32),          # edsr
        pltpu.VMEM((hkda, dh, dh), F32),               # st   : one layer's KDA state
        # With Q > 1 the state after every query is a checkpoint: the delta rule
        # has no inverse, so the next step resumes from the last accepted one.
        pltpu.VMEM(ck_st, F32),                        # stq  : state checkpoints
        # The conv state and weights are channels-last, [K-1, C] / [K, C]: as
        # [C, K-1] the short axis pads to 128 lanes (1.5 MiB per state buffer).
        pltpu.VMEM((spec.linear_conv_kernel_dim - 1, 3 * qsh), F32),   # cv
        pltpu.VMEM(ck_cv, F32),                        # cvq
        # The KV cache is streamed from HBM in attn_tile-token tiles (two
        # buffers each), so VMEM no longer grows with the context.
        pltpu.VMEM((2, cfg.attn_tile, kvl), BF16),     # kvt  : latent tiles
        pltpu.VMEM((32, idh), BF16),                   # ikb  : raw indexer key rows
        pltpu.VMEM((32, idh), BF16),                   # igb    around this step, to
                                                       #        close its pool(s)
        # One layer's pooled keys whole: 1.3 MiB at 10k tokens, 4 MiB at 32k.
        pltpu.VMEM((L // spec.index_kpool, idh), F32), # pka  : pooled indexer keys
        pltpu.VMEM((cfg.n_query, L // cfg.attn_tile,
                    cfg.attn_tile // spec.index_kpool), F32),   # scr: pool scores
        pltpu.VMEM((nh * (qk + vd), kvl), BF16),       # kvb  : absorbed both ways
        pltpu.VMEM((2 * qsh, dh), BF16),               # fgb  : forget/output gate b
        pltpu.VMEM((1, qsh), F32),                     # dtv  : dt_bias shard
        pltpu.VMEM((1, 3 * qsh + H), F32),             # ksc  : KDA fp8 channel scales
        pltpu.VMEM((spec.linear_conv_kernel_dim, 3 * qsh), BF16),  # cwv : conv shard
        pltpu.VMEM((hkda, H), BF16),                   # bpv  : b_proj shard
        pltpu.VMEM((1, hkda), F32),                    # alv  : A_log shard
        pltpu.VMEM((spec.n_routed_experts, H), BF16),  # rtr  : router, replicated
        pltpu.VMEM((1, H), BF16),                      # fnv  : final norm
        pltpu.VMEM((2, tp, cfg.n_query, H), F32),      # ar   : double-buffered pad
        pltpu.VMEM((cfg.n_query, H), F32),             # sendv: this device's rows
        pltpu.VMEM((cfg.n_query, H), F32),             # macc : MoE accumulator (Q > 1)
        pltpu.VMEM((2, tp, cfg.n_query, H) if BF16_COLL else (1, 1, 8, 128), BF16),  # arb
        pltpu.VMEM((cfg.n_query, H) if BF16_COLL else (8, 128), BF16),              # sendb
        pltpu.VMEM((8 if cfg.embed_2d else 1, H // layout.tp), BF16),   # embv : the
                                                       #        embedding row (with
                                                       #        embed_2d, its 8-row tile)
        # One buffer per scale shape, sized exactly: a DMA into a *slice* of a
        # VMEM buffer must be tile-aligned (8 sublanes x 128 lanes for f32), and a
        # block-scale grid is nothing like that shape.  Whole-array DMAs are fine.
        pltpu.VMEM((spec.q_lora_rank // 128, H // 128), F32),            # s_qa
        pltpu.VMEM((nh * qk // 128, -(-spec.q_lora_rank // 128)), F32),  # s_qb
        pltpu.VMEM((max(1, kvl // 128), H // 128), F32),                 # s_kva
        pltpu.VMEM((H // 128, max(1, nh * vd // 128)), F32),             # s_do
        pltpu.VMEM((max(1, 2 * moesh // 128), H // 128), F32),           # s_sgu
        pltpu.VMEM((H // 128, max(1, moesh // 128)), F32),               # s_sdn
        pltpu.VMEM((max(1, 2 * ffnsh // 128), H // 128), F32),           # s_dgu
        pltpu.VMEM((H // 128, max(1, ffnsh // 128)), F32),               # s_ddn
        pltpu.VMEM((spec.n_routed_experts, 2 * moesh // 128,
                    H // 128), F32),                   # egs  : all experts, one layer
        pltpu.VMEM((spec.n_routed_experts, H // 128,
                    max(1, moesh // 128)), F32),       # eds
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,                       # send
        pltpu.SemaphoreType.DMA,                       # recv
        pltpu.SemaphoreType.DMA((KM_SLOTS,)),          # sbf  : rbf ring
        pltpu.SemaphoreType.DMA((S,)),                 # s8a
        pltpu.SemaphoreType.DMA((S,)),                 # s8b
        pltpu.SemaphoreType.DMA((S,)),                 # s8d
        pltpu.SemaphoreType.DMA((8,)),                 # sgrp : batched small fetches
        pltpu.SemaphoreType.DMA((8,)),                 # sasy : early async fetches
        pltpu.SemaphoreType.DMA((8 * S,)),             # sbfn : rbf ring, narrow tiles
        pltpu.SemaphoreType.DMA((2, 2)),               # sslab: slab double buffer
        pltpu.SemaphoreType.DMA((S,)),                 # s8i  : r8i ring
        pltpu.SemaphoreType.DMA((2, S)),               # s8s  : expert scale rows
    ]


def build(spec, layout, cfg: KernelConfig, slab: SlabIndex):
    FP8_KDA, CH_EXPERTS, BF16_COLL, KM_TILE, KM_SLOTS, KM_PF = _flags(cfg)
    tp = layout.tp
    H = spec.hidden_size
    N = spec.hc_mult
    hsh = H // tp
    qsh = spec.kda_qkv_dim // tp
    hkda = spec.linear_num_heads // tp
    dh = spec.linear_head_dim
    nh = spec.num_attention_heads // tp
    qk, vd, kvl = spec.qk_head_dim, spec.v_head_dim, spec.kv_lora_rank
    idh, hi, kp = spec.index_head_dim, spec.index_n_heads, spec.index_kpool
    moesh = spec.moe_intermediate_size // tp
    ffnsh = spec.intermediate_size // tp
    blk = spec.weight_block_size
    eps = spec.rms_norm_eps
    kexp, nexp = spec.num_experts_per_tok, spec.n_routed_experts
    Lcap = cfg.max_len
    selk = spec.index_select_k
    TR = cfg.tile_rows
    f8 = jnp.float8_e4m3fn
    n_layers = cfg.layers(spec)
    Q = cfg.n_query

    def kernel(
        tok_ref, pos_ref, in_streams,
        w_embed, w_lm, w_fnorm, w_smalls, w_smallsf,
        w_kqkv, w_ko, w_kfg, w_kdt, w_kconv, w_kb, w_kalog,
        w_qa, w_qas, w_qb, w_qbs, w_kva, w_kvas, w_kvb, w_do, w_dos,
        w_iwq, w_router,
        w_egu, w_egus, w_edn, w_edns, w_sgu, w_sgus, w_sdn, w_sdns,
        w_dgu, w_dgus, w_ddn, w_ddns, w_ksc,
        # The DSA caches are read only: this step's rows go out through o_nkv /
        # o_nik / o_nig and `row_write_call` stores them.  A kernel with remote DMA
        # is a collective to XLA, which copies every aliased operand, and these
        # grow with the context.  The KDA states are read from the input refs and
        # written whole to the output refs; they are aliased only in a run capped
        # below the full depth, whose skipped layers must keep their state.
        c_kv, c_ik, c_ig, c_ip, _crec_in, _cconv_in,
        o_logits, o_streams, o_hidden, o_nkv, o_nik, o_nig, o_npk,
        c_recur, c_conv,
        sm2, smf2, rbf, r8a, r8b, r8d, r8i, egsr, edsr, st, stq, cv, cvq, kvt, ikb, igb, pka, scr,
        kvb, fgb, dtv, ksc,
        cwv, bpv, alv, rtr, fnv, ar, sendv, macc, arb, sendb, embv,
        s_qa, s_qb, s_kva, s_do, s_sgu, s_sdn, s_dgu, s_ddn, egs, eds,
        sem, sem2, ssem, rsem, sbf, s8a, s8b, s8d, sgrp, sasy, sbfn, sslab, s8i, s8s,
    ):
        me = jax.lax.axis_index(AXIS)
        pos = pos_ref[0]
        # With Q > 1 the KDA states carry one checkpoint per query; this step
        # resumes from checkpoint ck (the last token the previous step accepted).
        ck = pos_ref[1] if Q > 1 else None
        # `sm` / `smf` read the current layer's half of the slab double buffer;
        # layer() points them at it.
        sm, smf = _Half(sm2), _Half(smf2)

        def slab_copies(L, par):
            return [pltpu.make_async_copy(w_smalls.at[L], sm2.at[par], sslab.at[par, 0]),
                    pltpu.make_async_copy(w_smallsf.at[L], smf2.at[par], sslab.at[par, 1])]

        # --- in-kernel collectives -------------------------------------
        # Two correctness points, both of which cost results rather than crashing:
        #
        # 1. Each landing-pad slot has one writer per round: its owner's local
        #    or remote DMA. Do not clear the whole pad locally, since that can
        #    erase a peer's contribution that arrives before the local store.
        # 2. The pad is double buffered on a statically known parity.  A peer can
        #    be at most one collective ahead -- to finish round k+1 it needs this
        #    device's round k+1 contribution -- so alternating two pads keeps an
        #    early arrival for round k+1 away from round k's data.  The number of
        #    collectives per loop body is even, which is what makes the parity
        #    static inside the rolled layer loop.
        collective = [0]

        def _round(row_full, width):
            """`row_full` is [Q, H]; a shorter payload is zero-padded so the
            landing pad has one fixed shape across every collective."""
            par = collective[0] % 2
            collective[0] += 1
            if row_full.shape != (Q, H):
                row_full = jnp.pad(
                    row_full,
                    ((0, Q - row_full.shape[0]), (0, H - row_full.shape[1])))
            sbuf, rbuf = (sendb, arb) if BF16_COLL else (sendv, ar)
            sbuf[...] = row_full.astype(sbuf.dtype)
            # The local copy runs alongside the remote ones.
            local = pltpu.make_async_copy(sbuf, rbuf.at[par, me], sem)
            local.start()
            cps = []
            for d in range(1, tp):
                peer = jax.lax.rem(me + d, tp)
                cp = pltpu.make_async_remote_copy(
                    src_ref=sbuf, dst_ref=rbuf.at[par, me],
                    send_sem=ssem, recv_sem=rsem, device_id=peer)
                cp.start()
                cps.append(cp)
            local.wait()
            for cp in cps:
                cp.wait()
            return rbuf[par].astype(F32)                    # [tp, Q, H]

        def allgather(row, width):
            """[1, width] on each device -> [1, tp*width] in mesh rank order."""
            got = _round(row, width)                        # [tp, Q, H]
            return got[:, 0, :width].reshape(1, tp * width)

        def allreduce(rows):
            """[Q, H] partial sums on each device -> [Q, H] totals."""
            with _scope("allreduce"):
                return jnp.sum(_round(rows, H), axis=0)[: rows.shape[0]]

        def dma_batch(items):
            """Start every transfer, then wait for all of them.  Each item gets its
            own semaphore index: two in-flight transfers must never share one."""
            cps = [pltpu.make_async_copy(src, dst, sgrp.at[i])
                   for i, (src, dst) in enumerate(items)]
            for c in cps:
                c.start()
            for c in cps:
                _wait(c, "ld_wait")

        # --- streamed matvecs -------------------------------------------
        # Each projection streams row tiles through an S-slot ring, PF tiles ahead
        # (prefetch <= slots - 2, checked by the config); each ring has its own
        # semaphore array.
        S = cfg.slots if cfg.ring else 1
        PF = cfg.prefetch if cfg.ring else 0

        def _warm(n_tiles, make_copy, pf=None):
            """Start a stream's first transfers ahead of its compute (the weights
            do not depend on it); the stream itself is then run `warmed`."""
            pf = PF if pf is None else pf
            for t in range(min(pf + 1, n_tiles)):
                make_copy(t).start()

        def _stream(n_tiles, make_copy, consume, warmed=False, pf=None):
            pf = PF if pf is None else pf
            if not warmed:
                _warm(n_tiles, make_copy, pf)
            outs = []
            for t in range(n_tiles):
                _wait(make_copy(t), "w_wait")
                outs.append(consume(t, t % S))
                sinkhorn_bg()
                nxt = t + pf + 1
                if nxt < n_tiles:
                    make_copy(nxt).start()
            return outs

        # --- K-major dense stream ------------------------------------------
        # Weights the runner packs K-major (pack_kmajor): [L, n, rows, KM_TILE] where a
        # slot's rows hold rows // K column blocks of [K, KM_TILE] stacked, rows = H
        # (bf16) or 2H (fp8, through the bf16 ring by bitcast).  One contiguous
        # copy fills a slot; each block is a plain x[Q, K] @ W[K, KM_TILE] dot.
        def km_copies(w, idx, fp8):
            wv = w.bitcast(BF16) if fp8 else w
            n = w.shape[1]

            def mk(s_):
                return pltpu.make_async_copy(wv.at[idx, s_], rbf.at[s_ % KM_SLOTS], sbf.at[s_ % KM_SLOTS])

            return n, mk

        def km_block(s_, j, K_, fp8):
            ring = rbf.at[s_ % KM_SLOTS].bitcast(f8) if fp8 else rbf.at[s_ % KM_SLOTS]
            return ring[pl.ds(j * K_, K_), :]

        def warm_km(w, idx, fp8=False):
            n, mk = km_copies(w, idx, fp8)
            _warm(n, mk, KM_PF)

        def mv_km(w, idx, K_, x, fp8=False, warmed=False, scale=None):
            """x [Q, K] @ W [K, N] for a K-major packed W -> [Q, N] f32; `scale`
            [1, N] multiplies the result per output channel (fp8 weights)."""
            n, mk = km_copies(w, idx, fp8)
            k = (2 if fp8 else 1) * H // K_
            xb = x.astype(BF16)

            def use(s_, sl):
                return jnp.concatenate(
                    [jnp.dot(xb, km_block(s_, j, K_, fp8), preferred_element_type=F32)
                     for j in range(k)], axis=1)

            out = jnp.concatenate(_stream(n, mk, use, warmed, KM_PF), axis=1)
            return out if scale is None else out * scale

        def mv_km_allreduce(w, idx, K_, x, fp8=False, warmed=False, scale=None):
            """allreduce(mv_km(...)) with H outputs, each slot's columns sent to the
            peers as soon as they are computed (same pads and sum as `_round`)."""
            n, mk = km_copies(w, idx, fp8)
            k = (2 if fp8 else 1) * H // K_
            xb = x.astype(BF16)
            send, finish = _ar_pipe()
            if not warmed:
                _warm(n, mk, KM_PF)
            for s_ in range(n):
                _wait(mk(s_), "w_wait")
                for j in range(k):
                    c = s_ * k + j
                    part = jnp.dot(xb, km_block(s_, j, K_, fp8), preferred_element_type=F32)
                    if scale is not None:
                        part = part * scale[:, c * KM_TILE : (c + 1) * KM_TILE]
                    put_send(pl.ds(c * KM_TILE, KM_TILE), part)
                sinkhorn_bg()
                if s_ + KM_PF + 1 < n:
                    mk(s_ + KM_PF + 1).start()
                send(s_ * k * KM_TILE, k * KM_TILE)
            return finish(x.shape[0])

        def put_send(cols, v):
            """Columns of this device's partial, into the send buffer of the
            collective format (f32, or bf16 with bf16_collectives)."""
            buf = sendb if BF16_COLL else sendv
            buf[:, cols] = v.astype(buf.dtype)

        def _ar_pipe():
            """An allreduce of `sendv` sent in column groups as they are written.
            Returns (send(c0, width), finish(rows)); same pads, parity and sum as
            `_round`, so the result is identical to allreduce(sendv)."""
            par = collective[0] % 2
            collective[0] += 1
            cps = []

            sbuf, rbuf = (sendb, arb) if BF16_COLL else (sendv, ar)

            def send(c0, width):
                src = sbuf.at[:, pl.ds(c0, width)]
                dst = rbuf.at[par, me, :, pl.ds(c0, width)]
                cps.append(pltpu.make_async_copy(src, dst, sem))
                for d in range(1, tp):
                    cps.append(pltpu.make_async_remote_copy(
                        src_ref=src, dst_ref=dst, send_sem=ssem, recv_sem=rsem,
                        device_id=jax.lax.rem(me + d, tp)))
                for c in cps[-tp:]:
                    c.start()

            def finish(rows):
                with _scope("allreduce"):
                    for c in cps:
                        c.wait()
                    return jnp.sum(rbuf[par].astype(F32), axis=0)[:rows]

            return send, finish

        def load_scale(s_, idx, buf):
            """Block scales live in HBM, which only async_copy may touch."""
            _dma(s_.at[idx], buf, sem2)
            return buf[...]

        ring_in, sem_in = (r8i, s8i) if CH_EXPERTS else (r8a, s8a)

        def mv_fp8_in_h(w, s_, sbuf, idx, out_dim, in_dim, x, warmed=False):
            tile = pick_tile(out_dim, TR)
            rows_per = max(1, tile // blk[0])
            # The block scales are fetched alongside the first weight tiles.
            sc_cp = pltpu.make_async_copy(s_.at[idx], sbuf, sem2)
            sc_cp.start()
            n = out_dim // tile

            def mk(t):
                sl = t % S
                return pltpu.make_async_copy(
                    w.at[idx, pl.ds(t * tile, tile), pl.ds(0, in_dim)],
                    ring_in.at[sl, pl.ds(0, tile), pl.ds(0, in_dim)], sem_in.at[sl])

            def use(t, sl):
                if t == 0:
                    sc_cp.wait()
                return K.matvec_fp8(x, ring_in[sl, :tile, :in_dim],
                                    sbuf[t * rows_per : (t + 1) * rows_per], blk)

            return jnp.concatenate(_stream(n, mk, use, warmed), axis=1)

        def warm_fp8_in_h(w, idx, out_dim, in_dim):
            tile = pick_tile(out_dim, TR)

            def mk(t):
                sl = t % S
                return pltpu.make_async_copy(
                    w.at[idx, pl.ds(t * tile, tile), pl.ds(0, in_dim)],
                    ring_in.at[sl, pl.ds(0, tile), pl.ds(0, in_dim)], sem_in.at[sl])

            _warm(out_dim // tile, mk)

        def fp8_out_h_copies(w, idx, in_dim):
            tile = pick_tile(H, TR)

            def mk(t):
                sl = t % S
                return pltpu.make_async_copy(
                    w.at[idx, pl.ds(t * tile, tile), pl.ds(0, in_dim)],
                    r8b.at[sl, pl.ds(0, tile), pl.ds(0, in_dim)], s8b.at[sl])

            return H // tile, tile, mk

        def warm_fp8_out_h(w, idx, in_dim):
            n, _, mk = fp8_out_h_copies(w, idx, in_dim)
            _warm(n, mk)

        def mv_fp8_out_h(w, s_, sbuf, idx, in_dim, x, warmed=False):
            n, tile, mk = fp8_out_h_copies(w, idx, in_dim)
            rows_per = max(1, tile // blk[0])
            sc_cp = pltpu.make_async_copy(s_.at[idx], sbuf, sem2)
            sc_cp.start()

            def use(t, sl):
                if t == 0:
                    sc_cp.wait()
                return K.matvec_fp8(x, r8b[sl, :tile, :in_dim],
                                    sbuf[t * rows_per : (t + 1) * rows_per], blk)

            return jnp.concatenate(_stream(n, mk, use, warmed), axis=1)

        def mv_fp8_out_h_allreduce(w, s_, sbuf, idx, in_dim, x, warmed=False, group=4):
            """allreduce(mv_fp8_out_h(...)) with the reduction pipelined into the
            matvec, as `mv_slots_allreduce`: exact."""
            n, tile, mk = fp8_out_h_copies(w, idx, in_dim)
            rows_per = max(1, tile // blk[0])
            sc_cp = pltpu.make_async_copy(s_.at[idx], sbuf, sem2)
            sc_cp.start()
            send, finish = _ar_pipe()
            if not warmed:
                _warm(n, mk)
            for t in range(n):
                _wait(mk(t), "w_wait")
                if t == 0:
                    sc_cp.wait()
                put_send(pl.ds(t * tile, tile), K.matvec_fp8(
                    x, r8b[t % S, :tile, :in_dim],
                    sbuf[t * rows_per : (t + 1) * rows_per], blk))
                sinkhorn_bg()
                if t + PF + 1 < n:
                    mk(t + PF + 1).start()
                if (t + 1) % group == 0 or t == n - 1:
                    g0 = (t // group) * group
                    send(g0 * tile, (t + 1 - g0) * tile)
            return finish(x.shape[0])

        # ------------------------------------------------------ embedding
        if cfg.init_streams:
            streams = in_streams[...]                       # [Q*N, H]
        else:
            rows = []
            for qi in range(Q):
                if cfg.embed_2d:
                    # A 2-D table is tiled 8 rows at a time and a DMA offset must
                    # be tile-aligned: fetch the token's tile, pick its row here.
                    t8 = tok_ref[qi]
                    _dma(w_embed.at[pl.ds(pl.multiple_of((t8 // 8) * 8, 8), 8)], embv, sem)
                    r8 = jax.lax.broadcasted_iota(jnp.int32, (8, 1), 0)
                    row = jnp.sum(jnp.where(r8 == t8 % 8, embv[...].astype(F32), 0.0),
                                  axis=0, keepdims=True)
                else:
                    _dma(w_embed.at[tok_ref[qi]], embv, sem)
                    row = embv[...].astype(F32)
                emb = allgather(row, hsh)
                rows.append(jnp.broadcast_to(emb, (N, H)).astype(BF16))
            streams = jnp.concatenate(rows, axis=0)         # [Q*N, H]

        # Deferred Sinkhorn chains: each is [matrix, pairs left].  The serial
        # chain of tiny reductions runs between a stream's tiles, where the core
        # would otherwise wait on DMA; combine() drains whatever is left.
        pending_sinkhorn = []

        def sinkhorn_bg(n=1):
            for _ in range(n):
                for e in pending_sinkhorn:
                    if e[1] > 0:
                        e[0] = K.sinkhorn_step(e[0], spec.hc_eps)
                        e[1] -= 1
                        break

        def sinkhorn_done(e):
            while e[1] > 0:
                e[0] = K.sinkhorn_step(e[0], spec.hc_eps)
                e[1] -= 1
            pending_sinkhorn[:] = [p for p in pending_sinkhorn if p is not e]
            return e[0]

        def mhc(streams, tag, kind):
            """mHC is per token; its coefficient projection runs once for all Q
            tokens (hyper_connection_q), the small per-token math after it Q times."""
            iters = spec.hc_sinkhorn_iters
            res = K.hyper_connection_q(
                [streams[qi * N : (qi + 1) * N] for qi in range(Q)],
                slab.get(sm, f"hc_{tag}_fn"),
                slab.getf(smf, kind, f"hc_{tag}_base"),
                slab.getf(smf, kind, f"hc_{tag}_scale"),
                hc_mult=N, hc_eps=spec.hc_eps, iters=iters, rms_eps=eps,
                defer_sinkhorn=True)
            posts = [post for post, _, _ in res]
            cols = [col for _, _, col in res]
            # The Sinkhorn pairs run in the block's DMA waits (sinkhorn_bg), all
            # queries' matrices as one [Q, N, N] chain: the chain is serial, so
            # one per query would cost Q times its latency.
            entry = [jnp.stack([comb for _, comb, _ in res]) if Q > 1 else res[0][1],
                     iters - 1]
            pending_sinkhorn.append(entry)
            return posts, entry, jnp.concatenate(cols, axis=0)   # [Q, H]

        def combine(posts, combs, block_out, residual):
            if isinstance(combs, list) and len(combs) == 2 and isinstance(combs[1], int):
                done = sinkhorn_done(combs)       # one pending entry for all queries
                combs = [done[qi] for qi in range(Q)] if Q > 1 else [done]
            outs = []
            for qi in range(Q):
                comb = combs[qi] if not isinstance(combs[qi], list) else sinkhorn_done(combs[qi])
                outs.append(K.hyper_combine_vpu(
                    posts[qi], comb, block_out[qi : qi + 1],
                    residual[qi * N : (qi + 1) * N]))
            return jnp.concatenate(outs, axis=0)

        # --------------------------------------------------------- KDA
        def kda(x, k_i, kind):
            # The state and the small weights are in flight while qkv streams.
            loads = [pltpu.make_async_copy(src, dst, sgrp.at[i]) for i, (src, dst) in enumerate([
                (_crec_in.at[k_i] if Q == 1 else _crec_in.at[k_i, ck], st),
                (_cconv_in.at[k_i] if Q == 1 else _cconv_in.at[k_i, ck], cv),
                (w_kfg.at[k_i], fgb), (w_kdt.at[k_i], dtv),
                (w_kconv.at[k_i], cwv), (w_kb.at[k_i], bpv),
                (w_kalog.at[k_i], alv), (w_ksc.at[k_i], ksc),
            ])]
            for c in loads:
                c.start()

            # Projections see all Q tokens at once; the weight is streamed once.
            with _scope("kda_qkv"):
                qkv = mv_km(w_kqkv, k_i, H, x, FP8_KDA, warmed=True)    # [Q, 3*qsh]
            if FP8_KDA:
                # The scales arrive with the small loads: wait for them first.
                _wait(loads[-1], "ld_wait")
                qkv = qkv * ksc[:, : 3 * qsh]
            # The output projection's first tiles land during the recurrence.
            warm_km(w_ko, k_i, FP8_KDA)
            for c in loads[:-1] if FP8_KDA else loads:
                _wait(c, "ld_wait")
            f_low = K.matvec_bf16(x, slab.get(sm, "f_a_proj").astype(BF16))
            g_low = K.matvec_bf16(x, slab.get(sm, "g_a_proj").astype(BF16))
            f = K.matvec_bf16(f_low, fgb[:qsh])
            gg_all = K.matvec_bf16(g_low, fgb[qsh:])
            beta_all = jax.nn.sigmoid(K.matvec_bf16(x, bpv[...]))

            graw_all = f + dtv[...]
            decay = jnp.exp(alv[...]).reshape(hkda, 1)
            lb = spec.linear_lower_bound

            # The recurrence and the convolution are sequential in the token
            # index: there is no way to run query 1 without query 0's state.
            # With Q > 1 each query's state goes to checkpoint qi from a two-slot
            # ring (the small loads are all waited, so sgrp 0..3 are free).
            ys, ck_out = [], [[], []]
            for qi in range(Q):
                slot, prev = qi % 2, (qi - 1) % 2
                if Q > 1:
                    for c in ck_out[slot]:
                        c.wait()
                conv_out, new_conv = K.conv_step(
                    cv[...] if qi == 0 else cvq[prev], qkv[qi : qi + 1],
                    cwv[...].astype(F32))
                if Q == 1:
                    cv[...] = new_conv
                else:
                    cvq[slot] = new_conv
                qv = conv_out[0, :qsh].reshape(hkda, dh)
                kv_ = conv_out[0, qsh : 2 * qsh].reshape(hkda, dh)
                vv = conv_out[0, 2 * qsh :].reshape(hkda, dh)

                graw = graw_all[qi].reshape(hkda, dh)
                glog = (lb * jax.nn.sigmoid(decay * graw)) if lb is not None else (
                    -decay * jnp.where(graw > 20.0, graw,
                                       jnp.log1p(jnp.exp(jnp.minimum(graw, 20.0)))))
                new_state, out = K.kda_step(
                    st[...] if qi == 0 else stq[prev], qv, kv_, vv, glog,
                    beta_all[qi])
                if Q == 1:
                    st[...] = new_state
                else:
                    stq[slot] = new_state
                    ck_out[slot] = [
                        pltpu.make_async_copy(stq.at[slot], c_recur.at[k_i, qi],
                                              sgrp.at[2 * slot]),
                        pltpu.make_async_copy(cvq.at[slot], c_conv.at[k_i, qi],
                                              sgrp.at[2 * slot + 1])]
                    for c in ck_out[slot]:
                        c.start()

                y = out * jax.lax.rsqrt(
                    jnp.mean(jnp.square(out), axis=-1, keepdims=True) + eps)
                y = y * slab.get(sm, "o_norm").reshape(1, dh).astype(F32)
                y = y * jax.nn.sigmoid(gg_all[qi].reshape(hkda, dh))
                ys.append(y.reshape(1, qsh))

            # The state write-back overlaps the output projection.
            if Q == 1:
                saves = [pltpu.make_async_copy(st, c_recur.at[k_i], sgrp.at[0]),
                         pltpu.make_async_copy(cv, c_conv.at[k_i], sgrp.at[1])]
                for c in saves:
                    c.start()
            else:
                saves = ck_out[0] + ck_out[1]     # the last two checkpoints
            with _scope("kda_o"):
                out = mv_km_allreduce(w_ko, k_i, qsh, jnp.concatenate(ys, axis=0),
                                      FP8_KDA, warmed=True,
                                      scale=ksc[:, 3 * qsh :] if FP8_KDA else None)
            for c in saves:
                c.wait()
            return out

        # --------------------------------------------------------- DSA
        TK = cfg.attn_tile
        PT = TK // kp                      # pools per tile
        NT = Lcap // TK
        if Q + kp + 15 > 32:
            raise ValueError(f"n_query={Q}: the pools closed this step span > 32 rows")

        def closing_pools(p0):
            return _closing_pools(p0, Q, kp)

        def dsa(x, d_i):
            kvb_cp = pltpu.make_async_copy(w_kvb.at[d_i], kvb, sgrp.at[3])
            kvb_cp.start()
            # The layer's pooled indexer keys, whole, land during the projections.
            pka_cp = pltpu.make_async_copy(c_ip.at[d_i], pka, sgrp.at[7])
            pka_cp.start()
            # The output projection's ring is idle until the end of the block.
            warm_fp8_out_h(w_do, d_i, nh * vd)
            sc_proj = _scope("dsa_proj")
            sc_proj.__enter__()
            q_resid = K.rms_norm(
                mv_fp8_in_h(w_qa, w_qas, s_qa, d_i, spec.q_lora_rank, H, x, warmed=True),
                slab.get(sm, "q_a_layernorm").reshape(1, -1), eps, BF16)
            q_all = mv_fp8_in_h(w_qb, w_qbs, s_qb, d_i, nh * qk,
                                spec.q_lora_rank, q_resid)        # [Q, nh*qk]
            kv_new = K.rms_norm(
                mv_fp8_in_h(w_kva, w_kvas, s_kva, d_i, kvl, H, x),
                slab.get(sm, "kv_a_layernorm").reshape(1, -1), eps, BF16)
            ik_new = K.layer_norm(
                K.matvec_bf16(x, slab.get(sm, "indexer_wk").astype(BF16)),
                slab.get(sm, "indexer_k_norm_w").reshape(1, -1),
                slab.get(sm, "indexer_k_norm_b").reshape(1, -1), 1e-6)
            ig_new = K.matvec_bf16(
                x, slab.get(sm, "indexer_compress_gate").astype(BF16))
            sc_proj.__exit__(None, None, None)

            # This step's rows: written to the cache after the kernel (row_write_call),
            # and patched into the streamed tiles below so this step sees them.
            kv_new = kv_new.astype(BF16)
            ik_new = ik_new.astype(BF16)
            ig_new = ig_new.astype(BF16)
            o_nkv[d_i] = kv_new
            o_nik[d_i] = ik_new
            o_nig[d_i] = ig_new
            tok_col = jax.lax.broadcasted_iota(jnp.int32, (TK, 1), 0)

            def patch(tile, t, new):
                for qi in range(Q):
                    tile = jnp.where(tok_col + t * TK == pos + qi, new[qi], tile)
                return tile

            last = pos + Q - 1
            nt = last // TK + 1                       # tiles holding any visible token
            with _scope("dsa_indexq"):
                iq_all = mv_km(w_iwq, d_i, spec.q_lora_rank, q_resid)
            hw_all = K.matvec_bf16(
                x, slab.get(sm, "indexer_weights_proj").astype(BF16)) * (hi ** -0.5)
            ape = slab.get(sm, "indexer_ape").astype(F32)

            # Pools this step closes (their last token is one of the new rows):
            # their keys come from the raw rows around pos; every older closed
            # pool's key is read from the pooled-key cache c_ip.
            fp0, fresh = closing_pools(pos)
            r0 = jnp.minimum(((jnp.maximum(pos - kp + 1, 0)) // 16) * 16, Lcap - 32)
            raw = [pltpu.make_async_copy(c.at[d_i, pl.ds(r0, 32)], buf, sgrp.at[i])
                   for i, (c, buf) in enumerate(((c_ik, ikb), (c_ig, igb)))]
            for c in raw:
                c.start()
            for c in raw:
                _wait(c, "kv_wait")
            rr = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) + r0
            ikv, igv = ikb[...], igb[...]
            for qi in range(Q):
                ikv = jnp.where(rr == pos + qi, ik_new[qi], ikv)
                igv = jnp.where(rr == pos + qi, ig_new[qi], igv)
            pk_blk, _ = K.indexer_pool_keys(ikv, igv, ape, last, kp)   # [32/kp, idh]
            prow = jax.lax.broadcasted_iota(jnp.int32, (32 // kp, 1), 0) + r0 // kp
            fresh_keys = [jnp.sum(jnp.where(prow == fp0 + i, pk_blk, 0.0), axis=0,
                                  keepdims=True) for i in range(Q)]
            o_npk[d_i] = jnp.concatenate(fresh_keys, axis=0)
            pcol = jax.lax.broadcasted_iota(jnp.int32, (PT, 1), 0)

            # Pass 1: every pool's score, from the VMEM-resident pooled keys.
            _wait(pka_cp, "kv_wait")

            if Q > 1:
                # All queries' indexer heads in one operand [Q*hi, idh], and their
                # head weights block-diagonal [Q, Q*hi]: two dots per tile serve
                # every query (each tile's cost is the pooled keys, not the rows).
                iq_cat = jnp.concatenate(
                    [iq_all[qi].reshape(hi, idh) for qi in range(Q)], axis=0)
                hcol = jax.lax.broadcasted_iota(jnp.int32, (Q, Q * hi), 1)
                hrow = jax.lax.broadcasted_iota(jnp.int32, (Q, Q * hi), 0)
                hw_blk = jnp.where(hcol // hi == hrow,
                                   jnp.tile(hw_all, (1, Q)), 0.0).astype(F32)

            def score_tile(t, c):
                pk = pka[pl.ds(pl.multiple_of(t * PT, PT), PT), :]
                for i in range(Q):
                    pk = jnp.where((pcol + t * PT == fp0 + i) & (i < fresh), fresh_keys[i], pk)
                if Q > 1:
                    sc = K.indexer_scores(iq_cat, hw_blk, pk, idh ** -0.5)   # [Q, PT]
                    scr[:, pl.ds(t, 1), :] = sc.reshape(Q, 1, PT)
                    return c
                for qi in range(Q):
                    scr[qi, pl.ds(t, 1), :] = K.indexer_scores(
                        iq_all[qi].reshape(hi, idh), hw_all[qi], pk, idh ** -0.5)
                return c

            with _scope("dsa_scores"):
                jax.lax.fori_loop(0, nt, score_tile, 0)

            # Selection: each query sees closed pools ending at or before it.
            gidx = (jax.lax.broadcasted_iota(jnp.int32, (NT, PT), 0) * PT
                    + jax.lax.broadcasted_iota(jnp.int32, (NT, PT), 1))
            pool_end = (gidx + 1) * kp - 1
            # One bisection for all queries: its cost is the latency of a serial
            # chain of reductions, which batching the queries does not lengthen.
            eligs = [pool_end <= pos + qi for qi in range(Q)]   # unwritten tiles never
            if Q == 1:
                chosen = K.select_top_k_mask_2d(
                    jnp.where(eligs[0], scr[0], -jnp.inf), selk)[None]
            else:
                chosen = K.select_top_k_mask_3d(
                    jnp.where(jnp.stack(eligs), scr[...], -jnp.inf), selk)
            scr[...] = chosen.astype(F32)         # the scores are not needed again

            with _scope("dsa_absorb"):
                _wait(kvb_cp, "w_wait")
                kvbf = kvb[...].astype(F32)
                q_abs = [K.absorb_q(q_all[qi].reshape(nh, qk), kvbf, n_heads=nh,
                                    qk_dim=qk, v_dim=vd, kv_lora=kvl) for qi in range(Q)]

            # Pass 2: attention over the selected tokens, one latent tile at a
            # time with an online softmax.
            def kv_cp(t, b):
                return pltpu.make_async_copy(
                    c_kv.at[d_i, pl.ds(t * TK, TK)], kvt.at[b], sgrp.at[4 + b])

            kv_cp(0, 0).start()
            scaling = qk ** -0.5

            # The query and the probabilities enter their dots as bf16 against the
            # bf16 latent tiles (single-pass dots, f32 accumulation).
            def dot(a, b, dims):
                return jax.lax.dot_general(a.astype(BF16), b, (dims, ((), ())),
                                           preferred_element_type=F32)

            # All queries' heads in one operand [Q*nh, kvl]: one logits dot and one
            # probability-times-latent dot per tile serve every query (the MXU
            # cost is the KV tile, not the rows).
            q_all = jnp.concatenate(q_abs, axis=0) if Q > 1 else q_abs[0]
            q_bf = q_all.astype(BF16)

            def attend_tile(t, carry):
                b = jax.lax.rem(t, 2)
                _wait(kv_cp(t, b), "kv_wait")

                @pl.when(t + 1 < nt)
                def _():
                    kv_cp(t + 1, 1 - b).start()

                # This step's rows go into the one 32-row block of the tile that
                # holds them (a whole-tile select per tile costs more than the dots).
                @pl.when((t * TK <= last) & (pos < (t + 1) * TK))
                def _():
                    ws = pl.multiple_of(
                        jnp.clip((pos // 16) * 16 - t * TK, 0, TK - 32), 16)
                    rows = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) + t * TK + ws
                    blk = kvt[b, pl.ds(ws, 32), :]
                    for qi in range(Q):
                        blk = jnp.where(rows == pos + qi, kv_new[qi], blk)
                    kvt[b, pl.ds(ws, 32), :] = blk

                kvt_b = kvt[b]                                    # [TK, kvl] bf16
                tok = t * TK + jax.lax.broadcasted_iota(jnp.int32, (1, TK), 1)
                masks = []
                for qi in range(Q):
                    p_q = pos + qi
                    # Repeating the boolean directly lowers to an i1 transpose;
                    # go through float32 and compare.
                    allowed = jnp.repeat(scr[qi, pl.ds(t, 1), :], kp, axis=1) > 0.5
                    if spec.index_kpool_always_select_tail:
                        tail_count = jax.lax.rem(p_q + 1, kp)
                        tail_lo = p_q + 1 - tail_count
                        allowed = allowed | ((tok >= tail_lo) & (tail_count > 0))
                    allowed = allowed & (tok <= p_q)
                    masks.append(jnp.broadcast_to(allowed, (nh, TK)))
                allowed = jnp.concatenate(masks, axis=0) if Q > 1 else masks[0]
                m_o, l_o, a_o = carry
                lg = dot(q_bf, kvt_b, ((1,), (1,))) * scaling
                lg = jnp.where(allowed, lg, -jnp.inf)                # [Q*nh, TK]
                m_n = jnp.maximum(m_o, jnp.max(lg, axis=1, keepdims=True))
                shift = jnp.where(jnp.isfinite(m_n), m_n, 0.0)
                pr = jnp.where(allowed, jnp.exp(lg - shift), 0.0)
                alpha = jnp.exp(m_o - shift)
                l_n = alpha * l_o + jnp.sum(pr, axis=1, keepdims=True)
                a_n = alpha * a_o + dot(pr, kvt_b, ((1,), (0,)))
                return m_n, l_n, a_n

            init = (jnp.full((Q * nh, 1), -jnp.inf, F32), jnp.zeros((Q * nh, 1), F32),
                    jnp.zeros((Q * nh, kvl), F32))
            with _scope("dsa_attend"):
                _, l_all, a_all = jax.lax.fori_loop(0, nt, attend_tile, init)
            fin = [(None, l_all[qi * nh:(qi + 1) * nh], a_all[qi * nh:(qi + 1) * nh])
                   for qi in range(Q)]

            outs = []
            with _scope("dsa_unabsorb"):
                for qi in range(Q):
                    _, l_f, a_f = fin[qi]
                    lat = a_f / (l_f + 1e-30)
                    head_out = K.project_out(lat, kvbf, n_heads=nh, qk_dim=qk, v_dim=vd)
                    outs.append(head_out.reshape(1, nh * vd))
            with _scope("dsa_o"):
                return mv_fp8_out_h_allreduce(w_do, w_dos, s_do, d_i, nh * vd,
                                              jnp.concatenate(outs, axis=0), warmed=True)

        # --------------------------------------------------------- MLP
        def moe_early(m_i):
            """Router, every expert's block scales and the shared expert's scales,
            started before the FFN-side mHC; only the router is needed to route."""
            fetch = [(w_router.at[m_i], rtr)]
            if not CH_EXPERTS:
                fetch += [(w_egus.at[m_i], egs), (w_edns.at[m_i], eds),
                          (w_sgus.at[m_i], s_sgu), (w_sdns.at[m_i], s_sdn)]
            early = [pltpu.make_async_copy(src, dst, sasy.at[i])
                     for i, (src, dst) in enumerate(fetch)]
            for c in early:
                c.start()
            # The shared expert does not depend on routing: its weights take the
            # ring's first slot now, so they are in flight across the FFN-side mHC
            # and the router instead of queued behind the routed experts.
            for c in shared_copies(m_i):
                c.start()
            return early

        def expert_copies(sl, gu, dn, gu_s=None, dn_s=None):
            """The copies filling expert ring slot `sl`: gate/up and down slabs, and
            with CH_EXPERTS their scale rows."""
            if CH_EXPERTS:
                return [pltpu.make_async_copy(gu, r8a.at[sl], s8a.at[sl]),
                        pltpu.make_async_copy(dn, r8d.at[sl], s8d.at[sl]),
                        pltpu.make_async_copy(gu_s, egsr.at[sl], s8s.at[0, sl]),
                        pltpu.make_async_copy(dn_s, edsr.at[sl], s8s.at[1, sl])]
            return [pltpu.make_async_copy(gu, r8a.at[sl, pl.ds(0, 2 * moesh), pl.ds(0, H)],
                                          s8a.at[sl]),
                    pltpu.make_async_copy(dn, r8d.at[sl, pl.ds(0, H), pl.ds(0, moesh)],
                                          s8d.at[sl])]

        def shared_copies(m_i):
            if CH_EXPERTS:
                return expert_copies(0, w_sgu.at[m_i], w_sdn.at[m_i],
                                     w_sgus.at[pl.ds(m_i, 1)], w_sdns.at[pl.ds(m_i, 1)])
            return expert_copies(0, w_sgu.at[m_i], w_sdn.at[m_i])

        def moe(x, m_i, kind, early):
            with _scope("router_wait"):
                _wait(early[0], "e_wait")
            route_kw = dict(top_k=kexp, norm_topk_prob=spec.norm_topk_prob,
                            routed_scaling_factor=spec.routed_scaling_factor)
            sc_route = _scope("moe_route")
            sc_route.__enter__()
            if Q == 1 and kexp == 8:
                mask, wts, idxs = K.route_experts_ids(
                    x, rtr[...], slab.getf(smf, kind, "router_bias"), **route_kw)
            else:
                mask, wts = K.route_experts(
                    x, rtr[...], slab.getf(smf, kind, "router_bias"), **route_kw)
                idxs = None
            # With Q tokens the selected sets differ.  Fetch the union once and
            # apply each expert to whichever tokens chose it, weighted by that
            # token's own routing weight (zero where it did not).
            n_slot = kexp * Q
            union = mask if Q == 1 else jnp.max(mask.astype(jnp.int32), axis=0,
                                                keepdims=True) > 0
            if idxs is None:
                idxs = K.mask_to_indices_mxu(union, n_slot)
            base = m_i * nexp
            acc = jnp.zeros((Q, H), F32)
            # An unused union slot carries the sentinel `nexp`; clamp it for the
            # fetch (the bytes are harmless) and let the weight mask below give it
            # zero weight, since no token selected it.
            safe = jnp.minimum(idxs, nexp - 1)
            sc_route.__exit__(None, None, None)

            # The shared expert and then the routed experts stream through one
            # ring; the shared expert's copy was started by moe_early.
            entries = [None] + list(range(n_slot))

            def src_gu(e):
                return w_sgu.at[m_i] if e is None else w_egu.at[base + safe[e]]

            def src_dn(e):
                return w_sdn.at[m_i] if e is None else w_edn.at[base + safe[e]]

            def slot_copies(k):
                e = entries[k]
                if not CH_EXPERTS:
                    return expert_copies(k % S, src_gu(e), src_dn(e))
                if e is None:
                    return shared_copies(m_i)
                row = base + safe[e]
                return expert_copies(k % S, w_egu.at[row], w_edn.at[row],
                                     w_egus.at[pl.ds(row, 1)], w_edns.at[pl.ds(row, 1)])

            n_e = len(entries)
            # With several queries the union's ids are ascending with unused slots
            # last: only the n_u distinct experts are fetched and applied.  One
            # query always uses all of its slots, so that path is unguarded.
            guard = Q > 1
            n_u = jnp.sum(union.astype(jnp.int32)) if guard else None

            def when(k, fn):
                if not guard or entries[k] is None:
                    fn()
                else:
                    pl.when(entries[k] < n_u)(fn)

            def start(k):
                def go():
                    for c in slot_copies(k):
                        c.start()
                when(k, go)

            for k in range(1, min(PF + 1, n_e)):
                start(k)
            if n_e > 1 and not CH_EXPERTS:
                early[1].wait()
                early[2].wait()
            if guard:
                macc[...] = jnp.zeros(macc.shape, F32)
            for k in range(n_e):
                sl = k % S
                e = entries[k]

                def body(k=k, sl=sl, e=e):
                    for c in slot_copies(k):
                        _wait(c, "e_wait")
                    if CH_EXPERTS:
                        w_j = None
                        if e is not None:
                            hit = jax.lax.broadcasted_iota(jnp.int32, (Q, nexp), 1) == idxs[e]
                            w_j = jnp.sum(jnp.where(hit, wts, 0.0), axis=1, keepdims=True)
                        h = jnp.dot(x.astype(BF16), r8a[sl], preferred_element_type=F32) * egsr[sl]
                        act = K.swiglu_clamped(h[:, :moesh], h[:, moesh:], spec.swiglu_limit)
                        y = jnp.dot(act.astype(BF16), r8d[sl], preferred_element_type=F32) * edsr[sl]
                        return y if w_j is None else w_j * y
                    if e is None:
                        early[3].wait()
                        early[4].wait()
                        gs, dsc, w_j = s_sgu[...], s_sdn[...], None
                    else:
                        gs, dsc = egs[safe[e]], eds[safe[e]]
                        # Per-token weight for this expert, zero where the token did
                        # not select it; a masked sum, since a dynamic lane extract
                        # does not lower.
                        hit = jax.lax.broadcasted_iota(jnp.int32, (Q, nexp), 1) == idxs[e]
                        w_j = jnp.sum(jnp.where(hit, wts, 0.0), axis=1, keepdims=True)
                    h = K.matvec_fp8(x, r8a[sl, : 2 * moesh, :H], gs, blk)
                    act = K.swiglu_clamped(h[:, :moesh], h[:, moesh:], spec.swiglu_limit)
                    y = K.matvec_fp8(act, r8d[sl, :H, :moesh], dsc, blk)
                    return y if w_j is None else w_j * y

                if guard:
                    def guarded(body=body):
                        macc[...] += body()
                    when(k, guarded)
                else:
                    acc = acc + body()
                sinkhorn_bg(2)
                nxt = k + PF + 1
                if nxt < n_e:
                    start(nxt)
            return allreduce(macc[...] if guard else acc)

        def dense(x, d_i):
            h = mv_fp8_in_h(w_dgu, w_dgus, s_dgu, d_i, 2 * ffnsh, H, x)
            act = K.swiglu_clamped(h[:, :ffnsh], h[:, ffnsh:], spec.swiglu_limit)
            return mv_fp8_out_h_allreduce(w_ddn, w_ddns, s_ddn, d_i, ffnsh, act)

        def layer(streams, L, kind, mlp_kind, next_kda=None, prewarmed=False, par=None):
            """`next_kda`: the next layer's index when it is a KDA layer whose qkv
            may start during this layer's FFN block (the bf16 ring is idle then);
            `prewarmed`: this layer's qkv was started that way."""
            slab_kind = f"{kind}_{'sparse' if mlp_kind == 'moe' else 'dense'}"
            # This layer's slab was started by the previous layer (or here, for the
            # first); the next layer's goes into the other half now.
            par = L % 2 if par is None else par
            sm.i = smf.i = par
            with _scope("slab"):
                if isinstance(L, int) and L == 0:
                    for c in slab_copies(L, par):
                        c.start()
                for c in slab_copies(L, par):
                    c.wait()

                def prefetch():
                    for c in slab_copies(L + 1, 1 - par):
                        c.start()

                if not isinstance(L, int):
                    pl.when(L + 1 < n_layers)(prefetch)
                elif L + 1 < n_layers:
                    prefetch()
            # The block's first weight tiles stream while mHC runs: the weights do
            # not depend on it, and mHC is a serial chain of small reductions.
            if kind == "kda":
                if not prewarmed:
                    warm_km(w_kqkv, kda_ord(L), FP8_KDA)
            else:
                warm_fp8_in_h(w_qa, dsa_ord(L), spec.q_lora_rank, H)
            with _scope("mhc_attn"):
                posts, combs, x = mhc(streams, "attn", slab_kind)
                x = K.rms_norm(x, slab.get(sm, "input_layernorm").reshape(1, H), eps, BF16)
            with _scope(kind):
                a = (kda(x, kda_ord(L), slab_kind) if kind == "kda"
                     else dsa(x, dsa_ord(L)))
            with _scope("combine"):
                streams = combine(posts, combs, a, streams)

            early = moe_early(moe_ord(L)) if mlp_kind == "moe" else None
            if next_kda is not None:
                warm_km(w_kqkv, kda_ord(next_kda), FP8_KDA)
            with _scope("mhc_ffn"):
                posts, combs, x = mhc(streams, "ffn", slab_kind)
                x = K.rms_norm(
                    x, slab.get(sm, "post_attention_layernorm").reshape(1, H), eps, BF16)
            with _scope(mlp_kind):
                m = (moe(x, moe_ord(L), slab_kind, early) if mlp_kind == "moe"
                     else dense(x, L))
            with _scope("combine"):
                return combine(posts, combs, m, streams)

        o_nkv[...] = jnp.zeros(o_nkv.shape, o_nkv.dtype)
        o_npk[...] = jnp.zeros(o_npk.shape, o_npk.dtype)
        o_nik[...] = jnp.zeros(o_nik.shape, o_nik.dtype)
        o_nig[...] = jnp.zeros(o_nig.shape, o_nig.dtype)
        n_dense = min(spec.first_k_dense_replace, n_layers)
        for L in range(n_dense):
            nxt = L + 1 if (L + 1 < n_dense and spec.layer_types[L + 1] == "linear_attention") else None
            streams = layer(streams, L, "kda", "dense", next_kda=nxt, prewarmed=L > 0)

        def period(p, s_):   # noqa: ARG001 - p is the traced period index
            base = spec.first_k_dense_replace + 4 * p
            f0 = spec.first_k_dense_replace
            s_ = layer(s_, base, "dsa", "moe", next_kda=base + 1, par=f0 % 2)
            s_ = layer(s_, base + 1, "kda", "moe", next_kda=base + 2, prewarmed=True,
                       par=(f0 + 1) % 2)
            s_ = layer(s_, base + 2, "kda", "moe", next_kda=base + 3, prewarmed=True,
                       par=(f0 + 2) % 2)
            s_ = layer(s_, base + 3, "kda", "moe", prewarmed=True, par=(f0 + 3) % 2)
            return s_

        full_periods = min(cfg.n_periods,
                           max(0, (n_layers - spec.first_k_dense_replace) // 4))
        streams = jax.lax.fori_loop(0, full_periods, period, streams)

        for L in range(spec.first_k_dense_replace + 4 * full_periods, n_layers):
            kind = "kda" if spec.layer_types[L] == "linear_attention" else "dsa"
            streams = layer(streams, L, kind, "moe")

        # The residual stream is also emitted, so a disagreement can be localised
        # to a layer instead of being inferred from the logits.  It costs one
        # [N, H] write per step.
        o_streams[...] = streams
        _dma(w_fnorm, fnv, sem)
        hidden = K.rms_norm(
            jnp.concatenate([
                jnp.mean(streams[qi * N : (qi + 1) * N].astype(F32), axis=0,
                         keepdims=True) for qi in range(Q)], axis=0),
            fnv[...], eps, BF16)
        # The MTP draft reads this hidden (serving captures the same tensor).
        o_hidden[...] = hidden
        o_logits[...] = mv_km(w_lm, 0, H, hidden)

    return kernel


def _closing_pools(pos, Q, kp):
    """(first pool, count) of the pools whose last token is one of pos .. pos+Q-1."""
    first = (pos + kp) // kp - 1          # the pool containing pos
    first = jnp.where((first + 1) * kp - 1 >= pos, first, first + 1)
    last = (pos + Q) // kp - 1            # the last pool that ends by pos+Q-1
    return first, jnp.maximum(last - first + 1, 0)


def init_pool_keys(spec, ik, ig, apes):
    """The pooled-key cache [n_dsa, L/kpool, Dh] f32 from raw indexer caches
    [n_dsa, L, Dh]: same formula as the kernel (phases.indexer_pool_keys)."""
    kp = spec.index_kpool
    return jnp.stack([K.indexer_pool_keys(ik[d], ig[d], apes[d], ik.shape[1], kp)[0]
                      for d in range(ik.shape[0])])


def row_write_call(spec, cfg: KernelConfig):
    """``f(pos, nkv, nik, nig, npk, kv, idx_key, idx_gate, idx_pool) -> (kv, idx_key,
    idx_gate, idx_pool)``.

    Stores the decode step's new rows, and the keys of the pools it closed, in
    place.  It has no remote DMA, so XLA aliases the caches without a copy.  A
    dynamic single-row write into a tiled HBM array is not legal, so each layer's
    aligned block around the rows (32 bf16 rows, 8 f32 pool rows) is read,
    patched and written back."""
    Q, Lcap, kp = cfg.n_query, cfg.max_len, spec.index_kpool
    kvl, idh = spec.kv_lora_rank, spec.index_head_dim
    n_dsa = len(spec.dsa_layers)
    n_pool = Lcap // kp
    if Q > 16:
        raise ValueError(f"n_query={Q}: the 32-row block holds at most 16 new rows")

    def kernel(pos_ref, nkv, nik, nig, npk, _kv_in, _ik_in, _ig_in, _ip_in,
               kv, ik, ig, ip, bkv, bik, big, bip, sems):
        pos = pos_ref[0]
        r0 = jnp.minimum((pos // 16) * 16, Lcap - 32)
        fp0, fresh = _closing_pools(pos, Q, kp)
        p0 = jnp.minimum((fp0 // 8) * 8, n_pool - 8)
        blocks = [(kv, bkv, r0, 32), (ik, bik, r0, 32), (ig, big, r0, 32),
                  (ip, bip, p0, 8)]

        def copies(to_cache):
            cps = []
            for j, (c, buf, start, n) in enumerate(blocks):
                for d in range(n_dsa):
                    blk = c.at[d, pl.ds(start, n)]
                    src, dst = (buf.at[d], blk) if to_cache else (blk, buf.at[d])
                    cps.append(pltpu.make_async_copy(src, dst, sems.at[j * n_dsa + d]))
            for cp in cps:
                cp.start()
            for cp in cps:
                cp.wait()

        copies(False)
        rr = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) + r0
        for buf, new in ((bkv, nkv), (bik, nik), (big, nig)):
            for d in range(n_dsa):
                v = buf[d]
                for qi in range(Q):
                    v = jnp.where(rr == pos + qi, new[d, pl.ds(qi, 1), :], v)
                buf[d] = v
        pr = jax.lax.broadcasted_iota(jnp.int32, (8, 1), 0) + p0
        for d in range(n_dsa):
            v = bip[d]
            for i in range(min(Q, 8)):
                v = jnp.where((pr == fp0 + i) & (i < fresh), npk[d, pl.ds(i, 1), :], v)
            bip[d] = v
        copies(True)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)

    def call(pos, nkv, nik, nig, npk, kv, ik, ig, ip):
        return pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in (kv, ik, ig, ip)),
            in_specs=[smem] + [vmem] * 4 + [anyspace] * 4,
            out_specs=(anyspace,) * 4,
            scratch_shapes=[pltpu.VMEM((n_dsa, 32, kvl), BF16),
                            pltpu.VMEM((n_dsa, 32, idh), BF16),
                            pltpu.VMEM((n_dsa, 32, idh), BF16),
                            pltpu.VMEM((n_dsa, 8, idh), F32),
                            pltpu.SemaphoreType.DMA((4 * n_dsa,))],
            input_output_aliases={5: 0, 6: 1, 7: 2, 8: 3},
        )(pos, nkv, nik, nig, npk, kv, ik, ig, ip)

    return call
