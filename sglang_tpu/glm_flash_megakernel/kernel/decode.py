"""GLM-5.3-Flash megakernel: one Pallas program per device per target step.

A step is `n_req` requests of `n_query` rows each (row r = b * Q + qi). The kernel owns
nothing between steps: it reads serving's paged nope / raw / pooled-record pools and
the committed KDA state rows through the step's page table and row ids, and emits the
step's new rows and KDA checkpoints as outputs for `writeback.row_write_call`, because
XLA copies every operand that a kernel with remote DMA aliases.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import phases as K
from . import pools as PL

F32 = jnp.float32
BF16 = jnp.bfloat16
AXIS = "tp"


# Names in-kernel regions for a device trace (libtpu --xla_enable_custom_call_region_trace=true).
_TRACE = False


def _scope(name):
    return jax.named_scope(name) if _TRACE else contextlib.nullcontext()


def _wait(copy, name):
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
    max_len: int
    n_periods: int = 10
    tile_rows: int = 256
    vmem_mib: int = 110
    slots: int = 4
    prefetch: int = 2
    ring: bool = True
    n_query: int = 1
    n_req: int = 1
    page_size: int = 128
    # Stride of serving's page table (None: max_len / page_size).
    table_pages: int | None = None
    n_layers: int | None = None
    attn_tile: int = 512
    fp8_kda: bool = True
    ch_experts: bool = True
    # Round each rank's partial to bf16 before the sum, as serving's all-reduce does.
    bf16_collectives: bool = True
    km_slots: int = 4
    km_issue: int = 2
    embed_2d: bool = False
    init_streams: bool = False

    @property
    def km_cols(self) -> int:
        return 512 if self.rows == 1 else 256

    @property
    def rows(self) -> int:
        return self.n_req * self.n_query

    @property
    def pages_per_seq(self) -> int:
        if self.table_pages is not None:
            return self.table_pages
        return self.max_len // self.page_size

    def __post_init__(self):
        if self.max_len % self.attn_tile:
            raise ValueError(
                f"max_len={self.max_len} must be a multiple of "
                f"attn_tile={self.attn_tile}"
            )
        if self.attn_tile % self.page_size:
            raise ValueError(
                f"attn_tile={self.attn_tile} must be a multiple of "
                f"page_size={self.page_size}"
            )
        if self.ring and self.prefetch > self.slots - 2:
            raise ValueError(
                f"prefetch={self.prefetch} needs slots >= {self.prefetch + 2}, got "
                f"{self.slots}: issuing into a slot that still holds an unwaited "
                "transfer puts two DMAs on one semaphore, which corrupts results "
                "and halts the core -- and only once the tile count exceeds the "
                "ring, so a small shape will not reproduce it"
            )

    def layers(self, spec) -> int:
        """How many decoder layers this configuration actually runs."""
        n = min(
            spec.num_hidden_layers, spec.first_k_dense_replace + 4 * self.n_periods + 2
        )
        return n if self.n_layers is None else min(n, self.n_layers)


class SlabIndex:
    """Slab offsets keyed by layer kind, since the rolled loop's layer index is traced."""

    def __init__(self, small_map, small_f32_map, spec):
        self.m = {k: (v[0], v[1], tuple(v[2])) for k, v in small_map.items()}
        by_kind: dict[str, dict] = {}
        for L, m in small_f32_map.items():
            L = int(L)
            kind = self.kind_of(spec, L)
            if kind in by_kind and by_kind[kind] != m:
                raise ValueError(
                    f"layer {L} has different float32 slab offsets from another "
                    f"{kind} layer; the rolled loop assumes they match"
                )
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
        """Rank-2 on purpose: rank-1 slices feed shape casts that do not lower."""
        off, n = self.f[kind][tag]
        return ref[0:1, off : off + n]


class _Half:
    """One half of a two-slot VMEM buffer, chosen by `i`, indexed like a ref."""

    def __init__(self, buf):
        self.buf, self.i = buf, 0

    def __getitem__(self, idx):
        return self.buf.at[self.i][idx]


def dsa_scale_widths(spec, tp):
    """Columns of a DSA layer's scale row: [q_a | kv_a], q_b, o."""
    return (
        spec.q_lora_rank + spec.kv_lora_rank,
        spec.num_attention_heads // tp * spec.qk_head_dim,
        spec.hidden_size,
    )


def dense_scale_widths(spec, tp):
    """Columns of a dense-MLP layer's scale row: gate_up, down."""
    return 2 * spec.intermediate_size // tp, spec.hidden_size


def proj_scale_cols(spec, tp):
    return max(sum(dsa_scale_widths(spec, tp)), sum(dense_scale_widths(spec, tp)))


def km_blocks(K, N, rows, tile_n):
    """Column blocks per K-major slot: the most [K, tile_n] blocks that fit `rows` and divide N."""
    k = rows // K
    while k > 1 and N % (tile_n * k):
        k -= 1
    return k


def pack_kmajor(w, rows, tile_n):
    """[L, N, K] -> K-major ring slots [L, n, rows, tile_n]; a slot is one contiguous copy."""
    L, N, K = w.shape
    k = km_blocks(K, N, rows, tile_n)
    if k < 1 or N % (tile_n * k):
        raise ValueError(f"cannot pack [{N}, {K}] into [{rows}, {tile_n}] slots")
    n = N // (tile_n * k)
    out = (
        w.transpose(0, 2, 1)
        .reshape(L, K, n, k, tile_n)
        .transpose(0, 2, 3, 1, 4)
        .reshape(L, n, k * K, tile_n)
    )
    if k * K < rows:
        out = jnp.pad(out, ((0, 0), (0, 0), (0, rows - k * K), (0, 0)))
    return out


def pack_slots(w, tile_rows, hidden):
    """[L, out, in] -> [L, n, tile_rows, hidden]: k = hidden // in row tiles side by side."""
    L, out, inn = w.shape
    k = hidden // inn
    n = out // (tile_rows * k)
    return (
        w.reshape(L, n, k, tile_rows, inn)
        .transpose(0, 1, 3, 2, 4)
        .reshape(L, n, tile_rows, k * inn)
    )


def _dma(src, dst, sem):
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()


def pick_tile(out_dim: int, want: int) -> int:
    """Largest divisor of `out_dim` that is at most `want` and sublane-aligned."""
    for d in range(min(want, out_dim), 0, -8):
        if out_dim % d == 0:
            return d
    return out_dim


def _flags(cfg: KernelConfig):
    return (
        cfg.fp8_kda,
        cfg.ch_experts,
        cfg.bf16_collectives,
        cfg.km_cols,
        cfg.km_slots,
        cfg.km_issue,
    )


def scratch_shapes(spec, layout, cfg: KernelConfig):
    FP8_KDA, CH_EXPERTS, BF16_COLL, KM_TILE, KM_SLOTS, KM_PF = _flags(cfg)
    tp = layout.tp
    H = spec.hidden_size
    qsh = spec.kda_qkv_dim // tp
    hkda = spec.linear_num_heads // tp
    nh = spec.num_attention_heads // tp
    dh = spec.linear_head_dim
    qk, vd, kvl = spec.qk_head_dim, spec.v_head_dim, spec.kv_lora_rank
    idh, hi, kp = spec.index_head_dim, spec.index_n_heads, spec.index_kpool
    moesh = spec.moe_intermediate_size // tp
    TR = cfg.tile_rows
    L = cfg.max_len
    R, Q = cfg.rows, cfg.n_query
    TK = cfg.attn_tile
    PT = TK // kp
    slab_rows = layout.small_rows()
    k1 = spec.linear_conv_kernel_dim - 1

    f8 = jnp.float8_e4m3fn
    u32 = jnp.uint32
    S = cfg.slots if cfg.ring else 1
    small = (1, 1, 8, 128)
    # Positional: the order must match the kernel's scratch arguments.
    return [
        pltpu.VMEM((2, slab_rows, H), BF16),
        pltpu.VMEM((2, 1, H), F32),
        # Each ring has its own semaphore array: two in-flight DMAs must not share a semaphore.
        pltpu.VMEM((KM_SLOTS, H, KM_TILE), BF16),
        (
            pltpu.VMEM((S, H, 2 * moesh), f8)
            if CH_EXPERTS
            else pltpu.VMEM((S, max(TR, 2 * moesh), H), f8)
        ),
        (
            pltpu.VMEM((S, moesh, H), f8)
            if CH_EXPERTS
            else pltpu.VMEM((S, H, max(moesh, 128)), f8)
        ),
        pltpu.VMEM((S, 1, 2 * moesh) if CH_EXPERTS else (1, 1, 128), F32),
        pltpu.VMEM((S, 1, H) if CH_EXPERTS else (1, 1, 128), F32),
        # A request's KDA state lands in st / cv3; each query's state is then
        # checkpointed from the stq / cvq pair: the delta rule cannot be undone.
        pltpu.VMEM((hkda, dh, dh), F32),
        pltpu.VMEM((2, hkda, dh, dh), F32),
        # Conv state in the pool's channels-last [K-1, 1, C] form.
        pltpu.VMEM((k1, 1, 3 * qsh), F32),
        pltpu.VMEM((2, k1, 1, 3 * qsh), F32),
        # One attention tile of nope word rows, and its pooled records.
        pltpu.VMEM((2, TK, PL.NOPE_BYTES // PL.WORD_BYTES), u32),
        pltpu.VMEM((2, PT // PL.WORD_BYTES, PL.record_bytes(idh)), u32),
        pltpu.VMEM((32, 2 * idh), BF16),
        pltpu.VMEM((Q, L // TK, PT), F32),
        pltpu.VMEM((nh * (qk + vd), kvl), BF16),
        pltpu.VMEM((2 * qsh, dh), BF16),
        pltpu.VMEM((1, qsh), F32),
        pltpu.VMEM((1, 3 * qsh + H), F32),
        pltpu.VMEM((spec.linear_conv_kernel_dim, 1, 3 * qsh), BF16),
        pltpu.VMEM((hkda, H), BF16),
        pltpu.VMEM((1, hkda), F32),
        pltpu.VMEM((spec.n_routed_experts, H), BF16),
        pltpu.VMEM((1, H), BF16),
        pltpu.VMEM(small if BF16_COLL else (2, tp, R, H), F32),
        pltpu.VMEM((8, 128) if BF16_COLL else (R, H), F32),
        pltpu.VMEM((R, H), F32),
        pltpu.VMEM((2, tp, R, H) if BF16_COLL else small, BF16),
        pltpu.VMEM((R, H) if BF16_COLL else (8, 128), BF16),
        pltpu.VMEM((8 if cfg.embed_2d else 1, H // layout.tp), BF16),
        pltpu.VMEM((1, proj_scale_cols(spec, tp)), F32),
        # Exact-size scale buffers: a DMA into a slice of a VMEM buffer must be tile-aligned.
        pltpu.VMEM((max(1, 2 * moesh // 128), H // 128), F32),
        pltpu.VMEM((H // 128, max(1, moesh // 128)), F32),
        pltpu.VMEM((spec.n_routed_experts, 2 * moesh // 128, H // 128), F32),
        pltpu.VMEM((spec.n_routed_experts, H // 128, max(1, moesh // 128)), F32),
        # Per-row values the per-request loops read back by dynamic row.
        pltpu.VMEM((R, kvl), BF16),
        pltpu.VMEM((R, 2 * idh), BF16),
        pltpu.VMEM((R, nh * qk), F32),
        pltpu.VMEM((R * hi, idh), F32),
        pltpu.VMEM((R, hi), F32),
        pltpu.VMEM((R, nh * vd), BF16),
        pltpu.VMEM((R, qsh), F32),
        pltpu.VMEM((R, 3 * qsh), F32),
        pltpu.VMEM((R, 2 * qsh), F32),
        pltpu.VMEM((R, hkda), F32),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((KM_SLOTS,)),
        pltpu.SemaphoreType.DMA((S,)),
        pltpu.SemaphoreType.DMA((S,)),
        pltpu.SemaphoreType.DMA((8,)),
        pltpu.SemaphoreType.DMA((8,)),
        pltpu.SemaphoreType.DMA((8 * S,)),
        pltpu.SemaphoreType.DMA((2, 2)),
        pltpu.SemaphoreType.DMA((2, S)),
        pltpu.SemaphoreType.DMA((2, 2)),
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
    f8 = jnp.float8_e4m3fn
    n_layers = cfg.layers(spec)
    Q, B, R = cfg.n_query, cfg.n_req, cfg.rows
    ps, PPS = cfg.page_size, cfg.pages_per_seq
    n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)
    rec_bytes = PL.record_bytes(idh)
    RPP = ps // kp  # pooled records per page
    if kvl != PL.NOPE_BYTES:
        raise ValueError(f"kv_lora_rank={kvl}: the nope pool holds 512-byte tokens")

    def kernel(
        tok_ref,
        pos_ref,
        meta_ref,
        tab_ref,
        srow_ref,
        ksc_ref,
        in_streams,
        w_embed,
        w_lm,
        w_fnorm,
        w_smalls,
        w_smallsf,
        w_kqkv,
        w_ko,
        w_kfg,
        w_kdt,
        w_kconv,
        w_kb,
        w_ka_log,
        w_qakv,
        w_qb,
        w_kvb,
        w_do,
        w_dsc,
        w_iwq,
        w_router,
        w_egu,
        w_egus,
        w_edn,
        w_edns,
        w_sgu,
        w_sgus,
        w_sdn,
        w_sdns,
        w_dgu,
        w_ddn,
        w_dnsc,
        w_ksc,
        p_nope,
        p_raw,
        p_rec,
        p_tmp,
        p_cnv,
        o_logits,
        o_streams,
        o_hidden,
        o_nkv,
        o_nraw,
        o_npk,
        o_recur,
        o_conv,
        sm2,
        smf2,
        rbf,
        r8a,
        r8d,
        egsr,
        edsr,
        st,
        stq,
        cv3,
        cvq,
        kvw,
        rkw,
        rawb,
        scr,
        kvb,
        fgb,
        dtv,
        ksc,
        cwv,
        bpv,
        alv,
        rtr,
        fnv,
        ar,
        sendv,
        macc,
        arb,
        sendb,
        embv,
        psc,
        s_sgu,
        s_sdn,
        egs,
        eds,
        nkvb,
        nrawb,
        qallb,
        iqb,
        hwb,
        attb,
        ysb,
        qkvb,
        fgb2,
        btb,
        sem,
        sem2,
        ssem,
        rsem,
        sbf,
        s8a,
        s8d,
        sgrp,
        sem_async,
        sbfn,
        sslab,
        s8s,
        sck,
    ):
        me = jax.lax.axis_index(AXIS)
        sm, smf = _Half(sm2), _Half(smf2)

        def slab_copies(L, par):
            return [
                pltpu.make_async_copy(w_smalls.at[L], sm2.at[par], sslab.at[par, 0]),
                pltpu.make_async_copy(w_smallsf.at[L], smf2.at[par], sslab.at[par, 1]),
            ]

        # Each pad slot has one writer per round: never clear the pad locally, as a peer's
        # DMA may land first. Pads alternate on a static parity: a peer is at most one round
        # ahead, and each loop body runs an even number of collectives.
        collective = [0]

        def _round(row_full, width):
            """`row_full` is [R, H]; shorter payloads are zero-padded to the pad's fixed shape."""
            par = collective[0] % 2
            collective[0] += 1
            if row_full.shape != (R, H):
                row_full = jnp.pad(
                    row_full, ((0, R - row_full.shape[0]), (0, H - row_full.shape[1]))
                )
            sbuf, rbuf = (sendb, arb) if BF16_COLL else (sendv, ar)
            sbuf[...] = row_full.astype(sbuf.dtype)
            local = pltpu.make_async_copy(sbuf, rbuf.at[par, me], sem)
            local.start()
            cps = []
            for d in range(1, tp):
                peer = jax.lax.rem(me + d, tp)
                cp = pltpu.make_async_remote_copy(
                    src_ref=sbuf,
                    dst_ref=rbuf.at[par, me],
                    send_sem=ssem,
                    recv_sem=rsem,
                    device_id=peer,
                )
                cp.start()
                cps.append(cp)
            local.wait()
            for cp in cps:
                cp.wait()
            return rbuf[par].astype(F32)

        def allgather(rows, width):
            """[R, width] on each device -> [R, tp*width] in mesh rank order."""
            got = _round(rows, width)
            return jnp.concatenate([got[d, :, :width] for d in range(tp)], axis=1)

        def allreduce(rows):
            """[R, H] partial sums on each device -> [R, H] totals."""
            with _scope("allreduce"):
                return jnp.sum(_round(rows, H), axis=0)[: rows.shape[0]]

        S = cfg.slots if cfg.ring else 1
        PF = cfg.prefetch if cfg.ring else 0

        def _warm(n_tiles, make_copy, pf=None):
            """Start a stream's first transfers early; then run the stream with `warmed=True`."""
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

        def km_copies(w, idx, fp8):
            wv = w.bitcast(BF16) if fp8 else w
            n = w.shape[1]

            def mk(s_):
                return pltpu.make_async_copy(
                    wv.at[idx, s_], rbf.at[s_ % KM_SLOTS], sbf.at[s_ % KM_SLOTS]
                )

            return n, mk

        def km_block(s_, j, K_, fp8):
            ring = rbf.at[s_ % KM_SLOTS].bitcast(f8) if fp8 else rbf.at[s_ % KM_SLOTS]
            return ring[pl.ds(j * K_, K_), :]

        def warm_km(w, idx, fp8=False):
            n, mk = km_copies(w, idx, fp8)
            _warm(n, mk, KM_PF)

        def km_k(K_, fp8, n_out):
            """Blocks per ring slot, as pack_kmajor chose for an N = n_out weight."""
            rows = (2 if fp8 else 1) * H
            return rows // K_ if n_out is None else km_blocks(K_, n_out, rows, KM_TILE)

        def mv_km(w, idx, K_, x, fp8=False, warmed=False, scale=None, n_out=None):
            """x [R, K] @ K-major W [K, N] -> [R, N] f32, times the per-channel `scale` if given."""
            n, mk = km_copies(w, idx, fp8)
            k = km_k(K_, fp8, n_out)
            xb = x.astype(BF16)

            def use(s_, sl):
                return jnp.concatenate(
                    [
                        jnp.dot(
                            xb, km_block(s_, j, K_, fp8), preferred_element_type=F32
                        )
                        for j in range(k)
                    ],
                    axis=1,
                )

            out = jnp.concatenate(_stream(n, mk, use, warmed, KM_PF), axis=1)
            return out if scale is None else out * scale

        def mv_km_allreduce(w, idx, K_, x, fp8=False, warmed=False, scale=None):
            """allreduce(mv_km(...)), each slot's columns sent as soon as they are computed."""
            n, mk = km_copies(w, idx, fp8)
            k = km_k(K_, fp8, H)
            xb = x.astype(BF16)
            send, finish = _ar_pipe()
            if not warmed:
                _warm(n, mk, KM_PF)
            for s_ in range(n):
                _wait(mk(s_), "w_wait")
                for j in range(k):
                    c = s_ * k + j
                    part = jnp.dot(
                        xb, km_block(s_, j, K_, fp8), preferred_element_type=F32
                    )
                    if scale is not None:
                        part = part * scale[:, c * KM_TILE : (c + 1) * KM_TILE]
                    put_send(pl.ds(c * KM_TILE, KM_TILE), part)
                sinkhorn_bg()
                if s_ + KM_PF + 1 < n:
                    mk(s_ + KM_PF + 1).start()
                send(s_ * k * KM_TILE, k * KM_TILE)
            return finish(x.shape[0])

        def put_send(cols, v):
            buf = sendb if BF16_COLL else sendv
            buf[:, cols] = v.astype(buf.dtype)

        def _ar_pipe():
            """Allreduce sent in column groups: (send(c0, width), finish(rows)), same result as `_round`."""
            par = collective[0] % 2
            collective[0] += 1
            cps = []

            sbuf, rbuf = (sendb, arb) if BF16_COLL else (sendv, ar)

            def send(c0, width):
                src = sbuf.at[:, pl.ds(c0, width)]
                dst = rbuf.at[par, me, :, pl.ds(c0, width)]
                cps.append(pltpu.make_async_copy(src, dst, sem))
                for d in range(1, tp):
                    cps.append(
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=ssem,
                            recv_sem=rsem,
                            device_id=jax.lax.rem(me + d, tp),
                        )
                    )
                for c in cps[-tp:]:
                    c.start()

            def finish(rows):
                with _scope("allreduce"):
                    for c in cps:
                        c.wait()
                    return jnp.sum(rbuf[par].astype(F32), axis=0)[:rows]

            return send, finish

        if cfg.init_streams:
            streams = in_streams[...]
        else:
            rows = []
            for r in range(R):
                if cfg.embed_2d:
                    # DMA offsets must be tile-aligned: fetch the token's 8-row tile, pick the row.
                    t8 = tok_ref[r]
                    _dma(
                        w_embed.at[pl.ds(pl.multiple_of((t8 // 8) * 8, 8), 8)],
                        embv,
                        sem,
                    )
                    r8 = jax.lax.broadcasted_iota(jnp.int32, (8, 1), 0)
                    rows.append(
                        jnp.sum(
                            jnp.where(r8 == t8 % 8, embv[...].astype(F32), 0.0),
                            axis=0,
                            keepdims=True,
                        )
                    )
                else:
                    _dma(w_embed.at[tok_ref[r]], embv, sem)
                    rows.append(embv[...].astype(F32))
            emb = allgather(jnp.concatenate(rows, axis=0), hsh)
            streams = jnp.concatenate(
                [jnp.broadcast_to(emb[r : r + 1], (N, H)) for r in range(R)], axis=0
            ).astype(BF16)

        # Deferred Sinkhorn chains [matrix, pairs left]: stepped in DMA waits, drained in combine().
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
            iters = spec.hc_sinkhorn_iters
            res = K.hyper_connection_q(
                [streams[r * N : (r + 1) * N] for r in range(R)],
                slab.get(sm, f"hc_{tag}_fn"),
                slab.getf(smf, kind, f"hc_{tag}_base"),
                slab.getf(smf, kind, f"hc_{tag}_scale"),
                hc_mult=N,
                hc_eps=spec.hc_eps,
                iters=iters,
                rms_eps=eps,
                defer_sinkhorn=True,
            )
            posts = [post for post, _, _ in res]
            cols = [col for _, _, col in res]
            entry = [
                jnp.stack([comb for _, comb, _ in res]) if R > 1 else res[0][1],
                iters - 1,
            ]
            pending_sinkhorn.append(entry)
            return posts, entry, jnp.concatenate(cols, axis=0)

        def combine(posts, combs, block_out, residual):
            if (
                isinstance(combs, list)
                and len(combs) == 2
                and isinstance(combs[1], int)
            ):
                done = sinkhorn_done(combs)
                combs = [done[r] for r in range(R)] if R > 1 else [done]
            outs = []
            for r in range(R):
                comb = (
                    combs[r]
                    if not isinstance(combs[r], list)
                    else sinkhorn_done(combs[r])
                )
                outs.append(
                    K.hyper_combine_vpu(
                        posts[r],
                        comb,
                        block_out[r : r + 1],
                        residual[r * N : (r + 1) * N],
                    )
                )
            return jnp.concatenate(outs, axis=0)

        # KDA checkpoints: slot `qi % 2` of stq / cvq streams to the output rows; a
        # slot is reused only after its transfer is waited.
        def ck_start(slot, k_i, r):
            pltpu.make_async_copy(stq.at[slot], o_recur.at[k_i, r], sck.at[slot, 0]).start()
            pltpu.make_async_copy(cvq.at[slot], o_conv.at[k_i, r], sck.at[slot, 1]).start()

        def ck_wait(slot):
            PL.wait_dma(stq.at[slot], sck.at[slot, 0])
            PL.wait_dma(cvq.at[slot], sck.at[slot, 1])

        def kda(x, k_i, kind):
            loads = [
                pltpu.make_async_copy(src, dst, sgrp.at[2 + i])
                for i, (src, dst) in enumerate(
                    [
                        (w_kfg.at[k_i], fgb),
                        (w_kdt.at[k_i], dtv),
                        (w_kconv.at[k_i], cwv),
                        (w_kb.at[k_i], bpv),
                        (w_ka_log.at[k_i], alv),
                        (w_ksc.at[k_i], ksc),
                    ]
                )
            ]
            for c in loads:
                c.start()

            with _scope("kda_qkv"):
                qkv = mv_km(w_kqkv, k_i, H, x, FP8_KDA, warmed=True)
            if FP8_KDA:
                _wait(loads[-1], "ld_wait")
                qkv = qkv * ksc[:, : 3 * qsh]
            warm_km(w_ko, k_i, FP8_KDA)
            for c in loads[:-1] if FP8_KDA else loads:
                _wait(c, "ld_wait")
            f_low = K.matvec_bf16(x, slab.get(sm, "f_a_proj").astype(BF16))
            g_low = K.matvec_bf16(x, slab.get(sm, "g_a_proj").astype(BF16))
            f = K.matvec_bf16(f_low, fgb[:qsh])
            gg_all = K.matvec_bf16(g_low, fgb[qsh:])
            beta_all = jax.nn.sigmoid(K.matvec_bf16(x, bpv[...]))

            qkvb[...] = qkv
            fgb2[:, :qsh] = f + dtv[...]
            fgb2[:, qsh:] = gg_all
            btb[...] = beta_all
            decay = jnp.exp(alv[...]).reshape(hkda, 1)
            lb = spec.linear_lower_bound
            cw = cwv[...].astype(F32)
            o_norm = slab.get(sm, "o_norm").reshape(1, dh).astype(F32)

            def per_req(b, carry):
                r0 = pl.multiple_of(b * Q, Q)
                row = srow_ref[b]
                PL.dispatch(
                    k_i,
                    n_kda,
                    lambda k: pltpu.make_async_copy(
                        p_tmp[k].at[row], st, sgrp.at[0]
                    ).start(),
                )
                PL.dispatch(
                    k_i,
                    n_kda,
                    lambda k: pltpu.make_async_copy(
                        p_cnv[k].at[row], cv3, sgrp.at[1]
                    ).start(),
                )
                PL.wait_dma(st, sgrp.at[0])
                PL.wait_dma(cv3, sgrp.at[1])
                for qi in range(Q):
                    slot, prev = qi % 2, (qi - 1) % 2
                    if qi >= 2:
                        ck_wait(slot)
                    elif Q > qi:
                        pl.when(b > 0)(lambda slot=slot: ck_wait(slot))
                    r = r0 + qi
                    x1 = qkvb[pl.ds(r, 1), :].reshape(1, 1, 3 * qsh)
                    conv_out, new_conv = K.conv_step(
                        cv3[...] if qi == 0 else cvq[prev], x1, cw
                    )
                    cvq[slot] = new_conv
                    co = conv_out.reshape(1, 3 * qsh)
                    qv = co[0, :qsh].reshape(hkda, dh)
                    kv_ = co[0, qsh : 2 * qsh].reshape(hkda, dh)
                    vv = co[0, 2 * qsh :].reshape(hkda, dh)

                    graw = fgb2[pl.ds(r, 1), pl.ds(0, qsh)].reshape(hkda, dh)
                    glog = (
                        (lb * jax.nn.sigmoid(decay * graw))
                        if lb is not None
                        else (
                            -decay
                            * jnp.where(
                                graw > 20.0,
                                graw,
                                jnp.log1p(jnp.exp(jnp.minimum(graw, 20.0))),
                            )
                        )
                    )
                    beta = btb[pl.ds(r, 1), :][0]
                    new_state, out = K.kda_step(
                        st[...] if qi == 0 else stq[prev], qv, kv_, vv, glog, beta
                    )
                    stq[slot] = new_state
                    ck_start(slot, k_i, r)

                    y = out * jax.lax.rsqrt(
                        jnp.mean(jnp.square(out), axis=-1, keepdims=True) + eps
                    )
                    y = y * o_norm
                    gg = fgb2[pl.ds(r, 1), pl.ds(qsh, qsh)].reshape(hkda, dh)
                    y = y * jax.nn.sigmoid(gg)
                    ysb[pl.ds(r, 1), :] = y.reshape(1, qsh)
                return carry

            jax.lax.fori_loop(0, B, per_req, 0)
            for slot in range(min(Q, 2)):
                ck_wait(slot)
            with _scope("kda_o"):
                return mv_km_allreduce(
                    w_ko,
                    k_i,
                    qsh,
                    ysb[...],
                    FP8_KDA,
                    warmed=True,
                    scale=ksc[:, 3 * qsh :] if FP8_KDA else None,
                )

        TK = cfg.attn_tile
        PT = TK // kp
        NT = Lcap // TK
        TPG = TK // ps  # pages per attention tile
        if Q + kp + 15 > 32:
            raise ValueError(f"n_query={Q}: the pools closed this step span > 32 rows")
        if Q > 8:
            raise ValueError(f"n_query={Q}: a step closes at most 8 pools per request")
        per = qk + vd
        hmat = PL.hadamard(idh)

        def dot(a, b, dims):
            return jax.lax.dot_general(
                a.astype(BF16), b, (dims, ((), ())), preferred_element_type=F32
            )

        def dsa(x, d_i):
            kvb_cp = pltpu.make_async_copy(w_kvb.at[d_i], kvb, sgrp.at[3])
            kvb_cp.start()
            psc_cp = pltpu.make_async_copy(w_dsc.at[d_i], psc, sem_async.at[7])
            psc_cp.start()
            w_qakv_n, w_qb_n, _ = dsa_scale_widths(spec, tp)
            ql = spec.q_lora_rank
            sc_proj = _scope("dsa_proj")
            sc_proj.__enter__()
            qakv = mv_km(w_qakv, d_i, H, x, True, warmed=True, n_out=w_qakv_n)
            _wait(psc_cp, "ld_wait")
            qakv = qakv * psc[:, :w_qakv_n]
            q_resid = K.rms_norm(
                qakv[:, :ql], slab.get(sm, "q_a_layernorm").reshape(1, -1), eps, BF16
            )
            q_all = mv_km(
                w_qb,
                d_i,
                ql,
                q_resid,
                True,
                n_out=w_qb_n,
                scale=psc[:, w_qakv_n : w_qakv_n + w_qb_n],
            )
            kv_new = K.rms_norm(
                qakv[:, ql:], slab.get(sm, "kv_a_layernorm").reshape(1, -1), eps, BF16
            )
            ik_new = K.layer_norm(
                K.matvec_bf16(x, slab.get(sm, "indexer_wk").astype(BF16)),
                slab.get(sm, "indexer_k_norm_w").reshape(1, -1),
                slab.get(sm, "indexer_k_norm_b").reshape(1, -1),
                1e-6,
            )
            ig_new = K.matvec_bf16(
                x, slab.get(sm, "indexer_compress_gate").astype(BF16)
            )
            with _scope("dsa_indexq"):
                iq_all = mv_km(w_iwq, d_i, spec.q_lora_rank, q_resid)
            hw_all = K.matvec_bf16(
                x, slab.get(sm, "indexer_weights_proj").astype(BF16)
            ) * (hi**-0.5)
            ape = slab.get(sm, "indexer_ape").astype(F32)
            sc_proj.__exit__(None, None, None)

            # The step's rows leave as outputs; this step reads them from the row scratch.
            kv_new = kv_new.astype(BF16)
            raw_new = jnp.concatenate(
                [ik_new.astype(BF16), ig_new.astype(BF16)], axis=1
            )
            o_nkv[d_i] = kv_new
            o_nraw[d_i] = raw_new
            ks = ksc_ref[d_i]
            # In fp8 units, rounded as the pool will hold them.
            nkvb[...] = (kv_new.astype(F32) * (1.0 / ks)).astype(f8).astype(BF16)
            nrawb[...] = raw_new
            qallb[...] = q_all
            hwb[...] = hw_all
            for r in range(R):
                iqb[pl.ds(r * hi, hi), :] = PL.rotate(
                    iq_all[r].reshape(hi, idh), hmat
                )
            _wait(kvb_cp, "w_wait")
            kvbf = kvb[...].astype(F32)
            scaling = qk**-0.5 * ks
            pcol = jax.lax.broadcasted_iota(jnp.int32, (PT, 1), 0)
            gidx = jax.lax.broadcasted_iota(
                jnp.int32, (NT, PT), 0
            ) * PT + jax.lax.broadcasted_iota(jnp.int32, (NT, PT), 1)
            pool_end = (gidx + 1) * kp - 1
            tok = jax.lax.broadcasted_iota(jnp.int32, (1, TK), 1)
            lane = jax.lax.broadcasted_iota(jnp.int32, (1, PT), 1)

            def page(b, pg):
                return PL.page_of(tab_ref, b, pg, PPS)

            def fresh_pool_keys(b, r0, p, last):
                """Rotated keys of the pools this step closes, from the raw rows before `p`
                (two 16-row page reads) and the step's own rows."""
                fp0, fresh = PL.closing_pools(p, Q, kp)
                w0 = ((jnp.maximum(p - kp + 1, 0)) // 16) * 16
                for j in range(2):
                    row0 = w0 + 16 * j
                    pg, off = page(b, row0 // ps), pl.multiple_of(row0 % ps, 16)
                    PL.dispatch(
                        d_i,
                        n_dsa,
                        lambda d, j=j, pg=pg, off=off: pltpu.make_async_copy(
                            p_raw[d].at[pg, pl.ds(off, 16)],
                            rawb.at[pl.ds(16 * j, 16)],
                            sgrp.at[j],
                        ).start(),
                    )
                for j in range(2):
                    PL.wait_dma(rawb.at[pl.ds(16 * j, 16)], sgrp.at[j])
                rr = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0) + w0
                rawv = rawb[...]
                for qi in range(Q):
                    rawv = jnp.where(rr == p + qi, nrawb[pl.ds(r0 + qi, 1), :], rawv)
                pk_blk, _ = K.indexer_pool_keys(
                    rawv[:, :idh], rawv[:, idh:], ape, last, kp
                )
                pk_blk = PL.rotate(pk_blk, hmat)
                prow = jax.lax.broadcasted_iota(jnp.int32, (32 // kp, 1), 0) + w0 // kp
                keys = [
                    jnp.sum(
                        jnp.where(prow == fp0 + i, pk_blk, 0.0), axis=0, keepdims=True
                    )
                    for i in range(Q)
                ]
                for i in range(Q):
                    o_npk[d_i, pl.ds(r0 + i, 1), :] = keys[i]
                return fp0, fresh, keys

            def rec_fetch(b, t, slot):
                for i in range(TPG):
                    pg = page(b, t * TPG + i)
                    PL.dispatch(
                        d_i,
                        n_dsa,
                        lambda d, i=i, pg=pg: pltpu.make_async_copy(
                            PL.record_page(p_rec[d], pg, RPP, rec_bytes),
                            rkw.at[slot, pl.ds(i * (RPP // PL.WORD_BYTES), RPP // PL.WORD_BYTES)],
                            sgrp.at[6 + slot],
                        ).start(),
                    )

            def kv_fetch(b, t, slot):
                for i in range(TPG):
                    pg = page(b, t * TPG + i)
                    PL.dispatch(
                        d_i,
                        n_dsa,
                        lambda d, i=i, pg=pg: pltpu.make_async_copy(
                            PL.nope_page(p_nope[d], pg, ps),
                            kvw.at[slot, pl.ds(i * ps, ps)],
                            sgrp.at[4 + slot],
                        ).start(),
                    )

            def select(b, r0, p, last, fp0, fresh, keys):
                """Top-k pool mask of each query into `scr`, scoring the pooled records."""
                nts = last // TK + 1
                iq_cat = iqb[pl.ds(pl.multiple_of(r0 * hi, hi), Q * hi), :]
                hw_q = hwb[pl.ds(r0, Q), :]
                if Q > 1:
                    hcol = jax.lax.broadcasted_iota(jnp.int32, (Q, Q * hi), 1)
                    hrow = jax.lax.broadcasted_iota(jnp.int32, (Q, Q * hi), 0)
                    hw_blk = jnp.where(
                        hcol // hi == hrow, jnp.tile(hw_q, (1, Q)), 0.0
                    ).astype(F32)
                rec_fetch(b, 0, 0)

                def score_tile(t, c):
                    slot = jax.lax.rem(t, 2)
                    PL.wait_dma(rkw.at[slot], sgrp.at[6 + slot])

                    @pl.when(t + 1 < nts)
                    def _():
                        rec_fetch(b, t + 1, 1 - slot)

                    pk = PL.unpack_records(rkw[slot], idh)
                    for i in range(Q):
                        pk = jnp.where(
                            (pcol + t * PT == fp0 + i) & (i < fresh), keys[i], pk
                        )
                    if Q > 1:
                        sc = K.indexer_scores(iq_cat, hw_blk, pk, idh**-0.5)
                        scr[:, pl.ds(t, 1), :] = sc.reshape(Q, 1, PT)
                    else:
                        scr[0, pl.ds(t, 1), :] = K.indexer_scores(
                            iq_cat, hw_q, pk, idh**-0.5
                        )
                    return c

                with _scope("dsa_scores"):
                    jax.lax.fori_loop(0, nts, score_tile, 0)
                eligs = [pool_end <= p + qi for qi in range(Q)]
                if Q == 1:
                    chosen = K.select_top_k_mask_2d(
                        jnp.where(eligs[0], scr[0], -jnp.inf), selk
                    )[None]
                else:
                    chosen = K.select_top_k_mask_3d(
                        jnp.where(jnp.stack(eligs), scr[...], -jnp.inf), selk
                    )
                scr[...] = chosen.astype(F32)  # the scores are not needed again

            def allowed_rows(p, tok_ids, t):
                """[Q, width] token mask of each query over tile `t` (or the new rows when t is None)."""
                rows = []
                for qi in range(Q):
                    p_q = p + qi
                    if t is None:
                        sel = jnp.zeros(tok_ids.shape, jnp.bool_)
                        for i in range(Q):
                            ti, ci = (p + i) // TK, ((p + i) % TK) // kp
                            hit = (
                                jnp.sum(
                                    jnp.where(lane == ci, scr[qi, pl.ds(ti, 1), :], 0.0)
                                )
                                > 0.5
                            )
                            sel = sel | ((tok_ids == p + i) & hit)
                    else:
                        # Repeating a bool lowers to an i1 transpose; repeat the f32 and compare.
                        sel = jnp.repeat(scr[qi, pl.ds(t, 1), :], kp, axis=1) > 0.5
                    if spec.index_kpool_always_select_tail:
                        tail_count = jax.lax.rem(p_q + 1, kp)
                        tail_lo = p_q + 1 - tail_count
                        sel = sel | ((tok_ids >= tail_lo) & (tail_count > 0))
                    rows.append(sel & (tok_ids <= p_q))
                return jnp.concatenate(rows, axis=0) if Q > 1 else rows[0]

            def attend(b, r0, p):
                """Online softmax over the request's pool tiles, then its own new rows."""
                nta = (p + TK - 1) // TK
                qrows = qallb[pl.ds(r0, Q), :]
                q_abs = jnp.concatenate(
                    [
                        dot(
                            qrows[:, h * qk : (h + 1) * qk],
                            kvbf[h * per : h * per + qk].astype(BF16),
                            ((1,), (0,)),
                        )
                        for h in range(nh)
                    ],
                    axis=0,
                )
                q_bf = q_abs.astype(BF16)  # rows h * Q + qi

                pl.when(nta > 0)(lambda: kv_fetch(b, 0, 0))

                def step(carry, kvt_b, allowed):
                    m_o, l_o, a_o = carry
                    lg = dot(q_bf, kvt_b, ((1,), (1,))) * scaling
                    lg = jnp.where(allowed, lg, -jnp.inf)
                    m_n = jnp.maximum(m_o, jnp.max(lg, axis=1, keepdims=True))
                    shift = jnp.where(jnp.isfinite(m_n), m_n, 0.0)
                    pr = jnp.where(allowed, jnp.exp(lg - shift), 0.0)
                    alpha = jnp.exp(m_o - shift)
                    l_n = alpha * l_o + jnp.sum(pr, axis=1, keepdims=True)
                    a_n = alpha * a_o + dot(pr, kvt_b, ((1,), (0,)))
                    return m_n, l_n, a_n

                def attend_tile(t, carry):
                    slot = jax.lax.rem(t, 2)
                    PL.wait_dma(kvw.at[slot], sgrp.at[4 + slot])

                    @pl.when(t + 1 < nta)
                    def _():
                        kv_fetch(b, t + 1, 1 - slot)

                    kvt_b = PL.unpack_nope(kvw[slot])
                    tids = tok + t * TK
                    # Rows at or past p hold stale draft tokens; the step's own rows follow.
                    allowed = allowed_rows(p, tids, t) & (tids < p)
                    return step(carry, kvt_b, jnp.tile(allowed, (nh, 1)))

                init = (
                    jnp.full((nh * Q, 1), -jnp.inf, F32),
                    jnp.zeros((nh * Q, 1), F32),
                    jnp.zeros((nh * Q, kvl), F32),
                )
                with _scope("dsa_attend"):
                    carry = jax.lax.fori_loop(0, nta, attend_tile, init)
                # The new rows as one 16-row block (a one-row MXU operand does not lower).
                new_tok = p + jax.lax.broadcasted_iota(jnp.int32, (1, 16), 1)
                allowed = allowed_rows(p, new_tok, None)
                kn = jnp.pad(nkvb[pl.ds(r0, Q), :], ((0, 16 - Q), (0, 0)))
                _, l_f, a_f = step(carry, kn, jnp.tile(allowed, (nh, 1)))
                outs = []
                for h in range(nh):
                    lat = a_f[h * Q : (h + 1) * Q] / (l_f[h * Q : (h + 1) * Q] + 1e-30)
                    w_vc = kvbf[h * per + qk : (h + 1) * per].astype(BF16)
                    outs.append(dot(lat, w_vc, ((1,), (1,))) * ks)
                attb[pl.ds(r0, Q), :] = jnp.concatenate(outs, axis=1).astype(BF16)

            def per_req(b, carry):
                r0 = pl.multiple_of(b * Q, Q)
                p = pos_ref[b]
                last = p + Q - 1
                fp0, fresh, keys = fresh_pool_keys(b, r0, p, last)
                select(b, r0, p, last, fp0, fresh, keys)
                attend(b, r0, p)
                return carry

            jax.lax.fori_loop(0, B, per_req, 0)
            with _scope("dsa_o"):
                o_off = w_qakv_n + w_qb_n
                return mv_km_allreduce(
                    w_do,
                    d_i,
                    nh * vd,
                    attb[...],
                    True,
                    scale=psc[:, o_off : o_off + H],
                )

        def moe_early(m_i):
            """Start the router and scale fetches before the FFN-side mHC."""
            fetch = [(w_router.at[m_i], rtr)]
            if not CH_EXPERTS:
                fetch += [
                    (w_egus.at[m_i], egs),
                    (w_edns.at[m_i], eds),
                    (w_sgus.at[m_i], s_sgu),
                    (w_sdns.at[m_i], s_sdn),
                ]
            early = [
                pltpu.make_async_copy(src, dst, sem_async.at[i])
                for i, (src, dst) in enumerate(fetch)
            ]
            for c in early:
                c.start()
            for c in shared_copies(m_i):
                c.start()
            return early

        def expert_copies(sl, gu, dn, gu_s=None, dn_s=None):
            if CH_EXPERTS:
                return [
                    pltpu.make_async_copy(gu, r8a.at[sl], s8a.at[sl]),
                    pltpu.make_async_copy(dn, r8d.at[sl], s8d.at[sl]),
                    pltpu.make_async_copy(gu_s, egsr.at[sl], s8s.at[0, sl]),
                    pltpu.make_async_copy(dn_s, edsr.at[sl], s8s.at[1, sl]),
                ]
            return [
                pltpu.make_async_copy(
                    gu, r8a.at[sl, pl.ds(0, 2 * moesh), pl.ds(0, H)], s8a.at[sl]
                ),
                pltpu.make_async_copy(
                    dn, r8d.at[sl, pl.ds(0, H), pl.ds(0, moesh)], s8d.at[sl]
                ),
            ]

        def shared_copies(m_i):
            if CH_EXPERTS:
                return expert_copies(
                    0,
                    w_sgu.at[m_i],
                    w_sdn.at[m_i],
                    w_sgus.at[pl.ds(m_i, 1)],
                    w_sdns.at[pl.ds(m_i, 1)],
                )
            return expert_copies(0, w_sgu.at[m_i], w_sdn.at[m_i])

        def moe(x, m_i, kind, early, m_cands):
            """`m_cands`: the MoE ordinals m_i can take here; the expert refs are per layer."""
            with _scope("router_wait"):
                _wait(early[0], "e_wait")
            route_kw = dict(
                top_k=kexp,
                norm_topk_prob=spec.norm_topk_prob,
                routed_scaling_factor=spec.routed_scaling_factor,
            )
            sc_route = _scope("moe_route")
            sc_route.__enter__()
            if R == 1 and kexp == 8:
                mask, wts, idxs = K.route_experts_ids(
                    x, rtr[...], slab.getf(smf, kind, "router_bias"), **route_kw
                )
            else:
                mask, wts = K.route_experts(
                    x, rtr[...], slab.getf(smf, kind, "router_bias"), **route_kw
                )
                idxs = None
            # The union of the rows' experts: at most every expert once.
            n_slot = min(kexp * R, nexp)
            union = (
                mask
                if R == 1
                else jnp.max(mask.astype(jnp.int32), axis=0, keepdims=True) > 0
            )
            if idxs is None:
                idxs = K.mask_to_indices_mxu(union, n_slot)
            base = m_i * nexp
            acc = jnp.zeros((R, H), F32)
            # Unused union slots hold sentinel nexp: clamp it for the fetch; its weight is zero.
            safe = jnp.minimum(idxs, nexp - 1)
            sc_route.__exit__(None, None, None)

            # Entry 0 is the shared expert, whose copy moe_early already started.
            entries = [None] + list(range(n_slot))

            def slot_copies(k, m=None):
                """Slot k's copies from layer m's experts; any layer for a wait (same shapes)."""
                e = entries[k]
                if e is None:
                    return shared_copies(m_i)
                m = m_cands[0] if m is None else m
                return expert_copies(
                    k % S,
                    w_egu[m].at[safe[e]],
                    w_edn[m].at[safe[e]],
                    w_egus.at[pl.ds(base + safe[e], 1)],
                    w_edns.at[pl.ds(base + safe[e], 1)],
                )

            n_e = len(entries)
            # R > 1: union ids are ascending with unused slots last, so only the first n_u run.
            guard = R > 1
            n_u = jnp.sum(union.astype(jnp.int32)) if guard else None

            def when(k, fn):
                if not guard or entries[k] is None:
                    fn()
                else:
                    pl.when(entries[k] < n_u)(fn)

            def start(k):
                def go():
                    if entries[k] is None or len(m_cands) == 1:
                        for c in slot_copies(k):
                            c.start()
                        return
                    for m in m_cands:

                        @pl.when(m_i == m)
                        def _(m=m):
                            for c in slot_copies(k, m)[:2]:
                                c.start()

                    for c in slot_copies(k)[2:]:
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
                            hit = (
                                jax.lax.broadcasted_iota(jnp.int32, (R, nexp), 1)
                                == idxs[e]
                            )
                            w_j = jnp.sum(
                                jnp.where(hit, wts, 0.0), axis=1, keepdims=True
                            )
                        h = (
                            jnp.dot(x.astype(BF16), r8a[sl], preferred_element_type=F32)
                            * egsr[sl]
                        )
                        act = K.swiglu_clamped(
                            h[:, :moesh], h[:, moesh:], spec.swiglu_limit
                        )
                        y = (
                            jnp.dot(
                                act.astype(BF16), r8d[sl], preferred_element_type=F32
                            )
                            * edsr[sl]
                        )
                        return y if w_j is None else w_j * y
                    if e is None:
                        early[3].wait()
                        early[4].wait()
                        gs, dsc, w_j = s_sgu[...], s_sdn[...], None
                    else:
                        gs, dsc = egs[safe[e]], eds[safe[e]]
                        # A masked sum, since a dynamic lane extract does not lower.
                        hit = (
                            jax.lax.broadcasted_iota(jnp.int32, (R, nexp), 1) == idxs[e]
                        )
                        w_j = jnp.sum(jnp.where(hit, wts, 0.0), axis=1, keepdims=True)
                    h = K.matvec_fp8(x, r8a[sl, : 2 * moesh, :H], gs, blk)
                    act = K.swiglu_clamped(
                        h[:, :moesh], h[:, moesh:], spec.swiglu_limit
                    )
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
            _dma(w_dnsc.at[d_i], psc, sem2)
            h = mv_km(
                w_dgu, d_i, H, x, True, n_out=2 * ffnsh, scale=psc[:, : 2 * ffnsh]
            )
            act = K.swiglu_clamped(h[:, :ffnsh], h[:, ffnsh:], spec.swiglu_limit)
            return mv_km_allreduce(
                w_ddn, d_i, ffnsh, act, True, scale=psc[:, 2 * ffnsh : 2 * ffnsh + H]
            )

        def layer(
            streams,
            L,
            kind,
            mlp_kind,
            next_kda=None,
            prewarmed=False,
            par=None,
            moe_cands=None,
        ):
            """`next_kda`: start that KDA layer's qkv during this FFN; `prewarmed`: ours was started so."""
            slab_kind = f"{kind}_{'sparse' if mlp_kind == 'moe' else 'dense'}"
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
            if kind == "kda":
                if not prewarmed:
                    warm_km(w_kqkv, kda_ord(L), FP8_KDA)
            else:
                warm_km(w_qakv, dsa_ord(L), True)
            with _scope("mhc_attn"):
                posts, combs, x = mhc(streams, "attn", slab_kind)
                x = K.rms_norm(
                    x, slab.get(sm, "input_layernorm").reshape(1, H), eps, BF16
                )
            with _scope(kind):
                a = (
                    kda(x, kda_ord(L), slab_kind)
                    if kind == "kda"
                    else dsa(x, dsa_ord(L))
                )
            with _scope("combine"):
                streams = combine(posts, combs, a, streams)

            early = moe_early(moe_ord(L)) if mlp_kind == "moe" else None
            if next_kda is not None:
                warm_km(w_kqkv, kda_ord(next_kda), FP8_KDA)
            with _scope("mhc_ffn"):
                posts, combs, x = mhc(streams, "ffn", slab_kind)
                x = K.rms_norm(
                    x, slab.get(sm, "post_attention_layernorm").reshape(1, H), eps, BF16
                )
            with _scope(mlp_kind):
                m = (
                    moe(x, moe_ord(L), slab_kind, early, moe_cands or [moe_ord(L)])
                    if mlp_kind == "moe"
                    else dense(x, L)
                )
            with _scope("combine"):
                return combine(posts, combs, m, streams)

        o_nkv[...] = jnp.zeros(o_nkv.shape, o_nkv.dtype)
        o_nraw[...] = jnp.zeros(o_nraw.shape, o_nraw.dtype)
        o_npk[...] = jnp.zeros(o_npk.shape, o_npk.dtype)
        n_dense = min(spec.first_k_dense_replace, n_layers)
        # No next_kda: the dense MLP occupies the K-major ring.
        for L in range(n_dense):
            streams = layer(streams, L, "kda", "dense")

        full_periods = min(
            cfg.n_periods, max(0, (n_layers - spec.first_k_dense_replace) // 4)
        )
        f0 = spec.first_k_dense_replace

        def cands(j):
            return [moe_ord(f0 + 4 * q + j) for q in range(full_periods)]

        def period(p, s_):  # noqa: ARG001 - p is the traced period index
            base = f0 + 4 * p
            s_ = layer(
                s_,
                base,
                "dsa",
                "moe",
                next_kda=base + 1,
                par=f0 % 2,
                moe_cands=cands(0),
            )
            s_ = layer(
                s_,
                base + 1,
                "kda",
                "moe",
                next_kda=base + 2,
                prewarmed=True,
                par=(f0 + 1) % 2,
                moe_cands=cands(1),
            )
            s_ = layer(
                s_,
                base + 2,
                "kda",
                "moe",
                next_kda=base + 3,
                prewarmed=True,
                par=(f0 + 2) % 2,
                moe_cands=cands(2),
            )
            s_ = layer(
                s_,
                base + 3,
                "kda",
                "moe",
                prewarmed=True,
                par=(f0 + 3) % 2,
                moe_cands=cands(3),
            )
            return s_

        streams = jax.lax.fori_loop(0, full_periods, period, streams)

        for L in range(spec.first_k_dense_replace + 4 * full_periods, n_layers):
            kind = "kda" if spec.layer_types[L] == "linear_attention" else "dsa"
            streams = layer(streams, L, kind, "moe")

        o_streams[...] = streams
        _dma(w_fnorm, fnv, sem)
        hidden = K.rms_norm(
            jnp.concatenate(
                [
                    jnp.mean(
                        streams[r * N : (r + 1) * N].astype(F32),
                        axis=0,
                        keepdims=True,
                    )
                    for r in range(R)
                ],
                axis=0,
            ),
            fnv[...],
            eps,
            BF16,
        )
        # The MTP draft reads this hidden (serving captures the same tensor).
        o_hidden[...] = hidden
        o_logits[...] = mv_km(w_lm, 0, H, hidden)

    if not cfg.ch_experts:
        raise NotImplementedError("the per-layer expert refs need ch_experts")
    n_moe = sum(t == "sparse" for t in spec.mlp_layer_types)
    names = list(inspect.signature(kernel).parameters)
    n_in = names.index("o_logits")
    # Per-layer arrays arrive as flat runs, in this order, after the packed weights.
    lists = [
        ("p_nope", n_dsa),
        ("p_raw", n_dsa),
        ("p_rec", n_dsa),
        ("p_tmp", n_kda),
        ("p_cnv", n_kda),
        ("w_egu", n_moe),
        ("w_edn", n_moe),
    ]
    listed = {n for n, _ in lists}
    packed = [n for n in names[:n_in] if n not in listed]

    def entry(*refs):
        """Inputs, pools, then the per-layer expert refs; outputs and scratch."""
        k = len(packed)
        bound = dict(zip(packed, refs[:k]))
        for n, cnt in lists:
            bound[n] = list(refs[k : k + cnt])
            k += cnt
        bound.update(zip(names[n_in:], refs[k:]))
        return kernel(**bound)

    entry.lists = lists
    return entry
