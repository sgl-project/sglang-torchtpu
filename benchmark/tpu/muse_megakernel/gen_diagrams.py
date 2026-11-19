"""Generate the per-layer communication diagrams for sglang_tpu/muse_megakernel/README.md.

Run: python benchmark/tpu/muse_megakernel/gen_diagrams.py  (writes *.svg to assets/muse_megakernel/)
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import NamedTuple

OUT_DIR = Path(__file__).resolve().parents[3] / "assets" / "muse_megakernel"

# Muse Spark at TP 8; KiB per token row is what one rank puts on the wire (7/8 remote share).
PREFILL_BLOCK_ROWS = 4096
DECODE_BLOCK_ROWS = 64

DTYPE_COLORS = {
    "f32": "#d1242f",
    "bf16": "#1f6feb",
    "fp8 e4m3": "#bc4c00",
    "NVFP4 e2m1": "#8250df",
    "int32": "#57606a",
    "u32": "#0a7d62",
}
_DTYPE_RE = re.compile(
    "("
    + "|".join(re.escape(k) for k in sorted(DTYPE_COLORS, key=len, reverse=True))
    + ")"
)

SECTION_COLORS = {
    "Attention": ("#2f6fdb", "#f3f7fd", "#e3edfb"),
    "Pooler": ("#e8590c", "#fdf6f1", "#fbe6da"),
    "MoE": ("#2b9a66", "#f2faf6", "#ddf2e7"),
    "Unpooler": ("#e0a100", "#fdfaf0", "#faf0d2"),
}


# all of a layer's expert bytes one rank holds (fp4, QBLOCK=256), read once every expert is hit
EXPERT_ALL_MIB = 780
# EP worst-rank / mean expert reads under uniform routing (simulated), by rows
EP_IMBALANCE = ((8, 1.48), (32, 1.19), (64, 1.09), (128, 1.02), (512, 1.0))


def expert_mib(rows: int, layout: str) -> float:
    """Expert weights streamed per layer per rank: P(expert hit) = 1 - (31/32)^R (top 8 of 256)."""
    mean = EXPERT_ALL_MIB * (1 - (31 / 32) ** rows)
    if layout == "tp":
        return mean
    imb = next((f for r, f in EP_IMBALANCE if rows <= r), 1.0)
    return min(mean * imb, EXPERT_ALL_MIB)


KV_FP8 = "KV read: 244 MiB per 1M attended tokens (fp8 pool, required)"
KV_EITHER = "KV read: 488 MiB per 1M attended tokens (bf16 pool) / 244 (fp8 pool)"


class Box(NamedTuple):
    kind: str  # "compute" or "comm"
    title: str
    lines: tuple[str, ...]
    kib: float = 0.0
    note: str = ""
    formula: str = ""  # comm only: how kib is derived (FORMULAS)


class Column(NamedTuple):
    key: str
    title: str
    subtitle: tuple[str, ...]
    block_rows: int
    n_ops: int  # collectives per layer; one box may bundle several
    sections: dict[str, tuple[Box, ...]]
    # HBM<->VMEM DMA per layer per rank, estimate: section -> (fixed weights MiB,
    # activations + KV writes KiB per row); experts add expert_mib(R, expert_layout)
    dma: dict[str, tuple[float, float]]
    expert_layout: str  # tp: 1/8 of every expert; ep: 32 whole experts, worst rank
    dma_note: str = ""
    kv_note: str = KV_FP8


def C(title: str, *lines: str, note: str = "") -> Box:
    return Box("compute", title, lines, note=note)


def X(title: str, kib: float, *lines: str, note: str = "") -> Box:
    return Box("comm", title, lines, kib=kib, note=note)


_EXPERT_W = (
    "w13/w2: L1-60 NVFP4 e2m1 + bf16 block-16 scale",
    "        L0/61 fp8 e4m3 + f32 ch-scale; acc f32",
)
_Y_DIRECT_RS = X(
    "⇄ y direct-write RS (remote DMA inside GMM)",
    56,
    "partial [R*8, He] bf16 -> owner out_buf",
    "  [R/8, 8 slot, 8 shard, He] bf16",
    note="each gm tile is sent (make_async_remote_copy) as soon as it finishes; overlaps GMM",
)
_FUSED_EXPERTS = C(
    "experts (expert-TP, fused-RS GMM)",
    "all R*8 (token, slot) pairs x this rank's 1/8 I slice",
    *_EXPERT_W,
    "h = silu(g)*u -> bf16",
)
_FUSED_COMBINE = C(
    "combine (owner rows)",
    "sum 8 shards f32 -> bf16; post-expert norm f32",
    "top-k weighted sum bf16 -> y [R/8, He] bf16",
)

PREFILL_TP = Column(
    key="prefill_tp",
    dma={
        "Attention": (26, 77.25),
        "Pooler": (8, 9),
        "MoE": (0, 193),
        "Unpooler": (4, 65),
    },
    expert_layout="tp",
    dma_note="residual x [R, H] f32 read + written twice per layer: 128 KiB x R",
    title="(1) Prefill TP (default)",
    subtitle=(
        "Rows replicated: every rank holds all R rows; attn split by head, dense row/col",
        "parallel, expert-TP. GridColl grid route, pipelined over 64/128-row tiles",
    ),
    block_rows=PREFILL_BLOCK_ROWS,
    n_ops=7,
    sections={
        "Attention": (
            C(
                "input_layernorm + qkvg",
                "x [R, H] f32 -> a_in [R, H] bf16",
                "a_in @ wqkvg [H, 2304] fp8 e4m3 -> f32 x ch-scale",
                "-> q [R,1024] / kv [R,256] / gate [R,1024] bf16",
                note="fused into the tail of the previous layer's post_moe; head-RMS + RoPE (SWA)",
            ),
            C(
                "attention (RPA)",
                "q [R, 16, 64] bf16; new K/V bf16 -> fp8 e4m3 pool",
                "-> o [R, 1024] bf16",
                note="16 Q / 2 KV heads per rank, over all R rows",
            ),
            C(
                "o_proj (row-parallel)",
                "sigmoid(gate)*rms(o) bf16 @ wo [1024, H] fp8 e4m3",
                "-> partial [R, H] f32 -> cast bf16 before send",
            ),
            X(
                "⇄ o RS + AG (= all-reduce)",
                28,
                "RS: [R, H] bf16 -> [R, H/8]; f32 sum per hop, bf16 wire",
                "AG: [R, H/8] bf16 -> [R, H] bf16",
                note="attn_out kernel: rs_pair / rs2_* / ag_*, gap=1 pipeline",
            ),
            C("attention residual", "x = a*x + b*rms(attn)   [R, H] f32"),
        ),
        "Pooler": (
            C(
                "pre-FFN norm + router (weight replicated, all R rows)",
                "r_in [R, H] bf16 @ wrouter [H, 256] bf16 -> f32",
                "sigmoid + bias -> top8: ids int32, tw f32 [R, 8]",
                note="wrouter replicated; every rank routes all R rows redundantly",
            ),
            C(
                "pooler (column-parallel)",
                "r_in @ wpool [H, 512] fp8 e4m3 -> f32 -> [R, 512] bf16",
            ),
            X(
                "⇄ xe AG",
                7,
                "[R, He/8] bf16 -> [R, He] bf16",
                "FP8_GATHER=1: fp8 e4m3 + f32 block-scale (3.5 KiB)",
            ),
            C(
                "pre-expert norm",
                "xe = rms(xe)*w -> [R, He] bf16 (all R rows on every rank)",
            ),
        ),
        "MoE": (
            X(
                "⇄ routing psum (compiler workaround)",
                32 / 1024,
                "ids int32 / tw f32 -> bf16 [R, 8, 2], XLA psum",
                "routing already identical; only rank 0 is nonzero",
            ),
            _FUSED_EXPERTS,
            _Y_DIRECT_RS,
            _FUSED_COMBINE,
        ),
        "Unpooler": (
            X(
                "⇄ y AG (token dim)",
                7,
                "[R/8, He] bf16 -> [R, He] bf16 (always bf16)",
                note="post_moe kernel, tm=128, gap=0",
            ),
            C(
                "unpooler (column-parallel)",
                "y @ wunpool [He, 1024] fp8 e4m3 -> f32 -> [R, 1024] bf16",
            ),
            X(
                "⇄ u AG",
                14,
                "[R, H/8] bf16 -> [R, H] bf16",
                "FP8_GATHER=1: fp8 e4m3 + f32 block-scale (7 KiB)",
            ),
            C(
                "post-FFN norm + residual -> next layer qkvg",
                "x = a*x + b*rms(u*w)   [R, H] f32",
            ),
        ),
    },
)

PREFILL_SP = Column(
    key="prefill_sp",
    dma={
        "Attention": (26, 90),
        "Pooler": (36, 6),
        "MoE": (0, 110),
        "Unpooler": (32, 14),
    },
    expert_layout="ep",
    dma_note="unpack_rows decodes only received rows; the cap-sized receive buffer is allocated, not read",
    title="(2) Prefill SP + EP8, o RS variant (removed; reference for (3))",
    subtitle=(
        "Residual row-sharded (own R/8 rows); attn gathers all R rows, split by head. MoE:",
        "EP8, 32 whole experts per rank, remote-DMA a2a (common/a2a.py). fp8 policy dtypes",
    ),
    block_rows=PREFILL_BLOCK_ROWS,
    n_ops=5,
    sections={
        "Attention": (
            C(
                "input_layernorm (own rows)",
                "stream [R/8, H] f32 -> RMS f32 -> a_in fp8 e4m3 + row scale f32",
            ),
            X(
                "⇄ a_in AG (fp8 + row scale)",
                7,
                "[R/8, H] fp8 e4m3 + [R/8, 1] f32 -> [R, H]",
            ),
            C(
                "qkvg (fp8 x fp8) + attention (by head, all R rows)",
                "a_in fp8 e4m3 @ wqkvg fp8 e4m3 -> f32 x row x ch scales",
                "q bf16; K/V fp8 e4m3",
            ),
            C(
                "o_proj (fp8 x fp8)",
                "head-RMS f32 -> gated fp8 e4m3 + row scale @ wo fp8 e4m3",
                "-> f32 -> partial [R, H] bf16",
            ),
            X(
                "⇄ o RS (psum_scatter, bf16: feeds post-attn norm)",
                14,
                "partial [R, H] bf16 -> [R/8, H] bf16",
            ),
            C(
                "attention residual (own rows)",
                "stream = a*stream + b*rms(a_rs)   [R/8, H] f32",
            ),
        ),
        "Pooler": (
            C(
                "pre-FFN norm + router (bf16 until GSM8K A/B)",
                "ffn f32 -> bf16 @ wrouter [H, 256] bf16 -> f32",
                "-> ids int32, tw f32 [R/8, 8]",
            ),
            C(
                "pooler (weight replicated, fp8 x fp8)",
                "ffn fp8 e4m3 + row scale @ wpool [H, He] fp8 e4m3 -> f32",
                note="no collective: wpool is replicated",
            ),
            C(
                "pre-expert norm (own rows)",
                "RMS f32 -> xe [R/8, He] fp8 e4m3 + row scale f32",
            ),
        ),
        "MoE": (
            X(
                "⇄ count exchange (AG)",
                0,
                "per-global-expert counts [256] int32 -> [8, 256] int32",
                "7 KiB per layer, independent of R",
            ),
            C(
                "sort (row, slot) pairs by global expert",
                "one segment per (source, expert), padded to 8 rows",
                note="no per-rank dedup: a token sends one copy per slot",
            ),
            X(
                "⇄ dispatch a2a (remote DMA, expert-grouped)",
                4.91,
                "row = 1152 u32 (pack_rows): e4m3 bytes, f32 scale in word He/4, pad",
                "8-row chunk DMAs land in local-expert, then source order",
            ),
            C(
                "experts (EP, whole experts, gmm_v2 W4A8 / W8A8)",
                "recv already expert-grouped (no sort); Pallas unpack_rows -> bf16",
                "gmm_v2 requantizes x, h to fp8 e4m3",
                "silu(gate) * up fused in the w13 epilogue (1024-col g/u pairs)",
                "fp4 layers: NVFP4 e2m1, block-256 scales; acc f32",
                "out bf16 (feeds post-expert norm)",
            ),
            X(
                "⇄ combine a2a (remote DMA, bf16)",
                8.53,
                "y [cap, He] bf16, same 8-row segments back -> source's order",
            ),
            C(
                "combine (owner rows)",
                "post-expert RMS f32 x tw f32, sum f32 -> u f32",
                "-> u fp8 e4m3 + row scale",
            ),
        ),
        "Unpooler": (
            C(
                "unpooler (weight replicated, fp8 x fp8)",
                "u fp8 e4m3 @ wunpool [He, H] fp8 e4m3 -> f32",
                "-> f [R/8, H] bf16 (feeds post-FFN norm)",
                note="no collective: wunpool is replicated",
            ),
            C("post-FFN norm + residual (own rows)", "RMS f32; stream [R/8, H] f32"),
        ),
    },
)

DECODE_TP = Column(
    key="decode_tp",
    kv_note=KV_EITHER,
    dma={
        "Attention": (26, 96.25),
        "Pooler": (8, 16),
        "MoE": (0, 16),
        "Unpooler": (4, 96),
    },
    expert_layout="tp",
    dma_note="row-tiled: residual [R, H] f32 + o / f / pool / u AG pads in HBM; y pads in VMEM",
    title="(4) Decode TP (megakernel, row-tiled at R > 32)",
    subtitle=(
        "Rows replicated. R > 32 runs the row-tiled program (16-row tiles, residual + AG pads",
        "in HBM; needs MK_MAX_ROWS >= R). MK_OPT: pair-route AG, paired o RS, int8 y, fp8 expert x",
    ),
    block_rows=DECODE_BLOCK_ROWS,
    n_ops=6,
    sections={
        "Attention": (
            C(
                "input_layernorm + qkvg",
                "stream [R, H] f32 -> bf16 @ wqkvg [H, 2304] fp8 e4m3",
                "-> [R, 2304] f32 x ch-scale",
                note="tiled_fold: consumes the previous layer's f AG block by block as it lands",
            ),
            C(
                "attention",
                "q bf16, score / acc f32; KV pool bf16 or fp8 e4m3",
                "new K/V -> newkv -> kv_write_call; o [R, 1024] bf16",
            ),
            C(
                "o_proj (row-parallel)",
                "attn bf16 @ wo [1024, H] fp8 e4m3 -> partial [R, H] f32",
            ),
            X(
                "⇄ o RS (column-streamed) + o AG",
                28,
                "RS: paired o RS (pair_o_rs), bf16 (u32); owner f32 sum",
                "AG: [R, H/8] bf16 (u32) -> [R, H], pair route (pair_ag)",
            ),
            C(
                "attention residual",
                "stream = a*stream + b*bf16(rms(attn))   [R, H] f32",
            ),
        ),
        "Pooler": (
            C(
                "pre-FFN norm + pooler (column-parallel)",
                "ffn bf16 @ wpool [H, 512] fp8 e4m3 -> [R, 512] bf16",
            ),
            X(
                "⇄ pool AG (started early, overlaps router)",
                7,
                "[R, He/8] bf16 (u32) -> [R, He] bf16, pair route",
                "not gated by n_live: padding rows are sent too",
            ),
            C(
                "router (weight replicated, all R rows)",
                "ffn bf16 @ wrouter [H, 256] bf16 -> f32 -> top8",
                "ids int32 (SMEM route table), wsel f32",
            ),
            C("pre-expert norm", "xe [R, He] bf16 (all R rows on every rank)"),
        ),
        "MoE": (
            C(
                "experts (expert-TP, selected experts only)",
                *_EXPERT_W,
                "x -> fp8 e4m3 into gate/up (ex_lhs_fp8=1); h bf16 into w2",
                "y partial f32 -> int8 + per-lane f32 scale (y_quant)",
                note="expert weights stream through 4 VMEM slots (prefetch depth 3)",
            ),
            X(
                "⇄ y RS (by rows)",
                31.5,
                "ysend [R, 8, He] int8 + 128 f32 scales per slot row (u32)",
                "-> VMEM pad_y [n, R/8, 8, He] (tiled_vmem_y); owner f32 sum",
                "sent once an owner's last expert is done; own rows last",
            ),
            C(
                "combine (owner rows)",
                "sum n partials f32 -> post-expert norm -> x wsel",
                "-> u [R/8, He] bf16",
            ),
            X("⇄ u AG", 7, "[R/8, He] bf16 -> [R, He] bf16 (live mask)"),
        ),
        "Unpooler": (
            C(
                "unpooler (column-parallel)",
                "u bf16 @ wunpool [He, 1024] fp8 e4m3 -> [R, 1024] bf16",
            ),
            X(
                "⇄ f AG",
                14,
                "[R, H/8] bf16 (u32) -> [R, H] bf16, pair route",
                note="tiled_fold: next layer's qkvg consumes it as it lands",
            ),
            C("FFN residual", "stream = a*stream + b*bf16(rms(f*w))   [R, H] f32"),
        ),
    },
)

PREFILL_SP_TARGET = Column(
    key="prefill_sp_target",
    dma={
        "Attention": (82, 62),
        "Pooler": (36, 6),
        "MoE": (0, 110),
        "Unpooler": (32, 14),
    },
    expert_layout="ep",
    dma_note="wo replicated (+56 MiB/layer, 3.4 GiB over 62 layers); no [R, H] bf16 partial",
    title="(3) Prefill SP + EP8 (MK_PREFILL_SP=1, default SP path)",
    subtitle=(
        "(2) with the attention tail changed: gated output e4m3 row a2a to the row owner, then",
        "o_proj on own rows against a replicated wo. MoE as (2)",
    ),
    block_rows=PREFILL_BLOCK_ROWS,
    n_ops=5,
    sections={
        "Attention": (
            C(
                "input_layernorm (own rows)",
                "stream [R/8, H] f32 -> RMS f32 -> a_in fp8 e4m3 + row scale f32",
            ),
            X(
                "⇄ a_in AG (fp8 + row scale)",
                7,
                "[R/8, H] fp8 e4m3 + [R/8, 1] f32 -> [R, H]",
            ),
            C(
                "qkvg (fp8 x fp8) + attention (by head, all R rows)",
                "a_in fp8 e4m3 @ wqkvg fp8 e4m3 -> f32 x row x ch scales",
                "q fp8 e4m3 (unit scale); K/V fp8 e4m3",
            ),
            C(
                "gated output quant (this rank's 16 heads)",
                "head-RMS f32 -> gated [R, 1024] fp8 e4m3 + row scale f32",
            ),
            X(
                "⇄ o a2a (by row, fp8 + row scale)",
                0.875,
                "[R, 1024] fp8 e4m3 -> owner [R/8, 8 x 1024] fp8 e4m3",
                "each rank gets its own rows with every rank's heads",
            ),
            C(
                "o_proj (own rows, replicated wo, fp8 x fp8)",
                "requant [R/8, H] -> fp8 e4m3, one row scale f32",
                "@ wo [8192, H] fp8 e4m3 -> f32 acc (no partial sum)",
                note="wo replicated: 64 MiB per layer instead of an 8 MiB row slice",
            ),
            C(
                "attention residual (own rows)",
                "stream = a*stream + b*rms(o)   [R/8, H] f32",
            ),
        ),
        "Pooler": (
            C(
                "pre-FFN norm + router (bf16 until GSM8K A/B)",
                "ffn f32 -> bf16 @ wrouter [H, 256] bf16 -> f32",
                "-> ids int32, tw f32 [R/8, 8]",
            ),
            C(
                "pooler (weight replicated, fp8 x fp8)",
                "ffn fp8 e4m3 + row scale @ wpool [H, He] fp8 e4m3 -> f32",
                note="no collective: wpool is replicated",
            ),
            C(
                "pre-expert norm (own rows)",
                "RMS f32 -> xe [R/8, He] fp8 e4m3 + row scale f32",
            ),
        ),
        "MoE": (
            X(
                "⇄ count exchange (AG)",
                0,
                "per-global-expert counts [256] int32 -> [8, 256] int32",
                "7 KiB per layer, independent of R",
            ),
            C(
                "sort (row, slot) pairs by global expert",
                "one segment per (source, expert), padded to 8 rows",
                "expert groups 16-row aligned",
            ),
            X(
                "⇄ dispatch a2a (remote DMA, expert-grouped)",
                4.91,
                "row = 1152 u32 (pack_rows): e4m3 bytes, f32 scale in word He/4, pad",
                "streamed in 4 local-expert chunks, overlapped with the expert gmm",
            ),
            C(
                "experts (EP, whole experts, gmm_v2 W4A8 / W8A8)",
                "Pallas unpack_rows -> bf16; gmm_v2 requantizes x, h to fp8 e4m3",
                "gmm tiles (256, 4096, 2048); silu(gate) * up in the w13 epilogue",
                "fp4 layers: NVFP4 e2m1, block-256 scales; acc f32",
                "out bf16 (feeds post-expert norm)",
            ),
            X(
                "⇄ combine a2a (remote DMA, bf16)",
                8.53,
                "y [cap, He] bf16, same 8-row segments back -> source's order",
            ),
            C(
                "combine (owner rows)",
                "post-expert RMS f32 x tw f32, sum f32 -> u f32",
                "-> u fp8 e4m3 + row scale",
            ),
        ),
        "Unpooler": (
            C(
                "unpooler (weight replicated, fp8 x fp8)",
                "u fp8 e4m3 @ wunpool [He, H] fp8 e4m3 -> f32",
                "-> f [R/8, H] bf16 (feeds post-FFN norm)",
                note="no collective: wunpool is replicated",
            ),
            C("post-FFN norm + residual (own rows)", "RMS f32; stream [R/8, H] f32"),
        ),
    },
)

DECODE_EP_GATHER = Column(
    key="decode_ep_gather",
    kv_note=KV_EITHER,
    dma={
        "Attention": (26, 64.25),
        "Pooler": (8, 0),
        "MoE": (0, 0),
        "Unpooler": (4, 64),
    },
    expert_layout="ep",
    dma_note="experts x worst-rank imbalance (1.09 at R=64); residual f32 in HBM (row-tiled)",
    title="(5) Decode EP, gather arm",
    subtitle=(
        "EP_ARM=gather. EP weights (32 whole experts/rank, replicated pool/unpool); rows replicated.",
        "Row-tiled megakernel + EPHook. Not wired in serving (from_model raises on ep)",
    ),
    block_rows=DECODE_BLOCK_ROWS,
    n_ops=6,
    sections={
        "Attention": (
            C(
                "input_layernorm + qkvg (all R rows)",
                "stream [R, H] f32 -> bf16 @ wqkvg [H, 2304] fp8 e4m3",
                note="EP forces the row-tiled program (row_tile 16)",
            ),
            C(
                "attention",
                "q bf16, score / acc f32; KV pool bf16 or fp8 e4m3",
            ),
            C(
                "o_proj (row-parallel)",
                "attn bf16 @ wo [1024, H] fp8 e4m3 -> partial [R, H] f32",
            ),
            X(
                "⇄ o RS (by column) + o AG",
                28,
                "RS: wo tile [R, H/8] bf16 (u32) -> owner; f32 sum",
                "AG: [R, H/8] bf16 (u32) -> [R, H]; direct < 48 rows",
            ),
            C("attention residual (all rows)", "stream [R, H] f32"),
        ),
        "Pooler": (
            C(
                "pooler (column slice of replicated wpool)",
                "ffn bf16 @ wpool[:, 512 cols] fp8 e4m3 -> [R, 512] bf16",
                note="reads 4 MiB of the 32 MiB replicated weight",
            ),
            X("⇄ pool AG", 7, "[R, He/8] bf16 (u32) -> [R, He] bf16"),
            C(
                "router (weight replicated, all R rows, under pool AG)",
                "ffn bf16 @ wrouter bf16 -> f32 -> top8 for all R rows",
                "keep only slots routed to my 32 experts",
            ),
            C("pre-expert norm", "xe [R, He] bf16 (VMEM, all rows)"),
        ),
        "MoE": (
            C(
                "experts (EP, local slots only; no dispatch)",
                "items of <= 32 rows per expert, x from VMEM",
                "NVFP4 e2m1 + bf16 block scale (16 or QBLOCK); acc f32",
                "post-expert norm f32 x w per slot; u[row] += f32",
            ),
            X(
                "⇄ u RS (f32 wire)",
                14,
                "u [R, He] f32 -> owner [R, He/8] f32; f32 sum",
            ),
            X("⇄ u AG", 7, "[R, He/8] bf16 (u32) -> [R, He] bf16"),
        ),
        "Unpooler": (
            C(
                "unpooler (column slice of replicated wunpool)",
                "u bf16 @ wunpool[:, 1024 cols] fp8 e4m3 -> [R, 1024] bf16",
                note="reads 4 MiB of 32 MiB",
            ),
            X("⇄ f AG", 14, "[R, H/8] bf16 (u32) -> [R, H] bf16"),
            C("FFN residual (all rows)", "stream [R, H] f32; next pre-attn norm fused"),
        ),
    },
)

DECODE_EP_GATHER_TARGET = Column(
    key="decode_ep_gather_target",
    dma={
        "Attention": (26, 64.25),
        "Pooler": (8, 0),
        "MoE": (0, 0),
        "Unpooler": (4, 64),
    },
    expert_layout="ep",
    dma_note="weights and residual unchanged (fp8 / fp4 already; residual stays f32)",
    title="(6) Decode EP gather, target dtypes",
    subtitle=(
        "Same dataflow as (5). Policy: fp8 e4m3 by default; f32 / bf16 only at norms",
        "and for tensors that feed a norm. Not implemented; needs numerics + GSM8K A/B",
    ),
    block_rows=DECODE_BLOCK_ROWS,
    n_ops=6,
    sections={
        "Attention": (
            C(
                "input_layernorm + qkvg (fp8 x fp8)",
                "RMS f32 -> a_in [R, H] fp8 e4m3 @ wqkvg fp8 e4m3 -> f32 acc",
            ),
            C(
                "attention",
                "head-RMS f32 -> q fp8 e4m3; K/V fp8 e4m3",
                "P fp8 e4m3; QK / PV acc f32",
            ),
            C(
                "o_proj (fp8 x fp8)",
                "gated fp8 e4m3 @ wo fp8 e4m3 -> partial [R, H] f32",
            ),
            X(
                "⇄ o RS + o AG (bf16: feeds post-attn norm)",
                28,
                "RS: [R, H/8] bf16, f32 sum; AG: [R, H/8] bf16 -> [R, H]",
            ),
            C("attention residual", "post-attn RMS f32; stream [R, H] f32"),
        ),
        "Pooler": (
            C(
                "pooler (column slice, fp8 x fp8)",
                "ffn fp8 e4m3 @ wpool[:, 512 cols] fp8 e4m3 -> [R, 512] bf16",
            ),
            X(
                "⇄ pool AG (bf16: feeds pre-expert norm)",
                7,
                "[R, He/8] bf16 -> [R, He] bf16 (unchanged)",
            ),
            C(
                "router (open) + pre-expert norm",
                "router input / logits bf16 / f32 until GSM8K A/B",
                "RMS f32 -> xe [R, He] fp8 e4m3",
            ),
        ),
        "MoE": (
            C(
                "experts (EP, local slots)",
                "xe fp8 e4m3 x NVFP4 e2m1 / fp8 e4m3 weights; acc f32",
                "h -> fp8 e4m3 into w2; post-expert RMS f32 x w; u += f32",
            ),
            X(
                "⇄ u RS (bf16 wire)",
                7,
                "u [R, He] -> owner [R, He/8] bf16, f32 sum",
            ),
            X(
                "⇄ u AG (fp8 + scale)",
                3.5,
                "[R, He/8] fp8 e4m3 -> [R, He] fp8 e4m3 (into unpooler)",
            ),
        ),
        "Unpooler": (
            C(
                "unpooler (column slice, fp8 x fp8)",
                "u fp8 e4m3 @ wunpool[:, 1024 cols] fp8 e4m3 -> [R, 1024] bf16",
            ),
            X(
                "⇄ f AG (bf16: feeds post-FFN norm)",
                14,
                "[R, H/8] bf16 -> [R, H] bf16 (unchanged)",
            ),
            C("FFN residual", "post-FFN RMS f32; stream [R, H] f32"),
        ),
    },
)

DECODE_EP_A2A = Column(
    key="decode_ep_a2a",
    kv_note=KV_EITHER,
    dma={
        "Attention": (26, 40.25),
        "Pooler": (36, 0),
        "MoE": (0, 40),
        "Unpooler": (32, 8),
    },
    expert_layout="ep",
    dma_note="full wpool + wunpool (+56 MiB fixed vs gather); HBM staging ep_lhs_g, ep_xs/xr/yb",
    title="(7) Decode SP + EP, a2a arm",
    subtitle=(
        "EP_ARM=a2a. Residual row-sharded (own R/8 rows); attention on all R rows by head; MoE",
        "behind ragged a2a in HBM 8-row segments. Row-tiled + EPHook; not wired in serving",
    ),
    block_rows=DECODE_BLOCK_ROWS,
    n_ops=5,
    sections={
        "Attention": (
            C(
                "pre-attn input (all R rows) + qkvg",
                "ep_lhs_g [R, H] bf16 (u32) HBM -> VMEM",
                "@ wqkvg [H, 2304] fp8 e4m3 -> f32",
            ),
            C("attention", "q bf16, score / acc f32; KV pool bf16 or fp8 e4m3"),
            C(
                "o_proj (row-parallel)",
                "partial [R, H] f32, one wo column tile at a time",
            ),
            X(
                "⇄ o RS (by row; no AG)",
                14,
                "tile block [R/8, H/8] bf16 (u32) -> owner; f32 sum",
            ),
            C(
                "attention residual (own rows)",
                "residual + pre-FFN norm on own R/8 rows, f32",
            ),
        ),
        "Pooler": (
            C(
                "router (weight replicated)",
                "ffn bf16 @ wrouter bf16 -> f32 logits for all R rows",
                "top8 on own rows only (other rows' logits unused)",
            ),
            C(
                "pooler (replicated, full read)",
                "own rows bf16 @ wpool [H, He] fp8 e4m3 (all 32 MiB)",
                "-> pre-expert norm -> xe [R/8, He] bf16",
            ),
        ),
        "MoE": (
            X(
                "⇄ count exchange",
                7 / 64,
                "[256] int32 to 7 peers (1 KiB each, per layer)",
            ),
            X(
                "⇄ dispatch ragged a2a (HBM -> HBM)",
                9.9,
                "ep_xs [R/8*8, He] bf16 (u32) -> peers' ep_xr, 8-row segs",
                "one copy per (row, slot): no dedup",
            ),
            C(
                "experts (EP, whole experts)",
                "rows ep_xr HBM -> VMEM items (<= 32 rows); acc f32",
                "NVFP4 e2m1 / fp8 e4m3 weights, as (5)",
                note="an expert with n pairs re-streams its weights ceil(n/32) times",
            ),
            X(
                "⇄ y return",
                7,
                "per slot 1-row remote DMA, bf16 (u32) -> source ep_yb",
            ),
            C(
                "combine (own rows)",
                "post-expert norm f32 per slot, x w, sum f32",
                "-> u [R/8, He] bf16",
            ),
        ),
        "Unpooler": (
            C(
                "unpooler (replicated, full read) + residual",
                "u bf16 @ wunpool [He, H] fp8 e4m3 (all 32 MiB)",
                "-> post-FFN norm; residual own rows f32",
            ),
            X(
                "⇄ next-layer input AG",
                14,
                "own rows pre_attn-normed bf16 (u32) -> all ep_lhs_g",
                "pair relay at tp 8",
            ),
        ),
    },
)

DECODE_EP_A2A_TARGET = Column(
    key="decode_ep_a2a_target",
    dma={
        "Attention": (26, 24.25),
        "Pooler": (36, 0),
        "MoE": (0, 28),
        "Unpooler": (32, 8),
    },
    expert_layout="ep",
    dma_note="weights unchanged; fp8 halves the ep_lhs_g / ep_xs / ep_xr HBM staging",
    title="(8) Decode SP + EP a2a, target dtypes",
    subtitle=(
        "Same dataflow as (7). Policy: fp8 e4m3 by default; f32 / bf16 only at norms",
        "and for tensors that feed a norm. Not implemented; needs numerics + GSM8K A/B",
    ),
    block_rows=DECODE_BLOCK_ROWS,
    n_ops=5,
    sections={
        "Attention": (
            C(
                "pre-attn input (fp8) + qkvg (fp8 x fp8)",
                "ep_lhs_g [R, H] fp8 e4m3 + scale -> VMEM",
                "@ wqkvg fp8 e4m3 -> f32 acc",
            ),
            C(
                "attention",
                "head-RMS f32 -> q fp8 e4m3; K/V fp8 e4m3",
                "P fp8 e4m3; QK / PV acc f32",
            ),
            C(
                "o_proj (fp8 x fp8)",
                "gated fp8 e4m3 @ wo fp8 e4m3 -> partial [R, H] f32",
            ),
            X(
                "⇄ o RS (bf16: feeds post-attn norm)",
                14,
                "[R/8, H/8] bf16 blocks -> owner, f32 sum",
            ),
            C(
                "attention residual (own rows)",
                "post-attn RMS f32; stream [R/8, H] f32",
            ),
        ),
        "Pooler": (
            C(
                "router (open)",
                "router input / logits bf16 / f32 until GSM8K A/B",
            ),
            C(
                "pooler (replicated, fp8 x fp8)",
                "ffn fp8 e4m3 @ wpool [H, He] fp8 e4m3 -> f32 acc",
                "pre-expert RMS f32 -> xe [R/8, He] fp8 e4m3 + scale",
            ),
        ),
        "MoE": (
            X("⇄ count exchange", 7 / 64, "[256] int32 to 7 peers"),
            X(
                "⇄ dispatch ragged a2a (fp8 + scale)",
                4.9,
                "xe [R/8*8, He] fp8 e4m3 + row scale f32 -> peers' ep_xr",
            ),
            C(
                "experts (EP, whole experts)",
                "xe fp8 e4m3 x NVFP4 e2m1 / fp8 e4m3; h -> fp8 e4m3; acc f32",
                "y -> bf16 (feeds post-expert norm)",
            ),
            X(
                "⇄ y return (bf16)",
                7,
                "per slot bf16 -> source ep_yb",
            ),
            C(
                "combine (own rows)",
                "post-expert RMS f32 x w, sum f32 -> u [R/8, He] fp8 e4m3",
            ),
        ),
        "Unpooler": (
            C(
                "unpooler (replicated, fp8 x fp8) + residual",
                "u fp8 e4m3 @ wunpool [He, H] fp8 e4m3 -> f32 acc",
                "post-FFN RMS f32; residual own rows f32",
            ),
            X(
                "⇄ next-layer input AG (fp8 + scale)",
                7,
                "own rows pre_attn-normed fp8 e4m3 + scale -> all ep_lhs_g",
            ),
        ),
    },
)

DECODE_EP_A2A_SLICED = Column(
    key="decode_ep_a2a_sliced",
    dma={
        "Attention": (26, 24.25),
        "Pooler": (8, 16),
        "MoE": (0, 28),
        "Unpooler": (4, 8),
    },
    expert_layout="ep",
    dma_note="column slices of wpool / wunpool (4 + 4 MiB): TP's 38 MiB; ffn AG pad [R, H] e4m3 in HBM",
    title="(9) Decode SP + EP target, pool / unpool column-sliced",
    subtitle=(
        "(8) with pool / unpool on a 1/8 column slice of the replicated weights (as (5)) instead",
        "of in full on own rows: + ffn AG, pool a2a, u AG, f a2a. Slicing: a2a arm at R <= MK_EP_COL_POOL_MAX_ROWS",
    ),
    block_rows=DECODE_BLOCK_ROWS,
    n_ops=8,
    sections={
        "Attention": (
            X(
                "⇄ a_in AG (fp8 + row scale)",
                7,
                "own rows pre_attn-normed [R/8, H] fp8 e4m3 + f32 -> [R, H]",
            ),
            C(
                "qkvg (fp8 x fp8) + attention (by head, all R rows)",
                "a_in fp8 e4m3 @ wqkvg fp8 e4m3 -> f32 acc",
                "q fp8 e4m3; K/V fp8 e4m3; P fp8 e4m3; acc f32",
            ),
            C(
                "o_proj (row block of the replicated wo, fp8 x fp8)",
                "gated fp8 e4m3 @ wo[1024 rows, H] fp8 e4m3 -> partial [R, H] f32",
                note="the same replicated wo SP prefill reads; 8 of its 64 MiB",
            ),
            X(
                "⇄ o RS (by row, bf16: feeds post-attn norm)",
                14,
                "[R/8, H/8] bf16 blocks -> owner, f32 sum",
            ),
            C(
                "attention residual (own rows)",
                "post-attn RMS f32; stream [R/8, H] f32",
            ),
        ),
        "Pooler": (
            C(
                "pre-FFN norm + router (own rows, open)",
                "ffn bf16 @ wrouter bf16 -> f32 -> top8 [R/8, 8]",
                "ffn -> fp8 e4m3 + row scale f32",
            ),
            X(
                "⇄ ffn AG (fp8 + row scale)",
                7,
                "[R/8, H] fp8 e4m3 + [R/8, 1] f32 -> [R, H]",
            ),
            C(
                "pooler (column slice, fp8 x fp8, all R rows)",
                "ffn fp8 e4m3 @ wpool[:, 512 cols] fp8 e4m3 -> [R, He/8] f32",
                note="reads 4 MiB of the 32 MiB replicated weight",
            ),
            X(
                "⇄ pool a2a (by row, bf16: feeds pre-expert norm)",
                0.875,
                "[R, He/8] bf16 -> owner [R/8, 8 x He/8] bf16",
            ),
            C(
                "pre-expert norm (own rows)",
                "RMS f32 -> xe [R/8, He] fp8 e4m3 + row scale f32",
            ),
        ),
        "MoE": (
            X("⇄ count exchange", 7 / 64, "[256] int32 to 7 peers"),
            X(
                "⇄ dispatch a2a (remote DMA, fp8 u32-packed)",
                5.55,
                "row = 1152 u32 (pack_rows): e4m3 bytes, f32 scale, pad",
                "one segment per (source, destination), SEG=8-row DMAs",
                note="per-(source, expert) SEG=8 segments would pad 7.17x at R=64 (28 KiB/row)",
            ),
            C(
                "experts (EP, whole experts)",
                "xe fp8 e4m3 x NVFP4 e2m1 / fp8 e4m3; h -> fp8 e4m3; acc f32",
                "y -> bf16 (feeds post-expert norm)",
            ),
            X(
                "⇄ y return (bf16)",
                7,
                "per slot bf16 -> source ep_yb",
            ),
            C(
                "combine (own rows)",
                "post-expert RMS f32 x w, sum f32 -> u [R/8, He] fp8 e4m3 + scale",
            ),
        ),
        "Unpooler": (
            X(
                "⇄ u AG (fp8 + row scale)",
                3.5,
                "[R/8, He] fp8 e4m3 + [R/8, 1] f32 -> [R, He]",
            ),
            C(
                "unpooler (column slice, fp8 x fp8, all R rows)",
                "u fp8 e4m3 @ wunpool[:, 1024 cols] fp8 e4m3 -> [R, H/8] f32",
                note="reads 4 MiB of 32 MiB",
            ),
            X(
                "⇄ f a2a (by row, bf16: feeds post-FFN norm)",
                1.75,
                "[R, H/8] bf16 -> owner [R/8, 8 x H/8] bf16",
            ),
            C(
                "post-FFN norm + residual (own rows)",
                "RMS f32; stream [R/8, H] f32; next pre-attn norm fused",
            ),
        ),
    },
)

# How each comm box's KiB/row is derived: per rank, per token row, the 7/8 sent off-rank.
# "1/8 own rows": a rank sends only for its R/8 rows, averaged over all R rows.
FORMULAS = {
    (
        "prefill_tp",
        "⇄ o RS + AG (= all-reduce)",
    ): "RS 7/8 x H x 2B + AG 7 x (H/8) x 2B\n= 7/8 x 8192 x 2 + 7 x 1024 x 2 = 14 + 14 = 28 KiB",
    ("prefill_tp", "⇄ xe AG"): "7 x (He/8) x 2B = 7 x 512 x 2 = 7 KiB",
    (
        "prefill_tp",
        "⇄ routing psum (compiler workaround)",
    ): "[8 slots, 2] x 2B = 32 B (all-reduce)",
    (
        "prefill_tp",
        "⇄ y direct-write RS (remote DMA inside GMM)",
    ): "8 slots x He x 2B x 7/8 = 8 x 4096 x 2 x 7/8 = 56 KiB",
    ("prefill_tp", "⇄ y AG (token dim)"): "7/8 x He x 2B = 7/8 x 4096 x 2 = 7 KiB",
    ("prefill_tp", "⇄ u AG"): "7 x (H/8) x 2B = 7 x 1024 x 2 = 14 KiB",
    (
        "prefill_sp",
        "⇄ a_in AG (fp8 + row scale)",
    ): "7/8 x H x 1B = 7/8 x 8192 x 1 = 7 KiB (+ 4 B row scale)",
    (
        "prefill_sp",
        "⇄ o RS (psum_scatter, bf16: feeds post-attn norm)",
    ): "7/8 x H x 2B = 7/8 x 8192 x 2 = 14 KiB",
    (
        "prefill_sp",
        "⇄ count exchange (AG)",
    ): "7 peers x 256 x 4B = 7 KiB per layer, not per row",
    (
        "prefill_sp",
        "⇄ dispatch a2a (remote DMA, expert-grouped)",
    ): "1/8 own rows x 8 slots x 1152 u32 x 4B x 7/8 = 4.0 KiB\nx 1.22 (8-row chunks per expert, simulated at R=4096) = 4.91 KiB",
    (
        "prefill_sp",
        "⇄ combine a2a (remote DMA, bf16)",
    ): "1/8 own rows x 8 slots x He x 2B x 7/8 = 7 KiB\nx 1.22 (8-row chunks per expert) = 8.53 KiB",
    (
        "prefill_sp_target",
        "⇄ a_in AG (fp8 + row scale)",
    ): "7/8 x H x 1B = 7/8 x 8192 x 1 = 7 KiB (+ 4 B row scale)",
    (
        "prefill_sp_target",
        "⇄ o a2a (by row, fp8 + row scale)",
    ): "7/8 x (H/8) x 1B = 7/8 x 1024 x 1 = 0.875 KiB (+ 3.5 B row scale)",
    (
        "prefill_sp_target",
        "⇄ count exchange (AG)",
    ): "7 peers x 256 x 4B = 7 KiB per layer, not per row",
    (
        "prefill_sp_target",
        "⇄ dispatch a2a (remote DMA, expert-grouped)",
    ): "1/8 own rows x 8 slots x 1152 u32 x 4B x 7/8 = 4.0 KiB\nx 1.22 (8-row chunks per expert, simulated at R=4096) = 4.91 KiB",
    (
        "prefill_sp_target",
        "⇄ combine a2a (remote DMA, bf16)",
    ): "1/8 own rows x 8 slots x He x 2B x 7/8 = 7 KiB\nx 1.22 (8-row chunks per expert) = 8.53 KiB",
    (
        "decode_tp",
        "⇄ o RS (column-streamed) + o AG",
    ): "RS 7/8 x H x 2B + AG 7 x (H/8) x 2B\n= 7/8 x 8192 x 2 + 7 x 1024 x 2 = 14 + 14 = 28 KiB",
    (
        "decode_tp",
        "⇄ pool AG (started early, overlaps router)",
    ): "7 x (He/8) x 2B = 7 x 512 x 2 = 7 KiB",
    (
        "decode_tp",
        "⇄ y RS (by rows)",
    ): "8 slots x (He x 1B + 128 x 4B scales) x 7/8\n= 8 x 4.5 KiB x 7/8 = 31.5 KiB",
    ("decode_tp", "⇄ u AG"): "7/8 x He x 2B = 7/8 x 4096 x 2 = 7 KiB",
    ("decode_tp", "⇄ f AG"): "7 x (H/8) x 2B = 7 x 1024 x 2 = 14 KiB",
    (
        "decode_ep_gather",
        "⇄ o RS (by column) + o AG",
    ): "RS 7/8 x H x 2B + AG 7 x (H/8) x 2B\n= 7/8 x 8192 x 2 + 7 x 1024 x 2 = 14 + 14 = 28 KiB",
    ("decode_ep_gather", "⇄ pool AG"): "7 x (He/8) x 2B = 7 x 512 x 2 = 7 KiB",
    (
        "decode_ep_gather",
        "⇄ u RS (f32 wire)",
    ): "7/8 x He x 4B = 7/8 x 4096 x 4 = 14 KiB",
    ("decode_ep_gather", "⇄ u AG"): "7 x (He/8) x 2B = 7 x 512 x 2 = 7 KiB",
    ("decode_ep_gather", "⇄ f AG"): "7 x (H/8) x 2B = 7 x 1024 x 2 = 14 KiB",
    (
        "decode_ep_gather_target",
        "⇄ o RS + o AG (bf16: feeds post-attn norm)",
    ): "RS 7/8 x H x 2B + AG 7 x (H/8) x 2B\n= 7/8 x 8192 x 2 + 7 x 1024 x 2 = 14 + 14 = 28 KiB",
    (
        "decode_ep_gather_target",
        "⇄ pool AG (bf16: feeds pre-expert norm)",
    ): "7 x (He/8) x 2B = 7 x 512 x 2 = 7 KiB",
    (
        "decode_ep_gather_target",
        "⇄ u RS (bf16 wire)",
    ): "7/8 x He x 2B = 7/8 x 4096 x 2 = 7 KiB",
    (
        "decode_ep_gather_target",
        "⇄ u AG (fp8 + scale)",
    ): "7 x (He/8) x 1B = 7 x 512 x 1 = 3.5 KiB",
    (
        "decode_ep_gather_target",
        "⇄ f AG (bf16: feeds post-FFN norm)",
    ): "7 x (H/8) x 2B = 7 x 1024 x 2 = 14 KiB",
    (
        "decode_ep_a2a",
        "⇄ o RS (by row; no AG)",
    ): "7/8 x H x 2B = 7/8 x 8192 x 2 = 14 KiB",
    (
        "decode_ep_a2a",
        "⇄ count exchange",
    ): "7 peers x 256 x 4B = 7 KiB per layer; / R=64 = 0.11 KiB",
    (
        "decode_ep_a2a",
        "⇄ dispatch ragged a2a (HBM -> HBM)",
    ): "1/8 own rows x 8 slots x He x 2B x 7/8 = 7 KiB\nx 1.41 (8-row segment padding, simulated at R=64) = 9.9 KiB",
    (
        "decode_ep_a2a",
        "⇄ y return",
    ): "1/8 own rows x 8 slots x He x 2B x 7/8\n= 1/8 x 8 x 4096 x 2 x 7/8 = 7 KiB",
    (
        "decode_ep_a2a",
        "⇄ next-layer input AG",
    ): "7/8 x H x 2B = 7/8 x 8192 x 2 = 14 KiB",
    (
        "decode_ep_a2a_target",
        "⇄ o RS (bf16: feeds post-attn norm)",
    ): "7/8 x H x 2B = 7/8 x 8192 x 2 = 14 KiB",
    (
        "decode_ep_a2a_target",
        "⇄ count exchange",
    ): "7 peers x 256 x 4B = 7 KiB per layer; / R=64 = 0.11 KiB",
    (
        "decode_ep_a2a_target",
        "⇄ dispatch ragged a2a (fp8 + scale)",
    ): "1/8 own rows x 8 slots x He x 1B x 7/8 = 3.5 KiB\nx 1.41 (8-row segment padding, simulated at R=64) = 4.9 KiB",
    (
        "decode_ep_a2a_target",
        "⇄ y return (bf16)",
    ): "1/8 own rows x 8 slots x He x 2B x 7/8\n= 1/8 x 8 x 4096 x 2 x 7/8 = 7 KiB",
    (
        "decode_ep_a2a_target",
        "⇄ next-layer input AG (fp8 + scale)",
    ): "7/8 x H x 1B = 7/8 x 8192 x 1 = 7 KiB",
    (
        "decode_ep_a2a_sliced",
        "⇄ a_in AG (fp8 + row scale)",
    ): "7/8 x H x 1B = 7/8 x 8192 x 1 = 7 KiB (+ 3.5 B row scale)",
    (
        "decode_ep_a2a_sliced",
        "⇄ o RS (by row, bf16: feeds post-attn norm)",
    ): "7/8 x H x 2B = 7/8 x 8192 x 2 = 14 KiB",
    (
        "decode_ep_a2a_sliced",
        "⇄ ffn AG (fp8 + row scale)",
    ): "7/8 x H x 1B = 7/8 x 8192 x 1 = 7 KiB (+ 3.5 B row scale)",
    (
        "decode_ep_a2a_sliced",
        "⇄ pool a2a (by row, bf16: feeds pre-expert norm)",
    ): "7/8 x (He/8) x 2B = 7/8 x 512 x 2 = 0.875 KiB",
    (
        "decode_ep_a2a_sliced",
        "⇄ count exchange",
    ): "7 peers x 256 x 4B = 7 KiB per layer; / R=64 = 0.11 KiB",
    (
        "decode_ep_a2a_sliced",
        "⇄ dispatch a2a (remote DMA, fp8 u32-packed)",
    ): "1/8 own rows x 8 slots x 1152 u32 x 4B x 7/8 = 3.94 KiB\nx 1.41 (8-row segment padding, simulated at R=64) = 5.55 KiB",
    (
        "decode_ep_a2a_sliced",
        "⇄ y return (bf16)",
    ): "1/8 own rows x 8 slots x He x 2B x 7/8\n= 1/8 x 8 x 4096 x 2 x 7/8 = 7 KiB",
    (
        "decode_ep_a2a_sliced",
        "⇄ u AG (fp8 + row scale)",
    ): "7/8 x He x 1B = 7/8 x 4096 x 1 = 3.5 KiB (+ 3.5 B row scale)",
    (
        "decode_ep_a2a_sliced",
        "⇄ f a2a (by row, bf16: feeds post-FFN norm)",
    ): "7/8 x (H/8) x 2B = 7/8 x 1024 x 2 = 1.75 KiB",
}


def _with_formulas(col: Column) -> Column:
    """Attach FORMULAS to col's comm boxes; a comm box without one raises KeyError."""
    sections = {
        name: tuple(
            b._replace(formula=FORMULAS[(col.key, b.title)]) if b.kind == "comm" else b
            for b in boxes
        )
        for name, boxes in col.sections.items()
    }
    return col._replace(sections=sections)


COLUMNS = tuple(
    _with_formulas(c)
    for c in (
        PREFILL_TP,
        PREFILL_SP,
        PREFILL_SP_TARGET,
        DECODE_TP,
        DECODE_EP_GATHER,
        DECODE_EP_GATHER_TARGET,
        DECODE_EP_A2A,
        DECODE_EP_A2A_TARGET,
        DECODE_EP_A2A_SLICED,
    )
)

# ---------------------------------------------------------------- layout
COL_W = 560
BOX_W = 500
GAP_X = 30
LINE_H = 15
TITLE_H = 18
PAD = 8
BOX_GAP = 14
SEC_PAD = 14
SEC_LABEL_W = 34
HEADER_H = 124
TOP_H = 90
FONT = "'Inter','Helvetica Neue','Arial',sans-serif"
MONO = "'JetBrains Mono','DejaVu Sans Mono','Menlo',monospace"


def _box_h(b: Box) -> int:
    n_formula = len(b.formula.split("\n")) if b.formula else 0
    return (
        PAD * 2
        + TITLE_H
        + LINE_H * (len(b.lines) + n_formula)
        + (LINE_H if b.note else 0)
    )


def _section_h(boxes: tuple[Box, ...]) -> int:
    return SEC_PAD * 2 + sum(_box_h(b) for b in boxes) + BOX_GAP * (len(boxes) - 1) + 46


def _esc(s: str) -> str:
    return html.escape(s, quote=False)


def _rich(text: str, x: float, y: float, size: float, mono: bool, fill: str) -> str:
    """Text with dtype tokens colored and bold."""
    parts = []
    for tok in _DTYPE_RE.split(text):
        if not tok:
            continue
        if tok in DTYPE_COLORS:
            parts.append(
                f'<tspan fill="{DTYPE_COLORS[tok]}" font-weight="700">{_esc(tok)}</tspan>'
            )
        else:
            parts.append(_esc(tok))
    family = MONO if mono else FONT
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" xml:space="preserve">{"".join(parts)}</text>'
    )


def _mib(kib: float, rows: int) -> float:
    return kib * rows / 1024


def _fmt_kib(kib: float) -> str:
    if kib < 0.1:
        return f"{kib * 1024:.0f} B"
    return f"{kib:g} KiB" if kib == int(kib) else f"{kib:.2f} KiB".replace(".00", "")


def _draw_box(b: Box, x: float, y: float, color: str, comm_fill: str, rows: int) -> str:
    h = _box_h(b)
    out = []
    if b.kind == "comm":
        out.append(
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{h}" rx="5" fill="{comm_fill}" '
            f'stroke="{color}" stroke-width="1.4" stroke-dasharray="5 3"/>'
        )
        out.append(
            f'<rect x="{x + 10}" y="{y + PAD + TITLE_H - 3}" width="120" height="3" fill="{color}"/>'
        )
        out.append(
            f'<text x="{x + BOX_W - 10}" y="{y + PAD + 13}" text-anchor="end" font-family="{FONT}" '
            f'font-size="12.5" font-weight="700" fill="#24292f">{_fmt_kib(b.kib)}/row · '
            f"{_mib(b.kib, rows):.3g} MiB</text>"
        )
    else:
        out.append(
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{h}" rx="4" fill="#ffffff" '
            f'stroke="#d0d7de" stroke-width="1"/>'
        )
        out.append(
            f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2" fill="{color}"/>'
        )
    out.append(
        f'<text x="{x + 12}" y="{y + PAD + 13}" font-family="{FONT}" font-size="13" '
        f'font-weight="700" fill="#1f2328">{_esc(b.title)}</text>'
    )
    ty = y + PAD + TITLE_H + 11
    for line in b.lines:
        out.append(_rich(line, x + 14, ty, 11, mono=True, fill="#24292f"))
        ty += LINE_H
    for i, line in enumerate(b.formula.split("\n") if b.formula else ()):
        prefix = "= " if i == 0 else "  "
        out.append(
            f'<text x="{x + 14}" y="{ty}" font-family="{MONO}" font-size="11" font-weight="700" '
            f'fill="#0a3069" xml:space="preserve">{_esc(prefix + line)}</text>'
        )
        ty += LINE_H
    if b.note:
        out.append(_rich(b.note, x + 14, ty, 10.5, mono=False, fill="#57606a"))
    return "\n".join(out)


def _arrow(x: float, y0: float, y1: float) -> str:
    return (
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1 - 4}" stroke="#8c959f" stroke-width="1.2"/>'
        f'<path d="M{x - 4},{y1 - 6} L{x},{y1} L{x + 4},{y1 - 6} Z" fill="#8c959f"/>'
    )


def _legend(x: float, y: float) -> str:
    out = [
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="12.5" font-weight="700" '
        f'fill="#1f2328">dtype colors:</text>'
    ]
    cx = x + 92
    for name, color in DTYPE_COLORS.items():
        w = 10 + 7.2 * len(name)
        out.append(
            f'<rect x="{cx}" y="{y - 13}" width="{w}" height="18" rx="9" fill="{color}" '
            f'fill-opacity="0.12" stroke="{color}"/>'
        )
        out.append(
            f'<text x="{cx + w / 2}" y="{y}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="11.5" font-weight="700" fill="{color}">{name}</text>'
        )
        cx += w + 10
    out.append(
        f'<text x="{x}" y="{y + 22}" font-family="{FONT}" font-size="12" fill="#57606a">'
        f"Solid box = compute; dashed box = communication (⇄). Top right: wire bytes per rank "
        f"per row · per layer per block</text>"
    )
    return "\n".join(out)


def render(columns: tuple[Column, ...]) -> str:
    sec_names = list(SECTION_COLORS)
    sec_h = {s: max(_section_h(c.sections[s]) for c in columns) for s in sec_names}
    width = SEC_LABEL_W + len(columns) * COL_W + (len(columns) - 1) * GAP_X + 40
    body_h = sum(sec_h.values()) + 24 * (len(sec_names) - 1)
    height = TOP_H + HEADER_H + body_h + 230
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="20" y="28" font-family="{FONT}" font-size="17" font-weight="700" '
        f'fill="#1f2328">Muse Spark per-layer communication (TP 8)</text>',
        f'<text x="20" y="46" font-family="{MONO}" font-size="12" fill="#57606a">'
        f"H=8192 He=4096 E=256 top_k=8 Q128/KV16 hd64</text>",
        _legend(20, 70),
    ]
    x0 = SEC_LABEL_W + 20
    for ci, col in enumerate(columns):
        cx = x0 + ci * (COL_W + GAP_X)
        total = sum(b.kib for s in sec_names for b in col.sections[s])
        hy = TOP_H + 26
        out.append(
            f'<text x="{cx}" y="{hy}" font-family="{FONT}" font-size="16" font-weight="700" '
            f'fill="#1f2328">{_esc(col.title)}</text>'
        )
        for i, s in enumerate(col.subtitle):
            out.append(_rich(s, cx, hy + 20 + 16 * i, 11.5, mono=False, fill="#57606a"))
        out.append(
            f'<text x="{cx}" y="{hy + 20 + 16 * len(col.subtitle) + 4}" font-family="{FONT}" '
            f'font-size="12.5" font-weight="700" fill="#cf222e">'
            f"{total:.4g} KiB/row · {_mib(total, col.block_rows):.4g} MiB/layer "
            f"@R={col.block_rows} · {col.n_ops} collectives/layer</text>"
        )
        wt = sum(w for w, _ in col.dma.values())
        per_row = sum(a for _, a in col.dma.values())
        ex = expert_mib(col.block_rows, col.expert_layout)
        tot = wt + ex + per_row * col.block_rows / 1024
        dy = hy + 20 + 16 * len(col.subtitle) + 22
        out.append(
            f'<text x="{cx}" y="{dy}" font-family="{FONT}" font-size="12.5" font-weight="700" '
            f'fill="#8250df">HBM DMA = {wt:,.0f} MiB + E(R) + {per_row:,.4g} KiB x R '
            f"= ~{tot:,.0f} MiB/layer @R={col.block_rows} (E={ex:,.0f}) + KV</text>"
        )
        out.append(_rich(col.dma_note, cx, dy + 16, 11, mono=False, fill="#57606a"))
    y = TOP_H + HEADER_H + 12
    for s in sec_names:
        color, bg, comm_fill = SECTION_COLORS[s]
        h = sec_h[s]
        out.append(
            f'<rect x="12" y="{y}" width="{width - 24}" height="{h}" rx="6" fill="{bg}"/>'
        )
        out.append(
            f'<text x="30" y="{y + h / 2}" transform="rotate(-90 30 {y + h / 2})" '
            f'text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="700" '
            f'fill="{color}">{s}</text>'
        )
        for ci, col in enumerate(columns):
            bx = x0 + ci * (COL_W + GAP_X)
            by = y + SEC_PAD
            for bi, b in enumerate(col.sections[s]):
                out.append(_draw_box(b, bx, by, color, comm_fill, col.block_rows))
                by += _box_h(b)
                if bi < len(col.sections[s]) - 1:
                    out.append(_arrow(bx + BOX_W / 2, by, by + BOX_GAP))
                by += BOX_GAP
            sec_kib = sum(b.kib for b in col.sections[s])
            wt, per_row = col.dma[s]
            fixed = (
                f"E(R) = {expert_mib(col.block_rows, col.expert_layout):,.0f} MiB"
                if s == "MoE"
                else f"{wt:,.0f} MiB"
            )
            foot = [
                (
                    f"comm: {sec_kib:.4g} KiB/row · {_mib(sec_kib, col.block_rows):.4g} MiB",
                    "#57606a",
                ),
                (f"HBM DMA: weights {fixed} + act {per_row:,.4g} KiB x R", "#8250df"),
            ]
            if s == "Attention":
                foot.append((col.kv_note, "#8250df"))
            for i, (text, fill) in enumerate(reversed(foot)):
                out.append(
                    f'<text x="{bx + BOX_W}" y="{y + h - 8 - 14 * i}" text-anchor="end" '
                    f'font-family="{FONT}" font-size="11" font-weight="600" fill="{fill}">'
                    f"{_esc(text)}</text>"
                )
        y += h + 24
    notes = (
        "R = rows this step: prefill = chunk tokens (block size 4096); decode = B rounded up to 8",
        "R/8 = rows owned by one rank; n = 8; H = 8192; He = 4096; I = 4096 (512 per rank)",
        "pooler / unpooler = pre_expert_proj (H->He) / post_expert_proj (He->H)",
        "KiB/row = bytes one rank sends to the other 7 per token row (AG: own block x7; "
        "RS: 7/8 of partial)",
        "MiB = KiB/row x R / 1024, one layer, one rank",
        "HBM DMA = fixed weights + E(R) experts + per-row KiB x R, per layer per rank (estimate: each HBM",
        "  tensor written once, read once); E(R) = 780 MiB x (1 - (31/32)^R), x EP worst-rank imbalance;",
        "  tile re-reads and XLA fusion not modeled; fp4 experts at QBLOCK=256 (3.05 MiB per expert-TP slice)",
        "Decode columns (4)-(9) all use R=64 (spec decoding), so their numbers compare directly",
        "decode packs bf16 pairs into u32 words (common/packing.py)",
        "FP8_GATHER only affects the xe / u AG of (1); (2) rejects it",
        "(2) requires MOE_MODE=ep, which serving still rejects",
    )
    for i, line in enumerate(notes):
        out.append(_rich(line, 20, y + 20 + 18 * i, 11.5, mono=False, fill="#57606a"))
    out.append("</svg>\n")
    return "\n".join(out)


def main() -> None:
    (OUT_DIR / "comm_overview.svg").write_text(render(COLUMNS), encoding="utf-8")
    for col in COLUMNS:
        (OUT_DIR / f"comm_{col.key}.svg").write_text(render((col,)), encoding="utf-8")


if __name__ == "__main__":
    main()
