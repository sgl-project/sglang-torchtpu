# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decoder stack as one Pallas kernel (one decode row).

Layer 0 (KDA attention + dense MLP) is straight-line code; layers
``1..L-1`` run under ``pl.loop`` and pick KDA or MLA attention with
``lax.cond`` from SMEM layer tables, so 93 layers compile like 13. Weights
come in per-family stacks with a leading layer axis (see ``StackWeights``
for the order); streamed matmuls share named VMEM double buffers.

Per layer (sglang's attention-residual dataflow, see stages/attn_res.py):
  a_in = input_norm(agg(h));  bank write on write layers (old nvb first)
  attn = KDA | MLA (a_in);    P2 = attn on write layers, else bf16(h + attn)
  out  = bf16(MLP(post_norm(agg(P2))) + P2)
Final: final_norm(agg(h; output_attn_res)).

Schedule: every small per-layer tensor lives in one f32 slab per layer
(``layer_vec``, see ``VecLayout``), fetched with one DMA into a slot picked by
layer parity. The slab, the attention auxiliaries (KDA f_b or MLA w_kc/w_vc)
and the KDA state rows of layer ``l + 1`` are started from inside layer
``l``'s MoE, once its routed-expert DMAs are queued; KDA state write-back is
asynchronous and drained at that same point.
"""

import math

import jax
import jax.numpy as jnp
import msgspec
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import (
    K3MK_COLLECTIVE_ID,
    K3Dims,
    MoeLayout,
    Relax,
    Topology,
)
from sglang_tpu.k3_megakernel.kernel import collectives
from sglang_tpu.k3_megakernel.kernel.collectives import (
    AllGather,
    AllGatherSpec,
    AllReduce,
    RowScatterAllReduce,
    RowScatterSpec,
    entry_barrier,
)
from sglang_tpu.k3_megakernel.kernel.stages.arena import Arena, storage_bytes
from sglang_tpu.k3_megakernel.kernel.stages.attn_res import (
    MixOpts,
    aggregate_packed,
    aggregate_rows,
    bank_write,
    bf16,
    rmsnorm,
    row_scores,
)
from sglang_tpu.k3_megakernel.kernel.stages.common import (
    AllReduceProjection,
    StreamPool,
    flat_rows,
    phase,
    stream_chunk,
    trace_phases,
    tree_when,
)
from sglang_tpu.k3_megakernel.kernel.stages.dense_mlp import DenseMlpStage
from sglang_tpu.k3_megakernel.kernel.stages.head import (
    GreedyHead,
    HeadSpec,
    head_chunk,
)
from sglang_tpu.k3_megakernel.kernel.stages.kda import HEAD_DIM, KdaStage
from sglang_tpu.k3_megakernel.kernel.stages.mla import (
    PAGE,
    REPLICATED,
    ROW,
    MlaStage,
    SplitW1Projection,
    W1Split,
)
from sglang_tpu.k3_megakernel.kernel.stages.moe import _RING, MoeStage
from sglang_tpu.k3_megakernel.weights import DENSE_FP8_FAMILIES

BANK_ROWS = 8
NORMS = (
    "self_attention_res_proj",
    "self_attention_res_norm",
    "input_layernorm",
    "mlp_res_proj",
    "mlp_res_norm",
    "post_attention_layernorm",
)
_KDA, _MLA = 0, 1

# Per-rank kernel inputs, in order: (name, memory space).
SMEM_INPUTS = ("tables", "meta", "pages")
VMEM_INPUTS = ("x0", "final_w")
HEAD_INPUTS = ("embed", "lm_head")  # HBM, StackKernel(fused_head=True) only
# Kernel-entry meta: [kv_len, read_row, write_base, n_rows, write_stride, reset, fed token (fused_head)].
TOKEN_META = 6
_EMBED_BLOCK = (
    16  # embedding rows per DMA: a dynamic HBM row offset must be tile-aligned
)
HBM_INPUTS = (
    "layer_vec",
    "kda_proj",
    "kda_fb",
    "kda_oproj",
    "mla_w1",
    "mla_qb",
    "mla_wkc",
    "mla_wvc",
    "mla_oproj",
    "dense_gu",
    "dense_down",
    "router",
    "latent_down",
    "latent_up",
    "shared_gu",
    "shared_down",
    "expert_gu",
    "expert_gu_scale",
    "expert_down",
    "expert_down_scale",
)
# Stream keys whose buffers live in the VMEM arena (StackKernel._plan_arena).
_ARENA_KEYS = ("dense_gu", "kda_proj", "mla_w1", "mla_g", "mla_qb")
# Region M (``moe_region_m``): the MoE's dense streams in one raw buffer, latent_up overlaying
# router + latent_down (it is filled once both are consumed, see ``MoeStage(late_latent_up)``).
_REGION_M_KEYS = ("router", "latent_down", "shared_gu", "latent_up")
# Families packed into ``layer_vec`` by ``pack_inputs`` (loader output names).
SMALL_FAMILIES = (
    "norms",
    "kda_conv",
    "kda_dt",
    "kda_alog",
    "kda_onorm",
    "mla_qnorm",
    "mla_kvnorm",
    "router_bias",
    "latent_norm",
)


class VecLayout(msgspec.Struct, frozen=True):
    """Rows / columns of one layer's ``[ROWS, hidden]`` f32 slab. Column offsets
    are multiples of 128; unused cells are zero.

    rows 0..5  : the six attention-residual / norm vectors (``NORMS`` order)
    rows 6..9  : KDA conv taps ``[4, 3hd]`` (tap 0 oldest)
    row 10     : KDA dt_bias (hd) | A_log per channel (hd) | o_norm (128)
    row 11     : MoE latent norm (latent) | router bias (2 * router_pad), for this layer's MoE
    row 12     : MLA q_a norm (1536) | kv_a norm (512)
    """

    hidden: int
    kda_hd: int
    latent: int
    router_pad: int

    ROWS = 16
    NORMS, CONV, KDA, MOE, MLA = 0, 6, 10, 11, 12

    def __post_init__(self):
        assert (
            3 * self.kda_hd <= self.hidden and 2 * self.kda_hd + HEAD_DIM <= self.hidden
        )
        assert self.latent + 2 * self.router_pad <= self.hidden and 2048 <= self.hidden

    @property
    def cells(self) -> dict:
        """name -> (row, rows, col, cols)."""
        hd, lat = self.kda_hd, self.latent
        return {
            "conv": (self.CONV, 4, 0, 3 * hd),
            "dt": (self.KDA, 1, 0, hd),
            "a_log": (self.KDA, 1, hd, hd),
            "onorm": (self.KDA, 1, 2 * hd, HEAD_DIM),
            "latent_norm": (self.MOE, 1, 0, lat),
            "router_bias": (self.MOE, 1, lat, 2 * self.router_pad),
            "qnorm": (self.MLA, 1, 0, 1536),
            "kvnorm": (self.MLA, 1, 1536, 512),
        }


def state_inputs(spec, stacked: bool = False) -> tuple[str, ...]:
    """sglang's per-layer pools, passed one tensor per layer: KDA conv ``[slots, 3, 1, 3hd]`` f32,
    KDA recurrent ``[slots, h, 128, 128]`` f32, MLA latent ``[pages, 64, 2, 640]`` bf16.
    ``stacked``: one tensor per family with a leading layer axis (``[n_kda, slots, ...]``,
    ``[n_mla, pages, ...]``), indexed by the traced KDA / MLA index."""
    if stacked:
        return ("kda_conv", "kda_rec", "mla_cache")
    nk, nm = len(spec.kda_layers), len(spec.mla_layers)
    return (
        tuple(f"kda_conv_{k}" for k in range(nk))
        + tuple(f"kda_rec_{k}" for k in range(nk))
        + tuple(f"mla_cache_{j}" for j in range(nm))
    )


def scale_inputs(spec) -> tuple[str, ...]:
    """Column scales of the e4m3 dense families (``Relax.dense_fp8``), ``[layers, 1, N]`` f32."""
    if not spec.relax.dense_fp8:
        return ()
    return tuple(n + "_s" for n in HBM_INPUTS if n in DENSE_FP8_FAMILIES)


def input_names(
    spec, stacked: bool = False, fused_head: bool = False
) -> tuple[str, ...]:
    """``fused_head`` (see ``StackKernel``): ``embed`` / ``lm_head`` instead of ``x0``."""
    vmem = tuple(n for n in VMEM_INPUTS if not (fused_head and n == "x0"))
    head = HEAD_INPUTS if fused_head else ()
    return (
        SMEM_INPUTS
        + vmem
        + HBM_INPUTS
        + head
        + scale_inputs(spec)
        + state_inputs(spec, stacked)
    )


class StackSpec(msgspec.Struct, frozen=True):
    topo: Topology
    num_layers: int
    mla_layers: tuple[int, ...]
    dims: K3Dims = K3Dims()
    block_size: int = 12  # attn_res_block_size: bank write on layer % block_size == 0
    relax: Relax = Relax()

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_layers) if i not in self.mla_layers)

    def vec_layout(self) -> VecLayout:
        d = self.dims
        return VecLayout(
            d.hidden,
            96 // self.topo.world * HEAD_DIM,
            d.latent,
            MoeLayout(d, self.topo).router_pad,
        )

    def tables(self):
        """[4, L] int32: kind, kda index, mla index, bank-write flag."""
        kind, kda, mla, write = [], [], [], []
        for i in range(self.num_layers):
            is_mla = i in self.mla_layers
            kind.append(_MLA if is_mla else _KDA)
            kda.append(0 if is_mla else self.kda_layers.index(i))
            mla.append(self.mla_layers.index(i) if is_mla else 0)
            write.append(int(i % self.block_size == 0))
        return [kind, kda, mla, write]


def pack_inputs(spec: StackSpec, w: dict) -> dict:
    """Loader families -> kernel inputs: the ``SMALL_FAMILIES`` (each ``[R, n, ...]``,
    n = layers of that family) become one ``layer_vec`` ``[R, L, ROWS, hidden]`` f32."""
    vl = spec.vec_layout()
    ranks = w["norms"].shape[0]
    vec = np.zeros((ranks, spec.num_layers, vl.ROWS, vl.hidden), np.float32)
    vec[:, :, vl.NORMS : vl.NORMS + len(NORMS)] = np.asarray(w["norms"], np.float32)[
        :, :, :, 0
    ]
    cells = vl.cells

    def put(name, layers, src):
        row, rows, col, cols = cells[name]
        src = np.asarray(src, np.float32).reshape(ranks, len(layers), rows, cols)
        for i, layer in enumerate(layers):
            vec[:, layer, row : row + rows, col : col + cols] = src[:, i]

    put("conv", spec.kda_layers, w["kda_conv"])
    put("dt", spec.kda_layers, w["kda_dt"])
    put("a_log", spec.kda_layers, w["kda_alog"])
    put("onorm", spec.kda_layers, w["kda_onorm"])
    put("qnorm", spec.mla_layers, w["mla_qnorm"])
    put("kvnorm", spec.mla_layers, w["mla_kvnorm"])
    moe_layers = range(1, spec.num_layers)
    put("latent_norm", moe_layers, w["latent_norm"])
    put("router_bias", moe_layers, w["router_bias"])
    out = {k: v for k, v in w.items() if k not in SMALL_FAMILIES}
    out["layer_vec"] = vec
    return out


class _KdaStateIO:
    """Async copies between the KDA pools and the VMEM state buffers.

    Pools are either per-layer refs (sglang's layout; every DMA is emitted once
    per pool under ``pl.when``) or one stacked ref ``[n_kda, slots, ...]``
    indexed by the traced KDA index. ``sems`` is DMA ``(2, 2)``: [read/write, conv/rec].
    """

    def __init__(self, conv_pools, rec_pools, stacked: bool, row, conv_v, rec_v, sems):
        # Verify reads one checkpoint row and writes another (see ``StackKernel(verify_rows)``).
        self.conv, self.rec, self.stacked = conv_pools, rec_pools, stacked
        self.read_row = self.write_row = row
        self.conv_v, self.rec_v, self.sems = conv_v, rec_v, sems

    def _copies(self, k, write: bool, row=None):
        """(predicate, [conv copy, rec copy]) pairs; ``k`` None gives a wait-only descriptor."""
        d = 1 if write else 0
        out = []
        cands = (
            [(None, self.conv[0], self.rec[0])]
            if self.stacked
            else list(zip(range(len(self.conv)), self.conv, self.rec))
        )
        for kk, conv, rec in cands:
            if row is None:
                row = self.write_row if write else self.read_row
            if self.stacked:
                kidx = 0 if k is None else k
                conv_h, rec_h = conv.at[kidx, row], rec.at[kidx, row]
            else:
                conv_h, rec_h = conv.at[row], rec.at[row]
            pairs = ((conv_h, self.conv_v), (rec_h, self.rec_v))
            cps = [
                pltpu.make_async_copy(
                    *((v, h) if write else (h, v)), self.sems.at[d, i]
                )
                for i, (h, v) in enumerate(pairs)
            ]
            pred = None if (self.stacked or k is None) else (k == kk)
            out.append((pred, cps))
            if k is None:
                break
        return out

    def _start(self, k, write, row=None):
        copies = self._copies(k, write, row)
        if copies[0][0] is None:
            for cp in copies[0][1]:
                cp.start()
            return
        # Per-layer pools: a binary pl.when tree selects pool k (log2(n) tests instead of n).
        tree_when(k, len(copies), lambda kk: [cp.start() for cp in copies[kk][1]])

    def _wait(self, write):
        for cp in self._copies(None, write)[0][1]:
            cp.wait()

    def start_read(self, k):
        self._start(k, False)

    def wait_read(self):
        self._wait(False)

    def start_write(self, k, row=None):
        """``row``: this write's pool row (default ``write_row``), e.g. a verify checkpoint row."""
        self._start(k, True, row)

    def wait_write(self):
        self._wait(True)


class _SkippedStream:
    """Benchmark-only stand-in for a weight stream: consumes the chunks already started
    (keeps the DMA semaphores balanced) and returns zeros instead of ``x @ W``."""

    def __init__(self, st):
        self.st = st

    def __getattr__(self, name):
        return getattr(self.st, name)

    def matmul(self, x, *, prefetched):
        self.st.wait_prefetched()
        return jnp.zeros((x.shape[0], self.st.w.shape[-1]), jnp.float32)


# Ring-filling order of ``StackKernel(moe_fill_vmem=...)``: the routing path first.
_MOE_FILL = ("router", "latent_down", "latent_up", "shared_gu")
# Unpadded scratch bytes we allow out of the 64 MiB limit: measured at TP8, 53.6 MiB overflowed it
# by 0.2 MiB (tile padding and Mosaic's own VMEM), so this keeps ~1.5 MiB of margin.
_VMEM_SCRATCH_CAP = 52 << 20


class StackKernel:
    def __init__(
        self,
        spec: StackSpec,
        *,
        collective_id: int = K3MK_COLLECTIVE_ID,
        debug_repeat: int = 1,
        debug_skip: frozenset = frozenset(),
        stream_budget: int = 1 << 20,
        stream_slots: dict | None = None,
        stream_budgets: dict | None = None,
        prefetch_attn: bool = True,
        prefetch_moe: bool = True,
        stacked_states: bool = False,
        trace: bool = False,
        use_arena: bool = True,
        gather_bf16: bool = False,
        flat_gather: bool = True,
        wire_relax: tuple | str = (),
        layer_out: bool = False,
        row_gathers: bool = False,
        wire_scope: str = "",
        replica_balance: bool = True,
        moe_shared_after: int = -1,
        moe_debug: str = "",
        prefetch_latent_up: bool = False,
        moe_early_dma: bool | None = None,
        moe_pair_down: bool = True,
        moe_ring: int | None = None,
        moe_overlap_pair: bool = True,
        moe_stream_slots: dict | None = None,
        moe_fill_vmem: bool | str = "whole",
        moe_late_handoff: bool | str = True,
        moe_split_refill: bool = True,
        moe_region_m: bool = True,
        moe_up_fill: str = "reduce",
        moe_distinct_balance: bool = False,
        moe_single_rows: bool = True,
        moe_down_dequant: bool = True,
        oproj_full_gather: bool = True,
        far_first: bool = True,
        moe_fast_topk: bool = True,
        mixture_tree: bool = True,
        mixture_fused_score: bool = True,
        mixture_early_scores: bool = True,
        moe_prefetch_late: bool = False,
        mla_w1_split: bool = True,
        mla_early_kv: bool = True,
        mla_async_write: bool = False,
        attn_ring_mib: float = 24.5,
        mla_w1_chunk: int = 0,
        oproj_early: bool = True,
        mla_kv_window: bool = True,
        handoff: str = "experts",
        attn_region: str = "experts",
        attn_chunk_mib: float = 4.0,
        mla_kv_block_pages: int = 4,
        kda_mxu: bool = False,
        moe_up_tree: bool = False,
        kda_fast_core: bool = False,
        kda_ckpt_all: bool | str = True,
        fused_head: bool = True,
        head_vocab: int = 163840,
        verify_rows: int = 1,
        capture_layers: tuple = (),
        verify_mode: str = "loop",
        prefill_blocks: int = 0,
        capture_mixture: bool = False,
    ):
        """SMEM ``meta`` = [kv_len, KDA state row]; ``pages`` is the request's page table
        padded to a multiple of 16 entries.

        ``verify_rows`` = R > 1 (speculative verify, token loop): ``x0`` / the final output are
        ``[R, 1, H]`` bf16 in HBM; the stack runs once per row t at kv_len ``meta[0] + t`` (each row's
        MLA token is written before it attends: causal); ``meta`` = [kv_len, KDA read row, KDA
        write base]: row 0 reads the state from the read row, row t > 0 from ``base + t - 1``, and
        row t writes its state to ``base + t`` (sglang's per-row checkpoints). The second output
        holds the rows of ``capture_layers`` (``[len, R, 1, H]`` bf16, HBM) for the draft model.
        ``verify_mode="rows"``: the R rows go through every stage together (weights streamed once;
        see B8_VERIFY.md section 5); ``meta[0]`` is row 0's kv length. ``prefetch_attn`` starts layer ``l + 1``'s first
        attention projections from layer ``l``'s MoE; ``prefetch_moe`` starts layer ``l``'s
        router / latent_down / shared_gu when its attention begins (each fills its ring).
        ``stacked_states`` selects the state input form (see ``state_inputs``). ``use_arena``
        overlays buffers with disjoint lifetimes in one VMEM arena (see ``_plan_arena``); the KDA
        projection ring then gets as many slots as the MLA buffers it overlays.
        ``gather_bf16`` / ``flat_gather`` / ``row_gathers``: MoE gather options (exact), see ``MoeStage``.
        ``wire_relax``: transport relaxations (``bf16_transport``, ``host_bf16``, ``oproj_flat``) added to
        ``spec.relax``; they leave the weights alone, so a benchmark can switch them per job.
        ``wire_scope`` ("attn" | "moe", benchmark-only): the bf16 wires apply to that side only (the
        dense MLP counts as attention).
        ``layer_out`` fills the per-layer hidden-state output (tests); otherwise it is a single
        unwritten row. ``replica_balance=False`` never moves a route to an expert replica
        (``Relax.expert_replicas`` weights then compute exactly like the default layout).
        ``mixture_tree``: log-depth f32 sums in the attention-residual mixtures and their RMSNorms
        (shorter dependency chain; a different summation order, so not bit-identical to
        sglang's; False is exact). ``mixture_fused_score``: see ``attn_res.MixOpts`` (drops one
        bf16 round per mixture score). ``mixture_early_scores``: the MLP-side mixture's bank-row
        scores are computed before the attention (same values), leaving only the prefix after it.
        ``moe_prefetch_late`` starts the ``prefetch_moe`` rings
        when the attention output reduction starts instead of when the layer starts, so they do
        not compete with the attention weight streams. ``mla_w1_split`` splits MLA's replicated
        q_a / kv_a columns over the lanes of a host (``stages.mla.W1Split``). ``mla_early_kv`` /
        ``mla_async_write``: see ``MlaStage``; the async cache write is drained at the hand-off point.
        ``attn_ring_mib`` > 0 sizes the cross-layer-prefetched attention rings (KDA projection; MLA w1
        [and g_proj]) to about that many MiB of the arena instead of the default depths. ``oproj_early``
        gives the attention o_proj its own ring ("attn_out"), filled from the previous layer's MoE
        together with the input projections instead of at layer start. ``mla_kv_window``: see
        ``MlaStage`` (masks one page instead of the whole KV block; bit-identical).
        ``moe_up_tree``: ``MoeStage`` ``up_tree``. ``moe_single_rows`` / ``moe_down_dequant``: ``MoeStage``
        ``single_rows`` / ``down_dequant`` (verify rows; f32 reorder). ``oproj_full_gather`` (verify rows,
        pair o_proj): the row-scattered o_proj reduce gathers both halves in one host round (same bits).
        ``far_first``: every collective level issues its copies to other hosts first (same bits). ``fused_head``: the kernel also does the greedy
        LM head (``stages.head``; inputs ``embed`` [V, H] and ``lm_head`` [V / world, H] replace
        ``x0``, ``meta[TOKEN_META]`` is the token fed, the next token is the extra SMEM output 3)."""
        self.mix = MixOpts(tree=mixture_tree, fused_score=mixture_fused_score)
        self.mixture_early_scores = mixture_early_scores
        if isinstance(
            wire_relax, str
        ):  # e.g. "host_bf16,oproj_flat" from a shell-quoted CLI value
            wire_relax = tuple(x.strip(" ()'\"") for x in wire_relax.split(","))
        wire_relax = tuple(x for x in wire_relax if x)
        if wire_relax:
            assert set(wire_relax) <= {"bf16_transport", "host_bf16", "oproj_flat"}, (
                wire_relax
            )
            spec = msgspec.structs.replace(
                spec,
                relax=msgspec.structs.replace(
                    spec.relax, **{k: True for k in wire_relax}
                ),
            )
        # ``handoff``: where layer l starts layer l + 1's weights / state ("route": inside l's MoE once
        # its expert DMAs are queued; "experts": after its routed experts; "after_moe": after l's whole MoE). ``attn_region="experts"``
        # places the attention projection buffers (arena region A) over the routed-expert ring,
        # which is only valid when the hand-off comes after the experts.
        # "experts" = MoeStage(late_handoff): after the routed experts, before the reductions.
        assert handoff in ("route", "experts", "after_moe") and attn_region in (
            "separate",
            "experts",
        )
        if handoff == "route" and moe_late_handoff:
            handoff = "experts"
        assert attn_region == "separate" or handoff != "route", (
            "attn_region=experts needs a late hand-off"
        )
        self.handoff, self.attn_region = handoff, attn_region
        # ``attn_chunk_mib`` > 0: per-slot budget of the attention projection streams (bigger K-chunks,
        # fewer DMAs and dots; reorders the f32 accumulation). It also unpins the w1-split chunks,
        # which otherwise follow the unsplit stream's chunking.
        stream_budgets = dict(stream_budgets or {})
        if attn_chunk_mib:
            for key in ("kda_proj", "mla_w1", "mla_g", "mla_qb"):
                stream_budgets.setdefault(key, int(attn_chunk_mib * (1 << 20)))
        self.oproj_early = oproj_early
        self.attn_ring_bytes = int(attn_ring_mib * (1 << 20))
        self.moe_prefetch_late = moe_prefetch_late
        self.layer_out = layer_out
        self.fused_head, self.head_vocab = fused_head, head_vocab
        assert not fused_head or use_arena, (
            "fused_head streams lm_head through the arena"
        )
        if (
            verify_rows > 1
        ):  # several rows per call: the caller embeds and samples (x0 in, final rows out)
            fused_head = self.fused_head = False
        self.verify_rows, self.capture_layers = verify_rows, tuple(capture_layers)
        assert verify_mode in ("loop", "rows"), verify_mode
        self.rows_mode = verify_rows > 1 and verify_mode == "rows"
        # ``prefill_blocks`` = B > 0 (rows mode): a prefill chunk of up to B * R tokens as consecutive
        # R-row blocks; x0 / final are [B, R, H]. meta = [kv_len of token 0, KDA read row, KDA write base,
        # tokens n, write stride, reset]: block b starts at kv_len meta[0] + b * R and reads the state
        # from meta[1] (b = 0, zeroed when reset) else from the row of token b * R - 1; token g < n writes
        # base + g * stride, rows past n write the null slot 0.
        assert not prefill_blocks or self.rows_mode, "prefill_blocks: rows mode only"
        self.prefill_blocks, self._blk = prefill_blocks, None
        # ``capture_mixture``: a captured layer l gives what sglang's DSpark capture takes (KimiK3LinearModel.
        # _dspark_capture_stream): the attention-residual mixture layer l + 1's attention side computes before
        # its input norm (for the last layer: the output-side mixture before the final norm), not l's output.
        self.capture_mixture = capture_mixture
        self.R = verify_rows if self.rows_mode else 1
        if self.rows_mode:
            # Not implemented for rows yet: the KV window / early KV / async write.
            mla_kv_window = mla_early_kv = mla_async_write = False
            mixture_early_scores = self.mixture_early_scores = False
            assert not layer_out, "verify rows: layer_out unsupported"
        self.spec, self.collective_id, self.stacked = (
            spec,
            collective_id,
            stacked_states,
        )
        if "prefetch" in debug_skip:  # benchmark-only: no cross-layer weight prefetch
            prefetch_attn = prefetch_moe = False
        self.prefetch_attn, self.prefetch_moe = prefetch_attn, prefetch_moe
        # latent_up's ring is filled with router / latent_down / shared_gu at layer start.
        self._early_moe = ("router", "latent_down", "shared_gu") + (
            ("latent_up",) if prefetch_latent_up else ()
        )
        # Benchmark-only: named-scope phase markers in the kernel (see stages/common.phase).
        self.trace = trace
        # Benchmark-only knobs (outputs become meaningless): run the stack ``debug_repeat``
        # times per call; ``debug_skip`` drops "attn" and/or "moe" work from every layer;
        # "mixture": the attention-residual mixtures pass their prefix through and the bank is
        # never written; "compact" / "compact2": see ``MoeStage._compact_local``; "aux" (with
        # "attn" and prefetch_attn=False): no per-layer slab / f_b / KDA-state DMAs (every layer
        # uses layer 0's slab); "norm": the attention / MLP input RMSNorms pass through; "ids": no
        # per-layer copy of the route ids to the SMEM output. Attention phases:
        # "attn_proj" (input projections), "attn_core" (KDA recurrence / MLA attention + cache write),
        # "attn_oproj" (o_proj stream + dot), "attn_reduce" (= "attn_ar", its reduction), "prefetch"
        # (cross-layer weight prefetch off).
        self.debug_repeat, self.debug_skip = debug_repeat, debug_skip
        self._aux_off = {"aux", "attn"} <= debug_skip
        assert not (self._aux_off and prefetch_attn), (
            "debug_skip aux needs prefetch_attn=False"
        )
        d, topo = spec.dims, spec.topo
        rx = spec.relax
        self.layout = MoeLayout(d, topo, replicas=rx.expert_replicas)
        self.vl = spec.vec_layout()
        wires = {"bf16_transport": False, "host_bf16": False}
        rx_moe = msgspec.structs.replace(rx, **wires) if wire_scope == "attn" else rx
        rx_attn = msgspec.structs.replace(rx, **wires) if wire_scope == "moe" else rx
        if moe_early_dma is None:
            # Verify rows: starting expert DMAs from inside the grouped compaction cost ~12 us/layer at
            # TP32 (w10b_croute: 1.5045 -> 1.3085 ms/token without them, same tokens); B1 keeps them.
            moe_early_dma = self.R == 1
        self.moe = MoeStage(
            self.layout,
            rows=self.R,
            relax=rx_moe,
            gather_bf16=gather_bf16,
            flat_gather=flat_gather,
            shared_after=moe_shared_after,
            early_dma=moe_early_dma,
            pair_down=moe_pair_down,
            ring=moe_ring or _RING,
            overlap_pair=moe_overlap_pair,
            late_handoff=self.handoff == "experts",
            split_refill=moe_split_refill,
            late_latent_up=moe_region_m and moe_up_fill,
            row_gathers=row_gathers or self.rows_mode,
            up_tree=moe_up_tree,
            fast_topk=moe_fast_topk,
            single_rows=moe_single_rows,
            down_dequant=moe_down_dequant,
            debug=frozenset(x for x in moe_debug.split(",") if x)
            | frozenset(k for k in debug_skip if k.startswith("moe_")),
        )
        self.moe.replica_balance = replica_balance
        self.moe.distinct_balance = moe_distinct_balance
        self.moe.debug_compact = next(
            (k for k in ("compact", "compact2") if k in debug_skip), ""
        )
        self.kda = KdaStage(
            topo, hidden=d.hidden, eps=d.rms_eps, relax=rx_attn, rows=self.R
        )
        self.mla = MlaStage(
            topo,
            hidden=d.hidden,
            eps=d.rms_eps,
            relax=rx_attn,
            early_kv=mla_early_kv,
            async_write=mla_async_write and not {"attn", "attn_core"} & set(debug_skip),
            kv_window=mla_kv_window,
            block_pages=mla_kv_block_pages,
            rows=self.R,
        )
        self.kda.skip_core = self.mla.skip_core = "attn_core" in debug_skip
        # Remote copies to other hosts issued first (issue order only, same bits).
        collectives.FAR_FIRST[0] = far_first
        # Verify rows + pair_oproj: one host gather of both o_proj halves.
        if oproj_full_gather:
            for st in (self.kda, self.mla):
                st.out = msgspec.structs.replace(st.out, rows_full_gather=True)
        self.kda.mxu = kda_mxu  # KDA recurrence contractions on the MXU (f32 reorder)
        self.kda.fast_core = kda_fast_core
        # Verify rows: all R checkpoint states staged in VMEM and written by one DMA per family (pool rows
        # write_base .. write_base + R - 1 are contiguous) when they fit; prefill blocks keep the slots.
        ck_bytes = (
            self.R * 4 * (self.kda.heads * HEAD_DIM * HEAD_DIM + 8 * 3 * self.kda.hd)
        )
        self._ck_all = (
            bool(kda_ckpt_all)
            and self.rows_mode
            and not self.prefill_blocks
            and (ck_bytes <= 4 << 20 or kda_ckpt_all == "force")
        )
        self.dense = DenseMlpStage(d, topo, rows=self.R, relax=rx_attn)
        # Dense MLP and (without pair_oproj) attention o_proj share one all-reduce.
        self._hidden_spec = self.dense.ar if rx.pair_oproj else self.kda.out.full_spec
        if self.rows_mode:  # scattered by row: 1 / lanes of the reduce bytes, same sums
            ar = self._hidden_spec
            self._hidden_spec = RowScatterSpec(
                topo, self.R, d.hidden, ar.transport, ar.host_transport
            )
        lay, kd, ml = self.layout, self.kda, self.mla
        self.use_arena = use_arena
        self._attn_key = "attn_out" if rx.pair_oproj or oproj_early else "hidden_out"
        if oproj_early:
            # The whole o_proj in one DMA when it is small (TP32: 5.25 MiB BF16), else a 3-slot ring.
            k_o, n_o = kd.out.weight_shape
            whole = k_o * n_o * jnp.dtype(rx.dense_dtype).itemsize
            if whole <= 6 << 20:
                # pair_oproj: two chunks (own rows, peer rows) so the own-half dot hides the feature swap.
                parts = 2 if rx.pair_oproj else 1
                stream_budgets.setdefault("attn_out", whole // parts)
                stream_slots = {"attn_out": parts, **(stream_slots or {})}
            else:
                stream_slots = {"attn_out": 3, **(stream_slots or {})}
        self.w1_split = W1Split(topo.lanes) if mla_w1_split else None
        self.w1_gather = (
            AllGatherSpec(
                Topology(1, topo.lanes), flat_rows(self.R, self.w1_split.width)
            )
            if mla_w1_split
            else None
        )
        uses = [
            ("router", d.hidden, lay.router_pad),
            ("latent_down", d.hidden, lay.latent_chunk),
            ("shared_gu", d.hidden, 2 * lay.shared_cols),
            ("hidden_out", lay.shared_cols, d.hidden),
            ("hidden_out", self.dense.channels, d.hidden),
            (self._attn_key, *kd.out.weight_shape),
            (self._attn_key, *ml.out.weight_shape),
            ("latent_up", d.latent, lay.latent_up_cols),
            ("dense_gu", d.hidden, 2 * self.dense.channels),
            ("kda_proj", d.hidden, 4 * kd.hd + 256),
            ("mla_qb", 1536, 2 * ml.hd),
        ]
        # Ring depths of the MoE streams (``stream_slots`` wins); a ring as deep as the weight's
        # chunk count holds the whole weight, prefetched during the attention.
        stream_slots = {**(moe_stream_slots or {}), **(stream_slots or {})}
        chunks = {}
        if self.w1_split is None:
            uses.append(("mla_w1", d.hidden, REPLICATED + ml.hd))
        else:
            # Same K-chunks as the unsplit w1, so every column accumulates in the same order.
            budget = stream_budgets.get("mla_w1", stream_budget)
            c0 = stream_chunk(
                d.hidden, REPLICATED + ml.hd, jnp.dtype(rx.dense_dtype).itemsize, budget
            )
            uses += [
                ("mla_w1", d.hidden, self.w1_split.width),
                ("mla_g", d.hidden, ml.hd),
            ]
            if not attn_chunk_mib:
                # mla_w1_chunk (debug): other K-chunks.
                chunks.update(mla_w1=mla_w1_chunk or c0, mla_g=mla_w1_chunk or c0)
            if not self.attn_ring_bytes:
                stream_slots.setdefault("mla_w1", 8)
                stream_slots.setdefault("mla_g", 4)
        assert not (moe_region_m and prefetch_latent_up), (
            "region M fills latent_up after the routing"
        )
        self.region_m = None
        self._use_region_m = moe_region_m
        pool_args = dict(
            budget=stream_budget,
            budgets=stream_budgets,
            dtype=rx.dense_dtype,
            external=(_ARENA_KEYS if use_arena else ())
            + (_REGION_M_KEYS if moe_region_m else ()),
            chunks=chunks,
        )
        fill = lambda: self._fill_moe_rings(
            uses, stream_slots, pool_args, whole=moe_fill_vmem == "whole"
        )
        self._build_pool(uses, stream_slots, pool_args)
        if moe_fill_vmem:
            fill()
        if moe_ring is None:
            # 6 expert slots (the slowest TP32 host serves ~7 experts) when that costs no MoE stream
            # depth, e.g. inside a region E sized by the attention weights; else the default 4.
            base = dict(self.moe_slots) if moe_fill_vmem else None
            self.moe.ring = 6
            self._build_pool(uses, stream_slots, pool_args)
            if moe_fill_vmem:
                fill()
            if self._vmem_scratch_bytes() > _VMEM_SCRATCH_CAP or (
                moe_fill_vmem and self.moe_slots != base
            ):
                self.moe.ring = _RING
                self._build_pool(uses, stream_slots, pool_args)
                if moe_fill_vmem:
                    fill()

    def _build_pool(self, uses, slots: dict, pool_args: dict):
        self.pool = StreamPool(uses, slots=slots, **pool_args)
        if self.use_arena:
            self.pool = StreamPool(uses, slots=self._arena_slots(slots), **pool_args)
            self._plan_arena()
            if self.fused_head:
                self._plan_head()
        if self._use_region_m:
            self._plan_region_m()

    def _plan_region_m(self):
        """[router | latent_down | shared_gu] with latent_up from offset 0 (over router and
        latent_down); shared_gu starts after whichever ends later."""
        ar, pd = Arena(), self.pool.dtype
        place = lambda at, key: ar.place(at, self.pool.buf_shape(key), pd)
        router = place(0, "router")
        down = place(router[1], "latent_down")
        up = place(0, "latent_up")
        shared = place(max(down[1], up[1]), "shared_gu")
        self._region_m_at = {
            "router": router[0],
            "latent_down": down[0],
            "latent_up": up[0],
            "shared_gu": shared[0],
        }
        self.region_m = ar

    def _region_m_views(self, ref) -> dict:
        return {
            k: Arena.view(ref, at, self.pool.buf_shape(k), self.pool.dtype)
            for k, at in self._region_m_at.items()
        }

    def _vmem_scratch_bytes(self) -> int:
        """Unpadded bytes of the VMEM scratch (the unit ``_VMEM_SCRATCH_CAP`` was measured in)."""
        return sum(
            math.prod(x.shape) * jnp.dtype(x.dtype).itemsize
            for x in self._scratch()
            if str(x.memory_space) == "vmem"
        )

    def _fill_moe_rings(self, uses, explicit: dict, pool_args: dict, *, whole: bool):
        """Deepens the MoE stream rings in ``_MOE_FILL`` order, each up to its chunk count (the whole
        weight prefetched during the attention), while the scratch stays under ``_VMEM_SCRATCH_CAP``.
        ``whole``: first try one slot holding the whole weight (one DMA and one dot; the f32
        accumulation order of the dot changes)."""
        slots, args = (
            dict(explicit),
            dict(pool_args, budgets=dict(pool_args["budgets"] or {})),
        )
        itemsize = jnp.dtype(args["dtype"]).itemsize
        for key in _MOE_FILL:
            if key in explicit or key in args["budgets"]:
                continue
            ((k, n),) = [(k, n) for kk, k, n in uses if kk == key]
            trials = (
                [({**slots, key: 1}, {**args["budgets"], key: k * n * itemsize})]
                if whole
                else []
            )
            trials += [
                ({**slots, key: depth}, args["budgets"])
                for depth in range(self.pool.max_chunks[key], 2, -1)
            ]
            for t_slots, t_budgets in trials:
                self._build_pool(uses, t_slots, dict(args, budgets=t_budgets))
                if self._vmem_scratch_bytes() <= _VMEM_SCRATCH_CAP:
                    slots, args["budgets"] = t_slots, t_budgets
                    break
            self._build_pool(uses, slots, args)
        self.moe_slots = {k: self.pool.buf_shape(k)[:2] for k in _MOE_FILL}

    # ----- VMEM arena -----
    def _expert_ring_shapes(self):
        d, li = self.spec.dims, self.layout.lane_inter
        ring = self.moe.ring
        return [
            ((ring, d.latent, 2 * li), jnp.float8_e4m3fn),
            ((ring, 1, 2 * li), jnp.float32),
            ((ring, li // 2, d.latent), jnp.uint8),
            ((ring, li // 32, d.latent), jnp.uint8),
        ]

    def _mla_arena_shapes(self):
        """w1 ring, q_b ring, w_kc, w_vc[, g_proj ring]."""
        h, pd = self.mla.heads, self.pool.dtype
        out = [
            (self.pool.buf_shape("mla_w1"), pd),
            (self.pool.buf_shape("mla_qb"), pd),
            ((h, 128, 512), jnp.bfloat16),
            ((h, 512, 128), jnp.bfloat16),
        ]
        if self.w1_split is not None:
            out.append((self.pool.buf_shape("mla_g"), pd))
        return out

    def _arena_slots(self, explicit: dict) -> dict:
        """Default ring depths of the arena keys: the KDA projection ring fills the bytes of
        the MLA buffers it overlays, the dense gate/up ring those of the expert ring (<= 4)."""
        slots = dict(explicit)
        if self.attn_ring_bytes:
            self._ring_slots_by_bytes(slots, self.attn_ring_bytes)
        elif (
            self.attn_region == "experts"
        ):  # region A may use all of the expert ring's bytes
            self._ring_slots_by_bytes(
                slots,
                sum(storage_bytes(sh, dt) for sh, dt in self._expert_ring_shapes()),
            )
        mla = sum(storage_bytes(sh, dt) for sh, dt in self._mla_arena_shapes())
        experts = sum(storage_bytes(sh, dt) for sh, dt in self._expert_ring_shapes())
        kda_slot = storage_bytes(self.pool.buf_shape("kda_proj")[1:], self.pool.dtype)
        region = max(mla, experts) if self.attn_region == "experts" else mla
        slots.setdefault(
            "kda_proj",
            max(
                2,
                min(
                    self.spec.dims.hidden // self.pool.shape["kda_proj"][0],
                    region // kda_slot,
                ),
            ),
        )
        dense_slot = storage_bytes(self.pool.buf_shape("dense_gu")[1:], self.pool.dtype)
        slots.setdefault("dense_gu", max(2, min(4, experts // dense_slot)))
        return slots

    def _ring_slots_by_bytes(self, slots: dict, budget: int) -> None:
        """Default depths filling ``attn_ring_bytes``: all of it for the KDA ring; for MLA what
        q_b / w_kc / w_vc leave, shared by the w1 (and g_proj) rings. Never more than a weight's chunks."""
        pd, hidden = self.pool.dtype, self.spec.dims.hidden

        def fit(key, nbytes):
            rows, n = self.pool.shape[key]
            per = storage_bytes((rows, n), pd)
            slots.setdefault(key, int(max(2, min(hidden // rows, nbytes // per))))

        fit("kda_proj", budget)
        mla_fixed = sum(
            storage_bytes(sh, dt) for sh, dt in self._mla_arena_shapes()[1:4]
        )
        keys = ["mla_w1"] + (["mla_g"] if self.w1_split is not None else [])
        for key in keys:
            fit(key, (budget - mla_fixed) // len(keys))

    def _plan_arena(self):
        """Region X: layer 0's dense gate/up ring | the routed-expert ring (MoE only).
        Region A: the KDA projection ring | MLA w1 + q_b rings, w_kc, w_vc (one layer is one
        kind; a layer's region-A buffers are filled from the previous layer's MoE)."""
        ar = Arena()
        place = lambda at, items: [ar.place(at, sh, dt) for sh, dt in items]
        self._arena_x = place(0, [(self.pool.buf_shape("dense_gu"), self.pool.dtype)])
        ring, at = [], 0
        for sh, dt in self._expert_ring_shapes():
            off, at = ar.place(at, sh, dt)
            ring.append((off, sh, dt))
        self._arena_ring = ring
        a0 = 0 if self.attn_region == "experts" else ar.size
        self._arena_kda = ar.place(a0, self.pool.buf_shape("kda_proj"), self.pool.dtype)
        mla, at = [], a0
        for sh, dt in self._mla_arena_shapes():
            off, at = ar.place(at, sh, dt)
            mla.append((off, sh, dt))
        self._arena_mla = mla
        self.arena = ar

    def _plan_head(self):
        """The lm_head ring goes at the start of the arena (idle once the last layer is done)."""
        d, world = self.spec.dims, self.spec.topo.world
        vs = self.head_vocab // world
        chunk = head_chunk(vs, d.hidden, max(self.arena.size, 8 << 20))
        self.head = HeadSpec(self.spec.topo, vs, d.hidden, chunk)
        self._head_ring = ((2, chunk, d.hidden), jnp.bfloat16)
        self.arena.place(0, *self._head_ring)

    def _arena_views(self, ref):
        v = Arena.view
        ring = [v(ref, off, sh, dt) for off, sh, dt in self._arena_ring]
        mla = [v(ref, off, sh, dt) for off, sh, dt in self._arena_mla]
        pd = self.pool.dtype
        ext = {
            "dense_gu": v(
                ref, self._arena_x[0][0], self.pool.buf_shape("dense_gu"), pd
            ),
            "kda_proj": v(ref, self._arena_kda[0], self.pool.buf_shape("kda_proj"), pd),
            "mla_w1": mla[0],
            "mla_qb": mla[1],
        }
        if self.w1_split is not None:
            ext["mla_g"] = mla[4]
        return ext, ring, mla[2], mla[3]

    # ----- scratch -----
    def _scratch(self):
        d, lay, kd, ml, vl = self.spec.dims, self.layout, self.kda, self.mla, self.vl
        li, h_k, h_m = lay.lane_inter, kd.heads, ml.heads
        R = self.R
        smem_rows, smem_w = (
            self.moe.grouped_smem() if self.moe.grouped else (R * d.top_k, R * d.top_k)
        )
        bank = (
            (BANK_ROWS, R) if self.rows_mode else (BANK_ROWS,)
        )  # rows: slot-major, one bank per row
        s = {}
        s["pool"] = self.pool.scratch_shapes()
        s["arena"] = [self.arena.scratch_shape()] if self.use_arena else []
        s["region_m"] = (
            [self.region_m.scratch_shape()] if self.region_m is not None else []
        )
        ring = (
            []
            if self.use_arena
            else [pltpu.VMEM(sh, dt) for sh, dt in self._expert_ring_shapes()]
        )
        s["moe"] = ring + [
            pltpu.SemaphoreType.DMA((self.moe.ring, 4)),
            pltpu.VMEM((R, d.latent), jnp.float32),  # gathered latent
            pltpu.VMEM((R, d.latent), jnp.float32),  # routed accumulator
            pltpu.SMEM((R * d.top_k,), jnp.int32),
            pltpu.SMEM(
                (smem_rows,), jnp.int32
            ),  # route rows (grouped: expert slot -> distinct index)
            pltpu.SMEM(
                (smem_w,), jnp.float32
            ),  # route weights (grouped: distinct x row weights)
            pltpu.SMEM((2,), jnp.int32),
            pltpu.SMEM((R * d.top_k,), jnp.int32),  # selected expert ids
            pltpu.SMEM((R * d.top_k,), jnp.float32),  # route weights
        ]
        s["kda"] = [
            pltpu.VMEM((128, kd.hd), jnp.bfloat16),  # f_b
            pltpu.VMEM((3, 3 * kd.hd), jnp.float32),  # conv state
            pltpu.VMEM(
                (3, 1, 3 * kd.hd), jnp.float32
            ),  # conv state in the pool's layout
            pltpu.VMEM((h_k, HEAD_DIM, HEAD_DIM), jnp.float32),  # recurrent state
            pltpu.SemaphoreType.DMA((2, 2)),  # state [read/write, conv/rec]
        ]
        if (
            self.rows_mode
        ):  # two staging slots for the verify checkpoints of rows 0 .. R - 2
            for _ in range(2):
                s["kda"] += [
                    pltpu.VMEM((3, 1, 3 * kd.hd), jnp.float32),
                    pltpu.VMEM((h_k, HEAD_DIM, HEAD_DIM), jnp.float32),
                    pltpu.SemaphoreType.DMA((2, 2)),
                ]
            if self._ck_all:  # all R checkpoint states, written by one DMA per family after the last row
                s["kda"] += [
                    pltpu.VMEM((R, 3, 1, 3 * kd.hd), jnp.float32),
                    pltpu.VMEM((R, h_k, HEAD_DIM, HEAD_DIM), jnp.float32),
                    pltpu.SemaphoreType.DMA((2,)),
                ]
        s["mla"] = (
            []
            if self.use_arena
            else [
                pltpu.VMEM((h_m, 128, 512), jnp.bfloat16),
                pltpu.VMEM((h_m, 512, 128), jnp.bfloat16),
            ]
        ) + [
            pltpu.VMEM((2, self.mla.bp * PAGE // 2, 2, ROW), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2, self.mla.bp)),
            pltpu.VMEM((self.mla.word_rows, 2, ROW), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.VMEM((h_m * R, 512), jnp.float32),
        ]
        s["layer"] = [
            pltpu.VMEM(
                (2, vl.ROWS, d.hidden), jnp.float32
            ),  # layer_vec slabs by layer parity
            pltpu.VMEM(
                bank + (d.hidden,), jnp.float32
            ),  # attention-residual bank ([row,] slot, H)
            pltpu.VMEM((R, d.hidden), jnp.float32),  # stream head h
            pltpu.VMEM((R, d.hidden), jnp.bfloat16),  # MLP input
            pltpu.VMEM((R, d.hidden), jnp.bfloat16),  # MoE output
            pltpu.SMEM((1,), jnp.int32),  # nvb
            pltpu.SemaphoreType.DMA((2,)),  # layer_vec slots
            pltpu.SemaphoreType.DMA((2,)),  # attention auxiliaries (f_b | w_kc, w_vc)
            pltpu.VMEM(
                bank + (128,), jnp.float32
            ),  # inv_rms of each bank row (every lane)
        ]
        m = self.moe
        s["w1_gather"] = (
            self.w1_gather.scratch_shapes() if self.w1_split is not None else []
        )
        s["attn_out"] = (
            self.kda.out.scratch_shapes() if self.spec.relax.pair_oproj else []
        )
        s["colls"] = [x for spec in m.coll_specs() for x in spec.scratch_shapes()]
        s["head"] = (
            (
                [
                    pltpu.VMEM((_EMBED_BLOCK, d.hidden), jnp.bfloat16),
                    pltpu.SemaphoreType.DMA((1,)),
                ]
                + self.head.scratch_shapes()
            )
            if self.fused_head
            else []
        )
        s["hidden_ar"] = (
            self._hidden_spec.scratch_shapes()
        )  # dense MLP and (default) attention o_proj
        s["verify"] = (
            [pltpu.VMEM((R, d.hidden), jnp.bfloat16), pltpu.SemaphoreType.DMA((1,))]
            if self.verify_rows > 1
            else []
        )  # row staging for the HBM rows
        self._groups = [(k, len(v)) for k, v in s.items()]
        return [x for v in s.values() for x in v]

    def _split_scratch(self, scratch):
        out, i = {}, 0
        for k, n in self._groups:
            out[k] = scratch[i : i + n]
            i += n
        return out

    def input_names(self) -> tuple[str, ...]:
        return input_names(self.spec, self.stacked, self.fused_head)

    @property
    def n_fixed_outputs(self) -> int:
        """Outputs before the aliased states: final row, per-layer rows, route ids[, token]."""
        return 4 if self.fused_head else 3

    # ----- kernel -----
    def kernel(self, *refs):
        with trace_phases(self.trace):
            self._kernel(*refs)

    def _kernel(self, *refs):
        names = self.input_names()
        n_in = len(names)
        r = dict(zip(names, refs[:n_in]))
        nf = self.n_fixed_outputs
        outs = refs[n_in : n_in + nf]
        # Pool states are written through the input refs (any aliased outputs are the same buffers).
        sc = self._split_scratch(refs[n_in + nf + self._n_alias_out :])
        nk, nm = len(self.spec.kda_layers), len(self.spec.mla_layers)
        me = jax.lax.axis_index("x")

        moe_s, mla_s = list(sc["moe"]), list(sc["mla"])
        ext = (
            self._region_m_views(sc["region_m"][0]) if self.region_m is not None else {}
        )
        if self.use_arena:
            arena_ext, ring, wkc, wvc = self._arena_views(sc["arena"][0])
            self.pool.bind(sc["pool"], external={**arena_ext, **ext})
            moe_s = ring + moe_s
            mla_s = [wkc, wvc] + mla_s
        else:
            self.pool.bind(sc["pool"], external=ext)
        fb_v, conv_s, conv3_s, rec_s, state_sems = sc["kda"][:5]
        ck = sc["kda"][5:]
        colls = self.moe.bind_colls(me, sc["colls"])
        cls = RowScatterAllReduce if self.rows_mode else AllReduce
        self._hidden_ar = cls(self._hidden_spec, me, *sc["hidden_ar"])
        self._hidden_ar.init()
        if self.spec.relax.pair_oproj:
            self._attn_out = self.kda.out.bind(me, sc["attn_out"])
            self._attn_out.init()
        else:
            self._attn_out = AllReduceProjection(self.kda.out, self._hidden_ar)
            self._attn_out.skip_reduce = bool(
                {"attn_ar", "attn_reduce"} & set(self.debug_skip)
            )

        if self.w1_split is not None:
            lane = me % self.spec.topo.lanes
            self._w1_start = pl.multiple_of(self.w1_split.start_block(lane) * 128, 128)
            self._w1_gather = AllGather(self.w1_gather, me, *sc["w1_gather"])
            self._w1_gather.init()
        x0_copy = None
        if self.fused_head:
            self._x0_v, x0_sem, head_sems, *exchange = sc["head"]
            # The fed token's (tile-aligned) embedding block lands while the ranks meet at the barrier.
            start = pl.multiple_of(
                r["meta"][TOKEN_META] // _EMBED_BLOCK * _EMBED_BLOCK, _EMBED_BLOCK
            )
            x0_copy = pltpu.make_async_copy(
                r["embed"].at[pl.ds(start, _EMBED_BLOCK)], self._x0_v, x0_sem.at[0]
            )
            x0_copy.start()
            ring = Arena.view(sc["arena"][0], 0, *self._head_ring)
            self._greedy = GreedyHead(
                self.head, me, r["lm_head"], ring, head_sems, *exchange
            )
            self._greedy.init()
        entry_barrier(me, self.spec.topo)
        if x0_copy is not None:
            x0_copy.wait()
        self._r, self._me, self._colls = r, me, colls
        self._ring, self._ring_sems = list(moe_s[0:4]), moe_s[4]
        (
            self._lat_v,
            self._acc_v,
            self._lid,
            self._lrow,
            self._lw,
            self._count,
            self._ids_s,
            self._w_s,
        ) = moe_s[5:]
        self._fb_v, self._conv_s, self._conv3_s, self._rec_s = (
            fb_v,
            conv_s,
            conv3_s,
            rec_s,
        )
        (
            self._wkc_v,
            self._wvc_v,
            self._kv_buf,
            self._kv_sems,
            self._word_v,
            self._wsem,
            self._oacc_v,
        ) = mla_s
        (
            self._vec_v,
            self._bank_v,
            self._h_v,
            self._x_v,
            self._moe_out_v,
            self._nvb_s,
            self._vec_sems,
            self._aux_sems,
            self._bank_inv_v,
        ) = sc["layer"]
        self._outs = outs
        if self.stacked:
            conv_pools, rec_pools, self._mla_pools = (
                [r["kda_conv"]],
                [r["kda_rec"]],
                [r["mla_cache"]],
            )
        else:
            conv_pools = [r[f"kda_conv_{k}"] for k in range(nk)]
            rec_pools = [r[f"kda_rec_{k}"] for k in range(nk)]
            self._mla_pools = [r[f"mla_cache_{j}"] for j in range(nm)]
        self._state = _KdaStateIO(
            conv_pools,
            rec_pools,
            self.stacked,
            r["meta"][1],
            conv3_s,
            rec_s,
            state_sems,
        )
        self._ck_io = [
            _KdaStateIO(
                conv_pools,
                rec_pools,
                self.stacked,
                r["meta"][1],
                *ck[3 * i : 3 * i + 3],
            )
            for i in range(2 if ck else 0)
        ]
        self._ck_rows = tuple(ck[6:9]) if self._ck_all and ck else None
        self._kda_pools = (conv_pools, rec_pools)
        self._kv_len, self._t = r["meta"][0], None
        if self.rows_mode:
            self._row_v, self._row_sem = sc["verify"]
            self._state.write_row = r["meta"][2]
            if self.moe.grouped:

                @pl.loop(0, self.moe.grouped_smem()[0])
                def _(i):
                    self._lrow[i] = (
                        0  # the grouped compaction's "seen" table starts empty
                    )

            if self.prefill_blocks:
                meta, R = r["meta"], self.R
                n = jnp.minimum(meta[3], self.prefill_blocks * R)

                @pl.loop(0, (n + R - 1) // R)
                def _(b):
                    self._blk, self._kv_len = b, meta[0] + b * R
                    self._state.read_row = jnp.where(
                        b == 0, meta[1], meta[2] + (b * R - 1) * meta[4]
                    )
                    self._run()
            else:
                self._run()
        elif self.verify_rows > 1:
            self._row_v, self._row_sem = sc["verify"]

            meta = r[
                "meta"
            ]  # [kv_len, read row, write base, rows, write stride, reset] (see prefill_blocks)

            @pl.loop(0, jnp.minimum(meta[3], self.verify_rows))
            def _(t):
                self._t, self._kv_len = t, meta[0] + t
                self._state.read_row = jnp.where(
                    t == 0, meta[1], meta[2] + (t - 1) * meta[4]
                )
                self._state.write_row = meta[2] + t * meta[4]
                self._run()
        elif self.debug_repeat > 1:

            @pl.loop(0, self.debug_repeat)
            def _(i):
                self._run()
        else:
            self._run()

    def _stream(self, key, name, idx=None, prefetched: int = 0, cols=None):
        """A pool stream of weight ``name`` (layer ``idx``; ``cols`` = (start, width) of its
        columns), with its column scales in e4m3 mode."""

        def pick(ref):
            ref = ref if idx is None else ref.at[idx]
            return ref if cols is None else ref.at[:, pl.ds(cols[0], cols[1])]

        scale = pick(self._r[name + "_s"]) if self.spec.relax.dense_fp8 else None
        return self.pool.stream(
            key, pick(self._r[name]), prefetched, name=name, scale_ref=scale
        )

    # ----- per-layer prefetch -----
    def _vec_copy(self, l):
        slot = l % 2
        return pltpu.make_async_copy(
            self._r["layer_vec"].at[l], self._vec_v.at[slot], self._vec_sems.at[slot]
        )

    def _fb_copy(self, k):
        return pltpu.make_async_copy(
            self._r["kda_fb"].at[k], self._fb_v, self._aux_sems.at[0]
        )

    def _wkc_copies(self, j):
        r = self._r
        return (
            pltpu.make_async_copy(
                r["mla_wkc"].at[j], self._wkc_v, self._aux_sems.at[0]
            ),
            pltpu.make_async_copy(
                r["mla_wvc"].at[j], self._wvc_v, self._aux_sems.at[1]
            ),
        )

    def _prefetch_layer(self, l):
        """Start layer ``l``'s slab, attention auxiliaries and KDA state rows."""
        tables = self._r["tables"]
        if self._aux_off:
            return
        self._vec_copy(l).start()

        @pl.when(tables[0, l] == _KDA)
        def _():
            k = tables[1, l]
            self._fb_copy(k).start()
            self._state.start_read(k)
            for st in self._attn_streams(_KDA, k, prefetched=False) + self._early_oproj(
                _KDA, k
            ):
                st.prefetch(st.slots)

        @pl.when(tables[0, l] == _MLA)
        def _():
            j = tables[2, l]
            for cp in self._wkc_copies(j):
                cp.start()
            for st in self._attn_streams(_MLA, j, prefetched=False) + self._early_oproj(
                _MLA, j
            ):
                st.prefetch(st.slots)

    def _attn_streams(self, kind, idx, *, prefetched: bool):
        """The cross-layer-prefetched attention projections of a KDA / MLA layer."""
        if not self.prefetch_attn:
            return []
        return [
            self._attn_w(key, idx, prefetched=prefetched)
            for key in self._attn_keys(kind)
        ]

    def _early_oproj(self, kind, idx) -> list:
        if not (self.oproj_early and self.prefetch_attn):
            return []
        return [self._oproj_stream(kind, idx, prefetched=False)]

    def _attn_keys(self, kind) -> tuple[str, ...]:
        if kind == _KDA:
            return ("kda_proj",)
        return ("mla_w1", "mla_qb") + (("mla_g",) if self.w1_split is not None else ())

    def _attn_w(self, key, idx, *, prefetched: bool):
        """Attention projection stream ``key`` of layer ``idx`` (``prefetched``: its whole ring)."""
        name, cols = key, None
        if key == "mla_w1" and self.w1_split is not None:
            cols = (self._w1_start, self.w1_split.width)
        elif key == "mla_g":
            name, cols = "mla_w1", (REPLICATED, self.mla.hd)
        st = self._stream(key, name, idx, cols=cols)
        return (
            self._stream(key, name, idx, prefetched=st.slots, cols=cols)
            if prefetched
            else st
        )

    def _moe_streams(self, l, *, prefetched: bool):
        """This layer's MoE streams; the ones ``prefetch_moe`` starts early are marked prefetched."""
        r, pool, mo = self._r, self.pool, l - 1
        out = {}
        for name in ("router", "latent_down", "shared_gu", "latent_up"):
            st = self._stream(name, name, mo)
            early = prefetched and name in self._early_moe
            out[name] = self._stream(
                name, name, mo, prefetched=st.slots if early else 0
            )
        out["shared_down"] = self._stream("hidden_out", "shared_down", mo)
        return out

    def _after_route(self, l):
        """Layer ``l``'s hand-off point (inside its MoE once the expert DMAs are queued,
        or before the dense MLP): drain its KDA state write-back, prefetch layer ``l + 1``."""
        tables = self._r["tables"]

        with phase("prefetch_next"):

            @pl.when((tables[0, l] == _KDA) & (not self._aux_off))
            def _():
                if self._ck_rows is not None:
                    if (
                        "attn" in self.debug_skip
                    ):  # the skipped attention balanced B1's write-back instead
                        self._state.wait_write()
                    else:
                        self._ck_rows_wait()
                    return
                self._state.wait_write()
                if self.rows_mode and "attn" not in self.debug_skip:
                    for io in self._ck_io[: min(2, self.R - 1)]:
                        io.wait_write()

            if self.mla.async_write:

                @pl.when(tables[0, l] == _MLA)
                def _():
                    self.mla.wait_new(self._mla_pools, self._word_v, self._wsem)

            @pl.when(l + 1 < self.spec.num_layers)
            def _():
                self._prefetch_layer(l + 1)

    # ----- layer pieces -----
    def _vec(self, l, name):
        row, rows, col, cols = self.vl.cells[name]
        return self._vec_v.at[l % 2, pl.ds(row, rows), pl.ds(col, cols)]

    def _norm(self, l, i):
        return self._vec_v[l % 2, pl.ds(self.vl.NORMS + i, 1), :]

    def _attn_stream(self, key, idx):
        st = self._attn_w(key, idx, prefetched=self.prefetch_attn)
        return _SkippedStream(st) if "attn_proj" in self.debug_skip else st

    def _body_oproj(self, kind, idx):
        st = self._oproj_stream(kind, idx, prefetched=True)
        return _SkippedStream(st) if "attn_oproj" in self.debug_skip else st

    def _mla_w1(self, j):
        if self.w1_split is None:
            return self._attn_stream("mla_w1", j)
        return SplitW1Projection(
            self.w1_split,
            self._attn_stream("mla_w1", j),
            self._attn_stream("mla_g", j),
            self._w1_gather,
        )

    def _oproj_stream(self, kind, idx, *, prefetched: bool):
        name = "kda_oproj" if kind == _KDA else "mla_oproj"
        st = self._stream(self._attn_key, name, idx)
        if prefetched and self.prefetch_attn:
            st = self._stream(self._attn_key, name, idx, prefetched=st.slots)
        return st

    def _prefetch_oproj(self, l):
        """The attention o_proj shares ``hidden_out`` with the previous MoE's shared_down,
        so it starts once that MoE is done, i.e. when this layer begins."""
        tables = self._r["tables"]
        for kind, col in ((_KDA, 1), (_MLA, 2)):

            @pl.when(tables[0, l] == kind)
            def _(kind=kind, col=col):
                st = self._oproj_stream(kind, tables[col, l], prefetched=False)
                st.prefetch(st.slots)

    def _kda_attn(self, l, k, x):
        with phase("kda"):
            return self._kda_attn_body(l, k, x)

    def _kda_attn_body(self, l, k, x):
        with phase("wait_kda_aux"):
            self._fb_copy(k).wait()
            self._state.wait_read()
        first = (
            self._blk
            if self._blk is not None
            else (None if self.rows_mode else self._t)
        )
        if first is not None:  # prefill of a fresh request: KDA state from zero

            @pl.when((first == 0) & (self._r["meta"][5] != 0))
            def _():
                self._conv3_s[...] = jnp.zeros(self._conv3_s.shape, self._conv3_s.dtype)
                self._rec_s[...] = jnp.zeros(self._rec_s.shape, self._rec_s.dtype)

        self._conv_s[...] = self._conv3_s[:, 0, :]
        if self.rows_mode:
            return self._kda_rows(l, k, x)
        out = self.kda.body(
            x,
            self._attn_stream("kda_proj", k),
            self._fb_v,
            self._body_oproj(_KDA, k),
            self._vec(l, "conv")[...],
            self._vec(l, "dt")[...],
            self._vec(l, "a_log")[...],
            self._vec(l, "onorm")[...],
            self._conv_s,
            self._rec_s,
            self._attn_out,
        )
        self._conv3_s[:, 0, :] = self._conv_s[...]
        self._state.start_write(k)
        return out

    def _ck_rows_desc(self, conv, rec, kidx, base):
        cbuf, rbuf, sem = self._ck_rows
        rows = pl.ds(base, self.R)
        dst = (
            (conv.at[kidx, rows], rec.at[kidx, rows])
            if self.stacked
            else (conv.at[rows], rec.at[rows])
        )
        return [
            pltpu.make_async_copy(cbuf, dst[0], sem.at[0]),
            pltpu.make_async_copy(rbuf, dst[1], sem.at[1]),
        ]

    def _ck_rows_copies(self, k, base):
        """Starters of the R-row checkpoint DMAs of KDA layer ``k``: one flat ``pl.when`` per pool
        (measured at TP32 B8: 0.06 ms per accepted token faster than a ``tree_when`` selection here)."""
        conv_pools, rec_pools = self._kda_pools
        if self.stacked:
            return [
                lambda: [
                    cp.start()
                    for cp in self._ck_rows_desc(conv_pools[0], rec_pools[0], k, base)
                ]
            ]
        out = []
        for kk, (conv, rec) in enumerate(zip(conv_pools, rec_pools)):

            def start(conv=conv, rec=rec, kk=kk):
                @pl.when(k == kk)
                def _():
                    for cp in self._ck_rows_desc(conv, rec, 0, base):
                        cp.start()

            out.append(start)
        return out

    def _ck_rows_wait(self):
        conv_pools, rec_pools = self._kda_pools
        for cp in self._ck_rows_desc(conv_pools[0], rec_pools[0], 0, 0):
            cp.wait()

    def _kda_rows(self, l, k, x):
        """Verify rows: the state after row t goes to checkpoint row ``meta[2] + t``; the last
        write stays in flight (drained at the hand-off, like B1's write-back)."""

        def checkpoint(t, conv_t, rec_t):
            meta = self._r["meta"]
            if self._ck_rows is not None:
                cbuf, rbuf, _ = self._ck_rows
                cbuf[t, :, 0, :] = conv_t
                rbuf[t] = rec_t
                if t == self.R - 1:
                    for cp in self._ck_rows_copies(k, meta[2]):
                        cp()
                return
            if self._blk is None:
                row = meta[2] + t
            else:
                g = self._blk * self.R + t
                # Stride 0 (prefill: every token's state goes to one row): only a block's last row
                # (the next block reads it) and the last token write it, so no two in-flight
                # checkpoint DMAs share a destination; the other rows go to the null slot 0.
                keep = (g < meta[3]) & (
                    (meta[4] != 0) | (g == meta[3] - 1) | (t == self.R - 1)
                )
                row = jnp.where(keep, meta[2] + g * meta[4], 0)
            # Rows 0 .. R-2 alternate between two staging slots (a slot is reused once its write
            # from two rows earlier landed); the last row goes through the B1 write-back buffers.
            io = self._state if t == self.R - 1 else self._ck_io[t % 2]
            if io is not self._state and t >= 2:
                io.wait_write()
            io.conv_v[:, 0, :] = conv_t
            io.rec_v[...] = rec_t
            io.start_write(k, row=row)

        return self.kda.body_rows(
            x,
            self._attn_stream("kda_proj", k),
            self._fb_v,
            self._body_oproj(_KDA, k),
            self._vec(l, "conv")[...],
            self._vec(l, "dt")[...],
            self._vec(l, "a_log")[...],
            self._vec(l, "onorm")[...],
            self._conv_s,
            self._rec_s,
            self._attn_out,
            checkpoint,
        )

    def _mla_attn(self, l, j, x):
        with phase("mla"):
            return self._mla_attn_body(l, j, x)

    def _mla_attn_body(self, l, j, x):
        r = self._r
        with phase("wait_mla_aux"):
            for cp in self._wkc_copies(j):
                cp.wait()
        body = self.mla.body_rows if self.rows_mode else self.mla.body
        return body(
            x,
            self._mla_w1(j),
            self._vec(l, "qnorm")[...],
            self._vec(l, "kvnorm")[...],
            self._attn_stream("mla_qb", j),
            self._wkc_v,
            self._wvc_v,
            self._body_oproj(_MLA, j),
            self._mla_pools,
            j,
            r["pages"],
            self._kv_len,
            self._kv_buf,
            self._kv_sems,
            self._word_v,
            self._wsem,
            self._oacc_v,
            self._attn_out,
        )

    def _in_norm(self, x, w):
        if "norm" in self.debug_skip:
            return x
        return rmsnorm(x, w, self.spec.dims.rms_eps, tree=self.mix.tree)

    def _aggregate(self, prefix, nvb, proj_w, norm_w, bank_sc=None):
        """(mixture of the bank and ``prefix``, inv_rms(prefix) or None)."""
        if "mixture" in self.debug_skip:
            return prefix, None
        if self.rows_mode:  # one bank per row, all rows at once
            return aggregate_rows(
                prefix,
                self._bank_v[...],
                self._bank_inv_v[:, :, 0:1],
                nvb,
                proj_w,
                norm_w,
                self.spec.dims.rms_eps,
                self.mix,
            )
        return aggregate_packed(
            prefix,
            self._bank_v[...],
            self._bank_inv_v[:, 0:1],
            nvb,
            proj_w,
            norm_w,
            self.spec.dims.rms_eps,
            self.mix,
            bank_sc,
        )

    def _attention_side(self, l):
        """Returns (P2, MLP input) and updates the bank / nvb."""
        d, tables = self.spec.dims, self._r["tables"]
        nvb_s = self._nvb_s
        with phase("wait_vec"):
            if not self._aux_off:
                self._vec_copy(l).wait()
        if self.prefetch_attn and not self.oproj_early:
            self._prefetch_oproj(l)
        prefetch_moe = None
        if self.prefetch_moe and not isinstance(l, int):

            def prefetch_moe():
                for name in self._early_moe:
                    st = self._moe_streams(l, prefetched=False)[name]
                    st.prefetch(st.slots)

        late = self.moe_prefetch_late and "attn" not in self.debug_skip
        if prefetch_moe is not None and not late:
            prefetch_moe()
        # Late: the MoE rings start once the attention o_proj reduction is under way (HBM idle).
        self._attn_out.before_reduce = prefetch_moe if late else None
        h = self._h_v[...]
        nvb, write = nvb_s[0], tables[3, l]
        with phase("attn_in"):
            mixed, h_inv = self._aggregate(h, nvb, self._norm(l, 0), self._norm(l, 1))
            a_in = self._in_norm(mixed, self._norm(l, 2))
        if self.capture_mixture:
            self._capture(l - 1, mixed)

        @pl.when((write == 1) & ("mixture" not in self.debug_skip))
        def _():
            if self.rows_mode:  # slot nvb of every row's bank
                hit = jax.lax.broadcasted_iota(jnp.int32, (BANK_ROWS, 1, 1), 0) == nvb
                self._bank_v[...] = jnp.where(hit, h[None], self._bank_v[...])
                inv = h_inv if h_inv is not None else jnp.ones((self.R, 1), jnp.float32)
                self._bank_inv_v[...] = jnp.where(hit, inv[None], self._bank_inv_v[...])
            else:
                bank_write(
                    self._bank_v,
                    self._bank_inv_v,
                    h,
                    h_inv if h_inv is not None else 1.0,
                    nvb,
                )

        mlp_bank_sc = None
        if self.mixture_early_scores and "mixture" not in self.debug_skip:
            mlp_bank_sc = row_scores(
                self._bank_v[...],
                self._bank_inv_v[:, 0:1],
                self._norm(l, 3),
                self._norm(l, 4),
                d.rms_eps,
                self.mix,
            )
        nvb = nvb + write
        nvb_s[0] = nvb
        x = a_in.astype(jnp.bfloat16)
        if "attn" in self.debug_skip:
            # Keep the DMA accounting of the skipped attention balanced.
            @pl.when(tables[0, l] == _KDA)
            def _():
                if not self._aux_off:
                    self._fb_copy(0).wait()
                    self._state.wait_read()
                    self._state.start_write(tables[1, l])
                for st in self._attn_streams(_KDA, 0, prefetched=True) + [
                    self._oproj_stream(_KDA, 0, prefetched=True)
                ]:
                    st.wait_prefetched()

            @pl.when(tables[0, l] == _MLA)
            def _():
                for cp in self._wkc_copies(0) if not self._aux_off else ():
                    cp.wait()
                for st in self._attn_streams(_MLA, 0, prefetched=True) + [
                    self._oproj_stream(_MLA, 0, prefetched=True)
                ]:
                    st.wait_prefetched()

            attn = jnp.zeros_like(h)
        else:
            attn = jax.lax.cond(
                tables[0, l] == _MLA,
                lambda: self._mla_attn(l, tables[2, l], x),
                lambda: self._kda_attn(l, tables[1, l], x),
            )
        p2 = jnp.where(write == 1, attn, bf16(h + attn))
        with phase("mlp_in"):
            m_in = self._in_norm(
                self._aggregate(
                    p2, nvb, self._norm(l, 3), self._norm(l, 4), mlp_bank_sc
                )[0],
                self._norm(l, 5),
            )
        return p2, m_in

    def _finish_layer(self, l, out):
        self._h_v[...] = out
        if self.layer_out:
            self._outs[1][pl.ds(l, 1)] = out.astype(jnp.bfloat16)[None]
        if not self.capture_mixture:
            self._capture(l, out)

    def _capture(self, l, value):
        """Rows of ``value`` to the capture output when layer ``l`` is a captured layer."""
        for c, layer in enumerate(self.capture_layers if self.verify_rows > 1 else ()):

            def capture(c=c):
                self._row_v[...] = value.astype(jnp.bfloat16)
                if self._blk is not None:
                    dst = self._outs[1].at[c, self._blk]
                else:
                    dst = (
                        self._outs[1].at[c]
                        if self.rows_mode
                        else self._outs[1].at[c, self._t]
                    )
                self._row_copy(self._row_v, dst)

            if isinstance(l, int):
                if l == layer:
                    capture()
            else:
                pl.when(l == layer)(capture)

    def _moe_layer(self, l):
        r, m = self._r, self.moe
        p2, m_in = self._attention_side(l)
        mo = l - 1
        self._x_v[...] = m_in.astype(jnp.bfloat16)
        streams = self._moe_streams(l, prefetched=self.prefetch_moe)
        if self.prefetch_moe:
            # hidden_out is free once the attention o_proj has been consumed.
            streams["shared_down"].prefetch(streams["shared_down"].slots)
        if "moe" in self.debug_skip:
            if self.prefetch_moe:
                for name in self._early_moe + ("shared_down",):
                    streams[name].wait_prefetched()
            self._after_route(l)
            self._moe_out_v[...] = jnp.zeros(
                self._moe_out_v.shape, self._moe_out_v.dtype
            )
        else:
            with phase("moe"):
                m._body(
                    self._me,
                    self._x_v,
                    self._vec(l, "router_bias"),
                    self._vec(l, "latent_norm"),
                    r["expert_gu"].at[mo],
                    r["expert_gu_scale"].at[mo],
                    r["expert_down"].at[mo],
                    r["expert_down_scale"].at[mo],
                    self._moe_out_v,
                    self._ids_s,
                    self._w_s,
                    streams,
                    self._ring,
                    self._ring_sems,
                    self._lat_v,
                    self._acc_v,
                    self._lid,
                    self._lrow,
                    self._lw,
                    self._count,
                    self._colls,
                    after_route=(lambda: self._after_route(l))
                    if self.handoff != "after_moe"
                    else None,
                )
            if self.handoff == "after_moe":
                self._after_route(l)
        ids_out = self._outs[2]
        for k in range(
            self.R * self.spec.dims.top_k if "ids" not in self.debug_skip else 0
        ):
            ids_out[l, k] = self._ids_s[k]
        self._finish_layer(l, bf16(self._moe_out_v[...].astype(jnp.float32) + p2))

    def _embedded_x0(self):
        """The fed token's row of the DMA'd embedding block, ``[1, H]`` f32."""
        row = (
            jax.lax.broadcasted_iota(jnp.int32, (_EMBED_BLOCK, 1), 0)
            == self._r["meta"][TOKEN_META] % _EMBED_BLOCK
        )
        return jnp.sum(
            jnp.where(row, self._x0_v[...].astype(jnp.float32), 0.0),
            axis=0,
            keepdims=True,
        )

    def _run(self):
        spec, d, r = self.spec, self.spec.dims, self._r
        self._bank_v[...] = jnp.zeros(self._bank_v.shape, jnp.float32)
        self._bank_inv_v[...] = jnp.ones(self._bank_inv_v.shape, jnp.float32)
        self._nvb_s[0] = 0
        if self.rows_mode:  # x0 / final / captures are whole [R, H] blocks in HBM
            self._row_copy(
                r["x0"] if self._blk is None else r["x0"].at[self._blk], self._row_v
            )
            self._h_v[...] = self._row_v[...].astype(jnp.float32)
        elif self._t is None:
            self._h_v[...] = (
                self._embedded_x0()
                if self.fused_head
                else r["x0"][...].astype(jnp.float32)
            )
        else:
            self._row_copy(r["x0"].at[self._t], self._row_v)
            self._h_v[...] = self._row_v[...].astype(jnp.float32)
        if self.mla.kv_window:
            self.mla.clear_kv(self._kv_buf)
        self._prefetch_layer(0)
        if self._aux_off:
            for slot in range(2):
                cp = pltpu.make_async_copy(
                    r["layer_vec"].at[0], self._vec_v.at[slot], self._vec_sems.at[slot]
                )
                cp.start()
                cp.wait()

        # Layer 0: dense MLP.
        p2, m_in = self._attention_side(0)
        # Region A over the expert ring also covers layer 0's dense gate/up ring: hand off after it.
        late0 = self.attn_region == "experts"
        if not late0:
            self._after_route(0)
        with phase("dense"):
            dense = self.dense.body(
                m_in.astype(jnp.bfloat16),
                p2,
                self._stream("dense_gu", "dense_gu"),
                self._stream("hidden_out", "dense_down"),
                self._hidden_ar,
            )
        if late0:
            self._after_route(0)
        self._finish_layer(0, dense)

        @pl.loop(1, spec.num_layers)
        def _(l):
            self._moe_layer(l)

        if self.fused_head:
            self._greedy.prefetch()
        fw = r["final_w"]
        h = self._h_v[...]
        mixed = self._aggregate(h, self._nvb_s[0], fw[0], fw[1])[0]
        if self.capture_mixture:
            self._capture(spec.num_layers - 1, mixed)
        final = rmsnorm(mixed, fw[2], d.rms_eps, tree=self.mix.tree)
        if self.rows_mode:
            self._row_v[...] = final.astype(jnp.bfloat16)
            self._row_copy(
                self._row_v,
                self._outs[0] if self._blk is None else self._outs[0].at[self._blk],
            )
        elif self._t is None:
            self._outs[0][...] = final.astype(jnp.bfloat16)
        else:
            self._row_v[...] = final.astype(jnp.bfloat16)
            self._row_copy(self._row_v, self._outs[0].at[self._t])
        if self.fused_head:
            with phase("head"):
                self._outs[3][0] = self._greedy.token(final.astype(jnp.bfloat16))

    def _row_copy(self, src, dst):
        """Synchronous DMA of verify rows (``[1, H]`` per token in the loop mode, ``[R, H]`` in rows mode)."""
        cp = pltpu.make_async_copy(src, dst, self._row_sem.at[0])
        cp.start()
        cp.wait()

    def build(self, mesh, shapes: dict, *, alias_states: bool = True):
        """jit(shard_map(pallas_call)) for JAX callers; ``shapes`` maps every input name to its
        per-rank (shape, dtype). ``alias_states`` declares the pool writes as aliased outputs
        (pure JAX); the torch path passes False and relies on the in-place writes."""
        from jax.sharding import PartitionSpec as P

        call, n_out = self.pallas_call(shapes, alias_states=alias_states)
        n_in = len(self.input_names())

        def local(*args):
            outs = call(*[a[0] for a in args])
            return tuple(o[None] for o in outs)

        ps = P("x")
        return jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(ps,) * n_in,
                out_specs=(ps,) * n_out,
                check_vma=False,
            )
        )

    def pallas_call(self, shapes: dict, *, alias_states: bool):
        spec, d = self.spec, self.spec.dims
        names = self.input_names()
        states = state_inputs(spec, self.stacked)
        slots = shapes["expert_gu"][0][1]
        if slots != self.layout.expert_slots:
            raise ValueError(
                f"expert_gu has {slots} expert slots per layer, the layout expects "
                f"{self.layout.expert_slots} (Relax.expert_replicas must match the loaded experts)"
            )
        space = {
            **{n: pltpu.SMEM for n in SMEM_INPUTS},
            **{n: pltpu.VMEM for n in VMEM_INPUTS},
            **{
                n: pltpu.HBM
                for n in HBM_INPUTS + HEAD_INPUTS + scale_inputs(spec) + states
            },
        }
        if (
            self.fused_head
            and shapes["lm_head"][0][0] * spec.topo.world != self.head_vocab
        ):
            raise ValueError(
                f"lm_head shard {shapes['lm_head'][0]} != head_vocab {self.head_vocab} / world"
            )
        n_in = len(names)
        self._n_alias_out = len(states) if alias_states else 0
        out_shape = (
            jax.ShapeDtypeStruct((1, d.hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                (spec.num_layers if self.layer_out else 1, 1, d.hidden), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct((spec.num_layers, self.R * d.top_k), jnp.int32),
        )
        out_specs = (
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        )
        if self.verify_rows > 1:
            rows = self.verify_rows
            space["x0"] = pltpu.HBM
            row = (
                (1,) if not self.rows_mode else ()
            )  # loop mode: a leading untiled axis per row
            if self.prefill_blocks:  # [blocks, R, H]
                rows, row = self.prefill_blocks, (self.R,)
            out_shape = (
                jax.ShapeDtypeStruct((rows,) + row + (d.hidden,), jnp.bfloat16),
                jax.ShapeDtypeStruct(
                    (max(1, len(self.capture_layers)), rows) + row + (d.hidden,),
                    jnp.bfloat16,
                ),
                out_shape[2],
            )
            out_specs = (
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                out_specs[2],
            )
        if self.fused_head:
            out_shape += (jax.ShapeDtypeStruct((1,), jnp.int32),)
            out_specs += (pl.BlockSpec(memory_space=pltpu.SMEM),)
        aliases = {}
        if alias_states:
            out_shape += tuple(jax.ShapeDtypeStruct(*shapes[n]) for n in states)
            out_specs += (pl.BlockSpec(memory_space=pltpu.HBM),) * len(states)
            aliases = {
                n_in - len(states) + i: self.n_fixed_outputs + i
                for i in range(len(states))
            }
        call = pl.pallas_call(
            self.kernel,
            out_shape=out_shape,
            in_specs=[pl.BlockSpec(memory_space=space[n]) for n in names],
            out_specs=out_specs,
            scratch_shapes=self._scratch(),
            input_output_aliases=aliases,
            compiler_params=pltpu.CompilerParams(
                collective_id=self.collective_id, vmem_limit_bytes=64 * 1024 * 1024
            ),
        )
        return call, len(out_shape)
