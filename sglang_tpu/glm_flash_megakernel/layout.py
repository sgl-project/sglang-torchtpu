"""The single packed weight layout that prefill, decode and the megakernel all read.

Weight residency and transfer layout
------------------------------------
Prefill, reference decode and the megakernel consume one resident representation.
Budget its sharded and replicated arrays against usable HBM measured with all
devices active, including cache, scratch and loading transients.

1. Put dynamically indexed layers and experts on a leading axis of stacked
   `[N, rows, cols]` arrays so DMA operates on whole trailing tiles. Probe the
   exact shapes and dtypes under the pinned compiler.
2. Pack small replicated tensors into one contiguous block per layer to
   amortize transfer setup. Compare batched and separate fetches in the full step.
3. Replicate small mHC, router and low-rank projections when their additional
   weight traffic costs less than a reduction. Measure both terms with the
   target topology and payload sizes before changing the sharding.

Format versioning
-----------------
`FORMAT_VERSION` is written into the manifest and checked on load.  A packed
directory with an incompatible version is rejected before loading arrays.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Bump whenever the array set or any array's meaning changes. Validate the
# manifest before loading so incompatible packed arrays fail at the boundary.
FORMAT_VERSION = "glm53-flash-mk/2"

# Lane width on this target; a sharded dimension must stay a multiple of it or the
# tile planner has no legal tile to pick.
LANE = 128


def pad_to(n: int, m: int) -> int:
    return -(-n // m) * m


@dataclasses.dataclass(frozen=True)
class ShardPlan:
    """How the model is split across `tp` devices."""

    tp: int

    def split(self, n: int, what: str) -> int:
        if n % self.tp:
            raise ValueError(f"{what}: {n} not divisible by tp={self.tp}")
        per = n // self.tp
        if per % LANE and per > LANE:
            raise ValueError(
                f"{what}: per-device {per} is not a multiple of the lane width "
                f"{LANE}; pad the dimension to tp*{LANE} before sharding"
            )
        return per


@dataclasses.dataclass(frozen=True)
class PackedLayout:
    spec: object           # Glm53FlashSpec
    tp: int

    # ------------------------------------------------------------ derived
    @property
    def plan(self) -> ShardPlan:
        return ShardPlan(self.tp)

    @property
    def vocab_padded(self) -> int:
        """lm_head output must be shardable *and* tileable after sharding.

        154880 / 8 = 19360 and 19360 is not a multiple of 128, so the planner has no
        legal tile.  Pad to a multiple of tp * LANE and mask the added columns to a
        large negative sentinel at sampling time -- zeroing them is wrong, because a
        zero logit beats every genuinely negative one in an argmax.
        """
        return pad_to(self.spec.vocab_size, self.tp * LANE)

    @property
    def moe_layers(self) -> tuple[int, ...]:
        return tuple(
            i for i, t in enumerate(self.spec.mlp_layer_types) if t == "sparse"
        ) + (self.spec.mtp_layer_idx,)

    @property
    def expert_slots(self) -> int:
        return len(self.moe_layers) * self.spec.n_routed_experts

    def expert_slot(self, layer: int, expert: int) -> int:
        return self.moe_layers.index(layer) * self.spec.n_routed_experts + expert

    # ------------------------------------------------------- array catalogue
    def arrays(self) -> dict[str, tuple[tuple[int, ...], str, str]]:
        """name -> (per-device shape, dtype, one-line purpose).

        Shapes are per device.  `replicated` arrays hold the same bytes on every
        device; the rest hold that device's shard.
        """
        s, p = self.spec, self.plan
        H = s.hidden_size
        out: dict[str, tuple[tuple[int, ...], str, str]] = {}

        # ---- embedding and head -------------------------------------------
        h_sh = p.split(H, "hidden")
        out["embed_tokens"] = (
            (s.vocab_size, 1, h_sh), "bfloat16",
            "hidden-sharded embedding; the [V,1,D] form keeps the traced token id "
            "on a leading untiled axis so the row DMA is legal for bf16",
        )
        out["lm_head"] = (
            (p.split(self.vocab_padded, "vocab"), H), "bfloat16",
            "vocab-sharded output projection over the padded vocabulary",
        )
        out["final_norm"] = ((1, H), "bfloat16", "replicated: model.norm")

        # ---- MoE experts: 95% of the bytes --------------------------------
        I_sh = p.split(s.moe_intermediate_size, "moe_intermediate")
        bs0, bs1 = s.weight_block_size
        out["expert_gate_up"] = (
            (self.expert_slots, 2 * I_sh, H), "float8_e4m3fn",
            "gate and up stacked so one traced-index DMA fetches both",
        )
        out["expert_down"] = (
            (self.expert_slots, H, I_sh), "float8_e4m3fn", "expert down projection",
        )
        # Scales are stacked per *layer*, not per expert: 288 experts' scales are
        # 74 KiB, so one contiguous fetch replaces eight scattered ones.
        out["expert_gate_up_scale"] = (
            (len(self.moe_layers), s.n_routed_experts, 2 * I_sh // bs0, H // bs1),
            "float32", "all experts' gate/up block scales for a layer, one DMA",
        )
        out["expert_down_scale"] = (
            (len(self.moe_layers), s.n_routed_experts, H // bs0, max(1, I_sh // bs1)),
            "float32", "all experts' down block scales for a layer, one DMA",
        )
        out["shared_gate_up"] = (
            (len(self.moe_layers), 2 * I_sh, H), "float8_e4m3fn", "shared expert gate/up",
        )
        out["shared_down"] = (
            (len(self.moe_layers), H, I_sh), "float8_e4m3fn", "shared expert down",
        )
        # The shared expert is fp8 like the routed ones; its scales are plain
        # blockwise [16,32]/[32,16] in this checkpoint, with no blockwise-to-
        # channelwise conversion, which was verified against the headers rather
        # than assumed from other GLM loaders.
        out["shared_gate_up_scale"] = (
            (len(self.moe_layers), 2 * I_sh // bs0, H // bs1), "float32", "",
        )
        out["shared_down_scale"] = (
            (len(self.moe_layers), H // bs0, max(1, I_sh // bs1)), "float32", "",
        )

        # ---- KDA, sharded over heads --------------------------------------
        kda = s.kda_layers
        Q_sh = p.split(s.kda_qkv_dim, "kda_qkv")
        out["kda_qkv"] = (
            (len(kda), 3 * Q_sh, H), "bfloat16",
            "q,k,v column-parallel over heads, stacked in that order",
        )
        out["kda_o"] = ((len(kda), H, Q_sh), "bfloat16", "o_proj row-parallel")
        out["kda_fg_b"] = (
            (len(kda), 2 * Q_sh, s.linear_head_dim), "bfloat16",
            "forget-gate and output-gate 'b' projections, column-parallel",
        )
        out["kda_dt_bias"] = ((len(kda), 1, Q_sh), "float32", "per-channel, sharded with the heads")
        # These are per-head/per-channel quantities.  Replicating them and slicing
        # by device rank inside the kernel needs a dynamic slice of a *value*,
        # which does not lower in Pallas; sharding them here removes the slice and
        # cuts replicated traffic at the same time.
        out["kda_conv"] = (
            (len(kda), 3 * Q_sh, s.linear_conv_kernel_dim), "bfloat16",
            "q,k,v depthwise kernels, sharded with the channels",
        )
        out["kda_b"] = (
            (len(kda), p.split(s.linear_num_heads, "kda_heads"), H), "bfloat16",
            "b_proj, sharded with the heads",
        )
        out["kda_alog"] = (
            (len(kda), 1, p.split(s.linear_num_heads, "kda_heads")), "float32",
            "A_log, sharded with the heads",
        )

        # ---- DSA, sharded over heads --------------------------------------
        dsa = s.dsa_layers + (s.mtp_layer_idx,)
        qh_sh = p.split(s.num_attention_heads * s.qk_head_dim, "dsa_q")
        vh_sh = p.split(s.num_attention_heads * s.v_head_dim, "dsa_v")
        kvb_sh = p.split(s.num_attention_heads * (s.qk_nope_head_dim + s.v_head_dim), "dsa_kvb")
        out["dsa_q_b"] = ((len(dsa), qh_sh, s.q_lora_rank), "float8_e4m3fn", "q_b column-parallel")
        out["dsa_q_b_scale"] = (
            (len(dsa), qh_sh // bs0, pad_to(s.q_lora_rank, bs1) // bs1), "float32", "q_b block scales",
        )
        out["dsa_kv_b"] = ((len(dsa), kvb_sh, s.kv_lora_rank), "bfloat16", "kv_b column-parallel (BF16 in the checkpoint)")
        out["dsa_o"] = ((len(dsa), H, vh_sh), "float8_e4m3fn", "o_proj row-parallel")
        out["dsa_o_scale"] = ((len(dsa), H // bs0, vh_sh // bs1), "float32", "o_proj block scales")

        # ---- dense MLP prefix ---------------------------------------------
        dense = tuple(i for i, t in enumerate(s.mlp_layer_types) if t == "dense")
        F_sh = p.split(s.intermediate_size, "intermediate")
        out["dense_gate_up"] = ((len(dense), 2 * F_sh, H), "float8_e4m3fn", "dense gate/up stacked")
        out["dense_gate_up_scale"] = ((len(dense), 2 * F_sh // bs0, H // bs1), "float32", "")
        out["dense_down"] = ((len(dense), H, F_sh), "float8_e4m3fn", "dense down, row-parallel")
        out["dense_down_scale"] = ((len(dense), H // bs0, F_sh // bs1), "float32", "")

        # ---- replicated tensors that are too large for the shared slab -----
        # A fixed-height slab is addressed directly by the layer index, so its
        # height is paid 46 times.  Anything that would dominate it gets its own
        # array: `eh_proj` alone is 64 MiB and exists in exactly one layer, which
        # cost 4.2 GiB per device when it sat in the slab.
        out["moe_router"] = (
            (len(self.moe_layers), s.n_routed_experts, H), "bfloat16",
            "replicated: sharding 288x4096 would buy less than the collective costs",
        )
        out["dsa_q_a"] = (
            (len(dsa), s.q_lora_rank, H), "float8_e4m3fn",
            "replicated: q_a feeds both the attention q and the indexer q",
        )
        out["dsa_q_a_scale"] = (
            (len(dsa), s.q_lora_rank // bs0, H // bs1), "float32", "",
        )
        out["dsa_kv_a"] = (
            (len(dsa), s.kv_lora_rank + s.qk_rope_head_dim, H), "float8_e4m3fn",
            "replicated: its output is the compressed KV written to cache",
        )
        out["dsa_kv_a_scale"] = (
            (len(dsa), (s.kv_lora_rank + s.qk_rope_head_dim) // bs0, H // bs1), "float32", "",
        )
        out["indexer_wq_b"] = (
            (len(dsa), s.index_n_heads * s.index_head_dim, s.q_lora_rank), "bfloat16",
            "replicated: the per-head index scores are summed over all heads, so "
            "sharding heads would need an all-reduce of the whole pool-score vector",
        )
        out["mtp_eh_proj"] = (
            (1, H, p.split(2 * H, "mtp_eh_in")), "bfloat16",
            "MTP draft input projection, row-parallel over the concatenation",
        )

        # ---- the one replicated small block per layer ----------------------
        out["smalls"] = (
            (s.num_hidden_layers + 1, self.small_rows(), H), "bfloat16",
            "every replicated small tensor for a layer, contiguous, one DMA per layer",
        )
        out["smalls_f32"] = (
            (s.num_hidden_layers + 1, self.small_f32_rows(), H), "float32",
            "the float32 small tensors (mHC base/scale, A_log, router bias)",
        )
        return out

    # The small block is a fixed-height slab so that the layer index addresses it
    # directly; unused rows in a layer of the other type are zero.  The height is
    # the max over layer kinds, computed from the spec rather than hard-coded.
    def small_rows(self) -> int:
        """Height of the fixed slab, in rows of `hidden_size` bfloat16 values.

        Only tensors that stay small belong here; the slab height is paid once per
        layer slot whether that layer uses the row or not, so a single large tensor
        in one layer would be charged 46 times.  Computed from the spec, never
        hard-coded.
        """
        s = self.spec
        H = s.hidden_size
        nc = (2 + s.hc_mult) * s.hc_mult

        def rows(n_elems: int) -> int:
            return -(-n_elems // H)

        common = 2 + 2 * rows(nc * s.hc_mult * H)      # 2 norms + hc_attn_fn + hc_ffn_fn
        # b_proj, the conv kernels and A_log live in their own head-sharded arrays
        # (see `kda_conv` / `kda_b` / `kda_alog`), not here.
        kda = (
            rows(s.linear_head_dim * H)                # f_a_proj
            + rows(s.linear_head_dim * H)              # g_a_proj
            + rows(s.linear_head_dim)                  # o_norm
        )
        dsa = (
            rows(s.q_lora_rank)                        # q_a_layernorm
            + rows(s.kv_lora_rank)                     # kv_a_layernorm
            + rows(s.index_head_dim * H)               # indexer wk
            + 2 * rows(s.index_head_dim)               # indexer k_norm weight, bias
            + rows(s.index_n_heads * H)                # indexer weights_proj
            + rows(s.index_kpool * s.index_head_dim)   # compress ape
            + rows(s.index_head_dim * H)               # compress gate
        )
        mtp = 3                                        # enorm, hnorm, shared_head.norm
        return common + max(kda, dsa) + mtp

    def small_f32_rows(self) -> int:
        s = self.spec
        H = s.hidden_size
        nc = (2 + s.hc_mult) * s.hc_mult
        # hc base x2 and scale x2, plus the router bias.  A_log and dt_bias live in
        # their own head-sharded arrays.
        n = 2 * nc + 2 * 3 + s.n_routed_experts
        return -(-n // H)

    # ------------------------------------------------------------- budget
    def budget(self) -> dict[str, int]:
        """Per-device resident bytes by array."""
        sizes = {}
        for name, (shape, dt, _) in self.arrays().items():
            n = int(np.prod(shape))
            itemsize = {"bfloat16": 2, "float32": 4, "float8_e4m3fn": 1}[dt]
            sizes[name] = n * itemsize
        return sizes


def describe(spec, tp: int = 8) -> str:
    lay = PackedLayout(spec, tp)
    b = lay.budget()
    total = sum(b.values())
    lines = [f"packed layout {FORMAT_VERSION}, tp={tp}",
             f"vocab {spec.vocab_size} padded to {lay.vocab_padded} "
             f"({lay.vocab_padded // tp} per device = {lay.vocab_padded // tp // LANE} x {LANE})",
             f"expert slots: {lay.expert_slots} "
             f"({len(lay.moe_layers)} MoE layers x {spec.n_routed_experts})",
             "",
             f"{'array':<24} {'per-device shape':<34} {'dtype':<16} {'GiB':>9}"]
    for name, (shape, dt, _) in lay.arrays().items():
        lines.append(f"{name:<24} {str(tuple(shape)):<34} {dt:<16} {b[name]/2**30:9.3f}")
    lines.append(f"{'TOTAL':<24} {'':<34} {'':<16} {total/2**30:9.3f} GiB per device")
    return "\n".join(lines)


def main() -> int:
    import argparse

    from .config import Glm53FlashSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", nargs="?", default="./GLM-5.3-Flash")
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()
    print(describe(Glm53FlashSpec.from_checkpoint(args.ckpt_dir), args.tp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
