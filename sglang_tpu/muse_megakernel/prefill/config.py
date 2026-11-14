# SPDX-License-Identifier: Apache-2.0
"""PFConfig: the static description of one compiled prefill program."""

import msgspec

# The two-level collectives (coll.PairColl) route over a 2x2 grid of two-core chips.
PREFILL_TP = 8
# Row tiles of the fused kernels; larger ones overflow VMEM.
ATTN_OUT_TILE = 64
POST_MOE_TILE = 128
# post_moe tiles POST_MOE_TILE rows as POST_MOE_TILE / tp rows of each rank's token block.
ROW_ALIGN = POST_MOE_TILE


class PFConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Model geometry (per device) and chunk shape of one compiled prefill program."""

    hidden: int
    head_dim: int
    q_heads: int
    kv_heads: int
    n_layers: int
    n_experts: int
    top_k: int
    expert_hidden: int
    tp: int
    rms_eps: float
    post_eps: float
    scale_query_by: float
    attn_scale: float
    router_mult: float
    route_scale: float
    router_eps: float
    window: int
    global_layers: tuple
    capture_layers: tuple
    # padded token rows of the chunk
    rows: int
    # the expert-input and post-expert-projection all-gathers carry e4m3 blocks, one scale each
    fp8_gather: bool = False
    # [lo, hi) stacked expert slab groups, shared with the decode megakernel
    expert_groups: tuple = ()
    # layers whose fp4 experts are block-requantized (SGLANG_TPU_MUSE_MK_QBLOCK): W4A16 MoE
    w4a16_layers: tuple = ()

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkvg_dim(self) -> int:
        return 2 * self.q_dim + 2 * self.kv_dim

    def use_rope(self, layer: int) -> bool:
        return layer not in self.global_layers

    def validate(self) -> "PFConfig":
        if self.tp != PREFILL_TP:
            raise ValueError(
                f"the Muse prefill program runs at TP {PREFILL_TP}, got {self.tp}"
            )
        if self.rows % ROW_ALIGN:
            raise ValueError(
                f"{self.rows}-row extend step: the Muse prefill program needs multiples of "
                f"{ROW_ALIGN} (tpu_compile_tokens)"
            )
        return self
