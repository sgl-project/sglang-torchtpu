# SPDX-License-Identifier: Apache-2.0
"""Muse megakernel constants: lanes, fp4 packing, the smalls slab rows."""

LANE = 128
NEG = -1e30
LOG2E = 1.4426950408889634
# largest score bound MKConfig.attn_unshifted accepts
UNSHIFTED_MAX = 60.0
# Barrier semaphore ids; distinct from the fused-RS MoE (0) and two-shot all-reduce (7) kernels.
_COLLECTIVE_ID = 6
# The prefill program's attention-output (9) and post-MoE (10) kernels.
PREFILL_ATTN_OUT_COLLECTIVE_ID = 9
PREFILL_POST_MOE_COLLECTIVE_ID = 10
# The SP prefill program's MoE dispatch (11) and combine (12) all-to-alls.
SP_DISPATCH_COLLECTIVE_ID = 11
SP_COMBINE_COLLECTIVE_ID = 12
FP4_PACK = 8
FP4_GROUP = 16
E2M1_MAX = 6.0  # fp4 e2m1's largest magnitude
QBLOCK_SIZES = (64, 128, 256)  # SGLANG_TPU_MUSE_MK_QBLOCK: K rows per scale row


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
)
SMALL_ROWS = 16


def small_row(smr, name: str, n: int | None = None):
    """Row SMALL[name] of a smalls slab, as a [1, cols] slice (its first n columns when given)."""
    i = SMALL[name]
    return smr[i : i + 1] if n is None else smr[i : i + 1, :n]
