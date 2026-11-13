# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's unified pools at TP32 (the layout vLLM allocates with
``gdn_pool_layout`` and the Raiden connector moves).

24 pools, one per MLA layer (every fourth layer, and the last); pool ``p``
also holds KDA layers ``3p .. 3p + 2`` (the last pool none). A block is
``[128, 2, 640]`` bf16 (2560-byte rows; FP8 caches use :data:`FP8_LAYOUT`):

* an MLA page holds 256 tokens' latents, token ``t`` at ``[t // 2, t % 2]``
  (FP8: 512 tokens at ``[t // 4, t % 4]``), 576 lanes used (the 512-lane
  normalized latent, then the 64-lane rope part);
* a KDA state block holds a layer's state for this rank's 3 heads: rows 0-76
  the recurrent state (f32, ``[3, 128, 128]``), rows 77-80 the conv taps
  (bf16, ``[3, 3, 3, 128]``), heads in TP-rank order; rows 81-95 carry a
  fifteenth of the request's DSpark draft ring (``kernels.dspark.pack_ring``:
  the 69 blocks hold this rank's KV head's ring and its header), written with
  the state by a prefill that computes the ring.
"""

import dataclasses

from sglang_tpu.k3_programs.layout import KVLayout, StateRegion

NUM_LAYERS = 93
MLA = tuple(i for i in range(NUM_LAYERS) if i % 4 == 3 or i == NUM_LAYERS - 1)
KDA = tuple(i for i in range(NUM_LAYERS) if i not in MLA)
KDA_HEADS = 3  # per rank at TP32
HEAD_DIM = 128
CONV_TAPS = 3
# The decoder's conv-history slots keep each (tap, q/k/v) group's KDA_HEADS
# heads in four rows: with a second-minor dimension of three, the slot bank's
# default device layout puts its layer axis second-minor, and XLA relayouts
# the whole bank into and out of the decode kernels every step.
CONV_HEAD_ROWS = 4

LAYOUT = KVLayout(
    block_shape=(128, 2, 640),
    dtype="bfloat16",
    tokens_per_block=256,
    tokens_per_row=2,
    entry_lanes=576,
    attention_layers=MLA,
    state_layers=KDA,
    state_regions=(
        StateRegion("recurrent", 0, 77, "float32"),
        StateRegion("conv", 77, 4, "bfloat16"),
        StateRegion("draft", 81, 15, "bfloat16"),
    ),
)


FP8_LAYOUT = dataclasses.replace(
    LAYOUT,
    block_shape=(128, 4, 640),
    dtype="float8_e4m3fn",
    tokens_per_block=512,
    tokens_per_row=4,
)


def cache_layout(dtype: str = "auto") -> KVLayout:
    """The physical contract selected by vLLM's cache dtype.

    FP8 doubles tokens per page, preserving the 2560-byte physical row and
    every FP32/BF16 state region. Cache latents use E4M3FN with scale one.
    """
    dtype = str(dtype).removeprefix("torch.")
    if dtype in ("auto", "bfloat16"):
        return LAYOUT
    if dtype in ("fp8", "fp8_e4m3", "float8_e4m3fn"):
        return FP8_LAYOUT
    raise NotImplementedError(
        f"K3 cache dtype {dtype!r} is unsupported; use auto/bfloat16 or fp8/fp8_e4m3"
    )


# Kimi-K3.1 DSA index keys beside the latent pages (kimi_k3/dsa.py): per
# 128-token sub-page, 32 rows of four tokens' 128 fp8 key bytes and one row
# whose sublane 0 holds the 128 tokens' UE8M0 scale bytes (the compact layout
# of sglang_tpu/k3_megakernel/kernel/stages/dsa.py). A pool row holds the
# index keys of the latent row with the same block id.
INDEX_SUBPAGE_TOKENS = 128
INDEX_SUBPAGE_ROWS = 33


def index_block_shape(layout: KVLayout) -> tuple[int, int, int]:
    """One index-key pool row (uint8) for ``layout``'s latent rows."""
    subpages = layout.tokens_per_block // INDEX_SUBPAGE_TOKENS
    return (subpages * INDEX_SUBPAGE_ROWS, 4, 128)
