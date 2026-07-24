"""Tuned block sizes for the gmm_fused_rs (ICI direct-write) kernel ONLY.

This table is dedicated to ``gmm_fused_rs_nodedup.py`` and is intentionally
kept separate from the psum-based ``gmm_v2_fused`` kernel's tuning data.
Reasons:

1. The fused_rs kernel has a different optimization profile from gmm_v2_fused:
   - It runs an embedded ICI reduce-scatter, has more concurrent DMA streams
     (input gather + W1 + W2 + remote scatter + local writes), and uses
     same-expert weight caching when ``num_w*_bufs >= total_w*_steps``.
   - These differences mean the optimal (tile_m, tile_k1, tile_n1, tile_k2,
     tile_n2, num_w1_bufs, num_w2_bufs) tuple is generally NOT the same as
     for gmm_v2_fused, even at identical dimensions.

2. Auto-tuning for gmm_v2_fused (which writes to the shared table) can update
   entries in ways that regress fused_rs performance. By isolating the
   fused_rs tuning into its own file, we guarantee optimal block sizes are
   preserved regardless of what the shared table evolves to.

The dispatcher in ``tuned_block_sizes.py::get_fused_rs_tuned_block_sizes``
looks up this table FIRST, then falls back to the shared
``FUSED_TUNED_BLOCK_SIZES`` (so bf16 / other configs continue to work),
then to the caller-supplied defaults.

Entry format (tuple of 7 ints):
    (tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs)

Same-expert weight caching is automatically enabled when:
    num_w1_bufs >= total_w1_steps  AND  num_w2_bufs >= total_w2_steps
where total_w*_steps = (size_n* / tile_n*) * (size_k* / tile_k*).
"""

# =============================================================================
# Per-config tile selection rationale
# =============================================================================
#
# H4096-I4096 fp8 (H=4096, I=4096, E=256, top_k=8, EP=8 → g=32):
#   Dimensions: K1=4096, N1=8192 (gate+up), K2=4096, N2=4096
#   Chosen: (112, 4096, 2048, 4096, 2048, 2, 2)
#   - tile_k1 = K1 = 4096  → 1 K-step (no inner-K loop for W1)
#   - tile_n1 = 2048       → 2 N-steps for W1 (after silu split: N1/2=4096)
#   - tile_k2 = K2 = 4096  → 1 K-step for W2
#   - tile_n2 = 2048       → 2 N-steps for W2
#   - num_w1_bufs = 2      → covers all 2 W1 tiles → caching enabled
#   - num_w2_bufs = 2      → covers all 2 W2 tiles → caching enabled
#   VMEM (fp8 packed as uint32):
#     W1 buf: 2 × (4096/4) × (2*2048) × 4B = 32 MB
#     W2 buf: 2 × (4096/4) × 2048 × 4B    = 16 MB
#     Other: ~10 MB. Spill: ~6 MB. Total: ~64 MB (= VMEM limit).
#
# H3072-I3072 fp8 (H=3072, I=3072, E=256, top_k=8, EP=8 → g=32):
#   Dimensions: K1=3072, N1=6144 (gate+up), K2=3072, N2=3072 (aligned_n2=4096)
#   Chosen: (112, 3072, 1024, 3072, 1024, 3, 3)
#   - tile_k1 = K1 = 3072  → 1 K-step
#   - tile_n1 = 1024       → 3 N-steps (N1/2=3072, /1024 = 3 tiles)
#   - tile_k2 = K2 = 3072  → 1 K-step
#   - tile_n2 = 1024       → 3 N-steps (N2/1024 = 3)
#   - num_w1_bufs = 3      → covers all 3 W1 tiles → caching enabled
#   - num_w2_bufs = 3      → covers all 3 W2 tiles → caching enabled
#   VMEM (fp8 packed as uint32):
#     W1 buf: 3 × (3072/4) × (2*1024) × 4B = 18 MB
#     W2 buf: 3 × (3072/4) × 1024 × 4B    =  9 MB
#     Other: ~10 MB. Spill: ~6 MB. Total: ~43 MB (well under VMEM limit).
#
# Same tile config works for ALL M sizes (256-131072) because per-tile work
# is independent of M; M only changes the gm-loop iteration count.

_H4096_I4096_FP8 = (112, 4096, 4096, 4096, 4096, 1, 1)
# fp8 DECODE (M<=2048, HBM-bandwidth-bound): tile_m=32 + half-N + double-buffered
# weights. ~10% faster than the per-tensor default (112, full-N, 1 buf) across
# decode batch sizes (16-256 seqs). Used for both bf16 and fp8 output comm.
_H4096_I4096_FP8_DECODE = (32, 4096, 2048, 4096, 2048, 2, 2)
# fp8 direct-write prefill:
# fp8 direct-write adds VMEM staging tiles, so the bf16 tile_m=112 OOMs.
# tile_m=96 full-N is the largest that fits and is ~7% faster than the old
# unconditional min(tile_m, 64) clamp.
_H4096_I4096_FP8_FP8COMM_PREFILL = (96, 4096, 4096, 4096, 4096, 1, 1)
# H3072-I3072 fp8: TWO configs needed.
# - At small M (< 1024 tokens, < ~1 m-tile per expert): caching gives no
#   benefit since each expert is visited once. The smaller per-tile work
#   (tile_n=1024 → 3 N-steps) hurts due to per-step kernel overhead.
#   Use full-N tiles + single buffer to minimize iteration count.
# - At large M (>= 1024 tokens, multiple m-tiles per expert): caching
#   eliminates 90%+ of weight DMAs by skipping same-expert reloads.
#   Use 3-way buffered to enable caching.
_H3072_I3072_FP8_SMALL = (
    128,
    3072,
    3072,
    3072,
    3072,
    1,
    1,
)  # M < ~8192 (no tiling for small M)
_H3072_I3072_FP8_LARGE = (
    128,
    3072,
    3072,
    3072,
    1536,
    2,
    2,
)  # 8192 <= M < 131072 (N2-tiled, w1_bufs=2, w2_bufs=2) — autotuned 46.3ms
_H3072_I3072_FP8_XLARGE = (
    256,
    3072,
    3072,
    3072,
    1536,
    1,
    2,
)  # M >= 131072 (tile_m=256 wins at very large M) — autotuned 48.6ms
_H4096_I4096_BF16 = (128, 2048, 2048, 2048, 2048, 2, 2)
_H3072_I3072_BF16 = (128, 1024, 1024, 1024, 1024, 2, 2)

# FP4 (rhs_quant_block_size=16): weights are half FP8 → enough VMEM for
# num_w1_bufs=2, enabling full W1 caching that FP8 couldn't afford.
#
# H4096-I4096 fp4 (H=4096, I=4096, E=256, EP=8 → g=32):
#   Dimensions: K1=4096, N1=8192, K2=4096, N2=4096
#   Chosen: (128, 4096, 4096, 4096, 4096, 1, 1).
#   num_w1_bufs=2 was attempted but OOMs VMEM by ~6.6 MB due to FP4's
#   dense scale overhead (block_size=16 → (K/16)×N×4B scale per buffer,
#   which at 4096 dims adds ~4 MB/buf). The "Other" VMEM (LHS double-buf,
#   accumulators, ICI scatter bufs, spill) totals ~20-22 MB, leaving only
#   ~42 MB for weights+scales — not enough for 2 W1 + 1 W2 (~36 MB) with
#   the actual allocator overhead.
#   - tile_k1 = K1 = 4096    → 1 K-step
#   - tile_n1 = out_n1 = 4096 → 2 N-steps (aligned_n1=8192, /4096=2)
#   - tile_k2 = K2 = 4096    → 1 K-step
#   - tile_n2 = N2 = 4096    → 1 N-step
#   - num_w1_bufs = 1  → 1 < w1_steps=2 → no W1 caching
#   - num_w2_bufs = 1  → 1 ≥ w2_steps=1 → W2 caching enabled
#
# H3072-I3072 fp4 (H=3072, I=3072, E=256, EP=8 → g=32):
#   Dimensions: K1=3072, N1=6144, K2=3072, N2=3072
#   Chosen: (128, 3072, 3072, 3072, 3072, 2, 1)
#   - Full K, full out_N tiles. Same layout as FP8 small-M but with
#     num_w1_bufs=2 for caching (FP4's half-size weights make this fit).
#   VMEM:
#     W1 weight: 2 × 3072 × 3072 × 0.5B        =  9 MB
#     W1 scale:  2 × (3072/16) × 3072 × 4B      = 4.5 MB
#     W2 weight: 1 × 3072 × 3072 × 0.5B         = 4.5 MB
#     W2 scale:  1 × (3072/16) × 3072 × 4B       = 2.25 MB
#     Other + spill: ~16 MB. Total: ~36 MB (lots of headroom).
#   No size-dependent split needed — caching is essentially free with FP4.
_H4096_I4096_FP4_SMALL = (128, 4096, 4096, 4096, 4096, 1, 1)
_H4096_I4096_FP4_LARGE = (64, 4096, 2048, 4096, 2048, 2, 2)
_H4096_I4096_FP4_Q512_CACHED = (128, 4096, 4096, 4096, 4096, 2, 1)
_H4096_I4096_FP4_Q512_W4A8_SMALL = (64, 4096, 2048, 4096, 2048, 2, 2)
_H4096_I4096_FP8_Q512_SMALL = (64, 4096, 2048, 4096, 2048, 2, 2)
_H4096_I4096_FP8_Q512_FULL_112 = (112, 4096, 4096, 4096, 4096, 1, 1)
_H4096_I4096_FP8_Q512_FULL_128 = (128, 4096, 4096, 4096, 4096, 1, 1)
_H3072_I3072_FP4_SMALL = (128, 3072, 3072, 3072, 3072, 2, 1)
_H3072_I3072_FP4_LARGE = (64, 3072, 3072, 3072, 2048, 2, 2)
# H=I=2304, E=256, top_k=8 fp4 rhs_bs=512 (W4A8) at EP=2 -> g=128.
# H=2304 is not divisible by 512, so the weights use a ragged last scale block
# (4*512 + 256 -> ceil(2304/512)=5 blocks); the kernel recovers bs=512 from the
# block count and indexes the partial block with min(). Dims: k1=k2=2304,
# n1=4608 (gate+up, tile_n1 lives in this pre-silu space), aligned_n2=4096.
# Tuned on TPU7x via sweep_fused_rs_fp4 (real benchmark path; the shared
# autotune harness is FP8-only). Three zones:
#   SMALL (M<=4096): tn1=1536 (N1 in thirds, num_w1_bufs=3 caches W1) + full-N2.
#     +10-20% vs heuristic default.
#   MID (M==8192): full-N1 tm64 (matches the heuristic; this point is flat,
#     ~+0.1%) — avoids the SMALL config's ~1.6% regression here.
#   LARGE (M>=16384): tm128, tn1=2304 (halves) + full-N2, triple/double
#     buffered. +6-16%; robust (within +0.3% of best at M=16384/32768/65536).
_H2304_I2304_FP4_Q512_SMALL = (64, 2304, 1536, 2304, 4096, 3, 1)
_H2304_I2304_FP4_Q512_MID = (64, 2304, 4608, 2304, 4096, 1, 2)
_H2304_I2304_FP4_Q512_LARGE = (128, 2304, 2304, 2304, 4096, 3, 2)


# Apply the same config across all benchmarked M values for prefill_X with
# top_k=8: M = X * top_k. Covers prefill 32 → 16384.
_ALL_M_KEYS = [
    "128_silu",
    "256_silu",
    "512_silu",
    "1024_silu",
    "2048_silu",
    "4096_silu",
    "8192_silu",
    "16384_silu",
    "32768_silu",
    "65536_silu",
    "131072_silu",
]


def _all_m(tile):
    return {m_key: tile for m_key in _ALL_M_KEYS}


def _h4096_i4096_fp4_size_dependent():
    """H4096-I4096 fp4: full tiles for small M, smaller tiles for large M.

    Crossover at M=32768 (= prefill_4096 with top_k=8). At large M the
    full-N tiles (tile_n=4096) create too-large per-tile work that doesn't
    pipeline well across many m-tiles. Smaller tiles (tile_n=2048) with
    num_bufs=2 enable W2 caching + better DMA/compute overlap.
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        if m_val >= 32768:
            out[m_key] = _H4096_I4096_FP4_LARGE
        else:
            out[m_key] = _H4096_I4096_FP4_SMALL
    return out


def _h4096_i4096_fp4_q512_size_dependent():
    """H4096-I4096 fp4 rhs_bs=512.

    rhs_bs=512 matches the fused-rs LHS online-quant block, so the kernel uses
    W4A8. Half-N cached tm64 wins through M=8192; at M>=16384 the full-N
    W1-cached tile wins.
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        if m_val <= 8192:
            out[m_key] = _H4096_I4096_FP4_Q512_W4A8_SMALL
        else:
            out[m_key] = _H4096_I4096_FP4_Q512_CACHED
    return out


def _h4096_i4096_fp8_q512_size_dependent():
    """H4096-I4096 fp8 with rhs_bs=512."""
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        if m_val <= 8192:
            out[m_key] = _H4096_I4096_FP8_Q512_SMALL
        elif m_val == 32768:
            out[m_key] = _H4096_I4096_FP8_Q512_FULL_128
        else:
            out[m_key] = _H4096_I4096_FP8_Q512_FULL_112
    return out


def _h3072_i3072_fp4_size_dependent():
    """H3072-I3072 fp4: full tiles + W1 caching for small M, smaller N2 for large M.

    Crossover at M=32768 (= prefill_4096 with top_k=8). The small-M config
    (tile_n1=3072, tile_n2=3072, num_w1_bufs=2) enables full W1 caching —
    a clear win over default (8-10% faster). For large M, smaller tile_n2
    and tile_m=64 pipeline better across many m-tiles.
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        if m_val >= 32768:
            out[m_key] = _H3072_I3072_FP4_LARGE
        else:
            out[m_key] = _H3072_I3072_FP4_SMALL
    return out


def _h2304_i2304_fp4_q512_size_dependent():
    """H=I=2304 fp4 rhs_bs=512 (W4A8, ragged last block) at EP=2 (g=128).

    Three zones (M = num_tokens * top_k):
      M<=4096   -> SMALL: tn1=1536 (N1 thirds, W1 cached) + full-N2 at tm64.
      M==8192   -> MID:   full-N1 tm64 (flat region; ~heuristic default).
      M>=16384  -> LARGE: tm128, tn1=2304 (N1 halves) + full-N2, buffered.
    Tuned via sweep_fused_rs_fp4 on TPU7x. Crossovers at M=8192 and M=16384.
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        if m_val <= 4096:
            out[m_key] = _H2304_I2304_FP4_Q512_SMALL
        elif m_val <= 8192:
            out[m_key] = _H2304_I2304_FP4_Q512_MID
        else:
            out[m_key] = _H2304_I2304_FP4_Q512_LARGE
    return out


def _h3072_i3072_fp8_size_dependent():
    """H3072-I3072 fp8: full-N tiles for small M, cached config for large M.

    Crossover at M=8192 (= prefill_1024 with top_k=8) where caching's
    DMA savings start outweighing per-tile-step overhead.
    At M>=131072, tile_m=256 wins because more compute per tile hides scatter.
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        if m_val >= 131072:
            out[m_key] = _H3072_I3072_FP8_XLARGE
        elif m_val >= 8192:
            out[m_key] = _H3072_I3072_FP8_LARGE
        else:
            out[m_key] = _H3072_I3072_FP8_SMALL
    return out


def _h4096_i4096_fp8_perchannel_size_dependent():
    """H4096-I4096 per-tensor/per-channel fp8 (rhs_bs=K), bf16 output comm.

    Decode (M<=2048): tile_m=32 half-N double-buffered (bandwidth-bound, ~10%
    faster than the old single config). Prefill (M>=4096): full-N tile_m=112
    (compute-bound). Crossover at M=2048 (= decode 256 seqs / prefill 256 tok).
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        out[m_key] = _H4096_I4096_FP8_DECODE if m_val <= 2048 else _H4096_I4096_FP8
    return out


def _h4096_i4096_fp8_fp8comm_size_dependent():
    """H4096-I4096 fp8 WITH fp8 output comm (fp8_direct_write).

    Decode (M<=2048): tile_m=32 (well under the fp8-staging VMEM ceiling).
    Prefill (M>=4096): tile_m=96 full-N — the largest that fits with the fp8
    staging tiles (tile_m=112 OOMs), ~7% faster than the old tile_m<=64 clamp.
    """
    out = {}
    for m_key in _ALL_M_KEYS:
        m_val = int(m_key.split("_")[0])
        out[m_key] = (
            _H4096_I4096_FP8_DECODE
            if m_val <= 2048
            else _H4096_I4096_FP8_FP8COMM_PREFILL
        )
    return out


# =============================================================================
# Table — keyed by (device_name → dtype_key → joint_model_key → m_key)
# =============================================================================

FUSED_RS_TUNED_BLOCK_SIZES = {
    "TPU7x": {
        # ---------------------------------------------------------------------
        # FP8 (lhs bf16 × rhs fp8)
        # ---------------------------------------------------------------------
        "lhs_bfloat16_rhs_float8_e4m3fn": {
            # H4096-I4096: H=4096, I=4096, E=256, EP=8 → g=32
            # Validated: caching + double-buffered weights.
            # Beats gmm_ep_dma by 11-22% across all M.
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-256"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-256"
            ): _all_m(_H4096_I4096_FP8),
            # H4096-I4096 subchannel FP8 (block_size=512), EP=8 → g=32.
            # Tuned on TPU7x through the end-to-end serving path.
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-512"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-512"
            ): _h4096_i4096_fp8_q512_size_dependent(),
            # H4096-I4096 subchannel FP8 (block_size=512), EP=16 → g=16.
            # Reuse the EP=8 (g=32) q512 table — same tiles.
            (
                "k-4096_n-8192_g-16_lhs_quant_rhs_bs-512"
                "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-512"
            ): _h4096_i4096_fp8_q512_size_dependent(),
            # H4096-I4096 multi-host: H=4096, I=4096, E=256, EP=16 → g=16
            # Per-tensor FP8 (online quant, bs=K). Same tile as g=32.
            (
                "k-4096_n-8192_g-16_lhs_quant_rhs_bs-4096"
                "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-4096"
            ): _h4096_i4096_fp8_perchannel_size_dependent(),
            # H4096-I4096 single-host: per-tensor FP8, EP=8 → g=32
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-4096"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-4096"
            ): _h4096_i4096_fp8_perchannel_size_dependent(),
            # H3072-I3072: H=3072, I=3072, E=256, EP=8 → g=32
            # Full-N tiles (no N-tiling), no caching. tile_m=256 for large M
            # to minimize gm-loop iterations. Matches shared table values.
            (
                "k-3072_n-6144_g-32_lhs_quant_rhs_bs-256"
                "+k-3072_n-4096_g-32_lhs_quant_rhs_bs-256"
            ): {
                "128_silu": (128, 3072, 3072, 3072, 3072, 1, 1),
                "256_silu": (128, 3072, 3072, 3072, 3072, 1, 1),
                "512_silu": (128, 3072, 3072, 3072, 3072, 1, 1),
                "1024_silu": (128, 3072, 3072, 3072, 3072, 1, 1),
                "2048_silu": (128, 3072, 3072, 3072, 3072, 1, 1),
                "4096_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
                "8192_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
                "16384_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
                "32768_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
                "65536_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
                "131072_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
                "262144_silu": (256, 3072, 3072, 3072, 3072, 1, 1),
            },
            # H3072-I3072 per-channel FP8 (block_size=3072, aligned_n2=4096)
            (
                "k-3072_n-6144_g-32_lhs_quant_rhs_bs-3072"
                "+k-3072_n-4096_g-32_lhs_quant_rhs_bs-3072"
            ): _h3072_i3072_fp8_size_dependent(),
        },
        # ---------------------------------------------------------------------
        # FP4 (lhs bf16 × rhs fp4, rhs_quant_block_size=16)
        # ---------------------------------------------------------------------
        # FP4 weights are half the size of FP8, enabling full W1 caching
        # (num_w1_bufs=2) that FP8 couldn't afford. No size-dependent
        # split needed — the caching overhead is negligible at FP4 sizes.
        "lhs_bfloat16_rhs_float4_e2m1fn": {
            # H4096-I4096: H=4096, I=4096, E=256, EP=8 → g=32
            # Small M: full tiles (1,1). Large M (≥32768): smaller tiles
            # with W2 caching (2,2) for better pipelining.
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-16"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-16"
            ): _h4096_i4096_fp4_size_dependent(),
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-256"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-256"
            ): _h4096_i4096_fp4_q512_size_dependent(),
            # H4096-I4096 multi-host: H=4096, I=4096, E=256, EP=16 → g=16.
            # Reuse the EP=8 (g=32) rhs_bs=256 table — same tiles.
            (
                "k-4096_n-8192_g-16_lhs_quant_rhs_bs-256"
                "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-256"
            ): _h4096_i4096_fp4_q512_size_dependent(),
            # H4096-I4096 subchannel FP4 (block_size=512), EP=8 → g=32.
            # rhs_bs=512 matches the LHS online-quant block, so this uses the
            # W4A8-tuned table.
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-512"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-512"
            ): _h4096_i4096_fp4_q512_size_dependent(),
            # H4096-I4096 subchannel FP4 (block_size=512), EP=16 → g=16.
            # Reuse the EP=8 (g=32) q512 W4A8-tuned table — same tiles.
            (
                "k-4096_n-8192_g-16_lhs_quant_rhs_bs-512"
                "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-512"
            ): _h4096_i4096_fp4_q512_size_dependent(),
            # H3072-I3072: H=3072, I=3072, E=256, EP=8 → g=32
            # Small M: full tiles + W1 caching (2,1). Large M (≥32768):
            # smaller tile_n2/tile_m for better pipelining.
            (
                "k-3072_n-6144_g-32_lhs_quant_rhs_bs-16"
                "+k-3072_n-4096_g-32_lhs_quant_rhs_bs-16"
            ): _h3072_i3072_fp4_size_dependent(),
            # H3072-I3072 subchannel FP4 (block_size=512), EP=8 → g=32.
            # Reuse the H3072-I3072 fp4 tuned tiles (tiles are matmul-shape driven, not
            # scale-block driven) so qbs-512 uses tuned tiles instead of the
            # default fallback. (No separate q512 sweep for H3072-I3072 yet.)
            (
                "k-3072_n-6144_g-32_lhs_quant_rhs_bs-512"
                "+k-3072_n-4096_g-32_lhs_quant_rhs_bs-512"
            ): _h3072_i3072_fp4_size_dependent(),
            # H=I=2304, E=256, top_k=8 subchannel FP4 block_size=512,
            # EP=2 -> g=128. K=2304 is not a multiple of 512, so the scale uses a
            # ragged last block (ceil(2304/512)=5 blocks); the kernel recovers
            # bs=512 from the block count -> the key carries rhs_bs-512. W4A8
            # (rhs_bs 512 >= lhs online-quant block 512). Tuned on TPU7x via the
            # End-to-end serving path with an FP4 fused-RS sweep.
            (
                "k-2304_n-4608_g-128_lhs_quant_rhs_bs-512"
                "+k-2304_n-4096_g-128_lhs_quant_rhs_bs-512"
            ): _h2304_i2304_fp4_q512_size_dependent(),
        },
        # ---------------------------------------------------------------------
        # BF16 (lhs bf16 × rhs bf16)
        # ---------------------------------------------------------------------
        "lhs_bfloat16_rhs_bfloat16": {
            # H4096-I4096: H=4096, I=4096, E=256, EP=8 → g=32
            # Validated: tile_n2=2048 with num_w2_bufs=2 to eliminate W2 stall.
            # Caching disabled (num_bufs(2) < total_steps(4)).
            "k-4096_n-8192_g-32+k-4096_n-4096_g-32": _all_m(_H4096_I4096_BF16),
            # H3072-I3072: H=3072, I=3072, E=256, EP=8 → g=32
            # Smaller tiles for VMEM fit (bf16 uses 2× weight bytes vs fp8).
            "k-3072_n-6144_g-32+k-3072_n-3072_g-32": _all_m(_H3072_I3072_BF16),
        },
    }
}


# =============================================================================
# fp8 OUTPUT-COMM table.
#
# Selected by get_fused_rs_tuned_block_sizes(fp8_direct_write=True). Kept
# separate from the table above because fp8 direct-write adds VMEM staging tiles (an FP8
# DMA tile + per-row scale staging), which changes the optimal tile -- notably a
# lower prefill tile_m. Entries here are sized to fit that VMEM budget, so the
# selector returns them directly (no tile_m clamp). Shapes WITHOUT an entry here
# fall back to the bf16 table with tile_m clamped to <=64 for VMEM safety.
# =============================================================================
FUSED_RS_TUNED_BLOCK_SIZES_FP8COMM = {
    "TPU7x": {
        "lhs_bfloat16_rhs_float8_e4m3fn": {
            # H4096-I4096 per-tensor fp8, EP=16 -> g=16 (multi-host, prod).
            (
                "k-4096_n-8192_g-16_lhs_quant_rhs_bs-4096"
                "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-4096"
            ): _h4096_i4096_fp8_fp8comm_size_dependent(),
            # H4096-I4096 per-tensor fp8, EP=8 -> g=32 (single-host).
            (
                "k-4096_n-8192_g-32_lhs_quant_rhs_bs-4096"
                "+k-4096_n-4096_g-32_lhs_quant_rhs_bs-4096"
            ): _h4096_i4096_fp8_fp8comm_size_dependent(),
        },
    },
}
