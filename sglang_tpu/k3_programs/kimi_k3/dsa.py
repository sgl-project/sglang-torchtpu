# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's DSA ("lightning") indexer over decode context parallelism.

Kimi-K3.1 is Kimi-K3 with a DSA indexer on each of its 24 MLA layers: a query
attends only the ``index_topk`` (2048) earlier tokens with the highest index
scores

    score(t) = sum_h relu(q8_h . k8_t) * w_h * 2^e_t

with ``q8``/``k8`` fp8 (E4M3, UE8M0 power-of-two scales), 32 heads x 128 dims,
the key LayerNorm-ed and the head weights ``weights_proj(x) * 32^-0.5 * 128^-0.5``
times each query head's scale (sglang's native indexer; the in-kernel reference
is ``sglang_tpu/k3_megakernel/kernel/stages/dsa.py``). Ties at the threshold
keep the lowest positions.

Under DCP each rank holds the index keys of its own 512-token chunks (beside
their latent rows), scores only those, and the ranks agree on the exact global
threshold by a radix descent over the scores' monotone int32 keys: each round
every rank counts its keys per next digit under the prefix found so far, the
counts are summed over the group (one small all-reduce), and the digit where
the running count from the top reaches ``topk`` extends the prefix. A tie at
the threshold is broken by a second descent over the tied keys' positions.
Each rank then attends only its selected tokens.

This module is the host reference (numpy) of that selection; the kernels
implement the same rounds.
"""

from __future__ import annotations

import numpy as np

HEADS = 32
HEAD_DIM = 128
TOPK = 2048
FP8_MAX = 448.0
_MONO = 0x7FFFFFFF


# ------------------------------------------------------------------ quantization
def ue8m0_exponent(abs_max: np.ndarray) -> np.ndarray:
    """The UE8M0 scale exponent of an fp32 abs max: the smallest power of two
    with ``abs_max / 2^e <= 448`` (``quantize_fp8_ue8m0``)."""
    bits = np.asarray(abs_max, np.float32).view(np.int32)
    biased = bits >> 23
    exp = biased - (127 + 8) + ((bits & 0x7FFFFF) > 0x600000).astype(np.int32)
    exp = np.where(biased == 0, np.where(bits == 0, 0, -126), exp)
    return np.clip(exp, -126, 127).astype(np.int32)


def quantize_fp8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x`` f32 [..., D] -> (fp8-valued f32 [..., D], exponent i32 [..., 1])."""
    import ml_dtypes

    x = np.asarray(x, np.float32)
    exp = ue8m0_exponent(np.max(np.abs(x), axis=-1, keepdims=True))
    scaled = np.clip(x * np.exp2(-exp.astype(np.float32)), -FP8_MAX, FP8_MAX)
    return scaled.astype(ml_dtypes.float8_e4m3fn).astype(np.float32), exp


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, np.float32)
    mean = x.mean(-1, keepdims=True)
    var = (x * x).mean(-1, keepdims=True) - mean * mean
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def index_query(q: np.ndarray, gate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``q`` f32 [32 * 128] (the wq_b output of one row), ``gate`` f32 [32]
    (weights_proj output): the fp8 query heads [32, 128] and the head weights
    [32] (gate * 32^-0.5 * 128^-0.5 * 2^e_h)."""
    q8, exp = quantize_fp8(np.asarray(q, np.float32).reshape(HEADS, HEAD_DIM))
    w = gate * HEADS**-0.5 * HEAD_DIM**-0.5 * np.exp2(exp[:, 0].astype(np.float32))
    return q8, w.astype(np.float32)


def scores(q8: np.ndarray, w: np.ndarray, k8: np.ndarray, k_exp: np.ndarray) -> np.ndarray:
    """Index scores of keys ``k8`` f32 [N, 128] (exponents [N, 1])."""
    logits = np.maximum(k8 @ q8.T, 0.0)  # [N, 32]
    return (logits @ w) * np.exp2(k_exp[:, 0].astype(np.float32))


# ------------------------------------------------------------------ selection
def monotone_key(s: np.ndarray) -> np.ndarray:
    """f32 scores -> int32 keys of the same order."""
    bits = np.asarray(s, np.float32).view(np.int32)
    return bits ^ ((bits >> 31) & _MONO)


def topk_mask(keys: np.ndarray, positions: np.ndarray, topk: int = TOPK) -> np.ndarray:
    """Exact selection: the ``topk`` largest keys, ties keeping the lowest
    positions; everything when there are at most ``topk`` keys."""
    keys = np.asarray(keys, np.int64)
    if len(keys) <= topk:
        return np.ones(len(keys), bool)
    order = np.lexsort((positions, -keys))
    mask = np.zeros(len(keys), bool)
    mask[order[:topk]] = True
    return mask


def _unsigned(keys: np.ndarray) -> np.ndarray:
    """Signed int32 keys -> uint32 of the same order (for digit extraction)."""
    return (np.asarray(keys, np.int64) + (1 << 31)).astype(np.uint64)


def distributed_topk(
    rank_keys: list[np.ndarray],
    rank_positions: list[np.ndarray],
    topk: int = TOPK,
    digit_bits: int = 8,
    position_bits: int = 21,
) -> tuple[list[np.ndarray], int]:
    """The exact global selection over ranks that each hold some keys:
    ``(per-rank masks, all-reduces used)``. Each round sums one histogram of
    ``2^digit_bits`` counts over the ranks (simulated here by a sum)."""
    if sum(len(k) for k in rank_keys) <= topk:
        return [np.ones(len(k), bool) for k in rank_keys], 0
    ukeys = [_unsigned(k) for k in rank_keys]
    prefix, need, rounds = 0, topk, 0  # need: still to select at and below the prefix
    for shift, span in digits(32, digit_bits):
        hi_mask = (~((1 << (shift + span)) - 1)) & 0xFFFFFFFF
        hist = np.zeros(1 << span, np.int64)
        for k in ukeys:  # per rank, then summed: the all-reduce
            inside = (k & np.uint64(hi_mask)) == np.uint64(prefix)
            np.add.at(hist, ((k[inside] >> np.uint64(shift)) & np.uint64((1 << span) - 1)).astype(np.int64), 1)
        rounds += 1
        running = 0
        for digit in range((1 << span) - 1, -1, -1):
            if running + hist[digit] >= need:
                prefix |= digit << shift
                need -= running
                break
            running += hist[digit]
    tau = np.uint64(prefix)  # the threshold key; ``need`` of the keys == tau are selected
    tied = sum(int(np.sum(k == tau)) for k in ukeys)
    if tied == need:
        return [k >= tau for k in ukeys], rounds
    # A tie: keep the ``need`` tied keys with the lowest positions -- a second
    # descent, over the tied keys' position digits from the low end.
    pos_prefix, pos_need = 0, need
    for shift, span in digits(position_bits, digit_bits):
        hi_mask = ~((1 << (shift + span)) - 1)
        hist = np.zeros(1 << span, np.int64)
        for k, p in zip(ukeys, rank_positions):
            p = np.asarray(p, np.int64)
            inside = (k == tau) & ((p & hi_mask) == pos_prefix)
            np.add.at(hist, (p[inside] >> shift) & ((1 << span) - 1), 1)
        rounds += 1
        running = 0
        for digit in range(1 << span):
            if running + hist[digit] >= pos_need:
                pos_prefix |= digit << shift
                pos_need -= running
                break
            running += hist[digit]
    return [
        (k > tau) | ((k == tau) & (np.asarray(p, np.int64) <= pos_prefix))
        for k, p in zip(ukeys, rank_positions)
    ], rounds


def digits(bits: int, digit_bits: int) -> list[tuple[int, int]]:
    """``(shift, width)`` of each digit of a ``bits``-bit value, most significant first."""
    out, top = [], bits
    while top > 0:
        width = min(digit_bits, top)
        out.append((top - width, width))
        top -= width
    return out
