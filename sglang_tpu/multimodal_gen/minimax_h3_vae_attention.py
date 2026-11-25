# SPDX-License-Identifier: Apache-2.0
"""The MiniMax H3 video VAE's ViT attention as an fp16-faithful Pallas kernel.

Used by minimax_h3_decode (SGLANG_TPU_OPT_H3_VAE_FLASH_ATTENTION). The ViT
decoder runs full bidirectional attention over 1,797 tokens with 32 heads of
width 64 under fp16 autocast. torch_tpu has no flash path for fp16, so its
SDPA writes each [heads, tokens, tokens] score matrix through HBM, rounds the
scores to fp16, takes the softmax and rounds the probabilities to fp16.

Here every score stays in VMEM. XLA splits fp16 q, k and v exactly into bf16
high and low parts (an fp16 value has 11 significant bits, bf16 8: hi + lo is
exact) and rounds the float32 result to fp16 once; Mosaic has no fp16 vector
type on tpu7x, so the kernel only sees bf16 and float32. A grid cell takes one
unit, two heads (a 128-lane column block of the [B, S, heads * 64] layout) and
a block of queries against all keys:

- scores: q.k as qh.kh + qh.kl + ql.kh (the three bf16 MXU passes XLA runs for
  an fp16 dot; every product exact, float32 accumulation), packed with a
  shift lane into one 256-deep contraction a head, so the three products and
  the softmax shift cost one MXU pass (head_dim 64 alone fills a quarter of
  the MXU's depth). The scores are not rounded to fp16;
- softmax in float32 with a fixed shift instead of the row max: q and k are
  RMS-normalized per head without affine weights in the H3 decoder, so
  |q.k| / 8 <= 8 (Cauchy-Schwarz) and exp(s - 8) lies in [e^-16, 1] up to
  rounding; no overflow, underflow or online rescaling;
- the probabilities split into bf16 hi + lo (16 significant bits, finer than
  fp16's 11) and multiplied by [V hi | V lo | 1]: all four products exact,
  float32 accumulation, the row sums from the same pass;
- the output divided by the row sum; XLA rounds it to fp16 once.

So every product is exact or drops only the lo*lo term XLA's fp16 dot drops,
and the scores and the softmax stay float32 where the fp16 path rounds them
to fp16: the result is at least as accurate as the fp16 attention it
replaces (relative L2 2.08e-4 against float64 on the device precision
probe's inputs, the fp16 rounding of the exact result; the fp16 path:
6.63e-4 on TPU).
"""

# NOTE: no `from __future__ import annotations` here: torch_tpu's
# _infer_static_argnums identifies tensor arguments by `annotation is
# jax.Array`, which needs evaluated annotations on the registered function.

import functools
import math
import threading

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5
LOG2_E = math.log2(math.e)
NT_DIMS = (((1,), (1,)), ((), ()))
NN_DIMS = (((1,), (0,)), ((), ()))
# The softmax shift in score units: |q.k| * SCALE <= sqrt(HEAD_DIM) for q, k
# with per-head RMS at most 1. The kernel is exact for any shift above the
# row max by less than ~80 (exp(-80) is a normal float32).
SCORE_BOUND = math.sqrt(HEAD_DIM)
# Shift-lane value of padding keys: their scores become ~-2.4e38, exp 0.
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
# Query rows a block takes at most (VMEM holds a block's float32 scores and
# bf16 probabilities against every key).
MAX_QUERY_BLOCK = 640
# Keys the kernel takes (K/V parts, their input blocks and the score tiles of
# a query block grow with the key count).
MAX_KEYS = 2048
# VMEM the kernel may use (tpu7x has 64 MiB a TensorCore). At 1,797 tokens it
# needs 20-24 MiB; a 32 MiB limit adds VMEM-to-VMEM copy loops for the query
# blocks to every grid step (compile-only tpu7x), 48 MiB does not.
VMEM_LIMIT_BYTES = 48 << 20

_OP = None
_OP_LOCK = threading.Lock()


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def query_block(seq: int) -> int:
    """Query rows a grid cell takes: the fewest blocks of at most
    MAX_QUERY_BLOCK rows, each a multiple of 16 (bf16 sublane tiles),
    covering ``seq`` (1,797 tokens: 3 blocks of 608 rows)."""
    blocks = -(-seq // MAX_QUERY_BLOCK)
    return _round_up(-(-seq // blocks), 16)


def split_bf16(x):
    """fp16 (or float32 holding fp16 values) -> bf16 (hi, lo), hi + lo == x
    exactly (hi rounded to nearest; the remainder has at most 3 significant
    bits). An infinite x gives (inf, 0)."""
    hi = x.astype(jnp.bfloat16)
    lo = x.astype(jnp.float32) - hi.astype(jnp.float32)
    lo = jnp.where(jnp.isinf(x), 0.0, lo)
    return hi, lo.astype(jnp.bfloat16)


def _kernel(
    qh_ref, ql_ref, kh_ref, kl_ref, vh_ref, vl_ref, o_ref, kp_ref, vp_ref, *, seq
):
    """One (unit, head pair, query block) cell.

    q*_ref: [bq, 128] bf16 parts of the queries (lanes 0-63 head a, 64-127
    head b); k*_ref, v*_ref: [kp, 128] the keys' and values' parts, rows from
    ``seq`` on padding (undefined); o_ref: [bq, 128] float32. Scratch, filled
    at a head pair's first query block: kp_ref [2, kp, 256] bf16, a head's
    keys for the packed contraction; vp_ref [2, kp, 256] bf16, a head's
    [V hi | V lo | 1 0 ...] (head b: [V lo | V hi | 1 0 ...]).
    """
    kp = kh_ref.shape[0]
    half = HEAD_DIM
    # The shift lane adds -SCORE_BOUND / SCALE to q.k (exact in bf16: -64).
    shift = -SCORE_BOUND / SCALE
    to_exp2 = SCALE * LOG2_E

    @pl.when(pl.program_id(2) == 0)
    def _prepare():
        rows = lax.broadcasted_iota(jnp.int32, (kp, LANES), 0)
        lane = lax.broadcasted_iota(jnp.int32, (kp, LANES), 1)
        left = lane < half
        real = rows < seq

        def part(ref):
            # Padding rows may hold anything (NaN included): zero them.
            return jnp.where(real, ref[...].astype(jnp.float32), 0.0)

        kh, kl, vh, vl = part(kh_ref), part(kl_ref), part(vh_ref), part(vl_ref)
        kh_r = pltpu.roll(kh, half, 1)
        kl_r = pltpu.roll(kl, half, 1)
        vl_r = pltpu.roll(vl, half, 1)
        shift_lane = jnp.where(lane == half, jnp.where(real, shift, MASK_VALUE), 0.0)
        # Against the queries' [qh | qh] and [ql | 1 0 ...]: head a's
        # [kh_a | kl_a] and [kh_a | shift], head b's the same from its lanes.
        kp_ref[0] = jnp.concatenate(
            [jnp.where(left, kh, kl_r), jnp.where(left, kh, shift_lane)], axis=1
        ).astype(jnp.bfloat16)
        kp_ref[1] = jnp.concatenate(
            [jnp.where(left, kh_r, kl), jnp.where(left, kh_r, shift_lane)], axis=1
        ).astype(jnp.bfloat16)
        ones = jnp.where((lane == 0) & real, 1.0, 0.0)
        vp_ref[0] = jnp.concatenate([jnp.where(left, vh, vl_r), ones], axis=1).astype(
            jnp.bfloat16
        )
        vp_ref[1] = jnp.concatenate([jnp.where(left, vl_r, vh), ones], axis=1).astype(
            jnp.bfloat16
        )

    bq = qh_ref.shape[0]
    lane = lax.broadcasted_iota(jnp.int32, (bq, LANES), 1)
    left = lane < half
    qh = qh_ref[...].astype(jnp.float32)
    ql = ql_ref[...].astype(jnp.float32)
    qh_r = pltpu.roll(qh, half, 1)
    ql_r = pltpu.roll(ql, half, 1)
    one = jnp.where(lane == half, 1.0, 0.0)
    packed = (
        (jnp.where(left, qh, qh_r), jnp.where(left, ql, one)),
        (jnp.where(left, qh_r, qh), jnp.where(left, ql_r, one)),
    )

    def probabilities(head):
        qpack = jnp.concatenate(packed[head], axis=1).astype(jnp.bfloat16)
        u = lax.dot_general(
            qpack, kp_ref[head], NT_DIMS, preferred_element_type=jnp.float32
        )
        e = jnp.exp2(u * to_exp2)
        eh = e.astype(jnp.bfloat16)
        el = (e - eh.astype(jnp.float32)).astype(jnp.bfloat16)
        return jnp.concatenate([eh, el], axis=0)

    def output(head, p):
        pv = lax.dot_general(
            p, vp_ref[head], NN_DIMS, preferred_element_type=jnp.float32
        )
        pv = pv[:bq] + pv[bq:]
        row_sum = pv[:, LANES : LANES + 1]
        pv = pv[:, :LANES]
        # [P V hi | P V lo] (head b: [P V lo | P V hi]): the head's output is
        # the sum of the two lane halves.
        pv = pv + pltpu.roll(pv, half, 1)
        return pv / row_sum

    # Both heads' scores before either head's PV pass (a better schedule).
    p0 = probabilities(0)
    p1 = probabilities(1)
    out0 = output(0, p0)
    out1 = output(1, p1)
    o_ref[...] = jnp.where(left, out0, out1)


def flash_call(
    batch: int,
    seq: int,
    heads: int,
    *,
    rows=None,
    interpret=False,
    bq=None,
    blocks=None,
    buffers=2,
    bounds_checks=True,
):
    """``pallas_call`` over the bf16 parts (qh, ql, kh, kl, vh, vl), each
    [batch, rows, heads * 64] (rows: seq unless given; keys from seq on are
    masked) -> the float32 output, the same layout. bq (query rows a step,
    a multiple of 16; default :func:`query_block`) or blocks (query blocks
    a unit), buffers (the query blocks' DMA buffers) and bounds_checks
    leave every output element's arithmetic as it is."""
    rows = seq if rows is None else rows
    if heads % 2:
        raise ValueError(f"heads must be even, got {heads}")
    if not seq <= rows <= MAX_KEYS:
        raise ValueError(f"{seq} tokens in {rows} rows, at most {MAX_KEYS}")
    if blocks is not None:
        bq = _round_up(-(-rows // blocks), 16)
    bq = query_block(rows) if bq is None else bq
    kp = _round_up(seq, LANES)
    q_spec = pl.BlockSpec((None, bq, LANES), lambda b, p, i: (b, i, p))
    if buffers != 2:
        mode = pl.Buffered(buffers)
        q_in = pl.BlockSpec((None, bq, LANES), lambda b, p, i: (b, i, p), mode)
    else:
        q_in = q_spec
    checks = {} if bounds_checks else {"disable_bounds_checks": True}
    # The whole (padded) key range: rows from seq on are masked in-kernel.
    kv_spec = pl.BlockSpec((None, kp, LANES), lambda b, p, i: (b, 0, p))
    return pl.pallas_call(
        functools.partial(_kernel, seq=seq),
        grid=(batch, heads // 2, pl.cdiv(rows, bq)),
        in_specs=[q_in, q_in] + [kv_spec] * 4,
        out_specs=q_spec,
        out_shape=jax.ShapeDtypeStruct((batch, rows, heads * HEAD_DIM), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((2, kp, 2 * LANES), jnp.bfloat16),
            pltpu.VMEM((2, kp, 2 * LANES), jnp.bfloat16),
        ],
        compiler_params=pltpu.CompilerParams(
            # A head pair's query blocks reuse its K/V parts: in order.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            **checks,
        ),
        interpret=interpret,
        name="h3_vae_flash_attention",
    )


def flash_attention(q, k, v, *, interpret=False):
    """softmax(q k^T / 8) v per head over [batch, tokens, heads * 64] fp16
    (head h in columns 64h..64h+63) -> the same layout, fp16 (``interpret``:
    Pallas interpret mode, for checks off TPU). For q/k rows of RMS at most
    1 per head (see SCORE_BOUND)."""
    batch, seq, width = q.shape
    if width % (2 * HEAD_DIM):
        raise ValueError(f"width must be a multiple of {2 * HEAD_DIM}, got {width}")
    parts = [t for x in (q, k, v) for t in split_bf16(x)]
    out = flash_call(batch, seq, width // HEAD_DIM, interpret=interpret)(*parts)
    return out.astype(jnp.float16)


def flash_attention_jax(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """:func:`flash_attention` on TPU (the op's function)."""
    return flash_attention(q, k, v)


def op():
    """:func:`flash_attention_jax` as a torch op on TPU, registered once."""
    global _OP
    if _OP is None:
        with _OP_LOCK:
            if _OP is None:
                from sglang_tpu.pallas_jax_op import cpu_export_jax_op

                _OP = cpu_export_jax_op(
                    "sglang_tpu::h3_vae_flash_attention", flash_attention_jax
                )
    return _OP
