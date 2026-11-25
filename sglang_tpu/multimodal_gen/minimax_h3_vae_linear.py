# SPDX-License-Identifier: Apache-2.0
"""The MiniMax H3 video VAE's ViT block linears as an fp16-faithful Pallas kernel.

Used by minimax_h3_decode (SGLANG_TPU_OPT_H3_VAE_PALLAS_LINEAR) for to_qkv,
to_out, w1 and w2 (2048 -> 6144, 2048, 16384; 8192 -> 2048; 1,797 rows a
unit) under the decode's fp16 autocast. XLA runs that fp16 dot as three bf16
MXU passes (hi*hi + hi*lo + lo*hi), streaming the activations as unpacked
float32 rows, and rounds the product to fp16 before adding the fp16 bias.
Mosaic has no fp16 vector type on tpu7x, so the kernel takes fp16 bits
(int16) and

- decodes them to float32 exactly (subnormals through an integer to float
  conversion, which float32 flush-to-zero never sees) and splits each value
  into bf16 hi (rounded to nearest) plus lo (the exact rest): the activations
  once a row block into VMEM scratch, the weights at every grid step on the
  VPU, beside the MXU's work (no split weights in HBM);
- runs XLA's three products as bf16 MXU passes, float32 accumulation;
- adds the bias in float32 and rounds to fp16 once (to nearest even, with
  subnormals, overflow to inf and NaN), stored as int16 bits.

One rounding where today's path rounds twice: on the decoder's real
activations its relative L2 error against float64 is 0.72x today's TPU fp16
linear and within 1e-4 of an fp16 matmul with float32 accumulation (the
released decode's cuBLAS form). The dropped lo*lo (at most 2^-18 |x w| a
product) would cost a fourth MXU pass. A row with a non-finite input comes out
non-finite (NaN may stand for inf).
"""

# NOTE: no `from __future__ import annotations` here: torch_tpu's
# _infer_static_argnums identifies tensor arguments by `annotation is
# jax.Array`, which needs evaluated annotations on the registered function.

import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# An MXU pass is 256 wide: a grid step's output columns go through the MXU and
# the epilogue in chunks of it.
GRANULE = 256
# Contraction chunks of the splits (bounds the float32 temporaries); the first
# is a single pass deep, so the MXU starts after a short split.
K_CHUNK = 512
FIRST_K_CHUNK = 256
# Rows a grid step takes at most: 2 blocks a 1,797-row unit.
MAX_ROW_BLOCK = 912
# Row pieces of a step's last column chunk at k <= 2048: each piece's epilogue
# overlaps the next piece's passes (about 600 fewer cycles a step).
LAST_CHUNK_PIECES = 4
# VMEM a call may use (tpu7x has 64 MiB a TensorCore); the row block shrinks
# until its buffers fit in VMEM_BUDGET_BYTES (512 rows at k = 8192).
VMEM_LIMIT_BYTES = 56 << 20
VMEM_BUDGET_BYTES = 48 << 20
NT_DIMS = (((1,), (1,)), ((), ()))

_OP = None
_OP_LOCK = threading.Lock()


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def tiles(m: int, k: int, n: int) -> tuple[int, int]:
    """(row block, column block) for an [m, k] x [n, k]^T call: 1024 columns
    at k <= 2048 (512 at larger k, as n allows), and the fewest row blocks
    (multiples of 16, bf16 sublane tiles) whose buffers fit VMEM (x bits
    twice, its bf16 parts, the weight bits twice)."""
    tn = next(c for c in (1024, 512, GRANULE) if n % c == 0 and (c < 1024 or k <= 2048))
    cap = MAX_ROW_BLOCK
    while cap > 16 and cap * k * 8 + tn * k * 4 > VMEM_BUDGET_BYTES:
        cap -= 16
    blocks = -(-m // cap)
    return min(_round_up(-(-m // blocks), 16), _round_up(m, 16)), tn


def decode_fp16(h):
    """int32 holding fp16 bits -> float32, exactly."""
    a = h & 0x7FFF
    sign = (h & 0x8000) << 16
    # Rebias the exponent (inf/NaN to 255); subnormals are a * 2^-24.
    normal = (a << 13) + jnp.where(a >= 0x7C00, 0x70000000, 0x38000000)
    sub = lax.bitcast_convert_type(a.astype(jnp.float32) * 2.0**-24, jnp.int32)
    return lax.bitcast_convert_type(
        jnp.where(a < 0x400, sub, normal) | sign, jnp.float32
    )


def encode_fp16(f):
    """float32 -> int32 holding its fp16 rounding's bits (to nearest even)."""
    u = lax.bitcast_convert_type(f, jnp.int32)
    sign = (u >> 16) & 0x8000
    a = u & 0x7FFFFFFF
    normal = (a + ((a >> 13) & 1) + (0xFFF - 0x38000000)) >> 13
    # Below 2^-14: the float32 sum with 0.5 rounds to a multiple of 2^-24.
    sub = (
        lax.bitcast_convert_type(
            lax.bitcast_convert_type(a, jnp.float32) + 0.5, jnp.int32
        )
        - 0x3F000000
    )
    h = jnp.where(a < 0x38800000, sub, normal)
    h = jnp.where(a >= 0x477FF000, 0x7C00, h)
    h = jnp.where(a > 0x7F800000, 0x7E00, h)
    return h | sign


def split_bf16(f):
    """float32 holding an fp16 value -> bf16 (hi, lo), hi + lo == f exactly."""
    hi = f.astype(jnp.bfloat16)
    return hi, (f - hi.astype(jnp.float32)).astype(jnp.bfloat16)


def _parts(ref, rows, cols):
    return split_bf16(decode_fp16(ref[rows, cols].astype(jnp.int32)))


def _row_pieces(rows: int, pieces: int) -> list[tuple[int, int]]:
    size = rows // pieces // 16 * 16
    if size == 0:
        return [(0, rows)]
    cuts = [size * p for p in range(pieces)] + [rows]
    return list(zip(cuts[:-1], cuts[1:]))


def _kernel(x_ref, w_ref, b_ref, o_ref, xh_ref, xl_ref):
    """One (row block, column block) cell.

    x_ref: [tm, k] int16 fp16 bits (rows past m undefined: their outputs are
    dropped); w_ref: [tn, k] int16, the weight's rows (output columns);
    b_ref: [1, tn] int16; o_ref: [tm, tn] int16. Scratch xh_ref, xl_ref:
    [tm, k] bf16, the row block's parts, made at its first column block.
    """
    k = x_ref.shape[1]
    tm, tn = o_ref.shape
    starts = [0] + list(range(FIRST_K_CHUNK, k, K_CHUNK))
    chunks = [slice(c, e) for c, e in zip(starts, starts[1:] + [k])]
    pieces = _row_pieces(tm, LAST_CHUNK_PIECES if k <= 2048 else 1)

    @pl.when(pl.program_id(1) == 0)
    def _split_rows():
        # A rolled loop: the code stays one chunk long (instruction memory,
        # compile time); it runs once a row block.
        def chunk(c, carry):
            cols = pl.ds(pl.multiple_of(c * GRANULE, GRANULE), GRANULE)
            xh_ref[:, cols], xl_ref[:, cols] = _parts(x_ref, slice(None), cols)
            return carry

        lax.fori_loop(0, k // GRANULE, chunk, 0)

    def dot(a, b):
        return lax.dot_general(a, b, NT_DIMS, preferred_element_type=jnp.float32)

    bias = decode_fp16(b_ref[...].astype(jnp.int32))
    # Column chunks: one chunk's epilogue overlaps the next one's passes.
    for n0 in range(0, tn, GRANULE):
        cols = slice(n0, n0 + GRANULE)
        parts = [_parts(w_ref, cols, chunk) for chunk in chunks]
        for r0, r1 in pieces if n0 + GRANULE == tn else [(0, tm)]:
            rows = slice(r0, r1)
            acc = None
            for chunk, (wh, wl) in zip(chunks, parts):
                xh = xh_ref[rows, chunk]
                part = dot(xh, wh) + dot(xh, wl) + dot(xl_ref[rows, chunk], wh)
                acc = part if acc is None else acc + part
            o_ref[rows, cols] = encode_fp16(acc + bias[:, cols]).astype(jnp.int16)


def linear_call(m: int, k: int, n: int, *, interpret=False):
    """``pallas_call`` over fp16 bits (x [m, k], weight [n, k], bias [1, n],
    int16) -> the output's fp16 bits [m, n] (int16)."""
    if k % GRANULE or n % GRANULE:
        raise ValueError(f"k and n must be multiples of {GRANULE}, got {k}, {n}")
    tm, tn = tiles(m, k, n)
    return pl.pallas_call(
        _kernel,
        grid=(pl.cdiv(m, tm), n // tn),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((tn, k), lambda i, j: (j, 0)),
            pl.BlockSpec((1, tn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.int16),
        scratch_shapes=[pltpu.VMEM((tm, k), jnp.bfloat16)] * 2,
        compiler_params=pltpu.CompilerParams(
            # A row block's column blocks reuse its parts: in order.
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="h3_vae_linear",
    )


def pallas_linear(x, weight, bias, *, interpret=False):
    """fp16 x [m, k] times fp16 weight [n, k]^T plus fp16 bias [n] -> fp16
    [m, n] (``interpret``: Pallas interpret mode, for checks off TPU). XLA
    makes each bitcast a copy; the op takes the bits from torch instead."""

    def bits(t):
        return lax.bitcast_convert_type(t.astype(jnp.float16), jnp.int16)

    out = linear_bits(
        bits(x), bits(weight), bits(bias).reshape(1, -1), interpret=interpret
    )
    return lax.bitcast_convert_type(out, jnp.float16)


def linear_bits(x, weight, bias, *, interpret=False):
    """:func:`pallas_linear` on fp16 bits: x [m, k], weight [n, k], bias
    [1, n] int16 -> [m, n] int16."""
    m, k = x.shape
    return linear_call(m, k, weight.shape[0], interpret=interpret)(x, weight, bias)


def linear_bits_jax(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """:func:`linear_bits` on TPU (the op's function: a bare custom call;
    torch passes fp16 tensors as int16 views)."""
    return linear_bits(x, weight, bias)


def op():
    """:func:`linear_bits_jax` as a torch op on TPU, registered once."""
    global _OP
    if _OP is None:
        with _OP_LOCK:
            if _OP is None:
                from sglang_tpu.pallas_jax_op import cpu_export_jax_op

                _OP = cpu_export_jax_op("sglang_tpu::h3_vae_linear", linear_bits_jax)
    return _OP
