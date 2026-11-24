# SPDX-License-Identifier: Apache-2.0
"""The MiniMax H3 video VAE's ViT block as fused Pallas linears.

Used by minimax_h3_decode (SGLANG_TPU_OPT_H3_VAE_FUSED_LINEARS): a block (f32
residual stream, linears under fp16 autocast) is five custom calls, the flash
kernel of minimax_h3_vae_attention between them. qkv: RMSNorm prologue (rows
times the norm weight split into bf16 hi + lo in VMEM, their f32 1/RMS on the
sums), minimax_h3_vae_linear's three bf16 products, bias, per-head q/k RMSNorm
and NeoX RoPE in f32, written as the flash kernel's bf16 parts. out, w2: h +
(x W^T + b) * scale in f32. w1: RMSNorm prologue, gate and up from the
weight's halves, one fp16 rounding of SiLU(gate) * up. The stock block rounds
to fp16 13 times; here only the FFN activation (w2's fp16 input) is, and a
bf16 hi + lo operand keeps 16 significant bits (fp16: 11). Activations are
[batch, rows, width], rows a multiple of 8 (XLA's default layout is Mosaic's).
"""

# NOTE: no `from __future__ import annotations` here: torch_tpu's
# _infer_static_argnums identifies tensor arguments by `annotation is
# jax.Array`, which needs evaluated annotations on the registered function.

import dataclasses
import functools
import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.multimodal_gen import minimax_h3_vae_attention as attention
from sglang_tpu.multimodal_gen import minimax_h3_vae_linear as vl
from sglang_tpu.multimodal_gen.minimax_h3_vae_linear import GRANULE, decode_fp16

Array = jax.Array
HEAD = 64
LANES = 128
# Heads a qkv grid step takes: 256 output columns each of q, k and v.
STEP_HEADS = 4
MAX_ROW_BLOCK = 912
# VMEM of a call's operands, parts and epilogue tiles (minimax_h3_vae_linear
# leaves its output tiles out of its 48 MiB).
VMEM_BUDGET_BYTES = 50 << 20

_OPS = None
_OPS_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class Launch:
    """How a fused kernel's grid and steps are cut. Every field leaves each
    output element's arithmetic as it is (the K chunks and their order, the
    three products, the epilogue): it moves only which rows and columns a
    step and a dot take, the DMA buffering and the runtime checks. Defaults:
    the kernels' own choices.

    blocks: row blocks a unit (None: the fewest whose VMEM fits, at most
    MAX_ROW_BLOCK rows each); tn: output columns a grid step (qkv: heads
    a step x 64); granule: columns a dot and its epilogue take; pieces: row
    pieces of a step's last granule (None: LAST_CHUNK_PIECES where k <=
    2048, else 1); weight_buffers, x_buffers: DMA buffers of the weight and
    activation blocks (2: Pallas's double buffering; Mosaic takes 1 or 2,
    and no lookahead); weight_slots (gate, split weights): a ring
    of that many weight slots the kernel fills itself, slots - 1 steps
    ahead (deeper than Mosaic's two buffers); bounds_checks: the DMA
    bounds checks; vmem_mib: the kernel's VMEM limit (None: the default);
    split_epilogue (gate): write gate and up in f32 and take SiLU(gate) *
    up in a second kernel (:func:`silu_product`)."""

    blocks: int | None = None
    tn: int | None = None
    granule: int = GRANULE
    pieces: int | None = None
    weight_buffers: int = 2
    x_buffers: int = 2
    weight_slots: int | None = None
    bounds_checks: bool = True
    vmem_mib: int | None = None
    split_epilogue: bool = False

    def tm(self, rows: int, per_row: int, fixed: int) -> int:
        if self.blocks is None:
            return row_block(rows, per_row, fixed)
        return vl._round_up(-(-rows // self.blocks), 16)

    def spec(self, block, index, *, weight=False):
        buffers = self.weight_buffers if weight else self.x_buffers
        if buffers == 2:
            return pl.BlockSpec(block, index)
        return pl.BlockSpec(block, index, pipeline_mode=pl.Buffered(buffers))


DEFAULT = Launch()
_FIELDS = {f.name: f.type for f in dataclasses.fields(Launch)}


def parse_launches(spec: str) -> dict:
    """``kernel[@units]:field=value,...;...`` -> {(kernel, units or None):
    Launch}. Kernels: qkv, out, gate, w2, flash (flash takes bq, the query
    rows a step, or blocks, query blocks a unit; buffers; bounds_checks)."""
    table = {}
    for item in filter(None, (part.strip() for part in spec.split(";"))):
        head, _, body = item.partition(":")
        name, _, units = head.strip().partition("@")
        fields = {}
        for pair in filter(None, (p.strip() for p in body.split(","))):
            key, _, value = pair.partition("=")
            key, value = key.strip(), value.strip()
            if name == "flash":
                fields[key] = int(value)
                continue
            if key not in _FIELDS:
                raise ValueError(f"unknown fused launch field {key!r} in {item!r}")
            if key in ("bounds_checks", "split_epilogue"):
                fields[key] = value.lower() in ("1", "true", "on", "yes")
            else:
                fields[key] = None if value.lower() == "none" else int(value)
        if name not in ("qkv", "out", "gate", "w2", "flash"):
            raise ValueError(f"unknown fused kernel {name!r} in {item!r}")
        key = (name, int(units) if units else None)
        table[key] = fields if name == "flash" else Launch(**fields)
    return table


# The launches the ops use (SGLANG_TPU_OPT_H3_VAE_FUSED_LAUNCH, read when
# they are registered): {(kernel, units or None): Launch}.
_LAUNCHES: dict = {}

# The op bench's sweep (runner opbench/h3vae_fused.sh): a kernel's default
# and the launches timed against it, all exact (the bench and the unit tests
# check every output bit). Hypotheses for the gate's time over its static
# estimate: weight DMA (wb1 exposes it, ws3/ws4 fetch 2/3 steps ahead:
# Mosaic buffers only 1 or 2 deep), larger output tiles and dots (tn1024,
# g512), the epilogue in its own kernel (epi).
SWEEP = {
    "gate": {
        "base": DEFAULT,
        "b3": Launch(blocks=3),
        "b4": Launch(blocks=4),
        "b1x1": Launch(blocks=1, x_buffers=1),
        "tn256": Launch(tn=256),
        "tn1024": Launch(tn=1024),
        "tn1024x1": Launch(tn=1024, blocks=2, x_buffers=1),
        "g512": Launch(granule=512),
        "p1": Launch(pieces=1),
        "p2": Launch(pieces=2),
        "p8": Launch(pieces=8),
        "wb1": Launch(weight_buffers=1),
        "ws3": Launch(weight_slots=3),
        "ws4": Launch(weight_slots=4, blocks=3),
        "nochk": Launch(bounds_checks=False),
        "epi": Launch(split_epilogue=True),
    },
    "out": {
        "base": DEFAULT,
        "b3": Launch(blocks=3),
        "b4": Launch(blocks=4),
        "b1x1": Launch(blocks=1, x_buffers=1),
        "tn256": Launch(tn=256),
        "tn1024": Launch(tn=1024),
        "g512": Launch(granule=512),
        "p1": Launch(pieces=1),
        "p2": Launch(pieces=2),
        "p8": Launch(pieces=8),
        "wb1": Launch(weight_buffers=1),
        "nochk": Launch(bounds_checks=False),
    },
    "w2": {
        "base": DEFAULT,
        "b5": Launch(blocks=5),
        "b3x1": Launch(blocks=3, x_buffers=1),
        "tn512": Launch(tn=512),
        "p2": Launch(pieces=2),
        "p4": Launch(pieces=4),
        "wb1": Launch(weight_buffers=1),
        "nochk": Launch(bounds_checks=False),
    },
    "qkv": {
        "base": DEFAULT,
        "b3": Launch(blocks=3),
        "b4": Launch(blocks=4),
        "b1x1": Launch(blocks=1, x_buffers=1, vmem_mib=62),
        "h2": Launch(tn=2 * HEAD),
        "h8x1": Launch(tn=8 * HEAD, x_buffers=1),
        "p1": Launch(pieces=1),
        "p2": Launch(pieces=2),
        "p8": Launch(pieces=8),
        "wb1": Launch(weight_buffers=1),
        "nochk": Launch(bounds_checks=False),
    },
    "flash": {
        "base": {},
        "q6": {"blocks": 6},
        "q4": {"blocks": 4},
        "q2": {"blocks": 2},
        "q1": {"blocks": 1},
        "qb1": {"buffers": 1},
        "nochk": {"bounds_checks": 0},
    },
}


def _launch(kernel: str, units: int):
    found = _LAUNCHES.get((kernel, units), _LAUNCHES.get((kernel, None)))
    if kernel == "flash":
        return found or {}
    return found or DEFAULT


def row_block(rows: int, per_row: int, fixed: int) -> int:
    """Rows a grid step takes: the fewest blocks (multiples of 16) whose VMEM
    (``per_row`` bytes a row plus ``fixed``) fits VMEM_BUDGET_BYTES."""
    cap = MAX_ROW_BLOCK
    while cap > 16 and cap * per_row + fixed > VMEM_BUDGET_BYTES:
        cap -= 16
    blocks = -(-rows // cap)
    return min(vl._round_up(-(-rows // blocks), 16), vl._round_up(rows, 16))


def _rsqrt(a):
    """1/sqrt(a) to float32 accuracy whatever the EUP's: one Newton step."""
    r = lax.rsqrt(a)
    return r * (1.5 - 0.5 * a * r * r)


def _chunks(k: int) -> list[slice]:
    starts = [0] + list(range(vl.FIRST_K_CHUNK, k, vl.K_CHUNK))
    return [slice(c, e) for c, e in zip(starts, starts[1:] + [k])]


def _weight_parts(w_ref, *index):
    """The weight's bf16 (hi, lo) [256, c] at ``index``: fp16 bits split here,
    or its exact split held outside (a leading hi/lo axis, bf16): the same
    values either way."""
    if w_ref.dtype == jnp.bfloat16:
        hi, lo = (w_ref[(part, *index)] for part in (0, 1))
        return hi.reshape(-1, hi.shape[-1]), lo.reshape(-1, lo.shape[-1])
    f = decode_fp16(w_ref[index].astype(jnp.int32))
    return vl.split_bf16(f.reshape(-1, f.shape[-1]))


def _split_rows(x_ref, xh_ref, xl_ref, nw_ref=None, r_ref=None, eps=0.0):
    """The row block's bf16 parts into scratch at its first column block:
    fp16 bits decoded, or f32 as it is; with ``nw_ref`` (RMSNorm) the rows
    times the norm weight, and each row's 1/RMS into ``r_ref`` (the
    epilogue scales the sums by it: (x r w) W^T = r (x w) W^T)."""
    k = x_ref.shape[-1]
    if r_ref is not None:
        r_ref[...] = jnp.zeros(r_ref.shape, jnp.float32)

    def chunk(c, carry):
        cols = pl.ds(pl.multiple_of(c * GRANULE, GRANULE), GRANULE)
        f = x_ref[:, cols]
        if f.dtype == jnp.int16:
            f = decode_fp16(f.astype(jnp.int32))
        if r_ref is not None:
            r_ref[...] += f * f
            f = f * nw_ref[:, cols]
        xh_ref[:, cols], xl_ref[:, cols] = vl.split_bf16(f)
        return carry

    lax.fori_loop(0, k // GRANULE, chunk, 0)
    if r_ref is not None:
        mean = jnp.sum(r_ref[...], axis=1, keepdims=True) * (1.0 / k)
        r_ref[...] = jnp.broadcast_to(_rsqrt(mean + eps), r_ref.shape)


def _dot(a, b):
    return lax.dot_general(a, b, vl.NT_DIMS, preferred_element_type=jnp.float32)


def _sums(xh_ref, xl_ref, rows, chunks, parts):
    """hi*hi + hi*lo + lo*hi over the contraction chunks, f32."""
    acc = None
    for chunk, (wh, wl) in zip(chunks, parts):
        xh = xh_ref[rows, chunk]
        part = _dot(xh, wh) + _dot(xh, wl) + _dot(xl_ref[rows, chunk], wh)
        acc = part if acc is None else acc + part
    return acc


def _pieces(tm: int, k: int, last: bool = True, pieces: int | None = None) -> list:
    # Row pieces: each piece's epilogue overlaps the next piece's passes.
    if pieces is None:
        pieces = vl.LAST_CHUNK_PIECES if k <= 2048 else 1
    cuts = vl._row_pieces(tm, pieces if last else 1)
    return [slice(r0, r1) for r0, r1 in cuts]


def _first_column(*args):
    pl.when(pl.program_id(2) == 0)(functools.partial(_split_rows, *args))


def _row_scale(r_ref, rows, width):
    """The rows' 1/RMS (held GRANULE lanes wide) across ``width`` lanes."""
    r = r_ref[rows, :]
    if width < r.shape[1]:
        return r[:, :width]
    return r if width == r.shape[1] else jnp.tile(r, (1, width // r.shape[1]))


def _residual_kernel(
    x_ref, w_ref, b_ref, s_ref, h_ref, o_ref, xh_ref, xl_ref, *, launch=DEFAULT
):
    """o = h + (x W^T + b) * scale (x f32 or fp16 bits, h and o f32; W fp16
    bits, or its bf16 parts [2, tn, k] made once outside)."""
    k, (tm, tn), g = x_ref.shape[-1], o_ref.shape, launch.granule
    chunks = _chunks(k)
    _first_column(x_ref, xh_ref, xl_ref)
    bias = decode_fp16(b_ref[...].astype(jnp.int32))
    for n0 in range(0, tn, g):
        cols = slice(n0, n0 + g)
        parts = [_weight_parts(w_ref, cols, chunk) for chunk in chunks]
        for rows in _pieces(tm, k, n0 + g == tn, launch.pieces):
            acc = _sums(xh_ref, xl_ref, rows, chunks, parts) + bias[:, cols]
            o_ref[rows, cols] = h_ref[rows, cols] + acc * s_ref[:, cols]


def _silu_product(g, u):
    """fp16 bits of SiLU(g) * u (f32 g, u)."""
    return vl.encode_fp16(g * u / (1.0 + jnp.exp(-g))).astype(jnp.int16)


def _gate_kernel(
    x_ref, nw_ref, wg_ref, wu_ref, bg_ref, bu_ref, *refs, eps, launch=DEFAULT
):
    """o = fp16 bits of SiLU(g) * u, g and u = RMSNorm(x) W^T + b from the
    weight's gate and up rows (split_epilogue: g and u out as they are)."""
    outs, scratch = refs[:-3], refs[-3:]
    _gate_body(
        x_ref, nw_ref, wg_ref, wu_ref, bg_ref, bu_ref, outs, scratch, eps, launch
    )


def _gate_kernel_slots(x_ref, nw_ref, w_ref, bg_ref, bu_ref, *refs, eps, launch, up):
    """_gate_kernel with its weight blocks ([2, tn, k] bf16 hi/lo of the gate
    and up rows) fetched launch.weight_slots - 1 steps ahead into a ring of
    slots (the grid runs in order: steps are flattened (unit, rows, cols))."""
    outs, scratch, (ring, sems) = refs[:-5], refs[-5:-2], refs[-2:]
    slots, tn = launch.weight_slots, ring.shape[3]
    nj, ni = pl.num_programs(2), pl.num_programs(1)
    step = (pl.program_id(0) * ni + pl.program_id(1)) * nj + pl.program_id(2)
    total = pl.num_programs(0) * ni * nj

    def copies(t):
        col, slot = t % nj, t % slots
        return [
            pltpu.make_async_copy(
                w_ref.at[:, pl.ds((col + o) * tn, tn), :],
                ring.at[slot, half],
                sems.at[slot, half],
            )
            for half, o in enumerate((0, up))
        ]

    def start(t):
        @pl.when(t < total)
        def _():
            for copy in copies(t):
                copy.start()

    @pl.when(step == 0)
    def _():
        for t in range(slots - 1):
            start(jnp.int32(t))

    start(step + slots - 1)
    for copy in copies(step):
        copy.wait()
    slot = step % slots
    weights = (ring.at[slot, 0], ring.at[slot, 1])
    _gate_body(x_ref, nw_ref, *weights, bg_ref, bu_ref, outs, scratch, eps, launch)


def _gate_body(
    x_ref, nw_ref, wg_ref, wu_ref, bg_ref, bu_ref, outs, scratch, eps, launch
):
    xh_ref, xl_ref, r_ref = scratch
    k, (tm, tn), gr = x_ref.shape[-1], outs[0].shape, launch.granule
    chunks = _chunks(k)
    _first_column(x_ref, xh_ref, xl_ref, nw_ref, r_ref, eps)
    gate_bias, up_bias = (
        decode_fp16(b[...].astype(jnp.int32)) for b in (bg_ref, bu_ref)
    )
    for n0 in range(0, tn, gr):
        cols = slice(n0, n0 + gr)
        gate = [_weight_parts(wg_ref, cols, chunk) for chunk in chunks]
        up = [_weight_parts(wu_ref, cols, chunk) for chunk in chunks]
        for rows in _pieces(tm, k, n0 + gr == tn, launch.pieces):
            r = _row_scale(r_ref, rows, gr)
            g = _sums(xh_ref, xl_ref, rows, chunks, gate) * r + gate_bias[:, cols]
            u = _sums(xh_ref, xl_ref, rows, chunks, up) * r + up_bias[:, cols]
            if len(outs) == 2:
                outs[0][rows, cols], outs[1][rows, cols] = g, u
            else:
                outs[0][rows, cols] = _silu_product(g, u)


def _silu_kernel(g_ref, u_ref, o_ref):
    o_ref[...] = _silu_product(g_ref[...], u_ref[...])


def silu_product(g, u, *, interpret=False):
    """The gate kernel's epilogue alone: fp16 bits of SiLU(g) * u for f32
    [batch, rows, n] g and u (split_epilogue's second kernel)."""
    batch, rows, n = g.shape
    tm, tn = vl._round_up(-(-rows // 4), 16), min(n, 1024)
    tile = pl.BlockSpec((None, tm, tn), lambda b, i, j: (b, i, j))
    return pl.pallas_call(
        _silu_kernel,
        grid=(batch, pl.cdiv(rows, tm), pl.cdiv(n, tn)),
        in_specs=[tile, tile],
        out_specs=tile,
        out_shape=jax.ShapeDtypeStruct((batch, rows, n), jnp.int16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            vmem_limit_bytes=vl.VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="h3_vae_fused_silu",
    )(g, u)


def _head_sums(t):
    """[rows, 128] (two 64-wide heads) -> each lane its head's sum."""
    for shift in (1, 2, 4, 8, 16, 32):
        t = t + pltpu.roll(t, shift, 1)
    # Lane l now holds the sum of lanes l-63..l: heads end at 63 and 127.
    lane = lax.broadcasted_iota(jnp.int32, t.shape, 1)
    return jnp.where(lane < HEAD, t[:, HEAD - 1 : HEAD], t[:, LANES - 1 :])


def _norm_rope(y, cos, sin, eps, half):
    """Per 64-wide head of y [rows, heads * 64]: RMSNorm, then x * cos +
    partner * sin (partner: lane l + half below half, l - half up to 2 *
    half)."""
    out = []
    for lo in range(0, y.shape[1], LANES):
        t = y[:, lo : lo + LANES]
        t = t * _rsqrt(_head_sums(t * t) * (1.0 / HEAD) + eps)
        lane = lax.broadcasted_iota(jnp.int32, t.shape, 1) & (HEAD - 1)
        ahead, behind = pltpu.roll(t, LANES - half, 1), pltpu.roll(t, half, 1)
        out.append(t * cos + jnp.where(lane < half, ahead, behind) * sin)
    return jnp.concatenate(out, axis=1)


def _qkv_kernel(x_ref, nw_ref, *refs, eps, qk_eps, half, launch=DEFAULT):
    """q, k and v of a step's heads (STEP_HEADS by default) for a row block
    of one unit: x [tm, k] f32, nw [1, k] f32, w* [heads, 64, k] and b
    [8, heads * 64] (rows q, k, v) fp16 bits, tables [2, tm, 128] f32 (cos,
    sin); six outputs [tm, heads * 64] bf16."""
    weights, b_ref, t_ref, outs = refs[:3], refs[3], refs[4], refs[5:11]
    xh_ref, xl_ref, r_ref = refs[11:]
    k, tm = x_ref.shape[-1], x_ref.shape[0]
    chunks = _chunks(k)
    _first_column(x_ref, xh_ref, xl_ref, nw_ref, r_ref, eps)
    for part, w_ref in enumerate(weights):
        parts = [
            _weight_parts(w_ref, slice(None), slice(None), chunk) for chunk in chunks
        ]
        bias = decode_fp16(b_ref[part : part + 1, :].astype(jnp.int32))
        for rows in _pieces(tm, k, True, launch.pieces):
            y = _sums(xh_ref, xl_ref, rows, chunks, parts)
            y = y * _row_scale(r_ref, rows, bias.shape[1]) + bias
            if part < 2:  # q and k
                y = _norm_rope(y, t_ref[0, rows, :], t_ref[1, rows, :], qk_eps, half)
            outs[2 * part][rows, :], outs[2 * part + 1][rows, :] = vl.split_bf16(y)


def _call(kernel, grid, specs, outs, tm, k, norm, interpret, launch=DEFAULT, extra=()):
    """``pallas_call`` of a fused kernel; outs: (name, out specs, shapes);
    extra: more scratch (a kernel's own weight ring: steps in order)."""
    scratch = [pltpu.VMEM((tm, k), jnp.bfloat16)] * 2
    if norm:
        scratch.append(pltpu.VMEM((tm, GRANULE), jnp.float32))
    scratch += list(extra)
    order = ("arbitrary",) * 3 if extra else ("parallel", "parallel", "arbitrary")
    name, out_specs, out_shape = outs
    vmem = vl.VMEM_LIMIT_BYTES if launch.vmem_mib is None else launch.vmem_mib << 20
    checks = {} if launch.bounds_checks else {"disable_bounds_checks": True}
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            # A row block's column blocks reuse its parts: in order.
            dimension_semantics=order,
            vmem_limit_bytes=vmem,
            **checks,
        ),
        interpret=interpret,
        name=f"h3_vae_fused_{name}",
    )


def fused_residual(x, weight, bias, scale, h, *, launch=None, interpret=False):
    """h + (x weight^T + bias) * scale: x [batch, rows, k] f32 or fp16 bits
    (int16), weight [n, k] fp16 bits or [2, n, k] its bf16 hi and lo, bias
    [1, n] int16, scale [n] f32, h [batch, rows, n] f32 -> f32."""
    launch = launch or DEFAULT
    (batch, rows, k), n, split = x.shape, weight.shape[-2], weight.ndim == 3
    tn = launch.tn or (512 if n % 512 == 0 and not split else GRANULE)
    tm = launch.tm(
        rows, k * (2 * x.dtype.itemsize + 4) + tn * 16, tn * k * 4 * (1 + split)
    )
    tile = pl.BlockSpec((None, tm, tn), lambda b, i, j: (b, i, j))
    vector = pl.BlockSpec((1, tn), lambda b, i, j: (0, j))
    specs = [launch.spec((None, tm, k), lambda b, i, j: (b, i, 0))]
    if split:
        w = launch.spec((2, tn, k), lambda b, i, j: (0, j, 0), weight=True)
    else:
        w = launch.spec((tn, k), lambda b, i, j: (j, 0), weight=True)
    specs += [w, vector, vector, tile]
    out = ("residual", tile, jax.ShapeDtypeStruct((batch, rows, n), jnp.float32))
    grid = (batch, pl.cdiv(rows, tm), n // tn)
    kernel = _residual_kernel
    if launch != DEFAULT:
        kernel = functools.partial(_residual_kernel, launch=launch)
    call = _call(kernel, grid, specs, out, tm, k, False, interpret, launch)
    return call(x, weight, bias, scale.reshape(1, n), h)


def fused_gate(x, norm, weight, bias, eps, *, launch=None, interpret=False):
    """fp16 bits of SiLU(gate) * up for [gate | up] = RMSNorm(x) weight^T +
    bias: x [batch, rows, k] f32, norm [k] f32, weight [n, k] fp16 bits or
    [2, n, k] its bf16 hi and lo, bias [1, n] int16 -> [batch, rows, n / 2]
    int16."""
    launch = launch or DEFAULT
    if launch.split_epilogue:
        g, u = gate_sums(x, norm, weight, bias, eps, launch=launch, interpret=interpret)
        return silu_product(g, u, interpret=interpret)
    return _gate_call(x, norm, weight, bias, eps, launch, interpret, sums=False)


def gate_sums(x, norm, weight, bias, eps, *, launch=None, interpret=False):
    """The gate kernel without its epilogue: (gate, up) f32 [batch, rows,
    n / 2] each, as the kernel has them before SiLU(gate) * up."""
    return _gate_call(x, norm, weight, bias, eps, launch or DEFAULT, interpret, True)


def _gate_call(x, norm, weight, bias, eps, launch, interpret, sums):
    (batch, rows, k), half_n, split = x.shape, weight.shape[-2] // 2, weight.ndim == 3
    tn = launch.tn or (512 if half_n % 512 == 0 else GRANULE)
    tm = launch.tm(rows, k * 12 + GRANULE * 4 + tn * 4, 2 * tn * k * 4 * (1 + split))
    up = half_n // tn
    specs = [launch.spec((None, tm, k), lambda b, i, j: (b, i, 0))]
    specs += [pl.BlockSpec((1, k), lambda b, i, j: (0, 0))]
    if split:
        w_specs = [
            launch.spec((2, tn, k), lambda b, i, j, o=o: (0, j + o, 0), weight=True)
            for o in (0, up)
        ]
    else:
        w_specs = [
            launch.spec((tn, k), lambda b, i, j, o=o: (j + o, 0), weight=True)
            for o in (0, up)
        ]
    ring = bool(launch.weight_slots) and split  # int16 weights: Mosaic's buffers
    if ring:  # the kernel's own ring of weight slots
        w_specs = [pl.BlockSpec(memory_space=pl.ANY)]
    specs += w_specs
    specs += [pl.BlockSpec((1, tn), lambda b, i, j, o=o: (0, j + o)) for o in (0, up)]
    tile = pl.BlockSpec((None, tm, tn), lambda b, i, j: (b, i, j))
    if sums:
        part = jax.ShapeDtypeStruct((batch, rows, half_n), jnp.float32)
        out = ("gate_sums", [tile, tile], [part, part])
    else:
        out = ("gate", tile, jax.ShapeDtypeStruct((batch, rows, half_n), jnp.int16))
    kernel, extra, weights = functools.partial(_gate_kernel, eps=eps), (), (weight,) * 2
    if ring:
        slots = launch.weight_slots
        kernel = functools.partial(_gate_kernel_slots, eps=eps, launch=launch, up=up)
        extra = (
            pltpu.VMEM((slots, 2, 2, tn, k), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((slots, 2)),
        )
        weights = (weight,)
    elif launch != DEFAULT:
        kernel = functools.partial(_gate_kernel, eps=eps, launch=launch)
    grid = (batch, pl.cdiv(rows, tm), up)
    call = _call(kernel, grid, specs, out, tm, k, True, interpret, launch, extra)
    return call(x, norm.reshape(1, k), *weights, bias, bias)


def fused_qkv(
    x, norm, weight, bias, tables, eps, qk_eps, half, *, launch=None, interpret=False
):
    """The block's RMSNorm, to_qkv, q/k RMSNorm and RoPE: x [batch, rows, k]
    f32, norm [k] f32, weight [3 * heads * 64, k] fp16 bits (or [2, ...] its
    bf16 hi and lo), bias [8, heads * 64] (rows q, k, v, 0...) fp16 bits,
    tables [2, rows, 128] f32 for two
    heads' lanes (cos, 1 past the rotary lanes; sin signed as rotate_half:
    -sin below ``half``, sin up to 2 * half, 0 past) -> the flash kernel's
    parts (qh, ql, kh, kl, vh, vl), [batch, rows, heads * 64] bf16 each."""
    launch = launch or DEFAULT
    (batch, rows, k), heads = x.shape, weight.shape[-2] // (3 * HEAD)
    split = weight.ndim == 3
    width = launch.tn or STEP_HEADS * HEAD
    step = width // HEAD
    if heads % step:
        raise ValueError(f"heads must be a multiple of {step}, got {heads}")
    per_row = k * 12 + GRANULE * 4 + width * 24 + 2048
    tm = launch.tm(rows, per_row, 3 * width * k * 4 * (1 + split))
    w = weight.reshape(*weight.shape[:-2], heads, 3, HEAD, k)
    specs = [launch.spec((None, tm, k), lambda b_, i, j: (b_, i, 0))]
    specs += [pl.BlockSpec((1, k), lambda b_, i, j: (0, 0))]
    lead = (2,) if split else ()
    specs += [
        launch.spec(
            (*lead, step, None, HEAD, k),
            lambda b_, i, j, p=p: (*(0,) * len(lead), j, p, 0, 0),
            weight=True,
        )
        for p in range(3)
    ]
    specs += [pl.BlockSpec((8, width), lambda b_, i, j: (0, j))]
    specs += [pl.BlockSpec((2, tm, LANES), lambda b_, i, j: (0, i, 0))]
    tile = pl.BlockSpec((None, tm, width), lambda b_, i, j: (b_, i, j))
    part = jax.ShapeDtypeStruct((batch, rows, heads * HEAD), jnp.bfloat16)
    kernel = functools.partial(_qkv_kernel, eps=eps, qk_eps=qk_eps, half=half)
    if launch != DEFAULT:
        kernel = functools.partial(kernel, launch=launch)
    grid = (batch, pl.cdiv(rows, tm), heads // step)
    outs = ("qkv", [tile] * 6, [part] * 6)
    call = _call(kernel, grid, specs, outs, tm, k, True, interpret, launch)
    return tuple(call(x, norm.reshape(1, k), w, w, w, bias, tables))


def _split_kernel(w_ref, o_ref):
    o_ref[0], o_ref[1] = vl.split_bf16(decode_fp16(w_ref[...].astype(jnp.int32)))


def split_weight(bits, *, interpret=False):
    """fp16 bits [n, k] (int16) -> [2, n, k] bf16, their exact hi and lo (the
    kernels' own split) in one pass (XLA would write hi and lo, then copy)."""
    n, k = bits.shape
    tn = max(16, min(512, (2 << 20) // (2 * k)) // 16 * 16)
    return pl.pallas_call(
        _split_kernel,
        grid=(pl.cdiv(n, tn),),
        in_specs=[pl.BlockSpec((tn, k), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((2, tn, k), lambda i: (0, i, 0)),
        out_shape=jax.ShapeDtypeStruct((2, n, k), jnp.bfloat16),
        interpret=interpret,
        name="h3_vae_split_weight",
    )(bits)


def split_jax(bits: Array) -> Array:
    return split_weight(bits)


def qkv_jax(
    x: Array, nw: Array, w: Array, b: Array, t: Array, e: float, q: float, half: int
) -> tuple[Array, Array, Array, Array, Array, Array]:
    return fused_qkv(x, nw, w, b, t, e, q, half, launch=_launch("qkv", x.shape[0]))


def residual_jax(x: Array, w: Array, b: Array, scale: Array, h: Array) -> Array:
    # w2 takes the gate's fp16 bits; out the flash output (f32).
    name = "w2" if x.dtype == jnp.int16 else "out"
    return fused_residual(x, w, b, scale, h, launch=_launch(name, x.shape[0]))


def gate_jax(x: Array, nw: Array, w: Array, b: Array, eps: float) -> Array:
    return fused_gate(x, nw, w, b, eps, launch=_launch("gate", x.shape[0]))


def flash_parts_jax(
    qh: Array, ql: Array, kh: Array, kl: Array, vh: Array, vl: Array, seq: int
) -> Array:
    """The flash kernel on its bf16 parts [batch, rows, heads * 64] (keys
    from ``seq`` on masked) -> its f32 output, not rounded to fp16."""
    batch, rows, width = qh.shape
    launch = _launch("flash", batch)
    call = attention.flash_call(batch, seq, width // HEAD, rows=rows, **launch)
    return call(qh, ql, kh, kl, vh, vl)


def ops():
    """(qkv, residual, gate, flash_parts, split) as torch ops on TPU,
    registered once (with SGLANG_TPU_OPT_H3_VAE_FUSED_LAUNCH's launches)."""
    global _OPS, _LAUNCHES
    if _OPS is None:
        with _OPS_LOCK:
            if _OPS is None:
                from sglang_tpu.envs import envs as tpu_envs
                from sglang_tpu.pallas_jax_op import cpu_export_jax_op

                spec = tpu_envs.SGLANG_TPU_OPT_H3_VAE_FUSED_LAUNCH.get() or ""
                _LAUNCHES = parse_launches(spec)

                functions = (
                    qkv_jax,
                    residual_jax,
                    gate_jax,
                    flash_parts_jax,
                    split_jax,
                )
                names = ("qkv", "residual", "gate", "flash_parts", "split")
                _OPS = tuple(
                    cpu_export_jax_op(f"sglang_tpu::h3_vae_fused_{name}", fn)
                    for name, fn in zip(names, functions)
                )
    return _OPS
