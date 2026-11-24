# SPDX-License-Identifier: Apache-2.0
"""The block stack as one torch op over the Ulysses group (torch_tpu).

``cpu_export_jax_op`` exports ``kernel.stack`` under a ``shard_map`` over a
1-D ``"sp"`` mesh whose position ``i`` is world rank ``i``
(``get_pallas_mesh``). Each process passes its local tensors: its row shard
of ``x``, ``onehot``, ``cos`` and ``sin`` (``P("sp")``) and its full copy of
the tables, the live length and the packed weights (``P()``, global ==
local). The live length is a ``[1]`` int32 operand, so every prompt length
of one padded row count shares one program. Nothing is donated; the exchange
buffers live inside the call.

The vouched variant (``vouch_margin``, :func:`local_vouched_stack`) also
takes the heads a step tries at a fixed max and returns the heads its own
peaks vouch for next; it checks its result on device and recomputes it with
the proven layers only when a tried head left the fixed-max bound. With the
static-structure option too, it takes the static structure last and its
kernel call takes the structured op's options (``stack_kwargs``).
"""

# No `from __future__ import annotations`: cpu_export_jax_op reads the
# signature's jax.Array annotations to tell tensors from static arguments.

import dataclasses
import functools
import re

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu.multimodal_gen.h3_megakernel import kernel, pack
from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

AXIS = "sp"
# The kernel's tiling bounds for padding: padded_rows picks the packed row
# count from these (4,992 rows a device at 8 devices for the 1344x768x124
# request).
ROWS_TILING = tl.Tiling(sub_rows=640, stage_slots=24, block_kv_compute_online=1024)
# What the kernel runs with: those bounds plus the tuned MLP tiling (mlp_rows
# 640, chunk_rows 80); stack() cuts them down to divide each row count
# (Tiling.fit). qkv_rows 704: with 24 staging slots a 768-row phase-A tile
# needs ~69 MiB of the 63.94 MiB of VMEM (a 10 s video at 9,216 rows a
# device does not compile), 704 fits. The 1344x768x124 row counts (4,992,
# 4,864, 9,600, 18,944, 19,456 a device) fit the same tiles either way.
STACK_TILING = dataclasses.replace(
    ROWS_TILING, qkv_rows=704, mlp_rows=640, chunk_rows=80
)
# The call's softmax and precision options, pinned. With fixed and
# fixed_heads None the kernel runs a fixed-max softmax on exactly the layers
# its own fixed_max_layers proves from the packed q/k norm weights (a bound
# that holds for every input) and the online softmax elsewhere: nothing is
# vouched by measured activations, flags files or calibration. bf16 weights
# (no fp8), no sparse, windowed or far-cache attention, no peaks, the
# default fixed max of 17 (the one the norm bound vouches for).
STACK_KWARGS = {
    "fixed": None,
    "fixed_heads": None,
    "head_perm": None,
    "return_peaks": False,
    "fp8": False,
    "sparse": None,
    "window": None,
    "far": None,
    "far_mode": None,
    "tile_masks": None,
    "listed": None,
    "narrow_tail": False,
    "scaled_v": False,
    "fast_phase_c": False,
    "fixed_max": 17.0,
}
# The static-structure option (SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE)
# runs a static structure at these (devices, rows a device), the 1344x768x124
# request's row counts it was AOT-checked at (VMEM, loop sizes), and the
# generic walk at any other.
STRUCTURED_SHAPES = frozenset(
    {(8, 4992), (8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}
)
# And it pads to 19,456 rows a device at 2 devices where padded_rows gives
# 18,944 (the 1344x768x124 request): 2,432-row query tiles, the per-device tiles
# of 8 x 4,864, instead of 512-row ones.
STRUCTURED_ROWS = {(2, 18944): 19456}
# A structured call's own tiling at some (devices, rows a device), others
# STACK_TILING. 2 x 19,200 rows (SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS=19200 at
# 2 devices: 650 rows of padding where 19,456 carry 1,162) run the 4 x 9,600
# tiles (3,200-row query tiles, 640-row phase tiles) with 19,456's 2,048-key
# tiles (key_tile; 38,400 rows hold 18.75 of them): each real row adds
# 19,456's key sub-blocks in its order. Rounding-level against 19,456 rows
# on v7x: the compiler splits the P V sum of a query tile's last 128 lanes
# over both MXUs, and those lanes are other rows at 3,200-row tiles.
STRUCTURED_TILINGS = {
    (2, 19200): dataclasses.replace(STACK_TILING, block_q=3200, key_tile=2048)
}
# Every call's tiling at some (devices, rows a device) STACK_TILING does not
# cut. 8 x 4,800 rows (SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS=4800 at 8 devices:
# 650 rows of padding where 4,864 carry 1,162) run 4 x 9,600's attention
# tiles: 3,200-row query tiles that cross a
# device boundary at their middle (split_q_tiles; no 128-row tile divides
# 4,800) and 2,560-key tiles of 1,280- and 640-key sub-blocks, with 480-row
# phase tiles. Rounding-level against 4,864 rows.
SHAPE_TILINGS = {(8, 4800): dataclasses.replace(STACK_TILING, split_q_tiles=True)}
# Kernel levers of the structured op's calls, each behind its own flag
# (hook._levers), at the (devices, rows a device) it was AOT-checked at; other
# calls run without it (lean_cut and split_items shape only a static
# structure's loops, so its generic-walk calls run without them too).
#   lean_cut: kernel.stack cut_kv_compute (the fixed-max cut tile's last
#     real sub-block halved; bitwise), at the 2,048-key sub-blocks of 8 x
#     4,864, 2 x 19,456 and 2 x 19,200 rows (one 1,024-key sub-block for the
#     1344x768x124 structure) and the 1,280-key ones of 4 x 9,600 and 8 x 4,800
#     (640 keys of the cut tile's second).
#   split_items: split_items (one item loop per query width; bitwise).
#   epilogue_chunks: epilogue_chunk 64 at the 608- and 480-row phase A
#     tiles, whose epilogue rows (a divisor of the tile) are 32 and 48.
#   wide_kv: kv_group 2 (each key DMA and key step two 2,048-key tiles, run
#     as their 2,048- and 1,024-key sub-blocks; bitwise) at 38,912 rows: 18
#     key steps an item become 9. 9,600 rows' 2,560-key tiles would make an
#     online key step of eight 640-key sub-blocks; they keep their program.
#   wide_kv_online: the same for the online softmax only, kv_group (1, 2)
#     (at 3,200-row query tiles the grouped fixed-max step ran slower on
#     v7x, the online one faster); also at 4 x 9,600 rows, whose online key
#     step becomes eight 640-key sub-blocks of two 2,560-key tiles (the same
#     bits on TPU7x, and faster per online head). With 2,560-key tiles the
#     grouped fixed-max step (wide_kv) does change bits on TPU7x.
#   modulate_pieces: kernel.stack modulate_pieces, a piece a chunk
#     (``modulate_pieces``): the next row tile's modulate one chunk a phase A
#     step (19 steps of 32 rows at 608-row tiles, 8 of 80 at 640, 6 of 80 at
#     480) instead of one or two steps whose modulate outlasts their matmul
#     (bitwise).
#   prefetch_down: prefetch_down at 4 x 9,600 rows, where stack_kwargs turns it
#     off: the down projection's first weight tile loads during the up
#     projection (63.45 of the 63.94 MiB of VMEM there; the same bits).
#   uniform_adaln: uniform_adaln (phase C's uniform row tiles modulate the MLP
#     input with one lookup of its shift and scale, no one-hot matmul per
#     chunk; bitwise), at every structured shape.
#   zero_ahead: zero_ahead (each phase B item's running state zeroed in the
#     item before's epilogue block; bitwise), at every structured shape.
#   mask_scores: mask_scores (a masked sub-block subtracts a per-key column,
#     +inf at the keys past the live length, instead of selecting its
#     probabilities: exp2(-inf) is the select's +0; bitwise), at every
#     structured shape.
LEVER_SHAPES = {
    "lean_cut": frozenset({(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}),
    "split_items": frozenset({(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}),
    "epilogue_chunks": frozenset({(8, 4864), (8, 4800), (2, 19456)}),
    "wide_kv": frozenset({(8, 4864), (2, 19456), (2, 19200)}),
    "wide_kv_online": frozenset({(8, 4864), (4, 9600), (2, 19456), (2, 19200)}),
    "modulate_pieces": frozenset(
        {(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}
    ),
    "prefetch_down": frozenset({(4, 9600)}),
    "uniform_adaln": frozenset(
        {(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}
    ),
    "zero_ahead": frozenset({(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}),
    "mask_scores": frozenset({(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}),
}
# lean_cut's smallest sub-block: whole 256-key MXU passes.
CUT_KV_COMPUTE = 256
EPILOGUE_CHUNK = 64
KV_GROUP = 2


# Tiling bounds SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING may set (parse_tiling).
TILING_FIELDS = frozenset(
    {
        "block_q",
        "split_q_tiles",
        "key_tile",
        "block_kv",
        "block_kv_compute",
        "block_kv_compute_online",
        "qkv_rows",
        "mlp_rows",
        "sub_rows",
        "chunk_rows",
        "epilogue_rows",
        "stage_slots",
    }
)


def parse_tiling(text: str | None) -> tuple:
    """SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING's bounds: sorted (field, value)
    pairs (hashable), () when unset.

    Raises:
      ValueError: On a field outside ``TILING_FIELDS`` or a value that is not
        a whole number (split_q_tiles: 0 or 1).
    """
    pairs = {}
    for item in re.split(r"[,+\s]+", (text or "").strip()):
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep or name not in TILING_FIELDS or not value.isdigit():
            raise ValueError(
                f"tiling bound {item!r}: takes name=<whole number>, name one of"
                f" {', '.join(sorted(TILING_FIELDS))}"
            )
        pairs[name] = bool(int(value)) if name == "split_q_tiles" else int(value)
    return tuple(sorted(pairs.items()))


def shape_tiling(devices: int, rows: int, tiling: tl.Tiling = STACK_TILING):
    """The tiling of any call at ``rows`` a device: ``SHAPE_TILINGS``' at its
    shapes, else ``tiling``."""
    return SHAPE_TILINGS.get((devices, rows), tiling)


def structured_tiling(
    devices: int, rows: int, tiling: tl.Tiling = STACK_TILING, overrides=()
):
    """The tiling of a structured call (a nonzero static structure) at
    ``rows`` a device: ``STRUCTURED_TILINGS``' at its shapes, else
    ``shape_tiling``'s, with ``overrides`` (``parse_tiling``'s pairs)."""
    base = STRUCTURED_TILINGS.get((devices, rows)) or shape_tiling(
        devices, rows, tiling
    )
    return dataclasses.replace(base, **dict(overrides)) if overrides else base


def cut_kv_compute(devices: int, rows: int | None, levers=frozenset()) -> int:
    """kernel.stack's cut_kv_compute for a structured call at ``rows`` a
    device (0: off), also what its loop structure takes."""
    on = "lean_cut" in levers and (devices, rows) in LEVER_SHAPES["lean_cut"]
    return CUT_KV_COMPUTE if on else 0


def modulate_pieces(
    devices: int, rows: int, tiling: tl.Tiling | None = None, config=None
) -> int:
    """kernel.stack's modulate_pieces for a structured call at ``rows`` a
    device: one piece a chunk of its phase A row tile (``tiling``: the call's,
    default ``structured_tiling``'s), at most one a head group step."""
    config = config or ref.Config()
    tiling = (tiling or structured_tiling(devices, rows)).fit(config, rows, devices)
    chunks = tiling.qkv_rows // tiling.chunk_rows
    groups = config.heads // pack.tile_widths(config)["qkv_heads"]
    return max(p for p in range(1, min(chunks, groups) + 1) if chunks % p == 0)


def stack_kwargs(
    devices: int,
    structure: int | None = None,
    rows: int | None = None,
    levers=frozenset(),
    tiling: tl.Tiling | None = None,
    config=None,
) -> dict:
    """The call's kernel options; ``structure`` under the static-structure
    option: the static live length whose loop structure the loops take (0:
    none, the generic walk). That option also passes narrow_tail, which only
    a static structure shapes, and at 4 devices turns off the phase C
    down-weight prefetch: at 9,600 rows a device the prefetch leaves 0.49 MiB
    of the 63.94 MiB of VMEM, 2.36 MiB without it (the same bits). And each
    of ``levers`` (``LEVER_SHAPES``' names) that covers ``rows`` a device;
    ``tiling`` and ``config``: the call's (``modulate_pieces``)."""
    if structure is None:
        return STACK_KWARGS
    kwargs = dict(STACK_KWARGS, narrow_tail=True, static_structure=structure or None)
    if devices == 4:
        kwargs["prefetch_down"] = False
    # Only a static structure's calls (the generic walk keeps its program).
    on = {
        lever
        for lever in levers
        if structure and (devices, rows) in LEVER_SHAPES[lever]
    }
    if "lean_cut" in on:
        kwargs["cut_kv_compute"] = cut_kv_compute(devices, rows, levers)
    if "split_items" in on:
        kwargs["split_items"] = True
    if "epilogue_chunks" in on:
        kwargs["epilogue_chunk"] = EPILOGUE_CHUNK
    if "wide_kv" in on:
        kwargs["kv_group"] = KV_GROUP
    elif "wide_kv_online" in on:
        kwargs["kv_group"] = (1, KV_GROUP)
    if "modulate_pieces" in on:
        kwargs["modulate_pieces"] = modulate_pieces(devices, rows, tiling, config)
    if "prefetch_down" in on:
        kwargs["prefetch_down"] = True
    if "uniform_adaln" in on:
        kwargs["uniform_adaln"] = True
    if "zero_ahead" in on:
        kwargs["zero_ahead"] = True
    if "mask_scores" in on:
        kwargs["mask_scores"] = True
    return kwargs


def local_stack(
    x,
    onehot,
    cos,
    sin,
    tables,
    used,
    *weights,
    config: ref.Config,
    tiling: tl.Tiling,
    axis: str | None,
    interpret=False,
    options: dict = STACK_KWARGS,
):
    """One device's call: ``used`` is a ``[1]`` int32 array (traced),
    ``weights`` in ``pack.WEIGHT_ORDER``, ``options`` from ``stack_kwargs``."""
    return kernel.stack(
        x,
        dict(zip(pack.WEIGHT_ORDER, weights)),
        tables,
        onehot,
        cos,
        sin,
        used[0],
        config=config,
        tiling=tiling,
        axis=axis,
        interpret=interpret,
        **options,
    )


def local_vouched_stack(
    x,
    onehot,
    cos,
    sin,
    tables,
    used,
    vouch,
    tally,
    *weights,
    config: ref.Config,
    tiling: tl.Tiling,
    axis: str | None,
    margin: float,
    interpret=False,
    options: dict = STACK_KWARGS,
):
    """One device's call of a step that also tries ``vouch``'s heads at a
    fixed max: ``local_stack``'s kernel call (``options``), with
    ``fixed_heads`` and the kernel's peaks, in a loop that runs it at most
    twice.

    The kernel records, for every head of the layers the norm weights leave
    online, the largest squared norms of the bf16 q and k rows its phase B
    multiplies; by Cauchy-Schwarz no score of the head exceeds the root of
    their product, the largest over every device (``kernel.layer_bounds``).
    A fixed-max softmax is the softmax up to rounding while every score
    stays within ``kernel._FIXED_BOUND`` (``kernel.safe_heads``, the check
    ``stack`` asks of a caller's ``fixed_heads``). The first attempt runs
    the tried heads at a fixed max; if its own peaks leave any of them
    outside the bound, its result is dropped and the second attempt runs the
    proven layers only, the call ``local_stack`` makes, on the same rows.
    Both attempts run the one compiled kernel: the heads reach it as scalar
    flags.

    Args:
      vouch: ``[L, heads]`` int32, nonzero for each head to try at a fixed
        max (the caller's previous step's ``next`` output, zeros on a
        request's first step); heads of norm-proven layers are ignored, and
        so is any head a device was not given (the devices try the heads all
        of them were given, as the kernel's head order needs).
      tally: ``[2]`` int32, the request's recomputed steps and tried heads
        so far.
      margin: The factor on this step's bounds a head needs to be tried at
        the next step: ``kernel.safe_heads(peaks, margin)``.
      options: The kernel options of both attempts besides those two, from
        ``stack_kwargs`` (the structured op's under the static-structure
        option).

    Returns:
      ``[T_l, H]`` bf16 rows (the accepted attempt's), ``[L, heads]`` int32
      ``next`` (1: the heads this step's accepted peaks vouch for, outside
      the norm-proven layers) and ``tally`` plus this step's (1 if it was
      recomputed, the heads its accepted attempt ran at a fixed max).
    """
    named = dict(zip(pack.WEIGHT_ORDER, weights))
    unproven = ~kernel.fixed_max_layers(named, config)[:, None]
    tried = ((vouch != 0) & unproven).astype(jnp.int32)
    if axis is not None:
        tried = lax.pmin(tried, axis)
    tried = tried != 0
    options = dict(options, return_peaks=True)

    def attempt(carry):
        index = carry[0]
        fixed = tried & (index == 0)
        out, peaks = kernel.stack(
            x,
            named,
            tables,
            onehot,
            cos,
            sin,
            used[0],
            config=config,
            tiling=tiling,
            axis=axis,
            interpret=interpret,
            **dict(options, fixed_heads=fixed),
        )
        # Row 0 holds each head's (lane's) peaks; the bound takes them over
        # every device's rows.
        peaks = peaks[:, :, :1]
        if axis is not None:
            peaks = lax.pmax(peaks, axis)
        inside = kernel.safe_heads(peaks)[:, : config.heads]
        return index + 1, out, peaks, jnp.all(inside | ~fixed)

    layers = tables.shape[0]
    start = (
        jnp.int32(0),
        jnp.zeros_like(x),
        jnp.zeros((layers, 2, 1, kernel.TABLE_LANES), jnp.float32),
        jnp.bool_(False),
    )
    attempts, out, peaks, _ = lax.while_loop(lambda c: ~c[3], attempt, start)
    redone = attempts - 1
    vouched = kernel.safe_heads(peaks, margin)[:, : config.heads] & unproven
    ran = jnp.where(redone == 0, tried.sum(dtype=jnp.int32), 0)
    step = jnp.stack((redone, ran)).astype(jnp.int32)
    return out, vouched.astype(jnp.int32), tally + step


def partition_specs():
    """(input specs, output spec): rows split over ``AXIS`` for ``x``,
    ``onehot``, ``cos`` and ``sin``; the rest replicated."""
    from jax.sharding import PartitionSpec as P

    rows, replicated = P(AXIS), P()
    return (rows,) * 4 + (replicated,) * (2 + len(pack.WEIGHT_ORDER)), rows


def vouched_partition_specs():
    """(input specs, output specs) of the vouched op: ``partition_specs``'s,
    the heads to try and the tally replicated after the live length; the
    rows, then the vouched heads and the tally, replicated."""
    from jax.sharding import PartitionSpec as P

    rows, replicated = P(AXIS), P()
    return (
        (rows,) * 4 + (replicated,) * (4 + len(pack.WEIGHT_ORDER)),
        (rows, replicated, replicated),
    )


def sharded_stack(
    mesh,
    config: ref.Config,
    tiling: tl.Tiling,
    interpret=False,
    options: dict = STACK_KWARGS,
):
    """``local_stack`` under ``shard_map`` over ``mesh``'s ``"sp"`` axis."""
    in_specs, out_spec = partition_specs()
    return jax.shard_map(
        functools.partial(
            local_stack,
            config=config,
            tiling=tiling,
            axis=AXIS,
            interpret=interpret,
            options=options,
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_spec,
        check_vma=False,
    )


def sharded_vouched_stack(
    mesh,
    config: ref.Config,
    tiling: tl.Tiling,
    margin: float,
    interpret=False,
    options: dict = STACK_KWARGS,
):
    """``local_vouched_stack`` under ``shard_map`` over ``mesh``'s ``"sp"``
    axis."""
    in_specs, out_specs = vouched_partition_specs()
    return jax.shard_map(
        functools.partial(
            local_vouched_stack,
            config=config,
            tiling=tiling,
            axis=AXIS,
            margin=margin,
            interpret=interpret,
            options=options,
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )


def op_name(
    config: ref.Config,
    devices: int,
    structured: bool = False,
    vouched: bool = False,
    levers=frozenset(),
    tiled: bool = False,
    trace: int = 0,
) -> str:
    name = f"sglang_tpu_pallas::minimax_h3_stack_sp{devices}_L{config.layers}"
    name += ("_vouch" if vouched else "") + ("_structured" if structured else "")
    name += "".join(f"_{lever}" for lever in sorted(levers))
    return name + ("_tiled" if tiled else "") + (f"_trace{trace}" if trace else "")


def traced(options: dict, trace: int = 0) -> dict:
    """``options`` with kernel.stack's xprof trace regions at ``trace``
    (profiling only; 0 leaves them, and so the program, unchanged)."""
    return dict(options, trace=trace) if trace else options


def build_stack_op(
    config: ref.Config,
    devices: int,
    tiling: tl.Tiling = STACK_TILING,
    mesh=None,
    structured: bool = False,
    vouch_margin: float | None = None,
    levers=frozenset(),
    tiling_overrides=(),
    trace: int = 0,
):
    """Register the torch op; call on every rank, after distributed init.

    ``mesh`` defaults to the world's Pallas mesh (rank ``i`` at position
    ``i``). ``structured`` (the static-structure option): the op takes the
    static structure as a last, static int argument (``stack_kwargs``), one
    program per value. With ``vouch_margin`` the op is the vouched variant
    (``local_vouched_stack``): it takes the heads to try and the tally after
    the live length and returns the rows, the vouched heads and the tally;
    with ``structured`` too it also takes the static structure last and its
    kernel call takes the structured op's options. ``levers``: the kernel
    levers of the structured calls (``LEVER_SHAPES``' names);
    ``tiling_overrides``: their tiling bounds (``parse_tiling``). ``trace``:
    the kernel's xprof trace regions (``kernel.stack``'s ``trace``; profiling
    only, SGLANG_TPU_MK_TRACE_REGIONS).
    """
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    if mesh is None:
        from torch_tpu._internal.pallas import get_pallas_mesh

        mesh = get_pallas_mesh((AXIS,), (devices,))
    levers = frozenset(levers) if structured else frozenset()
    overrides = tuple(tiling_overrides) if structured else ()
    if vouch_margin is not None:
        return _build_vouched_op(
            config,
            devices,
            tiling,
            mesh,
            vouch_margin,
            structured=structured,
            levers=levers,
            overrides=overrides,
            trace=trace,
        )
    if structured:
        in_specs, out_spec = partition_specs()
        return cpu_export_jax_op(
            op_name(
                config,
                devices,
                structured=True,
                levers=levers,
                tiled=bool(overrides),
                trace=trace,
            ),
            _structured_stack(
                mesh, config, devices, tiling, levers, overrides, trace=trace
            ),
            mesh=mesh,
            # The static structure is a static argument: no partition.
            input_partition_specs=(*in_specs, None),
            output_partition_specs=(out_spec,),
        )
    # One program a row count: each with its shape_tiling.
    sharded = functools.lru_cache(maxsize=None)(
        lambda rows: sharded_stack(
            mesh,
            config,
            shape_tiling(devices, rows, tiling),
            options=traced(STACK_KWARGS, trace),
        )
    )

    def stack(
        x: jax.Array,
        onehot: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        tables: jax.Array,
        used: jax.Array,
        norm1: jax.Array,
        norm2: jax.Array,
        q_norm: jax.Array,
        k_norm: jax.Array,
        wqkv: jax.Array,
        wo: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
    ) -> jax.Array:
        weights = (norm1, norm2, q_norm, k_norm, wqkv, wo, w13, w2)
        program = sharded(x.shape[0] // devices)
        return program(x, onehot, cos, sin, tables, used, *weights)

    in_specs, out_spec = partition_specs()
    return cpu_export_jax_op(
        op_name(config, devices, trace=trace),
        stack,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=(out_spec,),
    )


def _structured_stack(
    mesh,
    config: ref.Config,
    devices: int,
    tiling: tl.Tiling,
    levers=frozenset(),
    overrides=(),
    trace: int = 0,
):
    """The structured op's function: ``build_stack_op``'s plus the static
    structure, a static int (one ``local_stack`` program per value and row
    count)."""
    sharded = functools.lru_cache(maxsize=None)(
        lambda structure, rows: sharded_stack(
            mesh,
            config,
            (
                structured_tiling(devices, rows, tiling, overrides)
                if structure
                else shape_tiling(devices, rows, tiling)
            ),
            options=traced(
                stack_kwargs(
                    devices,
                    structure,
                    rows=rows,
                    levers=levers,
                    tiling=structured_tiling(devices, rows, tiling, overrides),
                    config=config,
                ),
                trace,
            ),
        )
    )

    def stack(
        x: jax.Array,
        onehot: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        tables: jax.Array,
        used: jax.Array,
        norm1: jax.Array,
        norm2: jax.Array,
        q_norm: jax.Array,
        k_norm: jax.Array,
        wqkv: jax.Array,
        wo: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        structure: int,
    ) -> jax.Array:
        weights = (norm1, norm2, q_norm, k_norm, wqkv, wo, w13, w2)
        program = sharded(structure, x.shape[0] // devices)
        return program(x, onehot, cos, sin, tables, used, *weights)

    return stack


def _build_vouched_op(
    config: ref.Config,
    devices: int,
    tiling,
    mesh,
    margin: float,
    structured: bool = False,
    levers=frozenset(),
    overrides=(),
    trace: int = 0,
):
    """``build_stack_op``'s vouched variant; ``structured``: plus the static
    structure, last, as the structured op takes it."""
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    in_specs, out_specs = vouched_partition_specs()
    if structured:
        return cpu_export_jax_op(
            op_name(
                config,
                devices,
                structured=True,
                vouched=True,
                levers=levers,
                tiled=bool(overrides),
                trace=trace,
            ),
            _structured_vouched_stack(
                mesh, config, devices, tiling, margin, levers, overrides, trace=trace
            ),
            mesh=mesh,
            # The static structure is a static argument: no partition.
            input_partition_specs=(*in_specs, None),
            output_partition_specs=out_specs,
        )
    # One program a row count: each with its shape_tiling.
    sharded = functools.lru_cache(maxsize=None)(
        lambda rows: sharded_vouched_stack(
            mesh,
            config,
            shape_tiling(devices, rows, tiling),
            margin,
            options=traced(STACK_KWARGS, trace),
        )
    )

    def stack(
        x: jax.Array,
        onehot: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        tables: jax.Array,
        used: jax.Array,
        vouch: jax.Array,
        tally: jax.Array,
        norm1: jax.Array,
        norm2: jax.Array,
        q_norm: jax.Array,
        k_norm: jax.Array,
        wqkv: jax.Array,
        wo: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        weights = (norm1, norm2, q_norm, k_norm, wqkv, wo, w13, w2)
        program = sharded(x.shape[0] // devices)
        return program(x, onehot, cos, sin, tables, used, vouch, tally, *weights)

    return cpu_export_jax_op(
        op_name(config, devices, vouched=True, trace=trace),
        stack,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
    )


def _structured_vouched_stack(
    mesh,
    config: ref.Config,
    devices: int,
    tiling: tl.Tiling,
    margin: float,
    levers=frozenset(),
    overrides=(),
    trace: int = 0,
):
    """The vouched op's function under the static-structure option: plus the
    static structure, a static int (one ``local_vouched_stack`` program per
    value and row count), whose kernel call takes the structured op's
    options (``stack_kwargs``) besides the tried heads and the peaks."""
    sharded = functools.lru_cache(maxsize=None)(
        lambda structure, rows: sharded_vouched_stack(
            mesh,
            config,
            (
                structured_tiling(devices, rows, tiling, overrides)
                if structure
                else shape_tiling(devices, rows, tiling)
            ),
            margin,
            options=traced(
                stack_kwargs(
                    devices,
                    structure,
                    rows=rows,
                    levers=levers,
                    tiling=structured_tiling(devices, rows, tiling, overrides),
                    config=config,
                ),
                trace,
            ),
        )
    )

    def stack(
        x: jax.Array,
        onehot: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        tables: jax.Array,
        used: jax.Array,
        vouch: jax.Array,
        tally: jax.Array,
        norm1: jax.Array,
        norm2: jax.Array,
        q_norm: jax.Array,
        k_norm: jax.Array,
        wqkv: jax.Array,
        wo: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        structure: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        weights = (norm1, norm2, q_norm, k_norm, wqkv, wo, w13, w2)
        program = sharded(structure, x.shape[0] // devices)
        return program(x, onehot, cos, sin, tables, used, vouch, tally, *weights)

    return stack
