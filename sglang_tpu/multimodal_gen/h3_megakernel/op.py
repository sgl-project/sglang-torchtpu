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
import inspect
import re

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu.multimodal_gen.h3_megakernel import kernel, pack
from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref
from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

AXIS = "sp"
# The kernel's tiling bounds for padding: padded_rows picks the packed row
# count from these.
ROWS_TILING = tl.Tiling(sub_rows=640, stage_slots=24, block_kv_compute_online=1024)
# What the kernel runs with: those bounds plus the MLP tiling (mlp_rows 640,
# chunk_rows 80); stack() cuts them down to divide each row count
# (Tiling.fit). qkv_rows 704: with 24 staging slots a 768-row phase A tile
# does not fit VMEM.
STACK_TILING = dataclasses.replace(
    ROWS_TILING, qkv_rows=704, mlp_rows=640, chunk_rows=80
)
# The call's softmax options, pinned. With fixed and fixed_heads None the
# kernel runs a fixed-max softmax on exactly the layers its own
# fixed_max_layers proves from the packed q/k norm weights (a bound that holds
# for every input) and the online softmax elsewhere: nothing is vouched by
# measured activations, flags files or calibration. No peaks, the default
# fixed max of 17 (the one the norm bound vouches for).
STACK_KWARGS = {
    "fixed": None,
    "fixed_heads": None,
    "return_peaks": False,
    "narrow_tail": False,
    "fixed_max": 17.0,
}
# The static-structure option (SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE)
# runs a static structure at these (devices, rows a device), the row counts it
# was AOT-checked at (VMEM, loop sizes), and the generic walk at any other.
STRUCTURED_SHAPES = frozenset(
    {(8, 4992), (8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)}
)
# Row counts the static-structure option pads to in place of padded_rows':
# 19,456 rows at 2 devices take 2,432-row query tiles, not 512-row ones.
STRUCTURED_ROWS = {(2, 18944): 19456}
# A structured call's own tiling at some shapes (others STACK_TILING): 2 x
# 19,200 rows run 4 x 9,600's 3,200-row query tiles and 640-row phase tiles
# with 2,048-key tiles.
STRUCTURED_TILINGS = {
    (2, 19200): dataclasses.replace(STACK_TILING, block_q=3200, key_tile=2048)
}
# Every call's tiling at shapes STACK_TILING does not cut: 8 x 4,800 rows take
# 3,200-row query tiles that cross a device boundary at their middle.
SHAPE_TILINGS = {(8, 4800): dataclasses.replace(STACK_TILING, split_q_tiles=True)}
# The kernel levers (hook._levers, one flag each) and the shapes each was
# AOT-checked at; the structured calls at those shapes take it, every other call
# runs without it. Each maps to kernel.stack's option of the same idea (see its
# docstring): lean_cut cut_kv_compute, split_items, epilogue_chunks
# epilogue_chunk, wide_kv kv_group 2, wide_kv_online kv_group (1, 2) (the two
# are exclusive), modulate_pieces, prefetch_down (back on at 4 x 9,600, where
# stack_kwargs turns it off), uniform_adaln, zero_ahead, mask_scores,
# cast_first, step_tables, slab_modulate; sum_vouch: sum_bounds, the vouched op
# only. All but sum_vouch keep the bits.
_ALL = frozenset({(8, 4864), (8, 4800), (4, 9600), (2, 19456), (2, 19200)})
LEVER_SHAPES = {
    **{
        lever: _ALL
        for lever in (
            "lean_cut",
            "split_items",
            "modulate_pieces",
            "uniform_adaln",
            "zero_ahead",
            "mask_scores",
            "sum_vouch",
            "cast_first",
            "step_tables",
            "slab_modulate",
        )
    },
    "epilogue_chunks": frozenset({(8, 4864), (8, 4800), (2, 19456)}),
    "wide_kv": frozenset({(8, 4864), (2, 19456), (2, 19200)}),
    "wide_kv_online": frozenset({(8, 4864), (4, 9600), (2, 19456), (2, 19200)}),
    "prefetch_down": frozenset({(4, 9600)}),
}
# sum_vouch: the margin (log2 units) inside kernel.sum_safe_heads' window a
# head's sums need to be tried at a fixed max the next step.
SUM_VOUCH_MARGIN = 8.0
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
        "epilogue_chunk",
        "epilogue_chunk_peaks",
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


def check_levers(levers) -> None:
    """Refuses lever sets no call takes: wide_kv and wide_kv_online are
    exclusive (each sets the call's key groups).

    Raises:
      ValueError: If ``levers`` holds both.
    """
    if {"wide_kv", "wide_kv_online"} <= set(levers):
        raise ValueError(
            "MiniMax H3 megakernel levers wide_kv and wide_kv_online are"
            " exclusive: set SGLANG_TPU_OPT_H3_MEGAKERNEL_WIDE_KV or"
            " SGLANG_TPU_OPT_H3_MEGAKERNEL_WIDE_KV_ONLINE, not both"
        )


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
    down-weight prefetch (VMEM headroom at 9,600 rows a device; the same
    bits). And each of ``levers`` (``LEVER_SHAPES``' names) that covers
    ``rows`` a device; ``tiling`` and ``config``: the call's
    (``modulate_pieces``)."""
    check_levers(levers)
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
        # A tiling's own epilogue_chunk bound (SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING)
        # in place of the default; any chunk computes the same per-row values.
        kwargs["epilogue_chunk"] = (tiling and tiling.epilogue_chunk) or EPILOGUE_CHUNK
    if "wide_kv" in on:
        kwargs["kv_group"] = KV_GROUP
    elif "wide_kv_online" in on:
        kwargs["kv_group"] = (1, KV_GROUP)
    if structure and tiling and tiling.epilogue_chunk_peaks:
        # The TILING knob's chunk for the peaks layers' epilogue (any chunk
        # computes the same per-row values and the same peaks).
        kwargs["epilogue_chunk_peaks"] = tiling.epilogue_chunk_peaks
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
    if "sum_vouch" in on:
        kwargs["sum_bounds"] = True
    if "cast_first" in on:
        kwargs["cast_first"] = True
    if "step_tables" in on:
        kwargs["step_tables"] = True
    if "slab_modulate" in on:
        kwargs["slab_modulate"] = True
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
    # sum_vouch (options' sum_bounds): the peaks are the heads' fixed-max row
    # sums; a head is inside while they stay in kernel.sum_safe_heads' window,
    # and tried next while they stay SUM_VOUCH_MARGIN inside it.
    sums = bool(options.get("sum_bounds"))

    def safe(peaks, next_step=False):
        if sums:
            return kernel.sum_safe_heads(peaks, SUM_VOUCH_MARGIN if next_step else 0.0)
        return kernel.safe_heads(peaks, margin if next_step else 1.0)

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
        inside = safe(peaks)[:, : config.heads]
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
    vouched = safe(peaks, next_step=True)[:, : config.heads] & unproven
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


def call_tiling(
    devices: int, rows: int, structure: int | None, tiling=STACK_TILING, overrides=()
):
    """The tiling of one call at ``rows`` a device: a structured call's
    (``structured_tiling``, with the TILING overrides), else ``shape_tiling``'s.
    The overrides tune the structured shapes they were measured at; the
    generic walk (structure 0 or None) keeps the default tiling."""
    if structure:
        return structured_tiling(devices, rows, tiling, overrides)
    return shape_tiling(devices, rows, tiling)


# The op's tensor arguments, in order; the vouched op takes the tried heads and
# the tally after the live length; a structured op takes the static structure
# (an int, one program per value) last.
_TENSORS = ("x", "onehot", "cos", "sin", "tables", "used")
_VOUCH_TENSORS = ("vouch", "tally")


def _op_signature(vouched: bool, structured: bool) -> inspect.Signature:
    """The exported function's signature: cpu_export_jax_op tells tensors from
    static arguments by its annotations."""
    names = _TENSORS + (_VOUCH_TENSORS if vouched else ()) + pack.WEIGHT_ORDER
    params = [
        inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
        )
        for name in names
    ]
    if structured:
        params.append(
            inspect.Parameter(
                "structure", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
            )
        )
    returns = tuple[jax.Array, jax.Array, jax.Array] if vouched else jax.Array
    return inspect.Signature(params, return_annotation=returns)


def stack_function(
    mesh,
    config: ref.Config,
    devices: int,
    tiling: tl.Tiling = STACK_TILING,
    margin: float | None = None,
    structured: bool = False,
    levers=frozenset(),
    overrides=(),
    trace: int = 0,
):
    """The op's JAX function over ``mesh``: one ``sharded_stack`` program (or
    ``sharded_vouched_stack`` with ``margin``) per row count, and with
    ``structured`` per static structure (its last, static argument); each
    call's tiling from ``call_tiling`` and its options from ``stack_kwargs``."""

    @functools.lru_cache(maxsize=None)
    def program(structure, rows):
        tiling_ = call_tiling(devices, rows, structure, tiling, overrides)
        options = traced(
            stack_kwargs(
                devices,
                structure,
                rows=rows,
                levers=levers,
                tiling=tiling_,
                config=config,
            ),
            trace,
        )
        if margin is None:
            return sharded_stack(mesh, config, tiling_, options=options)
        return sharded_vouched_stack(mesh, config, tiling_, margin, options=options)

    def stack(*args):
        structure = args[-1] if structured else None
        tensors = args[:-1] if structured else args
        return program(structure, tensors[0].shape[0] // devices)(*tensors)

    stack.__signature__ = _op_signature(margin is not None, structured)
    return stack


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
    the live length and returns the rows, the vouched heads and the tally.
    ``levers``: the kernel levers of the structured calls (``LEVER_SHAPES``'
    names); ``tiling_overrides``: their tiling bounds (``parse_tiling``).
    ``trace``: the kernel's xprof trace regions (``kernel.stack``'s
    ``trace``; profiling only, SGLANG_TPU_MK_TRACE_REGIONS).
    """
    # At registration (server start), before any call is traced.
    check_levers(levers)
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    if mesh is None:
        from torch_tpu._internal.pallas import get_pallas_mesh

        mesh = get_pallas_mesh((AXIS,), (devices,))
    levers = frozenset(levers) if structured else frozenset()
    overrides = tuple(tiling_overrides) if structured else ()
    vouched = vouch_margin is not None
    in_specs, out_specs = vouched_partition_specs() if vouched else partition_specs()
    return cpu_export_jax_op(
        op_name(
            config,
            devices,
            structured=structured,
            vouched=vouched,
            levers=levers,
            tiled=bool(overrides),
            trace=trace,
        ),
        stack_function(
            mesh,
            config,
            devices,
            tiling,
            vouch_margin,
            structured=structured,
            levers=levers,
            overrides=overrides,
            trace=trace,
        ),
        mesh=mesh,
        # The static structure is a static argument: no partition.
        input_partition_specs=(*in_specs, None) if structured else in_specs,
        output_partition_specs=out_specs if vouched else (out_specs,),
    )
