# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/mla/sparse/native_sc_gather.py
"""Combined native-SparseCore NOPE/ROPE gather for GLM-5.2.

An optional traced live-row count (``num_live_indices``) bounds the outer
emit_pipeline grid so only the logical batches holding live rows are gathered;
rows beyond are left unwritten (the attention grid never reads them). Without
it every row is gathered.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sglang_tpu.moe.pallas.sparse_core import core_map_helper

NUM_CORES = 2
NUM_SUBCORES = 16
NUM_LANES = 16
ROPE_PACKING = 4
QUADS_PER_ATOM = 16
OCCURRENCES_PER_ATOM = QUADS_PER_ATOM * ROPE_PACKING
ROPE_WORDS = 32  # max words per rope token (128B padded view)


def _delta_swap(x, y, shift, mask):
    tmp = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(x, shift), y), mask)
    return (
        jnp.bitwise_xor(x, jnp.left_shift(tmp, shift)),
        jnp.bitwise_xor(y, tmp),
    )


def transpose_quad(data: list[Any]) -> list[Any]:
    """Four tokens' banded words -> four bands of token-interleaved words.

    In: `data[t]` word `w` byte `b` is token `t`'s byte `b * n + w` (the
    `pack_tokens` banding, `n` words per token: 32 padded, 16 packed). Out:
    `data[b]` word `w` byte `t` is that same byte, which bitcasts to the
    four-token uint8 tile the TensorCore consumer reads. Pure jnp, so tests
    run it on CPU too.
    """
    data = list(data)
    low_16 = jnp.uint32(0x0000FFFF)
    low_byte_each_half = jnp.uint32(0x00FF00FF)
    data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
    data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
    data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
    data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
    return data


def _kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    live_vmem_ref: Any = None,
    live_sem_ref: Any = None,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    wait_mode: str,
    serve_pipelined: bool,
    live_hbm_ref: Any = None,
    guard_tail: bool = False,
    rope_words: int = ROPE_WORDS,
    phase_stride: int = 0,
    query_batch_shift: int = 0,
    compact_rope_output: bool = False,
    rows_per_dma: int = NUM_LANES,
):
    streams = OCCURRENCES_PER_ATOM // rows_per_dma
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    live_batches = None
    if live_hbm_ref is not None:
        # The live logical-batch count arrives as a broadcast i32[NUM_LANES]
        # vector. On SparseCore a scalar load is legal only from SMEM and SMEM
        # only at rank 0 (jax/_src/pallas/mosaic/sc_lowering.py), so the count is
        # DMA'd into a VMEM vector and lane 0 extracted -- the same shape as the
        # in-repo dsa_gather.py:90 (`valid_ref[pl.ds(0, 16)][0]`).
        cp = pltpu.make_async_copy(live_hbm_ref, live_vmem_ref, live_sem_ref.at[0])
        cp.start()
        cp.wait()
        # Lanes [0,16) broadcast the live logical-batch count, lanes [16,32)
        # the per-owner trip count; only lane 0 of a vector is ever extracted.
        live_batches = live_vmem_ref[pl.ds(0, NUM_LANES)][0]

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners

        def scheduled_atom(atom):
            # Stagger owners serving different queries without rearranging indices
            # in HBM. Apply the same permutation to input and output addresses.
            if not phase_stride:
                return atom
            query = logical_batch >> query_batch_shift
            return (atom + query * phase_stride) & (atoms_per_batch - 1)

        def values(atom, stream):
            if rows_per_dma == NUM_LANES:
                return indices_ref[0, scheduled_atom(atom), stream, :]
            # Give the indirect DMA a rank-1 VMEM ref, so its offsets can
            # span more than one SparseCore SIMD register. The outer pipeline
            # owns this buffer until all DMAs for this batch have drained.
            return indices_ref.at[
                0, scheduled_atom(atom), pl.ds(stream * rows_per_dma, rows_per_dma)
            ]

        def destinations(step, atom, stream):
            occurrence_base = logical_batch * atoms_per_batch * OCCURRENCES_PER_ATOM
            return pl.ds(
                occurrence_base + atom * OCCURRENCES_PER_ATOM + stream * rows_per_dma,
                rows_per_dma,
            )

        def body(*refs):
            step = pl.program_id(0)
            atom = scheduled_atom(step)
            nope_inputs = refs[:streams]
            rope_inputs = refs[streams : 2 * streams]
            rope_output = refs[-1]

            nope_copies = []
            for stream in range(streams):
                copy = pltpu.make_async_copy(
                    nope_inputs[stream],
                    nope_out_hbm_ref.at[destinations(step, atom, stream), :],
                    sem_ref.at[stream],
                )
                copy.start()
                nope_copies.append(copy)
            if wait_mode == "before_rope":
                for copy in nope_copies:
                    copy.wait()

            for row in range(QUADS_PER_ATOM):
                data = []
                for token_slot in range(ROPE_PACKING):
                    logical_rank = row * ROPE_PACKING + token_slot
                    input_stream, input_lane = divmod(logical_rank, rows_per_dma)
                    data.append(
                        rope_inputs[input_stream][
                            pl.ds(input_lane, 1), pl.ds(0, rope_words)
                        ]
                    )

                data = transpose_quad(data)
                # Pair token groups [0:32] and [32:64] within the same atom.
                # Each original quad still uses its usual four-token bit transpose.
                out_row = row % (QUADS_PER_ATOM // 2) if compact_rope_output else row
                out_col = (
                    64 * (row // (QUADS_PER_ATOM // 2)) if compact_rope_output else 0
                )
                for band in range(ROPE_PACKING):
                    rope_output[
                        pl.ds(out_row, 1),
                        pl.ds(out_col + band * rope_words, rope_words),
                    ] = data[band]
                if compact_rope_output:
                    continue
                # Packed rope fills only the first ROPE_PACKING * rope_words
                # lanes; zero the rest, which the consumer reads as the
                # (zero-padded) rope tail, as the padded cache stores it.
                # Garbage there could decode to fp8 NaN, and NaN * 0 poisons
                # the score against q's zero pad.
                for band in range(ROPE_PACKING, 128 // rope_words):
                    rope_output[pl.ds(row, 1), pl.ds(band * rope_words, rope_words)] = (
                        jnp.zeros((1, rope_words), jnp.uint32)
                    )

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(rows_per_dma), 128),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(streams)
        )
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(rows_per_dma), rope_words),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(streams)
        )
        rope_out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM // (2 if compact_rope_output else 1), 128),
            lambda atom: (
                logical_batch * atoms_per_batch + scheduled_atom(atom),
                0,
            ),
        )

        def run_inner():
            pltpu.emit_pipeline(
                body,
                grid=(atoms_per_batch,),
                in_specs=nope_specs + rope_specs,
                out_specs=(rope_out_spec,),
                no_pipelining=not serve_pipelined,
            )(
                *([nope_hbm_ref] * streams),
                *([rope_hbm_ref] * streams),
                rope_out_hbm_ref,
            )

        if live_batches is not None and guard_tail:
            # Owners share a uniform traced grid of cdiv(live, num_owners);
            # the ragged last round is skipped per owner.
            pl.when(logical_batch < live_batches)(run_inner)
        else:
            run_inner()

    if rows_per_dma == NUM_LANES:
        indices_spec = pl.BlockSpec(
            (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
            lambda batch: (owner + batch * num_owners, 0, 0, 0),
        )
    else:
        indices_spec = pl.BlockSpec(
            (1, atoms_per_batch, OCCURRENCES_PER_ATOM),
            lambda batch: (owner + batch * num_owners, 0, 0),
        )
    if live_batches is None:
        pltpu.emit_pipeline(
            serve_batch,
            grid=(batches_per_owner,),
            in_specs=(indices_spec,),
        )(indices_hbm_ref)
    else:
        # Traced grid length for emit_pipeline on a VectorSubcoreMesh, and
        # pl.when around it for the zero-trip case -- both lower on v7x SC.
        # No upstream SparseCore test traces an emit_pipeline grid (the
        # tpu_sparsecore_pallas_test.py grids are static); the nearest
        # upstream form, tests/pallas/tpu_pallas_pipeline_test.py:2420-2452
        # @ce57049, is a TensorCore test. This SC form is measured in-repo.
        # Uniform per-owner trip count; no scalar division on SC: the wrapper
        # sends live_batches as a multiple-of-num_owners-rounded pair.
        owner_batches = live_vmem_ref[pl.ds(NUM_LANES, NUM_LANES)][0]

        @pl.when(owner_batches > 0)
        def _serve_live():
            pltpu.emit_pipeline(
                serve_batch,
                grid=(owner_batches,),
                in_specs=(indices_spec,),
            )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
        "live_multiple",
        "phase_stride",
        "compact_rope_output",
        "rows_per_dma",
    ),
)
def dsa_gather_native_sc(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    *,
    out_size: int,
    atoms_per_batch: int = 16,
    wait_mode: str = "overlap_rope",
    serve_pipelined: bool = True,
    num_live_indices: jax.Array | None = None,
    live_multiple: int = 1,
    phase_stride: int = 0,
    compact_rope_output: bool = False,
    rows_per_dma: int = NUM_LANES,
) -> tuple[jax.Array, jax.Array]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` packs
    ``128 // rope_words`` tokens into each 128-word row (e.g. eight 16-word
    64B tokens or four 32-word 128B tokens), with ``rope_words`` derived from
    the cache shapes. The returned ROPE is register-transposed into the grouped
    TensorCore format expected by the shared attention consumer, with any lanes
    past ``4 * rope_words`` zeroed.

    ``num_live_indices`` (traced i32 scalar, optional) gathers only the logical
    batches covering the first ``num_live_indices`` rows; later rows are left
    unwritten. ``live_multiple`` is a static promise that ``num_live_indices``
    is a multiple of it; when that multiple covers whole owner rounds the
    per-owner tail guard is dropped.

    A nonzero ``phase_stride`` rotates each query's atom issue order, retaining
    the original output order for both caches. The experiment requires a power
    of two atoms per owner batch and a power of two owner batches per query.
    """
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if rows_per_dma not in (16, 32, 64):
        raise ValueError("rows_per_dma must be 16, 32, or 64")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError("native-SC ROPE must be banded uint32[...,128]")
    if indices.dtype != jnp.int32 or indices.ndim != 2:
        raise ValueError("indices must be int32[query, topk]")
    if indices.size != out_size:
        raise ValueError(f"indices has {indices.size} values, expected {out_size}")

    nope_rows_view = nope_cache.reshape(-1, 128)
    num_tokens = nope_rows_view.shape[0]
    if num_tokens <= 0 or rope_cache.size % num_tokens != 0:
        raise ValueError(
            f"ROPE cache ({rope_cache.size} words) must divide evenly across "
            f"NOPE tokens ({num_tokens})"
        )
    rope_words = rope_cache.size // num_tokens
    if compact_rope_output and (
        rope_words != 16 or indices.shape[-1] % OCCURRENCES_PER_ATOM
    ):
        raise ValueError(
            "compact RoPE output requires 64-byte RoPE and full 64-occurrence atoms per query"
        )
    if (
        rope_words <= 0
        or rope_words % NUM_LANES != 0
        or ROPE_PACKING * rope_words > 128
        or 128 % rope_words != 0
    ):
        raise ValueError(
            f"derived rope_words ({rope_words}) must be a positive multiple of "
            f"{NUM_LANES} dividing 128 with {ROPE_PACKING} * rope_words <= 128"
        )
    rope_rows_view = rope_cache.reshape(-1, rope_words)

    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = sc_info.num_subcores if sc_info is not None else NUM_SUBCORES
    num_owners = num_cores * num_subcores

    max_possible_atoms = out_size // (num_owners * OCCURRENCES_PER_ATOM)
    if atoms_per_batch > max_possible_atoms > 0:
        atoms_per_batch = max_possible_atoms
    if atoms_per_batch <= 0:
        raise ValueError(
            f"atoms_per_batch ({atoms_per_batch}) must be > 0 "
            f"(out_size={out_size}, num_owners={num_owners})"
        )

    occurrences_per_batch = atoms_per_batch * OCCURRENCES_PER_ATOM
    query_batch_shift = 0
    if phase_stride:
        batches_per_query, tail = divmod(indices.shape[1], occurrences_per_batch)
        if (
            phase_stride < 0
            or atoms_per_batch & (atoms_per_batch - 1)
            or tail
            or batches_per_query < 1
            or batches_per_query & (batches_per_query - 1)
        ):
            raise ValueError(
                "phase rotation requires a nonnegative stride, power-of-two "
                "atoms and full power-of-two batches per query"
            )
        query_batch_shift = batches_per_query.bit_length() - 1
    owner_batch_size = num_owners * occurrences_per_batch
    if out_size % owner_batch_size != 0:
        raise ValueError(
            f"out_size ({out_size}) must be divisible by num_owners * "
            f"occurrences_per_batch ({owner_batch_size})"
        )
    num_batches = out_size // occurrences_per_batch
    batches_per_owner = num_batches // num_owners
    live = num_live_indices is not None
    streams = OCCURRENCES_PER_ATOM // rows_per_dma
    index_shape = (
        (ROPE_PACKING, NUM_LANES)
        if rows_per_dma == NUM_LANES
        else (OCCURRENCES_PER_ATOM,)
    )
    scratch_types = {"sem_ref": pltpu.SemaphoreType.DMA((streams,))}
    index_input = indices.reshape(num_batches, atoms_per_batch, *index_shape)
    operands = [nope_rows_view, rope_rows_view, index_input]
    guard_tail = False
    if live:
        # Live logical batches = cdiv(num_live_indices, occurrences/batch),
        # clamped to the static count; per-owner trip = cdiv(live, num_owners).
        live_b = jnp.clip(
            (jnp.asarray(num_live_indices, jnp.int32) + occurrences_per_batch - 1)
            // occurrences_per_batch,
            0,
            num_batches,
        )
        owner_b = (live_b + num_owners - 1) // num_owners
        # Lanes [0,16) = live logical batches, [16,32) = per-owner grid length.
        live_vec = jnp.concatenate(
            [
                jnp.full((NUM_LANES,), live_b, jnp.int32),
                jnp.full((NUM_LANES,), owner_b, jnp.int32),
            ]
        )
        operands.append(live_vec)
        scratch_types["live_vmem_ref"] = pltpu.VMEM((2 * NUM_LANES,), jnp.int32)
        scratch_types["live_sem_ref"] = pltpu.SemaphoreType.DMA((1,))
        # Per-owner guard is needed only if a live count can leave a ragged
        # last round, i.e. unless the caller promises num_live_indices is a
        # multiple of num_owners * occurrences_per_batch (at topk=2048,
        # atoms=1: 32 * 64 = 2048 = one token, so decode needs no guard).
        guard_tail = live_multiple % owner_batch_size != 0
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    body = functools.partial(
        _kernel,
        core_axis_name=mesh.core_axis_name,
        subcore_axis_name=mesh.subcore_axis_name,
        num_owners=num_owners,
        atoms_per_batch=atoms_per_batch,
        batches_per_owner=batches_per_owner,
        wait_mode=wait_mode,
        serve_pipelined=serve_pipelined,
        guard_tail=guard_tail,
        rope_words=rope_words,
        phase_stride=phase_stride,
        query_batch_shift=query_batch_shift,
        compact_rope_output=compact_rope_output,
        rows_per_dma=rows_per_dma,
    )
    if live:
        inner = body

        def body(nope, rope, idx, live_hbm, nope_o, rope_o, **scr):
            return inner(nope, rope, idx, nope_o, rope_o, live_hbm_ref=live_hbm, **scr)

    return core_map_helper.kernel(
        body,
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct(
                (out_size // (ROPE_PACKING * (2 if compact_rope_output else 1)), 128),
                jnp.uint32,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=(
            f"sc_native_sc_gather_a{atoms_per_batch}_rw{rope_words}_{wait_mode}_"
            f"{'pipe' if serve_pipelined else 'nopipe'}{'_live' if live else ''}"
            + (f"_phase{phase_stride}" if phase_stride else "")
            + ("_compactrope" if compact_rope_output else "")
            + (f"_rows{rows_per_dma}" if rows_per_dma != NUM_LANES else "")
        ),
    )(*operands)
