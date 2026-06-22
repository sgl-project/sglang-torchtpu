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
"""Pallas DMA gather/scatter kernels for TPU HiCache KV offload.

* gather (D2H): ``src_cache[block_ids[i]]`` -> contiguous output rows, a bulk
  HBM staging buffer for a single ``copy_`` to host.
* scatter (H2D): ``src_blocks[i]`` -> ``dest_cache[block_ids[i]]`` in place
  (``input_output_aliases``).

Both are HBM<->HBM only; the host<->HBM PCIe leg is a torch ``copy_`` in the
host pool. Block ids are scalar-prefetched (no recompile on id change) and the
last dim is padded to 128 for TPU DMA alignment.

Overlapped (layer-wise) usage: the H2D scatter writes the live ``kv_cache`` and
is issued WITHOUT a blocking sync, so the forward's read of that same tensor is
the dependency XLA schedules the DMA against.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas as tpu_pallas
from torch_tpu._internal.sync import synchronize as _tpu_sync

_DMA_ALIGNMENT = 128
MAX_H2D_PREWARM_BLOCKS = 2048
MAX_D2H_PREWARM_BLOCKS = 64


# ---------------------------------------------------------------------------
# Gather kernel (D2H)
# ---------------------------------------------------------------------------


def _gather_kernel(num_blocks_ref, block_ids_ref, src_ref, dst_ref_out, sem):
    n = num_blocks_ref[0]

    @pl.loop(0, n)
    def _start(i):
        src_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(src_idx, 1)],
            dst_ref_out.at[pl.ds(i, 1)],
            sem,
        ).start()

    @pl.loop(0, n)
    def _wait(i):
        src_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(src_idx, 1)],
            dst_ref_out.at[pl.ds(i, 1)],
            sem,
        ).wait()


def pallas_kv_gather(
    src_cache: jax.Array,
    block_ids: jax.Array,
    *,
    num_blocks: int | None = None,
) -> jax.Array:
    """Gather ``src_cache[block_ids[i]]`` -> ``output[i]`` for i in [0, num_blocks)."""
    if num_blocks is None:
        num_blocks = block_ids.shape[0]
    if src_cache.ndim < 2:
        raise ValueError(f"src_cache must be at least 2-D, got {src_cache.shape}")

    orig_last = src_cache.shape[-1]
    needs_pad = orig_last % _DMA_ALIGNMENT != 0
    if needs_pad:
        pad_amount = _DMA_ALIGNMENT - (orig_last % _DMA_ALIGNMENT)
        pad_widths = [(0, 0)] * (src_cache.ndim - 1) + [(0, pad_amount)]
        src_cache = jnp.pad(src_cache, pad_widths)

    max_blocks = block_ids.shape[0]
    num_blocks_arr = jnp.array([num_blocks], dtype=jnp.int32)
    block_ids = block_ids.astype(jnp.int32)
    if block_ids.shape[0] < max_blocks:
        block_ids = jnp.pad(
            block_ids, (0, max_blocks - block_ids.shape[0]), constant_values=0
        )

    block_shape = src_cache.shape[1:]
    dst_shape = (max_blocks,) + block_shape

    result = pl.pallas_call(
        _gather_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(dst_shape, src_cache.dtype),
    )(num_blocks_arr, block_ids, src_cache)

    if needs_pad:
        result = result[..., :orig_last]
    return result


# ---------------------------------------------------------------------------
# Scatter kernel (H2D)
# ---------------------------------------------------------------------------


def _scatter_kernel(
    num_blocks_ref, block_ids_ref, src_ref, dest_ref, dest_ref_out, sem
):
    n = num_blocks_ref[0]

    @pl.loop(0, n)
    def _start(i):
        dst_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(i, 1)],
            dest_ref.at[pl.ds(dst_idx, 1)],
            sem,
        ).start()

    @pl.loop(0, n)
    def _wait(i):
        dst_idx = block_ids_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(i, 1)],
            dest_ref.at[pl.ds(dst_idx, 1)],
            sem,
        ).wait()


def pallas_kv_scatter(
    dest_cache: jax.Array,
    src_blocks: jax.Array,
    block_ids: jax.Array,
    *,
    num_blocks: int | None = None,
) -> jax.Array:
    """In-place scatter ``src_blocks[i]`` -> ``dest_cache[block_ids[i]]``."""
    if num_blocks is None:
        num_blocks = src_blocks.shape[0]
    if dest_cache.ndim < 2:
        raise ValueError(f"dest_cache must be at least 2-D, got {dest_cache.shape}")
    if src_blocks.shape[1:] != dest_cache.shape[1:]:
        raise ValueError(
            f"Trailing dim mismatch: src {src_blocks.shape[1:]} "
            f"vs dest {dest_cache.shape[1:]}"
        )

    orig_last = dest_cache.shape[-1]
    needs_pad = orig_last % _DMA_ALIGNMENT != 0
    if needs_pad:
        pad_amount = _DMA_ALIGNMENT - (orig_last % _DMA_ALIGNMENT)
        pad_widths = [(0, 0)] * (dest_cache.ndim - 1) + [(0, pad_amount)]
        dest_cache = jnp.pad(dest_cache, pad_widths)
        src_blocks = jnp.pad(src_blocks, pad_widths)

    num_blocks_arr = jnp.array([num_blocks], dtype=jnp.int32)
    block_ids = block_ids.astype(jnp.int32)
    max_blocks = src_blocks.shape[0]
    if block_ids.shape[0] < max_blocks:
        block_ids = jnp.pad(
            block_ids, (0, max_blocks - block_ids.shape[0]), constant_values=0
        )

    dest_result = pl.pallas_call(
        _scatter_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # src_blocks
                pl.BlockSpec(
                    memory_space=pltpu.HBM
                ),  # dest_cache (aliased, written in place)
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(dest_cache.shape, dest_cache.dtype),
        input_output_aliases={3: 0},  # dest_cache (arg idx 3) -> output 0
    )(num_blocks_arr, block_ids, src_blocks, dest_cache)

    if needs_pad:
        dest_result = dest_result[..., :orig_last]
    return dest_result


# ---------------------------------------------------------------------------
# Torch <-> JAX bridges (compiled once per process)
# ---------------------------------------------------------------------------

_gather_fn = tpu_pallas.custom_jax_kernel(pallas_kv_gather)
_scatter_fn = tpu_pallas.custom_jax_kernel(pallas_kv_scatter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pad_to_pow2(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return ``(padded_ids, original_len)`` padded to the next power of two.

    Padding repeats the last real id so scattered writes stay idempotent (they
    just rewrite the last real block). Callers trim with ``original_len``.
    """
    n = len(ids)
    if n == 0:
        return ids, 0
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return ids, n
    padded = np.empty(p, dtype=ids.dtype)
    padded[:n] = ids
    padded[n:] = ids[n - 1]
    return padded, n


def _pow2_buckets(max_n: int) -> list[int]:
    buckets, p = [], 1
    while p <= max_n:
        buckets.append(p)
        p <<= 1
    return buckets


def prewarm_offload_kernels(
    kv_buffer_list: list[torch.Tensor],
    device: str,
    max_h2d: int = MAX_H2D_PREWARM_BLOCKS,
    max_d2h: int = MAX_D2H_PREWARM_BLOCKS,
) -> None:
    """Pre-compile the gather/scatter HLOs for power-of-2 block counts.

    First-call JIT tracing holds the GIL and would stall the scheduler mid-serving.
    HLO is shape-keyed, so one layer's buffer suffices; scatter prewarm only
    touches the dummy (loc==0) page.
    """
    if not kv_buffer_list:
        return
    kv0 = kv_buffer_list[0]
    num_blocks = kv0.shape[0]
    block_shape = tuple(kv0.shape[1:])

    for p in _pow2_buckets(min(num_blocks, max_h2d)):
        ids = torch.zeros(p, dtype=torch.int32, device=device)
        src = torch.zeros((p,) + block_shape, dtype=kv0.dtype, device=device)
        kv0[:] = _scatter_fn(kv0, src, ids)
        _tpu_sync([kv0], wait=True)
        del ids, src

    for p in _pow2_buckets(min(num_blocks, max_d2h)):
        ids = torch.zeros(p, dtype=torch.int32, device=device)
        gathered = _gather_fn(kv0, ids)
        _tpu_sync([gathered], wait=True)
        del ids, gathered
