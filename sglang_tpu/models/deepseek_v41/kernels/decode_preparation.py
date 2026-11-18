# SPDX-License-Identifier: Apache-2.0
"""Experimental decode preparation boundaries for measured decode buckets.

These functions keep communication and SparseCore producers outside the
TensorCore preparation call. Callers retain their general-shape paths.

Every fused region asks `admit` before taking its fused path. Token buckets
below are the counts whose fused lowering and native parity were measured on
v7; layout contracts are checked by the region and passed in as a problem
string. A miss takes the native path.
"""

import logging

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

logger = logging.getLogger(__name__)

# Measured buckets per region. A count above a region's largest bucket is a
# prefill-sized call and falls back silently.
SWA_PACK_TOKENS = frozenset({96, 192, 288})
# (tokens, requests): both regions build token x request matrices.
SWA_BATCHES = frozenset({(96, 16), (192, 32), (288, 32), (288, 48)})
MAIN_RECORD_BATCHES = SWA_BATCHES
# The main-record kernel's one-hot gather costs O(T * requests * columns);
# smaller page tables than the largest measured one are strictly cheaper.
MAIN_RECORD_MAX_COLUMNS = 772

_reported: set[tuple[str, str]] = set()


def admit(
    region: str,
    problem: str | None = None,
    *,
    tokens: int | tuple[int, ...] | None = None,
    buckets: frozenset | None = None,
    backend: bool = True,
    report: bool = True,
) -> bool:
    """Whether ``region`` takes its fused path.

    ``problem`` names a violated layout contract. ``tokens`` is checked
    against ``buckets``; any hashable bucket key works, with ``tokens`` an
    int or a tuple whose first item is the token count. ``backend`` requires
    a TPU. A decode-sized miss is logged once per reason. ``report=False``
    marks a region whose miss takes another tuned path, so it does not log.
    """
    if backend and jax.default_backend() != "tpu":
        return False  # interpret/CPU runs always take the native path
    if problem is None and buckets is not None and tokens not in buckets:
        count = tokens[0] if isinstance(tokens, tuple) else tokens
        largest = max(b[0] if isinstance(b, tuple) else b for b in buckets)
        if count is not None and count > largest:
            return False
        problem = f"{tokens} is not a measured bucket {sorted(buckets)}"
    if problem is None:
        return True
    if not report:
        return False
    if (region, problem) not in _reported:
        _reported.add((region, problem))
        logger.warning("DSv4.1 decode fusion '%s' falls back: %s", region, problem)
    return False


def first_problem(*checks: tuple[bool, str]) -> str | None:
    """Description of the first failed ``(ok, description)`` layout check."""
    return next((description for ok, description in checks if not ok), None)


def take_along(values, indices, axis):
    """Explicit batched gather; retain singleton dimensions for Mosaic."""
    batch = tuple(i for i in range(values.ndim) if i != axis)
    return lax.gather(
        values,
        indices[..., None],
        lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(axis,),
            start_index_map=(axis,),
            operand_batching_dims=batch,
            start_indices_batching_dims=batch,
        ),
        slice_sizes=(1,) * values.ndim,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def fuse_function(fn, args, *, name, interpret=False):
    """Real Pallas boundary, with the same externally visible array outputs."""
    out_shape = jax.eval_shape(fn, *args)
    leaves, tree = jax.tree.flatten(out_shape)

    def kernel(*refs):
        outputs = fn(*(r[...] for r in refs[: len(args)]))
        for ref, value in zip(refs[len(args) :], jax.tree.leaves(outputs)):
            ref[...] = value

    call = pl.pallas_call(
        kernel,
        out_shape=tuple(leaves),
        name=name,
        interpret=interpret,
        compiler_params=None if interpret else _half_vmem(),
    )
    return lambda *xs: jax.tree.unflatten(tree, call(*xs))


def _half_vmem() -> pltpu.CompilerParams:
    return pltpu.CompilerParams(
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2
    )


def pack_matrix(records, values, *, base, capacity, padded):
    n = values.shape[0]
    local = jnp.where(
        (max(base, 64) <= records.T) & (base + capacity > records.T),
        records.T - base,
        capacity,
    )
    idx = jnp.arange(n, dtype=jnp.int32)[:, None]
    ranks = jnp.sum(
        (local > local.T) | ((local == local.T) & (idx > idx.T)), axis=1, keepdims=True
    )
    permutation = idx == ranks.T
    ids = jnp.sum(jnp.where(permutation, local.T, 0), axis=1, keepdims=True)
    next_ids = jnp.concatenate((ids[1:], jnp.full((1, 1), capacity, jnp.int32)), axis=0)
    pairs = ids // 2
    boundary = (ids < capacity) & (
        jnp.concatenate(
            (jnp.ones((1, 1), jnp.int32), (pairs[1:] != pairs[:-1]).astype(jnp.int32)),
            axis=0,
        )
        != 0
    )
    even = ids % 2 == 0
    odd = (~even) | (next_ids // 2 == pairs)
    value_bits = lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.uint32)
    bits = jnp.dot(
        permutation.astype(jnp.float32),
        value_bits.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
    ).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)), axis=0)
    low = jnp.where(even, bits, jnp.uint32(0))
    high = jnp.where(odd, jnp.where(even, next_bits, bits), jnp.uint32(0))
    words = lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        odd, jnp.int32(-65536), jnp.int32(0)
    )
    return (
        jnp.pad(
            jnp.where(boundary, pairs, -1),
            ((0, padded - n), (0, 0)),
            constant_values=-1,
        ),
        jnp.broadcast_to(jnp.pad(masks, ((0, padded - n), (0, 0))), (padded, 512)),
        jnp.pad(words, ((0, padded - n), (0, 0))),
    )


def main_records_call(xs, *, page_size):
    selected, pages, starts = xs
    t, k = selected.shape
    requests, pps = pages.shape
    padded_pps = (pps + 127) // 128 * 128

    def body(sel, table, boundaries):
        token = jnp.arange(t, dtype=jnp.int32)
        req = jnp.sum(token[:, None] >= boundaries[None, 1:], axis=1)
        req = jnp.minimum(req, requests - 1)
        onehot = (req[:, None] == jnp.arange(requests)[None, :]).astype(jnp.bfloat16)
        table = jnp.pad(table, ((0, 0), (0, padded_pps - pps)))
        window = jnp.zeros((t, padded_pps), jnp.int32)
        # Four exact byte limbs preserve every INT32 page id, including -1.
        # 4*2*T*requests*padded_pps = 56.6 MFLOP for this shape.
        for shift in (0, 8, 16, 24):
            limb = ((table >> shift) & 255).astype(jnp.bfloat16)
            gathered = jnp.dot(onehot, limb, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            window = window | (gathered << shift)
        valid = (sel >= 0) & (token[:, None] < boundaries[-1])
        logical = jnp.maximum(sel, 0)
        page = jnp.clip(logical // page_size, 0, pps - 1)
        physical = jnp.zeros_like(page)
        for c in range(padded_pps // 128):
            local = page - c * 128
            chunk = window[:, c * 128 : (c + 1) * 128]
            value = take_along(chunk, jnp.clip(local, 0, 127), axis=1)
            physical = jnp.where((local >= 0) & (local < 128), value, physical)
        records = jnp.where(valid, physical * page_size + logical % page_size, -1)
        return records, jnp.sum(valid, axis=1, keepdims=True, dtype=jnp.int32).T

    fused = fuse_function(body, xs, name="dsv41_main_records_matrix")

    def call(*xs):
        records, counts = fused(*xs)
        return records, counts.reshape(t)

    return call


def _swa_capacities(banks):
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes_cores = info.num_lanes * info.num_cores * info.num_subcores
    return tuple(bank.shape[0] * bank.shape[1] for bank in banks), lanes_cores


def swa_pack(banks, records, values):
    capacities, lanes_cores = _swa_capacities(banks)
    padded = (values.shape[0] + lanes_cores - 1) // lanes_cores * lanes_cores

    def body(records, values):
        result, base = [], 0
        for capacity in capacities:
            result.append(
                pack_matrix(
                    records, values, base=base, capacity=capacity, padded=padded
                )
            )
            base += capacity
        return tuple(result)

    prepared = fuse_function(
        body, (records[None, :], values), name="decode_swa_sort_pack"
    )(records[None, :], values)
    return tuple((p[0].reshape(-1), *p[1:]) for p in prepared)
