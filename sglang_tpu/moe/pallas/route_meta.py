"""Pallas scalar kernel for bs=1 decode MoE routing metadata.

With one real token every routed row is the same activation, so the permute is a
broadcast and only the expert-major metadata is needed. XLA emits that (plus the
pad / mask / local-id glue) as ~10 small ops; this is one SMEM counting sort.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _route_meta_kernel(
    ids_ref,
    w_ref,
    offset_ref,
    ws_ref,
    gs_ref,
    cur_ref,
    *,
    k: int,
    n: int,
    num_experts: int,
) -> None:
    def zero_rows(r, carry):
        ws_ref[r] = 0.0
        return carry

    def zero_groups(e, carry):
        gs_ref[e] = 0
        return carry

    lax.fori_loop(0, n, zero_rows, 0)
    lax.fori_loop(0, num_experts, zero_groups, 0)
    offset = offset_ref[0]

    def local_id(i):
        e = ids_ref[0, i] - offset
        return e, (e >= 0) & (e < num_experts)

    def count(i, carry):
        e, ok = local_id(i)

        @pl.when(ok)
        def _():
            gs_ref[e] = gs_ref[e] + 1

        return carry

    lax.fori_loop(0, k, count, 0, unroll=True)

    def prefix(e, acc):
        cur_ref[e] = acc
        return acc + gs_ref[e]

    lax.fori_loop(0, num_experts, prefix, 0)

    # Slot order within a group keeps the sort stable.
    def place(i, carry):
        e, ok = local_id(i)

        @pl.when(ok)
        def _():
            pos = cur_ref[e]
            ws_ref[pos] = w_ref[0, i]
            cur_ref[e] = pos + 1

        return carry

    lax.fori_loop(0, k, place, 0, unroll=True)


@functools.partial(jax.jit, static_argnames=("n", "num_experts"))
def single_token_route_meta(
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    *,
    n: int,
    offset: int | jax.Array,
    num_experts: int,
) -> tuple[jax.Array, jax.Array]:
    """Expert-major metadata for the routed rows of one token.

    ``topk_ids`` / ``topk_weights`` are the token's [1, k] global ids and weights;
    ids in ``[offset, offset + num_experts)`` are local, every other id
    (non-local, or TopK's -1 for a padded token) is dropped. Returns f32[n]
    weights with the local rows as a sorted prefix (zeros after) and
    i32[num_experts] group sizes.
    """
    k = topk_ids.shape[1]
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    offset_arr = jnp.asarray(offset, dtype=jnp.int32).reshape(1)
    return pl.pallas_call(
        functools.partial(_route_meta_kernel, k=k, n=n, num_experts=num_experts),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((num_experts,), jnp.int32),
        ),
        in_specs=[smem, smem, smem],
        out_specs=(smem, smem),
        scratch_shapes=[pltpu.SMEM((num_experts,), jnp.int32)],
        name=f"moe_route_meta_{k}_{n}_{num_experts}",
    )(topk_ids.astype(jnp.int32), topk_weights.astype(jnp.float32), offset_arr)
