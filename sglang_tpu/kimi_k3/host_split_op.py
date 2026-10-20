# SPDX-License-Identifier: Apache-2.0
"""Torch ops for the Kimi-K3 host-split GEMV (sglang_tpu/kernels/host_split_gemv.py).

The static fp8 binder (static_fp8.py) asks for one op per split shape at bind
time, before torch.compile traces the model.
"""

# PEP 563 must stay off here: cpu_export_jax_op reads the wrappers' jax.Array
# annotations, and ``from __future__ import annotations`` turns them into
# strings it rejects (static_fp8.py has it, so the op build cannot live there).

import functools


def _build(world: int, local: int, cols: int, used: int, used_e: int):
    """The host-split GEMV for one split shape, as a torch op over the TP mesh."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.kernels import host_split_gemv
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("model",), (world,))
    spec = P("model")
    body = functools.partial(
        host_split_gemv.host_split_gemv_local,
        local=local,
        cols=cols,
        used=used,
        used_e=used_e,
    )
    if used_e:
        sharded = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec,) * 3,
            check_vma=False,
        )

        def _wrapped(
            x: jax.Array,
            w: jax.Array,
            out_scale: jax.Array,
            inv_x: jax.Array,
            router: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return sharded(x, w, out_scale, inv_x, router)

        n_in, n_out = 5, 3
    else:
        sharded = jax.shard_map(
            lambda *a: body(*a)[0],
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=spec,
            check_vma=False,
        )

        def _wrapped(
            x: jax.Array, w: jax.Array, out_scale: jax.Array, inv_x: jax.Array
        ) -> jax.Array:
            return sharded(x, w, out_scale, inv_x)

        n_in, n_out = 4, 1
    tag = f"c{cols}_u{used}" + (f"_r{used_e}" if used_e else "")
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::host_split_gemv_w{world}_l{local}_{tag}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(spec,) * n_in,
        output_partition_specs=(spec,) * n_out,
    )


# One op per split shape (the front and the latent up-proj), built at bind
# time, before torch.compile traces the model.
_OPS: dict = {}


def host_split_op(world: int, local: int, cols: int, used: int, used_e: int = 0):
    key = (world, local, cols, used, used_e)
    if key not in _OPS:
        _OPS[key] = _build(*key)
    return _OPS[key]
