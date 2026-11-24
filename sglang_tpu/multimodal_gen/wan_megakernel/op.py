# SPDX-License-Identifier: Apache-2.0
"""The block stack as one torch op over every rank (torch_tpu).

``cpu_export_jax_op`` exports ``kernel.stack`` under a ``shard_map`` over a
``("cfg", "sp")`` mesh whose row ``c`` is CFG group ``c`` and column ``u`` its
Ulysses rank ``u`` (sglang's rank order ``tp-sp-pp-cfg-dp`` puts world rank
``4 c + u`` there). Each process passes its local tensors: its rows of ``x``,
``cos`` and ``sin`` (``P(("cfg", "sp"))``), its branch's cross K/V and the keys
it attends to (``P("cfg")``), and its full copy of the modulation rows, the
real-row count and the packed weights (``P()``). Nothing is donated; the
exchange buffers live inside the call.
"""

# No `from __future__ import annotations`: cpu_export_jax_op reads the
# signature's jax.Array annotations to tell tensors from static arguments.

import functools

import jax

from sglang_tpu.multimodal_gen.wan_megakernel import kernel
from sglang_tpu.multimodal_gen.wan_megakernel import reference as ref
from sglang_tpu.multimodal_gen.wan_megakernel.tiling import Tiling

AXES = ("cfg", "sp")
# bf16 op names of the non-default row-sum floor layouts (Tiling.floor_guard).
_FLOOR_SUFFIX = {"shared": "", "inline": "_fi", "rowfix": "_fr"}


def local_stack(
    x,
    cos,
    sin,
    e,
    k2,
    v2,
    keys,
    key_weights,
    valid,
    *weights,
    cfg: ref.Config,
    tiling: Tiling,
    sp: int,
    softmax: str = "auto",
):
    """One device's call; ``keys`` and ``valid`` are ``[1]`` int32 arrays."""
    return kernel.stack(
        x,
        e,
        k2,
        v2,
        cos,
        sin,
        dict(zip(kernel.KERNEL_WEIGHTS, weights)),
        valid[0],
        keys[0],
        key_weights,
        cfg=cfg,
        tiling=tiling,
        axis="sp" if sp > 1 else None,
        devices=sp,
        softmax=softmax,
    )


def sharded_stack(mesh, cfg: ref.Config, tiling: Tiling, sp: int):
    from jax.sharding import PartitionSpec as P

    rows, branch, rep = P(AXES), P("cfg"), P()
    return jax.shard_map(
        functools.partial(local_stack, cfg=cfg, tiling=tiling, sp=sp),
        mesh=mesh,
        in_specs=(rows, rows, rows, rep)
        + (branch,) * 4
        + (rep,)
        + (rep,) * len(kernel.KERNEL_WEIGHTS),
        out_specs=rows,
        check_vma=False,
    )


def build_stack_op(cfg: ref.Config, cfg_ranks: int, sp: int, tiling: Tiling):
    """Register the torch op; call on every rank after distributed init."""
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(AXES, (cfg_ranks, sp))
    sharded = sharded_stack(mesh, cfg, tiling, sp)

    def stack(
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        e: jax.Array,
        k2: jax.Array,
        v2: jax.Array,
        keys: jax.Array,
        key_weights: jax.Array,
        valid: jax.Array,
        wqkv: jax.Array,
        wo: jax.Array,
        wq2: jax.Array,
        wo2: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        vecs: jax.Array,
    ) -> jax.Array:
        return sharded(
            x,
            cos,
            sin,
            e,
            k2,
            v2,
            keys,
            key_weights,
            valid,
            wqkv,
            wo,
            wq2,
            wo2,
            w1,
            w2,
            vecs,
        )

    rows, branch, rep = P(AXES), P("cfg"), P()
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::wan_stack_cfg{cfg_ranks}_sp{sp}_L{cfg.layers}"
        f"_q{tiling.block_q}_kv{tiling.block_kv}_s{tiling.sub_kv}_r{tiling.qkv_rows}x{tiling.mlp_rows}"
        f"{'_w8' if tiling.fp8 else ''}{'_qk8' if tiling.fp8_qk else ''}"
        f"{_FLOOR_SUFFIX[tiling.floor_guard] if not (tiling.fp8 or tiling.fp8_qk) else ''}"
        f"{tiling.opt_suffix()}",
        stack,
        mesh=mesh,
        input_partition_specs=(rows, rows, rows, rep)
        + (branch,) * 4
        + (rep,)
        + (rep,) * len(kernel.KERNEL_WEIGHTS),
        output_partition_specs=(rows,),
    )
