# SPDX-License-Identifier: Apache-2.0
"""In-place page writes into a fused TPU KV layer buffer.

``write_pages`` copies ``pages[i]`` into ``kv[block_ids[i]]`` for every
``i``. The Pallas kernel visits only the listed blocks and aliases its output
to the KV buffer, so the buffer is updated in place and no copy of the layer
is made. The torch custom op donates the KV buffer to the kernel for the same
reason. JAX is imported only when a kernel is built, so the CPU fallback in
`write_pages_torch` works without it.
"""

from __future__ import annotations

import torch

_OP_NAME = "sglang_tpu::kv_write_pages"
_OP = None


def _kernel(block_ids_ref, pages_ref, kv_ref, out_ref):
    del block_ids_ref, kv_ref
    out_ref[...] = pages_ref[...]


def write_pages(block_ids, pages, kv, *, interpret: bool = False):
    """Return ``kv`` with ``kv[block_ids[i]] = pages[i]``; the output aliases ``kv``."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    block = (None, *kv.shape[1:])
    zeros = (0,) * (kv.ndim - 1)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(pages.shape[0],),
        in_specs=[
            pl.BlockSpec(block, lambda i, ids: (i, *zeros)),
            pl.BlockSpec(block, lambda i, ids: (ids[i], *zeros)),
        ],
        out_specs=pl.BlockSpec(block, lambda i, ids: (ids[i], *zeros)),
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(kv.shape, kv.dtype),
        input_output_aliases={2: 0},
        interpret=interpret,
    )(block_ids, pages, kv)


def get_write_pages_op():
    """Torch custom op ``kv_write_pages(block_ids, pages, kv)`` mutating ``kv``."""
    global _OP
    if _OP is not None:
        return _OP

    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export

    jax_kernel = _torch_pallas.custom_jax_kernel(
        write_pages,
        name="jax_kv_write_pages",
        donate_argnums=(2,),
        input_output_aliases={2: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(_OP_NAME, mutates_args=("kv",), device_types=["tpu"])
    def kv_write_pages(
        block_ids: torch.Tensor, pages: torch.Tensor, kv: torch.Tensor
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(block_ids, pages, kv)

    @kv_write_pages.register_fake
    def _(block_ids: torch.Tensor, pages: torch.Tensor, kv: torch.Tensor) -> None:
        return None

    _OP = kv_write_pages
    return _OP


def write_pages_torch(block_ids: torch.Tensor, pages: torch.Tensor, kv: torch.Tensor):
    """``kv[block_ids[i]] = pages[i]`` in place, via the kernel on TPU."""
    if kv.device.type == "tpu":
        get_write_pages_op()(block_ids.to(torch.int32), pages, kv)
    else:
        kv[block_ids] = pages
