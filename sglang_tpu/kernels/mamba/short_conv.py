# SPDX-License-Identifier: Apache-2.0
"""torch_tpu bridge for state-mutating Pallas ShortConv."""

# Do not postpone annotations: custom_jax_kernel inspects jax.Array types.
from typing import Optional

import torch

_OP = None


def build_short_conv_op():
    """Register once, before tracing the model; state is a whole layer tensor."""
    global _OP
    if _OP is not None:
        return _OP

    import jax
    from torch_tpu._internal import pallas as torch_pallas

    from sglang_tpu.kernels.mamba.short_conv_pallas import short_conv_pallas
    from sglang_tpu.pallas_export import argnums, tpu_info_for_cpu_export

    def wrapped(
        x: jax.Array,
        weight: jax.Array,
        bias: Optional[jax.Array],
        state: jax.Array,
        slots: jax.Array,
        starts: Optional[jax.Array],
        has_prefix: Optional[jax.Array],
        decode: bool,
    ) -> tuple[jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            return short_conv_pallas(
                x, weight, bias, state, slots, starts, has_prefix, decode=decode
            )

    state_arg = argnums(wrapped, "state")[0]
    jax_kernel = torch_pallas.custom_jax_kernel(
        wrapped,
        name="jax_lfm2_short_conv",
        static_argnums=argnums(wrapped, "decode"),
        donate_argnums=(state_arg,),
        input_output_aliases={state_arg: 0},
        warn_deprecated=False,
    )
    exported_shapes = set()

    @torch.library.custom_op(
        "sglang_tpu_pallas::lfm2_short_conv",
        mutates_args=("state",),
        device_types=["tpu"],
    )
    def op(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        state: torch.Tensor,
        slots: torch.Tensor,
        starts: Optional[torch.Tensor],
        has_prefix: Optional[torch.Tensor],
        decode: bool,
    ) -> torch.Tensor:
        args = (x, weight, bias, state, slots, starts, has_prefix, decode)
        key = (
            tuple(None if t is None else (tuple(t.shape), t.dtype) for t in args[:-1]),
            decode,
        )
        if key in exported_shapes:
            _, output = jax_kernel(*args)
        else:
            # Mosaic lowering also queries TPU properties outside wrapped(),
            # during torch_tpu's CPU-mesh export. Guard only compilation: the
            # export config participates in JAX cache keys, not the hot path.
            with (
                tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                _, output = jax_kernel(*args)
            exported_shapes.add(key)
        return output

    @op.register_fake
    def fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        state: torch.Tensor,
        slots: torch.Tensor,
        starts: Optional[torch.Tensor],
        has_prefix: Optional[torch.Tensor],
        decode: bool,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    _OP = op
    return op
