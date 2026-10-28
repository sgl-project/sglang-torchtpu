# SPDX-License-Identifier: Apache-2.0
"""Inkling short-convolution (sconv) op and conv-state backend for torch_tpu.

Wraps :func:`sglang_tpu.attention.pallas.gdn.sconv_wrapper.fused_sconv` as a
torch custom op that updates one conv-state stream in place (donated, aliased),
and :class:`TpuInklingSconvBackend`, which reuses the GDN backend's per-forward
metadata (distribution, query_start_loc, state rows) for every sconv call.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here; see
# attention_gdn_pallas.py (custom_jax_kernel inspects real annotations).

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang_tpu.attention_gdn_pallas import TpuGDNAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_PALLAS_SCONV_OP = None


def build_sconv_pallas_op():
    """Lazily register ``sglang_tpu_pallas::inkling_sconv``.

    Signature::

        sconv(x, conv_state, weight, query_start_loc, state_indices,
              distribution, seq_lens, dyn=None, read_offsets=None,
              use_residual=True, batched_only=False, prefill_only=False,
              num_spec_tokens=0) -> y

    ``x`` is ``[T, dim]``; ``conv_state`` ``[slots, K - 1, 1, dim]`` is updated in
    place. Static ``weight`` is ``[dim, (1,) K]`` (oldest tap first); with
    ``dyn`` (``[T, n_heads, K]``, lag 0 = current token) ``weight`` is the
    lag-major static part ``[K, dim]``. Returns ``y`` ``[T, dim]`` in x.dtype.
    """
    global _PALLAS_SCONV_OP
    if _PALLAS_SCONV_OP is not None:
        return _PALLAS_SCONV_OP

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.gdn.sconv_wrapper import fused_sconv
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    _exported_shapes: set = set()
    op_name = "sglang_tpu_pallas::inkling_sconv"

    def _wrapped(
        x: jax.Array,
        conv_state: jax.Array,
        weight: jax.Array,
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        distribution: jax.Array,
        seq_lens: jax.Array,
        dyn: Optional[jax.Array],
        read_offsets: Optional[jax.Array],
        use_residual: bool,
        batched_only: bool,
        prefill_only: bool,
        num_spec_tokens: int,
    ) -> tuple[jax.Array, jax.Array]:
        # Slot 0 is the null slot; the DP-attention pad sentinel -1 maps there.
        state_indices = jax.numpy.maximum(state_indices, 0)
        # Tile sizing queries TPU info, which a CPU-mesh export cannot answer.
        with tpu_info_for_cpu_export():
            return fused_sconv(
                x,
                conv_state,
                weight,
                query_start_loc,
                state_indices,
                distribution,
                seq_lens,
                dyn,
                read_offsets,
                use_residual=use_residual,
                num_spec_tokens=num_spec_tokens,
                batched_only=batched_only,
                prefill_only=prefill_only,
            )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        # use_residual, batched_only, prefill_only, num_spec_tokens.
        static_argnums=(9, 10, 11, 12),
        donate_argnums=(1,),
        input_output_aliases={1: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name, mutates_args=("conv_state",), device_types=["tpu"]
    )
    def custom_op(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        dyn: Optional[torch.Tensor] = None,
        read_offsets: Optional[torch.Tensor] = None,
        use_residual: bool = True,
        batched_only: bool = False,
        prefill_only: bool = False,
        num_spec_tokens: int = 0,
    ) -> torch.Tensor:
        args = (
            x,
            conv_state,
            weight,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            dyn,
            read_offsets,
            use_residual,
            batched_only,
            prefill_only,
            num_spec_tokens,
        )
        # The CPU-export guard is a jit cache key; enter it only when compiling.
        key = tuple(
            (tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor) else a
            for a in args
        )
        if key in _exported_shapes:
            _new_state, y = jax_kernel(*args)
            return y
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            _new_state, y = jax_kernel(*args)
        _exported_shapes.add(key)
        return y

    @custom_op.register_fake
    def _fake_impl(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        dyn: Optional[torch.Tensor] = None,
        read_offsets: Optional[torch.Tensor] = None,
        use_residual: bool = True,
        batched_only: bool = False,
        prefill_only: bool = False,
        num_spec_tokens: int = 0,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    _PALLAS_SCONV_OP = custom_op
    return custom_op


class TpuInklingSconvBackend(TpuGDNAttnBackend):
    """Inkling's per-request conv-state streams on the fused sconv kernel.

    Inkling has no linear-attention layers: its modules call :meth:`run_sconv`
    directly. ``stream`` is a ``SconvType`` value indexing ``conv[stream][layer]``
    in :class:`~sglang_tpu.mamba_pool.TpuMambaPool`.
    """

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)
        self._sconv_op = build_sconv_pallas_op()

    def sconv_state(self, *, layer_id: int, stream: int) -> torch.Tensor:
        """``[slots, K - 1, 1, dim]`` conv state of one stream of ``layer_id``."""
        pool = self.req_to_token_pool
        return pool.mamba_pool.mamba_cache.conv[stream][
            pool.mamba2_layer_index(layer_id)
        ]

    def run_sconv(
        self,
        x: torch.Tensor,
        *,
        layer_id: int,
        stream: int,
        weight: torch.Tensor,
        dyn: Optional[torch.Tensor] = None,
        use_residual: bool = True,
    ) -> torch.Tensor:
        """y ``[T, dim]`` for x ``[T, dim]``; updates the stream's state slots."""
        state_rows, read_offsets = self._state_rows()
        return self._sconv_op(
            x.contiguous(),
            self.sconv_state(layer_id=layer_id, stream=stream),
            weight,
            self._query_start_loc,
            state_rows,
            self._distribution,
            self._seq_lens,
            dyn=None if dyn is None else dyn.contiguous(),
            read_offsets=read_offsets,
            use_residual=use_residual,
            batched_only=self._is_decode,
            prefill_only=not self._is_decode,
            num_spec_tokens=self._num_spec_tokens,
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("Inkling sconv streams run through run_sconv")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("Inkling sconv streams run through run_sconv")



class TpuInklingHybridAttnBackend(HybridLinearAttnBackend):
    """Every layer attends through the full-attention child; sconv runs on the sidecar."""

    def sconv_state(self, *, layer_id: int, stream: int) -> torch.Tensor:
        return self.linear_attn_backend.sconv_state(layer_id=layer_id, stream=stream)

    def run_sconv(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear_attn_backend.run_sconv(x, **kwargs)


def build_inkling_hybrid_backend(
    runner: "ModelRunner", full_attn_backend
) -> TpuInklingHybridAttnBackend:
    num_layers = runner.model_config.num_hidden_layers
    return TpuInklingHybridAttnBackend(
        full_attn_backend, TpuInklingSconvBackend(runner), list(range(num_layers))
    )
