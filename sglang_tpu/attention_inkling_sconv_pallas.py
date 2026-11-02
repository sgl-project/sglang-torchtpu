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
from sglang_tpu.envs import envs

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
        if not envs.SGLANG_TPU_ENABLE_INKLING_SCONV_PALLAS.get():
            if self._num_spec_tokens:
                xla_sconv = self._xla_sconv_verify
            elif self._is_decode:
                xla_sconv = self._xla_sconv_decode
            else:
                xla_sconv = self._xla_sconv
            return xla_sconv(
                x,
                state=self.sconv_state(layer_id=layer_id, stream=stream),
                weight=weight,
                dyn=dyn,
                use_residual=use_residual,
            )
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

    def _xla_sconv(self, x, *, state, weight, dyn, use_residual):
        """The sconv kernel's math in XLA ops; reads and writes ``state``'s slot rows."""
        assert self._num_spec_tokens == 0, "XLA sconv has no speculative verify path"
        num_tokens, dim = x.shape
        kernel_size = dyn.shape[2] if dyn is not None else weight.shape[-1]
        hist = kernel_size - 1
        qsl = self._query_start_loc.to(torch.int32)
        num_seqs = qsl.numel() - 1
        rows = torch.arange(num_tokens, device=x.device, dtype=torch.int32)
        seq = (rows[:, None] >= qsl[None, 1:]).sum(1).clamp(max=num_seqs - 1)
        pos = rows - qsl.index_select(0, seq)
        q_lens = qsl[1:] - qsl[:-1]
        # A request continues from its slot only if it had tokens before this forward.
        has_state = (self._seq_lens[:num_seqs] - q_lens) > 0
        slots = self._state_indices.clamp(0, state.shape[0] - 1).long()
        # [seqs * hist, dim], oldest row first per sequence.
        old = state.index_select(0, slots).reshape(num_seqs, hist, dim).float()
        # where, not a multiply: a recycled slot's stale rows must not leak NaN/Inf.
        old = torch.where(has_state[:, None, None], old, torch.zeros_like(old)).reshape(num_seqs * hist, dim)
        xf = x.float()
        acc = xf if use_residual else torch.zeros_like(xf)
        for lag in range(kernel_size):
            past = xf
            if lag:
                shifted = torch.cat((xf.new_zeros(lag, dim), xf))[:num_tokens]
                state_row = seq * hist + (hist - lag + pos).clamp(0, hist - 1)
                from_state = old.index_select(0, state_row.long())
                past = torch.where((pos >= lag)[:, None], shifted, from_state)
            if dyn is None:
                w = weight.reshape(dim, -1)[:, kernel_size - 1 - lag].float()[None]
            else:
                head_dim = dim // dyn.shape[1]
                w = weight[lag].float()[None] + dyn[:, :, lag].float().repeat_interleave(head_dim, dim=1)
            acc = acc + w * past
        # New state: the last `hist` inputs of [old state rows, this forward's rows].
        offs = q_lens[:, None] - hist + torch.arange(hist, device=x.device, dtype=torch.int32)[None]
        x_rows = (qsl[:-1, None] + offs).clamp(0, num_tokens - 1).reshape(-1).long()
        state_rows = (
            torch.arange(num_seqs, device=x.device, dtype=torch.int32)[:, None] * hist + (hist + offs).clamp(0, hist - 1)
        ).reshape(-1).long()
        new = torch.where(
            (offs >= 0).reshape(-1)[:, None], xf.index_select(0, x_rows), old.index_select(0, state_rows)
        )
        self._write_state(state, slots, new.reshape(num_seqs, hist * dim))
        scale = (kernel_size * (2 if dyn is not None else 1) + int(use_residual)) ** -0.5
        return (acc * scale).to(x.dtype)

    def _xla_sconv_decode(self, x, *, state, weight, dyn, use_residual):
        """``_xla_sconv`` for one new token per request: no per-row sequence lookup or gathers."""
        num_seqs, dim = x.shape
        kernel_size = dyn.shape[2] if dyn is not None else weight.shape[-1]
        hist = kernel_size - 1
        slots = self._state_indices.clamp(0, state.shape[0] - 1).long()
        old = state.index_select(0, slots).reshape(num_seqs, hist, dim).float()
        old = torch.where((self._seq_lens[:num_seqs] > 1)[:, None, None], old, torch.zeros_like(old))
        # [seqs, K, dim], oldest first; the new token is the last tap.
        window = torch.cat((old, x.float()[:, None]), dim=1)
        if dyn is None:
            w = weight.reshape(dim, kernel_size).float().T[None]
        else:
            # Lag-major (lag 0 = current token) -> window order.
            head_dim = dim // dyn.shape[1]
            lagged = weight.float()[None] + dyn.float().permute(0, 2, 1).repeat_interleave(head_dim, dim=2)
            w = lagged.flip(1)
        acc = (window * w).sum(1)
        if use_residual:
            acc = acc + x.float()
        self._write_state(state, slots, window[:, 1:].reshape(num_seqs, hist * dim))
        scale = (kernel_size * (2 if dyn is not None else 1) + int(use_residual)) ** -0.5
        return (acc * scale).to(x.dtype)

    def _xla_sconv_verify(self, x, *, state, weight, dyn, use_residual):
        """``_xla_sconv_decode`` over each request's verify window of ``_spec_window`` tokens.

        Reads the committed state row and writes the state after every window position
        into the request's checkpoint rows; ``commit_verified_states`` picks one.
        """
        width = self._spec_window
        num_seqs, dim = x.shape[0] // width, x.shape[1]
        kernel_size = dyn.shape[2] if dyn is not None else weight.shape[-1]
        hist = kernel_size - 1
        bases, offsets = self._state_rows()
        bases = bases[:num_seqs].long()
        read_rows = (bases + offsets[:num_seqs].long()).clamp(0, state.shape[0] - 1)
        old = state.index_select(0, read_rows).reshape(num_seqs, hist, dim).float()
        # [seqs, hist + width, dim], oldest first; a verify always continues a request.
        seq = torch.cat((old, x.float().view(num_seqs, width, dim)), dim=1)
        # [seqs, width, K, dim]: the conv window ending at each verify position.
        windows = torch.stack([seq[:, tap : tap + width] for tap in range(kernel_size)], dim=2)
        if dyn is None:
            w = weight.reshape(dim, kernel_size).float().T[None, None]
        else:
            head_dim = dim // dyn.shape[1]
            lagged = weight.float()[None] + dyn.float().permute(0, 2, 1).repeat_interleave(head_dim, dim=2)
            w = lagged.flip(1).view(num_seqs, width, kernel_size, dim)
        acc = (windows * w).sum(2).reshape(num_seqs * width, dim)
        if use_residual:
            acc = acc + x.float()
        # State after position j: the hist inputs ending at it.
        checkpoints = windows[:, :, 1:].reshape(num_seqs * width, hist * dim)
        steps = torch.arange(width, device=x.device)
        rows = (bases[:, None] + steps[None]).reshape(-1)
        live = (self._state_indices[:num_seqs] >= 0).repeat_interleave(width)
        self._write_state(state, rows, checkpoints, live=live)
        scale = (kernel_size * (2 if dyn is not None else 1) + int(use_residual)) ** -0.5
        return (acc * scale).to(x.dtype)

    def _write_state(self, state, slots, new, live=None):
        """Store each request's new window ``new`` [rows, K-1 * dim] in its row ``slots`` of ``state``."""
        # Dense one-hot write, not index_copy_: padded rows all clamp to slot 0, and the
        # duplicate-index scatter is nondeterministic once compiled.
        num_slots = state.shape[0]
        if live is None:
            live = self._state_indices >= 0
        onehot = (slots[:, None] == torch.arange(num_slots, device=new.device)[None]) & live[:, None]
        scattered = torch.einsum("ns,nf->sf", onehot.float(), new)
        flat = state.reshape(num_slots, -1)
        state.copy_(
            torch.where(onehot.any(0)[:, None], scattered, flat.float()).to(state.dtype).reshape(state.shape)
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
