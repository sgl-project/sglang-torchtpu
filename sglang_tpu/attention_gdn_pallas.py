# SPDX-License-Identifier: Apache-2.0
"""Pallas Gated-DeltaNet (GDN) linear-attention op for the TPU torch_tpu platform.

Wraps the fused Conv1D + gated-delta-rule JAX/Pallas kernel
(``sglang_tpu.attention.pallas.gdn.fused_conv1d_gdn``) as a PyTorch
custom op, so the hybrid Qwen3.5 / Qwen3-Next GDN layers can invoke it from
PyTorch and trace it through ``torch.compile`` with ``backend="tpu"``.

Design notes
------------

* The kernel runs conv1d AND the gated-delta recurrence in one call. It reads
  the per-request conv/recurrent state from a state cache (slot 0 is a reserved
  null block), and writes the updated state back into the same buffers via an
  input/output alias — mirroring how the RPA v3 kernel owns its KV-cache write.
  The backend therefore treats ``conv_state`` / ``recurrent_state`` as mutated
  in place and does not write them separately.
* Ragged (mixed prefill + decode) batching is resolved inside the kernel from
  ``query_start_loc`` / ``state_indices`` / ``distribution`` / ``seq_lens``;
  ``has_initial_state`` is derived as ``(seq_lens - query_lens) > 0``.

Sibling to :mod:`sglang_tpu.attention_pallas_rpa_v3`
(full attention). See that module for the shared torch<->JAX bridge rationale.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here.
# ``torch_tpu._internal.pallas.custom_jax_kernel`` inspects the wrapped JAX
# function's annotations at registration time (``param.annotation is
# jax.Array``); under PEP 563 those would be plain strings and validation would
# reject them.

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# One custom op per process. The kernel specialises on shapes internally, so no
# shape-dependent op specialisation is needed at the torch.library boundary.
_PALLAS_GDN_OP = None


def build_gdn_pallas_op():
    """Lazily register the fused Conv1D+GDN kernel as a torch custom op.

    Returns a ``torch.library.custom_op`` with signature::

        gdn(qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias,
            a_log, dt_bias, query_start_loc, state_indices, distribution,
            seq_lens, n_kq, n_v, d_k, d_v, kernel_size, kda=False,
            gate_lower_bound=None, batched_only=False, prefill_only=False,
            fused_qkvg=False, packed_weights=None,
            track_indices=None, track_end=None) -> output

    ``conv_state`` and ``recurrent_state`` are updated in place (declared as
    ``mutates_args``); the returned tensor is the fused attention output of
    shape ``[num_tokens, n_v, d_v]``. ``qkv`` is exactly the conv dim wide,
    unless ``fused_qkvg`` declares it a fused ``[q, k, v, g]`` projection (g
    ``n_v * d_v`` wide), whose leading conv-dim columns are read.

    ``track_indices`` and ``track_end`` are optional int32 tensors of shape
    ``[num_seqs]``, supplied together in the same order as ``state_indices``.
    They select physical snapshot destination slots in both state caches and
    exclusive checkpoint offsets in the packed input, respectively. An end
    of -1 disables the snapshot for that row; enabled ends must match kernel
    tile boundaries. Active state slots continue to update independently.
    """
    global _PALLAS_GDN_OP
    if _PALLAS_GDN_OP is not None:
        return _PALLAS_GDN_OP

    import jax  # noqa: F401  (force a loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
    from sglang_tpu.attention.pallas.gdn.config import AttentionMode
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # Shapes whose kernel is already compiled, so the export guard below can be
    # skipped on the hot path. Per-op (this builder runs once).
    _exported_shapes: set = set()

    op_name = "sglang_tpu_pallas::gdn_attention"

    def _wrapped(
        qkv: jax.Array,
        b: jax.Array,
        a: jax.Array,
        conv_state: jax.Array,
        recurrent_state: jax.Array,
        conv_weight: jax.Array,
        conv_bias: Optional[jax.Array],
        a_log: jax.Array,
        dt_bias: jax.Array,
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        distribution: jax.Array,
        seq_lens: jax.Array,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        kernel_size: int,
        kda: bool,
        gate_lower_bound: Optional[float],
        batched_only: bool,
        prefill_only: bool,
        fused_qkvg: bool,
        packed_weights: Optional[jax.Array] = None,
        track_indices: Optional[jax.Array] = None,
        track_end: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        conv_dim = n_kq * d_k * 2 + n_v * d_v
        if fused_qkvg:
            # Slicing here fuses with the kernel's input prep; a torch split
            # view would be copied at the op boundary.
            assert qkv.shape[1] == conv_dim + n_v * d_v, (qkv.shape, conv_dim)
            qkv = qkv[:, :conv_dim]
        elif qkv.shape[1] != conv_dim:
            raise ValueError(
                f"qkv is {qkv.shape[1]} wide but the conv dim is {conv_dim}; a "
                "fused [q, k, v, g] projection needs fused_qkvg=True."
            )
        # The GDN kernel sizes its tiling from `pltpu.get_tpu_info().num_lanes`
        # (gdn/wrapper.py), but distributed Pallas programs are exported through
        # a logical CPU mesh, so at trace time JAX reports device_kind="cpu" and
        # get_tpu_info() raises. Substitute the physical TPU metadata for the
        # trace, as RPA does (attention_pallas_rpa_base.py). The generic bridge
        # in pallas_jax_op.py does this automatically; this op registers through
        # torch_tpu's pallas helper directly and has to opt in.
        with tpu_info_for_cpu_export():
            (new_conv_state, new_recurrent_state), output = fused_conv1d_gdn(
                qkv,
                b,
                a,
                conv_state,
                recurrent_state,
                conv_weight,
                conv_bias,
                a_log,
                dt_bias,
                query_start_loc,
                state_indices,
                distribution,
                seq_lens,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                attention_mode=AttentionMode.KDA if kda else AttentionMode.GDN,
                gate_lower_bound=gate_lower_bound,
                batched_only=batched_only,
                prefill_only=prefill_only,
                packed_weights=packed_weights,
                track_indices=track_indices,
                track_end=track_end,
            )
        # Flatten the ((conv, rec), out) return so the bridge can alias the two
        # state buffers (outputs 0, 1) back onto the donated inputs (args 3, 4).
        # [T, n_v, d_v] is the kernel's layout; a flat [T, n_v * d_v] costs a
        # relayout, and every consumer splits heads again anyway.
        return new_conv_state, new_recurrent_state, output.reshape(-1, n_v, d_v)

    # Buffer donation: the kernel writes the updated state straight back into the
    # caller's conv/recurrent buffers via an input/output alias, no extra copy.
    # This requires each layer to donate a whole tensor, not a view of a shared
    # pool -- the bridge's write-back cannot alias a view and would fall back to
    # slice-in plus dynamic-update-slice per layer. `TpuPerLayerState` allocates
    # one tensor per layer for that reason.
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        # n_kq, n_v, d_k, d_v, kernel_size, kda, gate_lower_bound, batched_only,
        # prefill_only, fused_qkvg are static.
        static_argnums=(13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
        donate_argnums=(3, 4),
        input_output_aliases={3: 0, 4: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("conv_state", "recurrent_state"),
        device_types=["tpu"],
    )
    def custom_op(
        qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: Optional[torch.Tensor],
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        kernel_size: int,
        kda: bool = False,
        gate_lower_bound: Optional[float] = None,
        batched_only: bool = False,
        prefill_only: bool = False,
        fused_qkvg: bool = False,
        packed_weights: Optional[torch.Tensor] = None,
        track_indices: Optional[torch.Tensor] = None,
        track_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Second half of the CPU-mesh export guard (see `_wrapped`). Mosaic
        # lowering queries TPU metadata again (`_check_block_mappings` ->
        # get_tpu_info().num_sublanes) and runs here, outside `_wrapped`, so it
        # needs its own guard or precompile dies on device_kind="cpu".
        #
        # Enter it only on the compiling call:
        # `export_ignore_forward_compatibility` is a JAX config context that
        # participates in jit cache keys, so toggling it per forward thrashes
        # them (pallas_jax_op.py guards the same way). Keyed by every shape,
        # dtype and static the kernel specializes on rather than a one-shot
        # flag: each token/batch bucket compiles separately, and MIXED batches
        # share a token bucket across batch caps.
        tensors = (
            qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            packed_weights,
            track_indices,
            track_end,
        )
        key = (
            tuple(None if t is None else (tuple(t.shape), t.dtype) for t in tensors),
            n_kq,
            n_v,
            d_k,
            d_v,
            kernel_size,
            kda,
            gate_lower_bound,
            batched_only,
            prefill_only,
            fused_qkvg,
        )
        if key in _exported_shapes:
            _new_conv, _new_rec, output = jax_kernel(
                qkv,
                b,
                a,
                conv_state,
                recurrent_state,
                conv_weight,
                conv_bias,
                a_log,
                dt_bias,
                query_start_loc,
                state_indices,
                distribution,
                seq_lens,
                n_kq,
                n_v,
                d_k,
                d_v,
                kernel_size,
                kda,
                gate_lower_bound,
                batched_only,
                prefill_only,
                fused_qkvg,
                packed_weights,
                track_indices,
                track_end,
            )
        else:
            with (
                tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                _new_conv, _new_rec, output = jax_kernel(
                    qkv,
                    b,
                    a,
                    conv_state,
                    recurrent_state,
                    conv_weight,
                    conv_bias,
                    a_log,
                    dt_bias,
                    query_start_loc,
                    state_indices,
                    distribution,
                    seq_lens,
                    n_kq,
                    n_v,
                    d_k,
                    d_v,
                    kernel_size,
                    kda,
                    gate_lower_bound,
                    batched_only,
                    prefill_only,
                    fused_qkvg,
                    packed_weights,
                    track_indices,
                    track_end,
                )
            _exported_shapes.add(key)
        return output

    def _fake_impl(
        qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: Optional[torch.Tensor],
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        kernel_size: int,
        kda: bool = False,
        gate_lower_bound: Optional[float] = None,
        batched_only: bool = False,
        prefill_only: bool = False,
        fused_qkvg: bool = False,
        packed_weights: Optional[torch.Tensor] = None,
        track_indices: Optional[torch.Tensor] = None,
        track_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = qkv.shape[0]
        return qkv.new_empty((num_tokens, n_v, d_v))

    custom_op.register_fake(_fake_impl)

    _PALLAS_GDN_OP = custom_op
    return custom_op


class TpuGDNAttnBackend(MambaAttnBackendBase):
    """TPU GDN linear-attention backend backed by the fused Pallas kernel.

    The TPU sibling of the CUDA/Triton
    :class:`sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend`.
    Composed with the full-attention backend by
    :class:`~sglang.srt.layers.attention.hybrid_linear_attn_backend.HybridLinearAttnBackend`,
    which routes GDN layers here and full-attention layers to RPA v3.

    Prefill and decode share one code path: the fused kernel dispatches on the
    ``distribution`` vector (``[decode_end, prefill_end, mixed_end]``), whose
    semantics match the RPA v3 backend's distribution — ``[bs, bs, bs]`` for a
    pure-decode batch, ``[0, bs, bs]`` for extend/prefill. State read/write is
    owned by the kernel via a donated input/output alias, so ``save_kv_cache``
    is irrelevant here (state is always updated).
    """

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)
        # Build (and JAX-export) the custom op eagerly at construction time so
        # the heavyweight export does not happen inside a torch.compile trace.
        self._op = build_gdn_pallas_op()
        self._extra_buffer_enabled = get_exec().mamba.enable_mamba_extra_buffer
        if self._extra_buffer_enabled:
            self.req_to_token_pool.mamba_pool.warmup_state_copy(
                self.req_to_token_pool.size
            )
        # Per-forward GDN metadata (int32, TPU-native), computed once in
        # init_forward_metadata and read by every GDN layer's forward — not
        # carried by the shared ForwardMetadata dataclass.
        self._distribution: Optional[torch.Tensor] = None
        self._seq_lens: Optional[torch.Tensor] = None
        self._state_indices: Optional[torch.Tensor] = None
        self._query_start_loc: Optional[torch.Tensor] = None
        # Pure decode has no PER_SEQ rows and extend no BATCHED rows (see
        # _get_distribution), so each runs one kernel pass instead of both.
        self._is_decode = False
        # Cache the (shape-static) distribution vector across forwards, keyed by
        # (is_decode, bs, device) — same rationale as RPA v3's distribution cache.
        self._distribution_cache: dict = {}

    def _get_distribution(
        self, is_decode: bool, bs: int, device: torch.device
    ) -> torch.Tensor:
        key = (is_decode, bs, device)
        dist = self._distribution_cache.get(key)
        if dist is None:
            # `distribution` is (i, j, k): seqs[:i] decode, seqs[i:j] prefill,
            # seqs[j:k] mixed (kernels/causal_conv1d docstring). The GDN wrapper
            # only reads distribution[0] and distribution[-1], so the middle
            # element is inert; spell it [num_decode, num_decode, num_reqs] to
            # match tpu-inference's tpu_runner.py, where this kernel came from.
            num_decode = bs if is_decode else 0
            vals = [num_decode, num_decode, bs]
            dist = torch.tensor(vals, dtype=torch.int32, device=device)
            self._distribution_cache[key] = dist
        return dist

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        # `distribution` bounds which sequences the kernel walks, so give it the
        # pre-bucket row count: `forward_batch.seq_lens` is already padded to the
        # batch bucket here, and in decode `num_tiles = cdiv(distribution[0],
        # tile_size)` would put the padded rows in scope. They are inert either
        # way -- their state writes land in the reserved null slot -- but there is
        # no reason to compute them. Same source RPA reads
        # (attention_pallas_rpa_base._pre_bucket_padded_batch_size).
        bs = int(getattr(forward_batch, "_tpu_real_b", forward_batch.seq_lens.shape[0]))
        device = forward_batch.seq_lens.device
        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        self._distribution = self._get_distribution(is_decode, bs, device)
        self._is_decode = is_decode
        # Full per-request sequence length (context + current); the kernel uses
        # it to derive has_initial_state = (seq_lens - query_lens) > 0. Cast the
        # per-forward index/length metadata to int32 once here (shared by all
        # GDN layers) rather than per-layer in _run.
        self._seq_lens = forward_batch.seq_lens.to(torch.int32)
        # Build only TPU metadata. The GPU base constructs variable-length
        # tracked-row selections and reads device masks on the host. Here every
        # row retains its bucket slot and disabled snapshots use end=-1.
        slots = self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
        self._state_indices = (
            self._translate_mamba_indices(slots).to(torch.int32).clamp(min=0)
        )
        padded_bs = forward_batch.seq_lens.shape[0]
        if is_decode:
            self._query_start_loc = torch.arange(
                padded_bs + 1, dtype=torch.int32, device=device
            )
        else:
            self._query_start_loc = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.cumsum(forward_batch.extend_seq_lens, 0, dtype=torch.int32),
                )
            )
        self._track_indices = self._track_end = None
        if forward_batch.mamba_track_mask is not None:
            mask = forward_batch.mamba_track_mask
            self._track_indices = (
                self._translate_mamba_indices(forward_batch.mamba_track_indices)
                .to(torch.int32)
                .clamp(min=0)
            )
            if is_decode:
                ends = self._query_start_loc[1:]
            else:
                # Use the native scheduler's checkpoint grid and its encoding
                # of forced intermediate branching boundaries.
                lengths = forward_batch.mamba_track_aligned_lens()
                ends = self._query_start_loc[:-1] + lengths.to(torch.int32)
            valid = mask & (torch.arange(padded_bs, device=device) < bs)
            self._track_end = torch.where(valid, ends, -1).to(torch.int32)
        elif self._extra_buffer_enabled:
            # Warmup batches carry no tracking plan. Keep the same optional
            # tensor signature as live requests to precompile the cache path.
            self._track_indices = torch.zeros_like(self._state_indices)
            self._track_end = torch.full_like(self._state_indices, -1)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=self._query_start_loc,
            mamba_cache_indices=self._state_indices,
            mamba_track_indices=self._track_indices,
        )

    def _run(
        self,
        layer: "RadixLinearAttention",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_state = layer_cache.conv[0]  # [size+1, K-1, 1, dim]
        recurrent_state = layer_cache.temporal  # [size+1, n_v, d_k, d_v]

        state_indices = self._state_indices
        query_start_loc = self._query_start_loc

        # RadixLinearAttention stores conv weight as [dim, K]; the kernel wants
        # [dim, 1, K] (depthwise conv layout).
        conv_weight = layer.conv_weights.unsqueeze(1)
        kernel_size = layer.conv_weights.shape[-1]

        # Op arg order: (qkv, b, a, conv_state, recurrent_state, conv_weight,
        # conv_bias, a_log, dt_bias, query_start_loc, state_indices,
        # distribution, seq_lens, n_kq, n_v, d_k, d_v, kernel_size).
        out = self._op(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            layer.bias,
            layer.A_log,
            layer.dt_bias,
            query_start_loc,
            state_indices,
            self._distribution,
            self._seq_lens,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
            kernel_size,
            batched_only=self._is_decode,
            prefill_only=not self._is_decode,
            track_indices=self._track_indices,
            track_end=self._track_end,
        )
        # Per-head [num_tokens, n_v, d_v], as the model's output-gate norm expects.
        return out

    def forward_decode(
        self,
        q=None,
        k=None,
        v=None,
        layer: "RadixLinearAttention" = None,
        forward_batch: "ForwardBatch" = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._run(layer, mixed_qkv, a, b)

    def forward_extend(
        self,
        q=None,
        k=None,
        v=None,
        layer: "RadixLinearAttention" = None,
        forward_batch: "ForwardBatch" = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._run(layer, mixed_qkv, a, b)
