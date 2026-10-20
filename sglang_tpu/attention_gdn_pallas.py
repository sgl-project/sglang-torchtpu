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
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.runtime_context import get_disagg, get_exec
from sglang_tpu.mamba_pool import AT_SLOT

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
            fused_qkvg=False, packed_weights=None, track_indices=None,
            track_end=None, read_offsets=None, num_spec_tokens=0) -> output

    ``state_indices`` and ``track_indices`` are state rows. A verify window
    (``num_spec_tokens > 0``) reads row ``state_indices + read_offsets`` and
    writes its checkpoints from ``state_indices``; the backend maps slots and
    request windows to rows.

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
        read_offsets: Optional[jax.Array] = None,
        num_spec_tokens: int = 0,
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
        # The DP-attention pad sentinel -1 is not a skip for this kernel; slot 0
        # is the reserved null block. Clamped here, not as a per-forward launch.
        state_indices = jax.numpy.maximum(state_indices, 0)
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
                read_offsets,
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
                num_spec_tokens=num_spec_tokens,
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
        # prefill_only, fused_qkvg and num_spec_tokens are static.
        static_argnums=(13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27),
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
        read_offsets: Optional[torch.Tensor] = None,
        num_spec_tokens: int = 0,
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
            read_offsets,
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
            num_spec_tokens,
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
                read_offsets,
                num_spec_tokens,
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
                    read_offsets,
                    num_spec_tokens,
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
        read_offsets: Optional[torch.Tensor] = None,
        num_spec_tokens: int = 0,
    ) -> torch.Tensor:
        num_tokens = qkv.shape[0]
        return qkv.new_empty((num_tokens, n_v, d_v))

    custom_op.register_fake(_fake_impl)

    _PALLAS_GDN_OP = custom_op
    return custom_op


class GdnTracedMetadata(NamedTuple):
    """Per-forward tensors the GDN/KDA layers read; direct compile lifts them as inputs."""

    distribution: torch.Tensor
    seq_lens: torch.Tensor
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    track_indices: Optional[torch.Tensor]
    track_end: Optional[torch.Tensor]
    read_offsets: Optional[torch.Tensor]
    # A verify's request pool rows, which select its checkpoint windows.
    req_pool_indices: Optional[torch.Tensor] = None


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
        mamba_pool = self.req_to_token_pool.mamba_pool
        if mamba_pool.checkpoint_width > 1:
            # Verify windows are indexed by request pool row; a larger request
            # pool would write other requests' windows.
            num_req_rows = self.req_to_token_pool.req_to_token.shape[0]
            if num_req_rows > mamba_pool.spec_state_size + 1:
                raise ValueError(
                    f"TPU GDN checkpoint windows cover {mamba_pool.spec_state_size + 1} "
                    f"request rows, but the request pool has {num_req_rows}."
                )
        if self._extra_buffer_enabled:
            self.req_to_token_pool.mamba_pool.warmup_state_copy(
                self.req_to_token_pool.size
            )
        self._warmup_slot_bookkeeping()
        # Per-forward GDN metadata (int32, TPU-native), computed once in
        # init_forward_metadata and read by every GDN layer's forward — not
        # carried by the shared ForwardMetadata dataclass.
        self._distribution: Optional[torch.Tensor] = None
        self._seq_lens: Optional[torch.Tensor] = None
        self._state_indices: Optional[torch.Tensor] = None
        self._query_start_loc: Optional[torch.Tensor] = None
        self._track_indices: Optional[torch.Tensor] = None
        self._track_end: Optional[torch.Tensor] = None
        # Pure decode / verify has no PER_SEQ rows and extend no BATCHED rows
        # (see _get_distribution), so each runs one kernel pass instead of both.
        self._is_decode = False
        mamba_pool = self.req_to_token_pool.mamba_pool
        self._spec_window = mamba_pool.checkpoint_width
        self._read_offsets_pool = mamba_pool.checkpoint_read_offsets
        self._read_offsets: Optional[torch.Tensor] = None
        self._req_pool_indices: Optional[torch.Tensor] = None
        self._num_spec_tokens = 0
        # Cache the (shape-static) distribution vector across forwards, keyed by
        # (is_decode, bs, device) — same rationale as RPA v3's distribution cache.
        self._distribution_cache: dict = {}
        self._query_start_loc_cache: dict = {}
        self._slot_memo: Optional[tuple] = None

    def _slot_indices(
        self, req_pool_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(int64 slot ids, int32 slot ids) of the batch's requests.

        An unchanged batch reuses the last result: the gather and the cast are
        two launches per round otherwise. Valid only while every write to the
        request-to-slot table is an eager in-place op (it is written when a
        request is allocated), which bumps ``_version``; a write from inside a
        compiled graph would not, and must clear ``_slot_memo``.
        """
        pool = self.req_to_token_pool
        table = pool.req_index_to_mamba_index_mapping
        memo = self._slot_memo
        if (
            memo is not None
            and memo[0]() is req_pool_indices
            and memo[1] == req_pool_indices._version
            and memo[2] == table._version
        ):
            return memo[3], memo[4]
        slots = self._translate_mamba_indices(pool.get_mamba_indices(req_pool_indices))
        slots_i32 = slots.to(torch.int32)
        # int64 for the verify commit's index_copy_ (the table itself is int32).
        if slots.dtype != torch.int64:
            slots = slots.long()
        if pool.mamba_v2p_table is None:
            self._slot_memo = (
                weakref.ref(req_pool_indices),
                req_pool_indices._version,
                table._version,
                slots,
                slots_i32,
            )
        return slots, slots_i32

    def _warmup_slot_bookkeeping(self) -> None:
        """Slot allocation's mapping write and clear_slots, once per new-request count."""
        pool = self.req_to_token_pool
        mapping = pool.req_index_to_mamba_index_mapping
        with torch.no_grad():
            for count in range(1, pool.size + 1):
                # Allocator slot ids arrive as CPU int64; request row 0 and state
                # slot 0 are reserved, so these writes touch no request.
                slots = torch.zeros(count, dtype=torch.int64)
                mapping[[0] * count] = slots.to(
                    device=mapping.device, dtype=torch.int32
                )
                pool.mamba_pool.clear_slots(slots)
            mapping[0:1].cpu()
            if get_disagg().disaggregation_mode != "null":
                # The P/D state payload reads one row by Python int, a constant
                # baked into each program, so every row is a separate program.
                for row in range(mapping.shape[0]):
                    mapping[row].cpu()

    def _uniform_query_start_loc(
        self, padded_bs: int, step: int, device: torch.device
    ) -> torch.Tensor:
        # Shape-static, so cached like the distribution: building it every
        # forward is one more launch per round.
        key = (padded_bs, step, device)
        loc = self._query_start_loc_cache.get(key)
        if loc is None:
            loc = torch.arange(
                0, (padded_bs + 1) * step, step, dtype=torch.int32, device=device
            )
            self._query_start_loc_cache[key] = loc
        return loc

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
        is_verify = forward_batch.forward_mode.is_target_verify()
        is_decode = forward_batch.forward_mode.is_decode_or_idle() or is_verify
        self._distribution = self._get_distribution(is_decode, bs, device)
        self._is_decode = is_decode
        # Full per-request sequence length including this forward's tokens (the
        # verify window too); the kernel derives has_initial_state from it.
        self._seq_lens = forward_batch.seq_lens.to(torch.int32)
        if is_verify:
            self._seq_lens = self._seq_lens + self._spec_window
        # Build only TPU metadata. The GPU base constructs variable-length
        # tracked-row selections and reads device masks on the host. Here every
        # row retains its bucket slot and disabled snapshots use end=-1.
        # Slot ids; the op clamps the -1 pad in-graph (see build_gdn_pallas_op).
        slots, self._state_indices = self._slot_indices(forward_batch.req_pool_indices)
        padded_bs = forward_batch.seq_lens.shape[0]
        if is_decode:
            # One token per request, or the verify window per request.
            step = self._spec_window if is_verify else 1
            self._query_start_loc = self._uniform_query_start_loc(
                padded_bs, step, device
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
        elif self._extra_buffer_enabled and not is_verify:
            # Verify snapshots are taken after acceptance (commit_verified_states).
            # Warmup batches carry no tracking plan. Keep the same optional
            # tensor signature as live requests to precompile the cache path.
            self._track_indices = torch.zeros_like(self._state_indices)
            self._track_end = torch.full_like(self._state_indices, -1)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=self._query_start_loc,
            mamba_cache_indices=self._state_indices,
            mamba_track_indices=self._track_indices,
        )
        self._init_spec_metadata(forward_batch)

    def _init_spec_metadata(self, forward_batch: "ForwardBatch") -> None:
        self._read_offsets = None
        self._req_pool_indices = None
        self._num_spec_tokens = 0
        if self._spec_window == 1:
            return
        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            # Checkpoint rows are laid out per pool window; a different verify
            # width (adaptive spec) would read and commit the wrong rows.
            assert forward_batch.spec_info.num_tokens_per_req == self._spec_window
            self._read_offsets = self._read_offsets_pool
            self._req_pool_indices = forward_batch.req_pool_indices
            self._num_spec_tokens = self._spec_window - 1
        elif mode.is_extend():
            # Extend reads and writes the slot row, so that row is committed.
            self._read_offsets_pool.index_fill_(
                0, self._state_indices.clamp(min=0).long(), AT_SLOT
            )

    def _state_rows(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(base state rows, per-sequence read offsets) the kernel addresses.

        A verify writes its checkpoints into the request's window and reads the
        committed state: the slot row for AT_SLOT, else a window row.
        """
        if self._read_offsets is None:
            return self._state_indices, None
        slots = self._state_indices.clamp(min=0)
        bases = self.req_to_token_pool.mamba_pool.window_bases(
            self._req_pool_indices
        ).to(torch.int32)
        offsets = self._read_offsets.index_select(0, slots.to(torch.int64))
        return bases, torch.where(offsets < 0, slots - bases, offsets)

    def traced_metadata(self) -> GdnTracedMetadata:
        return GdnTracedMetadata(
            distribution=self._distribution,
            seq_lens=self._seq_lens,
            state_indices=self._state_indices,
            query_start_loc=self._query_start_loc,
            track_indices=self._track_indices,
            track_end=self._track_end,
            read_offsets=self._read_offsets,
            req_pool_indices=self._req_pool_indices,
        )

    @contextmanager
    def bind_traced_metadata(self, metadata: GdnTracedMetadata) -> Iterator[None]:
        """Install graph inputs in place of the eager metadata while tracing."""
        previous = self.traced_metadata()
        self._set_traced_metadata(metadata)
        try:
            yield
        finally:
            self._set_traced_metadata(previous)

    def _set_traced_metadata(self, metadata: GdnTracedMetadata) -> None:
        self._distribution = metadata.distribution
        self._seq_lens = metadata.seq_lens
        self._state_indices = metadata.state_indices
        self._query_start_loc = metadata.query_start_loc
        self._track_indices = metadata.track_indices
        self._track_end = metadata.track_end
        self._read_offsets = metadata.read_offsets
        self._req_pool_indices = metadata.req_pool_indices

    def commit_verified_states(
        self,
        mamba_caches,
        state_indices_tensor: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        src_indices_raw: Optional[torch.Tensor],
    ) -> None:
        """Point each verified slot's next read at its last accepted checkpoint,
        and snapshot boundary checkpoints into radix track slots."""
        mamba_pool = self.req_to_token_pool.mamba_pool
        mamba_pool.select_checkpoints(
            state_indices_tensor.clamp(min=0), last_correct_step_indices
        )
        if mamba_track_indices is None:
            return
        # As core's PLE commit, the caller resolves the boundary steps.
        assert mamba_steps_to_track is not None
        num_rows = last_correct_step_indices.shape[0]
        # Track ids come per live request; pad on host so device shapes stay bucketed.
        track_slots = mamba_track_indices.to(torch.int32)
        if track_slots.shape[0] < num_rows:
            track_slots = torch.nn.functional.pad(
                track_slots, (0, num_rows - track_slots.shape[0]), value=-1
            )
        mamba_pool.snapshot_checkpoints(
            track_slots, self._req_pool_indices, mamba_steps_to_track
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

        state_rows, read_offsets = self._state_rows()
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
            state_rows,
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
            read_offsets=read_offsets,
            num_spec_tokens=self._num_spec_tokens,
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
