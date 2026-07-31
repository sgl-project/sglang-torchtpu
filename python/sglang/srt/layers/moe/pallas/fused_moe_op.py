# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py
"""Pallas fused-MoE bridge for the TPU torch_tpu platform.

This module wraps the JAX/Pallas ``gmm_v2`` grouped-matmul kernel
(``sglang.srt.layers.moe.pallas.gmm_v2``) and assembles it into a
single-host fused-MoE op exposed to PyTorch via
``torch_tpu._internal.pallas.jax_op``. It mirrors the pattern used by the
ragged-paged-attention v3 backend in
``sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3``: register one custom
``torch.library`` op whose implementation is a ``jax.jit``-traced JAX
function. Per-shape / per-(topk, activation) specialisation happens
inside the JAX trace cache maintained by ``JaxCallable``; dynamo sees
the op as an opaque tensor-in / tensor-out call.

MoE math (single host, no shard_map)
------------------------------------
For each token, top-``k`` experts are activated. We do the standard
"permute / GMM1(+fused act) / GMM2 / un-permute / weighted-reduce"
pipeline:

1. Flatten ``topk_ids`` (``[m, topk]``) → ``[m * topk]``.
2. ``argsort`` the flattened expert ids; rows of the LHS become
   contiguous blocks per expert. ``group_sizes[e] = #(topk_ids == e)``.
3. Gather rows of ``hidden_states`` into the sorted order.
4. GMM1: ``x_sorted @ w1[e]`` with the gated activation fused in-kernel
   (``fuse_act``): the [.., 2*intermediate] gate||up accumulator is reduced
   to ``act(gate) * up -> [m*topk, intermediate]`` before write-out.
5. GMM2: ``gmm1_out @ w2[e] -> [m*topk, hidden]``.
6. Scatter back and reduce with ``topk_weights``.

Weight layout
-------------
Callers pre-transpose to gmm_v2's ``(g, k, n)`` LHS-RHS convention. The
unquant path supplies::

    w1 = layer.w13_weight.transpose(1, 2).contiguous()  # (g, hidden, 2*intermediate)
    w2 = layer.w2_weight.transpose(1, 2).contiguous()   # (g, intermediate, hidden)

Activation
----------
Only ``"silu"`` (SwiGLU: ``silu(gate) * up``) is exercised today, which
is what Qwen3-MoE uses. ``"gelu"`` is supported for parity with the
mxfp4 path. ``"swiglu_oai"`` is supported for GPT-OSS.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here —
# ``torch_tpu._internal.pallas.jax_op`` inspects ``param.annotation is
# jax.Array`` at registration time; PEP 563 would turn those into
# strings and validation would reject them.

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

logger = logging.getLogger(__name__)

# Singleton: one ``torch.library`` op for the whole process. The JaxCallable
# beneath does its own per-(static_arg, input-shape) trace caching, so a
# single op covers every (topk, activation, dtype, shape) combination the
# model will encounter.
_PALLAS_MOE_OP = None

# ---- JAX kernel body -------------------------------------------------------
#
# ``_fused_moe_jax`` (with ``_fuse_act_name``) is the
# function that ``torch_tpu._internal.pallas.jax_op`` exports and that all the
# per-shape JAX traces specialise. It lives at module scope (not in the
# ``_build_fused_moe_op`` closure) so unit tests can import and call it
# directly with JAX inputs — exercising the exact production execution path
# without going through the torch custom-op shuttle.
#
# JAX is imported inside each function body, not at module scope, so importing
# ``fused_moe_op`` on a non-TPU host stays cheap (the only inbound paths to
# this module are TPU-only: ``UnquantizedFusedMoEMethod.forward_tpu`` and the
# op registration in ``FusedMoE.__init__``, which is gated on ``_is_tpu``).


def _valid_rows_mask(group_sizes, expert_group_offset, local_num_experts, unsort_idx):
    """Return a mask of which rows are owned by this rank."""
    import jax.numpy as jnp

    gsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    row_start = gsum[expert_group_offset]
    row_end = gsum[expert_group_offset + local_num_experts]
    return (row_start <= unsort_idx) & (unsort_idx < row_end)


def _lookup_tile_info(lhs, rhs, rhs_scale, num_total_groups, packed_fp4=False):
    """Pick a ``TileSizes`` for this gmm_v2 call.

    ``get_tuned_block_sizes`` is a pure-Python lookup keyed on static
    shapes / dtypes, so it runs at JAX trace time (not at device exec
    time). On a hit we pin the kernel's tile sizes to the tuned triple;
    on a miss it returns ``get_default_gmm_block_sizes`` (a heuristic that
    respects the shapes) — either way this returns a concrete
    ``TileSizes``.

    ``packed_fp4`` marks a uint8-packed FP4 rhs (stored K is half the real K),
    so we key the lookup on the real K and the float4 dtype the kernel contracts
    in -- otherwise packed w2 mismatches lhs K and falls back to the heuristic.

    ``num_current_groups`` is set equal to ``num_total_groups``
    because this bridge is single-host (no expert parallelism); when
    EP is added the caller will need to pass the local group count
    through.
    """
    import jax.numpy as jnp

    from sglang.srt.layers.moe.pallas.gmm_v2 import TileSizes
    from sglang.srt.layers.moe.pallas.tuned_block_sizes import (
        get_tuned_block_sizes,
    )

    m, k = lhs.shape
    _g, _k_raw, n = rhs.shape
    # Packed FP4 stores 2 codes/byte along K, so real K = stored K * 2.
    _k = _k_raw * 2 if packed_fp4 else _k_raw
    assert _k == k, f"gmm_v2 lhs/rhs k mismatch: {k=} vs {_k=}"
    quant_block_size = k if rhs_scale is None else k // rhs_scale.shape[1]
    rhs_dtype = "float4_e2m1fn" if packed_fp4 else str(jnp.dtype(rhs.dtype).name)
    tm, tk, tn = get_tuned_block_sizes(
        m=int(m),
        k=int(k),
        n=int(n),
        num_total_groups=int(num_total_groups),
        num_current_groups=int(num_total_groups),
        lhs_dtype=str(jnp.dtype(lhs.dtype).name),
        rhs_dtype=rhs_dtype,
        quant_block_size=quant_block_size,
    )
    return TileSizes(tile_m=tm, tile_k=tk, tile_n=tn)


def _dividing_tile_k(base_tile_info):
    """Wrap a gmm_v2 ``tile_info`` so the resolved ``tile_k`` always divides
    ``size_k`` (mirrors the kernel's N-gate, which has no K equivalent).
    """
    import dataclasses

    from sglang.srt.layers.moe.pallas.gmm_v2 import (
        TileSizes,
        _largest_aligned_divisor_at_most,
    )

    def wrapped(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
        if isinstance(base_tile_info, TileSizes):
            tiles = base_tile_info
        else:
            tiles = base_tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        if dims.size_k % tiles.tile_k != 0:
            tiles = dataclasses.replace(
                tiles,
                tile_k=_largest_aligned_divisor_at_most(dims.size_k, tiles.tile_k, 128),
            )
        return tiles

    return wrapped


def _fuse_act_name(activation_name):
    """Map SGLang's activation name to gmm_v2's ``fuse_act`` string.

    The gated activation now runs *in-kernel* in GMM1 (gmm_v2 splits the
    ``2*n`` gate/up accumulator and applies ``act(gate) * up`` before writing
    only ``n`` columns to HBM). ``gmm_v2.apply_act_fn`` accepts
    ``silu`` / ``gelu`` / ``swigluoai``; SGLang spells the GPT-OSS one
    ``swiglu_oai``.
    """
    mapping = {
        "silu": "silu",
        "gelu": "gelu",
        "swiglu_oai": "swigluoai",
        "swigluoai": "swigluoai",
    }
    if activation_name not in mapping:
        raise NotImplementedError(
            f"Unsupported activation '{activation_name}' for the Pallas "
            "fused-MoE op. Supported: silu, gelu, swiglu_oai."
        )
    return mapping[activation_name]


def _fused_moe_jax(
    hidden_states,  # [m, k]
    w1,  # [g, k, 2 * n_intermediate]  bf16, OR [g, k/2, 2*n] uint8 (packed FP4)
    w2,  # [g, n_intermediate, k]      bf16, OR [g, n/2, k]    uint8 (packed FP4)
    topk_weights,  # [m, topk]
    topk_ids,  # i32[m, topk]
    w1_scale,  # Optional[g, 1, 1, 2 * n_intermediate] f32
    w2_scale,  # Optional[g, 1, 1, k] f32
    w1_bias,  # Optional[g, 1, 2 * n_intermediate] f32
    w2_bias,  # Optional[g, 1, k] f32
    topk,
    activation_name,
    no_combine=False,
    global_num_experts=0,
    expert_group_offset=0,
    use_sc_gather=0,
    permute_dense_threshold=0,
):
    """Production fused-MoE kernel body. Exported by ``_build_fused_moe_op``.

    Module-scoped (rather than nested in ``_build_fused_moe_op``) so unit
    tests can call it directly with JAX inputs and exercise the exact
    permute / GMM1 / activation / GMM2 / un-permute path that production
    uses.
    """
    import jax
    import jax.numpy as jnp

    from sglang.srt.layers.moe.pallas.gmm_v2 import (
        TileSizes,
        apply_act_fn,
        calculate_tiling,
        gmm_v2,
    )

    m, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    out_dtype = hidden_states.dtype

    # uint8 rhs = packed FP4 (2 float4 codes/byte along K); gmm_v2 unpacks to
    # float4_e2m1fn in-kernel when told packed_fp4=True (real K = shape[1] * 2).
    # This governs *any* packed-float4 rhs: today only NVFP4 emits uint8, but the
    # MXFP4-native path (not yet wired) would flow through here too — correct
    # (block size is data-driven from the scale), just untested.
    w1_packed = w1.dtype == jnp.uint8
    w2_packed = w2.dtype == jnp.uint8

    # gmm_v2 needs the routed-row count ``m*topk`` to be a
    # multiple of the bf16 sublane (16). Padding the *input* tokens here keeps
    # the permute/gather operating on an already-aligned, fixed-bucket shape
    # and removes the post-gather ``jnp.pad`` + ``group_sizes`` scatter that
    # XLA was offloading to the SparseCore (and crashing on). Pad tokens route
    # to expert 0 with zero weight, so they contribute nothing to any real
    # token and are sliced off at the very end.
    m_orig = m
    TOKEN_ALIGN = (
        16  # token bucket; m_pad % 16 == 0 => (m_pad*topk) % 16 == 0 for any topk
    )
    m_pad = ((m + TOKEN_ALIGN - 1) // TOKEN_ALIGN) * TOKEN_ALIGN
    if m_pad != m:
        hidden_states = jnp.pad(hidden_states, ((0, m_pad - m), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, m_pad - m), (0, 0)))  # pad -> expert 0
        topk_weights = jnp.pad(
            topk_weights, ((0, m_pad - m), (0, 0))
        )  # pad -> weight 0
        m = m_pad

    # Routing: build a permutation that puts rows of hidden_states in
    # expert-major order so a single grouped-matmul covers GMM1 / GMM2.
    topk_ids_flat = topk_ids.reshape(-1).astype(jnp.int32)  # [m*topk]
    # argsort is stable enough on TPU for our usage; the gmm kernel
    # only cares about contiguous expert runs.
    argsort_idx = jnp.argsort(topk_ids_flat)
    # Row-of-original-token for each sorted slot.
    token_indices = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
    sorted_token_ids = token_indices[argsort_idx]

    # Decide whether to use the dense permute/un-permute paths based on the
    # number of rows (m) and the number of routed rows (m*topk).
    use_dense_permute = m < permute_dense_threshold
    use_dense_unpermute = (m * topk) < permute_dense_threshold and not no_combine

    if use_dense_permute:
        permute_matrix = jax.nn.one_hot(sorted_token_ids, m, dtype=hidden_states.dtype)
        x_sorted = jax.lax.dot_general(
            permute_matrix,
            hidden_states,
            (((1,), (0,)), ((), ())),
            preferred_element_type=hidden_states.dtype,
        )
    elif use_sc_gather:
        from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_v2 import (
            ragged_gather_v2,
        )

        n_sorted = sorted_token_ids.shape[0]
        x_sorted = ragged_gather_v2(
            hidden_states,
            sorted_token_ids,
            jnp.array([0], dtype=jnp.int32),
            jnp.array([n_sorted], dtype=jnp.int32),
        )
    else:
        x_sorted = hidden_states[sorted_token_ids]  # [m*topk, k]
    # GMM1: gate/up projection. For per-column/per-tensor scales (and MXFP4), the
    # gated activation is fused in-kernel: ``fuse_act`` makes gmm_v2 split the
    # [.., 2*n] accumulator and write only [.., n] = act(gate) * up, avoiding a
    # full materialize of the 2*n tensor to HBM plus a separate activation pass.
    # Compact block-FP8 scales need different scale offsets for the gate and up
    # halves, so run an unfused GMM1 and apply the activation in JAX for
    # correctness.
    compact_block_scales = w1_scale is not None and w1_scale.shape[-1] != w1.shape[-1]
    # Packed FP4 fuses the SwiGLU in-kernel (gmm_v2 slices the up-half's per-block
    # scale). Only compact block scales, which need distinct gate/up scale offsets,
    # still run unfused with a separate JAX activation.
    unfused_gmm1 = compact_block_scales
    gmm1_fuse_act = None if unfused_gmm1 else _fuse_act_name(activation_name)

    gmm1_tile_info = None
    if compact_block_scales:
        gmm1_tile_info = TileSizes(
            tile_m=min(128, int(x_sorted.shape[0])),
            tile_k=int(w1.shape[1]),
            tile_n=min(896, int(w1.shape[2])),
        )

    # Weights arrive lane-aligned from the loader (bf16: pad activations to
    # match; quantized: unpadded-at-load dims are unsupported)
    gmm1_tiling = _dividing_tile_k(gmm1_tile_info or calculate_tiling)
    if w1_scale is None:
        assert w1.shape[1] % 128 == 0, (
            f"gmm_v2 w1 K={w1.shape[1]} is not lane-aligned; pad the weight K "
            "to a multiple of 128 at load time"
        )
        if w1.shape[1] != hidden_size:
            # activations are born at model width; zero-pad K to the
            # load-padded weight's width (0 * w = 0, exact)
            x_sorted = jnp.pad(x_sorted, ((0, 0), (0, w1.shape[1] - hidden_size)))
    elif hidden_size % 128 != 0:
        raise NotImplementedError(
            f"Quantized MoE with non-lane-aligned hidden dim ({hidden_size}) "
            "is unsupported: pad weights AND scales jointly at load time."
        )

    use_group_offset = global_num_experts > num_experts
    route_num_experts = global_num_experts if use_group_offset else num_experts
    # one_hot+sum is cheaper on TPU than bincount.
    group_sizes = jax.nn.one_hot(topk_ids_flat, route_num_experts, dtype=jnp.int32).sum(
        axis=0
    )
    group_offset = (
        jnp.array([expert_group_offset], dtype=jnp.int32) if use_group_offset else None
    )

    # The TensorCore path requires zero-initializing, while the SparseCore
    # path does not since it skips masked rows entirely — rows the GMM
    # never wrote are never read. no_combine gathers every row back, so it
    # zero-initializes on both paths. The dense un-permute multiplies every
    # row (unwritten ones by weight 0), so unwritten rows must be zeros,
    # not garbage — 0 * NaN would poison the sum.
    if use_sc_gather and not no_combine and not use_dense_unpermute:
        zero_init = False
    else:
        zero_init = True

    gmm1_out = gmm_v2(
        lhs=x_sorted,
        rhs=w1,
        group_sizes=group_sizes,
        rhs_scale=w1_scale,
        rhs_bias=w1_bias,
        group_offset=group_offset,
        zero_initialize=zero_init,
        preferred_element_type=out_dtype,
        # MXFP4 keeps the fuse_act-aware default tiling (tile_n is per-projection
        # when fused); compact FP8 pins an unfused tile.
        tile_info=gmm1_tiling,
        fuse_act=gmm1_fuse_act,
        packed_fp4=w1_packed,
    )
    if unfused_gmm1:
        gmm1_out = apply_act_fn(
            gmm1_out, int(gmm1_out.shape[-1] // 2), _fuse_act_name(activation_name)
        )
    # gmm1_out: [m*topk, n_intermediate]

    # same padding
    if w2_scale is not None and (
        gmm1_out.shape[1] % 128 != 0 or hidden_size % 128 != 0
    ):
        raise NotImplementedError(
            f"Quantized MoE with non-lane-aligned dims (intermediate "
            f"{gmm1_out.shape[1]}, hidden {hidden_size}) is unsupported: "
            "pad weights AND scales jointly at load time."
        )
    # Packed FP4 w2 has a halved (packed) K; ``_lookup_tile_info`` is packed-aware
    # (keys on the real K + the float4 dtype), so packed and bf16 w2 both go
    # through the tuned lookup (tuned triple on a hit, heuristic on a miss).
    gmm2_tiling = _dividing_tile_k(
        _lookup_tile_info(gmm1_out, w2, w2_scale, num_experts, packed_fp4=w2_packed)
    )
    gmm2_out = gmm_v2(
        lhs=gmm1_out,
        rhs=w2,
        group_sizes=group_sizes,
        rhs_scale=w2_scale,
        rhs_bias=w2_bias,
        group_offset=group_offset,
        zero_initialize=zero_init,
        preferred_element_type=out_dtype,
        tile_info=gmm2_tiling,
        packed_fp4=w2_packed,
    )[
        :,
        :hidden_size,
    ]  # [m*topk, k]

    # Un-permute back to token-major order. Models that apply a post-expert
    # norm ask the runner for no_combine output so they can RMSNorm each top-k
    # slot before the weighted reduction.
    if use_dense_unpermute:
        # Unpermute using dense matmul.
        w_sorted = topk_weights.reshape(-1)[argsort_idx].astype(jnp.float32)
        unpermute_matrix = (
            jax.nn.one_hot(sorted_token_ids, m, dtype=jnp.float32) * w_sorted[:, None]
        )
        combined = jax.lax.dot_general(
            unpermute_matrix,
            gmm2_out.astype(jnp.float32),
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return combined[:m_orig].astype(out_dtype)

    # TODO (Qinghan): check performance against scatter
    unsort_idx = jnp.argsort(argsort_idx)
    if use_sc_gather:
        if not no_combine:
            # Fused gather + weighted top-k sum.
            from sglang.srt.layers.moe.pallas.sparse_core.dense_gather_reduce import (
                dense_gather_reduce,
            )
            from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_reduce_v2 import (
                ragged_gather_reduce,
            )

            if use_group_offset:
                combined = ragged_gather_reduce(
                    gmm2_out,
                    unsort_idx.astype(jnp.int32),
                    topk_weights.reshape(-1),
                    _valid_rows_mask(
                        group_sizes,
                        expert_group_offset,
                        num_experts,
                        unsort_idx,
                    ),
                    topk,
                )
            else:
                # All experts local: every row is valid, so use the maskless
                # dense variant.
                combined = dense_gather_reduce(
                    gmm2_out,
                    unsort_idx.astype(jnp.int32),
                    topk_weights,
                    topk,
                )
            return combined[:m_orig].astype(out_dtype)

        # no_combine: just gather, without the weighted top-k sum.
        from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_v2 import (
            ragged_gather_v2,
        )

        n_sorted = unsort_idx.shape[0]
        unsorted = ragged_gather_v2(
            gmm2_out,
            unsort_idx.astype(jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([n_sorted], dtype=jnp.int32),
        )
    else:
        # Gather on the TensorCore.
        unsorted = gmm2_out[unsort_idx]
    unsorted = unsorted.reshape(m, topk, hidden_size)
    unsorted = unsorted[:m_orig].astype(out_dtype)
    if no_combine:
        # Zero the rows this rank did not compute, so later post-norm + combine
        # cannot consume expert-0 placeholder outputs with original router weights.
        if use_group_offset:
            valid_rows_mask = _valid_rows_mask(
                group_sizes, expert_group_offset, num_experts, unsort_idx
            )[: m_orig * topk].reshape(m_orig, topk, 1)
        else:
            valid_rows_mask = topk_weights[:m_orig].reshape(m_orig, topk, 1) != 0
        return jnp.where(valid_rows_mask, unsorted, jnp.zeros_like(unsorted))

    weights = topk_weights[:m_orig].reshape(m_orig, topk, 1).astype(unsorted.dtype)
    out = (unsorted * weights).sum(axis=1).astype(out_dtype)
    return out


def _build_fused_moe_op():
    """Register the fused-MoE kernel as a torch custom op (idempotent).

    Mirrors ``_build_pallas_kernel_op`` in attention_pallas_rpa_v3.py — registration
    is heavy (``inspect.signature`` on a JAX function + JAX export) and
    cannot run inside a dynamo trace, so ``FusedMoE.__init__``
    (fused_moe_triton/layer.py) calls this eagerly at layer construction,
    before torch.compile traces the model. The model code path just looks
    up the op cached in ``_PALLAS_MOE_OP``.
    """
    global _PALLAS_MOE_OP
    if _PALLAS_MOE_OP is not None:
        return _PALLAS_MOE_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)

    op_name = "sglang_tpu_pallas::fused_moe"

    # ``jax_op`` inspects annotations to decide which positional params
    # are tensors (jax.Array) vs. compile-time constants (int/str/...).
    # Keep annotations spelled out — no Optional aliases — so PEP-563 is
    # not needed.
    def _wrapped(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        w1_scale: Optional[jax.Array] = None,
        w2_scale: Optional[jax.Array] = None,
        w1_bias: Optional[jax.Array] = None,
        w2_bias: Optional[jax.Array] = None,
        topk: int = 1,
        activation: str = "silu",
        no_combine: bool = False,
        global_num_experts: int = 0,
        expert_group_offset: int = 0,
        use_sc_gather: int = 0,
        permute_dense_threshold: int = 0,
    ) -> jax.Array:
        return _fused_moe_jax(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            topk,
            activation,
            no_combine,
            global_num_experts,
            expert_group_offset,
            use_sc_gather,
            permute_dense_threshold,
        )

    custom_op = cpu_export_jax_op(op_name, _wrapped)

    def _fake_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        topk: int = 1,
        activation: str = "silu",
        no_combine: bool = False,
        global_num_experts: int = 0,
        expert_group_offset: int = 0,
        use_sc_gather: int = 0,
        permute_dense_threshold: int = 0,
    ):
        if no_combine:
            return hidden_states.new_empty(
                (hidden_states.shape[0], topk, hidden_states.shape[1])
            )
        # Output is [m, hidden] in the dtype of hidden_states. Using
        # ``empty_like`` lets ``torch.compile`` thread the dynamic token
        # dimension without invoking the JAX tracer.
        return torch.empty_like(hidden_states)

    custom_op.register_fake(_fake_impl)

    _PALLAS_MOE_OP = custom_op
    return custom_op


def fused_moe_call(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    activation: str,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    no_combine: bool = False,
    global_num_experts: int = 0,
    expert_group_offset: int = 0,
) -> torch.Tensor:
    """Invoke the Pallas fused-MoE op.

    Inputs use gmm_v2's ``(g, k, n)`` LHS-RHS convention:
      * ``hidden_states``: ``[num_tokens, hidden_size]``
      * ``w1``: ``[num_experts, hidden_size, 2 * intermediate]``  (gate || up)
      * ``w2``: ``[num_experts, intermediate, hidden_size]``
      * ``topk_weights``: ``[num_tokens, topk]``  (routing scores, dtype of x)
      * ``topk_ids``: ``[num_tokens, topk]``  (int32 expert ids)

    Returns ``[num_tokens, hidden_size]`` in ``hidden_states.dtype``,
    or the un-combined per-expert ``[num_tokens, topk, hidden_size]``
    when ``no_combine`` is True.
    """
    op = _build_fused_moe_op()
    return op(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk,
        activation,
        no_combine,
        global_num_experts,
        expert_group_offset,
        int(envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get()),
        int(envs.SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD.get()),
    )
