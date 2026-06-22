"""Pallas fused-MoE bridge for the TPU torch_tpu platform.

This module wraps the JAX/Pallas ``gmm_v2`` grouped-matmul kernel
(``sglang.srt.layers.moe.pallas.gmm_v2``) and assembles it into a
single-host fused-MoE op exposed to PyTorch via
``torch_tpu._internal.pallas.jax_op``. It mirrors the pattern used by the
ragged-paged-attention v3 backend in
``sglang.srt.layers.attention.pallas_backend``: register one custom
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

logger = logging.getLogger(__name__)

# Singleton: one ``torch.library`` op for the whole process. The JaxCallable
# beneath does its own per-(static_arg, input-shape) trace caching, so a
# single op covers every (topk, activation, dtype, shape) combination the
# model will encounter.
_PALLAS_MOE_OP = None


# ---- JAX kernel body -------------------------------------------------------
#
# ``_fused_moe_jax`` (with ``_fuse_act_name`` and ``_lookup_tile_info``) is the
# function that ``torch_tpu._internal.pallas.jax_op`` exports and that all the
# per-shape JAX traces specialise. It lives at module scope (not in the
# ``_build_fused_moe_op`` closure) so unit tests can import and call it
# directly with JAX inputs — exercising the exact production execution path
# without going through the torch custom-op shuttle.
#
# JAX is imported inside each function body, not at module scope, so importing
# ``fused_moe_op`` on a non-TPU host stays cheap (the only inbound paths to
# this module are TPU-only: ``UnquantizedFusedMoEMethod.forward_tpu`` and the
# TPU pre-warm hook).


def _lookup_tile_info(lhs, rhs, rhs_scale, num_total_groups):
    """Pick a ``TileSizes`` for this gmm_v2 call.

    ``get_tuned_block_sizes`` is a pure-Python lookup keyed on static
    shapes / dtypes, so it runs at JAX trace time (not at device exec
    time). On a hit we pin the kernel's tile sizes to the tuned
    triple; on a miss we hand ``None`` back to gmm_v2, which falls
    through to its built-in ``calculate_tiling`` heuristic.

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
    _g, _k, n = rhs.shape
    assert _k == k, f"gmm_v2 lhs/rhs k mismatch: {k=} vs {_k=}"
    quant_block_size = k if rhs_scale is None else k // rhs_scale.shape[1]
    tm, tk, tn = get_tuned_block_sizes(
        m=int(m),
        k=int(k),
        n=int(n),
        num_total_groups=int(num_total_groups),
        num_current_groups=int(num_total_groups),
        lhs_dtype=str(jnp.dtype(lhs.dtype).name),
        rhs_dtype=str(jnp.dtype(rhs.dtype).name),
        quant_block_size=quant_block_size,
    )
    return TileSizes(tile_m=tm, tile_k=tk, tile_n=tn)


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
    }
    if activation_name not in mapping:
        raise NotImplementedError(
            f"Unsupported activation '{activation_name}' for the Pallas "
            "fused-MoE op. Supported: silu, gelu, swiglu_oai."
        )
    return mapping[activation_name]


def _fused_moe_jax(
    hidden_states,  # [m, k]
    w1,  # [g, k, 2 * n_intermediate]
    w2,  # [g, n_intermediate, k]
    topk_weights,  # [m, topk]
    topk_ids,  # i32[m, topk]
    w1_scale,  # Optional[g, 1, 1, 2 * n_intermediate] f32
    w2_scale,  # Optional[g, 1, 1, k] f32
    w1_bias,  # Optional[g, 1, 2 * n_intermediate] f32
    w2_bias,  # Optional[g, 1, k] f32
    topk,
    activation_name,
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

    # one_hot+sum is cheaper on TPU than bincount.
    group_sizes = jax.nn.one_hot(topk_ids_flat, num_experts, dtype=jnp.int32).sum(
        axis=0
    )

    x_sorted = hidden_states[sorted_token_ids]  # [m*topk, k]

    # GMM1: gate/up projection. For per-column/per-tensor scales, the gated
    # activation can be fused in-kernel. Compact block-FP8 scales need different
    # scale offsets for the gate and up halves, so run an unfused GMM1 and apply
    # the activation in JAX for correctness.
    compact_block_scales = w1_scale is not None and w1_scale.shape[-1] != w1.shape[-1]
    gmm1_fuse_act = None if compact_block_scales else _fuse_act_name(activation_name)
    gmm1_tile_info = None
    if compact_block_scales:
        gmm1_tile_info = TileSizes(
            tile_m=min(128, int(x_sorted.shape[0])),
            tile_k=int(w1.shape[1]),
            tile_n=min(896, int(w1.shape[2])),
        )

    gmm1_out = gmm_v2(
        lhs=x_sorted,
        rhs=w1,
        group_sizes=group_sizes,
        rhs_scale=w1_scale,
        rhs_bias=w1_bias,
        preferred_element_type=out_dtype,
        tile_info=gmm1_tile_info or calculate_tiling,
        fuse_act=gmm1_fuse_act,
    )
    if compact_block_scales:
        gmm1_out = apply_act_fn(
            gmm1_out, int(gmm1_out.shape[-1] // 2), _fuse_act_name(activation_name)
        )
    # gmm1_out: [m*topk, n_intermediate]

    # GMM2: project back to hidden size.
    gmm2_out = gmm_v2(
        lhs=gmm1_out,
        rhs=w2,
        group_sizes=group_sizes,
        rhs_scale=w2_scale,
        rhs_bias=w2_bias,
        preferred_element_type=out_dtype,
        tile_info=_lookup_tile_info(gmm1_out, w2, w2_scale, num_experts),
    )  # [m*topk, k]

    # Un-permute back to token-major order and reduce across the top-k
    # experts with the routing weights.
    # TODO (Qinghan): check performance against scatter
    unsort_idx = jnp.argsort(argsort_idx)
    unsorted = gmm2_out[unsort_idx].reshape(m, topk, hidden_size)
    weights = topk_weights.reshape(m, topk, 1).astype(unsorted.dtype)
    out = (unsorted * weights).sum(axis=1).astype(out_dtype)
    # Slice off the padded tokens added at entry.
    return out[:m_orig]


def _build_fused_moe_op():
    """Lazily register the fused-MoE kernel as a torch custom op.

    Mirrors ``_build_pallas_kernel_op`` in pallas_backend.py — registration
    is heavy (``inspect.signature`` on a JAX function + JAX export) and
    cannot run inside a dynamo trace, so we call this eagerly from the
    TPU pre-warm hook and just look the op up from the model code path.
    """
    global _PALLAS_MOE_OP
    if _PALLAS_MOE_OP is not None:
        return _PALLAS_MOE_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

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
        )

    custom_op = _torch_pallas.jax_op(op_name, _wrapped)

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
    ):
        # Output is [m, hidden] in the dtype of hidden_states. Using
        # ``empty_like`` on a slice lets ``torch.compile`` thread the
        # dynamic token dimension without invoking the JAX tracer.
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
) -> torch.Tensor:
    """Invoke the Pallas fused-MoE op.

    Inputs use gmm_v2's ``(g, k, n)`` LHS-RHS convention:
      * ``hidden_states``: ``[num_tokens, hidden_size]``
      * ``w1``: ``[num_experts, hidden_size, 2 * intermediate]``  (gate || up)
      * ``w2``: ``[num_experts, intermediate, hidden_size]``
      * ``topk_weights``: ``[num_tokens, topk]``  (routing scores, dtype of x)
      * ``topk_ids``: ``[num_tokens, topk]``  (int32 expert ids)

    Returns ``[num_tokens, hidden_size]`` in ``hidden_states.dtype``.
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
    )
