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
import math
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


def _padded_token_count(m: int, topk: int, sublane: int) -> int:
    """Smallest ``m_pad >= m`` with ``(m_pad * topk) % sublane == 0``.

    gmm_v2 constrains the LHS row count, which after the permute is
    ``m_pad * topk``, not the token count -- so the minimal token alignment is
    ``sublane // gcd(sublane, topk)``, not a flat ``sublane``. When
    ``topk=10`` against the bf16 tiling of 16, giving gcd 2 and an alignment of
    8: a decode batch of 8 then pads to 8 instead of 16, and both the grouped
    matmul and the permute see 80 routed rows instead of 160. ``topk=1`` or any
    ``topk`` coprime with ``sublane`` resolves back to the full tiling, so this
    only ever loosens the padding, never tightens it.

    ``sublane`` is gmm_v2's ``lhs_sublane_tiling`` for the activation dtype --
    16 for bf16, but 8 for f32 and 32/64 for 8-/4-bit -- passed in rather than
    assumed, since a wider tiling than the one used here would under-pad.
    """
    token_align = sublane // math.gcd(sublane, topk)
    m_pad = ((m + token_align - 1) // token_align) * token_align

    return m_pad


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

    ``num_current_groups`` is set equal to ``num_total_groups`` because callers
    pass the LOCAL expert count (``w1.shape[0]``), which under EP is already
    this rank's share -- so both are the local count and the lookup is correct
    under expert parallelism as written.

    Keep it that way. The heuristic behind a table miss computes
    ``tm = round_up_to_multiple_of_128_within_limit(2 * m // g, 512)``, and with
    the local g (16 under EP16) ``2m/g`` is >= 512 for every prefill bucket, so
    ``tm`` saturates at 512 and is identical across buckets. Passing the GLOBAL
    count instead would make ``2m/g`` = 64/128/256 at pad_t 1024/2048/4096 and
    ``tm`` would become 128/128/256 -- a tiling that silently changes at one
    bucket and not the others, which is precisely the failure signature we are
    trying not to have.
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


def _quant_block_compatible(tile_k, quant_block_size):
    """Whether ``tile_k`` can index whole rhs scale blocks.

    ``IndexMaps.rhs_scale_index_map`` derives the scale window from
    ``b_id = (k_id * tile_k) // quant_block_size`` and
    ``b_tile_id = b_id // cdiv(tile_k, quant_block_size)``. That arithmetic is
    only exact when one of the two sizes divides the other; otherwise a tile
    straddles a block boundary and every ``k_id >= 1`` reads a window that has
    drifted behind the one it needs.
    """
    if quant_block_size is None:
        return True
    return tile_k % quant_block_size == 0 or quant_block_size % tile_k == 0


def _largest_compatible_tile_k(size_k, limit, quant_block_size, alignment=128):
    """Largest ``alignment``-aligned divisor of ``size_k`` that is ``<= limit``
    and quant-block compatible. Subsumes gmm_v2's
    ``_largest_aligned_divisor_at_most`` (identical when unquantized).
    """
    limit = min(size_k, limit)
    for candidate in range(limit - (limit % alignment), 0, -alignment):
        if size_k % candidate == 0 and _quant_block_compatible(
            candidate, quant_block_size
        ):
            return candidate
    raise ValueError(
        f"no {alignment}-aligned divisor of size_k={size_k} that is <= {limit} "
        f"is compatible with quant_block_size={quant_block_size}; pad K to an "
        "aligned, divisor-rich width at load time"
    )


def _gmm2_tile_source(lhs, w2, w2_scale, num_experts, *, w2_packed):
    """Pick the gmm2 ``tile_info`` source: tuned lookup for packed FP4 only.

    Packed FP4 w2 stores a halved K, which ``calculate_tiling`` would read as
    the real K and size tiles against half the contraction dim, so packed w2
    needs the packed-aware ``_lookup_tile_info`` (it keys on the real K and the
    float4 dtype the kernel contracts in).

    Every other dtype keeps the kernel's own adaptive ``calculate_tiling``.
    That is the only path that budgets the rhs tile against ``rhs_vmem_target``
    -- a pinned ``TileSizes`` is consumed verbatim with no clamp, so routing
    bf16/FP8 through the lookup would retune stock TPU MoE and could overflow
    VMEM at an untested shape. Widening this gate is a separate change that
    needs its own perf data.
    """
    from sglang.srt.layers.moe.pallas.gmm_v2 import calculate_tiling

    if not w2_packed:
        return calculate_tiling
    return _lookup_tile_info(lhs, w2, w2_scale, num_experts, packed_fp4=True)


def _dividing_tile_k(base_tile_info):
    """Wrap a gmm_v2 ``tile_info`` so the resolved ``tile_k`` always divides
    ``size_k`` (mirrors the kernel's N-gate, which has no K equivalent) and
    stays compatible with the rhs quantization block size.

    ``calculate_tiling`` enforces the quant-block constraint itself, but a
    pinned ``TileSizes`` (compact block-FP8 GMM1) or a tuned-lookup one
    (``_lookup_tile_info``, which every NVFP4 call reaches because
    ``TUNED_BLOCK_SIZES`` carries no fp4 entries) bypasses it. A straddling
    tile_k does not fail -- it silently loads the wrong per-block scales -- so
    re-apply the constraint here, on whatever tile_k the caller ended up with.

    This is a no-op for every configuration that already worked: unscaled bf16
    has no block size, channelwise scales have ``quant_block_size == size_k``
    (any divisor is compatible), block-FP8's 128 divides every 128-aligned
    tile, and NVFP4 at ``size_k`` a multiple of 2048 resolves ``tile_k=2048``,
    a multiple of 512. It only fires where the scale indexing would be wrong.
    """
    import dataclasses

    from sglang.srt.layers.moe.pallas.gmm_v2 import TileSizes

    def wrapped(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
        if isinstance(base_tile_info, TileSizes):
            tiles = base_tile_info
        else:
            tiles = base_tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        quant_block_size = rhs_cfgs.quant_block_size if rhs_cfgs.has_scale else None
        if dims.size_k % tiles.tile_k != 0 or not _quant_block_compatible(
            tiles.tile_k, quant_block_size
        ):
            tiles = dataclasses.replace(
                tiles,
                tile_k=_largest_compatible_tile_k(
                    dims.size_k, tiles.tile_k, quant_block_size, 128
                ),
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
    w1,  # [g,k,2*n] bf16 | [g,k/2,2*n] uint8 (packed FP4) | [g,k/8,2*n] int32 (packed int4)
    w2,  # [g,n,k]   bf16 | [g,n/2,k]   uint8 (packed FP4) | [g,n/8,k]   int32 (packed int4)
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
    w_quant_bits=0,  # 4 => w1/w2 are checkpoint-packed int4 words (W4A16)
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
        lhs_sublane_tiling,
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

    rhs_quant_dtype = None
    maybe_quantize_lhs = True
    if w_quant_bits:
        assert w_quant_bits == 4, f"unsupported w_quant_bits={w_quant_bits}"
        # W4A16: checkpoint-packed int4 rhs (8 per int32), bf16 activations.
        rhs_quant_dtype = jnp.int4
        maybe_quantize_lhs = False

    # gmm_v2 needs the routed-row count ``m*topk`` to be a multiple of its LHS
    # sublane tiling (16 for the bf16 activations this path runs on). Padding
    # the *input* tokens here keeps the permute/gather operating on an
    # already-aligned, fixed-bucket shape and removes the post-gather
    # ``jnp.pad`` + ``group_sizes`` scatter that XLA was offloading to the
    # SparseCore (and crashing on). Pad tokens route to expert 0 with zero
    # weight, so they contribute nothing to any real token and are sliced off
    # at the very end.
    m_orig = m
    m_pad = _padded_token_count(m, topk, lhs_sublane_tiling(hidden_states.dtype))
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
    # W4A16 runs GMM1 unfused: the fused gate/up split mis-applies multi-block
    # rhs scales, so the activation is applied in JAX below.
    unfused_gmm1 = compact_block_scales or bool(w_quant_bits)
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
        maybe_quantize_lhs=maybe_quantize_lhs,
        rhs_quant_dtype=rhs_quant_dtype,
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
    gmm2_tiling = _dividing_tile_k(
        _gmm2_tile_source(gmm1_out, w2, w2_scale, num_experts, w2_packed=w2_packed)
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
        maybe_quantize_lhs=maybe_quantize_lhs,
        rhs_quant_dtype=rhs_quant_dtype,
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
        w_quant_bits: int = 0,
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
            w_quant_bits,
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
        w_quant_bits: int = 0,
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
    w_quant_bits: int = 0,
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
        w_quant_bits,
        int(envs.SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD.get()),
    )
