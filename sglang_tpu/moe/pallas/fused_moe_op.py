# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py
"""Pallas fused-MoE bridge for the TPU torch_tpu platform.

This module wraps the JAX/Pallas ``gmm_v2`` grouped-matmul kernel
(``sglang_tpu.moe.pallas.gmm_v2``) and assembles it into a
single-host fused-MoE op exposed to PyTorch via
``torch_tpu._internal.pallas.jax_op``. It mirrors the pattern used by the
ragged-paged-attention v3 backend in
``sglang_tpu.attention_pallas_rpa_v3``: register one custom
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
mxfp4 path. ``"swiglu_oai"`` is supported for GPT-OSS, and
``"situ:{beta}:{linear_beta|none}"`` (built by ``fused_moe_call``) for Kimi-K3.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here —
# ``torch_tpu._internal.pallas.jax_op`` inspects ``param.annotation is
# jax.Array`` at registration time; PEP 563 would turn those into
# strings and validation would reject them.

import logging
import math
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch
from torch_tpu._internal.pallas import jax_op

from sglang.srt.environ import envs

if TYPE_CHECKING:
    import jax

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


def _shard_row_window(topk_ids_flat, expert_group_offset, local_num_experts):
    """Row range ``[start, end)`` of the expert-major sorted array this rank owns.

    Rows are sorted by expert id, so the count of rows before expert ``e`` is
    exactly ``count(topk_ids_flat < e)`` -- the same value as
    ``cumulative_sum(group_sizes, include_initial=True)[e]``, and the same bound
    ``gmm_v2`` derives internally.
    """
    import jax.numpy as jnp

    row_start = jnp.sum(topk_ids_flat < expert_group_offset).astype(jnp.int32)
    row_end = jnp.sum(topk_ids_flat < expert_group_offset + local_num_experts).astype(
        jnp.int32
    )
    return row_start, row_end


def _valid_rows_mask(row_start, row_end, unsort_idx):
    """Return a mask of which rows are owned by this rank."""
    return (row_start <= unsort_idx) & (unsort_idx < row_end)


def _drop_invalid_slots(
    hidden_states: "jax.Array",
    topk_ids: "jax.Array",
    topk_weights: "jax.Array",
    num_route_experts: int,
    zero_dropped_tokens: bool,
) -> tuple["jax.Array", "jax.Array", "jax.Array"]:
    """Route slots with an id outside ``[0, num_route_experts)`` to
    ``num_route_experts`` with zero weight (TopK marks padded tokens -1).

    That id sorts after every real row, has no group in ``group_sizes`` and
    lies outside the EP row window, so no GMM computes the slot.
    ``zero_dropped_tokens`` zeroes tokens with no live slot for the one-hot
    permute matmul, which reads every row (``0 * NaN`` would reach real rows).
    """
    import jax.numpy as jnp

    topk_ids = topk_ids.astype(jnp.int32)
    dropped = (topk_ids < 0) | (topk_ids >= num_route_experts)
    topk_ids = jnp.where(dropped, num_route_experts, topk_ids)
    topk_weights = jnp.where(dropped, 0, topk_weights)
    if zero_dropped_tokens:
        token_dropped = jnp.all(dropped, axis=1, keepdims=True)
        hidden_states = jnp.where(token_dropped, 0, hidden_states)
    return hidden_states, topk_ids, topk_weights


def _lookup_tile_info(
    lhs, rhs, rhs_scale, num_total_groups, packed_fp4=False, gmm_v2_only=False
):
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

    ``gmm_v2_only`` reads only the gmm_v2-measured table and returns None on
    a miss instead of the heuristic.

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

    from sglang_tpu.moe.pallas.gmm_v2 import TileSizes
    from sglang_tpu.moe.pallas.tuned_block_sizes import (
        get_tuned_block_sizes,
    )

    m, k = lhs.shape
    _g, _k_raw, n = rhs.shape
    # Packed FP4 stores 2 codes/byte along K, so real K = stored K * 2.
    _k = _k_raw * 2 if packed_fp4 else _k_raw
    assert _k == k, f"gmm_v2 lhs/rhs k mismatch: {k=} vs {_k=}"
    quant_block_size = k if rhs_scale is None else k // rhs_scale.shape[1]
    rhs_dtype = "float4_e2m1fn" if packed_fp4 else str(jnp.dtype(rhs.dtype).name)
    tiles = get_tuned_block_sizes(
        m=int(m),
        k=int(k),
        n=int(n),
        num_total_groups=int(num_total_groups),
        num_current_groups=int(num_total_groups),
        lhs_dtype=str(jnp.dtype(lhs.dtype).name),
        rhs_dtype=rhs_dtype,
        quant_block_size=quant_block_size,
        gmm_v2_only=gmm_v2_only,
    )
    if tiles is None:
        return None
    tm, tk, tn = tiles
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

    Every other dtype keeps the kernel's own adaptive ``calculate_tiling``
    unless gmm_v2 was measured at its exact shape (``_tuned_or_calculated``).
    """
    if not w2_packed:
        return _tuned_or_calculated(lhs, w2, w2_scale, num_experts)
    return _lookup_tile_info(lhs, w2, w2_scale, num_experts, packed_fp4=True)


def _tuned_or_calculated(lhs, rhs, rhs_scale, num_experts):
    """A ``GMM_V2_TUNED_BLOCK_SIZES`` entry for this exact shape, else
    ``calculate_tiling``. A pinned ``TileSizes`` skips the VMEM budget, so only
    shapes measured on gmm_v2 are pinned; the vendored v1 entries and the
    heuristic default never reach bf16/FP8."""
    from sglang_tpu.moe.pallas.gmm_v2 import calculate_tiling

    # Packed (INT4) weights store a different K and keep calculate_tiling.
    if rhs.shape[1] != lhs.shape[1]:
        return calculate_tiling
    tuned = _lookup_tile_info(lhs, rhs, rhs_scale, num_experts, gmm_v2_only=True)
    return tuned or calculate_tiling


# 3 pipelined rhs buffers + unpack staging tile + scales/lhs/acc/out.
_PACKED_RHS_VMEM_DIVISOR = 5


def _vmem_safe_packed_tile_n(base_tile_info):
    """Shrink ``tile_n`` for a packed sub-byte rhs so the unpack staging tile fits.

    ``calculate_tiling`` budgets the rhs from its stored bit width; the W4A16
    unpack needs about one more buffer, which overflowed VMEM at Kimi's GMM1
    (62.77 MB vs a 57.60 MB limit). No-op for bf16, FP8 and packed FP4.
    """
    import dataclasses

    import jax
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.gmm_v2 import (
        TileSizes,
        _largest_aligned_divisor_at_most,
    )

    def wrapped(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
        if isinstance(base_tile_info, TileSizes):
            tiles = base_tile_info
        else:
            tiles = base_tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        # Only pre-packed INT4 has the unpack staging tile; fp8 also reports
        # packing 4 (32-bit words) but needs none.
        if rhs_cfgs.quant_dtype != jnp.int4.dtype:
            return tiles

        rhs_bits = jax.dtypes.itemsize_bits(rhs_cfgs.dtype)
        budget = vmem_limit_bytes // _PACKED_RHS_VMEM_DIVISOR
        tile_n = tiles.tile_n
        while tile_n > 128 and tiles.tile_k * tile_n * rhs_bits // 8 > budget:
            try:
                tile_n = _largest_aligned_divisor_at_most(dims.size_n, tile_n // 2, 128)
            except ValueError:
                # Nothing smaller divides size_n; keep the kernel's choice.
                return tiles
        if tile_n == tiles.tile_n:
            return tiles
        return dataclasses.replace(tiles, tile_n=tile_n)

    return wrapped


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

    from sglang_tpu.moe.pallas.gmm_v2 import TileSizes

    def wrapped(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
        if isinstance(base_tile_info, TileSizes):
            tiles = base_tile_info
        else:
            tiles = base_tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        quant_block_size = rhs_cfgs.quant_block_size if rhs_cfgs.has_scale else None
        if getattr(rhs_cfgs, "requantize_in_vmem", False):
            quant_block_size = min(lhs_cfgs.quant_block_size, dims.size_k)
            # Activation amax spans the whole K=512 group; tile_k cannot split it.
            tiles = dataclasses.replace(
                tiles, tile_k=max(tiles.tile_k, quant_block_size)
            )
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
    ``silu`` / ``gelu`` / ``swigluoai`` / ``situ:{beta}:{linear_beta|none}``;
    SGLang spells the GPT-OSS one ``swiglu_oai``.
    """
    if activation_name.startswith(("situ:", "silu_and_mul_clamp:")):
        return activation_name
    mapping = {
        "silu": "silu",
        "silu_and_mul_clamp": "silu_and_mul_clamp",
        "gelu": "gelu",
        "swiglu_oai": "swigluoai",
        "swigluoai": "swigluoai",
    }
    if activation_name not in mapping:
        raise NotImplementedError(
            f"Unsupported activation '{activation_name}' for the Pallas "
            "fused-MoE op. Supported: silu, silu_and_mul_clamp, gelu, swiglu_oai."
        )
    return mapping[activation_name]


# Largest routed-row count for the rank permute. Its [rows, rows] compares grow
# quadratically (4 MiB of bools each at 2048 rows). On tpu7x under EP it beat
# the sort route at every measured shape up to 2048 rows (Kimi-K3, Qwen3-Coder,
# Qwen3.5) and by only ~10 us at 4096; raise the cap only with new measurements.
_RANK_PERMUTE_MAX_ROWS = 2048


class _DenseRoute(NamedTuple):
    sorted_token_ids: "jax.Array"  # i32[n]: owner token of each sorted row
    sorted_weights: "jax.Array"  # f32[n]: routing weight of each sorted row
    group_sizes: "jax.Array"  # i32[num_local_experts]
    num_owned: "jax.Array"  # i32[]: owned rows, a prefix of the sorted order


def _dense_rank_route(
    topk_ids_flat: "jax.Array",
    topk_weights: "jax.Array",
    num_experts: int,
    expert_group_offset: int,
) -> _DenseRoute:
    """Expert-major routing over this rank's local expert ids, without a sort.

    Ids outside ``[expert_group_offset, expert_group_offset + num_experts)``
    map to one sentinel past the last local expert, so the owned rows form a
    prefix of the sorted order and ``group_sizes`` covers only local experts.
    The rank of each slot comes from pairwise compares (stable, ties broken by
    slot), so the owned prefix matches a stable argsort over global ids. Exact
    integer compares rather than vllm-torchtpu's counting-sort rank matmul,
    which that repo keeps off the one-hot permute (it collapsed MTP acceptance)
    and off more than 512 buckets (an XLA miscompile).
    """
    import jax.numpy as jnp

    n = topk_ids_flat.shape[0]
    topk = n // topk_weights.shape[0]
    local_ids = topk_ids_flat - expert_group_offset
    owned = (local_ids >= 0) & (local_ids < num_experts)
    keys = jnp.where(owned, local_ids, num_experts)
    slot = jnp.arange(n, dtype=jnp.int32)
    before = (keys[None, :] < keys[:, None]) | (
        (keys[None, :] == keys[:, None]) & (slot[None, :] < slot[:, None])
    )
    sorted_pos = jnp.sum(before, axis=1, dtype=jnp.int32)
    # hit[p, s]: slot s lands at sorted position p.
    hit = sorted_pos[None, :] == slot[:, None]
    sorted_token_ids = jnp.sum(
        jnp.where(hit, (slot // topk)[None, :], 0), axis=1, dtype=jnp.int32
    )
    weights = topk_weights.reshape(-1).astype(jnp.float32)
    sorted_weights = jnp.sum(jnp.where(hit, weights[None, :], 0.0), axis=1)
    # The sentinel one-hots to an all-zero row.
    group_sizes = (keys[:, None] == jnp.arange(num_experts)[None, :]).sum(
        axis=0, dtype=jnp.int32
    )
    num_owned = jnp.sum(owned, dtype=jnp.int32)
    return _DenseRoute(sorted_token_ids, sorted_weights, group_sizes, num_owned)


def _dense_route_combine(
    gmm2_out: "jax.Array", route: _DenseRoute, num_tokens: int
) -> "jax.Array":
    """Weighted top-k combine of the expert-major GMM2 rows of ``route``.

    Rows past ``route.num_owned`` were never written (no zero-init), so they
    are masked before they can reach the sum as ``0 * NaN``.
    """
    import jax
    import jax.numpy as jnp

    owned_rows = jnp.arange(gmm2_out.shape[0]) < route.num_owned
    gmm2_out = jnp.where(owned_rows[:, None], gmm2_out, 0)
    unpermute_matrix = (
        jax.nn.one_hot(route.sorted_token_ids, num_tokens, dtype=jnp.float32)
        * route.sorted_weights[:, None]
    )
    return jax.lax.dot_general(
        unpermute_matrix,
        gmm2_out.astype(jnp.float32),
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


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
    w1_channel_scale=None,
    w2_channel_scale=None,
    expert_group_offset_tensor=None,
):
    """Production fused-MoE kernel body. Exported by ``_build_fused_moe_op``.

    Module-scoped (rather than nested in ``_build_fused_moe_op``) so unit
    tests can call it directly with JAX inputs and exercise the exact
    permute / GMM1 / activation / GMM2 / un-permute path that production
    uses.
    """
    import jax
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.gmm_v2 import (
        TileSizes,
        apply_act_fn,
        calculate_tiling,
        gmm_v2,
        lhs_sublane_tiling,
    )

    m, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    out_dtype = hidden_states.dtype
    if expert_group_offset_tensor is not None:
        # Rank-lifted EP window offset (see _resolve_expert_group_offset): a
        # traced scalar that every use below consumes through jnp ops, so the
        # MLIR is the same on every EP rank.
        expert_group_offset = expert_group_offset_tensor.astype(jnp.int32).reshape(())

    # One token with combine (bs=1 decode) on packed MXFP4: one kernel that
    # streams only the hit experts (single_token_moe.py) replaces route
    # metadata + two gmm_v2 calls + activation + combine. It reads topk_ids
    # directly, so it does not depend on the permute path chosen below
    # (permute_dense_threshold).
    if m == 1 and not no_combine:
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.moe.pallas import single_token_moe

        if (
            tpu_envs.SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL.get()
            and single_token_moe.validated(
                hidden_size,
                w1.shape[2] // 2,
                global_num_experts or num_experts,
                topk,
                activation_name,
            )
            and single_token_moe.eligible(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                w1_bias,
                w2_bias,
                w1_channel_scale,
                w2_channel_scale,
            )
        ):
            return single_token_moe.single_token_moe(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_ids,
                topk_weights,
                offset=(expert_group_offset if global_num_experts > num_experts else 0),
                act=_fuse_act_name(activation_name),
                hidden_size=hidden_size,
            ).astype(out_dtype)

    # uint8 rhs = packed FP4 (2 float4 codes/byte along K); gmm_v2 unpacks to
    # float4_e2m1fn in-kernel when told packed_fp4=True (real K = shape[1] * 2).
    # Block size (32 MXFP4, 512 NVFP4) is data-driven from the scale.
    w1_packed = w1.dtype == jnp.uint8
    w2_packed = w2.dtype == jnp.uint8

    rhs_quant_dtype = None
    maybe_quantize_lhs = True
    dequantize_int4_in_vmem = False
    use_vmem_fp8 = w1_channel_scale is not None or w2_channel_scale is not None
    if use_vmem_fp8 and (
        w_quant_bits != 4
        or w1_channel_scale is None
        or w2_channel_scale is None
        or w1_bias is not None
        or w2_bias is not None
    ):
        raise ValueError(
            "VMEM FP8 requires two channel-scale tensors, INT4 weights and no biases"
        )
    if w_quant_bits:
        assert w_quant_bits == 4, f"unsupported w_quant_bits={w_quant_bits}"
        # vmem-bf16: dequantize groups in VMEM, one dot per tile; default:
        # scale after each 32-K group product.
        from sglang.srt.layers.moe.tpu_moe_utils import (
            INT4_MOE_MODE_VMEM_BF16,
            resolve_int4_moe_mode,
        )

        dequantize_int4_in_vmem = (
            not use_vmem_fp8 and resolve_int4_moe_mode() == INT4_MOE_MODE_VMEM_BF16
        )
        rhs_quant_dtype = jnp.int4
        maybe_quantize_lhs = use_vmem_fp8

    from sglang_tpu.envs import envs as tpu_envs

    if tpu_envs.SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL.get():
        from sglang_tpu.moe.pallas.moe_stream import maybe_moe_stream

        # One kernel streaming the selected experts' weights; None outside its case.
        out = maybe_moe_stream(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            activation_name=activation_name,
            no_combine=no_combine,
            use_group_offset=global_num_experts > num_experts,
            w_quant_bits=w_quant_bits,
        )
        if out is not None:
            return out

    # gmm_v2 needs the routed-row count ``m*topk`` to be a multiple of its LHS
    # sublane tiling (16 for the bf16 activations this path runs on). Padding
    # the *input* tokens here keeps the permute/gather operating on an
    # already-aligned, fixed-bucket shape and removes the post-gather
    # ``jnp.pad`` + ``group_sizes`` scatter that XLA was offloading to the
    # SparseCore (and crashing on). Padding has zero weight.
    m_orig = m
    topk_ids_orig, topk_weights_orig = topk_ids, topk_weights
    m_pad = _padded_token_count(m, topk, lhs_sublane_tiling(hidden_states.dtype))
    if m_pad != m:
        hidden_states = jnp.pad(hidden_states, ((0, m_pad - m), (0, 0)))
        topk_ids = jnp.pad(
            topk_ids, ((0, m_pad - m), (0, 0)), constant_values=-1
        )  # pad -> dropped
        topk_weights = jnp.pad(
            topk_weights, ((0, m_pad - m), (0, 0))
        )  # pad -> weight 0
        m = m_pad
    hidden_states, topk_ids, topk_weights = _drop_invalid_slots(
        hidden_states,
        topk_ids,
        topk_weights,
        max(global_num_experts, num_experts),
        zero_dropped_tokens=m < permute_dense_threshold,
    )

    # Routing: build a permutation that puts rows of hidden_states in
    # expert-major order so a single grouped-matmul covers GMM1 / GMM2.
    topk_ids_flat = topk_ids.reshape(-1).astype(jnp.int32)  # [m*topk]
    use_group_offset = global_num_experts > num_experts

    # Decide whether to use the dense permute/un-permute paths based on the
    # number of rows (m) and the number of routed rows (m*topk).
    use_dense_permute = m < permute_dense_threshold
    use_dense_unpermute = (m * topk) < permute_dense_threshold and not no_combine
    # One real token (bs=1 decode) on the dense path (threshold > m_pad * topk,
    # combine on): every routed row is token 0, so the permute is a broadcast
    # and only the expert-major metadata is needed. One SMEM kernel builds it
    # in place of the sort/one-hot glue.
    single_token = use_dense_permute and use_dense_unpermute and m_orig == 1

    row_start = row_end = None
    if single_token:
        from sglang_tpu.moe.pallas.route_meta import single_token_route_meta

        # Local rows (stable, expert-major) form a prefix of the n routed rows;
        # group_sizes is over the local experts only, so no group_offset.
        w_sorted_f32, group_sizes = single_token_route_meta(
            topk_ids_orig,
            topk_weights_orig,
            n=m * topk,
            offset=expert_group_offset if use_group_offset else 0,
            num_experts=num_experts,
        )
    else:
        # argsort is stable enough on TPU for our usage; the gmm kernel
        # only cares about contiguous expert runs.
        argsort_idx = jnp.argsort(topk_ids_flat)
        # Slot i belongs to token i // topk; a divide avoids an XLA gather.
        sorted_token_ids = (argsort_idx // topk).astype(jnp.int32)

        # gmm_v2 only reads and writes the rows of expert groups at or after
        # ``group_offset``, so this rank's routed-row window is all the permute
        # has to produce. Computed here (not at the ``group_sizes`` site below)
        # so the gather can start as soon as the sort is done.
        if use_group_offset:
            row_start, row_end = _shard_row_window(
                topk_ids_flat, expert_group_offset, num_experts
            )

    if single_token:
        x_sorted = jnp.broadcast_to(hidden_states[:1], (m * topk, hidden_size))
    elif use_dense_permute:
        permute_matrix = jax.nn.one_hot(sorted_token_ids, m, dtype=hidden_states.dtype)
        x_sorted = jax.lax.dot_general(
            permute_matrix,
            hidden_states,
            (((1,), (0,)), ((), ())),
            preferred_element_type=hidden_states.dtype,
        )
    elif use_sc_gather:
        from sglang_tpu.moe.pallas.sparse_core.ragged_gather_v2 import (
            ragged_gather_v2,
        )

        n_sorted = sorted_token_ids.shape[0]
        gather_start = jnp.array([0], dtype=jnp.int32)
        gather_end = jnp.array([n_sorted], dtype=jnp.int32)
        x_sorted = ragged_gather_v2(
            hidden_states,
            sorted_token_ids,
            gather_start,
            gather_end,
        )
    else:
        x_sorted = hidden_states[sorted_token_ids]  # [m*topk, k]

    # Multi-token dense decode: route on local expert ids with a sort-free rank
    # permute. The routing above is then unused, and XLA drops it as dead code.
    # A single token keeps the path above.
    dense_route = None
    if (
        use_dense_permute
        and use_dense_unpermute
        and m_orig > 1
        and m * topk <= _RANK_PERMUTE_MAX_ROWS
    ):
        dense_route = _dense_rank_route(
            topk_ids_flat,
            topk_weights,
            num_experts,
            expert_group_offset if use_group_offset else 0,
        )
        x_sorted = jax.lax.dot_general(
            jax.nn.one_hot(dense_route.sorted_token_ids, m, dtype=hidden_states.dtype),
            hidden_states,
            (((1,), (0,)), ((), ())),
            preferred_element_type=hidden_states.dtype,
        )
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
    clamp_limit = (
        float(activation_name.split(":", 1)[1])
        if activation_name.startswith("silu_and_mul_clamp:")
        else None
    )
    unfused_gmm1 = compact_block_scales or bool(w_quant_bits) or clamp_limit is not None
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
    gmm1_tiling = _dividing_tile_k(
        _vmem_safe_packed_tile_n(
            gmm1_tile_info
            or (
                calculate_tiling
                if w1_packed
                else _tuned_or_calculated(x_sorted, w1, w1_scale, num_experts)
            )
        )
    )
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

    if single_token:
        group_offset = None
    else:
        route_num_experts = global_num_experts if use_group_offset else num_experts
        # one_hot+sum is cheaper on TPU than bincount.
        group_sizes = jax.nn.one_hot(
            topk_ids_flat, route_num_experts, dtype=jnp.int32
        ).sum(axis=0)
        group_offset = (
            jnp.asarray(expert_group_offset, dtype=jnp.int32).reshape(1)
            if use_group_offset
            else None
        )

    # The TensorCore path requires zero-initializing, while the SparseCore
    # combine does not: the masked EP combine skips masked rows entirely, and
    # the all-local combine zeroes rows of zero weight, which every dropped
    # slot has, so rows the GMM never wrote never reach the sum. no_combine
    # gathers every row back, so it zero-initializes on both paths. The dense
    # un-permute multiplies every row (unwritten ones by weight 0), so
    # unwritten rows must be zeros, not garbage: 0 * NaN would poison the sum.
    if use_sc_gather and not no_combine and not use_dense_unpermute:
        zero_init = False
    else:
        zero_init = True
    if single_token:
        # The single-token combine masks rows past the local prefix itself.
        zero_init = False

    if dense_route is not None:
        # Local groups only; the combine masks the rows past the owned prefix.
        group_sizes, group_offset, zero_init = dense_route.group_sizes, None, False

    gmm1_out = gmm_v2(
        lhs=x_sorted,
        rhs=w1,
        group_sizes=group_sizes,
        rhs_scale=w1_scale,
        rhs_bias=w1_bias,
        rhs_channel_scale=w1_channel_scale,
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
        dequantize_rhs=dequantize_int4_in_vmem,
    )
    if clamp_limit is not None:
        gate, up = jnp.split(gmm1_out, 2, axis=-1)
        gate = jnp.minimum(gate, clamp_limit)
        up = jnp.clip(up, -clamp_limit, clamp_limit)
        gmm1_out = (jax.nn.silu(gate.astype(jnp.float32)) * up).astype(out_dtype)
    elif unfused_gmm1:
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
        _vmem_safe_packed_tile_n(
            _gmm2_tile_source(gmm1_out, w2, w2_scale, num_experts, w2_packed=w2_packed)
        )
    )
    gmm2_out = gmm_v2(
        lhs=gmm1_out,
        rhs=w2,
        group_sizes=group_sizes,
        rhs_scale=w2_scale,
        rhs_bias=w2_bias,
        rhs_channel_scale=w2_channel_scale,
        group_offset=group_offset,
        zero_initialize=zero_init,
        preferred_element_type=out_dtype,
        tile_info=gmm2_tiling,
        packed_fp4=w2_packed,
        maybe_quantize_lhs=maybe_quantize_lhs,
        rhs_quant_dtype=rhs_quant_dtype,
        dequantize_rhs=dequantize_int4_in_vmem,
    )[
        :,
        :hidden_size,
    ]  # [m*topk, k]

    # Un-permute back to token-major order. Models that apply a post-expert
    # norm ask the runner for no_combine output so they can RMSNorm each top-k
    # slot before the weighted reduction.
    if single_token:
        # Rows past the local prefix carry weight 0 and are uninitialized; a
        # zero-weight row adds nothing, so the weight is an exact mask.
        gmm2_rows = jnp.where(w_sorted_f32[:, None] != 0, gmm2_out, 0)
        combined = jnp.sum(
            gmm2_rows.astype(jnp.float32) * w_sorted_f32[:, None],
            axis=0,
            keepdims=True,
        )
        return combined.astype(out_dtype)
    if use_dense_unpermute:
        if dense_route is not None:
            combined = _dense_route_combine(gmm2_out, dense_route, m)
            return combined[:m_orig].astype(out_dtype)
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
            from sglang_tpu.moe.pallas.sparse_core.dense_gather_reduce import (
                dense_gather_reduce,
            )
            from sglang_tpu.moe.pallas.sparse_core.ragged_gather_reduce_v2 import (
                ragged_gather_reduce,
            )

            if use_group_offset:
                combined = ragged_gather_reduce(
                    gmm2_out,
                    unsort_idx.astype(jnp.int32),
                    topk_weights.reshape(-1),
                    _valid_rows_mask(row_start, row_end, unsort_idx),
                    topk,
                )
            else:
                # All experts local: the maskless dense variant. Dropped slots
                # have zero weight and rows the GMM never wrote; zeroing
                # zero-weight rows keeps 0 * NaN out of the sum.
                combined = dense_gather_reduce(
                    gmm2_out,
                    unsort_idx.astype(jnp.int32),
                    topk_weights,
                    topk,
                    topk_wgt_zero_nan=True,
                )
            return combined[:m_orig].astype(out_dtype)

        # no_combine: just gather, without the weighted top-k sum.
        from sglang_tpu.moe.pallas.sparse_core.ragged_gather_v2 import (
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
        # Zero the rows this rank did not compute (dropped and non-local slots),
        # so a later post-norm + combine with the original router weights never
        # reads them.
        if use_group_offset:
            valid_rows_mask = _valid_rows_mask(row_start, row_end, unsort_idx)[
                : m_orig * topk
            ].reshape(m_orig, topk, 1)
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
        w1_channel_scale: Optional[jax.Array] = None,
        w2_channel_scale: Optional[jax.Array] = None,
        expert_group_offset_tensor: Optional[jax.Array] = None,
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
            w1_channel_scale,
            w2_channel_scale,
            expert_group_offset_tensor,
        )

    custom_op = jax_op(op_name, _wrapped)

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
        w1_channel_scale: Optional[torch.Tensor] = None,
        w2_channel_scale: Optional[torch.Tensor] = None,
        expert_group_offset_tensor: Optional[torch.Tensor] = None,
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


def _resolve_expert_group_offset(
    expert_group_offset: int,
    global_num_experts: int,
    num_local_experts: int,
) -> tuple[int, Optional[torch.Tensor]]:
    """Lift the EP rank's expert window offset into a tensor input under
    ``trace_rank`` so every EP rank traces identical MLIR (``spmd_safe``).

    Outside a rank-parameterized trace (eager, dynamo, EP=1) the offset stays a
    static int and the op signature is unchanged. The lift relies on
    ``moe_ep_rank == world_rank % ep_size``: ``direct_compile`` only enables
    ``lift_rank`` when the TP group is the whole world, so the EP group is a
    contiguous slice of it, and ``traced_rank`` raises if that ever breaks.
    """
    if global_num_experts <= num_local_experts:
        # EP=1: no window, and no new op input so the trace is unchanged.
        return expert_group_offset, None
    from sglang_tpu.rank_local import traced_rank

    ep_size = global_num_experts // num_local_experts
    ep_rank = expert_group_offset // num_local_experts
    rank = traced_rank(ep_rank, group_size=ep_size)
    if rank is None:
        return expert_group_offset, None
    return 0, (rank * num_local_experts).to(dtype=torch.int32).reshape(1)


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
    w1_channel_scale: Optional[torch.Tensor] = None,
    w2_channel_scale: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """Invoke the Pallas fused-MoE op.

    Inputs use gmm_v2's ``(g, k, n)`` LHS-RHS convention:
      * ``hidden_states``: ``[num_tokens, hidden_size]``
      * ``w1``: ``[num_experts, hidden_size, 2 * intermediate]``  (gate || up)
      * ``w2``: ``[num_experts, intermediate, hidden_size]``
      * ``topk_weights``: ``[num_tokens, topk]``  (routing scores, dtype of x)
      * ``topk_ids``: ``[num_tokens, topk]``  (int32 expert ids)

    Slots with a negative or out-of-range id (TopK's padded tokens) are
    dropped: they contribute zero, and a token with no live slot outputs zeros.

    Returns ``[num_tokens, hidden_size]`` in ``hidden_states.dtype``,
    or the un-combined per-expert ``[num_tokens, topk, hidden_size]``
    when ``no_combine`` is True.
    """
    if activation == "situ":
        # SiTU (beta, linear_beta) are the runner's gemm1 (alpha, clamp_limit).
        if gemm1_alpha is None:
            raise ValueError("SiTU fused MoE requires gemm1_alpha (SiTU beta).")
        linear_beta = "none" if gemm1_clamp_limit is None else gemm1_clamp_limit
        activation = f"situ:{gemm1_alpha}:{linear_beta}"
    op = _build_fused_moe_op()
    # Routed rows for this (bucket-padded) call; static per compiled graph.
    routed_rows = int(hidden_states.shape[0]) * int(topk)
    use_sc_gather = int(
        envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.get()
        and routed_rows >= envs.SGLANG_TPU_FUSED_MOE_SC_GATHER_MIN_ROWS.get()
    )
    expert_group_offset_int, expert_group_offset_tensor = _resolve_expert_group_offset(
        expert_group_offset,
        global_num_experts,
        int(w1.shape[0]),
    )
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
        expert_group_offset_int,
        use_sc_gather,
        w_quant_bits,
        int(envs.SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD.get()),
        w1_channel_scale,
        w2_channel_scale,
        expert_group_offset_tensor,
    )
