# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.environ import envs
from sglang.srt.layers.moe.pallas.gmm_v2 import TileSizes, recover_quant_block_size

# Hack to workaround libtpu bug in 0.0.46 and 0.0.47 that causes
# online-quantized LHS to be recomputed per use.
# TODO: Remove once the issue gets resolved in libtpu version > 0.0.47
# (likely 0.0.48).
_LIBTPU_WORKAROUND_FOR_ONLINE_QUANT_SCALE = (
    envs.SGLANG_TPU_LIBTPU_WORKAROUND_FOR_ONLINE_QUANT_SCALE.get()
)


def get_maybe_quantize_lhs(
    rhs_dtype=None,
    rhs_quant_block_size: int | None = None,
    lhs_quant_block_size: int | None = None,
) -> bool:
    """Return whether to online-quantize LHS activations.

    FP4 weights use W4A16 when RHS scale blocks are smaller than the LHS
    online-quant block; otherwise they use W4A8. Other quantized RHS dtypes keep
    the existing online-quantized LHS path.
    """
    if rhs_dtype is not None and jnp.dtype(rhs_dtype) == jnp.float4_e2m1fn:
        if rhs_quant_block_size is None or lhs_quant_block_size is None:
            return False
        return rhs_quant_block_size >= lhs_quant_block_size
    return True


# Util.
def align_to(x, a):
    return pl.cdiv(x, a) * a


# Define data classes.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef:
    weight: Any
    scale: Any | None
    bias: Any | None

    def get_weight(self):
        return self.weight[...]

    def get_scale(self):
        return self.scale[...]

    def get_bias(self):
        return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef:
    """Wraps gate and up WeightsRef for fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self):
        return jnp.concatenate([self.gate.weight[...], self.up.weight[...]], axis=-1)

    def get_scale(self):
        return jnp.concatenate([self.gate.scale[...], self.up.scale[...]], axis=-1)

    def get_bias(self):
        return jnp.concatenate([self.gate.bias[...], self.up.bias[...]], axis=-1)


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_bias: bool = False
    has_scale: bool = False
    packing: int = 1
    num_quant_blocks: int = 1


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool
    fuse_act: str | None = None
    has_post_norm: bool = False

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        if self.fuse_act is None:
            return self.dims.size_n
        return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes]


def apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
    """Apply fused activation (split gate/up, activate, multiply)."""
    if fuse_act is None:
        return acc
    acc_gate, acc_up = jnp.split(acc, 2, -1)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "silu_and_mul_clamp":
            limit = 10.0
            acc_gate = jnp.minimum(acc_gate, limit)
            acc_up = jnp.clip(acc_up, min=-limit, max=limit)
            return jax.nn.silu(acc_gate) * acc_up
        case str() if fuse_act.startswith("silu_and_mul_clamp:"):
            _, limit_str = fuse_act.split(":")
            limit = float(limit_str)
            acc_gate = jnp.minimum(acc_gate, limit)
            acc_up = jnp.clip(acc_up, min=-limit, max=limit)
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "swigluoai":
            limit = 7.0
            alpha = 1.702
            acc_gate = jnp.clip(acc_gate, max=limit)
            acc_up = jnp.clip(acc_up, min=-limit, max=limit)
            return (acc_gate * jax.nn.sigmoid(alpha * acc_gate)) * (acc_up + 1)
        case _:
            raise NotImplementedError(f"Unsupported fuse_act: {fuse_act}")


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, gm_id: jax.Array, _: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, gm_id: jax.Array, n_id: jax.Array, k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, gm_id: jax.Array, n_id: jax.Array, _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, gm_id: jax.Array, n_id: jax.Array, k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        b_id = (k_id * self.cfgs.tiles.tile_k) // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_id // self.cfgs.num_quant_blocks_per_tile_k
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, gm_id: jax.Array, n_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(0) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k // cfgs.rhs_cfgs.packing, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (None, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


# Define kernels.
def inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: WeightsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n] or [tile_m, tile_n] in scatter_mode
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, partial_out_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
    gs_gate_ref=None,
    gs_up_ref=None,
    scatter_mode: bool = False,
    _k_id: int | None = None,
    _num_k: int | None = None,
    _gm_id: int | None = None,
    _n_id: int | None = None,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

    tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
    Additionally, m_start and m_end does not have to align with tile boundaries
    [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
    [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
    invalid data and needs to be masked out.

    Args:
        tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
        tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
            g_id is the group associated with lhs[m_start:m_end, :]
        tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
        partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
            Will be initialized to zero if this is first tile for grid[n_id, :, :].
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """
    # Do not feed an unquantized LHS directly into MXU with a quantized RHS:
    # that mixed-dtype matmul has produced numerically wrong TPU results
    # (~70% cos-sim vs. truth). The quantized RHS path below avoids that in
    # both supported cases: W4A8/W8A8 online-quantizes LHS before matmul,
    # while W4A16 promotes/dequantizes RHS to bf16 before matmul.
    # Small FP4 RHS scale blocks are folded into RHS and requantized to the
    # LHS quant dtype before matmul so their per-block scales are not dropped.
    _resolved_gm_id = pl.program_id(0) if _gm_id is None else _gm_id

    def _matmul(is_first_k_step: bool, is_last_k_step: bool):
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[...]
        tiled_rhs = tiled_rhs_ref.get_weight()

        num_quant_blocks_per_tile_k = cfgs.num_quant_blocks_per_tile_k

        # When rhs is packed (quantized dtype packed into uint32), unpack it
        # back to the original dtype using pltpu.bitcast which operates on K
        # axis. This expands the K dimension back to tile_k.
        if cfgs.rhs_cfgs.packing > 1:
            tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.quant_dtype)

        def _valid_m_mask(shape):
            gm_id = _resolved_gm_id
            m_start = metadata_ref.gm_id_to_m_offset[gm_id]
            m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
            m_start_local = m_start - m_offset
            m_end_local = m_end - m_offset
            iota = lax.broadcasted_iota(jnp.int32, shape, 0)
            return jnp.logical_and(m_start_local <= iota, iota < m_end_local)

        def _get_scale_slice(b_id, start_n, end_n):
            rhs_scale = tiled_rhs_ref.get_scale()
            return rhs_scale[..., b_id, :, start_n:end_n]

        def _online_quantize(block, axis, quant_dtype):
            if jnp.issubdtype(quant_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(quant_dtype).max)
            else:
                dtype_max = float(jnp.iinfo(quant_dtype).max)

            block_f32 = block.astype(jnp.float32)
            amax = jnp.max(jnp.abs(block_f32), axis=axis, keepdims=True)
            amax_safe = jnp.where(amax == 0, jnp.ones_like(amax), amax)
            block_q = (block_f32 / amax_safe * dtype_max).astype(quant_dtype)
            block_scale = amax / dtype_max

            if _LIBTPU_WORKAROUND_FOR_ONLINE_QUANT_SCALE:
                # This is a noop to workaround libtpu bug in 0.0.46 and 0.0.47 that causes
                # "block_scale = amax / dtype_max" to be recomputed per use, which
                # leads to performance regression. The hack is to route the block scale
                # through a select, which breaks the pattern to prevent XLA from
                # rematerializing.
                # TODO: Remove once the issue gets resolved in libtpu version > 0.0.47
                # (likely 0.0.48).
                block_scale = jnp.where(
                    amax == 0, jnp.zeros_like(block_scale), block_scale
                )
            return block_q, block_scale

        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
            mask_lhs = lax.broadcasted_iota(jnp.int32, tiled_lhs.shape, 1) < valid_k
            tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)

        if cfgs.rhs_cfgs.quant_dtype is None:
            # Unquantized RHS matmul path.
            acc_list = []
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            rhs_qbs = cfgs.rhs_cfgs.quant_block_size
            rhs_tile_n = tiled_rhs.shape[-1]
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
                for b_id in range(num_quant_blocks_per_tile_k):
                    k_start = b_id * rhs_qbs
                    k_end = (b_id + 1) * rhs_qbs
                    partial_result = jnp.matmul(
                        tiled_lhs[:, k_start:k_end],
                        tiled_rhs[k_start:k_end, start_n:end_n],
                        preferred_element_type=jnp.float32,
                    )
                    if cfgs.rhs_cfgs.has_scale:
                        rhs_scale_slice = _get_scale_slice(b_id, start_n, end_n)
                        partial_result *= rhs_scale_slice
                    acc_n = acc_n + partial_result
                acc_list.append(acc_n.astype(acc_ref.dtype))
            acc = jnp.concatenate(acc_list, axis=1)
        else:
            # Quantized RHS path. When one stored RHS scale covers a full LHS
            # quant block, apply it after matmul. Otherwise fold the per-slice
            # RHS scales into RHS before matmul so small scale blocks are not
            # dropped. Online quantization avoids reciprocal scales; zero
            # blocks use amax_safe=1 and scale=0.
            q_block_size: int | None = cfgs.lhs_cfgs.quant_block_size
            fp8_activation_quant = cfgs.lhs_cfgs.quant_dtype is not None
            rhs_qbs = cfgs.rhs_cfgs.quant_block_size
            rhs_scale_matches_lhs_block = (
                rhs_qbs >= q_block_size and rhs_qbs % q_block_size == 0
            )
            apply_rhs_scale_after_matmul = cfgs.rhs_cfgs.has_scale and (
                rhs_scale_matches_lhs_block
            )

            # Without n outer loop, result of quantized matmul becomes available only
            # at the last iteration of the loop. This means [tile_m, tile_n] value
            # needs to be stored until the last iteration. By adding n outer loop,
            # result of [tile_m, mxu_size] becomes available at the end of every k
            # inner loop which can be used to pipeline subsequent VPU or VST ops with
            # MXU ops for the next [tile_m, mxu_size].
            acc_list = []
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            rhs_tile_n = tiled_rhs.shape[-1]
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)

                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                    if cfgs.rhs_cfgs.has_scale and not apply_rhs_scale_after_matmul:
                        scaled_rhs_slices = []
                        rhs_k_start = start_k
                        while rhs_k_start < end_k:
                            b_id = rhs_k_start // rhs_qbs
                            rhs_k_end = min((b_id + 1) * rhs_qbs, end_k)
                            local_k_start = rhs_k_start - start_k
                            local_k_end = rhs_k_end - start_k
                            rhs_scale_slice = _get_scale_slice(b_id, start_n, end_n)
                            block_rhs_slice = block_rhs[local_k_start:local_k_end]
                            scaled_rhs_slice = block_rhs_slice.astype(
                                jnp.bfloat16
                            ) * rhs_scale_slice.astype(jnp.bfloat16)
                            scaled_rhs_slices.append(scaled_rhs_slice)
                            rhs_k_start = rhs_k_end
                        block_rhs_for_quant = jnp.concatenate(scaled_rhs_slices, axis=0)
                    else:
                        block_rhs_for_quant = block_rhs.astype(jnp.bfloat16)

                    if fp8_activation_quant:
                        block_lhs_for_matmul, lhs_scale = _online_quantize(
                            block_lhs, axis=1, quant_dtype=cfgs.lhs_cfgs.quant_dtype
                        )
                        if cfgs.rhs_cfgs.has_scale and not apply_rhs_scale_after_matmul:
                            # Fold small RHS scale blocks into RHS first, then
                            # requantize to the activation quant dtype so
                            # matmul uses fp8 x fp8.
                            block_rhs_for_matmul, rhs_online_scale = _online_quantize(
                                block_rhs_for_quant,
                                axis=0,
                                quant_dtype=cfgs.lhs_cfgs.quant_dtype,
                            )
                        else:
                            block_rhs_for_matmul = block_rhs
                            rhs_online_scale = None
                    else:
                        block_lhs_for_matmul = block_lhs.astype(jnp.bfloat16)
                        block_rhs_for_matmul = block_rhs_for_quant

                    block_acc = jnp.matmul(
                        block_lhs_for_matmul,
                        block_rhs_for_matmul,
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)

                    if fp8_activation_quant:
                        block_acc *= lhs_scale.astype(acc_ref.dtype)
                        if rhs_online_scale is not None:
                            block_acc *= rhs_online_scale.astype(acc_ref.dtype)

                    if apply_rhs_scale_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_slice = _get_scale_slice(b_id, start_n, end_n)
                        block_acc *= rhs_scale_slice.astype(acc_ref.dtype)

                    acc_n += block_acc

                acc_list.append(acc_n)
            acc = jnp.concatenate(acc_list, axis=1)

        if not is_first_k_step:
            acc += acc_ref[...]

        if is_last_k_step:
            if cfgs.rhs_cfgs.has_scale and gs_gate_ref is not None:
                _gid = metadata_ref.gm_id_to_group_id[_resolved_gm_id]
                if cfgs.fuse_act is not None:
                    n_iota = lax.broadcasted_iota(jnp.int32, acc.shape, 1)
                    global_scale = jnp.where(
                        n_iota >= acc.shape[1] // 2,
                        gs_up_ref[_gid],
                        gs_gate_ref[_gid],
                    )
                else:
                    global_scale = gs_gate_ref[_gid]
                acc *= global_scale.astype(acc.dtype)

            if cfgs.rhs_cfgs.has_bias:
                acc += tiled_rhs_ref.get_bias().astype(acc.dtype)

            acc = apply_act_fn(acc, cfgs.fuse_act)

            gm_id = _resolved_gm_id
            # Mask out rows that does not belong to the current group.
            m_start = metadata_ref.gm_id_to_m_offset[gm_id]
            m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
            m_start_local = m_start - m_offset
            m_end_local = m_end - m_offset

            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)

            if scatter_mode:
                # In scatter mode, write masked acc directly to 2D output.
                # No 3D reshape needed — avoids VMEM tiling mismatch with DMA.
                # No partial_out needed — each scatter tile is independent.
                acc_masked = jnp.where(mask, acc, 0)
                tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)
            else:
                acc_masked = jnp.where(mask, acc, 0).reshape(tiled_out_ref.shape)

                # Write the final output to the output ref.
                tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

                # partial_out is n-aware: use n_id to index into the correct
                # n-tile slice of partial_out_ref (shape: sls × total_n).
                n_id = pl.program_id(1) if _n_id is None else _n_id
                n_offset = n_id * cfgs.tiles.tile_n
                tile_n = cfgs.tiles.tile_n
                sls = cfgs.dims.size_lhs_sublane
                partial_out_zeros = jnp.zeros(
                    (sls, tile_n), dtype=partial_out_ref.dtype
                )

                # Accumulate the partial output from the previous gm step
                # at the same n position.
                tiled_out_ref[0] += jnp.where(
                    gm_id == 0,
                    partial_out_zeros,
                    partial_out_ref[:, pl.ds(n_offset, tile_n)],
                )

                # Consider following case where size_lhs_sublane = 4, number denotes group
                # id and | denotes boundaries between sublanes:
                # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
                #
                # Assuming group id of current step is 1, current step will not completely
                # fill size_lhs_sublane rows and will be revisited at the next step. By
                # storing the partial rows into the partial_out_ref, the next step can
                # read them and accumulate to them.  Additionally, for group id of 2,
                # since it completely fills the size_lhs_sublane rows, we need to zero out
                # partial_out_ref to avoid numeric error for group 3.
                last_row = m_end_local // cfgs.dims.size_lhs_sublane
                partial_out_ref[:, pl.ds(n_offset, tile_n)] = jnp.where(
                    m_end_local % cfgs.dims.size_lhs_sublane == 0,
                    partial_out_zeros,
                    tiled_out_ref[last_row],
                )
        else:
            acc_ref[...] = acc

    # Define matmul wrapper functions.
    @jax.named_scope("matmul_first_last")
    def matmul_first_last():
        _matmul(is_first_k_step=True, is_last_k_step=True)

    @jax.named_scope("matmul_first")
    def matmul_first():
        _matmul(is_first_k_step=True, is_last_k_step=False)

    @jax.named_scope("matmul")
    def matmul():
        _matmul(is_first_k_step=False, is_last_k_step=False)

    @jax.named_scope("matmul_last")
    def matmul_last():
        _matmul(is_first_k_step=False, is_last_k_step=True)

    # Select and execute matmul function based on the current step.
    num_k = pl.num_programs(2) if _num_k is None else _num_k
    k_id = pl.program_id(2) if _k_id is None else _k_id

    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)

    lax.cond(
        is_first_k_step,
        lambda: lax.cond(
            is_last_k_step,
            matmul_first_last,
            matmul_first,
        ),
        lambda: lax.cond(
            is_last_k_step,
            matmul_last,
            matmul,
        ),
    )


@jax.named_scope("fill_metadata")
def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m offsets
            for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(
            cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset
        )

        metadata_ref.gm_id_to_group_id[tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


@jax.named_scope("zero_out_start")
def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
    """Zero out output rows that are not used in the computation."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    assert num_lanes == zero_ref.shape[-1]
    zero_ref[...] = jnp.zeros_like(zero_ref)

    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    row_size = zero_dma.shape[0]

    compute_start = metadata_ref.gm_id_to_m_offset[0]
    compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

    left_zero_start = 0
    left_zero_end = compute_start // dims.size_lhs_sublane
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = pl.cdiv(left_zero_size, row_size)

    right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    right_zero_size = right_zero_end - right_zero_start
    right_num_loops = pl.cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        # Static loop. Will be unrolled during compile time.
        for n_start in range(0, out_ref.shape[-1], num_lanes):
            n_end = n_start + num_lanes
            pltpu.make_async_copy(
                src_ref=zero_dma.at[pl.ds(0, dma_size)],
                dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
                sem=semaphore_ref.at[0],
            ).start(priority=1)

        return zero_size + dma_size

    @jax.named_scope("left_fill_zero")
    def left_fill_zero(i, zero_size):
        return fill_zero(i, zero_size, start=left_zero_start, end=left_zero_end)

    @jax.named_scope("right_fill_zero")
    def right_fill_zero(i, zero_size):
        return fill_zero(i, zero_size, start=right_zero_start, end=right_zero_end)

    zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
    zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
    return zero_size


@jax.named_scope("zero_out_end")
def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    pltpu.make_async_copy(
        src_ref=out_dma.at[pl.ds(0, zero_size)],
        dst_ref=out_dma.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


@jax.named_scope("zero_out_start_3d")
def zero_out_start_3d(
    out_ref: jax.Array,  # [size_m, size_n // num_lanes, num_lanes]
    zero_src_ref: jax.Array,  # [tile_rows, size_n // num_lanes, num_lanes]
    semaphore_ref: jax.Array,  # [1]
):
    """Zero out ALL output rows via DMA. Required for DMA scatter where output
    positions are non-contiguous (scattered).

    Reuses an existing 3D VMEM scratch (e.g. scatter_staging_ref) as the zero
    source — no dedicated zero_ref allocation needed. The caller must ensure
    zero_src_ref is not in use when this function runs.
    """

    zero_src_ref[...] = jnp.zeros_like(zero_src_ref)
    row_size = zero_src_ref.shape[0]

    total_rows = out_ref.shape[0]
    num_loops = pl.cdiv(total_rows, row_size)

    def fill_zero(i, zero_size):
        dma_start = i * row_size
        dma_end = jnp.minimum(dma_start + row_size, total_rows)
        dma_size = dma_end - dma_start

        pltpu.make_async_copy(
            src_ref=zero_src_ref.at[pl.ds(0, dma_size)],
            dst_ref=out_ref.at[pl.ds(dma_start, dma_size)],
            sem=semaphore_ref.at[0],
        ).start(priority=1)

        return zero_size + dma_size

    zero_size = lax.fori_loop(0, num_loops, fill_zero, 0)
    return zero_size


@jax.named_scope("zero_out_end_3d")
def zero_out_end_3d(
    out_ref: jax.Array,  # [size_m, size_n // num_lanes, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
):
    """Wait for all zero-fill DMAs to complete. Works with 3D output refs."""
    pltpu.make_async_copy(
        src_ref=out_ref.at[pl.ds(0, zero_size)],
        dst_ref=out_ref.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, partial_out_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
):
    """Entry point for GMM kernel.

    Computes metadata to determine which rows of lhs needs processing and how
    they will be tiled. And then, invoke inner kernel using metadata.

    Uses the following notation:
    - g: rhs group dimension
    - m: Batch dimension
    - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
        of tile_m. Skips over empty groups and accounts for revisited tiles.
    - k: in dimension
    - n: out dimension

    Args:
        lhs_group_sizes_ref: Reference to the group sizes of lhs.
        group_offset_ref: Reference to the group offset.
        lhs_ref: Reference to the lhs.
        rhs_ref: Reference to the rhs.
        out_ref: Reference to the out.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        zero_ref: Scratch memory for storing zero values used in initialization.
        semaphore_ref: Semaphore for zero initialization DMAs.
        cfgs: GmmConfigs.
    """

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    # Pack along K (2nd minor dim) so that pltpu.bitcast can unpack inside the
    # kernel.
    # [G, K, N] -> [G, K//packing, N] uint32
    if cfgs.rhs_cfgs.quant_dtype is not None:
        rhs_weight = rhs_ref.weight
        rhs_weight = rhs_weight.bitcast(jnp.uint32)
        assert rhs_weight.shape == (
            cfgs.dims.size_group,
            cfgs.dims.size_k // cfgs.rhs_cfgs.packing,
            cfgs.dims.size_n,
        )
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata buffer and return number of group & m iterations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    in_specs, out_specs = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        _, rhs_spec_orig = in_specs
        in_specs = (in_specs[0], FusedWeightsRef(gate=rhs_spec_orig, up=rhs_spec_orig))

    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_gm, num_n, num_k),
        in_specs=in_specs,
        out_specs=out_specs,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def _can_dma_rows(size, dtype):
    """Check if row-level DMA is possible (size divisible by num_lanes and size // num_lanes >= sublane tiling)."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    sls = pltpu.get_tpu_info().get_sublane_tiling(dtype)
    return (
        size % num_lanes == 0
        and size // num_lanes >= sls
        and (size // num_lanes) % sls == 0
    )


@jax.named_scope("dma_gather_gm_start")
def dma_gather_gm_start(
    src_ref, dst_ref, indices_ref, sem_ref, gm_id, metadata_ref, divisor: int = 1
):
    """Start gathering rows for a specific gm tile via DMA.

    src_ref and dst_ref must be 3D: (rows, k // num_lanes, num_lanes).
    No reshape — reshape on refs breaks dynamic pl.ds offsets.

    `divisor`: optional integer divisor applied to indices_ref values before
    use. Set > 1 when indices_ref contains packed values (e.g.,
    `combined = lhs_idx * divisor + extra_field`) and we need to recover the
    actual src_row via integer division. Default 1 = no unpacking.
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    sls = pltpu.get_tpu_info().get_sublane_tiling(src_ref.dtype)
    m_start_local = m_start % sls

    def _gather_body(i, _):
        row = m_start + i
        src_row = indices_ref[row]
        if divisor != 1:
            src_row = src_row // divisor
        pltpu.make_async_copy(
            src_ref=src_ref.at[pl.ds(src_row, 1), :, :],
            dst_ref=dst_ref.at[pl.ds(m_start_local + i, 1), :, :],
            sem=sem_ref,
        ).start()
        return _

    lax.fori_loop(0, m_end - m_start, _gather_body, 0)


@jax.named_scope("dma_gather_gm_wait")
def dma_gather_gm_wait(dst_ref, sem_ref, gm_id, metadata_ref):
    """Wait for all gather DMAs for a specific gm tile to complete.

    dst_ref must be 3D: (rows, k // num_lanes, num_lanes).
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    num_rows = m_end - m_start
    sls = pltpu.get_tpu_info().get_sublane_tiling(dst_ref.dtype)
    m_start_local = m_start % sls
    pltpu.make_async_copy(
        src_ref=dst_ref.at[pl.ds(m_start_local, num_rows), :, :],
        dst_ref=dst_ref.at[pl.ds(m_start_local, num_rows), :, :],
        sem=sem_ref,
    ).wait()


@jax.named_scope("dma_scatter_gm_start")
def dma_scatter_gm_start(src_ref, dst_ref, indices_ref, sem_ref, gm_id, metadata_ref):
    """Start scattering rows for a specific gm tile via DMA.

    src_ref and dst_ref must be 3D: (rows, n // num_lanes, num_lanes).
    No reshape — reshape on refs breaks dynamic pl.ds offsets.
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    sls = pltpu.get_tpu_info().get_sublane_tiling(src_ref.dtype)
    m_start_local = m_start % sls

    def _scatter_body(i, _):
        row = m_start + i
        dst_row = indices_ref[row]
        pltpu.make_async_copy(
            src_ref=src_ref.at[pl.ds(m_start_local + i, 1), :, :],
            dst_ref=dst_ref.at[pl.ds(dst_row, 1), :, :],
            sem=sem_ref,
        ).start()
        return _

    lax.fori_loop(0, m_end - m_start, _scatter_body, 0)


@jax.named_scope("dma_scatter_gm_wait")
def dma_scatter_gm_wait(src_ref, sem_ref, gm_id, metadata_ref):
    """Wait for all scatter DMAs for a specific gm tile to complete.

    src_ref must be 3D: (rows, n // num_lanes, num_lanes).
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    num_rows = m_end - m_start
    sls = pltpu.get_tpu_info().get_sublane_tiling(src_ref.dtype)
    m_start_local = m_start % sls
    pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(m_start_local, num_rows), :, :],
        dst_ref=src_ref.at[pl.ds(m_start_local, num_rows), :, :],
        sem=sem_ref,
    ).wait()


def inner_kernel_gather(
    # emit_pipeline tiled refs
    tiled_rhs_ref,
    tiled_out_ref,
    # emit_pipeline scratches
    partial_out_ref,
    acc_ref,
    metadata_ref,
    *,
    cfgs,
    # Captured via closure — 3D refs, no reshape for DMA
    hidden_states_ref,
    gathered_lhs_2x_ref,
    lhs_indices_ref,
    gather_sem_ref,
    gs_gate_ref=None,
    gs_up_ref=None,
):
    """Inner kernel with pipelined gather: prefetches next gm tile via DMA."""
    gm_id = pl.program_id(0)
    n_id = pl.program_id(1)
    k_id = pl.program_id(2)
    num_gm = pl.num_programs(0)
    num_lanes = pltpu.get_tpu_info().num_lanes

    sem_id = gm_id % 2

    # On first k step of first n pass, manage the gather pipeline.
    is_first_kn = jnp.logical_and(k_id == 0, n_id == 0)

    @pl.when(jnp.logical_and(is_first_kn, gm_id == 0))
    def _():
        dma_gather_gm_start(
            hidden_states_ref,
            gathered_lhs_2x_ref.at[sem_id],
            lhs_indices_ref,
            gather_sem_ref.at[sem_id],
            0,
            metadata_ref,
        )

    @pl.when(jnp.logical_and(is_first_kn, gm_id + 1 < num_gm))
    def _():
        dma_gather_gm_start(
            hidden_states_ref,
            gathered_lhs_2x_ref.at[1 - sem_id],
            lhs_indices_ref,
            gather_sem_ref.at[1 - sem_id],
            gm_id + 1,
            metadata_ref,
        )

    @pl.when(is_first_kn)
    def _():
        dma_gather_gm_wait(
            gathered_lhs_2x_ref.at[sem_id],
            gather_sem_ref.at[sem_id],
            gm_id,
            metadata_ref,
        )

    # Run the original inner kernel (matmul).
    # gathered_lhs_2x_ref is 4D: (2, tile_m, size_k // num_lanes, num_lanes)
    # Slice k, read as JAX array, then reshape for inner_kernel.
    tile_k = cfgs.tiles.tile_k
    gathered_lhs_k_slice = gathered_lhs_2x_ref.at[
        sem_id, :, pl.ds(k_id * (tile_k // num_lanes), tile_k // num_lanes), :
    ]
    # Read 3D ref → JAX array → reshape (avoids ref reshape tiling issues)
    gathered_lhs_data = gathered_lhs_k_slice[...].reshape(
        -1, cfgs.dims.size_lhs_sublane, tile_k
    )
    inner_kernel(
        gathered_lhs_data,
        tiled_rhs_ref,
        tiled_out_ref,
        partial_out_ref,
        acc_ref,
        metadata_ref,
        cfgs=cfgs,
        gs_gate_ref=gs_gate_ref,
        gs_up_ref=gs_up_ref,
    )


def inner_kernel_scatter(
    # emit_pipeline tiled refs
    tiled_lhs_ref,
    tiled_rhs_ref,
    # emit_pipeline scratches
    partial_out_ref,
    acc_ref,
    metadata_ref,
    *,
    cfgs,
    gs_gate_ref=None,
    gs_up_ref=None,
    # Captured via closure — scatter DMA refs
    tiled_out_2x_ref,  # 2D compute buffer: (2, tile_m, aligned_n)
    scatter_staging_ref,  # 3D DMA staging: (tile_m, aligned_n // num_lanes, num_lanes)
    final_out_ref,  # 3D HBM output: (size_m, aligned_n // num_lanes, num_lanes)
    output_indices_ref,
    staging_sem_ref,  # (1,) semaphore for staging/scatter buffer
    # Optional: weighted scatter-add refs (fuses reduce_sum)
    weighted_scatter=False,
    output_gather_ref=None,  # 3D VMEM for existing output gather
    topk_weights_ref=None,  # (M,) topk weights in SMEM
    output_sem_ref=None,  # (1,) semaphore for output gather DMA
    out_dtype=None,
    post_norm_ref=None,
):
    """Inner kernel with pipelined scatter using staging buffer.

    Compute writes to 2D tiled_out_2x_ref (any tile_n works).
    After all n-tiles done, copy to 3D staging buffer, then DMA scatter.
    Single staging buffer: wait for previous scatter before reuse.

    When weighted_scatter=True, performs weighted scatter-add:
        output[output_indices[i]] += matmul_result[i] * topk_weights[i]
    This fuses the topk weighting and reduce_sum into the scatter.
    """
    gm_id = pl.program_id(0)
    n_id = pl.program_id(1)
    k_id = pl.program_id(2)
    num_k = pl.num_programs(2)
    num_n = pl.num_programs(1)
    sem_id = gm_id % 2

    is_last_kn = jnp.logical_and(k_id == num_k - 1, n_id == num_n - 1)

    if weighted_scatter:
        # Weighted scatter-add: gather existing output on first step
        # to enable accumulation (output[i] += result[i] * weight[i]).
        is_first_kn = jnp.logical_and(k_id == 0, n_id == 0)

        @pl.when(is_first_kn)
        def _():
            @pl.when(gm_id > 0)
            def _():
                dma_scatter_gm_wait(
                    output_gather_ref,
                    staging_sem_ref.at[0],
                    gm_id - 1,
                    metadata_ref,
                )

            dma_gather_gm_start(
                final_out_ref,
                output_gather_ref,
                output_indices_ref,
                output_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

    # Run the original inner kernel (matmul) in scatter_mode.
    # tiled_out_2x_ref is 2D — n-slice works with any tile_n.
    tile_n = cfgs.tiles.tile_n
    tiled_out_n_slice = tiled_out_2x_ref.at[sem_id, :, pl.ds(n_id * tile_n, tile_n)]
    inner_kernel(
        tiled_lhs_ref,
        tiled_rhs_ref,
        tiled_out_n_slice,
        partial_out_ref,
        acc_ref,
        metadata_ref,
        cfgs=cfgs,
        gs_gate_ref=gs_gate_ref,
        gs_up_ref=gs_up_ref,
        scatter_mode=True,
    )

    if weighted_scatter:

        @pl.when(is_last_kn)
        def _():
            # Wait for output gather DMA to complete.
            dma_gather_gm_wait(
                output_gather_ref,
                output_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

            # Copy 2D compute result to 3D staging buffer.
            scatter_staging_ref[...] = tiled_out_2x_ref.at[sem_id][...].reshape(
                scatter_staging_ref.shape
            )

            # Per-row weighted accumulate:
            #   output_gather[row] += staging[row] * topk_weight
            m_st = metadata_ref.gm_id_to_m_offset[gm_id]
            m_en = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            _sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
            _ml = m_st % _sls

            _has_post_norm = cfgs.has_post_norm

            def _weighted_add_row(i, _):
                row_idx = _ml + i
                w = topk_weights_ref[m_st + i].astype(jnp.float32)
                old_r = output_gather_ref.at[pl.ds(row_idx, 1), :, :]
                new_r = scatter_staging_ref.at[pl.ds(row_idx, 1), :, :]
                new_r_f32 = new_r[...].astype(jnp.float32)
                if _has_post_norm:
                    # Input-scaled RMS norm: scale FIRST, normalize SECOND
                    # x = x * (weight + 1); x = rms_norm(x)
                    pnw_raw = post_norm_ref[...].astype(jnp.float32) + 1.0
                    norm_size = pnw_raw.size
                    row_size = new_r_f32.size
                    if pnw_raw.size < row_size:
                        n_cols_valid = norm_size // new_r_f32.shape[2]
                        col_idx = lax.broadcasted_iota(jnp.int32, new_r_f32.shape, 1)
                        new_r_f32 = jnp.where(
                            col_idx < n_cols_valid, new_r_f32, jnp.float32(0.0)
                        )
                        pnw = jnp.concatenate(
                            [pnw_raw.flatten(), jnp.zeros(row_size - pnw_raw.size)]
                        ).reshape(new_r_f32.shape)
                    else:
                        pnw = pnw_raw.flatten()[:row_size].reshape(new_r_f32.shape)
                    new_r_f32 = new_r_f32 * pnw
                    variance = jnp.sum(new_r_f32 * new_r_f32) / jnp.float32(norm_size)
                    new_r_f32 = new_r_f32 * jax.lax.rsqrt(variance + jnp.float32(1e-8))
                result = old_r[...].astype(jnp.float32) + new_r_f32 * w
                output_gather_ref.at[pl.ds(row_idx, 1), :, :][...] = result.astype(
                    out_dtype
                )
                return _

            lax.fori_loop(0, m_en - m_st, _weighted_add_row, 0)

            # DMA scatter the weighted result from output_gather_ref to HBM.
            dma_scatter_gm_start(
                output_gather_ref,
                final_out_ref,
                output_indices_ref,
                staging_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

    else:

        @pl.when(is_last_kn)
        def _():
            # Wait for previous scatter to free the staging buffer
            @pl.when(gm_id > 0)
            def _():
                dma_scatter_gm_wait(
                    scatter_staging_ref,
                    staging_sem_ref.at[0],
                    gm_id - 1,
                    metadata_ref,
                )

            # Copy 2D compute buffer to 3D staging buffer
            # Read as JAX array, reshape, write to 3D ref
            scatter_staging_ref[...] = tiled_out_2x_ref.at[sem_id][...].reshape(
                scatter_staging_ref.shape
            )

            # DMA scatter from 3D staging to 3D HBM output
            dma_scatter_gm_start(
                scatter_staging_ref,
                final_out_ref,
                output_indices_ref,
                staging_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )


def kernel_main_gather(
    # Scalar prefetch (5)
    lhs_group_sizes_ref,
    group_offset_ref,
    lhs_indices_ref,
    gs_gate_ref,
    gs_up_ref,
    # In (2)
    hidden_states_ref,
    rhs_ref,
    # Out (1)
    out_ref,
    # Scratch
    partial_out_ref,
    acc_ref,
    metadata_ref,
    zero_ref,
    semaphore_ref,
    gather_sem_ref,
    gathered_lhs_2x_ref,
    *,
    cfgs,
):
    """GMM kernel with pipelined gather: per-gm-tile DMA overlapped with compute."""
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    # Pack along K for quantized rhs (same as kernel_main).
    if cfgs.rhs_cfgs.quant_dtype is not None:
        rhs_weight = rhs_ref.weight
        rhs_weight = rhs_weight.bitcast(jnp.uint32)
        assert rhs_weight.shape == (
            cfgs.dims.size_group,
            cfgs.dims.size_k // cfgs.rhs_cfgs.packing,
            cfgs.dims.size_n,
        )
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata first (needed to know gm tile boundaries for per-tile gather).
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )
    in_specs, out_specs = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        _, rhs_spec_orig = in_specs
        in_specs = (in_specs[0], FusedWeightsRef(gate=rhs_spec_orig, up=rhs_spec_orig))

    # For gather, LHS is gathered via DMA scratch, not via pipeline.
    _, rhs_in_spec = in_specs

    # Start zero-out DMAs (overlapped with pre-gather, different semaphore).
    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )

    # Execute pipeline with gather-pipelined inner kernel.
    # inner_kernel_gather prefetches next gm tile's rows during compute.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            inner_kernel_gather,
            cfgs=cfgs,
            hidden_states_ref=hidden_states_ref,
            gathered_lhs_2x_ref=gathered_lhs_2x_ref,
            lhs_indices_ref=lhs_indices_ref,
            gather_sem_ref=gather_sem_ref,
            gs_gate_ref=gs_gate_ref,
            gs_up_ref=gs_up_ref,
        ),
        grid=(num_gm, num_n, num_k),
        in_specs=[rhs_in_spec],
        out_specs=out_specs,
    )
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [
        partial_out_ref,
        acc_ref,
        metadata_ref,
    ]
    pipeline_fn(rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def kernel_main_scatter(
    # Scalar prefetch (6)
    lhs_group_sizes_ref,
    group_offset_ref,
    output_indices_ref,
    topk_weights_ref,
    gs_gate_ref,
    gs_up_ref,
    # In (3)
    lhs_ref,
    rhs_ref,
    post_norm_ref,  # HBM: (N,) post_expert_norm weight
    # Out (1)
    final_out_ref,
    # Scratch
    partial_out_ref,
    acc_ref,
    metadata_ref,
    semaphore_ref,
    tiled_out_2x_ref,
    scatter_staging_ref,
    output_gather_ref,
    staging_sem_ref,
    output_sem_ref,
    *,
    cfgs,
    weighted_scatter=False,
    out_dtype=None,
):
    """GMM kernel with pipelined scatter, optionally weighted for reduce_sum fusion.

    When weighted_scatter=True, performs weighted scatter-add:
        output[output_indices[i]] += matmul_result[i] * topk_weights[i]
    This fuses the topk weighting and reduce_sum into the scatter, eliminating
    the post-kernel reshape + multiply + sum over the topk dimension.
    """
    # post_norm_ref is auto-DMA'd to VMEM via BlockSpec(memory_space=VMEM)

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    # Pack along K for quantized rhs (same as kernel_main).
    if cfgs.rhs_cfgs.quant_dtype is not None:
        rhs_weight = rhs_ref.weight
        rhs_weight = rhs_weight.bitcast(jnp.uint32)
        assert rhs_weight.shape == (
            cfgs.dims.size_group,
            cfgs.dims.size_k // cfgs.rhs_cfgs.packing,
            cfgs.dims.size_n,
        )
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )
    in_specs, out_specs = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        _, rhs_spec_orig = in_specs
        in_specs = (in_specs[0], FusedWeightsRef(gate=rhs_spec_orig, up=rhs_spec_orig))

    # Zero-init the 3D HBM output before the pipeline starts.
    # Blocking: start + wait completes before any scatter DMAs.
    if cfgs.zero_init:
        zero_size = zero_out_start_3d(
            final_out_ref,
            scatter_staging_ref,
            semaphore_ref,
        )
        zero_out_end_3d(final_out_ref, semaphore_ref, zero_size)

    # Execute pipeline with scatter-pipelined inner kernel.
    # inner_kernel_scatter scatters previous gm tile's output during compute.
    # Output is scattered via DMA scratch, not via pipeline.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            inner_kernel_scatter,
            cfgs=cfgs,
            gs_gate_ref=gs_gate_ref,
            gs_up_ref=gs_up_ref,
            tiled_out_2x_ref=tiled_out_2x_ref,
            scatter_staging_ref=scatter_staging_ref,
            final_out_ref=final_out_ref,
            output_indices_ref=output_indices_ref,
            staging_sem_ref=staging_sem_ref,
            weighted_scatter=weighted_scatter,
            output_gather_ref=output_gather_ref,
            topk_weights_ref=topk_weights_ref,
            output_sem_ref=output_sem_ref,
            out_dtype=out_dtype,
            post_norm_ref=post_norm_ref,
        ),
        grid=(num_gm, num_n, num_k),
        in_specs=in_specs,
        out_specs=[],
    )
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    scratches = [
        partial_out_ref,
        acc_ref,
        metadata_ref,
    ]
    pipeline_fn(lhs_in, rhs_ref, scratches=scratches)

    # Epilogue: wait for the last staging DMA to complete.
    if weighted_scatter:
        scatter_source_ref = output_gather_ref
    else:
        scatter_source_ref = scatter_staging_ref

    @pl.when(num_gm > 0)
    def _epilogue_wait():
        dma_scatter_gm_wait(
            scatter_source_ref,
            staging_sem_ref.at[0],
            num_gm - 1,
            metadata_ref,
        )


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_dtype = lhs_cfgs.quant_dtype or lhs_cfgs.dtype
    rhs_dtype = rhs_cfgs.dtype
    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if lower bitwidth is used, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod
    tile_m = min(tile_m, dims.size_m)

    # Subtract non-rhs VMEM overhead before computing per-buffer budget.
    # Overhead includes: gathered_lhs_2x (DMA gather), acc/partial_out buffers,
    # tiled_out_2x (DMA scatter), and compiler spill headroom.
    lhs_bits_item = jax.dtypes.itemsize_bits(lhs_cfgs.dtype)
    _overhead = (
        2 * tile_m * dims.size_k * lhs_bits_item // 8  # gathered_lhs_2x
        + 5 * 1024 * 1024  # acc, partial_out, spill headroom
    )
    rhs_vmem_budget = max(vmem_limit_bytes - _overhead, vmem_limit_bytes // 2)

    # Calculate vmem limit for a single rhs buffer when using triple buffers.
    num_rhs_buffers = 3
    rhs_vmem_target = rhs_vmem_budget // num_rhs_buffers
    base_rhs_size_bytes = dims.size_k * dims.size_n * rhs_bits // 8

    # To avoid stalling MXU, we add some buffer room where tile_n cannot go
    # smaller than 2x of mxu_column_size.
    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    # When fuse_act is set, tile_n tiles the output N (= size_n // 2).
    # base_rhs_size_bytes still uses size_n to account for loading both halves.
    size_n_per_rhs = dims.size_n
    if fuse_act is not None:
        size_n_per_rhs //= 2
        tile_n_limit = min(tile_n_limit, size_n_per_rhs)

    # Initialize tile_k and tile_n to their maximum valid values.
    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # to fit rhs into vmem by only adjusting tile_n.

    # Decrease tile_n until rhs fits in vmem target.
    while (
        pl.cdiv(base_rhs_size_bytes, num_n_tiles) > rhs_vmem_target
        and tile_n > tile_n_limit
    ):
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    # If decreasing tile_n is no longer possible, we decrease tile_k instead.
    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

        # Decrease tile_k until rhs fits in vmem target.
        base_rhs_size_bytes = pl.cdiv(base_rhs_size_bytes, num_n_tiles)
        while pl.cdiv(base_rhs_size_bytes, num_k_tiles) > rhs_vmem_target:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        raise ValueError(
            f"Could not find valid tile sizes for {dims=} and {rhs_vmem_target=}."
        )

    return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    packed_nvfp4: bool = False,
):
    """Validates the inputs for the GMM kernel."""

    size_m = lhs.shape[0]
    size_group, size_k_raw, size_n = rhs.shape
    size_k = size_k_raw * 2 if packed_nvfp4 else size_k_raw
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (
        size_m,
        size_k,
    ), (
        f"lhs.shape={lhs.shape} expected=({size_m}, {size_k}), rhs.shape={rhs.shape}, group_sizes.shape={group_sizes.shape}"
    )
    assert rhs.shape == (size_group, size_k_raw, size_n)
    if rhs_bias is not None:
        assert rhs_bias.shape == (size_group, 1, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
        # When K is zero-padded for DMA alignment, size_k may not divide
        # num_quant_blocks evenly.  The original (unpadded) K always does.
        # The inner kernel uses quant_block_size from make_gmm_configs which
        # is derived from the original K, so this is safe.

    assert group_offset.shape == (1,)

    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
    size_lhs_sublane = min(size_lhs_sublane, size_m)

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs):
    """Returns the cost estimate for the GMM kernel."""

    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.dtype

    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_bytes = (
        dims.size_group
        * dims.size_k
        * dims.size_n
        * jax.dtypes.itemsize_bits(rhs_dtype)
    ) // 8
    if cfgs.rhs_cfgs.has_scale:
        rhs_bytes += (
            dims.size_group
            * cfgs.rhs_cfgs.num_quant_blocks
            * dims.size_n
            * jnp.dtype(jnp.float32).itemsize
        )
    if cfgs.rhs_cfgs.has_bias:
        rhs_bytes += dims.size_group * dims.size_n * jnp.dtype(jnp.float32).itemsize

    out_bytes = dims.size_m * dims.size_n * jnp.dtype(cfgs.out_dtype).itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(dims: Dimensions, tiles: TileSizes) -> str:
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    lhs_indices: jax.Array | None = None,
    output_indices: jax.Array | None = None,
    original_k: int | None = None,
    fuse_act: str | None = None,
    post_expert_norm_weight: jax.Array | None = None,
    packed_nvfp4: bool = False,
):
    """Fills the GMM config for the GMM kernel."""

    dims = validate_inputs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        fuse_act=fuse_act,
        packed_nvfp4=packed_nvfp4,
    )

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = jnp.float4_e2m1fn.dtype if packed_nvfp4 else rhs.dtype
        num_blocks = rhs_scale.shape[1]
        # When K was zero-padded for DMA alignment, use the original K
        # to compute block_size so quant block boundaries stay correct.
        _k_for_blocks = original_k if original_k is not None else dims.size_k
        block_size = recover_quant_block_size(_k_for_blocks, num_blocks)
        rhs_packing = 8 if packed_nvfp4 else 32 // jax.dtypes.itemsize_bits(rhs.dtype)
    else:
        has_scale = False
        rhs_quant_dtype = None
        num_blocks = 1
        block_size = dims.size_k
        rhs_packing = 1

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
        packing=rhs_packing,
        num_quant_blocks=num_blocks,
    )

    lhs_quant_block_size = 256 if rhs_cfgs.quant_block_size < 512 else 512

    lhs_q_dtype = None
    if (
        maybe_quantize_lhs
        and get_maybe_quantize_lhs(
            rhs_quant_dtype,
            rhs_cfgs.quant_block_size,
            lhs_quant_block_size,
        )
        and rhs_quant_dtype is not None
    ):
        # Choose lhs quantization dtype based on TPU hardware support.
        is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
        tpu_info = pltpu.get_tpu_info()
        # Check if there is hardware compute support for rhs dtype group.
        if is_rhs_float:
            if tpu_info.fp8_ops_per_second > 0:
                lhs_q_dtype = jnp.float8_e4m3fn.dtype
        else:
            if tpu_info.int8_ops_per_second > 0:
                lhs_q_dtype = jnp.int8.dtype

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        # Input quantization involves reading all elements in a block to compute
        # scale value. Since this operation is very memory intensive, we use a
        # block size that is small enough to minimize memory overhead but large
        # enough to minimize compute overhead of quantization.
        quant_block_size=lhs_quant_block_size,
        dtype=lhs.dtype,
    )

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        acc_dtype = acc_dtype = jnp.float32.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        zero_init=zero_initialize,
        fuse_act=fuse_act,
        has_post_norm=post_expert_norm_weight is not None,
    )


def get_metadata(cfgs: GmmConfigs):
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if isinstance(val, jnp.dtype):
            val = val.name
        ret[key] = val
    return ret


@jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "maybe_quantize_lhs",
        "zero_initialize",
        "output_size",
        "fuse_act",
        "packed_nvfp4",
    ],
    compiler_options={
        # uncomment the following line for named_scopes to show up in traces
        # this incurs some performance loss in benchmarks, so only enable
        # when needed
        # "xla_enable_transpose_trace": True,
        # uncomment the following line to allow debug_print to show up in
        # logs, note that this changes the behavior of the kernel completely,
        # must turn off for accuracy tests
        # "xla_tpu_enable_log_recorder": True,
    },
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    rhs_global_scale: jax.Array | None = None,  # tiny per-expert global scale
    group_offset: jax.Array | None = None,  # int32[1]
    lhs_indices: jax.Array | None = None,  # optional gather indices for LHS rows
    output_indices: jax.Array | None = None,  # optional scatter indices for output rows
    topk_weights: jax.Array | None = None,  # optional per-row weights for scatter-add
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    output_size: int | None = None,
    fuse_act: str | None = None,
    post_expert_norm_weight: jax.Array | None = None,
    packed_nvfp4: bool = False,
) -> jax.Array:
    """GMM kernel with DMA gather/scatter fusion.

    Extends the base GMM kernel (emit_pipeline) with optional kernel-level
    DMA gather on LHS rows and DMA scatter on output rows. When DMA fusion
    is not possible (e.g. dimensions not aligned to TPU lanes), falls back
    to JAX-level gather/scatter.

    When topk_weights is provided along with output_indices, the scatter
    performs weighted scatter-add (fused reduce_sum):
        output[output_indices[i]] += matmul_result[i] * topk_weights[i]
    This eliminates the post-kernel reshape, multiply, and sum over the
    topk dimension, outputting directly to (output_size, N).

    Args:
        lhs: LHS matrix of shape [size_m, size_k]. When lhs_indices is set,
            this is the ungathered source and rows are gathered inside the
            kernel.
        rhs: RHS matrix of shape [size_group, size_k, size_n].
        group_sizes: Group sizes for LHS rows of shape [size_lhs_group,].
        rhs_scale: Optional per-block scale of shape
            [size_group, num_blocks, 1, out_size].
        rhs_bias: Optional bias of shape [size_group, 1, out_size].
        group_offset: Optional group offset of shape [1,]. Used with expert
            parallelism where group_sizes covers all experts but only a
            subset is processed on this shard.
        lhs_indices: Optional int32[size_m] gather indices. When set, the
            kernel gathers lhs[lhs_indices] via DMA (if aligned) or JAX
            fallback.
        output_indices: Optional int32[size_m] scatter indices. When set,
            output rows are scattered to output_indices positions in a
            buffer of size output_size. Uses DMA scatter (if aligned) or
            JAX fallback.
        topk_weights: Optional float[size_m] per-row weights for weighted
            scatter-add. When set along with output_indices, the scatter
            fuses topk weighting and reduce_sum.
        tile_info: Tile sizes or tiling function.
        vmem_limit_bytes: Optional VMEM limit in bytes.
        precision: Unused. Exists for compatibility reasons.
        preferred_element_type: Optional output dtype.
        acc_dtype: Optional accumulator dtype.
        maybe_quantize_lhs: Quantize LHS if True and RHS is quantized.
        zero_initialize: Whether to zero-initialize unvisited output
            elements.
        output_size: Required when output_indices is set. Total number of
            rows in the scatter output buffer.
        fuse_act: Optional activation fusion ("silu", "gelu", etc.).

    Returns:
        Output of shape [size_m, size_n] (or [output_size, size_n] when
        output_indices is set).
    """

    del precision

    # Guard: fp4 with unaligned K produces NaN due to a libtpu/Pallas
    # miscompilation when the kernel is lowered inside shard_map (EP).
    # Standalone gmm_v2 calls work, but the shard_map compilation path
    # generates incorrect code for this combination.  Block early with a
    # clear message so the caller can fall back to gmm_ep_fused.
    if rhs.dtype == jnp.float4_e2m1fn and rhs_scale is not None:
        _nl = pltpu.get_tpu_info().num_lanes
        _sls = pltpu.get_tpu_info().get_sublane_tiling(jnp.bfloat16)
        _k = rhs.shape[1]
        if _k % (_nl * _sls) != 0:
            raise ValueError(
                f"fp4 (float4_e2m1fn) gmm_v2 with K={_k} is not supported: "
                f"K must be a multiple of {_nl * _sls} "
                f"(num_lanes={_nl} * sublane_tiling={_sls}). "
                f"This combination triggers a known libtpu/Pallas "
                f"miscompilation under shard_map. "
                f"Use ep_backend='gmm_ep_fused' or 'gmm_ep_fused_v2' instead."
            )

    # Determine if kernel-level DMA fusion is possible for gather/scatter.
    # When K is large enough for DMA but not aligned to num_lanes, pad K with
    # zeros so the DMA gather reshape (M, K // num_lanes, num_lanes) succeeds.
    # Zero-padding the contraction dimension does not change the matmul result.
    # When K is large enough for DMA but not aligned to num_lanes *
    # sublane_tiling, pad K with zeros.  Zero-padding the contraction dim
    # does not change the matmul result.  For quantized rhs we keep the
    # original quant_block_size / num_blocks — the extra K elements are
    # zeros and their scale value is irrelevant.
    _use_dma_gather = lhs_indices is not None and _can_dma_rows(lhs.shape[1], lhs.dtype)
    _original_k = lhs.shape[1]
    if not _use_dma_gather and lhs_indices is not None:
        _nl = pltpu.get_tpu_info().num_lanes
        _sls = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
        _tile_k_unit = _nl * _sls
        _k_aligned = align_to(_original_k, _tile_k_unit)
        if _k_aligned != _original_k and _k_aligned // _nl >= _sls:
            _k_pad = _k_aligned - _original_k
            lhs = jnp.pad(lhs, ((0, 0), (0, _k_pad)))
            # rhs NOT padded: weight DMA (make_async_copy) doesn't need
            # the same 3D alignment as indexed DMA gather. Saves matmul
            # FLOPs and weight bandwidth on zero-padded K columns.
            _use_dma_gather = True
    _use_dma_scatter = output_indices is not None

    # JAX-level gather fallback when DMA fusion not possible.
    if lhs_indices is not None and not _use_dma_gather:
        lhs = lhs[lhs_indices]

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)

    # For DMA gather, use proxy LHS shape for config computation.
    # Use rhs K dimension (original, unpadded) — lhs may have been padded
    # for DMA alignment but rhs was NOT padded.
    _rhs_k = rhs.shape[1] * 2 if packed_nvfp4 else rhs.shape[1]
    if _use_dma_gather:
        lhs_for_config = jax.ShapeDtypeStruct((lhs_indices.shape[0], _rhs_k), lhs.dtype)
    else:
        lhs_for_config = lhs

    # Check fuse_act alignment: size_n must be divisible by 2*num_lanes.
    # If not, disable in-kernel fusion and fall back to JAX-level activation.
    _fallback_fuse_act = None
    if fuse_act is not None:
        _nl = pltpu.get_tpu_info().num_lanes
        if rhs.shape[2] % (2 * _nl) != 0:
            _fallback_fuse_act = fuse_act
            fuse_act = None

    cfgs = make_gmm_configs(
        lhs_for_config,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize,
        lhs_indices=lhs_indices,
        output_indices=output_indices,
        original_k=_original_k if _original_k != lhs.shape[1] else None,
        fuse_act=fuse_act,
        post_expert_norm_weight=post_expert_norm_weight,
        packed_nvfp4=packed_nvfp4,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    # DMA gather uses a 4D VMEM ref (2, tile_m, size_k // num_lanes, num_lanes).
    # The inner kernel slices dim 2 by tile_k // num_lanes, which must be a
    # multiple of the sublane tiling (e.g. 16 for bf16).  Round tile_k down
    # to the nearest valid value when it doesn't satisfy this.
    _use_dma_gather = lhs_indices is not None and _can_dma_rows(lhs.shape[1], lhs.dtype)
    if _use_dma_gather:
        _num_lanes = pltpu.get_tpu_info().num_lanes
        _sls = pltpu.get_tpu_info().get_sublane_tiling(cfgs.lhs_cfgs.dtype)
        tile_k_unit = _num_lanes * _sls
        if tiles.tile_k % tile_k_unit != 0:
            aligned_tile_k = (tiles.tile_k // tile_k_unit) * tile_k_unit
            if aligned_tile_k == 0:
                aligned_tile_k = tile_k_unit
            tiles = dataclasses.replace(tiles, tile_k=aligned_tile_k)
            cfgs = dataclasses.replace(cfgs, tiles=tiles)

    # Prepare block specs.
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    # Per-expert global_scale as scalar prefetch (SMEM): one scalar per expert.
    # Indexed in inner_kernel via group_id from metadata_ref. Avoids 22 GiB HLO temp.
    if rhs_global_scale is not None:
        if rhs_global_scale.ndim == 2:
            _gs_gate = rhs_global_scale[:, 0].astype(jnp.float32)
            _gs_up = rhs_global_scale[:, 1].astype(jnp.float32)
        else:
            _gs_gate = rhs_global_scale.astype(jnp.float32)
            _gs_up = _gs_gate
    else:
        _gs_gate = jnp.ones((dims.size_group,), dtype=jnp.float32)
        _gs_up = _gs_gate
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Compute aligned_n early -- needed for DMA scatter tile_n adjustment.
    # Align to num_lanes * sublane_tiling so that aligned_n // num_lanes is a
    # multiple of the sublane tiling, which Mosaic requires for DMA slices.
    num_lanes = pltpu.get_tpu_info().num_lanes
    _out_sls = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
    aligned_n = align_to(dims.size_n, num_lanes * _out_sls)

    # DMA scatter: ensure tile_n evenly divides aligned_n so that
    # tiled_out_2x_ref[sem_id, :, pl.ds(n_id * tile_n, tile_n)] never overflows.
    _use_dma_scatter = output_indices is not None
    if _use_dma_scatter and aligned_n % tiles.tile_n != 0:
        _mxu_size = pltpu.get_tpu_info().mxu_column_size
        _num_n = pl.cdiv(aligned_n, tiles.tile_n)
        _adj_tile_n = (aligned_n // _num_n // _mxu_size) * _mxu_size
        if _adj_tile_n > 0:
            tiles = dataclasses.replace(tiles, tile_n=_adj_tile_n)
            cfgs = dataclasses.replace(cfgs, tiles=tiles)

    # Initialize scratch shapes.
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1

    # partial_out is n-aware: stores partial rows for ALL n-tile positions so
    # that crossing gm boundaries in the (num_gm, num_n, num_k) grid reads
    # the correct partial data for each n position.
    num_n = pl.cdiv(cfgs.out_size_n, tiles.tile_n)
    partial_out_n = num_n * tiles.tile_n

    # When fuse_act is set, acc needs 2*tile_n columns (gate+up) before
    # activation halves it back to tile_n for the output.
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n

    scratch_shapes = [
        # partial_out_ref — widened to cover all n positions
        pltpu.VMEM((dims.size_lhs_sublane, partial_out_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]

    if cfgs.zero_init:
        # TODO(kyuyeunk): Create better heuristics for determining this value.
        target_zero_ref_bytes = 2 * 1024 * 1024

        # Zero initialization is done by tiling size_m dim where each tile invokes
        # zero initializing DMA for up-to tile_zero_m rows. This means larger
        # tile_zero_m will result in fewer number of tiles and lead to smaller
        # overhead. However, in order to invoke DMA call up-to tile_zero_m rows, we
        # need to store equivalent sized memory in VMEM buffer for the duration of
        # DMA. Storing [tile_zero_m, size_n] in buffer will trigger OOM if
        # tile_zero_m is too large. Instead, if we set column size as num_lanes
        # (which is smallest allowed column size for DMA) and reuse the buffer by
        # size_n//num_lanes times in a single tile, we can significantly increase
        # tile_zero_m without triggering OOM.
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims.size_m)

        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    else:
        scratch_shapes += [None, None]

    out_n = (
        aligned_n
        if cfgs.fuse_act is None
        else align_to(cfgs.out_size_n, num_lanes * _out_sls)
    )
    out_init = jax.ShapeDtypeStruct((dims.size_m, out_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    rhs_in_spec = WeightsRef(
        weight=pl.BlockSpec(memory_space=pltpu.HBM),
        scale=rhs_scale_spec,
        bias=rhs_bias_spec,
    )
    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    if _use_dma_gather:
        # Gather scratch shapes (maps to kernel_main_gather scratch params):
        #   [0] partial_out_ref:       VMEM((size_lhs_sublane, partial_out_n), out_dtype)
        #   [1] acc_ref:               VMEM((tile_m, tile_n), acc_dtype)
        #   [2] metadata_ref:          MetadataRef(gm_id_to_group_id=SMEM((max_num_gm,), int32),
        #                                          gm_id_to_m_offset=SMEM((max_num_gm+1,), int32))
        #   [3] zero_ref:              VMEM((tile_zero_m, num_lanes), out_dtype) or None
        #   [4] semaphore_ref:         SemaphoreType.DMA((1,)) or None
        #   [5] gather_sem_ref:        SemaphoreType.DMA((2,))
        #   [6] gathered_lhs_2x_ref:   VMEM((2, tile_m, padded_k // num_lanes, num_lanes), out_dtype)
        #                              — 4D double-buffered for DMA gather (uses padded K)
        _padded_k = lhs.shape[1]  # lhs was padded for DMA; rhs was NOT
        gather_scratch = scratch_shapes + [
            pltpu.SemaphoreType.DMA((2,)),  # gather_sem_ref
            pltpu.VMEM(
                (2, tiles.tile_m, _padded_k // num_lanes, num_lanes),
                cfgs.out_dtype,
            ),  # gathered_lhs_2x_ref — 4D for DMA (no reshape)
        ]

        # Reshape lhs to 3D for DMA gather (avoids reshape on ref inside kernel)
        lhs_3d = lhs.reshape(lhs.shape[0], lhs.shape[1] // num_lanes, num_lanes)
        result = pl.pallas_call(
            functools.partial(kernel_main_gather, cfgs=cfgs),
            out_shape=out_init,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    rhs_in_spec,
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=gather_scratch,
            ),
            compiler_params=compiler_params,
            name=get_scope_name(dims, tiles) + "-gather",
            cost_estimate=get_cost_estimate(cfgs),
            metadata=get_metadata(cfgs),
        )(group_sizes, group_offset, lhs_indices, _gs_gate, _gs_up, lhs_3d, rhs_weights)
        result = result[:, : cfgs.out_size_n]

    elif _use_dma_scatter and _can_dma_rows(aligned_n, cfgs.out_dtype):
        # Scatter fusion: scatter output rows inside the kernel via DMA.
        # When topk_weights is provided, performs weighted scatter-add
        # (fused reduce_sum): output[idx[i]] += result[i] * weight[i]
        assert output_size is not None
        _weighted_scatter = topk_weights is not None

        n_cols = aligned_n // num_lanes

        # Scatter scratch shapes (maps to kernel_main_scatter scratch params):
        #   [0] partial_out_ref:       VMEM((size_lhs_sublane, partial_out_n), out_dtype)
        #   [1] acc_ref:               VMEM((tile_m, acc_cols), acc_dtype)
        #   [2] metadata_ref:          MetadataRef
        #   [3] semaphore_ref:         SemaphoreType.DMA((1,)) — used by zero_out_3d
        #   [4] tiled_out_2x_ref:      VMEM((2, tile_m, aligned_n), out_dtype)
        #   [5] scatter_staging_ref:   VMEM((tile_m, n_cols, num_lanes), out_dtype)
        #   [6] output_gather_ref:     VMEM (weighted only) or None
        #   [7] staging_sem_ref:       SemaphoreType.DMA((1,))
        #   [8] output_sem_ref:        SemaphoreType.DMA (weighted only) or None
        scatter_scratch = [
            scratch_shapes[0],  # partial_out_ref
            scratch_shapes[1],  # acc_ref
            scratch_shapes[2],  # metadata_ref
            pltpu.SemaphoreType.DMA((1,)),  # semaphore_ref for zero_out_3d
            pltpu.VMEM(
                (2, tiles.tile_m, aligned_n), cfgs.out_dtype
            ),  # tiled_out_2x_ref — 2D compute buffer
            pltpu.VMEM(
                (tiles.tile_m, n_cols, num_lanes), cfgs.out_dtype
            ),  # scatter_staging_ref — 3D DMA staging, also reused as zero source
            (
                pltpu.VMEM((tiles.tile_m, n_cols, num_lanes), cfgs.out_dtype)
                if _weighted_scatter
                else None
            ),  # output_gather_ref
            pltpu.SemaphoreType.DMA((1,)),  # staging_sem_ref
            (
                pltpu.SemaphoreType.DMA((1,)) if _weighted_scatter else None
            ),  # output_sem_ref
        ]

        # 3D HBM output for DMA scatter (avoids reshape on ref)
        out_rows = output_size if _weighted_scatter else dims.size_m
        scatter_out_shape = (out_rows, n_cols, num_lanes)
        scatter_out_init = jax.ShapeDtypeStruct(scatter_out_shape, cfgs.out_dtype)

        # topk_weights passed as scalar prefetch (SMEM) for per-row access.

        # --- SMEM chunking for scatter ---
        # Scalar prefetch loads output_indices (M*4B) + topk_weights
        # (M*4B) into SMEM at kernel launch. At 16K+ tokens with
        # topk=8, M=131K and these arrays exceed the 1MB SMEM budget.
        # Fix: halve M until scalar prefetch fits, issuing multiple
        # pallas_calls that accumulate into the same output via
        # weighted scatter-add. Each chunk gets correctly computed
        # per-slice group_sizes from the cumulative sum of the
        # original group_sizes.
        import jax.experimental.pallas.tpu as _pltpu

        _smem_budget = int(_pltpu.get_tpu_info().smem_capacity_bytes)
        _size_m = output_indices.shape[0]
        _smem_per_m = 4 + 4  # output_indices + topk_weights
        _smem_overhead = 65536
        _smem_needed = _size_m * _smem_per_m + _smem_overhead
        _num_chunks = 1
        while _smem_needed > _smem_budget and _num_chunks < 16:
            _num_chunks *= 2
            _chunk_m = (_size_m + _num_chunks - 1) // _num_chunks
            _smem_needed = _chunk_m * _smem_per_m + _smem_overhead

        if _num_chunks > 1:
            # Split M for scatter kernel.
            _chunk_m = (_size_m + _num_chunks - 1) // _num_chunks
            _gs_cumsum = jnp.cumsum(group_sizes)
            _gs_start = jnp.concatenate(
                [jnp.zeros(1, dtype=jnp.int32), _gs_cumsum[:-1]]
            )
            out_rows = output_size if _weighted_scatter else dims.size_m
            n_cols = aligned_n // num_lanes
            result = jnp.zeros((out_rows, n_cols, num_lanes), dtype=cfgs.out_dtype)
            for _ci in range(_num_chunks):
                _m_lo = _ci * _chunk_m
                _m_hi = min(_m_lo + _chunk_m, _size_m)
                if _m_lo >= _size_m:
                    break
                _c_indices = output_indices[_m_lo:_m_hi]
                _c_topk = (
                    topk_weights.astype(jnp.float32).flatten()[_m_lo:_m_hi]
                    if _weighted_scatter
                    else jnp.zeros((1,), jnp.float32)
                )
                _ov_s = jnp.maximum(_gs_start, _m_lo)
                _ov_e = jnp.minimum(_gs_cumsum, _m_hi)
                _c_gs = jnp.maximum(0, _ov_e - _ov_s)
                _c_lhs = lhs[_m_lo:_m_hi]
                _c_size_m = _m_hi - _m_lo
                _c_dims = dataclasses.replace(dims, size_m=_c_size_m)
                _c_cfgs = dataclasses.replace(cfgs, dims=_c_dims)
                _c_max_num_gm = dims.size_group + pl.cdiv(_c_size_m, tiles.tile_m) - 1
                _c_scatter_scratch = list(scatter_scratch)
                _c_scatter_scratch[2] = MetadataRef(
                    gm_id_to_group_id=pltpu.SMEM((_c_max_num_gm,), jnp.int32),
                    gm_id_to_m_offset=pltpu.SMEM((_c_max_num_gm + 1,), jnp.int32),
                )
                _c_out_init = jax.ShapeDtypeStruct(
                    (out_rows, n_cols, num_lanes), cfgs.out_dtype
                )
                post_norm_w_flat = (
                    post_expert_norm_weight.astype(jnp.float32).flatten()
                    if post_expert_norm_weight is not None
                    else jnp.zeros((1,), jnp.float32)
                )
                _c_result = pl.pallas_call(
                    functools.partial(
                        kernel_main_scatter,
                        cfgs=_c_cfgs,
                        weighted_scatter=_weighted_scatter,
                        out_dtype=cfgs.out_dtype if _weighted_scatter else None,
                    ),
                    out_shape=_c_out_init,
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=6,
                        in_specs=[
                            pl.BlockSpec(memory_space=pltpu.HBM),
                            rhs_in_spec,
                            pl.BlockSpec(memory_space=pltpu.VMEM),
                        ],
                        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                        scratch_shapes=_c_scatter_scratch,
                    ),
                    compiler_params=compiler_params,
                    name=get_scope_name(_c_dims, tiles) + f"-reduce-chunk{_ci}",
                    cost_estimate=get_cost_estimate(_c_cfgs),
                    metadata=get_metadata(_c_cfgs),
                )(
                    _c_gs,
                    group_offset,
                    _c_indices,
                    _c_topk,
                    _gs_gate,
                    _gs_up,
                    _c_lhs,
                    rhs_weights,
                    post_norm_w_flat,
                )
                result = result + _c_result
            result = result.reshape(out_rows, aligned_n)[:, : cfgs.out_size_n]

        else:
            topk_w_flat = (
                topk_weights.astype(jnp.float32).flatten()
                if _weighted_scatter
                else jnp.zeros((1,), jnp.float32)
            )

            # post_expert_norm_weight for RMSNorm in weighted scatter
            post_norm_w_flat = (
                post_expert_norm_weight.astype(jnp.float32).flatten()
                if post_expert_norm_weight is not None
                else jnp.zeros((1,), jnp.float32)
            )

            result = pl.pallas_call(
                functools.partial(
                    kernel_main_scatter,
                    cfgs=cfgs,
                    weighted_scatter=_weighted_scatter,
                    out_dtype=cfgs.out_dtype if _weighted_scatter else None,
                ),
                out_shape=scatter_out_init,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=6,
                    in_specs=[
                        pl.BlockSpec(memory_space=pltpu.HBM),  # lhs
                        rhs_in_spec,  # rhs
                        pl.BlockSpec(
                            memory_space=pltpu.VMEM
                        ),  # post_expert_norm_weight (auto DMA to VMEM)
                    ],
                    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                    scratch_shapes=scatter_scratch,
                ),
                compiler_params=compiler_params,
                name=get_scope_name(dims, tiles)
                + ("-reduce" if _weighted_scatter else "-scatter"),
                cost_estimate=get_cost_estimate(cfgs),
                metadata=get_metadata(cfgs),
            )(
                group_sizes,
                group_offset,
                output_indices,
                topk_w_flat,
                _gs_gate,
                _gs_up,
                lhs,
                rhs_weights,
                post_norm_w_flat,
            )
            result = result.reshape(out_rows, aligned_n)[:, : cfgs.out_size_n]

    else:
        # Normal scratch shapes (maps to kernel_main scratch params):
        #   [0] partial_out_ref:  VMEM((size_lhs_sublane, partial_out_n), out_dtype)
        #   [1] acc_ref:          VMEM((tile_m, tile_n), acc_dtype)
        #   [2] metadata_ref:     MetadataRef(gm_id_to_group_id=SMEM((max_num_gm,), int32),
        #                                     gm_id_to_m_offset=SMEM((max_num_gm+1,), int32))
        #   [3] zero_ref:         VMEM((tile_zero_m, num_lanes), out_dtype) or None
        #   [4] semaphore_ref:    SemaphoreType.DMA((1,)) or None
        result = pl.pallas_call(
            functools.partial(kernel_main, cfgs=cfgs),
            out_shape=out_init,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    rhs_in_spec,
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=compiler_params,
            name=get_scope_name(dims, tiles),
            cost_estimate=get_cost_estimate(cfgs),
            metadata=get_metadata(cfgs),
        )(group_sizes, group_offset, lhs, rhs_weights)[:, : cfgs.out_size_n]

        # JAX-level scatter fallback.
        if output_indices is not None:
            assert output_size is not None
            full = jnp.zeros((output_size, result.shape[-1]), dtype=result.dtype)
            if topk_weights is not None:
                result = result * topk_weights[:, None].astype(result.dtype)
                result = full.at[output_indices].add(result)
            else:
                result = full.at[output_indices].set(result)

    # Fallback: apply activation at JAX level when kernel fusion was disabled.
    if _fallback_fuse_act is not None:
        gate, up = jnp.split(result, 2, axis=-1)
        result = apply_act_fn(jnp.concatenate([gate, up], axis=-1), _fallback_fuse_act)

    return result


# =============================================================================
# Fused kernel: gather + GMM1 + activation + GMM2 + scatter
# =============================================================================
# Time-shares the VMEM weight buffer between GMM1 and GMM2, keeping the
# intermediate activation result in VMEM to avoid an HBM round-trip.
# Prototype: bf16 only, synchronous weight reads (no pipelining).
# =============================================================================
@dataclasses.dataclass(frozen=True)
class FusedDims:
    """Dimensions for the fused gather+GMM1+act+GMM2+scatter kernel."""

    size_m: int  # Number of gathered/scattered rows
    size_k1: int  # K for GMM1 (= hidden_size H, original unpadded)
    size_n1: int  # N for GMM1 (= 2 * intermediate_size, gate+up)
    size_k2: int  # K for GMM2 (= intermediate_size I)
    size_n2: int  # N for GMM2 (= hidden_size H), aligned for DMA
    original_n2: int  # Original (unpadded) N for GMM2
    size_group: int  # Number of experts on this shard
    size_lhs_group: int  # Total number of expert groups
    size_lhs_sublane: int  # Sublane tiling for LHS dtype
    intermediate_size: int = 0  # Original I (= N1_orig // 2) for activation split
    has_bias: bool = False  # Whether bias refs are present
    quant_block_size: int | None = None  # RHS quantization block size
    num_scale_blocks1: int = 0  # K1 // quant_block_size (for scale DMA)
    num_scale_blocks2: int = 0  # K2 // quant_block_size (for scale DMA)
    padded_k1: int = 0  # DMA-aligned K1 (>= size_k1); 0 means no padding


def kernel_main_fused(
    # Scalar prefetch (8)
    lhs_group_sizes_ref,
    group_offset_ref,
    lhs_indices_ref,
    topk_weights_ref,  # Scalar prefetch: (M,) topk weights
    w1_gs_gate_ref,  # Scalar prefetch: (E,) GMM1 gate global_scale
    w1_gs_up_ref,  # Scalar prefetch: (E,) GMM1 up global_scale
    w2_gs_ref,  # Scalar prefetch: (E,) GMM2 global_scale
    # In (7)
    hidden_states_ref,  # 3D HBM: (M_src, K1//NL, NL)
    w1_ref,  # HBM: (E, K1, N1) — gate+up weights
    w2_ref,  # HBM: (E, K2, aligned_n2) — down weights
    w1_scale_ref,  # HBM: (E, total_blocks1, 1, N1) or placeholder
    w2_scale_ref,  # HBM: (E, total_blocks2, 1, aligned_n2) or placeholder
    w1_bias_ref,  # HBM: (E, 1, N1) or placeholder
    w2_bias_ref,  # HBM: (E, 1, aligned_n2) or placeholder
    post_norm_ref,  # VMEM: (N,) post_expert_norm weight or placeholder
    # Out (1)
    final_out_ref,  # 3D HBM: (output_size, N2//NL, NL)
    # Scratch
    metadata_ref,  # MetadataRef — full metadata for all gm tiles
    fused_metadata_ref,  # MetadataRef — single-slot for weight indexing
    gathered_lhs_2x_ref,  # VMEM: (2, tile_m, K1//NL, NL)
    gmm1_out_ref,  # VMEM: (tile_m, N1) — GMM1 accumulation buffer
    intermediate_ref,  # VMEM: (tile_m//sls, sls, K2) — after activation (scatter layout)
    tiled_out_2x_ref,  # VMEM: (2, tile_m, aligned_n2) — GMM2 output
    scatter_staging_ref,  # VMEM: 3D staging for scatter
    partial_out1_ref,  # VMEM: (sls, tile_n1) — dummy for scatter_mode
    partial_out2_ref,  # VMEM: (sls, tile_n2) — dummy for scatter_mode
    shared_acc_ref,  # VMEM: (tile_m, max(2*tile_n1, tile_n2)) — time-shared
    gather_sem_ref,  # DMA(2,)
    staging_sem_ref,  # DMA(1,)
    zero_sem_ref,  # DMA(1,)
    gm_id_ref,  # SMEM: (1,) int32
    w1_buf_ref,  # VMEM: (num_w1_bufs, packed_k1, tile_n1) — GMM1 weight buf
    w2_buf_ref,  # VMEM: (num_w2_bufs, packed_k2, tile_n2) — GMM2 weight buf
    w1_scale_buf_ref,  # VMEM: (num_w1_bufs, nqb1, 1, tile_n1) — GMM1 scale buf
    w2_scale_buf_ref,  # VMEM: (num_w2_bufs, nqb2, 1, tile_n2) — GMM2 scale buf
    w1_bias_buf_ref,  # VMEM: (1, w1_dma_n) — GMM1 bias tile
    w2_bias_buf_ref,  # VMEM: (1, tile_n2) — GMM2 bias tile
    w1_sem_ref,  # DMA(num_w1_bufs,) — GMM1 weight DMA semaphores
    w2_sem_ref,  # DMA(num_w2_bufs,) — GMM2 weight DMA semaphores
    output_sem_ref,  # DMA(1,) — output gather for weighted scatter-add
    *,
    fused_dims: FusedDims,
    tile_m: int,
    tile_k1: int,
    tile_k2: int,
    tile_n1: int,
    tile_n2: int,
    num_w1_bufs: int,
    num_w2_bufs: int,
    act_fn: str,
    out_dtype: jnp.dtype,
    has_post_norm: bool = False,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
):
    """Fused gather + GMM1 + activation + GMM2 + weighted scatter-add kernel.

    Combines two GMMs with gather/scatter into a single pallas_call,
    eliminating the HBM round-trip for intermediate activations.
    Topk weighting and cross-expert reduction are fused into the scatter,
    outputting directly to (num_tokens, N) and eliminating the post-kernel
    reshape + multiply + reduce_sum.

    Pipeline diagram (per gm tile):
    ┌────────────────────────────────────────────────────────────────┐
    │  1. Scatter wait    Wait for previous gm tile scatter DMA      │
    │  2. Gather DMA      Async gather hidden_states to VMEM         │
    │  3. Output gather   Load existing output for accumulation      │
    │  4. W2 prefetch     Start W2 weight DMA (cross-pipeline)       │
    │  5. Gather wait     Block until gather completes               │
    │  6. GMM1 loop       K-tiled matmul with W1 DMA pipelining      │
    │  7. W1 prefetch     Start next gm tile W1 DMA (cross-pipeline) │
    │  8. Interlude       Copy GMM1 output to intermediate (bf16)    │
    │  9. GMM2 loop       K-tiled matmul with W2 (already loaded)    │
    │ 10. Output wait     Wait for output gather to complete         │
    │ 11. Weighted add    out += gmm2_result * topk_weight           │
    │ 12. Scatter DMA     Async scatter result to HBM                │
    └────────────────────────────────────────────────────────────────┘

    Buffer sharing:
    - shared_acc: single buffer, sliced for GMM1 acc and GMM2 acc.
    - gathered_lhs_2x[sem_id]: reused for input gather (padded_k1 cols)
      during GMM1, then for output gather (aligned_n2 cols) during weighted
      scatter-add. Sized to max(padded_k1, aligned_n2) to fit both.
    - scatter_staging: separate buffer for GMM2 output reshape.
    - w1_buf / w2_buf: cannot share (W2 prefetched during GMM1).
    """
    dims = fused_dims
    num_lanes = pltpu.get_tpu_info().num_lanes

    # Build minimal GmmConfigs for fill_metadata.
    meta_dims = Dimensions(
        size_m=dims.size_m,
        size_k=dims.size_k1,
        size_n=dims.size_n1,
        size_group=dims.size_group,
        size_lhs_group=dims.size_lhs_group,
        size_lhs_sublane=dims.size_lhs_sublane,
    )
    meta_cfgs = GmmConfigs(
        tiles=TileSizes(tile_m=tile_m, tile_k=tile_k1, tile_n=dims.size_n1),
        dims=meta_dims,
        lhs_cfgs=InputConfigs(
            quant_dtype=None, quant_block_size=dims.size_k1, dtype=out_dtype
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=None, quant_block_size=dims.size_k1, dtype=out_dtype
        ),
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        zero_init=False,
    )
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=meta_cfgs,
    )

    # Zero-init scatter output (async start, wait before gm loop).
    zero_src = scatter_staging_ref
    zero_size = zero_out_start_3d(final_out_ref, zero_src, zero_sem_ref)

    # Pack fp8 weights if quantized.
    if cfgs1.rhs_cfgs.quant_dtype is not None:
        w1_packed = w1_ref.bitcast(jnp.uint32)
    else:
        w1_packed = w1_ref
    if cfgs2.rhs_cfgs.quant_dtype is not None:
        w2_packed = w2_ref.bitcast(jnp.uint32)
    else:
        w2_packed = w2_ref

    # Build WeightsRef for pipeline inputs.
    # Custom DMA pipeline: manual pltpu.make_async_copy with separate
    # w1/w2 buffers and cross-pipeline prefetch.

    num_k1 = dims.size_k1 // tile_k1
    num_k2 = dims.size_k2 // tile_k2
    out_n1 = dims.size_n1 // 2 if act_fn else dims.size_n1
    num_n1 = out_n1 // tile_n1
    # Use original (unpadded) N for GMM2 loop count, matching the DMA
    # gmm_v2 path which uses cfgs.out_size_n (not aligned_n).
    # This avoids iterating over padding N-tiles which can cause
    # incorrect results with fp8 weights.
    num_n2 = dims.original_n2 // tile_n2
    packing = cfgs1.rhs_cfgs.packing
    pk1 = tile_k1 // packing
    pk2 = tile_k2 // packing
    has_scale = cfgs1.rhs_cfgs.has_scale
    has_bias = cfgs1.rhs_cfgs.has_bias
    nqb1 = cfgs1.num_quant_blocks_per_tile_k if has_scale else 0
    nqb2 = cfgs2.num_quant_blocks_per_tile_k if has_scale else 0

    w1_dma_n = tile_n1 * 2 if act_fn else tile_n1

    @jax.named_scope("start_w1_dma")
    def start_w1_dma(buf_id, expert_id, n_id, k_id=0):
        if act_fn and num_n1 > 1:
            # Fix: w1 layout is [all_gate | all_up] along N axis.
            # Load tile_n1 from gate + tile_n1 from up into buf as [gate|up].
            _gate_off = n_id * tile_n1
            _up_off = out_n1 + n_id * tile_n1
            pltpu.make_async_copy(
                src_ref=w1_packed.at[
                    expert_id, pl.ds(k_id * pk1, pk1), pl.ds(_gate_off, tile_n1)
                ],
                dst_ref=w1_buf_ref.at[buf_id, :, :tile_n1],
                sem=w1_sem_ref.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                src_ref=w1_packed.at[
                    expert_id, pl.ds(k_id * pk1, pk1), pl.ds(_up_off, tile_n1)
                ],
                dst_ref=w1_buf_ref.at[buf_id, :, tile_n1:],
                sem=w1_sem_ref.at[buf_id],
            ).start()
            if has_scale:
                pltpu.make_async_copy(
                    src_ref=w1_scale_ref.at[
                        expert_id,
                        pl.ds(k_id * nqb1, nqb1),
                        :,
                        pl.ds(_gate_off, tile_n1),
                    ],
                    dst_ref=w1_scale_buf_ref.at[buf_id, :, :, :tile_n1],
                    sem=w1_sem_ref.at[buf_id],
                ).start()
                pltpu.make_async_copy(
                    src_ref=w1_scale_ref.at[
                        expert_id,
                        pl.ds(k_id * nqb1, nqb1),
                        :,
                        pl.ds(_up_off, tile_n1),
                    ],
                    dst_ref=w1_scale_buf_ref.at[buf_id, :, :, tile_n1:],
                    sem=w1_sem_ref.at[buf_id],
                ).start()
            if has_bias:
                pltpu.make_async_copy(
                    src_ref=w1_bias_ref.at[expert_id, :, pl.ds(_gate_off, tile_n1)],
                    dst_ref=w1_bias_buf_ref.at[:, :tile_n1],
                    sem=w1_sem_ref.at[buf_id],
                ).start()
                pltpu.make_async_copy(
                    src_ref=w1_bias_ref.at[expert_id, :, pl.ds(_up_off, tile_n1)],
                    dst_ref=w1_bias_buf_ref.at[:, tile_n1:],
                    sem=w1_sem_ref.at[buf_id],
                ).start()
        else:
            pltpu.make_async_copy(
                src_ref=w1_packed.at[
                    expert_id, pl.ds(k_id * pk1, pk1), pl.ds(n_id * w1_dma_n, w1_dma_n)
                ],
                dst_ref=w1_buf_ref.at[buf_id],
                sem=w1_sem_ref.at[buf_id],
            ).start()
            if has_scale:
                pltpu.make_async_copy(
                    src_ref=w1_scale_ref.at[
                        expert_id,
                        pl.ds(k_id * nqb1, nqb1),
                        :,
                        pl.ds(n_id * w1_dma_n, w1_dma_n),
                    ],
                    dst_ref=w1_scale_buf_ref.at[buf_id],
                    sem=w1_sem_ref.at[buf_id],
                ).start()

            if has_bias:
                pltpu.make_async_copy(
                    src_ref=w1_bias_ref.at[
                        expert_id, :, pl.ds(n_id * w1_dma_n, w1_dma_n)
                    ],
                    dst_ref=w1_bias_buf_ref,
                    sem=w1_sem_ref.at[buf_id],
                ).start()

    @jax.named_scope("start_w2_dma")
    def start_w2_dma(buf_id, expert_id, n_id, k_id=0):
        pltpu.make_async_copy(
            src_ref=w2_packed.at[
                expert_id, pl.ds(k_id * pk2, pk2), pl.ds(n_id * tile_n2, tile_n2)
            ],
            dst_ref=w2_buf_ref.at[buf_id],
            sem=w2_sem_ref.at[buf_id],
        ).start()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w2_scale_ref.at[
                    expert_id,
                    pl.ds(k_id * nqb2, nqb2),
                    :,
                    pl.ds(n_id * tile_n2, tile_n2),
                ],
                dst_ref=w2_scale_buf_ref.at[buf_id],
                sem=w2_sem_ref.at[buf_id],
            ).start()

        if has_bias:
            pltpu.make_async_copy(
                src_ref=w2_bias_ref.at[expert_id, :, pl.ds(n_id * tile_n2, tile_n2)],
                dst_ref=w2_bias_buf_ref,
                sem=w2_sem_ref.at[buf_id],
            ).start()

    @jax.named_scope("wait_w1_dma")
    def wait_w1_dma(buf_id):
        pltpu.make_async_copy(
            src_ref=w1_buf_ref.at[buf_id],
            dst_ref=w1_buf_ref.at[buf_id],
            sem=w1_sem_ref.at[buf_id],
        ).wait()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w1_scale_buf_ref.at[buf_id],
                dst_ref=w1_scale_buf_ref.at[buf_id],
                sem=w1_sem_ref.at[buf_id],
            ).wait()

        if has_bias:
            pltpu.make_async_copy(
                src_ref=w1_bias_buf_ref,
                dst_ref=w1_bias_buf_ref,
                sem=w1_sem_ref.at[buf_id],
            ).wait()

    @jax.named_scope("wait_w2_dma")
    def wait_w2_dma(buf_id):
        pltpu.make_async_copy(
            src_ref=w2_buf_ref.at[buf_id],
            dst_ref=w2_buf_ref.at[buf_id],
            sem=w2_sem_ref.at[buf_id],
        ).wait()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w2_scale_buf_ref.at[buf_id],
                dst_ref=w2_scale_buf_ref.at[buf_id],
                sem=w2_sem_ref.at[buf_id],
            ).wait()

        if has_bias:
            pltpu.make_async_copy(
                src_ref=w2_bias_buf_ref,
                dst_ref=w2_bias_buf_ref,
                sem=w2_sem_ref.at[buf_id],
            ).wait()

    @jax.named_scope("compute_gmm1_tile")
    def compute_gmm1_tile(buf_id, n_id, k_id, gm_id):
        sem_id = gm_id % 2
        k_cols = tile_k1 // num_lanes
        k_offset = k_id * k_cols
        lhs_data = gathered_lhs_2x_ref[sem_id, :, pl.ds(k_offset, k_cols), :].reshape(
            -1, dims.size_lhs_sublane, tile_k1
        )
        w1_tile = w1_buf_ref.at[buf_id]
        w1_sc = w1_scale_buf_ref.at[buf_id] if has_scale else None
        w1_bias_tile = w1_bias_buf_ref if has_bias else None
        w1_weights = WeightsRef(weight=w1_tile, scale=w1_sc, bias=w1_bias_tile)
        if cfgs1.fuse_act is not None:
            # Split gate/up for fused activation
            w1_up = WeightsRef(
                weight=w1_tile.at[:, tile_n1:],
                scale=w1_sc.at[:, :, tile_n1:] if w1_sc is not None else None,
                bias=w1_bias_tile.at[:, pl.ds(tile_n1, tile_n1)] if has_bias else None,
            )
            w1_gate = WeightsRef(
                weight=w1_tile.at[:, :tile_n1],
                scale=w1_sc.at[:, :, :tile_n1] if w1_sc is not None else None,
                bias=w1_bias_tile.at[:, pl.ds(0, tile_n1)] if has_bias else None,
            )
            w1_weights = FusedWeightsRef(gate=w1_gate, up=w1_up)
        inner_kernel(
            lhs_data,
            w1_weights,
            gmm1_out_ref.at[:, pl.ds(n_id * tile_n1, tile_n1)],
            partial_out1_ref,
            shared_acc_ref.at[:, pl.ds(0, tile_n1 * 2 if act_fn else tile_n1)],
            fused_metadata_ref,
            cfgs=cfgs1,
            gs_gate_ref=w1_gs_gate_ref,
            gs_up_ref=w1_gs_up_ref,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k1,
            _gm_id=0,
            _n_id=n_id,
        )

    @jax.named_scope("compute_gmm2_tile")
    def compute_gmm2_tile(buf_id, n_id, k_id, gm_id):
        sem_id = gm_id % 2
        # Intermediate is in scatter layout: (tile_m//sls, sls, K2)
        lhs_data = intermediate_ref[:, :, pl.ds(k_id * tile_k2, tile_k2)]
        w2_tile = w2_buf_ref.at[buf_id]
        w2_sc = w2_scale_buf_ref.at[buf_id] if has_scale else None
        inner_kernel(
            lhs_data,
            WeightsRef(
                weight=w2_tile, scale=w2_sc, bias=w2_bias_buf_ref if has_bias else None
            ),
            tiled_out_2x_ref.at[sem_id, :, pl.ds(n_id * tile_n2, tile_n2)],
            partial_out2_ref,
            shared_acc_ref.at[:, pl.ds(0, tile_n2)],
            fused_metadata_ref,
            cfgs=cfgs2,
            gs_gate_ref=w2_gs_ref,
            gs_up_ref=w2_gs_ref,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k2,
            _gm_id=0,
            _n_id=n_id,
        )

        @pl.when(jnp.logical_and(n_id == num_n2 - 1, k_id == num_k2 - 1))
        def _():
            # Wait for output gather.
            dma_gather_gm_wait(
                gathered_lhs_2x_ref.at[sem_id],
                output_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )
            # Weighted scatter: per-row multiply + accumulate + DMA.
            # Reads weight scalars from topk_weights_ref (scalar prefetch / SMEM).
            m_st_w = metadata_ref.gm_id_to_m_offset[gm_id]
            m_en_w = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            _sls_w = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
            _ml_w = m_st_w % _sls_w

            # Copy GMM2 to scatter_staging (3D reshape).
            # gathered_lhs_2x[sem_id] has old output from gather.
            scatter_staging_ref[...] = tiled_out_2x_ref[sem_id][...].reshape(
                scatter_staging_ref.shape
            )

            # Per-row weighted add using ref access only.
            def _weighted_add_row(i, _):
                row_idx = _ml_w + i
                w = topk_weights_ref[m_st_w + i].astype(jnp.float32)
                old_r = gathered_lhs_2x_ref.at[sem_id].at[pl.ds(row_idx, 1), :, :]
                new_r = scatter_staging_ref.at[pl.ds(row_idx, 1), :, :]
                new_r_f32 = new_r[...].astype(jnp.float32)
                if has_post_norm:
                    # Input-scaled RMS norm: scale FIRST, normalize SECOND
                    pnw_raw = post_norm_ref[...].astype(jnp.float32) + 1.0
                    norm_size = pnw_raw.size
                    row_size = new_r_f32.size
                    if pnw_raw.size < row_size:
                        n_cols_valid = norm_size // new_r_f32.shape[2]
                        col_idx = lax.broadcasted_iota(jnp.int32, new_r_f32.shape, 1)
                        new_r_f32 = jnp.where(
                            col_idx < n_cols_valid, new_r_f32, jnp.float32(0.0)
                        )
                        pnw = jnp.concatenate(
                            [pnw_raw.flatten(), jnp.zeros(row_size - pnw_raw.size)]
                        ).reshape(new_r_f32.shape)
                    else:
                        pnw = pnw_raw.flatten()[:row_size].reshape(new_r_f32.shape)
                    new_r_f32 = new_r_f32 * pnw
                    variance = jnp.sum(new_r_f32 * new_r_f32) / jnp.float32(norm_size)
                    new_r_f32 = new_r_f32 * jax.lax.rsqrt(variance + jnp.float32(1e-8))
                result = old_r[...].astype(jnp.float32) + new_r_f32 * w
                gathered_lhs_2x_ref.at[sem_id].at[pl.ds(row_idx, 1), :, :][...] = (
                    result.astype(out_dtype)
                )
                return _

            lax.fori_loop(0, m_en_w - m_st_w, _weighted_add_row, 0)
            # Scatter the weighted result.
            dma_scatter_gm_start(
                gathered_lhs_2x_ref.at[sem_id],
                final_out_ref,
                lhs_indices_ref,
                staging_sem_ref.at[0],
                gm_id,
                metadata_ref,
            )

    # Weight caching: skip DMA when consecutive gm tiles use the
    # same expert. Only safe when ALL tiles for that weight fit in its
    # buffer set (so the early-(n,k) tiles still live in the buffer at
    # gm-end and remain valid for the next gms first steps).
    _can_cache_w1 = num_n1 * num_k1 <= num_w1_bufs
    _can_cache_w2 = num_n2 * num_k2 <= num_w2_bufs
    _can_cache = _can_cache_w1 and _can_cache_w2

    @jax.named_scope("gm_loop_body")
    def gm_loop_body(gm_id, _):
        sem_id = gm_id % 2
        gm_id_ref[0] = gm_id
        fused_metadata_ref.gm_id_to_group_id[0] = metadata_ref.gm_id_to_group_id[gm_id]
        fused_metadata_ref.gm_id_to_m_offset[0] = metadata_ref.gm_id_to_m_offset[gm_id]
        fused_metadata_ref.gm_id_to_m_offset[1] = metadata_ref.gm_id_to_m_offset[
            gm_id + 1
        ]
        expert_id = fused_metadata_ref.gm_id_to_group_id[0]

        # Detect same expert via metadata (no extra SMEM scratch).
        # Per-weight predicates: w1 wait/cache and w2 wait/cache must use
        # their own _can_cache_wX flag, since one weight may fit (cache
        # valid) while the other does not. A combined flag would skip the
        # wait when one cache is valid but leave the other s sem unmatched
        # (e.g. some shapes have _can_cache_w1=True, _can_cache_w2=False: combined
        # _can_cache=False, but using it would make us always wait on w1
        # even though we ve correctly skipped its prefetch -> w1 sem
        # underflow -> deadlock).
        _same_expert_pred = (gm_id > 0) & (
            expert_id == metadata_ref.gm_id_to_group_id[gm_id - 1]
        )
        same_expert_w1 = _can_cache_w1 and _same_expert_pred
        same_expert_w2 = _can_cache_w2 and _same_expert_pred

        # Wait for previous scatter before reusing output_gather buffer.
        @pl.when(gm_id > 0)
        def _():
            dma_scatter_gm_wait(
                gathered_lhs_2x_ref.at[1 - sem_id],
                staging_sem_ref.at[0],
                gm_id - 1,
                metadata_ref,
            )

        # DMA gather
        @pl.when(gm_id == 0)
        def _():
            dma_gather_gm_start(
                hidden_states_ref,
                gathered_lhs_2x_ref.at[sem_id],
                lhs_indices_ref,
                gather_sem_ref.at[sem_id],
                0,
                metadata_ref,
            )

        @pl.when(gm_id + 1 < num_gm)
        def _():
            dma_gather_gm_start(
                hidden_states_ref,
                gathered_lhs_2x_ref.at[1 - sem_id],
                lhs_indices_ref,
                gather_sem_ref.at[1 - sem_id],
                gm_id + 1,
                metadata_ref,
            )

        # --- Weight DMA (overlapped with gather) ---
        # w1 prologue: load first num_w1_bufs tiles (gm==0 only;
        # for gm>0, tiles were prefetched during previous gm's GMM2).
        total_w1_steps = num_n1 * num_k1
        total_w2_steps = num_n2 * num_k2

        @pl.when(gm_id == 0)
        def _():
            for _i in range(min(num_w1_bufs, total_w1_steps)):
                start_w1_dma(_i, expert_id, _i // num_k1, _i % num_k1)

        # Early w2 prefetch — skip only when w2 cache is valid AND same expert.
        @pl.when(jnp.logical_not(same_expert_w2))
        def _():
            for _i in range(min(num_w2_bufs, total_w2_steps)):
                start_w2_dma(_i, expert_id, _i // num_k2, _i % num_k2)

        # Wait for gather (weight DMAs running in parallel).
        dma_gather_gm_wait(
            gathered_lhs_2x_ref.at[sem_id],
            gather_sem_ref.at[sem_id],
            gm_id,
            metadata_ref,
        )

        # --- GMM1 (N,K)-tile loop (flattened, buffered with num_w1_bufs slots) ---
        for step in range(total_w1_steps):
            _n1 = step // num_k1
            _k1 = step % num_k1
            buf_id = step % num_w1_bufs

            @pl.when(jnp.logical_not(same_expert_w1))
            def _():
                wait_w1_dma(buf_id)

            if step + num_w1_bufs < total_w1_steps:
                next_step = step + num_w1_bufs
                start_w1_dma(
                    next_step % num_w1_bufs,
                    expert_id,
                    next_step // num_k1,
                    next_step % num_k1,
                )
            compute_gmm1_tile(buf_id, _n1, _k1, gm_id)

        # Next-gm w1 prefetch. Refills ALL w1 buffers (range starts at 0)
        # for the next gms first GMM1 steps. When _can_cache_w1=False, the
        # w1 buffers got overwritten by later (n,k) tiles during this gms
        # GMM1 loop, so they must be refetched even when next_expert is
        # unchanged. Done strictly AFTER the GMM1 loop (not interleaved
        # with it) to avoid a write-after-write conflict on the buffer
        # slots in the nw1=1 case.
        @pl.when(gm_id + 1 < num_gm)
        def _():
            next_expert = metadata_ref.gm_id_to_group_id[gm_id + 1]
            if _can_cache_w1:

                @pl.when(next_expert != expert_id)
                def _():
                    for _i in range(min(num_w1_bufs, total_w1_steps)):
                        start_w1_dma(_i, next_expert, _i // num_k1, _i % num_k1)

            else:
                for _i in range(min(num_w1_bufs, total_w1_steps)):
                    start_w1_dma(_i, next_expert, _i // num_k1, _i % num_k1)

        with jax.named_scope("interlude"):
            # Copy gmm1_out (f32) to intermediate_ref (bf16, scatter layout).
            # Activation was already applied inside inner_kernel via fuse_act.
            gmm1_result = gmm1_out_ref[...]
            k2_pad = dims.size_k2 - gmm1_result.shape[1]
            if k2_pad > 0:
                gmm1_result = jnp.concatenate(
                    [gmm1_result, jnp.zeros((tile_m, k2_pad), dtype=gmm1_result.dtype)],
                    axis=1,
                )
            intermediate_ref[...] = gmm1_result.astype(out_dtype).reshape(
                intermediate_ref.shape
            )

        # Load existing output for weighted scatter-add.
        # Uses gathered_lhs_2x_ref (sized to max(padded_k1, aligned_n2)),
        # which can hold both input (padded_k1) and output (aligned_n2) gathers.
        dma_gather_gm_start(
            final_out_ref,
            gathered_lhs_2x_ref.at[sem_id],
            lhs_indices_ref,
            output_sem_ref.at[0],
            gm_id,
            metadata_ref,
        )

        # --- GMM2 (N,K)-tile loop (flattened, buffered with num_w2_bufs slots) ---
        for step in range(total_w2_steps):
            _n2 = step // num_k2
            _k2 = step % num_k2
            buf_id = step % num_w2_bufs

            @pl.when(jnp.logical_not(same_expert_w2))
            def _():
                wait_w2_dma(buf_id)

            if step + num_w2_bufs < total_w2_steps:
                next_step = step + num_w2_bufs
                start_w2_dma(
                    next_step % num_w2_bufs,
                    expert_id,
                    next_step // num_k2,
                    next_step % num_k2,
                )
            compute_gmm2_tile(buf_id, _n2, _k2, gm_id)

    # Wait for zero-fill to complete BEFORE the gm loop starts.
    # The output gather inside gm_loop_body reads from final_out_ref;
    # if zero-fill hasn't landed yet, the gather captures stale HBM data
    # instead of zeros, corrupting the weighted scatter-add accumulation.
    zero_out_end_3d(final_out_ref, zero_sem_ref, zero_size)

    lax.fori_loop(0, num_gm, gm_loop_body, None)

    # Epilogue: wait for last scatter.
    @pl.when(num_gm > 0)
    def _():
        last_sem_id = (num_gm - 1) % 2
        dma_scatter_gm_wait(
            gathered_lhs_2x_ref.at[last_sem_id],
            staging_sem_ref.at[0],
            num_gm - 1,
            metadata_ref,
        )


@jax.jit(
    static_argnames=[
        "act_fn",
        "output_size",
        "vmem_limit_bytes",
    ],
    compiler_options={
        # uncomment the following line for named_scopes to show up in traces
        # this incurs some performance loss in benchmarks, so only enable
        # when needed
        # "xla_enable_transpose_trace": True,
        # uncomment the following line to allow debug_print to show up in
        # logs, note that this changes the behavior of the kernel completely,
        # must turn off for accuracy tests
        # "xla_tpu_enable_log_recorder": True,
    },
)
def gmm_v2_fused(
    hidden_states: jax.Array,  # [size_m_src, size_k1]
    w1: jax.Array,  # [size_group, size_k1, size_n1] — gate+up weights
    w2: jax.Array,  # [size_group, size_k2, size_n2] — down weights
    group_sizes: jax.Array,  # int32[size_lhs_group]
    lhs_indices: jax.Array,  # int32[size_m] — gather indices for LHS rows
    output_indices: jax.Array,  # int32[size_m] — scatter indices for output rows
    *,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_global_scale: jax.Array | None = None,  # (E, 2) gate/up per-expert
    w2_global_scale: jax.Array | None = None,  # (E,) per-expert
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    act_fn: str = "silu",
    group_offset: jax.Array | None = None,
    output_size: int,
    vmem_limit_bytes: int | None = None,
    topk_weights: jax.Array,
    post_expert_norm_weight: jax.Array | None = None,
) -> jax.Array:
    """Fused gather + GMM1 + activation + GMM2 + weighted scatter-add.

    Uses custom DMA pipeline with separate double/triple-buffered weight VMEM
    for GMM1 and GMM2, with cross-pipeline prefetch for DMA overlap.

    The scatter stage performs weighted accumulation:
    output[token] += gmm2_result * weight, outputting directly to
    (num_tokens, N) instead of (num_tokens*topk, N).
    """
    size_group, size_k1, size_n1 = w1.shape
    _, size_k2, size_n2 = w2.shape
    size_m = lhs_indices.shape[0]

    is_quantized = w1_scale is not None
    if is_quantized:
        assert w2_scale is not None
        rhs1_quant_block_size = recover_quant_block_size(size_k1, w1_scale.shape[1])
        rhs2_quant_block_size = recover_quant_block_size(size_k2, w2_scale.shape[1])
        quant_block_size = rhs1_quant_block_size
        w1_scale = w1_scale.astype(jnp.float32)
        w2_scale = w2_scale.astype(jnp.float32)
    else:
        rhs1_quant_block_size = size_k1
        rhs2_quant_block_size = size_k2
        quant_block_size = None
    # Per-expert global_scale as scalar prefetch (SMEM): one scalar per expert,
    # indexed inside inner_kernel via metadata_ref.gm_id_to_group_id. Avoids
    # graph-level fp32 fold that would create a per-call ~0.4 GiB intermediate.
    if w1_global_scale is not None:
        if w1_global_scale.ndim == 2:
            _w1_gs_gate = w1_global_scale[:, 0].astype(jnp.float32)
            _w1_gs_up = w1_global_scale[:, 1].astype(jnp.float32)
        else:
            _w1_gs_gate = w1_global_scale.astype(jnp.float32)
            _w1_gs_up = _w1_gs_gate
    else:
        _w1_gs_gate = jnp.ones((size_group,), dtype=jnp.float32)
        _w1_gs_up = _w1_gs_gate
    if w2_global_scale is not None:
        _w2_gs = w2_global_scale.astype(jnp.float32)
    else:
        _w2_gs = jnp.ones((size_group,), dtype=jnp.float32)

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    num_lanes = pltpu.get_tpu_info().num_lanes
    sls = pltpu.get_tpu_info().get_sublane_tiling(hidden_states.dtype)
    out_dtype = hidden_states.dtype
    size_lhs_sublane = min(sls, size_m)

    # Pad K1 for DMA gather alignment.
    # Only pad hidden_states (LHS) — weights stay at original K1 to avoid
    # wasting matmul FLOPs and weight bandwidth on zero-padded columns.
    # The gathered LHS buffer uses padded_k1 for DMA shape, but the matmul
    # loop only iterates over original size_k1 columns.
    tile_k_unit = num_lanes * sls
    padded_k1 = align_to(size_k1, tile_k_unit)
    if padded_k1 != size_k1:
        k_pad = padded_k1 - size_k1
        hidden_states = jnp.pad(hidden_states, ((0, 0), (0, k_pad)))
        # w1 NOT padded: weight DMA (make_async_copy) doesn't need 3D alignment.
        # size_k1 stays at original value for matmul computation.

    # Pad N1 for weight DMA alignment (lane-aligned, same as gmm_v2).
    # Only need N1 % num_lanes == 0 (for weight DMA) and K2 = N1/2 % num_lanes == 0
    # (for intermediate scatter layout). So alignment = 2 * num_lanes, NOT
    # num_lanes * sublane_tiling (which is for 3D DMA refs, not weight tensors).
    n1_unit = 2 * num_lanes
    intermediate_size = size_n1 // 2
    aligned_n1 = align_to(size_n1, n1_unit)
    if aligned_n1 != size_n1:
        n1_pad = aligned_n1 - size_n1
        w1 = jnp.pad(w1, ((0, 0), (0, 0), (0, n1_pad)))
        if is_quantized:
            w1_scale = jnp.pad(w1_scale, ((0, 0), (0, 0), (0, 0), (0, n1_pad)))
        new_k2 = aligned_n1 // 2
        if new_k2 != size_k2:
            w2 = jnp.pad(w2, ((0, 0), (0, new_k2 - size_k2), (0, 0)))
            size_k2 = new_k2
        size_n1 = aligned_n1

    # Align N2 for DMA scatter (needed before tile computation).
    out_sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
    aligned_n2 = align_to(size_n2, num_lanes * out_sls)

    # N2 alignment note: aligned_n2 is used for OUTPUT buffer sizing (DMA
    # scatter requires 3D refs at aligned width), but w2 weights do NOT need
    # padding — the kernel iterates num_n2 = original_n2 // tile_n2 tiles,
    # so DMA only accesses w2 columns [0, original_n2).  Removing w2 pad
    # eliminates a large copy+pad (~33% of w2) on every forward pass.

    # ---- Joint tile selection for fused kernel ----
    # VMEM budget: subtract fixed buffers that don't depend on tile_k/tile_n,
    # then split remaining between the two GMMs' weight tiles.
    # Fixed buffers: gathered_lhs_2x, scatter_staging, tiled_out_2x,
    # gmm1_out, intermediate, misc.
    fixed_vmem = (
        2
        * 256
        * size_k1
        * jax.dtypes.itemsize_bits(out_dtype)
        // 8  # gathered_lhs_2x (tile_m=256 estimate)
        + 2
        * 256
        * aligned_n2
        * jax.dtypes.itemsize_bits(out_dtype)
        // 8  # tiled_out_2x
        + 256 * (size_n1 // 2 if act_fn else size_n1) * 4  # gmm1_out (f32)
        + 256 * size_k2 * jax.dtypes.itemsize_bits(out_dtype) // 8  # intermediate
        + 256
        * (aligned_n2 // num_lanes)
        * num_lanes
        * jax.dtypes.itemsize_bits(out_dtype)
        // 8  # scatter_staging
    )
    spill_budget = 6 * 1024 * 1024  # compiler spill overhead
    available_vmem = vmem_limit_bytes - fixed_vmem - spill_budget
    fused_vmem = max(available_vmem // 2, 1024 * 1024)  # at least 1 MB per GMM

    dims1 = Dimensions(
        size_m=size_m,
        size_k=size_k1,
        size_n=size_n1,
        size_group=size_group,
        size_lhs_group=group_sizes.shape[0],
        size_lhs_sublane=size_lhs_sublane,
    )
    rhs1_cfgs = InputConfigs(
        quant_dtype=w1.dtype if is_quantized else None,
        quant_block_size=rhs1_quant_block_size,
        dtype=w1.dtype,
    )
    tiles1 = calculate_tiling(
        dims1,
        InputConfigs(
            quant_dtype=None,
            quant_block_size=size_k1,
            dtype=out_dtype,
        ),
        rhs1_cfgs,
        fused_vmem,
    )

    dims2 = Dimensions(
        size_m=size_m,
        size_k=size_k2,
        size_n=aligned_n2,
        size_group=size_group,
        size_lhs_group=group_sizes.shape[0],
        size_lhs_sublane=size_lhs_sublane,
    )
    rhs2_cfgs = InputConfigs(
        quant_dtype=w2.dtype if is_quantized else None,
        quant_block_size=rhs2_quant_block_size,
        dtype=w2.dtype,
    )
    tiles2 = calculate_tiling(
        dims2,
        InputConfigs(
            quant_dtype=None,
            quant_block_size=size_k2,
            dtype=out_dtype,
        ),
        rhs2_cfgs,
        fused_vmem,
    )

    # Default tiles for the fused kernel.
    # For FP8 weights: avoid K- and N-tiling. The fused kernel has a
    # known accuracy bug when tile_n does not evenly divide the output
    # N dimension with FP8 quantization. No-tiling is safe; tuned
    # block sizes can re-enable tiling for validated configs.
    # For bf16 weights: use calculate_tiling defaults (K/N tiling is
    # safe without quantization block alignment concerns).
    out_n1_default = size_n1 // 2 if act_fn else size_n1
    if is_quantized:
        default_tile_m = min(tiles1.tile_m, 128)
        # Estimate VMEM for no-tiling: weight tiles dominate.
        _elem_bytes = jax.dtypes.itemsize_bits(w1.dtype) // 8
        _notile_weight_vmem = (
            size_k1 * out_n1_default * _elem_bytes + size_k2 * size_n2 * _elem_bytes
        )
        # Reduce tile_m until no-tiling fits in VMEM.
        # The gathered_lhs and tiled_out buffers scale with tile_m.
        _bytes_per_elem = jax.dtypes.itemsize_bits(out_dtype) // 8
        while default_tile_m > 16:
            _tile_m_vmem = (
                2 * default_tile_m * max(padded_k1, aligned_n2) * _bytes_per_elem
                + 2 * default_tile_m * aligned_n2 * _bytes_per_elem
                + default_tile_m * out_n1_default * 4
            )
            if _notile_weight_vmem + _tile_m_vmem <= vmem_limit_bytes - 6 * 1024 * 1024:
                break
            default_tile_m //= 2
        default_tiles = (
            default_tile_m,
            size_k1,
            out_n1_default,
            size_k2,
            size_n2,
            1,
            1,
        )
    else:
        # bf16: N-tiling is safe, avoid K-tiling only.
        default_tiles = (
            tiles1.tile_m,
            size_k1,  # no K1 tiling
            tiles1.tile_n,
            size_k2,  # no K2 tiling
            tiles2.tile_n,
            2,
            2,
        )

    # This non-RS path has no in-tree tuning table; keep its heuristic tiles.
    tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = default_tiles

    # Post-processing: alignment and divisibility
    k_align1 = num_lanes  # matmul only needs lane alignment, not sublane
    if tile_k1 % k_align1 != 0:
        tile_k1 = (tile_k1 // k_align1) * k_align1 or k_align1
    if size_k1 % tile_k1 != 0:
        tile_k1 = size_k1
    out_n1 = size_n1 // 2 if act_fn else size_n1  # fuse_act halves N
    if out_n1 % tile_n1 != 0:
        while tile_n1 > num_lanes and out_n1 % tile_n1 != 0:
            tile_n1 -= num_lanes
    # Adjust tile_n2 to divide size_n2 (not aligned_n2). The kernel
    # iterates num_n2 = size_n2 // tile_n2 real N-tiles only.
    if size_n2 % tile_n2 != 0:
        mxu_cols = pltpu.get_tpu_info().mxu_column_size
        nn2 = pl.cdiv(size_n2, tile_n2)
        adj_tn2 = (size_n2 // nn2 // mxu_cols) * mxu_cols
        if adj_tn2 > 0:
            tile_n2 = adj_tn2
    if size_n2 % tile_n2 != 0:
        while tile_n2 > num_lanes and size_n2 % tile_n2 != 0:
            tile_n2 -= num_lanes
    if tile_k2 % num_lanes != 0:
        tile_k2 = (tile_k2 // num_lanes) * num_lanes or num_lanes
    if size_k2 % tile_k2 != 0:
        tk2_adj = tile_k2
        while tk2_adj > num_lanes and size_k2 % tk2_adj != 0:
            tk2_adj -= num_lanes
        tile_k2 = tk2_adj if (tk2_adj > 0 and size_k2 % tk2_adj == 0) else size_k2

    assert size_k1 % tile_k1 == 0, f"tile_k1={tile_k1} must divide size_k1={size_k1}"
    assert size_k2 % tile_k2 == 0, f"tile_k2={tile_k2} must divide size_k2={size_k2}"
    assert padded_k1 % num_lanes == 0
    assert size_n1 % tile_n1 == 0, f"tile_n1={tile_n1} must divide size_n1={size_n1}"
    # tile_n2 must divide original size_n2. With num_n2 = original_n2 // tile_n2,
    # the kernel only iterates over real N-tiles, so alignment with aligned_n2
    # is not required.
    assert size_n2 % tile_n2 == 0, f"tile_n2={tile_n2} must divide size_n2={size_n2}"

    if tile_k1 < size_k1 or tile_k2 < size_k2:
        print(
            f"[GMM v2 fused] WARNING: K-tiling enabled: "
            f"tile_k1={tile_k1}/{size_k1} tile_k2={tile_k2}/{size_k2}. "
            f"K-tiling in the fused kernel may cause numerical "
            f"accuracy issues.",
            flush=True,
        )

    # Pad scale K-blocks (ceiling division for quant blocks per tile).
    if is_quantized:
        nqb1 = -(-tile_k1 // rhs1_quant_block_size)
        total_scale_blocks1 = (size_k1 // tile_k1) * nqb1
        if w1_scale.shape[1] < total_scale_blocks1:
            w1_scale = jnp.concatenate(
                [
                    w1_scale,
                    jnp.repeat(
                        w1_scale[:, -1:, :, :],
                        total_scale_blocks1 - w1_scale.shape[1],
                        axis=1,
                    ),
                ],
                axis=1,
            )
        nqb2 = -(-tile_k2 // rhs2_quant_block_size)
        total_scale_blocks2 = (size_k2 // tile_k2) * nqb2
        if w2_scale.shape[1] < total_scale_blocks2:
            w2_scale = jnp.concatenate(
                [
                    w2_scale,
                    jnp.repeat(
                        w2_scale[:, -1:, :, :],
                        total_scale_blocks2 - w2_scale.shape[1],
                        axis=1,
                    ),
                ],
                axis=1,
            )
    else:
        total_scale_blocks1 = 0
        total_scale_blocks2 = 0

    # Pad and reshape biases.
    has_bias = w1_bias is not None
    if has_bias:
        assert w2_bias is not None
        if w1_bias.shape[-1] != size_n1:
            w1_bias = jnp.pad(
                w1_bias, ((0, 0), (0, 0), (0, size_n1 - w1_bias.shape[-1]))
            )
        if w2_bias.shape[-1] != size_n2:
            w2_bias = jnp.pad(
                w2_bias, ((0, 0), (0, 0), (0, size_n2 - w2_bias.shape[-1]))
            )
        w1_bias = w1_bias.astype(jnp.float32)
        w2_bias = w2_bias.astype(jnp.float32)

    # Reshape hidden_states for DMA gather: (M_src, padded_K1) -> 3D.
    # Uses padded_k1 for the reshape since hidden_states was padded for DMA.
    hidden_3d = hidden_states.reshape(
        hidden_states.shape[0], padded_k1 // num_lanes, num_lanes
    )

    fused_dims = FusedDims(
        size_m=size_m,
        size_k1=size_k1,
        padded_k1=padded_k1,
        size_n1=size_n1,
        size_k2=size_k2,
        size_n2=aligned_n2,
        original_n2=size_n2,
        size_group=size_group,
        size_lhs_group=group_sizes.shape[0],
        size_lhs_sublane=size_lhs_sublane,
        intermediate_size=intermediate_size,
        has_bias=has_bias,
        quant_block_size=quant_block_size,
        num_scale_blocks1=total_scale_blocks1,
        num_scale_blocks2=total_scale_blocks2,
    )

    # Build GmmConfigs for each GMM (used by inner_kernel and generate_block_specs).
    rhs1_packing = (32 // jax.dtypes.itemsize_bits(w1.dtype)) if is_quantized else 1
    rhs2_packing = (32 // jax.dtypes.itemsize_bits(w2.dtype)) if is_quantized else 1
    lhs1_quant_block_size = min(256 if rhs1_quant_block_size < 512 else 512, size_k1)
    lhs2_quant_block_size = min(256 if rhs2_quant_block_size < 512 else 512, size_k2)

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=size_m,
            size_k=size_k1,
            size_n=size_n1,
            size_group=size_group,
            size_lhs_group=group_sizes.shape[0],
            size_lhs_sublane=size_lhs_sublane,
        ),
        tiles=TileSizes(tile_m=tile_m, tile_k=tile_k1, tile_n=tile_n1),
        lhs_cfgs=InputConfigs(
            quant_dtype=(
                jnp.float8_e4m3fn.dtype
                if is_quantized
                and get_maybe_quantize_lhs(
                    w1.dtype, rhs1_quant_block_size, lhs1_quant_block_size
                )
                else None
            ),
            quant_block_size=lhs1_quant_block_size,
            dtype=out_dtype,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=w1.dtype if is_quantized else None,
            quant_block_size=rhs1_quant_block_size,
            dtype=w1.dtype,
            has_bias=has_bias,
            has_scale=is_quantized,
            packing=rhs1_packing,
            num_quant_blocks=total_scale_blocks1 if is_quantized else 1,
        ),
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=act_fn,
    )

    cfgs2 = GmmConfigs(
        dims=Dimensions(
            size_m=size_m,
            size_k=size_k2,
            size_n=aligned_n2,
            size_group=size_group,
            size_lhs_group=group_sizes.shape[0],
            size_lhs_sublane=size_lhs_sublane,
        ),
        tiles=TileSizes(tile_m=tile_m, tile_k=tile_k2, tile_n=tile_n2),
        lhs_cfgs=InputConfigs(
            quant_dtype=(
                jnp.float8_e4m3fn.dtype
                if is_quantized
                and get_maybe_quantize_lhs(
                    w2.dtype, rhs2_quant_block_size, lhs2_quant_block_size
                )
                else None
            ),
            quant_block_size=lhs2_quant_block_size,
            dtype=out_dtype,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=w2.dtype if is_quantized else None,
            quant_block_size=rhs2_quant_block_size,
            dtype=w2.dtype,
            has_bias=has_bias,
            has_scale=is_quantized,
            packing=rhs2_packing,
            num_quant_blocks=total_scale_blocks2 if is_quantized else 1,
        ),
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        zero_init=False,
    )

    # Scratch shapes.
    max_num_gm = size_group + pl.cdiv(size_m, tile_m) - 1
    n2_cols = aligned_n2 // num_lanes

    scratch_shapes = [
        # metadata_ref — full metadata for all gm tiles
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
        # fused_metadata_ref — single-slot for emit_pipeline index maps
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((1,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((2,), jnp.int32),
        ),
        # gathered_lhs_2x_ref — double-buffered, reused for input gather (padded_k1
        # columns) and output gather (aligned_n2 columns). Sized to
        # max(padded_k1, aligned_n2) to handle both correctly.
        pltpu.VMEM(
            (2, tile_m, max(padded_k1, aligned_n2) // num_lanes, num_lanes), out_dtype
        ),
        # gmm1_out_ref — GMM1 output (N1/2 with fuse_act)
        pltpu.VMEM((tile_m, size_n1 // 2 if act_fn else size_n1), jnp.float32),
        # intermediate_ref — after activation, scatter layout for k-slicing
        pltpu.VMEM((tile_m // size_lhs_sublane, size_lhs_sublane, size_k2), out_dtype),
        # tiled_out_2x_ref — GMM2 output, double-buffered
        pltpu.VMEM((2, tile_m, aligned_n2), out_dtype),
        # scatter_staging_ref — separate buffer for 3D DMA staging
        pltpu.VMEM((tile_m, n2_cols, num_lanes), out_dtype),
        # partial_out1_ref (2*tile_n1 with fuse_act)
        pltpu.VMEM(
            (size_lhs_sublane, tile_n1 * 2 if act_fn else tile_n1),
            jnp.float32,
        ),
        # partial_out2_ref — not used in scatter_mode but required by inner_kernel signature
        pltpu.VMEM((size_lhs_sublane, tile_n2), jnp.float32),
        # shared_acc_ref — time-shared between GMM1 and GMM2
        pltpu.VMEM(
            (tile_m, max(tile_n1 * 2 if act_fn else tile_n1, tile_n2)), jnp.float32
        ),
        # gather_sem_ref
        pltpu.SemaphoreType.DMA((2,)),
        # staging_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # zero_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # gm_id_ref
        pltpu.SMEM((1,), jnp.int32),
        # w1_buf_ref — GMM1 weight buffer (2*tile_n1 with fuse_act)
        pltpu.VMEM(
            (
                num_w1_bufs,
                (
                    tile_k1 // (32 // jax.dtypes.itemsize_bits(w1.dtype))
                    if is_quantized
                    else tile_k1
                ),
                tile_n1 * 2 if act_fn else tile_n1,
            ),
            jnp.uint32 if is_quantized else w1.dtype,
        ),
        # w2_buf_ref — GMM2 weight buffer: (num_w2_bufs, packed_k2, tile_n2)
        pltpu.VMEM(
            (
                num_w2_bufs,
                (
                    tile_k2 // (32 // jax.dtypes.itemsize_bits(w2.dtype))
                    if is_quantized
                    else tile_k2
                ),
                tile_n2,
            ),
            jnp.uint32 if is_quantized else w2.dtype,
        ),
        # w1_scale_buf_ref — GMM1 scale buffer (2*tile_n1 with fuse_act)
        pltpu.VMEM(
            (
                num_w1_bufs,
                (-(-tile_k1 // rhs1_quant_block_size) if is_quantized else 1),
                1,
                tile_n1 * 2 if act_fn else tile_n1,
            ),
            jnp.float32,
        ),
        # w2_scale_buf_ref — GMM2 scale buffer: (num_w2_bufs, nqb2, 1, tile_n2)
        pltpu.VMEM(
            (
                num_w2_bufs,
                (-(-tile_k2 // rhs2_quant_block_size) if is_quantized else 1),
                1,
                tile_n2,
            ),
            jnp.float32,
        ),
        # w1_bias_buf_ref — GMM1 bias tile
        pltpu.VMEM((1, tile_n1 * 2 if act_fn else tile_n1), jnp.float32),
        # w2_bias_buf_ref — GMM2 bias tile
        pltpu.VMEM((1, tile_n2), jnp.float32),
        # w1_sem_ref — GMM1 weight DMA semaphores
        pltpu.SemaphoreType.DMA((num_w1_bufs,)),
        # w2_sem_ref — GMM2 weight DMA semaphores
        pltpu.SemaphoreType.DMA((num_w2_bufs,)),
        # output_sem_ref — output gather/scatter DMA
        pltpu.SemaphoreType.DMA((1,)),
    ]

    # --- SMEM-aware splitting ---
    # Scalar prefetch loads lhs_indices (M*4B) + topk_weights (M*4B)
    # into SMEM. At 16K+ tokens with topk=8, M=131K exceeds the 1MB
    # SMEM budget. Halve M until it fits, issuing multiple pallas_calls
    # that accumulate results. Each chunk gets per-slice group_sizes
    # and a fresh tuned block size lookup for the smaller M.
    # Scalar prefetch loads lhs_indices (M*4B) + topk_weights (M*4B)
    # into SMEM. When M is large, this exceeds the 1MB SMEM budget.
    # Split into chunks that fit.
    _smem_budget = int(pltpu.get_tpu_info().smem_capacity_bytes)
    _smem_per_m = 4 + 4  # lhs_indices(int32) + topk_weights(float32)
    _smem_overhead = (
        (size_group + (size_m + tile_m - 1) // tile_m + 1) * 4 * 2  # metadata
        + group_sizes.shape[0] * 4  # group_sizes
        + group_offset.shape[0] * 4  # group_offset
        + 4096  # misc (gm_id_ref, semaphores, etc)
    )
    _smem_avail = _smem_budget - _smem_overhead
    _smem_needed = size_m * _smem_per_m + _smem_overhead
    _num_splits = 1
    while _smem_needed > _smem_budget and _num_splits < 16:
        _num_splits *= 2
        _chunk_m = (size_m + _num_splits - 1) // _num_splits
        _chunk_m = (_chunk_m + tile_m - 1) // tile_m * tile_m
        _smem_needed = _chunk_m * _smem_per_m + _smem_overhead

    topk_w_flat = topk_weights.astype(jnp.float32).flatten()

    # Single call — fits in SMEM.
    scatter_out_init = jax.ShapeDtypeStruct(
        (output_size, n2_cols, num_lanes),
        out_dtype,
    )

    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

    # Build input arrays. Weights stay in (E, K, N) shape for emit_pipeline.
    # Scale stays in (E, num_blocks, 1, N) shape.
    w1_scale_input = w1_scale if is_quantized else jnp.zeros((1, 1, 1, 1), jnp.float32)
    w2_scale_input = w2_scale if is_quantized else jnp.zeros((1, 1, 1, 1), jnp.float32)
    w1_bias_input = w1_bias if has_bias else jnp.zeros((1, 1, 1), jnp.float32)
    w2_bias_input = w2_bias if has_bias else jnp.zeros((1, 1, 1), jnp.float32)
    # topk_weights passed as scalar prefetch (SMEM) for per-row scalar access.

    post_norm_w_flat = (
        post_expert_norm_weight.astype(jnp.float32).flatten()
        if post_expert_norm_weight is not None
        else jnp.zeros((1,), jnp.float32)
    )
    _has_post_norm_fused = post_expert_norm_weight is not None

    if _num_splits == 1:
        result = pl.pallas_call(
            functools.partial(
                kernel_main_fused,
                fused_dims=fused_dims,
                tile_m=tile_m,
                tile_k1=tile_k1,
                tile_k2=tile_k2,
                tile_n1=tile_n1,
                tile_n2=tile_n2,
                num_w1_bufs=num_w1_bufs,
                num_w2_bufs=num_w2_bufs,
                act_fn=act_fn,
                out_dtype=out_dtype,
                has_post_norm=_has_post_norm_fused,
                cfgs1=cfgs1,
                cfgs2=cfgs2,
            ),
            out_shape=scatter_out_init,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=7,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states_3d
                    pl.BlockSpec(memory_space=pltpu.HBM),  # w1
                    pl.BlockSpec(memory_space=pltpu.HBM),  # w2
                    pl.BlockSpec(memory_space=pltpu.HBM),  # w1_scale
                    pl.BlockSpec(memory_space=pltpu.HBM),  # w2_scale
                    pl.BlockSpec(memory_space=pltpu.HBM),  # w1_bias
                    pl.BlockSpec(memory_space=pltpu.HBM),  # w2_bias
                    pl.BlockSpec(memory_space=pltpu.VMEM),  # post_expert_norm_weight
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=compiler_params,
            name=(
                f"gmm_v2_fused-E_{size_group}-M_{size_m}-K1_{size_k1}-N1_{size_n1}"
                f"-K2_{size_k2}-N2_{size_n2}-TN1_{tile_n1}-TN2_{tile_n2}"
            ),
        )(
            group_sizes,
            group_offset,
            lhs_indices,
            topk_w_flat,
            _w1_gs_gate,
            _w1_gs_up,
            _w2_gs,
            hidden_3d,
            w1,
            w2,
            w1_scale_input,
            w2_scale_input,
            w1_bias_input,
            w2_bias_input,
            post_norm_w_flat,
        )

        return result.reshape(output_size, aligned_n2)[:, :size_n2]

    else:
        # Multi-call: split M into chunks.
        result = jnp.zeros((output_size, n2_cols, num_lanes), dtype=out_dtype)
        _chunk = (size_m + _num_splits - 1) // _num_splits
        _chunk = (_chunk + tile_m - 1) // tile_m * tile_m
        _gs_cumsum = jnp.cumsum(group_sizes)
        _gs_start = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), _gs_cumsum[:-1]])
        for _si in range(_num_splits):
            _m_lo = _si * _chunk
            _m_hi = min(_m_lo + _chunk, size_m)
            if _m_lo >= size_m:
                break
            _s_indices = lhs_indices[_m_lo:_m_hi]
            _s_topk = topk_w_flat[_m_lo:_m_hi]
            # Compute split group_sizes for this M-slice.
            _ov_start = jnp.maximum(_gs_start, _m_lo)
            _ov_end = jnp.minimum(_gs_cumsum, _m_hi)
            _s_group_sizes = jnp.maximum(0, _ov_end - _ov_start)
            _s_size_m = _m_hi - _m_lo
            # Recompute scratch shapes for smaller M.
            _s_max_num_gm = size_group + pl.cdiv(_s_size_m, tile_m) - 1
            _s_scratch = list(scratch_shapes)  # copy
            # Update metadata_ref shapes (index 0 in scratch_shapes).
            _s_scratch[0] = MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((_s_max_num_gm,), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((_s_max_num_gm + 1,), jnp.int32),
            )
            _s_out_init = jax.ShapeDtypeStruct(
                (output_size, n2_cols, num_lanes), out_dtype
            )
            _s_result = pl.pallas_call(
                functools.partial(
                    kernel_main_fused,
                    fused_dims=dataclasses.replace(fused_dims, size_m=_s_size_m),
                    tile_m=tile_m,
                    tile_k1=tile_k1,
                    tile_k2=tile_k2,
                    tile_n1=tile_n1,
                    tile_n2=tile_n2,
                    num_w1_bufs=num_w1_bufs,
                    num_w2_bufs=num_w2_bufs,
                    act_fn=act_fn,
                    out_dtype=out_dtype,
                    has_post_norm=_has_post_norm_fused,
                    cfgs1=cfgs1,
                    cfgs2=cfgs2,
                ),
                out_shape=_s_out_init,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=7,
                    in_specs=[
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                    ],
                    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                    scratch_shapes=_s_scratch,
                ),
                compiler_params=compiler_params,
                name=(
                    f"gmm_v2_fused-E_{size_group}-M_{_s_size_m}"
                    f"-K1_{size_k1}-N1_{size_n1}"
                    f"-K2_{size_k2}-N2_{size_n2}"
                    f"-split_{_si}of{_num_splits}"
                ),
            )(
                _s_group_sizes,
                group_offset,
                _s_indices,
                _s_topk,
                _w1_gs_gate,
                _w1_gs_up,
                _w2_gs,
                hidden_3d,
                w1,
                w2,
                w1_scale_input,
                w2_scale_input,
                w1_bias_input,
                w2_bias_input,
                post_norm_w_flat,
            )
            result = result + _s_result

    return result.reshape(output_size, aligned_n2)[:, :size_n2]
