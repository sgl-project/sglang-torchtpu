import logging

import jax
import jax.numpy as jnp

from .v2_fused_rs_tuned_block_sizes import (
    FUSED_RS_TUNED_BLOCK_SIZES,
    FUSED_RS_TUNED_BLOCK_SIZES_FP8COMM,
)

logger = logging.getLogger(__name__)


def _device_kind() -> str:
    """Return the physical TPU kind when JAX is only an export frontend."""
    device_kind = jax.devices()[0].device_kind
    if device_kind != "cpu":
        return device_kind

    try:
        from torch_tpu._internal.utils.hardware import get_tpu_device_name

        device_kind = get_tpu_device_name()
        return "TPU7x" if device_kind == "TPU v7" else device_kind
    except (ImportError, RuntimeError):
        return device_kind


def _dtype_key(lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype) -> str:
    return f"lhs_{jnp.dtype(lhs_dtype).name}_rhs_{jnp.dtype(rhs_dtype).name}"


def _model_key(
    k: int,
    n: int,
    groups: int,
    *,
    lhs_quantized: bool,
    rhs_quant_block_size: int | None,
) -> str:
    lhs_quant_key = "_lhs_quant" if lhs_quantized else ""
    rhs_block_key = (
        f"_rhs_bs-{rhs_quant_block_size}" if rhs_quant_block_size is not None else ""
    )
    return f"k-{k}_n-{n}_g-{groups}{lhs_quant_key}{rhs_block_key}"


def get_tuned_block_sizes_v2(
    m: int,
    k: int,
    n: int,
    num_current_groups: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    maybe_quantize_lhs: bool,
    rhs_quant_block_size: int | None,
    default_block_sizes: tuple[int, int, int],
    lhs_indices: jax.Array | None = None,
    output_indices: jax.Array | None = None,
    fuse_act: str | None = None,
    fused: bool = False,
) -> tuple[int, int, int]:
    del (
        m,
        k,
        n,
        num_current_groups,
        lhs_dtype,
        rhs_dtype,
        maybe_quantize_lhs,
        rhs_quant_block_size,
        lhs_indices,
        output_indices,
        fuse_act,
        fused,
    )
    return default_block_sizes


def get_fused_tuned_block_sizes(
    m: int,
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    num_current_groups: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    rhs_quant_block_size: int | None,
    default_block_sizes: tuple[int, int, int, int, int, int, int],
    fuse_act: str | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    del (
        m,
        k1,
        n1,
        k2,
        n2,
        num_current_groups,
        lhs_dtype,
        rhs_dtype,
        rhs_quant_block_size,
        fuse_act,
    )
    return default_block_sizes


def get_fused_rs_tuned_block_sizes(
    m: int,
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    num_current_groups: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    rhs_quant_block_size: int | None,
    default_block_sizes: tuple[int, int, int, int, int, int, int],
    fuse_act: str | None = None,
    fp8_direct_write: bool = False,
) -> tuple[int, int, int, int, int, int, int]:
    device_name = _device_kind()
    dtype_key = _dtype_key(lhs_dtype, rhs_dtype)
    lhs_quantized = rhs_dtype != jnp.bfloat16
    block_size = rhs_quant_block_size if lhs_quantized else None
    model_key1 = _model_key(
        k1,
        n1,
        num_current_groups,
        lhs_quantized=lhs_quantized,
        rhs_quant_block_size=block_size,
    )
    model_key2 = _model_key(
        k2,
        n2,
        num_current_groups,
        lhs_quantized=lhs_quantized,
        rhs_quant_block_size=block_size,
    )
    joint_key = f"{model_key1}+{model_key2}"
    m_key = f"{m}_{fuse_act}"

    if fp8_direct_write:
        try:
            return FUSED_RS_TUNED_BLOCK_SIZES_FP8COMM[device_name][dtype_key][
                joint_key
            ][m_key]
        except KeyError:
            pass

    try:
        result = FUSED_RS_TUNED_BLOCK_SIZES[device_name][dtype_key][joint_key][m_key]
    except KeyError:
        logger.warning(
            "[GMM v2 fused_rs] using default block sizes for %s %s %s %s: %s",
            device_name,
            dtype_key,
            joint_key,
            m_key,
            default_block_sizes,
        )
        result = default_block_sizes

    if fp8_direct_write and result[0] > 64:
        result = (64,) + tuple(result[1:])
    return result
