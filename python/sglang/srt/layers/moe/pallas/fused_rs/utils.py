"""Shared utilities for fused MoE GMM kernels."""

from typing import Union

import jax
from jax import numpy as jnp
from jax.sharding import Mesh


class ShardingAxisName:
    """Sharding axis names for MoE kernel."""

    SEQUENCE = "data"
    ATTN_DATA = "data"
    MLP_DATA = "data"
    ATTN_HEAD = "model"
    ATTN_TENSOR = None
    MLP_TENSOR = "model"
    MOE_TENSOR = "model"
    EXPERT = "model"
    EXPERT_DATA = ("data", "model")
    VOCAB = ("data", "model")


def get_mesh_shape_product(
    mesh: Mesh,
    axes: Union[str, tuple, list[str], None],
) -> int:
    """
    Get the product of mesh dimensions for one or more axes.

    Examples:
        # Single axis (defaults to 1 if not present)
        get_mesh_shape_product(mesh, "model")

        # Multiple axes - computes product of their sizes
        get_mesh_shape_product(mesh, ["model", "attn_dp"])

        # None means no sharding on this dimension
        get_mesh_shape_product(mesh, None)  # returns 1
    """
    if axes is None:
        return 1

    if isinstance(axes, str):
        axes = [axes]

    product = 1
    for axis in axes:
        product *= mesh.shape.get(axis, 1)

    return product


def flatten_partition_axes(*axis_specs):
    """Flatten nested mesh axis specs into a single PartitionSpec axis."""
    axes = ()
    for axis_spec in axis_specs:
        if axis_spec is None:
            continue
        if isinstance(axis_spec, tuple):
            axes += flatten_partition_axes(*axis_spec)
        else:
            axes += (axis_spec,)
    if len(axes) == 1:
        return axes[0]
    return axes


def combine_partition_axes(*axis_specs):
    """Flatten axis specs and drop duplicates while preserving first use."""
    axes = flatten_partition_axes(*axis_specs)
    if axes is None:
        return None
    if isinstance(axes, str):
        return axes
    combined = ()
    for axis in axes:
        if axis not in combined:
            combined += (axis,)
    if len(combined) == 1:
        return combined[0]
    return combined


def get_moe_expert_axis(mesh, default_axis=None):
    """Return the runtime FFN expert axis for MoE kernels.

    Under attention-DP, the attention mesh is attn_dp x model while FFN
    experts should still shard over all chips. Represent that as one logical
    EP axis spanning the physical mesh axes:
    attn_dp x attn_dp_expert x expert x model.
    """
    default_axis = ShardingAxisName.EXPERT if default_axis is None else default_axis
    axis_order = ("attn_dp", "attn_dp_expert", "expert", "model")
    axes = tuple(axis for axis in axis_order if axis in mesh.shape)
    if not axes:
        return default_axis
    return flatten_partition_axes(*axes)


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {scoring_fn} scoring function"
            )


def apply_act_fn(activation: str, x1: jax.Array, x2: jax.Array) -> jax.Array:
    match activation:
        case "silu":
            return jax.nn.silu(x1) * x2
        case "swigluoai":
            return _swigluoai(x1, x2)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {activation} activation function"
            )


def _swigluoai(x1: jax.Array, x2: jax.Array, alpha=1.702, limit=7.0) -> jax.Array:
    x1 = jnp.clip(x1, max=limit)
    x2 = jnp.clip(x2, min=-limit, max=limit)

    gated_activation = x1 * jax.nn.sigmoid(alpha * x1)

    return gated_activation * (x2 + 1)
