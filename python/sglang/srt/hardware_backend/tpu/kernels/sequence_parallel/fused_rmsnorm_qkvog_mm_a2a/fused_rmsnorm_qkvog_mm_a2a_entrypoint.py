# SPDX-License-Identifier: Apache-2.0

"""Public entrypoint for fused SP QKVOG projection and attention epilogue.

The public contract is deliberately independent of the internal TPU schedule:
the activation is token-sharded, the weight and RMSNorm gain are replicated,
and the result is sharded by QKVOG destination.  The static input shape selects
a schedule before Pallas tracing, so callers see one operation while each token
range still gets the measured-fastest TPU program.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ._contract import COLUMNS_PER_DESTINATION, HEAD_DIM, HIDDEN_SIZE, TP_SIZE
from ._epilogue import qkvog_epilogue, split_epilogued_qkvog
from ._kernel_a2a import run_a2a
from ._kernel_hybrid_gather_a2a import run_hybrid_gather_a2a
from ._schedule import (
    AllToAllPlan,
    HybridGatherAllToAllPlan,
    SchedulePlan,
    make_schedule_plan,
)


def pack_qkvog_weight_by_destination(
    q_weight: jax.Array,
    k_weight: jax.Array,
    v_weight: jax.Array,
    og_weight: jax.Array,
    *,
    tp_size: int,
) -> jax.Array:
    """Pack canonical 3-D Q/K/V/OG weights into destination-major blocks.

    Canonical weights group all query heads, then all key heads, and so on.
    The distributed projection instead needs one contiguous
    ``[Q_local | K_local | V_local | OG_local]`` block per TP destination so
    each completed block can be sent directly to its owner.
    """
    weights = (q_weight, k_weight, v_weight, og_weight)
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if any(weight.ndim != 3 for weight in weights):
        raise ValueError("Q/K/V/OG weights must all have shape [hidden, heads, dim]")
    hidden_size, _, head_dim = q_weight.shape
    if any(
        weight.shape[0] != hidden_size or weight.shape[2] != head_dim
        for weight in weights
    ):
        raise ValueError("Q/K/V/OG weights must share hidden and head dimensions")
    if any(weight.shape[1] % tp_size for weight in weights):
        raise ValueError(f"every head count must be divisible by TP{tp_size}")

    grouped = tuple(
        weight.reshape(hidden_size, tp_size, weight.shape[1] // tp_size, head_dim)
        for weight in weights
    )
    return jnp.concatenate(grouped, axis=2).reshape(hidden_size, -1)


def _run_local_schedule(
    local_x: jax.Array,
    replicated_weight: jax.Array,
    replicated_gain: jax.Array,
    local_cos: jax.Array,
    local_sin: jax.Array,
    local_rank_metadata: jax.Array,
    *,
    axis_name: str,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    kv_dtype: Any,
    plan: SchedulePlan,
    collective_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Trace the selected projection schedule and return final components."""
    if isinstance(plan, HybridGatherAllToAllPlan):
        projected = run_hybrid_gather_a2a(
            local_x,
            replicated_weight,
            replicated_gain,
            local_cos,
            local_sin,
            local_rank_metadata,
            axis_name=axis_name,
            eps=eps,
            eps_qk=eps_qk,
            scale_query_by=scale_query_by,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
            plan=plan,
            collective_id=collective_id,
        )
        return split_epilogued_qkvog(projected, kv_dtype=kv_dtype)
    elif isinstance(plan, AllToAllPlan):
        outputs = run_a2a(
            local_x,
            replicated_weight,
            replicated_gain,
            local_cos,
            local_sin,
            local_rank_metadata,
            axis_name=axis_name,
            eps=eps,
            eps_qk=eps_qk,
            scale_query_by=scale_query_by,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
            kv_dtype=kv_dtype,
            plan=plan,
            collective_id=collective_id,
        )
        return tuple(
            output.reshape(output.shape[0], output.shape[1] // HEAD_DIM, HEAD_DIM)
            for output in outputs
        )
    else:
        raise TypeError(f"unsupported schedule plan: {type(plan).__name__}")


def _validate_rope_inputs(
    x: jax.Array,
    cos: jax.Array | None,
    sin: jax.Array | None,
    *,
    use_rope: bool,
) -> tuple[jax.Array, jax.Array]:
    if not use_rope and cos is None and sin is None:
        zeros = jnp.zeros((x.shape[0], HEAD_DIM), dtype=jnp.float32)
        return zeros, zeros
    if cos is None or sin is None:
        raise ValueError("cos and sin are both required when RoPE is enabled")
    expected = (x.shape[0], HEAD_DIM)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(
            f"cos and sin must both have global shape {expected}, "
            f"got {cos.shape} and {sin.shape}"
        )
    return cos, sin


def fused_rmsnorm_qkvog_mm_a2a(
    x: jax.Array,
    weight: jax.Array,
    gain: jax.Array,
    cos: jax.Array | None,
    sin: jax.Array | None,
    mesh: Mesh,
    axis_name: str,
    *,
    rank_metadata: jax.Array | None = None,
    eps: float = 1e-5,
    eps_qk: float = 1e-5,
    scale_query_by: float = 1.0,
    use_qk_norm: bool = True,
    use_rope: bool = True,
    kv_dtype: Any = jnp.bfloat16,
    collective_id: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the tuned TP16 fused SP attention-entry operation.

    Args:
        x: Token-sharded ``[tokens, 8192]`` activation.
        weight: Replicated ``[8192, 16 * 1152]`` packed QKVOG weight.
        gain: Replicated ``[8192]`` RMSNorm gain.
        cos: Token-sharded global ``[tokens, 64]`` pair-repeated RoPE cosines.
        sin: Token-sharded global ``[tokens, 64]`` pair-repeated RoPE sines.
        mesh: Device mesh containing a size-16 tensor-parallel axis.
        axis_name: Name of the tensor-parallel mesh axis.
        eps: RMSNorm epsilon.
        rank_metadata: Token-sharded ``[16 * 18]`` int32 metadata. Each local
            row is ``[physical_id, logical_id, *physical_to_logical]``. ``None``
            selects identity placement for native JAX callers.
        collective_id: Optional Pallas collective channel override. The default
            uses the schedule's stable channel so equivalent layer invocations
            share one compiled kernel safely.

    Returns:
        Attention-ready ``(q, k, v, og)`` with global shapes
        ``[tokens,128,64]``, ``[tokens,16,64]``, ``[tokens,16,64]``, and
        ``[tokens,128,64]``. Heads are sharded over ``axis_name``. Q/OG use
        BF16 and K/V use ``kv_dtype``.

    The local token count selects a static schedule: 64 rows gathers eight
    source shards before projecting two destinations; 128 rows gathers two
    on-chip source shards before projecting eight destinations; larger inputs
    project local rows and push completed attention-ready outputs directly.
    """
    if axis_name not in mesh.axis_names:
        raise ValueError(f"mesh has no axis named {axis_name!r}")
    axis_size = mesh.shape[axis_name]
    if axis_size != TP_SIZE:
        raise ValueError(f"the tuned kernel requires TP{TP_SIZE}, got TP{axis_size}")
    if x.ndim != 2 or x.shape[0] % TP_SIZE:
        raise ValueError(
            f"x must be [tokens, hidden] with tokens divisible by TP{TP_SIZE}"
        )
    if x.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"the tuned kernel requires hidden size {HIDDEN_SIZE}")
    expected_weight_shape = (
        HIDDEN_SIZE,
        TP_SIZE * COLUMNS_PER_DESTINATION,
    )
    if weight.shape != expected_weight_shape:
        raise ValueError(
            f"weight must be the replicated packed {expected_weight_shape} QKVOG matrix"
        )
    if gain.shape != (HIDDEN_SIZE,):
        raise ValueError(f"gain must have shape ({HIDDEN_SIZE},)")
    if x.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("x must use float32 or bfloat16")
    if any(value.dtype != jnp.bfloat16 for value in (weight, gain)):
        raise ValueError("weight and gain must use bfloat16")
    cos, sin = _validate_rope_inputs(x, cos, sin, use_rope=use_rope)
    if rank_metadata is None:
        logical_ranks = np.arange(TP_SIZE, dtype=np.int32)
        rank_metadata = jax.device_put(
            np.concatenate(
                (
                    logical_ranks[:, None],
                    logical_ranks[:, None],
                    np.broadcast_to(logical_ranks, (TP_SIZE, TP_SIZE)),
                ),
                axis=1,
            ).reshape(-1),
            NamedSharding(mesh, P(axis_name)),
        )
    if rank_metadata.shape != (TP_SIZE * (TP_SIZE + 2),):
        raise ValueError(
            f"rank_metadata must have shape ({TP_SIZE * (TP_SIZE + 2)},), "
            f"got {rank_metadata.shape}"
        )
    if rank_metadata.dtype != jnp.int32:
        raise ValueError("rank_metadata must use int32")

    local_rows = x.shape[0] // TP_SIZE
    plan = make_schedule_plan(local_rows)
    if collective_id is None:
        collective_id = plan.collective_id
    local_operation = functools.partial(
        _run_local_schedule,
        axis_name=axis_name,
        eps=eps,
        eps_qk=eps_qk,
        scale_query_by=scale_query_by,
        use_qk_norm=use_qk_norm,
        use_rope=use_rope,
        kv_dtype=kv_dtype,
        plan=plan,
        collective_id=collective_id,
    )
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(
            P(axis_name, None),
            P(None, None),
            P(None),
            P(axis_name, None),
            P(axis_name, None),
            P(axis_name),
        ),
        out_specs=(
            P(None, axis_name, None),
            P(None, axis_name, None),
            P(None, axis_name, None),
            P(None, axis_name, None),
        ),
        check_vma=False,
    )(x, weight, gain, cos, sin, rank_metadata)


def jax_rmsnorm_qkvog_mm_a2a(
    x: jax.Array,
    weight: jax.Array,
    gain: jax.Array,
    cos: jax.Array | None,
    sin: jax.Array | None,
    mesh: Mesh,
    axis_name: str,
    *,
    eps: float = 1e-5,
    eps_qk: float = 1e-5,
    scale_query_by: float = 1.0,
    use_qk_norm: bool = True,
    use_rope: bool = True,
    kv_dtype: Any = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure-JAX same-contract numerical oracle and performance baseline."""

    cos, sin = _validate_rope_inputs(x, cos, sin, use_rope=use_rope)

    def _local(local_x, replicated_weight, replicated_gain, local_cos, local_sin):
        if (
            replicated_weight.ndim != 2
            or replicated_weight.shape[1] % mesh.shape[axis_name]
        ):
            raise ValueError("flat weight output width must divide the collective size")
        grouped_weight = replicated_weight.reshape(
            replicated_weight.shape[0],
            mesh.shape[axis_name],
            replicated_weight.shape[1] // mesh.shape[axis_name],
        )
        x_f32 = local_x.astype(jnp.float32)
        inverse_rms = lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
        normalized = (
            x_f32 * inverse_rms * replicated_gain.astype(jnp.float32)[jnp.newaxis, :]
        ).astype(jnp.bfloat16)
        projected = lax.dot_general(
            normalized,
            grouped_weight,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        exchanged = lax.all_to_all(
            projected,
            axis_name,
            split_axis=1,
            concat_axis=0,
            tiled=True,
        )
        if use_rope:
            full_cos = lax.all_gather(local_cos, axis_name, axis=0, tiled=True)
            full_sin = lax.all_gather(local_sin, axis_name, axis=0, tiled=True)
        else:
            full_cos, full_sin = local_cos, local_sin
        return qkvog_epilogue(
            exchanged[:, 0, :],
            full_cos,
            full_sin,
            eps_qk=eps_qk,
            scale_query_by=scale_query_by,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
            kv_dtype=kv_dtype,
        )

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(axis_name, None),
            P(None, None),
            P(None),
            P(axis_name, None),
            P(axis_name, None),
        ),
        out_specs=(
            P(None, axis_name, None),
            P(None, axis_name, None),
            P(None, axis_name, None),
            P(None, axis_name, None),
        ),
        check_vma=False,
    )(x, weight, gain, cos, sin)
