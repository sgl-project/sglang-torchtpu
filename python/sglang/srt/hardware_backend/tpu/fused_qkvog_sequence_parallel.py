# SPDX-License-Identifier: Apache-2.0
"""TorchTPU adapter for the tuned TP16 sequence-parallel QKVOG entry."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Sequence
from threading import Lock
from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.tpu.kernels.sequence_parallel.fused_rmsnorm_qkvog_mm_a2a._contract import (
    COLUMNS_PER_DESTINATION,
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_HEADS_PER_DESTINATION,
    OUTPUT_GATE_HEADS_PER_DESTINATION,
    QUERY_HEADS_PER_DESTINATION,
    TP_SIZE,
    TPU_DEVICE_KIND,
)
from sglang.srt.hardware_backend.tpu.kernels.sequence_parallel.fused_rmsnorm_qkvog_mm_a2a._schedule import (
    supports_token_count,
)

if TYPE_CHECKING:
    import jax

_RANK_METADATA_WIDTH = TP_SIZE + 2
_FUSED_QKVOG_OP: Callable | None = None
_FUSED_QKVOG_LOCK = Lock()


def tpu_sp_qkvog_config_unsupported_reason(
    *,
    tp_size: int,
    world_size: int | None = None,
    hidden_size: int,
    head_dim: int,
    query_heads_per_rank: int,
    kv_heads_per_rank: int,
    output_gate_heads_per_rank: int,
) -> str | None:
    """Return why model metadata cannot use this shape-specialized kernel."""
    if world_size is not None and world_size != TP_SIZE:
        return (
            f"the tuned QKVOG kernel requires DP1/TP{TP_SIZE} "
            f"(WORLD_SIZE={TP_SIZE}), got WORLD_SIZE={world_size}"
        )
    actual = (
        tp_size,
        hidden_size,
        head_dim,
        query_heads_per_rank,
        kv_heads_per_rank,
        output_gate_heads_per_rank,
    )
    required = (
        TP_SIZE,
        HIDDEN_SIZE,
        HEAD_DIM,
        QUERY_HEADS_PER_DESTINATION,
        KV_HEADS_PER_DESTINATION,
        OUTPUT_GATE_HEADS_PER_DESTINATION,
    )
    if actual == required:
        return None
    return (
        "the tuned QKVOG kernel requires "
        f"TP{TP_SIZE}, hidden size {HIDDEN_SIZE}, head size {HEAD_DIM}, "
        f"{QUERY_HEADS_PER_DESTINATION} query heads, "
        f"{KV_HEADS_PER_DESTINATION} KV heads, and "
        f"{OUTPUT_GATE_HEADS_PER_DESTINATION} output-gate heads per rank; got "
        f"TP{tp_size}, hidden size {hidden_size}, head size {head_dim}, "
        f"{query_heads_per_rank} query heads, {kv_heads_per_rank} KV heads, and "
        f"{output_gate_heads_per_rank} output-gate heads per rank"
    )


def validate_physical_to_logical(
    physical_to_logical: Sequence[int],
) -> tuple[int, ...]:
    """Validate the TP16 permutation required by physical-DMA scheduling."""
    mapping = tuple(int(rank) for rank in physical_to_logical)
    if len(mapping) != TP_SIZE or sorted(mapping) != list(range(TP_SIZE)):
        raise ValueError(
            f"physical_to_logical must be a TP{TP_SIZE} permutation, got {mapping}"
        )
    # The 1K hybrid schedule transfers its final two same-chip source shards
    # as one contiguous block. Reject a topology that would make the block land
    # in non-contiguous or reversed logical rows.
    for physical_id in range(0, TP_SIZE, 2):
        if (
            mapping[physical_id] % 2
            or mapping[physical_id + 1] != mapping[physical_id] + 1
        ):
            raise ValueError(
                "physical_to_logical must preserve ordered two-core pairs for "
                f"the 1K schedule, got {mapping}"
            )
    return mapping


def make_local_rank_metadata(
    *,
    physical_id: int,
    logical_id: int,
    physical_to_logical: Sequence[int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build this rank's scalar-prefetch row for the Pallas kernel."""
    mapping = validate_physical_to_logical(physical_to_logical)
    if not 0 <= physical_id < TP_SIZE:
        raise ValueError(f"physical_id must be in [0, {TP_SIZE}), got {physical_id}")
    if not 0 <= logical_id < TP_SIZE:
        raise ValueError(f"logical_id must be in [0, {TP_SIZE}), got {logical_id}")
    if mapping[physical_id] != logical_id:
        raise ValueError(
            f"physical rank {physical_id} owns logical rank {mapping[physical_id]}, "
            f"not {logical_id}"
        )
    return torch.tensor(
        [physical_id, logical_id, *mapping],
        dtype=torch.int32,
        device=device,
    )


def pack_local_qkvog_weights_by_physical_destination(
    gathered_logical_weights: torch.Tensor,
    physical_to_logical: Sequence[int],
) -> torch.Tensor:
    """Convert gathered local ``[Q|K|V|OG]`` blocks to kernel layout.

    ``gathered_logical_weights`` is the result of a dim-0 TP all-gather and has
    shape ``[16 * 1152, 8192]`` in logical-rank order. The returned replicated
    matrix is ``[8192, 16 * 1152]`` with destination blocks in physical order,
    matching the Pallas DMA schedule.
    """
    mapping = validate_physical_to_logical(physical_to_logical)
    expected = (TP_SIZE * COLUMNS_PER_DESTINATION, HIDDEN_SIZE)
    if tuple(gathered_logical_weights.shape) != expected:
        raise ValueError(
            f"gathered QKVOG weight must have shape {expected}, got "
            f"{tuple(gathered_logical_weights.shape)}"
        )
    if gathered_logical_weights.dtype != torch.bfloat16:
        raise ValueError(
            "the fused QKVOG kernel requires BF16 weights, got "
            f"{gathered_logical_weights.dtype}"
        )
    physical_order = torch.tensor(
        mapping,
        dtype=torch.long,
        device=gathered_logical_weights.device,
    )
    blocks = gathered_logical_weights.reshape(
        TP_SIZE,
        COLUMNS_PER_DESTINATION,
        HIDDEN_SIZE,
    )
    return blocks.index_select(0, physical_order).reshape(expected).t().contiguous()


def prepare_tpu_sp_qkvog_weight(
    local_weight: torch.Tensor,
    tp_group=None,
    *,
    hidden_size: int,
    head_dim: int,
    query_heads_per_rank: int,
    kv_heads_per_rank: int,
    output_gate_heads_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather and pack one layer's QKVOG weight once, after checkpoint load."""
    if tp_group is None:
        from sglang.srt.runtime_context import get_parallel

        tp_group = get_parallel().attn_tp_group
    config_reason = tpu_sp_qkvog_config_unsupported_reason(
        tp_size=tp_group.world_size,
        world_size=int(os.environ.get("WORLD_SIZE", tp_group.world_size)),
        hidden_size=hidden_size,
        head_dim=head_dim,
        query_heads_per_rank=query_heads_per_rank,
        kv_heads_per_rank=kv_heads_per_rank,
        output_gate_heads_per_rank=output_gate_heads_per_rank,
    )
    if config_reason is not None:
        raise ValueError(config_reason)
    expected_local = (COLUMNS_PER_DESTINATION, HIDDEN_SIZE)
    if tuple(local_weight.shape) != expected_local:
        raise ValueError(
            f"local QKVOG weight must have shape {expected_local}, got "
            f"{tuple(local_weight.shape)}"
        )
    if local_weight.dtype != torch.bfloat16:
        raise ValueError(
            f"the fused QKVOG kernel requires BF16 weights, got {local_weight.dtype}"
        )

    from sglang.srt.hardware_backend.tpu.rank_topology import (
        get_tpu_physical_rank_metadata,
    )

    physical_id, physical_to_logical, logical_to_physical = (
        get_tpu_physical_rank_metadata()
    )
    physical_to_logical = validate_physical_to_logical(physical_to_logical)
    logical_id = physical_to_logical[physical_id]
    if (
        logical_to_physical[logical_id] != physical_id
        or logical_id != tp_group.rank_in_group
    ):
        raise RuntimeError("inconsistent TPU logical/physical rank metadata")
    gathered = tp_group.all_gather(local_weight, dim=0).contiguous()
    packed = pack_local_qkvog_weights_by_physical_destination(
        gathered,
        physical_to_logical,
    )
    metadata = make_local_rank_metadata(
        physical_id=physical_id,
        logical_id=logical_id,
        physical_to_logical=physical_to_logical,
        device=local_weight.device,
    )
    return packed, metadata


def tpu_sp_qkvog_device_unsupported_reason() -> str | None:
    """Return why the current TPU generation cannot use the tuned schedules."""
    try:
        from sglang.srt.hardware_backend.tpu.pallas_export import (
            get_pallas_tpu_device_kind,
        )

        device_kind = get_pallas_tpu_device_kind()
    except (ImportError, RuntimeError) as error:
        return f"could not identify the TPU device kind: {error}"
    if device_kind != TPU_DEVICE_KIND:
        return (
            f"the tuned QKVOG schedules require {TPU_DEVICE_KIND}, "
            f"got {device_kind}"
        )
    return None


def supports_tpu_sp_qkvog_device() -> bool:
    """Whether this process is running on the topology the schedules target."""
    return tpu_sp_qkvog_device_unsupported_reason() is None


@functools.cache
def _get_mesh():
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != TP_SIZE:
        raise RuntimeError(
            f"the tuned QKVOG kernel requires WORLD_SIZE={TP_SIZE}, got {world_size}"
        )
    device_reason = tpu_sp_qkvog_device_unsupported_reason()
    if device_reason is not None:
        raise RuntimeError(device_reason)
    export_devices = np.asarray(get_torchtpu_export_devices(world_size)).reshape(1, -1)
    return Mesh(export_devices, ("data", "model"))


def _build_fused_qkvog_op() -> Callable:
    global _FUSED_QKVOG_OP
    if _FUSED_QKVOG_OP is not None:
        return _FUSED_QKVOG_OP

    with _FUSED_QKVOG_LOCK:
        if _FUSED_QKVOG_OP is not None:
            return _FUSED_QKVOG_OP

        import jax
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec as P

        from sglang.srt.hardware_backend.tpu.kernels.sequence_parallel.fused_rmsnorm_qkvog_mm_a2a import (
            fused_rmsnorm_qkvog_mm_a2a,
        )
        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        mesh = _get_mesh()

        def _wrapped(
            hidden_states: jax.Array,
            packed_weight: jax.Array,
            gain: jax.Array,
            cos: jax.Array,
            sin: jax.Array,
            rank_metadata: jax.Array,
            eps: float,
            eps_qk: float,
            scale_query_by: float,
            use_qk_norm: bool,
            use_rope: bool,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return fused_rmsnorm_qkvog_mm_a2a(
                hidden_states,
                packed_weight,
                gain,
                cos,
                sin,
                mesh,
                "model",
                rank_metadata=rank_metadata,
                eps=eps,
                eps_qk=eps_qk,
                scale_query_by=scale_query_by,
                use_qk_norm=use_qk_norm,
                use_rope=use_rope,
                kv_dtype=jnp.bfloat16,
            )

        # ``from __future__ import annotations`` stringizes the inline types,
        # while TorchTPU's JAX bridge deliberately requires the concrete JAX
        # and POD classes to determine tensor versus static arguments.
        _wrapped.__annotations__ = {
            "hidden_states": jax.Array,
            "packed_weight": jax.Array,
            "gain": jax.Array,
            "cos": jax.Array,
            "sin": jax.Array,
            "rank_metadata": jax.Array,
            "eps": float,
            "eps_qk": float,
            "scale_query_by": float,
            "use_qk_norm": bool,
            "use_rope": bool,
            "return": tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        }

        op = cpu_export_jax_op(
            "sglang_tpu_pallas::fused_sp_rmsnorm_qkvog",
            _wrapped,
            mesh=mesh,
            input_partition_specs=(
                P("model", None),
                P(),
                P(),
                P("model", None),
                P("model", None),
                P("model"),
                None,
                None,
                None,
                None,
                None,
            ),
            output_partition_specs=(
                P(None, "model", None),
                P(None, "model", None),
                P(None, "model", None),
                P(None, "model", None),
            ),
        )

        # cpu_export_jax_op installs the fake implementation from the exported
        # output shardings; registering another fake here would conflict with
        # that bridge-owned registration.
        _FUSED_QKVOG_OP = op
        return op


def register_tpu_sp_fused_qkvog_op() -> None:
    """Register the custom op before torch.compile captures a model."""
    _build_fused_qkvog_op()


def can_use_tpu_sp_fused_qkvog(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor | None,
    gain: torch.Tensor | None,
    rank_metadata: torch.Tensor | None,
) -> bool:
    """Return whether tensors satisfy the tuned kernel contract."""
    if hidden_states.ndim != 2:
        return False

    global_tokens = hidden_states.shape[0] * TP_SIZE
    return bool(
        hidden_states.shape[1] == HIDDEN_SIZE
        and hidden_states.dtype in (torch.float32, torch.bfloat16)
        and supports_token_count(global_tokens)
        and packed_weight is not None
        and tuple(packed_weight.shape)
        == (HIDDEN_SIZE, TP_SIZE * COLUMNS_PER_DESTINATION)
        and packed_weight.dtype == torch.bfloat16
        and gain is not None
        and tuple(gain.shape) == (HIDDEN_SIZE,)
        and gain.dtype == torch.bfloat16
        and rank_metadata is not None
        and tuple(rank_metadata.shape) == (_RANK_METADATA_WIDTH,)
        and rank_metadata.dtype == torch.int32
    )


def tpu_sp_fused_qkvog(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rank_metadata: torch.Tensor,
    *,
    eps: float,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    collective_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused RMSNorm, QKVOG projection, Q/K epilogue, and token/head A2A.

    ``collective_id`` remains accepted for compatibility with serving callers,
    but is intentionally not forwarded to the exported custom op. The Pallas
    implementation owns a finite channel namespace by schedule so layer IDs do
    not create otherwise-identical compiler specializations.
    """
    if not can_use_tpu_sp_fused_qkvog(
        hidden_states,
        packed_weight,
        gain,
        rank_metadata,
    ):
        raise ValueError("inputs do not satisfy the tuned TP16 QKVOG contract")
    expected_rope = (hidden_states.shape[0], HEAD_DIM)
    if tuple(cos.shape) != expected_rope or tuple(sin.shape) != expected_rope:
        raise ValueError(
            f"local cos/sin must have shape {expected_rope}, got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    return _build_fused_qkvog_op()(
        hidden_states,
        packed_weight,
        gain,
        cos,
        sin,
        rank_metadata,
        eps,
        eps_qk,
        scale_query_by,
        use_qk_norm,
        use_rope,
    )


__all__ = [
    "can_use_tpu_sp_fused_qkvog",
    "make_local_rank_metadata",
    "pack_local_qkvog_weights_by_physical_destination",
    "prepare_tpu_sp_qkvog_weight",
    "register_tpu_sp_fused_qkvog_op",
    "supports_tpu_sp_qkvog_device",
    "tpu_sp_qkvog_config_unsupported_reason",
    "tpu_sp_qkvog_device_unsupported_reason",
    "tpu_sp_fused_qkvog",
    "validate_physical_to_logical",
]
