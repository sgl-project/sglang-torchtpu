# SPDX-License-Identifier: Apache-2.0
"""TPU MoE with fused ICI reduce-scatter."""

import functools
import math
import os
from typing import Any, Optional

import torch

_FUSED_RS_ROUTED_ROW_ALIGNMENT = 16



def fused_moe_rs_padded_tokens(num_tokens: int, topk: int, ep_size: int) -> int:
    """Pad tokens for both EP sharding and the fused kernel's routed rows."""
    routed_row_token_alignment = _FUSED_RS_ROUTED_ROW_ALIGNMENT // math.gcd(
        _FUSED_RS_ROUTED_ROW_ALIGNMENT, topk
    )
    token_alignment = math.lcm(ep_size, routed_row_token_alignment)
    return ((num_tokens + token_alignment - 1) // token_alignment) * token_alignment


def _get_current_tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_moe_expert_parallel_rank,
        )

        r = get_moe_expert_parallel_rank()
        if isinstance(r, (int, str)):
            return int(r)
    except Exception:
        pass
    try:
        from sglang.srt.distributed import get_tp_group
        return get_tp_group().rank
    except Exception:
        pass
    try:
        from sglang.srt.distributed.parallel_state import get_parallel
        return get_parallel().tp_rank
    except Exception:
        pass
    try:
        from sglang.srt.runtime_context import get_parallel
        p = get_parallel()
        if p is not None:
            r = getattr(p, "tp_rank", None)
            if r is not None and not hasattr(r, "_mock_return_value"):
                return int(r)
    except Exception:
        pass
    if "TP_RANK" in os.environ:
        return int(os.environ["TP_RANK"])
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "TPU_WORKER_ID" in os.environ:
        return int(os.environ["TPU_WORKER_ID"])
    if "CLOUD_TPU_TASK_ID" in os.environ:
        return int(os.environ["CLOUD_TPU_TASK_ID"])
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _get_current_global_device_id() -> int:
    try:
        from sglang.srt.runtime_context import get_parallel

        p = get_parallel()
        if p is not None:
            r = getattr(p, "gpu_id", None)
            if isinstance(r, (int, str)) and not hasattr(r, "_mock_return_value"):
                return int(r)
    except Exception:
        pass
    try:
        import torch_tpu._internal.distributed.tpu_distributed as tpu_dist

        if hasattr(tpu_dist, "global_device_id") and not hasattr(
            tpu_dist.global_device_id, "_mock_return_value"
        ):
            return int(tpu_dist.global_device_id())
    except Exception:
        pass
    return _get_current_tp_rank()


_PHYSICAL_RANK_METADATA_CACHE: dict[
    tuple[int, int, int], tuple[int, tuple[int, ...], tuple[int, ...]]
] = {}


def _get_physical_rank_metadata(
    world_size: int = 8,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    global _PHYSICAL_RANK_METADATA_CACHE
    global_device_id = _get_current_global_device_id()
    cache_key = (os.getpid(), global_device_id, world_size)
    if cache_key in _PHYSICAL_RANK_METADATA_CACHE:
        return _PHYSICAL_RANK_METADATA_CACHE[cache_key]

    tp_group = None
    try:
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            cpu_group = getattr(tp_group, "cpu_group", None)
            if cpu_group is not None and not hasattr(
                cpu_group, "_mock_return_value"
            ):
                physical_ids = [None] * world_size
                dist.all_gather_object(physical_ids, global_device_id, group=cpu_group)
                if (
                    len(physical_ids) == world_size
                    and len(set(physical_ids)) == world_size
                ):
                    physical_device_ids = tuple(physical_ids)
                    order_map = [0] * world_size
                    for logical_r, phys_id in enumerate(physical_device_ids):
                        order_map[phys_id] = logical_r
                    physical_order = tuple(order_map)
                    res = (global_device_id, physical_order, physical_device_ids)
                    _PHYSICAL_RANK_METADATA_CACHE[cache_key] = res
                    return res

            # Fallback: All-gather device IDs via TPU tensor communication on the active process group
            if dist.get_world_size() == world_size:
                device = (
                    torch.device("tpu")
                    if (hasattr(torch, "tpu") or dist.get_backend() == "tpu_dist")
                    else torch.device("cpu")
                )
                local_dev_tensor = torch.tensor(
                    [global_device_id], dtype=torch.int32, device=device
                )
                gathered_tensors = [
                    torch.zeros(1, dtype=torch.int32, device=device)
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_tensors, local_dev_tensor)
                gathered_ids = [int(t.item()) for t in gathered_tensors]
                if (
                    len(gathered_ids) == world_size
                    and len(set(gathered_ids)) == world_size
                ):
                    physical_device_ids = tuple(gathered_ids)
                    order_map = [0] * world_size
                    for logical_r, phys_id in enumerate(physical_device_ids):
                        order_map[phys_id] = logical_r
                    physical_order = tuple(order_map)
                    res = (global_device_id, physical_order, physical_device_ids)
                    _PHYSICAL_RANK_METADATA_CACHE[cache_key] = res
                    return res
    except Exception:
        pass

    if world_size == 8:
        physical_device_ids = (0, 1, 6, 7, 2, 3, 4, 5)
        order_map = [0] * world_size
        for logical_r, phys_id in enumerate(physical_device_ids):
            order_map[phys_id] = logical_r
        physical_order = tuple(order_map)
    else:
        physical_device_ids = tuple(range(world_size))
        physical_order = tuple(range(world_size))

    res = (global_device_id, physical_order, physical_device_ids)
    _PHYSICAL_RANK_METADATA_CACHE[cache_key] = res
    return res


def sync_fused_moe_rs_rank_buffers(
    model: torch.nn.Module,
    tp_rank: Optional[int] = None,
    global_device_id: Optional[int] = None,
) -> None:
    """Synchronize fused MoE RS rank and device buffers across all MoE layers in the model."""
    if model is None:
        return

    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        model = model.model
    elif not isinstance(model, torch.nn.Module):
        return

    current_tp_rank = _get_current_tp_rank() if tp_rank is None else tp_rank
    current_global_device_id = (
        _get_current_global_device_id() if global_device_id is None else global_device_id
    )

    global_device_id_meta, physical_order, physical_device_ids = _get_physical_rank_metadata()
    actual_global_device_id = (
        current_global_device_id if global_device_id is not None else global_device_id_meta
    )

    for module in model.modules():
        if (
            hasattr(module, "w13_weight")
            or hasattr(module, "_tpu_fused_rs_group_offset")
            or hasattr(module, "experts")
        ):
            device = None
            if (
                hasattr(module, "_tpu_fused_rs_group_offset")
                and getattr(module, "_tpu_fused_rs_group_offset") is not None
            ):
                device = module._tpu_fused_rs_group_offset.device
            elif (
                hasattr(module, "w13_weight")
                and getattr(module, "w13_weight") is not None
            ):
                device = module.w13_weight.device
            elif (
                hasattr(module, "_tpu_fused_rs_physical_order")
                and getattr(module, "_tpu_fused_rs_physical_order") is not None
            ):
                device = module._tpu_fused_rs_physical_order.device

            if device is None:
                continue

            num_local_experts = getattr(module, "num_local_experts", None)
            if num_local_experts is None:
                if hasattr(module, "w13_weight") and getattr(module, "w13_weight") is not None:
                    num_local_experts = module.w13_weight.shape[0]
                else:
                    num_local_experts = 0

            group_offset = torch.tensor(
                [[current_tp_rank * num_local_experts, actual_global_device_id, *physical_device_ids]],
                dtype=torch.int32,
                device=device,
            )
            module.register_buffer(
                "_tpu_fused_rs_group_offset", group_offset, persistent=False
            )
            module.register_buffer(
                "_tpu_fused_rs_physical_order",
                torch.tensor(physical_order, dtype=torch.int64, device=device),
                persistent=False,
            )


def prepare_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Prepare physical-rank metadata before Dynamo captures the MoE layer."""
    current_tp_rank = _get_current_tp_rank()
    global_device_id, physical_order, physical_device_ids = _get_physical_rank_metadata()
    device = layer.w13_weight.device

    physical_order_tensor = torch.tensor(
        physical_order,
        dtype=torch.int64,
        device=device,
    )
    local_group_offset = torch.tensor(
        [[current_tp_rank * layer.num_local_experts, global_device_id, *physical_device_ids]],
        dtype=torch.int32,
        device=device,
    )
    layer.register_buffer(
        "_tpu_fused_rs_physical_order", physical_order_tensor, persistent=False
    )
    layer.register_buffer(
        "_tpu_fused_rs_group_offset", local_group_offset, persistent=False
    )
    _build_fused_moe_rs_op()


@functools.cache
def _get_mesh(ep_size: int = 8) -> Any:
    import numpy as np
    from jax.sharding import Mesh
    from sglang.srt.hardware_backend.tpu import pallas_export

    try:
        devices = pallas_export.get_jax_export_devices(ep_size)
        return Mesh(np.asarray(devices), ("model",))
    except Exception:
        import jax
        devices = jax.devices()
        if len(devices) >= ep_size:
            return Mesh(np.asarray(devices[:ep_size]), ("model",))
        return Mesh(np.asarray(devices), ("model",))


_CANONICAL_TOKEN_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 8192]

_REGISTERED_OPS: dict[tuple, Any] = {}


def _register_fused_moe_rs_op(
    num_tokens: int,
    topk: int,
    activation: str = "silu",
    has_post_expert_norm: bool = False,
    fp8_direct_write: bool = False,
) -> Any:
    """Dynamically specialized custom op per token size and top-k."""
    cache_key = (num_tokens, topk, activation, has_post_expert_norm, fp8_direct_write)
    if cache_key in _REGISTERED_OPS:
        return _REGISTERED_OPS[cache_key]

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import _MOE_GMM_LOCAL_RS
    from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
        precompute_peer_recv_counts,
    )

    op_name = f"sglang_tpu_pallas::fused_moe_rs_{num_tokens}_{topk}"
    ep_size = 8
    mesh = _get_mesh(ep_size)

    def _wrapped(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        group_offset: jax.Array,
        post_expert_norm_weight: jax.Array,
    ) -> jax.Array:
        ep_size = 8
        num_local_exp = int(w1.shape[0]) if hasattr(w1, "shape") and len(w1.shape) > 0 else 20
        num_experts = num_local_exp * ep_size
        _has_pn_rs = bool(has_post_expert_norm)
        _pnw_rs = (
            post_expert_norm_weight
            if _has_pn_rs and post_expert_norm_weight is not None
            else jnp.zeros((1,), jnp.bfloat16)
        )

        topk_indices_flat = topk_ids.reshape(-1)
        M = topk_indices_flat.shape[0]
        deterministic_key = topk_indices_flat.astype(jnp.int32) * jnp.int32(M) + jnp.arange(M, dtype=jnp.int32)
        output_indices = jnp.argsort(deterministic_key).astype(jnp.int32)

        expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
        group_sizes = jnp.sum(
            (topk_indices_flat[:, None] == expert_ids[None, :]).astype(jnp.int32),
            axis=0,
        )

        def _sharded_call(
            hs_local,
            w1_local,
            w1_scale_local,
            w1_bias_local,
            w2_local,
            w2_scale_local,
            w2_bias_local,
            w1_gs_local,
            w2_gs_local,
            topk_weights_local,
            output_indices_local,
            group_sizes_local,
            group_offset_local,
            pnw_local,
        ):
            go_flat = group_offset_local.reshape(-1)
            expert_offset = go_flat[0]
            my_id = go_flat[1]
            ep_size = 8
            physical_device_ids = (
                go_flat[2 : 2 + ep_size]
                if go_flat.shape[0] >= 2 + ep_size
                else jnp.arange(ep_size, dtype=jnp.int32)
            )
            cumsum_sizes = jnp.cumsum(group_sizes_local)
            cumsum_with_zero = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), cumsum_sizes])
            local_start_local = jax.lax.dynamic_slice(
                cumsum_with_zero, (expert_offset,), (1,)
            )
            num_tokens = hs_local.shape[0]
            chunk_size = (num_tokens + ep_size - 1) // ep_size
            peer_recv_counts_local = precompute_peer_recv_counts(
                group_sizes=group_sizes_local,
                chunk_size=chunk_size,
                ep_size=ep_size,
                my_id=my_id,
                pack_indices=False,
                top_k=topk,
                topk_argsort_indices=output_indices_local,
                physical_device_ids=physical_device_ids,
            )
            from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
                compute_num_gm,
            )

            shard_num_gms = []
            for i in range(ep_size):
                s_go = i * num_local_exp
                s_start = cumsum_with_zero[s_go]
                s_gs = jax.lax.dynamic_slice(group_sizes_local, (s_go,), (num_local_exp,))
                shard_num_gms.append(
                    compute_num_gm(s_gs, 64, 64, start_offset=s_start)
                )
            dynamic_max_num_gm = jnp.max(jnp.stack(shard_num_gms))

            return _MOE_GMM_LOCAL_RS(
                hidden_states_local=hs_local,
                w1=w1_local,
                w1_scale=w1_scale_local,
                w1_bias=None,
                w2=w2_local,
                w2_scale=w2_scale_local,
                w2_bias=None,
                w1_global_scale=None,
                w2_global_scale=None,
                group_offset=group_offset_local,
                topk_weights=topk_weights_local,
                output_indices=output_indices_local,
                group_sizes=group_sizes_local,
                peer_recv_counts=peer_recv_counts_local,
                local_start=local_start_local,
                post_expert_norm_weight_input=pnw_local,
                activation=activation,
                topk=topk,
                ep_size=ep_size,
                ep_axis_name="model",
                has_post_norm=_has_pn_rs,
                sp_enabled=False,
                fp8_direct_write=fp8_direct_write,
                num_tokens=num_tokens,
                max_num_gm=dynamic_max_num_gm,
            )

        sharded_kernel = jax.shard_map(
            _sharded_call,
            mesh=mesh,
            in_specs=(
                P(),                 # hs_local
                P(),                 # w1_local
                P(),                 # w1_scale_local
                None,                # w1_bias_local
                P(),                 # w2_local
                P(),                 # w2_scale_local
                None,                # w2_bias_local
                None,                # w1_gs_local
                None,                # w2_gs_local
                P(),                 # topk_weights_local
                P(),                 # output_indices_local
                P(),                 # group_sizes_local
                P(),                 # group_offset_local
                P(),                 # pnw_local
            ),
            out_specs=P(),
            check_vma=False,
        )

        result = sharded_kernel(
            hidden_states,
            w1,
            w1_scale,
            None,
            w2,
            w2_scale,
            None,
            None,
            None,
            topk_weights,
            output_indices,
            group_sizes,
            group_offset,
            _pnw_rs,
        )

        chunk_size = (num_tokens + ep_size - 1) // ep_size
        return result[:chunk_size]

    # Bridge the JAX kernel into Torch with model-sharded weights and output.
    custom_op = cpu_export_jax_op(
        op_name,
        _wrapped,
        mesh=mesh,
        input_partition_specs=(
            P(),                 # hidden_states
            P(),                 # w1
            P(),                 # w2
            P(),                 # w1_scale
            P(),                 # w2_scale
            P(),                 # topk_weights
            P(),                 # topk_ids
            P(),                 # group_offset
            P(),                 # post_expert_norm_weight
        ),
        output_partition_specs=(P(),),
        donate_argnums=None,
    )

    def _fake_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        group_offset: torch.Tensor,
        post_expert_norm_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_tok = hidden_states.shape[0]
        hidden_size = w2.shape[-1] if w2.ndim == 3 else hidden_states.shape[-1]
        ep_size = 8
        chunk_size = (n_tok + ep_size - 1) // ep_size
        return torch.empty(
            (chunk_size, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    custom_op.register_fake(_fake_impl)
    _REGISTERED_OPS[cache_key] = custom_op
    return custom_op


_get_fused_moe_rs_op = _register_fused_moe_rs_op


def _build_fused_moe_rs_op():
    """Pre-register all canonical SGLang token buckets (both decode batch sizes and extend buckets)."""
    for bucket in _CANONICAL_TOKEN_BUCKETS:
        _register_fused_moe_rs_op(
            num_tokens=bucket,
            topk=8,
            activation="silu",
            has_post_expert_norm=False,
            fp8_direct_write=False,
        )
    return _register_fused_moe_rs_op(
        num_tokens=128, topk=8, fp8_direct_write=False
    )


def fused_moe_rs_call(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    group_offset: Optional[torch.Tensor] = None,
    topk: int,
    activation: str,
    has_post_expert_norm: bool,
    fp8_direct_write: bool = False,
) -> torch.Tensor:
    """Execute the fused Pallas MoE GMM kernel with ICI direct-write reduce-scatter.

    Returns:
        torch.Tensor: Local token shard of shape (chunk_size, hidden_size),
        where chunk_size = (num_tokens + ep_size - 1) // ep_size.
    """
    num_local_exp = int(w1.shape[0]) if hasattr(w1, "shape") and len(w1.shape) > 0 else 20
    ep_size = 8

    device = hidden_states.device
    if group_offset is None:
        tp_rank = _get_current_tp_rank()
        global_device_id, physical_order, physical_device_ids = _get_physical_rank_metadata()
        local_group_offset = torch.tensor(
            [[tp_rank * num_local_exp, global_device_id, *physical_device_ids]],
            dtype=torch.int32,
            device=device,
        )
    elif group_offset.ndim == 1:
        local_group_offset = group_offset.unsqueeze(0).to(torch.int32)
    elif group_offset.ndim == 2 and group_offset.shape[0] == ep_size:
        tp_rank = _get_current_tp_rank()
        local_group_offset = group_offset[tp_rank : tp_rank + 1].to(torch.int32)
    else:
        local_group_offset = group_offset.to(torch.int32)

    num_tokens = int(hidden_states.size(0))

    op_name = f"fused_moe_rs_{num_tokens}_{topk}"
    if not hasattr(torch.ops.sglang_tpu_pallas, op_name):
        _register_fused_moe_rs_op(
            num_tokens=num_tokens,
            topk=topk,
            activation=activation,
            has_post_expert_norm=has_post_expert_norm,
            fp8_direct_write=fp8_direct_write,
        )
    op = getattr(torch.ops.sglang_tpu_pallas, op_name)
    return op(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids.to(torch.int32),
        local_group_offset.to(torch.int32),
        post_expert_norm_weight,
    )


def fused_moe_rs_gather(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    group_offset: torch.Tensor,
    topk: int,
    activation: str,
    has_post_expert_norm: bool,
    fp8_direct_write: bool = False,
    physical_order_to_tp_rank: torch.Tensor,
) -> torch.Tensor:
    """Execute fused MoE RS and all-gather the output across the TP process group.

    Reorders gathered shards from physical device order back to logical TP rank order
    and slices to the unpadded token count.
    """
    from sglang.srt.distributed import get_tp_group

    token_shard = fused_moe_rs_call(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=post_expert_norm_weight,
        group_offset=group_offset,
        topk=topk,
        activation=activation,
        has_post_expert_norm=has_post_expert_norm,
        fp8_direct_write=fp8_direct_write,
    )

    shard_tokens = token_shard.shape[0]
    hidden_size = token_shard.shape[1]
    tp_group = get_tp_group()
    tp_size = physical_order_to_tp_rank.shape[0]

    gathered = tp_group.all_gather(token_shard.contiguous(), dim=0)
    gathered_3d = gathered.reshape(tp_size, shard_tokens, hidden_size)
    gathered_3d = torch.index_select(
        gathered_3d, 0, physical_order_to_tp_rank
    ).contiguous()
    output = gathered_3d.reshape(-1, hidden_size).contiguous()
    if output.shape[0] != hidden_states.shape[0]:
        output = torch.narrow(output, 0, 0, hidden_states.shape[0]).contiguous()
    return output
