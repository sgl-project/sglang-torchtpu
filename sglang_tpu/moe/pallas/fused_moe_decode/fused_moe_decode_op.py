# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# NOTE: do *not* add `from __future__ import annotations` here —
# `sglang_tpu.pallas_jax_op.cpu_export_jax_op` / `torch_tpu._internal.pallas.jax_op`
# inspects `param.annotation is jax.Array` at registration time.
"""PyTorch custom-op registration, config helpers, and model integration for `fused_moe_decode`.

Enables switching MoE implementations via either:
  1. Server argument: `--moe-runner-backend pallas_tpu_fused_decode`
  2. Environment variable: `SGLANG_TPU_MOE_IMPLEMENTATION=fused_moe_decode`
     or `SGLANG_TPU_ENABLE_FUSED_MOE_DECODE_KERNEL=1`

For decode batches (`num_tokens <= SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS`,
default `256` as an example threshold), eligible MoE layers execute the
single-kernel `fused_moe_decode` Pallas op (fusing TopK mask + sparse expert
gather + routed gate/up/SwiGLU/down + shared expert MLP + cross-device
All-Reduce). Batches with `num_tokens > SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS`
automatically use the standard `gmm_v2` prefill path.
"""

import functools
import logging
from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.rank_local import traced_rank

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_OPS: dict[tuple, Callable] = {}
_OPS_LOCK = Lock()

_DEFAULT_MESH_AXIS = "model"

_FUSED_DECODE_BACKEND_NAMES = frozenset(
    {
        "fused_moe_decode",
        "pallas_tpu_fused_decode",
        "tpu_fused_moe_decode",
    }
)


def _own_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure `tensor` owns a standalone, contiguous storage buffer for PJRT/Pallas.

    When TorchTPU exports a `torch.Tensor` as a PJRT buffer for a custom Pallas/JAX
    op, it requires:
      1. `tensor.is_contiguous()` (row-major layout with no transposed strides),
      2. `tensor.storage_offset() == 0` (starts at the beginning of the allocation), and
      3. `tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()`
         (not a narrow/slice view into a larger storage buffer).
    If any of these conditions is not met, `.clone()` allocates a dedicated contiguous
    device buffer so TorchTPU does not reject the view or insert a runtime relay copy
    on every compiled call.
    """
    if (
        not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
    ):
        return tensor.clone()
    return tensor


def is_fused_moe_decode_enabled() -> bool:
    """Return True if the fused MoE decode kernel is enabled via env or runner config."""
    from sglang_tpu.envs import envs as tpu_envs

    if tpu_envs.SGLANG_TPU_ENABLE_FUSED_MOE_DECODE_KERNEL.get():
        return True
    moe_impl = tpu_envs.SGLANG_TPU_MOE_IMPLEMENTATION.get().strip().lower()
    if moe_impl in _FUSED_DECODE_BACKEND_NAMES:
        return True

    try:
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        runner_backend = get_moe_runner_backend()
        if runner_backend is not None:
            if getattr(runner_backend, "is_pallas_tpu_fused_decode", lambda: False)():
                return True
            if str(runner_backend).strip().lower() in _FUSED_DECODE_BACKEND_NAMES:
                return True
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning(
            "Failed to query get_moe_runner_backend() for fused_moe_decode: %s",
            exc,
        )

    return False


def get_fused_moe_decode_max_tokens() -> int:
    """Maximum token count (`num_tokens <= max_tokens`) that uses `fused_moe_decode`."""
    from sglang_tpu.envs import envs as tpu_envs

    return int(tpu_envs.SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS.get())


@functools.cache
def _get_mesh(world_size: int, axis_name: str = _DEFAULT_MESH_AXIS):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh((axis_name,), (world_size,))


def _build_op(
    world_size: int,
    routed_scaling_factor: float,
    swiglu_limit: Optional[float] = None,
    has_shared: bool = True,
    reduce_output: bool = True,
    axis_name: str = _DEFAULT_MESH_AXIS,
) -> Callable:
    import jax
    from jax.sharding import PartitionSpec

    from sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_kernel import (
        fused_moe_decode,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size, axis_name=axis_name)
    spec = PartitionSpec(axis_name)
    scale_tag = f"{routed_scaling_factor:.6g}".replace(".", "_").replace("-", "m")
    limit_tag = (
        "none"
        if swiglu_limit is None
        else f"{swiglu_limit:.6g}".replace(".", "_").replace("-", "m")
    )
    mode_tag = f"s{int(has_shared)}_r{int(reduce_output)}"
    op_name = (
        f"sglang_tpu_pallas::fused_moe_decode_tp{world_size}_s{scale_tag}"
        f"_l{limit_tag}_{mode_tag}_{axis_name}"
    )

    if has_shared:

        def _shared_body(
            x: jax.Array,
            topk_ids: jax.Array,
            topk_weights: jax.Array,
            w1: jax.Array,
            w1_scale: jax.Array,
            w2: jax.Array,
            w2_scale: jax.Array,
            s_w1: jax.Array,
            s_w1_scale: jax.Array,
            s_w2: jax.Array,
            s_w2_scale: jax.Array,
            rank: jax.Array,
        ) -> jax.Array:
            expert_start = rank[0] * w2.shape[0] if world_size > 1 else None
            return fused_moe_decode(
                rank,
                x,
                w1,
                w2,
                w1_scale,
                w2_scale,
                s_w1,
                s_w1_scale,
                s_w2,
                s_w2_scale,
                None,
                topk_weights,
                topk_ids,
                n_ranks=world_size,
                expert_start=expert_start,
                routed_scale=routed_scaling_factor,
                swiglu_limit=swiglu_limit,
                has_shared=True,
                reduce_output=reduce_output,
            )

        sharded_fn = jax.shard_map(
            _shared_body,
            mesh=mesh,
            in_specs=(spec,) * 12,
            out_specs=spec,
            check_vma=False,
        )

        def _jax_fn(
            x: jax.Array,
            topk_ids: jax.Array,
            topk_weights: jax.Array,
            w1: jax.Array,
            w1_scale: jax.Array,
            w2: jax.Array,
            w2_scale: jax.Array,
            s_w1: jax.Array,
            s_w1_scale: jax.Array,
            s_w2: jax.Array,
            s_w2_scale: jax.Array,
            rank: jax.Array,
        ) -> jax.Array:
            return sharded_fn(
                x,
                topk_ids,
                topk_weights,
                w1,
                w1_scale,
                w2,
                w2_scale,
                s_w1,
                s_w1_scale,
                s_w2,
                s_w2_scale,
                rank,
            )

        return cpu_export_jax_op(
            op_name,
            _jax_fn,
            mesh=mesh,
            input_partition_specs=(spec,) * 12,
            output_partition_specs=(spec,),
        )

    def _routed_only_body(
        x: jax.Array,
        topk_ids: jax.Array,
        topk_weights: jax.Array,
        w1: jax.Array,
        w1_scale: jax.Array,
        w2: jax.Array,
        w2_scale: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        expert_start = rank[0] * w2.shape[0] if world_size > 1 else None
        return fused_moe_decode(
            rank,
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            n_ranks=world_size,
            expert_start=expert_start,
            routed_scale=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            has_shared=False,
            reduce_output=reduce_output,
        )

    sharded_routed_fn = jax.shard_map(
        _routed_only_body,
        mesh=mesh,
        in_specs=(spec,) * 8,
        out_specs=spec,
        check_vma=False,
    )

    def _jax_routed_fn(
        x: jax.Array,
        topk_ids: jax.Array,
        topk_weights: jax.Array,
        w1: jax.Array,
        w1_scale: jax.Array,
        w2: jax.Array,
        w2_scale: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return sharded_routed_fn(
            x,
            topk_ids,
            topk_weights,
            w1,
            w1_scale,
            w2,
            w2_scale,
            rank,
        )

    return cpu_export_jax_op(
        op_name,
        _jax_routed_fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 8,
        output_partition_specs=(spec,),
    )


def _get_op(
    world_size: int,
    routed_scaling_factor: float,
    swiglu_limit: Optional[float] = None,
    has_shared: bool = True,
    reduce_output: bool = True,
    axis_name: str = _DEFAULT_MESH_AXIS,
) -> Callable:
    key = (
        world_size,
        float(routed_scaling_factor),
        None if swiglu_limit is None else float(swiglu_limit),
        bool(has_shared),
        bool(reduce_output),
        str(axis_name),
    )
    op = _OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(key)
        if op is None:
            op = _build_op(*key)
            _OPS[key] = op
    return op


def _resolve_swiglu_limit(
    moe_layer: torch.nn.Module,
    experts: Optional[torch.nn.Module] = None,
    shared: Optional[torch.nn.Module] = None,
) -> Optional[float]:
    """Resolve optional SwiGLU activation clamping threshold (`swiglu_limit`).

    Some MoE architectures (such as models setting `moe_runner_config.swiglu_limit`
    or `gemm1_clamp_limit`) clamp the gate and up projections before the SwiGLU
    nonlinearity (`gate = min(gate, limit)`, `up = clip(up, -limit, limit)`) to
    prevent activation overflow in low-precision formats. Returns `None` when
    standard unclamped SwiGLU (`silu(gate) * up`) is used (e.g., DeepSeek-V2/V3,
    GLM-5.3-Flash).
    """
    for obj in (
        moe_layer,
        getattr(experts, "moe_runner_config", None),
        experts,
        shared,
    ):
        if obj is None:
            continue
        for attr in ("swiglu_limit", "gemm1_clamp_limit"):
            val = getattr(obj, attr, None)
            if val is not None:
                return float(val)
    return None


def _resolve_rank_tensor(
    layer: torch.nn.Module,
    ep_rank: int,
    ep_size: int,
) -> torch.Tensor:
    if ep_size > 1:
        traced = traced_rank(ep_rank, group_size=ep_size)
        if traced is not None:
            return traced.to(dtype=torch.int32).reshape(1)
    return layer._fmd_rank


def prepare_fp8_fused_moe_decode(layer: torch.nn.Module) -> bool:
    """Prepare routed FP8 MoE scale & rank buffers on `layer` (`FusedMoE`)."""
    if getattr(layer, "_fmd_routed_prepared", False):
        return True
    if not is_fused_moe_decode_enabled():
        return False
    if getattr(layer, "weight_block_size", None) is not None:
        return False

    w13 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    w13_scale = getattr(layer, "w13_weight_scale", None)
    w2_scale = getattr(layer, "w2_weight_scale", None)
    if w13 is None or w2 is None or w13_scale is None or w2_scale is None:
        return False
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False
    if w13.dim() != 3 or w2.dim() != 3:
        return False

    # Keep `w13_weight_scale` / `w2_weight_scale` in bfloat16 for `gmm_v2`
    # prefill (> max_tokens), and register float32 buffers for Pallas async DMA.
    w1_s_f32 = _own_storage(w13_scale.data.to(torch.float32).contiguous())
    w2_s_f32 = _own_storage(w2_scale.data.to(torch.float32).contiguous())
    parallel = get_parallel()
    ep_rank = int(parallel.moe_ep_rank)
    world_size = int(max(1, parallel.moe_ep_size))
    rank_tensor = torch.tensor([ep_rank], dtype=torch.int32, device=w13.data.device)

    layer.register_buffer("_fmd_w1_scale", w1_s_f32, persistent=False)
    layer.register_buffer("_fmd_w2_scale", w2_s_f32, persistent=False)
    layer.register_buffer("_fmd_rank", rank_tensor, persistent=False)
    layer._fmd_routed_prepared = True
    layer._fmd_max_tokens = get_fused_moe_decode_max_tokens()
    layer._fmd_ep_rank = ep_rank
    layer._fmd_world_size = world_size
    return True


def prepare_moe_layer_for_fused_decode(moe_layer: torch.nn.Module) -> bool:
    """Prepare an MoE layer (routed `experts` + optional `shared_experts`) after weight loading."""
    if getattr(moe_layer, "_tpu_fused_moe_decode_ready", False):
        return True
    moe_layer._tpu_fused_moe_decode_checked = True
    if not is_fused_moe_decode_enabled():
        return False

    parallel = get_parallel()
    tp_size = parallel.tp_size
    ep_size = parallel.moe_ep_size
    if tp_size < 1 or ep_size != tp_size or parallel.pp_size > 1:
        return False

    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        return False
    if not prepare_fp8_fused_moe_decode(experts):
        return False

    h = int(experts.w13_weight.shape[1])
    i_routed = int(experts.w2_weight.shape[1])
    if h % 512 != 0 or h % (tp_size * 128) != 0 or i_routed % 128 != 0:
        return False

    shared = getattr(moe_layer, "shared_experts", None)
    has_shared = shared is not None
    if has_shared:
        gate_up = getattr(shared, "gate_up_proj", None)
        down = getattr(shared, "down_proj", None)
        if gate_up is None or down is None:
            return False
        s_w1 = getattr(gate_up, "weight", None)
        s_w1_scale = getattr(gate_up, "weight_scale", None)
        s_w2 = getattr(down, "weight", None)
        s_w2_scale = getattr(down, "weight_scale", None)
        if s_w1 is None or s_w1_scale is None or s_w2 is None or s_w2_scale is None:
            return False
        if s_w1.dtype != torch.float8_e4m3fn or s_w2.dtype != torch.float8_e4m3fn:
            return False
        # Verify post-load K-major layout: s_w1 is [H, 2*I_S], s_w2 is [I_S, H]
        if int(s_w1.shape[0]) != h or int(s_w2.shape[1]) != h:
            return False
        i_s = int(s_w2.shape[0])
        if int(s_w1.shape[1]) != 2 * i_s:
            return False

        # Pad `i_s` to a multiple of 128 if needed (e.g. GLM-5.3-Flash TP8 has i_s=192 -> 256).
        # Zero-padding gate/up columns and down rows is mathematically exact (0 * 0 = 0).
        i_s_pad = ((i_s + 127) // 128) * 128
        s1s_raw = s_w1_scale.data.reshape(-1).to(torch.float32)
        s2s_raw = s_w2_scale.data.reshape(1, h).to(torch.float32)
        if i_s_pad != i_s:
            sw1_buf = torch.zeros(
                (h, 2 * i_s_pad), dtype=s_w1.dtype, device=s_w1.data.device
            )
            sw1_buf[:, :i_s] = s_w1.data[:, :i_s]
            sw1_buf[:, i_s_pad : i_s_pad + i_s] = s_w1.data[:, i_s : 2 * i_s]

            s1s_buf = torch.zeros(
                (1, 2 * i_s_pad), dtype=torch.float32, device=s_w1.data.device
            )
            s1s_buf[0, :i_s] = s1s_raw[:i_s]
            s1s_buf[0, i_s_pad : i_s_pad + i_s] = s1s_raw[i_s : 2 * i_s]

            sw2_buf = torch.zeros(
                (i_s_pad, h), dtype=s_w2.dtype, device=s_w2.data.device
            )
            sw2_buf[:i_s, :] = s_w2.data
        else:
            sw1_buf = _own_storage(s_w1.data.contiguous())
            s1s_buf = _own_storage(s1s_raw.reshape(1, 2 * i_s).contiguous())
            sw2_buf = _own_storage(s_w2.data.contiguous())

        s2s_buf = _own_storage(s2s_raw.contiguous())
        experts.register_buffer("_fmd_sw1", sw1_buf, persistent=False)
        experts.register_buffer("_fmd_s1s", s1s_buf, persistent=False)
        experts.register_buffer("_fmd_sw2", sw2_buf, persistent=False)
        experts.register_buffer("_fmd_s2s", s2s_buf, persistent=False)

    routed_scaling_factor = (
        1.0
        if getattr(experts, "should_fuse_routed_scaling_factor_in_topk", False)
        else float(getattr(moe_layer, "routed_scaling_factor", 1.0))
    )
    swiglu_limit = _resolve_swiglu_limit(moe_layer, experts, shared)

    # Eagerly build the custom op outside torch.compile / Dynamo tracing.
    _get_op(
        tp_size,
        routed_scaling_factor,
        swiglu_limit,
        has_shared=has_shared,
        reduce_output=True,
    )

    moe_layer._tpu_fused_moe_decode_ready = True
    moe_layer._tpu_fused_moe_decode_has_shared = has_shared
    moe_layer._tpu_fused_moe_decode_world_size = int(tp_size)
    moe_layer._tpu_fused_moe_decode_ep_rank = int(parallel.moe_ep_rank)
    moe_layer._tpu_fused_moe_decode_routed_scale = routed_scaling_factor
    moe_layer._tpu_fused_moe_decode_swiglu_limit = swiglu_limit
    moe_layer._tpu_fused_moe_decode_max_tokens = get_fused_moe_decode_max_tokens()
    experts._tpu_fused_moe_decode_layer_prepared = True
    return True


def prepare_model_for_fused_moe_decode(model: torch.nn.Module) -> int:
    """Prepare all eligible MoE modules in `model` after weight postprocessing."""
    if not is_fused_moe_decode_enabled():
        return 0

    prepared_moe_layer_count = 0
    prepared_routed_count = 0
    for module in model.modules():
        if hasattr(module, "experts") and hasattr(module, "topk"):
            if prepare_moe_layer_for_fused_decode(module):
                prepared_moe_layer_count += 1

    parallel = get_parallel()
    world_size = max(1, parallel.moe_ep_size)
    for module in model.modules():
        if getattr(module, "_tpu_fused_moe_decode_layer_prepared", False):
            continue
        if hasattr(module, "w13_weight") and hasattr(module, "w2_weight"):
            if prepare_fp8_fused_moe_decode(module):
                swiglu_limit = _resolve_swiglu_limit(module, module, None)
                routed_scale = float(getattr(module, "routed_scaling_factor", 1.0))
                _get_op(
                    world_size,
                    routed_scale,
                    swiglu_limit,
                    has_shared=False,
                    reduce_output=False,
                )
                module._fmd_standalone_ready = True
                module._fmd_world_size = world_size
                prepared_routed_count += 1

    total = prepared_moe_layer_count + prepared_routed_count
    if total > 0:
        logger.info(
            "Prepared TPU fused_moe_decode kernel on %d MoE layers "
            "and %d standalone FusedMoE layers (max_tokens=%d).",
            prepared_moe_layer_count,
            prepared_routed_count,
            get_fused_moe_decode_max_tokens(),
        )
    return total


def should_use_fused_moe_decode(
    moe_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> bool:
    """Return True when `moe_layer` should execute `fused_moe_decode` for `hidden_states`."""
    if not getattr(moe_layer, "_tpu_fused_moe_decode_ready", False):
        if not getattr(moe_layer, "_tpu_fused_moe_decode_checked", False):
            prepare_moe_layer_for_fused_decode(moe_layer)
        if not getattr(moe_layer, "_tpu_fused_moe_decode_ready", False):
            return False

    num_tokens = int(hidden_states.shape[0])
    max_tokens = int(getattr(moe_layer, "_tpu_fused_moe_decode_max_tokens", 256))
    if num_tokens <= 0 or num_tokens > max_tokens:
        return False

    if get_attn_tp_context().input_scattered:
        return False

    return True


def apply_fused_moe_decode(
    moe_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_output,
) -> torch.Tensor:
    """Execute the fused MoE + Shared Expert + All-Reduce decode kernel."""
    topk_weights, topk_ids, _ = topk_output
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    experts = moe_layer.experts
    world_size = int(moe_layer._tpu_fused_moe_decode_world_size)
    routed_scale = float(moe_layer._tpu_fused_moe_decode_routed_scale)
    swiglu_limit = moe_layer._tpu_fused_moe_decode_swiglu_limit
    has_shared = bool(moe_layer._tpu_fused_moe_decode_has_shared)

    ep_rank = int(moe_layer._tpu_fused_moe_decode_ep_rank)
    rank = _resolve_rank_tensor(experts, ep_rank, world_size)

    op = _get_op(
        world_size,
        routed_scale,
        swiglu_limit,
        has_shared=has_shared,
        reduce_output=True,
    )
    if has_shared:
        return op(
            hidden_states,
            topk_ids,
            topk_weights,
            experts.w13_weight,
            experts._fmd_w1_scale,
            experts.w2_weight,
            experts._fmd_w2_scale,
            experts._fmd_sw1,
            experts._fmd_s1s,
            experts._fmd_sw2,
            experts._fmd_s2s,
            rank,
        )
    return op(
        hidden_states,
        topk_ids,
        topk_weights,
        experts.w13_weight,
        experts._fmd_w1_scale,
        experts.w2_weight,
        experts._fmd_w2_scale,
        rank,
    )


def should_use_routed_fused_moe_decode(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    block_quant: bool,
    no_combine: bool,
) -> bool:
    """Return True for standalone `FusedMoE` layers (when not fused at parent MoE block)."""
    if block_quant or no_combine:
        return False
    if getattr(layer, "_tpu_fused_moe_decode_layer_prepared", False):
        return False
    if not getattr(layer, "_fmd_routed_prepared", False):
        return False
    num_tokens = int(hidden_states.shape[0])
    max_tokens = int(getattr(layer, "_fmd_max_tokens", 256))
    return 0 < num_tokens <= max_tokens


def apply_routed_fused_moe_decode(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """Execute routed-only `fused_moe_decode` (without shared expert or all-reduce)."""
    world_size = int(layer._fmd_world_size)
    ep_rank = int(layer._fmd_ep_rank)
    rank = _resolve_rank_tensor(layer, ep_rank, world_size)
    routed_scale = 1.0  # StandardRunner scales topk_weights or post-scales outside
    if swiglu_limit is not None:
        swiglu_limit = float(swiglu_limit)
    op = _get_op(
        world_size,
        routed_scale,
        swiglu_limit,
        has_shared=False,
        reduce_output=False,
    )
    return op(
        hidden_states,
        topk_ids.to(torch.int32),
        topk_weights.to(torch.float32),
        layer.w13_weight,
        layer._fmd_w1_scale,
        layer.w2_weight,
        layer._fmd_w2_scale,
        rank,
    )
