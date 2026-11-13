# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# NOTE: do *not* add `from __future__ import annotations` here —
# `sglang_tpu.pallas_jax_op.cpu_export_jax_op` / `torch_tpu._internal.pallas.jax_op`
# inspects `param.annotation is jax.Array` at registration time.
"""PyTorch custom-op registration and model integration for `fused_moe_decode`.

Enabled with `--moe-runner-backend pallas_tpu_fused_decode`.

Eligible MoE layers then run the single-kernel `fused_moe_decode` Pallas op
(fusing TopK mask + sparse expert gather + routed gate/up/SwiGLU/down + shared
expert MLP + cross-device all-reduce) for `ForwardMode.DECODE` batches with
`num_tokens <= SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS` (default `256`). Every
other batch (extend/prefill, mixed chunked prefill, speculative target-verify
and draft-extend, and decode batches above the threshold) uses the regular
`gmm_v2` path.

Layers with tensor-parallel experts (`moe_ep_size == 1`, attention TP over the
whole world) run `moe_tp_decode` instead, for decode batches up to the largest
row bucket that kernel admits (64 tokens); see `sglang_tpu.moe.pallas.moe_tp_decode`.

Because the kernel returns an already all-reduced output, the decoder layer
must tell its FFN boundary not to sum it again (`summed_by_compute`). Only MoE
blocks whose decoder layer does that set `allow_tpu_fused_moe_decode = True`
(`DeepseekV2DecoderLayer` and `Glm5NextDecoderLayer`); others are never
prepared.
"""

import functools
import logging
from threading import Lock
from typing import Callable, Optional

import torch

from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.rank_local import traced_rank

logger = logging.getLogger(__name__)

_OPS: dict[tuple, Callable] = {}
_OPS_LOCK = Lock()

_DEFAULT_MESH_AXIS = "model"


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
    """Return True if `--moe-runner-backend pallas_tpu_fused_decode` is selected."""
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    return get_moe_runner_backend().is_pallas_tpu_fused_decode()


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
    mode_tag = f"s{int(has_shared)}"
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
                None,  # shared_gate_weight: no sigmoid shared-expert gate
                topk_weights,
                topk_ids,
                n_ranks=world_size,
                expert_start=expert_start,
                routed_scale=routed_scaling_factor,
                swiglu_limit=swiglu_limit,
                has_shared=True,
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
            None,  # shared_gate_weight
            topk_weights,
            topk_ids,
            n_ranks=world_size,
            expert_start=expert_start,
            routed_scale=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            has_shared=False,
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
    axis_name: str = _DEFAULT_MESH_AXIS,
) -> Callable:
    key = (
        world_size,
        float(routed_scaling_factor),
        None if swiglu_limit is None else float(swiglu_limit),
        bool(has_shared),
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
    """Resolve the optional SwiGLU clamp threshold (`swiglu_limit`).

    Models with `config.swiglu_limit` clamp the gate and up projections before
    the SwiGLU nonlinearity (`gate = min(gate, limit)`,
    `up = clip(up, -limit, limit)`); `DeepseekV2MoE` forwards it to `FusedMoE`,
    where it lands in `moe_runner_config.swiglu_limit`. GLM-5.3-Flash sets
    `swiglu_limit = 10.0`. Returns `None` for unclamped SwiGLU
    (`silu(gate) * up`), e.g. DeepSeek-V2/V3.
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


def _prepare_fp8_routed_experts(layer: torch.nn.Module) -> bool:
    """Prepare routed FP8 MoE scale & rank buffers on `layer` (`FusedMoE`)."""
    if getattr(layer, "_fmd_routed_prepared", False):
        return True
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
    ep_rank = int(get_parallel().moe_ep_rank)
    rank_tensor = torch.tensor([ep_rank], dtype=torch.int32, device=w13.data.device)

    layer.register_buffer("_fmd_w1_scale", w1_s_f32, persistent=False)
    layer.register_buffer("_fmd_w2_scale", w2_s_f32, persistent=False)
    layer.register_buffer("_fmd_rank", rank_tensor, persistent=False)
    layer._fmd_routed_prepared = True
    return True


def _static_ineligibility_reason(moe_layer: torch.nn.Module) -> Optional[str]:
    """Why `moe_layer`'s forward could never take the fused path, or None.

    These are per-layer properties fixed at construction, so they are checked
    once here instead of at every forward.
    """
    if not getattr(moe_layer, "allow_tpu_fused_moe_decode", False):
        return "decoder layer does not pass summed_by_compute to its FFN boundary"
    if getattr(moe_layer, "is_hash", False):
        return "hash-routed MoE layer"
    gate = getattr(moe_layer, "gate", None)
    if getattr(gate, "e_score_correction_bias_vl", None) is not None:
        return "vision top-k routing"
    if getattr(moe_layer, "_enable_a2a_moe", False):
        return "all-to-all (DeepEP/mega MoE) dispatch"
    if getattr(moe_layer, "_shared_expert_tp1", False):
        return "replicated (TP1) shared expert"
    return None


def prepare_moe_layer_for_fused_decode(moe_layer: torch.nn.Module) -> bool:
    """Prepare an MoE layer (routed `experts` + optional `shared_experts`) after weight loading."""
    if getattr(moe_layer, "_tpu_fused_moe_decode_ready", False):
        return True
    moe_layer._tpu_fused_moe_decode_checked = True
    if not is_fused_moe_decode_enabled():
        return False
    reason = _static_ineligibility_reason(moe_layer)
    if reason is not None:
        logger.debug(
            "fused_moe_decode disabled for %s: %s", type(moe_layer).__name__, reason
        )
        return False

    parallel = get_parallel()
    tp_size = parallel.tp_size
    ep_size = parallel.moe_ep_size
    if tp_size < 1 or parallel.pp_size > 1:
        return False
    if ep_size == tp_size:
        mode = "ep"
    else:
        # Tensor-parallel experts run the moe_tp_decode kernel instead.
        from sglang_tpu.moe.pallas.moe_tp_decode.op import moe_tp_layout_reason

        reason = moe_tp_layout_reason()
        if reason is not None:
            logger.debug("fused_moe_decode disabled: %s", reason)
            return False
        mode = "tp"

    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        return False
    if not _prepare_fp8_routed_experts(experts):
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
        # torch_tpu re-materializes these device-built [1, N] scales with an
        # eager `execute_as_strided` program before every compiled call (a
        # device clone does not avoid it); a buffer moved from the host does.
        s1s_buf = s1s_buf.cpu().to(s1s_buf.device)
        s2s_buf = s2s_buf.cpu().to(s2s_buf.device)
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

    max_tokens = get_fused_moe_decode_max_tokens()
    if mode == "tp":
        from sglang_tpu.moe.pallas.moe_tp_decode.op import prepare_moe_tp_decode

        if not has_shared:
            return False
        admitted = prepare_moe_tp_decode(
            moe_layer, experts, routed_scaling_factor, swiglu_limit
        )
        if admitted == 0:
            return False
        max_tokens = min(max_tokens, admitted)
    else:
        # Eagerly build the custom op outside torch.compile / Dynamo tracing.
        _get_op(tp_size, routed_scaling_factor, swiglu_limit, has_shared=has_shared)

    moe_layer._tpu_fused_moe_decode_mode = mode
    moe_layer._tpu_fused_moe_decode_ready = True
    moe_layer._tpu_fused_moe_decode_has_shared = has_shared
    moe_layer._tpu_fused_moe_decode_world_size = int(tp_size)
    moe_layer._tpu_fused_moe_decode_ep_rank = int(parallel.moe_ep_rank)
    moe_layer._tpu_fused_moe_decode_routed_scale = routed_scaling_factor
    moe_layer._tpu_fused_moe_decode_swiglu_limit = swiglu_limit
    moe_layer._tpu_fused_moe_decode_max_tokens = max_tokens
    return True


def prepare_model_for_fused_moe_decode(model: torch.nn.Module) -> int:
    """Prepare all eligible MoE modules in `model` after weight postprocessing.

    Every eligible module is visited even when the fused decode backend is off,
    so that `_tpu_fused_moe_decode_checked` is set before torch.compile runs.
    Otherwise `should_use_fused_moe_decode` sets it lazily on the first call,
    which changes a guarded module attribute and forces a recompile; after
    warmup this fails under the `fail_on_recompile` stance.
    """
    prepared = []
    for module in model.modules():
        if getattr(module, "allow_tpu_fused_moe_decode", False):
            if prepare_moe_layer_for_fused_decode(module):
                prepared.append(module)

    if prepared:
        modes = sorted({m._tpu_fused_moe_decode_mode for m in prepared})
        logger.info(
            "Prepared TPU fused_moe_decode kernel on %d MoE layers (mode=%s, max_tokens=%d).",
            len(prepared),
            ",".join(modes),
            min(m._tpu_fused_moe_decode_max_tokens for m in prepared),
        )
    return len(prepared)


def should_use_fused_moe_decode(
    moe_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    forward_batch,
) -> bool:
    """Return True when `moe_layer` should execute `fused_moe_decode` for this batch.

    Callers must use the same answer both to dispatch the MoE forward and as the
    FFN boundary's `summed_by_compute`; see `DeepseekV2MoE.use_tpu_fused_moe_decode`.
    """
    if not getattr(moe_layer, "_tpu_fused_moe_decode_ready", False):
        if not getattr(moe_layer, "_tpu_fused_moe_decode_checked", False):
            prepare_moe_layer_for_fused_decode(moe_layer)
        if not getattr(moe_layer, "_tpu_fused_moe_decode_ready", False):
            return False

    # Only plain decode steps: extend / mixed chunked-prefill / speculative
    # verify batches can also be <= max_tokens but have not been benchmarked
    # with this kernel, so they keep using gmm_v2.
    if forward_batch is None or not forward_batch.forward_mode.is_decode():
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

    if moe_layer._tpu_fused_moe_decode_mode == "tp":
        from sglang_tpu.moe.pallas.moe_tp_decode.op import apply_moe_tp_decode

        return apply_moe_tp_decode(moe_layer, hidden_states, topk_ids, topk_weights)

    experts = moe_layer.experts
    world_size = int(moe_layer._tpu_fused_moe_decode_world_size)
    routed_scale = float(moe_layer._tpu_fused_moe_decode_routed_scale)
    swiglu_limit = moe_layer._tpu_fused_moe_decode_swiglu_limit
    has_shared = bool(moe_layer._tpu_fused_moe_decode_has_shared)

    ep_rank = int(moe_layer._tpu_fused_moe_decode_ep_rank)
    rank = _resolve_rank_tensor(experts, ep_rank, world_size)

    op = _get_op(world_size, routed_scale, swiglu_limit, has_shared=has_shared)
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
