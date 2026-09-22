# SPDX-License-Identifier: Apache-2.0
"""Experimental lossless GPT-OSS MXFP4 adapter for the fused EP kernel.

Weight conversion is a one-time operation; the existing fused kernel unpacks
original block-32 weights into BF16 in VMEM for both decode and prefill.
"""

import torch

_OP = None
_OWNER_OP = None

_PARAMETERS = (
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_bias",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_bias",
)


def configure_gptoss_native_mxfp4(layer):
    """Validate the model contract and register the packed operation before tracing."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
    from sglang.srt.runtime_context import get_parallel, get_spec

    parallel = get_parallel()
    if (
        get_spec().speculative_algorithm is not None
        or parallel.attn_tp_size not in (4, 8)
        or parallel.dwdp_size != 1
        or envs.SGLANG_TPU_GPTOSS_OWNER_BASELINE_MOE.get()
        or layer.reduce_results
        or layer.handles_moe_reduction
        or layer.moe_runner_config.no_combine
        or not isinstance(layer.dispatcher, StandardDispatcher)
        or type(layer.quant_method).__name__ != "Mxfp4MoEMethod"
        or layer.moe_runner_config.activation != "swigluoai"
    ):
        raise NotImplementedError(
            "Native GPT-OSS MXFP4 requires original MXFP4, unfused reduction, "
            "attention TP4/8, standard EP dispatch and non-speculative execution."
        )
    layer._tpu_native_mxfp4_configured = True
    build_packed_moe_op()
    build_packed_owner_moe_op()


def prepare_gptoss_native_mxfp4(layer):
    """Replace one layer's checkpoint parameters with native packed parameters."""
    from sglang.srt.hardware_backend.tpu.fp4_moe import _materialize_packed_fp4
    from sglang.srt.layers.utils.common import copy_or_rebind_param

    if not getattr(layer, "_tpu_native_mxfp4_configured", False):
        raise ValueError("Native MXFP4 must be configured by the GPT-OSS model.")
    if getattr(layer, "_tpu_native_mxfp4_loaded", False):
        raise ValueError("Native MXFP4 parameters have already been converted.")
    device = layer.w13_weight.device
    checkpoint = tuple(getattr(layer, name).detach().cpu() for name in _PARAMETERS)
    if checkpoint[0].shape[1:] != (5760, 1440):
        raise ValueError("Native GPT-OSS MXFP4 requires original H/I2880 weights.")
    original_shapes = {
        name: list(value.shape) for name, value in zip(_PARAMETERS, checkpoint)
    }
    scale_ranges = {
        name: [int(value.min()), int(value.max())]
        for name, value in zip(_PARAMETERS, checkpoint)
        if name.endswith("_scale")
    }
    prepared = prepare_packed_weights(*checkpoint, alignment=256)
    for name, value in zip(_PARAMETERS, prepared):
        value = value.to(device)
        if name in ("w13_weight", "w2_weight"):
            value = _materialize_packed_fp4(value)
        copy_or_rebind_param(layer, name, value)
    layer._tpu_native_mxfp4_loaded = True
    layer._tpu_native_mxfp4_receipt = dict(
        alignment=256,
        logical_hidden=2880,
        logical_intermediate=2880,
        padded_hidden=3072,
        padded_intermediate=3072,
        block_size=32,
        original_shapes=original_shapes,
        scale_ranges=scale_ranges,
        parameters={
            name: dict(
                shape=list(getattr(layer, name).shape),
                dtype=str(getattr(layer, name).dtype),
                bytes=getattr(layer, name).numel()
                * getattr(layer, name).element_size(),
            )
            for name in _PARAMETERS
        },
    )


def forward_owner_native_mxfp4_mlp(
    owner_hidden: torch.Tensor, mlp: torch.nn.Module
) -> torch.Tensor:
    """Compute router+topk on 64 owner rows, pre-pad to [64, 3072] with packed routing, and all-gather once."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp = parallel.tp_group
    layer = mlp.experts
    owner_rows, hidden_dim = owner_hidden.shape
    if (
        not getattr(layer, "_tpu_native_mxfp4_loaded", False)
        or owner_hidden.dtype != torch.bfloat16
        or hidden_dim != 2880
        or layer.w13_weight.dtype != torch.float4_e2m1fn_x2
        or layer.w2_weight.dtype != torch.float4_e2m1fn_x2
    ):
        raise ValueError(
            "Owner native GPT-OSS MoE requires loaded packed weights and BF16 H2880."
        )

    with torch.profiler.record_function("tp8dp2_owner_router_topk_64"):
        router_logits, _ = mlp.router(owner_hidden)
        topk_weights, topk_ids, _ = mlp.topk(owner_hidden, router_logits)
        valid = (topk_ids >= 0) & (topk_ids < 128)
        clean_ids = torch.where(valid, topk_ids, 0)
        owner_weights_bf16 = topk_weights.to(torch.bfloat16) * valid.to(torch.bfloat16)
        owner_ids_bf16 = clean_ids.to(torch.bfloat16)

    with torch.profiler.record_function("tp8dp2_owner_pack_all_gather"):
        owner_padded_2944 = torch.nn.functional.pad(owner_hidden, (0, 64)).view(
            owner_rows, 23, 128
        )
        tile_23 = torch.nn.functional.pad(owner_ids_bf16, (0, 124)).view(
            owner_rows, 1, 128
        )
        packed_owner_3d = torch.cat((owner_padded_2944, tile_23), dim=1)
        packed_global_3d = tp.all_gather(packed_owner_3d, dim=0)

    return build_packed_owner_moe_op()(
        packed_global_3d,
        owner_weights_bf16,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w13_weight_bias,
        layer.w2_weight,
        layer.w2_weight_scale,
        layer.w2_weight_bias,
        layer._tpu_fused_rs_group_offset,
        layer._tpu_fused_rs_tp_rank_to_device_id,
    )


def gptoss_native_mxfp4_moe(x, layer, topk_weights, topk_ids, *, output_layout):
    """Return combined expert rows in full, token-owner, or attention-island order."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp = parallel.tp_group
    if (
        not getattr(layer, "_tpu_native_mxfp4_loaded", False)
        or x.dtype != torch.bfloat16
        or x.shape[1] != 2880
        or layer.w13_weight.dtype != torch.float4_e2m1fn_x2
        or layer.w2_weight.dtype != torch.float4_e2m1fn_x2
        or topk_ids.shape != (x.shape[0], 4)
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError(
            "Native GPT-OSS MoE requires loaded packed weights and BF16 H2880/top4."
        )
    if output_layout not in ("owner", "island", "full"):
        raise ValueError("Unknown native GPT-OSS MoE output layout.")
    if output_layout != "full" and (not x.shape[0] or x.shape[0] % tp.world_size):
        raise ValueError("Token owners require a nonempty EP-divisible global batch.")
    ids = topk_ids.to(torch.int32)
    valid = (ids >= 0) & (ids < 128)
    ids = torch.where(valid, ids, 0)
    weights = topk_weights.to(x.dtype) * valid.to(x.dtype)

    def run(input_rows, route_weights, route_ids):
        return build_packed_moe_op()(
            input_rows,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w13_weight_bias,
            layer.w2_weight,
            layer.w2_weight_scale,
            layer.w2_weight_bias,
            route_weights,
            route_ids,
            layer._tpu_fused_rs_group_offset,
            layer._tpu_fused_rs_tp_rank_to_device_id,
        )

    if output_layout != "full":
        shard = run(x, weights, ids)
        return (
            shard
            if output_layout == "owner"
            else parallel.attn_tp_group.all_gather(shard, dim=0)
        )
    outputs = []
    for start in range(0, x.shape[0], 8192):
        rows = min(8192, x.shape[0] - start)
        pad = (-rows) % tp.world_size
        chunk = torch.nn.functional.pad(x[start : start + rows], (0, 0, 0, pad))
        route_weights = torch.nn.functional.pad(
            weights[start : start + rows], (0, 0, 0, pad)
        )
        route_ids = torch.nn.functional.pad(ids[start : start + rows], (0, 0, 0, pad))
        shard = run(chunk, route_weights, route_ids)
        outputs.append(tp.all_gather(shard, dim=0)[:rows])
    return torch.cat(outputs, dim=0) if outputs else x


def prepare_packed_weights(w1, s1, b1, w2, s2, b2, *, alignment=256):
    """Return N-packed bytes, FP32 block scales and biases from HF CPU tensors."""
    from sglang.srt.hardware_backend.tpu.fp4_moe import _transpose_packed_fp4
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        _deinterleave_axis1,
        _e8m0_to_fp32,
        _pad_gate_up_halves_axis1,
        _pad_last_axis,
    )

    if alignment not in (128, 256):
        raise ValueError("Prototype alignment must be 128 or 256.")
    if any(t.device.type != "cpu" for t in (w1, s1, b1, w2, s2, b2)):
        raise ValueError("Checkpoint conversion must happen on CPU before tracing.")
    if any(t.dtype != torch.uint8 for t in (w1, s1, w2, s2)):
        raise TypeError("Checkpoint codes and E8M0 scales must be uint8.")
    experts, two_i, half_h = w1.shape
    hidden, intermediate = 2 * half_h, two_i // 2
    if two_i % 2 or hidden % 32 or intermediate % 32:
        raise ValueError(
            "Original weight dimensions must contain complete MXFP4 blocks."
        )
    if (
        w2.shape != (experts, hidden, intermediate // 2)
        or s1.shape != (experts, two_i, hidden // 32)
        or s2.shape != (experts, hidden, intermediate // 32)
        or b1.shape != (experts, two_i)
        or b2.shape != (experts, hidden)
    ):
        raise ValueError("Checkpoint weights, scales and biases disagree.")
    if any(bool(((s < 1) | (s > 253)).any()) for s in (s1, s2)):
        raise ValueError("Prototype requires finite normal BF16 block scales.")
    hp = ((hidden + alignment - 1) // alignment) * alignment
    ip = ((intermediate + alignment - 1) // alignment) * alignment
    gate = _pad_gate_up_halves_axis1(
        _deinterleave_axis1(w1), intermediate, ip - intermediate
    )
    gate = _pad_last_axis(gate, (hp - hidden) // 2)
    down = _pad_last_axis(w2, (ip - intermediate) // 2)
    down = torch.nn.functional.pad(down, (0, 0, 0, hp - hidden))
    scale1 = _pad_gate_up_halves_axis1(
        _deinterleave_axis1(_e8m0_to_fp32(s1)), intermediate, ip - intermediate
    )
    scale1 = _pad_last_axis(scale1, (hp - hidden) // 32)
    scale2 = _pad_last_axis(_e8m0_to_fp32(s2), (ip - intermediate) // 32)
    scale2 = torch.nn.functional.pad(scale2, (0, 0, 0, hp - hidden))
    bias1 = _pad_gate_up_halves_axis1(
        _deinterleave_axis1(b1.unsqueeze(-1)), intermediate, ip - intermediate
    ).squeeze(-1)
    bias2 = _pad_last_axis(b2, hp - hidden)
    return (
        _transpose_packed_fp4(gate),
        scale1.permute(0, 2, 1).unsqueeze(2).contiguous(),
        bias1.float().unsqueeze(1).contiguous(),
        _transpose_packed_fp4(down),
        scale2.permute(0, 2, 1).unsqueeze(2).contiguous(),
        bias2.float().unsqueeze(1).contiguous(),
    )


def build_packed_moe_op():
    """Register before Dynamo captures the experimental fused boundary."""
    global _OP
    if _OP is not None:
        return _OP
    import jax
    from torch_tpu._internal.pallas import jax_op

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import fused_moe_func_rs

    mesh = _get_mesh()

    def fused(
        x: jax.Array,
        w1: jax.Array,
        s1: jax.Array,
        b1: jax.Array,
        w2: jax.Array,
        s2: jax.Array,
        b2: jax.Array,
        weights: jax.Array,
        ids: jax.Array,
        group: jax.Array,
        destinations: jax.Array,
    ) -> jax.Array:
        if x.shape[0] % mesh.shape["model"]:
            raise ValueError("Prototype batch must divide evenly across EP ranks.")
        if w1.dtype != jax.numpy.float4_e2m1fn or w2.dtype != w1.dtype:
            raise TypeError("Packed custom op requires logical native FP4 weights.")
        if w1.shape[1] // s1.shape[1] != 32 or w2.shape[1] // s2.shape[1] != 32:
            raise ValueError(
                "Prototype preserves the checkpoint block-32 scale contract."
            )
        return fused_moe_func_rs(
            hidden_states=x,
            w1=w1,
            w2=w2,
            w1_scale=s1,
            w2_scale=s2,
            w1_bias=b1,
            w2_bias=b2,
            gating_output=None,
            topk=4,
            renormalize=False,
            mesh=mesh,
            activation="swigluoai",
            scoring_fn="softmax",
            topk_weights=weights,
            topk_indices=ids,
            group_offset_input=group,
            fp8_direct_write=False,
            tp_rank_to_device_id=destinations,
        )

    op = jax_op("sglang_tpu_pallas::gptoss_packed_mxfp4", fused)

    @op.register_fake
    def fake(x, w1, s1, b1, w2, s2, b2, weights, ids, group, destinations):
        return x.new_empty((x.shape[0] // destinations.shape[0], x.shape[1]))

    _OP = op
    return op


def build_packed_owner_moe_op():
    """Register owner-packed [1024, 3072] custom op before Dynamo capture."""
    global _OWNER_OP
    if _OWNER_OP is not None:
        return _OWNER_OP
    import jax
    from torch_tpu._internal.pallas import jax_op

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import _get_mesh
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import fused_moe_func_rs

    mesh = _get_mesh()

    def fused_owner(
        packed_x: jax.Array,
        owner_weights: jax.Array,
        w1: jax.Array,
        s1: jax.Array,
        b1: jax.Array,
        w2: jax.Array,
        s2: jax.Array,
        b2: jax.Array,
        group: jax.Array,
        destinations: jax.Array,
    ) -> jax.Array:
        if (
            packed_x.ndim != 3
            or packed_x.shape[0] % mesh.shape["model"]
            or packed_x.shape[1:] != (24, 128)
        ):
            raise ValueError(
                "Packed owner MoE requires [M, 24, 128] EP-divisible input."
            )
        with jax.named_scope("tp8dp2_unpack_routing"):
            ids_bf16 = packed_x[:, 23, :4]
            ids = ids_bf16.astype(jax.numpy.int32)
        return fused_moe_func_rs(
            hidden_states=packed_x,
            w1=w1,
            w2=w2,
            w1_scale=s1,
            w2_scale=s2,
            w1_bias=b1,
            w2_bias=b2,
            gating_output=None,
            topk=4,
            renormalize=False,
            mesh=mesh,
            activation="swigluoai",
            scoring_fn="softmax",
            topk_weights=ids_bf16,
            topk_indices=ids,
            group_offset_input=group,
            fp8_direct_write=False,
            tp_rank_to_device_id=destinations,
            owner_topk_weights=owner_weights,
            skip_routing_psum=True,
            unpadded_hidden_size=2880,
        )

    op = jax_op("sglang_tpu_pallas::gptoss_packed_owner_mxfp4", fused_owner)

    @op.register_fake
    def fake(packed_x, owner_weights, w1, s1, b1, w2, s2, b2, group, destinations):
        return owner_weights.new_empty((owner_weights.shape[0], 2880))

    _OWNER_OP = op
    return op
