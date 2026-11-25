# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 Q-B fast path for an unsharded channelwise-FP8 projection."""

import functools

import torch

# T=48/64 use a different FP8 GEMM lowering in the unfused path and produce
# different query values. Admit only the independently compared verify/prefill
# buckets; small decode shapes also see little benefit.
_VALIDATED_TOKENS = frozenset({24, 96, 192, 288, 384, 1024, 2048})


@functools.cache
def register_qb_rope_op():
    from sglang_tpu.models.deepseek_v41.kernels.qb_rope import project_q
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    return cpu_export_jax_op("sglang_tpu::dsv41_project_q_rope", project_q)


def supports_qb_rope(layer, x, positions, table) -> bool:
    from sglang.srt.layers.linear import ColumnParallelLinear
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    linear = layer.wq_b
    # A wrapper or a different method can implement adapters, communication or
    # another weight/activation format. Its own forward must run in those cases.
    if (
        type(linear) is not ColumnParallelLinear
        or type(linear.quant_method) is not Fp8LinearMethod
    ):
        return False
    method = linear.quant_method
    if (
        linear.bias is not None
        or linear.gather_output
        or linear.output_size != linear.output_size_per_partition
        or linear._forward_hooks
        or linear._forward_pre_hooks
        or method.use_marlin
        or method.use_mxfp8
        or method.block_fp8_as_mxfp8
        or (method.block_quant and linear.weight_block_size is not None)
    ):
        return False

    scale = linear.weight_scale
    weight = linear.weight
    return (
        weight.device == scale.device == positions.device == table.device == x.device
        and x.ndim == 2
        and x.shape[0] in _VALIDATED_TOKENS
        and x.shape[1] == 1280
        and x.dtype == torch.bfloat16
        and layer.n_local_heads == 64
        and layer.head_dim == 512
        and weight.shape == (1280, 32768)
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and scale.shape in ((32768,), (32768, 1))
        and scale.dtype == torch.float32
        and positions.shape == (x.shape[0],)
        and positions.dtype in (torch.int32, torch.int64)
        and table.ndim == 2
        and table.shape[0] > 0
        and table.shape[1] in (64, 128)
        and table.dtype == torch.float32
    )
