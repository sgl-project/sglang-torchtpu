# SPDX-License-Identifier: Apache-2.0
"""FP8 grouped ``wo_a`` contraction for DeepSeek-V4.1 on TPU.

On TPU, ``disable_tpu_dsv4_wo_a_fp8_gemm`` keeps ``wo_a`` in BF16 (~67 MB per
layer x 43 layers = 2.88 GB/rank), making ``torch.einsum("tgd,grd->tgr")`` one
of the single largest bottlenecks at low-latency decode (~20.7% of active time
at c8). Requantizing ``wo_a`` to per-channel ``float8_e4m3fn`` (matching
``sglang_tpu/fp8_linear.py`` used by every other dense linear in V4.1) halves
weight HBM footprint (-1.44 GB/rank) and runs an XLA FP8 einsum that is
2.5x-3.4x faster at decode and 1.6x faster at prefill.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import logging
from threading import Lock

import jax
import jax.numpy as jnp
import torch

logger = logging.getLogger(__name__)

_OP_NAME = "sglang_tpu_pallas::dsv41_wo_a_fp8_einsum"
_OP = None
_OP_LOCK = Lock()

SCALE_ATTR = "_tpu_wo_a_fp8_scale"


def _wo_a_fp8_jax(
    o: jax.Array, w_fp8: jax.Array, w_scale: jax.Array
) -> jax.Array:
    """``o [T, G, D]`` BF16, ``w_fp8 [G, R, D]`` FP8, ``w_scale [G, R]`` FP32 -> ``[T, G, R]`` BF16."""
    fp8_dtype = w_fp8.dtype
    fp8_max = float(jnp.finfo(fp8_dtype).max)
    xf = o.astype(jnp.float32)
    xs = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True) / fp8_max, 1e-12)
    xq = (xf / xs).astype(fp8_dtype)
    acc = jnp.einsum("tgd,grd->tgr", xq, w_fp8, preferred_element_type=jnp.float32)
    return (acc * xs * w_scale[None, :, :]).astype(o.dtype)


def _fake_wo_a_fp8(o, w_fp8, w_scale):
    return o.new_empty((o.shape[0], w_fp8.shape[0], w_fp8.shape[1]), dtype=o.dtype)


def _get_op():
    global _OP
    if _OP is not None:
        return _OP
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OP_LOCK:
        if _OP is None:
            op = cpu_export_jax_op(_OP_NAME, _wo_a_fp8_jax)
            op.register_fake(_fake_wo_a_fp8)
            _OP = op
    return _OP


def register_wo_a_fp8_op() -> None:
    """Register the FP8 wo_a op before torch.compile traces."""
    _get_op()


def wo_a_fp8_matmul(
    o: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    """Run FP8 grouped ``wo_a`` contraction: ``[T, G, D] @ [G, R, D] -> [T, G, R]``."""
    return _get_op()(o.contiguous(), w_fp8, w_scale)


def _release(param: torch.nn.Parameter) -> None:
    old = param.data
    param.data = torch.empty(0, dtype=old.dtype, device=old.device)
    storage = old.untyped_storage()
    if storage.data_ptr() != param.data.untyped_storage().data_ptr():
        try:
            storage.resize_(0)
        except RuntimeError:
            pass
    del old


def quantize_wo_a_module(attn) -> bool:
    """Convert ``attn.wo_a.weight`` (BF16 ``[G*R, D]``) to per-channel FP8 ``[G, R, D]``."""
    wo_a = getattr(attn, "wo_a", None)
    if wo_a is None or getattr(wo_a, SCALE_ATTR, None) is not None:
        return False
    weight = getattr(wo_a, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or weight.device.type != "tpu"
    ):
        return False
    groups = int(getattr(attn, "n_local_groups", 0))
    rank = int(getattr(attn, "o_lora_rank", 0))
    if groups <= 0 or rank <= 0 or weight.shape[0] != groups * rank:
        return False
    dim = weight.shape[1]
    if dim % 128 != 0 or rank % 128 != 0:
        return False

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    with torch.no_grad():
        wf = weight.data.view(groups, rank, dim).float()
        scale = torch.clamp(wf.abs().amax(dim=-1) / fp8_max, min=1e-12)  # [G, R] f32
        wq = (wf / scale.unsqueeze(-1)).to(fp8_dtype).contiguous()
        scale = scale.contiguous()
        del wf
        _release(weight)
        wo_a.weight = torch.nn.Parameter(wq, requires_grad=False)
        wo_a.register_parameter(
            SCALE_ATTR, torch.nn.Parameter(scale, requires_grad=False)
        )
    return True


def prepare_model_for_dsv41_wo_a_fp8(model) -> int:
    """Quantize every eligible V4.1 ``wo_a`` module (target + DSpark draft) to FP8."""
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_DSV41_WO_A_FP8.get():
        return 0
    register_wo_a_fp8_op()
    converted = 0
    for module in model.modules():
        if hasattr(module, "wo_a") and hasattr(module, "n_local_groups") and hasattr(module, "o_lora_rank"):
            if quantize_wo_a_module(module):
                converted += 1
    if converted:
        logger.info(
            "SGLANG_TPU_DSV41_WO_A_FP8: quantized %d wo_a modules to per-channel FP8.",
            converted,
        )
    return converted
