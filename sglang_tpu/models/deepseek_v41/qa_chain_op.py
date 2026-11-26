# SPDX-License-Identifier: Apache-2.0
"""G-B-lite: the MoE -> next attention mHC boundary with the next layer's
attention input chain (wqkv_a W8A8 + q/kv RMSNorm) fused into the same Pallas
call (SGLANG_TPU_DSV41_MHC_QA_EPILOGUE).

The mHC dispatch on ``deepseek_v4_mhc._post_fusion`` calls ``try_boundary_qa``
first for single-tile row counts. It reuses #2411's packed mix-weight buffer
and owner reference on the next norm (ops/mhc.py), runs the boundary kernel
with the qa_chain epilogue (kernels/qa_chain.py), and stashes (q_lora, kv) for
the next attention's ``prepare_mqa_qkv`` under the identity of the normed
input it returns. Eligible target verify buckets also rotate the KV here.

c4 XProf (6 rows/rank, padded to 16): -0.41 ms per verify; GSM8K-1319 0.919.

PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.
"""

import logging
import weakref
from threading import Lock
from typing import Tuple

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

# The kernel takes one token tile; the boundary kernel's largest is 128 rows.
MAX_ROWS = 128
_OP = None
_OP_LOCK = Lock()
_ENGAGEMENT_LOGGED: set = set()
# id(normed attention input) -> (weakref to it, q_lora, kv): the boundary's
# chain results, taken by that attention's prepare_mqa_qkv in the same trace.
_PRECOMPUTED: dict = {}


def padded_rows(rows: int) -> int:
    return -(-rows // 16) * 16


def _log_engagement(rows: int, used: bool, why: str = "") -> None:
    """Once per (rows, used, why): a shape gate falling back silently has hidden
    a kernel arm before."""
    key = (rows, used, why)
    if key not in _ENGAGEMENT_LOGGED:
        _ENGAGEMENT_LOGGED.add(key)
        logger.info(
            "[dsv41] site=mhc_boundary_qa rows=%d used=%s%s",
            rows,
            used,
            f" why={why}" if why else "",
        )


def _boundary_qa(
    y: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    packed: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    pre: jax.Array,
    norm_weight: jax.Array,
    qa_weight: jax.Array,
    qa_weight_scale: jax.Array,
    q_norm_weight: jax.Array,
    kv_norm_weight: jax.Array,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    q_rank: int,
    qa_eps: float,
    positions: jax.Array | None = None,
    table: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """#2411's ``_shifted_jax`` plus the qa_chain epilogue, rows padded to 16."""
    from sglang_tpu.models.deepseek_v41.kernels.mhc import shifted_post_pre_rms as mhc
    from sglang_tpu.models.deepseek_v41.kernels.qa_chain import qa_chain_epilogue

    t = residual.shape[0]
    rows = padded_rows(t)
    if rows != t:
        # Zero rows stay finite (rsqrt(0 + eps) * 0) and are sliced off.
        y, residual, post, comb, pre = (
            jnp.pad(a, ((0, rows - t),) + ((0, 0),) * (a.ndim - 1))
            for a in (y, residual, post, comb, pre)
        )
    cos_sin = None
    if positions is not None:
        from sglang_tpu.dsv4.rope import cos_sin_rows

        cos_sin = cos_sin_rows(table, positions)
    epilogue = qa_chain_epilogue(
        rows,
        qa_weight,
        qa_weight_scale,
        q_norm_weight,
        kv_norm_weight,
        cos_sin,
        q_rank=q_rank,
        eps=qa_eps,
    )
    outs = mhc.shifted_post_pre_rms(
        y,
        residual,
        post,
        comb,
        packed,
        scale,
        base,
        pre,
        norm_weight=norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        sinkhorn_repeat=sinkhorn_iters,
        norm_eps=norm_eps,
        gate_input_mode=mhc.GateInputMode.COMPILER,
        unpadded_gates=True,
        early_prep=True,
        epilogue=epilogue,
    )
    return tuple(o[:t] for o in outs)


def _boundary_qa_fake(
    y,
    residual,
    post,
    comb,
    packed,
    scale,
    base,
    pre,
    norm_weight,
    qa_weight,
    qa_weight_scale,
    q_norm_weight,
    *args,
):
    tokens, mult = pre.shape
    q_rank = q_norm_weight.shape[0]
    return (
        torch.empty_like(residual),
        post.new_empty((tokens, mult, 1)),
        torch.empty_like(comb),
        torch.empty_like(y),
        torch.empty_like(pre),
        y.new_empty((tokens, q_rank)),
        y.new_empty((tokens, qa_weight.shape[1] - q_rank)),
    )


def _op():
    global _OP
    if _OP is None:
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        with _OP_LOCK:
            if _OP is None:
                op = cpu_export_jax_op(
                    "sglang_tpu_pallas::dsv41_mhc_boundary_qa", _boundary_qa
                )
                op.register_fake(_boundary_qa_fake)
                _OP = op
    return _OP


def register_boundary_qa_op() -> None:
    """Register the op before the model is traced (flag on only)."""
    if tpu_envs.SGLANG_TPU_DSV41_MHC_QA_EPILOGUE.get():
        _op()


def _next_attention(next_boundary):
    """(next HcSubLayer, its decoder's attention) from #2411's owner reference
    on the next norm, or None. Resolved per call so direct compile sees the
    live parameters."""
    from sglang_tpu.models.deepseek_v41.ops import mhc as shifted

    owner = getattr(next_boundary.norm, shifted._OWNER, None)
    if owner is None or owner[1] != "attn_hc":
        return None
    layer = owner[0]()
    if layer is None:
        return None
    attn = layer._modules.get("self_attn")
    if attn is None or "wqkv_a" not in attn._modules or not attn.fuse_wqa_wkv:
        return None
    return getattr(layer, owner[1]), attn


def _verify_rotary_inputs(attn, y):
    # The target model binds its actual forward argument for this trace. Metadata
    # may instead hold a separately padded/cast tensor or the compact extend tail.
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        has_forward_context,
    )

    if y.shape[0] not in (48, 96) or not has_forward_context():
        return None
    positions = getattr(get_attn_backend(), "verify_positions", None)
    table = getattr(attn, "freqs_cis", None)
    if (
        positions is None
        or positions.shape != (y.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or table is None
        or table.ndim != 2
        or table.shape[0] == 0
        or table.shape[1] != 128
        or table.dtype != torch.float32
        or positions.device != y.device
        or table.device != y.device
    ):
        return None
    return positions, table


def try_boundary_qa(hc, y, residual, coefficients, next_boundary):
    """The boundary + next attention chain as one call, or None when it does
    not apply (flag off, no attention after the seam, multi-tile rows, ...)."""
    if not tpu_envs.SGLANG_TPU_DSV41_MHC_QA_EPILOGUE.get():
        return None
    rows = int(y.shape[0])
    if next_boundary is None or not next_boundary.norm_fusable:
        return None
    resolved = _next_attention(next_boundary)
    if resolved is None:
        _log_engagement(rows, False, "layer")
        return None
    next_hc, attn = resolved
    from sglang_tpu.models.deepseek_v41.kernels.mhc.shifted_post_pre_rms import (
        _token_tile,
    )
    from sglang_tpu.models.deepseek_v41.kernels.qa_chain import supports
    from sglang_tpu.models.deepseek_v41.ops import mhc as shifted

    packed = getattr(next_hc.norm, shifted._PACKED, None)
    weight = attn.wqkv_a.weight
    padded = padded_rows(rows)
    if (
        rows == 0
        or padded > MAX_ROWS
        or _token_tile(padded) != padded
        or packed is None
        or y.device.type != "tpu"
        or y.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or tuple(residual.shape[1:]) != (hc.cfg.mult, hc.cfg.hidden)
        or any(gate.dtype != torch.float32 for gate in coefficients)
        or next_hc.scale.dtype != torch.float32
        or next_hc.base.dtype != torch.float32
        or weight.dtype != torch.float8_e4m3fn
        or attn.wqkv_a.weight_scale.numel() != weight.shape[1]
        or not supports(
            padded,
            weight.shape[0],
            weight.shape[1],
            attn.q_lora_rank,
            attn.rope_head_dim,
        )
    ):
        _log_engagement(rows, False, "shape")
        return None
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if is_batch_invariant_mode_enabled():
        _log_engagement(rows, False, "batch_invariant")
        return None

    from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcState

    pre, post, comb = coefficients
    rotary = _verify_rotary_inputs(attn, y)
    rotary_args = () if rotary is None else (rotary[0].to(torch.int32), rotary[1])
    updated, next_post, next_comb, normalized, next_pre, q_lora, kv = _op()(
        y,
        residual.flatten(1),
        post,
        comb,
        packed,
        next_hc.scale,
        next_hc.base,
        pre,
        next_boundary.norm.weight,
        attn.wqkv_a.weight,
        attn.wqkv_a.weight_scale,
        attn.q_norm.weight,
        attn.kv_norm.weight,
        next_hc.cfg.rms_eps,
        next_hc.cfg.eps,
        next_hc.cfg.sinkhorn_iters,
        next_boundary.norm.variance_epsilon,
        int(attn.q_lora_rank),
        float(attn.eps),
        *rotary_args,
    )
    _log_engagement(rows, True, "with_kv_rope" if rotary else "")
    stash(normalized, q_lora, kv, rotary)
    return HcState(
        updated.view_as(residual),
        pre,
        HcNormed(normalized),
        (next_pre, next_post.squeeze(-1), next_comb),
    )


def stash(
    normalized: torch.Tensor, q_lora: torch.Tensor, kv: torch.Tensor, rotary=None
) -> None:
    # An entry nobody took (the attention sliced its input) dies with its input.
    for key in [k for k, v in _PRECOMPUTED.items() if v[0]() is None]:
        del _PRECOMPUTED[key]
    _PRECOMPUTED[id(normalized)] = (weakref.ref(normalized), q_lora, kv, rotary)


def take_precomputed(x: torch.Tensor, positions=None, table=None):
    """Take Q and KV only for the original activation and rotary inputs."""
    entry = _PRECOMPUTED.pop(id(x), None)
    if entry is None or entry[0]() is not x:
        return None
    _, q, kv, rotary = entry
    if rotary is not None and not (rotary[0] is positions and rotary[1] is table):
        return None
    return q, kv, rotary is not None
