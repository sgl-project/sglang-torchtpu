# SPDX-License-Identifier: Apache-2.0
"""PyTorch custom-op wrapper for the fused Muse Spark decode QKV epilogue (D4).

Exports two TorchTPU JAX/Pallas custom ops (preserving ``torch.compile`` FakeTensor
tracing and CPU-only StableHLO export):
  - ``sglang_tpu_pallas::muse_decode_rope_step_tables``: builds ``fp32[B_pad32, 2, 128]``
    signed interleaved RoPE tables once per decode step across all 46 RoPE layers.
  - ``sglang_tpu_pallas::muse_decode_qkv_epilogue_{rope,nope}``: fuses ``qkv.split`` +
    ``q/k fp32 RMSNorm`` + interleaved ``fp32`` RoPE + ``KV_PACK_HEADS=1`` layout +
    per-tensor ``fp8_e4m3fn`` KV quantization + ``merge_kv`` in one pure-VMEM kernel.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here;
# ``torch_tpu._internal.pallas.custom_jax_kernel`` inspects real ``jax.Array`` types.

from threading import Lock
from typing import Optional

import torch

from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    has_forward_context,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_LOCK = Lock()
_BUILT_OPS: dict[str, object] = {}
_ROPE_STEP_OP = None
_EPI_ROPE_OP = None
_EPI_NOPE_OP = None


def _align32(n: int) -> int:
    return ((max(n, 1) + 31) // 32) * 32


def _custom_op_device_types() -> Optional[list[str]]:
    try:
        torch._C._dispatch_key_for_device("tpu")
        return ["tpu"]
    except RuntimeError:
        return None


def get_decode_rope_step_op():
    global _ROPE_STEP_OP
    if _ROPE_STEP_OP is not None:
        return _ROPE_STEP_OP
    with _LOCK:
        if _ROPE_STEP_OP is not None:
            return _ROPE_STEP_OP
        op_name = "sglang_tpu_pallas::muse_decode_rope_step_tables"
        cached = _BUILT_OPS.get(op_name)
        if cached is not None:
            _ROPE_STEP_OP = cached
            return _ROPE_STEP_OP
        import jax
        from torch_tpu._internal import pallas as _torch_pallas

        from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
            build_decode_rope_step_tables,
        )

        def _jax_fn(positions: jax.Array, cos_sin_pairs: jax.Array) -> jax.Array:
            return build_decode_rope_step_tables(positions, cos_sin_pairs)

        jax_kernel = _torch_pallas.custom_jax_kernel(
            _jax_fn,
            name=f"jax_{op_name}",
            static_argnums=(),
            warn_deprecated=False,
        )

        @torch.library.custom_op(
            op_name,
            mutates_args=(),
            device_types=_custom_op_device_types(),
        )
        def rope_step_op(
            positions: torch.Tensor, cos_sin_pairs: torch.Tensor
        ) -> torch.Tensor:
            with tpu_info_for_cpu_export():
                return jax_kernel(positions, cos_sin_pairs)

        def _fake(positions: torch.Tensor, cos_sin_pairs: torch.Tensor) -> torch.Tensor:
            b_pad = _align32(positions.shape[0])
            return cos_sin_pairs.new_empty((b_pad, 2, 128), dtype=torch.float32)

        rope_step_op.register_fake(_fake)
        _BUILT_OPS[op_name] = rope_step_op
        _ROPE_STEP_OP = rope_step_op
        return _ROPE_STEP_OP


def get_decode_qkv_epilogue_op(has_rope: bool):
    global _EPI_ROPE_OP, _EPI_NOPE_OP
    cur = _EPI_ROPE_OP if has_rope else _EPI_NOPE_OP
    if cur is not None:
        return cur
    with _LOCK:
        cur = _EPI_ROPE_OP if has_rope else _EPI_NOPE_OP
        if cur is not None:
            return cur
        op_name = (
            "sglang_tpu_pallas::muse_decode_qkv_epilogue_rope"
            if has_rope
            else "sglang_tpu_pallas::muse_decode_qkv_epilogue_nope"
        )
        cached = _BUILT_OPS.get(op_name)
        if cached is not None:
            if has_rope:
                _EPI_ROPE_OP = cached
            else:
                _EPI_NOPE_OP = cached
            return cached
        import jax
        from torch_tpu._internal import pallas as _torch_pallas

        from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
            run_decode_qkv_epilogue,
        )

        if has_rope:

            def _jax_rope(
                qkv: jax.Array,
                cos_sin_step: jax.Array,
                norm_eps: float = 1e-6,
                query_scale: float = 1.0,
                k_scale: float = 1.0,
                v_scale: float = 1.0,
            ) -> tuple[jax.Array, jax.Array]:
                q_3d, kv_m = run_decode_qkv_epilogue(
                    qkv,
                    cos_sin_step,
                    norm_eps=norm_eps,
                    query_scale=query_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                return q_3d.reshape(1, qkv.shape[0], 8, 2, 128), kv_m

            jax_k = _torch_pallas.custom_jax_kernel(
                _jax_rope,
                name=f"jax_{op_name}",
                static_argnums=(2, 3, 4, 5),
                warn_deprecated=False,
            )

            @torch.library.custom_op(
                op_name,
                mutates_args=(),
                device_types=_custom_op_device_types(),
            )
            def epi_rope_op(
                qkv: torch.Tensor,
                cos_sin_step: torch.Tensor,
                norm_eps: float = 1e-6,
                query_scale: float = 1.0,
                k_scale: float = 1.0,
                v_scale: float = 1.0,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                with tpu_info_for_cpu_export():
                    return jax_k(
                        qkv, cos_sin_step, norm_eps, query_scale, k_scale, v_scale
                    )

            def _fake_rope(
                qkv: torch.Tensor,
                cos_sin_step: torch.Tensor,
                norm_eps: float = 1e-6,
                query_scale: float = 1.0,
                k_scale: float = 1.0,
                v_scale: float = 1.0,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                b_tok = qkv.shape[0]
                return (
                    qkv.new_empty((1, b_tok, 8, 2, 128), dtype=torch.bfloat16),
                    qkv.new_empty((b_tok, 1, 4, 128), dtype=torch.float8_e4m3fn),
                )

            epi_rope_op.register_fake(_fake_rope)
            _BUILT_OPS[op_name] = epi_rope_op
            _EPI_ROPE_OP = epi_rope_op
            return _EPI_ROPE_OP

        def _jax_nope(
            qkv: jax.Array,
            norm_eps: float = 1e-6,
            query_scale: float = 1.0,
            k_scale: float = 1.0,
            v_scale: float = 1.0,
        ) -> tuple[jax.Array, jax.Array]:
            q_3d, kv_m = run_decode_qkv_epilogue(
                qkv,
                None,
                norm_eps=norm_eps,
                query_scale=query_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            return q_3d.reshape(1, qkv.shape[0], 8, 2, 128), kv_m

        jax_k_nope = _torch_pallas.custom_jax_kernel(
            _jax_nope,
            name=f"jax_{op_name}",
            static_argnums=(1, 2, 3, 4),
            warn_deprecated=False,
        )

        @torch.library.custom_op(
            op_name,
            mutates_args=(),
            device_types=_custom_op_device_types(),
        )
        def epi_nope_op(
            qkv: torch.Tensor,
            norm_eps: float = 1e-6,
            query_scale: float = 1.0,
            k_scale: float = 1.0,
            v_scale: float = 1.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            with tpu_info_for_cpu_export():
                return jax_k_nope(qkv, norm_eps, query_scale, k_scale, v_scale)

        def _fake_nope(
            qkv: torch.Tensor,
            norm_eps: float = 1e-6,
            query_scale: float = 1.0,
            k_scale: float = 1.0,
            v_scale: float = 1.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            b_tok = qkv.shape[0]
            return (
                qkv.new_empty((1, b_tok, 8, 2, 128), dtype=torch.bfloat16),
                qkv.new_empty((b_tok, 1, 4, 128), dtype=torch.float8_e4m3fn),
            )

        epi_nope_op.register_fake(_fake_nope)
        _BUILT_OPS[op_name] = epi_nope_op
        _EPI_NOPE_OP = epi_nope_op
        return _EPI_NOPE_OP


def warm_decode_qkv_epilogue_ops() -> None:
    """Eagerly register decode RoPE + QKV epilogue custom ops outside ``torch.compile``."""
    get_decode_rope_step_op()
    get_decode_qkv_epilogue_op(True)
    get_decode_qkv_epilogue_op(False)


def _resolve_decode_attn_backend(forward_batch: object):
    """Resolve active decode attention backend via ``forward_batch`` or ``ForwardContext``."""
    backend = getattr(forward_batch, "attn_backend", None)
    if backend is None and has_forward_context():
        backend = get_attn_backend()
    while backend is not None:
        select = getattr(backend, "_select_backend", None)
        if select is None or not hasattr(forward_batch, "forward_mode"):
            break
        nxt = select(forward_batch.forward_mode)
        if nxt is backend:
            break
        backend = nxt
    return backend


def _effective_decode_qkv_max_tokens() -> int:
    b384_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_B384", None)
    lift_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", None)
    if (b384_env is not None and b384_env.get()) or (
        lift_env is not None and lift_env.get()
    ):
        return 384
    return 256


def can_use_muse_decode_qkv_epilogue(
    qkv: torch.Tensor,
    rotary_emb: Optional[object],
    attn_layer: object,
    forward_batch: object,
    *,
    max_tokens: Optional[int] = None,
) -> bool:
    """True when the current forward step can execute fused D4 + prepared RPAd."""
    if forward_batch is None:
        return False
    if max_tokens is None:
        max_tokens = _effective_decode_qkv_max_tokens()
    verify_ok = (
        forward_batch.forward_mode.is_target_verify()
        and tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.get()
    )
    if (
        not (forward_batch.forward_mode.is_decode() or verify_ok)
        or qkv.ndim != 2
        or qkv.shape[1] != 1280
        or qkv.shape[0] > max_tokens
        or qkv.dtype != torch.bfloat16
    ):
        return False
    if rotary_emb is not None and getattr(rotary_emb, "cos_sin_pairs", None) is None:
        return False
    backend = _resolve_decode_attn_backend(forward_batch)
    fn = getattr(backend, "can_forward_decode_prepared", None)
    return bool(fn is not None and fn(attn_layer, forward_batch))


def can_use_muse_decode_rope_step(
    num_tokens: int,
    rotary_emb: Optional[object],
    attn_layer: object,
    forward_batch: object,
) -> bool:
    """True when ``MuseSparkTextModel`` can hoist ``cos_sin_step`` once for all RoPE layers."""
    if (
        forward_batch is None
        or rotary_emb is None
        or getattr(rotary_emb, "cos_sin_pairs", None) is None
    ):
        return False
    max_tokens = _effective_decode_qkv_max_tokens()
    verify_ok = (
        forward_batch.forward_mode.is_target_verify()
        and tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.get()
    )
    if (
        not (forward_batch.forward_mode.is_decode() or verify_ok)
        or num_tokens <= 0
        or num_tokens > max_tokens
    ):
        return False
    backend = _resolve_decode_attn_backend(forward_batch)
    fn = getattr(backend, "can_forward_decode_prepared", None)
    return bool(fn is not None and fn(attn_layer, forward_batch))


def muse_decode_fused_qkv_attn(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    *,
    rotary_emb: Optional[object],
    attn_layer: object,
    forward_batch: object,
    norm_eps: float,
    query_scale: float,
    cos_sin_step: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    return_post_fused: bool = False,
):
    """Turnkey helper replacing ``qkv.split -> q/k norm -> RoPE -> self.attn`` on decode."""
    k_scale = float(getattr(attn_layer, "k_scale_float", None) or 1.0)
    v_scale = float(getattr(attn_layer, "v_scale_float", None) or 1.0)
    if rotary_emb is not None:
        if cos_sin_step is None:
            cos_sin_step = get_decode_rope_step_op()(
                positions.to(torch.int32), rotary_emb.cos_sin_pairs
            )
        q_5d, kv_merged = get_decode_qkv_epilogue_op(True)(
            qkv,
            cos_sin_step,
            float(norm_eps),
            float(query_scale),
            k_scale,
            v_scale,
        )
    else:
        q_5d, kv_merged = get_decode_qkv_epilogue_op(False)(
            qkv,
            float(norm_eps),
            float(query_scale),
            k_scale,
            v_scale,
        )
    backend = _resolve_decode_attn_backend(forward_batch)
    if backend is None:
        raise RuntimeError(
            "muse_decode_fused_qkv_attn requires an active attention backend "
            "on forward_batch or ForwardContext"
        )
    if gate is not None:
        fn_gated = getattr(backend, "forward_decode_prepared_gated", None)
        if fn_gated is not None:
            out_g = fn_gated(
                q_5d, kv_merged, gate, attn_layer, forward_batch
            )
            if out_g is not None:
                return (out_g, True) if return_post_fused else out_g
    out = backend.forward_decode_prepared(
        q_5d, kv_merged, attn_layer, forward_batch
    )
    return (out, False) if return_post_fused else out
