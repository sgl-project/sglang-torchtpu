# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax-H3 per-head QK RMSNorm + partial half-split RoPE in one Pallas pass.

The kernel is adapted from the MiniMax-H3 QK RMSNorm + RoPE Pallas kernel in
vllm-torchtpu (Apache-2.0).

Runtime patch, tracked debt:
- Why: on TPU the H3 attention takes the unfused path (the CUDA fused kernel is
  gated on CUDA, and the ROCm aten gate is skipped under ``torch.compile``):
  ``nn.RMSNorm`` on q and k, then neox RoPE over the first 96 of 128 head dims.
  The RoPE slices at 48 and 96 lanes and concatenates, so every block relayouts
  q and k around the rotation.
- How: :func:`install` rebinds ``_apply_qk_norm_rope`` in the H3 DiT module.
  Eligible TPU calls (bf16 ``[tokens, heads, 128]`` with heads a multiple of 8,
  a ``[tokens, 96]`` cos|sin cache) run one kernel that reads q and k once:
  RMSNorm in float32, rounded to bf16 as the CUDA fused kernel does
  (``round_norm_before_rope=True``), then the half-split rotation as an exact
  signed-permutation matmul on the MXU. Other calls keep the original function.
- Upstream PR or issue: none.
- Remove when: the H3 DiT dispatches QK norm + RoPE through a platform op.
"""

import functools
import threading

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

HEAD_DIM = 128
ROTARY_DIM = 96
SUBLANES = 8
# Row x head slots of q plus k per grid step: 32 rows at H3's 56 + 56 heads.
# Fewer heads take more rows, so a step does the same work (8 heads: 224 rows).
# On TPU7x twice this budget still fits the 32 MiB VMEM limit; four times does not.
ROW_HEADS_PER_STEP = 32 * 112


def _block_rows(rows: int, q_heads: int, k_heads: int) -> int:
    """Rows per grid step: a multiple of 8, at most ``rows`` rounded up to 8."""
    budget = ROW_HEADS_PER_STEP // (q_heads + k_heads)
    block = max(SUBLANES, budget // SUBLANES * SUBLANES)
    return min(block, -(-rows // SUBLANES) * SUBLANES)


_OP = None
_OP_LOCK = threading.Lock()
_INSTALLED = False


@functools.cache
def _rotation_matrix(head_dim: int, rot_dim: int):
    """``P`` with ``x @ P = [-x[h:2h], x[:h], 0]`` for ``h = rot_dim // 2``."""
    import numpy as np

    half = rot_dim // 2
    p = np.zeros((head_dim, head_dim), dtype=np.float32)
    for j in range(half):
        p[half + j, j] = -1.0
        p[j, half + j] = 1.0
    return p


def _kernel(
    q_ref, k_ref, qw_ref, kw_ref, cos_ref, sin_ref, rot_ref, qo_ref, ko_ref, *, eps
):
    import jax
    import jax.numpy as jnp

    f32 = jnp.float32
    cos = cos_ref[...][:, None, :]
    sin = sin_ref[...][:, None, :]
    rot = rot_ref[...]

    def norm_rope(x_ref, w_ref, out_ref):
        x = x_ref[...].astype(f32)
        w = w_ref[...][:1, None, :]
        normed = (
            (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w)
            .astype(jnp.bfloat16)
            .astype(f32)
        )
        rotated = jax.lax.dot_general(
            normed,
            rot,
            (((2,), (0,)), ((), ())),
            preferred_element_type=f32,
            precision=jax.lax.Precision.HIGHEST,
        )
        out_ref[...] = (normed * cos + rotated * sin).astype(out_ref.dtype)

    norm_rope(q_ref, qw_ref, qo_ref)
    norm_rope(k_ref, kw_ref, ko_ref)


def _fused_qk_norm_rope_jax(
    q, k, q_weight, k_weight, cos_sin, eps: float, *, interpret: bool = False
):
    """``q, k [T, H, 128]`` bf16, H a multiple of 8; ``cos_sin [T, 96]``: cos | sin."""
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    f32 = jnp.float32
    rows, q_heads, head_dim = q.shape
    k_heads = k.shape[1]
    block_rows = _block_rows(rows, q_heads, k_heads)
    rows_pad = -(-rows // block_rows) * block_rows
    q_p = jnp.pad(q, ((0, rows_pad - rows), (0, 0), (0, 0)))
    k_p = jnp.pad(k, ((0, rows_pad - rows), (0, 0), (0, 0)))

    half = ROTARY_DIM // 2
    table = cos_sin.astype(f32)
    cos_h, sin_h = table[:, :half], table[:, half:ROTARY_DIM]
    pass_dims = head_dim - ROTARY_DIM
    # Pass-through dims rotate by angle 0: cos 1, sin 0.
    cos = jnp.concatenate([cos_h, cos_h, jnp.ones((rows, pass_dims), f32)], axis=-1)
    sin = jnp.concatenate([sin_h, sin_h, jnp.zeros((rows, pass_dims), f32)], axis=-1)
    cos = jnp.pad(cos, ((0, rows_pad - rows), (0, 0)), constant_values=1.0)
    sin = jnp.pad(sin, ((0, rows_pad - rows), (0, 0)))
    q_w = jnp.broadcast_to(q_weight.astype(f32)[None, :], (SUBLANES, head_dim))
    k_w = jnp.broadcast_to(k_weight.astype(f32)[None, :], (SUBLANES, head_dim))
    rot = jnp.asarray(_rotation_matrix(head_dim, ROTARY_DIM))

    q_spec = pl.BlockSpec((block_rows, q_heads, head_dim), lambda i: (i, 0, 0))
    k_spec = pl.BlockSpec((block_rows, k_heads, head_dim), lambda i: (i, 0, 0))
    w_spec = pl.BlockSpec((SUBLANES, head_dim), lambda i: (0, 0))
    row_spec = pl.BlockSpec((block_rows, head_dim), lambda i: (i, 0))
    rot_spec = pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0))
    q_out, k_out = pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        grid=(rows_pad // block_rows,),
        in_specs=[q_spec, k_spec, w_spec, w_spec, row_spec, row_spec, rot_spec],
        out_specs=[q_spec, k_spec],
        out_shape=[
            _jax_struct(q_p.shape, q.dtype),
            _jax_struct(k_p.shape, k.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        interpret=interpret,
        name="sglang_tpu_minimax_h3_qk_norm_rope",
    )(q_p, k_p, q_w, k_w, cos, sin, rot)
    return q_out[:rows], k_out[:rows]


def _jax_struct(shape, dtype):
    import jax

    return jax.ShapeDtypeStruct(shape, dtype)


def _ensure_op():
    global _OP
    if _OP is not None:
        return _OP
    with _OP_LOCK:
        if _OP is None:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            # cpu_export_jax_op reads the annotations to tell arrays from static args.
            def fused(
                q: jax.Array,
                k: jax.Array,
                q_weight: jax.Array,
                k_weight: jax.Array,
                cos_sin: jax.Array,
                eps: float,
            ) -> tuple[jax.Array, jax.Array]:
                return _fused_qk_norm_rope_jax(q, k, q_weight, k_weight, cos_sin, eps)

            def fused_fake(q, k, q_weight, k_weight, cos_sin, eps):
                return q.new_empty(q.shape), k.new_empty(k.shape)

            op = cpu_export_jax_op("sglang_tpu_pallas::minimax_h3_qk_norm_rope", fused)
            op.register_fake(fused_fake)
            _OP = op
    return _OP


def _eligible(q, k, q_norm, k_norm, head_dim, cos_sin_cache) -> bool:
    return (
        q.device.type == "tpu"
        and head_dim == HEAD_DIM
        and q.dim() == 3
        and k.dim() == 3
        and q.shape[0] == k.shape[0] == cos_sin_cache.shape[0]
        and q.shape[-1] == k.shape[-1] == HEAD_DIM
        # Other head counts need a padded copy of q and k, which costs more
        # than the fusion saves (7 heads: slower than the unfused path).
        and q.shape[1] % SUBLANES == 0
        and k.shape[1] % SUBLANES == 0
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[-1] == ROTARY_DIM
        and q.dtype == k.dtype == torch.bfloat16
        and q_norm.weight is not None
        and k_norm.weight is not None
        and q_norm.eps is not None
        and q_norm.eps == k_norm.eps
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit

    original = dit._apply_qk_norm_rope
    # Register outside any compiled region: the op is traced from the DiT blocks.
    # Every TPU diffusion worker runs this, so a failure must not stop the
    # worker: H3 then keeps the unfused path.
    try:
        _ensure_op()
    except Exception as exc:
        logger.warning(
            "MiniMax-H3 fused QK norm + RoPE unavailable (%s); using the unfused path.",
            exc,
        )
        return

    def apply_qk_norm_rope(
        q, k, q_norm, k_norm, head_dim, cos_sin_cache, positions, rotary_emb
    ):
        if not _eligible(q, k, q_norm, k_norm, head_dim, cos_sin_cache):
            return original(
                q, k, q_norm, k_norm, head_dim, cos_sin_cache, positions, rotary_emb
            )
        return _OP(
            q.contiguous(),
            k.contiguous(),
            q_norm.weight,
            k_norm.weight,
            cos_sin_cache.contiguous(),
            float(q_norm.eps),
        )

    dit._apply_qk_norm_rope = apply_qk_norm_rope
    _INSTALLED = True
