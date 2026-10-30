# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas attention backend (``--attention-backend tpu_splash_attn``).

A flash attention kernel with an online exp2 softmax and a v7x block-size
model for non-causal attention over at least 512 keys, and JAX's
``make_splash_mha`` for causal attention, queries shorter than 256 rows and
cross-attention to fewer than 512 aligned keys. Both run as ``cpu_export_jax_op`` torch ops; block sizes,
padding and query scaling are resolved inside the jitted JAX function from the
static shapes.
"""

import functools
import math
import threading

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

LOG2_E = math.log2(math.e)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
# Block sizes and the VMEM model below are tuned and validated on v7x only.
_SUPPORTED_PALLAS_DEVICE_KINDS = frozenset({"TPU7x"})

_SPLASH_AVAILABLE: bool | None = None
_logged_sdpa_fallbacks: set[tuple] = set()

_FLASH_OP = None
_SPLASH_MHA_OP = None
_OPS_LOCK = threading.Lock()
_resolved_mode_logged = False


def _log_resolved_mode_once(use_splash: bool, dropout: float) -> None:
    global _resolved_mode_logged
    if _resolved_mode_logged:
        return
    _resolved_mode_logged = True
    if use_splash:
        logger.info("TPU splash attention: splash=on (per-shape SDPA fallback).")
    else:
        reason = f"dropout={dropout}" if dropout != 0.0 else "kernels unavailable"
        logger.info(
            "TPU splash attention: splash=off (%s); every call uses SDPA.", reason
        )


def _check_splash_available() -> bool:
    global _SPLASH_AVAILABLE
    if _SPLASH_AVAILABLE is not None:
        return _SPLASH_AVAILABLE
    try:
        import jax  # noqa: F401
        from jax.experimental import pallas  # noqa: F401
        from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
            splash_attention_kernel,
        )

        from sglang_tpu.pallas_export import (
            get_pallas_tpu_device_kind,
        )
        from sglang_tpu.pallas_jax_op import (  # noqa: F401
            cpu_export_jax_op,
        )

        device_kind = get_pallas_tpu_device_kind()
        if device_kind not in _SUPPORTED_PALLAS_DEVICE_KINDS:
            raise RuntimeError(f"kernels are tuned for TPU v7x only, not {device_kind}")
        _SPLASH_AVAILABLE = True
    except Exception as exc:
        logger.warning("TPU Pallas splash attention unavailable (%s); using SDPA.", exc)
        _SPLASH_AVAILABLE = False
    return _SPLASH_AVAILABLE


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _round_down(x: int, multiple: int) -> int:
    """Round ``x`` down to a multiple, but never below one ``multiple``.

    For ``x < multiple`` this rounds *up* to ``multiple``.
    """
    return max((x // multiple) * multiple, multiple)


# TPU v7x VMEM per TensorCore. The kernel passes the live
# ``get_tpu_info().vmem_capacity_bytes`` to both the block-size model and
# Mosaic's ``vmem_limit_bytes``; this default serves CPU-side callers.
_V7X_VMEM_CAPACITY_BYTES = 64 * 2**20
# Headroom for buffers the model does not count, chiefly the double-buffered
# bf16 output block (3.1 MiB at block_q=6400).
_VMEM_BUDGET_FRACTION = 0.93


def _score_tile_bytes(
    block_q: int,
    block_kv_compute: int,
    head_dim: int = 128,
    block_kv: int = 2048,
    bytes_per_element: int = 2,
) -> int:
    score_f32 = block_kv_compute * block_q * 4
    qkv_2x = 2 * (block_q + 2 * block_kv) * head_dim * bytes_per_element
    output_f32 = block_q * head_dim * 4
    max_sum_scratch = 2 * 8 * block_q * 4
    return score_f32 + qkv_2x + output_f32 + max_sum_scratch


def _autotune_block_sizes(
    q_seq: int,
    kv_seq: int,
    head_dim: int = 128,
    vmem_capacity_bytes: int = _V7X_VMEM_CAPACITY_BYTES,
) -> tuple[int, int, int]:
    """``(block_q, block_kv, block_kv_compute)`` of the flash kernel on TPU v7x.

    Minimizes grid-cell cost under a VMEM budget, keeping at least two query
    blocks per head. The VMEM model holds on v7x only.
    """
    mxu_dim = 256
    safe_kv_block = 2048
    vmem_budget = int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION)
    # Cap block_q: the VMEM model leaves out the double-buffered bf16 output
    # block (4 * block_q * head_dim bytes). For short KV the model alone would
    # allow block_q up to ~19.7k at block_kv=512 and overflow once that block
    # is added. 9472 is the largest size validated on device (37800x512,
    # 18900x600).
    max_block_q = 9472

    if q_seq < mxu_dim or kv_seq < mxu_dim:
        return 256, 256, 256

    block_kv = _round_down(min(safe_kv_block, kv_seq), mxu_dim)
    block_kv_compute = block_kv

    vmem_max = mxu_dim
    bq = mxu_dim
    while bq <= min(q_seq, max_block_q):
        if _score_tile_bytes(bq, block_kv_compute, head_dim, block_kv) <= vmem_budget:
            vmem_max = bq
            bq += mxu_dim
        else:
            break

    candidates = [
        c for c in range(mxu_dim, vmem_max + 1, mxu_dim) if math.ceil(q_seq / c) >= 2
    ]
    if not candidates:
        candidates = [min(vmem_max, _round_down(q_seq, mxu_dim))]

    cell_overhead = mxu_dim * safe_kv_block

    def _cost(cand_bq: int) -> tuple[int, int]:
        cells = math.ceil(q_seq / cand_bq) * math.ceil(kv_seq / block_kv)
        return (cells * (cand_bq * block_kv + cell_overhead), -cand_bq)

    best_bq = min(candidates, key=_cost)
    return best_bq, block_kv, block_kv_compute


def _use_custom_kernel(q_seq: int, kv_seq: int, is_causal: bool) -> bool:
    """Non-causal attention with at least 512 keys: long-KV self-attention, SP
    query shards and text cross-attention. The kernel masks ragged edge
    blocks, so no query or KV padding is needed."""
    return (not is_causal) and q_seq >= 256 and kv_seq >= 512


def _cross_attn_block_kv(kv_seq: int) -> int:
    """KV block dividing ``kv_seq``, a multiple of 128 below 512.

    Only kv in {128, 256, 384} reaches here: 512 or more keys go to the
    custom kernel.
    """
    return 256 if kv_seq % 256 == 0 else 128


def _splash_mha_block_sizes(
    q_seq: int, kv_seq: int, is_causal: bool
) -> tuple[int, int]:
    """``(block_q, block_kv)`` of ``make_splash_mha`` for a non-custom shape."""
    # Long query against short 128-aligned KV: block_kv divides kv_seq.
    if not is_causal and q_seq >= 2048 and kv_seq % 128 == 0:
        return 2048, _cross_attn_block_kv(kv_seq)
    # Causal attention, and short queries against long KV.
    if q_seq % 1024 == 0 and kv_seq % 1024 == 0:
        return 1024, 1024
    if q_seq >= 512 and kv_seq >= 512:
        return 512, 512
    return 256, 256


def _flash_attention(q, k, v, scale):
    """Flash attention with an online exp2 softmax on ``[B, H, S, D]`` arrays.

    Takes the unscaled query: the softmax scale is applied to the float32
    scores. Inputs are not padded; Mosaic masks the ragged query edge block and
    the last KV block reads only real keys.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    num_q_heads, q_seq, head_dim = q.shape[1:]
    num_kv_heads, kv_seq = k.shape[1], k.shape[2]
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    # One VMEM figure feeds both the block-size model and Mosaic's limit.
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    block_q, block_kv, block_kv_compute = _autotune_block_sizes(
        q_seq, kv_seq, head_dim, vmem_capacity_bytes
    )
    grid_width = pl.cdiv(kv_seq, block_kv)
    qk_scale = scale * LOG2_E
    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)
    logger.info(
        "TPU splash attention: q_seq=%d kv_seq=%d -> flash kernel blocks=%s",
        q_seq,
        kv_seq,
        (block_q, block_kv, block_kv_compute),
    )

    def kernel(q_ref, k_ref, v_ref, o_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref):
        float32 = jnp.float32
        sv_dims = (((0,), (0,)), ((), ()))
        j = pl.program_id(2)

        @pl.when(j == 0)
        def init():
            o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

        def update(slice_k):
            q_tile = q_ref[...]
            k_chunk = k_ref[slice_k, :]
            v_chunk = v_ref[slice_k, :]
            qk = lax.dot_general(
                k_chunk, q_tile, NT_DIM_NUMBERS, preferred_element_type=float32
            )
            m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
            # Max of raw scores (qk_scale > 0): qk * c - m is one FMA.
            m_curr = qk.max(axis=0)[None, :] * qk_scale
            m_next = jnp.maximum(m_prev, m_curr)
            s_curr = jnp.exp2(qk * qk_scale - m_next[0:1])
            l_curr = s_curr.sum(axis=0, keepdims=True)
            alpha = jnp.exp2(m_prev - m_next)
            o_curr = lax.dot_general(
                v_chunk,
                s_curr.astype(jnp.bfloat16),
                sv_dims,
                preferred_element_type=float32,
            )
            o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
            m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

        def compute_body(kv_compute_index, _):
            update(pl.ds(kv_compute_index * block_kv_compute, block_kv_compute))

        @pl.when(j != grid_width - 1)
        def body():
            lax.fori_loop(
                0, block_kv // block_kv_compute, compute_body, None, unroll=True
            )

        # The last KV block reads only real keys.
        @pl.when(j == grid_width - 1)
        def last_body():
            remain = kv_seq - (grid_width - 1) * block_kv
            full, tail = divmod(remain, block_kv_compute)
            lax.fori_loop(0, full, compute_body, None, unroll=True)
            if tail:
                update(pl.ds(full * block_kv_compute, tail))

        @pl.when(j == grid_width - 1)
        def end():
            # l is replicated across the sublanes; tile it to the [D, block_q]
            # scratch and write the [block_q, D] output tile.
            l_inv = jnp.tile(1.0 / l_scratch_ref[...], (head_dim // NUM_SUBLANES, 1))
            o_ref[...] = (o_scratch_ref[...] * l_inv).T.astype(o_ref.dtype)

    def attend(q_3d, k_3d, v_3d):
        kv_spec = pl.BlockSpec(
            (None, block_kv, head_dim),
            lambda h, i, j: (h // q_heads_per_kv_head, j, 0),
        )
        return pl.pallas_call(
            kernel,
            grid=(num_q_heads, pl.cdiv(q_seq, block_q), grid_width),
            in_specs=[
                pl.BlockSpec((None, block_q, head_dim), lambda h, i, j: (h, i, 0)),
                kv_spec,
                kv_spec,
            ],
            out_specs=pl.BlockSpec(
                (None, block_q, head_dim), lambda h, i, j: (h, i, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
                pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
                pltpu.VMEM((head_dim, block_q), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
                # Autotuned score tiles exceed Mosaic's 32 MiB default.
                vmem_limit_bytes=vmem_capacity_bytes,
            ),
            out_shape=jax.ShapeDtypeStruct((num_q_heads, q_seq, head_dim), q_3d.dtype),
            name="diffusion_flash_attention",
        )(q_3d, k_3d, v_3d)

    return jax.vmap(attend)(q, k, v)


def _splash_mha(q, k, v, scale, is_causal):
    """Stock ``make_splash_mha`` on ``[B, H, S, D]`` arrays.

    The query is scaled and padded to a block multiple; zero-padded keys are
    excluded by segment ids. Causal attention only reaches here with
    ``q_seq == kv_seq`` and equal blocks, so real rows never see padded keys.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as sam,
    )

    num_q_heads, q_seq = q.shape[1], q.shape[2]
    num_kv_heads, kv_seq = k.shape[1], k.shape[2]
    block_q, block_kv = _splash_mha_block_sizes(q_seq, kv_seq, is_causal)
    q_padded, kv_padded = _round_up(q_seq, block_q), _round_up(kv_seq, block_kv)
    logger.info(
        "TPU splash attention: q_seq=%d kv_seq=%d causal=%s -> make_splash_mha "
        "blocks=(%d, %d)",
        q_seq,
        kv_seq,
        is_causal,
        block_q,
        block_kv,
    )

    # make_splash_mha takes no scale, so q is pre-scaled and rounded to bf16:
    # up to 2^-9 relative error per element, which the flash kernel avoids by
    # scaling its f32 scores. The extra error grows with the logit spread
    # (test_splash_high_logit_gain bounds it).
    q = (q.astype(jnp.float32) * scale).astype(k.dtype)
    if num_q_heads != num_kv_heads:
        k = jnp.repeat(k, num_q_heads // num_kv_heads, axis=1)
        v = jnp.repeat(v, num_q_heads // num_kv_heads, axis=1)
    q = jnp.pad(q, ((0, 0), (0, 0), (0, q_padded - q_seq), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, 0), (0, kv_padded - kv_seq), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_padded - kv_seq), (0, 0)))

    mask_cls = sam.CausalMask if is_causal else sam.FullMask
    kernel = splash.make_splash_mha(
        sam.MultiHeadMask([mask_cls((q_padded, kv_padded))] * num_q_heads),
        head_shards=1,
        q_seq_shards=1,
        block_sizes=splash.BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=min(block_kv, 512),
        ),
    )
    if kv_padded == kv_seq:
        out = jax.vmap(kernel)(q, k, v)
    else:
        segment_ids = splash.SegmentIds(
            q=jnp.ones((q_padded,), jnp.int32),
            kv=(jnp.arange(kv_padded) < kv_seq).astype(jnp.int32),
        )
        out = jax.vmap(functools.partial(kernel, segment_ids=segment_ids))(q, k, v)
    return out[:, :, :q_seq]


def _register_ops() -> None:
    """Create the kernel ops; must run before torch.compile traces a model.

    A failure marks the kernels unavailable, so callers fall back to SDPA
    instead of retrying a half-registered ``torch.library`` op.
    """
    global _FLASH_OP, _SPLASH_MHA_OP, _SPLASH_AVAILABLE
    if _FLASH_OP is not None or _SPLASH_AVAILABLE is False:
        return
    with _OPS_LOCK:
        if _FLASH_OP is not None or _SPLASH_AVAILABLE is False:
            return
        try:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            def flash(
                q: jax.Array, k: jax.Array, v: jax.Array, scale: float
            ) -> jax.Array:
                return _flash_attention(q, k, v, scale)

            def splash_mha(
                q: jax.Array, k: jax.Array, v: jax.Array, scale: float, is_causal: bool
            ) -> jax.Array:
                return _splash_mha(q, k, v, scale, is_causal)

            def flash_fake(q, k, v, scale):
                return q.new_empty(q.shape)

            def splash_mha_fake(q, k, v, scale, is_causal):
                return q.new_empty(q.shape)

            splash_mha_op = cpu_export_jax_op(
                "sglang_tpu_pallas::diffusion_splash_mha", splash_mha
            )
            splash_mha_op.register_fake(splash_mha_fake)
            flash_op = cpu_export_jax_op(
                "sglang_tpu_pallas::diffusion_flash_attention", flash
            )
            flash_op.register_fake(flash_fake)
        except Exception as exc:
            logger.warning(
                "TPU Pallas splash attention ops failed to register (%s); using SDPA.",
                exc,
            )
            _SPLASH_AVAILABLE = False
            return
        _SPLASH_MHA_OP = splash_mha_op
        _FLASH_OP = flash_op


def _sdpa_fallback_reason(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool
) -> str | None:
    q_seq, kv_seq = query.shape[2], key.shape[2]
    if q_seq == 0 or kv_seq == 0:
        return "empty"
    # Short sequences, and cross-attention to fewer than 512 keys that
    # make_splash_mha cannot block without padding, stay on SDPA.
    if (q_seq <= 512 and kv_seq <= 512) or (
        not is_causal
        and q_seq != kv_seq
        and kv_seq < 512
        and (q_seq < 2048 or kv_seq % 128 != 0)
    ):
        return "short"
    if is_causal and q_seq != kv_seq:
        return "ragged causal"
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        return f"head_dim={query.shape[-1]}/{value.shape[-1]}"
    if query.shape[1] % key.shape[1] != 0:
        return f"heads={query.shape[1]}/{key.shape[1]}"
    # SDPA reports a malformed value tensor with a clear shape error.
    if value.shape[:3] != key.shape[:3]:
        return f"value shape={tuple(value.shape)} key shape={tuple(key.shape)}"
    # The kernels compute in bf16.
    if not query.dtype == key.dtype == value.dtype == torch.bfloat16:
        return f"dtype={query.dtype}/{key.dtype}/{value.dtype}"
    if not _check_splash_available():
        return "pallas unavailable"
    return None


def tpu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """``[B, H, S, D]`` attention on TPU through the Pallas kernels, else SDPA.

    Ragged causal attention (``q_seq != kv_seq``) is rejected: SDPA's
    top-left causal mask differs from ``SDPAImpl``'s bottom-right one, so
    callers pick the semantics (``TpuSplashAttentionImpl`` routes it to
    ``SDPAImpl``).
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    # The kernels take the row max of the raw scores, which is the max of the
    # scaled scores only for a positive scale.
    if scale <= 0:
        raise ValueError(f"tpu_attention needs a positive scale, got {scale}")
    q_seq, kv_seq = query.shape[2], key.shape[2]
    if is_causal and q_seq != kv_seq:
        raise NotImplementedError(
            f"tpu_attention: ragged causal attention (q_seq={q_seq}, "
            f"kv_seq={kv_seq}) is ambiguous; use SDPAImpl's bottom-right mask."
        )

    reason = _sdpa_fallback_reason(query, key, value, is_causal)
    if reason is None:
        _register_ops()
        if _FLASH_OP is None:
            reason = "pallas registration failed"
    if reason is None:
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        if _use_custom_kernel(q_seq, kv_seq, is_causal):
            return _FLASH_OP(q, k, v, scale)
        return _SPLASH_MHA_OP(q, k, v, scale, is_causal)

    log_key = (q_seq, kv_seq, is_causal, reason)
    # Logging breaks a torch.compile graph; eager runs report the fallback.
    if not torch.compiler.is_compiling() and log_key not in _logged_sdpa_fallbacks:
        _logged_sdpa_fallbacks.add(log_key)
        logger.info(
            "TPU splash attention: q_seq=%d kv_seq=%d causal=%s -> sdpa (%s)",
            q_seq,
            kv_seq,
            is_causal,
            reason,
        )
    # torch_tpu's bf16 SDPA is 1.6-2x flash attention's error (4.2e-3 against
    # 2.6e-3 relative at a 512-token causal text-encoder shape); the fallback
    # shapes are short, so float32 costs ~0.3 ms a call and lands at 3e-5.
    return F.scaled_dot_product_attention(
        query.float(),
        key.float(),
        value.float(),
        is_causal=is_causal,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
    ).to(query.dtype)


class TpuSplashAttentionBackend(SDPABackend):
    """TPU Pallas attention, tuned on v7x; uncovered shapes fall back to SDPA."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        # The platform's own backend; TpuOOTPlatform names it tpu_splash_attn.
        return AttentionBackendEnum.OOT

    @staticmethod
    def get_impl_cls() -> type["TpuSplashAttentionImpl"]:
        return TpuSplashAttentionImpl


class TpuSplashAttentionImpl(SDPAImpl):
    """Routes eligible attention to the Pallas kernels, else to SDPA."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        if self.dropout != 0.0:
            logger.info(
                "%s: dropout=%s, using SDPA instead of the TPU splash kernels.",
                prefix or type(self).__name__,
                self.dropout,
            )
        self.use_splash = self.dropout == 0.0 and _check_splash_available()
        if self.use_splash:
            _register_ops()
            self.use_splash = _FLASH_OP is not None
        _log_resolved_mode_once(self.use_splash, self.dropout)
        # forward_varlen keeps packed segments on plain SDPA.
        self._sdpa = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # Inputs are [b, s, h, d]. Ragged causal attention needs SDPAImpl's
        # bottom-right mask, which the kernels do not take.
        if (
            not self.use_splash
            or query.device.type != "tpu"
            or (self.causal and query.shape[1] != key.shape[1])
        ):
            return super().forward(query, key, value, attn_metadata)

        output = tpu_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
            is_causal=self.causal,
        )
        return output.transpose(1, 2)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        # Packed segment lengths vary per request. Each new length compiles
        # under either backend (dynamic=False); SDPA skips the Pallas trace,
        # export and Mosaic compile on top of that, so segments stay on SDPA.
        return self._sdpa.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )
