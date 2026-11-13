# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas attention backend (``--attention-backend tpu_splash_attn``).

A flash attention kernel with an online exp2 softmax and a v7x block-size
model for non-causal attention over at least 512 keys, and JAX's
``make_splash_mha`` for causal attention, queries shorter than 256 rows and
cross-attention to fewer than 512 aligned keys. Both run as ``cpu_export_jax_op`` torch ops; block sizes,
padding and query scaling are resolved inside the jitted JAX function from the
static shapes.

Against stock ``make_splash_mha`` with its best of 9 tuned block sizes (one
tpu7x core, 10 heads, bf16, non-causal; libtpu 0.0.50): 56.6 vs 97.1 ms at
75600 tokens, 13.2 vs 18.5 ms at 32760, 0.98 vs 1.92 ms at 9680. Worth keeping
across jax upgrades: the LLO scheduler flag (about a third of the gap), query
blocks above splash's 2048-4096 that the VMEM model allows, and the softmax
scale applied to the float32 scores inside an exp2 softmax.
"""

import functools
import math
import threading
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_replica_group,
    get_sp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.envs import envs as tpu_envs

logger = init_logger(__name__)

LOG2_E = math.log2(math.e)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
# Finite stand-in for -inf scores: exp2 of it is 0 without inf - inf.
_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
# The kernels mask raw scores with _MASK_VALUE before scaling: for a scale
# above ~1.43 the product overflows to -inf, which exp2 still maps to 0.
# Block sizes and the VMEM model below are tuned and validated on v7x only.
_SUPPORTED_PALLAS_DEVICE_KINDS = frozenset({"TPU7x"})

_SPLASH_AVAILABLE: bool | None = None
_logged_sdpa_fallbacks: set[tuple] = set()

_FLASH_OP = None
_FLASH_RING_OP = None
_FLASH_LIVE_LENGTH_OP = None
_SPLASH_MHA_OP = None
# Below 4096 rows per rank, pure-ring per-head ICI hops cost more than intra-chip D2D all_to_all + bulk ppermute.
_SHORT_SHARD_2D_RING_MAX = 4096
_RING_OPS: dict[tuple[int, int, int, int, float, bool], Callable | None] = {}
_SHORT_2D_RING_OPS: dict[tuple[int, int, int, float], Callable | None] = {}
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
        logger.warning(
            "TPU Pallas splash attention unavailable (%s); using XLA attention.", exc
        )
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
# Live-length tiles (packed rows with trailing padding): compile-only tpu7x
# schedules at 32768 rows, 7 heads are within 2% of the autotuned 4096x2048
# tiles in a quarter of the instructions.
_LIVE_LENGTH_BLOCK_Q = 2048
_LIVE_LENGTH_BLOCK_KV = 1024
# Headroom for buffers the model does not count, chiefly the double-buffered
# bf16 output block (3.1 MiB at block_q=6400).
_VMEM_BUDGET_FRACTION = 0.93
# Shapes where a block sweep on device beat the model's pick, keyed by
# (q_seq, kv_seq, carry_bytes_per_q_row). Wan2.2-TI2V-5B 1280x704x121 ring
# resume step: 2.47 vs 2.88 ms.
_MEASURED_BLOCK_SIZES = {(13640, 13640, 2176): (6912, 1024, 1024)}


def _score_tile_bytes(
    block_q: int,
    block_kv_compute: int,
    head_dim: int = 128,
    block_kv: int = 2048,
    bytes_per_element: int = 2,
    carry_bytes_per_q_row: int = 0,
) -> int:
    score_f32 = block_kv_compute * block_q * 4
    qkv_2x = 2 * (block_q + 2 * block_kv) * head_dim * bytes_per_element
    output_f32 = block_q * head_dim * 4
    max_sum_scratch = 2 * 8 * block_q * 4
    carry = carry_bytes_per_q_row * block_q
    return score_f32 + qkv_2x + output_f32 + max_sum_scratch + carry


def _autotune_block_sizes(
    q_seq: int,
    kv_seq: int,
    head_dim: int = 128,
    vmem_capacity_bytes: int = _V7X_VMEM_CAPACITY_BYTES,
    carry_bytes_per_q_row: int = 0,
) -> tuple[int, int, int]:
    """``(block_q, block_kv, block_kv_compute)`` of the flash kernel on TPU v7x.

    Minimizes grid-cell cost under a VMEM budget, keeping at least two query
    blocks per head. The VMEM model holds on v7x only.
    ``carry_bytes_per_q_row`` is VMEM per query row beyond the score tile,
    e.g. a ring step's float32 carry blocks.
    """
    measured = _MEASURED_BLOCK_SIZES.get((q_seq, kv_seq, carry_bytes_per_q_row))
    if measured is not None:
        return measured
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
        tile_bytes = _score_tile_bytes(
            bq, block_kv_compute, head_dim, block_kv, 2, carry_bytes_per_q_row
        )
        if tile_bytes <= vmem_budget:
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


def _carry_bytes_per_q_row(
    head_dim: int, *, f32_out: bool = False, resume: bool = False
) -> int:
    # A bf16 output + LSE block fits in _VMEM_BUDGET_FRACTION's headroom; ring
    # steps that write a float32 output carry (f32_out) and/or read a previous
    # float32 output + LSE carry (resume) allocate double-buffered blocks per row.
    return (int(f32_out) + int(resume)) * 2 * (head_dim + NUM_SUBLANES) * 4


def _flash_call(
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    scale: float,
    *,
    num_rows: int,
    heads: int,
    kv_heads: int,
    out_dtype,
    return_lse: bool = False,
    resume: bool = False,
    runtime_kv: bool = False,
    rows_in: bool = False,
    block_sizes: tuple[int, int, int] | None = None,
):
    """``pallas_call`` of the flash kernel over ``[rows, S, D]`` arrays.

    Query row ``h`` is (batch ``h // heads``, head ``h % heads``); it reads KV
    row ``(h // heads) * kv_heads + head // (heads // kv_heads)``.

    - ``return_lse``: also return the base-2 log-sum-exp of the scaled scores,
      ``[rows, NUM_SUBLANES, q_seq]`` float32 with identical sublane rows.
    - ``resume``: two more inputs, the output and base-2 log-sum-exp of
      attention over other keys; the softmax continues from them.
    - ``runtime_kv``: a leading int32 ``[row offset, real keys]`` input read at
      run time: KV rows shift by ``offset`` and only their first ``real keys``
      keys count. A rank-dependent shard choice then compiles one program.
    - ``rows_in``: q/k/v and the output are ``[S, rows * D]``, row ``h`` the
      ``h``-th ``D``-wide column block, read in place (no transposes).
    - ``block_sizes``: ``(block_q, block_kv, block_kv_compute)`` instead of the
      autotuned ones.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    q_heads_per_kv_head = heads // kv_heads
    carry_bytes_per_q_row = _carry_bytes_per_q_row(
        head_dim, f32_out=(out_dtype == jnp.float32), resume=resume
    )
    # One VMEM figure feeds both the block-size model and Mosaic's limit.
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    if rows_in and (return_lse or resume):
        raise ValueError("rows_in takes neither return_lse nor resume")
    block_q, block_kv, block_kv_compute = block_sizes or _autotune_block_sizes(
        q_seq, kv_seq, head_dim, vmem_capacity_bytes, carry_bytes_per_q_row
    )
    grid_width = pl.cdiv(kv_seq, block_kv)
    qk_scale = scale * LOG2_E
    logger.info(
        "TPU splash attention: q_seq=%d kv_seq=%d -> flash kernel blocks=%s%s",
        q_seq,
        kv_seq,
        (block_q, block_kv, block_kv_compute),
        " (live length)" if rows_in else " (ring)" if runtime_kv else "",
    )

    def kernel(*refs):
        if runtime_kv:
            kv_info_ref, *refs = refs
        q_ref, k_ref, v_ref, *refs = refs
        if resume:
            o_prev_ref, lse_prev_ref, *refs = refs
        o_ref, *refs = refs
        if return_lse:
            lse_ref, *refs = refs
        m_scratch_ref, l_scratch_ref, o_scratch_ref = refs
        float32 = jnp.float32
        sv_dims = (((0,), (0,)), ((), ()))
        j = pl.program_id(2)

        @pl.when(j == 0)
        def init():
            if resume:
                # The earlier keys enter as one pseudo-key with score lse_prev
                # and value o_prev: sum(exp2(s) v) = exp2(lse_prev) o_prev.
                o_scratch_ref[...] = o_prev_ref[...].astype(float32).T
                l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
                m_scratch_ref[...] = lse_prev_ref[...]
            else:
                o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
                l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
                m_scratch_ref[...] = jnp.full_like(m_scratch_ref, _MASK_VALUE)

        def update(slice_k, real=None):
            q_tile = q_ref[...]
            k_chunk = k_ref[slice_k, :]
            v_chunk = v_ref[slice_k, :]
            qk = lax.dot_general(
                k_chunk, q_tile, NT_DIM_NUMBERS, preferred_element_type=float32
            )
            if real is not None:
                # Rows from `real` on are padding or past the array: drop their
                # scores and zero their values (they may hold NaN).
                rows = lax.broadcasted_iota(jnp.int32, (qk.shape[0], 1), 0)
                qk = jnp.where(rows < real, qk, _MASK_VALUE)
                v_chunk = jnp.where(rows < real, v_chunk, jnp.zeros_like(v_chunk))
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

        def last_body():
            # The ragged last KV block: read only the keys the array holds.
            remain = kv_seq - (grid_width - 1) * block_kv
            full, tail = divmod(remain, block_kv_compute)
            lax.fori_loop(0, full, compute_body, None, unroll=True)
            if tail:
                update(pl.ds(full * block_kv_compute, tail))

        if runtime_kv:
            # A count past the array would read stale VMEM rows unmasked.
            real_keys = jnp.minimum(kv_info_ref[1], kv_seq)
            start = j * block_kv

            @pl.when(start + block_kv <= real_keys)
            def body():
                lax.fori_loop(
                    0, block_kv // block_kv_compute, compute_body, None, unroll=True
                )

            if kv_seq % block_kv:
                # A shard without padding takes the static ragged tail.
                pl.when((j == grid_width - 1) & (real_keys == kv_seq))(last_body)

            # In a padded shard, the block holding the last real key masks the
            # rest; blocks past it are skipped.
            @pl.when(
                (start < real_keys)
                & (start + block_kv > real_keys)
                & (real_keys < kv_seq)
            )
            def edge_body():
                for c in range(block_kv // block_kv_compute):
                    chunk_start = start + c * block_kv_compute

                    @pl.when(chunk_start < real_keys)
                    def _():
                        update(
                            pl.ds(c * block_kv_compute, block_kv_compute),
                            real=real_keys - chunk_start,
                        )

        elif kv_seq % block_kv == 0:
            # Every KV block is whole: one body, not a second identical copy.
            lax.fori_loop(
                0, block_kv // block_kv_compute, compute_body, None, unroll=True
            )
        else:

            @pl.when(j != grid_width - 1)
            def body():
                lax.fori_loop(
                    0, block_kv // block_kv_compute, compute_body, None, unroll=True
                )

            pl.when(j == grid_width - 1)(last_body)

        @pl.when(j == grid_width - 1)
        def end():
            # No real keys (an all-padding shard) leaves l == 0: output 0 and a
            # _MASK_VALUE log-sum-exp, which a later resume treats exactly.
            l = l_scratch_ref[...]
            l_safe = jnp.where(l > 0, l, 1.0)
            # l is replicated across the sublanes; tile it to the [D, block_q]
            # scratch and write the [block_q, D] output tile.
            l_inv = jnp.tile(1.0 / l_safe, (head_dim // NUM_SUBLANES, 1))
            o_ref[...] = (o_scratch_ref[...] * l_inv).T.astype(o_ref.dtype)
            if return_lse:
                lse_ref[...] = jnp.where(
                    l > 0, m_scratch_ref[...] + jnp.log2(l_safe), _MASK_VALUE
                )

    def q_index(h, i, j, *_):
        return (i, h) if rows_in else (h, i, 0)

    def kv_index(h, i, j, *scalars):
        row = (h // heads) * kv_heads + (h % heads) // q_heads_per_kv_head
        if scalars:
            # Blocks past the last real key skip compute; repeating the last
            # real block's index there lets the pipeline skip their K/V copies.
            real = jnp.minimum(scalars[0][1], kv_seq)
            last = jnp.maximum((real + block_kv - 1) // block_kv - 1, 0)
            row, j = row + scalars[0][0], jnp.minimum(j, last)
        return (j, row) if rows_in else (row, j, 0)

    def lse_index(h, i, j, *_):
        return (h, 0, i)

    q_block = (block_q, head_dim) if rows_in else (None, block_q, head_dim)
    kv_block = (block_kv, head_dim) if rows_in else (None, block_kv, head_dim)
    in_specs = [
        pl.BlockSpec(q_block, q_index),
        pl.BlockSpec(kv_block, kv_index),
        pl.BlockSpec(kv_block, kv_index),
    ]
    if resume:
        in_specs += [
            pl.BlockSpec((None, block_q, head_dim), q_index),
            pl.BlockSpec((None, NUM_SUBLANES, block_q), lse_index),
        ]
    out_specs = pl.BlockSpec(q_block, q_index)
    out_shape = jax.ShapeDtypeStruct(
        (q_seq, num_rows * head_dim) if rows_in else (num_rows, q_seq, head_dim),
        out_dtype,
    )
    if return_lse:
        out_specs = [out_specs, pl.BlockSpec((None, NUM_SUBLANES, block_q), lse_index)]
        out_shape = [
            out_shape,
            jax.ShapeDtypeStruct((num_rows, NUM_SUBLANES, q_seq), jnp.float32),
        ]
    grid = (num_rows, pl.cdiv(q_seq, block_q), grid_width)
    scratch_shapes = [
        pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
        pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
        pltpu.VMEM((head_dim, block_q), jnp.float32),
    ]
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel", "arbitrary"),
        # Mosaic's LLO scheduler: 56.6 vs 87.4 ms at 75600 tokens, 13.2 vs
        # 16.3 ms at 32760 (one tpu7x core, libtpu 0.0.50, identical output).
        flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
        # Autotuned score tiles exceed Mosaic's 32 MiB default.
        vmem_limit_bytes=vmem_capacity_bytes,
    )
    name = (
        "diffusion_flash_attention_live_length"
        if rows_in
        else "diffusion_flash_attention_ring"
        if runtime_kv
        else "diffusion_flash_attention"
    )
    if runtime_kv:
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=compiler_params,
            out_shape=out_shape,
            name=name,
        )
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=compiler_params,
        out_shape=out_shape,
        name=name,
    )


def _flash_attention(q, k, v, scale):
    """Flash attention with an online exp2 softmax on ``[B, H, S, D]`` arrays.

    Takes the unscaled query: the softmax scale is applied to the float32
    scores. Inputs are not padded; Mosaic masks the ragged query edge block and
    the last KV block reads only real keys.
    """
    import jax

    heads, q_seq, head_dim = q.shape[1:]
    call = _flash_call(
        q_seq,
        k.shape[2],
        head_dim,
        scale,
        num_rows=heads,
        heads=heads,
        kv_heads=k.shape[1],
        out_dtype=q.dtype,
    )
    return jax.vmap(call)(q, k, v)


def _flash_attention_ring(q, k, v, kv_info, out_prev, lse_prev, scale):
    """One ring step of flash attention with a run-time KV shard choice.

    ``q`` is ``[B, H, S, D]``; ``k``/``v`` stack shards ``[N, H_kv, S_kv, D]``
    (``N >= B``), and ``kv_info`` = int32 ``[row offset, real keys]`` picks KV
    rows ``offset .. offset + B * H_kv`` of the batch-flattened stack (``None``:
    ``k``/``v`` are ``[B, H_kv, S_kv, D]`` and every key counts). With
    ``out_prev``/``lse_prev`` the softmax resumes from earlier shards. Returns
    the output, float32 for a ring step (``kv_info`` given) so the carry is
    not rounded between steps and ``q``'s dtype otherwise, and the base-2
    log-sum-exp ``[B, H, NUM_SUBLANES, S]`` float32.
    """
    import jax.numpy as jnp

    batch, heads, q_seq, head_dim = q.shape
    resume = out_prev is not None
    runtime_kv = kv_info is not None
    call = _flash_call(
        q_seq,
        k.shape[2],
        head_dim,
        scale,
        num_rows=batch * heads,
        heads=heads,
        kv_heads=k.shape[1],
        out_dtype=jnp.float32 if runtime_kv else q.dtype,
        return_lse=True,
        resume=resume,
        runtime_kv=runtime_kv,
    )

    # The offset may address any batch row of the stack, so batch folds into
    # the row axis instead of being mapped over.
    def flat(t):
        return t.reshape(-1, *t.shape[2:])

    info = (kv_info,) if runtime_kv else ()
    prev = (flat(out_prev), flat(lse_prev)) if resume else ()
    out, lse = call(*info, flat(q), flat(k), flat(v), *prev)
    return out.reshape(q.shape), lse.reshape(batch, heads, NUM_SUBLANES, q_seq)


def _flash_attention_live_length(q, k, v, live, scale, num_heads):
    """Flash attention over one packed sequence's ``[S, H * D]`` rows whose
    rows from ``live`` (int32 ``[1]``) on are padding.

    A ``runtime_kv`` call with row offset 0 and ``live`` real keys: keys past
    it are masked and KV blocks past it skipped. Padding rows output zeros. The
    length is read on device, so one kernel serves every live length of a
    padded shape.
    """
    import jax.numpy as jnp

    q_seq, kv_seq = q.shape[0], k.shape[0]
    head_dim = q.shape[1] // num_heads
    blocks = None
    if q_seq >= _LIVE_LENGTH_BLOCK_Q and kv_seq >= _LIVE_LENGTH_BLOCK_KV:
        blocks = (_LIVE_LENGTH_BLOCK_Q, _LIVE_LENGTH_BLOCK_KV, _LIVE_LENGTH_BLOCK_KV)
    call = _flash_call(
        q_seq,
        kv_seq,
        head_dim,
        scale,
        num_rows=num_heads,
        heads=num_heads,
        kv_heads=k.shape[1] // head_dim,
        out_dtype=q.dtype,
        runtime_kv=True,
        rows_in=True,
        block_sizes=blocks,
    )
    live = live.astype(jnp.int32).reshape(1)
    out = call(jnp.concatenate([jnp.zeros_like(live), live]), q, k, v)
    rows = jnp.arange(q_seq, dtype=jnp.int32)[:, None]
    return jnp.where(rows < live[0], out, jnp.zeros_like(out))


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

    # make_splash_mha takes no scale. Keep the scaled query in f32: rounding
    # it to bf16 perturbs logits enough to change high-gain attention weights.
    output_dtype = q.dtype
    q = q.astype(jnp.float32) * scale
    k = k.astype(jnp.float32)
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
    with jax.default_matmul_precision("highest"):
        if kv_padded == kv_seq:
            out = jax.vmap(kernel)(q, k, v)
        else:
            segment_ids = splash.SegmentIds(
                q=jnp.ones((q_padded,), jnp.int32),
                kv=(jnp.arange(kv_padded) < kv_seq).astype(jnp.int32),
            )
            out = jax.vmap(functools.partial(kernel, segment_ids=segment_ids))(q, k, v)
    return out[:, :, :q_seq].astype(output_dtype)


def _register_ops() -> None:
    """Create the kernel ops; must run before torch.compile traces a model.

    A failure marks the kernels unavailable, so callers fall back to SDPA
    instead of retrying a half-registered ``torch.library`` op.
    """
    global _FLASH_OP, _FLASH_RING_OP, _FLASH_LIVE_LENGTH_OP, _SPLASH_MHA_OP
    global _SPLASH_AVAILABLE
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

            def flash_ring(
                q: jax.Array,
                k: jax.Array,
                v: jax.Array,
                kv_info: jax.Array | None,
                out_prev: jax.Array | None,
                lse_prev: jax.Array | None,
                scale: float,
            ) -> tuple[jax.Array, jax.Array]:
                return _flash_attention_ring(
                    q, k, v, kv_info, out_prev, lse_prev, scale
                )

            def flash_live_length(
                q: jax.Array,
                k: jax.Array,
                v: jax.Array,
                live: jax.Array,
                scale: float,
                num_heads: int,
            ) -> jax.Array:
                return _flash_attention_live_length(q, k, v, live, scale, num_heads)

            def splash_mha(
                q: jax.Array, k: jax.Array, v: jax.Array, scale: float, is_causal: bool
            ) -> jax.Array:
                return _splash_mha(q, k, v, scale, is_causal)

            def flash_fake(q, k, v, scale):
                return q.new_empty(q.shape)

            def flash_ring_fake(q, k, v, kv_info, out_prev, lse_prev, scale):
                batch, heads, q_seq = q.shape[:3]
                lse_shape = (batch, heads, NUM_SUBLANES, q_seq)
                # A ring step (kv_info given) returns its float32 carry.
                out_dtype = torch.float32 if kv_info is not None else q.dtype
                return (
                    q.new_empty(q.shape, dtype=out_dtype),
                    q.new_empty(lse_shape, dtype=torch.float32),
                )

            def flash_live_length_fake(q, k, v, live, scale, num_heads):
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
            flash_ring_op = cpu_export_jax_op(
                "sglang_tpu_pallas::diffusion_flash_attention_ring", flash_ring
            )
            flash_ring_op.register_fake(flash_ring_fake)
            flash_live_length_op = cpu_export_jax_op(
                "sglang_tpu_pallas::diffusion_flash_attention_live_length",
                flash_live_length,
            )
            flash_live_length_op.register_fake(flash_live_length_fake)
        except Exception as exc:
            logger.warning(
                "TPU Pallas splash attention ops failed to register (%s); using SDPA.",
                exc,
            )
            _SPLASH_AVAILABLE = False
            return
        _SPLASH_MHA_OP = splash_mha_op
        _FLASH_RING_OP = flash_ring_op
        _FLASH_LIVE_LENGTH_OP = flash_live_length_op
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
    return _kernel_input_fallback_reason(query, key, value)


def _kernel_input_fallback_reason(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> str | None:
    """Why the kernels cannot take these ``[B, H, S, D]`` inputs, else None."""
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


def _ring_kernel_fallback_reason(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> str | None:
    if not _use_custom_kernel(query.shape[2], key.shape[2], False):
        return "short"
    return _sdpa_fallback_reason(query, key, value, False)


# float32 score bytes per query chunk of the reference attention.
_REFERENCE_SCORE_BYTES = 256 << 20


def _reference_chunk_rows(query: torch.Tensor, key: torch.Tensor) -> int:
    batch, heads, kv_seq = query.shape[0], query.shape[1], key.shape[2]
    return max(1, _REFERENCE_SCORE_BYTES // (4 * batch * heads * max(kv_seq, 1)))


def _attention_with_lse_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    real_keys: torch.Tensor | int | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 ``[B, H, S, D]`` attention and its natural-log LSE ``[B, H, S]``.

    Query rows run in chunks, so the float32 scores stay bounded.
    """
    if query.shape[1] != key.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    batch, heads, q_seq, _ = query.shape
    kv_seq = key.shape[2]
    key_t, value = key.float().transpose(-1, -2), value.float()
    cols = torch.arange(kv_seq, device=query.device)
    if real_keys is not None:
        real = cols < real_keys
        # Keys past the real count may hold NaN; they must add nothing.
        value = torch.where(real[:, None], value, 0.0)
    chunk = _reference_chunk_rows(query, key)
    outs, lses = [], []
    for start in range(0, q_seq, chunk):
        scores = torch.matmul(query[:, :, start : start + chunk].float(), key_t) * scale
        if real_keys is not None:
            scores = scores.masked_fill(~real, float("-inf"))
        if is_causal:
            rows = torch.arange(start, start + scores.shape[2], device=query.device)
            scores = scores.masked_fill(cols > rows[:, None], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        # Rows without a real key: output 0, log-sum-exp -inf.
        probs = torch.exp(scores - lse[..., None]).nan_to_num(0.0)
        outs.append(torch.matmul(probs, value))
        lses.append(lse)
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


def _merge_attention(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_prev: torch.Tensor,
    lse_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over two key sets from each set's output and natural-log LSE."""
    peak = torch.maximum(lse, lse_prev)
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)
    w, w_prev = torch.exp(lse - peak), torch.exp(lse_prev - peak)
    total = w + w_prev
    merged = out.float() * w[..., None] + out_prev.float() * w_prev[..., None]
    merged = merged / torch.where(total > 0, total, 1.0)[..., None]
    return merged, peak + torch.log(total)


def tpu_attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, H, S, D]`` attention and its natural-log log-sum-exp ``[B, H, S]``
    float32, the ``return_softmax_lse`` contract of the ring merge."""
    if not is_causal and _ring_kernel_fallback_reason(query, key, value) is None:
        _register_ops()
        out, lse2 = _FLASH_RING_OP(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            None,
            None,
            None,
            scale,
        )
        return out, lse2[:, :, 0] * math.log(2.0)
    if is_causal and query.shape[2] != key.shape[2]:
        raise NotImplementedError("ragged causal attention with LSE")
    out, lse = _attention_with_lse_reference(
        query, key, value, scale, is_causal=is_causal
    )
    return out.to(query.dtype), lse


def tpu_ring_attention_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_info: torch.Tensor,
    carry: tuple[torch.Tensor, torch.Tensor] | None,
    scale: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One all-gather ring step; see ``AttentionImpl.forward_ring_resume``.

    Steps carry the float32 partial output and its log-sum-exp (base 2 from
    the kernel; natural log from the reference, which serves the shapes the
    kernel does not take), so only the returned output is rounded to the
    query dtype. The carry layouts differ by path, chosen statically from the
    shapes: kernel ``(out [B, H, S, D], lse [B, H, NUM_SUBLANES, S])`` with
    identical sublane rows, reference ``(out, lse [B, H, S])``.
    """
    assert kv_info.dtype == torch.int32 and tuple(kv_info.shape) == (2,), (
        f"kv_info must be int32 [row offset, real keys], got "
        f"{kv_info.dtype} {tuple(kv_info.shape)}"
    )
    reason = _ring_kernel_fallback_reason(query, key, value)
    if reason is None:
        _register_ops()
        out_prev, lse_prev = carry if carry is not None else (None, None)
        out, lse2 = _FLASH_RING_OP(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            kv_info,
            out_prev,
            lse_prev,
            scale,
        )
        return out.to(query.dtype), (out, lse2)
    log_key = ("ring", query.shape[2], key.shape[2], reason)
    # Logging breaks a torch.compile graph; eager runs report the fallback.
    if not torch.compiler.is_compiling() and log_key not in _logged_sdpa_fallbacks:
        _logged_sdpa_fallbacks.add(log_key)
        logger.info(
            "TPU ring attention step: q_seq=%d kv_seq=%d -> float32 reference "
            "(%s); %d query chunks per step.",
            query.shape[2],
            key.shape[2],
            reason,
            -(-query.shape[2] // _reference_chunk_rows(query, key)),
        )
    # Same shard selection as the kernel's index map, as device data.
    batch, kv_heads = query.shape[0], key.shape[1]
    rows = kv_info[0] + torch.arange(batch * kv_heads, device=key.device)
    shard = (batch, kv_heads, *key.shape[2:])
    key = key.reshape(-1, *key.shape[2:]).index_select(0, rows).view(shard)
    value = value.reshape(-1, *value.shape[2:]).index_select(0, rows).view(shard)
    out, lse = _attention_with_lse_reference(
        query, key, value, scale, real_keys=kv_info[1]
    )
    if carry is not None:
        out, lse = _merge_attention(out, lse, *carry)
    return out.to(query.dtype), (out, lse)


def _resolve_ring_replica_ranks() -> tuple[
    list[int], torch.distributed.ProcessGroup | None
]:
    world_size = torch.distributed.get_world_size()
    replica_group = get_replica_group()
    if replica_group.world_size < world_size:
        pg = replica_group.device_group
        ranks = (
            list(torch.distributed.get_process_group_ranks(pg))
            if pg is not None
            else list(replica_group.ranks)
        )
        return ranks, pg
    return list(range(world_size)), None


def _resolve_ring_mesh_spec() -> tuple[int, int, int] | None:
    """Return ``(outer_size, ring_size, inner_size)`` when ring ranks form a regular grid in the replica."""
    if not torch.distributed.is_initialized() or not model_parallel_is_initialized():
        return None
    sp_group = get_sp_group()
    ring_size = sp_group.ring_world_size
    if ring_size <= 1:
        return None
    replica_ranks, _ = _resolve_ring_replica_ranks()
    replica_size = len(replica_ranks)
    ring_ranks = list(torch.distributed.get_process_group_ranks(sp_group.ring_group))
    if len(ring_ranks) != ring_size or replica_size % ring_size != 0:
        return None
    rank_to_local = {rank: idx for idx, rank in enumerate(replica_ranks)}
    try:
        local_ring_ranks = [rank_to_local[r] for r in ring_ranks]
    except KeyError:
        return None
    inner_size = local_ring_ranks[1] - local_ring_ranks[0]
    group_span = ring_size * inner_size
    if inner_size <= 0 or replica_size % group_span != 0:
        return None
    outer_idx, inner_idx = divmod(local_ring_ranks[0], group_span)
    if inner_idx >= inner_size:
        return None
    expected = [
        outer_idx * group_span + r * inner_size + inner_idx for r in range(ring_size)
    ]
    if local_ring_ranks != expected:
        return None
    return replica_size // group_span, ring_size, inner_size


def _get_ring_op(
    outer_size: int,
    ring_size: int,
    inner_size: int,
    scale: float,
    use_fixed_m: bool,
    *,
    process_group: torch.distributed.ProcessGroup | None = None,
    replica_lead: int = 0,
) -> Callable | None:
    key = (outer_size, ring_size, inner_size, replica_lead, scale, use_fixed_m)
    if key in _RING_OPS:
        return _RING_OPS[key]
    with _OPS_LOCK:
        if key in _RING_OPS:
            return _RING_OPS[key]
        op: Callable | None = None
        try:
            import jax
            from jax.sharding import PartitionSpec
            from torch_tpu._internal.pallas import get_pallas_mesh

            from sglang_tpu.multimodal_gen.diffusion_ring_attention import (
                ring_attention_sharded,
            )
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            axis_names = ("batch", "ring", "inner")
            mesh = get_pallas_mesh(
                axis_names,
                (outer_size, ring_size, inner_size),
                process_group=process_group,
            )
            spec = PartitionSpec(*axis_names)
            sharded = jax.shard_map(
                functools.partial(
                    ring_attention_sharded,
                    ring_axis="ring",
                    axis_names=axis_names,
                    scale=scale,
                    use_fixed_m=use_fixed_m,
                ),
                mesh=mesh,
                in_specs=(spec, spec, spec),
                out_specs=spec,
                check_vma=False,
            )

            def _wrapped(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
                return sharded(q, k, v)

            def _ring_fake(q, k, v):
                return q.new_empty(q.shape)

            scale_hex = f"{int(np.float64(scale).view(np.uint64)):016x}"
            fm_tag = "fm1" if use_fixed_m else "fm0"
            rep_tag = f"r{replica_lead}_" if process_group is not None else ""
            op_name = (
                f"sglang_tpu_pallas::diffusion_ring_attn_"
                f"{rep_tag}{outer_size}_{ring_size}_{inner_size}_{fm_tag}_{scale_hex}"
            )
            op = cpu_export_jax_op(
                op_name,
                _wrapped,
                mesh=mesh,
                input_partition_specs=(spec, spec, spec),
                output_partition_specs=(spec,),
            )
            op.register_fake(_ring_fake)
        except Exception as exc:
            logger.warning(
                "Failed to register TPU Pallas ring attention op at init (%s); "
                "leaving ring kernel disabled (first-call trace/lowering errors are not caught here).",
                exc,
            )
        _RING_OPS[key] = op
        return op


def _ranks_pair_on_chip() -> bool:
    from sglang_tpu.attention.pallas.util import get_tpu_version

    # get_pallas_mesh puts ranks r, r ^ 1 on one tpu7x chip (k3_megakernel/config.py);
    # short-2D runs its Ulysses all_to_all over those pairs.
    return get_tpu_version() == 7


def _short_2d_ring_sharded(q, k, v, *, scale: float):
    """Run intra-chip Ulysses=2 + bulk-ppermute ring attention on short shards."""
    import jax
    import jax.numpy as jnp

    lead_axes = (0, 1, 2)
    q_s = jnp.squeeze(q, axis=lead_axes)
    k_s = jnp.squeeze(k, axis=lead_axes)
    v_s = jnp.squeeze(v, axis=lead_axes)

    q_u = jax.lax.all_to_all(q_s, "uly", split_axis=2, concat_axis=1, tiled=True)
    k_u = jax.lax.all_to_all(k_s, "uly", split_axis=2, concat_axis=1, tiled=True)
    v_u = jax.lax.all_to_all(v_s, "uly", split_axis=2, concat_axis=1, tiled=True)

    batch, seq_u, heads_u, head_dim = q_u.shape
    kv_heads_u = k_u.shape[2]
    q_flat = jnp.swapaxes(q_u, 1, 2).reshape(batch * heads_u, seq_u, head_dim)
    k_flat = jnp.swapaxes(k_u, 1, 2).reshape(batch * kv_heads_u, seq_u, head_dim)
    v_flat = jnp.swapaxes(v_u, 1, 2).reshape(batch * kv_heads_u, seq_u, head_dim)

    ring_2d = jax.lax.axis_size("ring")
    perm = [(i, (i + 1) % ring_2d) for i in range(ring_2d)]
    step0 = _flash_call(
        seq_u,
        seq_u,
        head_dim,
        scale,
        num_rows=batch * heads_u,
        heads=heads_u,
        kv_heads=kv_heads_u,
        out_dtype=jnp.float32,
        return_lse=True,
        resume=False,
    )
    step_mid = (
        _flash_call(
            seq_u,
            seq_u,
            head_dim,
            scale,
            num_rows=batch * heads_u,
            heads=heads_u,
            kv_heads=kv_heads_u,
            out_dtype=jnp.float32,
            return_lse=True,
            resume=True,
        )
        if ring_2d > 2
        else None
    )
    step_last = _flash_call(
        seq_u,
        seq_u,
        head_dim,
        scale,
        num_rows=batch * heads_u,
        heads=heads_u,
        kv_heads=kv_heads_u,
        out_dtype=q_u.dtype,
        return_lse=False,
        resume=True,
    )

    k_cur, v_cur = k_flat, v_flat
    k_nxt = jax.lax.ppermute(k_cur, "ring", perm)
    v_nxt = jax.lax.ppermute(v_cur, "ring", perm)
    out_acc, lse_acc = step0(q_flat, k_cur, v_cur)
    for _ in range(1, ring_2d - 1):
        k_cur, v_cur = k_nxt, v_nxt
        k_nxt = jax.lax.ppermute(k_cur, "ring", perm)
        v_nxt = jax.lax.ppermute(v_cur, "ring", perm)
        out_acc, lse_acc = step_mid(q_flat, k_cur, v_cur, out_acc, lse_acc)
    out_flat = step_last(q_flat, k_nxt, v_nxt, out_acc, lse_acc)
    out_u = jnp.swapaxes(out_flat.reshape(batch, heads_u, seq_u, head_dim), 1, 2)
    out_s = jax.lax.all_to_all(out_u, "uly", split_axis=1, concat_axis=2, tiled=True)
    return jnp.expand_dims(out_s, axis=lead_axes)


def _get_short_2d_ring_op(
    outer_size: int,
    ring_size: int,
    scale: float,
    *,
    process_group: torch.distributed.ProcessGroup | None = None,
    replica_lead: int = 0,
) -> Callable | None:
    key = (outer_size, ring_size, replica_lead, scale)
    if key in _SHORT_2D_RING_OPS:
        return _SHORT_2D_RING_OPS[key]
    with _OPS_LOCK:
        if key in _SHORT_2D_RING_OPS:
            return _SHORT_2D_RING_OPS[key]
        op: Callable | None = None
        try:
            import jax
            from jax.sharding import PartitionSpec
            from torch_tpu._internal.pallas import get_pallas_mesh

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            axis_names = ("batch", "ring", "uly")
            mesh = get_pallas_mesh(
                axis_names,
                (outer_size, ring_size // 2, 2),
                process_group=process_group,
            )
            spec = PartitionSpec(*axis_names)
            sharded = jax.shard_map(
                functools.partial(_short_2d_ring_sharded, scale=scale),
                mesh=mesh,
                in_specs=(spec, spec, spec),
                out_specs=spec,
                check_vma=False,
            )

            def _wrapped(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
                return sharded(q, k, v)

            def _short_2d_fake(q, k, v):
                return q.new_empty(q.shape)

            scale_hex = f"{int(np.float64(scale).view(np.uint64)):016x}"
            rep_tag = f"r{replica_lead}_" if process_group is not None else ""
            op_name = (
                f"sglang_tpu_pallas::diffusion_short_2d_ring_attn_"
                f"{rep_tag}{outer_size}_{ring_size}_{scale_hex}"
            )
            op = cpu_export_jax_op(
                op_name,
                _wrapped,
                mesh=mesh,
                input_partition_specs=(spec, spec, spec),
                output_partition_specs=(spec,),
            )
            op.register_fake(_short_2d_fake)
        except Exception as exc:
            logger.warning(
                "Failed to register TPU Pallas short-2D ring attention op at init (%s); "
                "leaving short-2D ring op disabled (first-call trace/lowering errors are not caught here).",
                exc,
            )
        _SHORT_2D_RING_OPS[key] = op
        return op


class TpuSplashAttentionBackend(SDPABackend):
    """TPU Pallas attention, tuned on v7x; uncovered shapes fall back to SDPA."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        # The platform's own backend; TpuOOTPlatform names it tpu_splash_attn.
        return AttentionBackendEnum.OOT

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        # Ring by K/V all-gather (usp.allgather_kv_ring_attn), not p2p.
        return True

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
        self._ring_op: Callable | None = None
        self._short_2d_ring_op: Callable | None = None
        if (
            self.use_splash
            and not self.causal
            and self.softmax_scale > 0
            and tpu_envs.SGLANG_TPU_ENABLE_DIFFUSION_RING_KERNEL.get()
        ):
            mesh_spec = _resolve_ring_mesh_spec()
            if mesh_spec is not None:
                outer_size, ring_size, inner_size = mesh_spec
                replica_ranks, replica_pg = _resolve_ring_replica_ranks()
                replica_lead = replica_ranks[0] if replica_pg is not None else 0
                self._ring_op = _get_ring_op(
                    outer_size,
                    ring_size,
                    inner_size,
                    self.softmax_scale,
                    tpu_envs.SGLANG_TPU_ENABLE_DIFFUSION_RING_FIXED_M.get(),
                    process_group=replica_pg,
                    replica_lead=replica_lead,
                )
                if (
                    inner_size == 1
                    and ring_size >= 4
                    and ring_size % 2 == 0
                    and _ranks_pair_on_chip()
                ):
                    self._short_2d_ring_op = _get_short_2d_ring_op(
                        outer_size,
                        ring_size,
                        self.softmax_scale,
                        process_group=replica_pg,
                        replica_lead=replica_lead,
                    )
        _log_resolved_mode_once(self.use_splash, self.dropout)
        # One live segment plus trailing padding (MiniMax H3's packed layout)
        # runs as one kernel call that reads the live length on device, so
        # callers pass no host bounds and a padded shape compiles once.
        self.trailing_padding_on_device = self.use_splash and bool(
            extra_impl_args.get("packed_trailing_padding", False)
        )
        # Other packed layouts stay on plain SDPA, one call per segment.
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
        return_softmax_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_softmax_lse:
            # [b, s, h, d] output and [b, h, s] natural-log LSE.
            if not self.use_splash or query.device.type != "tpu":
                output, lse = _attention_with_lse_reference(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    self.softmax_scale,
                    is_causal=self.causal,
                )
                return output.to(query.dtype).transpose(1, 2), lse
            output, lse = tpu_attention_with_lse(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                scale=self.softmax_scale,
                is_causal=self.causal,
            )
            return output.transpose(1, 2), lse
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

    def forward_ring(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._ring_op is None or self.causal or query.device.type != "tpu":
            return None
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return None
        batch, q_seq, num_q_heads, head_dim = query.shape
        kv_batch, kv_seq, num_kv_heads, k_dim = key.shape
        if batch != kv_batch or q_seq != kv_seq or q_seq < 256:
            return None
        if head_dim != 128 or k_dim != 128 or value.shape != key.shape:
            return None
        if num_kv_heads == 0 or num_q_heads % num_kv_heads != 0:
            return None
        if not query.dtype == key.dtype == value.dtype == torch.bfloat16:
            return None
        q = query.contiguous().reshape(1, 1, 1, *query.shape)
        k = key.contiguous().reshape(1, 1, 1, *key.shape)
        v = value.contiguous().reshape(1, 1, 1, *value.shape)
        if (
            self._short_2d_ring_op is not None
            and q_seq <= _SHORT_SHARD_2D_RING_MAX
            and num_q_heads % 2 == 0
            and num_kv_heads % 2 == 0
        ):
            return self._short_2d_ring_op(q, k, v).reshape(query.shape)
        return self._ring_op(q, k, v).reshape(query.shape)

    def forward_ring_resume(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_info: torch.Tensor,
        carry: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # carry = (float32 partial output, log-sum-exp); its layout depends on
        # whether the kernel or the reference ran (tpu_ring_attention_step).
        # Treat it as opaque.
        if self.causal:
            raise NotImplementedError("the all-gather ring is non-causal only")
        return tpu_ring_attention_step(
            query, key, value, kv_info, carry, scale=self.softmax_scale
        )

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
        # cu_seqlens = [0, used, total] without host bounds: rows [used, total)
        # are padding. Callers that pass host bounds (e.g. H3's token refiner,
        # whose length varies per prompt) keep the per-segment SDPA path.
        if self.trailing_padding_on_device and cu_seqlens_host is None:
            # [S, H, D] views in the kernel's [B, H, S, D] terms, for the checks.
            q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (query, key, value))
            if (
                not self.causal
                and query.device.type == "tpu"
                and cu_seqlens.shape == (3,)
                and query.shape[0] >= 512
                and key.shape[0] == query.shape[0]
                and _kernel_input_fallback_reason(q, k, v) is None
            ):
                # [S, H * D] rows: one lane-dense copy of a strided view, where
                # .contiguous() pads 7 heads to 8 sublanes and the kernel's
                # reshape then relayouts again.
                rows = (t.reshape(t.shape[0], -1) for t in (query, key, value))
                output = _FLASH_LIVE_LENGTH_OP(
                    *rows,
                    cu_seqlens[1:2].to(torch.int32),
                    self.softmax_scale,
                    query.shape[1],
                )
                return output.view(query.shape)
            # The caller dropped its host bounds for this kernel; SDPA would
            # read them back and compile once per live length.
            raise ValueError(
                "live-length attention without host bounds needs the kernel: "
                f"q {tuple(query.shape)} {query.dtype} on {query.device}, "
                f"k {tuple(key.shape)}, cu_seqlens {tuple(cu_seqlens.shape)}, "
                f"causal={self.causal}"
            )
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
